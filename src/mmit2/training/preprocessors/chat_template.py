"""Chat-template preprocessor with explicit prompt masking."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image

from mmit2.data.types import CanonicalSample
from mmit2.training.preprocessors.base import IGNORE_INDEX, Preprocessor


def _load_image(sample: CanonicalSample, image_root: str = "") -> Optional[Image.Image]:
    if not sample.image_path:
        return None
    pil_image = sample.metadata.get("_pil_image") if sample.metadata else None
    if pil_image is not None:
        return pil_image.convert("RGB")
    img_path = os.path.join(image_root, sample.image_path) if image_root else sample.image_path
    if not os.path.isfile(img_path):
        return None
    return Image.open(img_path).convert("RGB")


def _build_messages(sample: CanonicalSample, has_image: bool) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for turn in sample.turns:
        role = "user" if turn.role == "human" else "assistant"
        if role == "user" and has_image and not messages:
            content = [{"type": "image"}, {"type": "text", "text": turn.content}]
        else:
            content = [{"type": "text", "text": turn.content}]
        messages.append({"role": role, "content": content})
    return messages


def _build_prompt_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last_assistant_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "assistant":
            last_assistant_idx = idx
            break
    return messages if last_assistant_idx < 0 else messages[:last_assistant_idx]


class ChatTemplatePreprocessor(Preprocessor):
    def tokenize(
        self,
        sample: CanonicalSample,
        processor: Any,
        image_root: str = "",
        max_length: int = 2048,
        debug_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        # `processor` must be a Hugging Face multimodal processor that supports
        # both `apply_chat_template(...)` and `processor(text=..., images=...)`.
        image = _load_image(sample, image_root)
        messages = _build_messages(sample, image is not None)
        if not messages:
            raise ValueError(f"Sample {sample.id} has no turns")

        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        prompt_messages = _build_prompt_messages(messages)
        prompt_text = ""
        if prompt_messages:
            prompt_text = processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )

        if debug_sink is not None:
            debug_sink({
                "sample_id": sample.id,
                "has_image": image is not None,
                "message_count": len(messages),
                "full_text": full_text,
                "prompt_text": prompt_text,
            })

        images = [image] if image is not None else None
        full_inputs = processor(
            text=full_text,
            images=images,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs.get("attention_mask", torch.ones_like(input_ids))
        if attention_mask.dim() > 1:
            attention_mask = attention_mask.squeeze(0)

        labels = input_ids.clone()
        prompt_len = 0
        if prompt_text:
            prompt_inputs = processor(
                text=prompt_text,
                images=images,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            prompt_len = min(prompt_inputs["input_ids"].shape[1], input_ids.size(0))
            labels[:prompt_len] = IGNORE_INDEX

        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if prompt_len:
            prompt_mask[:prompt_len] = True
        prompt_mask &= attention_mask.bool()

        result: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "prompt_mask": prompt_mask,
        }

        if "pixel_values" in full_inputs:
            pv = full_inputs["pixel_values"]
            while pv.dim() > 3 and pv.shape[0] == 1:
                pv = pv.squeeze(0)
            result["pixel_values"] = pv
        if "image_sizes" in full_inputs:
            result["image_sizes"] = full_inputs["image_sizes"]
        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = full_inputs["image_grid_thw"]
        return result

    def collate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            return {}

        max_len = max(s["input_ids"].size(0) for s in samples)
        batch_size = len(samples)

        batch_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        batch_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        batch_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        batch_prompt = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for i, sample in enumerate(samples):
            seq_len = sample["input_ids"].size(0)
            batch_ids[i, :seq_len] = sample["input_ids"]
            batch_labels[i, :seq_len] = sample["labels"]
            batch_mask[i, :seq_len] = sample["attention_mask"]
            batch_prompt[i, :seq_len] = sample["prompt_mask"]

        batch: Dict[str, Any] = {
            "input_ids": batch_ids,
            "labels": batch_labels,
            "attention_mask": batch_mask,
            "prompt_mask": batch_prompt,
        }

        if "pixel_values" in samples[0]:
            pvs = [s["pixel_values"] for s in samples]
            try:
                batch["pixel_values"] = torch.stack(pvs)
            except RuntimeError:
                batch["pixel_values"] = torch.cat(pvs, dim=0)

        for key in ("image_sizes", "image_grid_thw"):
            if key in samples[0]:
                vals = [s[key] for s in samples]
                if isinstance(vals[0], torch.Tensor):
                    try:
                        batch[key] = torch.cat(vals, dim=0)
                    except RuntimeError:
                        batch[key] = vals
                else:
                    batch[key] = vals

        return batch
