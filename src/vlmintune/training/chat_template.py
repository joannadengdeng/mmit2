"""Chat-template preprocessor for single-sample tokenization and batch collation."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from PIL import Image

from vlmintune.data.datasets.base import (
    build_full_inputs,
    build_prompt_inputs,
    load_sample_image,
)
from vlmintune.data.types import CanonicalSample

IGNORE_INDEX = -100

class ChatTemplatePreprocessor:
    def __init__(
        self,
        enable_instruction_supervision: bool = False,
        enable_mores_intervention: bool = False,
        append_eos_to_training_answer: bool = False,
    ) -> None:
        self.enable_instruction_supervision = enable_instruction_supervision
        self.enable_mores_intervention = enable_mores_intervention
        self.append_eos_to_training_answer = append_eos_to_training_answer

    @staticmethod
    def _is_image_token_truncation_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            "Mismatch in `image` token count" in message
            and "truncation='max_length'" in message
        )

    @staticmethod
    def _resize_image(image: Image.Image, max_side: int) -> Image.Image:
        resized = image.copy()
        resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return resized.convert("RGB")

    def _tokenize_with_image_retries(
        self,
        *,
        processor: Any,
        sample: CanonicalSample,
        image: Any,
        max_length: int,
    ) -> tuple[str, Dict[str, Any], str, Dict[str, Any], int, Any]:
        candidates: list[tuple[int, Any]] = [(0, image)]
        if isinstance(image, Image.Image):
            seen_sizes = {image.size}
            for max_side in (896, 768, 640, 512, 384):
                if max(image.size) <= max_side:
                    continue
                resized = self._resize_image(image, max_side)
                if resized.size in seen_sizes:
                    continue
                seen_sizes.add(resized.size)
                candidates.append((len(candidates), resized))

        last_error: Exception | None = None
        for retry_index, candidate_image in candidates:
            try:
                full_text, full_inputs = build_full_inputs(
                    processor,
                    sample.question,
                    sample.train_answer,
                    candidate_image,
                    append_eos_if_missing=self.append_eos_to_training_answer,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                prompt_text, prompt_inputs = build_prompt_inputs(
                    processor,
                    sample.question,
                    candidate_image,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                return (
                    full_text,
                    full_inputs,
                    prompt_text,
                    prompt_inputs,
                    retry_index,
                    candidate_image,
                )
            except Exception as exc:
                if not self._is_image_token_truncation_error(exc):
                    raise
                last_error = exc

        assert last_error is not None
        raise last_error

    def tokenize(
        self,
        sample: CanonicalSample,
        processor: Any,
        model_config: Any,
        max_length: int = 2048,
    ) -> Dict[str, Any]:
        # `processor` must be a Hugging Face multimodal processor that supports
        # both `apply_chat_template(...)` and `processor(text=..., images=...)`.
        image = load_sample_image(sample)
        (
            full_text,
            full_inputs,
            prompt_text,
            prompt_inputs,
            image_retry_count,
            tokenized_image,
        ) = self._tokenize_with_image_retries(
            processor=processor,
            sample=sample,
            image=image,
            max_length=max_length,
        )

        prompt_preview = {
            "sample_id": sample.id,
            "has_image": image is not None,
            "message_count": 1 + int(bool(sample.train_answer.strip())),
            "full_text": full_text,
            "prompt_text": prompt_text,
            "image_tokenization_retry_count": image_retry_count,
        }
        if isinstance(image, Image.Image):
            prompt_preview["original_image_size"] = list(image.size)
        if isinstance(tokenized_image, Image.Image):
            prompt_preview["tokenized_image_size"] = list(tokenized_image.size)

        input_ids = full_inputs["input_ids"].squeeze(0)

        labels = input_ids.clone()
        prompt_len = min(prompt_inputs["input_ids"].shape[1], input_ids.size(0))
        labels[:prompt_len] = IGNORE_INDEX

        result: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_preview": prompt_preview,
        }

        if self.enable_mores_intervention:
            from vlmintune.training.methods.mores import build_mores_intervention_mask

            intervention_mask = build_mores_intervention_mask(
                model_config=model_config,
                input_ids=input_ids,
            )
            prompt_preview["intervention_mask"] = intervention_mask.tolist()
            result["intervention_mask"] = intervention_mask

        if self.enable_instruction_supervision:
            from vlmintune.training.methods.l2t import (
                build_instruction_debug_preview,
                build_instruction_supervision_mask,
                extract_instruction_texts,
            )

            instruction_supervision_mask = build_instruction_supervision_mask(
                processor=processor,
                sample=sample,
                input_ids=input_ids,
                prompt_len=prompt_len,
                max_length=max_length,
            )
            prompt_preview["instruction_texts"] = extract_instruction_texts(sample)
            prompt_preview["instruction_supervision_spans"] = build_instruction_debug_preview(
                processor=processor,
                input_ids=input_ids,
                instruction_supervision_mask=instruction_supervision_mask,
            )
            result["instruction_supervision_mask"] = instruction_supervision_mask

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

        for i, sample in enumerate(samples):
            seq_len = sample["input_ids"].size(0)
            batch_ids[i, :seq_len] = sample["input_ids"]
            batch_labels[i, :seq_len] = sample["labels"]
            batch_mask[i, :seq_len] = 1

        batch: Dict[str, Any] = {
            "input_ids": batch_ids,
            "labels": batch_labels,
            "attention_mask": batch_mask,
        }

        if all("instruction_supervision_mask" in sample for sample in samples):
            batch_instruction = torch.zeros(batch_size, max_len, dtype=torch.bool)
            for i, sample in enumerate(samples):
                seq_len = sample["instruction_supervision_mask"].size(0)
                batch_instruction[i, :seq_len] = sample["instruction_supervision_mask"]
            batch["instruction_supervision_mask"] = batch_instruction

        if all("intervention_mask" in sample for sample in samples):
            batch_intervention_mask = torch.zeros(
                (batch_size, max_len),
                dtype=torch.bool,
            )
            for sample_idx, sample in enumerate(samples):
                intervention_mask = sample["intervention_mask"]
                if intervention_mask.dim() != 1:
                    raise ValueError(
                        "MoReS expects per-sample intervention_mask to have shape [seq_len]."
                    )
                batch_intervention_mask[sample_idx, :intervention_mask.size(0)] = (
                    intervention_mask.to(dtype=torch.bool)
                )
            batch["intervention_mask"] = batch_intervention_mask

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


__all__ = [
    "ChatTemplatePreprocessor",
    "IGNORE_INDEX",
    "build_full_inputs",
    "build_prompt_inputs",
]
