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
        enable_reft_intervention: bool = False,
        append_eos_to_training_answer: bool = False,
    ) -> None:
        self.enable_instruction_supervision = enable_instruction_supervision
        self.enable_mores_intervention = enable_mores_intervention
        self.enable_reft_intervention = enable_reft_intervention
        self.append_eos_to_training_answer = append_eos_to_training_answer

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
        full_text, full_inputs = build_full_inputs(
            processor,
            sample.question,
            sample.train_answer,
            image,
            append_eos_if_missing=self.append_eos_to_training_answer,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        prompt_text, prompt_inputs = build_prompt_inputs(
            processor,
            sample.question,
            image,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        prompt_preview = {
            "sample_id": sample.id,
            "has_image": image is not None,
            "message_count": 1 + int(bool(sample.train_answer.strip())),
            "full_text": full_text,
            "prompt_text": prompt_text,
        }
        if isinstance(image, Image.Image):
            prompt_preview["original_image_size"] = list(image.size)

        input_ids = full_inputs["input_ids"].squeeze(0)

        labels = input_ids.clone()
        prompt_len = min(prompt_inputs["input_ids"].shape[1], input_ids.size(0))
        labels[:prompt_len] = IGNORE_INDEX
        if self.enable_instruction_supervision and not (labels != IGNORE_INDEX).any():
            raise ValueError(
                "L2T requires answer tokens after the prompt; increase max_length "
                "or remove the invalid sample."
            )

        result: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_preview": prompt_preview,
        }

        if self.enable_reft_intervention:
            from vlmintune.training.methods.reft import build_reft_position_mask

            reft_intervention_mask = build_reft_position_mask(
                input_ids.unsqueeze(0),
                torch.tensor([prompt_len], device=input_ids.device),
            ).squeeze(0)
            prompt_preview["reft_intervention_mask"] = reft_intervention_mask.tolist()
            result["reft_intervention_mask"] = reft_intervention_mask

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
            prompt_preview["removed_task_templates"] = list(
                sample.l2t_removed_task_templates
            )
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
        if "mm_token_type_ids" in full_inputs:
            mm_token_type_ids = full_inputs["mm_token_type_ids"]
            if mm_token_type_ids.dim() > 1 and mm_token_type_ids.shape[0] == 1:
                mm_token_type_ids = mm_token_type_ids.squeeze(0)
            result["mm_token_type_ids"] = mm_token_type_ids
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

        if all("reft_intervention_mask" in sample for sample in samples):
            batch_reft_intervention_mask = torch.zeros(
                (batch_size, max_len),
                dtype=torch.bool,
            )
            for sample_idx, sample in enumerate(samples):
                reft_intervention_mask = sample["reft_intervention_mask"]
                batch_reft_intervention_mask[
                    sample_idx, :reft_intervention_mask.size(0)
                ] = reft_intervention_mask.to(dtype=torch.bool)
            batch["reft_intervention_mask"] = batch_reft_intervention_mask

        if all("mm_token_type_ids" in sample for sample in samples):
            batch_mm_token_type_ids = torch.zeros(
                (batch_size, max_len),
                dtype=samples[0]["mm_token_type_ids"].dtype,
            )
            for sample_idx, sample in enumerate(samples):
                token_type_ids = sample["mm_token_type_ids"]
                batch_mm_token_type_ids[
                    sample_idx, :token_type_ids.size(0)
                ] = token_type_ids
            batch["mm_token_type_ids"] = batch_mm_token_type_ids

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
