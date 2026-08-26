"""Chat-template preprocessor for single-sample tokenization and batch collation."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

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
        method_cls,
        append_eos_to_training_answer: bool = False,
    ) -> None:
        self.method_cls = method_cls
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
        _, full_inputs = build_full_inputs(
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

        input_ids = full_inputs["input_ids"].squeeze(0)

        labels = input_ids.clone()
        prompt_len = min(prompt_inputs["input_ids"].shape[1], input_ids.size(0))
        labels[:prompt_len] = IGNORE_INDEX

        result: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
        }

        method_mask = self.method_cls.build_method_mask(
            sample=sample,
            processor=processor,
            model_config=model_config,
            input_ids=input_ids,
            prompt_text=prompt_text,
            prompt_len=prompt_len,
            image=image,
            max_length=max_length,
        )
        if method_mask is not None:
            result["method_mask"] = method_mask

        if "pixel_values" in full_inputs:
            result["pixel_values"] = full_inputs["pixel_values"]
        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = full_inputs["image_grid_thw"]
        if "mm_token_type_ids" in full_inputs:
            result["mm_token_type_ids"] = full_inputs["mm_token_type_ids"].squeeze(0)
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

        if "method_mask" in samples[0]:
            batch["method_mask"] = pad_sequence(
                [sample["method_mask"].bool() for sample in samples],
                batch_first=True,
                padding_value=False,
            )

        if "mm_token_type_ids" in samples[0]:
            batch["mm_token_type_ids"] = pad_sequence(
                [sample["mm_token_type_ids"] for sample in samples],
                batch_first=True,
                padding_value=0,
            )

        if "pixel_values" in samples[0]:
            batch["pixel_values"] = torch.cat(
                [sample["pixel_values"] for sample in samples],
                dim=0,
            )

        if "image_grid_thw" in samples[0]:
            batch["image_grid_thw"] = torch.cat(
                [sample["image_grid_thw"] for sample in samples],
                dim=0,
            )

        return batch


__all__ = [
    "ChatTemplatePreprocessor",
    "IGNORE_INDEX",
    "build_full_inputs",
    "build_prompt_inputs",
]
