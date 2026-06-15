"""L2T: supervise both instruction and response sequences."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from vlmintune.data.types import CanonicalSample

from vlmintune.training.methods.lora import LoRAMethod
from vlmintune.training.methods.base import TrainingMethod


def extract_instruction_texts(sample: CanonicalSample) -> List[str]:
    question = sample.question.strip()
    return [question] if question else []


def tokenized_text_variants(
    tokenizer: Any,
    text: str,
    max_length: int,
) -> List[List[int]]:
    variants: List[List[int]] = []
    seen = set()
    for candidate in (text, "\n" + text, " " + text):
        tokenized = tokenizer(
            candidate,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        text_ids = tokenized["input_ids"]
        if text_ids.dim() > 1:
            text_ids = text_ids.squeeze(0)
        ids = [int(token_id) for token_id in text_ids.tolist()]
        key = tuple(ids)
        if ids and key not in seen:
            variants.append(ids)
            seen.add(key)
    return variants


def build_instruction_supervision_mask(
    processor: Any,
    sample: CanonicalSample,
    input_ids: torch.Tensor,
    prompt_len: int,
    max_length: int,
) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if prompt_len <= 0:
        return mask

    instruction_texts = extract_instruction_texts(sample)
    prompt_ids = input_ids[:prompt_len].tolist()
    tokenizer = getattr(processor, "tokenizer", processor)
    cursor = 0
    for text in instruction_texts:
        start_idx = -1
        matched_ids: List[int] = []
        for text_ids_list in tokenized_text_variants(tokenizer, text, max_length):
            limit = len(prompt_ids) - len(text_ids_list) + 1
            for idx in range(max(0, cursor), max(0, limit)):
                if prompt_ids[idx:idx + len(text_ids_list)] == text_ids_list:
                    start_idx = idx
                    matched_ids = text_ids_list
                    break
            if start_idx >= 0:
                break
        if start_idx < 0:
            continue

        end_idx = start_idx + len(matched_ids)
        mask[start_idx:end_idx] = True
        cursor = end_idx
    return mask


def decode_token_ids(processor: Any, token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "decode"):
        return str(
            tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    return " ".join(str(token_id) for token_id in token_ids)


def build_instruction_debug_preview(
    processor: Any,
    input_ids: torch.Tensor,
    instruction_supervision_mask: torch.Tensor,
) -> List[Dict[str, Any]]:
    previews: List[Dict[str, Any]] = []
    span_start: Optional[int] = None
    mask_list = instruction_supervision_mask.tolist()
    token_ids = input_ids.tolist()

    for idx, is_selected in enumerate(mask_list + [False]):
        if is_selected and span_start is None:
            span_start = idx
        if not is_selected and span_start is not None:
            span_token_ids = token_ids[span_start:idx]
            previews.append(
                {
                    "start": span_start,
                    "end": idx,
                    "token_count": len(span_token_ids),
                    "text": decode_token_ids(processor, span_token_ids),
                }
            )
            span_start = None

    return previews


class L2TMethod(TrainingMethod):
    name = "l2t"
    display_name = "L2T (Zhou et al. 2025)"

    def __init__(self):
        self.base = LoRAMethod()
        self.last_config: Dict[str, Any] = {}

    def default_config(self):
        defaults = self.base.default_config()
        defaults.pop("train_layer_range", None)
        return defaults

    def requires_quantization(self, config=None):
        return self.base.requires_quantization(config)

    def prepare_model_impl(self, model, processor, config, model_spec):
        self.last_config = dict(config)
        return self.base.prepare_model(model, processor, config, model_spec=model_spec)

    def preprocess_labels(self, input_ids, labels, batch_meta=None):
        if not batch_meta:
            return labels
        instruction_mask = batch_meta.get("instruction_supervision_mask")
        if instruction_mask is None:
            return labels
        mask = instruction_mask.bool()
        attention_mask = batch_meta.get("attention_mask")
        if attention_mask is not None:
            mask &= attention_mask.bool()
        if not mask.any():
            return labels
        updated = labels.clone()
        updated[mask] = input_ids[mask]
        return updated

    def compute_loss(self, model, batch, outputs):
        return self.base.compute_loss(model, batch, outputs)

    def get_trainable_params(self, model):
        return self.base.get_trainable_params(model)

    def save_checkpoint(self, model, processor, path, metadata):
        metadata = {**metadata, "ft_method": self.name, "config": self.last_config}
        self.base.save_checkpoint(model, processor, path, metadata)

    def load_for_inference(self, path, model_name, **kwargs):
        return self.base.load_for_inference(path, model_name, **kwargs)
