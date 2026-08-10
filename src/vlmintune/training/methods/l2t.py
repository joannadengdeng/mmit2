"""L2T: learn from informative instruction tokens and response tokens."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch

from vlmintune.data.types import CanonicalSample
from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.base import TrainingMethod
from vlmintune.training.methods.base import load_processor, load_vlm
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss


L2T_CHECKPOINT_NAME = "l2t_tuned.pt"
_CROSS_ENTROPY_LOSS = CrossEntropyLoss()


def extract_instruction_texts(sample: CanonicalSample) -> List[str]:
    return [
        text
        for raw_text in sample.l2t_instruction_texts
        if (text := str(raw_text).strip())
    ]


def tokenized_text_variants(
    tokenizer: Any,
    text: str,
    max_length: int,
) -> List[List[int]]:
    variants: List[List[int]] = []
    seen = set()
    # Token boundaries depend on both sides of a fragment.  For example, a
    # ScienceQA question is rendered as ``Question: <text>\nOptions:`` by the
    # chat prompt.  Qwen therefore tokenizes its first word with a leading
    # space and may merge the final punctuation with the trailing newline.
    # Keep the alignment strict, but cover the whitespace contexts used by
    # the built-in prompt renderers.
    for candidate in (
        text,
        "\n" + text,
        " " + text,
        text + "\n",
        "\n" + text + "\n",
        " " + text + "\n",
    ):
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


def find_token_span(
    sequence: List[int],
    candidates: List[List[int]],
    start_at: int = 0,
) -> Optional[tuple[int, int]]:
    search_start = max(0, start_at)
    earliest: Optional[tuple[int, int]] = None
    for candidate in candidates:
        if not candidate:
            continue
        last_start = len(sequence) - len(candidate)
        for idx in range(search_start, last_start + 1):
            if sequence[idx:idx + len(candidate)] == candidate:
                span = (idx, idx + len(candidate))
                if earliest is None or span[0] < earliest[0]:
                    earliest = span
                break
    return earliest


def build_instruction_supervision_mask(
    processor: Any,
    sample: CanonicalSample,
    input_ids: torch.Tensor,
    prompt_len: int,
    max_length: int,
) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if prompt_len <= 0:
        raise ValueError(f"L2T sample {sample.id!r} has no prompt tokens.")

    instruction_texts = extract_instruction_texts(sample)
    if not instruction_texts:
        raise ValueError(
            f"L2T sample {sample.id!r} has no dataset-defined instruction supervision text."
        )
    prompt_ids = input_ids[:prompt_len].tolist()
    tokenizer = getattr(processor, "tokenizer", processor)
    cursor = 0
    for text in instruction_texts:
        text_variants = tokenized_text_variants(tokenizer, text, max_length)
        span = find_token_span(prompt_ids, text_variants, start_at=cursor)
        if span is None:
            raise ValueError(
                f"L2T could not align instruction text for sample {sample.id!r}: {text!r}"
            )

        start_idx, end_idx = span
        mask[start_idx:end_idx] = True
        cursor = end_idx
    if not mask.any():
        raise ValueError(f"L2T sample {sample.id!r} produced an empty instruction mask.")
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
    display_name = "L2T (full-SFT v1)"

    @staticmethod
    def trainable_modules(model, model_spec):
        if model_spec.name == "qwen25vl_3b_instruct":
            return (
                model.model.language_model,
                model.lm_head,
                model.model.visual.merger,
            )
        if model_spec.name == "llava15_7b":
            return (
                model.model.language_model,
                model.lm_head,
                model.model.multi_modal_projector,
            )
        raise ValueError(
            "L2T full-SFT v1 supports only qwen25vl_3b_instruct and llava15_7b."
        )

    def prepare_model_impl(self, model, processor, model_spec):
        del processor
        model.requires_grad_(False)
        for module in self.trainable_modules(model, model_spec):
            module.requires_grad_(True)

        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        info = (
            "L2T full-SFT v1: bf16 base; language model + lm_head + visual "
            f"projection trainable; vision encoder frozen; trainable={trainable:,}/{total:,} "
            f"({100 * trainable / total:.4f}%)"
        )
        return model, info

    def preprocess_labels(self, input_ids, labels, batch_meta=None):
        if not batch_meta:
            raise ValueError("L2T requires instruction supervision batch metadata.")
        has_answer = (labels != -100).flatten(start_dim=1).any(dim=1)
        if not has_answer.all():
            raise ValueError(
                "L2T requires at least one supervised answer token in every sample; "
                "the answer may have been truncated by max_length."
            )
        instruction_mask = batch_meta.get("instruction_supervision_mask")
        if instruction_mask is None:
            raise ValueError("L2T requires an instruction_supervision_mask.")
        mask = instruction_mask.bool()
        attention_mask = batch_meta.get("attention_mask")
        if attention_mask is not None:
            mask &= attention_mask.bool()
        if not mask.any():
            raise ValueError("L2T instruction_supervision_mask contains no active tokens.")
        updated = labels.clone()
        updated[mask] = input_ids[mask]
        return updated

    def compute_loss(self, model, batch, outputs):
        return _CROSS_ENTROPY_LOSS.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        return [
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
            }
        ]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        state_dict = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        torch.save(state_dict, os.path.join(path, L2T_CHECKPOINT_NAME))
        processor.save_pretrained(path)
        metadata = {
            **metadata,
            "ft_method": self.name,
            "recipe": "l2t_full_sft_v1",
        }
        with open(os.path.join(path, "vlmintune_meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, model_name, **kwargs):
        del kwargs
        model_spec = get_model_spec(model_name)
        processor = load_processor(model_spec.hf_model_id)
        model = load_vlm(
            model_spec.hf_model_id,
            quantize_4bit=False,
            torch_dtype=torch.bfloat16,
        )
        model, _ = self.prepare_model(model, processor, model_spec=model_spec)
        state_dict = torch.load(
            os.path.join(path, L2T_CHECKPOINT_NAME),
            map_location="cpu",
            weights_only=True,
        )
        expected = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        if set(state_dict) != expected:
            raise ValueError(
                "L2T checkpoint does not match the fixed full-SFT v1 trainable state."
            )
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        checkpoint_name = os.path.basename(os.path.normpath(path))
        info = {
            "model_id": f"{model_spec.hf_model_id} (L2T full-SFT: {checkpoint_name})"
        }
        return model, processor, info


__all__ = [
    "L2T_CHECKPOINT_NAME",
    "L2TMethod",
    "build_instruction_debug_preview",
    "build_instruction_supervision_mask",
    "extract_instruction_texts",
    "find_token_span",
    "tokenized_text_variants",
]
