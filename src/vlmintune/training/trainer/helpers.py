"""Shared helper functions for the trainer package."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR

from vlmintune.data.hf_datasets import HFDatasetsAdapter

DEBUG_EXAMPLE_LIMIT = 5
IGNORE_INDEX = -100


# Event helpers

def emit(event_type: str, data: dict) -> None:
    """Print a JSON event line to stdout."""
    print(json.dumps({"type": event_type, "data": data}), flush=True)


# Training runtime helpers

def cosine_schedule(optimizer, num_warmup: int, num_total: int):
    """Cosine LR schedule with linear warmup."""

    def lr_lambda(step):
        if step < num_warmup:
            return step / max(1, num_warmup)
        progress = (step - num_warmup) / max(1, num_total - num_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            out[key] = [tensor.to(device) for tensor in value]
        else:
            out[key] = value
    return out


def shape_str(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return "x".join(str(dim) for dim in value.shape)
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        first = value[0]
        return f"list[{len(value)}]:" + "x".join(str(dim) for dim in first.shape)
    return type(value).__name__


def describe_batch(batch: Dict[str, Any]) -> str:
    parts = [
        f"input_ids={shape_str(batch['input_ids'])}",
        f"labels={shape_str(batch['labels'])}",
        f"attention_mask={shape_str(batch['attention_mask'])}",
    ]
    for key in ("pixel_values", "image_grid_thw", "image_sizes"):
        if key in batch:
            parts.append(f"{key}={shape_str(batch[key])}")
    return "First batch shapes: " + ", ".join(parts)


def decode_debug_tokens(processor: Any, token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    if hasattr(processor, "decode"):
        return str(
            processor.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
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


def build_supervised_span_preview(
    processor: Any,
    input_ids: torch.Tensor,
    supervised_mask: torch.Tensor,
) -> List[Dict[str, Any]]:
    token_ids = input_ids.detach().cpu().tolist()
    mask_list = supervised_mask.detach().cpu().tolist()
    previews: List[Dict[str, Any]] = []
    span_start: Optional[int] = None

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
                    "text": decode_debug_tokens(processor, span_token_ids),
                }
            )
            span_start = None

    return previews


def build_label_supervision_debug(
    processor: Any,
    input_ids: torch.Tensor,
    labels_before: torch.Tensor,
    labels_after: torch.Tensor,
    instruction_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    restored_mask = (labels_before == IGNORE_INDEX) & (labels_after != IGNORE_INDEX)
    debug: Dict[str, Any] = {
        "kind": "label_supervision",
        "supervised_tokens_before": int((labels_before != IGNORE_INDEX).sum().item()),
        "supervised_tokens_after": int((labels_after != IGNORE_INDEX).sum().item()),
        "restored_tokens_into_loss": int(restored_mask.sum().item()),
        "first_sample_supervised_spans_before": build_supervised_span_preview(
            processor,
            input_ids[0],
            labels_before[0] != IGNORE_INDEX,
        ),
        "first_sample_supervised_spans_after": build_supervised_span_preview(
            processor,
            input_ids[0],
            labels_after[0] != IGNORE_INDEX,
        ),
        "first_sample_restored_spans": build_supervised_span_preview(
            processor,
            input_ids[0],
            restored_mask[0],
        ),
    }
    if instruction_mask is not None:
        debug["instruction_mask_tokens"] = int(instruction_mask.sum().item())
        debug["first_sample_instruction_spans"] = build_supervised_span_preview(
            processor,
            input_ids[0],
            instruction_mask[0],
        )
    return debug


# Dataset helpers

def build_dataset(config: Any):
    data_cfg = dict(config.data_config)
    max_samples = int(data_cfg.pop("max_samples", 0) or 0)
    data_cfg.pop("image_root", None)

    if "dataset_name" not in data_cfg:
        if "data_path" in data_cfg:
            data_cfg["dataset_name"] = data_cfg.pop("data_path")
        elif "dataset" in data_cfg:
            data_cfg["dataset_name"] = data_cfg.pop("dataset")

    adapter = HFDatasetsAdapter(
        max_samples=max_samples if max_samples > 0 else None,
        **data_cfg,
    )
    dataset_len = len(adapter)
    if dataset_len < 0:
        raise ValueError(
            "Could not determine dataset length for training. "
            "Please provide a dataset/split with a known size or set max_samples."
        )

    emit("log", {"message": f"{dataset_len} samples", "level": "INFO"})
    emit(
        "log",
        {
            "message": (
                "Dataset resolved to "
                f"{adapter.dataset_name} split={adapter.split} "
                f"streaming={adapter.streaming} max_samples={adapter.max_samples or 'full'}"
            ),
            "level": "INFO",
        },
    )
    return adapter, dataset_len


# Debug helpers

def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def sample_debug_record(sample) -> Dict[str, Any]:
    record = asdict(sample)
    metadata = dict(record.get("metadata") or {})
    metadata.pop("_pil_image", None)
    record["metadata"] = json_safe(metadata)
    return record


class DebugRecorder:
    """Capture a tiny debug snapshot of the training input pipeline."""

    def __init__(self) -> None:
        self.samples = []
        self.prompts = []
        self.total_skipped = 0
        self.skip_examples = []

    def record_sample(self, sample) -> None:
        if len(self.samples) < DEBUG_EXAMPLE_LIMIT:
            self.samples.append(sample_debug_record(sample))

    def record_prompt(self, preview: Dict[str, Any]) -> None:
        if len(self.prompts) < DEBUG_EXAMPLE_LIMIT:
            self.prompts.append(json_safe(preview))

    def record_skip(self, sample_id: Any, exc: Exception) -> None:
        self.total_skipped += 1
        if len(self.skip_examples) < DEBUG_EXAMPLE_LIMIT:
            self.skip_examples.append({
                "sample_id": str(sample_id),
                "error": str(exc),
            })

    def emit_run_log(self) -> None:
        if self.samples:
            emit(
                "debug",
                {
                    "kind": "canonical_samples",
                    "limit": DEBUG_EXAMPLE_LIMIT,
                    "examples": self.samples,
                },
            )
        if self.prompts:
            emit(
                "debug",
                {
                    "kind": "rendered_prompts",
                    "limit": DEBUG_EXAMPLE_LIMIT,
                    "examples": self.prompts,
                },
            )
        emit(
            "debug",
            {
                "kind": "skip_summary",
                "total_skipped": self.total_skipped,
                "first_errors": self.skip_examples,
            },
        )


def build_skip_logger(debug_recorder: DebugRecorder) -> Callable[[Any, Exception], None]:
    def log_skip(sample_id: Any, exc: Exception) -> None:
        debug_recorder.record_skip(sample_id, exc)
        emit(
            "log",
            {"message": f"Skipping sample {sample_id}: {exc}", "level": "WARNING"},
        )

    return log_skip
