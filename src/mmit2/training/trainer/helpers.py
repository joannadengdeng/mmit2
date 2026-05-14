"""Shared helper functions for the trainer package."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from typing import Any, Callable, Dict

import torch
from torch.optim.lr_scheduler import LambdaLR

from mmit2.data.adapters.hf_datasets import HFDatasetsAdapter


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


# Dataset helpers

def build_dataset(config: Any):
    data_cfg = dict(config.data_config)
    adapter_name = data_cfg.pop("adapter", "hf_datasets")
    max_samples = int(data_cfg.pop("max_samples", 0) or 0)
    data_cfg.pop("image_root", None)
    if adapter_name != "hf_datasets":
        raise ValueError(
            f"Unsupported data adapter '{adapter_name}'. Only 'hf_datasets' is supported."
        )

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


def resolve_debug_dir(config: Any, tracker: Any) -> str:
    if tracker is not None:
        return os.path.join(tracker.meta.exp_dir, "debug")
    return os.path.join(config.output_dir, "debug")


# Debug artifact helpers

def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sample_debug_record(sample) -> Dict[str, Any]:
    record = asdict(sample)
    metadata = dict(record.get("metadata") or {})
    metadata.pop("_pil_image", None)
    record["metadata"] = json_safe(metadata)
    return record


class DebugRecorder:
    """Capture a tiny debug snapshot of the training input pipeline."""

    def __init__(self, limit: int = 5) -> None:
        self.limit = limit
        self.samples = []
        self.prompts = []
        self.total_skipped = 0
        self.skip_examples = []

    def record_sample(self, sample) -> None:
        if len(self.samples) < self.limit:
            self.samples.append(sample_debug_record(sample))

    def record_prompt(self, preview: Dict[str, Any]) -> None:
        if len(self.prompts) < self.limit:
            self.prompts.append(json_safe(preview))

    def record_skip(self, sample_id: Any, exc: Exception) -> None:
        self.total_skipped += 1
        if len(self.skip_examples) < self.limit:
            self.skip_examples.append({
                "sample_id": str(sample_id),
                "error": str(exc),
            })

    def flush(self, debug_dir: str) -> None:
        os.makedirs(debug_dir, exist_ok=True)
        if self.samples:
            write_json(os.path.join(debug_dir, "first_5_canonical_samples.json"), self.samples)
        if self.prompts:
            write_json(os.path.join(debug_dir, "first_5_rendered_prompts.json"), self.prompts)
        write_json(
            os.path.join(debug_dir, "skip_summary.json"),
            {
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
