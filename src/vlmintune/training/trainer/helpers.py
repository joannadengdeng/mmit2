"""Shared helper functions for the trainer package."""
from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict

import torch
from torch.optim.lr_scheduler import LambdaLR

from vlmintune.data.hf_datasets import HFDatasetsAdapter


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


# Dataset helpers

def build_dataset(config: Any):
    dataset_name = str(config.data_config["dataset_name"])
    max_samples = int(config.data_config.get("max_samples", 0) or 0)
    sample_seed = int(config.data_config.get("sample_seed", config.seed))
    adapter = HFDatasetsAdapter(
        dataset_name=dataset_name,
        usage="train",
        max_samples=max_samples or None,
        sample_seed=sample_seed,
    )

    dataset_len = len(adapter)
    if dataset_len < 0:
        raise ValueError(
            "Could not determine dataset length for training. "
            "The initial release requires a dataset with a known training size."
        )

    emit("log", {"message": f"{dataset_len} samples", "level": "INFO"})
    emit(
        "log",
        {
            "message": (
                "Dataset resolved to "
                f"{adapter.dataset_name} split={adapter.split} "
                f"streaming={adapter.streaming} "
                f"max_samples={max_samples or 'full'} sample_seed={sample_seed}"
            ),
            "level": "INFO",
        },
    )
    return adapter, dataset_len


# Preprocessing coverage helpers


class PreprocessingCoverage:
    """Track aggregate preprocessing coverage without retaining sample data."""

    def __init__(self) -> None:
        self.total_skipped = 0

    def mark_skipped(self) -> None:
        self.total_skipped += 1

    def emit_summary(self) -> None:
        emit(
            "data_summary",
            {
                "kind": "preprocessing_coverage",
                "total_skipped": self.total_skipped,
            },
        )


def build_skip_logger(coverage: PreprocessingCoverage) -> Callable[[Any, Exception], None]:
    def log_skip(sample_id: Any, exc: Exception) -> None:
        coverage.mark_skipped()
        emit(
            "log",
            {"message": f"Skipping sample {sample_id}: {exc}", "level": "WARNING"},
        )

    return log_skip
