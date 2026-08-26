"""Strict public training configuration for the initial release."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict

import yaml

from vlmintune.models.registry import get_model_spec, list_model_names
from vlmintune.training.methods.registry import (
    get_training_method_cls,
    list_training_methods,
)


PUBLIC_CONFIG_FIELDS = {
    "model",
    "dataset",
    "method",
    "epochs",
    "learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "max_length",
    "max_samples",
    "seed",
    "output_dir",
}


@dataclass
class TrainingConfig:
    model: str = ""
    dataset: str = ""
    method: str = ""
    epochs: int = 1
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    max_samples: int = 0
    seed: int = 42
    output_dir: str = "output"


def load_config_dict(raw: Dict[str, Any]) -> TrainingConfig:
    """Load one strict flat config mapping."""

    raw = raw or {}
    unknown = set(raw) - PUBLIC_CONFIG_FIELDS
    if unknown:
        raise ValueError(f"Unknown training config fields: {sorted(unknown)}")

    config = TrainingConfig(
        model=str(raw.get("model", "")).strip(),
        dataset=str(raw.get("dataset", "")).strip(),
        method=str(raw.get("method", "")).strip(),
        epochs=int(raw.get("epochs", 1)),
        learning_rate=float(raw.get("learning_rate", 2e-4)),
        batch_size=int(raw.get("batch_size", 4)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 4)),
        max_length=int(raw.get("max_length", 2048)),
        max_samples=int(raw.get("max_samples", 0)),
        seed=int(raw.get("seed", 42)),
        output_dir=str(raw.get("output_dir", "output")).strip(),
    )
    validate(config)
    return config


def load_config(path: str) -> TrainingConfig:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return load_config_dict(yaml.safe_load(file) or {})


def validate(config: TrainingConfig) -> None:
    errors = []
    if not config.model:
        errors.append("model: required field is empty")
    else:
        try:
            get_model_spec(config.model)
        except KeyError:
            errors.append(
                f"model: unknown built-in model '{config.model}'. Available: {list_model_names()}"
            )

    if not config.dataset:
        errors.append("dataset: required field is empty")

    available_methods = list_training_methods()
    if not config.method:
        errors.append("method: required field is empty")
    elif config.method not in available_methods:
        errors.append(
            f"method: '{config.method}' is not registered. Available: {available_methods}"
        )
    else:
        supported_models = get_training_method_cls(config.method).supported_model_names
        if supported_models is not None and config.model not in supported_models:
            supported = ", ".join(repr(model) for model in supported_models)
            errors.append(
                f"method: '{config.method}' only supports model(s): {supported}"
            )

    if not config.output_dir:
        errors.append("output_dir: required field is empty")

    if config.max_samples < 0:
        errors.append("max_samples: must be >= 0 (0 means the full training split)")

    if errors:
        raise ValueError("Config validation errors:\n" + "\n".join(f"  - {error}" for error in errors))


def config_to_trainer_dict(config: TrainingConfig) -> dict:
    """Return the strict flat representation consumed by the CLI."""

    return asdict(config)
