"""Training configuration: YAML loading, validation, and conversion.

Loads a YAML config file and produces the single-stage dict structure expected
by ``vlmintune.training.__main__.main()``.

Usage::

    from vlmintune.config.training_config import load_config, config_to_trainer_dict

    cfg = load_config("experiment_setup/my_experiment/train_config.yaml")
    trainer_dict = config_to_trainer_dict(cfg)
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml

from vlmintune.config.model_layouts import list_model_layouts
from vlmintune.training.methods.registry import (
    get_training_method_defaults,
    list_training_methods,
)

_LORA_FAMILY_METHODS = {"lora", "qlora", "dora", "l2t"}


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    model_path: str = ""


@dataclass
class TrainingParams:
    ft_method: str = "qlora"
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    save_steps: int = 500
    output_dir: str = "output"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str = ""
    base_dir: str = "experiments"
    setup_dir: str = ""


@dataclass
class DataConfig:
    data_path: str = ""
    split: str = "train"
    image_root: str = ""
    max_samples: int = 0


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingParams = field(default_factory=TrainingParams)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)


# ── YAML loading ─────────────────────────────────────────────────────


def load_config_dict(raw: Dict[str, Any]) -> TrainingConfig:
    """Load and validate a training config from an in-memory mapping."""
    raw = raw or {}

    raw_model = raw.get("model", {})
    raw_training = raw.get("training", {})
    raw_experiment = raw.get("experiment", {})
    raw_data = raw.get("data", {})

    cfg = TrainingConfig(
        model=ModelConfig(
            model_path=str(raw_model.get("model_path", "")),
        ),
        training=TrainingParams(
            ft_method=str(raw_training.get("ft_method", "qlora")),
            num_epochs=int(raw_training.get("num_epochs", 3)),
            per_device_batch_size=int(raw_training.get("per_device_batch_size", 4)),
            gradient_accumulation_steps=int(raw_training.get("gradient_accumulation_steps", 4)),
            max_length=int(raw_training.get("max_length", 2048)),
            dataloader_num_workers=int(raw_training.get("dataloader_num_workers", 0)),
            dataloader_pin_memory=bool(raw_training.get("dataloader_pin_memory", False)),
            dataloader_persistent_workers=bool(raw_training.get("dataloader_persistent_workers", False)),
            learning_rate=float(raw_training.get("learning_rate", 2e-4)),
            warmup_ratio=float(raw_training.get("warmup_ratio", 0.03)),
            weight_decay=float(raw_training.get("weight_decay", 0.0)),
            max_grad_norm=float(raw_training.get("max_grad_norm", 1.0)),
            save_steps=int(raw_training.get("save_steps", 500)),
            output_dir=str(raw_training.get("output_dir", "output")),
            params=dict(raw_training.get("params", {})),
        ),
        experiment=ExperimentConfig(
            name=str(raw_experiment.get("name", "")).strip(),
            base_dir=str(raw_experiment.get("base_dir", "experiments")).strip() or "experiments",
            setup_dir=str(raw_experiment.get("setup_dir", "")).strip(),
        ),
        data=DataConfig(
            data_path=str(raw_data.get("data_path", "")),
            split=str(raw_data.get("split", "train")),
            image_root=str(raw_data.get("image_root", "")),
            max_samples=int(raw_data.get("max_samples", 0)),
        ),
    )

    validate(cfg)
    merge_method_defaults(cfg)
    return cfg


def load_config(path: str) -> TrainingConfig:
    """Load and validate a YAML training config file.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    TrainingConfig
        Validated config with method defaults merged into ``training.params``.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required fields are missing or invalid.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = load_config_dict(raw)
    if not cfg.experiment.setup_dir:
        cfg.experiment.setup_dir = infer_setup_dir_from_config_path(path)
    return cfg


def infer_setup_dir_from_config_path(path: str) -> str:
    config_dir = os.path.dirname(os.path.normpath(path))
    if not config_dir:
        return ""

    cwd = os.path.normpath(os.getcwd())
    abs_config_dir = os.path.normpath(os.path.abspath(config_dir))
    try:
        rel = os.path.relpath(abs_config_dir, cwd)
    except ValueError:
        return config_dir
    if rel == ".":
        return config_dir
    if not rel.startswith(".."):
        return rel
    return config_dir


def validate(cfg: TrainingConfig) -> None:
    """Validate config fields; raise ValueError with all issues at once."""
    errors: List[str] = []

    if not cfg.model.model_path:
        errors.append("model.model_path: required field is empty")

    if not cfg.data.data_path:
        errors.append("data.data_path: required field is empty")

    if not cfg.experiment.name:
        errors.append("experiment.name: required field is empty")

    available = list_training_methods()
    if available and cfg.training.ft_method not in available:
        errors.append(
            f"training.ft_method: '{cfg.training.ft_method}' is not registered. "
            f"Available: {available}"
        )

    method_name = cfg.training.ft_method
    method_params = cfg.training.params
    requires_targets = method_name in _LORA_FAMILY_METHODS
    if requires_targets and not method_params.get("target_modules"):
        errors.append(
            "training.params.target_modules: required non-empty list for "
            f"method '{method_name}'"
        )

    if method_name == "freeze" and not str(method_params.get("model_layout", "")).strip():
        errors.append(
            "training.params.model_layout: required non-empty string for method "
            f"'{method_name}'. Available: {list_model_layouts()}"
        )

    if errors:
        msg = "Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)


def merge_method_defaults(cfg: TrainingConfig) -> None:
    """Merge method default params into cfg.training.params."""
    defaults = get_training_method_defaults(cfg.training.ft_method)
    unknown = set(cfg.training.params) - set(defaults)
    if unknown:
        warnings.warn(
            f"Unknown params for method '{cfg.training.ft_method}': {unknown}. "
            f"Known params: {set(defaults)}",
            stacklevel=3,
        )
    cfg.training.params = {**defaults, **cfg.training.params}


# ── Conversion ───────────────────────────────────────────────────────

def config_to_trainer_dict(cfg: TrainingConfig) -> dict:
    """Convert TrainingConfig to the trainer dict format expected by __main__.py."""
    data_config = {
        "data_path": cfg.data.data_path,
        "split": cfg.data.split,
        "image_root": cfg.data.image_root,
    }
    if cfg.data.max_samples:
        data_config["max_samples"] = cfg.data.max_samples

    return {
        "model": {
            "model_path": cfg.model.model_path,
        },
        "experiment": {
            "name": cfg.experiment.name,
            "base_dir": cfg.experiment.base_dir,
            "setup_dir": cfg.experiment.setup_dir,
        },
        "data": data_config,
        "training_method": cfg.training.ft_method,
        "method_params": cfg.training.params,
        "training": {
            "num_epochs": cfg.training.num_epochs,
            "per_device_batch_size": cfg.training.per_device_batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "max_length": cfg.training.max_length,
            "dataloader_num_workers": cfg.training.dataloader_num_workers,
            "dataloader_pin_memory": cfg.training.dataloader_pin_memory,
            "dataloader_persistent_workers": cfg.training.dataloader_persistent_workers,
            "learning_rate": cfg.training.learning_rate,
            "warmup_ratio": cfg.training.warmup_ratio,
            "weight_decay": cfg.training.weight_decay,
            "max_grad_norm": cfg.training.max_grad_norm,
            "save_steps": cfg.training.save_steps,
            "output_dir": cfg.training.output_dir,
        },
    }
