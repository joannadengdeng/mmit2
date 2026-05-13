"""Training configuration: YAML loading, validation, and conversion.

Loads a YAML config file and produces the single-stage dict structure expected
by ``mmit2.training.__main__.main()``.

Usage::

    from mmit2.config.training_config import load_config, config_to_trainer_dict

    cfg = load_config("configs/ssh_qlora.yaml")
    trainer_dict = config_to_trainer_dict(cfg)
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml

from mmit2.config.model_layouts import list_model_layouts
from mmit2.config.runtime import RuntimeConfig, SSHConfig
from mmit2.training.registry import (
    get_training_method_defaults,
    list_training_methods,
)

_LORA_FAMILY_METHODS = {"lora", "qlora", "dora"}


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
    base_dir: str = ""


@dataclass
class DataConfig:
    adapter: str = "hf_datasets"
    data_path: str = ""
    split: str = "train"
    image_root: str = ""
    max_samples: int = 0


@dataclass
class TrainingConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingParams = field(default_factory=TrainingParams)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)


# ── YAML loading ─────────────────────────────────────────────────────

def _parse_ssh(raw: dict) -> SSHConfig:
    if not raw:
        return SSHConfig()
    return SSHConfig(
        host=str(raw.get("host", "")),
        port=int(raw.get("port", 22)),
        username=str(raw.get("username", "")),
        key_path=str(raw.get("key_path", "")),
        password=str(raw.get("password", "")),
        conda_env=str(raw.get("conda_env", "")),
    )


def _raw_ssh_section(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw.get("ssh"), dict):
        return raw.get("ssh") or {}

    raw_runtime = raw.get("runtime", {})
    if isinstance(raw_runtime, dict) and isinstance(raw_runtime.get("ssh"), dict):
        return raw_runtime.get("ssh") or {}

    return {}


def load_runtime_config_dict(raw: Dict[str, Any]) -> RuntimeConfig:
    """Parse just the runtime section from an in-memory config mapping."""
    raw = raw or {}
    raw_runtime = raw.get("runtime", {})
    runtime_mode = "ssh"
    if isinstance(raw_runtime, dict):
        runtime_mode = str(raw_runtime.get("mode", "ssh") or "ssh")
    runtime = RuntimeConfig(
        mode=runtime_mode,
        ssh=_parse_ssh(_raw_ssh_section(raw)),
    )
    _validate_runtime(runtime)
    return runtime


def load_config_dict(raw: Dict[str, Any]) -> TrainingConfig:
    """Load and validate a training config from an in-memory mapping."""
    raw = raw or {}

    raw_model = raw.get("model", {})
    raw_training = raw.get("training", {})
    raw_experiment = raw.get("experiment", {})
    raw_data = raw.get("data", {})

    cfg = TrainingConfig(
        runtime=load_runtime_config_dict(raw),
        model=ModelConfig(
            model_path=str(raw_model.get("model_path", "")),
        ),
        training=TrainingParams(
            ft_method=str(raw_training.get("ft_method", "qlora")),
            num_epochs=int(raw_training.get("num_epochs", 3)),
            per_device_batch_size=int(raw_training.get("per_device_batch_size", 4)),
            gradient_accumulation_steps=int(raw_training.get("gradient_accumulation_steps", 4)),
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
            base_dir=str(raw_experiment.get("base_dir", "")).strip(),
        ),
        data=DataConfig(
            adapter=str(raw_data.get("adapter", "hf_datasets")),
            data_path=str(raw_data.get("data_path", "")),
            split=str(raw_data.get("split", "train")),
            image_root=str(raw_data.get("image_root", "")),
            max_samples=int(raw_data.get("max_samples", 0)),
        ),
    )

    _validate(cfg)
    _merge_method_defaults(cfg)
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
    return load_config_dict(raw)


def _validate_runtime(runtime: RuntimeConfig) -> None:
    if runtime.mode != "ssh":
        raise ValueError(
            f"runtime.mode: '{runtime.mode}' is not supported. Only 'ssh' mode is supported."
        )
    if not runtime.ssh.host:
        raise ValueError("runtime.ssh.host: required")
    if not runtime.ssh.username:
        raise ValueError("runtime.ssh.username: required")


def _validate(cfg: TrainingConfig) -> None:
    """Validate config fields; raise ValueError with all issues at once."""
    errors: List[str] = []

    if not cfg.model.model_path:
        errors.append("model.model_path: required field is empty")

    if not cfg.data.data_path:
        errors.append("data.data_path: required field is empty")

    if cfg.data.adapter != "hf_datasets":
        errors.append(
            f"data.adapter: '{cfg.data.adapter}' is not supported. "
            "Only 'hf_datasets' is supported."
        )

    try:
        _validate_runtime(cfg.runtime)
    except ValueError as exc:
        errors.append(str(exc))

    available = list_training_methods()
    if available and cfg.training.ft_method not in available:
        errors.append(
            f"training.ft_method: '{cfg.training.ft_method}' is not registered. "
            f"Available: {available}"
        )

    method_name = cfg.training.ft_method
    method_params = cfg.training.params
    requires_targets = method_name in _LORA_FAMILY_METHODS
    if method_name == "l2t":
        base_method = method_params.get("base_method", "lora")
        requires_targets = base_method in _LORA_FAMILY_METHODS
    if requires_targets and not method_params.get("target_modules"):
        errors.append(
            "training.params.target_modules: required non-empty list for "
            f"method '{method_name}'"
        )

    if method_name == "freeze" and not str(method_params.get("model_layout", "")).strip():
        errors.append(
            "training.params.model_layout: required non-empty string for method "
            f"'freeze'. Available: {list_model_layouts()}"
        )

    if errors:
        msg = "Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)


def _merge_method_defaults(cfg: TrainingConfig) -> None:
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
        "adapter": cfg.data.adapter,
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
        },
        "data": data_config,
        "training_method": cfg.training.ft_method,
        "method_params": cfg.training.params,
        "training": {
            "num_epochs": cfg.training.num_epochs,
            "per_device_batch_size": cfg.training.per_device_batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "learning_rate": cfg.training.learning_rate,
            "warmup_ratio": cfg.training.warmup_ratio,
            "weight_decay": cfg.training.weight_decay,
            "max_grad_norm": cfg.training.max_grad_norm,
            "save_steps": cfg.training.save_steps,
            "output_dir": cfg.training.output_dir,
        },
    }
