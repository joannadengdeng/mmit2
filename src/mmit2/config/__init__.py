"""Configuration package exports."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "TrainingConfig": ("mmit2.config.training_config", "TrainingConfig"),
    "RuntimeConfig": ("mmit2.config.runtime", "RuntimeConfig"),
    "SSHConfig": ("mmit2.config.runtime", "SSHConfig"),
    "ModelConfig": ("mmit2.config.training_config", "ModelConfig"),
    "TrainingParams": ("mmit2.config.training_config", "TrainingParams"),
    "ExperimentConfig": ("mmit2.config.training_config", "ExperimentConfig"),
    "DataConfig": ("mmit2.config.training_config", "DataConfig"),
    "run_remote_module": ("mmit2.config.runtime", "run_remote_module"),
    "load_runtime_config_dict": ("mmit2.config.training_config", "load_runtime_config_dict"),
    "load_config": ("mmit2.config.training_config", "load_config"),
    "load_config_dict": ("mmit2.config.training_config", "load_config_dict"),
    "config_to_trainer_dict": ("mmit2.config.training_config", "config_to_trainer_dict"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
