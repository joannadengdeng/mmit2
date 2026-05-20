"""Simple lookup table for built-in training methods."""
from __future__ import annotations

from typing import Any

from vlmintune.training.methods.base import TrainingMethod
from vlmintune.training.methods.dora import DoRAMethod
from vlmintune.training.methods.freeze import FreezeTuningMethod
from vlmintune.training.methods.l2t import L2TMethod
from vlmintune.training.methods.lora import LoRAMethod, QLoRAMethod
from vlmintune.training.methods.mores import MoReSMethod

TrainingMethodType = type[TrainingMethod]

_TRAINING_METHODS: dict[str, TrainingMethodType] = {
    "qlora": QLoRAMethod,
    "lora": LoRAMethod,
    "dora": DoRAMethod,
    "freeze": FreezeTuningMethod,
    "l2t": L2TMethod,
    "mores": MoReSMethod,
}


def list_training_methods() -> list[str]:
    """Return the registered training method names."""
    return list(_TRAINING_METHODS)


def get_training_method_cls(name: str) -> TrainingMethodType:
    """Return the training method class for ``name``."""
    try:
        return _TRAINING_METHODS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown training method '{name}'. Available: {sorted(_TRAINING_METHODS)}"
        ) from exc


def build_training_method(name: str, **kwargs: Any) -> TrainingMethod:
    """Instantiate a training method."""
    return get_training_method_cls(name)(**kwargs)


def get_training_method_defaults(name: str) -> dict[str, Any]:
    """Return a copy of the built-in default config for ``name``."""
    return get_training_method_cls(name)().default_config()


__all__ = [
    "list_training_methods",
    "get_training_method_cls",
    "build_training_method",
    "get_training_method_defaults",
]
