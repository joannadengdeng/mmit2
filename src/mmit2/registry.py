"""Simple lookup table for built-in training methods."""
from __future__ import annotations

from typing import Any

from mmit2.training.methods.dora import DoRAMethod
from mmit2.training.methods.freeze import FreezeTuningMethod
from mmit2.training.methods.l2t import L2TMethod
from mmit2.training.methods.lora import LoRAMethod, QLoRAMethod

_TRAINING_METHODS: dict[str, type[Any]] = {
    "qlora": QLoRAMethod,
    "lora": LoRAMethod,
    "dora": DoRAMethod,
    "freeze": FreezeTuningMethod,
    "l2t": L2TMethod,
}

_REGISTERED_DEFAULTS: dict[str, dict[str, Any]] = {}


def register_training_method(
    name: str,
    cls: type[Any],
    defaults: dict[str, Any] | None = None,
) -> None:
    """Register a training method by name."""
    _TRAINING_METHODS[name] = cls
    if defaults is not None:
        _REGISTERED_DEFAULTS[name] = dict(defaults)
    elif name in _REGISTERED_DEFAULTS:
        del _REGISTERED_DEFAULTS[name]


def _resolve_training_method(name: str) -> type[Any]:
    try:
        return _TRAINING_METHODS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown training method '{name}'. Available: {sorted(_TRAINING_METHODS)}"
        ) from exc


def list_training_methods() -> list[str]:
    """Return the registered training method names."""
    return list(_TRAINING_METHODS)


def get_training_method_cls(name: str) -> type[Any]:
    """Return the training method class for ``name``."""
    return _resolve_training_method(name)


def build_training_method(name: str, **kwargs: Any) -> Any:
    """Instantiate a training method."""
    return get_training_method_cls(name)(**kwargs)


def get_training_method_defaults(name: str) -> dict[str, Any]:
    """Return a copy of the registered default config for ``name``."""
    if name not in _TRAINING_METHODS:
        raise KeyError(
            f"Unknown training method '{name}'. Available: {sorted(_TRAINING_METHODS)}"
        )
    if name in _REGISTERED_DEFAULTS:
        return dict(_REGISTERED_DEFAULTS[name])
    return get_training_method_cls(name)().default_config()


__all__ = [
    "register_training_method",
    "list_training_methods",
    "get_training_method_cls",
    "build_training_method",
    "get_training_method_defaults",
]
