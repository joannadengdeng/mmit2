"""Built-in training method exports."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "build_training_method": ("vlmintune.training.methods.registry", "build_training_method"),
    "get_training_method_cls": ("vlmintune.training.methods.registry", "get_training_method_cls"),
    "get_training_method_defaults": ("vlmintune.training.methods.registry", "get_training_method_defaults"),
    "list_training_methods": ("vlmintune.training.methods.registry", "list_training_methods"),
    "QLoRAMethod": ("vlmintune.training.methods.lora", "QLoRAMethod"),
    "LoRAMethod": ("vlmintune.training.methods.lora", "LoRAMethod"),
    "DoRAMethod": ("vlmintune.training.methods.dora", "DoRAMethod"),
    "FreezeTuningMethod": ("vlmintune.training.methods.freeze", "FreezeTuningMethod"),
    "L2TMethod": ("vlmintune.training.methods.l2t", "L2TMethod"),
    "MoReSMethod": ("vlmintune.training.methods.mores", "MoReSMethod"),
    "MoLEMethod": ("vlmintune.training.methods.mole", "MoLEMethod"),
    "ReFTMethod": ("vlmintune.training.methods.reft", "ReFTMethod"),
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
