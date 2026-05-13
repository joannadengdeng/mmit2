"""Built-in training method exports."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "QLoRAMethod": ("mmit2.training.methods.lora", "QLoRAMethod"),
    "LoRAMethod": ("mmit2.training.methods.lora", "LoRAMethod"),
    "DoRAMethod": ("mmit2.training.methods.dora", "DoRAMethod"),
    "FreezeTuningMethod": ("mmit2.training.methods.freeze", "FreezeTuningMethod"),
    "L2TMethod": ("mmit2.training.methods.l2t", "L2TMethod"),
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
