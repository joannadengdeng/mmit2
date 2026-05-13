"""mmit2 public package exports.

The package keeps top-level imports lazy so method registration and config
helpers do not fight each other during import-time initialization.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "registry": ("mmit2.training.registry", None),
    "Method": ("mmit2.eval.methods.base", "Method"),
    "LocalMethod": ("mmit2.eval.methods.local_method", "LocalMethod"),
    "CanonicalSample": ("mmit2.data.types", "CanonicalSample"),
    "EvalSample": ("mmit2.data.types", "EvalSample"),
    "Turn": ("mmit2.data.types", "Turn"),
    "HFDatasetsAdapter": ("mmit2.data.adapters.hf_datasets", "HFDatasetsAdapter"),
    "DatasetProfile": ("mmit2.data.adapters.hf_datasets", "DatasetProfile"),
    "QLoRAMethod": ("mmit2.training.methods.lora", "QLoRAMethod"),
    "LoRAMethod": ("mmit2.training.methods.lora", "LoRAMethod"),
    "DoRAMethod": ("mmit2.training.methods.dora", "DoRAMethod"),
    "FreezeTuningMethod": ("mmit2.training.methods.freeze", "FreezeTuningMethod"),
    "L2TMethod": ("mmit2.training.methods.l2t", "L2TMethod"),
    "ChatTemplatePreprocessor": (
        "mmit2.training.preprocessors.chat_template",
        "ChatTemplatePreprocessor",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__ if module_name.startswith(".") else None)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
