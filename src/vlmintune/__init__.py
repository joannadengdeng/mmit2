"""vlmintune public package exports.

The package keeps top-level imports lazy so method registration and config
helpers do not fight each other during import-time initialization.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "registry": ("vlmintune.training.methods.registry", None),
    "LocalMethod": ("vlmintune.eval.method", "LocalMethod"),
    "CanonicalSample": ("vlmintune.data.types", "CanonicalSample"),
    "EvalSample": ("vlmintune.data.types", "EvalSample"),
    "HFDatasetsAdapter": ("vlmintune.data.hf_datasets", "HFDatasetsAdapter"),
    "DatasetProfile": ("vlmintune.data.hf_datasets", "DatasetProfile"),
    "QLoRAMethod": ("vlmintune.training.methods.lora", "QLoRAMethod"),
    "LoRAMethod": ("vlmintune.training.methods.lora", "LoRAMethod"),
    "DoRAMethod": ("vlmintune.training.methods.dora", "DoRAMethod"),
    "L2TMethod": ("vlmintune.training.methods.l2t", "L2TMethod"),
    "MoReSMethod": ("vlmintune.training.methods.mores", "MoReSMethod"),
    "MoReSLoRAMethod": (
        "vlmintune.training.methods.mores_lora",
        "MoReSLoRAMethod",
    ),
    "MoReSDoRAMethod": (
        "vlmintune.training.methods.mores_dora",
        "MoReSDoRAMethod",
    ),
    "ReFTMethod": ("vlmintune.training.methods.reft", "ReFTMethod"),
    "ReFTLoRAMethod": (
        "vlmintune.training.methods.reft_lora",
        "ReFTLoRAMethod",
    ),
    "VLAdapterMethod": ("vlmintune.training.methods.vl_adapter", "VLAdapterMethod"),
    "ChatTemplatePreprocessor": (
        "vlmintune.training.chat_template",
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
