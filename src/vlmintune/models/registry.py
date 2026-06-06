"""Registry for built-in model specs."""
from __future__ import annotations

from vlmintune.models.base import ModelSpec
from vlmintune.models.llava15 import LLAVA15_SPEC
from vlmintune.models.qwen25vl import QWEN25VL_SPEC

_MODEL_SPECS: dict[str, ModelSpec] = {
    QWEN25VL_SPEC.name: QWEN25VL_SPEC,
    LLAVA15_SPEC.name: LLAVA15_SPEC,
}


def list_model_names() -> list[str]:
    return sorted(_MODEL_SPECS)


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _MODEL_SPECS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model.name '{name}'. Available: {sorted(_MODEL_SPECS)}"
        ) from exc


__all__ = [
    "get_model_spec",
    "list_model_names",
]
