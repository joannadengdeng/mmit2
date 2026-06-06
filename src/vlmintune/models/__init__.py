"""Built-in model specs for structure-sensitive tuning methods."""
from vlmintune.models.base import ModelSpec
from vlmintune.models.registry import get_model_spec, list_model_names

__all__ = [
    "ModelSpec",
    "get_model_spec",
    "list_model_names",
]
