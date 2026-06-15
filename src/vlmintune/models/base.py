"""Model-spec base classes and helpers for built-in VLM backbones."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch.nn as nn


def resolve_attr_path(obj: object, attr_path: str) -> object:
    value = obj
    for part in attr_path.split("."):
        value = getattr(value, part)
    return value


def resolve_int_attr(spec: "ModelSpec", obj: object, attr_path: str, label: str) -> int:
    try:
        return int(resolve_attr_path(obj, attr_path))
    except AttributeError as exc:
        raise ValueError(
            f"model '{spec.name}' expects {label} at '{attr_path}', "
            f"but that path was not found on {obj.__class__.__name__}."
        ) from exc


def resolve_module_sequence(
    spec: "ModelSpec",
    obj: object,
    attr_path: str,
    label: str,
) -> Sequence[nn.Module]:
    try:
        modules = list(resolve_attr_path(obj, attr_path))
    except AttributeError as exc:
        raise ValueError(
            f"model '{spec.name}' expects {label} at '{attr_path}', "
            f"but that path was not found on {obj.__class__.__name__}."
        ) from exc
    if not modules:
        raise ValueError(
            f"model '{spec.name}' resolved '{attr_path}', "
            f"but no {label} were found."
        )
    return modules


class ModelSpec(ABC):
    """Runtime contract for one built-in VLM backbone."""

    name: str = ""
    hf_model_id: str = ""
    transformer_layer_path: str = ""
    append_eos_to_training_answer: bool = False

    @abstractmethod
    def get_transformer_layers(self, model: nn.Module) -> Sequence[nn.Module]:
        """Return the ordered transformer layers for this backbone."""

    @abstractmethod
    def get_hidden_size(self, model: nn.Module) -> int:
        """Return the language hidden size for this backbone."""

    @abstractmethod
    def get_image_token_id(self, model: nn.Module) -> int:
        """Return the special image token id for this backbone."""
