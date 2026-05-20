"""Explicit model layout registry used by freeze tuning."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Callable, Dict, Iterable, Sequence

import torch.nn as nn
import yaml


@dataclass(frozen=True)
class ModelLayout:
    name: str
    description: str
    model_ids: tuple[str, ...]
    model_types: tuple[str, ...]
    transformer_layer_path: str


TransformerLayerResolver = Callable[[nn.Module, ModelLayout], Sequence[nn.Module]]


def _iter_strs(values: Iterable[Any] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


def _load_layouts() -> dict[str, ModelLayout]:
    layout_path = resources.files("vlmintune.config").joinpath("model_layouts.yaml")
    raw = yaml.safe_load(layout_path.read_text(encoding="utf-8")) or {}
    raw_layouts: Dict[str, Dict[str, Any]] = raw.get("layouts", {})
    layouts: dict[str, ModelLayout] = {}
    for name, item in raw_layouts.items():
        matches = item.get("matches", {})
        layouts[name] = ModelLayout(
            name=name,
            description=str(item.get("description", "")).strip(),
            model_ids=_iter_strs(matches.get("model_ids")),
            model_types=_iter_strs(matches.get("model_types")),
            transformer_layer_path=str(item["transformer_layer_path"]).strip(),
        )
    return layouts


_MODEL_LAYOUTS = _load_layouts()


def list_model_layouts() -> list[str]:
    return sorted(_MODEL_LAYOUTS)


def get_model_layout(name: str) -> ModelLayout:
    try:
        return _MODEL_LAYOUTS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model_layout '{name}'. Available: {sorted(_MODEL_LAYOUTS)}"
        ) from exc


def resolve_attr_path(model: nn.Module, attr_path: str) -> object:
    obj: object = model
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_layers_from_attr_path(model: nn.Module, layout: ModelLayout) -> Sequence[nn.Module]:
    try:
        layers = list(resolve_attr_path(model, layout.transformer_layer_path))
    except AttributeError as exc:
        raise ValueError(
            f"model_layout '{layout.name}' expects transformer layers at "
            f"'{layout.transformer_layer_path}', but that path was not found on "
            f"{model.__class__.__name__}."
        ) from exc
    if not layers:
        raise ValueError(
            f"model_layout '{layout.name}' resolved '{layout.transformer_layer_path}', "
            "but no transformer layers were found."
        )
    return layers


def resolve_qwen2_5_vl_transformer_layers(
    model: nn.Module,
    layout: ModelLayout,
) -> Sequence[nn.Module]:
    return resolve_layers_from_attr_path(model, layout)


_TRANSFORMER_LAYER_RESOLVERS: dict[str, TransformerLayerResolver] = {
    "qwen2_5_vl": resolve_qwen2_5_vl_transformer_layers,
}


def resolve_transformer_layers(model: nn.Module, model_layout: str) -> Sequence[nn.Module]:
    layout = get_model_layout(model_layout)
    resolver = _TRANSFORMER_LAYER_RESOLVERS.get(layout.name, resolve_layers_from_attr_path)
    return resolver(model, layout)


__all__ = [
    "ModelLayout",
    "get_model_layout",
    "list_model_layouts",
    "resolve_attr_path",
    "resolve_transformer_layers",
    "resolve_qwen2_5_vl_transformer_layers",
]
