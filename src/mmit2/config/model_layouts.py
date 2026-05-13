"""Explicit model layout registry used by freeze tuning."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Dict, Iterable

import yaml


@dataclass(frozen=True)
class ModelLayout:
    name: str
    description: str
    model_ids: tuple[str, ...]
    model_types: tuple[str, ...]
    transformer_layer_path: str


def _iter_strs(values: Iterable[Any] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


def _load_layouts() -> dict[str, ModelLayout]:
    layout_path = resources.files("mmit2.config").joinpath("model_layouts.yaml")
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


__all__ = ["ModelLayout", "get_model_layout", "list_model_layouts"]
