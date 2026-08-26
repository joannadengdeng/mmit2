"""Fixed MoReS + DoRA joint adaptation recipe."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
from peft import PeftModel

from vlmintune.training.methods.dora import DoRAMethod
from vlmintune.training.methods.mores import (
    MORES_CHECKPOINT_FORMAT,
    MoReSMethod,
    compact_mores_state,
    load_compact_mores_state,
)


MORES_DORA_RECIPE = "mores_dora_v1"
MORES_DORA_CHECKPOINT_COMPONENTS = {
    "dora": "adapter_model.safetensors",
    "mores": "mores_tuned.pt",
}


def _mores_adapters(model: nn.Module) -> nn.ModuleList:
    """Resolve the MoReS stack through a PEFT wrapper."""

    try:
        adapters = model.mores_adapters
    except AttributeError as exc:
        raise RuntimeError(
            "MoReS + DoRA model is missing its MoReS adapter stack."
        ) from exc
    if not isinstance(adapters, nn.ModuleList) or not adapters:
        raise RuntimeError("MoReS + DoRA requires a non-empty MoReS adapter stack.")
    return adapters


def _dora_parameter_family(name: str) -> str | None:
    if "lora_A" in name:
        return "DoRA A"
    if "lora_B" in name:
        return "DoRA B"
    if "lora_magnitude_vector" in name:
        return "DoRA magnitude"
    return None


def _is_mores_parameter(name: str) -> bool:
    return "mores_adapters" in name


def _validate_mores_dora_checkpoint(path: str) -> Dict[str, Any]:
    """Reject incomplete or mismatched combination checkpoints."""

    metadata_path = os.path.join(path, "vlmintune_meta.json")
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    expected_fields = {
        "ft_method": "mores_dora",
        "recipe": MORES_DORA_RECIPE,
        "combination_recipe": MORES_DORA_RECIPE,
        "structure_methods": ["mores", "dora"],
        "composition_order": ["mores", "dora"],
        "component_recipes": {
            "mores": MORES_CHECKPOINT_FORMAT,
            "dora": "dora_v1",
        },
        "checkpoint_components": MORES_DORA_CHECKPOINT_COMPONENTS,
    }
    for field, expected in expected_fields.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Invalid MoReS + DoRA checkpoint metadata {field}: "
                f"expected {expected!r}, got {metadata.get(field)!r}."
            )

    required_files = (
        "adapter_config.json",
        MORES_DORA_CHECKPOINT_COMPONENTS["dora"],
        MORES_DORA_CHECKPOINT_COMPONENTS["mores"],
    )
    for filename in required_files:
        component_path = os.path.join(path, filename)
        if not os.path.isfile(component_path) or os.path.getsize(component_path) <= 0:
            raise ValueError(
                f"MoReS + DoRA checkpoint component is missing or empty: {filename}"
            )

    return metadata


class MoReSDoRAMethod(MoReSMethod, DoRAMethod):
    """Jointly train the MoReS and DoRA v1 structures."""

    name = "mores_dora"
    display_name = "MoReS + DoRA"

    def prepare_model_impl(self, model, processor, model_spec):
        # Install the output hooks on the real layers first.  DoRA then wraps
        # their linear projections and freezes all non-PEFT parameters, so the
        # MoReS stack must be explicitly re-enabled afterward.
        model, mores_info = MoReSMethod.prepare_model_impl(
            self,
            model,
            processor,
            model_spec,
        )
        model, dora_info = DoRAMethod.prepare_model_impl(
            self,
            model,
            processor,
            model_spec,
        )
        _mores_adapters(model).requires_grad_(True)

        self.get_trainable_params(model)
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        mores_recipe = mores_info.rsplit("\nTrainable:", 1)[0]
        dora_recipe = dora_info.rsplit("\nTrainable:", 1)[0]
        info = (
            f"{self.display_name} fixed joint recipe ({MORES_DORA_RECIPE})\n"
            "Composition order: MoReS hooks, then DoRA injection; both train jointly\n"
            f"{mores_recipe}\n"
            f"{dora_recipe}\n"
            f"Joint trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.4f}%)"
        )
        return model, info

    def get_trainable_params(self, model: nn.Module) -> List[Dict[str, Any]]:
        trainable_named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        unexpected = [
            name
            for name, _ in trainable_named
            if _dora_parameter_family(name) is None
            and not _is_mores_parameter(name)
        ]
        if unexpected:
            raise RuntimeError(
                "MoReS + DoRA found trainable parameters outside the joint "
                f"adapter families: {unexpected[:5]}"
            )

        missing_families = [
            family
            for family in ("DoRA A", "DoRA B", "DoRA magnitude")
            if not any(
                _dora_parameter_family(name) == family
                for name, _ in trainable_named
            )
        ]
        if not any(_is_mores_parameter(name) for name, _ in trainable_named):
            missing_families.append("MoReS")
        if missing_families:
            raise RuntimeError(
                "MoReS + DoRA is missing trainable adapter family/families: "
                + ", ".join(missing_families)
            )

        params = [parameter for _, parameter in trainable_named]
        parameter_ids = [id(parameter) for parameter in params]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError("MoReS + DoRA optimizer contains duplicate parameters.")
        if set(parameter_ids) != {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }:
            raise RuntimeError(
                "MoReS + DoRA optimizer parameters do not exactly match the "
                "model's trainable parameters."
            )
        return [{"params": params}]

    def _save_weights(self, model, path):
        model.save_pretrained(path)
        torch.save(
            compact_mores_state(_mores_adapters(model)),
            os.path.join(path, MORES_DORA_CHECKPOINT_COMPONENTS["mores"]),
        )

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "recipe": MORES_DORA_RECIPE,
            "combination_recipe": MORES_DORA_RECIPE,
            "structure_methods": ["mores", "dora"],
            "composition_order": ["mores", "dora"],
            "component_recipes": {
                "mores": MORES_CHECKPOINT_FORMAT,
                "dora": "dora_v1",
            },
            "checkpoint_components": dict(MORES_DORA_CHECKPOINT_COMPONENTS),
        }

    def _restore_model(self, model, processor, model_spec, path):
        _validate_mores_dora_checkpoint(path)
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
        model, _ = MoReSMethod.prepare_model_impl(
            self,
            model,
            processor,
            model_spec,
        )
        state = torch.load(
            os.path.join(path, MORES_DORA_CHECKPOINT_COMPONENTS["mores"]),
            map_location="cpu",
            weights_only=True,
        )
        load_compact_mores_state(_mores_adapters(model), state)
        return model


__all__ = [
    "MORES_DORA_CHECKPOINT_COMPONENTS",
    "MORES_DORA_RECIPE",
    "MoReSDoRAMethod",
    "_validate_mores_dora_checkpoint",
]
