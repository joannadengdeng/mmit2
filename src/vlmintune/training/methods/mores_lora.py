"""Fixed MoReS + LoRA joint adaptation recipe.

MoReS owns the representation hooks and sparse visual-token mask while LoRA
owns the in-layer low-rank updates.  Both parameter families are optimized in
one training run and are persisted as separate, explicit checkpoint
components.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
from peft import PeftModel

from vlmintune.training.methods.lora import LoRAMethod
from vlmintune.training.methods.mores import (
    MORES_CHECKPOINT_FORMAT,
    MoReSMethod,
    compact_mores_state,
    load_compact_mores_state,
)


MORES_LORA_RECIPE = "mores_lora_v1"
MORES_LORA_CHECKPOINT_COMPONENTS = {
    "lora": "adapter_model.safetensors",
    "mores": "mores_tuned.pt",
}


def _mores_adapters(model: nn.Module) -> nn.ModuleList:
    """Resolve MoReS adapters through a PEFT wrapper with a clear failure."""

    try:
        adapters = model.mores_adapters
    except AttributeError as exc:
        raise RuntimeError(
            "MoReS + LoRA model is missing its MoReS adapter stack."
        ) from exc
    if not isinstance(adapters, nn.ModuleList) or not adapters:
        raise RuntimeError("MoReS + LoRA requires a non-empty MoReS adapter stack.")
    return adapters


def _is_lora_parameter(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name


def _is_mores_parameter(name: str) -> bool:
    return "mores_adapters" in name


def _validate_mores_lora_checkpoint(path: str) -> Dict[str, Any]:
    """Reject incomplete or cross-run component mixtures before loading."""

    metadata_path = os.path.join(path, "vlmintune_meta.json")
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    expected_fields = {
        "ft_method": "mores_lora",
        "recipe": MORES_LORA_RECIPE,
        "combination_recipe": MORES_LORA_RECIPE,
        "structure_methods": ["mores", "lora"],
        "composition_order": ["mores", "lora"],
        "component_recipes": {
            "mores": MORES_CHECKPOINT_FORMAT,
            "lora": "lora_v1",
        },
        "checkpoint_components": MORES_LORA_CHECKPOINT_COMPONENTS,
    }
    for field, expected in expected_fields.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Invalid MoReS + LoRA checkpoint metadata {field}: "
                f"expected {expected!r}, got {metadata.get(field)!r}."
            )

    required_files = (
        "adapter_config.json",
        MORES_LORA_CHECKPOINT_COMPONENTS["lora"],
        MORES_LORA_CHECKPOINT_COMPONENTS["mores"],
    )
    for filename in required_files:
        component_path = os.path.join(path, filename)
        if not os.path.isfile(component_path) or os.path.getsize(component_path) <= 0:
            raise ValueError(
                f"MoReS + LoRA checkpoint component is missing or empty: {filename}"
            )
    return metadata


class MoReSLoRAMethod(MoReSMethod, LoRAMethod):
    """Jointly train the MoReS and LoRA v1 structures."""

    name = "mores_lora"
    display_name = "MoReS + LoRA"

    def prepare_model_impl(self, model, processor, model_spec):
        # Install MoReS first so its hooks remain on the actual VLM layers.
        # PEFT then freezes all non-LoRA parameters, so explicitly re-enable
        # the MoReS stack after LoRA injection.
        model, mores_info = MoReSMethod.prepare_model_impl(
            self,
            model,
            processor,
            model_spec,
        )
        model, lora_info = LoRAMethod.prepare_model_impl(
            self,
            model,
            processor,
            model_spec,
        )
        _mores_adapters(model).requires_grad_(True)

        # Validate the exact optimizer ownership during preparation as well as
        # when the trainer later requests parameter groups.
        self.get_trainable_params(model)
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        mores_recipe = mores_info.rsplit("\nTrainable:", 1)[0]
        lora_recipe = lora_info.rsplit("\nTrainable:", 1)[0]
        info = (
            f"{self.display_name} fixed joint recipe ({MORES_LORA_RECIPE})\n"
            "Composition order: MoReS hooks, then LoRA injection; both train jointly\n"
            f"{mores_recipe}\n"
            f"{lora_recipe}\n"
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
            if not _is_lora_parameter(name) and not _is_mores_parameter(name)
        ]
        if unexpected:
            raise RuntimeError(
                "MoReS + LoRA found trainable parameters outside the joint "
                f"adapter families: {unexpected[:5]}"
            )

        lora_named = [item for item in trainable_named if _is_lora_parameter(item[0])]
        mores_named = [item for item in trainable_named if _is_mores_parameter(item[0])]
        if not lora_named or not mores_named:
            missing = []
            if not lora_named:
                missing.append("LoRA")
            if not mores_named:
                missing.append("MoReS")
            raise RuntimeError(
                "MoReS + LoRA is missing trainable adapter family/families: "
                + ", ".join(missing)
            )

        params = [parameter for _, parameter in trainable_named]
        parameter_ids = [id(parameter) for parameter in params]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError("MoReS + LoRA optimizer contains duplicate parameters.")
        if set(parameter_ids) != {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }:
            raise RuntimeError(
                "MoReS + LoRA optimizer parameters do not exactly match the "
                "model's trainable parameters."
            )
        return [{"params": params}]

    def _save_weights(self, model, path):
        model.save_pretrained(path)
        torch.save(
            compact_mores_state(_mores_adapters(model)),
            os.path.join(path, MORES_LORA_CHECKPOINT_COMPONENTS["mores"]),
        )

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "recipe": MORES_LORA_RECIPE,
            "combination_recipe": MORES_LORA_RECIPE,
            "structure_methods": ["mores", "lora"],
            "composition_order": ["mores", "lora"],
            "component_recipes": {
                "mores": MORES_CHECKPOINT_FORMAT,
                "lora": "lora_v1",
            },
            "checkpoint_components": dict(MORES_LORA_CHECKPOINT_COMPONENTS),
        }

    def _restore_model(self, model, processor, model_spec, path):
        _validate_mores_lora_checkpoint(path)
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
        model, _ = MoReSMethod.prepare_model_impl(
            self,
            model,
            processor,
            model_spec,
        )
        state = torch.load(
            os.path.join(path, MORES_LORA_CHECKPOINT_COMPONENTS["mores"]),
            map_location="cpu",
            weights_only=True,
        )
        load_compact_mores_state(_mores_adapters(model), state)
        return model


__all__ = [
    "MORES_LORA_CHECKPOINT_COMPONENTS",
    "MORES_LORA_RECIPE",
    "MoReSLoRAMethod",
    "_validate_mores_lora_checkpoint",
]
