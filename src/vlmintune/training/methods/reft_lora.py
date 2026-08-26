"""Fixed ReFT + LoRA joint adaptation recipe."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
from peft import PeftModel

from vlmintune.training.methods.lora import LoRAMethod
from vlmintune.training.methods.reft import (
    REFT_CHECKPOINT_FORMAT,
    ReFTMethod,
    compact_reft_state,
    load_compact_reft_state,
)


REFT_LORA_RECIPE = "reft_lora_v1"
REFT_LORA_CHECKPOINT_COMPONENTS = {
    "lora": "adapter_model.safetensors",
    "reft": "reft_tuned.pt",
}


def _reft_adapters(model: nn.Module) -> nn.ModuleList:
    """Resolve the ReFT stack through a PEFT wrapper."""

    try:
        adapters = model.reft_adapters
    except AttributeError as exc:
        raise RuntimeError(
            "ReFT + LoRA model is missing its ReFT adapter stack."
        ) from exc
    if not isinstance(adapters, nn.ModuleList) or not adapters:
        raise RuntimeError("ReFT + LoRA requires a non-empty ReFT adapter stack.")
    return adapters


def _lora_parameter_family(name: str) -> str | None:
    if "lora_A" in name:
        return "LoRA A"
    if "lora_B" in name:
        return "LoRA B"
    return None


def _is_reft_parameter(name: str) -> bool:
    return "reft_adapters" in name


def _validate_reft_lora_checkpoint(path: str) -> Dict[str, Any]:
    """Reject incomplete or cross-run component mixtures before loading."""

    metadata_path = os.path.join(path, "vlmintune_meta.json")
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    expected_fields = {
        "ft_method": "reft_lora",
        "recipe": REFT_LORA_RECIPE,
        "combination_recipe": REFT_LORA_RECIPE,
        "structure_methods": ["reft", "lora"],
        "composition_order": ["reft", "lora"],
        "component_recipes": {
            "reft": REFT_CHECKPOINT_FORMAT,
            "lora": "lora_v1",
        },
        "checkpoint_components": REFT_LORA_CHECKPOINT_COMPONENTS,
    }
    for field, expected in expected_fields.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Invalid ReFT + LoRA checkpoint metadata {field}: "
                f"expected {expected!r}, got {metadata.get(field)!r}."
            )

    required_files = (
        "adapter_config.json",
        REFT_LORA_CHECKPOINT_COMPONENTS["lora"],
        REFT_LORA_CHECKPOINT_COMPONENTS["reft"],
    )
    for filename in required_files:
        component_path = os.path.join(path, filename)
        if not os.path.isfile(component_path) or os.path.getsize(component_path) <= 0:
            raise ValueError(
                f"ReFT + LoRA checkpoint component is missing or empty: {filename}"
            )

    return metadata


class ReFTLoRAMethod(ReFTMethod, LoRAMethod):
    """Jointly train the fixed ReFT v1 and LoRA v1 structures."""

    name = "reft_lora"
    display_name = "ReFT + LoRA"

    def prepare_model_impl(self, model, processor, model_spec):
        # ReFT must own hooks on the real Transformer layers.  PEFT injection
        # follows, then the ReFT stack is re-enabled after PEFT freezes it.
        model, reft_info = ReFTMethod.prepare_model_impl(
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
        _reft_adapters(model).requires_grad_(True)

        self.get_trainable_params(model)
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        reft_recipe = reft_info.rsplit("\nTrainable:", 1)[0]
        lora_recipe = lora_info.rsplit("\nTrainable:", 1)[0]
        info = (
            f"{self.display_name} fixed joint recipe ({REFT_LORA_RECIPE})\n"
            "Composition order: ReFT hooks, then LoRA injection; both train jointly\n"
            f"{reft_recipe}\n"
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
            if _lora_parameter_family(name) is None
            and not _is_reft_parameter(name)
        ]
        if unexpected:
            raise RuntimeError(
                "ReFT + LoRA found trainable parameters outside the joint "
                f"adapter families: {unexpected[:5]}"
            )

        missing = [
            family
            for family in ("LoRA A", "LoRA B")
            if not any(
                _lora_parameter_family(name) == family
                for name, _ in trainable_named
            )
        ]
        if not any(_is_reft_parameter(name) for name, _ in trainable_named):
            missing.append("ReFT")
        if missing:
            raise RuntimeError(
                "ReFT + LoRA is missing trainable adapter family/families: "
                + ", ".join(missing)
            )

        params = [parameter for _, parameter in trainable_named]
        parameter_ids = [id(parameter) for parameter in params]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError("ReFT + LoRA optimizer contains duplicate parameters.")
        if set(parameter_ids) != {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }:
            raise RuntimeError(
                "ReFT + LoRA optimizer parameters do not exactly match the "
                "model's trainable parameters."
            )
        return [{"params": params}]

    def _save_weights(self, model, path):
        model.save_pretrained(path)
        torch.save(
            compact_reft_state(_reft_adapters(model)),
            os.path.join(path, REFT_LORA_CHECKPOINT_COMPONENTS["reft"]),
        )

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "recipe": REFT_LORA_RECIPE,
            "combination_recipe": REFT_LORA_RECIPE,
            "structure_methods": ["reft", "lora"],
            "composition_order": ["reft", "lora"],
            "component_recipes": {
                "reft": REFT_CHECKPOINT_FORMAT,
                "lora": "lora_v1",
            },
            "checkpoint_components": dict(REFT_LORA_CHECKPOINT_COMPONENTS),
        }

    def _restore_model(self, model, processor, model_spec, path):
        _validate_reft_lora_checkpoint(path)
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
        model, _ = ReFTMethod.prepare_model_impl(
            self,
            model,
            processor,
            model_spec,
        )
        state = torch.load(
            os.path.join(path, REFT_LORA_CHECKPOINT_COMPONENTS["reft"]),
            map_location="cpu",
            weights_only=True,
        )
        load_compact_reft_state(_reft_adapters(model), state)
        return model


__all__ = [
    "REFT_LORA_CHECKPOINT_COMPONENTS",
    "REFT_LORA_RECIPE",
    "ReFTLoRAMethod",
    "_validate_reft_lora_checkpoint",
]
