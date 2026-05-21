"""Freeze Tuning — train only the requested module prefixes."""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from vlmintune.config.model_layouts import (
    get_model_layout,
    list_model_layouts,
    resolve_transformer_layers,
)
from vlmintune.training.methods.base import TrainingMethod, load_processor, load_vlm
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss

_ce_loss = CrossEntropyLoss()
_GRAD_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def can_update(param: nn.Parameter) -> bool:
    return param.dtype in _GRAD_DTYPES


def has_updatable_params(module: nn.Module) -> bool:
    return any(can_update(param) for param in module.parameters(recurse=True))


def list_tunable_modules(model: nn.Module, model_layout: str) -> list[str]:
    candidates = set()
    for name, module in model.named_modules():
        if not name or name.count(".") > 1:
            continue
        if has_updatable_params(module):
            candidates.add(name)

    layout = get_model_layout(model_layout)
    layers = resolve_transformer_layers(model, model_layout)
    candidates.add(layout.transformer_layer_path)
    for idx in range(len(layers)):
        candidates.add(f"{layout.transformer_layer_path}.{idx}")

    return sorted(candidates)


def matches_prefix(param_name: str, prefix: str) -> bool:
    return param_name == prefix or param_name.startswith(prefix + ".")


class FreezeTuningMethod(TrainingMethod):
    """Freeze Tuning: unfreeze the requested module prefixes and keep the rest frozen."""

    name = "freeze"
    display_name = "Freeze Tuning"

    def default_config(self):
        return {
            "model_layout": "",
            "unfreeze_modules": [],
        }

    def prepare_model_impl(self, model, processor, config):
        model_layout = str(config.get("model_layout", "")).strip()
        if not model_layout:
            raise ValueError(
                "Freeze Tuning requires training.params.model_layout. "
                f"Available layouts: {list_model_layouts()}"
            )

        available = list_tunable_modules(model, model_layout)
        unfreeze_modules = [str(name).strip() for name in config["unfreeze_modules"] if str(name).strip()]
        if not unfreeze_modules:
            available_lines = "\n".join(f"  - {name}" for name in available) or "  (no parameterized modules found)"
            raise ValueError(
                "Freeze Tuning requires a non-empty 'unfreeze_modules' list.\n"
                f"Model layout: {model_layout}\n"
                "Available module prefixes you can unfreeze:\n"
                f"{available_lines}"
            )

        for param in model.parameters():
            if can_update(param):
                param.requires_grad = False

        matched_modules = set()
        unfrozen_params = 0
        for name, param in model.named_parameters():
            if not can_update(param):
                continue
            for prefix in unfreeze_modules:
                if matches_prefix(name, prefix):
                    param.requires_grad = True
                    matched_modules.add(prefix)
                    unfrozen_params += 1
                    break

        unknown = [prefix for prefix in unfreeze_modules if prefix not in matched_modules]
        if unknown:
            available_lines = "\n".join(f"  - {name}" for name in available) or "  (no parameterized modules found)"
            raise ValueError(
                "Unknown unfreeze_modules entries: "
                f"{unknown}\nModel layout: {model_layout}\n"
                f"Available module prefixes you can unfreeze:\n{available_lines}"
            )

        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        info = (
            f"Freeze Tuning ({model_layout}): unfrozen [{', '.join(sorted(matched_modules))}]\n"
            f"Trainable parameter tensors: {unfrozen_params}\n"
            f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
        )
        return model, info

    def compute_loss(self, model, batch, outputs):
        return _ce_loss.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        return [{"params": [param for param in model.parameters() if param.requires_grad]}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        trained_names = {name for name, param in model.named_parameters() if param.requires_grad}
        trainable_state = {key: value for key, value in model.state_dict().items() if key in trained_names}
        torch.save(trainable_state, os.path.join(path, "freeze_tuned.pt"))
        processor.save_pretrained(path)
        metadata["ft_method"] = self.name
        with open(os.path.join(path, "vlmintune_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, base_model_id, **kwargs):
        processor = load_processor(base_model_id)
        model = load_vlm(base_model_id, quantize_4bit=False, torch_dtype=torch.bfloat16)
        state = torch.load(
            os.path.join(path, "freeze_tuned.pt"),
            map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state, strict=False)
        model.eval()

        adapter_name = os.path.basename(path)
        info = {"model_id": f"{base_model_id} (Freeze: {adapter_name})"}
        return model, processor, info
