"""MoLE: sparse mixture of LoRA experts for linear layers."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.base import TrainingMethod, load_processor, load_vlm
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss

CROSS_ENTROPY_LOSS = CrossEntropyLoss()
MOLE_CHECKPOINT_FORMAT = "mole_v1"


class MoELoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float, num_experts: int):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        device = base.weight.device
        self.router = nn.Linear(base.in_features, num_experts, bias=False, device=device)
        self.lora_a = nn.ModuleList([
            nn.Linear(base.in_features, rank, bias=False, device=device) for _ in range(num_experts)
        ])
        self.lora_b = nn.ModuleList([
            nn.Linear(rank, base.out_features, bias=False, device=device) for _ in range(num_experts)
        ])
        for expert_b in self.lora_b:
            nn.init.zeros_(expert_b.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_out = self.base(hidden_states)
        route_probs = torch.softmax(self.router(hidden_states.to(self.router.weight.dtype)), dim=-1)
        route = route_probs.argmax(dim=-1)
        hard_gate = torch.nn.functional.one_hot(route, num_classes=len(self.lora_a)).to(route_probs.dtype)
        gate = hard_gate + route_probs - route_probs.detach()
        expert_outputs = []
        dropped = self.dropout(hidden_states)
        for expert_a, expert_b in zip(self.lora_a, self.lora_b):
            expert_in = dropped.to(expert_a.weight.dtype)
            expert_outputs.append(expert_b(expert_a(expert_in)).to(base_out.dtype))
        update = (torch.stack(expert_outputs, dim=-2) * gate.to(base_out.dtype).unsqueeze(-1)).sum(dim=-2)
        return base_out + update * self.scaling


def parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_target_linears(
    model: nn.Module,
    targets: list[str],
    rank: int,
    alpha: int,
    dropout: float,
    num_experts: int,
) -> list[str]:
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name == target or name.endswith("." + target) for target in targets):
            continue
        parent, child_name = parent_module(model, name)
        setattr(parent, child_name, MoELoRALinear(module, rank, alpha, dropout, num_experts))
        replaced.append(name)
    return replaced


class MoLEMethod(TrainingMethod):
    name = "mole"
    display_name = "MoLE"

    def __init__(self) -> None:
        self.last_config: Dict[str, Any] = {}

    def default_config(self) -> Dict[str, Any]:
        return {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [],
            "num_experts": 3,
        }

    def prepare_model_impl(self, model, processor, config, model_spec):
        del processor, model_spec
        self.last_config = dict(config)
        targets = [str(target) for target in config["target_modules"]]
        if not targets:
            raise ValueError("MoLE requires a non-empty 'target_modules' list.")

        for param in model.parameters():
            param.requires_grad = False
        replaced = replace_target_linears(
            model,
            targets,
            int(config["lora_r"]),
            int(config["lora_alpha"]),
            float(config["lora_dropout"]),
            int(config["num_experts"]),
        )
        if not replaced:
            raise ValueError(f"MoLE found no nn.Linear target modules for {targets}.")

        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        return model, (
            f"MoLE: experts={int(config['num_experts'])}, r={int(config['lora_r'])}, "
            f"targets={targets}, replaced={len(replaced)}\n"
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)"
        )

    def compute_loss(self, model, batch, outputs):
        return CROSS_ENTROPY_LOSS.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        return [{"params": [param for param in model.parameters() if param.requires_grad]}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
        state = {key: value for key, value in model.state_dict().items() if key in trainable_names}
        torch.save({"format": MOLE_CHECKPOINT_FORMAT, "state_dict": state}, os.path.join(path, "mole_tuned.pt"))
        processor.save_pretrained(path)
        metadata = {**metadata, "ft_method": self.name, "config": dict(self.last_config)}
        with open(os.path.join(path, "vlmintune_meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, model_name, **kwargs):
        del kwargs
        model_spec = get_model_spec(model_name)
        processor = load_processor(model_spec.hf_model_id)
        model = load_vlm(
            model_spec.hf_model_id,
            quantize_4bit=False,
            torch_dtype=torch.bfloat16,
        )
        with open(os.path.join(path, "vlmintune_meta.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f) or {}
        model, _ = self.prepare_model(model, processor, metadata.get("config") or self.default_config(), model_spec)
        state = torch.load(os.path.join(path, "mole_tuned.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("state_dict", state), strict=False)
        model.eval()
        return model, processor, {"model_id": f"{model_spec.hf_model_id} (MoLE: {os.path.basename(path)})"}


__all__ = ["MOLE_CHECKPOINT_FORMAT", "MoELoRALinear", "MoLEMethod", "replace_target_linears"]
