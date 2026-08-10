"""Fixed MoReS v1 recipe for sparse visual-token representation steering."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.base import TrainingMethod, load_processor, load_vlm
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss


CROSS_ENTROPY_LOSS = CrossEntropyLoss()
MORES_RANK = 1
MORES_VISUAL_TOKEN_INDICES = (1, 2, 3, 4, -5, -4, -3, -2, -1)
MORES_CHECKPOINT_FORMAT = "mores_v1"


def build_mores_intervention_mask(
    model_config: Any,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Select the fixed first four and last five visual-token positions."""
    image_token_id = int(model_config.image_token_id)
    visual_positions = [
        index
        for index, token_id in enumerate(input_ids.tolist())
        if token_id == image_token_id
    ]

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for visual_index in MORES_VISUAL_TOKEN_INDICES:
        position_index = (
            visual_index - 1
            if visual_index > 0
            else len(visual_positions) + visual_index
        )
        if 0 <= position_index < len(visual_positions):
            mask[visual_positions[position_index]] = True
    return mask


def first_parameter_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


class MoReSAdapter(nn.Module):
    """Rank-one MoReS map: h + W_up(Linear(h) - W_down(h))."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        w_down = nn.Linear(hidden_size, MORES_RANK, bias=False, dtype=torch.float32)
        nn.init.orthogonal_(w_down.weight)
        self.w_down = torch.nn.utils.parametrizations.orthogonal(
            w_down,
            orthogonal_map="householder",
        )
        self.w_down.parametrizations.weight.original.data = (
            self.w_down.parametrizations.weight.original.data.to(torch.float32)
        )
        self.linear = nn.Linear(hidden_size, MORES_RANK, dtype=torch.float32)

    @property
    def w_up(self) -> torch.Tensor:
        return self.w_down.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = hidden_states.to(torch.float32)
        update = torch.matmul(self.linear(base) - self.w_down(base), self.w_up)
        return (base + update).to(hidden_states.dtype)


def compact_mores_state(adapters: nn.ModuleList) -> Dict[str, Any]:
    return {
        "format": MORES_CHECKPOINT_FORMAT,
        "layers": [
            {
                "w_down_weight": adapter.w_down.weight.detach().cpu(),
                "linear_weight": adapter.linear.weight.detach().cpu(),
                "linear_bias": adapter.linear.bias.detach().cpu(),
            }
            for adapter in adapters
        ],
    }


def load_compact_mores_state(adapters: nn.ModuleList, state: Dict[str, Any]) -> None:
    if state.get("format") != MORES_CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not the fixed MoReS v1 format.")
    layers = state["layers"]
    if len(layers) != len(adapters):
        raise ValueError(
            f"MoReS layer count mismatch: {len(layers)} checkpoint layers for "
            f"{len(adapters)} model layers."
        )
    with torch.no_grad():
        for adapter, layer_state in zip(adapters, layers):
            adapter.w_down.weight = layer_state["w_down_weight"].to(
                device=adapter.w_down.weight.device,
                dtype=adapter.w_down.weight.dtype,
            )
            adapter.linear.weight.copy_(
                layer_state["linear_weight"].to(
                    device=adapter.linear.weight.device,
                    dtype=adapter.linear.weight.dtype,
                )
            )
            adapter.linear.bias.copy_(
                layer_state["linear_bias"].to(
                    device=adapter.linear.bias.device,
                    dtype=adapter.linear.bias.dtype,
                )
            )


class MoReSMethod(TrainingMethod):
    """Fixed v1: rank one, visual f4+l5, and every language layer."""

    name = "mores"
    display_name = "MoReS (fixed sparse v1)"

    def __init__(self) -> None:
        self.current_intervention_mask: Optional[torch.Tensor] = None
        self.hook_call_count = 0
        self.hook_layer_indices: List[int] = []
        self.hook_intervention_tokens: Optional[int] = None

    def layer_hook(self, adapter: MoReSAdapter, layer_index: int):
        def hook(module, args, output):
            del module, args
            if self.current_intervention_mask is None:
                return output

            is_tuple_output = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple_output else output
            mask = self.current_intervention_mask.to(hidden_states.device)
            self.hook_call_count += 1
            if len(self.hook_layer_indices) < 16:
                self.hook_layer_indices.append(layer_index)
            if self.hook_intervention_tokens is None:
                self.hook_intervention_tokens = int(mask.sum().item())

            updated = hidden_states.clone()
            updated[mask] = adapter(hidden_states[mask])
            if is_tuple_output:
                return (updated, *output[1:])
            return updated

        return hook

    def forward_pre_hook(self, module, args, kwargs):
        if module.training:
            return None
        past_key_values = kwargs.get("past_key_values")
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            self.current_intervention_mask = None
            return None
        input_ids = kwargs["input_ids"]
        self.current_intervention_mask = torch.stack(
            [build_mores_intervention_mask(module.config, row) for row in input_ids],
            dim=0,
        ).to(input_ids.device)
        return None

    def prepare_model_impl(self, model, processor, model_spec):
        del processor
        self.current_intervention_mask = None
        self.hook_call_count = 0
        self.hook_layer_indices = []
        self.hook_intervention_tokens = None

        for parameter in model.parameters():
            parameter.requires_grad = False

        hidden_size = int(model_spec.get_hidden_size(model))
        layers = list(model_spec.get_transformer_layers(model))
        adapters: list[MoReSAdapter] = []
        for layer_index, layer in enumerate(layers):
            adapter = MoReSAdapter(hidden_size).to(first_parameter_device(layer))
            adapters.append(adapter)
            layer.register_forward_hook(self.layer_hook(adapter, layer_index))

        model.mores_adapters = nn.ModuleList(adapters)
        model.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        model.vlmintuneMoresMethod = self

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info = (
            f"MoReS v1: backbone={model_spec.name}, rank={MORES_RANK}, "
            f"visual positions=f4+l5 {list(MORES_VISUAL_TOKEN_INDICES)}, "
            f"layers=all ({len(layers)}), init=random, scale=1.0\n"
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)"
        )
        return model, info

    def build_forward_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.current_intervention_mask = batch["intervention_mask"].bool()
        excluded = {
            "instruction_supervision_mask",
            "intervention_mask",
            "reft_intervention_mask",
        }
        return {key: value for key, value in batch.items() if key not in excluded}

    def runtime_debug(self) -> Dict[str, Any]:
        return {
            "kind": "mores_runtime",
            "hook_call_count": self.hook_call_count,
            "hook_layer_indices_preview": self.hook_layer_indices,
            "hook_intervention_tokens": self.hook_intervention_tokens,
        }

    def compute_loss(self, model, batch, outputs):
        return CROSS_ENTROPY_LOSS.compute(model, batch, outputs)

    def prepare_inference_inputs(self, model, processor, inputs):
        del model, processor
        return inputs

    def get_trainable_params(self, model: nn.Module) -> List[Dict[str, Any]]:
        return [{"params": list(model.mores_adapters.parameters())}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        torch.save(
            compact_mores_state(model.mores_adapters),
            os.path.join(path, "mores_tuned.pt"),
        )
        processor.save_pretrained(path)
        metadata = {**metadata, "ft_method": self.name, "recipe": "mores_v1"}
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
        model, _ = self.prepare_model(model, processor, model_spec=model_spec)
        state = torch.load(
            os.path.join(path, "mores_tuned.pt"),
            map_location="cpu",
            weights_only=True,
        )
        load_compact_mores_state(model.mores_adapters, state)
        model.eval()

        adapter_name = os.path.basename(path)
        info = {"model_id": f"{model_spec.hf_model_id} (MoReS: {adapter_name})"}
        return model, processor, info


__all__ = [
    "MORES_CHECKPOINT_FORMAT",
    "MORES_RANK",
    "MORES_VISUAL_TOKEN_INDICES",
    "MoReSAdapter",
    "MoReSMethod",
    "build_mores_intervention_mask",
    "compact_mores_state",
    "load_compact_mores_state",
]
