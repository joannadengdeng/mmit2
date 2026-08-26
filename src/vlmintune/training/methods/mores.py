"""MoReS recipe for sparse visual-token representation steering."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from vlmintune.training.methods.base import TrainingMethod
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss


CROSS_ENTROPY_LOSS = CrossEntropyLoss()
MORES_RANK = 1
MORES_VISUAL_TOKEN_INDICES = (1, 2, 3, 4, -5, -4, -3, -2, -1)
MORES_CHECKPOINT_FORMAT = "mores"


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
        raise ValueError("Checkpoint is not a MoReS checkpoint.")
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
    """Rank one, visual f4+l5, and every language layer."""

    name = "mores"
    display_name = "MoReS"

    @staticmethod
    def build_method_mask(*, model_config, input_ids, **_):
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

    def __init__(self) -> None:
        self.current_intervention_mask: Optional[torch.Tensor] = None

    def layer_hook(self, adapter: MoReSAdapter):
        def hook(module, args, output):
            if self.current_intervention_mask is None:
                return output

            is_tuple_output = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple_output else output
            mask = self.current_intervention_mask.to(hidden_states.device)

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
            [
                self.build_method_mask(
                    model_config=module.config,
                    input_ids=row,
                )
                for row in input_ids
            ],
            dim=0,
        )
        return None

    def prepare_model_impl(self, model, processor, model_spec):
        self.current_intervention_mask = None

        for parameter in model.parameters():
            parameter.requires_grad = False

        hidden_size = int(model_spec.get_hidden_size(model))
        layers = list(model_spec.get_transformer_layers(model))
        adapters: list[MoReSAdapter] = []
        for layer in layers:
            adapter = MoReSAdapter(hidden_size).to(first_parameter_device(layer))
            adapters.append(adapter)
            layer.register_forward_hook(self.layer_hook(adapter))

        model.mores_adapters = nn.ModuleList(adapters)
        model.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info = (
            f"MoReS: backbone={model_spec.name}, rank={MORES_RANK}, "
            f"visual positions=f4+l5 {list(MORES_VISUAL_TOKEN_INDICES)}, "
            f"layers=all ({len(layers)}), init=random, scale=1.0\n"
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)"
        )
        return model, info

    def build_forward_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.current_intervention_mask = batch["method_mask"].bool()
        return super().build_forward_batch(batch)

    def compute_loss(self, model, batch, outputs):
        return CROSS_ENTROPY_LOSS.compute(model, batch, outputs)

    def get_trainable_params(self, model: nn.Module) -> List[Dict[str, Any]]:
        return [{"params": list(model.mores_adapters.parameters())}]

    def _save_weights(self, model, path):
        torch.save(
            compact_mores_state(model.mores_adapters),
            os.path.join(path, "mores_tuned.pt"),
        )

    def _restore_model(self, model, processor, model_spec, path):
        model, _ = self.prepare_model(model, processor, model_spec=model_spec)
        state = torch.load(
            os.path.join(path, "mores_tuned.pt"),
            map_location="cpu",
            weights_only=True,
        )
        load_compact_mores_state(model.mores_adapters, state)
        return model

    def _checkpoint_metadata(self):
        return {"recipe": MORES_CHECKPOINT_FORMAT}


__all__ = [
    "MORES_CHECKPOINT_FORMAT",
    "MORES_RANK",
    "MORES_VISUAL_TOKEN_INDICES",
    "MoReSAdapter",
    "MoReSMethod",
    "compact_mores_state",
    "load_compact_mores_state",
]
