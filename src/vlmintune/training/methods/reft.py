"""ReFT / LoReFT-style representation finetuning."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from vlmintune.training.methods.base import TrainingMethod
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss

CROSS_ENTROPY_LOSS = CrossEntropyLoss()
REFT_RANK = 4
REFT_PREFIX_POSITIONS = 4
REFT_SUFFIX_POSITIONS = 4
REFT_CHECKPOINT_FORMAT = "reft_tied_rank4_p4_s4_all_layers_v1"


def first_parameter_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


class LoReFTAdapter(nn.Module):
    """Low-rank linear subspace intervention: h + R^T(W h + b - R h)."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        projection = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        nn.init.orthogonal_(projection.weight)
        projection = torch.nn.utils.parametrizations.orthogonal(
            projection,
            orthogonal_map="householder",
        )
        projection.parametrizations.weight.original.data = (
            projection.parametrizations.weight.original.data.to(torch.float32)
        )
        self.projection = projection
        self.source = nn.Linear(hidden_size, rank, dtype=torch.float32)

    @property
    def r(self) -> torch.Tensor:
        return self.projection.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = hidden_states.to(torch.float32)
        projected_base = self.projection(base)
        projected_source = self.source(base)
        update = torch.matmul(projected_source - projected_base, self.r)
        return (base + update).to(hidden_states.dtype)


def compact_reft_state(adapters: nn.ModuleList) -> Dict[str, Any]:
    layers = []
    for adapter in adapters:
        layers.append(
            {
                "projection_weight": adapter.projection.weight.detach().cpu(),
                "source_weight": adapter.source.weight.detach().cpu(),
                "source_bias": adapter.source.bias.detach().cpu(),
            }
        )
    return {"format": REFT_CHECKPOINT_FORMAT, "layers": layers}


def load_compact_reft_state(adapters: nn.ModuleList, state: Dict[str, Any]) -> None:
    if state.get("format") != REFT_CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not the fixed tied ReFT v1 format.")
    layers = state["layers"]
    if len(layers) != len(adapters):
        raise ValueError(
            f"ReFT layer count mismatch: {len(layers)} checkpoint layers for "
            f"{len(adapters)} model layers."
        )
    with torch.no_grad():
        for adapter, layer_state in zip(adapters, layers):
            adapter.projection.weight = layer_state["projection_weight"].to(
                device=adapter.projection.weight.device,
                dtype=adapter.projection.weight.dtype,
            )
            adapter.source.weight.copy_(
                layer_state["source_weight"].to(
                    device=adapter.source.weight.device,
                    dtype=adapter.source.weight.dtype,
                )
            )
            adapter.source.bias.copy_(
                layer_state["source_bias"].to(
                    device=adapter.source.bias.device,
                    dtype=adapter.source.bias.dtype,
                )
            )


class ReFTMethod(TrainingMethod):
    name = "reft"
    display_name = "ReFT / LoReFT"

    @staticmethod
    def build_method_mask(*, input_ids, prompt_len, **_):
        """Select the fixed first four and last four positions of one prompt."""
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        prefix = min(REFT_PREFIX_POSITIONS, prompt_len // 2)
        suffix = min(REFT_SUFFIX_POSITIONS, prompt_len - prefix)
        mask[:prefix] = True
        mask[prompt_len - suffix:prompt_len] = True
        return mask

    def __init__(self) -> None:
        self.current_intervention_mask: Optional[torch.Tensor] = None

    def layer_hook(self, adapter: LoReFTAdapter):
        def hook(module, args, output):
            del module, args
            if self.current_intervention_mask is None:
                return output

            hidden_states = output[0] if isinstance(output, tuple) else output
            intervention_mask = self.current_intervention_mask.to(hidden_states.device)

            updated = hidden_states.clone()
            updated[intervention_mask] = adapter(hidden_states[intervention_mask])
            if isinstance(output, tuple):
                return (updated, *output[1:])
            return updated

        return hook

    def forward_pre_hook(self, module, args, kwargs):
        del args
        if module.training:
            return None

        past_key_values = kwargs.get("past_key_values")
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            self.current_intervention_mask = None
            return None

        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        if attention_mask.dim() == 2:
            prompt_lengths = attention_mask.sum(dim=1)
        else:
            prompt_lengths = torch.full(
                (input_ids.size(0),),
                input_ids.size(1),
                dtype=torch.long,
                device=input_ids.device,
            )
        self.current_intervention_mask = torch.stack(
            [
                self.build_method_mask(
                    input_ids=row,
                    prompt_len=int(prompt_length.item()),
                )
                for row, prompt_length in zip(input_ids, prompt_lengths)
            ],
            dim=0,
        )
        return None

    def prepare_model_impl(self, model, processor, model_spec):
        del processor
        self.current_intervention_mask = None

        for param in model.parameters():
            param.requires_grad = False

        hidden_size = model_spec.get_hidden_size(model)
        layers = list(model_spec.get_transformer_layers(model))
        adapters: list[LoReFTAdapter] = []
        for layer in layers:
            adapter = LoReFTAdapter(int(hidden_size), REFT_RANK)
            adapter = adapter.to(first_parameter_device(layer))
            adapters.append(adapter)
            layer.register_forward_hook(self.layer_hook(adapter))

        model.reft_adapters = nn.ModuleList(adapters)
        model.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)

        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        info = (
            f"ReFT v1: backbone={model_spec.name}, rank={REFT_RANK}, "
            f"layers=all ({len(layers)}), tied positions, "
            f"prompt positions=p{REFT_PREFIX_POSITIONS}+s{REFT_SUFFIX_POSITIONS}\n"
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)"
        )
        return model, info

    def build_forward_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.current_intervention_mask = batch["method_mask"].bool()
        return super().build_forward_batch(batch)

    def prepare_inference_inputs(self, model, processor, inputs):
        del model, processor
        return {**inputs, "use_cache": True}

    def compute_loss(self, model, batch, outputs):
        return CROSS_ENTROPY_LOSS.compute(model, batch, outputs)

    def get_trainable_params(self, model: nn.Module) -> List[Dict[str, Any]]:
        params = [
            param for param in model.reft_adapters.parameters() if param.requires_grad
        ]
        return [{"params": params}]

    def _save_weights(self, model, path):
        torch.save(
            compact_reft_state(model.reft_adapters),
            os.path.join(path, "reft_tuned.pt"),
        )

    def _restore_model(self, model, processor, model_spec, path):
        model, _ = self.prepare_model(model, processor, model_spec=model_spec)
        state = torch.load(
            os.path.join(path, "reft_tuned.pt"),
            map_location="cpu",
            weights_only=True,
        )
        load_compact_reft_state(model.reft_adapters, state)
        return model

    def _checkpoint_metadata(self):
        return {"recipe": REFT_CHECKPOINT_FORMAT}


__all__ = [
    "LoReFTAdapter",
    "REFT_CHECKPOINT_FORMAT",
    "REFT_PREFIX_POSITIONS",
    "REFT_RANK",
    "REFT_SUFFIX_POSITIONS",
    "ReFTMethod",
    "compact_reft_state",
    "load_compact_reft_state",
]
