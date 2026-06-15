"""ReFT / LoReFT-style representation finetuning."""
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
REFT_CHECKPOINT_FORMAT = "reft_compact_v1"
REFT_DEBUG_LAYER_PREVIEW_LIMIT = 16


def first_parameter_device(module: nn.Module) -> torch.device | None:
    for param in module.parameters(recurse=True):
        return param.device
    return None


def build_reft_position_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    prefix_positions: int,
    suffix_positions: int,
) -> torch.Tensor:
    """Build a prefix/suffix intervention mask for each sequence in a batch."""
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for batch_idx in range(input_ids.size(0)):
        if attention_mask is None:
            valid_len = input_ids.size(1)
        else:
            valid_len = int(attention_mask[batch_idx].bool().sum().item())
        if valid_len <= 0:
            continue

        prefix = int(prefix_positions)
        suffix = int(suffix_positions)
        if valid_len < prefix + suffix:
            prefix = min(prefix, valid_len // 2)
            suffix = min(suffix, valid_len - prefix)

        if prefix > 0:
            mask[batch_idx, :prefix] = True
        if suffix > 0:
            mask[batch_idx, valid_len - suffix:valid_len] = True
    return mask


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
    return {
        "format": REFT_CHECKPOINT_FORMAT,
        "layers": layers,
    }


def load_compact_reft_state(adapters: nn.ModuleList, state: Dict[str, Any]) -> None:
    layers = state.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Invalid compact ReFT checkpoint: missing layers list.")
    if len(layers) != len(adapters):
        raise ValueError(
            "Invalid compact ReFT checkpoint: "
            f"expected {len(adapters)} layers, found {len(layers)}."
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
    """Freeze the backbone and learn low-rank hidden-state interventions."""

    name = "reft"
    display_name = "ReFT / LoReFT"

    def __init__(self) -> None:
        self.last_config: Dict[str, Any] = {}
        self.current_intervention_mask: Optional[torch.Tensor] = None
        self.hook_call_count = 0
        self.hook_layer_indices: List[int] = []
        self.hook_intervention_tokens: Optional[int] = None

    def default_config(self) -> Dict[str, Any]:
        return {
            "rank": 4,
            "layers": [],
            "prefix_positions": 4,
            "suffix_positions": 4,
        }

    def _selected_layer_indices(self, model, model_spec, config) -> list[int]:
        if config.get("layers"):
            indices = [int(idx) for idx in config["layers"]]
            layer_count = len(model_spec.get_transformer_layers(model))
            invalid = [idx for idx in indices if idx < 0 or idx >= layer_count]
            if invalid:
                raise ValueError(
                    f"ReFT layers {invalid} are invalid for model '{model_spec.name}' "
                    f"with {layer_count} transformer layers."
                )
            return indices
        return []

    def layer_hook(self, adapter: LoReFTAdapter, layer_index: int):
        def hook(module, args, output):
            del module, args
            if self.current_intervention_mask is None:
                return output

            hidden_states = output[0] if isinstance(output, tuple) else output
            intervention_mask = self.current_intervention_mask.to(hidden_states.device)
            self.hook_call_count += 1
            if len(self.hook_layer_indices) < REFT_DEBUG_LAYER_PREVIEW_LIMIT:
                self.hook_layer_indices.append(layer_index)
            if self.hook_intervention_tokens is None:
                self.hook_intervention_tokens = int(intervention_mask.sum().item())

            updated = hidden_states.clone()
            updated[intervention_mask] = adapter(hidden_states[intervention_mask])
            if isinstance(output, tuple):
                return (updated, *output[1:])
            return updated

        return hook

    def forward_pre_hook(self, module, args, kwargs):
        del module, args
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        attention_mask = kwargs.get("attention_mask")
        if input_ids is None:
            if inputs_embeds is None:
                raise ValueError("ReFT requires input_ids or inputs_embeds during forward.")
            input_ids = torch.zeros(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        self.current_intervention_mask = build_reft_position_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_positions=int(self.last_config["prefix_positions"]),
            suffix_positions=int(self.last_config["suffix_positions"]),
        )
        return None

    def prepare_model_impl(self, model, processor, config, model_spec):
        del processor
        self.last_config = dict(config)
        self.hook_call_count = 0
        self.hook_layer_indices = []
        self.hook_intervention_tokens = None
        rank = int(config["rank"])
        if rank <= 0:
            raise ValueError("ReFT rank must be positive.")
        layer_indices = self._selected_layer_indices(model, model_spec, config)
        if not layer_indices:
            raise ValueError("ReFT requires non-empty 'layers'.")

        for param in model.parameters():
            param.requires_grad = False

        hidden_size = model_spec.get_hidden_size(model)
        layers = list(model_spec.get_transformer_layers(model))
        adapters: list[LoReFTAdapter] = []
        for layer_index in layer_indices:
            layer = layers[layer_index]
            adapter = LoReFTAdapter(int(hidden_size), rank)
            layer_device = first_parameter_device(layer)
            if layer_device is not None:
                adapter = adapter.to(layer_device)
            adapters.append(adapter)
            layer.register_forward_hook(self.layer_hook(adapter, layer_index))

        model.reft_adapters = nn.ModuleList(adapters)
        model.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        model.vlmintuneReFTMethod = self

        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        info = (
            f"ReFT: backbone={model_spec.name}, rank={rank}, layers={layer_indices}, "
            f"positions=p{int(config['prefix_positions'])}+s{int(config['suffix_positions'])}\n"
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)"
        )
        return model, info

    def runtime_debug(self) -> Dict[str, Any]:
        return {
            "kind": "reft_runtime",
            "hook_call_count": self.hook_call_count,
            "hook_layer_indices_preview": self.hook_layer_indices,
            "hook_intervention_tokens": self.hook_intervention_tokens,
        }

    def compute_loss(self, model, batch, outputs):
        return CROSS_ENTROPY_LOSS.compute(model, batch, outputs)

    def get_trainable_params(self, model: nn.Module) -> List[Dict[str, Any]]:
        adapters = getattr(model, "reft_adapters", None)
        if adapters is None:
            return [{"params": []}]
        params = [param for param in adapters.parameters() if param.requires_grad]
        return [{"params": params}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        adapters = getattr(model, "reft_adapters", None)
        if adapters is None:
            raise ValueError("ReFT checkpoint save failed: model.reft_adapters is missing.")
        torch.save(compact_reft_state(adapters), os.path.join(path, "reft_tuned.pt"))
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

        meta_path = os.path.join(path, "vlmintune_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f) or {}
        config = dict(metadata.get("config") or self.default_config())

        model, _ = self.prepare_model(model, processor, config, model_spec=model_spec)
        state = torch.load(
            os.path.join(path, "reft_tuned.pt"),
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(state, dict) and state.get("format") == REFT_CHECKPOINT_FORMAT:
            load_compact_reft_state(model.reft_adapters, state)
        else:
            model.reft_adapters.load_state_dict(state)
        model.eval()

        adapter_name = os.path.basename(path)
        info = {"model_id": f"{model_spec.hf_model_id} (ReFT: {adapter_name})"}
        return model, processor, info


__all__ = [
    "LoReFTAdapter",
    "REFT_CHECKPOINT_FORMAT",
    "ReFTMethod",
    "build_reft_position_mask",
    "compact_reft_state",
    "load_compact_reft_state",
]
