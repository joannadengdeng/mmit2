"""MoReS: modality linear representation steering for visual tokens."""
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
MORES_LOW_RANK_DIMENSION = 1
MORES_FIRST_VISUAL_TOKEN_COUNT = 4
MORES_LAST_VISUAL_TOKEN_COUNT = 5
MORES_CHECKPOINT_FORMAT = "mores_compact_v1"


def build_mores_intervention_mask(
    model_config: Any,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    image_token_id = int(model_config.image_token_id)
    visual_token_indices = []
    for idx, token_id in enumerate(input_ids.tolist()):
        if token_id == image_token_id:
            visual_token_indices.append(idx)

    intervention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for idx in visual_token_indices[:MORES_FIRST_VISUAL_TOKEN_COUNT]:
        intervention_mask[idx] = True
    for idx in visual_token_indices[-MORES_LAST_VISUAL_TOKEN_COUNT:]:
        intervention_mask[idx] = True
    return intervention_mask


def first_parameter_device(module: nn.Module) -> torch.device | None:
    for param in module.parameters(recurse=True):
        return param.device
    return None


class MoReSAdapter(nn.Module):
    """Residual steering module: h + W_up(Linear(h) - W_down(h))."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
    ) -> None:
        super().__init__()
        w_down = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        nn.init.orthogonal_(w_down.weight)
        w_down = torch.nn.utils.parametrizations.orthogonal(
            w_down,
            orthogonal_map="householder",
        )
        w_down.parametrizations.weight.original.data = (
            w_down.parametrizations.weight.original.data.to(torch.float32)
        )
        self.w_down = w_down
        self.linear = nn.Linear(hidden_size, rank, dtype=torch.float32)

    @property
    def w_up(self) -> torch.Tensor:
        return self.w_down.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = hidden_states.to(torch.float32)
        w_down_h = self.w_down(base)
        linear_h = self.linear(base)
        update = torch.matmul(linear_h - w_down_h, self.w_up)
        return (base + update).to(hidden_states.dtype)


def compact_mores_state(adapters: nn.ModuleList) -> Dict[str, Any]:
    """Return a compact MoReS checkpoint without orthogonal parametrization buffers."""
    layers = []
    for adapter in adapters:
        layers.append(
            {
                "w_down_weight": adapter.w_down.weight.detach().cpu(),
                "linear_weight": adapter.linear.weight.detach().cpu(),
                "linear_bias": adapter.linear.bias.detach().cpu(),
            }
        )
    return {
        "format": MORES_CHECKPOINT_FORMAT,
        "layers": layers,
    }


def load_compact_mores_state(adapters: nn.ModuleList, state: Dict[str, Any]) -> None:
    layers = state.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Invalid compact MoReS checkpoint: missing layers list.")
    if len(layers) != len(adapters):
        raise ValueError(
            "Invalid compact MoReS checkpoint: "
            f"expected {len(adapters)} layers, found {len(layers)}."
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
    """Freeze the backbone and steer sparse visual tokens in every transformer layer."""

    name = "mores"
    display_name = "MoReS (Bi et al. 2025)"

    def __init__(self) -> None:
        self.last_config: Dict[str, Any] = {}
        self.image_token_id: Optional[int] = None
        self.current_intervention_mask: Optional[torch.Tensor] = None
        self.hook_call_count = 0
        self.hook_layer_indices: List[int] = []
        self.hook_intervention_tokens: Optional[int] = None

    def default_config(self) -> Dict[str, Any]:
        return {}

    def layer_hook(self, adapter: MoReSAdapter, layer_index: int):
        def hook(module, args, output):
            del module, args

            is_tuple_output = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple_output else output
            intervention_mask = self.current_intervention_mask.to(hidden_states.device)
            self.hook_call_count += 1
            if len(self.hook_layer_indices) < 16:
                self.hook_layer_indices.append(layer_index)
            if self.hook_intervention_tokens is None:
                self.hook_intervention_tokens = int(intervention_mask.sum().item())
            updated = hidden_states.clone()
            updated[intervention_mask] = adapter(hidden_states[intervention_mask])
            if is_tuple_output:
                return (updated,) + output[1:]
            return updated

        return hook

    def forward_pre_hook(self, module, args, kwargs):
        del args
        input_ids = kwargs["input_ids"]
        if "intervention_mask" in kwargs:
            intervention_mask = kwargs["intervention_mask"].to(
                device=input_ids.device,
                dtype=torch.bool,
            )
        else:
            intervention_mask = torch.stack(
                [
                    build_mores_intervention_mask(module.config, row)
                    for row in input_ids
                ],
                dim=0,
            ).to(device=input_ids.device)

        if intervention_mask.shape != input_ids.shape[:2]:
            raise ValueError(
                "MoReS intervention_mask must have shape [batch_size, seq_len]."
            )
        self.current_intervention_mask = intervention_mask
        return None

    def prepare_model_impl(self, model, processor, config, model_spec):
        del processor
        self.last_config = dict(config)
        self.hook_call_count = 0
        self.hook_layer_indices = []
        self.hook_intervention_tokens = None

        self.image_token_id = model_spec.get_image_token_id(model)

        for param in model.parameters():
            param.requires_grad = False

        hidden_size = model_spec.get_hidden_size(model)
        layers = list(model_spec.get_transformer_layers(model))
        if not layers:
            raise ValueError(
                f"MoReS found no transformer layers for model '{model_spec.name}'."
            )

        adapters: list[MoReSAdapter] = []
        for layer_index, layer in enumerate(layers):
            adapter = MoReSAdapter(
                int(hidden_size),
                MORES_LOW_RANK_DIMENSION,
            )
            layer_device = first_parameter_device(layer)
            if layer_device is not None:
                adapter = adapter.to(layer_device)
            adapters.append(adapter)
            layer.register_forward_hook(self.layer_hook(adapter, layer_index))

        if not adapters:
            raise ValueError("MoReS did not activate any transformer layers.")

        model.mores_adapters = nn.ModuleList(adapters)
        model.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        model.vlmintuneMoresMethod = self

        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        info = (
            f"MoReS: backbone={model_spec.name}, "
            f"rank={MORES_LOW_RANK_DIMENSION}, "
            f"positions=f{MORES_FIRST_VISUAL_TOKEN_COUNT}+l{MORES_LAST_VISUAL_TOKEN_COUNT}, "
            f"image_token_id={self.image_token_id}\n"
            f"Transformer layers: {len(layers)}\n"
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)"
        )
        return model, info

    def runtime_debug(self) -> Dict[str, Any]:
        return {
            "kind": "mores_runtime",
            "hook_call_count": self.hook_call_count,
            "hook_layer_indices_preview": self.hook_layer_indices,
            "hook_intervention_tokens": self.hook_intervention_tokens,
        }

    def compute_loss(self, model, batch, outputs):
        return CROSS_ENTROPY_LOSS.compute(model, batch, outputs)

    def prepare_inference_inputs(
        self,
        model: nn.Module,
        processor: Any,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        del model, processor
        return inputs

    def get_trainable_params(self, model: nn.Module) -> List[Dict[str, Any]]:
        adapters = getattr(model, "mores_adapters", None)
        if adapters is None:
            return [{"params": []}]
        params = [param for param in adapters.parameters() if param.requires_grad]
        return [{"params": params}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        adapters = getattr(model, "mores_adapters", None)
        if adapters is None:
            raise ValueError("MoReS checkpoint save failed: model.mores_adapters is missing.")
        torch.save(compact_mores_state(adapters), os.path.join(path, "mores_tuned.pt"))
        processor.save_pretrained(path)
        metadata = {**metadata, "ft_method": self.name, "config": dict(self.last_config)}
        with open(os.path.join(path, "vlmintune_meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, model_name, **kwargs):
        quantize_4bit = bool(kwargs.get("quantize_4bit", False))
        model_spec = get_model_spec(model_name)
        processor = load_processor(model_spec.hf_model_id)
        model = load_vlm(
            model_spec.hf_model_id,
            quantize_4bit=quantize_4bit,
            torch_dtype=torch.float16 if quantize_4bit else torch.bfloat16,
        )

        meta_path = os.path.join(path, "vlmintune_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f) or {}
        config = dict(metadata.get("config") or self.default_config())

        model, _ = self.prepare_model(model, processor, config, model_spec=model_spec)
        state = torch.load(
            os.path.join(path, "mores_tuned.pt"),
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(state, dict) and state.get("format") == MORES_CHECKPOINT_FORMAT:
            load_compact_mores_state(model.mores_adapters, state)
        else:
            model.mores_adapters.load_state_dict(state)
        model.eval()

        adapter_name = os.path.basename(path)
        info = {"model_id": f"{model_spec.hf_model_id} (MoReS: {adapter_name})"}
        return model, processor, info
