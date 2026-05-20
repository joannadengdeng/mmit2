"""MoReS: modality linear representation steering for visual tokens.

Implements a minimal LLaVA-Steering style recipe:
freeze the backbone, then inject a tiny low-rank residual steering module
after each decoder layer and apply it only to selected visual tokens.
"""
from __future__ import annotations

import json
import math
import os
import re
import types
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from vlmintune.config.model_layouts import (
    list_model_layouts,
    resolve_transformer_layers,
)
from vlmintune.training.losses.ce_loss import CrossEntropyLoss
from vlmintune.training.methods.base import TrainingMethod, load_processor, load_vlm

_ce_loss = CrossEntropyLoss()


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


_FIRST_LAST_PATTERN = re.compile(r"^f(\d+)\+l(\d+)$")
_UNIFORM_PATTERN = re.compile(r"^uniform(\d+)$")


def normalize_intervention_positions(intervention_positions: str) -> str:
    normalized = str(intervention_positions).strip().lower()
    if _FIRST_LAST_PATTERN.fullmatch(normalized) or _UNIFORM_PATTERN.fullmatch(normalized):
        return normalized
    raise ValueError(
        "MoReS intervention_positions must be like 'f4+l5' or 'uniform9'"
    )


def uniform_sample_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count == 1:
        return [0]
    if length < count:
        return list(range(length))
    interval = (length - 1) // (count - 1)
    return [i * interval for i in range(count)]


def select_visual_tokens(
    visual_mask: torch.Tensor,
    *,
    intervention_positions: str,
) -> torch.Tensor:
    normalized = normalize_intervention_positions(intervention_positions)
    fl_match = _FIRST_LAST_PATTERN.fullmatch(normalized)
    uniform_match = _UNIFORM_PATTERN.fullmatch(normalized)
    assert fl_match is not None or uniform_match is not None

    selected = torch.zeros_like(visual_mask, dtype=torch.bool)
    for row_idx in range(visual_mask.shape[0]):
        token_positions = torch.nonzero(visual_mask[row_idx], as_tuple=False).flatten()
        if token_positions.numel() == 0:
            continue

        if uniform_match:
            count = int(uniform_match.group(1))
            chosen = token_positions[uniform_sample_indices(token_positions.numel(), count)]
        else:
            first_n = int(fl_match.group(1))
            last_n = int(fl_match.group(2))
            left = token_positions[:first_n] if first_n > 0 else token_positions[:0]
            right = token_positions[-last_n:] if last_n > 0 else token_positions[:0]
            chosen = torch.cat([left, right]).unique(sorted=True)
        selected[row_idx, chosen] = True
    return selected


class MoReSLayer(nn.Module):
    """Low-rank residual steering block applied only to selected visual tokens."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden_states))


class MoReSRuntime:
    """Forward-hook runtime that applies steering after every decoder layer."""

    def __init__(
        self,
        *,
        model: nn.Module,
        layers: Sequence[nn.Module],
        steering_layers: nn.ModuleList,
        intervention_positions: str,
    ) -> None:
        self.model = model
        self.layers = list(layers)
        self.steering_layers = steering_layers
        self.intervention_positions = normalize_intervention_positions(intervention_positions)
        self.current_mask: torch.Tensor | None = None
        self.original_forward = model.forward
        self.handles = [
            layer.register_forward_hook(self._build_hook(steering_layer))
            for layer, steering_layer in zip(self.layers, self.steering_layers)
        ]
        model.forward = types.MethodType(self._patched_forward, model)

    def _patched_forward(self, model_self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        if input_ids is None and args:
            input_ids = args[0]
        self.current_mask = self._build_visual_mask(input_ids, attention_mask)
        try:
            return self.original_forward(*args, **kwargs)
        finally:
            self.current_mask = None

    def _build_visual_mask(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if input_ids is None:
            return None

        image_token_id = getattr(self.model.config, "image_token_id", None)
        video_token_id = getattr(self.model.config, "video_token_id", None)
        visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if image_token_id is not None:
            visual_mask |= input_ids == int(image_token_id)
        if video_token_id is not None:
            visual_mask |= input_ids == int(video_token_id)
        if attention_mask is not None:
            visual_mask &= attention_mask.bool()
        if not visual_mask.any():
            return None
        return select_visual_tokens(
            visual_mask,
            intervention_positions=self.intervention_positions,
        )

    def _build_hook(self, steering_layer: MoReSLayer):
        def hook(module, args, output):
            token_mask = self.current_mask
            if token_mask is None:
                return output

            hidden_states = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
                return output
            if hidden_states.shape[:2] != token_mask.shape:
                return output
            if not token_mask.any():
                return output

            updated = hidden_states.clone()
            delta = steering_layer(hidden_states[token_mask]).to(hidden_states.dtype)
            updated[token_mask] = updated[token_mask] + delta
            if isinstance(output, tuple):
                return (updated, *output[1:])
            return updated

        return hook


class MoReSMethod(TrainingMethod):
    """MoReS / LLaVA Steering style visual representation steering."""

    name = "mores"
    display_name = "MoReS (LLaVA Steering)"

    def __init__(self) -> None:
        self._last_config: dict[str, Any] = {}

    def default_config(self):
        return {
            "model_layout": "",
            "hidden_size": 0,
            "steering_rank": 1,
            "intervention_positions": "f4+l5",
        }

    def prepare_model_impl(self, model, processor, config):
        self._last_config = dict(config)
        model_layout = str(config.get("model_layout", "")).strip()
        if not model_layout:
            raise ValueError(
                "MoReS requires training.params.model_layout. "
                f"Available layouts: {list_model_layouts()}"
            )

        rank = int(config.get("steering_rank", 1))
        if rank <= 0:
            raise ValueError("MoReS requires steering_rank >= 1")

        hidden_size = int(config.get("hidden_size", 0))
        if hidden_size <= 0:
            raise ValueError("MoReS requires hidden_size >= 1 in training.params.hidden_size")

        intervention_positions = normalize_intervention_positions(
            config.get("intervention_positions", "f4+l5")
        )

        layers = resolve_transformer_layers(model, model_layout)
        freeze_backbone(model)

        steering_layers = nn.ModuleList([MoReSLayer(hidden_size, rank) for _ in layers])
        model.mores_layers = steering_layers
        model._vlmintune_mores_runtime = MoReSRuntime(
            model=model,
            layers=layers,
            steering_layers=steering_layers,
            intervention_positions=intervention_positions,
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info = (
            f"MoReS ({model_layout}): layers={len(layers)}, rank={rank}, "
            f"intervention_positions={intervention_positions}\n"
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)"
        )
        return model, info

    def compute_loss(self, model, batch, outputs):
        return _ce_loss.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        params = [param for param in model.parameters() if param.requires_grad]
        return [{"params": params}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        if not hasattr(model, "mores_layers"):
            raise ValueError("MoReS checkpoint save expected model.mores_layers to exist")

        torch.save(
            {
                "config": self._last_config,
                "state_dict": model.mores_layers.state_dict(),
            },
            os.path.join(path, "mores_tuned.pt"),
        )
        processor.save_pretrained(path)
        metadata["ft_method"] = self.name
        with open(os.path.join(path, "vlmintune_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, base_model_id, **kwargs):
        processor = load_processor(base_model_id)
        quantize_4bit = bool(kwargs.get("quantize_4bit", False))
        model = load_vlm(
            base_model_id,
            quantize_4bit=quantize_4bit,
            torch_dtype=torch.float16 if quantize_4bit else torch.bfloat16,
        )

        payload = torch.load(os.path.join(path, "mores_tuned.pt"), map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            config = {**self.default_config(), **(payload.get("config") or {})}
            state_dict = payload["state_dict"]
        else:
            config = self.default_config()
            state_dict = payload

        model, _ = self.prepare_model(model, processor, config)
        model.mores_layers.load_state_dict(state_dict)
        model.eval()

        adapter_name = os.path.basename(path)
        info = {"model_id": f"{base_model_id} (MoReS: {adapter_name})"}
        return model, processor, info
