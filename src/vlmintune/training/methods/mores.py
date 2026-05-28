"""MoReS: modality linear representation steering for visual tokens."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from vlmintune.training.methods.base import TrainingMethod, load_processor, load_vlm
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss

CROSS_ENTROPY_LOSS = CrossEntropyLoss()
MORES_LOW_RANK_DIMENSION = 1
MORES_FIRST_VISUAL_TOKEN_COUNT = 4
MORES_LAST_VISUAL_TOKEN_COUNT = 5


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


class MoReSProjector(nn.Module):
    """Shared downsample/upsample projector with orthogonal rows."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        projector = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        nn.init.orthogonal_(projector.weight)
        projector = torch.nn.utils.parametrizations.orthogonal(
            projector,
            orthogonal_map="householder",
        )
        projector.parametrizations.weight.original.data = (
            projector.parametrizations.weight.original.data.to(torch.float32)
        )
        self.projector = projector

    @property
    def weight(self) -> torch.Tensor:
        return self.projector.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projector(hidden_states.to(torch.float32))


class MoReSAdapter(nn.Module):
    """Residual steering module: h + W_up(Linear(h) - W_down(h))."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.projector = MoReSProjector(hidden_size, rank)
        self.learned_source = nn.Linear(hidden_size, rank, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = hidden_states.to(torch.float32)
        projected = self.projector(base)
        steered = self.learned_source(base) - projected
        update = torch.matmul(steered, self.projector.weight)
        return (base + update).to(hidden_states.dtype)


class MoReSMethod(TrainingMethod):
    """Freeze the backbone and steer sparse visual tokens in every transformer layer."""

    name = "mores"
    display_name = "MoReS (Bi et al. 2025)"

    def __init__(self) -> None:
        self.last_config: Dict[str, Any] = {}
        self.image_token_id: Optional[int] = None
        self.current_intervention_mask: Optional[torch.Tensor] = None

    def default_config(self) -> Dict[str, Any]:
        return {}

    def layer_hook(self, adapter: MoReSAdapter):
        def hook(module, args, output):
            del module, args

            intervention_mask = self.current_intervention_mask.to(output.device)
            updated = output.clone()
            updated[intervention_mask] = adapter(output[intervention_mask])
            return updated

        return hook

    def forward_pre_hook(self, module, args, kwargs):
        del module, args
        input_ids = kwargs["input_ids"]
        intervention_mask = kwargs["intervention_mask"].to(
            device=input_ids.device,
            dtype=torch.bool,
        )

        if intervention_mask.shape != input_ids.shape[:2]:
            raise ValueError(
                "MoReS intervention_mask must have shape [batch_size, seq_len]."
            )
        self.current_intervention_mask = intervention_mask
        return None

    def prepare_model_impl(self, model, processor, config):
        del processor, config
        self.last_config = {}

        self.image_token_id = int(model.config.image_token_id)

        for param in model.parameters():
            param.requires_grad = False

        hidden_size = int(model.config.text_config.hidden_size)
        layers = list(model.model.language_model.layers)
        if not layers:
            raise ValueError("MoReS found no Qwen2.5-VL transformer layers.")

        adapters: list[MoReSAdapter] = []
        for layer in layers:
            adapter = MoReSAdapter(
                int(hidden_size),
                MORES_LOW_RANK_DIMENSION,
            )
            adapters.append(adapter)
            layer.register_forward_hook(self.layer_hook(adapter))

        if not adapters:
            raise ValueError("MoReS did not activate any transformer layers.")

        model.mores_adapters = nn.ModuleList(adapters)
        model.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        model.vlmintuneMoresMethod = self

        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        info = (
            f"MoReS: backbone=Qwen2.5-VL, rank={MORES_LOW_RANK_DIMENSION}, "
            f"positions=f{MORES_FIRST_VISUAL_TOKEN_COUNT}+l{MORES_LAST_VISUAL_TOKEN_COUNT}, "
            f"image_token_id={self.image_token_id}\n"
            f"Transformer layers: {len(layers)}\n"
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)"
        )
        return model, info

    def compute_loss(self, model, batch, outputs):
        return CROSS_ENTROPY_LOSS.compute(model, batch, outputs)

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
        torch.save(adapters.state_dict(), os.path.join(path, "mores_tuned.pt"))
        processor.save_pretrained(path)
        metadata = {**metadata, "ft_method": self.name, "config": dict(self.last_config)}
        with open(os.path.join(path, "vlmintune_meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, base_model_id, **kwargs):
        quantize_4bit = bool(kwargs.get("quantize_4bit", False))
        processor = load_processor(base_model_id)
        model = load_vlm(
            base_model_id,
            quantize_4bit=quantize_4bit,
            torch_dtype=torch.float16 if quantize_4bit else torch.bfloat16,
        )

        meta_path = os.path.join(path, "vlmintune_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f) or {}
        config = dict(metadata.get("config") or self.default_config())

        model, _ = self.prepare_model(model, processor, config)
        state = torch.load(
            os.path.join(path, "mores_tuned.pt"),
            map_location="cpu",
            weights_only=True,
        )
        model.mores_adapters.load_state_dict(state)
        model.eval()

        adapter_name = os.path.basename(path)
        info = {"model_id": f"{base_model_id} (MoReS: {adapter_name})"}
        return model, processor, info
