"""Fixed Single Adapter recipe inspired by VL-Adapter for Qwen2.5-VL."""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.base import TrainingMethod, load_processor, load_vlm
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss


CROSS_ENTROPY_LOSS = CrossEntropyLoss()
VL_ADAPTER_CHECKPOINT_NAME = "vl_adapter_tuned.pt"
VL_ADAPTER_REDUCTION_FACTOR = 8
VL_ADAPTER_SUPPORTED_MODEL = "qwen25vl_3b_instruct"


class VLAdapterBlock(nn.Module):
    """Fixed residual bottleneck: x + Up(GELU-new(Down(x)))."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        bottleneck_size = int(hidden_size) // VL_ADAPTER_REDUCTION_FACTOR
        self.down = nn.Linear(int(hidden_size), bottleneck_size)
        self.activation = nn.GELU(approximate="tanh")
        self.up = nn.Linear(bottleneck_size, int(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.up(self.activation(self.down(hidden_states)))


class VLAdapterLayer(nn.Module):
    """Independent attention and FFN adapters for one language layer."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = VLAdapterBlock(hidden_size)
        self.mlp = VLAdapterBlock(hidden_size)


class VLAdapterMethod(TrainingMethod):
    """Qwen2.5-VL-only Single Adapter v1."""

    name = "vl_adapter"
    display_name = "Single Adapter (VL-Adapter style)"

    @staticmethod
    def attention_hook(adapter: VLAdapterBlock):
        def hook(module, args, output):
            del module, args
            return (adapter(output[0]), *output[1:])

        return hook

    @staticmethod
    def mlp_hook(adapter: VLAdapterBlock):
        def hook(module, args, output):
            del module, args
            return adapter(output)

        return hook

    def prepare_model_impl(self, model, processor, model_spec):
        del processor
        if model_spec.name != VL_ADAPTER_SUPPORTED_MODEL:
            raise ValueError(
                "Single Adapter (VL-Adapter style) v1 only supports Qwen2.5-VL."
            )
        if hasattr(model, "vl_adapter_layers"):
            raise RuntimeError("Single Adapter v1 is already installed on this model.")

        layers = list(model_spec.get_transformer_layers(model))
        layer_norms: list[nn.Module] = []
        for layer in layers:
            layer_norms.extend([layer.input_layernorm, layer.post_attention_layernorm])

        final_norm = model.model.language_model.norm
        visual_merger = model.model.visual.merger
        model.requires_grad_(False)

        hidden_size = int(model_spec.get_hidden_size(model))
        adapter_layers: list[VLAdapterLayer] = []
        for layer in layers:
            reference = next(layer.parameters())
            adapter_layers.append(
                VLAdapterLayer(hidden_size).to(
                    device=reference.device,
                    dtype=reference.dtype,
                )
            )

        model.vl_adapter_layers = nn.ModuleList(adapter_layers)
        for layer, adapters in zip(layers, model.vl_adapter_layers):
            layer.self_attn.register_forward_hook(
                self.attention_hook(adapters.attention)
            )
            layer.mlp.register_forward_hook(self.mlp_hook(adapters.mlp))

        for module in [*layer_norms, final_norm, visual_merger]:
            module.requires_grad_(True)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info = (
            "Single Adapter (VL-Adapter style) v1: "
            f"backbone={model_spec.name}, layers={len(layers)}, "
            f"reduction_factor={VL_ADAPTER_REDUCTION_FACTOR}, activation=gelu_new, "
            "train=adapters+LayerNorm+visual_merger\n"
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)"
        )
        return model, info

    def compute_loss(self, model, batch, outputs):
        return CROSS_ENTROPY_LOSS.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        state_dict = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        torch.save(state_dict, os.path.join(path, VL_ADAPTER_CHECKPOINT_NAME))
        processor.save_pretrained(path)
        metadata = {
            **metadata,
            "ft_method": self.name,
            "recipe": "single_vl_adapter_v1",
        }
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
        state_dict = torch.load(
            os.path.join(path, VL_ADAPTER_CHECKPOINT_NAME),
            map_location="cpu",
            weights_only=True,
        )
        expected = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        if set(state_dict) != expected:
            raise ValueError(
                "Single Adapter checkpoint does not match the fixed v1 trainable state."
            )
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        checkpoint_name = os.path.basename(os.path.normpath(path))
        info = {
            "model_id": (
                f"{model_spec.hf_model_id} "
                f"(Single Adapter, VL-Adapter style: {checkpoint_name})"
            )
        }
        return model, processor, info


__all__ = [
    "VL_ADAPTER_CHECKPOINT_NAME",
    "VL_ADAPTER_REDUCTION_FACTOR",
    "VL_ADAPTER_SUPPORTED_MODEL",
    "VLAdapterBlock",
    "VLAdapterLayer",
    "VLAdapterMethod",
]
