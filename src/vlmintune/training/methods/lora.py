"""LoRA / QLoRA / DoRA fine-tuning methods.

These three methods share the same core logic (PEFT LoraConfig), differing only
in quantization (QLoRA) and weight decomposition (DoRA).
"""
from __future__ import annotations

import json
import os

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

from vlmintune.models.base import ModelSpec
from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.base import TrainingMethod, load_processor, load_vlm
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss

IGNORE_INDEX = -100

_ce_loss = CrossEntropyLoss()


def resolve_layer_indices(config: dict, model_spec: ModelSpec, model) -> list[int]:
    raw_range = config.get("train_layer_range")
    if not raw_range:
        return []
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        raise ValueError("train_layer_range must be a two-item list: [start, end].")

    start = int(raw_range[0])
    end = int(raw_range[1])
    layers = model_spec.get_transformer_layers(model)
    if start < 0 or end < start or end >= len(layers):
        raise ValueError(
            f"train_layer_range={list(raw_range)} is invalid for model "
            f"'{model_spec.name}' with {len(layers)} transformer layers."
        )
    return list(range(start, end + 1))


def layer_pattern_from_spec(model_spec: ModelSpec) -> str:
    return model_spec.transformer_layer_path.split(".")[-1]


class LoRAMethod(TrainingMethod):
    """Standard LoRA fine-tuning (bf16 precision)."""

    name = "lora"
    display_name = "LoRA"

    def default_config(self):
        defaults = {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [],
        }
        if self.supports_train_layer_range():
            defaults["train_layer_range"] = []
        return defaults

    def lora_kwargs(self) -> dict:
        """Extra kwargs for LoraConfig. Override in subclasses."""
        return {}

    def supports_train_layer_range(self) -> bool:
        return self.name in {"lora", "qlora"}

    def prepare_model_impl(self, model, processor, config, model_spec):
        r = int(config["lora_r"])
        alpha = int(config["lora_alpha"])
        dropout = float(config["lora_dropout"])
        targets = list(config["target_modules"])
        if not targets:
            raise ValueError(
                f"{self.display_name} requires a non-empty 'target_modules' list."
            )
        layer_indices = []
        if self.supports_train_layer_range() and config.get("train_layer_range"):
            layer_indices = resolve_layer_indices(config, model_spec, model)
        if self.requires_quantization(config):
            model = prepare_model_for_kbit_training(model)

        lora_config_kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "target_modules": targets,
            "task_type": TaskType.CAUSAL_LM,
            **self.lora_kwargs(),
        }
        if layer_indices:
            lora_config_kwargs["layers_to_transform"] = layer_indices
            lora_config_kwargs["layers_pattern"] = layer_pattern_from_spec(model_spec)
        lora_config = LoraConfig(**lora_config_kwargs)
        peft_model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        info = (
            f"{self.display_name}: r={r}, alpha={alpha}, dropout={dropout}\n"
            f"Target modules: {targets}\n"
            f"Train layer range: {config.get('train_layer_range') or 'all'}\n"
            f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
        )
        return peft_model, info

    def compute_loss(self, model, batch, outputs):
        return _ce_loss.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        processor.save_pretrained(path)
        metadata.setdefault("ft_method", self.name)
        with open(os.path.join(path, "vlmintune_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, model_name, **kwargs):
        del kwargs
        model_spec = get_model_spec(model_name)
        processor = load_processor(model_spec.hf_model_id)
        quantize_4bit = self.requires_quantization()
        model = load_vlm(
            model_spec.hf_model_id,
            quantize_4bit=quantize_4bit,
            torch_dtype=torch.float16 if quantize_4bit else torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, path)
        model.eval()
        if not quantize_4bit:
            try:
                model = model.merge_and_unload()
            except Exception:
                pass

        adapter_name = os.path.basename(path)
        info = {"model_id": f"{model_spec.hf_model_id} ({self.display_name}: {adapter_name})"}
        return model, processor, info


class QLoRAMethod(LoRAMethod):
    """QLoRA: LoRA with 4-bit quantized base model."""

    name = "qlora"
    display_name = "QLoRA"

    def requires_quantization(self, config=None):
        return True
