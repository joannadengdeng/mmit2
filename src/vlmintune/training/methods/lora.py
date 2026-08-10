"""Fixed LoRA and QLoRA recipes for the initial vlmintune release."""
from __future__ import annotations

import json
import os
import re

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.base import TrainingMethod, load_processor, load_vlm
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss


LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

QLORA_R = 64
QLORA_ALPHA = 16
QLORA_DROPOUT = 0.0

LANGUAGE_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_ce_loss = CrossEntropyLoss()


def language_lora_target_pattern(model_spec) -> str:
    """Match only the seven supported linear modules inside language blocks."""

    layer_path = re.escape(model_spec.transformer_layer_path)
    attention_targets = "|".join(LANGUAGE_LORA_TARGETS[:4])
    mlp_targets = "|".join(LANGUAGE_LORA_TARGETS[4:])
    return (
        rf"{layer_path}\.\d+\."
        rf"(?:self_attn\.(?:{attention_targets})|mlp\.(?:{mlp_targets}))"
    )


class LoRAMethod(TrainingMethod):
    """Fixed v1 LoRA over every language Transformer block."""

    name = "lora"
    display_name = "LoRA"
    rank = LORA_R
    alpha = LORA_ALPHA
    dropout = LORA_DROPOUT

    def lora_kwargs(self) -> dict:
        return {}

    def prepare_model_impl(self, model, processor, model_spec):
        del processor
        target_pattern = language_lora_target_pattern(model_spec)
        if self.requires_quantization():
            model = prepare_model_for_kbit_training(
                model,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=target_pattern,
            task_type=TaskType.CAUSAL_LM,
            **self.lora_kwargs(),
        )
        peft_model = get_peft_model(model, lora_config)

        trainable = sum(param.numel() for param in peft_model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in peft_model.parameters())
        recipe = (
            f"{self.display_name} v1: r={self.rank}, alpha={self.alpha}, "
            f"dropout={self.dropout}, layers=all language Transformer layers\n"
            f"Targets: {list(LANGUAGE_LORA_TARGETS)}\n"
            f"Target scope: {model_spec.transformer_layer_path}"
        )
        if self.requires_quantization():
            recipe += (
                "\nQuantization: NF4 4-bit, double_quant=True, compute_dtype=BF16"
                "\nOptimizer: PagedAdamW8bit"
            )
        info = (
            f"{recipe}\n"
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )
        return peft_model, info

    def compute_loss(self, model, batch, outputs):
        return _ce_loss.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        params = [param for param in model.parameters() if param.requires_grad]
        return [{"params": params}]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        processor.save_pretrained(path)
        metadata = {**metadata, "ft_method": self.name}
        with open(os.path.join(path, "vlmintune_meta.json"), "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, model_name, **kwargs):
        del kwargs
        model_spec = get_model_spec(model_name)
        processor = load_processor(model_spec.hf_model_id)
        quantize_4bit = self.requires_quantization()
        model = load_vlm(
            model_spec.hf_model_id,
            quantize_4bit=quantize_4bit,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, path)
        model.eval()
        if not quantize_4bit:
            model = model.merge_and_unload()

        adapter_name = os.path.basename(path)
        info = {"model_id": f"{model_spec.hf_model_id} ({self.display_name}: {adapter_name})"}
        return model, processor, info


class QLoRAMethod(LoRAMethod):
    """Fixed v1 QLoRA: NF4 + double quantization + BF16 + paged AdamW 8-bit."""

    name = "qlora"
    display_name = "QLoRA"
    rank = QLORA_R
    alpha = QLORA_ALPHA
    dropout = QLORA_DROPOUT

    def requires_quantization(self):
        return True
