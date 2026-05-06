"""LoRA / QLoRA / DoRA fine-tuning methods.

These three methods share the same core logic (PEFT LoraConfig), differing only
in quantization (QLoRA) and weight decomposition (DoRA).
"""
from __future__ import annotations

import json
import os

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from mmit2.modeling import load_processor, load_vlm
from mmit2.training.methods.base import TrainingMethod
from mmit2.training.losses.ce_loss import CrossEntropyLoss

IGNORE_INDEX = -100

_ce_loss = CrossEntropyLoss()


class LoRAMethod(TrainingMethod):
    """Standard LoRA fine-tuning (bf16 precision)."""

    name = "lora"
    display_name = "LoRA"

    def default_config(self):
        return {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [],
        }

    def _lora_kwargs(self) -> dict:
        """Extra kwargs for LoraConfig. Override in subclasses."""
        return {}

    def _prepare_model_impl(self, model, processor, config):
        r = int(config["lora_r"])
        alpha = int(config["lora_alpha"])
        dropout = float(config["lora_dropout"])
        targets = list(config["target_modules"])
        if not targets:
            raise ValueError(
                f"{self.display_name} requires a non-empty 'target_modules' list."
            )

        lora_config = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
            **self._lora_kwargs(),
        )
        try:
            peft_model = get_peft_model(model, lora_config)
        except ImportError as exc:
            if "torchao" in str(exc).lower():
                raise ImportError(
                    "LoRA adapter injection failed because PEFT detected an incompatible 'torchao' "
                    "installation in the environment. This project does not require torchao for the "
                    "current LoRA path. In Colab, the simplest fix is:\n"
                    "pip uninstall -y torchao\n"
                    "and then rerun the command."
                ) from exc
            raise

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        info = (
            f"{self.display_name}: r={r}, alpha={alpha}, dropout={dropout}\n"
            f"Target modules: {targets}\n"
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
        metadata["ft_method"] = self.name
        with open(os.path.join(path, "mmit_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, base_model_id, **kwargs):
        processor = load_processor(base_model_id)
        model = load_vlm(base_model_id, quantize_4bit=True, torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, path)
        model.eval()
        try:
            model = model.merge_and_unload()
        except Exception:
            pass

        adapter_name = os.path.basename(path)
        info = {"model_id": f"{base_model_id} ({self.display_name}: {adapter_name})"}
        return model, processor, info


class QLoRAMethod(LoRAMethod):
    """QLoRA: LoRA with 4-bit quantized base model."""

    name = "qlora"
    display_name = "QLoRA"

    def requires_quantization(self, config=None):
        return True
