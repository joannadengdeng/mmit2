"""Initial local-only model invocation for evaluation."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch

from vlmintune.data.datasets.base import build_prompt_inputs, load_sample_image
from vlmintune.data.types import CanonicalSample, EvalSample
from vlmintune.training.methods.base import load_processor, load_vlm
from vlmintune.training.methods.registry import build_training_method


class LocalMethod:
    """Inference with a locally loaded VLM model."""

    def __init__(self, model, processor, device=None, inference_method=None):
        self.model = model
        self.processor = processor
        self.device = device or next(model.parameters()).device
        self.inference_method = inference_method
        self.model.eval()

    @classmethod
    def from_base_model(
        cls,
        base_model_id: str,
        quantize_4bit: bool = True,
    ) -> "LocalMethod":
        processor = load_processor(base_model_id)
        model = load_vlm(
            base_model_id,
            quantize_4bit=quantize_4bit,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        return cls(model, processor)

    @classmethod
    def from_checkpoint(
        cls,
        base_model_id: str,
        checkpoint_path: str = "",
        ft_method: str = "",
        quantize_4bit: bool = True,
        **kwargs,
    ) -> "LocalMethod":
        if checkpoint_path and os.path.isdir(checkpoint_path):
            if not ft_method:
                meta_path = os.path.join(checkpoint_path, "vlmintune_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    ft_method = meta.get("ft_method", "")

            if not ft_method:
                raise ValueError(
                    f"Could not determine ft_method for checkpoint: {checkpoint_path}"
                )
            method = build_training_method(ft_method)
            model, processor, _ = method.load_for_inference(
                checkpoint_path,
                base_model_id,
                quantize_4bit=quantize_4bit,
                **kwargs,
            )
            return cls(model, processor, inference_method=method)

        return cls.from_base_model(
            base_model_id,
            quantize_4bit=quantize_4bit,
        )

    def prepare_eval_input(
        self,
        sample: EvalSample,
    ) -> Dict[str, Any]:
        cs = CanonicalSample(
            id=sample.id,
            image_path=sample.image_path,
            question=sample.question,
            metadata=sample.metadata,
        )
        return self.prepare_input(cs)

    def prepare_input(
        self,
        sample: CanonicalSample,
    ) -> Dict[str, Any]:
        image = load_sample_image(sample)

        _, inputs = build_prompt_inputs(
            self.processor,
            sample.question,
            image,
            return_tensors="pt",
        )
        if self.inference_method is not None:
            inputs = self.inference_method.prepare_inference_inputs(
                self.model,
                self.processor,
                inputs,
            )
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def generate(
        self,
        prepared: Dict[str, Any],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        with torch.no_grad():
            output = self.model.generate(
                **prepared,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        prompt_len = prepared["input_ids"].shape[1]
        response = self.processor.decode(
            output[0][prompt_len:], skip_special_tokens=True,
        )
        return response.strip()


__all__ = [
    "LocalMethod",
]
