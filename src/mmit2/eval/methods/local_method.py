"""LocalMethod — run inference with a locally loaded model (base or PEFT checkpoint).

Used for evaluating trained models on TextVQA-style runs.

Usage:
    method = LocalMethod.from_checkpoint(
        base_model_id="llava-hf/llava-1.5-7b-hf",
        checkpoint_path="output/qlora/final",
        ft_method="qlora",
    )
    # Or just a base model:
    method = LocalMethod(model, processor)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch
from PIL import Image

from mmit2.data.types import CanonicalSample, EvalSample, Turn
from mmit2.eval.methods.base import Method
from mmit2.training.methods.base import load_processor, load_vlm
from mmit2.training.registry import build_training_method

_SHORT_ANSWER_INSTRUCTION = "Answer with a single short answer only. Do not use a full sentence."


def _build_eval_question(question: str) -> str:
    question = question.strip() or "Describe this image."
    return f"{question}\n{_SHORT_ANSWER_INSTRUCTION}"


class LocalMethod(Method):
    """Inference with a locally loaded VLM model.

    Optimized for eval: short max_new_tokens, greedy decoding.
    """

    def __init__(self, model, processor, device=None):
        self.model = model
        self.processor = processor
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @classmethod
    def from_base_model(
        cls,
        base_model_id: str,
        quantize_4bit: bool = True,
    ) -> "LocalMethod":
        """Load an unfine-tuned base model for baseline evaluation."""
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
        """Load a base model + optional PEFT checkpoint."""
        if checkpoint_path and os.path.isdir(checkpoint_path):
            if not ft_method:
                meta_path = os.path.join(checkpoint_path, "mmit_meta.json")
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
            return cls(model, processor)

        return cls.from_base_model(
            base_model_id,
            quantize_4bit=quantize_4bit,
        )

    def prepare_eval_input(
        self,
        sample: EvalSample,
        image_root: str = "",
    ) -> Dict[str, Any]:
        cs = CanonicalSample(
            id=sample.id,
            image_path=sample.image_path,
            turns=[Turn(role="human", content=_build_eval_question(sample.question))],
            metadata=sample.metadata,
        )
        return self.prepare_input(cs, image_root=image_root)

    def prepare_input(
        self,
        sample: CanonicalSample,
        image_root: str = "",
    ) -> Dict[str, Any]:
        """Prepare input for generation."""
        image = None
        if sample.image_path:
            pil = (sample.metadata or {}).get("_pil_image")
            if pil is not None:
                image = pil.convert("RGB")
            else:
                img_path = os.path.join(image_root, sample.image_path) if image_root else sample.image_path
                if os.path.isfile(img_path):
                    image = Image.open(img_path).convert("RGB")

        question = ""
        for turn in sample.turns:
            if turn.role == "human":
                question = turn.content
                break
        if not question:
            question = "Describe this image."

        if image is not None:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}]
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": question},
            ]}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        images = [image] if image is not None else None
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}

    def generate(
        self,
        prepared: Dict[str, Any],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response. Default max_new_tokens=32 for short VQA answers."""
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
