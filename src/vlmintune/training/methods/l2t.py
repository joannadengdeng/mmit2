"""Full user-prompt supervision and the standalone L2T training recipe."""
from __future__ import annotations

from pathlib import Path

import torch

from vlmintune.data.datasets.base import build_processor_images
from vlmintune.training.methods.base import TrainingMethod
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss


L2T_CHECKPOINT_NAME = "l2t_tuned.pt"
L2T_SUPERVISION_RECIPE = "l2t_full_user_prompt_v2"
_LOSS = CrossEntropyLoss()


class L2TMethod(TrainingMethod):
    """Train the language side on the complete user text and answer."""

    name = "l2t"
    display_name = "L2T (full user prompt)"

    @staticmethod
    def build_method_mask(
        *,
        sample,
        processor,
        input_ids,
        prompt_text,
        image,
        max_length,
        **_,
    ):
        question = sample.question.strip()
        question_start = prompt_text.rindex(question)
        processor_kwargs = {
            "images": build_processor_images(image),
            "return_tensors": "pt",
            "truncation": True,
            "max_length": max_length,
        }
        question_token_start = processor(
            text=prompt_text[:question_start],
            **processor_kwargs,
        )["input_ids"].size(1)
        question_token_end = processor(
            text=prompt_text[:question_start + len(question)],
            **processor_kwargs,
        )["input_ids"].size(1)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[question_token_start:question_token_end] = True
        return mask

    @staticmethod
    def trainable_modules(model, model_spec):
        if model_spec.name == "qwen25vl_3b_instruct":
            return model.model.language_model, model.lm_head, model.model.visual.merger
        if model_spec.name == "llava15_7b":
            return (
                model.model.language_model,
                model.lm_head,
                model.model.multi_modal_projector,
            )
        raise ValueError(
            "L2T supports only qwen25vl_3b_instruct and llava15_7b."
        )

    def prepare_model_impl(self, model, processor, model_spec):
        del processor
        model.requires_grad_(False)
        for module in self.trainable_modules(model, model_spec):
            module.requires_grad_(True)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info = (
            "L2T full-user-prompt supervision: language model + lm_head + visual projection "
            f"trainable; vision encoder frozen; trainable={trainable:,}/{total:,} "
            f"({100 * trainable / total:.4f}%)"
        )
        return model, info

    def preprocess_labels(self, input_ids, labels, batch_meta):
        return torch.where(batch_meta["method_mask"], input_ids, labels)

    def compute_loss(self, model, batch, outputs):
        return _LOSS.compute(model, batch, outputs)

    def get_trainable_params(self, model):
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]

    def _save_weights(self, model, path):
        state_dict = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        torch.save(state_dict, Path(path) / L2T_CHECKPOINT_NAME)

    def _restore_model(self, model, processor, model_spec, path):
        model, _ = self.prepare_model(model, processor, model_spec=model_spec)
        state_dict = torch.load(
            Path(path) / L2T_CHECKPOINT_NAME,
            map_location="cpu",
            weights_only=True,
        )
        expected = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if set(state_dict) != expected:
            raise ValueError(
                "L2T checkpoint does not match the fixed trainable state."
            )
        model.load_state_dict(state_dict, strict=False)
        return model

    def _checkpoint_metadata(self):
        return {
            "recipe": "l2t_full_sft_v1",
            "supervision_recipe": L2T_SUPERVISION_RECIPE,
        }


__all__ = [
    "L2T_CHECKPOINT_NAME",
    "L2T_SUPERVISION_RECIPE",
    "L2TMethod",
]
