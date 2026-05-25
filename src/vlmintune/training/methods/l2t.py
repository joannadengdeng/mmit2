"""L2T: supervise both instruction and response sequences."""
from __future__ import annotations

from typing import Any, Dict

from vlmintune.training.methods.lora import LoRAMethod
from vlmintune.training.methods.base import TrainingMethod


class L2TMethod(TrainingMethod):
    name = "l2t"
    display_name = "L2T (Zhou et al. 2025)"

    def __init__(self):
        self.base = LoRAMethod()
        self.last_config: Dict[str, Any] = {}

    def default_config(self):
        return self.base.default_config()

    def requires_quantization(self, config=None):
        return self.base.requires_quantization(config)

    def prepare_model_impl(self, model, processor, config):
        self.last_config = dict(config)
        return self.base.prepare_model(model, processor, config)

    def preprocess_labels(self, input_ids, labels, batch_meta=None):
        if not batch_meta:
            return labels
        instruction_mask = batch_meta.get("instruction_supervision_mask")
        if instruction_mask is None:
            return labels
        mask = instruction_mask.bool()
        attention_mask = batch_meta.get("attention_mask")
        if attention_mask is not None:
            mask &= attention_mask.bool()
        if not mask.any():
            return labels
        updated = labels.clone()
        updated[mask] = input_ids[mask]
        return updated

    def compute_loss(self, model, batch, outputs):
        return self.base.compute_loss(model, batch, outputs)

    def get_trainable_params(self, model):
        return self.base.get_trainable_params(model)

    def save_checkpoint(self, model, processor, path, metadata):
        metadata = {**metadata, "ft_method": self.name, "config": self.last_config}
        self.base.save_checkpoint(model, processor, path, metadata)

    def load_for_inference(self, path, base_model_id, **kwargs):
        return self.base.load_for_inference(path, base_model_id, **kwargs)
