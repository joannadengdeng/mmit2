"""LLaVA-1.5 model spec."""
from __future__ import annotations

import torch.nn as nn

from vlmintune.models.base import ModelSpec, resolve_int_attr, resolve_module_sequence


class Llava15Spec(ModelSpec):
    name = "llava15_7b"
    hf_model_id = "llava-hf/llava-1.5-7b-hf"
    transformer_layer_path = "model.language_model.layers"

    def get_transformer_layers(self, model: nn.Module):
        return resolve_module_sequence(self, model, self.transformer_layer_path, "transformer layers")

    def get_hidden_size(self, model: nn.Module) -> int:
        return resolve_int_attr(self, model, "config.text_config.hidden_size", "hidden size")

    def get_image_token_id(self, model: nn.Module) -> int:
        return resolve_int_attr(self, model, "config.image_token_id", "image token id")


LLAVA15_SPEC = Llava15Spec()
