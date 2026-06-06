"""Qwen2.5-VL model spec."""
from __future__ import annotations

import torch.nn as nn

from vlmintune.models.base import ModelSpec, resolve_int_attr, resolve_module_sequence


class Qwen25VLSpec(ModelSpec):
    name = "qwen25vl_3b_instruct"
    hf_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    transformer_layer_path = "model.language_model.layers"

    def get_transformer_layers(self, model: nn.Module):
        return resolve_module_sequence(self, model, self.transformer_layer_path, "transformer layers")

    def get_hidden_size(self, model: nn.Module) -> int:
        return resolve_int_attr(self, model, "config.text_config.hidden_size", "hidden size")

    def get_image_token_id(self, model: nn.Module) -> int:
        return resolve_int_attr(self, model, "config.image_token_id", "image token id")


QWEN25VL_SPEC = Qwen25VLSpec()
