"""L2T: supervise both instruction and response sequences."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch

from mmit2.training.methods.dora import DoRAMethod
from mmit2.training.methods.lora import LoRAMethod, QLoRAMethod
from mmit2.training.methods.base import TrainingMethod

_ALLOWED_BASE_METHODS = {"lora", "qlora", "dora"}
_BASE_METHODS = {
    "lora": LoRAMethod,
    "qlora": QLoRAMethod,
    "dora": DoRAMethod,
}


class L2TMethod(TrainingMethod):
    name = "l2t"
    display_name = "L2T (Zhou et al. 2025)"

    def __init__(self):
        self._base: Optional[TrainingMethod] = None
        self._base_name = "lora"
        self._last_config: Dict[str, Any] = {}
        self._special_token_ids: set[int] = set()

    @staticmethod
    def _collect_special_token_ids(processor: Any) -> set[int]:
        tokenizer = getattr(processor, "tokenizer", processor)
        ids: set[int] = set()
        for attr in ("all_special_ids", "additional_special_tokens_ids"):
            value = getattr(tokenizer, attr, None)
            if value:
                ids.update(int(v) for v in value if v is not None)
        for attr in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
            value = getattr(tokenizer, attr, None)
            if value is not None:
                ids.add(int(value))
        return ids

    def _get_base(self, config: Optional[Dict[str, Any]] = None) -> TrainingMethod:
        if config:
            base_name = config["base_method"]
            if base_name not in _ALLOWED_BASE_METHODS:
                raise ValueError(
                    f"L2T base_method must be one of {sorted(_ALLOWED_BASE_METHODS)}; got '{base_name}'"
                )
            if self._base is None or base_name != self._base_name:
                self._base_name = base_name
                self._base = _BASE_METHODS[base_name]()
        elif self._base is None:
            return self._get_base(self.default_config())
        return self._base  # type: ignore[return-value]

    def default_config(self):
        return {
            "base_method": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [],
        }

    def requires_quantization(self, config: Optional[Dict[str, Any]] = None):
        return self._get_base(config).requires_quantization(config)

    def _prepare_model_impl(self, model, processor, config):
        self._last_config = dict(config)
        self._special_token_ids = self._collect_special_token_ids(processor)
        return self._get_base(config).prepare_model(model, processor, config)

    def preprocess_labels(self, input_ids, labels, batch_meta=None):
        if not batch_meta:
            return labels
        prompt_mask = batch_meta.get("prompt_mask")
        if prompt_mask is None:
            return labels
        mask = prompt_mask.bool()
        attention_mask = batch_meta.get("attention_mask")
        if attention_mask is not None:
            mask &= attention_mask.bool()
        if self._special_token_ids:
            special_ids = torch.tensor(
                sorted(self._special_token_ids),
                device=input_ids.device,
            )
            mask &= ~torch.isin(input_ids, special_ids)
        if not mask.any():
            return labels
        updated = labels.clone()
        updated[mask] = input_ids[mask]
        return updated

    def compute_loss(self, model, batch, outputs):
        return self._get_base(self._last_config).compute_loss(model, batch, outputs)

    def get_trainable_params(self, model):
        return self._get_base(self._last_config).get_trainable_params(model)

    def save_checkpoint(self, model, processor, path, metadata):
        base = self._get_base(self._last_config)
        metadata = {**metadata, "ft_method": self.name, "l2t_base_method": base.name, "config": self._last_config}
        base.save_checkpoint(model, processor, path, metadata)

    def load_for_inference(self, path, base_model_id, **kwargs):
        meta_path = os.path.join(path, "mmit_meta.json")
        base_name = "lora"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            base_name = meta.get("l2t_base_method", "lora")
        if base_name not in _ALLOWED_BASE_METHODS:
            raise ValueError(
                f"L2T checkpoint expects base_method in {sorted(_ALLOWED_BASE_METHODS)}; got '{base_name}'"
            )

        base = _BASE_METHODS[base_name]()
        return base.load_for_inference(path, base_model_id, **kwargs)
