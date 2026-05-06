"""Shared HuggingFace model-loading helpers."""
from __future__ import annotations

import torch
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLM


def load_processor(model_id: str):
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def load_vlm(
    model_id: str,
    *,
    quantize_4bit: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if quantize_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch_dtype

    return AutoVLM.from_pretrained(model_id, **load_kwargs)
