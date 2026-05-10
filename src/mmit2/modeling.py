"""Shared HuggingFace model-loading helpers."""
from __future__ import annotations

import os

import torch
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLM


def _disable_hf_progress_bars() -> None:
    """Silence noisy HF model-loading progress unless the user explicitly opted in."""
    env_value = os.getenv("HF_HUB_DISABLE_PROGRESS_BARS")
    if env_value is None:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    elif env_value.strip().lower() in {"0", "false", "no", "off"}:
        return

    try:
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()
    except Exception:
        pass

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass


def load_processor(model_id: str):
    _disable_hf_progress_bars()
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

    _disable_hf_progress_bars()
    return AutoVLM.from_pretrained(model_id, **load_kwargs)
