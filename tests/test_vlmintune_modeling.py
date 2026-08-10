import importlib
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_modeling_with_stubs():
    calls = {
        "processor_loads": [],
        "model_loads": [],
    }

    transformers_mod = types.ModuleType("transformers")

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["processor_loads"].append((args, kwargs))
            return {"kind": "processor", "args": args, "kwargs": kwargs}

    class _FakeAutoVLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["model_loads"].append((args, kwargs))
            return {"kind": "model", "args": args, "kwargs": kwargs}

    class _FakeBitsAndBytesConfig(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    transformers_mod.AutoProcessor = _FakeAutoProcessor
    transformers_mod.AutoModelForImageTextToText = _FakeAutoVLM
    transformers_mod.BitsAndBytesConfig = _FakeBitsAndBytesConfig

    module_names = [
        "transformers",
        "vlmintune.training.methods.base",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}

    sys.modules["transformers"] = transformers_mod
    sys.modules.pop("vlmintune.training.methods.base", None)

    modeling = importlib.import_module("vlmintune.training.methods.base")
    return modeling, calls, saved_modules


def restore_modules(saved_modules):
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_load_processor_uses_trust_remote_code():
    modeling, calls, saved_modules = load_modeling_with_stubs()

    try:
        processor = modeling.load_processor("fake/model")

        assert processor["kind"] == "processor"
        args, kwargs = calls["processor_loads"][0]
        assert args == ("fake/model",)
        assert kwargs["trust_remote_code"] is True
    finally:
        restore_modules(saved_modules)


def test_qwen_processor_caps_high_resolution_images():
    modeling, _, saved_modules = load_modeling_with_stubs()

    try:
        processor = types.SimpleNamespace(
            image_processor=types.SimpleNamespace(
                size=types.SimpleNamespace(longest_edge=12_845_056),
            ),
        )

        configured = modeling.configure_processor_image_budget(
            processor,
            "Qwen/Qwen2.5-VL-3B-Instruct",
        )

        assert configured is processor
        assert (
            processor.image_processor.size.longest_edge
            == modeling.QWEN25VL_IMAGE_MAX_PIXELS
            == 1_003_520
        )
    finally:
        restore_modules(saved_modules)


def test_image_budget_does_not_change_non_qwen_processors():
    modeling, _, saved_modules = load_modeling_with_stubs()

    try:
        processor = types.SimpleNamespace(
            image_processor=types.SimpleNamespace(
                size=types.SimpleNamespace(longest_edge=12_845_056),
            ),
        )

        modeling.configure_processor_image_budget(
            processor,
            "llava-hf/llava-1.5-7b-hf",
        )

        assert processor.image_processor.size.longest_edge == 12_845_056
    finally:
        restore_modules(saved_modules)


def test_load_vlm_builds_quantized_kwargs():
    modeling, calls, saved_modules = load_modeling_with_stubs()

    try:
        model = modeling.load_vlm("fake/model", quantize_4bit=True)

        assert model["kind"] == "model"
        _, kwargs = calls["model_loads"][0]
        assert kwargs["device_map"] == "auto"
        assert kwargs["trust_remote_code"] is True
        assert isinstance(kwargs["quantization_config"], dict)
        assert kwargs["quantization_config"]["load_in_4bit"] is True
        assert kwargs["quantization_config"]["bnb_4bit_compute_dtype"] is torch.bfloat16
        assert kwargs["quantization_config"]["bnb_4bit_quant_type"] == "nf4"
        assert kwargs["quantization_config"]["bnb_4bit_use_double_quant"] is True
    finally:
        restore_modules(saved_modules)


def test_load_vlm_quantized_compute_dtype_can_be_overridden():
    modeling, calls, saved_modules = load_modeling_with_stubs()

    try:
        modeling.load_vlm("fake/model", quantize_4bit=True, torch_dtype=torch.float16)

        _, kwargs = calls["model_loads"][0]
        assert kwargs["quantization_config"]["bnb_4bit_compute_dtype"] is torch.float16
    finally:
        restore_modules(saved_modules)


def test_load_vlm_uses_transformers_5_dtype_keyword_when_not_quantized():
    modeling, calls, saved_modules = load_modeling_with_stubs()

    try:
        modeling.load_vlm("fake/model", quantize_4bit=False)

        _, kwargs = calls["model_loads"][0]
        assert kwargs["dtype"] is torch.bfloat16
        assert "torch_dtype" not in kwargs
    finally:
        restore_modules(saved_modules)
