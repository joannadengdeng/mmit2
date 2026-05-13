import importlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _load_modeling_with_stubs():
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
        "mmit2.training.methods.base",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}

    sys.modules["transformers"] = transformers_mod
    sys.modules.pop("mmit2.training.methods.base", None)

    modeling = importlib.import_module("mmit2.training.methods.base")
    return modeling, calls, saved_modules


def _restore_modules(saved_modules):
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_load_processor_uses_trust_remote_code():
    modeling, calls, saved_modules = _load_modeling_with_stubs()

    try:
        processor = modeling.load_processor("fake/model")

        assert processor["kind"] == "processor"
        args, kwargs = calls["processor_loads"][0]
        assert args == ("fake/model",)
        assert kwargs["trust_remote_code"] is True
    finally:
        _restore_modules(saved_modules)


def test_load_vlm_builds_quantized_kwargs():
    modeling, calls, saved_modules = _load_modeling_with_stubs()

    try:
        model = modeling.load_vlm("fake/model", quantize_4bit=True)

        assert model["kind"] == "model"
        _, kwargs = calls["model_loads"][0]
        assert kwargs["device_map"] == "auto"
        assert kwargs["trust_remote_code"] is True
        assert isinstance(kwargs["quantization_config"], dict)
        assert kwargs["quantization_config"]["load_in_4bit"] is True
    finally:
        _restore_modules(saved_modules)
