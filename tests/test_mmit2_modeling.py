import importlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _load_modeling_with_stubs():
    calls = {
        "disable_transformers": 0,
        "disable_hub": 0,
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

    transformers_utils_mod = types.ModuleType("transformers.utils")
    transformers_logging_mod = types.ModuleType("transformers.utils.logging")

    def _disable_transformers_progress_bar():
        calls["disable_transformers"] += 1

    transformers_logging_mod.disable_progress_bar = _disable_transformers_progress_bar

    huggingface_hub_mod = types.ModuleType("huggingface_hub")
    huggingface_hub_utils_mod = types.ModuleType("huggingface_hub.utils")

    def _disable_hub_progress_bars():
        calls["disable_hub"] += 1

    huggingface_hub_utils_mod.disable_progress_bars = _disable_hub_progress_bars
    huggingface_hub_mod.utils = huggingface_hub_utils_mod

    module_names = [
        "transformers",
        "transformers.utils",
        "transformers.utils.logging",
        "huggingface_hub",
        "huggingface_hub.utils",
        "mmit2.training.modeling",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}

    sys.modules["transformers"] = transformers_mod
    sys.modules["transformers.utils"] = transformers_utils_mod
    sys.modules["transformers.utils.logging"] = transformers_logging_mod
    sys.modules["huggingface_hub"] = huggingface_hub_mod
    sys.modules["huggingface_hub.utils"] = huggingface_hub_utils_mod
    sys.modules.pop("mmit2.training.modeling", None)

    modeling = importlib.import_module("mmit2.training.modeling")
    return modeling, calls, saved_modules


def _restore_modules(saved_modules):
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_load_processor_disables_progress_bars_by_default():
    previous_env = os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    modeling, calls, saved_modules = _load_modeling_with_stubs()

    try:
        processor = modeling.load_processor("fake/model")

        assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
        assert calls["disable_transformers"] == 1
        assert calls["disable_hub"] == 1
        assert processor["kind"] == "processor"
        assert calls["processor_loads"][0][0] == ("fake/model",)
    finally:
        _restore_modules(saved_modules)
        if previous_env is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_env


def test_load_vlm_respects_explicit_progress_bar_opt_in():
    previous_env = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    modeling, calls, saved_modules = _load_modeling_with_stubs()

    try:
        model = modeling.load_vlm("fake/model", quantize_4bit=True)

        assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "0"
        assert calls["disable_transformers"] == 0
        assert calls["disable_hub"] == 0
        assert model["kind"] == "model"
        _, kwargs = calls["model_loads"][0]
        assert kwargs["device_map"] == "auto"
        assert kwargs["trust_remote_code"] is True
        assert isinstance(kwargs["quantization_config"], dict)
        assert kwargs["quantization_config"]["load_in_4bit"] is True
    finally:
        _restore_modules(saved_modules)
        if previous_env is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = previous_env
