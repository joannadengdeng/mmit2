import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods import lora as lora_mod
from vlmintune.training.methods.lora import LoRAMethod, QLoRAMethod


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)


class _ToyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(), _ToyBlock(), _ToyBlock()])


class _ToyQwenVL(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel()


class _FakeLoraConfig:
    kwargs = None

    def __init__(self, **kwargs):
        _FakeLoraConfig.kwargs = kwargs


def test_lora_train_layer_range_is_passed_to_peft(monkeypatch):
    monkeypatch.setattr(lora_mod, "LoraConfig", _FakeLoraConfig)
    monkeypatch.setattr(lora_mod, "get_peft_model", lambda model, config: model)
    model = _ToyQwenVL()

    _, info = LoRAMethod().prepare_model(
        model,
        processor=None,
        config={
            "lora_r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "train_layer_range": [1, 2],
        },
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    assert _FakeLoraConfig.kwargs["layers_to_transform"] == [1, 2]
    assert _FakeLoraConfig.kwargs["layers_pattern"] == "layers"
    assert "Train layer range: [1, 2]" in info


def test_qlora_prepares_model_for_kbit_training(monkeypatch):
    calls = []
    monkeypatch.setattr(lora_mod, "LoraConfig", _FakeLoraConfig)
    monkeypatch.setattr(lora_mod, "get_peft_model", lambda model, config: model)
    monkeypatch.setattr(
        lora_mod,
        "prepare_model_for_kbit_training",
        lambda model: calls.append(model) or model,
    )
    model = _ToyQwenVL()

    QLoRAMethod().prepare_model(
        model,
        processor=None,
        config={
            "lora_r": 2,
            "lora_alpha": 2,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
        },
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    assert calls == [model]
