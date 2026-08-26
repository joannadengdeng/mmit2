import os
import re
import sys

import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods import lora as lora_mod
from vlmintune.training.methods.dora import DoRAMethod
from vlmintune.training.methods.lora import (
    LANGUAGE_LORA_TARGETS,
    LoRAMethod,
    QLoRAMethod,
)


class _ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)


class _ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4)
        self.up_proj = nn.Linear(4, 4)
        self.down_proj = nn.Linear(4, 4)


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _ToyAttention()
        self.mlp = _ToyMLP()


class _ToyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(), _ToyBlock()])


class _ToyQwenVL(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel()
        self.visual = nn.Module()
        self.visual.q_proj = nn.Linear(4, 4)
        self.visual.mlp = _ToyMLP()


class _FakeLoraConfig:
    kwargs = None

    def __init__(self, **kwargs):
        _FakeLoraConfig.kwargs = kwargs


def _stub_peft(monkeypatch):
    monkeypatch.setattr(lora_mod, "LoraConfig", _FakeLoraConfig)
    monkeypatch.setattr(lora_mod, "get_peft_model", lambda model, config: model)


def test_lora_v1_uses_fixed_recipe_and_language_only_targets(monkeypatch):
    _stub_peft(monkeypatch)
    model = _ToyQwenVL()

    _, info = LoRAMethod().prepare_model(
        model,
        processor=None,
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    kwargs = _FakeLoraConfig.kwargs
    assert kwargs["r"] == 8
    assert kwargs["lora_alpha"] == 16
    assert kwargs["lora_dropout"] == 0.05
    pattern = kwargs["target_modules"]
    for target in LANGUAGE_LORA_TARGETS:
        container = "self_attn" if target in LANGUAGE_LORA_TARGETS[:4] else "mlp"
        assert re.fullmatch(
            pattern,
            f"model.language_model.layers.1.{container}.{target}",
        )
    assert not re.fullmatch(pattern, "visual.q_proj")
    assert not re.fullmatch(pattern, "visual.mlp.down_proj")
    assert not re.fullmatch(pattern, "model.language_model.lm_head")
    assert "layers=all language Transformer layers" in info


def test_qlora_v1_prepares_kbit_model_and_uses_fixed_recipe(monkeypatch):
    calls = []
    _stub_peft(monkeypatch)
    monkeypatch.setattr(
        lora_mod,
        "prepare_model_for_kbit_training",
        lambda model, **kwargs: calls.append((model, kwargs)) or model,
    )
    model = _ToyQwenVL()

    _, info = QLoRAMethod().prepare_model(
        model,
        processor=None,
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    assert calls == [
        (model, {"gradient_checkpointing_kwargs": {"use_reentrant": False}})
    ]
    assert _FakeLoraConfig.kwargs["r"] == 64
    assert _FakeLoraConfig.kwargs["lora_alpha"] == 16
    assert _FakeLoraConfig.kwargs["lora_dropout"] == 0.0
    assert "NF4 4-bit" in info
    assert "double_quant=True" in info
    assert "compute_dtype=BF16" in info
    assert "PagedAdamW8bit" in info


def test_dora_v1_is_fixed_lora_with_weight_decomposition(monkeypatch):
    _stub_peft(monkeypatch)

    DoRAMethod().prepare_model(
        _ToyQwenVL(),
        processor=None,
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    assert _FakeLoraConfig.kwargs["r"] == 8
    assert _FakeLoraConfig.kwargs["lora_alpha"] == 16
    assert _FakeLoraConfig.kwargs["lora_dropout"] == 0.05
    assert _FakeLoraConfig.kwargs["use_dora"] is True
