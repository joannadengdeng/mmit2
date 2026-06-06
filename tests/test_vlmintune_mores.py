import os
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.config.training_config import load_config_dict
from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.mores import MoReSMethod
from vlmintune.training.methods.registry import list_training_methods


class _FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 42}[token]

    def get_added_vocab(self):
        return {"<|image_pad|>": 42}


class _FakeProcessor:
    def __init__(self):
        self.image_token_id = 42
        self.tokenizer = _FakeTokenizer()


class _ToyBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.linear.weight)

    def forward(self, hidden_states):
        return self.linear(hidden_states)


class _ToyLanguageModel(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(hidden_size) for _ in range(num_layers)])


class _ToyQwenVL(nn.Module):
    def __init__(self, hidden_size: int = 4, vocab_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=hidden_size),
            image_token_id=42,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            self.embed_tokens.weight.zero_()
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel(hidden_size, num_layers)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        for layer in self.model.language_model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _ToyLlava(nn.Module):
    def __init__(self, hidden_size: int = 4, vocab_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=hidden_size),
            image_token_id=32000,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            self.embed_tokens.weight.zero_()
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel(hidden_size, num_layers)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        for layer in self.model.language_model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_mores_registers_as_built_in_training_method():
    assert "mores" in list_training_methods()


def test_mores_prepare_model_freezes_backbone_and_adds_adapters():
    model = _ToyQwenVL(num_layers=2)
    method = MoReSMethod()

    prepared_model, info = method.prepare_model(
        model,
        _FakeProcessor(),
        {},
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    assert prepared_model is model
    assert hasattr(model, "mores_adapters")
    assert len(model.mores_adapters) == 2
    assert "backbone=qwen25vl_3b_instruct" in info
    assert "Transformer layers: 2" in info
    assert "positions=f4+l5" in info

    base_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("mores_adapters")
    ]
    assert base_params
    assert all(not param.requires_grad for param in base_params)
    assert all(param.requires_grad for param in model.mores_adapters.parameters())
    assert all(
        next(adapter.parameters()).device == next(layer.parameters()).device
        for adapter, layer in zip(model.mores_adapters, model.model.language_model.layers)
    )


def test_mores_only_steers_visual_tokens_during_forward():
    model = _ToyQwenVL(num_layers=1)
    method = MoReSMethod()
    method.prepare_model(
        model,
        _FakeProcessor(),
        {},
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    adapter = model.mores_adapters[0]
    with torch.no_grad():
        adapter.w_down.weight.zero_()
        adapter.w_down.weight[0, 0] = 1.0
        adapter.linear.weight.zero_()
        adapter.linear.bias.fill_(2.0)

    batch = {
        "input_ids": torch.tensor([[3, 42, 5, 42]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "intervention_mask": torch.tensor([[False, True, False, False]]),
    }
    forward_batch = method.build_forward_batch(batch)
    output = model(**forward_batch)

    assert output.shape == (1, 4, 4)
    assert torch.allclose(output[0, 0], torch.zeros(4))
    assert torch.allclose(output[0, 2], torch.zeros(4))
    assert not torch.allclose(output[0, 1], torch.zeros(4))
    assert torch.allclose(output[0, 3], torch.zeros(4))


def test_mores_build_forward_batch_keeps_intervention_mask():
    method = MoReSMethod()
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "instruction_supervision_mask": torch.tensor([[False, True]]),
        "intervention_mask": torch.tensor([[True, False]]),
    }

    forward_batch = method.build_forward_batch(batch)

    assert torch.equal(forward_batch["intervention_mask"], batch["intervention_mask"])
    assert "instruction_supervision_mask" not in forward_batch

def test_mores_supports_llava_layout():
    model = _ToyLlava(num_layers=2)
    method = MoReSMethod()

    prepared_model, info = method.prepare_model(
        model,
        _FakeProcessor(),
        {},
        model_spec=get_model_spec("llava15_7b"),
    )

    assert prepared_model is model
    assert hasattr(model, "mores_adapters")
    assert len(model.mores_adapters) == 2
    assert "backbone=llava15_7b" in info


def test_mores_config_accepts_model_name_only():
    cfg = load_config_dict(
        {
            "model": {"name": "qwen25vl_3b_instruct"},
            "experiment": {"name": "demo"},
            "training": {"ft_method": "mores"},
            "data": {"dataset_name": "lmms-lab/textvqa", "split": "train"},
        }
    )

    assert cfg.model.name == "qwen25vl_3b_instruct"
    assert cfg.training.params == {}
