import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.reft import (
    REFT_CHECKPOINT_FORMAT,
    LoReFTAdapter,
    ReFTMethod,
    build_reft_position_mask,
    compact_reft_state,
    load_compact_reft_state,
)
from vlmintune.training.methods.registry import list_training_methods


class _FakeProcessor:
    pass


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
    def __init__(self, hidden_size: int = 4, vocab_size: int = 64, num_layers: int = 2):
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


def test_reft_registers_as_built_in_training_method():
    assert "reft" in list_training_methods()


def test_reft_position_mask_handles_short_sequences():
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    mask = build_reft_position_mask(
        input_ids,
        attention_mask,
        prefix_positions=2,
        suffix_positions=2,
    )

    assert mask.tolist() == [
        [True, True, True],
        [True, True, False],
    ]


def test_reft_prepare_model_freezes_backbone_and_adds_adapters():
    model = _ToyQwenVL(num_layers=2)
    method = ReFTMethod()

    prepared_model, info = method.prepare_model(
        model,
        _FakeProcessor(),
        {
            "rank": 1,
            "layers": [0, 1],
            "prefix_positions": 1,
            "suffix_positions": 1,
        },
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    assert prepared_model is model
    assert hasattr(model, "reft_adapters")
    assert len(model.reft_adapters) == 2
    assert "layers=[0, 1]" in info
    base_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("reft_adapters")
    ]
    assert base_params
    assert all(not param.requires_grad for param in base_params)
    assert all(param.requires_grad for param in model.reft_adapters.parameters())


def test_reft_only_steers_prefix_and_suffix_tokens_during_forward():
    model = _ToyQwenVL(num_layers=1)
    method = ReFTMethod()
    method.prepare_model(
        model,
        _FakeProcessor(),
        {
            "rank": 1,
            "layers": [0],
            "prefix_positions": 1,
            "suffix_positions": 1,
        },
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    adapter = model.reft_adapters[0]
    with torch.no_grad():
        adapter.projection.weight.zero_()
        adapter.projection.weight[0, 0] = 1.0
        adapter.source.weight.zero_()
        adapter.source.bias.fill_(2.0)

    output = model(
        input_ids=torch.tensor([[3, 4, 5, 6]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
    )

    assert output.shape == (1, 4, 4)
    assert not torch.allclose(output[0, 0], torch.zeros(4))
    assert torch.allclose(output[0, 1], torch.zeros(4))
    assert torch.allclose(output[0, 2], torch.zeros(4))
    assert not torch.allclose(output[0, 3], torch.zeros(4))


def test_reft_compact_checkpoint_round_trips_adapter_outputs():
    source = nn.ModuleList([LoReFTAdapter(hidden_size=4, rank=1)])
    target = nn.ModuleList([LoReFTAdapter(hidden_size=4, rank=1)])
    inputs = torch.randn(3, 4)

    with torch.no_grad():
        source[0].projection.weight.zero_()
        source[0].projection.weight[0, 2] = 1.0
        source[0].source.weight.fill_(0.25)
        source[0].source.bias.fill_(0.5)

    state = compact_reft_state(source)
    expected = source[0](inputs)
    load_compact_reft_state(target, state)

    assert state["format"] == REFT_CHECKPOINT_FORMAT
    assert torch.allclose(target[0](inputs), expected, atol=1e-6, rtol=1e-5)
