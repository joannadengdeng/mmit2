import os
import sys
import types

import pytest
import torch
import torch.nn as nn
from transformers import DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.reft import (
    LoReFTAdapter,
    REFT_CHECKPOINT_FORMAT,
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
        self.cache_lengths = []
        self.forward_outputs = []

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        del attention_mask, kwargs
        if past_key_values is not None:
            self.cache_lengths.append(past_key_values.get_seq_length())
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        for layer in self.model.language_model.layers:
            hidden_states = layer(hidden_states)
        self.forward_outputs.append(hidden_states.detach().clone())
        if past_key_values is not None and use_cache:
            cache_states = torch.zeros(
                hidden_states.size(0),
                1,
                hidden_states.size(1),
                1,
                device=hidden_states.device,
            )
            past_key_values.update(cache_states, cache_states.clone(), layer_idx=0)
        return hidden_states


def test_reft_registers_as_built_in_training_method():
    assert "reft" in list_training_methods()


def test_reft_position_mask_handles_short_sequences():
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    mask = build_reft_position_mask(
        input_ids,
        torch.tensor([10, 3]),
    )

    assert mask.tolist() == [
        [True, True, True, True, False, False, True, True, True, True, False, False],
        [True, True, True, False, False, False, False, False, False, False, False, False],
    ]


def test_reft_prepare_model_freezes_backbone_and_adds_adapters():
    model = _ToyQwenVL(num_layers=2)
    method = ReFTMethod()

    prepared_model, info = method.prepare_model(
        model,
        _FakeProcessor(),
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    assert prepared_model is model
    assert hasattr(model, "reft_adapters")
    assert len(model.reft_adapters) == 2
    assert "rank=4" in info
    assert "layers=all (2)" in info
    assert "tied positions" in info
    assert all(adapter.source.out_features == 4 for adapter in model.reft_adapters)
    base_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("reft_adapters")
    ]
    assert base_params
    assert all(not param.requires_grad for param in base_params)
    assert all(param.requires_grad for param in model.reft_adapters.parameters())

def test_reft_training_mask_only_steers_prompt_prefix_and_suffix():
    model = _ToyQwenVL(num_layers=1)
    method = ReFTMethod()
    method.prepare_model(
        model,
        _FakeProcessor(),
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    adapter = model.reft_adapters[0]
    with torch.no_grad():
        adapter.source.weight.zero_()
        adapter.source.bias.fill_(2.0)

    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
    reft_mask = build_reft_position_mask(input_ids, torch.tensor([10]))
    forward_batch = method.build_forward_batch(
        {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
            "reft_intervention_mask": reft_mask,
        }
    )
    output = model(**forward_batch)

    assert "reft_intervention_mask" not in forward_batch
    assert output.shape == (1, 12, 4)
    for token_idx in (0, 1, 2, 3, 6, 7, 8, 9):
        assert not torch.allclose(output[0, token_idx], torch.zeros(4))
    for token_idx in (4, 5, 10, 11):
        assert torch.allclose(output[0, token_idx], torch.zeros(4))


def test_reft_generation_intervenes_on_prefill_but_not_cached_decode():
    model = _ToyQwenVL(num_layers=1)
    method = ReFTMethod()
    method.prepare_model(
        model,
        _FakeProcessor(),
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )
    model.eval()

    with torch.no_grad():
        model.reft_adapters[0].source.weight.zero_()
        model.reft_adapters[0].source.bias.fill_(2.0)

    prompt_ids = torch.arange(1, 11).unsqueeze(0)
    prepared = method.prepare_inference_inputs(
        model,
        _FakeProcessor(),
        {
            "input_ids": prompt_ids,
            "attention_mask": torch.ones_like(prompt_ids),
        },
    )
    cache = DynamicCache()
    prefill = model(**prepared, past_key_values=cache)

    assert prepared["use_cache"] is True
    assert prefill.abs().sum(dim=-1).ne(0).tolist() == [[
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
    ]]

    first_decode = model(
        input_ids=torch.tensor([[11]]),
        attention_mask=torch.ones(1, 11, dtype=torch.long),
        past_key_values=cache,
        use_cache=True,
    )
    second_decode = model(
        input_ids=torch.tensor([[12]]),
        attention_mask=torch.ones(1, 12, dtype=torch.long),
        past_key_values=cache,
        use_cache=True,
    )

    assert model.cache_lengths == [0, 10, 11]
    assert torch.allclose(first_decode, torch.zeros_like(first_decode))
    assert torch.allclose(second_decode, torch.zeros_like(second_decode))
    assert method.hook_call_count == 1
    assert method.hook_intervention_tokens == 8


def test_reft_static_cache_prefill_accepts_four_dimensional_attention_mask():
    model = _ToyQwenVL(num_layers=1)
    method = ReFTMethod()
    method.prepare_model(
        model,
        _FakeProcessor(),
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )
    model.eval()
    with torch.no_grad():
        model.reft_adapters[0].source.weight.zero_()
        model.reft_adapters[0].source.bias.fill_(2.0)

    input_ids = torch.arange(1, 11).unsqueeze(0)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.zeros(1, 1, 10, 16),
    )

    assert output.abs().sum(dim=-1).ne(0).tolist() == [[
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
    ]]


def test_reft_compact_checkpoint_round_trips_adapter_outputs():
    source = nn.ModuleList([LoReFTAdapter(hidden_size=4, rank=4)])
    target = nn.ModuleList([LoReFTAdapter(hidden_size=4, rank=4)])
    inputs = torch.randn(3, 4)

    with torch.no_grad():
        source[0].source.weight.fill_(0.25)
        source[0].source.bias.fill_(0.5)

    state = compact_reft_state(source)
    expected = source[0](inputs)
    load_compact_reft_state(target, state)

    assert state["format"] == REFT_CHECKPOINT_FORMAT
    assert torch.allclose(target[0](inputs), expected, atol=1e-6, rtol=1e-5)

    with pytest.raises(ValueError, match="fixed tied ReFT v1"):
        load_compact_reft_state(target, {"layers": state["layers"]})
