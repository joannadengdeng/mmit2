import os
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.mores import (
    MORES_CHECKPOINT_FORMAT,
    MORES_RANK,
    MORES_VISUAL_TOKEN_INDICES,
    MoReSAdapter,
    MoReSMethod,
    compact_mores_state,
    load_compact_mores_state,
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


class _ToyVLM(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4,
        vocab_size: int = 64,
        num_layers: int = 1,
        image_token_id: int = 42,
    ):
        super().__init__()
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=hidden_size),
            image_token_id=image_token_id,
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


def prepare(model=None, model_name="qwen25vl_3b_instruct"):
    model = model or _ToyVLM()
    method = MoReSMethod()
    prepared, info = method.prepare_model(
        model,
        _FakeProcessor(),
        model_spec=get_model_spec(model_name),
    )
    return prepared, method, info


def test_mores_is_registered_with_fixed_recipe():
    assert "mores" in list_training_methods()
    assert MORES_RANK == 1


def test_mores_mask_is_fixed_first_four_last_five_without_duplicates():
    config = types.SimpleNamespace(image_token_id=42)
    input_ids = torch.tensor([7, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 8])

    mask = MoReSMethod.build_method_mask(
        model_config=config,
        input_ids=input_ids,
    )

    assert MORES_VISUAL_TOKEN_INDICES == (1, 2, 3, 4, -5, -4, -3, -2, -1)
    assert mask.tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
    ]

    short_mask = MoReSMethod.build_method_mask(
        model_config=config,
        input_ids=torch.tensor([42, 3, 42, 42]),
    )
    assert short_mask.tolist() == [True, False, True, True]


def test_mores_freezes_backbone_and_installs_rank_one_adapter_on_every_layer():
    model, _, info = prepare(_ToyVLM(num_layers=3))

    assert len(model.mores_adapters) == 3
    assert all(adapter.linear.out_features == MORES_RANK for adapter in model.mores_adapters)
    assert "rank=1" in info
    assert "layers=all (3)" in info
    assert "visual positions=f4+l5" in info

    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("mores_adapters")
    ]
    assert base_parameters
    assert all(not parameter.requires_grad for parameter in base_parameters)
    assert all(parameter.requires_grad for parameter in model.mores_adapters.parameters())


def test_mores_training_only_changes_selected_visual_rows_and_drops_runtime_mask():
    model, method, _ = prepare()
    with torch.no_grad():
        model.mores_adapters[0].linear.weight.zero_()
        model.mores_adapters[0].linear.bias.fill_(2.0)

    input_ids = torch.tensor([[3, 42, 4, 42]])
    runtime_mask = torch.tensor([[False, True, False, False]])
    forward_batch = method.build_forward_batch(
        {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
            "method_mask": runtime_mask,
        }
    )
    output = model(**forward_batch)

    assert "method_mask" not in forward_batch
    assert not torch.allclose(output[0, 1], torch.zeros(4))
    for token_index in (0, 2, 3):
        assert torch.allclose(output[0, token_index], torch.zeros(4))


def test_mores_eval_prefill_builds_mask_and_decode_token_is_not_steered():
    model, method, _ = prepare()
    model.eval()
    with torch.no_grad():
        model.mores_adapters[0].linear.weight.zero_()
        model.mores_adapters[0].linear.bias.fill_(2.0)

    prefill = model(
        input_ids=torch.tensor([[3, 42, 42, 4]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
    )

    class _Cache:
        @staticmethod
        def get_seq_length():
            return 4

    decode = model(
        input_ids=torch.tensor([[5]]),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        past_key_values=_Cache(),
    )

    assert prefill.abs().sum(dim=-1).ne(0).tolist() == [[False, True, True, False]]
    assert torch.allclose(decode, torch.zeros_like(decode))


def test_mores_compact_checkpoint_has_only_fixed_recipe_tensors_and_round_trips():
    source = nn.ModuleList([MoReSAdapter(hidden_size=4)])
    target = nn.ModuleList([MoReSAdapter(hidden_size=4)])
    inputs = torch.randn(3, 4)
    with torch.no_grad():
        source[0].linear.weight.fill_(0.25)
        source[0].linear.bias.fill_(0.5)

    state = compact_mores_state(source)
    expected = source[0](inputs)
    load_compact_mores_state(target, state)

    assert state["format"] == MORES_CHECKPOINT_FORMAT
    assert set(state["layers"][0]) == {
        "w_down_weight",
        "linear_weight",
        "linear_bias",
    }
    assert torch.allclose(target[0](inputs), expected, atol=1e-6, rtol=1e-5)

    with pytest.raises(ValueError, match="MoReS checkpoint"):
        load_compact_mores_state(target, {"layers": state["layers"]})


def test_mores_supports_llava_language_layer_layout():
    model = _ToyVLM(image_token_id=32)
    prepared, _, info = prepare(model, model_name="llava15_7b")

    assert prepared is model
    assert len(model.mores_adapters) == 1
    assert "backbone=llava15_7b" in info
