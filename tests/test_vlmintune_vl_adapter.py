import json
import os
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.eval.method import LocalMethod
from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.registry import build_training_method, list_training_methods
from vlmintune.training.methods.vl_adapter import (
    VL_ADAPTER_CHECKPOINT_NAME,
    VL_ADAPTER_REDUCTION_FACTOR,
    VLAdapterBlock,
    VLAdapterMethod,
)


class _FakeProcessor:
    def save_pretrained(self, path):
        self.saved_path = str(path)


class _ToySelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, hidden_states, **kwargs):
        del kwargs
        weights = torch.ones(
            hidden_states.shape[:2],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return self.proj(hidden_states), weights


class _ToyMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class _ToyDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = _ToySelfAttention(hidden_size)
        self.mlp = _ToyMLP(hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        attention_output, _ = self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + attention_output
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class _ToyLanguageModel(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [_ToyDecoderLayer(hidden_size) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)


class _ToyVisual(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(hidden_size, hidden_size)])
        self.merger = nn.Linear(hidden_size, hidden_size)


class _ToyQwenVL(nn.Module):
    def __init__(self, hidden_size=8, vocab_size=16, num_layers=2):
        super().__init__()
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=hidden_size),
            image_token_id=42,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel(hidden_size, num_layers)
        self.model.visual = _ToyVisual(hidden_size)

    def forward(self, input_ids=None, inputs_embeds=None):
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_tokens(input_ids)
        )
        for layer in self.model.language_model.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(self.model.language_model.norm(hidden_states))


def prepare(model=None):
    model = model or _ToyQwenVL()
    method = VLAdapterMethod()
    prepared, info = method.prepare_model(
        model,
        _FakeProcessor(),
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )
    return prepared, method, info


def allowed_trainable_name(name):
    return (
        name.startswith("vl_adapter_layers.")
        or name.startswith("model.language_model.layers.")
        and (
            ".input_layernorm." in name
            or ".post_attention_layernorm." in name
        )
        or name.startswith("model.language_model.norm.")
        or name.startswith("model.visual.merger.")
    )


def test_vl_adapter_registry_name_and_fixed_recipe():
    assert "vl_adapter" in list_training_methods()
    method = build_training_method("vl_adapter")
    assert isinstance(method, VLAdapterMethod)
    assert method.display_name == "Single Adapter (VL-Adapter style)"
    assert VL_ADAPTER_REDUCTION_FACTOR == 8

    block = VLAdapterBlock(hidden_size=16)
    assert block.down.in_features == 16
    assert block.down.out_features == 2
    assert block.up.out_features == 16


def test_vl_adapter_rejects_non_qwen_model():
    with pytest.raises(ValueError, match="only supports Qwen2.5-VL"):
        VLAdapterMethod().prepare_model(
            _ToyQwenVL(),
            _FakeProcessor(),
            model_spec=get_model_spec("llava15_7b"),
        )


def test_vl_adapter_cannot_be_installed_twice():
    model, method, _ = prepare()

    with pytest.raises(RuntimeError, match="already installed"):
        method.prepare_model(
            model,
            _FakeProcessor(),
            model_spec=get_model_spec("qwen25vl_3b_instruct"),
        )


def test_vl_adapter_installs_attention_and_mlp_adapter_per_layer():
    model, _, info = prepare(_ToyQwenVL(num_layers=3))

    assert len(model.vl_adapter_layers) == 3
    assert "layers=3" in info
    assert "reduction_factor=8" in info
    for adapter_pair in model.vl_adapter_layers:
        assert set(dict(adapter_pair.named_children())) == {"attention", "mlp"}

    layer = model.model.language_model.layers[0]
    inputs = torch.randn(2, 4, 8)
    original_attention = layer.self_attn.proj(inputs)
    attention_output = layer.self_attn(inputs)
    assert isinstance(attention_output, tuple)
    assert attention_output[0].shape == inputs.shape
    assert not torch.allclose(attention_output[0], original_attention)

    original_mlp = layer.mlp.proj(inputs)
    mlp_output = layer.mlp(inputs)
    assert mlp_output.shape == inputs.shape
    assert not torch.allclose(mlp_output, original_mlp)


def test_vl_adapter_trainable_scope_and_gradients_are_strict():
    model, _, _ = prepare()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert trainable_names
    assert all(allowed_trainable_name(name) for name in trainable_names)
    assert any(name.startswith("vl_adapter_layers.0.attention") for name in trainable_names)
    assert any(name.startswith("vl_adapter_layers.0.mlp") for name in trainable_names)
    assert any(name.startswith("model.language_model.norm") for name in trainable_names)
    assert any(name.startswith("model.visual.merger") for name in trainable_names)
    assert not model.embed_tokens.weight.requires_grad
    assert not model.lm_head.weight.requires_grad
    assert not model.model.visual.blocks[0].weight.requires_grad
    assert not model.model.language_model.layers[0].self_attn.proj.weight.requires_grad
    assert not model.model.language_model.layers[0].mlp.proj.weight.requires_grad

    logits = model(inputs_embeds=torch.randn(2, 3, 8))
    merger_output = model.model.visual.merger(torch.randn(2, 8))
    (logits.square().mean() + merger_output.square().mean()).backward()

    for adapter_pair in model.vl_adapter_layers:
        for branch in (adapter_pair.attention, adapter_pair.mlp):
            assert all(parameter.grad is not None for parameter in branch.parameters())
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None


def test_vl_adapter_checkpoint_round_trip_and_local_loader(tmp_path, monkeypatch):
    torch.manual_seed(123)
    source, method, _ = prepare()
    with torch.no_grad():
        for index, parameter in enumerate(
            parameter for parameter in source.parameters() if parameter.requires_grad
        ):
            parameter.fill_(0.01 * (index + 1))
    inputs = torch.randn(1, 3, 8)
    expected = source(inputs_embeds=inputs).detach()

    method.save_checkpoint(
        source,
        _FakeProcessor(),
        str(tmp_path),
        {"model_name": "qwen25vl_3b_instruct", "final_loss": 0.25},
    )
    state_dict = torch.load(
        tmp_path / VL_ADAPTER_CHECKPOINT_NAME,
        map_location="cpu",
        weights_only=True,
    )
    assert state_dict
    assert all(allowed_trainable_name(name) for name in state_dict)

    metadata = json.loads((tmp_path / "vlmintune_meta.json").read_text())
    assert metadata["ft_method"] == "vl_adapter"
    assert metadata["recipe"] == "single_vl_adapter_v1"
    assert "config" not in metadata

    import vlmintune.training.methods.vl_adapter as vl_adapter_module

    def fresh_base_model():
        torch.manual_seed(123)
        return _ToyQwenVL()

    load_calls = []

    def load_base_model(model_id, **kwargs):
        del model_id
        load_calls.append(kwargs)
        return fresh_base_model()

    monkeypatch.setattr(vl_adapter_module, "load_processor", lambda model_id: _FakeProcessor())
    monkeypatch.setattr(vl_adapter_module, "load_vlm", load_base_model)

    loaded, _, info = VLAdapterMethod().load_for_inference(
        str(tmp_path),
        "qwen25vl_3b_instruct",
    )
    assert torch.allclose(loaded(inputs_embeds=inputs).detach(), expected, atol=1e-6)
    assert "Single Adapter" in info["model_id"]
    assert load_calls[-1] == {
        "quantize_4bit": False,
        "torch_dtype": torch.bfloat16,
    }

    local_method = LocalMethod.from_checkpoint(
        "qwen25vl_3b_instruct",
        checkpoint_path=str(tmp_path),
    )
    assert isinstance(local_method.inference_method, VLAdapterMethod)
    assert torch.allclose(
        local_method.model(inputs_embeds=inputs).detach(),
        expected,
        atol=1e-6,
    )
