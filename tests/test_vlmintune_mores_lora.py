import json
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.training.methods import mores_lora as combo_module
from vlmintune.training.methods import base as method_base
from vlmintune.training.methods.lora import LoRAMethod
from vlmintune.training.methods.mores import MoReSMethod, compact_mores_state
from vlmintune.training.methods.mores_lora import (
    MORES_LORA_CHECKPOINT_COMPONENTS,
    MORES_LORA_RECIPE,
    MoReSLoRAMethod,
    _validate_mores_lora_checkpoint,
)
from vlmintune.training.methods.registry import build_training_method


class _FakeProcessor:
    pass


class _ToyBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class _ToyVLM(nn.Module):
    def __init__(self, hidden_size=4, vocab_size=64, num_layers=2):
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size),
            image_token_id=42,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [_ToyBlock(hidden_size) for _ in range(num_layers)]
        )

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        del kwargs
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_tokens(input_ids)
        )
        for layer in self.model.language_model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _ToyPeftModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.text_config.hidden_size
        self.lora_A = nn.Linear(hidden_size, 2, bias=False)
        self.lora_B = nn.Linear(2, hidden_size, bias=False)
        self.config = base_model.config

    @property
    def mores_adapters(self):
        return self.base_model.mores_adapters

    def forward(self, *args, **kwargs):
        hidden_states = self.base_model(*args, **kwargs)
        return hidden_states + self.lora_B(self.lora_A(hidden_states))

    def save_pretrained(self, path):
        with open(
            os.path.join(path, "adapter_config.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump({"peft_type": "LORA"}, file)
        torch.save(
            {
                "lora_A": self.lora_A.state_dict(),
                "lora_B": self.lora_B.state_dict(),
            },
            os.path.join(path, MORES_LORA_CHECKPOINT_COMPONENTS["lora"]),
        )


_TOY_SPEC = SimpleNamespace(
    name="toy_vlm",
    hf_model_id="local/toy-vlm",
    transformer_layer_path="model.language_model.layers",
    get_hidden_size=lambda model: model.config.text_config.hidden_size,
    get_transformer_layers=lambda model: list(model.model.language_model.layers),
)


def _fake_lora_prepare(self, model, processor, model_spec):
    del self, processor, model_spec
    model.requires_grad_(False)
    return _ToyPeftModel(model), "LoRA v1 test recipe\nTrainable: fake"


def _prepare(monkeypatch):
    monkeypatch.setattr(LoRAMethod, "prepare_model_impl", _fake_lora_prepare)
    method = MoReSLoRAMethod()
    model, info = method.prepare_model(
        _ToyVLM(),
        _FakeProcessor(),
        model_spec=_TOY_SPEC,
    )
    return model, method, info


def test_mores_lora_inherits_the_mores_method_mask():
    method = build_training_method("mores_lora")

    assert type(method) is MoReSLoRAMethod
    assert MoReSLoRAMethod.build_method_mask is MoReSMethod.build_method_mask
    assert method.requires_quantization() is False


def test_mores_lora_prepares_both_trainable_families_and_exact_optimizer(
    monkeypatch,
):
    model, method, info = _prepare(monkeypatch)
    trainable_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert any("lora_A" in name for name, _ in trainable_named)
    assert any("lora_B" in name for name, _ in trainable_named)
    assert any("mores_adapters" in name for name, _ in trainable_named)
    assert all(
        "lora_A" in name or "lora_B" in name or "mores_adapters" in name
        for name, _ in trainable_named
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_A" not in name
        and "lora_B" not in name
        and "mores_adapters" not in name
    )
    assert MORES_LORA_RECIPE in info
    assert "both train jointly" in info

    groups = method.get_trainable_params(model)
    optimizer_parameters = [
        parameter
        for group in groups
        for parameter in group["params"]
    ]
    assert len({id(parameter) for parameter in optimizer_parameters}) == len(
        optimizer_parameters
    )
    assert {id(parameter) for parameter in optimizer_parameters} == {
        id(parameter)
        for _, parameter in trainable_named
    }


def test_mores_lora_backward_reaches_both_adapter_families(monkeypatch):
    model, method, _ = _prepare(monkeypatch)
    input_ids = torch.tensor([[3, 42, 7, 42]])
    forward_batch = method.build_forward_batch(
        {
            "input_ids": input_ids,
            "method_mask": torch.tensor(
                [[False, True, False, False]],
                dtype=torch.bool,
            ),
        }
    )

    model(**forward_batch).square().mean().backward()

    lora_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]
    mores_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "mores_adapters" in name
    ]
    assert lora_gradients and all(gradient is not None for gradient in lora_gradients)
    assert mores_gradients and all(
        gradient is not None for gradient in mores_gradients
    )
    assert all(
        torch.isfinite(gradient).all()
        for gradient in [*lora_gradients, *mores_gradients]
    )


def test_mores_lora_rejects_missing_family_and_rogue_trainable_parameter(
    monkeypatch,
):
    model, method, _ = _prepare(monkeypatch)
    model.lora_A.requires_grad_(False)
    model.lora_B.requires_grad_(False)
    with pytest.raises(RuntimeError, match="missing trainable.*LoRA"):
        method.get_trainable_params(model)

    model.lora_A.requires_grad_(True)
    model.lora_B.requires_grad_(True)
    rogue_name, rogue_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_A" not in name
        and "lora_B" not in name
        and "mores_adapters" not in name
    )
    rogue_parameter.requires_grad_(True)
    with pytest.raises(RuntimeError, match="outside the joint") as exc_info:
        method.get_trainable_params(model)
    assert rogue_name in str(exc_info.value)


def test_mores_lora_checkpoint_saves_both_components_and_explicit_metadata(
    monkeypatch,
    tmp_path,
):
    model, method, _ = _prepare(monkeypatch)

    method.save_checkpoint(
        model,
        str(tmp_path),
        {"model_name": "qwen25vl_3b_instruct", "final_loss": 0.25},
    )

    metadata = json.loads((tmp_path / "vlmintune_meta.json").read_text())
    assert (tmp_path / MORES_LORA_CHECKPOINT_COMPONENTS["lora"]).is_file()
    assert (tmp_path / MORES_LORA_CHECKPOINT_COMPONENTS["mores"]).is_file()
    assert metadata["ft_method"] == "mores_lora"
    assert metadata["recipe"] == MORES_LORA_RECIPE
    assert metadata["combination_recipe"] == MORES_LORA_RECIPE
    assert metadata["structure_methods"] == ["mores", "lora"]
    assert metadata["composition_order"] == ["mores", "lora"]
    assert metadata["component_recipes"] == {
        "mores": "mores",
        "lora": "lora_v1",
    }
    assert metadata["checkpoint_components"] == MORES_LORA_CHECKPOINT_COMPONENTS
    assert _validate_mores_lora_checkpoint(str(tmp_path)) == metadata


def test_mores_lora_checkpoint_validation_rejects_mixed_components(
    monkeypatch,
    tmp_path,
):
    model, method, _ = _prepare(monkeypatch)
    method.save_checkpoint(
        model,
        str(tmp_path),
        {"model_name": "qwen25vl_3b_instruct", "final_loss": 0.25},
    )

    metadata_path = tmp_path / "vlmintune_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["composition_order"] = ["lora", "mores"]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="composition_order"):
        _validate_mores_lora_checkpoint(str(tmp_path))


class _FakeMergedPeft:
    load_calls = []

    def __init__(self, base_model):
        self.base_model = base_model

    @classmethod
    def from_pretrained(cls, model, path):
        cls.load_calls.append((model, path))
        return cls(model)

    def merge_and_unload(self):
        return self.base_model


def test_mores_lora_inference_merges_lora_then_restores_mores(
    monkeypatch,
    tmp_path,
):
    source = _ToyVLM()
    source_method = MoReSMethod()
    source, _ = source_method.prepare_model(
        source,
        _FakeProcessor(),
        model_spec=_TOY_SPEC,
    )
    with torch.no_grad():
        source.mores_adapters[0].linear.weight.fill_(0.125)
        source.mores_adapters[0].linear.bias.fill_(0.375)
    torch.save(
        compact_mores_state(source.mores_adapters),
        tmp_path / MORES_LORA_CHECKPOINT_COMPONENTS["mores"],
    )
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}),
        encoding="utf-8",
    )
    (tmp_path / MORES_LORA_CHECKPOINT_COMPONENTS["lora"]).write_bytes(b"lora")
    (tmp_path / "vlmintune_meta.json").write_text(
        json.dumps(
            {
                "ft_method": "mores_lora",
                "recipe": MORES_LORA_RECIPE,
                "combination_recipe": MORES_LORA_RECIPE,
                "structure_methods": ["mores", "lora"],
                "composition_order": ["mores", "lora"],
                "component_recipes": {"mores": "mores", "lora": "lora_v1"},
                "checkpoint_components": MORES_LORA_CHECKPOINT_COMPONENTS,
            }
        ),
        encoding="utf-8",
    )

    load_calls = []

    def load_base(model_id, *, quantize_4bit=False, torch_dtype=None):
        load_calls.append((model_id, quantize_4bit, torch_dtype))
        return _ToyVLM()

    monkeypatch.setattr(method_base, "get_model_spec", lambda name: _TOY_SPEC)
    monkeypatch.setattr(method_base, "load_processor", lambda model_id: _FakeProcessor())
    monkeypatch.setattr(method_base, "load_vlm", load_base)
    monkeypatch.setattr(combo_module, "PeftModel", _FakeMergedPeft)
    _FakeMergedPeft.load_calls.clear()

    method = MoReSLoRAMethod()
    loaded, _, info = method.load_for_inference(
        str(tmp_path),
        "qwen25vl_3b_instruct",
    )

    assert _FakeMergedPeft.load_calls == [(loaded, str(tmp_path))]
    assert load_calls == [("local/toy-vlm", False, torch.bfloat16)]
    assert loaded.training is False
    assert torch.allclose(
        loaded.mores_adapters[0].linear.weight,
        source.mores_adapters[0].linear.weight,
    )
    assert torch.allclose(
        loaded.mores_adapters[0].linear.bias,
        source.mores_adapters[0].linear.bias,
    )
    assert "MoReS + LoRA" in info["model_id"]
