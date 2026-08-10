import json
import os
import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import vlmintune.training.methods.mole as mole_module
from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.mole import (
    ATTENTION_PROJECTIONS,
    FFN_PROJECTIONS,
    MOLE_ALPHA,
    MOLE_BALANCE_LOSS_WEIGHT,
    MOLE_CHECKPOINT_FILENAME,
    MOLE_CHECKPOINT_FORMAT,
    MOLE_DROPOUT,
    MOLE_NUM_EXPERTS,
    MOLE_RANK,
    LoRALinear,
    MoELoRALinear,
    MoLEMethod,
)
from vlmintune.training.methods.registry import list_training_methods


class _FakeProcessor:
    def __init__(self):
        self.saved_path = None

    def save_pretrained(self, path):
        self.saved_path = path


class _ToySelfAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        combined = (
            self.q_proj(hidden_states)
            + self.k_proj(hidden_states)
            + self.v_proj(hidden_states)
        ) / 3.0
        return self.o_proj(combined)


class _ToyMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class _ToyBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.self_attn = _ToySelfAttention(hidden_size)
        self.mlp = _ToyMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(hidden_states)
        return hidden_states + self.mlp(hidden_states)


class _ToyLanguageModel(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_ToyBlock(hidden_size, intermediate_size) for _ in range(num_layers)]
        )


class _ToyLlava(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4,
        intermediate_size: int = 7,
        vocab_size: int = 17,
        num_layers: int = 1,
    ):
        super().__init__()
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=hidden_size),
            image_token_id=32000,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel(
            hidden_size,
            intermediate_size,
            num_layers,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids=None, labels=None, **kwargs):
        del labels, kwargs
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.model.language_model.layers:
            hidden_states = layer(hidden_states)
        return types.SimpleNamespace(logits=self.lm_head(hidden_states), loss=None)


def _prepare(num_layers: int = 1):
    torch.manual_seed(7)
    model = _ToyLlava(num_layers=num_layers)
    method = MoLEMethod()
    prepared, info = method.prepare_model(
        model,
        _FakeProcessor(),
        model_spec=get_model_spec("llava15_7b"),
    )
    return method, prepared, info


def _batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
    }


def test_mole_registers_with_fixed_recipe():
    assert "mole" in list_training_methods()
    assert (MOLE_RANK, MOLE_ALPHA, MOLE_DROPOUT, MOLE_NUM_EXPERTS) == (32, 16, 0.05, 3)


def test_mole_explicitly_rejects_non_llava_backbones():
    with pytest.raises(ValueError, match="only supports model.name='llava15_7b'"):
        MoLEMethod().prepare_model(
            _ToyLlava(),
            _FakeProcessor(),
            model_spec=get_model_spec("qwen25vl_3b_instruct"),
        )


def test_mole_installs_ordinary_attention_lora_and_shared_ffn_router():
    _, model, info = _prepare(num_layers=2)
    layers = model.model.language_model.layers

    assert "r=32, alpha=16, dropout=0.05, experts=3, top_k=1" in info
    assert len(model.mole_routers) == 2
    for layer_index, layer in enumerate(layers):
        assert all(
            isinstance(getattr(layer.self_attn, name), LoRALinear)
            for name in ATTENTION_PROJECTIONS
        )
        assert all(
            isinstance(getattr(layer.mlp, name), MoELoRALinear)
            for name in FFN_PROJECTIONS
        )
        router = layer.mlp.mole_router
        assert router is model.mole_routers[layer_index]
        assert layer.mlp.gate_proj.router is router
        assert layer.mlp.up_proj.router is router
        assert layer.mlp.down_proj.router is router

    assert layers[0].mlp.mole_router is not layers[1].mlp.mole_router


def test_mole_computes_router_once_and_reuses_route_for_all_ffn_linears():
    _, model, _ = _prepare()
    router = model.mole_routers[0]
    calls = []
    handle = router.register_forward_hook(lambda module, args, output: calls.append(output[0].detach()))

    model(**_batch())
    handle.remove()

    assert len(calls) == 1
    assert router.last_counts.sum().item() == 4


def test_mole_groups_tokens_and_never_executes_unselected_experts():
    _, model, _ = _prepare()
    layer = model.model.language_model.layers[0]
    with torch.no_grad():
        # Equal logits make torch.argmax select expert zero for every token.
        layer.mlp.mole_router.proj.weight.zero_()

    execution_counts = {
        projection: [0 for _ in range(MOLE_NUM_EXPERTS)]
        for projection in FFN_PROJECTIONS
    }
    handles = []
    for projection in FFN_PROJECTIONS:
        wrapped = getattr(layer.mlp, projection)
        for expert_index, expert in enumerate(wrapped.experts):
            handles.append(
                expert.register_forward_hook(
                    lambda module, args, output, projection=projection, expert_index=expert_index: (
                        execution_counts[projection].__setitem__(
                            expert_index,
                            execution_counts[projection][expert_index] + 1,
                        )
                    )
                )
            )

    model(**_batch())
    for handle in handles:
        handle.remove()

    assert execution_counts == {
        "gate_proj": [1, 0, 0],
        "up_proj": [1, 0, 0],
        "down_proj": [1, 0, 0],
    }


def test_mole_excludes_padding_from_routing_and_expert_execution():
    _, model, _ = _prepare()
    layer = model.model.language_model.layers[0]
    with torch.no_grad():
        layer.mlp.mole_router.proj.weight.zero_()

    routed_token_counts = []
    handles = [
        getattr(layer.mlp, projection).experts[0].register_forward_pre_hook(
            lambda module, args: routed_token_counts.append(args[0].shape[0])
        )
        for projection in FFN_PROJECTIONS
    ]
    model(
        input_ids=torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]]),
        labels=torch.tensor([[1, 2, 3, 4], [5, 6, -100, -100]]),
        attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
    )
    for handle in handles:
        handle.remove()

    assert layer.mlp.mole_router.last_counts.sum().item() == 6
    assert routed_token_counts == [6, 6, 6]


def test_mole_balance_loss_matches_paper_formula_and_trains_router():
    method, model, _ = _prepare(num_layers=2)
    for router in model.mole_routers:
        with torch.no_grad():
            router.proj.weight.zero_()

    batch = _batch()
    outputs = model(**batch)
    ce_loss = F.cross_entropy(
        outputs.logits[..., :-1, :].reshape(-1, outputs.logits.shape[-1]),
        batch["labels"][..., 1:].reshape(-1),
    )
    expected_layer_losses = [
        (router.last_counts * router.last_probability_totals).sum()
        for router in model.mole_routers
    ]
    expected_balance = torch.stack(expected_layer_losses).mean()

    loss, metrics = method.compute_loss(model, batch, outputs)
    loss.backward()

    assert torch.allclose(
        loss.detach(),
        ce_loss.detach() + MOLE_BALANCE_LOSS_WEIGHT * expected_balance,
    )
    assert metrics["mole_balance_loss"] == pytest.approx(float(expected_balance))
    assert metrics["mole_auxiliary_loss"] == pytest.approx(
        MOLE_BALANCE_LOSS_WEIGHT * float(expected_balance)
    )
    for router in model.mole_routers:
        assert router.proj.weight.grad is not None
        assert router.proj.weight.grad.abs().sum() > 0


def test_mole_trainable_scope_is_only_attention_lora_ffn_experts_and_router():
    method, model, _ = _prepare()
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert len(trainable) == 27
    assert all(".base." not in name for name in trainable)
    assert all(
        (
            ".self_attn." in name and ".expert.lora_" in name
        )
        or (
            ".mlp." in name and ".experts." in name and ".lora_" in name
        )
        or name.endswith(".mlp.mole_router.proj.weight")
        for name in trainable
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name not in trainable
    )

    optimizer_params = method.get_trainable_params(model)[0]["params"]
    assert {id(parameter) for parameter in optimizer_params} == {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }


def test_mole_fixed_checkpoint_round_trip(monkeypatch, tmp_path):
    source_method, source_model, _ = _prepare()
    source_processor = _FakeProcessor()
    with torch.no_grad():
        for parameter_index, parameter in enumerate(
            parameter
            for parameter in source_model.parameters()
            if parameter.requires_grad
        ):
            parameter.fill_((parameter_index + 1) / 100.0)

    source_method.save_checkpoint(
        source_model,
        source_processor,
        str(tmp_path),
        {"model_name": "llava15_7b"},
    )
    assert source_processor.saved_path == str(tmp_path)
    checkpoint = torch.load(
        tmp_path / MOLE_CHECKPOINT_FILENAME,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["format"] == MOLE_CHECKPOINT_FORMAT
    with open(tmp_path / "vlmintune_meta.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["mole_checkpoint_format"] == MOLE_CHECKPOINT_FORMAT

    target_processor = _FakeProcessor()
    monkeypatch.setattr(mole_module, "load_processor", lambda model_id: target_processor)
    monkeypatch.setattr(
        mole_module,
        "load_vlm",
        lambda model_id, **kwargs: _ToyLlava(),
    )
    loaded_model, loaded_processor, info = MoLEMethod().load_for_inference(
        str(tmp_path),
        "llava15_7b",
    )

    source_state = {
        name: parameter.detach()
        for name, parameter in source_model.named_parameters()
        if parameter.requires_grad
    }
    loaded_state = {
        name: parameter.detach()
        for name, parameter in loaded_model.named_parameters()
        if parameter.requires_grad
    }
    assert source_state.keys() == loaded_state.keys()
    assert all(torch.equal(source_state[name], loaded_state[name]) for name in source_state)
    assert loaded_processor is target_processor
    assert not loaded_model.training
    assert "LLaVA-MoLE fixed v1" in info["model_id"]


def test_mole_checkpoint_loader_rejects_legacy_unversioned_weights(tmp_path):
    method, model, _ = _prepare()
    method.save_checkpoint(
        model,
        _FakeProcessor(),
        str(tmp_path),
        {"model_name": "llava15_7b"},
    )
    torch.save(
        {"state_dict": {}},
        tmp_path / MOLE_CHECKPOINT_FILENAME,
    )

    with pytest.raises(ValueError, match="not fixed LLaVA-MoLE v1 format"):
        MoLEMethod().load_for_inference(str(tmp_path), "llava15_7b")
