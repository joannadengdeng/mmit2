import json
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods import base as method_base
from vlmintune.training.methods.l2t import (
    L2T_CHECKPOINT_NAME,
    L2T_SUPERVISION_RECIPE,
    L2TMethod,
)


def test_l2t_unmasks_instruction_only():
    method = L2TMethod()
    input_ids = torch.tensor([[11, 12, 13, 14, 0]])
    labels = torch.tensor([[-100, -100, -100, 14, -100]])
    batch_meta = {
        "method_mask": torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.bool),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
    }

    updated = method.preprocess_labels(input_ids, labels, batch_meta=batch_meta)
    forward_batch = method.build_forward_batch(
        {
            "input_ids": input_ids,
            "labels": updated,
            **batch_meta,
        }
    )

    assert updated.tolist() == [[11, -100, 13, 14, -100]]
    assert "method_mask" not in forward_batch
    assert torch.equal(forward_batch["labels"], updated)


def test_l2t_is_standalone():
    method = L2TMethod()

    assert not hasattr(method, "base")


class _FakeProcessor:
    pass


class _ToyQwenL2T(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Linear(hidden_size, hidden_size)
        self.model.visual = nn.Module()
        self.model.visual.encoder = nn.Linear(hidden_size, hidden_size)
        self.model.visual.merger = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        hidden_states = self.model.visual.merger(inputs)
        hidden_states = self.model.language_model(hidden_states)
        return self.lm_head(hidden_states)


class _ToyLlavaL2T(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Linear(hidden_size, hidden_size)
        self.model.vision_tower = nn.Linear(hidden_size, hidden_size)
        self.model.multi_modal_projector = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        hidden_states = self.model.multi_modal_projector(inputs)
        hidden_states = self.model.language_model(hidden_states)
        return self.lm_head(hidden_states)


def _prepare_l2t(model, model_name):
    method = L2TMethod()
    prepared, info = method.prepare_model(
        model,
        _FakeProcessor(),
        model_spec=get_model_spec(model_name),
    )
    return prepared, method, info


@pytest.mark.parametrize(
    ("model_name", "model_factory", "trainable_prefixes", "frozen_prefix"),
    [
        (
            "qwen25vl_3b_instruct",
            _ToyQwenL2T,
            ("model.language_model.", "model.visual.merger.", "lm_head."),
            "model.visual.encoder.",
        ),
        (
            "llava15_7b",
            _ToyLlavaL2T,
            ("model.language_model.", "model.multi_modal_projector.", "lm_head."),
            "model.vision_tower.",
        ),
    ],
)
def test_l2t_fixed_full_sft_trainable_scope(
    model_name,
    model_factory,
    trainable_prefixes,
    frozen_prefix,
):
    model, _, info = _prepare_l2t(model_factory(), model_name)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert trainable_names
    assert all(name.startswith(trainable_prefixes) for name in trainable_names)
    assert all(any(name.startswith(prefix) for name in trainable_names) for prefix in trainable_prefixes)
    assert not any(name.startswith(frozen_prefix) for name in trainable_names)
    assert "vision encoder frozen" in info

def test_l2t_checkpoint_round_trip_uses_bf16_unquantized_base(tmp_path, monkeypatch):
    torch.manual_seed(123)
    source, method, _ = _prepare_l2t(_ToyQwenL2T(), "qwen25vl_3b_instruct")
    with torch.no_grad():
        for index, parameter in enumerate(
            parameter for parameter in source.parameters() if parameter.requires_grad
        ):
            parameter.fill_(0.01 * (index + 1))

    inputs = torch.randn(2, 4)
    expected = source(inputs).detach()
    method.save_checkpoint(
        source,
        str(tmp_path),
        {"model_name": "qwen25vl_3b_instruct", "final_loss": 0.25},
    )

    state_dict = torch.load(
        tmp_path / L2T_CHECKPOINT_NAME,
        map_location="cpu",
        weights_only=True,
    )
    assert state_dict
    assert not any(name.startswith("model.visual.encoder.") for name in state_dict)
    metadata = json.loads((tmp_path / "vlmintune_meta.json").read_text())
    assert metadata["ft_method"] == "l2t"
    assert metadata["recipe"] == "l2t_full_sft_v1"
    assert metadata["supervision_recipe"] == L2T_SUPERVISION_RECIPE
    assert "config" not in metadata

    load_calls = []

    def load_base_model(model_id, **kwargs):
        load_calls.append({"model_id": model_id, **kwargs})
        torch.manual_seed(123)
        return _ToyQwenL2T()

    monkeypatch.setattr(method_base, "load_processor", lambda model_id: _FakeProcessor())
    monkeypatch.setattr(method_base, "load_vlm", load_base_model)

    loaded, _, info = L2TMethod().load_for_inference(
        str(tmp_path),
        "qwen25vl_3b_instruct",
    )

    actual = loaded(inputs).detach()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert load_calls[-1]["quantize_4bit"] is False
    assert load_calls[-1]["torch_dtype"] is torch.bfloat16
    assert "L2T" in info["model_id"]
