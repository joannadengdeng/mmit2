import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.training.methods.mores import MoReSMethod
from vlmintune.training.registry import build_training_method, list_training_methods


class _ToyDecoderLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _ToyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ToyDecoderLayer(), _ToyDecoderLayer()])

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return types.SimpleNamespace(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class _ToyVLCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _ToyLanguageModel()
        self.embeddings = nn.Embedding(128, 2)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        return self.language_model(input_ids=None, inputs_embeds=inputs_embeds, **kwargs)


class _ToyProcessor:
    def save_pretrained(self, path):
        with open(os.path.join(path, "processor.txt"), "w", encoding="utf-8") as f:
            f.write("ok")


class _ToyTopModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ToyVLCore()
        self.config = types.SimpleNamespace(
            model_type="qwen2_5_vl",
            image_token_id=99,
            video_token_id=100,
            text_config=types.SimpleNamespace(hidden_size=2),
        )

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        outputs = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        return (outputs.last_hidden_state,)


def _set_known_embeddings(model: _ToyTopModel) -> None:
    with torch.no_grad():
        model.model.embeddings.weight.zero_()
        model.model.embeddings.weight[1] = torch.tensor([1.0, 2.0])
        model.model.embeddings.weight[2] = torch.tensor([5.0, 6.0])
        model.model.embeddings.weight[99] = torch.tensor([3.0, 4.0])


def _zero_all_mores_layers(model: _ToyTopModel) -> None:
    with torch.no_grad():
        for layer in model.mores_layers:
            layer.down.weight.zero_()
            layer.up.weight.zero_()


def test_mores_is_registered():
    assert "mores" in list_training_methods()
    assert build_training_method("mores").name == "mores"


def test_mores_only_updates_selected_visual_tokens():
    model = _ToyTopModel()
    _set_known_embeddings(model)

    method = MoReSMethod()
    model, info = method.prepare_model(
        model,
        _ToyProcessor(),
        {
            "model_layout": "qwen2_5_vl",
            "hidden_size": 2,
            "steering_rank": 1,
            "intervention_positions": "f1+l0",
        },
    )

    assert "rank=1" in info
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names
    assert all(name.startswith("mores_layers.") for name in trainable_names)

    _zero_all_mores_layers(model)
    with torch.no_grad():
        model.mores_layers[0].down.weight[0, 0] = 1.0
        model.mores_layers[0].up.weight[0, 0] = 1.0

    hidden_states = model(input_ids=torch.tensor([[1, 99, 2]], dtype=torch.long))[0]
    assert torch.allclose(hidden_states[0, 0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(hidden_states[0, 1], torch.tensor([6.0, 4.0]))
    assert torch.allclose(hidden_states[0, 2], torch.tensor([5.0, 6.0]))


def test_mores_first_last_positions_can_limit_steered_tokens():
    model = _ToyTopModel()
    _set_known_embeddings(model)

    method = MoReSMethod()
    model, _ = method.prepare_model(
        model,
        _ToyProcessor(),
        {
            "model_layout": "qwen2_5_vl",
            "hidden_size": 2,
            "steering_rank": 1,
            "intervention_positions": "f1+l0",
        },
    )

    _zero_all_mores_layers(model)
    with torch.no_grad():
        model.mores_layers[0].down.weight[0, 0] = 1.0
        model.mores_layers[0].up.weight[0, 0] = 1.0

    hidden_states = model(input_ids=torch.tensor([[99, 99, 1]], dtype=torch.long))[0]
    assert torch.allclose(hidden_states[0, 0], torch.tensor([6.0, 4.0]))
    assert torch.allclose(hidden_states[0, 1], torch.tensor([3.0, 4.0]))
    assert torch.allclose(hidden_states[0, 2], torch.tensor([1.0, 2.0]))


def test_mores_uniform_positions_select_evenly_spaced_visual_tokens():
    model = _ToyTopModel()
    _set_known_embeddings(model)

    method = MoReSMethod()
    model, _ = method.prepare_model(
        model,
        _ToyProcessor(),
        {
            "model_layout": "qwen2_5_vl",
            "hidden_size": 2,
            "steering_rank": 1,
            "intervention_positions": "uniform2",
        },
    )

    _zero_all_mores_layers(model)
    with torch.no_grad():
        model.mores_layers[0].down.weight[0, 0] = 1.0
        model.mores_layers[0].up.weight[0, 0] = 1.0

    hidden_states = model(input_ids=torch.tensor([[99, 99, 99, 1]], dtype=torch.long))[0]
    assert torch.allclose(hidden_states[0, 0], torch.tensor([6.0, 4.0]))
    assert torch.allclose(hidden_states[0, 1], torch.tensor([3.0, 4.0]))
    assert torch.allclose(hidden_states[0, 2], torch.tensor([6.0, 4.0]))
    assert torch.allclose(hidden_states[0, 3], torch.tensor([1.0, 2.0]))


def test_mores_checkpoint_round_trip(monkeypatch, tmp_path):
    model = _ToyTopModel()
    _set_known_embeddings(model)
    processor = _ToyProcessor()
    method = MoReSMethod()
    model, _ = method.prepare_model(
        model,
        processor,
        {
            "model_layout": "qwen2_5_vl",
            "hidden_size": 2,
            "steering_rank": 2,
            "intervention_positions": "uniform2",
        },
    )

    with torch.no_grad():
        model.mores_layers[0].down.weight.fill_(0.5)
        model.mores_layers[0].up.weight.fill_(0.25)

    ckpt_dir = tmp_path / "checkpoint"
    method.save_checkpoint(model, processor, str(ckpt_dir), {"base_model": "toy/model"})

    monkeypatch.setattr("vlmintune.training.methods.mores.load_processor", lambda _: _ToyProcessor())
    monkeypatch.setattr("vlmintune.training.methods.mores.load_vlm", lambda *args, **kwargs: _ToyTopModel())

    loaded_model, loaded_processor, info = method.load_for_inference(str(ckpt_dir), "toy/model")

    assert isinstance(loaded_processor, _ToyProcessor)
    assert "MoReS" in info["model_id"]
    assert torch.equal(model.mores_layers[0].down.weight, loaded_model.mores_layers[0].down.weight)
    assert torch.equal(model.mores_layers[0].up.weight, loaded_model.mores_layers[0].up.weight)


def test_mores_requires_explicit_hidden_size():
    model = _ToyTopModel()

    method = MoReSMethod()

    try:
        method.prepare_model(
            model,
            _ToyProcessor(),
            {
                "model_layout": "qwen2_5_vl",
                "steering_rank": 1,
                "intervention_positions": "f1+l0",
            },
        )
    except ValueError as exc:
        assert "training.params.hidden_size" in str(exc)
    else:
        raise AssertionError("Expected MoReS to require explicit hidden_size")
