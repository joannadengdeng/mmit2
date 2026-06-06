import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.models.registry import get_model_spec, list_model_names
from vlmintune.training.methods.freeze import (
    list_tunable_modules,
)


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _ToyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(), _ToyBlock()])


class _ToyQwenVL(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel()


class _ToyLlava(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel()


def test_freeze_lists_qwen_language_model_layer_prefixes():
    model = _ToyQwenVL()

    tunable_modules = list_tunable_modules(model, "qwen25vl_3b_instruct")

    assert "model.language_model.layers" in tunable_modules
    assert "model.language_model.layers.0" in tunable_modules
    assert "model.language_model.layers.1" in tunable_modules


def test_model_spec_resolves_qwen_transformer_layers():
    model = _ToyQwenVL()
    spec = get_model_spec("qwen25vl_3b_instruct")

    layers = spec.get_transformer_layers(model)

    assert len(layers) == 2
    assert all(isinstance(layer, _ToyBlock) for layer in layers)


def test_registry_lists_qwen_and_llava_model_names():
    model_names = list_model_names()

    assert "qwen25vl_3b_instruct" in model_names
    assert "llava15_7b" in model_names


def test_model_spec_resolves_llava_transformer_layers():
    model = _ToyLlava()
    spec = get_model_spec("llava15_7b")

    layers = spec.get_transformer_layers(model)

    assert len(layers) == 2
    assert all(isinstance(layer, _ToyBlock) for layer in layers)
