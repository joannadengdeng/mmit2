import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmit2.training.methods.freeze import _find_transformer_layers, _list_tunable_modules


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


def test_freeze_finds_qwen_language_model_layers():
    model = _ToyQwenVL()

    prefix, layers = _find_transformer_layers(model)

    assert prefix == "model.language_model.layers"
    assert len(layers) == 2


def test_freeze_lists_qwen_language_model_layer_prefixes():
    model = _ToyQwenVL()

    tunable_modules = _list_tunable_modules(model)

    assert "model.language_model.layers" in tunable_modules
    assert "model.language_model.layers.0" in tunable_modules
    assert "model.language_model.layers.1" in tunable_modules
