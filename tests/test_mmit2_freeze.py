import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmit2.training.methods.freeze import (
    _list_tunable_modules,
    _restore_trainable_flags,
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


def test_freeze_lists_qwen_language_model_layer_prefixes():
    model = _ToyQwenVL()

    tunable_modules = _list_tunable_modules(model, "qwen2_5_vl")

    assert "model.language_model.layers" in tunable_modules
    assert "model.language_model.layers.0" in tunable_modules
    assert "model.language_model.layers.1" in tunable_modules


def test_restore_trainable_flags_marks_only_saved_params():
    model = _ToyQwenVL()

    trained_names = [
        "model.language_model.layers.0.linear.weight",
        "model.language_model.layers.0.linear.bias",
    ]
    _restore_trainable_flags(model, trained_names)

    assert model.model.language_model.layers[0].linear.weight.requires_grad is True
    assert model.model.language_model.layers[0].linear.bias.requires_grad is True
    assert model.model.language_model.layers[1].linear.weight.requires_grad is False
    assert model.model.language_model.layers[1].linear.bias.requires_grad is False
