import os
import sys
import types

import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.models.registry import get_model_spec


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
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=16),
            image_token_id=42,
        )
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel()


class _ToyLlava(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=32),
            image_token_id=32000,
        )
        self.model = nn.Module()
        self.model.language_model = _ToyLanguageModel()


def test_qwen_model_spec_returns_hidden_size_and_image_token_id():
    spec = get_model_spec("qwen25vl_3b_instruct")
    model = _ToyQwenVL()

    assert spec.name == "qwen25vl_3b_instruct"
    assert spec.get_hidden_size(model) == 16
    assert spec.get_image_token_id(model) == 42


def test_llava_model_spec_returns_hidden_size_and_image_token_id():
    spec = get_model_spec("llava15_7b")
    model = _ToyLlava()

    assert spec.name == "llava15_7b"
    assert spec.get_hidden_size(model) == 32
    assert spec.get_image_token_id(model) == 32000
