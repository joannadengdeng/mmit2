import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.training.methods.mole import MoELoRALinear, MoLEMethod
from vlmintune.training.methods.registry import list_training_methods
from vlmintune.models.registry import get_model_spec


class _FakeProcessor:
    def save_pretrained(self, path):
        self.path = path


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.other = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.proj(x)


def test_mole_registers_as_built_in_training_method():
    assert "mole" in list_training_methods()


def test_mole_linear_routes_each_token_to_top1_expert():
    base = nn.Linear(2, 2, bias=False)
    nn.init.zeros_(base.weight)
    layer = MoELoRALinear(base, rank=1, alpha=1, dropout=0.0, num_experts=2)
    with torch.no_grad():
        layer.router.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        layer.lora_a[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        layer.lora_b[0].weight.copy_(torch.tensor([[2.0], [0.0]]))
        layer.lora_a[1].weight.copy_(torch.tensor([[-1.0, 0.0]]))
        layer.lora_b[1].weight.copy_(torch.tensor([[0.0], [3.0]]))

    output = layer(torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]))

    assert torch.allclose(output, torch.tensor([[[2.0, 0.0], [0.0, 3.0]]]))


def test_mole_router_gets_straight_through_gradient():
    base = nn.Linear(2, 2, bias=False)
    nn.init.zeros_(base.weight)
    layer = MoELoRALinear(base, rank=1, alpha=1, dropout=0.0, num_experts=2)
    with torch.no_grad():
        for expert_a in layer.lora_a:
            expert_a.weight.fill_(1.0)
        layer.lora_b[0].weight.copy_(torch.tensor([[1.0], [0.0]]))
        layer.lora_b[1].weight.copy_(torch.tensor([[0.0], [1.0]]))

    layer(torch.tensor([[[1.0, 2.0]]]))[..., 0].sum().backward()

    assert layer.router.weight.grad is not None
    assert layer.router.weight.grad.abs().sum() > 0


def test_mole_prepare_model_replaces_targets_and_freezes_base():
    model = _ToyModel()

    prepared, info = MoLEMethod().prepare_model(
        model,
        _FakeProcessor(),
        {
            "lora_r": 1,
            "lora_alpha": 1,
            "lora_dropout": 0.0,
            "target_modules": ["proj"],
            "num_experts": 2,
        },
        model_spec=get_model_spec("qwen25vl_3b_instruct"),
    )

    assert prepared is model
    assert isinstance(model.proj, MoELoRALinear)
    assert isinstance(model.other, nn.Linear)
    assert "replaced=1" in info
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names == {
        "proj.router.weight",
        "proj.lora_a.0.weight",
        "proj.lora_a.1.weight",
        "proj.lora_b.0.weight",
        "proj.lora_b.1.weight",
    }
