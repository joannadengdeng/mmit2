import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.training.methods import lora as lora_mod
from vlmintune.training.methods.dora import DoRAMethod
from vlmintune.training.methods.l2t import L2TMethod
from vlmintune.training.methods.lora import LoRAMethod, QLoRAMethod


class _FakeModel:
    def eval(self):
        return self


class _FakePeftModel:
    def __init__(self, base_model):
        self.base_model = base_model

    @classmethod
    def from_pretrained(cls, model, path):
        return cls(model)

    def eval(self):
        return self

    def merge_and_unload(self):
        return self


def test_lora_family_inference_quantizes_only_qlora(monkeypatch, tmp_path):
    load_calls = []

    def fake_load_processor(model_id):
        return {"model_id": model_id}

    def fake_load_vlm(model_id, *, quantize_4bit=False, torch_dtype=None):
        load_calls.append(
            {
                "model_id": model_id,
                "quantize_4bit": quantize_4bit,
                "torch_dtype": torch_dtype,
            }
        )
        return _FakeModel()

    monkeypatch.setattr(lora_mod, "load_processor", fake_load_processor)
    monkeypatch.setattr(lora_mod, "load_vlm", fake_load_vlm)
    monkeypatch.setattr(lora_mod, "PeftModel", _FakePeftModel)

    methods = [
        (LoRAMethod(), False),
        (DoRAMethod(), False),
        (L2TMethod(), False),
        (QLoRAMethod(), True),
    ]

    for method, expected_quantized in methods:
        load_calls.clear()
        method.load_for_inference(str(tmp_path), "llava15_7b", quantize_4bit=True)
        assert load_calls[-1]["quantize_4bit"] is expected_quantized
