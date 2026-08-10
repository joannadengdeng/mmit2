import os
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import vlmintune.training.methods.registry as method_registry
import vlmintune.training.trainer.trainer as trainer_mod
from vlmintune.training.trainer.trainer import Trainer, TrainerConfig
from vlmintune.training.trainer.helpers import DebugRecorder


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = types.SimpleNamespace()


class _FakeQLoRAMethod:
    saved_path = None

    def requires_quantization(self):
        return True

    def prepare_model(self, model, processor, model_spec):
        return model, "QLoRA fixed recipe"

    def get_trainable_params(self, model):
        return [{"params": [model.weight]}]

    def save_checkpoint(self, model, processor, path, metadata):
        self.saved_path = path


class _FakeFixedMethod:
    def __init__(self):
        self.saved_metadata = None

    def requires_quantization(self):
        return False

    def prepare_model(self, model, processor, model_spec):
        return model, "fixed recipe"

    def preprocess_labels(self, input_ids, labels, batch_meta=None):
        return labels

    def build_forward_batch(self, batch):
        return batch

    def compute_loss(self, model, batch, outputs):
        return outputs, {}

    def get_trainable_params(self, model):
        return [{"params": [model.weight]}]

    def save_checkpoint(self, model, processor, path, metadata):
        self.saved_metadata = metadata


def test_debug_recorder_compacts_sparse_intervention_masks():
    recorder = DebugRecorder()
    recorder.record_prompt(
        {
            "sample_id": "demo",
            "intervention_mask": [False, True, False, True, False],
            "reft_intervention_mask": [True, False, False, False, True],
        }
    )

    prompt = recorder.prompts[0]
    assert prompt["intervention_mask"] == {
        "length": 5,
        "selected_count": 2,
        "selected_positions": [1, 3],
        "selected_positions_truncated": False,
    }
    assert prompt["reft_intervention_mask"]["selected_positions"] == [0, 4]


def test_trainer_uses_paged_adamw8bit_for_qlora(monkeypatch, tmp_path):
    optimizer_calls = []

    class _FakePagedAdamW8bit:
        def __init__(self, param_groups, **kwargs):
            optimizer_calls.append((param_groups, kwargs))

    method = _FakeQLoRAMethod()
    monkeypatch.setattr(method_registry, "build_training_method", lambda name: method)
    monkeypatch.setattr(trainer_mod, "build_dataset", lambda config: (types.SimpleNamespace(streaming=False), 1))
    monkeypatch.setattr(trainer_mod, "build_tokenized_dataset", lambda **kwargs: ([], object()))
    monkeypatch.setattr(trainer_mod, "DataLoader", lambda *args, **kwargs: [])
    monkeypatch.setattr(trainer_mod, "cosine_schedule", lambda *args, **kwargs: object())
    monkeypatch.setattr("bitsandbytes.optim.PagedAdamW8bit", _FakePagedAdamW8bit)

    trainer = Trainer("qwen25vl_3b_instruct")
    trainer.model = _ToyModel()
    trainer.processor = object()
    output_dir = str(tmp_path / "checkpoint")

    with pytest.raises(RuntimeError, match="zero optimizer steps"):
        trainer.train(
            TrainerConfig(
                data_config={"dataset_name": "lmms-lab/textvqa"},
                training_method="qlora",
                per_device_batch_size=1,
                gradient_accumulation_steps=1,
                output_dir=output_dir,
            )
        )

    assert len(optimizer_calls) == 1
    param_groups, kwargs = optimizer_calls[0]
    assert param_groups[0]["lr"] == 2e-5
    assert kwargs == {"weight_decay": 0.0}
    assert method.saved_path is None


def test_trainer_flushes_partial_gradient_accumulation_group(monkeypatch, tmp_path):
    method = _FakeFixedMethod()
    batches = [
        {
            "input_ids": torch.ones(1, 2),
            "labels": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
        for _ in range(3)
    ]
    scheduler_steps = []

    class _Scheduler:
        def step(self):
            scheduler_steps.append(1)

        def get_last_lr(self):
            return [2e-5]

    class _TrainingModel(_ToyModel):
        def forward(self, **batch):
            del batch
            return self.weight.square().sum()

    monkeypatch.setattr(method_registry, "build_training_method", lambda name: method)
    monkeypatch.setattr(
        trainer_mod,
        "build_dataset",
        lambda config: (types.SimpleNamespace(streaming=False), len(batches)),
    )
    monkeypatch.setattr(
        trainer_mod,
        "build_tokenized_dataset",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(trainer_mod, "DataLoader", lambda *args, **kwargs: batches)
    monkeypatch.setattr(trainer_mod, "cosine_schedule", lambda *args, **kwargs: _Scheduler())
    monkeypatch.setattr(trainer_mod, "describe_batch", lambda batch: "batch")
    monkeypatch.setattr(
        trainer_mod,
        "build_label_supervision_debug",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(trainer_mod, "gradient_debug", lambda *args, **kwargs: {})

    trainer = Trainer("qwen25vl_3b_instruct")
    trainer.model = _TrainingModel()
    trainer.processor = object()
    trainer.train(
        TrainerConfig(
            data_config={"dataset_name": "lmms-lab/textvqa"},
            training_method="lora",
            per_device_batch_size=1,
            gradient_accumulation_steps=2,
            output_dir=str(tmp_path / "checkpoint"),
        )
    )

    assert len(scheduler_steps) == 2
    assert method.saved_metadata["final_loss"] > 0
