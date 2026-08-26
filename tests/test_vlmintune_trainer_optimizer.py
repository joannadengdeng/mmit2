import json
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
from vlmintune.training.trainer.helpers import PreprocessingCoverage, build_skip_logger


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = types.SimpleNamespace()


class _FakeQLoRAMethod:
    last_instance = None

    def __init__(self):
        self.saved_path = None
        type(self).last_instance = self

    def requires_quantization(self):
        return True

    def prepare_model(self, model, processor, model_spec):
        return model, "QLoRA fixed recipe"

    def get_trainable_params(self, model):
        return [{"params": [model.weight]}]

    def save_checkpoint(self, model, path, metadata):
        self.saved_path = path


class _FakeFixedMethod:
    last_instance = None

    def __init__(self):
        self.saved_metadata = None
        type(self).last_instance = self

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

    def save_checkpoint(self, model, path, metadata):
        self.saved_metadata = metadata


def test_preprocessing_coverage_reports_only_aggregate_skip_count(capsys):
    coverage = PreprocessingCoverage()
    build_skip_logger(coverage)("demo", ValueError("invalid sample"))
    coverage.emit_summary()

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert coverage.total_skipped == 1
    assert events[-1] == {
        "type": "data_summary",
        "data": {"kind": "preprocessing_coverage", "total_skipped": 1},
    }


def test_trainer_uses_paged_adamw8bit_for_qlora(monkeypatch, tmp_path):
    optimizer_calls = []
    tokenization_calls = []

    class _FakePagedAdamW8bit:
        def __init__(self, param_groups, **kwargs):
            optimizer_calls.append((param_groups, kwargs))

    monkeypatch.setattr(
        method_registry,
        "get_training_method_cls",
        lambda name: _FakeQLoRAMethod,
    )
    monkeypatch.setattr(trainer_mod, "build_dataset", lambda config: (types.SimpleNamespace(streaming=False), 1))

    def fake_build_tokenized_dataset(**kwargs):
        tokenization_calls.append(kwargs)
        return [], object()

    monkeypatch.setattr(
        trainer_mod,
        "build_tokenized_dataset",
        fake_build_tokenized_dataset,
    )
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
    assert tokenization_calls[0]["method_cls"] is _FakeQLoRAMethod
    assert "method" not in tokenization_calls[0]
    assert _FakeQLoRAMethod.last_instance.saved_path is None


def test_trainer_flushes_partial_gradient_accumulation_group(monkeypatch, tmp_path):
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

    monkeypatch.setattr(
        method_registry,
        "get_training_method_cls",
        lambda name: _FakeFixedMethod,
    )
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
    assert _FakeFixedMethod.last_instance.saved_metadata["final_loss"] > 0
