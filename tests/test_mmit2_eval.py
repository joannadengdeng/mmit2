import os
import sys
import json

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmit2.data.adapters.hf_datasets import HFDatasetsAdapter
from mmit2.data.types import EvalSample
from mmit2.eval.methods.local_method import LocalMethod
from mmit2.eval.run import (
    EvalTarget,
    _evaluate_vqa_dataset,
    _resolve_baseline_source,
    parse_eval_target,
)


def test_parse_eval_target_infers_defaults():
    target = parse_eval_target(
        {
            "dataset_name": "lmms-lab/textvqa",
            "max_new_tokens": 12,
            "max_samples": 25,
        }
    )

    assert target.dataset_name == "lmms-lab/textvqa"
    assert target.split == "validation"
    assert target.max_new_tokens == 12
    assert target.max_samples == 25


def test_parse_eval_target_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unsupported eval.dataset_name"):
        parse_eval_target({"dataset_name": "foo/bar"})


def test_parse_eval_target_rejects_non_textvqa_dataset():
    with pytest.raises(ValueError, match="Unsupported eval.dataset_name"):
        parse_eval_target({"dataset_name": "lmms-lab/VQAv2"})


def test_parse_eval_target_rejects_multi_target_legacy_config():
    with pytest.raises(ValueError, match="exactly one eval dataset"):
        parse_eval_target(
            {
                "targets": [
                    {"dataset_name": "lmms-lab/textvqa"},
                    {"dataset_name": "lmms-lab/textvqa"},
                ]
            }
        )


def test_resolve_baseline_source_uses_base_model_only(tmp_path):
    source = _resolve_baseline_source(
        {
            "model": {"model_path": "Qwen/Qwen2.5-VL-3B-Instruct"},
            "eval": {
                "dataset_name": "lmms-lab/textvqa",
                "output_dir": str(tmp_path / "baseline_eval"),
            },
        },
        "lmms-lab/textvqa",
    )

    assert source.kind == "baseline"
    assert source.base_model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert source.checkpoint_path == ""
    assert source.ft_method == ""


def test_resolve_baseline_source_rejects_checkpoint_config():
    with pytest.raises(ValueError, match="Baseline eval only supports an unfine-tuned base model"):
        _resolve_baseline_source(
            {
                "model": {
                    "model_path": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "checkpoint_path": "some/checkpoint",
                }
            },
            "lmms-lab/textvqa",
        )


class _DummyMethod:
    def prepare_eval_input(self, sample, image_root=""):
        return sample

    def generate(self, prepared, max_new_tokens, temperature):
        return "cat"


class _FakeEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))


class _FakeEvalProcessor:
    def __init__(self):
        self.last_text = ""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_text = messages[0]["content"][-1]["text"]
        return self.last_text

    def __call__(self, text, images=None, return_tensors="pt"):
        self.last_text = text
        return {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }


def test_evaluate_textvqa_uses_multi_annotator_answers(monkeypatch, tmp_path):
    rows = [
        {
            "question_id": 123,
            "image": None,
            "question": "What animal is shown?",
            "answers": (
                [{"answer": "cat"}] * 3
                + [{"answer": "dog"}] * 7
            ),
        }
    ]

    def fake_load_dataset(self, datasets_mod, load_pos, split, streaming, trust_remote_code):
        self._num_examples = len(rows)
        return rows

    monkeypatch.setattr(HFDatasetsAdapter, "_load_dataset", fake_load_dataset)

    result = _evaluate_vqa_dataset(
        _DummyMethod(),
        EvalTarget(
            name="textvqa_validation",
            dataset_name="lmms-lab/textvqa",
            split="validation",
            max_samples=1,
            streaming=True,
        ),
        str(tmp_path),
    )

    assert result["primary_metric"] == "vqa_accuracy"
    assert result["metrics"]["vqa_accuracy"] == 100.0

    with open(result["prediction_file"], "r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    assert len(record["ground_truth"]) == 10
    assert record["ground_truth"].count("cat") == 3


def test_local_method_eval_prompt_requests_short_answer():
    processor = _FakeEvalProcessor()
    method = LocalMethod(_FakeEvalModel(), processor)

    method.prepare_eval_input(
        EvalSample(
            id="1",
            image_path="",
            question="What number is on the player's jersey?",
        )
    )

    assert "single short answer only" in processor.last_text.lower()
    assert "do not use a full sentence" in processor.last_text.lower()
