import os
import sys
import json

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.hf_datasets import HFDatasetsAdapter
from vlmintune.data.types import EvalSample
from vlmintune.eval.method import (
    LocalMethod,
)
from vlmintune.eval.run import (
    EvalTarget,
    evaluate_vqa_dataset,
    parse_eval_target,
    resolve_experiment_source,
)
from vlmintune.eval.vqa import normalize_answer, vqa_accuracy
from vlmintune.training.experiment import ExperimentTracker


def test_parse_eval_target_requires_explicit_split():
    with pytest.raises(ValueError, match="eval.split is required"):
        parse_eval_target(
            {
                "dataset_name": "lmms-lab/textvqa",
                "max_new_tokens": 12,
                "max_samples": 25,
            }
        )


def test_parse_eval_target_accepts_explicit_split():
    target = parse_eval_target(
        {
            "dataset_name": "lmms-lab/textvqa",
            "split": "validation",
            "source": "trained",
            "metric": "vqa_accuracy",
            "max_new_tokens": 12,
            "max_samples": 25,
        }
    )

    assert target.dataset_name == "lmms-lab/textvqa"
    assert target.split == "validation"
    assert target.source == "trained"
    assert target.metric == "vqa_accuracy"
    assert target.max_new_tokens == 12
    assert target.max_samples == 25


def test_parse_eval_target_requires_metric():
    with pytest.raises(ValueError, match="eval.metric is required"):
        parse_eval_target(
            {
                "dataset_name": "lmms-lab/textvqa",
                "split": "validation",
                "source": "trained",
            }
        )


def test_parse_eval_target_rejects_non_string_metric():
    with pytest.raises(ValueError, match="eval.metric is required"):
        parse_eval_target(
            {
                "dataset_name": "lmms-lab/textvqa",
                "split": "validation",
                "source": "trained",
                "metric": ["vqa_accuracy"],
            }
        )


def test_parse_eval_target_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unsupported eval.dataset_name"):
        parse_eval_target({"dataset_name": "foo/bar"})


def test_parse_eval_target_rejects_non_textvqa_dataset():
    with pytest.raises(ValueError, match="Unsupported eval.dataset_name"):
        parse_eval_target({"dataset_name": "lmms-lab/VQAv2"})


def test_parse_eval_target_rejects_unsupported_metric():
    with pytest.raises(ValueError, match="eval.metric must be exactly"):
        parse_eval_target(
            {
                "dataset_name": "lmms-lab/textvqa",
                "split": "validation",
                "source": "trained",
                "metric": "anls",
            }
        )


def test_parse_eval_target_rejects_metric_with_whitespace():
    with pytest.raises(ValueError, match="eval.metric must be exactly"):
        parse_eval_target(
            {
                "dataset_name": "lmms-lab/textvqa",
                "split": "validation",
                "source": "trained",
                "metric": " vqa_accuracy ",
            }
        )


def test_parse_eval_target_requires_source():
    with pytest.raises(ValueError, match="eval.source is required"):
        parse_eval_target(
            {
                "dataset_name": "lmms-lab/textvqa",
                "split": "validation",
                "metric": "vqa_accuracy",
            }
        )


def test_parse_eval_target_rejects_unknown_source():
    with pytest.raises(ValueError, match="eval.source must be exactly"):
        parse_eval_target(
            {
                "dataset_name": "lmms-lab/textvqa",
                "split": "validation",
                "source": "baseline",
                "metric": "vqa_accuracy",
            }
        )


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


def test_normalize_answer_matches_official_vqa_processing():
    assert normalize_answer("The two, cats.") == "2 cats"
    assert normalize_answer("dont") == "don't"
    assert normalize_answer("2:50 pm") == "2:50 pm"


def test_vqa_accuracy_matches_official_leave_one_out_scoring():
    assert vqa_accuracy("cat", ["cat"] * 10) == 1.0
    assert vqa_accuracy("cat", ["cat"] + ["dog"] * 9) == 0.3
    assert vqa_accuracy("cat", ["cat"] * 2 + ["dog"] * 8) == 0.6
    assert vqa_accuracy("cat", ["cat"] * 3 + ["dog"] * 7) == 0.9
    assert vqa_accuracy("cat", ["cat"] * 4 + ["dog"] * 6) == 1.0


def test_resolve_experiment_source_uses_trained_checkpoint(tmp_path):
    tracker = ExperimentTracker.create(exp_name="demo_exp", base_dir=str(tmp_path))
    meta_path = os.path.join(tracker.get_checkpoint_dir(), "vlmintune_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                "ft_method": "lora",
            },
            f,
        )

    source, loaded_tracker = resolve_experiment_source(
        {
            "experiment": {"name": "demo_exp", "base_dir": str(tmp_path)},
            "eval": {"dataset_name": "lmms-lab/textvqa", "split": "validation", "source": "trained", "metric": "vqa_accuracy"},
        },
        EvalTarget(
            name="textvqa_validation",
            dataset_name="lmms-lab/textvqa",
            split="validation",
            source="trained",
            metric="vqa_accuracy",
        ),
    )

    assert source.kind == "trained"
    assert source.base_model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert source.checkpoint_path == tracker.get_checkpoint_dir()
    assert source.ft_method == "lora"
    assert source.output_dir == tracker.get_eval_dir("trained")
    assert loaded_tracker.exp_name == "demo_exp"


def test_resolve_experiment_source_uses_base_eval_dir(tmp_path):
    tracker = ExperimentTracker.create(exp_name="demo_exp", base_dir=str(tmp_path))
    meta_path = os.path.join(tracker.get_checkpoint_dir(), "vlmintune_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"base_model": "Qwen/Qwen2.5-VL-3B-Instruct"}, f)

    source, _ = resolve_experiment_source(
        {
            "experiment": {"name": "demo_exp", "base_dir": str(tmp_path)},
            "eval": {"dataset_name": "lmms-lab/textvqa", "split": "validation", "source": "base", "metric": "vqa_accuracy"},
        },
        EvalTarget(
            name="textvqa_validation",
            dataset_name="lmms-lab/textvqa",
            split="validation",
            source="base",
            metric="vqa_accuracy",
        ),
    )

    assert source.kind == "base"
    assert source.base_model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert source.checkpoint_path == ""
    assert source.output_dir == tracker.get_eval_dir("base")


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

    monkeypatch.setattr(HFDatasetsAdapter, "load_dataset", fake_load_dataset)

    result = evaluate_vqa_dataset(
        _DummyMethod(),
        EvalTarget(
            name="textvqa_validation",
            dataset_name="lmms-lab/textvqa",
            split="validation",
            source="trained",
            metric="vqa_accuracy",
            max_samples=1,
            streaming=True,
        ),
        str(tmp_path),
    )

    assert result["metrics"]["vqa_accuracy"] == 90.0

    with open(result["prediction_file"], "r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    assert os.path.basename(result["prediction_file"]) == "predictions.jsonl"
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
