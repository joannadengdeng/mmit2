import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmit2.data.adapters.hf_datasets import HFDatasetsAdapter
from mmit2.eval.run import (
    EvalTarget,
    _evaluate_vqa_dataset,
    _resolve_baseline_source,
    parse_eval_target,
)


def test_parse_eval_target_infers_defaults():
    target = parse_eval_target(
        {
            "dataset_name": "lmms-lab/VQAv2",
            "max_new_tokens": 12,
            "max_samples": 25,
        }
    )

    assert target.dataset_name == "lmms-lab/VQAv2"
    assert target.split == "validation"
    assert target.max_new_tokens == 12
    assert target.max_samples == 25


def test_parse_eval_target_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unsupported eval.dataset_name"):
        parse_eval_target({"dataset_name": "foo/bar"})


def test_parse_eval_target_rejects_removed_eval_dataset():
    with pytest.raises(ValueError, match="Unsupported eval.dataset_name"):
        parse_eval_target({"dataset_name": "lmms-lab/POPE"})


def test_parse_eval_target_rejects_multi_target_legacy_config():
    with pytest.raises(ValueError, match="exactly one eval dataset"):
        parse_eval_target(
            {
                "targets": [
                    {"dataset_name": "lmms-lab/VQAv2"},
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


def test_evaluate_vqav2_uses_multi_annotator_answers(monkeypatch, tmp_path):
    rows = [
        {
            "question_id": 123,
            "image": None,
            "question": "What animal is shown?",
            "multiple_choice_answer": "dog",
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
            name="vqav2_validation",
            dataset_name="lmms-lab/VQAv2",
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
