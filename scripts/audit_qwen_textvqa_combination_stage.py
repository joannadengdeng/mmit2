#!/usr/bin/env python3
"""Strictly validate one completed Qwen TextVQA combination stage."""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import sys
from typing import Any

import yaml

METHOD_SPECS = {
    "mores_lora": ("adapter_config.json", 2e-4),
    "mores_dora": ("adapter_config.json", 2e-4),
    "reft_lora": ("adapter_config.json", 2e-4),
}
ADAPTER_METHODS = {
    "mores_lora",
    "mores_dora",
    "reft_lora",
}
JOINT_METHOD_SPECS = {
    "mores_lora": {
        "recipe": "mores_lora_v1",
        "structure_methods": ["mores", "lora"],
        "composition_order": ["mores", "lora"],
        "component_recipes": {"mores": "mores", "lora": "lora_v1"},
        "checkpoint_components": {
            "lora": "adapter_model.safetensors",
            "mores": "mores_tuned.pt",
        },
    },
    "mores_dora": {
        "recipe": "mores_dora_v1",
        "structure_methods": ["mores", "dora"],
        "composition_order": ["mores", "dora"],
        "component_recipes": {"mores": "mores", "dora": "dora_v1"},
        "checkpoint_components": {
            "dora": "adapter_model.safetensors",
            "mores": "mores_tuned.pt",
        },
    },
    "reft_lora": {
        "recipe": "reft_lora_v1",
        "structure_methods": ["reft", "lora"],
        "composition_order": ["reft", "lora"],
        "component_recipes": {"reft": "reft_tied_rank4_p4_s4_all_layers_v1", "lora": "lora_v1"},
        "checkpoint_components": {
            "lora": "adapter_model.safetensors",
            "reft": "reft_tuned.pt",
        },
    },
}
def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is non-finite: {number!r}")
    return number


def _preprocessing_total_skipped(record: Any) -> Any:
    if not isinstance(record, dict) or record.get("type") != "data_summary":
        return None
    data = record.get("data")
    if not isinstance(data, dict) or data.get("kind") != "preprocessing_coverage":
        return None
    return data.get("total_skipped")


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty {label}: {path}")


def audit_run(
    *,
    experiments_dir: Path,
    run_prefix: str,
    method: str,
    train_samples: int,
    eval_samples: int,
    grad_acc: int,
    max_length: int,
    epochs: int,
    seed: int,
    scope: str = "all",
) -> dict[str, Any]:
    if method not in METHOD_SPECS:
        raise ValueError(f"unsupported TextVQA combination: {method}")
    if scope not in {"all", "checkpoint"}:
        raise ValueError(f"unsupported audit scope: {scope}")

    marker_name, expected_learning_rate = METHOD_SPECS[method]
    run_name = f"{run_prefix}_{method}_n{train_samples}_s{seed}"
    run_dir = experiments_dir / run_name
    checkpoint_dir = run_dir / "checkpoint"
    eval_dir = run_dir / "eval_trained"

    config_path = checkpoint_dir / "train_config.yaml"
    metadata_path = checkpoint_dir / "vlmintune_meta.json"
    marker_path = checkpoint_dir / marker_name
    eval_config_path = run_dir / "eval_trained_config.yaml"
    eval_path = eval_dir / "eval.json"
    predictions_path = eval_dir / "predictions.jsonl"
    eval_ids_path = eval_dir / "eval_ids.json"

    _require_nonempty_file(config_path, "training config")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected_config = {
        "model": "qwen25vl_3b_instruct",
        "dataset": "lmms-lab/textvqa",
        "method": method,
        "epochs": epochs,
        "learning_rate": expected_learning_rate,
        "batch_size": 1,
        "gradient_accumulation_steps": grad_acc,
        "max_length": max_length,
        "max_samples": train_samples,
        "seed": seed,
        "output_dir": str(checkpoint_dir),
    }
    if not isinstance(config, dict) or set(config) != set(expected_config):
        raise ValueError(
            f"{run_name}: training config keys={sorted(config or {})}, "
            f"expected={sorted(expected_config)}"
        )
    for key, expected in expected_config.items():
        actual = config.get(key)
        if key == "learning_rate":
            if not math.isclose(_finite(actual, f"{run_name}: learning_rate"), expected):
                raise ValueError(
                    f"{run_name}: learning_rate={actual!r}, expected {expected!r}"
                )
        elif actual != expected:
            raise ValueError(
                f"{run_name}: config {key}={actual!r}, expected {expected!r}"
            )

    _require_nonempty_file(marker_path, "checkpoint marker")
    if method in ADAPTER_METHODS:
        adapter_weights = (
            checkpoint_dir / "adapter_model.safetensors",
            checkpoint_dir / "adapter_model.bin",
        )
        if not any(path.is_file() and path.stat().st_size > 0 for path in adapter_weights):
            raise ValueError(f"{run_name}: adapter checkpoint has no weight file")
    joint_spec = JOINT_METHOD_SPECS[method]
    for component, filename in joint_spec["checkpoint_components"].items():
        _require_nonempty_file(
            checkpoint_dir / filename,
            f"{component} checkpoint component",
        )
    _require_nonempty_file(metadata_path, "checkpoint metadata")
    metadata = _load_json(metadata_path)
    if metadata.get("model_name") != "qwen25vl_3b_instruct":
        raise ValueError(f"{run_name}: wrong model metadata: {metadata}")
    if metadata.get("ft_method") != method:
        raise ValueError(f"{run_name}: wrong method metadata: {metadata}")
    final_loss = _finite(metadata.get("final_loss"), f"{run_name}: final_loss")
    expected_joint_metadata = {
        "recipe": joint_spec["recipe"],
        "combination_recipe": joint_spec["recipe"],
        "structure_methods": joint_spec["structure_methods"],
        "composition_order": joint_spec["composition_order"],
        "component_recipes": joint_spec["component_recipes"],
        "checkpoint_components": joint_spec["checkpoint_components"],
    }
    field_labels = {
        "recipe": "recipe",
        "combination_recipe": "combination",
        "structure_methods": "structure",
        "composition_order": "composition",
        "component_recipes": "component recipes",
        "checkpoint_components": "checkpoint components",
    }
    for field, expected in expected_joint_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"{run_name}: wrong {field_labels[field]} metadata: "
                f"expected={expected!r}, actual={metadata.get(field)!r}"
            )

    train_logs = sorted(glob.glob(str(run_dir / "train" / "run_*.log")))
    if not train_logs:
        raise ValueError(f"{run_name}: missing training log")
    total_skipped = None
    with open(train_logs[-1], "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "preprocessing_coverage" not in line:
                continue
            try:
                candidate = _preprocessing_total_skipped(json.loads(line))
            except json.JSONDecodeError:
                continue
            if candidate is not None:
                total_skipped = candidate
    if total_skipped != 0:
        raise ValueError(f"{run_name}: total_skipped={total_skipped!r}, expected 0")

    checkpoint_result = {
        "run_name": run_name,
        "final_loss": final_loss,
        "total_skipped": 0,
    }
    if scope == "checkpoint":
        return checkpoint_result

    _require_nonempty_file(eval_config_path, "evaluation config")
    with eval_config_path.open("r", encoding="utf-8") as handle:
        eval_config = yaml.safe_load(handle)
    expected_eval_config = {
        "model": {"name": "qwen25vl_3b_instruct"},
        "experiment": {"name": run_name, "base_dir": str(experiments_dir)},
        "eval": {
            "source": "trained",
            "dataset_name": "lmms-lab/textvqa",
            "split": "validation",
            "max_samples": eval_samples,
            "sample_seed": seed,
            "shuffle_buffer_size": 10_000,
            "max_new_tokens": 16,
            "temperature": 0.0,
        },
    }
    if eval_config != expected_eval_config:
        raise ValueError(f"{run_name}: wrong evaluation config: {eval_config}")

    _require_nonempty_file(eval_path, "evaluation summary")
    summary = _load_json(eval_path)
    expected_summary_fields = {
        "experiment_name": run_name,
        "source": "trained",
        "model_name": "qwen25vl_3b_instruct",
        "dataset_name": "lmms-lab/textvqa",
        "split": "validation",
        "metric": "vqa_accuracy",
        "num_predictions": eval_samples,
        "sample_seed": seed,
        "shuffle_buffer_size": 10_000,
    }
    for key, expected in expected_summary_fields.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{run_name}: eval {key}={summary.get(key)!r}, expected {expected!r}"
            )
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict) or "vqa_accuracy" not in metrics:
        raise ValueError(f"{run_name}: missing vqa_accuracy: {metrics!r}")
    vqa_accuracy = _finite(metrics["vqa_accuracy"], f"{run_name}: vqa_accuracy")

    _require_nonempty_file(predictions_path, "predictions")
    prediction_ids: list[str] = []
    prediction_score_sum = 0.0
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{run_name}: blank prediction line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{run_name}: invalid prediction JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{run_name}: prediction line {line_number} is not an object")
            if record.get("id") is None or not str(record["id"]).strip():
                raise ValueError(f"{run_name}: missing prediction id at line {line_number}")
            if not isinstance(record.get("question"), str) or not record["question"].strip():
                raise ValueError(f"{run_name}: invalid question at line {line_number}")
            if not isinstance(record.get("prediction"), str):
                raise ValueError(f"{run_name}: invalid prediction at line {line_number}")
            if not isinstance(record.get("ground_truth"), list) or not record["ground_truth"]:
                raise ValueError(f"{run_name}: invalid ground truth at line {line_number}")
            scores = record.get("scores")
            if not isinstance(scores, dict) or "vqa_accuracy" not in scores:
                raise ValueError(f"{run_name}: missing row vqa_accuracy at line {line_number}")
            score = _finite(
                scores["vqa_accuracy"],
                f"{run_name}: row {line_number} vqa_accuracy",
            )
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"{run_name}: row {line_number} vqa_accuracy out of range: {score}"
                )
            prediction_ids.append(str(record["id"]))
            prediction_score_sum += score
    if len(prediction_ids) != eval_samples:
        raise ValueError(
            f"{run_name}: predictions={len(prediction_ids)}, expected {eval_samples}"
        )
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError(f"{run_name}: duplicate prediction ids")
    calculated_accuracy = round(100.0 * prediction_score_sum / eval_samples, 2)
    if not math.isclose(vqa_accuracy, calculated_accuracy, abs_tol=1e-9):
        raise ValueError(
            f"{run_name}: summary vqa_accuracy={vqa_accuracy}, "
            f"calculated={calculated_accuracy}"
        )

    _require_nonempty_file(eval_ids_path, "evaluation ids")
    eval_ids = _load_json(eval_ids_path)
    expected_eval_id_fields = {
        "dataset_name": "lmms-lab/textvqa",
        "split": "validation",
        "sample_seed": seed,
        "shuffle_buffer_size": 10_000,
        "max_samples": eval_samples,
    }
    for key, expected in expected_eval_id_fields.items():
        if eval_ids.get(key) != expected:
            raise ValueError(
                f"{run_name}: eval_ids {key}={eval_ids.get(key)!r}, expected {expected!r}"
            )
    if [str(value) for value in eval_ids.get("ids", [])] != prediction_ids:
        raise ValueError(f"{run_name}: eval_ids do not match prediction ids")

    return {
        **checkpoint_result,
        "vqa_accuracy": vqa_accuracy,
        "predictions": len(prediction_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", required=True, type=Path)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--train-samples", required=True, type=int)
    parser.add_argument("--eval-samples", required=True, type=int)
    parser.add_argument("--grad-acc", required=True, type=int)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--scope", choices=("all", "checkpoint"), default="all")
    parser.add_argument("methods", nargs="+")
    args = parser.parse_args()
    if min(
        args.train_samples,
        args.eval_samples,
        args.grad_acc,
        args.max_length,
        args.epochs,
    ) <= 0:
        parser.error("sample counts and training dimensions must be positive")
    invalid = [method for method in args.methods if method not in METHOD_SPECS]
    if invalid:
        parser.error(f"unsupported methods: {invalid}")
    return args


def main() -> None:
    args = parse_args()
    for method in args.methods:
        try:
            result = audit_run(
                experiments_dir=args.experiments_dir.resolve(),
                run_prefix=args.run_prefix,
                method=method,
                train_samples=args.train_samples,
                eval_samples=args.eval_samples,
                grad_acc=args.grad_acc,
                max_length=args.max_length,
                epochs=args.epochs,
                seed=args.seed,
                scope=args.scope,
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"STRICT FAIL {method} scope={args.scope}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if args.scope == "checkpoint":
            print(
                "STRICT CHECKPOINT PASS {run_name}: final_loss={final_loss:.6f} "
                "skip=0".format(**result)
            )
        else:
            print(
                "STRICT PASS {run_name}: final_loss={final_loss:.6f} "
                "vqa_accuracy={vqa_accuracy:.2f} predictions={predictions} skip=0".format(
                    **result
                )
            )


if __name__ == "__main__":
    main()
