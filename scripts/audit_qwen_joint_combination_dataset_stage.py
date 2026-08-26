#!/usr/bin/env python3
"""Strictly audit a joint-combination run on a built-in image dataset."""
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
        "component_recipes": {
            "reft": "reft_tied_rank4_p4_s4_all_layers_v1",
            "lora": "lora_v1",
        },
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


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty {label}: {path}")


def _preprocessing_total_skipped(record: Any) -> Any:
    if not isinstance(record, dict) or record.get("type") != "data_summary":
        return None
    data = record.get("data")
    if not isinstance(data, dict) or data.get("kind") != "preprocessing_coverage":
        return None
    return data.get("total_skipped")


def audit_run(
    *,
    experiments_dir: Path,
    run_prefix: str,
    dataset: str,
    eval_split: str,
    metric: str,
    method: str,
    train_samples: int,
    eval_samples: int,
    grad_acc: int,
    max_length: int,
    epochs: int,
    seed: int,
    model: str = "qwen25vl_3b_instruct",
    learning_rate: float = 2e-4,
    scope: str = "all",
) -> dict[str, Any]:
    if method not in METHOD_SPECS:
        raise ValueError(f"unsupported joint combination: {method}")
    if scope not in {"all", "checkpoint"}:
        raise ValueError(f"unsupported audit scope: {scope}")

    spec = METHOD_SPECS[method]
    run_name = f"{run_prefix}_{method}_n{train_samples}_s{seed}"
    run_dir = experiments_dir / run_name
    checkpoint_dir = run_dir / "checkpoint"
    eval_dir = run_dir / "eval_trained"
    config_path = checkpoint_dir / "train_config.yaml"
    metadata_path = checkpoint_dir / "vlmintune_meta.json"

    _require_file(config_path, "training config")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected_config = {
        "model": model,
        "dataset": dataset,
        "method": method,
        "epochs": epochs,
        "learning_rate": learning_rate,
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
                raise ValueError(f"{run_name}: learning_rate={actual!r}, expected {expected!r}")
        elif actual != expected:
            raise ValueError(f"{run_name}: config {key}={actual!r}, expected {expected!r}")

    for component, filename in spec["checkpoint_components"].items():
        _require_file(checkpoint_dir / filename, f"{component} checkpoint component")
    _require_file(checkpoint_dir / "adapter_config.json", "adapter config")

    _require_file(metadata_path, "checkpoint metadata")
    metadata = _load_json(metadata_path)
    if metadata.get("model_name") != model:
        raise ValueError(f"{run_name}: wrong model metadata: {metadata}")
    if metadata.get("ft_method") != method:
        raise ValueError(f"{run_name}: wrong method metadata: {metadata}")
    final_loss = _finite(metadata.get("final_loss"), f"{run_name}: final_loss")
    expected_metadata = {
        "recipe": spec["recipe"],
        "combination_recipe": spec["recipe"],
        "structure_methods": spec["structure_methods"],
        "composition_order": spec["composition_order"],
        "component_recipes": spec["component_recipes"],
        "checkpoint_components": spec["checkpoint_components"],
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"{run_name}: metadata {field}={metadata.get(field)!r}, expected {expected!r}"
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

    result = {"run_name": run_name, "final_loss": final_loss, "total_skipped": 0}
    if scope == "checkpoint":
        return result

    eval_config_path = run_dir / "eval_trained_config.yaml"
    _require_file(eval_config_path, "evaluation config")
    with eval_config_path.open("r", encoding="utf-8") as handle:
        eval_config = yaml.safe_load(handle)
    expected_eval_config = {
        "model": {"name": model},
        "experiment": {"name": run_name, "base_dir": str(experiments_dir)},
        "eval": {
            "source": "trained",
            "dataset_name": dataset,
            "split": eval_split,
            "max_samples": eval_samples,
            "sample_seed": seed,
            "shuffle_buffer_size": 10_000,
            "max_new_tokens": 16,
            "temperature": 0.0,
        },
    }
    if eval_config != expected_eval_config:
        raise ValueError(f"{run_name}: wrong evaluation config: {eval_config}")

    eval_path = eval_dir / "eval.json"
    predictions_path = eval_dir / "predictions.jsonl"
    eval_ids_path = eval_dir / "eval_ids.json"
    _require_file(eval_path, "evaluation summary")
    summary = _load_json(eval_path)
    expected_summary = {
        "experiment_name": run_name,
        "source": "trained",
        "model_name": model,
        "dataset_name": dataset,
        "split": eval_split,
        "metric": metric,
        "num_predictions": eval_samples,
        "sample_seed": seed,
        "shuffle_buffer_size": 10_000,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"{run_name}: eval {field}={summary.get(field)!r}, expected {expected!r}"
            )
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict) or metric not in metrics:
        raise ValueError(f"{run_name}: missing summary metric {metric}: {metrics!r}")
    metric_value = _finite(metrics[metric], f"{run_name}: {metric}")

    _require_file(predictions_path, "predictions")
    prediction_ids: list[str] = []
    score_sum = 0.0
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{run_name}: blank prediction line {line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{run_name}: prediction line {line_number} is not an object")
            prediction_id = str(record.get("id", "")).strip()
            if not prediction_id:
                raise ValueError(f"{run_name}: missing prediction id at line {line_number}")
            if not isinstance(record.get("question"), str) or not record["question"].strip():
                raise ValueError(f"{run_name}: invalid question at line {line_number}")
            if not isinstance(record.get("prediction"), str):
                raise ValueError(f"{run_name}: invalid prediction at line {line_number}")
            if not isinstance(record.get("ground_truth"), list) or not record["ground_truth"]:
                raise ValueError(f"{run_name}: invalid ground truth at line {line_number}")
            scores = record.get("scores")
            if not isinstance(scores, dict) or metric not in scores:
                raise ValueError(f"{run_name}: missing row metric {metric} at line {line_number}")
            score = _finite(scores[metric], f"{run_name}: row {line_number} {metric}")
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{run_name}: row {line_number} {metric} out of range: {score}")
            prediction_ids.append(prediction_id)
            score_sum += score
    if len(prediction_ids) != eval_samples:
        raise ValueError(f"{run_name}: predictions={len(prediction_ids)}, expected {eval_samples}")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError(f"{run_name}: duplicate prediction ids")
    calculated_metric = round(100.0 * score_sum / eval_samples, 2)
    if not math.isclose(metric_value, calculated_metric, abs_tol=1e-9):
        raise ValueError(
            f"{run_name}: summary {metric}={metric_value}, calculated={calculated_metric}"
        )

    _require_file(eval_ids_path, "evaluation ids")
    eval_ids = _load_json(eval_ids_path)
    expected_eval_ids = {
        "dataset_name": dataset,
        "split": eval_split,
        "sample_seed": seed,
        "shuffle_buffer_size": 10_000,
        "max_samples": eval_samples,
    }
    for field, expected in expected_eval_ids.items():
        if eval_ids.get(field) != expected:
            raise ValueError(
                f"{run_name}: eval_ids {field}={eval_ids.get(field)!r}, expected {expected!r}"
            )
    if [str(value) for value in eval_ids.get("ids", [])] != prediction_ids:
        raise ValueError(f"{run_name}: eval_ids do not match prediction ids")
    return {**result, "metric": metric, "metric_value": metric_value, "predictions": eval_samples}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", required=True, type=Path)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--model", default="qwen25vl_3b_instruct")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval-split", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--train-samples", required=True, type=int)
    parser.add_argument("--eval-samples", required=True, type=int)
    parser.add_argument("--grad-acc", required=True, type=int)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--scope", choices=("all", "checkpoint"), default="all")
    parser.add_argument("method", choices=tuple(METHOD_SPECS))
    args = parser.parse_args()
    if min(
        args.train_samples,
        args.eval_samples,
        args.grad_acc,
        args.max_length,
        args.epochs,
    ) <= 0:
        parser.error("sample counts and training dimensions must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        result = audit_run(
            experiments_dir=args.experiments_dir.resolve(),
            run_prefix=args.run_prefix,
            model=args.model,
            dataset=args.dataset,
            eval_split=args.eval_split,
            metric=args.metric,
            method=args.method,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            grad_acc=args.grad_acc,
            max_length=args.max_length,
            epochs=args.epochs,
            seed=args.seed,
            learning_rate=args.learning_rate,
            scope=args.scope,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"STRICT FAIL {args.method} scope={args.scope}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.scope == "checkpoint":
        print(
            f"STRICT CHECKPOINT PASS {result['run_name']}: "
            f"final_loss={result['final_loss']:.6f} skip=0"
        )
    else:
        print(
            f"STRICT PASS {result['run_name']}: final_loss={result['final_loss']:.6f} "
            f"{result['metric']}={result['metric_value']:.2f} "
            f"predictions={result['predictions']} skip=0"
        )


if __name__ == "__main__":
    main()
