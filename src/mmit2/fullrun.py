"""Full LoRA run: train once, then evaluate full datasets and benchmarks.

Usage:
    python -m mmit2.fullrun --config configs/colab_lora_full_eval.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import yaml

try:
    from google.colab import drive as colab_drive  # type: ignore[import-not-found]
except ImportError:
    colab_drive = None


_ALLOWED_TARGET_TYPES = {"hf_vqa", "pope_hf", "mme_hf"}


@dataclass
class EvalTarget:
    type: str
    name: str
    dataset_name: str
    split: str
    task_type: str = "open_vqa"
    metrics: List[str] = field(default_factory=lambda: ["auto"])
    max_new_tokens: int = 16
    temperature: float = 0.0
    max_samples: Optional[int] = None
    streaming: bool = True


def _default_target_name(target_type: str, dataset_name: str, split: str) -> str:
    dataset_short = dataset_name.rstrip("/").split("/")[-1].replace(".", "_").replace("-", "_")
    return f"{target_type}_{dataset_short}_{split}".lower()


def _parse_eval_targets(raw_eval: Dict[str, Any]) -> List[EvalTarget]:
    raw_targets = raw_eval.get("targets", [])
    if not raw_targets:
        raise ValueError("eval.targets must contain at least one target")

    default_max_new_tokens = int(raw_eval.get("max_new_tokens", 16))
    default_temperature = float(raw_eval.get("temperature", 0.0))
    parsed: List[EvalTarget] = []

    for idx, raw in enumerate(raw_targets):
        if not isinstance(raw, dict):
            raise ValueError(f"eval.targets[{idx}] must be a mapping")

        target_type = str(raw.get("type", "")).strip()
        if target_type not in _ALLOWED_TARGET_TYPES:
            raise ValueError(
                f"eval.targets[{idx}].type must be one of {sorted(_ALLOWED_TARGET_TYPES)}; "
                f"got '{target_type}'"
            )

        dataset_name = str(raw.get("dataset_name", "")).strip()
        if not dataset_name:
            if target_type == "pope_hf":
                dataset_name = "lmms-lab/POPE"
            elif target_type == "mme_hf":
                dataset_name = "lmms-lab/MME"
            else:
                raise ValueError(f"eval.targets[{idx}].dataset_name is required for '{target_type}'")

        split_default = "validation" if target_type == "hf_vqa" else "test"
        split = str(raw.get("split", split_default)).strip() or split_default
        name = str(raw.get("name", "")).strip() or _default_target_name(target_type, dataset_name, split)

        task_type = str(raw.get("task_type", "open_vqa")).strip() or "open_vqa"
        if target_type in {"pope_hf", "mme_hf"}:
            task_type = "yes_no"

        metrics = raw.get("metrics", ["auto"])
        if not isinstance(metrics, list) or not metrics:
            metrics = ["auto"]

        max_samples_raw = raw.get("max_samples")
        max_samples = int(max_samples_raw) if max_samples_raw not in (None, "", 0) else None

        parsed.append(
            EvalTarget(
                type=target_type,
                name=name,
                dataset_name=dataset_name,
                split=split,
                task_type=task_type,
                metrics=[str(metric) for metric in metrics],
                max_new_tokens=int(raw.get("max_new_tokens", default_max_new_tokens)),
                temperature=float(raw.get("temperature", default_temperature)),
                max_samples=max_samples,
                streaming=bool(raw.get("streaming", True)),
            )
        )

    return parsed


def _load_raw_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _prepare_runtime(cfg) -> None:
    runtime_mode = cfg.runtime.mode
    if runtime_mode == "local":
        return
    if runtime_mode != "colab":
        raise ValueError(f"mmit2.fullrun currently supports local or colab mode; got '{runtime_mode}'")

    colab_cfg = cfg.runtime.colab
    if colab_cfg.pip_install:
        print("[mmit2] Installing packages...")
        for pkg in colab_cfg.pip_install:
            print(f"  pip install {pkg}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", str(pkg)],
                check=False,
            )
        print()

    if colab_cfg.mount_drive:
        drive_root = os.path.join(colab_cfg.drive_mount_point, "MyDrive")
        try:
            if colab_drive is None:
                raise ImportError
            if os.path.isdir(drive_root):
                print(f"[mmit2] Google Drive already mounted at {colab_cfg.drive_mount_point}")
            else:
                print(f"[mmit2] Mounting Google Drive at {colab_cfg.drive_mount_point}")
                colab_drive.mount(colab_cfg.drive_mount_point)
        except ImportError:
            print("[mmit2] WARNING: Not running inside Colab; skipping Drive mount.")
        except Exception as exc:
            print(f"[mmit2] WARNING: Drive mount failed: {exc}")

    if colab_cfg.output_to_drive:
        drive_output = os.path.join(
            colab_cfg.drive_mount_point,
            "MyDrive",
            "mmit2_output",
            cfg.training.output_dir,
        )
        print(f"[mmit2] Output redirected to Drive: {drive_output}")
        cfg.training.output_dir = drive_output


def _build_trainer_config(cfg):
    from mmit2.config.training_config import config_to_trainer_dict
    from mmit2.training.trainer import TrainerConfig

    trainer_dict = config_to_trainer_dict(cfg)
    training_cfg = trainer_dict["training"]
    return trainer_dict, TrainerConfig(
        data_config=trainer_dict["data"],
        training_method=trainer_dict["training_method"],
        method_params=trainer_dict.get("method_params", {}),
        num_epochs=training_cfg["num_epochs"],
        per_device_batch_size=training_cfg["per_device_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        warmup_ratio=training_cfg["warmup_ratio"],
        weight_decay=training_cfg["weight_decay"],
        max_grad_norm=training_cfg["max_grad_norm"],
        save_steps=training_cfg["save_steps"],
        output_dir=training_cfg["output_dir"],
    )


def _ensure_prediction_dir(output_dir: str) -> str:
    prediction_dir = os.path.join(output_dir, "eval_predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return prediction_dir


def _iter_with_progress(items: Iterable[Any], total: Optional[int], desc: str) -> Iterator[Any]:
    try:
        from tqdm import tqdm

        yield from tqdm(items, total=total, desc=desc)
    except Exception:
        count = 0
        for item in items:
            count += 1
            if count == 1 or count % 100 == 0:
                if total and total > 0:
                    print(f"[mmit2] {desc}: {count}/{total}")
                else:
                    print(f"[mmit2] {desc}: {count}")
            yield item


def _prediction_path(prediction_dir: str, target_name: str) -> str:
    safe_name = target_name.replace("/", "_").replace(" ", "_")
    return os.path.join(prediction_dir, f"{safe_name}.jsonl")


def _build_vqa_column_map(dataset_name: str):
    from mmit2.data.datasets.base import ColumnMapping

    if dataset_name == "lmms-lab/VQAv2":
        return ColumnMapping(
            id_col="question_id",
            image_col="image",
            question_col="question",
            answer_col="answers",
        )
    if dataset_name == "lmms-lab/textvqa":
        return ColumnMapping(
            id_col="question_id",
            image_col="image",
            question_col="question",
            answer_col="answers",
        )
    if dataset_name == "lmms-lab/VizWiz-VQA":
        return ColumnMapping(
            id_col="question_id",
            image_col="image",
            question_col="question",
            answer_col="answers",
        )
    return None


def _evaluate_hf_vqa_target(method, target: EvalTarget, prediction_dir: str) -> Dict[str, Any]:
    from mmit2.data.adapters.hf_datasets import HFDatasetsAdapter
    from mmit2.data.types import EvalSample
    from mmit2.eval.metrics.scoring import auto_select_metric, score_prediction_multi

    adapter = HFDatasetsAdapter(
        dataset_name=target.dataset_name,
        split=target.split,
        column_map=_build_vqa_column_map(target.dataset_name),
        max_samples=target.max_samples,
        streaming=target.streaming,
        load_images=True,
    )
    total = len(adapter)
    total = total if total >= 0 else None
    prediction_path = _prediction_path(prediction_dir, target.name)

    metric_sums: Dict[str, float] = {}
    num_predictions = 0
    primary_metric = ""
    primary_reason = ""

    with open(prediction_path, "w", encoding="utf-8") as f:
        for sample in _iter_with_progress(adapter, total, f"Evaluating {target.name}"):
            ground_truth = sample.metadata.get("raw_answers") or ([sample.first_answer] if sample.first_answer else [])
            if not primary_metric:
                primary_metric, primary_reason = auto_select_metric(target.task_type, ground_truth)

            eval_sample = EvalSample(
                id=sample.id,
                image_path=sample.image_path,
                question=sample.first_question,
                ground_truth=ground_truth,
                metadata=sample.metadata,
            )
            prepared = method.prepare_eval_input(eval_sample, image_root="")
            prediction = method.generate(
                prepared,
                max_new_tokens=target.max_new_tokens,
                temperature=target.temperature,
            )
            scores = score_prediction_multi(
                prediction,
                ground_truth,
                task_type=target.task_type,
                metrics=target.metrics,
            )
            for metric_name, value in scores.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + float(value)

            record = {
                "id": sample.id,
                "question": sample.first_question,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "scores": scores,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_predictions += 1

    metrics = {
        metric_name: round(100.0 * total_value / max(1, num_predictions), 2)
        for metric_name, total_value in sorted(metric_sums.items())
    }
    return {
        "type": target.type,
        "dataset_name": target.dataset_name,
        "split": target.split,
        "num_predictions": num_predictions,
        "primary_metric": primary_metric,
        "primary_metric_reason": primary_reason,
        "metrics": metrics,
        "prediction_file": prediction_path,
    }


def _load_hf_rows(dataset_name: str, split: str, streaming: bool, max_samples: Optional[int]) -> tuple[Iterable[Dict[str, Any]], Optional[int]]:
    import datasets

    dataset = datasets.load_dataset(dataset_name, split=split, streaming=streaming)
    if max_samples is not None and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    total = max_samples if streaming and max_samples is not None else (len(dataset) if not streaming else None)
    return dataset, total


def _evaluate_pope_hf_target(method, target: EvalTarget, prediction_dir: str) -> Dict[str, Any]:
    from mmit2.data.datasets.base import handle_image_value
    from mmit2.data.types import EvalSample
    from mmit2.eval.metrics.vqa import normalize_answer

    rows, total = _load_hf_rows(target.dataset_name, target.split, target.streaming, target.max_samples)
    prediction_path = _prediction_path(prediction_dir, target.name)

    tp = fp = tn = fn = 0
    num_predictions = 0

    with open(prediction_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(_iter_with_progress(rows, total, f"Evaluating {target.name}")):
            if target.max_samples is not None and num_predictions >= target.max_samples:
                break

            image_path, image_meta = handle_image_value(row.get("image"), load_images=True)
            question = str(row.get("question", row.get("text", ""))).strip()
            ground_truth = str(row.get("answer", row.get("label", ""))).strip()
            eval_sample = EvalSample(
                id=str(row.get("question_id", row.get("id", idx))),
                image_path=image_path,
                question=question,
                ground_truth=ground_truth,
                metadata={**image_meta, "category": row.get("category", "")},
            )
            prepared = method.prepare_eval_input(eval_sample, image_root="")
            prediction = method.generate(
                prepared,
                max_new_tokens=target.max_new_tokens,
                temperature=target.temperature,
            )

            pred_ans = normalize_answer(prediction)
            pred_yes = pred_ans.startswith("yes")
            gt_yes = normalize_answer(ground_truth).startswith("yes")
            exact_match = 1.0 if pred_yes == gt_yes else 0.0

            if pred_yes and gt_yes:
                tp += 1
            elif pred_yes and not gt_yes:
                fp += 1
            elif not pred_yes and not gt_yes:
                tn += 1
            else:
                fn += 1

            record = {
                "id": eval_sample.id,
                "question": question,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "category": row.get("category", ""),
                "scores": {"exact_match": exact_match},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_predictions += 1

    total_count = tp + fp + tn + fn
    accuracy = (tp + tn) / total_count if total_count else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    yes_rate = (tp + fp) / total_count if total_count else 0.0

    return {
        "type": target.type,
        "dataset_name": target.dataset_name,
        "split": target.split,
        "num_predictions": num_predictions,
        "primary_metric": "f1",
        "metrics": {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "yes_rate": round(yes_rate * 100, 2),
        },
        "prediction_file": prediction_path,
    }


def _evaluate_mme_hf_target(method, target: EvalTarget, prediction_dir: str) -> Dict[str, Any]:
    from collections import defaultdict

    from mmit2.data.datasets.base import handle_image_value
    from mmit2.data.types import EvalSample
    from mmit2.eval.metrics.vqa import normalize_answer

    rows, total = _load_hf_rows(target.dataset_name, target.split, target.streaming, target.max_samples)
    prediction_path = _prediction_path(prediction_dir, target.name)

    category_results: Dict[str, List[bool]] = defaultdict(list)
    num_predictions = 0

    with open(prediction_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(_iter_with_progress(rows, total, f"Evaluating {target.name}")):
            if target.max_samples is not None and num_predictions >= target.max_samples:
                break

            image_path, image_meta = handle_image_value(row.get("image"), load_images=True)
            question = str(row.get("question", row.get("text", ""))).strip()
            ground_truth = str(row.get("answer", "")).strip()
            category = str(row.get("category", "unknown")).strip() or "unknown"

            eval_sample = EvalSample(
                id=str(row.get("question_id", row.get("id", idx))),
                image_path=image_path,
                question=question,
                ground_truth=ground_truth,
                metadata={**image_meta, "category": category},
            )
            prepared = method.prepare_eval_input(eval_sample, image_root="")
            prediction = method.generate(
                prepared,
                max_new_tokens=target.max_new_tokens,
                temperature=target.temperature,
            )

            pred_yes = normalize_answer(prediction).startswith("yes")
            gt_yes = normalize_answer(ground_truth).startswith("yes")
            is_correct = pred_yes == gt_yes
            category_results[category].append(is_correct)

            record = {
                "id": eval_sample.id,
                "question": question,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "category": category,
                "scores": {"exact_match": 1.0 if is_correct else 0.0},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_predictions += 1

    perception_cats = {
        "existence", "count", "position", "color", "poster",
        "celebrity", "scene", "landmark", "artwork", "ocr",
    }
    cognition_cats = {
        "commonsense_reasoning", "numerical_calculation",
        "text_translation", "code_reasoning",
    }

    subtask_scores: Dict[str, float] = {}
    perception_total = 0.0
    cognition_total = 0.0
    for category, results in sorted(category_results.items()):
        score = (sum(results) / len(results) * 100.0) if results else 0.0
        subtask_scores[f"subtask_{category}"] = round(score, 2)
        cat_key = category.lower().replace(" ", "_")
        if cat_key in perception_cats:
            perception_total += score
        elif cat_key in cognition_cats:
            cognition_total += score

    total_score = perception_total + cognition_total
    metrics = {
        "perception": round(perception_total, 2),
        "cognition": round(cognition_total, 2),
        "total": round(total_score, 2),
        **subtask_scores,
    }
    return {
        "type": target.type,
        "dataset_name": target.dataset_name,
        "split": target.split,
        "num_predictions": num_predictions,
        "primary_metric": "total",
        "metrics": metrics,
        "prediction_file": prediction_path,
    }


def _evaluate_target(method, target: EvalTarget, prediction_dir: str) -> Dict[str, Any]:
    if target.type == "hf_vqa":
        return _evaluate_hf_vqa_target(method, target, prediction_dir)
    if target.type == "pope_hf":
        return _evaluate_pope_hf_target(method, target, prediction_dir)
    if target.type == "mme_hf":
        return _evaluate_mme_hf_target(method, target, prediction_dir)
    raise ValueError(f"Unsupported target type: {target.type}")


def run(config_path: str) -> Dict[str, Any]:
    from mmit2.config.training_config import load_config
    from mmit2.eval.methods.local_method import LocalMethod
    from mmit2.training.trainer import Trainer

    raw_cfg = _load_raw_config(config_path)
    targets = _parse_eval_targets(raw_cfg.get("eval", {}))
    cfg = load_config(config_path)
    _prepare_runtime(cfg)
    trainer_dict, trainer_config = _build_trainer_config(cfg)

    print("=" * 80)
    print("mmit2 Full LoRA Run")
    print("=" * 80)
    print(f"Model: {cfg.model.model_path}")
    print(f"Training method: {trainer_config.training_method}")
    print(f"Training dataset: {trainer_config.data_config.get('data_path')} ({trainer_config.data_config.get('split')})")
    print(f"Output dir: {trainer_config.output_dir}")
    print(f"Eval targets: {[target.name for target in targets]}")
    print()

    trainer = Trainer(cfg.model.model_path)
    trainer.train(trainer_config)

    final_checkpoint = os.path.join(trainer_config.output_dir, "final")
    if not os.path.isdir(final_checkpoint):
        raise FileNotFoundError(f"Final checkpoint not found: {final_checkpoint}")

    method = LocalMethod.from_checkpoint(
        base_model_id=cfg.model.model_path,
        checkpoint_path=final_checkpoint,
        ft_method=trainer_dict["training_method"],
    )
    prediction_dir = _ensure_prediction_dir(trainer_config.output_dir)

    eval_results = {}
    for target in targets:
        print()
        print("=" * 80)
        print(f"Eval Target: {target.name}")
        print("=" * 80)
        result = _evaluate_target(method, target, prediction_dir)
        eval_results[target.name] = result
        print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))

    summary = {
        "training": {
            "model_path": cfg.model.model_path,
            "training_method": trainer_config.training_method,
            "training_dataset": trainer_config.data_config,
            "output_dir": trainer_config.output_dir,
            "final_checkpoint": final_checkpoint,
        },
        "eval_targets": [asdict(target) for target in targets],
        "eval_results": eval_results,
    }
    summary_path = os.path.join(trainer_config.output_dir, "fullrun_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("Full Run Summary")
    print("=" * 80)
    print(f"Summary JSON: {summary_path}")
    print(json.dumps({name: result["metrics"] for name, result in eval_results.items()}, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full LoRA train + full eval flow")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
