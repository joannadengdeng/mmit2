"""Evaluation execution flow for the initial release."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, Optional

from vlmintune.training.experiment import ExperimentTracker
from vlmintune.training.peft_env import ensure_peft_runtime_compatible


@dataclass(frozen=True)
class EvalTarget:
    name: str
    dataset_name: str
    split: str
    max_new_tokens: int = 16
    temperature: float = 0.0
    max_samples: Optional[int] = None
    streaming: bool = True
    metric: str = ""


@dataclass(frozen=True)
class EvalSource:
    kind: str
    base_model_id: str
    output_dir: str
    checkpoint_path: str = ""
    ft_method: str = ""
    experiment_name: str = ""


SUPPORTED_EVAL_DATASETS = {
    "lmms-lab/textvqa",
}


def iter_with_progress(items: Iterable[Any], total: Optional[int], desc: str) -> Iterator[Any]:
    try:
        from tqdm import tqdm

        yield from tqdm(items, total=total, desc=desc)
    except Exception:
        count = 0
        for item in items:
            count += 1
            if count == 1 or count % 100 == 0:
                if total and total > 0:
                    print(f"[vlmintune] {desc}: {count}/{total}")
                else:
                    print(f"[vlmintune] {desc}: {count}")
            yield item


def default_eval_name(dataset_name: str, split: str) -> str:
    dataset_short = dataset_name.rstrip("/").split("/")[-1].replace(".", "_").replace("-", "_")
    return f"{dataset_short}_{split}".lower()


def parse_eval_target(raw_eval: Dict[str, Any]) -> EvalTarget:
    from vlmintune.eval.metrics.vqa import validate_metric

    raw_eval = raw_eval or {}
    if "targets" in raw_eval:
        raw_targets = raw_eval.get("targets", [])
        if len(raw_targets) != 1:
            raise ValueError(
                "Initial release supports exactly one eval dataset per run. "
                "Set eval.dataset_name / eval.split / eval.max_samples instead of eval.targets."
            )
        raw = raw_targets[0] or {}
    else:
        raw = raw_eval

    dataset_name = str(raw.get("dataset_name", "")).strip()
    if not dataset_name:
        raise ValueError("eval.dataset_name is required")

    if dataset_name not in SUPPORTED_EVAL_DATASETS:
        raise ValueError(
            f"Unsupported eval.dataset_name '{dataset_name}'. "
            f"Supported: {sorted(SUPPORTED_EVAL_DATASETS)}"
        )
    split = str(raw.get("split", "")).strip()
    if not split:
        raise ValueError("eval.split is required")

    name = str(raw.get("name", "")).strip() or default_eval_name(dataset_name, split)
    raw_metric = raw.get("metric")
    if not isinstance(raw_metric, str):
        raise ValueError("eval.metric is required and must be a string")
    metric = validate_metric(raw_metric)
    max_samples_raw = raw.get("max_samples")
    max_samples = int(max_samples_raw) if max_samples_raw not in (None, "", 0) else None

    return EvalTarget(
        name=name,
        dataset_name=dataset_name,
        split=split,
        max_new_tokens=int(raw.get("max_new_tokens", raw_eval.get("max_new_tokens", 16))),
        temperature=float(raw.get("temperature", raw_eval.get("temperature", 0.0))),
        max_samples=max_samples,
        streaming=bool(raw.get("streaming", True)),
        metric=metric,
    )


def create_eval_output_dir(
    *,
    base_dir: str = "eval_outputs",
    model_path: str,
    dataset_name: str,
    explicit_dir: str = "",
) -> str:
    if explicit_dir:
        os.makedirs(explicit_dir, exist_ok=True)
        return explicit_dir

    model_short = os.path.basename(model_path.rstrip("/")) or "model"
    dataset_short = os.path.basename(dataset_name.rstrip("/")) or "dataset"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{model_short}_{dataset_short}_{ts}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def prediction_path(output_dir: str, target_name: str) -> str:
    pred_dir = os.path.join(output_dir, "eval_predictions")
    os.makedirs(pred_dir, exist_ok=True)
    safe_name = target_name.replace("/", "_").replace(" ", "_")
    return os.path.join(pred_dir, f"{safe_name}.jsonl")


def evaluate_vqa_dataset(method, target: EvalTarget, output_dir: str) -> Dict[str, Any]:
    from vlmintune.data.adapters.hf_datasets import HFDatasetsAdapter
    from vlmintune.data.types import EvalSample
    from vlmintune.eval.metrics.vqa import score_textvqa_prediction
    adapter = HFDatasetsAdapter(
        dataset_name=target.dataset_name,
        split=target.split,
        max_samples=target.max_samples,
        streaming=target.streaming,
        load_images=True,
    )
    total = len(adapter) if len(adapter) >= 0 else None
    prediction_file = prediction_path(output_dir, target.name)

    metric_sums: Dict[str, float] = {}
    num_predictions = 0

    with open(prediction_file, "w", encoding="utf-8") as f:
        for sample in iter_with_progress(adapter, total, f"Evaluating {target.name}"):
            ground_truth = sample.metadata.get("raw_answers") or ([sample.first_answer] if sample.first_answer else [])

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
            scores = score_textvqa_prediction(
                prediction=prediction,
                ground_truth=ground_truth,
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
        "dataset_name": target.dataset_name,
        "split": target.split,
        "num_predictions": num_predictions,
        "metrics": metrics,
        "prediction_file": prediction_file,
    }


def resolve_experiment_source(
    raw_cfg: Dict[str, Any],
) -> tuple[EvalSource, ExperimentTracker]:
    experiment_cfg = raw_cfg.get("experiment", {}) or {}
    experiment_name = str(experiment_cfg.get("name", "")).strip()
    base_dir = str(experiment_cfg.get("base_dir", "")).strip()
    if not experiment_name:
        raise ValueError("experiment.name is required when evaluating a saved experiment")
    if not base_dir:
        raise ValueError("experiment.base_dir is required when evaluating a saved experiment")

    tracker = ExperimentTracker.load_by_name(base_dir, experiment_name)
    base_model_id = (
        str((raw_cfg.get("model", {}) or {}).get("model_path", "")).strip()
        or tracker.meta.model
    )
    if not base_model_id:
        raise ValueError(
            "Could not determine base model id. Set model.model_path in the eval config "
            "or ensure the experiment summary contains it."
        )

    tracker_training_cfg = tracker.meta.config.get("training", {}) or {}
    tracker_method_params = (
        tracker.meta.config.get("method_params", {})
        or tracker_training_cfg.get("params", {})
        or {}
    )
    ensure_peft_runtime_compatible(tracker.meta.method, tracker_method_params)
    source = EvalSource(
        kind="experiment",
        base_model_id=base_model_id,
        output_dir=tracker.meta.exp_dir,
        checkpoint_path=tracker.resolve_checkpoint_path(),
        ft_method=tracker.meta.method,
        experiment_name=tracker.meta.exp_id,
    )
    return source, tracker


def resolve_baseline_source(raw_cfg: Dict[str, Any], dataset_name: str) -> EvalSource:
    model_cfg = raw_cfg.get("model", {}) or {}
    base_model_id = str(model_cfg.get("model_path", "")).strip()
    if not base_model_id:
        raise ValueError(
            "Set experiment.name to evaluate a saved run, or set model.model_path "
            "to evaluate a base-model baseline"
        )
    if str(model_cfg.get("checkpoint_path", "")).strip():
        raise ValueError(
            "Baseline eval only supports an unfine-tuned base model. "
            "Use experiment.name / experiment.base_dir to evaluate a trained run."
        )
    if str(model_cfg.get("ft_method", "")).strip():
        raise ValueError(
            "Baseline eval does not accept model.ft_method. "
            "Use experiment.name / experiment.base_dir to evaluate a trained run."
        )
    if model_cfg.get("method_params"):
        raise ValueError(
            "Baseline eval does not accept model.method_params. "
            "Use experiment.name / experiment.base_dir to evaluate a trained run."
        )

    eval_cfg = raw_cfg.get("eval", {}) or {}
    output_dir = create_eval_output_dir(
        base_dir=str(eval_cfg.get("base_dir", "eval_outputs")).strip() or "eval_outputs",
        model_path=base_model_id,
        dataset_name=dataset_name,
        explicit_dir=str(eval_cfg.get("output_dir", "")).strip(),
    )
    return EvalSource(
        kind="baseline",
        base_model_id=base_model_id,
        output_dir=output_dir,
    )


def run_eval_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    from vlmintune.eval.methods.local_method import LocalMethod

    eval_target = parse_eval_target(raw_cfg.get("eval", {}))
    experiment_name = str((raw_cfg.get("experiment", {}) or {}).get("name", "")).strip()
    tracker = None
    if experiment_name:
        source, tracker = resolve_experiment_source(raw_cfg)
        if source.checkpoint_path and not os.path.isdir(source.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {source.checkpoint_path}")
    else:
        source = resolve_baseline_source(raw_cfg, eval_target.dataset_name)

    print("=" * 80)
    print("vlmintune Eval Run")
    print("=" * 80)
    if source.kind == "experiment":
        assert tracker is not None
        print("Source: experiment")
        print(f"Experiment: {source.experiment_name}")
        print(f"Experiment dir: {source.output_dir}")
    else:
        print("Source: baseline")
        print(f"Output dir: {source.output_dir}")
    print(f"Model: {source.base_model_id}")
    print(f"Checkpoint: {source.checkpoint_path or '<base model only>'}")
    print(f"Eval dataset: {eval_target.dataset_name} ({eval_target.split})")
    print()

    if source.kind == "experiment":
        method = LocalMethod.from_checkpoint(
            base_model_id=source.base_model_id,
            checkpoint_path=source.checkpoint_path,
            ft_method=source.ft_method,
        )
    else:
        method = LocalMethod.from_base_model(source.base_model_id)

    eval_result = evaluate_vqa_dataset(method, eval_target, source.output_dir)
    if tracker is not None:
        tracker.log_eval(eval_target.name, eval_result["metrics"])
    print(json.dumps(eval_result["metrics"], indent=2, ensure_ascii=False))

    summary = {
        "source": {
            "kind": source.kind,
            "experiment_name": source.experiment_name,
            "output_dir": source.output_dir,
            "checkpoint": source.checkpoint_path,
        },
        "model": {
            "model_path": source.base_model_id,
            "ft_method": source.ft_method,
        },
        "eval_target": asdict(eval_target),
        "eval_result": eval_result,
    }
    summary_path = os.path.join(source.output_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("Eval Summary")
    print("=" * 80)
    print(f"Summary JSON: {summary_path}")
    return summary


__all__ = [
    "EvalSource",
    "EvalTarget",
    "parse_eval_target",
    "run_eval_config",
]
