"""Local-only evaluation flow under a single experiment folder."""
from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional

from vlmintune.data.datasets.registry import DATASET_SPECS, get_dataset_spec
from vlmintune.training.experiment import ExperimentTracker

from vlmintune.eval.method import LocalMethod
from vlmintune.eval.vqa import score_prediction


@dataclass(frozen=True)
class EvalTarget:
    name: str
    dataset_name: str
    split: str
    source: str
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


SUPPORTED_EVAL_SOURCES = {
    "trained",
    "base",
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
    raw_eval = raw_eval or {}
    raw = raw_eval

    dataset_name = str(raw.get("dataset_name", "")).strip()
    if not dataset_name:
        raise ValueError("eval.dataset_name is required")

    spec = get_dataset_spec(dataset_name)
    if spec is None or spec.data_model is None:
        raise ValueError(
            f"Unsupported eval.dataset_name '{dataset_name}'. "
            f"Supported: {sorted(DATASET_SPECS)}"
        )

    split = str(raw.get("split", "")).strip() or str(spec.data_model.default_eval_split).strip()
    if not split:
        raise ValueError(f"eval split could not be resolved for dataset '{dataset_name}'")

    raw_source = raw.get("source")
    if not isinstance(raw_source, str):
        raise ValueError("eval.source is required and must be a string")
    if raw_source not in SUPPORTED_EVAL_SOURCES:
        raise ValueError(
            f"eval.source must be exactly one of {sorted(SUPPORTED_EVAL_SOURCES)}"
        )

    metric = spec.data_model.metric_family

    max_samples_raw = raw.get("max_samples")
    max_samples = int(max_samples_raw) if max_samples_raw not in (None, "", 0) else None
    name = str(raw.get("name", "")).strip() or default_eval_name(dataset_name, split)

    return EvalTarget(
        name=name,
        dataset_name=dataset_name,
        split=split,
        source=raw_source,
        max_new_tokens=int(raw.get("max_new_tokens", raw_eval.get("max_new_tokens", 16))),
        temperature=float(raw.get("temperature", raw_eval.get("temperature", 0.0))),
        max_samples=max_samples,
        streaming=bool(raw.get("streaming", True)),
        metric=metric,
    )


def prediction_path(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "predictions.jsonl")


def emit_eval_debug_examples(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    print()
    print("First 5 eval examples")
    print(json.dumps(records, indent=2, ensure_ascii=False))


def evaluate_dataset(method, target: EvalTarget, output_dir: str) -> Dict[str, Any]:
    from vlmintune.data.hf_datasets import HFDatasetsAdapter
    from vlmintune.data.types import EvalSample

    adapter = HFDatasetsAdapter(
        dataset_name=target.dataset_name,
        split=target.split,
        max_samples=target.max_samples,
        streaming=target.streaming,
        load_images=True,
        usage="eval",
    )
    data_model = getattr(adapter.profile, "data_model", None)
    if data_model is None:
        raise ValueError(f"Dataset '{target.dataset_name}' is missing eval data model metadata.")
    total = len(adapter) if len(adapter) >= 0 else None
    prediction_file = prediction_path(output_dir)

    metric_sums: Dict[str, float] = {}
    num_predictions = 0
    debug_records: list[dict[str, Any]] = []

    with open(prediction_file, "w", encoding="utf-8") as f:
        for sample in iter_with_progress(adapter, total, f"Evaluating {target.name}"):
            eval_answers = sample.eval_answers

            eval_sample = EvalSample(
                id=sample.id,
                image_path=sample.image_path,
                question=sample.question,
                eval_answers=eval_answers,
                metadata=sample.metadata,
            )
            prepared = method.prepare_eval_input(eval_sample)
            prediction = method.generate(
                prepared,
                max_new_tokens=target.max_new_tokens,
                temperature=target.temperature,
            )
            scores = score_prediction(target.metric, prediction=prediction, ground_truth=eval_answers)
            for metric_name, value in scores.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + float(value)

            record = {
                "id": sample.id,
                "question": sample.question,
                "prediction": prediction,
                "ground_truth": eval_answers,
                "scores": scores,
            }
            if len(debug_records) < 5:
                debug_records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_predictions += 1

    emit_eval_debug_examples(debug_records)

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


def load_checkpoint_meta(checkpoint_path: str) -> Dict[str, Any]:
    meta_path = os.path.join(checkpoint_path, "vlmintune_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def resolve_experiment_source(
    raw_cfg: Dict[str, Any],
    target: EvalTarget,
) -> tuple[EvalSource, ExperimentTracker]:
    experiment_cfg = raw_cfg.get("experiment", {}) or {}
    experiment_name = str(experiment_cfg.get("name", "")).strip()
    base_dir = str(experiment_cfg.get("base_dir", "")).strip() or "experiments"
    if not experiment_name:
        raise ValueError("experiment.name is required for evaluation")

    tracker = ExperimentTracker.load_by_name(base_dir, experiment_name)
    model_cfg = raw_cfg.get("model", {}) or {}
    configured_base_model_id = str(model_cfg.get("model_path", "")).strip()
    checkpoint_path = tracker.get_checkpoint_dir()

    checkpoint_meta: Dict[str, Any] = {}
    if os.path.isdir(checkpoint_path):
        try:
            checkpoint_meta = load_checkpoint_meta(checkpoint_path)
        except FileNotFoundError:
            checkpoint_meta = {}

    if target.source == "trained":
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        base_model_id = configured_base_model_id or str(checkpoint_meta.get("base_model", "")).strip()
        if not base_model_id:
            raise ValueError(
                "Could not determine base model id for eval.source='trained'. "
                "Set model.model_path in the eval config or ensure the checkpoint has "
                "vlmintune_meta.json with a base_model entry."
            )
        ft_method = str(checkpoint_meta.get("ft_method", "")).strip()
        if not ft_method:
            raise ValueError(
                f"Could not determine ft_method for checkpoint: {checkpoint_path}"
            )
        source = EvalSource(
            kind="trained",
            base_model_id=base_model_id,
            output_dir=tracker.get_eval_dir("trained"),
            checkpoint_path=checkpoint_path,
            ft_method=ft_method,
            experiment_name=tracker.exp_name,
        )
        return source, tracker

    if not configured_base_model_id:
        raise ValueError("model.model_path is required when eval.source='base'")

    source = EvalSource(
        kind="base",
        base_model_id=configured_base_model_id,
        output_dir=tracker.get_eval_dir("base"),
        checkpoint_path="",
        ft_method="",
        experiment_name=tracker.exp_name,
    )
    return source, tracker


def run_eval_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    eval_target = parse_eval_target(raw_cfg.get("eval", {}))
    source, tracker = resolve_experiment_source(raw_cfg, eval_target)

    with tracker.capture_eval_log(source.kind):
        try:
            print("=" * 80)
            print("vlmintune Eval Run")
            print("=" * 80)
            print(f"Source: {source.kind}")
            print(f"Experiment: {source.experiment_name}")
            print(f"Output dir: {source.output_dir}")
            print(f"Model: {source.base_model_id}")
            print(f"Checkpoint: {source.checkpoint_path or '<base model only>'}")
            print(f"Eval dataset: {eval_target.dataset_name} ({eval_target.split})")
            print()

            if source.kind == "trained":
                method = LocalMethod.from_checkpoint(
                    base_model_id=source.base_model_id,
                    checkpoint_path=source.checkpoint_path,
                    ft_method=source.ft_method,
                )
            else:
                method = LocalMethod.from_base_model(source.base_model_id)

            eval_result = evaluate_dataset(method, eval_target, source.output_dir)
            summary = {
                "experiment_name": source.experiment_name,
                "source": source.kind,
                "model_path": source.base_model_id,
                "dataset_name": eval_result["dataset_name"],
                "split": eval_result["split"],
                "metric": eval_target.metric,
                "num_predictions": eval_result["num_predictions"],
                "metrics": eval_result["metrics"],
            }
            tracker.write_eval_summary(source.kind, summary)

            print(json.dumps(eval_result["metrics"], indent=2, ensure_ascii=False))
            print()
            print("=" * 80)
            print("Eval Summary")
            print("=" * 80)
            print(f"Summary JSON: {tracker.get_eval_summary_path(source.kind)}")
            print(f"Predictions: {tracker.get_predictions_path(source.kind)}")
            return summary
        except Exception:
            print(traceback.format_exc())
            raise


__all__ = [
    "EvalSource",
    "EvalTarget",
    "evaluate_dataset",
    "parse_eval_target",
    "resolve_experiment_source",
    "run_eval_config",
]
