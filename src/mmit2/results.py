"""Minimal persistence for evaluation predictions and summary metrics."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class PredictionRecord:
    """A single inference result."""

    id: str
    question: str
    prediction: str
    ground_truth: Any = None
    status: str = "ok"
    error: str | None = None
    latency_s: float = 0.0
    score: float | None = None
    scores: dict[str, float] | None = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("ground_truth", "error", "score", "scores"):
            if data[key] is None:
                data.pop(key)
        if not data["latency_s"]:
            data.pop("latency_s")
        if not data["timestamp"]:
            data.pop("timestamp")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictionRecord":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclass
class RunHeader:
    """Stored metadata for one evaluation run."""

    run_id: str = ""
    status: str = "running"
    failure_reason: str = ""
    created_at: str = ""
    updated_at: str = ""
    model: dict[str, str] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics_summary: dict[str, Any] = field(default_factory=dict)


_DEFAULT_RESULTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "results")


class ResultsManager:
    """Save predictions, parameters, and aggregate metrics for one run."""

    def __init__(
        self,
        path: str,
        header: RunHeader,
        predictions: list[PredictionRecord],
    ) -> None:
        self.path = path
        self.header = header
        self.predictions = predictions

    @classmethod
    def create(
        cls,
        model_id: str,
        model_family: str,
        dataset_name: str,
        split: str = "validation",
        max_samples: int | None = None,
        streaming: bool = False,
        parameters: dict[str, Any] | None = None,
        expected_total: int | None = None,
        results_dir: str = _DEFAULT_RESULTS_DIR,
        run_id: str | None = None,
    ) -> "ResultsManager":
        """Create a new results file for an evaluation run."""
        now = datetime.now()
        if run_id is None:
            ts = now.strftime("%Y%m%d_%H%M%S")
            model_short = os.path.basename(model_id.rstrip("/"))
            dataset_short = os.path.basename(dataset_name.rstrip("/"))
            run_id = f"{model_short}_{dataset_short}_{ts}"

        os.makedirs(results_dir, exist_ok=True)
        dataset = {
            "name": dataset_name,
            "split": split,
        }
        if max_samples is not None:
            dataset["max_samples"] = max_samples
        if streaming:
            dataset["streaming"] = streaming
        if expected_total is not None:
            dataset["expected_total"] = expected_total

        header = RunHeader(
            run_id=run_id,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            model={"model_id": model_id, "family": model_family},
            dataset=dataset,
            parameters=parameters or {},
        )
        mgr = cls(
            path=os.path.join(results_dir, f"{run_id}.json"),
            header=header,
            predictions=[],
        )
        mgr.save()
        return mgr

    @classmethod
    def load(cls, path: str) -> "ResultsManager":
        """Load an existing results file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        header = RunHeader(**{
            key: value
            for key, value in data.get("header", {}).items()
            if key in RunHeader.__dataclass_fields__
        })
        predictions = [
            PredictionRecord.from_dict(item)
            for item in data.get("predictions", [])
        ]
        return cls(path=path, header=header, predictions=predictions)

    def add_prediction(self, record: PredictionRecord) -> None:
        """Append one prediction record."""
        if not record.timestamp:
            record.timestamp = datetime.now().isoformat()
        self.predictions.append(record)

    def save(self) -> None:
        """Atomically persist header and predictions to disk."""
        self.header.updated_at = datetime.now().isoformat()
        self.header.metrics_summary = self.compute_metrics()
        payload = {
            "header": asdict(self.header),
            "predictions": [record.to_dict() for record in self.predictions],
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    def mark_completed(self) -> None:
        self.header.status = "completed"
        self.header.failure_reason = ""
        self.save()

    def mark_failed(self, error: str = "") -> None:
        self.header.status = "failed"
        self.header.failure_reason = error
        self.save()

    def compute_metrics(self) -> dict[str, Any]:
        """Return aggregate metrics for the current predictions."""
        total = len(self.predictions)
        ok_predictions = [record for record in self.predictions if record.status == "ok"]
        summary: dict[str, Any] = {
            "num_predictions": total,
            "num_success": len(ok_predictions),
            "num_errors": total - len(ok_predictions),
        }
        if total:
            summary["success_rate"] = round(len(ok_predictions) / total, 4)

        latencies = [record.latency_s for record in ok_predictions if record.latency_s > 0]
        if latencies:
            summary["mean_latency_s"] = round(mean(latencies), 3)
            summary["total_latency_s"] = round(sum(latencies), 3)

        scored = [record.score for record in ok_predictions if record.score is not None]
        if scored:
            summary["accuracy"] = round(mean(scored) * 100, 2)

        metric_names = sorted({
            metric_name
            for record in ok_predictions
            for metric_name in (record.scores or {})
        })
        if metric_names:
            summary["metric_details"] = {
                metric_name: round(
                    mean(
                        record.scores[metric_name]
                        for record in ok_predictions
                        if record.scores and metric_name in record.scores
                    ) * 100,
                    2,
                )
                for metric_name in metric_names
            }

        return summary
