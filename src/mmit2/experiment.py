"""Minimal experiment tracking: parameters, summaries, and checkpoint path."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMeta:
    """Single persisted experiment record."""
    exp_id: str = ""
    status: str = "running"
    created_at: str = ""
    completed_at: str = ""
    error: str = ""
    method: str = ""
    model: str = ""
    dataset: str = ""
    num_samples: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    train_summary: Dict[str, Any] = field(default_factory=dict)
    eval_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    checkpoint_path: str = ""
    exp_dir: str = ""


class ExperimentTracker:
    """Persist one experiment's parameters and aggregate results."""

    def __init__(self, meta: ExperimentMeta):
        self.meta = meta

    @classmethod
    def create(
        cls,
        base_dir: str = "experiments",
        method: str = "",
        model: str = "",
        dataset: str = "",
        num_samples: int = 0,
        config: Optional[Dict[str, Any]] = None,
        exp_id: Optional[str] = None,
    ) -> "ExperimentTracker":
        """Create a new experiment directory and initial summary."""
        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")

        if exp_id is None:
            os.makedirs(base_dir, exist_ok=True)
            existing = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
            next_num = len(existing) + 1
            samples_label = f"{num_samples // 1000}k" if num_samples >= 1000 else str(num_samples)
            exp_id = f"exp_{next_num:03d}_{method}_{samples_label}_{ts}"

        exp_dir = os.path.join(base_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoint"), exist_ok=True)

        meta = ExperimentMeta(
            exp_id=exp_id,
            status="running",
            created_at=now.isoformat(),
            method=method,
            model=model,
            dataset=dataset,
            num_samples=num_samples,
            config=config or {},
            exp_dir=exp_dir,
        )

        tracker = cls(meta=meta)
        tracker._save_summary()
        return tracker

    @classmethod
    def load(cls, exp_dir: str) -> "ExperimentTracker":
        """Load a persisted experiment."""
        summary_path = os.path.join(exp_dir, "summary.json")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"No summary.json in {exp_dir}")
        with open(summary_path) as f:
            data = json.load(f)
        meta = ExperimentMeta(**{
            k: v for k, v in data.items()
            if k in ExperimentMeta.__dataclass_fields__
        })
        return cls(meta=meta)

    def log_train_summary(
        self,
        avg_loss: float = 0.0,
        total_steps: int = 0,
        train_time_s: float = 0.0,
        trainable_params: int = 0,
        total_params: int = 0,
        **extra,
    ) -> None:
        """Record final training summary."""
        self.meta.train_summary = {
            "avg_loss": round(avg_loss, 6),
            "total_steps": total_steps,
            "train_time_s": round(train_time_s, 1),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": round(100 * trainable_params / max(1, total_params), 4),
            **extra,
        }
        self._save_summary()

    def log_eval(
        self,
        benchmark: str,
        scores: Dict[str, float],
    ) -> None:
        """Record aggregate evaluation scores for one benchmark."""
        self.meta.eval_results[benchmark] = scores
        self._save_summary()

    def set_checkpoint_path(self, path: str) -> None:
        """Record the saved checkpoint directory."""
        self.meta.checkpoint_path = path
        self._save_summary()

    def get_checkpoint_dir(self) -> str:
        """Return the default checkpoint directory for this experiment."""
        return os.path.join(self.meta.exp_dir, "checkpoint")

    def finalize(self, status: str = "completed") -> None:
        """Mark the experiment complete and persist it."""
        self.meta.status = status
        self.meta.completed_at = datetime.now().isoformat()
        self._save_summary()

    def fail(self, error: str = "") -> None:
        """Mark the experiment as failed."""
        self.meta.error = error
        self.finalize(status="failed")

    def _save_summary(self) -> None:
        """Write the full summary file."""
        summary_path = os.path.join(self.meta.exp_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(asdict(self.meta), f, indent=2, ensure_ascii=False)
