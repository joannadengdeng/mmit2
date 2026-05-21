"""Deterministic experiment layout for training and evaluation runs."""
from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterator

_EVAL_DIRS = {
    "trained": "eval_trained",
    "base": "eval_base",
}


class TeeStream:
    """Mirror writes to the terminal and a log file."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_run_output(log_path: str) -> Iterator[None]:
    """Capture stdout/stderr into a run log while preserving terminal output."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


@dataclass(frozen=True)
class ExperimentTracker:
    """Small helper for one experiment folder and its fixed output layout."""

    exp_name: str
    base_dir: str
    exp_dir: str

    @classmethod
    def create(
        cls,
        exp_name: str,
        base_dir: str = "experiments",
    ) -> "ExperimentTracker":
        exp_name = str(exp_name)
        if not exp_name:
            raise ValueError("experiment.name is required")
        base_dir = str(base_dir or "experiments")
        os.makedirs(base_dir, exist_ok=True)
        exp_dir = os.path.join(base_dir, exp_name)
        if os.path.exists(exp_dir):
            raise FileExistsError(
                f"Experiment '{exp_name}' already exists at {exp_dir}. "
                "Choose a different experiment.name."
            )
        tracker = cls(exp_name=exp_name, base_dir=base_dir, exp_dir=exp_dir)
        tracker.ensure_layout()
        return tracker

    @classmethod
    def load_by_name(cls, base_dir: str, exp_name: str) -> "ExperimentTracker":
        exp_name = str(exp_name)
        if not exp_name:
            raise ValueError("experiment.name is required")
        base_dir = str(base_dir or "experiments")
        exp_dir = os.path.join(base_dir, exp_name)
        if not os.path.isdir(exp_dir):
            raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")
        tracker = cls(exp_name=exp_name, base_dir=base_dir, exp_dir=exp_dir)
        tracker.ensure_layout()
        return tracker

    def ensure_layout(self) -> None:
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.get_checkpoint_dir(), exist_ok=True)
        os.makedirs(self.get_train_dir(), exist_ok=True)
        for source in _EVAL_DIRS:
            os.makedirs(self.get_eval_dir(source), exist_ok=True)

    def get_checkpoint_dir(self) -> str:
        return os.path.join(self.exp_dir, "checkpoint")

    def get_train_dir(self) -> str:
        return os.path.join(self.exp_dir, "train")

    def get_train_summary_path(self) -> str:
        return os.path.join(self.get_train_dir(), "train_summary.json")

    def get_train_log_path(self) -> str:
        return os.path.join(self.get_train_dir(), "run.log")

    def get_eval_dir(self, source: str) -> str:
        if source not in _EVAL_DIRS:
            raise ValueError(
                f"Unsupported eval source '{source}'. "
                f"Expected one of {sorted(_EVAL_DIRS)}."
            )
        return os.path.join(self.exp_dir, _EVAL_DIRS[source])

    def get_eval_summary_path(self, source: str) -> str:
        return os.path.join(self.get_eval_dir(source), "eval.json")

    def get_eval_log_path(self, source: str) -> str:
        return os.path.join(self.get_eval_dir(source), "run.log")

    def get_predictions_path(self, source: str) -> str:
        return os.path.join(self.get_eval_dir(source), "predictions.jsonl")

    def write_train_summary(self, payload: dict[str, Any]) -> None:
        self.write_json(self.get_train_summary_path(), payload)

    def write_eval_summary(self, source: str, payload: dict[str, Any]) -> None:
        self.write_json(self.get_eval_summary_path(source), payload)

    def write_json(self, path: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def capture_train_log(self) -> contextlib.AbstractContextManager[None]:
        return tee_run_output(self.get_train_log_path())

    def capture_eval_log(self, source: str) -> contextlib.AbstractContextManager[None]:
        return tee_run_output(self.get_eval_log_path(source))
