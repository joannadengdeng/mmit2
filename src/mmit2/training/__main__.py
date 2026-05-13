"""Headless CLI trainer for single-stage multimodal fine-tuning.

Usage::

    # SSH YAML config (delegates to the SSH runner):
    python -m mmit2.training --config configs/ssh_qlora.yaml

    # JSON config (used by the remote machine after SSH dispatch):
    python -m mmit2.training --config-json '{"model": {...}, "data": {...}, ...}'

Config schema::

    model:
      model_path: "Qwen/Qwen2.5-VL-3B-Instruct"
    data:
      adapter: "hf_datasets"
      data_path: "..."
      split: "train"
    training_method: "qlora"
    method_params: {lora_r: 8}
    training:
      num_epochs: 1
      learning_rate: 2e-5
      per_device_batch_size: 4

Output format (one JSON object per line)::

    {"type":"status","data":{"status":"loading"}}
    {"type":"metric","data":{"step":1,"loss":2.34,...}}
    {"type":"status","data":{"status":"completed"}}
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from mmit2.training.experiment import ExperimentTracker
from mmit2.training.trainer import Trainer, TrainerConfig, _emit


def _parse_train_config(config: dict) -> tuple[str, TrainerConfig]:
    """Parse a single-stage config dict into TrainerConfig."""
    model_cfg = config.get("model", {})
    model_path = model_cfg.get("model_path", "")

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    train_config = TrainerConfig(
        data_config=data_cfg,
        training_method=config.get("training_method", "qlora"),
        method_params=config.get("method_params", {}),
        num_epochs=training_cfg.get("num_epochs", 1),
        per_device_batch_size=training_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-5),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        save_steps=training_cfg.get("save_steps", 500),
        output_dir=training_cfg.get("output_dir", "output"),
    )
    return model_path, train_config


def _create_experiment_tracker(config: dict, model_path: str, train_config: TrainerConfig) -> ExperimentTracker:
    experiment_cfg = config.get("experiment", {}) or {}
    exp_name = str(experiment_cfg.get("name", "")).strip() or None
    base_dir = str(experiment_cfg.get("base_dir", "")).strip() or train_config.output_dir
    data_cfg = train_config.data_config or {}
    dataset_name = (
        str(data_cfg.get("data_path", "")).strip()
        or str(data_cfg.get("dataset_name", "")).strip()
    )
    num_samples = int(data_cfg.get("max_samples", 0) or 0)
    tracker = ExperimentTracker.create(
        base_dir=base_dir,
        method=train_config.training_method,
        model=model_path,
        dataset=dataset_name,
        num_samples=num_samples,
        config=config,
        exp_id=exp_name,
    )
    train_config.output_dir = tracker.meta.exp_dir
    return tracker


def main():
    parser = argparse.ArgumentParser(description="mmit2 headless trainer")
    parser.add_argument(
        "--config-json",
        default=None,
        help="Normalized training config as a JSON string",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to an SSH training YAML config file",
    )
    args = parser.parse_args()

    if args.config_json:
        config = json.loads(args.config_json)
    elif args.config:
        from mmit2.training.runner import run as run_over_ssh

        run_over_ssh(args.config)
        return
    else:
        parser.error("Either --config or --config-json is required")

    tracker = None
    try:
        if "data" not in config:
            _emit("error", {"message": "config must contain 'data' key"})
            sys.exit(1)

        model_path, train_config = _parse_train_config(config)

        if not model_path:
            _emit("error", {"message": "model.model_path is required"})
            sys.exit(1)

        tracker = _create_experiment_tracker(config, model_path, train_config)
        _emit(
            "log",
            {
                "message": (
                    f"Experiment: {tracker.meta.exp_id} "
                    f"(dir={tracker.meta.exp_dir})"
                ),
                "level": "INFO",
            },
        )

        trainer = Trainer(model_path, experiment_tracker=tracker)
        trainer.train(train_config)
        tracker.finalize(status="completed")

    except Exception as e:
        if tracker is not None:
            tracker.fail(str(e))
        _emit("error", {"message": str(e), "traceback": traceback.format_exc()})
        _emit("status", {"status": "failed"})
        sys.exit(1)


if __name__ == "__main__":
    main()
