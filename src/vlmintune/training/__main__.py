"""Headless CLI trainer for single-stage multimodal fine-tuning.

Usage::

    python -m vlmintune.training --config experiment_setup/my_experiment/train_config.yaml

    # JSON config (used internally after YAML normalization):
    python -m vlmintune.training --config-json '{"model": {...}, "data": {...}, ...}'

Config schema::

    model:
      model_path: "Qwen/Qwen2.5-VL-3B-Instruct"
    data:
      dataset_name: "..."
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
import os
import sys
import traceback

from vlmintune.config.training_config import config_to_trainer_dict, load_config
from vlmintune.training.experiment import ExperimentTracker
from vlmintune.training.trainer import Trainer, TrainerConfig, emit


def apply_hf_token(token: str | None, token_file: str | None) -> None:
    token = (token or "").strip()
    token_file = os.path.expanduser((token_file or "").strip())
    if not token and token_file:
        with open(token_file, "r", encoding="utf-8") as f:
            token = f.read().strip()
    if token:
        os.environ["HF_TOKEN"] = token


def parse_train_config(config: dict) -> tuple[str, TrainerConfig]:
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
        max_length=training_cfg.get("max_length", 2048),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 0),
        dataloader_pin_memory=training_cfg.get("dataloader_pin_memory", False),
        dataloader_persistent_workers=training_cfg.get("dataloader_persistent_workers", False),
        learning_rate=training_cfg.get("learning_rate", 2e-5),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        save_steps=training_cfg.get("save_steps", 500),
        output_dir=training_cfg.get("output_dir", "output"),
    )
    return model_path, train_config


def create_experiment_tracker(config: dict, train_config: TrainerConfig) -> ExperimentTracker:
    experiment_cfg = config.get("experiment", {}) or {}
    exp_name = str(experiment_cfg.get("name", "")).strip()
    if not exp_name:
        raise ValueError("experiment.name is required")
    base_dir = str(experiment_cfg.get("base_dir", "")).strip() or "experiments"
    tracker = ExperimentTracker.create(
        exp_name=exp_name,
        base_dir=base_dir,
    )
    train_config.output_dir = tracker.get_train_dir()
    return tracker


def write_failed_train_summary(
    tracker: ExperimentTracker | None,
    model_path: str,
    train_config: TrainerConfig | None,
    error: Exception,
) -> None:
    if tracker is None or train_config is None:
        return
    tracker.write_train_summary(
        {
            "experiment_name": tracker.exp_name,
            "model_path": model_path,
            "training_method": train_config.training_method,
            "training_params": {
                "num_epochs": train_config.num_epochs,
                "per_device_batch_size": train_config.per_device_batch_size,
                "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                "max_length": train_config.max_length,
                "dataloader_num_workers": train_config.dataloader_num_workers,
                "dataloader_pin_memory": train_config.dataloader_pin_memory,
                "dataloader_persistent_workers": train_config.dataloader_persistent_workers,
                "learning_rate": train_config.learning_rate,
                "warmup_ratio": train_config.warmup_ratio,
                "weight_decay": train_config.weight_decay,
                "max_grad_norm": train_config.max_grad_norm,
                "save_steps": train_config.save_steps,
                "method_params": dict(train_config.method_params),
            },
            "data": dict(train_config.data_config),
            "result": {
                "status": "failed",
                "error": str(error),
            },
        }
    )


def main():
    parser = argparse.ArgumentParser(description="vlmintune headless trainer")
    parser.add_argument(
        "--config-json",
        default=None,
        help="Normalized training config as a JSON string",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a training YAML config file",
    )
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token")
    parser.add_argument("--hf-token-file", default=None, help="Path to a file containing a Hugging Face token")
    args = parser.parse_args()
    apply_hf_token(args.hf_token, args.hf_token_file)

    if args.config_json:
        config = json.loads(args.config_json)
    elif args.config:
        config = config_to_trainer_dict(load_config(args.config))
    else:
        parser.error("Either --config or --config-json is required")

    tracker = None
    model_path = ""
    train_config = None
    try:
        if "data" not in config:
            emit("error", {"message": "config must contain 'data' key"})
            sys.exit(1)

        model_path, train_config = parse_train_config(config)

        if not model_path:
            emit("error", {"message": "model.model_path is required"})
            sys.exit(1)

        tracker = create_experiment_tracker(config, train_config)
        with tracker.capture_train_log():
            try:
                emit(
                    "log",
                    {
                        "message": (
                            f"Experiment: {tracker.exp_name} "
                            f"(dir={tracker.exp_dir})"
                        ),
                        "level": "INFO",
                    },
                )

                trainer = Trainer(model_path, experiment_tracker=tracker)
                trainer.train(train_config)
            except Exception as e:
                write_failed_train_summary(tracker, model_path, train_config, e)
                emit("error", {"message": str(e), "traceback": traceback.format_exc()})
                emit("status", {"status": "failed"})
                raise

    except Exception as e:
        if tracker is None:
            emit("error", {"message": str(e), "traceback": traceback.format_exc()})
            emit("status", {"status": "failed"})
        sys.exit(1)


if __name__ == "__main__":
    main()
