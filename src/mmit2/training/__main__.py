"""Headless CLI trainer for single-stage multimodal fine-tuning.

Usage::

    # YAML config:
    python -m mmit2.training --config configs/local_qlora.yaml

    # JSON config (from subprocess):
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

import yaml

from mmit2.config.training_config import config_to_trainer_dict, load_config
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


def main():
    parser = argparse.ArgumentParser(description="mmit2 headless trainer")
    parser.add_argument("--config-json", default=None,
                        help="Full training config as JSON string")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            raw_config = yaml.safe_load(f) or {}
        # Support both:
        # 1) the normalized trainer dict schema (training_method / method_params)
        # 2) the higher-level YAML schema used by repo configs (training.ft_method / training.params)
        if "training_method" in raw_config or "method_params" in raw_config:
            config = raw_config
        else:
            config = config_to_trainer_dict(load_config(args.config))
    elif args.config_json:
        config = json.loads(args.config_json)
    else:
        parser.error("Either --config or --config-json is required")

    try:
        if "data" not in config:
            _emit("error", {"message": "config must contain 'data' key"})
            sys.exit(1)

        model_path, train_config = _parse_train_config(config)

        if not model_path:
            _emit("error", {"message": "model.model_path is required"})
            sys.exit(1)

        trainer = Trainer(model_path)
        trainer.train(train_config)

    except Exception as e:
        _emit("error", {"message": str(e), "traceback": traceback.format_exc()})
        _emit("status", {"status": "failed"})
        sys.exit(1)


if __name__ == "__main__":
    main()
