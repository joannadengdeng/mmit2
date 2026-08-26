"""Headless trainer for the strict initial-release configuration.

Example YAML::

    model: qwen25vl_3b_instruct
    dataset: lmms-lab/textvqa
    method: qlora
    epochs: 1
    learning_rate: 0.0002
    batch_size: 4
    gradient_accumulation_steps: 4
    max_length: 2048
    max_samples: 0
    seed: 42
    output_dir: output/textvqa_qlora
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from vlmintune.config.training_config import (
    config_to_trainer_dict,
    load_config,
    load_config_dict,
)
from vlmintune.training.trainer import Trainer, TrainerConfig, emit


def apply_hf_token(token: str | None, token_file: str | None) -> None:
    token = (token or "").strip()
    token_file = os.path.expanduser((token_file or "").strip())
    if not token and token_file:
        with open(token_file, "r", encoding="utf-8") as file:
            token = file.read().strip()
    if token:
        os.environ["HF_TOKEN"] = token


def parse_train_config(config: dict) -> tuple[str, TrainerConfig]:
    """Convert a validated flat public config into the internal trainer config."""

    normalized = config_to_trainer_dict(load_config_dict(config))
    return normalized["model"], TrainerConfig(
        data_config={
            "dataset_name": normalized["dataset"],
            "max_samples": normalized["max_samples"],
            "sample_seed": normalized["seed"],
        },
        training_method=normalized["method"],
        num_epochs=normalized["epochs"],
        per_device_batch_size=normalized["batch_size"],
        gradient_accumulation_steps=normalized["gradient_accumulation_steps"],
        max_length=normalized["max_length"],
        learning_rate=normalized["learning_rate"],
        save_steps=0,
        seed=normalized["seed"],
        output_dir=normalized["output_dir"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="vlmintune headless trainer")
    parser.add_argument("--config-json", default=None, help="Strict flat training config as JSON")
    parser.add_argument("--config", default=None, help="Path to a strict flat training YAML file")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token")
    parser.add_argument("--hf-token-file", default=None, help="Path to a Hugging Face token file")
    args = parser.parse_args()
    apply_hf_token(args.hf_token, args.hf_token_file)

    if not args.config_json and not args.config:
        parser.error("Either --config or --config-json is required")

    try:
        if args.config_json:
            raw_config = json.loads(args.config_json)
        else:
            raw_config = config_to_trainer_dict(load_config(args.config))
        model_name, train_config = parse_train_config(raw_config)
        Trainer(model_name).train(train_config)
    except Exception as error:
        emit("error", {"message": str(error), "traceback": traceback.format_exc()})
        emit("status", {"status": "failed"})
        sys.exit(1)


if __name__ == "__main__":
    main()
    if os.environ.get("VLMINTUNE_FAST_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
