"""CLI entry point for evaluation."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import yaml

from mmit2.config.runtime import run_remote_module
from mmit2.config.training_config import load_runtime_config_dict
from mmit2.eval.run import run_eval_config


def _load_raw_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_hf_token(token: str | None, token_file: str | None) -> None:
    token = (token or "").strip()
    token_file = os.path.expanduser((token_file or "").strip())
    if not token and token_file:
        with open(token_file, "r", encoding="utf-8") as f:
            token = f.read().strip()
    if token:
        os.environ["HF_TOKEN"] = token


def run(config_path: str) -> None:
    raw_cfg = _load_raw_config(config_path)
    runtime = load_runtime_config_dict(raw_cfg)
    run_remote_module(
        runtime.ssh,
        module_name="mmit2.eval",
        payload=raw_cfg,
        task_label="evaluation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved experiment or base-model baseline")
    parser.add_argument("--config", default=None, help="Path to an SSH eval YAML config file")
    parser.add_argument("--config-json", default=None, help="Full eval config as JSON string")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token")
    parser.add_argument("--hf-token-file", default=None, help="Path to a file containing a Hugging Face token")
    args = parser.parse_args()
    _apply_hf_token(args.hf_token, args.hf_token_file)
    if args.config_json:
        run_eval_config(json.loads(args.config_json))
        return
    if args.config:
        run(args.config)
        return
    parser.error("Either --config or --config-json is required")


if __name__ == "__main__":
    main()
