"""SSH training runner.

Usage::

    # From Python:
    from mmit2.training.runner import run
    run("configs/ssh_qlora.yaml")

    # From CLI:
    python -m mmit2.training.runner --config configs/ssh_qlora.yaml
"""
from __future__ import annotations

import argparse
import json

from mmit2.config.training_config import (
    config_to_trainer_dict,
    load_config,
)
from mmit2.config.runtime import run_remote_module
from mmit2.training.peft_env import ensure_peft_runtime_compatible


def run(config_path: str) -> None:
    """Load config and run training on the configured SSH server."""
    cfg = load_config(config_path)
    ensure_peft_runtime_compatible(cfg.training.ft_method, cfg.training.params)
    run_remote_module(
        cfg.runtime.ssh,
        module_name="mmit2.training",
        payload=config_to_trainer_dict(cfg),
        task_label="training",
        line_handler=_handle_remote_line,
    )


# ── Event formatting ─────────────────────────────────────────────────

def _handle_remote_line(line: str) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        print(line)
        return
    _print_event(event)


def _print_event(event: dict) -> None:
    """Pretty-print a JSON-line training event for the terminal."""
    etype = event.get("type", "")
    data = event.get("data", {})

    if etype == "metric":
        step = data.get("step", 0)
        total = data.get("total", 0)
        epoch = data.get("epoch", 0)
        total_epochs = data.get("total_epochs", 0)
        loss = data.get("loss", 0)
        avg = data.get("avg_loss", 0)
        lr = data.get("lr", 0)
        eta = data.get("eta", 0)

        m, s = divmod(int(eta), 60)
        eta_str = f"{m}m{s:02d}s" if m else f"{s}s"
        epoch_str = f"E{epoch+1}/{total_epochs} " if total_epochs else ""

        print(
            f"  {epoch_str}Step {step}/{total} | "
            f"Loss: {loss:.4f} | Avg: {avg:.4f} | "
            f"LR: {lr:.2e} | ETA: {eta_str}"
        )

    elif etype == "log":
        level = data.get("level", "INFO")
        message = data.get("message", "")
        print(f"  [{level}] {message}")

    elif etype == "status":
        status = data.get("status", "")
        result = data.get("result", "")
        if result:
            print(f"  [STATUS] {status} — {result}")
        else:
            print(f"  [STATUS] {status}")

    elif etype == "error":
        message = data.get("message", "")
        print(f"  [ERROR] {message}")
        tb = data.get("traceback", "")
        if tb:
            print(tb)


def main() -> None:
    parser = argparse.ArgumentParser(description="mmit2 SSH training runner")
    parser.add_argument("--config", required=True, help="Path to SSH training YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
