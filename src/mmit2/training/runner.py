"""Training runner — dispatches config to the appropriate execution mode.

Supports three modes:

- **local**: Run training directly on the local machine (GPU required).
- **colab**: Colab environment setup (pip install, Drive mount) then local run.
- **ssh**: Run training on a remote server via SSH, streaming events back.

Usage::

    # From Python:
    from mmit2.training.runner import run
    run("configs/local_qlora.yaml")

    # From CLI:
    python -m mmit2.training --config configs/local_qlora.yaml
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import metadata

try:
    from google.colab import drive as colab_drive  # type: ignore[import-not-found]
except ImportError:
    colab_drive = None

try:
    import paramiko
except ImportError:
    paramiko = None

from mmit2.config.training_config import (
    TrainingConfig,
    config_to_trainer_dict,
    load_config,
)


def _parse_version_tuple(version_text: str) -> tuple[int, ...]:
    parts = []
    for chunk in version_text.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _ensure_compatible_torchao() -> None:
    """Remove preinstalled torchao versions that PEFT cannot work with."""
    try:
        version_text = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        return

    if _parse_version_tuple(version_text) >= (0, 16, 0):
        return

    print(
        "[mmit2] Removing incompatible torchao "
        f"{version_text} (PEFT LoRA requires >=0.16.0 if torchao is installed)"
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        check=False,
    )
    print()


def _maybe_mount_drive(colab_cfg) -> bool:
    """Attempt to mount Google Drive and report whether MyDrive is usable."""
    drive_root = os.path.join(colab_cfg.drive_mount_point, "MyDrive")
    if not colab_cfg.mount_drive:
        return os.path.isdir(drive_root)

    if os.path.isdir(drive_root):
        print(f"[mmit2] Google Drive already mounted at {colab_cfg.drive_mount_point}")
        return True

    try:
        if colab_drive is None:
            raise ImportError
        print(f"[mmit2] Mounting Google Drive at {colab_cfg.drive_mount_point}")
        colab_drive.mount(colab_cfg.drive_mount_point)
    except ImportError:
        print("[mmit2] WARNING: Not running in Colab, skipping Drive mount.")
    except Exception as exc:
        print(
            "[mmit2] WARNING: Drive mount failed. If you want Drive output, run this "
            "first in a notebook cell:\n"
            "from google.colab import drive\n"
            f"drive.mount('{colab_cfg.drive_mount_point}')\n"
            f"Underlying error: {exc}"
        )

    return os.path.isdir(drive_root)


def run(config_path: str) -> None:
    """Load config and dispatch to the appropriate runner.

    Parameters
    ----------
    config_path : str
        Path to a YAML config file.
    """
    cfg = load_config(config_path)

    if cfg.runtime.mode == "local":
        _run_local(cfg)
    elif cfg.runtime.mode == "colab":
        _run_colab(cfg)
    elif cfg.runtime.mode == "ssh":
        _run_ssh(cfg)
    else:
        raise ValueError(f"Unknown runtime mode: {cfg.runtime.mode}")


# ── Local mode ───────────────────────────────────────────────────────

def _run_local(cfg: TrainingConfig) -> None:
    """Run training directly on the local machine as a subprocess."""
    trainer_dict = config_to_trainer_dict(cfg)
    config_json = json.dumps(trainer_dict)
    cmd = [sys.executable, "-m", "mmit2.training", "--config-json", config_json]

    print(f"[mmit2] Starting local training: {cfg.training.ft_method}")
    print(f"[mmit2] Model: {cfg.model.model_path}")
    print(f"[mmit2] Output: {cfg.training.output_dir}")
    print()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    try:
        for line in iter(proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                _print_event(event)
            except json.JSONDecodeError:
                print(line)
    finally:
        proc.wait()

    if proc.returncode != 0:
        stderr_out = proc.stderr.read().strip()
        if stderr_out:
            print(f"\n[ERROR] {stderr_out}")
        sys.exit(proc.returncode)

    print("\n[mmit2] Training complete.")


# ── Colab mode ───────────────────────────────────────────────────────

def _run_colab(cfg: TrainingConfig) -> None:
    """Run training in a Google Colab environment.

    Steps:
      1. pip install missing packages
      2. Mount Google Drive if requested
      3. Adjust output_dir to point to Drive
      4. Run training locally (Colab has a local GPU)
    """
    colab_cfg = cfg.runtime.colab

    # 1. pip install
    if colab_cfg.pip_install:
        print("[mmit2] Installing packages...")
        for pkg in colab_cfg.pip_install:
            print(f"  pip install {pkg}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", str(pkg)],
                check=False,
            )
        print()

    _ensure_compatible_torchao()

    # 2. Mount Drive
    drive_available = _maybe_mount_drive(colab_cfg)

    # 3. Adjust output_dir
    if colab_cfg.output_to_drive:
        if drive_available:
            drive_output = os.path.join(
                colab_cfg.drive_mount_point, "MyDrive", "mmit2_output",
                cfg.training.output_dir,
            )
            print(f"[mmit2] Output redirected to Drive: {drive_output}")
            cfg.training.output_dir = drive_output
        else:
            print(
                "[mmit2] WARNING: Drive output requested, but Google Drive is not mounted. "
                f"Keeping local output at: {cfg.training.output_dir}"
            )

    # 4. Run locally
    _run_local(cfg)


# ── SSH mode ─────────────────────────────────────────────────────────

def _run_ssh(cfg: TrainingConfig) -> None:
    """Run training on a remote server via SSH.

    Sends the JSON config via ``--config-json`` and streams JSON-line events back.
    """
    if paramiko is None:
        print("[ERROR] SSH mode requires paramiko: pip install 'mmit2[remote]'")
        sys.exit(1)

    ssh_cfg = cfg.runtime.ssh
    trainer_dict = config_to_trainer_dict(cfg)
    config_json = json.dumps(trainer_dict)

    print(f"[mmit2] Connecting to {ssh_cfg.username}@{ssh_cfg.host}:{ssh_cfg.port}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs = {
        "hostname": ssh_cfg.host,
        "port": ssh_cfg.port,
        "username": ssh_cfg.username,
        "timeout": 15,
    }
    if ssh_cfg.key_path:
        connect_kwargs["key_filename"] = os.path.expanduser(ssh_cfg.key_path)
    elif ssh_cfg.password:
        connect_kwargs["password"] = ssh_cfg.password

    try:
        client.connect(**connect_kwargs)
    except Exception as e:
        print(f"[ERROR] SSH connection failed: {e}")
        sys.exit(1)

    print("[mmit2] Connected. Starting remote training...")

    # Build remote command
    config_json_escaped = config_json.replace("'", "'\"'\"'")
    cmd = f"python3 -m mmit2.training --config-json '{config_json_escaped}'"
    if ssh_cfg.conda_env:
        cmd = f"conda run -n {ssh_cfg.conda_env} {cmd}"

    _, stdout, stderr = client.exec_command(cmd, get_pty=False)

    # Stream JSON-line events
    try:
        for line in iter(stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                _print_event(event)
            except json.JSONDecodeError:
                print(line)
    except KeyboardInterrupt:
        print("\n[mmit2] Interrupted by user.")

    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0:
        err = stderr.read().decode() if hasattr(stderr, "read") else ""
        err = err.strip() if err else ""
        if err:
            print(f"\n[ERROR] Remote training failed (exit {exit_code}):\n{err}")
        else:
            print(f"\n[ERROR] Remote training failed with exit code {exit_code}")
        client.close()
        sys.exit(exit_code)

    client.close()
    print("\n[mmit2] Remote training complete.")


# ── Event formatting ─────────────────────────────────────────────────

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
