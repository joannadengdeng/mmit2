"""Runtime config types and shared SSH execution helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import sys
from typing import Any, Callable

try:
    import paramiko
except ImportError:
    paramiko = None


@dataclass
class SSHConfig:
    host: str = ""
    port: int = 22
    username: str = ""
    key_path: str = ""
    password: str = ""
    conda_env: str = ""


@dataclass
class RuntimeConfig:
    mode: str = "ssh"
    ssh: SSHConfig = field(default_factory=SSHConfig)


def _escape_single_quotes(text: str) -> str:
    return text.replace("'", "'\"'\"'")


def _build_connect_kwargs(ssh_cfg: Any) -> dict[str, Any]:
    connect_kwargs: dict[str, Any] = {
        "hostname": ssh_cfg.host,
        "port": ssh_cfg.port,
        "username": ssh_cfg.username,
        "timeout": 15,
    }
    if ssh_cfg.key_path:
        connect_kwargs["key_filename"] = os.path.expanduser(ssh_cfg.key_path)
    elif ssh_cfg.password:
        connect_kwargs["password"] = ssh_cfg.password
    return connect_kwargs


def run_remote_module(
    ssh_cfg: Any,
    *,
    module_name: str,
    payload: dict[str, Any],
    task_label: str,
    line_handler: Callable[[str], None] | None = None,
) -> None:
    """Run a remote mmit2 module over SSH and stream its stdout back locally."""
    if paramiko is None:
        print(f"[ERROR] SSH {task_label} requires paramiko: pip install 'mmit2[remote]'")
        sys.exit(1)

    print(f"[mmit2] Connecting to {ssh_cfg.username}@{ssh_cfg.host}:{ssh_cfg.port}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(**_build_connect_kwargs(ssh_cfg))
    except Exception as exc:
        print(f"[ERROR] SSH connection failed: {exc}")
        sys.exit(1)

    payload_json = json.dumps(payload, ensure_ascii=False)
    command = f"python3 -m {module_name} --config-json '{_escape_single_quotes(payload_json)}'"
    if ssh_cfg.conda_env:
        command = f"conda run -n {ssh_cfg.conda_env} {command}"

    print(f"[mmit2] Connected. Starting remote {task_label}...")
    _, stdout, stderr = client.exec_command(command, get_pty=False)

    try:
        for raw_line in iter(stdout.readline, ""):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line_handler is None:
                print(line)
            else:
                line_handler(line)
    except KeyboardInterrupt:
        print(f"\n[mmit2] Remote {task_label} interrupted by user.")

    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0:
        err = stderr.read().decode() if hasattr(stderr, "read") else ""
        err = err.strip() if err else ""
        if err:
            print(f"\n[ERROR] Remote {task_label} failed (exit {exit_code}):\n{err}")
        else:
            print(f"\n[ERROR] Remote {task_label} failed with exit code {exit_code}")
        client.close()
        sys.exit(exit_code)

    client.close()
    print(f"\n[mmit2] Remote {task_label} complete.")
