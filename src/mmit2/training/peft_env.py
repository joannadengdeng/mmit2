"""PEFT runtime compatibility checks.

Environment validation belongs at the runner layer, not inside individual
training methods. This module centralizes lightweight preflight checks for
LoRA-like methods.
"""
from __future__ import annotations

from importlib import metadata
from typing import Any, Mapping

_LORA_FAMILY_METHODS = {"lora", "qlora", "dora"}


def _parse_version_tuple(version_text: str) -> tuple[int, ...]:
    parts = []
    for chunk in version_text.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def method_uses_peft(method_name: str, method_params: Mapping[str, Any] | None = None) -> bool:
    method_name = str(method_name).strip().lower()
    params = method_params or {}
    if method_name in _LORA_FAMILY_METHODS:
        return True
    if method_name == "l2t":
        base_method = str(params.get("base_method", "lora")).strip().lower()
        return base_method in _LORA_FAMILY_METHODS
    return False


def ensure_peft_runtime_compatible(
    method_name: str,
    method_params: Mapping[str, Any] | None = None,
) -> None:
    """Raise early if the environment is known-incompatible with PEFT LoRA paths."""
    if not method_uses_peft(method_name, method_params):
        return

    try:
        version_text = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        return

    if _parse_version_tuple(version_text) >= (0, 16, 0):
        return

    raise RuntimeError(
        "Detected incompatible torchao "
        f"{version_text}. PEFT LoRA paths require torchao >= 0.16.0 if torchao is "
        "installed. This project does not require torchao, so the simplest fix is:\n"
        "pip uninstall -y torchao\n"
        "Then rerun the command."
    )
