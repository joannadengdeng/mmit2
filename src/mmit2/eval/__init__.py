"""Evaluation package exports."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "Method": ("mmit2.eval.methods.base", "Method"),
    "LocalMethod": ("mmit2.eval.methods.local_method", "LocalMethod"),
    "EvalTarget": ("mmit2.eval.run", "EvalTarget"),
    "parse_eval_target": ("mmit2.eval.run", "parse_eval_target"),
    "run_eval_config": ("mmit2.eval.run", "run_eval_config"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
