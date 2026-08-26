"""Simple lookup table for built-in training methods."""
from __future__ import annotations

from vlmintune.training.methods.base import TrainingMethod
from vlmintune.training.methods.dora import DoRAMethod
from vlmintune.training.methods.l2t import L2TMethod
from vlmintune.training.methods.lora import LoRAMethod, QLoRAMethod
from vlmintune.training.methods.mores import MoReSMethod
from vlmintune.training.methods.mores_dora import MoReSDoRAMethod
from vlmintune.training.methods.mores_lora import MoReSLoRAMethod
from vlmintune.training.methods.reft import ReFTMethod
from vlmintune.training.methods.reft_lora import ReFTLoRAMethod
from vlmintune.training.methods.vl_adapter import VLAdapterMethod

TrainingMethodType = type[TrainingMethod]

_TRAINING_METHODS: dict[str, TrainingMethodType] = {
    "qlora": QLoRAMethod,
    "lora": LoRAMethod,
    "dora": DoRAMethod,
    "l2t": L2TMethod,
    "mores": MoReSMethod,
    "mores_lora": MoReSLoRAMethod,
    "mores_dora": MoReSDoRAMethod,
    "reft": ReFTMethod,
    "reft_lora": ReFTLoRAMethod,
    "vl_adapter": VLAdapterMethod,
}


def list_training_methods() -> list[str]:
    """Return the registered training method names."""
    return list(_TRAINING_METHODS)


def get_training_method_cls(name: str) -> TrainingMethodType:
    """Return the training method class for ``name``."""
    try:
        return _TRAINING_METHODS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown training method '{name}'. Available: {sorted(_TRAINING_METHODS)}"
        ) from exc


def build_training_method(name: str) -> TrainingMethod:
    """Instantiate a training method."""
    return get_training_method_cls(name)()


__all__ = [
    "list_training_methods",
    "get_training_method_cls",
    "build_training_method",
]
