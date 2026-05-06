"""mmit2 — Multimodal Instruction Tuning library.

Quick start
-----------
>>> from mmit2 import Method
>>> method = Method.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
"""
from __future__ import annotations

from . import registry
from mmit2.data import CanonicalSample, EvalSample, Turn
from mmit2.data.adapters.hf_datasets import DatasetProfile, HFDatasetsAdapter
from mmit2.eval.methods.base import Method
from mmit2.eval.methods.local_method import LocalMethod
from mmit2.results import PredictionRecord, ResultsManager
from mmit2.training.methods.dora import DoRAMethod
from mmit2.training.methods.freeze import FreezeTuningMethod
from mmit2.training.methods.l2t import L2TMethod
from mmit2.training.methods.lora import LoRAMethod, QLoRAMethod
from mmit2.training.preprocessors.chat_template import ChatTemplatePreprocessor

__all__ = [
    "registry",
    "Method",
    "LocalMethod",
    "ResultsManager",
    "PredictionRecord",
    "CanonicalSample",
    "EvalSample",
    "Turn",
    "HFDatasetsAdapter",
    "DatasetProfile",
    "QLoRAMethod",
    "LoRAMethod",
    "DoRAMethod",
    "FreezeTuningMethod",
    "L2TMethod",
    "ChatTemplatePreprocessor",
]
