"""Preprocessors: CanonicalSample + HF Processor → model-ready tensors."""

from vlmintune.training.preprocessors.base import Preprocessor
from vlmintune.training.preprocessors.chat_template import ChatTemplatePreprocessor

__all__ = ["Preprocessor", "ChatTemplatePreprocessor"]
