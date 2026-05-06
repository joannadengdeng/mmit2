"""Preprocessors: CanonicalSample + HF Processor → model-ready tensors."""

from mmit2.training.preprocessors.base import Preprocessor
from mmit2.training.preprocessors.chat_template import ChatTemplatePreprocessor

__all__ = ["Preprocessor", "ChatTemplatePreprocessor"]
