"""TrainingMethod ABC for the built-in fine-tuning methods.

Each method defines how to prepare a model, compute loss, and save/load checkpoints.

Built-in methods:
  - QLoRA, LoRA, DoRA          — parameter-efficient LoRA variants
  - FreezeTuning               — train selected modules only
  - L2T                        — instruction-aware loss masking (Zhou et al. 2025)

To add a custom method:
  1. Subclass ``TrainingMethod``
  2. Implement all abstract methods
  3. Register: ``register_training_method("my-method", MyMethod)``
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class TrainingMethod(ABC):
    """Base class for all fine-tuning methods.

    A TrainingMethod encapsulates the complete recipe for fine-tuning a VLM:
    how to load the model, what to freeze/adapt, how to compute loss,
    and how to save/load the result.
    """

    name: str = ""              # registry key: "qlora", "lora", "l2t", ...
    display_name: str = ""      # Human-readable label: "QLoRA", "L2T (Zhou et al. 2025)", ...
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @abstractmethod
    def default_config(self) -> Dict[str, Any]:
        """Return default hyperparameters for this method."""

    def requires_quantization(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Whether the base model should be loaded in 4-bit quantization.

        Only QLoRA returns True. All other methods load in bf16/fp16.
        """
        return False

    # ------------------------------------------------------------------
    # Model preparation
    # ------------------------------------------------------------------

    def prepare_model(
        self,
        model: nn.Module,
        processor: Any,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, str]:
        """Prepare the model for training.

        Parameters
        ----------
        model : nn.Module
            The base VLM loaded from HuggingFace.
        processor :
            The tokenizer / processor.
        config : dict
            Method-specific config (from YAML or CLI).

        Returns
        -------
        (prepared_model, info_str)
        """
        return self._prepare_model_impl(model, processor, config)

    @abstractmethod
    def _prepare_model_impl(
        self,
        model: nn.Module,
        processor: Any,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, str]:
        """Subclass implementation of model preparation.

        This is where PEFT injection, hook registration, freezing, etc. happen.
        """

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def preprocess_labels(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        batch_meta: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Optionally modify labels before loss computation.

        Override for methods that need custom masking (e.g., L2T unmasks
        instruction tokens). Default: return labels unchanged.
        """
        return labels

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        outputs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the training loss.

        Returns
        -------
        (loss, metrics_dict) — scalar loss and optional logging metrics.
        """

    @abstractmethod
    def get_trainable_params(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Return optimizer parameter groups.

        Each dict has "params" (list of Parameters) and optionally "lr".
        """

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @abstractmethod
    def save_checkpoint(
        self,
        model: nn.Module,
        processor: Any,
        path: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Save trained weights/adapter to disk."""

    @abstractmethod
    def load_for_inference(
        self,
        path: str,
        base_model_id: str,
        **kwargs,
    ) -> Tuple[nn.Module, Any, Dict[str, str]]:
        """Load a saved checkpoint for inference.

        Returns
        -------
        (model, processor, info_dict)
        """
