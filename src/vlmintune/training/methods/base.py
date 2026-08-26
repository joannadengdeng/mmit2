"""TrainingMethod ABC for the built-in fine-tuning methods.

Each method defines model preparation, loss, and method-specific checkpoint state.

The initial release exposes fixed recipes. Method-specific configuration is
deliberately not part of this interface.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoProcessor, BitsAndBytesConfig

from vlmintune.models.base import ModelSpec
from vlmintune.models.registry import get_model_spec

try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLM


QWEN25VL_IMAGE_TOKEN_BUDGET = 1280
QWEN25VL_IMAGE_MAX_PIXELS = QWEN25VL_IMAGE_TOKEN_BUDGET * 28 * 28


def configure_processor_image_budget(processor, model_id: str):
    """Bound Qwen visual tokens so high-resolution images fit training prompts.

    Qwen's upstream processor permits up to 12,845,056 pixels (16,384 merged
    visual tokens).  That silently makes ordinary phone photos incompatible
    with vlmintune's 1,536/2,048-token training recipes: text truncation then
    raises an image-token mismatch and the affected sample is skipped.  The
    1,280-token image budget is Qwen's documented practical setting and leaves
    room for the chat template, question, and answer.
    """

    if not str(model_id).lower().startswith("qwen/qwen2.5-vl-"):
        return processor

    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if size is None or not hasattr(size, "longest_edge"):
        raise ValueError(
            "Qwen2.5-VL processor does not expose image_processor.size.longest_edge."
        )
    current = getattr(size, "longest_edge")
    if current is None or int(current) > QWEN25VL_IMAGE_MAX_PIXELS:
        size.longest_edge = QWEN25VL_IMAGE_MAX_PIXELS
    return processor


def load_processor(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return configure_processor_image_budget(processor, model_id)


def load_vlm(
    model_id: str,
    *,
    quantize_4bit: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if quantize_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        # Transformers 5 uses ``dtype`` and warns when the legacy
        # ``torch_dtype`` keyword is forwarded to from_pretrained().
        load_kwargs["dtype"] = torch_dtype

    return AutoVLM.from_pretrained(model_id, **load_kwargs)


class TrainingMethod(ABC):
    """Base class for all fine-tuning methods.

    A TrainingMethod encapsulates the complete recipe for fine-tuning a VLM:
    what to freeze or adapt, how to compute loss, and which state to persist.
    """

    name: str = ""              # registry key: "qlora", "lora", "l2t", ...
    display_name: str = ""      # Human-readable label: "QLoRA", "L2T (Zhou et al. 2025)", ...
    supported_model_names: Optional[Tuple[str, ...]] = None

    def requires_quantization(self) -> bool:
        """Whether the base model should be loaded in 4-bit quantization.

        QLoRA-based methods return True. All other methods load in bf16/fp16.
        """
        return False

    # ------------------------------------------------------------------
    # Model preparation
    # ------------------------------------------------------------------

    def prepare_model(
        self,
        model: nn.Module,
        processor: Any,
        model_spec: ModelSpec,
    ) -> Tuple[nn.Module, str]:
        """Prepare the model for training.

        Parameters
        ----------
        model : nn.Module
            The base VLM loaded from HuggingFace.
        processor :
            The tokenizer / processor.
        Returns
        -------
        (prepared_model, info_str)
        """
        return self.prepare_model_impl(model, processor, model_spec=model_spec)

    @abstractmethod
    def prepare_model_impl(
        self,
        model: nn.Module,
        processor: Any,
        model_spec: ModelSpec,
    ) -> Tuple[nn.Module, str]:
        """Subclass implementation of model preparation.

        This is where PEFT injection, hook registration, freezing, etc. happen.
        """

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def build_method_mask(**_) -> Optional[torch.Tensor]:
        """Return the method-owned token mask, or ``None`` when unused."""
        return None

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

    def build_forward_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Select the subset of batch keys passed into model(...).

        Most methods only need the standard tensors. Methods with extra runtime
        metadata can override this to stash metadata and/or drop unsupported keys.
        """
        excluded_keys = {"method_mask"}
        return {key: value for key, value in batch.items() if key not in excluded_keys}

    def prepare_inference_inputs(
        self,
        model: nn.Module,
        processor: Any,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Optionally augment inference-time model inputs.

        Override for methods that need runtime-only tensors during generation
        (e.g. MoReS intervention masks). Default: return inputs unchanged.
        """
        del model, processor
        return inputs

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

    def save_checkpoint(
        self,
        model: nn.Module,
        path: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Save method-specific weights and shared checkpoint metadata."""
        os.makedirs(path, exist_ok=True)
        self._save_weights(model, path)
        checkpoint_metadata = {
            **metadata,
            **self._checkpoint_metadata(),
            "ft_method": self.name,
        }
        metadata_path = os.path.join(path, "vlmintune_meta.json")
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(checkpoint_metadata, file, indent=2, ensure_ascii=False)

    def load_for_inference(
        self,
        path: str,
        model_name: str,
    ) -> Tuple[nn.Module, Any, Dict[str, str]]:
        """Load a saved checkpoint for inference.

        Returns
        -------
        (model, processor, info_dict)
        """
        model_spec = get_model_spec(model_name)
        processor = load_processor(model_spec.hf_model_id)
        model = load_vlm(
            model_spec.hf_model_id,
            quantize_4bit=self.requires_quantization(),
            torch_dtype=torch.bfloat16,
        )
        model = self._restore_model(model, processor, model_spec, path)
        model.eval()
        checkpoint_name = os.path.basename(path)
        info = {
            "model_id": (
                f"{model_spec.hf_model_id} ({self.display_name}: {checkpoint_name})"
            )
        }
        return model, processor, info

    @abstractmethod
    def _save_weights(self, model: nn.Module, path: str) -> None:
        """Save the method-specific trainable state."""

    @abstractmethod
    def _restore_model(
        self,
        model: nn.Module,
        processor: Any,
        model_spec: ModelSpec,
        path: str,
    ) -> nn.Module:
        """Install and restore the method-specific inference state."""

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        """Return method-specific metadata stored with the checkpoint."""
        return {}
