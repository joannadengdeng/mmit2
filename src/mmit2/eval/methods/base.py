"""Method ABC — the central interface for multimodal inference in mmit2."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from mmit2.data.types import CanonicalSample, EvalSample, Turn


class Method(ABC):
    """Full multimodal inference pipeline for one model variant.

    Lifecycle
    ---------
    1. ``Method.from_pretrained(path)``  — easiest entry point
    2. ``method.prepare_input(sample)``  — per-sample preprocessing
    3. ``method.generate(prepared)``     — autoregressive decoding
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        **kwargs,
    ) -> "Method":
        """Load a HuggingFace model locally for inference."""
        return LocalMethod.from_checkpoint(base_model_id=model_path, **kwargs)

    @abstractmethod
    def prepare_input(
        self,
        sample: CanonicalSample,
        image_root: str = "",
    ) -> Dict[str, Any]:
        """Return a dict of tensors ready for ``generate``."""

    def prepare_eval_input(
        self,
        sample: EvalSample,
        image_root: str = "",
    ) -> Dict[str, Any]:
        """Convenience wrapper: build a CanonicalSample from an EvalSample."""
        cs = CanonicalSample(
            id=sample.id,
            image_path=sample.image_path,
            turns=[Turn(role="human", content=sample.question)],
            metadata=sample.metadata,
        )
        return self.prepare_input(cs, image_root=image_root)

    @abstractmethod
    def generate(
        self,
        prepared: Dict[str, Any],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response string from preprocessed inputs."""


from mmit2.eval.methods.local_method import LocalMethod
