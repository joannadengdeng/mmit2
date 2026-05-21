"""Shared causal LM cross-entropy helper for training methods."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


class CrossEntropyLoss:
    """Standard causal language modeling cross-entropy loss."""

    def compute(
        self,
        model: Any,
        batch: Dict[str, Any],
        outputs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss, {}

        logits = outputs.logits
        labels = batch["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
        return loss, {}
