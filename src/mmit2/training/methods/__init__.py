"""Training methods for multimodal fine-tuning.

Classic methods:
  - QLoRA, LoRA          — parameter-efficient LoRA variants
  - DoRA                 — weight-decomposed LoRA (Liu et al. ICML 2024)
  - FreezeTuning         — train selected modules only

Paper methods:
  - L2T          — instruction-aware loss masking (Zhou et al. 2025)
"""

from mmit2.training.methods.dora import DoRAMethod
from mmit2.training.methods.freeze import FreezeTuningMethod
from mmit2.training.methods.l2t import L2TMethod
from mmit2.training.methods.lora import LoRAMethod, QLoRAMethod

__all__ = [
    "QLoRAMethod",
    "LoRAMethod",
    "DoRAMethod",
    "FreezeTuningMethod",
    "L2TMethod",
]
