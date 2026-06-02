"""Core data types shared across all mmit components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CanonicalSample:
    """Unified training/inference sample used throughout vlmintune."""
    id: str
    image_path: str
    question: str
    train_answer: str = ""
    eval_answers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSample:
    """A single benchmark question plus eval-time answer strings for scoring."""
    id: str
    image_path: str
    question: str
    eval_answers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
