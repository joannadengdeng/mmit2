"""VisNec-style dataset filtering utilities."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from vlmintune.data.types import CanonicalSample


@dataclass(frozen=True)
class VisNecSelection:
    """A fixed set of sample ids selected by visual-necessity score."""

    score_file: str
    top_ratio: float
    selected_ids: set[str]


def _score_from_record(record: dict[str, Any]) -> tuple[str, float]:
    sample_id = record.get("sample_id", record.get("id"))
    if sample_id is None:
        raise ValueError("VisNec score record must contain 'sample_id' or 'id'.")
    score = record.get("visnec_score", record.get("score"))
    if score is None:
        raise ValueError("VisNec score record must contain 'visnec_score' or 'score'.")
    return str(sample_id), float(score)


def _read_score_records(path: str) -> list[tuple[str, float]]:
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            payload = json.load(f)
            if isinstance(payload, dict):
                return [(str(sample_id), float(score)) for sample_id, score in payload.items()]
            if not isinstance(payload, list):
                raise ValueError("VisNec JSON score file must be a list or object.")
            return [_score_from_record(record) for record in payload]

        records = []
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(_score_from_record(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"VisNec score file must be JSONL; invalid JSON at line {line_no}."
                ) from exc
        return records


def load_visnec_selection(score_file: str, top_ratio: float) -> VisNecSelection:
    """Load score records and keep the highest-scoring ``top_ratio`` sample ids."""
    if not score_file:
        raise ValueError("visnec_score_file is required when VisNec filtering is enabled.")
    top_ratio = float(top_ratio)
    if top_ratio <= 0 or top_ratio > 1:
        raise ValueError("visnec_top_ratio must be in the interval (0, 1].")

    records = _read_score_records(score_file)
    if not records:
        raise ValueError(f"VisNec score file has no records: {score_file}")

    records.sort(key=lambda item: item[1], reverse=True)
    keep_count = max(1, int(math.ceil(len(records) * top_ratio)))
    return VisNecSelection(
        score_file=score_file,
        top_ratio=top_ratio,
        selected_ids={sample_id for sample_id, _ in records[:keep_count]},
    )


class VisNecFilteredAdapter:
    """Filter an existing adapter to samples selected by VisNec score."""

    def __init__(
        self,
        base_adapter: Any,
        selection: VisNecSelection,
        max_samples: int | None = None,
    ) -> None:
        self.base_adapter = base_adapter
        self.selection = selection
        self.streaming = bool(getattr(base_adapter, "streaming", False))
        self.dataset_name = getattr(base_adapter, "dataset_name", "")
        self.split = getattr(base_adapter, "split", "")
        self.max_samples = max_samples
        self._indices: list[int] | None = None
        if not self.streaming:
            self._indices = self._build_indices()

    def _build_indices(self) -> list[int]:
        indices = []
        for idx in range(len(self.base_adapter)):
            sample_id = str(self.base_adapter.get_sample_id(idx))
            if sample_id in self.selection.selected_ids:
                indices.append(idx)
                if self.max_samples is not None and len(indices) >= self.max_samples:
                    break
        return indices

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        if self.max_samples is not None:
            return min(self.max_samples, len(self.selection.selected_ids))
        return len(self.selection.selected_ids)

    def __iter__(self) -> Iterator[CanonicalSample]:
        if self._indices is not None:
            for idx in self._indices:
                yield self.base_adapter[idx]
            return
        yielded = 0
        for sample in self.base_adapter:
            if str(sample.id) in self.selection.selected_ids:
                yield sample
                yielded += 1
                if self.max_samples is not None and yielded >= self.max_samples:
                    break

    def __getitem__(self, idx: int) -> CanonicalSample:
        if self._indices is None:
            raise TypeError("__getitem__ is not supported for streaming VisNec filtering.")
        return self.base_adapter[self._indices[idx]]

    @property
    def column_names(self) -> Sequence[str]:
        return getattr(self.base_adapter, "column_names", [])

    @property
    def mapping(self) -> Any:
        return self.base_adapter.mapping

    @property
    def profile(self) -> Any:
        return self.base_adapter.profile


__all__ = [
    "VisNecFilteredAdapter",
    "VisNecSelection",
    "load_visnec_selection",
]
