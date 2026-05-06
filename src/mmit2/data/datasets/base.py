"""Base classes and shared helpers for HuggingFace VQA dataset specs."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image

from mmit2.data.types import CanonicalSample, Turn


@dataclass(frozen=True)
class ColumnMapping:
    """Map dataset columns to canonical VQA fields."""

    id_col: str = "id"
    image_col: str = "image"
    question_col: str = "question"
    answer_col: str = "answer"


def handle_image_value(image_val, load_images: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Convert a HuggingFace image field into ``(image_path, metadata)``."""

    metadata: Dict[str, Any] = {}
    image_path = ""

    if image_val is None:
        pass
    elif isinstance(image_val, str):
        image_path = image_val
    elif isinstance(image_val, dict) and ("bytes" in image_val or "path" in image_val):
        if load_images and image_val.get("bytes"):
            try:
                pil_img = Image.open(io.BytesIO(image_val["bytes"]))
                metadata["_pil_image"] = pil_img
                image_path = "<in_memory>"
            except Exception:
                image_path = image_val.get("path", "<deferred>")
        else:
            image_path = image_val.get("path", "<deferred>")
            if image_val.get("bytes"):
                metadata["_image_bytes"] = image_val["bytes"]
    else:
        if load_images:
            if isinstance(image_val, Image.Image):
                metadata["_pil_image"] = image_val
                image_path = "<in_memory>"
        else:
            metadata["_raw_image"] = image_val
            image_path = "<deferred>"

    return image_path, metadata


class HFDatasetSpec:
    """Dataset-specific mapping and row parsing for a HF VQA dataset."""

    dataset_name: str = ""
    mapping: ColumnMapping = ColumnMapping()
    prefer_streaming: bool = False

    def parse_question(self, row: dict) -> str:
        raw = row.get(self.mapping.question_col, "") if self.mapping.question_col else ""
        if isinstance(raw, list):
            return str(raw[0]).strip() if raw else ""
        return str(raw).strip()

    def parse_answer(self, row: dict) -> tuple[str, Any]:
        raw = row.get(self.mapping.answer_col, "") if self.mapping.answer_col else ""
        if isinstance(raw, list):
            if raw and isinstance(raw[0], dict):
                return raw[0].get("answer", str(raw[0])), raw
            return (str(raw[0]) if raw else ""), raw
        return str(raw).strip(), raw

    def build_metadata(self, answer_raw: Any) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if isinstance(answer_raw, list):
            if answer_raw and isinstance(answer_raw[0], dict):
                metadata["raw_answers"] = [item.get("answer", str(item)) for item in answer_raw]
            else:
                metadata["raw_answers"] = [str(item) for item in answer_raw]
        elif answer_raw:
            metadata["raw_answers"] = [str(answer_raw).strip()]
        return metadata

    def parse_row(self, row: dict, idx: int, load_images: bool = True) -> CanonicalSample:
        question = self.parse_question(row)
        answer, answer_raw = self.parse_answer(row)

        turns = []
        if question:
            turns.append(Turn(role="human", content=question))
        if answer:
            turns.append(Turn(role="assistant", content=answer))

        image_val = row.get(self.mapping.image_col)
        image_path, image_meta = handle_image_value(image_val, load_images)
        metadata = {**image_meta, **self.build_metadata(answer_raw)}

        sample_id = str(row.get(self.mapping.id_col, idx)) if self.mapping.id_col else str(idx)
        return CanonicalSample(
            id=sample_id,
            image_path=image_path,
            turns=turns,
            metadata=metadata,
        )


class ConfiguredVQASpec(HFDatasetSpec):
    """Runtime-built spec for auto-detected or user-overridden mappings."""

    def __init__(
        self,
        dataset_name: str,
        mapping: ColumnMapping,
        *,
        prefer_streaming: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.mapping = mapping
        self.prefer_streaming = prefer_streaming
