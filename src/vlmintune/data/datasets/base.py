"""Base classes and shared helpers for HuggingFace VQA dataset specs."""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from PIL import Image

from vlmintune.data.types import CanonicalSample


@dataclass(frozen=True)
class ColumnMapping:
    """Map dataset columns to canonical VQA fields."""

    id_col: str = "id"
    image_col: str = "image"
    question_col: str = "question"
    answer_col: str = "answer"


@dataclass(frozen=True)
class DatasetDataModel:
    """Minimal per-dataset contract used by training/eval loaders."""

    dataset_name: str
    hf_dataset_name: str = ""
    config_name: str = ""
    default_train_split: str = "train"
    default_eval_split: str = "validation"
    metric_family: str = "vqa_accuracy"

    @property
    def resolved_hf_dataset_name(self) -> str:
        return self.hf_dataset_name or self.dataset_name


@dataclass(frozen=True)
class AnswerBundle:
    train_answer: str = ""
    eval_answers: List[str] = field(default_factory=list)


def parse_image_field(image_val, load_images: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Parse a HF row image field into ``(image_path, metadata)``."""

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


def load_sample_image(sample: CanonicalSample) -> Image.Image | None:
    """Load a runtime image from cached sample metadata or a sample image path."""

    metadata = sample.metadata or {}
    pil_image = metadata.get("_pil_image")
    if isinstance(pil_image, Image.Image):
        return pil_image.convert("RGB")

    image_bytes = metadata.get("_image_bytes")
    if image_bytes:
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            pass

    raw_image = metadata.get("_raw_image")
    if isinstance(raw_image, Image.Image):
        return raw_image.convert("RGB")

    if not sample.image_path or sample.image_path in {"<in_memory>", "<deferred>"}:
        return None

    if not os.path.isfile(sample.image_path):
        return None
    return Image.open(sample.image_path).convert("RGB")


def extract_answer_strings(raw: Any, *, answer_key: str = "answer") -> List[str]:
    answers: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                text = item.get(answer_key, str(item))
            else:
                text = str(item)
            text = text.strip()
            if text:
                answers.append(text)
        return answers
    if raw not in (None, ""):
        text = str(raw).strip()
        if text:
            answers.append(text)
    return answers


def majority_vote_answer(answers: List[str]) -> str:
    if not answers:
        return ""
    counts: Dict[str, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(
        counts,
        key=lambda answer: (counts[answer], -answers.index(answer)),
    )


def build_question_message(question: str, has_image: bool) -> Dict[str, Any]:
    text = question.strip()
    if not text:
        raise ValueError("Question text is empty.")
    if has_image:
        content = [{"type": "image"}, {"type": "text", "text": text}]
    else:
        content = [{"type": "text", "text": text}]
    return {"role": "user", "content": content}


def build_answer_message(answer: str) -> Dict[str, Any] | None:
    text = answer.strip()
    if not text:
        return None
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    }


def build_prompt_messages(question: str, has_image: bool) -> List[Dict[str, Any]]:
    return [build_question_message(question, has_image)]


def build_full_messages(
    question: str,
    train_answer: str,
    has_image: bool,
) -> List[Dict[str, Any]]:
    messages = build_prompt_messages(question, has_image)
    answer_message = build_answer_message(train_answer)
    if answer_message is not None:
        messages.append(answer_message)
    return messages


def build_processor_images(image: Any) -> List[Any] | None:
    return [image] if image is not None else None


def render_prompt_text(processor: Any, question: str, has_image: bool) -> str:
    messages = build_prompt_messages(question, has_image)
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def render_full_text(
    processor: Any,
    question: str,
    train_answer: str,
    has_image: bool,
    append_eos_if_missing: bool = False,
) -> str:
    messages = build_full_messages(question, train_answer, has_image)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if append_eos_if_missing:
        tokenizer = getattr(processor, "tokenizer", processor)
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token and not str(text).rstrip().endswith(str(eos_token)):
            text = f"{text}{eos_token}"
    return text


def build_prompt_inputs(
    processor: Any,
    question: str,
    image: Any,
    **processor_kwargs: Any,
) -> Tuple[str, Dict[str, Any]]:
    text = render_prompt_text(processor, question, image is not None)
    inputs = processor(
        text=text,
        images=build_processor_images(image),
        **processor_kwargs,
    )
    return text, inputs


def build_full_inputs(
    processor: Any,
    question: str,
    train_answer: str,
    image: Any,
    append_eos_if_missing: bool = False,
    **processor_kwargs: Any,
) -> Tuple[str, Dict[str, Any]]:
    text = render_full_text(
        processor,
        question,
        train_answer,
        image is not None,
        append_eos_if_missing=append_eos_if_missing,
    )
    inputs = processor(
        text=text,
        images=build_processor_images(image),
        **processor_kwargs,
    )
    return text, inputs


class HFDatasetSpec:
    """Dataset-specific mapping and row parsing for a HF VQA dataset."""

    dataset_name: str = ""
    mapping: ColumnMapping = ColumnMapping()
    prefer_streaming: bool = False
    data_model: DatasetDataModel | None = None

    def parse_question(self, row: dict) -> str:
        raw = row.get(self.mapping.question_col, "") if self.mapping.question_col else ""
        if isinstance(raw, list):
            return str(raw[0]).strip() if raw else ""
        return str(raw).strip()

    def parse_answers(self, row: dict) -> AnswerBundle:
        raw = row.get(self.mapping.answer_col, "") if self.mapping.answer_col else ""
        answers = extract_answer_strings(raw)
        if answers:
            return AnswerBundle(
                train_answer=majority_vote_answer(answers),
                eval_answers=answers,
            )
        return AnswerBundle()

    def build_metadata(self, row: dict) -> Dict[str, Any]:
        del row
        return {}

    def parse_id(self, row: dict, idx: int) -> str:
        return str(row.get(self.mapping.id_col, idx)) if self.mapping.id_col else str(idx)

    def parse_row(self, row: dict, idx: int, load_images: bool = True) -> CanonicalSample:
        question = self.parse_question(row)
        answers = self.parse_answers(row)

        image_val = row.get(self.mapping.image_col)
        image_path, image_meta = parse_image_field(image_val, load_images)
        metadata = {**image_meta, **self.build_metadata(row)}

        return CanonicalSample(
            id=self.parse_id(row, idx),
            image_path=image_path,
            question=question,
            train_answer=answers.train_answer,
            eval_answers=answers.eval_answers,
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
        self.data_model = DatasetDataModel(dataset_name=dataset_name)
