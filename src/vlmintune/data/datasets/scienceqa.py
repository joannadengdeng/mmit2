"""ScienceQA image-only dataset spec."""
from __future__ import annotations

from typing import Any, Dict, List

from vlmintune.data.datasets.base import (
    AnswerBundle,
    ColumnMapping,
    DatasetDataModel,
    HFDatasetSpec,
)


class ScienceQAImageSpec(HFDatasetSpec):
    dataset_name = "scienceqa_image"
    data_model = DatasetDataModel(
        dataset_name=dataset_name,
        hf_dataset_name="derek-thomas/ScienceQA",
        default_train_split="train",
        default_eval_split="validation",
        metric_family="normalized_exact_match",
    )
    mapping = ColumnMapping(
        id_col="pid",
        image_col="image",
        question_col="question",
        answer_col="answer",
    )

    def parse_id(self, row: dict, idx: int) -> str:
        for key in ("pid", "id", "question_id"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
        return str(idx)

    def _choices(self, row: dict) -> List[str]:
        raw = row.get("choices", [])
        if not isinstance(raw, list):
            return []
        return [str(choice).strip() for choice in raw]

    def _answer_index(self, row: dict) -> int:
        raw = row.get(self.mapping.answer_col)
        if raw is None:
            raise ValueError("ScienceQA row is missing answer index.")
        return int(raw)

    def parse_question(self, row: dict) -> str:
        question = str(row.get(self.mapping.question_col, "")).strip()
        if not question:
            raise ValueError("ScienceQA question text is empty.")

        lines = [f"Question: {question}"]
        choices = self._choices(row)
        if choices:
            lines.append("Options:")
            lines.extend(f"{idx}. {choice}" for idx, choice in enumerate(choices))
        lines.append("Answer with only the option index.")
        return "\n".join(lines)

    def parse_answers(self, row: dict) -> AnswerBundle:
        answer = str(self._answer_index(row))
        return AnswerBundle(train_answer=answer, eval_answers=[answer])

    def build_metadata(self, row: dict) -> Dict[str, Any]:
        choices = self._choices(row)
        answer_index = self._answer_index(row)
        answer_text = choices[answer_index] if 0 <= answer_index < len(choices) else ""
        metadata: Dict[str, Any] = {
            "choices": choices,
            "answer_index": answer_index,
            "answer_text": answer_text,
        }
        for key in (
            "hint",
            "lecture",
            "solution",
            "task",
            "grade",
            "subject",
            "topic",
            "category",
            "skill",
        ):
            if key in row:
                metadata[key] = row.get(key)
        return metadata
