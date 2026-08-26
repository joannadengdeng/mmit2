"""ScienceQA image-only dataset spec."""
from __future__ import annotations

from vlmintune.data.datasets.base import (
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

    def _choices(self, row: dict) -> list[str]:
        raw = row.get("choices", [])
        if not isinstance(raw, list):
            return []
        return [str(choice).strip() for choice in raw]

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
