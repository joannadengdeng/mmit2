"""TextVQA dataset spec."""
from __future__ import annotations

from collections import Counter

from mmit2.data.datasets.base import ColumnMapping, HFDatasetSpec


class TextVQASpec(HFDatasetSpec):
    dataset_name = "lmms-lab/textvqa"
    mapping = ColumnMapping(
        id_col="question_id",
        image_col="image",
        question_col="question",
        answer_col="answers",
    )

    def parse_answer(self, row: dict) -> tuple[str, object]:
        raw = row.get(self.mapping.answer_col, "")
        if not isinstance(raw, list):
            return super().parse_answer(row)

        answers: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                text = item.get("answer", str(item))
            else:
                text = str(item)
            text = text.strip()
            if text:
                answers.append(text)

        if not answers:
            return "", raw

        counts = Counter(answers)
        winner_answer = max(
            counts,
            key=lambda answer: (counts[answer], -answers.index(answer)),
        )
        return winner_answer, raw
