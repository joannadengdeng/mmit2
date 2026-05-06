"""TextVQA dataset spec."""
from __future__ import annotations

from mmit2.data.datasets.base import ColumnMapping, HFDatasetSpec


class TextVQASpec(HFDatasetSpec):
    dataset_name = "lmms-lab/textvqa"
    mapping = ColumnMapping(
        id_col="question_id",
        image_col="image",
        question_col="question",
        answer_col="answers",
    )
