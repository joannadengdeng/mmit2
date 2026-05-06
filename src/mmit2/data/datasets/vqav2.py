"""VQAv2 dataset specs."""
from __future__ import annotations

from mmit2.data.datasets.base import ColumnMapping, HFDatasetSpec


class VQAv2Spec(HFDatasetSpec):
    dataset_name = "lmms-lab/VQAv2"
    mapping = ColumnMapping(
        id_col="question_id",
        image_col="image",
        question_col="question",
        answer_col="multiple_choice_answer",
    )
    prefer_streaming = True
