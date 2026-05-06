"""VizWiz-VQA dataset spec."""
from __future__ import annotations

from mmit2.data.datasets.base import ColumnMapping, HFDatasetSpec


class VizWizVQASpec(HFDatasetSpec):
    dataset_name = "lmms-lab/VizWiz-VQA"
    mapping = ColumnMapping(
        id_col="question_id",
        image_col="image",
        question_col="question",
        answer_col="answers",
    )
