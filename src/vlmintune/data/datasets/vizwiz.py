"""VizWiz dataset spec."""
from __future__ import annotations

from vlmintune.data.datasets.base import ColumnMapping, DatasetDataModel, HFDatasetSpec


class VizWizSpec(HFDatasetSpec):
    dataset_name = "HuggingFaceM4/VizWiz"
    data_model = DatasetDataModel(
        dataset_name=dataset_name,
        default_train_split="train",
        default_eval_split="validation",
        metric_family="vqa_accuracy",
    )
    mapping = ColumnMapping(
        id_col="question_id",
        image_col="image",
        question_col="question",
        answer_col="answers",
    )
