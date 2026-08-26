"""VQAv2 dataset spec."""
from __future__ import annotations

from vlmintune.data.datasets.base import ColumnMapping, DatasetDataModel, HFDatasetSpec


class VQAv2Spec(HFDatasetSpec):
    dataset_name = "pingzhili/vqa_v2"
    data_model = DatasetDataModel(
        dataset_name=dataset_name,
        default_train_split="train",
        default_eval_split="validation",
        metric_family="vqa_accuracy",
        split_file_pattern="data/{split}-*",
    )
    mapping = ColumnMapping(
        id_col="question_id",
        image_col="image",
        question_col="question",
        answer_col="answers",
    )
