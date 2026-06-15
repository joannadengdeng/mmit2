"""VizWiz dataset spec."""
from __future__ import annotations

from vlmintune.data.datasets.base import ColumnMapping, DatasetDataModel, HFDatasetSpec


class VizWizSpec(HFDatasetSpec):
    dataset_name = "ebrukilic/vizwiz_vqa_dataset"
    data_model = DatasetDataModel(
        dataset_name=dataset_name,
        default_train_split="train",
        default_eval_split="validation",
        metric_family="vqa_accuracy",
    )
    mapping = ColumnMapping(
        id_col="id",
        image_col="image",
        question_col="question",
        answer_col="answers",
    )

    def parse_id(self, row: dict, idx: int) -> str:
        for key in ("id", "filename"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
        return str(idx)
