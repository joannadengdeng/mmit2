"""GQA dataset spec."""
from __future__ import annotations

from vlmintune.data.datasets.base import ColumnMapping, DatasetDataModel, HFDatasetSpec


class GQASpec(HFDatasetSpec):
    dataset_name = "Mineru/GQA"
    data_model = DatasetDataModel(
        dataset_name=dataset_name,
        default_train_split="train_balanced",
        default_eval_split="val_balanced",
        metric_family="normalized_exact_match",
    )
    mapping = ColumnMapping(
        id_col="question_id",
        image_col="image",
        question_col="question",
        answer_col="answer",
    )

    def build_l2t_instruction_texts(self, row: dict, rendered_question: str):
        del row
        return [rendered_question]
