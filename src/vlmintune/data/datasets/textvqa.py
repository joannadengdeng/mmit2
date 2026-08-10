"""TextVQA dataset spec."""
from __future__ import annotations

from vlmintune.data.datasets.base import ColumnMapping, DatasetDataModel, HFDatasetSpec


class TextVQASpec(HFDatasetSpec):
    dataset_name = "lmms-lab/textvqa"
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

    def build_l2t_instruction_texts(self, row: dict, rendered_question: str):
        del row
        return [rendered_question]
