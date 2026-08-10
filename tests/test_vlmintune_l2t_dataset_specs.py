import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.datasets.base import ColumnMapping
from vlmintune.data.datasets.registry import build_configured_spec, get_dataset_spec


@pytest.mark.parametrize(
    ("dataset_name", "row", "expected_question"),
    [
        (
            "lmms-lab/textvqa",
            {
                "question_id": 1,
                "image": None,
                "question": "What word is shown?",
                "answers": [{"answer": "stop"}],
            },
            "What word is shown?",
        ),
        (
            "pingzhili/vqa_v2",
            {
                "question_id": 2,
                "image": None,
                "question": "What animal is shown?",
                "answers": [{"answer": "cat"}],
            },
            "What animal is shown?",
        ),
        (
            "ebrukilic/vizwiz_vqa_dataset",
            {
                "question_id": 3,
                "image": None,
                "question": "What is visible?",
                "answers": [{"answer": "nothing"}],
            },
            "What is visible?",
        ),
        (
            "Mineru/GQA",
            {
                "question_id": "gqa-1",
                "image": None,
                "question": "What color is the car?",
                "answer": "red",
            },
            "What color is the car?",
        ),
    ],
)
def test_builtin_vqa_specs_define_explicit_l2t_instruction_contract(
    dataset_name,
    row,
    expected_question,
):
    sample = get_dataset_spec(dataset_name).parse_row(
        row,
        idx=0,
        load_images=False,
    )

    assert sample.l2t_instruction_texts == [expected_question]
    assert sample.l2t_removed_task_templates == []


def test_scienceqa_l2t_contract_excludes_wrappers_and_option_numbers():
    sample = get_dataset_spec("scienceqa_image").parse_row(
        {
            "pid": "science-1",
            "image": None,
            "question": "Which object is magnetic?",
            "choices": ["wooden spoon", "iron nail", "plastic cup"],
            "answer": 1,
        },
        idx=0,
        load_images=False,
    )

    assert sample.l2t_instruction_texts == [
        "Which object is magnetic?",
        "wooden spoon",
        "iron nail",
        "plastic cup",
    ]
    assert sample.l2t_removed_task_templates == [
        "Question:",
        "Options:",
        "Answer with only the option index.",
        "0.",
        "1.",
        "2.",
    ]


def test_custom_dataset_has_no_implicit_l2t_instruction_contract():
    spec = build_configured_spec(
        "owner/custom-vqa",
        ColumnMapping(
            id_col="id",
            image_col="image",
            question_col="question",
            answer_col="answer",
        ),
    )
    sample = spec.parse_row(
        {
            "id": "custom-1",
            "image": None,
            "question": "What is shown?",
            "answer": "a cat",
        },
        idx=0,
        load_images=False,
    )

    assert sample.l2t_instruction_texts == []
    assert sample.l2t_removed_task_templates == []
