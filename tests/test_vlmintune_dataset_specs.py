import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.datasets.registry import get_dataset_spec


def test_textvqa_spec_majority_vote_and_default_splits():
    spec = get_dataset_spec("lmms-lab/textvqa")
    sample = spec.parse_row(
        {
            "question_id": 10,
            "image": None,
            "question": "What word is shown?",
            "answers": [{"answer": "stop"}, {"answer": "stop"}, {"answer": "shop"}],
        },
        idx=0,
        load_images=False,
    )

    assert spec.data_model.default_train_split == "train"
    assert spec.data_model.default_eval_split == "validation"
    assert sample.question == "What word is shown?"
    assert sample.train_answer == "stop"
    assert sample.eval_answers == ["stop", "stop", "shop"]


def test_vqav2_spec_uses_majority_vote_for_train_answer():
    spec = get_dataset_spec("pingzhili/vqa_v2")
    sample = spec.parse_row(
        {
            "question_id": 20,
            "image": None,
            "question": "What animal is shown?",
            "multiple_choice_answer": "dog",
            "answers": [{"answer": "cat"}, {"answer": "cat"}, {"answer": "dog"}],
            "answer_type": "other",
            "question_type": "what is this",
        },
        idx=0,
        load_images=False,
    )

    assert sample.question == "What animal is shown?"
    assert sample.train_answer == "cat"
    assert sample.eval_answers == ["cat", "cat", "dog"]
    assert sample.metadata == {}


def test_vizwiz_spec_uses_answer_contract_without_extra_metadata():
    spec = get_dataset_spec("ebrukilic/vizwiz_vqa_dataset")
    sample = spec.parse_row(
        {
            "question_id": 30,
            "image": None,
            "question": "What is visible?",
            "answers": [{"answer": "nothing"}, {"answer": "nothing"}, {"answer": "blur"}],
            "answer_type": "unanswerable",
            "answerable": False,
        },
        idx=0,
        load_images=False,
    )

    assert sample.question == "What is visible?"
    assert sample.train_answer == "nothing"
    assert sample.eval_answers == ["nothing", "nothing", "blur"]
    assert sample.metadata == {}


def test_vizwiz_spec_uses_replacement_dataset_directly():
    spec = get_dataset_spec("ebrukilic/vizwiz_vqa_dataset")

    assert spec is get_dataset_spec("ebrukilic/vizwiz_vqa_dataset")
    assert spec.data_model.resolved_hf_dataset_name == "ebrukilic/vizwiz_vqa_dataset"
    assert spec.data_model.default_train_split == "train"
    assert spec.data_model.default_eval_split == "validation"


def test_gqa_spec_uses_one_row_one_question_shape():
    spec = get_dataset_spec("Mineru/GQA")
    sample = spec.parse_row(
        {
            "question_id": "n1",
            "question": "What color is the car?",
            "answer": "red",
            "image": None,
            "fullAnswer": "The car is red.",
        },
        idx=0,
        load_images=False,
    )

    assert spec.data_model.default_train_split == "train_balanced"
    assert spec.data_model.default_eval_split == "val_balanced"
    assert sample.question == "What color is the car?"
    assert sample.train_answer == "red"
    assert sample.eval_answers == ["red"]
    assert "fullAnswer" not in sample.metadata


def test_scienceqa_image_spec_uses_choice_index_answer():
    spec = get_dataset_spec("scienceqa_image")
    sample = spec.parse_row(
        {
            "pid": "sci-1",
            "image": {"path": "science.png"},
            "question": "Which object is magnetic?",
            "choices": ["wooden spoon", "iron nail", "plastic cup"],
            "answer": 1,
            "hint": "Think about metals.",
        },
        idx=0,
        load_images=False,
    )

    assert spec.data_model.resolved_hf_dataset_name == "derek-thomas/ScienceQA"
    assert spec.data_model.default_train_split == "train"
    assert spec.data_model.default_eval_split == "validation"
    assert spec.data_model.metric_family == "normalized_exact_match"
    assert sample.id == "sci-1"
    assert sample.image_path == "science.png"
    assert sample.question == "\n".join(
        [
            "Question: Which object is magnetic?",
            "Options:",
            "0. wooden spoon",
            "1. iron nail",
            "2. plastic cup",
            "Answer with only the option index.",
        ]
    )
    assert sample.train_answer == "1"
    assert sample.eval_answers == ["1"]
    assert sample.metadata["answer_index"] == 1
    assert sample.metadata["answer_text"] == "iron nail"
    assert sample.metadata["choices"] == ["wooden spoon", "iron nail", "plastic cup"]
    assert sample.metadata["hint"] == "Think about metals."
