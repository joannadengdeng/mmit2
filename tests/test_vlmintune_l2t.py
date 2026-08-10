import json
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.types import CanonicalSample
from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.l2t import (
    L2T_CHECKPOINT_NAME,
    L2TMethod,
    build_instruction_supervision_mask,
    find_token_span,
)
from vlmintune.training.trainer.helpers import build_label_supervision_debug


def test_l2t_unmasks_instruction_only():
    method = L2TMethod()
    input_ids = torch.tensor([[11, 12, 13, 14, 0]])
    labels = torch.tensor([[-100, -100, -100, 14, -100]])
    batch_meta = {
        "instruction_supervision_mask": torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.bool),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
    }

    updated = method.preprocess_labels(input_ids, labels, batch_meta=batch_meta)

    assert updated.tolist() == [[11, -100, 13, 14, -100]]


@pytest.mark.parametrize(
    ("batch_meta", "message"),
    [
        (None, "batch metadata"),
        ({}, "batch metadata"),
        ({"attention_mask": torch.ones(1, 3)}, "instruction_supervision_mask"),
        (
            {"instruction_supervision_mask": torch.zeros(1, 3, dtype=torch.bool)},
            "contains no active tokens",
        ),
    ],
)
def test_l2t_never_silently_degrades_to_answer_only(batch_meta, message):
    input_ids = torch.tensor([[11, 12, 13]])
    labels = torch.tensor([[-100, -100, 13]])

    with pytest.raises(ValueError, match=message):
        L2TMethod().preprocess_labels(input_ids, labels, batch_meta=batch_meta)


def test_l2t_rejects_instruction_only_sample_when_answer_was_truncated():
    input_ids = torch.tensor([[11, 12, 13]])
    labels = torch.full_like(input_ids, -100)
    batch_meta = {
        "instruction_supervision_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        "attention_mask": torch.ones_like(input_ids),
    }

    with pytest.raises(ValueError, match="supervised answer token"):
        L2TMethod().preprocess_labels(input_ids, labels, batch_meta=batch_meta)


def test_l2t_is_standalone():
    method = L2TMethod()

    assert not hasattr(method, "base")


class _PrefixSensitiveTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_tensors=None,
        truncation=None,
        max_length=None,
    ):
        del add_special_tokens, return_tensors, truncation, max_length
        if text == "what brand?":
            return {"input_ids": torch.tensor([[20, 21]])}
        if text == "\nwhat brand?":
            return {"input_ids": torch.tensor([[12, 13]])}
        if text == " what brand?":
            return {"input_ids": torch.tensor([[30, 31]])}
        return {"input_ids": torch.tensor([[]], dtype=torch.long)}


class _PrefixSensitiveProcessor:
    tokenizer = _PrefixSensitiveTokenizer()


def test_l2t_instruction_mask_matches_newline_tokenized_question_variant():
    sample = CanonicalSample(
        id="1",
        image_path="",
        question="what brand?",
        l2t_instruction_texts=["what brand?"],
    )
    input_ids = torch.tensor([99, 12, 13, 98, 77])

    mask = build_instruction_supervision_mask(
        processor=_PrefixSensitiveProcessor(),
        sample=sample,
        input_ids=input_ids,
        prompt_len=4,
        max_length=16,
    )

    assert mask.tolist() == [False, True, True, False, False]


class _BothBoundariesSensitiveTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_tensors=None,
        truncation=None,
        max_length=None,
    ):
        del add_special_tokens, return_tensors, truncation, max_length
        if text == " what brand?\n":
            return {"input_ids": torch.tensor([[12, 13]])}
        return {"input_ids": torch.tensor([[]], dtype=torch.long)}


class _BothBoundariesSensitiveProcessor:
    tokenizer = _BothBoundariesSensitiveTokenizer()


def test_l2t_instruction_mask_matches_leading_space_and_trailing_newline_variant():
    sample = CanonicalSample(
        id="both-boundaries",
        image_path="",
        question="what brand?",
        l2t_instruction_texts=["what brand?"],
    )

    mask = build_instruction_supervision_mask(
        processor=_BothBoundariesSensitiveProcessor(),
        sample=sample,
        input_ids=torch.tensor([99, 12, 13, 98, 77]),
        prompt_len=4,
        max_length=16,
    )

    assert mask.tolist() == [False, True, True, False, False]


@pytest.mark.parametrize(
    ("sequence", "candidates", "expected"),
    [
        (
            [99, 472, 19, 98, 393, 17, 39, 19, 97],
            [[39, 19], [472, 19]],
            (1, 3),
        ),
        (
            [99, 63503, 6008, 98, 2477, 32845, 6008, 97],
            [[32845, 6008], [63503, 6008]],
            (1, 3),
        ),
    ],
)
def test_l2t_find_token_span_selects_earliest_position_across_variants(
    sequence,
    candidates,
    expected,
):
    assert find_token_span(sequence, candidates) == expected


def test_l2t_instruction_mask_rejects_missing_dataset_contract():
    sample = CanonicalSample(id="missing", image_path="", question="what brand?")

    with pytest.raises(ValueError, match="dataset-defined"):
        build_instruction_supervision_mask(
            processor=_PrefixSensitiveProcessor(),
            sample=sample,
            input_ids=torch.tensor([99, 12, 13, 98, 77]),
            prompt_len=4,
            max_length=16,
        )


def test_l2t_instruction_mask_rejects_unaligned_required_fragment():
    sample = CanonicalSample(
        id="unaligned",
        image_path="",
        question="what brand?",
        l2t_instruction_texts=["what brand?", "missing fragment"],
    )

    with pytest.raises(ValueError, match="could not align"):
        build_instruction_supervision_mask(
            processor=_PrefixSensitiveProcessor(),
            sample=sample,
            input_ids=torch.tensor([99, 12, 13, 98, 77]),
            prompt_len=4,
            max_length=16,
        )


class _ScienceInstructionTokenizer:
    _mapping = {
        "Which object is magnetic?": [20, 21, 22],
        "wooden spoon": [30, 31],
        "iron nail": [40, 41],
        "plastic cup": [50, 51],
    }

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_tensors=None,
        truncation=None,
        max_length=None,
    ):
        del add_special_tokens, return_tensors, truncation, max_length
        ids = self._mapping.get(text.strip(), [])
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _ScienceInstructionProcessor:
    tokenizer = _ScienceInstructionTokenizer()


def test_l2t_scienceqa_mask_supervises_content_but_not_wrappers_or_answer_format():
    input_ids = torch.tensor(
        [
            1,  # system
            2,  # image
            3,  # USER
            10,  # Question:
            20,
            21,
            22,
            11,  # Options:
            12,  # 0.
            30,
            31,
            13,  # 1.
            40,
            41,
            14,  # 2.
            50,
            51,
            15,  # Answer with only the option index.
            4,  # ASSISTANT
            60,  # answer
        ]
    )
    sample = CanonicalSample(
        id="science",
        image_path="",
        question="science prompt",
        train_answer="1",
        l2t_instruction_texts=[
            "Which object is magnetic?",
            "wooden spoon",
            "iron nail",
            "plastic cup",
        ],
        l2t_removed_task_templates=[
            "Question:",
            "Options:",
            "Answer with only the option index.",
            "0.",
            "1.",
            "2.",
        ],
    )

    mask = build_instruction_supervision_mask(
        processor=_ScienceInstructionProcessor(),
        sample=sample,
        input_ids=input_ids,
        prompt_len=19,
        max_length=64,
    )

    assert mask.nonzero(as_tuple=False).flatten().tolist() == [
        4,
        5,
        6,
        9,
        10,
        12,
        13,
        15,
        16,
    ]

    labels = torch.full((1, input_ids.numel()), -100, dtype=torch.long)
    labels[0, -1] = input_ids[-1]
    updated = L2TMethod().preprocess_labels(
        input_ids.unsqueeze(0),
        labels,
        batch_meta={
            "instruction_supervision_mask": mask.unsqueeze(0),
            "attention_mask": torch.ones_like(labels),
        },
    )
    assert updated[0, -1].item() == 60
    assert updated[0, mask].tolist() == input_ids[mask].tolist()
    assert updated[0, [0, 1, 2, 3, 7, 8, 11, 14, 17, 18]].tolist() == [-100] * 10


class _FakeDecodeProcessor:
    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        vocab = {
            11: "what",
            12: "brand",
            13: "is",
            14: "this",
            15: "nokia",
        }
        return " ".join(vocab.get(token_id, str(token_id)) for token_id in token_ids)


def test_build_label_supervision_debug_shows_restored_text():
    processor = _FakeDecodeProcessor()
    input_ids = torch.tensor([[11, 12, 13, 14, 15]])
    labels_before = torch.tensor([[-100, -100, -100, -100, 15]])
    labels_after = torch.tensor([[-100, 12, 13, 14, 15]])
    instruction_mask = torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.bool)

    debug = build_label_supervision_debug(
        processor,
        input_ids,
        labels_before,
        labels_after,
        instruction_mask,
    )

    assert debug["supervised_tokens_before"] == 1
    assert debug["supervised_tokens_after"] == 4
    assert debug["restored_tokens_into_loss"] == 3
    assert debug["instruction_mask_tokens"] == 3
    assert debug["first_sample_supervised_spans_before"] == [
        {"start": 4, "end": 5, "token_count": 1, "text": "nokia"}
    ]
    assert debug["first_sample_supervised_spans_after"] == [
        {"start": 1, "end": 5, "token_count": 4, "text": "brand is this nokia"}
    ]
    assert debug["first_sample_restored_spans"] == [
        {"start": 1, "end": 4, "token_count": 3, "text": "brand is this"}
    ]


class _FakeProcessor:
    def save_pretrained(self, path):
        self.saved_path = str(path)


class _ToyQwenL2T(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Linear(hidden_size, hidden_size)
        self.model.visual = nn.Module()
        self.model.visual.encoder = nn.Linear(hidden_size, hidden_size)
        self.model.visual.merger = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        hidden_states = self.model.visual.merger(inputs)
        hidden_states = self.model.language_model(hidden_states)
        return self.lm_head(hidden_states)


class _ToyLlavaL2T(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Linear(hidden_size, hidden_size)
        self.model.vision_tower = nn.Linear(hidden_size, hidden_size)
        self.model.multi_modal_projector = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        hidden_states = self.model.multi_modal_projector(inputs)
        hidden_states = self.model.language_model(hidden_states)
        return self.lm_head(hidden_states)


def _prepare_l2t(model, model_name):
    method = L2TMethod()
    prepared, info = method.prepare_model(
        model,
        _FakeProcessor(),
        model_spec=get_model_spec(model_name),
    )
    return prepared, method, info


@pytest.mark.parametrize(
    ("model_name", "model_factory", "trainable_prefixes", "frozen_prefix"),
    [
        (
            "qwen25vl_3b_instruct",
            _ToyQwenL2T,
            ("model.language_model.", "model.visual.merger.", "lm_head."),
            "model.visual.encoder.",
        ),
        (
            "llava15_7b",
            _ToyLlavaL2T,
            ("model.language_model.", "model.multi_modal_projector.", "lm_head."),
            "model.vision_tower.",
        ),
    ],
)
def test_l2t_fixed_full_sft_trainable_scope(
    model_name,
    model_factory,
    trainable_prefixes,
    frozen_prefix,
):
    model, _, info = _prepare_l2t(model_factory(), model_name)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert trainable_names
    assert all(name.startswith(trainable_prefixes) for name in trainable_names)
    assert all(any(name.startswith(prefix) for name in trainable_names) for prefix in trainable_prefixes)
    assert not any(name.startswith(frozen_prefix) for name in trainable_names)
    assert "vision encoder frozen" in info

def test_l2t_checkpoint_round_trip_uses_bf16_unquantized_base(tmp_path, monkeypatch):
    torch.manual_seed(123)
    source, method, _ = _prepare_l2t(_ToyQwenL2T(), "qwen25vl_3b_instruct")
    with torch.no_grad():
        for index, parameter in enumerate(
            parameter for parameter in source.parameters() if parameter.requires_grad
        ):
            parameter.fill_(0.01 * (index + 1))

    inputs = torch.randn(2, 4)
    expected = source(inputs).detach()
    method.save_checkpoint(
        source,
        _FakeProcessor(),
        str(tmp_path),
        {"model_name": "qwen25vl_3b_instruct", "final_loss": 0.25},
    )

    state_dict = torch.load(
        tmp_path / L2T_CHECKPOINT_NAME,
        map_location="cpu",
        weights_only=True,
    )
    assert state_dict
    assert not any(name.startswith("model.visual.encoder.") for name in state_dict)
    metadata = json.loads((tmp_path / "vlmintune_meta.json").read_text())
    assert metadata["ft_method"] == "l2t"
    assert metadata["recipe"] == "l2t_full_sft_v1"
    assert "config" not in metadata

    import vlmintune.training.methods.l2t as l2t_module

    load_calls = []

    def load_base_model(model_id, **kwargs):
        load_calls.append({"model_id": model_id, **kwargs})
        torch.manual_seed(123)
        return _ToyQwenL2T()

    monkeypatch.setattr(l2t_module, "load_processor", lambda model_id: _FakeProcessor())
    monkeypatch.setattr(l2t_module, "load_vlm", load_base_model)

    loaded, _, info = L2TMethod().load_for_inference(
        str(tmp_path),
        "qwen25vl_3b_instruct",
        quantize_4bit=True,
    )

    actual = loaded(inputs).detach()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert load_calls[-1]["quantize_4bit"] is False
    assert load_calls[-1]["torch_dtype"] is torch.bfloat16
    assert "L2T full-SFT" in info["model_id"]
