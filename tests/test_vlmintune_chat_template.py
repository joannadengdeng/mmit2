import os
import sys
import types

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.types import CanonicalSample, Turn
from vlmintune.training.chat_template import ChatTemplatePreprocessor


class _FakeProcessor:
    tokenizer = None

    def __init__(self):
        self.tokenizer = self

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        return "PROMPT" if add_generation_prompt else "FULL"

    def __call__(
        self,
        text,
        images=None,
        return_tensors=None,
        truncation=None,
        max_length=None,
        add_special_tokens=True,
    ):
        del images, return_tensors, truncation, max_length, add_special_tokens
        if text == "Question?":
            return {"input_ids": torch.tensor([[12]])}
        if text == "PROMPT":
            return {"input_ids": torch.tensor([[11, 12]])}
        return {
            "input_ids": torch.tensor([[99, 12, 13, 14]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        vocab = {
            99: "<image>",
            12: "Question?",
            13: "assistant:",
            14: "Answer.",
        }
        return " ".join(vocab.get(token_id, str(token_id)) for token_id in token_ids)


class _FakeImageProcessor(_FakeProcessor):
    image_token_id = 99

    def __call__(
        self,
        text,
        images=None,
        return_tensors=None,
        truncation=None,
        max_length=None,
        add_special_tokens=True,
    ):
        del images, return_tensors, truncation, max_length, add_special_tokens
        if text == "What is shown?":
            return {"input_ids": torch.tensor([[12, 13, 14]])}
        if text == "PROMPT":
            return {"input_ids": torch.tensor([[99, 99, 99, 12, 13, 14]])}
        return {
            "input_ids": torch.tensor([[99, 99, 99, 12, 13, 14, 15]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]]),
        }

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        vocab = {
            99: "<|image_pad|>",
            12: "What",
            13: "is",
            14: "shown?",
            15: "Answer.",
        }
        return " ".join(vocab.get(token_id, str(token_id)) for token_id in token_ids)


def test_chat_template_tokenize_includes_rendered_prompt_preview():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-1",
        image_path="",
        turns=[
            Turn(role="user", content="Question?"),
            Turn(role="assistant", content="Answer."),
        ],
    )

    result = ChatTemplatePreprocessor(enable_instruction_supervision=True).tokenize(
        sample,
        _FakeProcessor(),
        model_config,
    )

    assert result["input_ids"].tolist() == [99, 12, 13, 14]
    assert result["prompt_preview"]["sample_id"] == "sample-1"
    assert result["prompt_preview"]["full_text"] == "FULL"
    assert result["prompt_preview"]["prompt_text"] == "PROMPT"
    assert result["prompt_preview"]["has_image"] is False
    assert "attention_mask" not in result
    assert result["prompt_preview"]["instruction_texts"] == ["Question?"]
    assert result["prompt_preview"]["instruction_supervision_spans"] == [
        {"start": 1, "end": 2, "token_count": 1, "text": "Question?"}
    ]
    assert result["instruction_supervision_mask"].tolist() == [False, True, False, False]


def test_chat_template_tokenize_emits_mores_intervention_mask_for_visual_tokens():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-image",
        image_path="",
        turns=[
            Turn(role="user", content="What is shown?"),
            Turn(role="assistant", content="Answer."),
        ],
        metadata={"_pil_image": Image.new("RGB", (4, 4), color="white")},
    )

    result = ChatTemplatePreprocessor(enable_mores_intervention=True).tokenize(
        sample,
        _FakeImageProcessor(),
        model_config,
    )

    assert result["intervention_mask"].tolist() == [True, True, True, False, False, False, False]
    assert result["prompt_preview"]["intervention_mask"] == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_chat_template_collate_pads_intervention_mask():
    preprocessor = ChatTemplatePreprocessor(
        enable_instruction_supervision=True,
        enable_mores_intervention=True,
    )
    batch = preprocessor.collate(
        [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([1, 2, 3]),
                "instruction_supervision_mask": torch.tensor([False, True, False]),
                "intervention_mask": torch.tensor([False, True, True]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([4, 5]),
                "instruction_supervision_mask": torch.tensor([False, True]),
                "intervention_mask": torch.tensor([True, False]),
            },
        ]
    )

    assert batch["attention_mask"].tolist() == [
        [1, 1, 1],
        [1, 1, 0],
    ]
    assert batch["intervention_mask"].tolist() == [
        [False, True, True],
        [True, False, False],
    ]


def test_chat_template_default_preprocessor_skips_l2t_only_fields():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-plain",
        image_path="",
        turns=[
            Turn(role="user", content="Question?"),
            Turn(role="assistant", content="Answer."),
        ],
    )

    result = ChatTemplatePreprocessor().tokenize(
        sample,
        _FakeProcessor(),
        model_config,
    )

    assert "instruction_supervision_mask" not in result
    assert "intervention_mask" not in result
    assert "instruction_texts" not in result["prompt_preview"]
    assert "instruction_supervision_spans" not in result["prompt_preview"]
    assert "intervention_mask" not in result["prompt_preview"]
