import os
import sys
import types

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.types import CanonicalSample
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


class _FakeMMTokenTypeProcessor(_FakeProcessor):
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        if result["input_ids"].shape[1] == 4:
            result["mm_token_type_ids"] = torch.tensor([[1, 0, 0, 0]])
        return result


class _FakeAnswerTruncatedProcessor(_FakeProcessor):
    def __call__(self, text, **kwargs):
        if text == "PROMPT":
            return {"input_ids": torch.tensor([[99, 12, 13, 14]])}
        return super().__call__(text, **kwargs)


class _FakeLargeImageProcessor(_FakeImageProcessor):
    def __call__(
        self,
        text,
        images=None,
        return_tensors=None,
        truncation=None,
        max_length=None,
        add_special_tokens=True,
    ):
        image = (images or [None])[0]
        if isinstance(image, Image.Image) and max(image.size) > 512:
            raise ValueError(
                "Mismatch in `image` token count between text and `input_ids`. "
                "Likely due to `truncation='max_length'`."
            )
        return super().__call__(
            text=text,
            images=images,
            return_tensors=return_tensors,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )


class _FakeProcessorWithoutTemplateEos(_FakeProcessor):
    eos_token = "</s>"
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        if add_generation_prompt:
            return "USER: Question? ASSISTANT:"
        return "USER: Question? ASSISTANT: Answer. "

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
        if text == "USER: Question? ASSISTANT:":
            return {"input_ids": torch.tensor([[11, 12, 13]])}
        if text == "USER: Question? ASSISTANT: Answer. </s>":
            return {
                "input_ids": torch.tensor([[11, 12, 13, 14, 2]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            }
        raise AssertionError(f"unexpected text: {text!r}")

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        vocab = {11: "USER:", 12: "Question?", 13: "ASSISTANT:", 14: "Answer.", 2: "</s>"}
        return " ".join(vocab.get(token_id, str(token_id)) for token_id in token_ids)


class _FakeProcessorWithTemplateEos(_FakeProcessorWithoutTemplateEos):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        if add_generation_prompt:
            return "USER: Question? ASSISTANT:"
        return "USER: Question? ASSISTANT: Answer.</s>\n"

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
        if text == "USER: Question? ASSISTANT:":
            return {"input_ids": torch.tensor([[11, 12, 13]])}
        if text == "USER: Question? ASSISTANT: Answer.</s>\n":
            return {
                "input_ids": torch.tensor([[11, 12, 13, 14, 2, 15]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
            }
        raise AssertionError(f"unexpected text: {text!r}")


def test_chat_template_tokenize_includes_rendered_prompt_preview():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-1",
        image_path="",
        question="Question?",
        train_answer="Answer.",
        l2t_instruction_texts=["Question?"],
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
    assert result["prompt_preview"]["removed_task_templates"] == []
    assert result["prompt_preview"]["instruction_supervision_spans"] == [
        {"start": 1, "end": 2, "token_count": 1, "text": "Question?"}
    ]
    assert result["instruction_supervision_mask"].tolist() == [False, True, False, False]


def test_chat_template_preserves_qwen_mm_token_type_ids():
    sample = CanonicalSample(
        id="sample-mm-rope",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor().tokenize(
        sample,
        _FakeMMTokenTypeProcessor(),
        types.SimpleNamespace(image_token_id=99),
    )

    assert result["mm_token_type_ids"].tolist() == [1, 0, 0, 0]


def test_chat_template_l2t_rejects_sample_without_dataset_instruction_contract():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-no-l2t-contract",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    with pytest.raises(ValueError, match="dataset-defined"):
        ChatTemplatePreprocessor(enable_instruction_supervision=True).tokenize(
            sample,
            _FakeProcessor(),
            model_config,
        )


def test_chat_template_l2t_rejects_answer_truncated_by_max_length():
    sample = CanonicalSample(
        id="sample-truncated-answer",
        image_path="",
        question="Question?",
        train_answer="Answer.",
        l2t_instruction_texts=["Question?"],
    )

    with pytest.raises(ValueError, match="requires answer tokens"):
        ChatTemplatePreprocessor(enable_instruction_supervision=True).tokenize(
            sample,
            _FakeAnswerTruncatedProcessor(),
            types.SimpleNamespace(image_token_id=99),
        )


def test_chat_template_appends_eos_when_full_template_omits_it():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-eos",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor(append_eos_to_training_answer=True).tokenize(
        sample,
        _FakeProcessorWithoutTemplateEos(),
        model_config,
    )

    assert result["prompt_preview"]["full_text"].endswith("</s>")
    assert result["labels"].tolist() == [-100, -100, -100, 14, 2]


def test_chat_template_does_not_duplicate_existing_template_eos():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-existing-eos",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor(append_eos_to_training_answer=True).tokenize(
        sample,
        _FakeProcessorWithTemplateEos(),
        model_config,
    )

    assert result["prompt_preview"]["full_text"].count("</s>") == 1


def test_chat_template_tokenize_emits_mores_intervention_mask_for_visual_tokens():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-image",
        image_path="",
        question="What is shown?",
        train_answer="Answer.",
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


def test_chat_template_reft_mask_uses_prompt_boundary_not_full_sequence():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-reft",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor(enable_reft_intervention=True).tokenize(
        sample,
        _FakeProcessor(),
        model_config,
    )

    assert result["reft_intervention_mask"].tolist() == [True, True, False, False]
    assert result["prompt_preview"]["reft_intervention_mask"] == [
        True,
        True,
        False,
        False,
    ]


def test_chat_template_propagates_image_token_truncation_error():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-large-image",
        image_path="",
        question="What is shown?",
        train_answer="Answer.",
        metadata={"_pil_image": Image.new("RGB", (1200, 800), color="white")},
    )

    with pytest.raises(ValueError, match="Mismatch in `image` token count"):
        ChatTemplatePreprocessor().tokenize(
            sample,
            _FakeLargeImageProcessor(),
            model_config,
            max_length=4096,
        )


def test_chat_template_collate_pads_intervention_mask():
    preprocessor = ChatTemplatePreprocessor(
        enable_instruction_supervision=True,
        enable_mores_intervention=True,
        enable_reft_intervention=True,
    )
    batch = preprocessor.collate(
        [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([1, 2, 3]),
                "instruction_supervision_mask": torch.tensor([False, True, False]),
                "intervention_mask": torch.tensor([False, True, True]),
                "reft_intervention_mask": torch.tensor([True, False, True]),
                "mm_token_type_ids": torch.tensor([1, 0, 0]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([4, 5]),
                "instruction_supervision_mask": torch.tensor([False, True]),
                "intervention_mask": torch.tensor([True, False]),
                "reft_intervention_mask": torch.tensor([True, True]),
                "mm_token_type_ids": torch.tensor([1, 0]),
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
    assert batch["reft_intervention_mask"].tolist() == [
        [True, False, True],
        [True, True, False],
    ]
    assert batch["mm_token_type_ids"].tolist() == [
        [1, 0, 0],
        [1, 0, 0],
    ]


def test_chat_template_default_preprocessor_skips_l2t_only_fields():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-plain",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor().tokenize(
        sample,
        _FakeProcessor(),
        model_config,
    )

    assert "instruction_supervision_mask" not in result
    assert "intervention_mask" not in result
    assert "reft_intervention_mask" not in result
    assert "instruction_texts" not in result["prompt_preview"]
    assert "removed_task_templates" not in result["prompt_preview"]
    assert "instruction_supervision_spans" not in result["prompt_preview"]
    assert "intervention_mask" not in result["prompt_preview"]
    assert "reft_intervention_mask" not in result["prompt_preview"]


def test_chat_template_tokenize_rejects_empty_question():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-empty",
        image_path="",
        question="   ",
        train_answer="Answer.",
    )

    with pytest.raises(ValueError, match="Question text is empty"):
        ChatTemplatePreprocessor().tokenize(
            sample,
            _FakeProcessor(),
            model_config,
        )
