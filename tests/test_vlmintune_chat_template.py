import os
import sys
import types

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.types import CanonicalSample
from vlmintune.training.chat_template import ChatTemplatePreprocessor
from vlmintune.training.methods.base import TrainingMethod
from vlmintune.training.methods.l2t import L2TMethod
from vlmintune.training.methods.mores import MoReSMethod
from vlmintune.training.methods.reft import ReFTMethod


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


class _FakePrefixL2TProcessor(_FakeProcessor):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        if add_generation_prompt:
            return "PREFIXQuestion?SUFFIX"
        return "PREFIXQuestion?SUFFIXAnswer."

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
        input_ids = {
            "PREFIX": [11],
            "PREFIXQuestion?": [11, 12],
            "PREFIXQuestion?SUFFIX": [11, 12, 20],
            "PREFIXQuestion?SUFFIXAnswer.": [11, 12, 20, 13, 14],
        }[text]
        result = {"input_ids": torch.tensor([input_ids])}
        if text == "PREFIXQuestion?SUFFIXAnswer.":
            result["attention_mask"] = torch.ones(1, len(input_ids), dtype=torch.long)
        return result


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

class _FakeMMTokenTypeProcessor(_FakeProcessor):
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        if result["input_ids"].shape[1] == 4:
            result["mm_token_type_ids"] = torch.tensor([[1, 0, 0, 0]])
        return result


class _FakeLlavaPixelProcessor(_FakeProcessor):
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        if result["input_ids"].shape[1] == 4:
            result["pixel_values"] = torch.arange(12).reshape(1, 3, 2, 2)
        return result


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


def test_chat_template_tokenize_emits_l2t_method_mask():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-1",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor(L2TMethod).tokenize(
        sample,
        _FakePrefixL2TProcessor(),
        model_config,
    )

    assert result["input_ids"].tolist() == [11, 12, 20, 13, 14]
    assert result["labels"].tolist() == [-100, -100, -100, 13, 14]
    assert "attention_mask" not in result
    assert result["method_mask"].tolist() == [
        False,
        True,
        False,
        False,
        False,
    ]


def test_chat_template_preserves_qwen_mm_token_type_ids():
    sample = CanonicalSample(
        id="sample-mm-rope",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor(TrainingMethod).tokenize(
        sample,
        _FakeMMTokenTypeProcessor(),
        types.SimpleNamespace(image_token_id=99),
    )

    assert result["mm_token_type_ids"].tolist() == [1, 0, 0, 0]


def test_chat_template_preserves_processor_pixel_batch_axis():
    sample = CanonicalSample(
        id="sample-pixels",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor(TrainingMethod).tokenize(
        sample,
        _FakeLlavaPixelProcessor(),
        types.SimpleNamespace(image_token_id=99),
    )

    assert result["pixel_values"].shape == (1, 3, 2, 2)


def test_chat_template_appends_eos_when_full_template_omits_it():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-eos",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor(
        TrainingMethod,
        append_eos_to_training_answer=True,
    ).tokenize(
        sample,
        _FakeProcessorWithoutTemplateEos(),
        model_config,
    )

    assert result["input_ids"].tolist() == [11, 12, 13, 14, 2]
    assert result["labels"].tolist() == [-100, -100, -100, 14, 2]


def test_chat_template_does_not_duplicate_existing_template_eos():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-existing-eos",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor(
        TrainingMethod,
        append_eos_to_training_answer=True,
    ).tokenize(
        sample,
        _FakeProcessorWithTemplateEos(),
        model_config,
    )

    assert result["input_ids"].tolist() == [11, 12, 13, 14, 2, 15]
    assert result["labels"].tolist() == [-100, -100, -100, 14, 2, 15]


def test_chat_template_tokenize_emits_mores_method_mask_for_visual_tokens():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-image",
        image_path="",
        question="What is shown?",
        train_answer="Answer.",
        metadata={"_pil_image": Image.new("RGB", (4, 4), color="white")},
    )

    result = ChatTemplatePreprocessor(MoReSMethod).tokenize(
        sample,
        _FakeImageProcessor(),
        model_config,
    )

    assert result["method_mask"].tolist() == [True, True, True, False, False, False, False]


def test_chat_template_reft_method_mask_uses_prompt_boundary_not_full_sequence():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-reft",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor(ReFTMethod).tokenize(
        sample,
        _FakeProcessor(),
        model_config,
    )

    assert result["method_mask"].tolist() == [True, True, False, False]


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
        ChatTemplatePreprocessor(TrainingMethod).tokenize(
            sample,
            _FakeLargeImageProcessor(),
            model_config,
            max_length=4096,
        )


def test_chat_template_collate_pads_method_mask():
    preprocessor = ChatTemplatePreprocessor(TrainingMethod)
    batch = preprocessor.collate(
        [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([1, 2, 3]),
                "method_mask": torch.tensor([False, True, True]),
                "mm_token_type_ids": torch.tensor([1, 0, 0]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([4, 5]),
                "method_mask": torch.tensor([True, False]),
                "mm_token_type_ids": torch.tensor([1, 0]),
            },
        ]
    )

    assert batch["attention_mask"].tolist() == [
        [1, 1, 1],
        [1, 1, 0],
    ]
    assert batch["method_mask"].tolist() == [
        [False, True, True],
        [True, False, False],
    ]
    assert batch["mm_token_type_ids"].tolist() == [
        [1, 0, 0],
        [1, 0, 0],
    ]


def test_chat_template_collate_concatenates_qwen_visual_inputs():
    preprocessor = ChatTemplatePreprocessor(TrainingMethod)
    first_pixels = torch.arange(8).reshape(2, 4)
    second_pixels = torch.arange(8, 16).reshape(2, 4)

    batch = preprocessor.collate(
        [
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([1, 2]),
                "pixel_values": first_pixels,
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            },
            {
                "input_ids": torch.tensor([3, 4]),
                "labels": torch.tensor([3, 4]),
                "pixel_values": second_pixels,
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            },
        ]
    )

    assert torch.equal(batch["pixel_values"], torch.cat([first_pixels, second_pixels]))
    assert batch["pixel_values"].shape == (4, 4)
    assert batch["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 2]]


def test_chat_template_collate_concatenates_llava_visual_inputs():
    preprocessor = ChatTemplatePreprocessor(TrainingMethod)
    first_pixels = torch.zeros(1, 3, 2, 2)
    second_pixels = torch.ones(1, 3, 2, 2)

    batch = preprocessor.collate(
        [
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([1, 2]),
                "pixel_values": first_pixels,
            },
            {
                "input_ids": torch.tensor([3, 4]),
                "labels": torch.tensor([3, 4]),
                "pixel_values": second_pixels,
            },
        ]
    )

    assert torch.equal(batch["pixel_values"], torch.cat([first_pixels, second_pixels]))
    assert batch["pixel_values"].shape == (2, 3, 2, 2)


def test_chat_template_default_preprocessor_skips_l2t_only_fields():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-plain",
        image_path="",
        question="Question?",
        train_answer="Answer.",
    )

    result = ChatTemplatePreprocessor(TrainingMethod).tokenize(
        sample,
        _FakeProcessor(),
        model_config,
    )

    assert "method_mask" not in result


def test_chat_template_tokenize_rejects_empty_question():
    model_config = types.SimpleNamespace(image_token_id=99)
    sample = CanonicalSample(
        id="sample-empty",
        image_path="",
        question="   ",
        train_answer="Answer.",
    )

    with pytest.raises(ValueError, match="Question text is empty"):
        ChatTemplatePreprocessor(TrainingMethod).tokenize(
            sample,
            _FakeProcessor(),
            model_config,
        )
