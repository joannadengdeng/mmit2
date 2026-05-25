import os
import sys

import torch

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


def test_chat_template_tokenize_includes_rendered_prompt_preview():
    sample = CanonicalSample(
        id="sample-1",
        image_path="",
        turns=[
            Turn(role="human", content="Question?"),
            Turn(role="assistant", content="Answer."),
        ],
    )

    result = ChatTemplatePreprocessor().tokenize(
        sample,
        _FakeProcessor(),
    )

    assert result["input_ids"].tolist() == [99, 12, 13, 14]
    assert result["prompt_preview"]["sample_id"] == "sample-1"
    assert result["prompt_preview"]["full_text"] == "FULL"
    assert result["prompt_preview"]["prompt_text"] == "PROMPT"
    assert result["prompt_preview"]["has_image"] is False
    assert result["prompt_preview"]["instruction_texts"] == ["Question?"]
    assert result["prompt_preview"]["instruction_supervision_spans"] == [
        {"start": 1, "end": 2, "token_count": 1, "text": "Question?"}
    ]
    assert result["prompt_mask"].tolist() == [True, True, False, False]
    assert result["instruction_supervision_mask"].tolist() == [False, True, False, False]
