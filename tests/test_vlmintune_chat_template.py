import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.types import CanonicalSample, Turn
from vlmintune.training.chat_template import ChatTemplatePreprocessor


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        return "PROMPT" if add_generation_prompt else "FULL"

    def __call__(self, text, images=None, return_tensors=None, truncation=None, max_length=None):
        del images, return_tensors, truncation, max_length
        if text == "PROMPT":
            return {"input_ids": torch.tensor([[11, 12]])}
        return {
            "input_ids": torch.tensor([[11, 12, 13, 14]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }


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

    assert result["input_ids"].tolist() == [11, 12, 13, 14]
    assert result["prompt_preview"]["sample_id"] == "sample-1"
    assert result["prompt_preview"]["full_text"] == "FULL"
    assert result["prompt_preview"]["prompt_text"] == "PROMPT"
    assert result["prompt_preview"]["has_image"] is False
