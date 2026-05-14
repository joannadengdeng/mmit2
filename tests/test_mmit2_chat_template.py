import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmit2.data.types import CanonicalSample, Turn
from mmit2.training.preprocessors.chat_template import ChatTemplatePreprocessor


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


def test_chat_template_debug_sink_receives_rendered_prompt():
    sample = CanonicalSample(
        id="sample-1",
        image_path="",
        turns=[
            Turn(role="human", content="Question?"),
            Turn(role="assistant", content="Answer."),
        ],
    )
    records = []

    result = ChatTemplatePreprocessor().tokenize(
        sample,
        _FakeProcessor(),
        debug_sink=records.append,
    )

    assert result["input_ids"].tolist() == [11, 12, 13, 14]
    assert len(records) == 1
    assert records[0]["sample_id"] == "sample-1"
    assert records[0]["full_text"] == "FULL"
    assert records[0]["prompt_text"] == "PROMPT"
    assert records[0]["has_image"] is False
