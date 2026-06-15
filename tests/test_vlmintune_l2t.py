import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.types import CanonicalSample
from vlmintune.training.methods.l2t import L2TMethod, build_instruction_supervision_mask
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


def test_l2t_defaults_match_lora_shape():
    defaults = L2TMethod().default_config()

    assert "base_method" not in defaults
    assert "train_layer_range" not in defaults
    assert defaults["target_modules"] == []


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
    sample = CanonicalSample(id="1", image_path="", question="what brand?")
    input_ids = torch.tensor([99, 12, 13, 98, 77])

    mask = build_instruction_supervision_mask(
        processor=_PrefixSensitiveProcessor(),
        sample=sample,
        input_ids=input_ids,
        prompt_len=4,
        max_length=16,
    )

    assert mask.tolist() == [False, True, True, False, False]


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
