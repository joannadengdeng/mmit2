import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmit2.training.methods.l2t import L2TMethod


def test_l2t_unmasks_prompt_only():
    method = L2TMethod()
    method._special_token_ids = {12}
    input_ids = torch.tensor([[11, 12, 13, 14, 0]])
    labels = torch.tensor([[-100, -100, -100, 14, -100]])
    batch_meta = {
        "prompt_mask": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
    }

    updated = method.preprocess_labels(input_ids, labels, batch_meta=batch_meta)

    assert updated.tolist() == [[11, -100, 13, 14, -100]]
