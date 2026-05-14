import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmit2.eval.__main__ import apply_hf_token as apply_eval_hf_token
from mmit2.training.__main__ import apply_hf_token as apply_train_hf_token


def test_eval_cli_applies_direct_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    apply_eval_hf_token("hf_direct_token", None)

    assert os.environ["HF_TOKEN"] == "hf_direct_token"


def test_training_cli_reads_hf_token_file(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    token_file = tmp_path / "hf_token.txt"
    token_file.write_text("hf_file_token\n", encoding="utf-8")

    apply_train_hf_token(None, str(token_file))

    assert os.environ["HF_TOKEN"] == "hf_file_token"
