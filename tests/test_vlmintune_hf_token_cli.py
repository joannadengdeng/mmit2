import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.config.training_config import config_to_trainer_dict, load_config
from vlmintune.eval.__main__ import apply_hf_token as apply_eval_hf_token
from vlmintune.eval.__main__ import run as run_eval_from_config
from vlmintune.training.__main__ import apply_hf_token as apply_train_hf_token


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


def test_training_config_loads_local_yaml_for_current_machine(tmp_path):
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        """
model: qwen25vl_3b_instruct
dataset: lmms-lab/textvqa
method: lora
output_dir: output/demo
""".strip()
        + "\n",
        encoding="utf-8",
    )

    loaded = config_to_trainer_dict(load_config(str(config_path)))

    assert loaded["method"] == "lora"
    assert loaded["output_dir"] == "output/demo"


def test_eval_cli_runs_local_yaml(monkeypatch, tmp_path):
    config_path = tmp_path / "eval_config.yaml"
    config_path.write_text(
        """
experiment:
  name: "demo_exp"
  base_dir: "experiments"
model:
  name: "qwen25vl_3b_instruct"
eval:
  source: "base"
  dataset_name: "lmms-lab/textvqa"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    captured = {}

    def fake_run_eval_config(raw_cfg):
        captured["cfg"] = raw_cfg

    monkeypatch.setattr("vlmintune.eval.__main__.run_eval_config", fake_run_eval_config)

    run_eval_from_config(str(config_path))

    assert captured["cfg"]["eval"]["source"] == "base"
