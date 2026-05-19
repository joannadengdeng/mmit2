import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.eval.__main__ import apply_hf_token as apply_eval_hf_token
from vlmintune.eval.__main__ import run as run_eval_over_ssh
from vlmintune.training.__main__ import apply_hf_token as apply_train_hf_token
from vlmintune.training.runner import run as run_train_over_ssh


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


def test_training_runner_host_override_wins_over_yaml(monkeypatch, tmp_path):
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        """
runtime:
  mode: ssh
  ssh:
    username: "root"
model:
  model_path: "Qwen/Qwen2.5-VL-3B-Instruct"
experiment:
  name: "demo_exp"
  base_dir: "experiments"
training:
  ft_method: freeze
  params:
    model_layout: "qwen2_5_vl"
    unfreeze_modules: ["model.language_model.layers.0"]
data:
  adapter: hf_datasets
  data_path: "lmms-lab/textvqa"
  split: train
""".strip()
        + "\n",
        encoding="utf-8",
    )

    captured = {}

    def fake_run_remote_module(ssh_cfg, **kwargs):
        captured["host"] = ssh_cfg.host
        captured["payload"] = kwargs["payload"]

    monkeypatch.setattr("vlmintune.training.runner.run_remote_module", fake_run_remote_module)
    monkeypatch.setattr("vlmintune.training.runner.ensure_peft_runtime_compatible", lambda *args, **kwargs: None)

    run_train_over_ssh(str(config_path), host_override="10.0.0.8")

    assert captured["host"] == "10.0.0.8"
    assert "runtime" not in captured["payload"]


def test_eval_cli_host_override_wins_over_yaml(monkeypatch, tmp_path):
    config_path = tmp_path / "eval_config.yaml"
    config_path.write_text(
        """
runtime:
  mode: ssh
  ssh:
    username: "root"
model:
  model_path: "Qwen/Qwen2.5-VL-3B-Instruct"
eval:
  dataset_name: "lmms-lab/textvqa"
  split: validation
""".strip()
        + "\n",
        encoding="utf-8",
    )

    captured = {}

    def fake_run_remote_module(ssh_cfg, **kwargs):
        captured["host"] = ssh_cfg.host
        captured["payload"] = kwargs["payload"]

    monkeypatch.setattr("vlmintune.eval.__main__.run_remote_module", fake_run_remote_module)

    run_eval_over_ssh(str(config_path), host_override="10.0.0.9")

    assert captured["host"] == "10.0.0.9"


def test_training_config_requires_host_without_override(tmp_path):
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        """
runtime:
  mode: ssh
  ssh:
    username: "root"
model:
  model_path: "Qwen/Qwen2.5-VL-3B-Instruct"
experiment:
  name: "demo_exp"
  base_dir: "experiments"
training:
  ft_method: freeze
  params:
    model_layout: "qwen2_5_vl"
    unfreeze_modules: ["model.language_model.layers.0"]
data:
  adapter: hf_datasets
  data_path: "lmms-lab/textvqa"
  split: train
""".strip()
        + "\n",
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="runtime.ssh.host: required"):
        run_train_over_ssh(str(config_path))
