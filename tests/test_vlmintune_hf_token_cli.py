import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.eval.__main__ import apply_hf_token as apply_eval_hf_token
from vlmintune.eval.__main__ import run as run_eval_from_config
from vlmintune.training.__main__ import apply_hf_token as apply_train_hf_token
from vlmintune.training.__main__ import load_config_or_dispatch
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


def test_training_main_loads_local_yaml_when_host_is_missing(tmp_path):
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

    loaded = load_config_or_dispatch(str(config_path))

    assert loaded is not None
    assert loaded["training_method"] == "freeze"
    assert loaded["experiment"]["name"] == "demo_exp"


def test_training_main_dispatches_when_host_is_present(monkeypatch, tmp_path):
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        """
runtime:
  mode: ssh
  ssh:
    host: "10.0.0.8"
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

    def fake_run(config_path_arg: str):
        captured["config_path"] = config_path_arg

    monkeypatch.setattr("vlmintune.training.runner.run", fake_run)

    loaded = load_config_or_dispatch(str(config_path))

    assert loaded is None
    assert captured["config_path"] == str(config_path)


def test_eval_cli_runs_local_yaml_when_host_is_missing(monkeypatch, tmp_path):
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
  metric: "vqa_accuracy"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    captured = {}

    def fake_run_eval_config(raw_cfg):
        captured["cfg"] = raw_cfg

    monkeypatch.setattr("vlmintune.eval.__main__.run_eval_config", fake_run_eval_config)

    run_eval_from_config(str(config_path))

    assert captured["cfg"]["eval"]["split"] == "validation"


def test_eval_cli_dispatches_when_host_is_present(monkeypatch, tmp_path):
    config_path = tmp_path / "eval_config.yaml"
    config_path.write_text(
        """
runtime:
  mode: ssh
  ssh:
    host: "10.0.0.9"
    username: "root"
model:
  model_path: "Qwen/Qwen2.5-VL-3B-Instruct"
eval:
  dataset_name: "lmms-lab/textvqa"
  split: validation
  metric: "vqa_accuracy"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    captured = {}

    def fake_run_remote_module(ssh_cfg, **kwargs):
        captured["host"] = ssh_cfg.host
        captured["payload"] = kwargs["payload"]

    monkeypatch.setattr("vlmintune.eval.__main__.run_remote_module", fake_run_remote_module)

    run_eval_from_config(str(config_path))

    assert captured["host"] == "10.0.0.9"


def test_training_runner_still_requires_host_in_yaml(tmp_path):
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

    with pytest.raises(ValueError, match="runtime.ssh.host: required"):
        run_train_over_ssh(str(config_path))
