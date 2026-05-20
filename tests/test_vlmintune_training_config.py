import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.config.training_config import load_config


def test_load_config_infers_experiment_setup_dir(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    setup_dir = repo_root / "experiment_setup" / "demo_exp"
    setup_dir.mkdir(parents=True)
    config_path = setup_dir / "train_config.yaml"
    config_path.write_text(
        """
runtime:
  mode: ssh
  ssh:
    host: "127.0.0.1"
    username: "root"
model:
  model_path: "Qwen/Qwen2.5-VL-3B-Instruct"
experiment:
  name: "demo_exp"
  base_dir: "experiments"
training:
  ft_method: lora
  params:
    target_modules: ["q_proj", "v_proj"]
data:
  adapter: hf_datasets
  data_path: "lmms-lab/textvqa"
  split: train
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo_root)
    cfg = load_config(str(config_path.relative_to(repo_root)))

    assert cfg.experiment.setup_dir == "experiment_setup/demo_exp"


def test_load_config_accepts_mores_with_model_layout(tmp_path):
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        """
runtime:
  mode: ssh
  ssh:
    host: "127.0.0.1"
    username: "root"
model:
  model_path: "Qwen/Qwen2.5-VL-3B-Instruct"
experiment:
  name: "demo_exp"
  base_dir: "experiments"
training:
  ft_method: mores
  params:
    model_layout: "qwen2_5_vl"
    hidden_size: 2048
    intervention_positions: "uniform9"
data:
  adapter: hf_datasets
  data_path: "lmms-lab/textvqa"
  split: train
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(str(config_path))

    assert cfg.training.ft_method == "mores"
    assert cfg.training.params["steering_rank"] == 1
    assert cfg.training.params["intervention_positions"] == "uniform9"


def test_load_config_rejects_mores_without_hidden_size(tmp_path):
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        """
runtime:
  mode: ssh
  ssh:
    host: "127.0.0.1"
    username: "root"
model:
  model_path: "Qwen/Qwen2.5-VL-3B-Instruct"
experiment:
  name: "demo_exp"
  base_dir: "experiments"
training:
  ft_method: mores
  params:
    model_layout: "qwen2_5_vl"
data:
  adapter: hf_datasets
  data_path: "lmms-lab/textvqa"
  split: train
""".strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        load_config(str(config_path))
    except ValueError as exc:
        assert "training.params.hidden_size" in str(exc)
    else:
        raise AssertionError("Expected missing MoReS hidden_size to fail validation")


def test_load_config_rejects_invalid_mores_intervention_positions(tmp_path):
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        """
runtime:
  mode: ssh
  ssh:
    host: "127.0.0.1"
    username: "root"
model:
  model_path: "Qwen/Qwen2.5-VL-3B-Instruct"
experiment:
  name: "demo_exp"
  base_dir: "experiments"
training:
  ft_method: mores
  params:
    model_layout: "qwen2_5_vl"
    hidden_size: 2048
    intervention_positions: "first"
data:
  adapter: hf_datasets
  data_path: "lmms-lab/textvqa"
  split: train
""".strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        load_config(str(config_path))
    except ValueError as exc:
        assert "training.params.intervention_positions" in str(exc)
    else:
        raise AssertionError("Expected invalid MoReS intervention_positions to fail validation")
