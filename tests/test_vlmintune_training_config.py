import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.config.training_config import config_to_trainer_dict, load_config


def test_load_config_infers_experiment_setup_dir(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    setup_dir = repo_root / "experiment_setup" / "demo_exp"
    setup_dir.mkdir(parents=True)
    config_path = setup_dir / "train_config.yaml"
    config_path.write_text(
        """
model:
  name: "qwen25vl_3b_instruct"
experiment:
  name: "demo_exp"
  base_dir: "experiments"
training:
  ft_method: lora
  params:
    target_modules: ["q_proj", "v_proj"]
data:
  dataset_name: "lmms-lab/textvqa"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo_root)
    cfg = load_config(str(config_path.relative_to(repo_root)))

    assert cfg.experiment.setup_dir == "experiment_setup/demo_exp"


def test_load_config_requires_experiment_name(tmp_path):
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        """
model:
  name: "qwen25vl_3b_instruct"
training:
  ft_method: lora
  params:
    target_modules: ["q_proj", "v_proj"]
data:
  dataset_name: "lmms-lab/textvqa"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        load_config(str(config_path))
    except ValueError as exc:
        assert "experiment.name" in str(exc)
        return
    raise AssertionError("load_config should require experiment.name")


def test_load_config_preserves_training_perf_overrides(tmp_path):
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        """
model:
  name: "qwen25vl_3b_instruct"
experiment:
  name: "demo_exp"
training:
  ft_method: lora
  per_device_batch_size: 2
  gradient_accumulation_steps: 2
  max_length: 1536
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  dataloader_persistent_workers: true
  params:
    target_modules: ["q_proj", "v_proj"]
data:
  dataset_name: "lmms-lab/textvqa"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(str(config_path))
    trainer_dict = config_to_trainer_dict(cfg)

    assert cfg.training.per_device_batch_size == 2
    assert cfg.training.gradient_accumulation_steps == 2
    assert cfg.training.max_length == 1536
    assert cfg.training.dataloader_num_workers == 4
    assert cfg.training.dataloader_pin_memory is True
    assert cfg.training.dataloader_persistent_workers is True
    assert trainer_dict["training"]["max_length"] == 1536
    assert trainer_dict["training"]["dataloader_num_workers"] == 4
    assert trainer_dict["training"]["dataloader_pin_memory"] is True
    assert trainer_dict["training"]["dataloader_persistent_workers"] is True
    assert "split" not in trainer_dict["data"]
