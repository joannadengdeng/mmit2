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
  data_path: "lmms-lab/textvqa"
  split: train
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo_root)
    cfg = load_config(str(config_path.relative_to(repo_root)))

    assert cfg.experiment.setup_dir == "experiment_setup/demo_exp"
