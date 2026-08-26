import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "scripts" / "audit_qwen_textvqa_combination_stage.py"
STAGE_PATH = ROOT / "scripts" / "run_qwen_textvqa_combination_stage.sh"
SPEC = importlib.util.spec_from_file_location("combination_audit", AUDIT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_valid_run(tmp_path: Path, method: str) -> dict[str, Path]:
    experiments_dir = (tmp_path / "experiments").resolve()
    run_prefix = "qwen_textvqa_combo"
    run_name = f"{run_prefix}_{method}_n8_s42"
    run_dir = experiments_dir / run_name
    checkpoint_dir = run_dir / "checkpoint"
    eval_dir = run_dir / "eval_trained"
    checkpoint_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    marker_name, learning_rate = AUDIT.METHOD_SPECS[method]
    (checkpoint_dir / marker_name).write_bytes(b"marker")
    if method in AUDIT.ADAPTER_METHODS:
        (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"weights")
    joint_spec = AUDIT.JOINT_METHOD_SPECS[method]
    for filename in joint_spec["checkpoint_components"].values():
        (checkpoint_dir / filename).write_bytes(b"component weights")

    train_config = {
        "model": "qwen25vl_3b_instruct",
        "dataset": "lmms-lab/textvqa",
        "method": method,
        "epochs": 1,
        "learning_rate": learning_rate,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_length": 1536,
        "max_samples": 8,
        "seed": 42,
        "output_dir": str(checkpoint_dir),
    }
    (checkpoint_dir / "train_config.yaml").write_text(
        yaml.safe_dump(train_config, sort_keys=False),
        encoding="utf-8",
    )

    metadata = {
        "model_name": "qwen25vl_3b_instruct",
        "ft_method": method,
        "final_loss": 0.25,
    }
    metadata["recipe"] = joint_spec["recipe"]
    metadata["combination_recipe"] = joint_spec["recipe"]
    metadata["structure_methods"] = joint_spec["structure_methods"]
    metadata["composition_order"] = joint_spec["composition_order"]
    metadata["component_recipes"] = joint_spec["component_recipes"]
    metadata["checkpoint_components"] = joint_spec["checkpoint_components"]
    _write_json(checkpoint_dir / "vlmintune_meta.json", metadata)

    train_dir = run_dir / "train"
    train_dir.mkdir()
    (train_dir / "run_20260811T000000Z.log").write_text(
        json.dumps({
            "type": "data_summary",
            "data": {"kind": "preprocessing_coverage", "total_skipped": 0},
        }) + "\n",
        encoding="utf-8",
    )

    eval_config = {
        "model": {"name": "qwen25vl_3b_instruct"},
        "experiment": {"name": run_name, "base_dir": str(experiments_dir)},
        "eval": {
            "source": "trained",
            "dataset_name": "lmms-lab/textvqa",
            "split": "validation",
            "max_samples": 2,
            "sample_seed": 42,
            "shuffle_buffer_size": 10_000,
            "max_new_tokens": 16,
            "temperature": 0.0,
        },
    }
    (run_dir / "eval_trained_config.yaml").write_text(
        yaml.safe_dump(eval_config, sort_keys=False),
        encoding="utf-8",
    )

    records = [
        {
            "id": "q1",
            "question": "First question?",
            "prediction": "one",
            "ground_truth": ["one"],
            "scores": {"vqa_accuracy": 0.3},
        },
        {
            "id": "q2",
            "question": "Second question?",
            "prediction": "two",
            "ground_truth": ["two"],
            "scores": {"vqa_accuracy": 0.6},
        },
    ]
    predictions_path = eval_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "experiment_name": run_name,
        "source": "trained",
        "model_name": "qwen25vl_3b_instruct",
        "hf_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "dataset_name": "lmms-lab/textvqa",
        "split": "validation",
        "metric": "vqa_accuracy",
        "num_predictions": 2,
        "metrics": {"vqa_accuracy": 45.0},
        "sample_seed": 42,
        "shuffle_buffer_size": 10_000,
        "diagnostics": {},
    }
    _write_json(eval_dir / "eval.json", summary)
    _write_json(
        eval_dir / "eval_ids.json",
        {
            "dataset_name": "lmms-lab/textvqa",
            "split": "validation",
            "sample_seed": 42,
            "shuffle_buffer_size": 10_000,
            "max_samples": 2,
            "ids": ["q1", "q2"],
        },
    )
    return {
        "experiments_dir": experiments_dir,
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "eval_dir": eval_dir,
        "predictions_path": predictions_path,
    }


def _audit(paths: dict[str, Path], method: str):
    return AUDIT.audit_run(
        experiments_dir=paths["experiments_dir"],
        run_prefix="qwen_textvqa_combo",
        method=method,
        train_samples=8,
        eval_samples=2,
        grad_acc=1,
        max_length=1536,
        epochs=1,
        seed=42,
    )


@pytest.mark.parametrize("method", sorted(AUDIT.METHOD_SPECS))
def test_strict_audit_accepts_every_fixed_combination(tmp_path, method):
    paths = _build_valid_run(tmp_path, method)

    result = _audit(paths, method)

    assert result == {
        "run_name": f"qwen_textvqa_combo_{method}_n8_s42",
        "final_loss": 0.25,
        "vqa_accuracy": 45.0,
        "predictions": 2,
        "total_skipped": 0,
    }


def test_strict_audit_rejects_wrong_eval_identity(tmp_path):
    paths = _build_valid_run(tmp_path, "mores_lora")
    summary_path = paths["eval_dir"] / "eval.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source"] = "base"
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="eval source='base', expected 'trained'"):
        _audit(paths, "mores_lora")


def test_strict_audit_rejects_duplicate_prediction_ids(tmp_path):
    paths = _build_valid_run(tmp_path, "mores_dora")
    records = [
        json.loads(line)
        for line in paths["predictions_path"].read_text(encoding="utf-8").splitlines()
    ]
    records[1]["id"] = records[0]["id"]
    paths["predictions_path"].write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate prediction ids"):
        _audit(paths, "mores_dora")


def test_strict_audit_rejects_summary_score_mismatch(tmp_path):
    paths = _build_valid_run(tmp_path, "reft_lora")
    summary_path = paths["eval_dir"] / "eval.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"]["vqa_accuracy"] = 99.0
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="calculated=45.0"):
        _audit(paths, "reft_lora")


def test_joint_audit_requires_the_shared_combination_recipe(tmp_path):
    paths = _build_valid_run(tmp_path, "mores_dora")
    metadata_path = paths["checkpoint_dir"] / "vlmintune_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("combination_recipe")
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="wrong combination metadata"):
        _audit(paths, "mores_dora")


def test_checkpoint_scope_accepts_valid_training_without_eval(tmp_path):
    paths = _build_valid_run(tmp_path, "reft_lora")
    for child in paths["eval_dir"].iterdir():
        child.unlink()
    paths["eval_dir"].rmdir()
    (paths["run_dir"] / "eval_trained_config.yaml").unlink()

    result = AUDIT.audit_run(
        experiments_dir=paths["experiments_dir"],
        run_prefix="qwen_textvqa_combo",
        method="reft_lora",
        train_samples=8,
        eval_samples=2,
        grad_acc=1,
        max_length=1536,
        epochs=1,
        seed=42,
        scope="checkpoint",
    )

    assert result == {
        "run_name": "qwen_textvqa_combo_reft_lora_n8_s42",
        "final_loss": 0.25,
        "total_skipped": 0,
    }


def test_mores_lora_audit_requires_both_structural_weight_files(tmp_path):
    paths = _build_valid_run(tmp_path, "mores_lora")
    (paths["checkpoint_dir"] / "mores_tuned.pt").unlink()

    with pytest.raises(ValueError, match="mores checkpoint component"):
        _audit(paths, "mores_lora")


def test_mores_lora_audit_rejects_wrong_structure_order(tmp_path):
    paths = _build_valid_run(tmp_path, "mores_lora")
    metadata_path = paths["checkpoint_dir"] / "vlmintune_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["structure_methods"] = ["lora", "mores"]
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="wrong structure metadata"):
        _audit(paths, "mores_lora")


def _make_fake_runtime(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    fake_venv = tmp_path / "fake_venv"
    fake_bin = fake_venv / "bin"
    fake_bin.mkdir(parents=True)
    eval_marker = tmp_path / "fake_eval_invoked"
    train_marker = tmp_path / "fake_train_invoked"
    eval_writer = tmp_path / "fake_eval_writer.py"
    eval_writer.write_text(
        """
import argparse
import json
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
run_name = config["experiment"]["name"]
base_dir = Path(config["experiment"]["base_dir"])
eval_config = config["eval"]
count = int(eval_config["max_samples"])
eval_dir = base_dir / run_name / "eval_trained"
eval_dir.mkdir(parents=True, exist_ok=True)
records = []
ids = []
for index in range(count):
    sample_id = f"recovered-{index}"
    ids.append(sample_id)
    records.append({
        "id": sample_id,
        "question": f"Question {index}?",
        "prediction": "answer",
        "ground_truth": ["answer"],
        "scores": {"vqa_accuracy": 0.5},
    })
(eval_dir / "predictions.jsonl").write_text(
    "".join(json.dumps(record) + "\\n" for record in records),
    encoding="utf-8",
)
(eval_dir / "eval.json").write_text(json.dumps({
    "experiment_name": run_name,
    "source": "trained",
    "model_name": "qwen25vl_3b_instruct",
    "hf_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "dataset_name": "lmms-lab/textvqa",
    "split": "validation",
    "metric": "vqa_accuracy",
    "num_predictions": count,
    "metrics": {"vqa_accuracy": 50.0},
    "sample_seed": eval_config["sample_seed"],
    "shuffle_buffer_size": eval_config["shuffle_buffer_size"],
    "diagnostics": {},
}) + "\\n", encoding="utf-8")
(eval_dir / "eval_ids.json").write_text(json.dumps({
    "dataset_name": "lmms-lab/textvqa",
    "split": "validation",
    "sample_seed": eval_config["sample_seed"],
    "shuffle_buffer_size": eval_config["shuffle_buffer_size"],
    "max_samples": count,
    "ids": ids,
}) + "\\n", encoding="utf-8")
Path(__import__("os").environ["FAKE_EVAL_MARKER"]).write_text("called\\n")
""".lstrip(),
        encoding="utf-8",
    )
    train_writer = tmp_path / "fake_train_writer.py"
    train_writer.write_text(
        """
import argparse
import json
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args, _ = parser.parse_known_args()
config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
checkpoint = Path(config["output_dir"])
checkpoint.mkdir(parents=True, exist_ok=True)
method = config["method"]
markers = {
    "mores_lora": "adapter_config.json",
    "mores_dora": "adapter_config.json",
    "reft_lora": "adapter_config.json",
}
(checkpoint / markers[method]).write_bytes(b"marker")
(checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
if method in {"mores_lora", "mores_dora"}:
    (checkpoint / "mores_tuned.pt").write_bytes(b"mores weights")
if method == "reft_lora":
    (checkpoint / "reft_tuned.pt").write_bytes(b"reft weights")
metadata = {
    "model_name": "qwen25vl_3b_instruct",
    "hf_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "ft_method": method,
    "final_loss": 0.125,
}
joint_specs = {
    "mores_lora": {
        "structure_methods": ["mores", "lora"],
        "component_recipes": {"mores": "mores", "lora": "lora_v1"},
        "checkpoint_components": {
            "lora": "adapter_model.safetensors",
            "mores": "mores_tuned.pt",
        },
    },
    "mores_dora": {
        "structure_methods": ["mores", "dora"],
        "component_recipes": {"mores": "mores", "dora": "dora_v1"},
        "checkpoint_components": {
            "dora": "adapter_model.safetensors",
            "mores": "mores_tuned.pt",
        },
    },
    "reft_lora": {
        "structure_methods": ["reft", "lora"],
        "component_recipes": {
            "reft": "reft_tied_rank4_p4_s4_all_layers_v1",
            "lora": "lora_v1",
        },
        "checkpoint_components": {
            "lora": "adapter_model.safetensors",
            "reft": "reft_tuned.pt",
        },
    },
}
spec = joint_specs[method]
metadata["recipe"] = f"{method}_v1"
metadata["combination_recipe"] = f"{method}_v1"
metadata["structure_methods"] = spec["structure_methods"]
metadata["composition_order"] = spec["structure_methods"]
metadata["component_recipes"] = spec["component_recipes"]
metadata["checkpoint_components"] = spec["checkpoint_components"]
(checkpoint / "vlmintune_meta.json").write_text(
    json.dumps(metadata) + "\\n", encoding="utf-8"
)
print(json.dumps({
    "type": "data_summary",
    "data": {"kind": "preprocessing_coverage", "total_skipped": 0},
}), flush=True)
Path(__import__("os").environ["FAKE_TRAIN_MARKER"]).write_text("called\\n")
""".lstrip(),
        encoding="utf-8",
    )
    fake_benchmark = tmp_path / "fake_benchmark.sh"
    fake_benchmark.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${DRY_RUN:-}" != "0" ]]; then
  echo "stage failed to force DRY_RUN=0" >&2
  exit 42
fi
mkdir -p "$(dirname "$CONFIG_PATH")"
cat > "$CONFIG_PATH" <<YAML
model: "$MODEL"
dataset: "$DATASET"
method: "$METHOD"
epochs: $EPOCHS
learning_rate: $LEARNING_RATE
batch_size: $BATCH_SIZE
gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS
max_length: $MAX_LENGTH
max_samples: $MAX_SAMPLES
seed: $SEED
output_dir: "$OUTPUT_DIR"
YAML
python -m vlmintune.training --config "$CONFIG_PATH"
""",
        encoding="utf-8",
    )
    fake_benchmark.chmod(0o755)
    python_wrapper = fake_bin / "python"
    python_wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-m" && "${2:-}" == "vlmintune.eval" ]]; then
  shift 2
  exec "$REAL_PYTHON" "$FAKE_EVAL_WRITER" "$@"
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "vlmintune.training" ]]; then
  shift 2
  exec "$REAL_PYTHON" "$FAKE_TRAIN_WRITER" "$@"
fi
exec "$REAL_PYTHON" "$@"
""",
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)

    hf_cache = tmp_path / "hf_cache"
    (hf_cache / "hub" / "models--Qwen--Qwen2.5-VL-3B-Instruct").mkdir(
        parents=True
    )
    return fake_venv, hf_cache, eval_marker, train_marker


def _run_stage(tmp_path: Path, paths: dict[str, Path], method: str):
    fake_venv, hf_cache, eval_marker, train_marker = _make_fake_runtime(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "VENV_DIR": str(fake_venv),
            "HF_CACHE_ROOT": str(hf_cache),
            "EXPERIMENTS_DIR": str(paths["experiments_dir"]),
            "STAGE_SAMPLES": "8",
            "EVAL_SAMPLES": "2",
            "GRADIENT_ACCUMULATION_STEPS": "1",
            "METHODS": method,
            "RUN_PREFIX": "qwen_textvqa_combo",
            "REAL_PYTHON": sys.executable,
            "FAKE_EVAL_WRITER": str(tmp_path / "fake_eval_writer.py"),
            "FAKE_EVAL_MARKER": str(eval_marker),
            "FAKE_TRAIN_WRITER": str(tmp_path / "fake_train_writer.py"),
            "FAKE_TRAIN_MARKER": str(train_marker),
            "BENCHMARK_SCRIPT": str(tmp_path / "fake_benchmark.sh"),
            "DRY_RUN": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(STAGE_PATH)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, eval_marker, train_marker


def test_stage_skips_a_fully_strict_completed_run(tmp_path):
    paths = _build_valid_run(tmp_path, "mores_dora")

    result, eval_marker, train_marker = _run_stage(tmp_path, paths, "mores_dora")

    assert result.returncode == 0, result.stderr
    assert "SKIP strict completed run" in result.stdout
    assert not eval_marker.exists()
    assert not train_marker.exists()


def test_stage_recovers_only_eval_and_preserves_invalid_outputs(tmp_path):
    paths = _build_valid_run(tmp_path, "mores_lora")
    summary_path = paths["eval_dir"] / "eval.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source"] = "base"
    _write_json(summary_path, summary)

    result, eval_marker, train_marker = _run_stage(tmp_path, paths, "mores_lora")

    assert result.returncode == 0, result.stderr
    assert eval_marker.is_file()
    assert "REUSE strict checkpoint" in result.stdout
    assert "Preserved prior evaluation directory" in result.stdout
    assert len(list(paths["run_dir"].glob("eval_trained.invalid_*"))) == 1
    assert len(list(paths["run_dir"].glob("eval_trained_config.yaml.invalid_*"))) == 1
    recovered = _audit(paths, "mores_lora")
    assert recovered["vqa_accuracy"] == 50.0
    assert not train_marker.exists()


def test_stage_stops_without_overwriting_invalid_checkpoint(tmp_path):
    paths = _build_valid_run(tmp_path, "reft_lora")
    config_path = paths["checkpoint_dir"] / "train_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["learning_rate"] = 9e-4
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result, eval_marker, train_marker = _run_stage(tmp_path, paths, "reft_lora")

    assert result.returncode == 6
    assert "preserving them and stopping" in result.stderr
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["learning_rate"] == 9e-4
    assert not eval_marker.exists()
    assert not train_marker.exists()


def test_stage_forces_real_training_when_parent_exports_dry_run(tmp_path):
    experiments_dir = (tmp_path / "experiments").resolve()
    experiments_dir.mkdir()
    paths = {"experiments_dir": experiments_dir}

    result, eval_marker, train_marker = _run_stage(tmp_path, paths, "mores_dora")

    assert result.returncode == 0, result.stderr
    assert train_marker.is_file()
    assert eval_marker.is_file()
    assert "TRAIN qwen_textvqa_combo_mores_dora_n8_s42" in result.stdout
    run_dir = experiments_dir / "qwen_textvqa_combo_mores_dora_n8_s42"
    assert (run_dir / "checkpoint" / "vlmintune_meta.json").is_file()
    assert (run_dir / "eval_trained" / "eval.json").is_file()


def test_stage_runs_and_strictly_audits_mores_lora(tmp_path):
    experiments_dir = (tmp_path / "experiments").resolve()
    experiments_dir.mkdir()
    paths = {"experiments_dir": experiments_dir}

    result, eval_marker, train_marker = _run_stage(tmp_path, paths, "mores_lora")

    assert result.returncode == 0, result.stderr
    assert train_marker.is_file()
    assert eval_marker.is_file()
    run_dir = experiments_dir / "qwen_textvqa_combo_mores_lora_n8_s42"
    assert (run_dir / "checkpoint" / "adapter_model.safetensors").is_file()
    assert (run_dir / "checkpoint" / "mores_tuned.pt").is_file()
    assert _audit(
        {
            "experiments_dir": experiments_dir,
            "run_dir": run_dir,
            "checkpoint_dir": run_dir / "checkpoint",
            "eval_dir": run_dir / "eval_trained",
            "predictions_path": run_dir / "eval_trained" / "predictions.jsonl",
        },
        "mores_lora",
    )["vqa_accuracy"] == 50.0
