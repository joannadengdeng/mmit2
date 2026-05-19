import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.results.bundle import build_bundle


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def test_bundle_results_collects_configs_scripts_and_local_paths(tmp_path):
    experiments_dir = tmp_path / "experiments"
    experiment_dir = experiments_dir / "demo_exp"
    checkpoint_dir = experiment_dir / "checkpoint"
    debug_dir = experiment_dir / "debug"
    eval_pred_dir = experiment_dir / "eval_predictions"
    checkpoint_dir.mkdir(parents=True)
    debug_dir.mkdir(parents=True)
    eval_pred_dir.mkdir(parents=True)

    _write_json(
        experiment_dir / "summary.json",
        {
            "exp_id": "demo_exp",
            "status": "completed",
            "created_at": "2026-05-19T10:00:00",
            "completed_at": "2026-05-19T11:00:00",
            "error": "",
            "method": "lora",
            "model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "dataset": "lmms-lab/textvqa",
            "num_samples": 100,
            "config": {
                "model": {"model_path": "Qwen/Qwen2.5-VL-3B-Instruct"},
                "data": {
                    "adapter": "hf_datasets",
                    "data_path": "lmms-lab/textvqa",
                    "split": "train",
                    "max_samples": 100,
                },
                "training_method": "lora",
                "method_params": {
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "v_proj"],
                },
                "training": {
                    "num_epochs": 1,
                    "per_device_batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "learning_rate": 2e-4,
                    "warmup_ratio": 0.03,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "save_steps": 0,
                    "output_dir": "/remote/experiments",
                },
            },
            "train_summary": {
                "avg_loss": 0.22,
                "total_steps": 50,
                "train_time_s": 12.3,
                "trainable_params": 100,
                "total_params": 1000,
                "trainable_pct": 10.0,
            },
            "eval_results": {
                "textvqa_validation": {
                    "vqa_accuracy": 81.0,
                }
            },
            "checkpoint_path": "/remote/experiments/demo_exp/checkpoint",
            "exp_dir": "/remote/experiments/demo_exp",
        },
    )
    _write_json(
        debug_dir / "first_5_canonical_samples.json",
        [
            {"id": "a", "question": "What word is shown?"},
            {"id": "b", "question": "What number is shown?"},
        ],
    )
    _write_json(
        debug_dir / "first_5_rendered_prompts.json",
        [
            {"prompt": "Answer with a single short answer."},
            {"prompt": "Read the text in the image."},
        ],
    )
    _write_json(
        debug_dir / "skip_summary.json",
        {
            "total_skipped": 1,
            "first_errors": [{"sample_id": "bad-1", "error": "broken image"}],
        },
    )

    (checkpoint_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    (checkpoint_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (eval_pred_dir / "textvqa_validation.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")

    baseline_eval_dir = tmp_path / "eval_outputs" / "baseline_demo"
    baseline_pred_dir = baseline_eval_dir / "eval_predictions"
    baseline_pred_dir.mkdir(parents=True)
    _write_json(
        baseline_eval_dir / "eval_summary.json",
        {"metrics": {"vqa_accuracy": 74.33}},
    )
    (baseline_pred_dir / "textvqa_validation.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")

    run_logs_dir = tmp_path / "run_logs"
    run_logs_dir.mkdir()
    (run_logs_dir / "demo_exp.log").write_text(
        "\n".join(
            [
                "[vlmintune] Starting JarvisLabs LoRA run",
                "[vlmintune] Model: Qwen/Qwen2.5-VL-3B-Instruct",
                "[vlmintune] Dataset: lmms-lab/textvqa (train)",
                "[vlmintune] Samples: 100",
                "[vlmintune] Experiment: demo_exp",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    setup_dir = tmp_path / "experiment_setup" / "demo_setup"
    (setup_dir / "train_config.yaml").parent.mkdir(parents=True)
    (setup_dir / "train_config.yaml").write_text("model:\n  model_path: demo\n", encoding="utf-8")
    (setup_dir / "run_train.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    bundle_base_dir = tmp_path / "experiment_results"
    bundle_dir = build_bundle(
        experiment_name="demo_exp",
        experiment_base_dir=str(experiments_dir),
        setup_base_dir=str(tmp_path / "experiment_setup"),
        eval_outputs_dir=str(tmp_path / "eval_outputs"),
        run_logs_dir=str(run_logs_dir),
        bundle_base_dir=str(bundle_base_dir),
        baseline_eval_dir=str(baseline_eval_dir),
        bundle_name="demo_bundle",
        setup_dir=str(setup_dir),
    )
    assert bundle_dir == bundle_base_dir / "demo_bundle"

    assert (bundle_dir / "experiment_setup" / "train_config.yaml").is_file()
    assert (bundle_dir / "experiment_setup" / "run_train.sh").is_file()
    assert (bundle_dir / "baseline_eval" / "eval_summary.json").is_file()
    assert (bundle_dir / "train.log").is_file()

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_name"] == "demo_exp"
    assert manifest["experiment_source_dir"] == str(experiment_dir)
    assert manifest["setup_source_dir"] == str(setup_dir)
    assert manifest["baseline_eval_source_dir"] == str(baseline_eval_dir)
    assert manifest["train_log_source"] == str(run_logs_dir / "demo_exp.log")
    assert manifest["copied_entries"]["experiment_setup"] == str(bundle_dir / "experiment_setup")
    assert "experiment_setup/train_config.yaml" in manifest["files"]
    assert "baseline_eval/eval_summary.json" in manifest["files"]
    assert "train.log" in manifest["files"]

    train_log = (bundle_dir / "train.log").read_text(encoding="utf-8")
    assert "Starting JarvisLabs LoRA run" in train_log
    assert "Experiment: demo_exp" in train_log
