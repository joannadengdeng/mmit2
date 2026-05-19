#!/usr/bin/env bash
set -euo pipefail

# Collect one experiment, its checkpoint/debug artifacts, and an optional
# baseline eval into a single bundle directory.
#
# Defaults:
#   - experiment: latest directory under ./experiments
#   - baseline eval: latest directory under ./eval_outputs (if present)
#   - bundle dir: ./result_bundles/<experiment>_bundle_<timestamp>
#
# Common overrides:
#   EXPERIMENT_NAME=20260513_lora_textvqa_3b_full ./scripts/jarvislabs_bundle_results.sh
#   BASELINE_EVAL_DIR=./eval_outputs/Qwen2.5-VL-3B-Instruct_textvqa_20260519_173002 ./scripts/jarvislabs_bundle_results.sh
#   BUNDLE_NAME=my_textvqa_results ./scripts/jarvislabs_bundle_results.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

EXPERIMENT_BASE_DIR="${EXPERIMENT_BASE_DIR:-$ROOT_DIR/experiments}"
EVAL_OUTPUTS_DIR="${EVAL_OUTPUTS_DIR:-$ROOT_DIR/eval_outputs}"
RUN_LOGS_DIR="${RUN_LOGS_DIR:-$ROOT_DIR/run_logs}"
BUNDLE_BASE_DIR="${BUNDLE_BASE_DIR:-$ROOT_DIR/result_bundles}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
BASELINE_EVAL_DIR="${BASELINE_EVAL_DIR:-}"
BUNDLE_NAME="${BUNDLE_NAME:-}"

if [[ -z "$EXPERIMENT_NAME" ]]; then
  if [[ ! -d "$EXPERIMENT_BASE_DIR" ]]; then
    echo "[mmit2] No experiments directory found at $EXPERIMENT_BASE_DIR" >&2
    exit 1
  fi
  EXPERIMENT_NAME="$(basename "$(ls -td "$EXPERIMENT_BASE_DIR"/* 2>/dev/null | head -n 1)")"
fi

if [[ -z "$EXPERIMENT_NAME" ]]; then
  echo "[mmit2] Could not determine experiment name. Set EXPERIMENT_NAME explicitly." >&2
  exit 1
fi

EXPERIMENT_DIR="$EXPERIMENT_BASE_DIR/$EXPERIMENT_NAME"
SUMMARY_PATH="$EXPERIMENT_DIR/summary.json"

if [[ ! -f "$SUMMARY_PATH" ]]; then
  echo "[mmit2] Experiment summary not found: $SUMMARY_PATH" >&2
  exit 1
fi

if [[ -z "$BASELINE_EVAL_DIR" && -d "$EVAL_OUTPUTS_DIR" ]]; then
  BASELINE_EVAL_DIR="$(ls -td "$EVAL_OUTPUTS_DIR"/* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$BUNDLE_NAME" ]]; then
  BUNDLE_NAME="${EXPERIMENT_NAME}_bundle_$(date +%Y%m%d_%H%M%S)"
fi

BUNDLE_DIR="$BUNDLE_BASE_DIR/$BUNDLE_NAME"
if [[ -e "$BUNDLE_DIR" ]]; then
  echo "[mmit2] Bundle directory already exists: $BUNDLE_DIR" >&2
  echo "[mmit2] Set BUNDLE_NAME explicitly or remove the existing directory." >&2
  exit 1
fi

mkdir -p "$BUNDLE_DIR"

cp -R "$EXPERIMENT_DIR" "$BUNDLE_DIR/experiment"

RUN_LOG_PATH="$RUN_LOGS_DIR/$EXPERIMENT_NAME.log"
if [[ -f "$RUN_LOG_PATH" ]]; then
  cp "$RUN_LOG_PATH" "$BUNDLE_DIR/train.log"
fi

BASELINE_SUMMARY_PATH=""
if [[ -n "$BASELINE_EVAL_DIR" ]]; then
  if [[ ! -d "$BASELINE_EVAL_DIR" ]]; then
    echo "[mmit2] Baseline eval directory not found: $BASELINE_EVAL_DIR" >&2
    exit 1
  fi
  cp -R "$BASELINE_EVAL_DIR" "$BUNDLE_DIR/baseline_eval"
  BASELINE_SUMMARY_PATH="$BASELINE_EVAL_DIR/eval_summary.json"
fi

export EXPERIMENT_NAME
export EXPERIMENT_DIR
export SUMMARY_PATH
export BUNDLE_DIR
export BASELINE_EVAL_DIR
export BASELINE_SUMMARY_PATH

python3 - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path


def load_json(path: str):
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_primary_metric(summary: dict | None):
    if not summary:
        return "", None
    eval_result = summary.get("eval_result", {}) or {}
    metrics = eval_result.get("metrics", {}) or {}
    primary = str(eval_result.get("primary_metric", "")).strip()
    if primary and primary in metrics:
        return primary, metrics[primary]
    if len(metrics) == 1:
        metric_name, metric_value = next(iter(metrics.items()))
        return metric_name, metric_value
    return primary, None


bundle_dir = Path(os.environ["BUNDLE_DIR"])
experiment_dir = Path(os.environ["EXPERIMENT_DIR"])
experiment_summary = load_json(os.environ["SUMMARY_PATH"]) or {}
experiment_eval_summary = load_json(str(experiment_dir / "eval_summary.json"))
baseline_dir = os.environ.get("BASELINE_EVAL_DIR", "").strip()
baseline_eval_summary = load_json(os.environ.get("BASELINE_SUMMARY_PATH", "").strip()) if baseline_dir else None

training_config = experiment_summary.get("config", {}) or {}
train_summary = experiment_summary.get("train_summary", {}) or {}
overview = {
    "experiment_name": experiment_summary.get("exp_id", experiment_dir.name),
    "status": experiment_summary.get("status", ""),
    "method": experiment_summary.get("method", ""),
    "model": experiment_summary.get("model", ""),
    "dataset": experiment_summary.get("dataset", ""),
    "num_samples": experiment_summary.get("num_samples", 0),
    "checkpoint_path": experiment_summary.get("checkpoint_path", ""),
    "created_at": experiment_summary.get("created_at", ""),
    "completed_at": experiment_summary.get("completed_at", ""),
}

write_json(bundle_dir / "experiment_summary.json", experiment_summary)
write_json(bundle_dir / "training_config.json", training_config)
write_json(bundle_dir / "train_summary.json", train_summary)
write_json(bundle_dir / "experiment_overview.json", overview)

if experiment_eval_summary:
    write_json(bundle_dir / "experiment_eval_summary.json", experiment_eval_summary)
if baseline_eval_summary:
    write_json(bundle_dir / "baseline_eval_summary.json", baseline_eval_summary)

exp_metric_name, exp_metric_value = extract_primary_metric(experiment_eval_summary)
base_metric_name, base_metric_value = extract_primary_metric(baseline_eval_summary)
comparison = {
    "created_at": datetime.now().isoformat(),
    "experiment_name": overview["experiment_name"],
    "experiment_dir": str(experiment_dir),
    "baseline_eval_dir": baseline_dir,
    "experiment_metric": {
        "name": exp_metric_name,
        "value": exp_metric_value,
    },
    "baseline_metric": {
        "name": base_metric_name,
        "value": base_metric_value,
    },
    "delta": None,
}

if (
    exp_metric_name
    and base_metric_name
    and exp_metric_name == base_metric_name
    and exp_metric_value is not None
    and base_metric_value is not None
):
    comparison["delta"] = round(float(exp_metric_value) - float(base_metric_value), 2)

write_json(bundle_dir / "comparison.json", comparison)

lines = [
    f"# Results Bundle: {overview['experiment_name']}",
    "",
    "## Experiment",
    f"- Method: {overview['method']}",
    f"- Model: {overview['model']}",
    f"- Dataset: {overview['dataset']}",
    f"- Samples: {overview['num_samples']}",
    f"- Checkpoint: {overview['checkpoint_path']}",
]

if exp_metric_name and exp_metric_value is not None:
    lines.extend(
        [
            "",
            "## Fine-tuned Eval",
            f"- {exp_metric_name}: {exp_metric_value}",
        ]
    )

if base_metric_name and base_metric_value is not None:
    lines.extend(
        [
            "",
            "## Baseline Eval",
            f"- {base_metric_name}: {base_metric_value}",
        ]
    )

if comparison["delta"] is not None:
    lines.extend(
        [
            "",
            "## Comparison",
            f"- Delta ({exp_metric_name}): {comparison['delta']}",
        ]
    )

lines.extend(
    [
        "",
        "## Included Files",
        "- experiment/ : full experiment snapshot including checkpoint, debug artifacts, and eval outputs",
        "- train.log : training log when available",
        "- baseline_eval/ : copied baseline eval output when available",
        "- training_config.json : config snapshot used for training",
        "- train_summary.json : persisted train summary",
        "- comparison.json : machine-readable fine-tuned vs baseline comparison",
    ]
)

(bundle_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

manifest = {
    "bundle_dir": str(bundle_dir),
    "experiment_name": overview["experiment_name"],
    "experiment_source_dir": str(experiment_dir),
    "baseline_source_dir": baseline_dir,
    "files": sorted(p.name for p in bundle_dir.iterdir()),
}
write_json(bundle_dir / "manifest.json", manifest)
PY

echo "[mmit2] Created results bundle: $BUNDLE_DIR"
echo "[mmit2] Experiment snapshot: $BUNDLE_DIR/experiment"
if [[ -n "$BASELINE_EVAL_DIR" ]]; then
  echo "[mmit2] Baseline snapshot: $BUNDLE_DIR/baseline_eval"
fi
echo "[mmit2] Training config: $BUNDLE_DIR/training_config.json"
echo "[mmit2] Train summary: $BUNDLE_DIR/train_summary.json"
echo "[mmit2] Comparison: $BUNDLE_DIR/comparison.json"
