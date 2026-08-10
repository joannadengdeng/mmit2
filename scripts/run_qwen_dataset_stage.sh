#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-/root/autodl-tmp/venvs/vlmintune-torch211}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/root/autodl-tmp/hf_cache}"
DATASET="${DATASET:-}"
STAGE_SAMPLES="${STAGE_SAMPLES:-8}"
EVAL_SAMPLES="${EVAL_SAMPLES:-$STAGE_SAMPLES}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
METHODS="${METHODS:-lora mores reft dora vl_adapter qlora l2t}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$ROOT_DIR/experiments}"
FORCE="${FORCE:-0}"

case "$DATASET" in
  pingzhili/vqa_v2)
    DATASET_SLUG="vqav2"
    EVAL_SPLIT="validation"
    ;;
  ebrukilic/vizwiz_vqa_dataset)
    DATASET_SLUG="vizwiz"
    EVAL_SPLIT="validation"
    ;;
  Mineru/GQA)
    DATASET_SLUG="gqa"
    EVAL_SPLIT="val_balanced"
    ;;
  scienceqa_image)
    DATASET_SLUG="scienceqa_image"
    EVAL_SPLIT="validation"
    ;;
  lmms-lab/textvqa)
    DATASET_SLUG="textvqa"
    EVAL_SPLIT="validation"
    ;;
  "")
    echo "DATASET is required." >&2
    exit 2
    ;;
  *)
    echo "Unsupported built-in dataset: $DATASET" >&2
    exit 2
    ;;
esac

case "$DATASET_SLUG" in
  vizwiz|vqav2|gqa)
    # These datasets contain phone/original-resolution images.  The suffix
    # keeps capped-image runs distinct from older checkpoints that may have
    # silently skipped samples after visual-token truncation.
    DEFAULT_RUN_PREFIX="qwen_${DATASET_SLUG}_px1003520"
    ;;
  *)
    DEFAULT_RUN_PREFIX="qwen_${DATASET_SLUG}"
    ;;
esac
RUN_PREFIX="${RUN_PREFIX:-$DEFAULT_RUN_PREFIX}"

if [[ "$STAGE_SAMPLES" -le 8 ]]; then
  DEFAULT_GRAD_ACC=1
else
  DEFAULT_GRAD_ACC=4
fi
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRAD_ACC}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing experiment Python: $VENV_DIR/bin/python" >&2
  exit 2
fi

for numeric_value in \
  "$STAGE_SAMPLES" "$EVAL_SAMPLES" "$MAX_LENGTH" "$EPOCHS" \
  "$GRADIENT_ACCUMULATION_STEPS"; do
  if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Stage sizes, max length, epochs, and gradient accumulation must be positive integers." >&2
    exit 2
  fi
done

export PATH="$VENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export HF_XET_CACHE="$HF_CACHE_ROOT/xet"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLMINTUNE_FAST_EXIT=1
export TOKENIZERS_PARALLELISM=false

MODEL_CACHE="$HF_HUB_CACHE/models--Qwen--Qwen2.5-VL-3B-Instruct"
if [[ ! -e "$MODEL_CACHE" ]]; then
  echo "Qwen cache is not visible at $MODEL_CACHE" >&2
  exit 2
fi

mkdir -p "$EXPERIMENTS_DIR"
read -r -a method_list <<< "$METHODS"

checkpoint_marker() {
  local method="$1"
  case "$method" in
    lora|qlora|dora) echo "adapter_config.json" ;;
    mores) echo "mores_tuned.pt" ;;
    reft) echo "reft_tuned.pt" ;;
    vl_adapter) echo "vl_adapter_tuned.pt" ;;
    l2t) echo "l2t_tuned.pt" ;;
    mole)
      echo "MoLE is LLaVA-only and must not be included in this Qwen run." >&2
      return 2
      ;;
    *)
      echo "Unsupported method in METHODS: $method" >&2
      return 2
      ;;
  esac
}

method_learning_rate() {
  local method="$1"
  case "$method" in
    qlora) echo "${LEARNING_RATE_QLORA:-5e-5}" ;;
    vl_adapter) echo "${LEARNING_RATE_VL_ADAPTER:-1e-4}" ;;
    l2t) echo "${LEARNING_RATE_L2T:-2e-5}" ;;
    *) echo "${LEARNING_RATE_DEFAULT:-2e-4}" ;;
  esac
}

validate_dataset_sample() {
  python - "$DATASET" "$EVAL_SPLIT" "$SEED" <<'PY'
import os
import sys

from vlmintune.data.datasets import get_dataset_spec
from vlmintune.data.hf_datasets import HFDatasetsAdapter
from vlmintune.data.datasets.base import load_sample_image

dataset_name, eval_split, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
spec = get_dataset_spec(dataset_name)
if spec is None:
    raise SystemExit(f"missing built-in dataset spec: {dataset_name}")

for usage, split in (("train", None), ("eval", eval_split)):
    adapter = HFDatasetsAdapter(
        dataset_name=dataset_name,
        split=split,
        usage=usage,
        max_samples=1,
        streaming=True,
        sample_seed=seed,
    )
    sample = next(iter(adapter))
    if not sample.question.strip():
        raise SystemExit(f"{usage} sample has an empty question")
    if usage == "train" and not sample.train_answer.strip():
        raise SystemExit("train sample has an empty answer")
    if usage == "eval" and not sample.eval_answers:
        raise SystemExit("eval sample has no ground truth")
    if load_sample_image(sample) is None:
        raise SystemExit(f"{usage} sample has no loadable image: id={sample.id}")
    print(
        f"dataset preflight OK: usage={usage} split={adapter.split} "
        f"id={sample.id} metric={spec.data_model.metric_family}"
    )

# Streaming Parquet readers can leave native worker state alive until Python
# teardown.  The train/eval CLIs already use the same guarded fast exit to
# avoid an intermittent post-success abort in third-party destructors.
sys.stdout.flush()
sys.stderr.flush()
if os.environ.get("VLMINTUNE_FAST_EXIT") == "1":
    os._exit(0)
PY
}

validate_checkpoint() {
  local metadata_path="$1"
  local config_path="$2"
  local expected_method="$3"
  python - \
    "$metadata_path" "$config_path" "$expected_method" "$DATASET" \
    "$STAGE_SAMPLES" "$MAX_LENGTH" "$SEED" "$GRADIENT_ACCUMULATION_STEPS" <<'PY'
import json
import math
import sys

import yaml

(
    metadata_path,
    config_path,
    expected_method,
    expected_dataset,
    expected_samples,
    expected_max_length,
    expected_seed,
    expected_grad_acc,
) = sys.argv[1:]
with open(metadata_path, "r", encoding="utf-8") as handle:
    metadata = json.load(handle)
with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

expected = {
    "model": "qwen25vl_3b_instruct",
    "dataset": expected_dataset,
    "method": expected_method,
    "max_samples": int(expected_samples),
    "max_length": int(expected_max_length),
    "seed": int(expected_seed),
    "gradient_accumulation_steps": int(expected_grad_acc),
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"wrong train config {key}: expected={value!r}, actual={config.get(key)!r}")
if metadata.get("model_name") != "qwen25vl_3b_instruct":
    raise SystemExit(f"wrong model metadata: {metadata}")
if metadata.get("ft_method") != expected_method:
    raise SystemExit(f"wrong method metadata: {metadata}")
loss = float(metadata["final_loss"])
if not math.isfinite(loss):
    raise SystemExit(f"non-finite final loss: {loss}")
print(f"checkpoint metadata OK: method={expected_method}, final_loss={loss:.6f}")
PY
}

validate_eval() {
  local summary_path="$1"
  local predictions_path="$2"
  python - \
    "$summary_path" "$predictions_path" "$DATASET" "$EVAL_SPLIT" \
    "$EVAL_SAMPLES" "$SEED" <<'PY'
import json
import math
import sys

summary_path, predictions_path, dataset, split, expected_count, seed = sys.argv[1:]
expected_count = int(expected_count)
seed = int(seed)
with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("dataset_name") != dataset:
    raise SystemExit(f"wrong eval dataset: {summary.get('dataset_name')!r}")
if summary.get("split") != split:
    raise SystemExit(f"wrong eval split: {summary.get('split')!r}")
if int(summary.get("sample_seed", -1)) != seed:
    raise SystemExit(f"wrong eval seed: {summary.get('sample_seed')!r}")
if int(summary.get("num_predictions", -1)) != expected_count:
    raise SystemExit(
        f"expected {expected_count} predictions, got {summary.get('num_predictions')!r}"
    )
metrics = summary.get("metrics") or {}
if not metrics:
    raise SystemExit("evaluation produced no metrics")
for metric_name, value in metrics.items():
    if not math.isfinite(float(value)):
        raise SystemExit(f"non-finite metric {metric_name}={value}")

line_count = 0
with open(predictions_path, "r", encoding="utf-8") as handle:
    for line in handle:
        json.loads(line)
        line_count += 1
if line_count != expected_count:
    raise SystemExit(f"prediction JSONL has {line_count} rows, expected {expected_count}")
print(f"evaluation OK: predictions={line_count}, metrics={metrics}")
PY
}

validate_training_log() {
  local train_log="$1"
  python - "$train_log" <<'PY'
import json
import sys

path = sys.argv[1]
skip_summaries = []
with open(path, "r", encoding="utf-8", errors="replace") as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = record.get("data") or {}
        if record.get("type") == "debug" and data.get("kind") == "skip_summary":
            skip_summaries.append(data)

if not skip_summaries:
    raise SystemExit(f"training log has no skip_summary: {path}")
skipped = int(skip_summaries[-1].get("total_skipped", -1))
if skipped != 0:
    raise SystemExit(
        f"training skipped {skipped} samples; refusing to accept a partial run: {path}"
    )
print("training data coverage OK: skipped=0")
PY
}

validate_dataset_sample

for method in "${method_list[@]}"; do
  marker="$(checkpoint_marker "$method")"
  learning_rate="$(method_learning_rate "$method")"
  run_name="${RUN_PREFIX}_${method}_n${STAGE_SAMPLES}_s${SEED}"
  experiment_dir="$EXPERIMENTS_DIR/$run_name"
  checkpoint_dir="$experiment_dir/checkpoint"
  train_dir="$experiment_dir/train"
  eval_dir="$experiment_dir/eval_trained"
  metadata_path="$checkpoint_dir/vlmintune_meta.json"
  train_config="$checkpoint_dir/train_config.yaml"
  eval_summary="$eval_dir/eval.json"
  predictions_path="$eval_dir/predictions.jsonl"
  latest_train_log=""
  if [[ -d "$train_dir" ]]; then
    latest_train_log="$(find "$train_dir" -maxdepth 1 -type f -name 'run_*.log' 2>/dev/null | sort | tail -n 1)"
  fi

  checkpoint_ready=0
  if [[ "$FORCE" != "1" ]]; then
    checkpoint_artifact_found=0
    for checkpoint_artifact in \
      "$checkpoint_dir/$marker" "$metadata_path" "$train_config"; do
      if [[ -e "$checkpoint_artifact" ]]; then
        checkpoint_artifact_found=1
      fi
    done

    if [[ -f "$checkpoint_dir/$marker" \
          && -f "$metadata_path" \
          && -f "$train_config" \
          && -n "$latest_train_log" ]]; then
      if validate_checkpoint "$metadata_path" "$train_config" "$method" \
          && validate_training_log "$latest_train_log"; then
        checkpoint_ready=1
      else
        echo "Existing training artifacts failed validation for $run_name; refusing to overwrite them." >&2
        exit 1
      fi

      if [[ -f "$eval_summary" \
            && -f "$predictions_path" ]] \
          && validate_eval "$eval_summary" "$predictions_path"; then
        echo "SKIP completed run: $run_name"
        continue
      fi

      echo "RESUME evaluation from validated checkpoint: $run_name"
    elif [[ "$checkpoint_artifact_found" == "1" \
            || -n "$latest_train_log" ]]; then
      echo "Incomplete training artifacts exist for $run_name; refusing to overwrite them." >&2
      exit 1
    fi
  fi

  if [[ "$checkpoint_ready" != "1" ]]; then
    mkdir -p "$train_dir" "$checkpoint_dir"
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    train_log="$train_dir/run_${timestamp}.log"

    echo "================================================================================"
    echo "TRAIN $run_name"
    echo "dataset=$DATASET method=$method samples=$STAGE_SAMPLES eval_samples=$EVAL_SAMPLES"
    echo "lr=$learning_rate max_length=$MAX_LENGTH grad_acc=$GRADIENT_ACCUMULATION_STEPS"
    echo "================================================================================"

    MODEL=qwen25vl_3b_instruct \
    DATASET="$DATASET" \
    METHOD="$method" \
    MAX_SAMPLES="$STAGE_SAMPLES" \
    EPOCHS="$EPOCHS" \
    LEARNING_RATE="$learning_rate" \
    BATCH_SIZE=1 \
    GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
    MAX_LENGTH="$MAX_LENGTH" \
    SEED="$SEED" \
    RUN_NAME="$run_name" \
    OUTPUT_DIR="$checkpoint_dir" \
    CONFIG_PATH="$train_config" \
    bash experiment_setup/paper_benchmark/run_paper_benchmark.sh \
      2>&1 | tee "$train_log"

    if [[ ! -f "$checkpoint_dir/$marker" \
          || ! -f "$metadata_path" \
          || ! -f "$train_config" ]]; then
      echo "Training finished without the expected checkpoint files for $method." >&2
      exit 1
    fi
    validate_checkpoint "$metadata_path" "$train_config" "$method"
    validate_training_log "$train_log"
  fi

  eval_config="$experiment_dir/eval_trained_config.yaml"
  cat > "$eval_config" <<YAML
model:
  name: qwen25vl_3b_instruct
experiment:
  name: "$run_name"
  base_dir: "$EXPERIMENTS_DIR"
eval:
  source: trained
  dataset_name: "$DATASET"
  split: "$EVAL_SPLIT"
  max_samples: $EVAL_SAMPLES
  sample_seed: $SEED
  shuffle_buffer_size: 10000
  max_new_tokens: 16
  temperature: 0.0
YAML

  echo "EVAL $run_name"
  python -m vlmintune.eval --config "$eval_config"
  validate_eval "$eval_summary" "$predictions_path"
  echo "PASS $run_name"
done

echo "All requested methods passed dataset=$DATASET stage n=$STAGE_SAMPLES."
