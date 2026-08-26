#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-/root/autodl-tmp/venvs/vlmintune-torch211}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/root/autodl-tmp/hf_cache}"
STAGE_SAMPLES="${STAGE_SAMPLES:-8}"
EVAL_SAMPLES="${EVAL_SAMPLES:-$STAGE_SAMPLES}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
METHODS="${METHODS:-lora mores reft dora vl_adapter qlora l2t}"
RUN_PREFIX="${RUN_PREFIX:-qwen_textvqa}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$ROOT_DIR/experiments}"
FORCE="${FORCE:-0}"

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
  echo "Link the legacy cache into HF_HUB_CACHE before running this script." >&2
  exit 2
fi

mkdir -p "$EXPERIMENTS_DIR"
read -r -a method_list <<< "$METHODS"

checkpoint_marker() {
  local method="$1"
  case "$method" in
    lora|qlora|dora|mores_lora) echo "adapter_config.json" ;;
    mores) echo "mores_tuned.pt" ;;
    reft) echo "reft_tuned.pt" ;;
    vl_adapter) echo "vl_adapter_tuned.pt" ;;
    l2t) echo "l2t_tuned.pt" ;;
    *)
      echo "Unsupported method in METHODS: $method" >&2
      return 2
      ;;
  esac
}

checkpoint_files_exist() {
  local checkpoint_dir="$1"
  local method="$2"
  local marker="$3"

  [[ -s "$checkpoint_dir/$marker" ]] || return 1
  if [[ "$method" == "mores_lora" ]]; then
    [[ -s "$checkpoint_dir/mores_tuned.pt" ]] || return 1
    [[ -s "$checkpoint_dir/adapter_model.safetensors" ]] || return 1
  fi
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

validate_checkpoint() {
  local metadata_path="$1"
  local expected_method="$2"
  python - "$metadata_path" "$expected_method" <<'PY'
import json
import sys

metadata_path, expected_method = sys.argv[1:]
with open(metadata_path, "r", encoding="utf-8") as handle:
    metadata = json.load(handle)
if metadata.get("model_name") != "qwen25vl_3b_instruct":
    raise SystemExit(f"wrong model metadata: {metadata}")
if metadata.get("ft_method") != expected_method:
    raise SystemExit(f"wrong method metadata: {metadata}")
loss = float(metadata["final_loss"])
if not (loss == loss and abs(loss) != float("inf")):
    raise SystemExit(f"non-finite final loss: {loss}")
print(f"checkpoint metadata OK: method={expected_method}, final_loss={loss:.6f}")
PY
}

validate_eval() {
  local summary_path="$1"
  python - "$summary_path" "$EVAL_SAMPLES" <<'PY'
import json
import sys

summary_path, expected_count = sys.argv[1], int(sys.argv[2])
with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)
actual_count = int(summary.get("num_predictions", -1))
if actual_count != expected_count:
    raise SystemExit(f"expected {expected_count} predictions, got {actual_count}")
metrics = summary.get("metrics") or {}
if not metrics:
    raise SystemExit("evaluation produced no metrics")
for metric_name, value in metrics.items():
    numeric_value = float(value)
    if not (numeric_value == numeric_value and abs(numeric_value) != float("inf")):
        raise SystemExit(f"non-finite metric {metric_name}={value}")
print(f"evaluation OK: predictions={actual_count}, metrics={metrics}")
PY
}

for method in "${method_list[@]}"; do
  marker="$(checkpoint_marker "$method")"
  learning_rate="$(method_learning_rate "$method")"
  run_name="${RUN_PREFIX}_${method}_n${STAGE_SAMPLES}_s${SEED}"
  experiment_dir="$EXPERIMENTS_DIR/$run_name"
  checkpoint_dir="$experiment_dir/checkpoint"
  train_dir="$experiment_dir/train"
  eval_dir="$experiment_dir/eval_trained"
  metadata_path="$checkpoint_dir/vlmintune_meta.json"
  eval_summary="$eval_dir/eval.json"

  if [[ "$FORCE" != "1" \
        && -f "$metadata_path" \
        && -f "$eval_summary" ]] \
      && checkpoint_files_exist "$checkpoint_dir" "$method" "$marker"; then
    if validate_checkpoint "$metadata_path" "$method" \
        && validate_eval "$eval_summary"; then
      echo "SKIP completed run: $run_name"
      continue
    fi
  fi

  mkdir -p "$train_dir" "$checkpoint_dir"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  train_log="$train_dir/run_${timestamp}.log"

  echo "================================================================================"
  echo "TRAIN $run_name"
  echo "method=$method samples=$STAGE_SAMPLES eval_samples=$EVAL_SAMPLES"
  echo "lr=$learning_rate max_length=$MAX_LENGTH grad_acc=$GRADIENT_ACCUMULATION_STEPS"
  echo "================================================================================"

  MODEL=qwen25vl_3b_instruct \
  DATASET=lmms-lab/textvqa \
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
  CONFIG_PATH="$checkpoint_dir/train_config.yaml" \
  bash experiment_setup/paper_benchmark/run_paper_benchmark.sh \
    2>&1 | tee "$train_log"

  if ! checkpoint_files_exist "$checkpoint_dir" "$method" "$marker" \
      || [[ ! -f "$metadata_path" ]]; then
    echo "Training finished without the expected checkpoint files for $method." >&2
    exit 1
  fi
  validate_checkpoint "$metadata_path" "$method"

  eval_config="$experiment_dir/eval_trained_config.yaml"
  cat > "$eval_config" <<YAML
model:
  name: qwen25vl_3b_instruct
experiment:
  name: "$run_name"
  base_dir: "$EXPERIMENTS_DIR"
eval:
  source: trained
  dataset_name: lmms-lab/textvqa
  split: validation
  max_samples: $EVAL_SAMPLES
  sample_seed: $SEED
  shuffle_buffer_size: 10000
  max_new_tokens: 16
  temperature: 0.0
YAML

  echo "EVAL $run_name"
  python -m vlmintune.eval --config "$eval_config"
  validate_eval "$eval_summary"
  echo "PASS $run_name"
done

echo "All requested methods passed stage n=$STAGE_SAMPLES."
