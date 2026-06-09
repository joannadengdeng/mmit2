#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METHODS="${METHODS:-qlora lora dora freeze l2t mores}"
DATASETS="${DATASETS:-textvqa vizwiz vqav2 gqa}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export DATALOADER_NUM_WORKERS DATALOADER_PIN_MEMORY DATALOADER_PERSISTENT_WORKERS

GENERATED_DIR="$SETUP_DIR/generated/$RUN_STAMP"
SUMMARY_PATH="$SETUP_DIR/llava_method_matrix_${RUN_STAMP}_summary.tsv"
mkdir -p "$GENERATED_DIR"

cd "$ROOT_DIR"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

HF_TOKEN_ARGS=()
if [[ -n "${HF_TOKEN_FILE:-}" ]]; then
  HF_TOKEN_ARGS=(--hf-token-file "$HF_TOKEN_FILE")
elif [[ -s "$ROOT_DIR/.hf_token" ]]; then
  HF_TOKEN_ARGS=(--hf-token-file "$ROOT_DIR/.hf_token")
elif [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARGS=(--hf-token "$HF_TOKEN")
fi

dataset_name() {
  case "$1" in
    textvqa) echo "lmms-lab/textvqa" ;;
    vizwiz) echo "HuggingFaceM4/VizWiz" ;;
    vqav2) echo "pingzhili/vqa_v2" ;;
    gqa) echo "Mineru/GQA" ;;
    *)
      echo "Unsupported dataset key: $1" >&2
      return 2
      ;;
  esac
}

write_config() {
  local path="$1"
  local method="$2"
  local dataset_key="$3"
  local dataset="$4"
  local exp_name="$5"

  python - "$path" "$method" "$dataset_key" "$dataset" "$exp_name" "$MAX_SAMPLES" "$MAX_LENGTH" <<'PY'
from pathlib import Path
import sys

path, method, dataset_key, dataset, exp_name, max_samples, max_length = sys.argv[1:]

params = ""
if method in {"qlora", "lora", "dora", "l2t"}:
    params = """  params:
    lora_r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    target_modules: ["q_proj", "v_proj"]
"""
elif method == "freeze":
    params = """  params:
    unfreeze_modules: ["model.language_model.layers.31"]
"""
elif method == "mores":
    params = ""
else:
    raise SystemExit(f"Unsupported method: {method}")

Path(path).write_text(
    f"""model:
  name: "llava15_7b"

experiment:
  name: "{exp_name}"
  base_dir: "experiments"

training:
  ft_method: {method}
  num_epochs: 1
  per_device_batch_size: 1
  gradient_accumulation_steps: 4
  max_length: {int(max_length)}
  dataloader_num_workers: {int(__import__("os").environ.get("DATALOADER_NUM_WORKERS", "0"))}
  dataloader_pin_memory: {__import__("os").environ.get("DATALOADER_PIN_MEMORY", "false").lower()}
  dataloader_persistent_workers: {__import__("os").environ.get("DATALOADER_PERSISTENT_WORKERS", "false").lower()}
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  save_steps: 0
  output_dir: "experiments"
{params}
data:
  dataset_name: "{dataset}"
  max_samples: {int(max_samples)}
""",
    encoding="utf-8",
)
PY
}

printf "method\tdataset\texperiment\texit_code\tstatus\tavg_loss\tsteps\ttrain_time_s\n" > "$SUMMARY_PATH"

for method in $METHODS; do
  for dataset_key in $DATASETS; do
    dataset="$(dataset_name "$dataset_key")"
    exp_name="llava15_7b_${method}_${dataset_key}_${MAX_SAMPLES}_len${MAX_LENGTH}_${RUN_STAMP}"
    config_path="$GENERATED_DIR/${exp_name}.yaml"
    write_config "$config_path" "$method" "$dataset_key" "$dataset" "$exp_name"

    echo
    echo "### Running method=$method dataset=$dataset_key samples=$MAX_SAMPLES max_length=$MAX_LENGTH"
    set +e
    VLMINTUNE_FAST_EXIT="${VLMINTUNE_FAST_EXIT:-1}" \
      python -m vlmintune.training "${HF_TOKEN_ARGS[@]}" --config "$config_path"
    exit_code=$?
    set -e

    train_summary="experiments/$exp_name/train/train_summary.json"
    if [[ -f "$train_summary" ]]; then
      python - "$SUMMARY_PATH" "$method" "$dataset_key" "$exp_name" "$exit_code" "$train_summary" <<'PY'
import json
import sys
from pathlib import Path

summary_path, method, dataset, exp_name, exit_code, train_summary = sys.argv[1:]
data = json.loads(Path(train_summary).read_text(encoding="utf-8"))
result = data.get("result", {})
row = [
    method,
    dataset,
    exp_name,
    str(exit_code),
    str(result.get("status", "")),
    str(result.get("avg_loss", "")),
    str(result.get("total_steps", "")),
    str(result.get("train_time_s", "")),
]
with Path(summary_path).open("a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
PY
    else
      printf "%s\t%s\t%s\t%s\tmissing_summary\t\t\t\n" \
        "$method" "$dataset_key" "$exp_name" "$exit_code" >> "$SUMMARY_PATH"
    fi

    if [[ "$exit_code" != "0" && "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "Stopping after failure because CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR" >&2
      exit "$exit_code"
    fi
  done
done

echo
echo "=== LLaVA method matrix summary ==="
cat "$SUMMARY_PATH"
