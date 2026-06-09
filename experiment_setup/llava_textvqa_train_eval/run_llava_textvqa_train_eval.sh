#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METHODS="${METHODS:-mores}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
export DATALOADER_NUM_WORKERS DATALOADER_PIN_MEMORY DATALOADER_PERSISTENT_WORKERS

GENERATED_DIR="$SETUP_DIR/generated/$RUN_STAMP"
SUMMARY_PATH="$SETUP_DIR/llava_textvqa_train_eval_${RUN_STAMP}_summary.tsv"
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

sample_slug() {
  if [[ "$1" == "0" ]]; then
    echo "full"
  else
    echo "$1"
  fi
}

write_train_config() {
  local path="$1"
  local method="$2"
  local exp_name="$3"

  python - "$path" "$method" "$exp_name" "$TRAIN_MAX_SAMPLES" "$MAX_LENGTH" <<'PY'
from pathlib import Path
import os
import sys

path, method, exp_name, train_max_samples, max_length = sys.argv[1:]

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
  dataloader_num_workers: {int(os.environ.get("DATALOADER_NUM_WORKERS", "0"))}
  dataloader_pin_memory: {os.environ.get("DATALOADER_PIN_MEMORY", "false").lower()}
  dataloader_persistent_workers: {os.environ.get("DATALOADER_PERSISTENT_WORKERS", "false").lower()}
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  save_steps: 0
  output_dir: "experiments"
{params}
data:
  dataset_name: "lmms-lab/textvqa"
  max_samples: {int(train_max_samples)}
""",
    encoding="utf-8",
)
PY
}

write_eval_config() {
  local path="$1"
  local exp_name="$2"

  python - "$path" "$exp_name" "$EVAL_MAX_SAMPLES" "$MAX_NEW_TOKENS" <<'PY'
from pathlib import Path
import sys

path, exp_name, eval_max_samples, max_new_tokens = sys.argv[1:]

Path(path).write_text(
    f"""model:
  name: "llava15_7b"

experiment:
  name: "{exp_name}"
  base_dir: "experiments"

eval:
  source: "trained"
  dataset_name: "lmms-lab/textvqa"
  max_samples: {int(eval_max_samples)}
  max_new_tokens: {int(max_new_tokens)}
  temperature: 0.0
""",
    encoding="utf-8",
)
PY
}

printf "method\texperiment\ttrain_exit\teval_exit\ttrain_status\tavg_loss\ttrain_steps\ttrain_time_s\teval_predictions\tvqa_accuracy\n" > "$SUMMARY_PATH"

for method in $METHODS; do
  train_slug="$(sample_slug "$TRAIN_MAX_SAMPLES")"
  eval_slug="$(sample_slug "$EVAL_MAX_SAMPLES")"
  exp_name="llava15_7b_${method}_textvqa_train${train_slug}_eval${eval_slug}_len${MAX_LENGTH}_${RUN_STAMP}"
  train_config="$GENERATED_DIR/${exp_name}_train.yaml"
  eval_config="$GENERATED_DIR/${exp_name}_eval.yaml"
  write_train_config "$train_config" "$method" "$exp_name"
  write_eval_config "$eval_config" "$exp_name"

  echo
  echo "### Training method=$method dataset=textvqa train_samples=$TRAIN_MAX_SAMPLES max_length=$MAX_LENGTH"
  set +e
  VLMINTUNE_FAST_EXIT="${VLMINTUNE_FAST_EXIT:-1}" \
    python -m vlmintune.training "${HF_TOKEN_ARGS[@]}" --config "$train_config"
  train_exit=$?
  set -e

  eval_exit=""
  if [[ "$train_exit" == "0" ]]; then
    echo
    echo "### Evaluating method=$method dataset=textvqa eval_samples=$EVAL_MAX_SAMPLES"
    set +e
    VLMINTUNE_FAST_EXIT="${VLMINTUNE_FAST_EXIT:-1}" \
      python -m vlmintune.eval "${HF_TOKEN_ARGS[@]}" --config "$eval_config"
    eval_exit=$?
    set -e
  else
    eval_exit="skipped"
  fi

  python - "$SUMMARY_PATH" "$method" "$exp_name" "$train_exit" "$eval_exit" <<'PY'
import json
import sys
from pathlib import Path

summary_path, method, exp_name, train_exit, eval_exit = sys.argv[1:]
train_path = Path("experiments") / exp_name / "train" / "train_summary.json"
eval_path = Path("experiments") / exp_name / "eval_trained" / "eval.json"

train_result = {}
if train_path.exists():
    train_result = (json.loads(train_path.read_text(encoding="utf-8")).get("result") or {})

eval_result = {}
if eval_path.exists():
    raw_eval = json.loads(eval_path.read_text(encoding="utf-8"))
    eval_result = {
        "num_predictions": raw_eval.get("num_predictions", ""),
        "vqa_accuracy": (raw_eval.get("metrics") or {}).get("vqa_accuracy", ""),
    }

row = [
    method,
    exp_name,
    str(train_exit),
    str(eval_exit),
    str(train_result.get("status", "")),
    str(train_result.get("avg_loss", "")),
    str(train_result.get("total_steps", "")),
    str(train_result.get("train_time_s", "")),
    str(eval_result.get("num_predictions", "")),
    str(eval_result.get("vqa_accuracy", "")),
]
with Path(summary_path).open("a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
PY

  if [[ "$train_exit" != "0" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    echo "Stopping after train failure because CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR" >&2
    exit "$train_exit"
  fi
  if [[ "$eval_exit" != "0" && "$eval_exit" != "skipped" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    echo "Stopping after eval failure because CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR" >&2
    exit "$eval_exit"
  fi
done

echo
echo "=== LLaVA TextVQA train+eval summary ==="
cat "$SUMMARY_PATH"
