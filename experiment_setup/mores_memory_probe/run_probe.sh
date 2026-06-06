#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_KEY="${1:-}"
DATASET_KEY="${2:-}"
MAX_SAMPLES="${3:-}"
MAX_LENGTH="${4:-}"

if [[ -z "$MODEL_KEY" || -z "$DATASET_KEY" || -z "$MAX_SAMPLES" || -z "$MAX_LENGTH" ]]; then
  echo "Usage: bash $0 <qwen|llava> <textvqa|vizwiz|vqav2|gqa> <max_samples|0> <max_length>" >&2
  exit 2
fi

case "$MODEL_KEY" in
  qwen)
    MODEL_NAME="qwen25vl_3b_instruct"
    MODEL_SLUG="qwen25vl3b"
    ;;
  llava)
    MODEL_NAME="llava15_7b"
    MODEL_SLUG="llava15_7b"
    ;;
  *)
    echo "Unsupported model key: $MODEL_KEY" >&2
    exit 2
    ;;
esac

case "$DATASET_KEY" in
  textvqa)
    DATASET_NAME="lmms-lab/textvqa"
    ;;
  vizwiz)
    DATASET_NAME="HuggingFaceM4/VizWiz"
    ;;
  vqav2)
    DATASET_NAME="pingzhili/vqa_v2"
    ;;
  gqa)
    DATASET_NAME="Mineru/GQA"
    ;;
  *)
    echo "Unsupported dataset key: $DATASET_KEY" >&2
    exit 2
    ;;
esac

if [[ "$MAX_SAMPLES" == "0" ]]; then
  SAMPLE_SLUG="full"
else
  SAMPLE_SLUG="$MAX_SAMPLES"
fi

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="memprobe_${MODEL_SLUG}_mores_${DATASET_KEY}_${SAMPLE_SLUG}_len${MAX_LENGTH}_${RUN_STAMP}"
GENERATED_DIR="$SETUP_DIR/generated/$EXP_NAME"
CONFIG_PATH="$GENERATED_DIR/train_config.yaml"
LOG_DIR="$ROOT_DIR/memory_probe_logs"
CSV_PATH="$LOG_DIR/${EXP_NAME}_nvidia_smi.csv"
SUMMARY_PATH="$LOG_DIR/${EXP_NAME}_summary.txt"

mkdir -p "$GENERATED_DIR" "$LOG_DIR"

python - "$CONFIG_PATH" "$MODEL_NAME" "$EXP_NAME" "$MAX_LENGTH" "$DATASET_NAME" "$MAX_SAMPLES" <<'PY'
from pathlib import Path
import sys

path, model_name, exp_name, max_length, dataset_name, max_samples = sys.argv[1:]
Path(path).write_text(
    f"""model:
  name: "{model_name}"

experiment:
  name: "{exp_name}"
  base_dir: "experiments"

training:
  ft_method: mores
  num_epochs: 1
  per_device_batch_size: 1
  gradient_accumulation_steps: 4
  max_length: {int(max_length)}
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  dataloader_persistent_workers: true
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  save_steps: 0
  output_dir: "experiments"

data:
  dataset_name: "{dataset_name}"
  max_samples: {int(max_samples)}
""",
    encoding="utf-8",
)
PY

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

echo "=== MoReS memory probe ==="
echo "experiment=$EXP_NAME"
echo "config=$CONFIG_PATH"
echo "model=$MODEL_NAME"
echo "dataset=$DATASET_NAME"
echo "max_samples=$MAX_SAMPLES"
echo "max_length=$MAX_LENGTH"
echo "csv=$CSV_PATH"

nvidia-smi \
  --query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu \
  --format=csv \
  -l 1 > "$CSV_PATH" &
MONITOR_PID=$!

cleanup_monitor() {
  if kill -0 "$MONITOR_PID" >/dev/null 2>&1; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_monitor EXIT

set +e
python -m vlmintune.training "${HF_TOKEN_ARGS[@]}" --config "$CONFIG_PATH"
TRAIN_EXIT=$?
set -e

cleanup_monitor
trap - EXIT

python - "$CSV_PATH" "$SUMMARY_PATH" "$EXP_NAME" "$TRAIN_EXIT" <<'PY'
import csv
import json
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
exp_name = sys.argv[3]
train_exit = int(sys.argv[4])

peak_mib = None
gpu_name = ""
total_mib = None

if csv_path.exists() and csv_path.stat().st_size > 0:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    for row in rows:
        used_key = next((key for key in row if "memory.used" in key), None)
        total_key = next((key for key in row if "memory.total" in key), None)
        name_key = next((key for key in row if key.strip() == "name"), None)
        if used_key is None:
            continue
        used = int(str(row[used_key]).split()[0])
        peak_mib = used if peak_mib is None else max(peak_mib, used)
        if total_key is not None:
            total_mib = int(str(row[total_key]).split()[0])
        if name_key is not None:
            gpu_name = str(row[name_key]).strip()

lines = [
    f"experiment={exp_name}",
    f"train_exit_code={train_exit}",
    f"gpu_name={gpu_name}",
]
if total_mib is not None:
    lines.append(f"gpu_total_gib={total_mib / 1024:.2f}")
if peak_mib is not None:
    lines.append(f"peak_memory_used_mib={peak_mib}")
    lines.append(f"peak_memory_used_gib={peak_mib / 1024:.2f}")
else:
    lines.append("peak_memory_used_mib=UNKNOWN")
    lines.append("peak_memory_used_gib=UNKNOWN")

run_log_path = Path("experiments") / exp_name / "train" / "run.log"
train_summary_path = Path("experiments") / exp_name / "train" / "train_summary.json"

first_batch_shapes = ""
if run_log_path.exists():
    for raw_line in run_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "First batch shapes" not in raw_line:
            continue
        first_batch_shapes = raw_line
        try:
            payload = json.loads(raw_line)
            first_batch_shapes = payload.get("data", {}).get("message", raw_line)
        except Exception:
            pass
        break
if first_batch_shapes:
    lines.append(f"first_batch_shapes={first_batch_shapes}")

if train_summary_path.exists():
    try:
        summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
        result = summary.get("result", {})
        lines.append(f"train_status={result.get('status', '')}")
        if "total_steps" in result:
            lines.append(f"total_steps={result['total_steps']}")
        if "train_time_s" in result:
            lines.append(f"train_time_s={result['train_time_s']}")
        if "trainable_params" in result:
            lines.append(f"trainable_params={result['trainable_params']}")
        if "total_params" in result:
            lines.append(f"total_params={result['total_params']}")
    except Exception as exc:
        lines.append(f"train_summary_parse_error={exc}")

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"summary={summary_path}")
PY

echo "run_log=experiments/${EXP_NAME}/train/run.log"
echo "train_summary=experiments/${EXP_NAME}/train/train_summary.json"

exit "$TRAIN_EXIT"
