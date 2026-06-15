#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS="${MODELS:-qwen25vl_3b_instruct llava15_7b}"
DATASETS="${DATASETS:-textvqa vqav2 vizwiz gqa scienceqa}"
METHODS="${METHODS:-base qlora lora dora freeze l2t mole reft mores lora_layer}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
QWEN_VIZWIZ_MAX_LENGTH="${QWEN_VIZWIZ_MAX_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-20260611}"
EVAL_SHUFFLE_BUFFER_SIZE="${EVAL_SHUFFLE_BUFFER_SIZE:-10000}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
export DATALOADER_NUM_WORKERS DATALOADER_PIN_MEMORY DATALOADER_PERSISTENT_WORKERS

REFT_LAYERS="${REFT_LAYERS:-16 24 31}"
VISNEC_SCORE_FILE="${VISNEC_SCORE_FILE:-}"
VISNEC_TOP_RATIO="${VISNEC_TOP_RATIO:-1.0}"
FREEZE_UNFREEZE_MODULES="${FREEZE_UNFREEZE_MODULES:-}"
FREEZE_LAYERS="${FREEZE_LAYERS:-31}"
FREEZE_LAYER_RANGE="${FREEZE_LAYER_RANGE:-}"
LORA_TRAIN_LAYER_RANGE="${LORA_TRAIN_LAYER_RANGE:-}"
QLORA_TRAIN_LAYER_RANGE="${QLORA_TRAIN_LAYER_RANGE:-}"
LORA_LAYER_TRAIN_LAYER_RANGE="${LORA_LAYER_TRAIN_LAYER_RANGE:-16:31}"
LORA_VISNEC_SCORE_FILE="${LORA_VISNEC_SCORE_FILE:-}"
LORA_VISNEC_TOP_RATIO="${LORA_VISNEC_TOP_RATIO:-0.3}"
export REFT_LAYERS VISNEC_SCORE_FILE VISNEC_TOP_RATIO
export FREEZE_UNFREEZE_MODULES FREEZE_LAYERS FREEZE_LAYER_RANGE
export LORA_TRAIN_LAYER_RANGE QLORA_TRAIN_LAYER_RANGE LORA_LAYER_TRAIN_LAYER_RANGE
export LORA_VISNEC_SCORE_FILE LORA_VISNEC_TOP_RATIO

GENERATED_DIR="$SETUP_DIR/generated/$RUN_STAMP"
SUMMARY_PATH="$SETUP_DIR/paper_benchmark_${RUN_STAMP}_summary.csv"
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

dataset_id() {
  case "$1" in
    textvqa) echo "lmms-lab/textvqa" ;;
    vqav2) echo "pingzhili/vqa_v2" ;;
    vizwiz) echo "ebrukilic/vizwiz_vqa_dataset" ;;
    gqa) echo "Mineru/GQA" ;;
    scienceqa|scienceqa_image) echo "scienceqa_image" ;;
    *) echo "Unknown dataset key: $1" >&2; exit 1 ;;
  esac
}

sample_slug() {
  if [[ "$1" == "0" ]]; then
    echo "full"
  else
    echo "$1"
  fi
}

train_max_length_for() {
  local model="$1"
  local dataset_key="$2"
  if [[ "$model" == "qwen25vl_3b_instruct" && "$dataset_key" == "vizwiz" ]]; then
    echo "$QWEN_VIZWIZ_MAX_LENGTH"
  else
    echo "$MAX_LENGTH"
  fi
}

condition_slug() {
  local slug=""
  if [[ -n "$VISNEC_SCORE_FILE" ]]; then
    slug="${slug}_visnec$(python - <<'PY'
import os
ratio = float(os.environ.get("VISNEC_TOP_RATIO", "1.0") or 1.0)
text = f"{ratio:g}".replace(".", "p")
print(text)
PY
)"
  fi
  echo "$slug"
}

write_train_config() {
  local path="$1"
  local model="$2"
  local dataset="$3"
  local method="$4"
  local exp_name="$5"
  local max_length="$6"

  python - "$path" "$model" "$dataset" "$method" "$exp_name" "$TRAIN_MAX_SAMPLES" "$max_length" <<'PY'
from pathlib import Path
import os
import sys

path, model, dataset, method, exp_name, train_max_samples, max_length = sys.argv[1:]
visnec_score_file = os.environ.get("VISNEC_SCORE_FILE", "")
visnec_top_ratio = float(os.environ.get("VISNEC_TOP_RATIO", "1.0") or 1.0)
method_slug = method
train_method = {
    "lora_layer": "lora",
    "lora_visnec": "lora",
}.get(method_slug, method_slug)
method_visnec_score_file = visnec_score_file
method_visnec_top_ratio = visnec_top_ratio
if method_slug == "lora_visnec":
    method_visnec_score_file = os.environ.get("LORA_VISNEC_SCORE_FILE", "") or visnec_score_file
    method_visnec_top_ratio = float(
        os.environ.get(
            "LORA_VISNEC_TOP_RATIO",
            os.environ.get("VISNEC_TOP_RATIO", "0.3"),
        )
        or 0.3
    )
    if not method_visnec_score_file:
        raise SystemExit(
            "lora_visnec requires LORA_VISNEC_SCORE_FILE or VISNEC_SCORE_FILE."
        )

def parse_int_list(raw: str) -> list[int]:
    return [
        int(item)
        for item in raw.replace(",", " ").split()
        if item.strip()
    ]

def parse_inclusive_range(raw: str) -> list[int]:
    text = raw.strip().strip("[]")
    if not text:
        return []
    if ":" in text:
        start_text, end_text = text.split(":", 1)
    elif "-" in text:
        start_text, end_text = text.split("-", 1)
    else:
        values = parse_int_list(text)
        if len(values) == 1:
            return values
        if len(values) == 2:
            start_text, end_text = str(values[0]), str(values[1])
        else:
            return values
    start = int(start_text.strip())
    end = int(end_text.strip())
    if end < start:
        raise SystemExit(f"Invalid inclusive layer range: {raw!r}")
    return list(range(start, end + 1))

def layer_range_pair(raw: str) -> list[int]:
    values = parse_inclusive_range(raw)
    if not values:
        return []
    return [values[0], values[-1]]

def transformer_layer_prefix(model_name: str) -> str:
    # Current built-in specs use the same language-layer path for Qwen2.5-VL and LLaVA.
    # Keeping this in generated YAML avoids importing model code in the setup script.
    if model_name in {"qwen25vl_3b_instruct", "llava15_7b"}:
        return "model.language_model.layers"
    raise SystemExit(f"Unsupported model for layer freeze: {model_name}")

learning_rate = "2.0e-4"
max_grad_norm = "1.0"
params = ""
if train_method == "qlora":
    learning_rate = os.environ.get("QLORA_LEARNING_RATE", "2.0e-4")
    max_grad_norm = os.environ.get("QLORA_MAX_GRAD_NORM", "1.0")
    train_layer_range = layer_range_pair(os.environ.get("QLORA_TRAIN_LAYER_RANGE", ""))
    layer_line = f"    train_layer_range: {train_layer_range}\n" if train_layer_range else ""
    params = """  params:
    lora_r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    target_modules: ["q_proj", "v_proj"]
""" + layer_line
elif train_method in {"lora", "dora", "l2t"}:
    if method_slug == "lora_layer":
        raw_layer_range = os.environ.get("LORA_LAYER_TRAIN_LAYER_RANGE", "24:31")
    else:
        raw_layer_range = os.environ.get("LORA_TRAIN_LAYER_RANGE", "")
    train_layer_range = layer_range_pair(raw_layer_range)
    layer_line = f"    train_layer_range: {train_layer_range}\n" if train_method == "lora" and train_layer_range else ""
    params = """  params:
    lora_r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    target_modules: ["q_proj", "v_proj"]
""" + layer_line
elif train_method == "mole":
    params = """  params:
    lora_r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    target_modules: ["q_proj", "v_proj"]
    num_experts: 3
"""
elif train_method == "reft":
    layers = [
        int(item)
        for item in os.environ.get("REFT_LAYERS", "0").replace(",", " ").split()
    ]
    if not layers:
        layers = [0]
    params = f"""  params:
    rank: 4
    layers: {layers}
    prefix_positions: 4
    suffix_positions: 4
"""
elif train_method == "freeze":
    explicit_modules = os.environ.get("FREEZE_UNFREEZE_MODULES", "").strip()
    if explicit_modules:
        modules = [item.strip() for item in explicit_modules.replace(",", " ").split() if item.strip()]
    else:
        layers = parse_inclusive_range(os.environ.get("FREEZE_LAYER_RANGE", ""))
        if not layers:
            layers = parse_int_list(os.environ.get("FREEZE_LAYERS", "31"))
        prefix = transformer_layer_prefix(model)
        modules = [f"{prefix}.{idx}" for idx in layers]
    params = f"""  params:
    unfreeze_modules: {modules}
"""
elif train_method == "mores":
    params = ""
else:
    raise SystemExit(f"Unsupported train method: {method_slug}")

Path(path).write_text(
    f"""model:
  name: "{model}"

experiment:
  name: "{exp_name}"
  base_dir: "experiments"

training:
  ft_method: {train_method}
  num_epochs: 1
  per_device_batch_size: 1
  gradient_accumulation_steps: 4
  max_length: {int(max_length)}
  dataloader_num_workers: {int(os.environ.get("DATALOADER_NUM_WORKERS", "0"))}
  dataloader_pin_memory: {os.environ.get("DATALOADER_PIN_MEMORY", "false").lower()}
  dataloader_persistent_workers: {os.environ.get("DATALOADER_PERSISTENT_WORKERS", "false").lower()}
  learning_rate: {learning_rate}
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: {max_grad_norm}
  save_steps: 0
  output_dir: "experiments"
{params}
data:
  dataset_name: "{dataset}"
  max_samples: {int(train_max_samples)}
  visnec_score_file: "{method_visnec_score_file}"
  visnec_top_ratio: {method_visnec_top_ratio}
""",
    encoding="utf-8",
)
PY
}

write_eval_config() {
  local path="$1"
  local model="$2"
  local dataset="$3"
  local source="$4"
  local exp_name="$5"

  python - "$path" "$model" "$dataset" "$source" "$exp_name" "$EVAL_MAX_SAMPLES" "$MAX_NEW_TOKENS" "$EVAL_SAMPLE_SEED" "$EVAL_SHUFFLE_BUFFER_SIZE" <<'PY'
from pathlib import Path
import sys

path, model, dataset, source, exp_name, eval_max_samples, max_new_tokens, sample_seed, shuffle_buffer_size = sys.argv[1:]
sample_seed_line = f"  sample_seed: {int(sample_seed)}\n" if str(sample_seed).strip() else ""

Path(path).write_text(
    f"""model:
  name: "{model}"

experiment:
  name: "{exp_name}"
  base_dir: "experiments"

eval:
  source: "{source}"
  dataset_name: "{dataset}"
  max_samples: {int(eval_max_samples)}
{sample_seed_line}  shuffle_buffer_size: {int(shuffle_buffer_size)}
  max_new_tokens: {int(max_new_tokens)}
  temperature: 0.0
""",
    encoding="utf-8",
)
PY
}

append_summary_row() {
  local model="$1"
  local dataset_key="$2"
  local method="$3"
  local exp_name="$4"
  local base_exp="$5"
  local train_exit="$6"
  local base_eval_exit="$7"
  local eval_trained_exit="$8"

  python - "$SUMMARY_PATH" "$model" "$dataset_key" "$method" "$exp_name" "$base_exp" "$train_exit" "$base_eval_exit" "$eval_trained_exit" <<'PY'
import csv
import json
import difflib
import sys
from pathlib import Path

summary_path, model, dataset, method, exp_name, base_exp, train_exit, base_eval_exit, eval_trained_exit = sys.argv[1:]
exp_dir = Path("experiments") / exp_name
base_exp_dir = Path("experiments") / base_exp

def load_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def first_metric(payload):
    metrics = payload.get("metrics") or {}
    if metrics:
        name, value = next(iter(metrics.items()))
        return name, value
    return payload.get("metric", ""), ""

def load_predictions(path):
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records

def prediction_similarity(left_records, right_records):
    if not left_records or not right_records:
        return ""
    count = min(len(left_records), len(right_records))
    if count <= 0:
        return ""
    ratios = []
    for left, right in zip(left_records[:count], right_records[:count]):
        left_text = str(left.get("prediction", ""))
        right_text = str(right.get("prediction", ""))
        ratios.append(difflib.SequenceMatcher(None, left_text, right_text).ratio())
    return round(sum(ratios) / count, 4)

def short_text(value, limit=80):
    text = " ".join(str(value or "").split())
    return text[:limit]

def expected_steps(train_payload):
    cfg = train_payload.get("training_params") or {}
    data = train_payload.get("data") or {}
    try:
        max_samples = int(data.get("max_samples") or 0)
        batch = int(cfg.get("per_device_batch_size") or 1)
        grad_accum = int(cfg.get("gradient_accumulation_steps") or 1)
        epochs = int(cfg.get("num_epochs") or 1)
    except Exception:
        return ""
    if max_samples <= 0:
        return ""
    return max(1, (max_samples * epochs) // max(1, batch * grad_accum))

def suspicious_flags(row, train_payload, base_records, tuned_records, diagnostics, train_exit, eval_exit):
    flags = []
    if train_exit not in {"0", "skipped"}:
        flags.append(f"train_exit={train_exit}")
    if eval_exit not in {"0", "skipped"}:
        flags.append(f"eval_exit={eval_exit}")
    result = train_payload.get("result") or {}
    avg_loss = float(result.get("avg_loss") or 0)
    if avg_loss >= 5:
        flags.append(f"high_loss={avg_loss:g}")
    skipped = int(result.get("skipped_samples") or 0)
    max_samples = int((train_payload.get("data") or {}).get("max_samples") or 0)
    if max_samples and skipped / max_samples > 0.1:
        flags.append(f"high_skip={skipped}/{max_samples}")
    steps = int(result.get("total_steps") or 0)
    exp_steps = expected_steps(train_payload)
    if exp_steps and steps and steps < max(1, int(0.8 * exp_steps)):
        flags.append(f"low_steps={steps}/{exp_steps}")
    if float(diagnostics.get("top_prediction_ratio") or 0) >= 0.5:
        flags.append(f"repeated_prediction={diagnostics.get('top_prediction_ratio')}")
    if float(diagnostics.get("avg_prediction_words") or 0) > 5:
        flags.append(f"long_predictions={diagnostics.get('avg_prediction_words')}")
    sim = prediction_similarity(base_records, tuned_records)
    if sim != "" and float(sim) >= 0.8:
        flags.append(f"base_like={sim}")
    return ";".join(flags)

train = load_json(exp_dir / "train" / "train_summary.json")
base = load_json(base_exp_dir / "eval" / "eval.json")
tuned = load_json(exp_dir / "eval_trained" / "eval.json")
base_records = load_predictions(base_exp_dir / "eval" / "predictions.jsonl")
tuned_records = load_predictions(exp_dir / "eval_trained" / "predictions.jsonl")
metric, base_score = first_metric(base)
tuned_metric, tuned_score = first_metric(tuned)
metric = tuned_metric or metric
train_result = train.get("result") or {}
diag_source = tuned if method != "base" else base
diagnostics = diag_source.get("diagnostics") or {}
similarity = prediction_similarity(base_records, tuned_records)

row = {
    "model": model,
    "dataset": dataset,
    "method": method,
    "experiment": exp_name,
    "train_exit": train_exit,
    "base_eval_exit": base_eval_exit,
    "eval_trained_exit": eval_trained_exit,
    "metric": metric,
    "base_score": base_score,
    "tuned_score": tuned_score if method != "base" else "",
    "eval_predictions": tuned.get("num_predictions", base.get("num_predictions", "")),
    "avg_loss": train_result.get("avg_loss", ""),
    "train_steps": train_result.get("total_steps", ""),
    "train_time_s": train_result.get("train_time_s", ""),
    "trainable_params": train_result.get("trainable_params", ""),
    "trainable_pct": train_result.get("trainable_pct", ""),
    "skipped_samples": train_result.get("skipped_samples", ""),
    "avg_prediction_words": diagnostics.get("avg_prediction_words", ""),
    "long_prediction_count": diagnostics.get("long_prediction_count", ""),
    "top_prediction": short_text(diagnostics.get("top_prediction", "")),
    "top_prediction_ratio": diagnostics.get("top_prediction_ratio", ""),
    "gt_unanswerable_items": diagnostics.get("ground_truth_items_with_unanswerable", ""),
    "base_similarity": similarity if method != "base" else "",
    "suspicious_flags": suspicious_flags(
        row={},
        train_payload=train,
        base_records=base_records,
        tuned_records=tuned_records,
        diagnostics=diagnostics,
        train_exit=train_exit,
        eval_exit=eval_trained_exit,
    ) if method != "base" else "",
}

write_header = not Path(summary_path).exists()
with Path(summary_path).open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

run_or_continue() {
  local exit_code="$1"
  local label="$2"
  if [[ "$exit_code" != "0" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    echo "Stopping after failure in $label because CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR" >&2
    exit "$exit_code"
  fi
}

train_slug="$(sample_slug "$TRAIN_MAX_SAMPLES")"
eval_slug="$(sample_slug "$EVAL_MAX_SAMPLES")"
cond_slug="$(condition_slug)"

for model in $MODELS; do
  for dataset_key in $DATASETS; do
    dataset="$(dataset_id "$dataset_key")"
    base_exp="paper_${model}_${dataset_key}_base_eval${eval_slug}${cond_slug}_${RUN_STAMP}"
    base_eval_config="$GENERATED_DIR/${base_exp}_base_eval.yaml"
    write_eval_config "$base_eval_config" "$model" "$dataset" "base" "$base_exp"

    echo
    echo "### Base eval model=$model dataset=$dataset_key eval_samples=$EVAL_MAX_SAMPLES"
    mkdir -p "experiments/$base_exp"
    set +e
    VLMINTUNE_FAST_EXIT="${VLMINTUNE_FAST_EXIT:-1}" \
      python -m vlmintune.eval "${HF_TOKEN_ARGS[@]}" --config "$base_eval_config"
    base_exit=$?
    set -e
    append_summary_row "$model" "$dataset_key" "base" "$base_exp" "$base_exp" "skipped" "$base_exit" "skipped"
    run_or_continue "$base_exit" "base eval $base_exp"

    for method in $METHODS; do
      if [[ "$method" == "base" ]]; then
        continue
      fi

      effective_max_length="$(train_max_length_for "$model" "$dataset_key")"
      exp_name="paper_${model}_${method}_${dataset_key}_train${train_slug}_eval${eval_slug}_len${effective_max_length}${cond_slug}_${RUN_STAMP}"
      train_config="$GENERATED_DIR/${exp_name}_train.yaml"
      eval_trained_config="$GENERATED_DIR/${exp_name}_eval_trained.yaml"
      write_train_config "$train_config" "$model" "$dataset" "$method" "$exp_name" "$effective_max_length"
      write_eval_config "$eval_trained_config" "$model" "$dataset" "trained" "$exp_name"

      echo
      echo "### Training model=$model method=$method dataset=$dataset_key train_samples=$TRAIN_MAX_SAMPLES"
      set +e
      VLMINTUNE_FAST_EXIT="${VLMINTUNE_FAST_EXIT:-1}" \
        python -m vlmintune.training "${HF_TOKEN_ARGS[@]}" --config "$train_config"
      train_exit=$?
      set -e
      run_or_continue "$train_exit" "train $exp_name"

      eval_exit="skipped"
      if [[ "$train_exit" == "0" ]]; then
        echo
        echo "### Eval trained model=$model method=$method dataset=$dataset_key eval_samples=$EVAL_MAX_SAMPLES"
        set +e
        VLMINTUNE_FAST_EXIT="${VLMINTUNE_FAST_EXIT:-1}" \
          python -m vlmintune.eval "${HF_TOKEN_ARGS[@]}" --config "$eval_trained_config"
        eval_exit=$?
        set -e
        run_or_continue "$eval_exit" "trained eval $exp_name"
      fi

      append_summary_row "$model" "$dataset_key" "$method" "$exp_name" "$base_exp" "$train_exit" "$base_exit" "$eval_exit"
    done
  done
done

echo
echo "=== Paper benchmark summary ==="
cat "$SUMMARY_PATH"
