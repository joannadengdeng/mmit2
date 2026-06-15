# Paper Benchmark Runner

This runner is a thin scheduler for the benchmark matrix in
`docs/paper_experiment_requirements.md`. It uses the normal `vlmintune` training
and evaluation CLIs, keeps core code paths unchanged, and writes an aggregate
CSV.

Training runs get the standard training artifacts, and evaluation outputs are
written only for the source that was actually evaluated:

```text
train/train_summary.json
train/run.log
checkpoint/
eval_trained/eval.json
eval_trained/predictions.jsonl
eval/eval.json
eval/predictions.jsonl
```

## Smoke test

Full smoke matrix: every model, every dataset, every method, 8 train samples and
8 eval samples each. This is the first thing to run after code changes.

```bash
bash experiment_setup/paper_benchmark/run_smoke_all.sh
```

Narrow smoke test:

```bash
MODELS="qwen25vl_3b_instruct" \
DATASETS="textvqa" \
METHODS="base lora" \
TRAIN_MAX_SAMPLES=8 \
EVAL_MAX_SAMPLES=8 \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

## Priority 1: Qwen TextVQA full matrix

```bash
MODELS="qwen25vl_3b_instruct" \
DATASETS="textvqa" \
METHODS="base qlora lora dora freeze l2t mole reft mores lora_layer lora_visnec" \
TRAIN_MAX_SAMPLES=0 \
EVAL_MAX_SAMPLES=0 \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

## Current TextVQA 1000 Matrix

This is the focused experiment set used for follow-up work: Qwen and LLaVA,
TextVQA only, 1000 train samples and 1000 eval samples. It includes the 8 base
tuning methods plus LoRA layer selection and LoRA data selection.

```bash
MODELS="qwen25vl_3b_instruct llava15_7b" \
DATASETS="textvqa" \
METHODS="base qlora lora dora freeze l2t mole reft mores lora_layer lora_visnec" \
TRAIN_MAX_SAMPLES=1000 \
EVAL_MAX_SAMPLES=1000 \
LORA_LAYER_TRAIN_LAYER_RANGE="24:31" \
LORA_VISNEC_SCORE_FILE=/root/autodl-tmp/visnec_scores/textvqa.jsonl \
LORA_VISNEC_TOP_RATIO=0.3 \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

## Full paper matrix

```bash
MODELS="qwen25vl_3b_instruct llava15_7b" \
DATASETS="textvqa vqav2 vizwiz gqa" \
METHODS="base qlora lora dora freeze l2t mole reft mores" \
TRAIN_MAX_SAMPLES=0 \
EVAL_MAX_SAMPLES=0 \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

## Save terminal output

```bash
mkdir -p run_logs

MODELS="qwen25vl_3b_instruct" \
DATASETS="textvqa" \
METHODS="base qlora lora dora freeze l2t mole reft mores" \
TRAIN_MAX_SAMPLES=0 \
EVAL_MAX_SAMPLES=0 \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh \
  2>&1 | tee run_logs/paper_benchmark_$(date +%Y%m%d_%H%M%S).log
```

The aggregate summary is written to:

```text
experiment_setup/paper_benchmark/paper_benchmark_<stamp>_summary.csv
```

## Optional VisNec filtering

VisNec is a data-selection condition, not a tuning method. If you have a score
file, pass it through the data config environment. Experiment names include a
`_visnec<ratio>` suffix when this condition is enabled.

```bash
VISNEC_SCORE_FILE=/root/autodl-tmp/visnec_scores/textvqa.jsonl \
VISNEC_TOP_RATIO=0.3 \
MODELS="qwen25vl_3b_instruct" \
DATASETS="textvqa" \
METHODS="base qlora lora dora freeze l2t mole reft mores" \
TRAIN_MAX_SAMPLES=0 \
EVAL_MAX_SAMPLES=0 \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

## Layer-selection variants

The `freeze` method supports selecting which transformer layers to unfreeze. By
default it unfreezes the last Qwen/LLaVA language layer, matching the 1000
baseline. For follow-up 1000 experiments, use one of:

```bash
FREEZE_LAYERS="24 31" \
METHODS="freeze" \
TRAIN_MAX_SAMPLES=1000 \
EVAL_MAX_SAMPLES=1000 \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

```bash
FREEZE_LAYER_RANGE="24:31" \
METHODS="freeze" \
TRAIN_MAX_SAMPLES=1000 \
EVAL_MAX_SAMPLES=1000 \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

You can also override the exact prefixes:

```bash
FREEZE_UNFREEZE_MODULES="model.language_model.layers.30 model.language_model.layers.31" \
METHODS="freeze" \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

LoRA and QLoRA can be restricted to an inclusive transformer-layer range:

```bash
LORA_TRAIN_LAYER_RANGE="24:31" \
QLORA_TRAIN_LAYER_RANGE="24:31" \
METHODS="lora qlora" \
TRAIN_MAX_SAMPLES=1000 \
EVAL_MAX_SAMPLES=1000 \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

The `lora_layer` method slug is a convenience wrapper around `ft_method: lora`
with a default `train_layer_range` of `24:31`. Override it with:

```bash
LORA_LAYER_TRAIN_LAYER_RANGE="20:31" \
METHODS="lora_layer" \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

The `lora_visnec` method slug is a convenience wrapper around `ft_method: lora`
with VisNec filtering applied to training data only. It requires
`LORA_VISNEC_SCORE_FILE` or `VISNEC_SCORE_FILE`.

## Debug artifacts to check

Training debug is in each experiment's `train/run.log`. The smoke run should
show:

```text
canonical_samples
rendered_prompts
trainable_parameters
label_supervision
gradient
mores_intervention
mores_runtime
```

`mores_intervention` and `mores_runtime` only apply to MoReS. Base-model
evaluation outputs are in `eval/predictions.jsonl`; trained-checkpoint outputs
are in `eval_trained/predictions.jsonl` with matching `eval.json` files.
