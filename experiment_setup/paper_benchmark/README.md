# Minimal Training Runner

This directory contains the initial-release training entry point. It intentionally
does not contain method sweeps, target-module variants, layer-selection variants,
multi-dataset mixtures, or evaluation orchestration.

Each invocation trains exactly one model, one dataset, and one of ten fixed
release recipes: seven standalone methods and three joint structural recipes.
Method internals are fixed in code; the runner only writes the eleven public
training fields.

## Run one training job

```bash
MODEL=qwen25vl_3b_instruct \
DATASET=lmms-lab/textvqa \
METHOD=lora \
RUN_NAME=textvqa_lora \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

The generated config is saved with the checkpoint at
`experiments/textvqa_lora/checkpoint/train_config.yaml`. The default output path
is `experiments/<RUN_NAME>/checkpoint`, which is also compatible with the current
experiment-based evaluation loader.

Supported environment variables are the eleven public config values, with
`RUN_NAME` and `CONFIG_PATH` as runner-only path conveniences:

- `MODEL`
- `DATASET`
- `METHOD`
- `EPOCHS`
- `LEARNING_RATE`
- `BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`
- `MAX_LENGTH`
- `MAX_SAMPLES` (`0` means the full split)
- `SEED`
- `OUTPUT_DIR`
- `RUN_NAME`
- `CONFIG_PATH`

`HF_TOKEN_FILE` or `HF_TOKEN` may be supplied for gated Hugging Face assets.

## Model restrictions

- `vl_adapter` requires `MODEL=qwen25vl_3b_instruct`.
- The remaining nine recipes support both built-in models.

Examples:

```bash
MODEL=qwen25vl_3b_instruct METHOD=vl_adapter RUN_NAME=textvqa_single_adapter \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

## Fixed structural combinations

The fixed structural combinations are `mores_lora`, `mores_dora`, and
`reft_lora`. They install the representation intervention before weight-adapter
injection and train and save both parameter families jointly.
There is no free-form combination syntax.

```bash
MODEL=qwen25vl_3b_instruct METHOD=mores_lora RUN_NAME=textvqa_mores_lora \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

## Configuration-only smoke

The smoke wrapper generates and validates a strict config for every release
method. It does not load models, datasets, or GPUs:

```bash
bash experiment_setup/paper_benchmark/run_smoke_all.sh
```

For one dry run:

```bash
METHOD=qlora DRY_RUN=1 \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

Use `MAX_SAMPLES` for staged smoke runs before committing to the full split:

```bash
MAX_SAMPLES=8 METHOD=lora RUN_NAME=textvqa_lora_smoke \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```
