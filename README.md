# vlmintune

`vlmintune` is a compact initial-release toolkit for parameter-efficient and
supervision-based visual instruction tuning. A training run uses one built-in
vision-language model, one dataset, and one fixed method recipe.

Chinese documentation: [Library usage and development guide](docs/library_guide_zh.html).

The public training configuration is deliberately small. Method architecture,
target modules, ranks, intervention positions, expert layout, and model-specific
compatibility are fixed in code rather than exposed as user options.

## Installation

Python 3.10 or newer and a CUDA GPU are recommended for training.

```bash
pip install -e .
```

QLoRA requires bitsandbytes, included in the fine-tuning extra:

```bash
pip install -e ".[finetune]"
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Built-in models and datasets

Models:

- `qwen25vl_3b_instruct` — Qwen2.5-VL-3B-Instruct
- `llava15_7b` — LLaVA-1.5-7B

Datasets:

- `lmms-lab/textvqa`
- `pingzhili/vqa_v2`
- `ebrukilic/vizwiz_vqa_dataset`
- `Mineru/GQA`
- `scienceqa_image`

Each run trains exactly one dataset. Its built-in dataset spec selects the default
training split.

## Seven standalone methods plus three fixed combinations

| Config name | Fixed initial-release recipe | Model support |
| --- | --- | --- |
| `lora` | Rank 8, alpha 16, dropout 0.05; q/k/v/o and gate/up/down in every language Transformer layer | Qwen and LLaVA |
| `qlora` | Rank 64, alpha 16, dropout 0; NF4 4-bit base, double quantization, BF16 compute, paged AdamW 8-bit | Qwen and LLaVA |
| `dora` | The fixed LoRA recipe with DoRA weight decomposition | Qwen and LLaVA |
| `reft` | Tied LoReFT, rank 4, prompt prefix 4 plus suffix 4, every language layer; generation intervenes only during prompt prefill | Qwen and LLaVA |
| `mores` | Rank 1 steering at the first 4 and last 5 visual tokens in every language layer | Qwen and LLaVA |
| `vl_adapter` | Single Adapter (VL-Adapter style), reduction factor 8, GELU-new, separate attention/FFN adapters; trains adapters, LayerNorm, and visual merger | **Qwen only** |
| `l2t` | Standalone full-SFT over the complete user-message text and answer; vision encoder frozen | Qwen and LLaVA |
| `mores_lora` | Joint MoReS representation steering and LoRA weight adaptation | Qwen and LLaVA |
| `mores_dora` | Joint MoReS representation steering and DoRA weight adaptation | Qwen and LLaVA |
| `reft_lora` | Joint ReFT representation intervention and LoRA weight adaptation | Qwen and LLaVA |

`vl_adapter` is a Qwen structural adaptation named Single Adapter (VL-Adapter
style), not a full reproduction of the original VL-T5/VL-BART multi-task system.
The release supports the three fixed structural combinations `mores_lora`,
`mores_dora`, and `reft_lora`. Free-form and arbitrary combinations are rejected.

## Strict training configuration

Training accepts exactly these eleven top-level fields:

```yaml
model: qwen25vl_3b_instruct
dataset: lmms-lab/textvqa
method: lora
epochs: 1
learning_rate: 0.0002
batch_size: 1
gradient_accumulation_steps: 4
max_length: 2048
max_samples: 0  # 0 means the full training split
seed: 42
output_dir: experiments/textvqa_lora/checkpoint
```

Run it with:

```bash
python -m vlmintune.training --config train_config.yaml
```

Unknown fields are rejected. In particular, the initial release has no
`method_params`, `target_modules`, layer ranges, method-specific aliases,
multi-dataset list, or legacy nested `model/training/data/experiment` schema.

The checkpoint and its `vlmintune_meta.json` are written directly to
`output_dir`. Using `experiments/<run-name>/checkpoint` keeps the checkpoint
compatible with the current experiment-based evaluation loader.

## Minimal runner

The repository includes a small wrapper that creates the strict config and starts
one training run:

```bash
MODEL=qwen25vl_3b_instruct \
DATASET=lmms-lab/textvqa \
METHOD=lora \
RUN_NAME=textvqa_lora \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

Single Adapter example:

```bash
MODEL=qwen25vl_3b_instruct METHOD=vl_adapter RUN_NAME=textvqa_single_adapter \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

Standalone L2T example:

```bash
MODEL=qwen25vl_3b_instruct METHOD=l2t RUN_NAME=textvqa_l2t \
bash experiment_setup/paper_benchmark/run_paper_benchmark.sh
```

Validate generated configs for all ten release recipes without loading a model or GPU:

```bash
bash experiment_setup/paper_benchmark/run_smoke_all.sh
```

## AutoDL sync and staged Qwen TextVQA validation

The AutoDL helper synchronizes the working tree without copying local secrets,
model weights, experiment outputs, or Git metadata, then runs the remote tests in
the dedicated Torch 2.11 environment:

```bash
AUTODL_HOST=your-host.example AUTODL_PORT=22 \
AUTODL_KEY="$HOME/.ssh/id_ed25519" \
scripts/autodl_sync_test.sh
```

On the AutoDL machine, run the standalone Qwen-compatible recipes with a small
stage first. Every method must train, save, reload, and
produce the requested number of validation predictions before the next method
starts:

```bash
STAGE_SAMPLES=8 EVAL_SAMPLES=8 \
bash scripts/run_qwen_textvqa_stage.sh

STAGE_SAMPLES=256 EVAL_SAMPLES=32 GRADIENT_ACCUMULATION_STEPS=4 \
bash scripts/run_qwen_textvqa_stage.sh

STAGE_SAMPLES=1000 EVAL_SAMPLES=100 GRADIENT_ACCUMULATION_STEPS=4 \
bash scripts/run_qwen_textvqa_stage.sh
```

MoReS + LoRA has its own strict progressive pipeline:

```bash
bash scripts/run_qwen_textvqa_mores_lora_pipeline.sh
```

It runs `8/8`, `256/32`, `1000/100`, and full `34602/5000` stages, requires
both LoRA and MoReS checkpoint components, and shares the TextVQA combination
lock so it cannot contend with another combination run on the same GPU.

Joint-method names are included in run names, so their checkpoints and evaluation
artifacts cannot collide with the corresponding standalone method.

Completed runs are skipped when their checkpoint and evaluation summary pass
validation. Set `FORCE=1` or change `RUN_PREFIX` after changing learning rates,
sequence length, epochs, or other run-defining settings. For a full TextVQA run,
use the explicit split sizes `STAGE_SAMPLES=34602 EVAL_SAMPLES=5000`; the public
trainer does not yet provide periodic checkpoint/resume within a method.

The initial release does not include target sweeps, layer sweeps, free-form method
combinations, or multi-dataset benchmark orchestration.

## Evaluation

Evaluation still uses one dataset per run. If training wrote to
`experiments/textvqa_lora/checkpoint`, a trained-checkpoint eval config can use:

```yaml
model:
  name: qwen25vl_3b_instruct
experiment:
  name: textvqa_lora
  base_dir: experiments
eval:
  source: trained
  checkpoint_path: experiments/textvqa_lora/checkpoint
  dataset_name: lmms-lab/textvqa
  max_new_tokens: 32
  temperature: 0.0
```

```bash
python -m vlmintune.eval --config eval_config.yaml
```

The evaluation schema is separate from the strict eleven-field training schema.
`eval.checkpoint_path` may point directly at any training `output_dir`; if it is
omitted, evaluation falls back to `experiments/<experiment.name>/checkpoint`.

## Development

```bash
python -m pytest -q
```

No pre-trained or fine-tuned checkpoints are currently published from this
repository.
