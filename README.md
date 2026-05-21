# vlmintune

This repository is the official implementation of **vlmintune**, a compact multimodal instruction tuning toolkit for vision-language models. The project is organized around reproducible local fine-tuning and evaluation rather than a single monolithic training script.

`vlmintune` currently supports LoRA-family fine-tuning, freeze tuning, label-to-target (L2T) composition, Hugging Face VQA-style datasets, and a small experiment workflow for training and evaluation.

## Requirements

- Python `>=3.9`
- A CUDA-capable GPU for practical training and evaluation
- Core dependencies:
  - `torch~=2.11.0`
  - `torchvision~=0.26.0`
  - `transformers~=5.7.0`
  - `peft~=0.19.1`
  - `accelerate~=1.13.0`
  - `datasets>=2.14`
  - `pillow>=9.0`
  - `pyyaml>=6.0`

The package pins the tested LoRA-family stack directly, and `qlora` additionally expects `bitsandbytes~=0.49.2`. This repository does not depend on `torchao`.

To install the package in editable mode:

```bash
pip install -e .
```

To install fine-tuning extras:

```bash
pip install -e ".[finetune]"
```

To install development extras:

```bash
pip install -e ".[dev]"
```

## What Is Included

### Training methods

- `lora`
- `qlora`
- `dora`
- `freeze`
- `l2t`

### Dataset support

Built-in dataset specs currently cover:

- `lmms-lab/textvqa`

Training uses the built-in Hugging Face dataset loader in `src/vlmintune/data/hf_datasets.py`. The example training configs default to `lmms-lab/textvqa` because it exposes a `train` split on Hugging Face.

### Evaluation support

- `TextVQA`

## Repository Layout

```text
src/vlmintune/
  training/      fine-tuning methods, trainer, local CLI entry point
  eval/          inference methods and scoring helpers
  data/          dataset specs, Hugging Face dataset loader, canonical sample types
experiment_setup/
  <experiment>/  per-experiment configs and run scripts
tests/           lightweight regression tests
```

## Training

All training flows are config-driven. Each experiment keeps its own YAML configs and run scripts under `experiment_setup/<experiment_name>/`.

### Local training

To run one experiment:

```bash
python -m vlmintune.training --config experiment_setup/textvqa_qwen25vl3b_lora_full/train_config.yaml
```

The recommended setup layout is:

```text
experiment_setup/<experiment_name>/
  train_config.yaml
  eval_config.yaml
  base_eval_config.yaml
  run_train.sh
  run_eval_trained.sh
  run_eval_base.sh
```

Notes:

- Run the commands on the machine that actually has the model weights, GPU, and dependencies installed.
- Training dataset selection lives under `data.data_path`.
- Training sample count lives under `data.max_samples`. Set it to `0` or omit it for the full split.
- `experiment.name` is required and becomes the experiment folder name under `experiment.base_dir` (default `experiments/`).
- Training writes into one fixed experiment layout:

```text
experiments/<experiment_name>/
  checkpoint/
  train/
    train_summary.json
    run.log
  eval_trained/
    eval.json
    predictions.jsonl
    run.log
  eval_base/
    eval.json
    predictions.jsonl
    run.log
```

- The trainer emits a small amount of runtime information by design, including dataset resolution, estimated training plan, and the first batch tensor shapes.
- The first 5 debug examples are written into `run.log`; there is no separate `debug/` folder.
- There is no separate `fullrun` command in the initial release. Training the full dataset is just a normal training run with `data.max_samples` omitted or set to `0`.
- Use the experiment-local wrappers in `experiment_setup/<experiment_name>/` when you want one-command train / eval runs.

## Evaluation

The intended workflow is:

1. Run `python -m vlmintune.training --config ...`
2. Run `python -m vlmintune.eval --config ...` for the trained checkpoint
3. Run `python -m vlmintune.eval --config ...` for the base-model comparison

### Evaluate A Saved Experiment

To evaluate the trained checkpoint inside an experiment folder:

```bash
python -m vlmintune.eval --config experiment_setup/textvqa_qwen25vl3b_lora_full/eval_config.yaml
```

In that config:

- `experiment.name` selects the saved experiment
- `experiment.base_dir` points at the experiment root directory and defaults to `experiments`
- `eval.source` is required and must be `"trained"`
- `eval.dataset_name` selects the eval dataset
- `eval.split` is required
- `eval.metric` is required and must be `"vqa_accuracy"`
- `eval.max_samples` limits the eval sample count

### Evaluate A Base-Model Baseline

To evaluate the corresponding unfine-tuned base model under the same experiment folder:

```bash
python -m vlmintune.eval --config experiment_setup/textvqa_qwen25vl3b_lora_full/base_eval_config.yaml
```

In that config:

- `experiment.name` is required
- `experiment.base_dir` points at the experiment root directory and defaults to `experiments`
- `model.model_path` is required
- `eval.source` is required and must be `"base"`
- `eval.dataset_name` selects the eval dataset
- `eval.split` is required
- `eval.metric` is required and must be `"vqa_accuracy"`
- `eval.max_samples` limits the eval sample count

This initial release intentionally evaluates one dataset per run. If you want multiple eval datasets, run `vlmintune.eval` multiple times with different configs.

## Pre-trained Models

No pre-trained or fine-tuned checkpoints are currently published from this repository.

Produced adapters and checkpoints are written to `experiments/<experiment_name>/checkpoint/` on the current machine.

## Results

This repository does not currently ship a paper-specific leaderboard table or released benchmark checkpoints.

## Contributing

Issues and pull requests are welcome, especially for:

- new training methods
- additional dataset specs or adapters
- benchmark integrations
- regression tests
- documentation and reproducibility improvements

For development work, install the dev extras and run targeted tests:

```bash
pip install -e ".[dev]"
python -m pytest tests
```

If you change training or serialization behavior, please run an appropriate targeted validation command and include the result in your PR description.

## Project Status

`vlmintune` is usable today for small-to-medium multimodal fine-tuning experiments, but it is still early-stage infrastructure. In particular:

- the repository does not yet ship published pre-trained adapters
- benchmark result tables are not yet curated in the README
- the repository does not currently include a standalone `LICENSE` file

If you are adopting the code in a downstream project, it is worth checking the configs and output conventions directly rather than assuming a fully stabilized release contract.
