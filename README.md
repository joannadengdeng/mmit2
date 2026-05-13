# mmit2

This repository is the official implementation of **mmit2**, a compact multimodal instruction tuning toolkit for vision-language models. The project is organized around reproducible fine-tuning, evaluation, and SSH-based runtime orchestration rather than a single monolithic training script.

`mmit2` currently supports LoRA-family fine-tuning, freeze tuning, label-to-target (L2T) composition, Hugging Face VQA-style datasets, and a small SSH-based experiment workflow for training and evaluation.

## Requirements

- Python `>=3.9`
- A CUDA-capable GPU for practical training and evaluation
- Core dependencies:
  - `torch>=2.0`
  - `torchvision>=0.15`
  - `transformers>=4.37`
  - `peft>=0.7`
  - `accelerate>=0.21`
  - `datasets>=2.14`
  - `pillow>=9.0`
  - `pyyaml>=6.0`

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

To install SSH / remote execution extras:

```bash
pip install -e ".[remote]"
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

- `lmms-lab/VQAv2`
- `lmms-lab/textvqa`
- `lmms-lab/VizWiz-VQA`

Training uses the Hugging Face adapter path in `src/mmit2/data/adapters/hf_datasets.py`. The example training configs default to `lmms-lab/textvqa` because it exposes a `train` split on Hugging Face.

### Evaluation support

- `VQAv2`
- `TextVQA`
- `VizWiz`

### Runtime mode

- `ssh`

## Repository Layout

```text
src/mmit2/
  training/      fine-tuning methods, trainer, runtime runner
  eval/          inference methods and scoring helpers
  data/          dataset specs, adapters, canonical sample types
configs/         ready-to-run YAML configs
examples/        inspection helpers and package notes
tests/           lightweight regression tests
```

## Training

All training flows are config-driven. The YAML files in `configs/` define the SSH target, model, method, optimizer settings, training dataset, and training sample count.

### SSH training

To run QLoRA training through the SSH runner:

```bash
python -m mmit2.training --config configs/ssh_qlora.yaml
```

Other built-in SSH configs:

- `configs/ssh_lora.yaml`
- `configs/ssh_lora_textvqa.yaml`
- `configs/ssh_dora.yaml`
- `configs/ssh_freeze.yaml`
- `configs/ssh_l2t.yaml`

Notes:

- The local machine only needs the SSH client-side dependency path, for example `pip install -e ".[remote]"`.
- The remote machine needs the actual training dependencies, for example `pip install -e ".[finetune]"`.
- Training dataset selection lives under `data.data_path`.
- Training sample count lives under `data.max_samples`. Set it to `0` or omit it for the full split.
- Each training run creates an experiment directory with a saved `summary.json`, checkpoint path, config snapshot, and train summary.
- The trainer emits a small amount of runtime information by design, including dataset resolution, estimated training plan, and the first batch tensor shapes.
- There is no separate `fullrun` command in the initial release. Training the full dataset is just a normal training run with `data.max_samples` omitted or set to `0`.
- JarvisLabs helper scripts are available in `scripts/`:
  - `./scripts/jarvislabs_lora_100.sh` for a 100-sample smoke run
  - `./scripts/jarvislabs_lora_full.sh` for a full-split LoRA run

## Evaluation

The intended workflow is:

1. Run `python -m mmit2.training --config ...`
2. Run `python -m mmit2.eval --config ...`

### Evaluate A Saved Experiment

To evaluate a previously saved experiment without retraining:

```bash
python -m mmit2.eval --config configs/ssh_experiment_eval.yaml
```

In that config:

- `experiment.name` selects the saved experiment
- `experiment.base_dir` points at the experiment root directory
- `eval.dataset_name` selects the eval dataset
- `eval.max_samples` limits the eval sample count

### Evaluate A Base-Model Baseline

To evaluate an unfine-tuned Hugging Face base model as a baseline:

```bash
python -m mmit2.eval --config configs/ssh_baseline_eval.yaml
```

In that config:

- `model.model_path` is required
- `eval.dataset_name` selects the eval dataset
- `eval.max_samples` limits the eval sample count
- `eval.output_dir` controls where the summary and predictions are written

This initial release intentionally evaluates one dataset per run. If you want multiple eval datasets, run `mmit2.eval` multiple times with different configs.

## Pre-trained Models

No pre-trained or fine-tuned checkpoints are currently published from this repository.

Produced adapters and checkpoints are written to the configured `output_dir` on the remote machine.

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

If you change training, runtime, or serialization behavior, please run an appropriate targeted validation command and include the result in your PR description.

## Project Status

`mmit2` is usable today for small-to-medium multimodal fine-tuning experiments, but it is still early-stage infrastructure. In particular:

- the repository does not yet ship published pre-trained adapters
- benchmark result tables are not yet curated in the README
- the repository does not currently include a standalone `LICENSE` file

If you are adopting the code in a downstream project, it is worth checking the configs and output conventions directly rather than assuming a fully stabilized release contract.
