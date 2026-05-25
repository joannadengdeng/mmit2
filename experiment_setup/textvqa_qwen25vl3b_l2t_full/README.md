# textvqa_qwen25vl3b_l2t_full

This setup directory contains one complete L2T experiment definition:

- `train_config.yaml`: train into `experiments/<experiment_name>/train/` and `checkpoint/`
- `eval_config.yaml`: evaluate the trained checkpoint into `experiments/<experiment_name>/eval_trained/`
- `base_eval_config.yaml`: evaluate the unfine-tuned base model into `experiments/<experiment_name>/eval_base/`
- `run_*.sh`: thin wrappers to run the configs

This setup uses `training.ft_method: l2t` and shares the same LoRA adapter
hyperparameters as the LoRA baseline.

This is the full-run setup:

- training: `data.max_samples: 0`
- eval: `eval.max_samples: 0`

To avoid exporting `HF_TOKEN` every time, put your token once in a local file at:

- `/Users/dengqiuyu/Documents/New project/vlmintune/.hf_token`

That file is gitignored, and the `run_*.sh` wrappers in this setup will pass it
automatically via `--hf-token-file` when it exists and is non-empty.

Run on the machine with the GPU and dependencies installed:

- `./run_train.sh`
- `./run_eval_trained.sh`
- `./run_eval_base.sh`
