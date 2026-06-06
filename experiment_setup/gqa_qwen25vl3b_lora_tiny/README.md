# gqa_qwen25vl3b_lora_tiny

This setup directory contains a tiny LoRA smoke run for GQA.

- `train_config.yaml`: train into `experiments/<experiment_name>/train/` and `checkpoint/`
- `eval_config.yaml`: evaluate the trained checkpoint into `experiments/<experiment_name>/eval_trained/`
- `base_eval_config.yaml`: evaluate the unfine-tuned base model into `experiments/<experiment_name>/eval_base/`
- `run_*.sh`: thin wrappers to run the configs

This setup is intentionally small so you can quickly validate the LoRA
training/eval wiring:

- training: `data.max_samples: 8`
- eval: `eval.max_samples: 4`
- default train split: `train_balanced`
- default eval split: `val_balanced`
- metric: `normalized_exact_match`

Run on the machine with the GPU and dependencies installed:

- `./run_train.sh`
- `./run_eval_trained.sh`
- `./run_eval_base.sh`
