# textvqa_qwen25vl3b_freeze_debug

This setup directory contains a debug-friendly freeze tuning experiment for TextVQA.

- `train_config.yaml`: train into `experiments/<experiment_name>/train/` and `checkpoint/`
- `eval_config.yaml`: evaluate the trained checkpoint into `experiments/<experiment_name>/eval_trained/`
- `base_eval_config.yaml`: evaluate the unfine-tuned base model into `experiments/<experiment_name>/eval_base/`
- `run_*.sh`: thin wrappers to run each config

The sample counts are intentionally small so you can quickly inspect a few outputs:

- training: `data.max_samples: 100`
- eval: `eval.max_samples: 20`

For a full run later, set those values to `0`.

Run on the machine with the GPU and dependencies installed:

- `./run_train.sh`
- `./run_eval_trained.sh`
- `./run_eval_base.sh`
