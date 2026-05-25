# textvqa_qwen25vl3b_lora_debug

This setup directory contains a debug-friendly LoRA experiment for TextVQA.

- `train_config.yaml`: train into `experiments/<experiment_name>/train/` and `checkpoint/`
- `eval_config.yaml`: evaluate the trained checkpoint into `experiments/<experiment_name>/eval_trained/`
- `base_eval_config.yaml`: evaluate the unfine-tuned base model into `experiments/<experiment_name>/eval_base/`
- `run_*.sh`: thin wrappers to run the configs

The sample counts are intentionally small so you can quickly validate training,
evaluation, and prediction outputs:

- training: `data.max_samples: 100`
- eval: `eval.max_samples: 20`

For a larger run later, raise those values or set training to `0`.

Run on the machine with the GPU and dependencies installed:

- `./run_train.sh`
- `./run_eval_trained.sh`
- `./run_eval_base.sh`
