# textvqa_qwen25vl3b_freeze_debug

This setup directory contains a debug-friendly freeze tuning experiment for TextVQA.

- `train_config.yaml`: train a freeze-tuned checkpoint and write debug artifacts to `experiments/<experiment_name>/debug/`
- `eval_config.yaml`: evaluate the trained checkpoint and write predictions to `experiments/<experiment_name>/eval_predictions/`
- `base_eval_config.yaml`: evaluate the unfine-tuned base model on the same small validation slice
- `run_*.sh`: thin wrappers to run each config

The sample counts are intentionally small so you can quickly inspect a few outputs:

- training: `data.max_samples: 100`
- eval: `eval.max_samples: 20`

For a full run later, set those values to `0`.

Run on the SSH machine:

- `./run_train.sh`
- `./run_eval_trained.sh`
- `./run_eval_base.sh`
