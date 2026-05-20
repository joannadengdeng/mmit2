# textvqa_qwen25vl3b_freeze_full

This setup directory contains a full-dataset freeze tuning experiment for TextVQA.

- `train_config.yaml`: train a freeze-tuned checkpoint and write debug artifacts to `experiments/<experiment_name>/debug/`
- `eval_config.yaml`: evaluate the trained checkpoint on the full validation split
- `base_eval_config.yaml`: evaluate the unfine-tuned base model on the same full validation split
- `run_*.sh`: thin wrappers to run each config

This setup keeps the same freeze recipe as the debug run, but removes the sample caps:

- training: `data.max_samples: 0`
- eval: `eval.max_samples: 0`

Run on the SSH machine:

- `./run_train.sh`
- `./run_eval_trained.sh`
- `./run_eval_base.sh`
