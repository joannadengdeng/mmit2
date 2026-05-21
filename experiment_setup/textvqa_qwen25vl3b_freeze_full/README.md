# textvqa_qwen25vl3b_freeze_full

This setup directory contains a full-dataset freeze tuning experiment for TextVQA.

- `train_config.yaml`: train into `experiments/<experiment_name>/train/` and `checkpoint/`
- `eval_config.yaml`: evaluate the trained checkpoint into `experiments/<experiment_name>/eval_trained/`
- `base_eval_config.yaml`: evaluate the unfine-tuned base model into `experiments/<experiment_name>/eval_base/`
- `run_*.sh`: thin wrappers to run each config

This setup keeps the same freeze recipe as the debug run, but removes the sample caps:

- training: `data.max_samples: 0`
- eval: `eval.max_samples: 0`

Run on the machine with the GPU and dependencies installed:

- `./run_train.sh`
- `./run_eval_trained.sh`
- `./run_eval_base.sh`
