# textvqa_qwen25vl3b_l2t_debug

This setup directory contains a debug-friendly L2T experiment for TextVQA.

- `train_config.yaml`: train into `experiments/<experiment_name>/train/` and `checkpoint/`
- `eval_config.yaml`: evaluate the trained checkpoint into `experiments/<experiment_name>/eval_trained/`
- `base_eval_config.yaml`: evaluate the unfine-tuned base model into `experiments/<experiment_name>/eval_base/`
- `run_*.sh`: thin wrappers to run the configs

The sample counts are intentionally small so you can quickly validate:

- training: `data.max_samples: 500`
- eval: `eval.max_samples: 50`

This setup uses `training.ft_method: l2t` and writes the L2T supervision debug
preview into `experiments/<experiment_name>/train/run.log`.

Run on the machine with the GPU and dependencies installed:

- `./run_train.sh`
- `./run_eval_trained.sh`
- `./run_eval_base.sh`
