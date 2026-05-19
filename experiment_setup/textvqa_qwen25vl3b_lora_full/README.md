# textvqa_qwen25vl3b_lora_full

This setup directory contains one complete experiment definition:

- `train_config.yaml`: SSH training config
- `eval_config.yaml`: evaluate the trained checkpoint
- `base_eval_config.yaml`: evaluate the unfine-tuned base model
- `run_*.sh`: thin wrappers to run the configs

Adjust the SSH section and any model/dataset/training values here when cloning
this setup into a new experiment directory.
