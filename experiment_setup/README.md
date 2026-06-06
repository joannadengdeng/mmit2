# Experiment Setup

Each experiment lives in its own subdirectory under `experiment_setup/`.

Recommended layout:

```text
experiment_setup/<experiment_name>/
  train_config.yaml
  eval_config.yaml
  base_eval_config.yaml
  run_train.sh
  run_eval_trained.sh
  run_eval_base.sh
```

Workflow:

1. Create one setup directory for the experiment.
2. Put that experiment's config files and run scripts there.
3. Run training, trained-model eval, and base-model eval on the machine with the GPU and dependencies installed.
4. All three commands write into the same experiment folder under `experiments/<experiment_name>/`.

Tiny LoRA smoke setups included in this repo:

- `textvqa_qwen25vl3b_lora_tiny`
- `textvqa_llava15_7b_lora_tiny`
- `vqav2_qwen25vl3b_lora_tiny`
- `vizwiz_qwen25vl3b_lora_tiny`
- `gqa_qwen25vl3b_lora_tiny`

These are intentionally tiny end-to-end validation setups:

- training uses `data.max_samples: 8`
- eval uses `eval.max_samples: 4`
- dataset default train/eval splits are selected automatically by the built-in dataset specs

MoReS memory probes for JarvisLabs A100 are under `mores_memory_probe/`. Start with:

```bash
bash experiment_setup/mores_memory_probe/run_5090_check.sh
```
