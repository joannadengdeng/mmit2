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
