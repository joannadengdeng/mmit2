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
3. Run training and evaluation from that setup directory on the SSH machine, for example `./run_train.sh`.
4. If you want to archive the setup, manually copy it into `experiment_results/`
   together with the experiment artifacts you want to keep.
