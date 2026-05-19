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
  bundle_results.sh
```

Workflow:

1. Create one setup directory for the experiment.
2. Put that experiment's config files and run scripts there.
3. Run training and evaluation from that setup directory.
4. Bundle results after the run.

The results bundler copies the matching experiment setup directory into
`experiment_results/<bundle_name>/experiment_setup/`, so each archived result
keeps its own config and scripts together with the copied experiment directory.
