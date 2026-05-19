"""Minimal experiment bundling for vlmintune."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def latest_subdir(base_dir: Path) -> Optional[Path]:
    if not base_dir.is_dir():
        return None
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_experiment_dir(experiment_base_dir: Path, experiment_name: str) -> Path:
    if experiment_name:
        experiment_dir = experiment_base_dir / experiment_name
    else:
        latest = latest_subdir(experiment_base_dir)
        if latest is None:
            raise FileNotFoundError(f"No experiments found in {experiment_base_dir}")
        experiment_dir = latest

    summary_path = experiment_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Experiment summary not found: {summary_path}")
    return experiment_dir


def resolve_setup_dir(
    *,
    explicit_setup_dir: str,
    setup_base_dir: Path,
    experiment_name: str,
    experiment_summary: Dict[str, Any],
) -> Optional[Path]:
    candidates = []
    if explicit_setup_dir.strip():
        candidates.append(Path(explicit_setup_dir.strip()))

    summary_setup_dir = str(
        ((experiment_summary.get("config", {}) or {}).get("experiment", {}) or {}).get("setup_dir", "")
    ).strip()
    if summary_setup_dir:
        candidates.append(Path(summary_setup_dir))

    candidates.append(setup_base_dir / experiment_name)

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else Path.cwd() / candidate
        if resolved.is_dir():
            return resolved
    return None


def resolve_baseline_eval_dir(explicit_dir: str, eval_outputs_dir: Path) -> Optional[Path]:
    if explicit_dir.strip():
        candidate = Path(explicit_dir.strip())
        resolved = candidate if candidate.is_absolute() else Path.cwd() / candidate
        if not resolved.is_dir():
            raise FileNotFoundError(f"Baseline eval directory not found: {resolved}")
        return resolved
    return latest_subdir(eval_outputs_dir)


def copy_if_exists(src: Optional[Path], dst: Path) -> None:
    if src is None:
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_bundle(
    *,
    experiment_name: str,
    experiment_base_dir: str = "experiments",
    setup_base_dir: str = "experiment_setup",
    eval_outputs_dir: str = "eval_outputs",
    run_logs_dir: str = "run_logs",
    bundle_base_dir: str = "experiment_results",
    baseline_eval_dir: str = "",
    bundle_name: str = "",
    setup_dir: str = "",
) -> Path:
    experiment_base_path = Path(experiment_base_dir)
    experiment_dir = resolve_experiment_dir(experiment_base_path, experiment_name)
    experiment_name = experiment_dir.name
    experiment_summary = load_json(experiment_dir / "summary.json")

    if not bundle_name.strip():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bundle_name = f"{experiment_name}_bundle_{stamp}"

    bundle_dir = Path(bundle_base_dir) / bundle_name
    if bundle_dir.exists():
        raise FileExistsError(f"Bundle directory already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True)

    shutil.copytree(experiment_dir, bundle_dir / "experiment")

    setup_path = resolve_setup_dir(
        explicit_setup_dir=setup_dir,
        setup_base_dir=Path(setup_base_dir),
        experiment_name=experiment_name,
        experiment_summary=experiment_summary,
    )
    copy_if_exists(setup_path, bundle_dir / "experiment_setup")

    baseline_path = resolve_baseline_eval_dir(baseline_eval_dir, Path(eval_outputs_dir))
    copy_if_exists(baseline_path, bundle_dir / "baseline_eval")

    run_log_path = Path(run_logs_dir) / f"{experiment_name}.log"
    copy_if_exists(run_log_path if run_log_path.is_file() else None, bundle_dir / "train.log")

    manifest = {
        "experiment_name": experiment_name,
        "bundle_dir": str(bundle_dir),
        "experiment_source_dir": str(experiment_dir),
        "setup_source_dir": str(setup_path) if setup_path is not None else "",
        "baseline_eval_source_dir": str(baseline_path) if baseline_path is not None else "",
        "train_log_source": str(run_log_path) if run_log_path.is_file() else "",
        "copied_entries": {
            "experiment": str(bundle_dir / "experiment"),
            "experiment_setup": str(bundle_dir / "experiment_setup") if (bundle_dir / "experiment_setup").is_dir() else "",
            "baseline_eval": str(bundle_dir / "baseline_eval") if (bundle_dir / "baseline_eval").is_dir() else "",
            "train_log": str(bundle_dir / "train.log") if (bundle_dir / "train.log").is_file() else "",
        },
        "files": sorted(
            str(path.relative_to(bundle_dir))
            for path in bundle_dir.rglob("*")
            if path.is_file()
        ),
    }
    write_json(bundle_dir / "manifest.json", manifest)

    return bundle_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle one experiment into experiment_results")
    parser.add_argument("--experiment-name", default="", help="Experiment name under the experiment base directory")
    parser.add_argument("--experiment-base-dir", default="experiments", help="Directory containing experiment runs")
    parser.add_argument("--setup-base-dir", default="experiment_setup", help="Directory containing experiment setup folders")
    parser.add_argument("--eval-outputs-dir", default="eval_outputs", help="Directory containing baseline eval outputs")
    parser.add_argument("--run-logs-dir", default="run_logs", help="Directory containing training logs")
    parser.add_argument("--bundle-base-dir", default="experiment_results", help="Directory where results bundles are written")
    parser.add_argument("--baseline-eval-dir", default="", help="Optional explicit baseline eval directory")
    parser.add_argument("--bundle-name", default="", help="Optional explicit bundle directory name")
    parser.add_argument("--setup-dir", default="", help="Optional explicit experiment setup directory")
    args = parser.parse_args()

    bundle_dir = build_bundle(
        experiment_name=args.experiment_name,
        experiment_base_dir=args.experiment_base_dir,
        setup_base_dir=args.setup_base_dir,
        eval_outputs_dir=args.eval_outputs_dir,
        run_logs_dir=args.run_logs_dir,
        bundle_base_dir=args.bundle_base_dir,
        baseline_eval_dir=args.baseline_eval_dir,
        bundle_name=args.bundle_name,
        setup_dir=args.setup_dir,
    )

    print(f"[vlmintune] Created results bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
