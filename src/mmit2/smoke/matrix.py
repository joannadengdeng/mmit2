"""Run a tiny smoke-test matrix across multiple mmit2 training paths.

This entry point is designed for path coverage, not metric quality. It
intentionally uses very small train/eval slices so that we can quickly validate
multiple training and debug flows from a single command.

Typical usage:
    python -m mmit2.smoke --suite quick
    python -m mmit2.smoke --suite full
    python -m mmit2.smoke --scenario freeze_debug_tiny
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List

from mmit2.config.training_config import config_to_trainer_dict, load_config


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str  # "debug" | "cli"
    config: str
    description: str
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    name: str
    kind: str
    config: str
    description: str
    command: list[str]
    status: str
    returncode: int
    duration_s: float
    output_dir: str
    final_checkpoint: str
    debug_report: str


SCENARIOS: dict[str, Scenario] = {
    "inspect_lora_debug": Scenario(
        name="inspect_lora_debug",
        kind="debug",
        config="configs/colab_lora_tiny_smoke.yaml",
        description="Only inspect preprocessor/runtime path; no model training.",
        extra_args=["--inspect-only"],
    ),
    "lora_debug_tiny": Scenario(
        name="lora_debug_tiny",
        kind="debug",
        config="configs/colab_lora_tiny_smoke.yaml",
        description="Full LoRA debug flow: inspect, pretrain eval, train, reload, post-train eval.",
        extra_args=["--eval-count", "4", "--max-new-tokens", "8"],
    ),
    "qlora_cli_tiny": Scenario(
        name="qlora_cli_tiny",
        kind="cli",
        config="configs/colab_qlora_smoke.yaml",
        description="Headless Trainer/CLI path with 4-bit quantized base model.",
    ),
    "dora_cli_tiny": Scenario(
        name="dora_cli_tiny",
        kind="cli",
        config="configs/colab_dora_smoke.yaml",
        description="Headless Trainer/CLI path for DoRA adapter injection.",
    ),
    "freeze_debug_tiny": Scenario(
        name="freeze_debug_tiny",
        kind="debug",
        config="configs/colab_freeze_smoke.yaml",
        description="Freeze Tuning path with custom save/load and post-train eval.",
        extra_args=["--eval-count", "4", "--max-new-tokens", "8"],
    ),
    "l2t_lora_debug_tiny": Scenario(
        name="l2t_lora_debug_tiny",
        kind="debug",
        config="configs/colab_l2t_lora_smoke.yaml",
        description="L2T label-rewrite path on top of LoRA, with pre/post eval.",
        extra_args=["--eval-count", "4", "--max-new-tokens", "8"],
    ),
    "l2t_qlora_cli_tiny": Scenario(
        name="l2t_qlora_cli_tiny",
        kind="cli",
        config="configs/colab_l2t_qlora_smoke.yaml",
        description="Headless Trainer/CLI path for L2T composed with QLoRA.",
    ),
}

SUITES: dict[str, list[str]] = {
    "quick": [
        "inspect_lora_debug",
        "lora_debug_tiny",
        "qlora_cli_tiny",
        "freeze_debug_tiny",
        "l2t_lora_debug_tiny",
    ],
    "full": [
        "inspect_lora_debug",
        "lora_debug_tiny",
        "qlora_cli_tiny",
        "dora_cli_tiny",
        "freeze_debug_tiny",
        "l2t_lora_debug_tiny",
        "l2t_qlora_cli_tiny",
    ],
}


def _discover_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    return cwd


ROOT = _discover_repo_root()


def _resolve_paths(config_path: str, debug_mode: bool) -> tuple[str, str, str]:
    cfg = load_config(config_path)
    trainer_dict = config_to_trainer_dict(cfg)
    output_dir = trainer_dict["training"]["output_dir"]
    if debug_mode and cfg.runtime.mode == "colab" and cfg.runtime.colab.output_to_drive:
        output_dir = os.path.join(
            cfg.runtime.colab.drive_mount_point,
            "MyDrive",
            "mmit2_output",
            output_dir,
        )
    final_checkpoint = os.path.join(output_dir, "final")
    debug_report = os.path.join(output_dir, "debug_report.json")
    return output_dir, final_checkpoint, debug_report


def _build_command(scenario: Scenario) -> list[str]:
    config_path = str(ROOT / scenario.config)
    if scenario.kind == "debug":
        return [
            sys.executable,
            "-m",
            "mmit2.debug",
            "--config",
            config_path,
            *scenario.extra_args,
        ]
    if scenario.kind == "cli":
        return [
            sys.executable,
            "-m",
            "mmit2.training",
            "--config",
            config_path,
            *scenario.extra_args,
        ]
    raise ValueError(f"Unknown scenario kind: {scenario.kind}")


def _iter_scenarios(suite: str, scenario_names: list[str]) -> Iterable[Scenario]:
    if scenario_names:
        for name in scenario_names:
            try:
                yield SCENARIOS[name]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown scenario '{name}'. Available: {sorted(SCENARIOS)}"
                ) from exc
        return

    try:
        names = SUITES[suite]
    except KeyError as exc:
        raise KeyError(f"Unknown suite '{suite}'. Available: {sorted(SUITES)}") from exc
    for name in names:
        yield SCENARIOS[name]


def _run_command(command: list[str], cwd: Path) -> int:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return process.wait()


def _summary_path() -> Path:
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return out_dir / f"colab_smoke_matrix_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny smoke-test matrix")
    parser.add_argument(
        "--suite",
        default="quick",
        choices=sorted(SUITES),
        help="Predefined scenario set to run",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Run only the named scenario(s); may be passed multiple times",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop immediately after the first failing scenario",
    )
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for scenario in SCENARIOS.values():
            print(f"- {scenario.name}: {scenario.description} [{scenario.kind}]")
        print()
        print("Suites:")
        for name, scenario_names in SUITES.items():
            print(f"- {name}: {', '.join(scenario_names)}")
        return

    scenarios = list(_iter_scenarios(args.suite, args.scenario))
    if not scenarios:
        raise ValueError("No scenarios selected")

    results: List[ScenarioResult] = []
    total_start = time.time()

    print("=" * 80)
    print("mmit2 Smoke Matrix")
    print("=" * 80)
    print(f"Repository root: {ROOT}")
    print(f"Selected scenarios: {[scenario.name for scenario in scenarios]}")
    print()

    for idx, scenario in enumerate(scenarios, start=1):
        config_path = str(ROOT / scenario.config)
        output_dir, final_checkpoint, debug_report = _resolve_paths(
            config_path,
            debug_mode=(scenario.kind == "debug"),
        )
        command = _build_command(scenario)

        print("=" * 80)
        print(f"[{idx}/{len(scenarios)}] {scenario.name}")
        print("=" * 80)
        print(f"Kind: {scenario.kind}")
        print(f"Config: {config_path}")
        print(f"Description: {scenario.description}")
        print(f"Expected output_dir: {output_dir}")
        if scenario.kind == "debug":
            print(f"Expected debug report: {debug_report}")
        else:
            print(f"Expected final checkpoint: {final_checkpoint}")
        print(f"Command: {' '.join(command)}")
        print("-" * 80)

        start = time.time()
        returncode = _run_command(command, cwd=ROOT)
        duration_s = time.time() - start
        status = "passed" if returncode == 0 else "failed"

        result = ScenarioResult(
            name=scenario.name,
            kind=scenario.kind,
            config=config_path,
            description=scenario.description,
            command=command,
            status=status,
            returncode=returncode,
            duration_s=round(duration_s, 1),
            output_dir=output_dir,
            final_checkpoint=final_checkpoint,
            debug_report=debug_report,
        )
        results.append(result)

        print("-" * 80)
        print(
            f"Scenario {scenario.name}: {status.upper()} "
            f"(returncode={returncode}, duration={duration_s:.1f}s)"
        )
        print()

        if returncode != 0 and args.stop_on_failure:
            break

    passed = sum(1 for result in results if result.status == "passed")
    failed = len(results) - passed
    total_duration = round(time.time() - total_start, 1)

    summary = {
        "suite": args.suite,
        "selected_scenarios": [scenario.name for scenario in scenarios],
        "passed": passed,
        "failed": failed,
        "total_duration_s": total_duration,
        "results": [asdict(result) for result in results],
    }
    summary_path = _summary_path()
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Smoke Matrix Summary")
    print("=" * 80)
    for result in results:
        print(
            f"- {result.name}: {result.status} "
            f"(kind={result.kind}, duration={result.duration_s}s)"
        )
    print()
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total duration: {total_duration}s")
    print(f"Summary JSON: {summary_path}")

    if failed:
        sys.exit(1)
