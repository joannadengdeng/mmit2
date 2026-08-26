import os
from pathlib import Path
import re
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
QWEN_STAGE_SCRIPTS = (
    ROOT / "scripts" / "run_qwen_dataset_stage.sh",
    ROOT / "scripts" / "run_qwen_textvqa_stage.sh",
)
LLAVA_STAGE_SCRIPT = ROOT / "scripts" / "run_llava_textvqa_stage.sh"
RUNNER = ROOT / "experiment_setup" / "paper_benchmark" / "run_paper_benchmark.sh"
SMOKE_RUNNER = ROOT / "experiment_setup" / "paper_benchmark" / "run_smoke_all.sh"
MORES_LORA_PIPELINE = ROOT / "scripts" / "run_qwen_textvqa_mores_lora_pipeline.sh"
COMBINATION_AUDIT = ROOT / "scripts" / "audit_qwen_textvqa_combination_stage.py"
COMBINATION_STAGE = ROOT / "scripts" / "run_qwen_textvqa_combination_stage.sh"

COMBINATIONS = {
    "mores_lora",
    "mores_dora",
    "reft_lora",
}
RELEASE_RECIPES = {
    "lora",
    "qlora",
    "dora",
    "reft",
    "mores",
    "vl_adapter",
    "l2t",
    *COMBINATIONS,
}


def _extract_shell_function(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n.*?^\}}\n",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing shell function {name} in {path}"
    return match.group(0)


def _call_shell_function(
    path: Path,
    function_names: tuple[str, ...],
    command: str,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    definitions = "\n".join(
        _extract_shell_function(path, name) for name in function_names
    )
    script = f"set -euo pipefail\n{definitions}\n{command}\n"
    process_env = os.environ.copy()
    process_env.update(env or {})
    return subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env=process_env,
        check=False,
    )


@pytest.mark.parametrize(
    ("script", "expected_default"),
    [
        (
            ROOT / "scripts" / "run_qwen_dataset_stage.sh",
            'METHODS="${METHODS:-lora mores reft dora vl_adapter qlora l2t}"',
        ),
        (
            ROOT / "scripts" / "run_qwen_textvqa_stage.sh",
            'METHODS="${METHODS:-lora mores reft dora vl_adapter qlora l2t}"',
        ),
        (
            ROOT / "scripts" / "run_qwen_vqav2_pipeline.sh",
            'METHODS="${METHODS:-lora mores reft dora vl_adapter qlora l2t}"',
        ),
        (
            ROOT / "scripts" / "run_qwen_other_datasets_smoke.sh",
            'METHODS="${METHODS:-lora mores reft dora vl_adapter qlora l2t}"',
        ),
        (
            ROOT / "scripts" / "run_llava_textvqa_stage.sh",
            'METHODS="${METHODS:-lora mores reft dora qlora l2t}"',
        ),
        (
            ROOT / "scripts" / "run_llava_textvqa_pipeline.sh",
            'METHODS="${METHODS:-lora mores reft dora qlora l2t}"',
        ),
    ],
)
def test_remote_pipeline_defaults_do_not_silently_add_combinations(
    script,
    expected_default,
):
    method_lines = [
        line
        for line in script.read_text(encoding="utf-8").splitlines()
        if line.startswith("METHODS=")
    ]

    assert method_lines == [expected_default]
    assert all(combo not in method_lines[0] for combo in COMBINATIONS)


def test_textvqa_stage_supports_mores_lora_without_changing_default_methods():
    marker = _call_shell_function(
        ROOT / "scripts" / "run_qwen_textvqa_stage.sh",
        ("checkpoint_marker",),
        "checkpoint_marker mores_lora",
    )
    rate = _call_shell_function(
        ROOT / "scripts" / "run_qwen_textvqa_stage.sh",
        ("method_learning_rate",),
        "method_learning_rate mores_lora",
        env={"LEARNING_RATE_DEFAULT": "structural-default"},
    )

    assert marker.returncode == 0, marker.stderr
    assert marker.stdout.strip() == "adapter_config.json"
    assert rate.returncode == 0, rate.stderr
    assert rate.stdout.strip() == "structural-default"


def test_textvqa_stage_requires_both_mores_lora_weight_formats(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    checkpoint_dir = shlex.quote(str(tmp_path))
    command = (
        f"checkpoint_files_exist {checkpoint_dir} mores_lora adapter_config.json"
    )

    missing_both = _call_shell_function(
        ROOT / "scripts" / "run_qwen_textvqa_stage.sh",
        ("checkpoint_files_exist",),
        command,
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter weights")
    missing_mores = _call_shell_function(
        ROOT / "scripts" / "run_qwen_textvqa_stage.sh",
        ("checkpoint_files_exist",),
        command,
    )
    (tmp_path / "mores_tuned.pt").write_bytes(b"mores weights")
    complete = _call_shell_function(
        ROOT / "scripts" / "run_qwen_textvqa_stage.sh",
        ("checkpoint_files_exist",),
        command,
    )

    assert missing_both.returncode == 1
    assert missing_mores.returncode == 1
    assert complete.returncode == 0


def test_minimal_runner_whitelists_exactly_the_fixed_release_recipes():
    text = RUNNER.read_text(encoding="utf-8")
    match = re.search(r'case "\$METHOD" in\n\s+([^)]*)\) ;;', text)

    assert match is not None
    assert set(match.group(1).split("|")) == RELEASE_RECIPES
    assert '"$METHOD" == "vl_adapter"' in text
    assert '"$MODEL" != "qwen25vl_3b_instruct"' in text


def test_configuration_smoke_covers_all_ten_fixed_recipes():
    text = SMOKE_RUNNER.read_text(encoding="utf-8")
    match = re.search(r"for method in\s+(.*?); do", text, flags=re.DOTALL)

    assert match is not None
    recipes = set(match.group(1).replace("\\", "").split())
    assert recipes == RELEASE_RECIPES
    assert "all ten release recipes" in text
    assert '"$method" == "vl_adapter"' in text


def test_mores_lora_has_a_separate_progressive_pipeline_and_shared_lock():
    text = MORES_LORA_PIPELINE.read_text(encoding="utf-8")

    assert 'METHODS="mores_lora"' in text
    assert 'STAGES="8:8 256:32 1000:100 34602:5000"' in text
    assert 'RUN_PREFIX="qwen_textvqa_combo"' in text
    assert 'LEARNING_RATE_DEFAULT=2e-4' in text
    assert 'qwen_textvqa_combinations.lock' in text
    assert "flock -n 9" in text
    assert "run_qwen_textvqa_combination_stage.sh" in text
    assert "audit_qwen_textvqa_combination_stage.py" in text


def test_combination_audit_requires_strict_textvqa_identity_and_scores():
    text = COMBINATION_AUDIT.read_text(encoding="utf-8")

    for required in (
        '"source": "trained"',
        '"model_name": "qwen25vl_3b_instruct"',
        '"dataset_name": "lmms-lab/textvqa"',
        '"split": "validation"',
        '"metric": "vqa_accuracy"',
        '"sample_seed": seed',
        '"shuffle_buffer_size": 10_000',
        '"vqa_accuracy" not in scores',
        "duplicate prediction ids",
        "eval_ids do not match prediction ids",
    ):
        assert required in text


def test_combination_stage_has_strict_checkpoint_and_eval_only_recovery():
    text = COMBINATION_STAGE.read_text(encoding="utf-8")

    assert 'audit_method checkpoint "$method"' in text
    assert 'if audit_method all "$method"; then' in text
    assert "Checkpoint artifacts are incomplete or invalid" in text
    assert "No training or evaluation output was overwritten" in text
    assert 'mv "$eval_dir" "$invalid_eval_dir"' in text
    assert 'mv "$eval_config" "$invalid_eval_config"' in text
    assert "Preserved prior evaluation directory" in text
    assert "DRY_RUN=0" in text


@pytest.mark.parametrize(
    "script",
    [
        *QWEN_STAGE_SCRIPTS,
        LLAVA_STAGE_SCRIPT,
        RUNNER,
        SMOKE_RUNNER,
        MORES_LORA_PIPELINE,
        COMBINATION_STAGE,
    ],
)
def test_combination_runner_scripts_have_valid_bash_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
