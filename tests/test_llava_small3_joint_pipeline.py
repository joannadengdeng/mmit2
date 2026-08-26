from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "run_llava_small3_joint_combinations_pipeline.sh"
JOINT_STAGE = ROOT / "scripts" / "run_qwen_joint_combination_dataset_stage.sh"
DATASET_STAGE = ROOT / "scripts" / "run_qwen_dataset_stage.sh"
AUDIT = ROOT / "scripts" / "audit_qwen_joint_combination_dataset_stage.py"


def test_llava_small3_pipeline_has_exact_progressive_matrix_and_order():
    text = PIPELINE.read_text(encoding="utf-8")

    assert 'MODEL="llava15_7b"' in text
    assert 'METHODS="mores_lora mores_dora reft_lora"' in text
    assert 'TEXTVQA_STAGES="8:8 256:32 1000:100 34602:5000"' in text
    assert 'VIZWIZ_STAGES="8:8 256:32 1000:100 20523:4319"' in text
    assert 'SCIENCEQA_STAGES="8:8 256:32 1000:100 6218:2097"' in text
    assert text.index('"lmms-lab/textvqa"') < text.index(
        '"ebrukilic/vizwiz_vqa_dataset"'
    ) < text.index('"scienceqa_image"')
    assert 'MAX_LENGTH=1536' in text
    assert 'EPOCHS=1' in text
    assert 'SEED=42' in text
    assert 'LEARNING_RATE_DEFAULT=2e-4' in text


def test_llava_small3_pipeline_is_offline_isolated_and_pause_guarded():
    text = PIPELINE.read_text(encoding="utf-8")

    for setting in (
        "HF_HUB_OFFLINE=1",
        "HF_HUB_DISABLE_XET=1",
        "TRANSFORMERS_OFFLINE=1",
    ):
        assert setting in text
    assert "paused_by_user_dora_step_11788_of_110940" in text
    assert "verify_idle_gpu_processes" in text
    assert "flock -n 9" in text
    assert "VLMINTUNE_GPU_LOCK_HELD=1" in text
    assert "FORCE=0" in text
    assert "progressive20260822" in text
    assert "llava_textvqa_pipeline" not in text


def test_joint_stage_passes_model_to_generic_stage_and_strict_audit():
    text = JOINT_STAGE.read_text(encoding="utf-8")

    assert 'MODEL="${MODEL:-qwen25vl_3b_instruct}"' in text
    assert "qwen25vl_3b_instruct|llava15_7b" in text
    assert '--model "$MODEL"' in text
    assert 'MODEL="$MODEL"' in text
    assert "lmms-lab/textvqa" in text


def test_generic_dataset_stage_uses_selected_model_for_train_and_eval():
    text = DATASET_STAGE.read_text(encoding="utf-8")

    assert 'MODEL="${MODEL:-qwen25vl_3b_instruct}"' in text
    assert 'MODEL="$MODEL" \\' in text
    assert 'name: "$MODEL"' in text
    assert "models--llava-hf--llava-1.5-7b-hf" in text
    assert "models--Qwen--Qwen2.5-VL-3B-Instruct" in text


def test_strict_audit_accepts_explicit_model_identity():
    text = AUDIT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--model", default="qwen25vl_3b_instruct")' in text
    assert '"model": model' in text
    assert '"model_name": model' in text


def test_new_shell_scripts_parse():
    result = subprocess.run(
        [
            "bash",
            "-n",
            str(PIPELINE),
            str(JOINT_STAGE),
            str(DATASET_STAGE),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_no_unexpanded_qwen_model_in_runtime_blocks():
    text = DATASET_STAGE.read_text(encoding="utf-8")
    hardcoded_runtime = re.findall(
        r"(?:MODEL=|name: )qwen25vl_3b_instruct", text
    )
    assert hardcoded_runtime == []
