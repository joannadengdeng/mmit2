import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.config.training_config import (
    PUBLIC_CONFIG_FIELDS,
    config_to_trainer_dict,
    load_config,
    load_config_dict,
)
from vlmintune.training.__main__ import parse_train_config
from vlmintune.training.methods.registry import list_training_methods


def test_initial_release_registry_contains_exactly_eight_paper_methods():
    assert set(list_training_methods()) == {
        "lora",
        "qlora",
        "dora",
        "mole",
        "reft",
        "mores",
        "vl_adapter",
        "l2t",
    }


def test_flat_training_config_contains_only_public_v1_fields(tmp_path):
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        """
model: qwen25vl_3b_instruct
dataset: lmms-lab/textvqa
method: qlora
epochs: 2
learning_rate: 0.0001
batch_size: 3
gradient_accumulation_steps: 5
max_length: 1536
max_samples: 17
seed: 7
output_dir: runs/demo
""".strip()
        + "\n",
        encoding="utf-8",
    )

    loaded = config_to_trainer_dict(load_config(str(config_path)))

    assert set(loaded) == PUBLIC_CONFIG_FIELDS
    assert loaded == {
        "model": "qwen25vl_3b_instruct",
        "dataset": "lmms-lab/textvqa",
        "method": "qlora",
        "epochs": 2,
        "learning_rate": 0.0001,
        "batch_size": 3,
        "gradient_accumulation_steps": 5,
        "max_length": 1536,
        "max_samples": 17,
        "seed": 7,
        "output_dir": "runs/demo",
    }


def test_flat_training_config_applies_simple_runtime_defaults():
    config = load_config_dict(
        {
            "model": "qwen25vl_3b_instruct",
            "dataset": "lmms-lab/textvqa",
            "method": "lora",
        }
    )

    assert config.epochs == 1
    assert config.learning_rate == 2e-4
    assert config.batch_size == 4
    assert config.gradient_accumulation_steps == 4
    assert config.max_length == 2048
    assert config.max_samples == 0
    assert config.seed == 42
    assert config.output_dir == "output"


@pytest.mark.parametrize("old_field", ["training", "data", "experiment", "method_params"])
def test_flat_training_config_rejects_removed_fields(old_field):
    raw = {
        "model": "qwen25vl_3b_instruct",
        "dataset": "lmms-lab/textvqa",
        "method": "lora",
        old_field: {},
    }

    with pytest.raises(ValueError, match="Unknown training config fields"):
        load_config_dict(raw)


def test_flat_training_config_requires_model_dataset_and_method():
    with pytest.raises(ValueError) as exc_info:
        load_config_dict({})

    message = str(exc_info.value)
    assert "model: required" in message
    assert "dataset: required" in message
    assert "method: required" in message


@pytest.mark.parametrize(
    ("model", "method", "supported_model"),
    [
        ("qwen25vl_3b_instruct", "mole", "llava15_7b"),
        ("llava15_7b", "vl_adapter", "qwen25vl_3b_instruct"),
    ],
)
def test_flat_training_config_rejects_the_two_fixed_model_mismatches(
    model,
    method,
    supported_model,
):
    with pytest.raises(ValueError, match=supported_model):
        load_config_dict(
            {
                "model": model,
                "dataset": "lmms-lab/textvqa",
                "method": method,
            }
        )


def test_parse_train_config_maps_flat_names_without_method_params():
    model_name, trainer_config = parse_train_config(
        {
            "model": "llava15_7b",
            "dataset": "lmms-lab/textvqa",
            "method": "dora",
            "epochs": 2,
            "learning_rate": 3e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_length": 1024,
            "max_samples": 23,
            "seed": 123,
            "output_dir": "runs/dora",
        }
    )

    assert model_name == "llava15_7b"
    assert trainer_config.data_config == {
        "dataset_name": "lmms-lab/textvqa",
        "max_samples": 23,
        "sample_seed": 123,
    }
    assert trainer_config.training_method == "dora"
    assert trainer_config.num_epochs == 2
    assert trainer_config.per_device_batch_size == 2
    assert trainer_config.gradient_accumulation_steps == 8
    assert trainer_config.max_length == 1024
    assert trainer_config.learning_rate == 3e-5
    assert trainer_config.seed == 123
    assert trainer_config.output_dir == "runs/dora"
    assert not hasattr(trainer_config, "method_params")


def test_flat_training_config_rejects_negative_max_samples():
    with pytest.raises(ValueError, match="max_samples"):
        load_config_dict(
            {
                "model": "qwen25vl_3b_instruct",
                "dataset": "lmms-lab/textvqa",
                "method": "lora",
                "max_samples": -1,
            }
        )
