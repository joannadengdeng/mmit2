import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


_FULLRUN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "mmit2",
    "fullrun.py",
)
_SPEC = importlib.util.spec_from_file_location("mmit2_fullrun_test", _FULLRUN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_parse_eval_targets = _MODULE._parse_eval_targets


def test_parse_eval_targets_applies_defaults():
    targets = _parse_eval_targets(
        {
            "max_new_tokens": 12,
            "targets": [
                {
                    "type": "hf_vqa",
                    "dataset_name": "lmms-lab/VQAv2",
                },
                {
                    "type": "pope_hf",
                },
                {
                    "type": "mme_hf",
                    "name": "mme_custom",
                    "max_new_tokens": 8,
                },
            ],
        }
    )

    assert [target.type for target in targets] == ["hf_vqa", "pope_hf", "mme_hf"]
    assert targets[0].split == "validation"
    assert targets[0].max_new_tokens == 12
    assert targets[0].dataset_name == "lmms-lab/VQAv2"
    assert targets[1].dataset_name == "lmms-lab/POPE"
    assert targets[1].split == "test"
    assert targets[1].task_type == "yes_no"
    assert targets[2].dataset_name == "lmms-lab/MME"
    assert targets[2].name == "mme_custom"
    assert targets[2].max_new_tokens == 8


def test_parse_eval_targets_rejects_unknown_type():
    with pytest.raises(ValueError):
        _parse_eval_targets(
            {
                "targets": [
                    {
                        "type": "unknown_target",
                        "dataset_name": "foo/bar",
                    }
                ]
            }
        )
