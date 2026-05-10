import importlib.util
import os
import sys
import types

import pytest


_ADAPTER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "mmit2",
    "data",
    "adapters",
    "hf_datasets.py",
)


def _load_adapter_module():
    saved_modules = {}
    module_names = [
        "datasets",
        "mmit2.data.adapters.base",
        "mmit2.data.datasets",
        "mmit2.data.types",
        "mmit2_test_hf_adapter",
    ]
    for name in module_names:
        saved_modules[name] = sys.modules.get(name)

    datasets_mod = types.ModuleType("datasets")

    base_mod = types.ModuleType("mmit2.data.adapters.base")
    base_mod.DatasetAdapter = object

    data_datasets_mod = types.ModuleType("mmit2.data.datasets")
    data_datasets_mod.DATASET_SPECS = {}
    data_datasets_mod.ColumnMapping = object
    data_datasets_mod.HFDatasetSpec = object
    data_datasets_mod.build_configured_spec = lambda *args, **kwargs: None
    data_datasets_mod.get_dataset_spec = lambda *args, **kwargs: None

    data_types_mod = types.ModuleType("mmit2.data.types")
    data_types_mod.CanonicalSample = object

    sys.modules["datasets"] = datasets_mod
    sys.modules["mmit2.data.adapters.base"] = base_mod
    sys.modules["mmit2.data.datasets"] = data_datasets_mod
    sys.modules["mmit2.data.types"] = data_types_mod

    spec = importlib.util.spec_from_file_location("mmit2_test_hf_adapter", _ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, saved_modules


def _restore_modules(saved_modules):
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_rejects_unavailable_requested_split():
    module, saved_modules = _load_adapter_module()

    class _SplitInfo:
        def __init__(self, num_examples):
            self.num_examples = num_examples

    class _Builder:
        info = types.SimpleNamespace(
            splits={
                "validation": _SplitInfo(214354),
                "test": _SplitInfo(447793),
            }
        )

    class _DatasetsMod:
        @staticmethod
        def load_dataset_builder(*args):
            return _Builder()

        @staticmethod
        def load_dataset(*args, **kwargs):
            raise AssertionError("load_dataset should not be called for an unavailable split")

    try:
        adapter = module.HFDatasetsAdapter.__new__(module.HFDatasetsAdapter)
        adapter.dataset_name = "lmms-lab/VQAv2"
        with pytest.raises(ValueError, match="Requested split 'train'"):
            adapter._load_dataset(_DatasetsMod, ("lmms-lab/VQAv2",), "train", True, True)
    finally:
        _restore_modules(saved_modules)


def test_does_not_fallback_to_other_split_when_requested_split_fails():
    module, saved_modules = _load_adapter_module()

    class _SplitInfo:
        def __init__(self, num_examples):
            self.num_examples = num_examples

    class _Builder:
        info = types.SimpleNamespace(
            splits={
                "train": _SplitInfo(443757),
                "validation": _SplitInfo(214354),
            }
        )

    calls = []

    class _DatasetsMod:
        @staticmethod
        def load_dataset_builder(*args):
            return _Builder()

        @staticmethod
        def load_dataset(*args, **kwargs):
            calls.append(kwargs["split"])
            raise RuntimeError("train split failed")

    try:
        adapter = module.HFDatasetsAdapter.__new__(module.HFDatasetsAdapter)
        adapter.dataset_name = "lmms-lab/VQAv2"
        with pytest.raises(RuntimeError, match="Failed to load dataset 'lmms-lab/VQAv2' split 'train'"):
            adapter._load_dataset(_DatasetsMod, ("lmms-lab/VQAv2",), "train", True, True)
        assert calls
        assert set(calls) == {"train"}
    finally:
        _restore_modules(saved_modules)
