import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.hf_datasets import HFDatasetsAdapter
import vlmintune.data.hf_datasets as hf_datasets_module


def test_vqav2_loads_only_the_requested_split_files(monkeypatch):
    calls = []

    class FakeDatasets:
        @staticmethod
        def load_dataset_builder(*args, **kwargs):
            calls.append(("builder", args, kwargs))
            return SimpleNamespace(info=SimpleNamespace(splits={}))

        @staticmethod
        def load_dataset(*args, **kwargs):
            calls.append(("dataset", args, kwargs))
            return []

    monkeypatch.setattr(hf_datasets_module, "datasets", FakeDatasets)

    HFDatasetsAdapter(
        dataset_name="pingzhili/vqa_v2",
        split="validation",
        max_samples=1,
        load_images=False,
    )

    expected = {"validation": "data/validation-*"}
    assert calls[0][0] == "builder"
    assert calls[0][2]["data_files"] == expected
    assert calls[1][0] == "dataset"
    assert calls[1][2]["data_files"] == expected


def _make_vqav2_snapshot(tmp_path, revision="test-revision"):
    snapshot = (
        tmp_path
        / "hub"
        / "datasets--pingzhili--vqa_v2"
        / "snapshots"
        / revision
    )
    (snapshot / "data").mkdir(parents=True)
    (snapshot / "README.md").write_text("cached", encoding="utf-8")
    return snapshot


def test_vqav2_local_snapshot_uses_parquet_loader(monkeypatch, tmp_path):
    snapshot = _make_vqav2_snapshot(tmp_path)
    shard_a = snapshot / "data" / "validation-00001-of-00002.parquet"
    shard_b = snapshot / "data" / "validation-00000-of-00002.parquet"
    shard_a.write_bytes(b"a")
    shard_b.write_bytes(b"b")
    calls = []

    class FakeDatasets:
        @staticmethod
        def load_dataset_builder(*args, **kwargs):
            calls.append(("builder", args, kwargs))
            return SimpleNamespace(info=SimpleNamespace(splits={}))

        @staticmethod
        def load_dataset(*args, **kwargs):
            calls.append(("dataset", args, kwargs))
            return []

    def fake_snapshot_download(**kwargs):
        assert kwargs["repo_id"] == "pingzhili/vqa_v2"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == "test-revision"
        assert kwargs["local_files_only"] is True
        assert kwargs["allow_patterns"] == ["README.md", "data/validation-*"]
        return str(snapshot)

    monkeypatch.setenv("VLMINTUNE_VQAV2_SNAPSHOT", str(snapshot))
    monkeypatch.setattr(hf_datasets_module, "datasets", FakeDatasets)
    monkeypatch.setattr(hf_datasets_module, "snapshot_download", fake_snapshot_download)

    HFDatasetsAdapter(
        dataset_name="pingzhili/vqa_v2",
        split="validation",
        max_samples=1,
        load_images=False,
    )

    expected_files = [str(shard_b.absolute()), str(shard_a.absolute())]
    assert calls[0][1] == ("parquet",)
    assert calls[0][2]["data_files"] == {"validation": expected_files}
    assert calls[1][1] == ("parquet",)
    assert calls[1][2]["data_files"] == {"validation": expected_files}


def test_vqav2_local_snapshot_missing_split_fails_without_hub_fallback(
    monkeypatch, tmp_path
):
    snapshot = _make_vqav2_snapshot(tmp_path)
    dataset_calls = []

    monkeypatch.setenv("VLMINTUNE_VQAV2_SNAPSHOT", str(snapshot))
    monkeypatch.setattr(
        hf_datasets_module,
        "snapshot_download",
        lambda **kwargs: str(snapshot),
    )
    monkeypatch.setattr(
        hf_datasets_module.datasets,
        "load_dataset",
        lambda *args, **kwargs: dataset_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="no non-empty Parquet shards"):
        HFDatasetsAdapter(
            dataset_name="pingzhili/vqa_v2",
            split="validation",
            max_samples=1,
            load_images=False,
        )

    assert dataset_calls == []


def test_vqav2_local_parquet_streaming_integration(monkeypatch, tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    snapshot = _make_vqav2_snapshot(tmp_path)
    shard = snapshot / "data" / "train-00000-of-00001.parquet"
    parquet.write_table(
        pyarrow.table({"question_id": [7], "question": ["What is shown?"]}),
        shard,
    )

    monkeypatch.setenv("VLMINTUNE_VQAV2_SNAPSHOT", str(snapshot))
    monkeypatch.setattr(
        hf_datasets_module,
        "snapshot_download",
        lambda **kwargs: str(snapshot),
    )

    adapter = HFDatasetsAdapter(
        dataset_name="pingzhili/vqa_v2",
        split="train",
        max_samples=1,
        load_images=False,
    )

    assert adapter.streaming is True
    assert next(iter(adapter._hf_dataset))["question_id"] == 7


def test_other_datasets_do_not_override_data_files(monkeypatch):
    calls = []

    class FakeDatasets:
        @staticmethod
        def load_dataset_builder(*args, **kwargs):
            calls.append(("builder", args, kwargs))
            return SimpleNamespace(info=SimpleNamespace(splits={}))

        @staticmethod
        def load_dataset(*args, **kwargs):
            calls.append(("dataset", args, kwargs))
            return []

    monkeypatch.setattr(hf_datasets_module, "datasets", FakeDatasets)

    HFDatasetsAdapter(
        dataset_name="lmms-lab/textvqa",
        split="validation",
        max_samples=1,
        load_images=False,
    )

    assert "data_files" not in calls[0][2]
    assert "data_files" not in calls[1][2]


def test_limited_sample_training_prefers_streaming(monkeypatch):
    calls = []

    def fake_load_dataset(self, datasets_mod, load_pos, split, streaming, trust_remote_code):
        calls.append(
            {
                "load_pos": load_pos,
                "split": split,
                "streaming": streaming,
            }
        )
        return []

    monkeypatch.setattr(HFDatasetsAdapter, "load_dataset", fake_load_dataset)

    adapter = HFDatasetsAdapter(
        dataset_name="lmms-lab/textvqa",
        split="train",
        max_samples=100,
        load_images=False,
    )

    assert calls[0]["streaming"] is True
    assert adapter.streaming is True
    assert len(adapter) == 100


def test_omitted_split_uses_dataset_default_for_training(monkeypatch):
    calls = []

    def fake_load_dataset(self, datasets_mod, load_pos, split, streaming, trust_remote_code):
        calls.append(
            {
                "load_pos": load_pos,
                "split": split,
                "streaming": streaming,
            }
        )
        return []

    monkeypatch.setattr(HFDatasetsAdapter, "load_dataset", fake_load_dataset)

    adapter = HFDatasetsAdapter(
        dataset_name="lmms-lab/textvqa",
        max_samples=10,
        load_images=False,
        usage="train",
    )

    assert calls[0]["split"] == "train"
    assert adapter.split == "train"


def test_vizwiz_uses_parquet_dataset_and_val_split_by_default(monkeypatch):
    calls = []

    def fake_load_dataset(self, datasets_mod, load_pos, split, streaming, trust_remote_code):
        calls.append(
            {
                "load_pos": load_pos,
                "split": split,
                "streaming": streaming,
            }
        )
        return []

    monkeypatch.setattr(HFDatasetsAdapter, "load_dataset", fake_load_dataset)

    adapter = HFDatasetsAdapter(
        dataset_name="ebrukilic/vizwiz_vqa_dataset",
        max_samples=10,
        load_images=False,
        usage="train",
    )

    assert calls[0]["load_pos"] == ("ebrukilic/vizwiz_vqa_dataset",)
    assert calls[0]["split"] == "train"
    assert adapter.split == "train"


def test_vizwiz_eval_uses_validation_split(monkeypatch):
    calls = []

    def fake_load_dataset(self, datasets_mod, load_pos, split, streaming, trust_remote_code):
        calls.append(
            {
                "load_pos": load_pos,
                "split": split,
                "streaming": streaming,
            }
        )
        return []

    monkeypatch.setattr(HFDatasetsAdapter, "load_dataset", fake_load_dataset)

    adapter = HFDatasetsAdapter(
        dataset_name="ebrukilic/vizwiz_vqa_dataset",
        max_samples=10,
        load_images=False,
        usage="eval",
    )

    assert calls[0]["load_pos"] == ("ebrukilic/vizwiz_vqa_dataset",)
    assert calls[0]["split"] == "validation"
    assert adapter.split == "validation"


class _FakeDataset(list):
    column_names = []
    features = {}

    def select(self, indices):
        return _FakeDataset([self[idx] for idx in indices])


def test_scienceqa_image_filters_rows_without_images(monkeypatch):
    def fake_load_dataset(self, datasets_mod, load_pos, split, streaming, trust_remote_code):
        assert load_pos == ("derek-thomas/ScienceQA",)
        assert split == "train"
        assert streaming is False
        return _FakeDataset(
            [
                {
                    "pid": "no-image",
                    "image": None,
                    "question": "No image?",
                    "choices": ["yes", "no"],
                    "answer": 0,
                },
                {
                    "pid": "has-image",
                    "image": {"path": "science.png"},
                    "question": "Which object is magnetic?",
                    "choices": ["wood", "iron"],
                    "answer": 1,
                },
            ]
        )

    monkeypatch.setattr(HFDatasetsAdapter, "load_dataset", fake_load_dataset)

    adapter = HFDatasetsAdapter(
        dataset_name="scienceqa_image",
        load_images=False,
        usage="train",
    )

    assert len(adapter) == 1
    sample = adapter[0]
    assert sample.id == "has-image"
    assert sample.train_answer == "1"
    assert sample.eval_answers == ["1"]
