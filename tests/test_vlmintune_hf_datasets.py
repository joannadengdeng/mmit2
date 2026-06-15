import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.hf_datasets import HFDatasetsAdapter


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
