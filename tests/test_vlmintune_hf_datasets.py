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
