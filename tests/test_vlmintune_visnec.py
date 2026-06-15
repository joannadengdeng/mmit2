import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vlmintune.data.types import CanonicalSample
from vlmintune.data.visnec import VisNecFilteredAdapter, load_visnec_selection


class _FakeAdapter:
    streaming = False
    dataset_name = "fake"
    split = "train"
    mapping = {}
    profile = {}

    def __init__(self):
        self.getitem_calls = 0
        self.samples = [
            CanonicalSample(id="a", image_path="", question="qa"),
            CanonicalSample(id="b", image_path="", question="qb"),
            CanonicalSample(id="c", image_path="", question="qc"),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        self.getitem_calls += 1
        return self.samples[idx]

    def get_sample_id(self, idx):
        return self.samples[idx].id

    def __iter__(self):
        return iter(self.samples)


def test_visnec_selection_keeps_highest_scoring_ids(tmp_path):
    score_file = tmp_path / "scores.jsonl"
    records = [
        {"sample_id": "a", "visnec_score": 0.1},
        {"sample_id": "b", "visnec_score": 0.9},
        {"sample_id": "c", "visnec_score": 0.8},
    ]
    score_file.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    selection = load_visnec_selection(str(score_file), top_ratio=2 / 3)

    assert selection.selected_ids == {"b", "c"}


def test_visnec_filtered_adapter_preserves_dataset_order(tmp_path):
    score_file = tmp_path / "scores.jsonl"
    score_file.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "a", "score": 0.1}),
                json.dumps({"sample_id": "b", "score": 0.9}),
                json.dumps({"sample_id": "c", "score": 0.8}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    base_adapter = _FakeAdapter()
    adapter = VisNecFilteredAdapter(
        base_adapter,
        load_visnec_selection(str(score_file), top_ratio=2 / 3),
    )

    assert len(adapter) == 2
    assert base_adapter.getitem_calls == 0
    assert [sample.id for sample in adapter] == ["b", "c"]
    assert adapter[0].id == "b"


def test_visnec_filtered_adapter_applies_max_samples_after_filtering(tmp_path):
    score_file = tmp_path / "scores.jsonl"
    score_file.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "a", "score": 0.1}),
                json.dumps({"sample_id": "b", "score": 0.9}),
                json.dumps({"sample_id": "c", "score": 0.8}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    adapter = VisNecFilteredAdapter(
        _FakeAdapter(),
        load_visnec_selection(str(score_file), top_ratio=2 / 3),
        max_samples=1,
    )

    assert len(adapter) == 1
    assert [sample.id for sample in adapter] == ["b"]
