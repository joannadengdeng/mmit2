"""Hugging Face dataset loader backed by per-dataset VQA specs."""
from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional

import datasets

from vlmintune.data.datasets import (
    DATASET_SPECS,
    ColumnMapping,
    HFDatasetSpec,
    build_configured_spec,
    get_dataset_spec,
)
from vlmintune.data.types import CanonicalSample

DatasetProfile = HFDatasetSpec


class HFDatasetsAdapter:
    """Load a HuggingFace VQA dataset and yield ``CanonicalSample`` rows."""

    def __init__(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        column_map: Optional[ColumnMapping] = None,
        max_samples: Optional[int] = None,
        streaming: bool = False,
        trust_remote_code: bool = True,
        load_images: bool = True,
        config_name: Optional[str] = None,
        usage: str = "train",
        sample_seed: Optional[int] = None,
        shuffle_buffer_size: int = 10_000,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = str(split or "").strip()
        self.usage = usage
        self.max_samples = max_samples
        self.streaming = streaming
        self.load_images = load_images
        self.sample_seed = sample_seed
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self._num_examples: Optional[int] = None

        spec = get_dataset_spec(dataset_name)
        prefer_streaming = spec.prefer_streaming if spec is not None else False
        data_model = getattr(spec, "data_model", None)
        if not self.split:
            if data_model is None:
                raise ValueError(
                    f"Dataset '{dataset_name}' requires an explicit split because no built-in data model was found."
                )
            if usage == "eval":
                self.split = str(data_model.default_eval_split).strip()
            else:
                self.split = str(data_model.default_train_split).strip()
            if not self.split:
                raise ValueError(
                    f"Dataset '{dataset_name}' does not define a default split for usage='{usage}'."
                )

        resolved_dataset_name = (
            data_model.resolved_hf_dataset_name if data_model is not None else dataset_name
        )
        self.hf_dataset_name = resolved_dataset_name
        config_arg = config_name or (data_model.config_name if data_model is not None else "")
        load_pos = (resolved_dataset_name, config_arg) if config_arg else (resolved_dataset_name,)

        # For initial-release smoke runs, loading a fixed subset should not force
        # downloading the full dataset shards before sampling.
        if not streaming and (prefer_streaming or max_samples is not None):
            streaming = True
            self.streaming = True

        self._hf_dataset = self.load_dataset(
            datasets,
            load_pos,
            self.split,
            streaming,
            trust_remote_code,
        )

        if sample_seed is not None:
            self.apply_seeded_sampling(int(sample_seed))

        if not self.load_images and not self.streaming:
            self.disable_eager_image_decode(datasets)

        if column_map is not None:
            self._spec = build_configured_spec(
                dataset_name,
                column_map,
                prefer_streaming=prefer_streaming,
            )
        elif spec is not None:
            self._spec = spec
        else:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                "Use one of the built-in dataset specs or pass column_map explicitly. "
                f"Built-ins: {sorted(DATASET_SPECS)}"
            )

        if not self.streaming and self.dataset_name == "scienceqa_image":
            keep_indices = [
                idx
                for idx, row in enumerate(self._hf_dataset)
                if self._scienceqa_image_has_image(row)
            ]
            if max_samples is not None:
                keep_indices = keep_indices[:max_samples]
            self._hf_dataset = self._hf_dataset.select(keep_indices)
        elif max_samples is not None and not self.streaming:
            self._hf_dataset = self._hf_dataset.select(range(min(max_samples, len(self._hf_dataset))))

    def apply_seeded_sampling(self, sample_seed: int) -> None:
        if self.streaming:
            if hasattr(self._hf_dataset, "shuffle"):
                self._hf_dataset = self._hf_dataset.shuffle(
                    seed=sample_seed,
                    buffer_size=max(1, self.shuffle_buffer_size),
                )
            return

        total = len(self._hf_dataset)
        indices = list(range(total))
        random.Random(sample_seed).shuffle(indices)
        self._hf_dataset = self._hf_dataset.select(indices)

    def load_dataset(
        self,
        datasets_mod,
        load_pos: tuple,
        split: str,
        streaming: bool,
        trust_remote_code: bool,
    ):
        splits_to_try = [split]
        split_sizes: Dict[str, int] = {}
        available: List[str] = []
        try:
            try:
                ds_info = datasets_mod.load_dataset_builder(
                    *load_pos,
                    trust_remote_code=trust_remote_code,
                ).info
            except TypeError:
                ds_info = datasets_mod.load_dataset_builder(*load_pos).info
            if ds_info.splits:
                split_sizes = {
                    name: int(split_info.num_examples)
                    for name, split_info in ds_info.splits.items()
                    if split_info.num_examples is not None
                }
            available = list(ds_info.splits.keys()) if ds_info.splits else []
        except Exception:
            pass

        if available and split not in available:
            raise ValueError(
                f"Requested split '{split}' is not available for dataset '{self.dataset_name}'. "
                f"Available splits: {available}"
            )

        first_err = None
        for try_split in splits_to_try:
            load_kwargs: Dict[str, object] = {"split": try_split, "streaming": streaming}
            for kwargs in (load_kwargs, {**load_kwargs, "trust_remote_code": trust_remote_code}):
                try:
                    dataset = datasets_mod.load_dataset(*load_pos, **kwargs)
                    self.split = try_split
                    self._num_examples = split_sizes.get(try_split)
                    return dataset
                except Exception as exc:
                    if first_err is None:
                        first_err = exc

        if not streaming:
            for try_split in splits_to_try:
                for kwargs in (
                    {"split": try_split, "streaming": True},
                    {"split": try_split, "streaming": True, "trust_remote_code": trust_remote_code},
                ):
                    try:
                        dataset = datasets_mod.load_dataset(*load_pos, **kwargs)
                        self.streaming = True
                        self.split = try_split
                        self._num_examples = split_sizes.get(try_split)
                        return dataset
                    except Exception:
                        pass

        if first_err is not None:
            raise RuntimeError(
                f"Failed to load dataset '{self.dataset_name}' split '{split}' "
                f"(streaming={streaming})."
            ) from first_err
        raise RuntimeError(
            f"Failed to load dataset '{self.dataset_name}' split '{split}' "
            f"(streaming={streaming})."
        )

    def disable_eager_image_decode(self, datasets_mod) -> None:
        try:
            for col_name in self._hf_dataset.column_names:
                feat = self._hf_dataset.features.get(col_name)
                if isinstance(feat, datasets_mod.Image):
                    self._hf_dataset = self._hf_dataset.cast_column(
                        col_name,
                        datasets_mod.Image(decode=False),
                    )
        except Exception:
            pass

    def _scienceqa_image_has_image(self, row: dict) -> bool:
        image_val = row.get("image")
        if image_val is None:
            return False
        if isinstance(image_val, dict):
            return bool(image_val.get("bytes") or image_val.get("path"))
        return bool(image_val)

    def __len__(self) -> int:
        if self.streaming:
            if self.max_samples is not None:
                if self._num_examples is not None:
                    return min(self.max_samples, self._num_examples)
                return self.max_samples
            if self._num_examples is not None:
                return self._num_examples
            return -1
        return len(self._hf_dataset)

    def __iter__(self) -> Iterator[CanonicalSample]:
        count = 0
        for idx, row in enumerate(self._hf_dataset):
            if self.dataset_name == "scienceqa_image" and not self._scienceqa_image_has_image(row):
                continue
            if self.max_samples is not None and count >= self.max_samples:
                break
            yield self._spec.parse_row(row, idx, load_images=self.load_images)
            count += 1

    def __getitem__(self, idx: int) -> CanonicalSample:
        if self.streaming:
            raise TypeError("__getitem__ is not supported in streaming mode.")
        row = self._hf_dataset[idx]
        return self._spec.parse_row(row, idx, load_images=self.load_images)

    def get_sample_id(self, idx: int) -> str:
        if self.streaming:
            raise TypeError("get_sample_id is not supported in streaming mode.")
        row = self._hf_dataset[idx]
        return self._spec.parse_id(row, idx)

    @property
    def column_names(self) -> List[str]:
        return self._hf_dataset.column_names

    @property
    def mapping(self) -> ColumnMapping:
        return self._spec.mapping

    @property
    def profile(self) -> HFDatasetSpec:
        return self._spec


__all__ = ["DatasetProfile", "HFDatasetsAdapter"]
