"""HuggingFace dataset adapter backed by per-dataset VQA specs."""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import datasets

from mmit2.data.adapters.base import DatasetAdapter
from mmit2.data.datasets import (
    DATASET_SPECS,
    ColumnMapping,
    HFDatasetSpec,
    build_configured_spec,
    get_dataset_spec,
)
from mmit2.data.types import CanonicalSample

DatasetProfile = HFDatasetSpec


class HFDatasetsAdapter(DatasetAdapter):
    """Load a HuggingFace VQA dataset and yield ``CanonicalSample`` rows."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        column_map: Optional[ColumnMapping] = None,
        max_samples: Optional[int] = None,
        streaming: bool = False,
        trust_remote_code: bool = True,
        load_images: bool = True,
        config_name: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.streaming = streaming
        self.load_images = load_images
        self._num_examples: Optional[int] = None

        spec = get_dataset_spec(dataset_name)
        prefer_streaming = spec.prefer_streaming if spec is not None else False
        config_arg = config_name or ""
        load_pos = (dataset_name, config_arg) if config_arg else (dataset_name,)

        if not streaming and (prefer_streaming or (max_samples is not None and max_samples <= 50)):
            streaming = True
            self.streaming = True

        self._hf_dataset = self._load_dataset(
            datasets,
            load_pos,
            split,
            streaming,
            trust_remote_code,
        )

        if not self.load_images and not self.streaming:
            self._disable_eager_image_decode(datasets)

        if max_samples is not None and not self.streaming:
            self._hf_dataset = self._hf_dataset.select(range(min(max_samples, len(self._hf_dataset))))

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

    def _load_dataset(
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

    def _disable_eager_image_decode(self, datasets_mod) -> None:
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
            if self.max_samples is not None and count >= self.max_samples:
                break
            yield self._spec.parse_row(row, idx, load_images=self.load_images)
            count += 1

    def __getitem__(self, idx: int) -> CanonicalSample:
        if self.streaming:
            raise TypeError("__getitem__ is not supported in streaming mode.")
        row = self._hf_dataset[idx]
        return self._spec.parse_row(row, idx, load_images=self.load_images)

    @property
    def column_names(self) -> List[str]:
        return self._hf_dataset.column_names

    @property
    def mapping(self) -> ColumnMapping:
        return self._spec.mapping

    @property
    def profile(self) -> HFDatasetSpec:
        return self._spec
