"""Tokenization and dataset-wrapping helpers for training."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from torch.utils.data import Dataset, IterableDataset

from mmit2.training.preprocessors.chat_template import ChatTemplatePreprocessor
from mmit2.training.trainer.helpers import DebugRecorder


class TokenizedDatasetBase:
    def __init__(
        self,
        adapter,
        preprocessor,
        processor,
        image_root: str,
        skip_logger: Callable[[Any, Exception], None],
        debug_recorder: Optional[DebugRecorder] = None,
    ) -> None:
        self.adapter = adapter
        self.preprocessor = preprocessor
        self.processor = processor
        self.image_root = image_root
        self.skip_logger = skip_logger
        self.debug_recorder = debug_recorder

    def tokenize_sample(self, sample):
        if self.debug_recorder is not None:
            self.debug_recorder.record_sample(sample)
        try:
            return self.preprocessor.tokenize(
                sample,
                self.processor,
                image_root=self.image_root,
                max_length=2048,
                debug_sink=(
                    self.debug_recorder.record_prompt
                    if self.debug_recorder is not None
                    else None
                ),
            )
        except Exception as exc:
            self.skip_logger(sample.id, exc)
            return None


class TokenizedMapDataset(TokenizedDatasetBase, Dataset):
    def __len__(self) -> int:
        return len(self.adapter)

    def __getitem__(self, idx: int):
        return self.tokenize_sample(self.adapter[idx])


class TokenizedIterableDataset(TokenizedDatasetBase, IterableDataset):
    def __iter__(self):
        for sample in self.adapter:
            if (result := self.tokenize_sample(sample)) is not None:
                yield result


def safe_collate(preprocessor: ChatTemplatePreprocessor, samples) -> Dict[str, Any]:
    valid = [sample for sample in samples if sample is not None]
    if not valid:
        return {}
    return preprocessor.collate(valid)


def build_tokenized_dataset(
    *,
    adapter,
    processor,
    image_root: str,
    skip_logger: Callable[[Any, Exception], None],
    debug_recorder: Optional[DebugRecorder] = None,
):
    preprocessor = ChatTemplatePreprocessor()
    dataset_cls = TokenizedIterableDataset if getattr(adapter, "streaming", False) else TokenizedMapDataset
    dataset = dataset_cls(
        adapter,
        preprocessor,
        processor,
        image_root,
        skip_logger,
        debug_recorder,
    )
    return dataset, preprocessor
