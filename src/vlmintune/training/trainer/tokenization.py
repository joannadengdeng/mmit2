"""Tokenization and dataset-wrapping helpers for training."""
from __future__ import annotations

from typing import Any, Callable, Dict

from torch.utils.data import Dataset, IterableDataset

from vlmintune.training.chat_template import ChatTemplatePreprocessor


class TokenizedDatasetBase:
    def __init__(
        self,
        adapter,
        preprocessor,
        processor,
        model_config,
        max_length: int,
        skip_logger: Callable[[Any, Exception], None],
    ) -> None:
        self.adapter = adapter
        self.preprocessor = preprocessor
        self.processor = processor
        self.model_config = model_config
        self.max_length = max_length
        self.skip_logger = skip_logger

    def tokenize_sample(self, sample):
        try:
            return self.preprocessor.tokenize(
                sample,
                self.processor,
                self.model_config,
                max_length=self.max_length,
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
    return preprocessor.collate(valid)


def build_tokenized_dataset(
    *,
    adapter,
    processor,
    model_spec,
    model_config,
    method_cls,
    max_length: int,
    skip_logger: Callable[[Any, Exception], None],
):
    preprocessor = ChatTemplatePreprocessor(
        method_cls=method_cls,
        append_eos_to_training_answer=model_spec.append_eos_to_training_answer,
    )
    dataset_cls = TokenizedIterableDataset if adapter.streaming else TokenizedMapDataset
    dataset = dataset_cls(
        adapter,
        preprocessor,
        processor,
        model_config,
        max_length,
        skip_logger,
    )
    return dataset, preprocessor
