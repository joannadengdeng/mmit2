"""Tokenization and dataset-wrapping helpers for training."""
from __future__ import annotations

from typing import Any, Callable, Dict

from torch.utils.data import Dataset, IterableDataset

from vlmintune.training.chat_template import ChatTemplatePreprocessor
from vlmintune.training.trainer.helpers import DebugRecorder


class TokenizedDatasetBase:
    def __init__(
        self,
        adapter,
        preprocessor,
        processor,
        model_config,
        max_length: int,
        skip_logger: Callable[[Any, Exception], None],
        debug_recorder: DebugRecorder,
    ) -> None:
        self.adapter = adapter
        self.preprocessor = preprocessor
        self.processor = processor
        self.model_config = model_config
        self.max_length = max_length
        self.skip_logger = skip_logger
        self.debug_recorder = debug_recorder

    def tokenize_sample(self, sample):
        self.debug_recorder.record_sample(sample)
        try:
            result = self.preprocessor.tokenize(
                sample,
                self.processor,
                self.model_config,
                max_length=self.max_length,
            )
            self.debug_recorder.record_prompt(result.pop("prompt_preview"))
            return result
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
    model_config,
    enable_instruction_supervision: bool = False,
    enable_mores_intervention: bool = False,
    max_length: int,
    skip_logger: Callable[[Any, Exception], None],
    debug_recorder: DebugRecorder,
):
    preprocessor = ChatTemplatePreprocessor(
        enable_instruction_supervision=enable_instruction_supervision,
        enable_mores_intervention=enable_mores_intervention,
    )
    dataset_cls = TokenizedIterableDataset if getattr(adapter, "streaming", False) else TokenizedMapDataset
    dataset = dataset_cls(
        adapter,
        preprocessor,
        processor,
        model_config,
        max_length,
        skip_logger,
        debug_recorder,
    )
    return dataset, preprocessor
