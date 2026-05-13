"""Inspect a Hugging Face dataset row alongside mmit2's canonical sample.

Examples:
    python examples/inspect_hf_dataset.py \
        --dataset-name lmms-lab/textvqa \
        --split train \
        --num-samples 1

    python examples/inspect_hf_dataset.py \
        --dataset-name lmms-lab/textvqa \
        --split train \
        --num-samples 1 \
        --tokenize \
        --model-path Qwen/Qwen2.5-VL-3B-Instruct
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from pprint import pprint
from typing import Any

import datasets

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmit2.data.adapters.hf_datasets import HFDatasetsAdapter
from mmit2.training.modeling import load_processor
from mmit2.training.preprocessors.chat_template import ChatTemplatePreprocessor


def _preview_value(value: Any, *, max_str: int = 140, max_items: int = 4) -> Any:
    if isinstance(value, str):
        if len(value) <= max_str:
            return value
        return value[:max_str] + "...<truncated>"

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, dict):
        preview = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                preview["..."] = f"{len(value) - max_items} more keys"
                break
            preview[key] = _preview_value(item, max_str=max_str, max_items=max_items)
        return preview

    if isinstance(value, (list, tuple)):
        items = [
            _preview_value(item, max_str=max_str, max_items=max_items)
            for item in list(value[:max_items])
        ]
        if len(value) > max_items:
            items.append(f"... {len(value) - max_items} more items")
        return items

    cls_name = value.__class__.__name__
    size = getattr(value, "size", None)
    mode = getattr(value, "mode", None)
    if size is not None or mode is not None:
        return {
            "type": cls_name,
            "size": size,
            "mode": mode,
        }

    return f"<{cls_name}>"


def _canonical_summary(sample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "image_path": sample.image_path,
        "first_question": sample.first_question,
        "first_answer": sample.first_answer,
        "turns": [
            {"role": turn.role, "content": _preview_value(turn.content)}
            for turn in sample.turns
        ],
        "metadata": {
            key: _preview_value(value)
            for key, value in sample.metadata.items()
            if not key.startswith("_")
        },
        "has_pil_image": "_pil_image" in sample.metadata,
        "metadata_keys": sorted(sample.metadata.keys()),
    }


def _tokenized_summary(sample, processor, image_root: str) -> dict[str, Any]:
    preprocessor = ChatTemplatePreprocessor()
    tokenized = preprocessor.tokenize(
        sample,
        processor,
        image_root=image_root,
        max_length=2048,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    supervised_ids = input_ids[labels != -100]

    summary = {
        "input_ids_shape": tuple(input_ids.shape),
        "labels_shape": tuple(labels.shape),
        "attention_mask_shape": tuple(tokenized["attention_mask"].shape),
        "prompt_token_count": int(tokenized["prompt_mask"].sum().item()),
        "supervised_token_count": int((labels != -100).sum().item()),
        "decoded_input": _preview_value(
            tokenizer.decode(input_ids, skip_special_tokens=False),
            max_str=500,
            max_items=6,
        ),
        "decoded_supervised_text": _preview_value(
            tokenizer.decode(supervised_ids, skip_special_tokens=False)
            if supervised_ids.numel()
            else "",
            max_str=500,
            max_items=6,
        ),
    }
    for key in ("pixel_values", "image_sizes", "image_grid_thw"):
        if key in tokenized:
            value = tokenized[key]
            if hasattr(value, "shape"):
                summary[f"{key}_shape"] = tuple(value.shape)
            else:
                summary[f"{key}_type"] = type(value).__name__
    return summary


def _take_rows(dataset_name: str, split: str, streaming: bool, num_samples: int):
    dataset = datasets.load_dataset(dataset_name, split=split, streaming=streaming)
    return list(itertools.islice(iter(dataset), num_samples))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HF rows and mmit2 canonical samples")
    parser.add_argument("--dataset-name", required=True, help="Hugging Face dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=1, help="How many samples to inspect")
    parser.add_argument("--streaming", action="store_true", help="Force streaming for raw HF rows")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional adapter max_samples cap")
    parser.add_argument("--image-root", default="", help="Optional image root for tokenization")
    parser.add_argument("--tokenize", action="store_true", help="Also run ChatTemplatePreprocessor.tokenize()")
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model id used only when --tokenize is enabled",
    )
    args = parser.parse_args()

    adapter = HFDatasetsAdapter(
        dataset_name=args.dataset_name,
        split=args.split,
        max_samples=args.max_samples or None,
        streaming=args.streaming,
        load_images=True,
    )
    canonical_samples = list(itertools.islice(iter(adapter), args.num_samples))
    raw_rows = _take_rows(
        args.dataset_name,
        adapter.split,
        adapter.streaming,
        args.num_samples,
    )

    print("=" * 80)
    print("Adapter Summary")
    print("=" * 80)
    pprint(
        {
            "dataset_name": adapter.dataset_name,
            "requested_split": args.split,
            "resolved_split": adapter.split,
            "streaming": adapter.streaming,
            "max_samples": adapter.max_samples,
            "column_names": adapter.column_names,
            "mapping": {
                "id_col": adapter.mapping.id_col,
                "image_col": adapter.mapping.image_col,
                "question_col": adapter.mapping.question_col,
                "answer_col": adapter.mapping.answer_col,
            },
        }
    )

    processor = None
    if args.tokenize:
        print()
        print(f"[inspect] loading processor: {args.model_path}")
        processor = load_processor(args.model_path)

    for idx, (raw_row, sample) in enumerate(zip(raw_rows, canonical_samples)):
        print()
        print("=" * 80)
        print(f"Sample {idx}")
        print("=" * 80)
        print("Raw Hugging Face Row")
        pprint({key: _preview_value(value) for key, value in raw_row.items()})
        print()
        print("CanonicalSample")
        pprint(_canonical_summary(sample))

        if processor is not None:
            print()
            print("Tokenized View")
            pprint(_tokenized_summary(sample, processor, args.image_root))


if __name__ == "__main__":
    main()
