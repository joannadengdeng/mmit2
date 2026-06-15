#!/usr/bin/env python3
"""Generate VisNec-style visual-necessity scores for VQA training samples.

The score is the marginal answer loss reduction from providing the image:

    visnec_score = loss_without_image - loss_with_image

Higher scores indicate samples where visual context helps the model predict the
training answer more than text alone.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from vlmintune.data.hf_datasets import HFDatasetsAdapter
from vlmintune.models.registry import get_model_spec
from vlmintune.training.chat_template import ChatTemplatePreprocessor
from vlmintune.training.methods.base import load_processor, load_vlm
from vlmintune.training.trainer.helpers import to_device

DATASET_ALIASES = {
    "textvqa": "lmms-lab/textvqa",
    "vqav2": "lmms-lab/VQAv2",
    "vizwiz": "lmms-lab/VizWiz-VQA",
    "gqa": "lmms-lab/GQA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="qwen25vl_3b_instruct")
    parser.add_argument("--dataset-name", default="textvqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def answer_loss(
    model: Any,
    processor: Any,
    preprocessor: ChatTemplatePreprocessor,
    sample: Any,
    max_length: int,
) -> float:
    tokenized = preprocessor.tokenize(
        sample,
        processor=processor,
        model_config=model.config,
        max_length=max_length,
    )
    labels = tokenized["labels"]
    if int((labels != -100).sum().item()) == 0:
        raise ValueError(f"Sample {sample.id} has no supervised answer tokens.")

    batch = preprocessor.collate([tokenized])
    device = next(model.parameters()).device
    batch = to_device(batch, device)
    with torch.inference_mode():
        outputs = model(**batch)
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise RuntimeError("Model output did not include loss.")
    return float(loss.detach().float().cpu().item())


def make_text_only_sample(sample: Any) -> Any:
    return replace(sample, image_path="", metadata={})


def main() -> int:
    args = parse_args()
    dataset_name = DATASET_ALIASES.get(args.dataset_name, args.dataset_name)
    output_path = Path(os.path.expanduser(args.output))
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists; pass --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_spec = get_model_spec(args.model_name)
    processor = load_processor(model_spec.hf_model_id)
    model = load_vlm(
        model_spec.hf_model_id,
        quantize_4bit=False,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    adapter = HFDatasetsAdapter(
        dataset_name=dataset_name,
        split=args.split,
        max_samples=None,
        streaming=False,
        trust_remote_code=args.trust_remote_code,
        load_images=True,
        usage="train",
    )
    preprocessor = ChatTemplatePreprocessor(
        append_eos_to_training_answer=bool(model_spec.append_eos_to_training_answer),
    )

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    processed = 0
    skipped = 0
    with tmp_path.open("w", encoding="utf-8") as out:
        for sample in adapter:
            if args.max_samples > 0 and processed >= args.max_samples:
                break
            processed += 1
            try:
                loss_with_image = answer_loss(
                    model,
                    processor,
                    preprocessor,
                    sample,
                    args.max_length,
                )
                loss_without_image = answer_loss(
                    model,
                    processor,
                    preprocessor,
                    make_text_only_sample(sample),
                    args.max_length,
                )
                score = loss_without_image - loss_with_image
                record = {
                    "sample_id": str(sample.id),
                    "visnec_score": score,
                    "loss_with_image": loss_with_image,
                    "loss_without_image": loss_without_image,
                    "question": sample.question,
                    "train_answer": sample.train_answer,
                    "model_name": args.model_name,
                    "dataset_name": dataset_name,
                    "split": args.split,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            except Exception as exc:
                skipped += 1
                print(
                    json.dumps(
                        {
                            "type": "skip",
                            "sample_id": str(getattr(sample, "id", "")),
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            if processed % 25 == 0:
                print(
                    json.dumps(
                        {
                            "type": "progress",
                            "processed": processed,
                            "written": processed - skipped,
                            "skipped": skipped,
                        }
                    ),
                    flush=True,
                )

    tmp_path.replace(output_path)
    print(
        json.dumps(
            {
                "type": "completed",
                "processed": processed,
                "written": processed - skipped,
                "skipped": skipped,
                "output": str(output_path),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
