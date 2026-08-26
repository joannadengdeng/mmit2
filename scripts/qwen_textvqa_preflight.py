#!/usr/bin/env python3
"""Offline BF16 Qwen/TextVQA load-and-generate check for the AutoDL worker."""
from __future__ import annotations

import json
import os
import sys
import traceback

import torch

from vlmintune.data.hf_datasets import HFDatasetsAdapter
from vlmintune.data.types import EvalSample
from vlmintune.eval.method import LocalMethod
from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.base import load_processor, load_vlm


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    model_spec = get_model_spec("qwen25vl_3b_instruct")
    processor = load_processor(model_spec.hf_model_id)
    model = load_vlm(
        model_spec.hf_model_id,
        quantize_4bit=False,
        torch_dtype=torch.bfloat16,
    )
    method = LocalMethod(model, processor)

    adapter = HFDatasetsAdapter(
        dataset_name="lmms-lab/textvqa",
        split="validation",
        usage="eval",
        max_samples=1,
        streaming=True,
        sample_seed=42,
    )
    sample = next(iter(adapter))
    eval_sample = EvalSample(
        id=sample.id,
        image_path=sample.image_path,
        question=sample.question,
        eval_answers=sample.eval_answers,
        metadata=sample.metadata,
    )
    prepared = method.prepare_eval_input(eval_sample)
    prediction = method.generate(prepared, max_new_tokens=16, temperature=0.0)
    if not prediction.strip():
        raise RuntimeError("BF16 preflight generated an empty prediction")
    if set(prediction.strip()) == {"!"}:
        raise RuntimeError(f"BF16 preflight generated a degenerate prediction: {prediction!r}")

    print(
        json.dumps(
            {
                "status": "ok",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "model": model_spec.hf_model_id,
                "dtype": str(next(model.parameters()).dtype),
                "sample_id": str(sample.id),
                "question": sample.question,
                "prediction": prediction,
                "ground_truth": sample.eval_answers,
                "cuda_memory_allocated_gib": round(
                    torch.cuda.memory_allocated() / (1024**3),
                    3,
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("VLMINTUNE_FAST_EXIT") == "1":
        os._exit(exit_code)
    raise SystemExit(exit_code)
