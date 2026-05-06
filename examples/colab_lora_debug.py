"""Colab-friendly LoRA debug entry point.

Usage in Colab after cloning the repo:

    !pip install -q ".[finetune]"
    !python examples/colab_lora_debug.py --config configs/colab_lora_debug.yaml

This script prints:
  - the normalized training config
  - one raw training sample
  - the chat-template messages / rendered prompt / token stats
  - a small held-out eval set summary before training
  - a small held-out eval set summary after training (unless --inspect-only)

It also writes a JSON report to ``<output_dir>/debug_report.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from platform import python_version
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

import torch

try:
    from google.colab import drive as colab_drive  # type: ignore[import-not-found]
except ImportError:
    colab_drive = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmit2.config.training_config import config_to_trainer_dict, load_config
from mmit2.data.adapters.hf_datasets import HFDatasetsAdapter
from mmit2.data.types import CanonicalSample, EvalSample
from mmit2.eval.methods.local_method import LocalMethod
from mmit2.eval.metrics.scoring import (
    METRIC_LABELS,
    auto_select_metric,
    score_prediction_multi,
)
from mmit2.modeling import load_processor, load_vlm
from mmit2.registry import build_training_method
from mmit2.training.preprocessors.chat_template import (
    ChatTemplatePreprocessor,
    _build_messages,
    _build_prompt_messages,
)
from mmit2.training.trainer import Trainer, TrainerConfig

VQAV2_SINGLE_SAMPLE_INSTRUCTION = "Answer the question using a single word or phrase."


def _preview_list(values: List[Any], limit: int = 8) -> List[Any]:
    if len(values) <= limit:
        return values
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return [*values[:head], "...", *values[-tail:]]


def _json_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return repr(value)


def _section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _build_trainer_config(trainer_dict: Dict[str, Any]) -> TrainerConfig:
    training_cfg = trainer_dict["training"]
    return TrainerConfig(
        data_config=trainer_dict["data"],
        training_method=trainer_dict["training_method"],
        method_params=trainer_dict.get("method_params", {}),
        num_epochs=training_cfg["num_epochs"],
        per_device_batch_size=training_cfg["per_device_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        warmup_ratio=training_cfg["warmup_ratio"],
        weight_decay=training_cfg["weight_decay"],
        max_grad_norm=training_cfg["max_grad_norm"],
        save_steps=training_cfg["save_steps"],
        output_dir=training_cfg["output_dir"],
    )


def _build_adapter_kwargs(
    trainer_config: TrainerConfig,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    data_cfg = dict(trainer_config.data_config)
    adapter_name = data_cfg.pop("adapter", "hf_datasets")
    data_cfg.pop("image_root", None)
    if adapter_name != "hf_datasets":
        raise ValueError(f"Unsupported adapter '{adapter_name}'")

    dataset_name = data_cfg.pop("data_path")
    requested_max = int(data_cfg.pop("max_samples", 0) or 0)
    resolved_max = max_samples if max_samples is not None else requested_max
    return {
        "dataset_name": dataset_name,
        "max_samples": resolved_max if resolved_max and resolved_max > 0 else None,
        **data_cfg,
    }


def _load_samples(trainer_config: TrainerConfig) -> List[CanonicalSample]:
    adapter = HFDatasetsAdapter(**_build_adapter_kwargs(trainer_config))
    return [sample for sample in adapter]


def _load_eval_samples(
    trainer_config: TrainerConfig,
    eval_count: int,
    eval_offset: int,
) -> List[CanonicalSample]:
    if eval_count <= 0:
        return []
    fetch_count = eval_offset + eval_count if eval_offset > 0 else eval_count
    adapter = HFDatasetsAdapter(**_build_adapter_kwargs(trainer_config, max_samples=fetch_count))
    samples = [sample for sample in adapter]
    if eval_offset >= len(samples):
        raise IndexError(
            f"eval_offset {eval_offset} is out of range for {len(samples)} loaded samples",
        )
    return samples[eval_offset:eval_offset + eval_count]


def _sample_summary(sample: CanonicalSample) -> Dict[str, Any]:
    pil_image = sample.metadata.get("_pil_image") if sample.metadata else None
    image_info = None
    if pil_image is not None:
        image_info = {
            "mode": getattr(pil_image, "mode", None),
            "size": getattr(pil_image, "size", None),
        }
    return {
        "id": sample.id,
        "image_path": sample.image_path,
        "first_question": sample.first_question,
        "first_answer": sample.first_answer,
        "turns": [
            {"role": turn.role, "content": turn.content}
            for turn in sample.turns
        ],
        "raw_answers": list(sample.metadata.get("raw_answers", [])) if sample.metadata else [],
        "image_info": image_info,
        "metadata": _json_ready(sample.metadata),
    }


def _sample_short_summary(sample: CanonicalSample) -> Dict[str, Any]:
    return {
        "id": sample.id,
        "question": sample.first_question,
        "answer": sample.first_answer,
        "image_path": sample.image_path,
    }


def _dataset_overview(
    train_samples: List[CanonicalSample],
    eval_samples: List[CanonicalSample],
) -> Dict[str, Any]:
    return {
        "train_count": len(train_samples),
        "eval_count": len(eval_samples),
        "train_preview": [_sample_short_summary(sample) for sample in train_samples[:3]],
        "eval_preview": [_sample_short_summary(sample) for sample in eval_samples[:3]],
    }


def _runtime_summary() -> Dict[str, Any]:
    device_names = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            device_names.append(torch.cuda.get_device_name(idx))
    return {
        "python_version": python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": device_names,
    }


def _processor_summary(processor: Any) -> Dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)
    chat_template = getattr(tokenizer, "chat_template", None)
    image_processor = getattr(processor, "image_processor", None)
    return {
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "image_processor_class": type(image_processor).__name__ if image_processor is not None else None,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "chat_template_preview": chat_template[:500] if isinstance(chat_template, str) else None,
    }


def _model_summary(model: Any) -> Dict[str, Any]:
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        count = int(param.numel())
        total_params += count
        if param.requires_grad:
            trainable_params += count
    return {
        "model_class": type(model).__name__,
        "dtype": str(getattr(model, "dtype", None)),
        "device": str(getattr(model, "device", None)),
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def _inspect_chat_template_sample(
    preprocessor: ChatTemplatePreprocessor,
    sample: CanonicalSample,
    processor: Any,
    image_root: str = "",
    max_length: int = 2048,
) -> Dict[str, Any]:
    has_image = bool(sample.image_path) or bool(sample.metadata and sample.metadata.get("_pil_image"))
    messages = _build_messages(sample, has_image)
    if not messages:
        raise ValueError(f"Sample {sample.id} has no turns")

    prompt_messages = _build_prompt_messages(messages)
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prompt_text = ""
    if prompt_messages:
        prompt_text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )

    tokenized = preprocessor.tokenize(
        sample,
        processor,
        image_root=image_root,
        max_length=max_length,
    )
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    prompt_mask = tokenized["prompt_mask"]
    supervised_ids = input_ids[labels != -100]
    tokenizer = getattr(processor, "tokenizer", processor)
    supervised_positions = (labels != -100).nonzero(as_tuple=False).flatten().tolist()
    prompt_positions = prompt_mask.nonzero(as_tuple=False).flatten().tolist()
    attention_mask = tokenized["attention_mask"]

    return {
        "sample_id": sample.id,
        "messages": messages,
        "prompt_messages": prompt_messages,
        "full_text": full_text,
        "prompt_text": prompt_text,
        "input_length": int(input_ids.numel()),
        "prompt_token_count": int(prompt_mask.sum().item()),
        "supervised_token_count": int((labels != -100).sum().item()),
        "input_ids": input_ids.tolist(),
        "labels": labels.tolist(),
        "attention_mask": attention_mask.tolist(),
        "prompt_mask": prompt_mask.tolist(),
        "supervised_positions": supervised_positions,
        "prompt_positions": prompt_positions,
        "decoded_input": tokenizer.decode(input_ids, skip_special_tokens=False),
        "decoded_supervised_text": tokenizer.decode(
            supervised_ids,
            skip_special_tokens=False,
        ) if supervised_ids.numel() else "",
    }


def _apply_colab_runtime(cfg: Any, trainer_config: TrainerConfig) -> None:
    if cfg.runtime.mode != "colab":
        return

    colab_cfg = cfg.runtime.colab
    drive_root = os.path.join(colab_cfg.drive_mount_point, "MyDrive")
    if colab_cfg.mount_drive and colab_drive is not None:
        if os.path.isdir(drive_root):
            print(f"[mmit2] Google Drive already mounted at {colab_cfg.drive_mount_point}")
        else:
            print(f"[mmit2] Mounting Google Drive at {colab_cfg.drive_mount_point}")
            try:
                colab_drive.mount(colab_cfg.drive_mount_point)
            except Exception as exc:
                raise RuntimeError(
                    "Google Drive auto-mount failed. In Colab, run:\n"
                    "from google.colab import drive\n"
                    f"drive.mount('{colab_cfg.drive_mount_point}')\n"
                    "in a Python cell first, then rerun this script. "
                    "If you do not want Drive output, set mount_drive=false and output_to_drive=false "
                    "in the config."
                ) from exc

    if colab_cfg.output_to_drive:
        trainer_config.output_dir = os.path.join(
            colab_cfg.drive_mount_point,
            "MyDrive",
            "mmit2_output",
            trainer_config.output_dir,
        )


def _evaluate_single_sample(
    method: LocalMethod,
    sample: CanonicalSample,
    image_root: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    ground_truths = list(sample.metadata.get("raw_answers", [])) if sample.metadata else []
    if not ground_truths and sample.first_answer:
        ground_truths = [sample.first_answer]

    eval_prompt = sample.first_question + "\n" + VQAV2_SINGLE_SAMPLE_INSTRUCTION
    eval_sample = EvalSample(
        id=sample.id,
        image_path=sample.image_path,
        question=eval_prompt,
        ground_truth=ground_truths,
        metadata=sample.metadata,
    )
    prepared = method.prepare_eval_input(eval_sample, image_root=image_root)
    prediction = method.generate(prepared, max_new_tokens=max_new_tokens, temperature=0.0)

    metric_key, metric_reason = auto_select_metric("open_vqa", ground_truths)
    scores = score_prediction_multi(
        prediction,
        ground_truths,
        task_type="open_vqa",
        metrics=[metric_key],
    )
    return {
        "sample_id": sample.id,
        "question": sample.first_question,
        "eval_prompt": eval_prompt,
        "ground_truths": ground_truths,
        "prediction": prediction,
        "metric_key": metric_key,
        "metric_label": METRIC_LABELS.get(metric_key, metric_key),
        "metric_reason": metric_reason,
        "scores": scores,
    }


def _summarize_eval_runs(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_totals: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    for result in results:
        for metric_key, value in result.get("scores", {}).items():
            metric_totals[metric_key] = metric_totals.get(metric_key, 0.0) + float(value)
            metric_counts[metric_key] = metric_counts.get(metric_key, 0) + 1

    metric_averages = {
        metric_key: metric_totals[metric_key] / metric_counts[metric_key]
        for metric_key in metric_totals
        if metric_counts[metric_key] > 0
    }
    return {
        "count": len(results),
        "metric_averages": metric_averages,
        "results": results,
    }


def _evaluate_sample_batch(
    method: LocalMethod,
    samples: List[CanonicalSample],
    image_root: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    results = [
        _evaluate_single_sample(
            method,
            sample,
            image_root=image_root,
            max_new_tokens=max_new_tokens,
        )
        for sample in samples
    ]
    return _summarize_eval_runs(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab LoRA debug runner")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "colab_lora_debug.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Which sample to inspect")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Eval generation length")
    parser.add_argument(
        "--eval-count",
        type=int,
        default=10,
        help="How many held-out samples to evaluate before/after training",
    )
    parser.add_argument(
        "--eval-offset",
        type=int,
        default=-1,
        help="Start eval from this sample offset; default uses training max_samples as the hold-out boundary",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only print intermediate states; skip training and post-train eval",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer_dict = config_to_trainer_dict(cfg)
    trainer_config = _build_trainer_config(trainer_dict)
    _apply_colab_runtime(cfg, trainer_config)
    samples = _load_samples(trainer_config)
    if not samples:
        raise ValueError("No samples loaded from dataset")
    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise IndexError(f"sample_index {args.sample_index} out of range for {len(samples)} samples")

    sample = samples[args.sample_index]
    method_obj = build_training_method(trainer_config.training_method)
    method_config = {**method_obj.default_config(), **trainer_config.method_params}
    image_root = trainer_config.data_config.get("image_root", "")
    train_sample_count = int(trainer_config.data_config.get("max_samples", 0) or 0)
    eval_offset = args.eval_offset if args.eval_offset >= 0 else train_sample_count
    eval_samples = _load_eval_samples(
        trainer_config,
        eval_count=args.eval_count,
        eval_offset=eval_offset,
    )
    runtime_summary = _runtime_summary()

    _section("Runtime Summary")
    pprint(runtime_summary)

    if not args.inspect_only and not runtime_summary["cuda_available"]:
        raise RuntimeError(
            "CUDA is not available in this Python environment, so the model would run on CPU and "
            "be extremely slow. In Colab this usually means the GPU build of torch is not active. "
            "Restart the runtime, then reinstall without replacing Colab's torch. Recommended steps:\n"
            "1. Runtime -> Restart runtime\n"
            "2. Clone the repo again\n"
            "3. Run: pip install -q transformers peft accelerate datasets pyyaml pillow bitsandbytes\n"
            "4. Run: pip install -q -e . --no-deps\n"
            "5. Verify with: import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
        )

    _section("Training Config")
    pprint(_json_ready({
        "config_path": args.config,
        "training_config": trainer_dict,
        "effective_output_dir": trainer_config.output_dir,
        "method_config": method_config,
        "train_sample_count": train_sample_count,
        "eval_count": args.eval_count,
        "eval_offset": eval_offset,
    }))

    _section("Dataset Overview")
    pprint(_dataset_overview(samples, eval_samples))

    _section("One Raw Sample")
    pprint(_sample_summary(sample))

    processor = load_processor(cfg.model.model_path)
    _section("Processor Summary")
    pprint(_processor_summary(processor))

    preprocessor = ChatTemplatePreprocessor()
    preprocessor_debug = _inspect_chat_template_sample(
        preprocessor,
        sample,
        processor,
        image_root=image_root,
        max_length=2048,
    )

    _section("Chat Template / Tokenization")
    pprint({
        "messages": preprocessor_debug["messages"],
        "prompt_messages": preprocessor_debug["prompt_messages"],
        "full_text": preprocessor_debug["full_text"],
        "prompt_text": preprocessor_debug["prompt_text"],
        "input_length": preprocessor_debug["input_length"],
        "prompt_token_count": preprocessor_debug["prompt_token_count"],
        "supervised_token_count": preprocessor_debug["supervised_token_count"],
        "input_ids_preview": preprocessor_debug["input_ids"][:64],
        "input_ids_tail": preprocessor_debug["input_ids"][-32:],
        "labels_preview": preprocessor_debug["labels"][:64],
        "labels_tail": preprocessor_debug["labels"][-32:],
        "attention_mask_preview": preprocessor_debug["attention_mask"][:64],
        "prompt_mask_preview": preprocessor_debug["prompt_mask"][:64],
        "supervised_positions_preview": _preview_list(preprocessor_debug["supervised_positions"], limit=12),
        "prompt_positions_preview": _preview_list(preprocessor_debug["prompt_positions"], limit=12),
        "decoded_input": preprocessor_debug["decoded_input"],
        "decoded_supervised_text": preprocessor_debug["decoded_supervised_text"],
    })

    report: Dict[str, Any] = {
        "config": _json_ready(trainer_dict),
        "method_config": _json_ready(method_config),
        "dataset_overview": _dataset_overview(samples, eval_samples),
        "sample": _sample_summary(sample),
        "processor": _processor_summary(processor),
        "preprocessor": _json_ready(preprocessor_debug),
    }

    if args.inspect_only:
        os.makedirs(trainer_config.output_dir, exist_ok=True)
        report_path = os.path.join(trainer_config.output_dir, "debug_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        _section("Inspect-Only Complete")
        print(f"Saved debug report to {report_path}")
        return

    _section("Eval Sample Set")
    pprint({
        "count": len(eval_samples),
        "offset": eval_offset,
        "sample_ids": [sample.id for sample in eval_samples],
        "questions": [sample.first_question for sample in eval_samples],
        "summaries": [_sample_short_summary(sample) for sample in eval_samples],
    })

    quantize_4bit = method_obj.requires_quantization(method_config)
    model = load_vlm(
        cfg.model.model_path,
        quantize_4bit=quantize_4bit,
        torch_dtype=torch.bfloat16,
    )
    _section("Model Summary Before Training")
    pprint(_model_summary(model))
    base_method = LocalMethod(model, processor)
    pretrain_eval = _evaluate_single_sample(
        base_method,
        sample,
        image_root=image_root,
        max_new_tokens=args.max_new_tokens,
    )
    eval_before_train = _evaluate_sample_batch(
        base_method,
        eval_samples,
        image_root=image_root,
        max_new_tokens=args.max_new_tokens,
    )

    _section("Single-Sample Eval Before Training")
    pprint(pretrain_eval)
    _section("Held-Out Eval Before Training")
    pprint(eval_before_train)

    report["eval_before_train"] = _json_ready(pretrain_eval)
    report["held_out_eval_before_train"] = _json_ready(eval_before_train)
    report["model_before_train"] = _model_summary(model)

    del base_method
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _section("Training")
    trainer = Trainer(cfg.model.model_path)
    trainer.train(trainer_config)

    final_checkpoint = os.path.join(trainer_config.output_dir, "final")
    if os.path.isdir(final_checkpoint):
        tuned_method = LocalMethod.from_checkpoint(
            base_model_id=cfg.model.model_path,
            checkpoint_path=final_checkpoint,
            ft_method=trainer_config.training_method,
        )
        _section("Model Summary After Training")
        pprint(_model_summary(tuned_method.model))
        posttrain_eval = _evaluate_single_sample(
            tuned_method,
            sample,
            image_root=image_root,
            max_new_tokens=args.max_new_tokens,
        )
        eval_after_train = _evaluate_sample_batch(
            tuned_method,
            eval_samples,
            image_root=image_root,
            max_new_tokens=args.max_new_tokens,
        )
        _section("Single-Sample Eval After Training")
        pprint(posttrain_eval)
        _section("Held-Out Eval After Training")
        pprint(eval_after_train)
        report["eval_after_train"] = _json_ready(posttrain_eval)
        report["held_out_eval_after_train"] = _json_ready(eval_after_train)
        report["final_checkpoint"] = final_checkpoint
        report["model_after_train"] = _model_summary(tuned_method.model)
        del tuned_method
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"[WARN] Final checkpoint not found: {final_checkpoint}")

    output_dir = Path(trainer_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "debug_report.json"
    report_path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _section("Report Saved")
    print(report_path)


if __name__ == "__main__":
    main()
