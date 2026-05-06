"""Colab-friendly LoRA debug entry point.

Usage in Colab after cloning the repo:

    !pip install -q ".[finetune]"
    !python examples/colab_lora_debug.py --config configs/colab_lora_debug.yaml

This script prints:
  - the normalized training config
  - one raw training sample
  - the chat-template messages / rendered prompt / token stats
  - one single-sample VQAv2-style eval result before training
  - one single-sample eval result after training (unless --inspect-only)

It also writes a JSON report to ``<output_dir>/debug_report.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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


def _load_samples(trainer_config: TrainerConfig) -> List[CanonicalSample]:
    data_cfg = dict(trainer_config.data_config)
    max_samples = int(data_cfg.pop("max_samples", 0) or 0)
    adapter_name = data_cfg.pop("adapter", "hf_datasets")
    if adapter_name != "hf_datasets":
        raise ValueError(f"Unsupported adapter '{adapter_name}'")

    dataset_name = data_cfg.pop("data_path")
    adapter = HFDatasetsAdapter(dataset_name=dataset_name, **data_cfg)
    samples = list(adapter)
    return samples[:max_samples] if max_samples > 0 else samples


def _sample_summary(sample: CanonicalSample) -> Dict[str, Any]:
    return {
        "id": sample.id,
        "image_path": sample.image_path,
        "first_question": sample.first_question,
        "first_answer": sample.first_answer,
        "metadata": _json_ready(sample.metadata),
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
        "attention_mask": tokenized["attention_mask"].tolist(),
        "prompt_mask": prompt_mask.tolist(),
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
        "question": sample.first_question,
        "eval_prompt": eval_prompt,
        "ground_truths": ground_truths,
        "prediction": prediction,
        "metric_key": metric_key,
        "metric_label": METRIC_LABELS.get(metric_key, metric_key),
        "metric_reason": metric_reason,
        "scores": scores,
    }


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

    _section("Training Config")
    pprint(_json_ready({
        "config_path": args.config,
        "training_config": trainer_dict,
        "effective_output_dir": trainer_config.output_dir,
        "method_config": method_config,
    }))

    _section("One Raw Sample")
    pprint(_sample_summary(sample))

    processor = load_processor(cfg.model.model_path)
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
        "labels_preview": preprocessor_debug["labels"][:64],
        "decoded_supervised_text": preprocessor_debug["decoded_supervised_text"],
    })

    quantize_4bit = method_obj.requires_quantization(method_config)
    model = load_vlm(
        cfg.model.model_path,
        quantize_4bit=quantize_4bit,
        torch_dtype=torch.bfloat16,
    )
    base_method = LocalMethod(model, processor)
    pretrain_eval = _evaluate_single_sample(
        base_method,
        sample,
        image_root=image_root,
        max_new_tokens=args.max_new_tokens,
    )

    _section("Single-Sample Eval Before Training")
    pprint(pretrain_eval)

    report: Dict[str, Any] = {
        "config": _json_ready(trainer_dict),
        "method_config": _json_ready(method_config),
        "sample": _sample_summary(sample),
        "preprocessor": _json_ready(preprocessor_debug),
        "eval_before_train": _json_ready(pretrain_eval),
    }

    if not args.inspect_only:
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
            posttrain_eval = _evaluate_single_sample(
                tuned_method,
                sample,
                image_root=image_root,
                max_new_tokens=args.max_new_tokens,
            )
            _section("Single-Sample Eval After Training")
            pprint(posttrain_eval)
            report["eval_after_train"] = _json_ready(posttrain_eval)
            report["final_checkpoint"] = final_checkpoint
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
