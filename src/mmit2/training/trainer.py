"""Single-stage trainer for multimodal instruction tuning."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, IterableDataset

from mmit2.data.adapters.hf_datasets import HFDatasetsAdapter
from mmit2.training.modeling import load_processor, load_vlm
from mmit2.training.registry import build_training_method
from mmit2.training.preprocessors.chat_template import ChatTemplatePreprocessor


_NON_FORWARD_BATCH_KEYS = {
    "prompt_mask",
}


def _emit(event_type: str, data: dict) -> None:
    """Print a JSON event line to stdout."""
    print(json.dumps({"type": event_type, "data": data}), flush=True)


def _cosine_schedule(optimizer, num_warmup: int, num_total: int):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < num_warmup:
            return step / max(1, num_warmup)
        progress = (step - num_warmup) / max(1, num_total - num_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            out[key] = [tensor.to(device) for tensor in value]
        else:
            out[key] = value
    return out


def _shape_str(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return "x".join(str(dim) for dim in value.shape)
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        first = value[0]
        return f"list[{len(value)}]:" + "x".join(str(dim) for dim in first.shape)
    return type(value).__name__


def _describe_batch(batch: Dict[str, Any]) -> str:
    parts = [
        f"input_ids={_shape_str(batch['input_ids'])}",
        f"labels={_shape_str(batch['labels'])}",
        f"attention_mask={_shape_str(batch['attention_mask'])}",
    ]
    for key in ("pixel_values", "image_grid_thw", "image_sizes"):
        if key in batch:
            parts.append(f"{key}={_shape_str(batch[key])}")
    return "First batch shapes: " + ", ".join(parts)


class _TokenizedDatasetBase:
    def __init__(self, adapter, preprocessor, processor, image_root: str, logger) -> None:
        self._adapter = adapter
        self._preprocessor = preprocessor
        self._processor = processor
        self._image_root = image_root
        self._logger = logger

    def _tokenize(self, sample):
        try:
            return self._preprocessor.tokenize(
                sample,
                self._processor,
                image_root=self._image_root,
                max_length=2048,
            )
        except Exception as exc:
            self._logger(sample.id, exc)
            return None


class _TokenizedMapDataset(_TokenizedDatasetBase, Dataset):
    def __len__(self) -> int:
        return len(self._adapter)

    def __getitem__(self, idx: int):
        return self._tokenize(self._adapter[idx])


class _TokenizedIterableDataset(_TokenizedDatasetBase, IterableDataset):
    def __iter__(self):
        for sample in self._adapter:
            if (result := self._tokenize(sample)) is not None:
                yield result


def _safe_collate(preprocessor: ChatTemplatePreprocessor, samples) -> Dict[str, Any]:
    valid = [sample for sample in samples if sample is not None]
    if not valid:
        return {}
    return preprocessor.collate(valid)


@dataclass
class TrainerConfig:
    """Configuration for a single training run."""
    data_config: Dict[str, Any] = field(default_factory=dict)
    training_method: str = "qlora"
    method_params: Dict[str, Any] = field(default_factory=dict)
    num_epochs: int = 1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    save_steps: int = 500
    output_dir: str = "output"


class Trainer:
    """Run a single training configuration end to end."""

    def __init__(self, model_path: str, experiment_tracker=None):
        self.model_path = model_path
        self._model = None
        self._processor = None
        self._tracker = experiment_tracker

    def _load_model(self, method_obj, method_config: Optional[Dict[str, Any]] = None) -> None:
        _emit("log", {"message": f"Loading model: {self.model_path}", "level": "INFO"})
        self._processor = load_processor(self.model_path)
        self._model = load_vlm(
            self.model_path,
            quantize_4bit=method_obj.requires_quantization(method_config),
            torch_dtype=torch.bfloat16,
        )

    def _build_dataset(self, config: TrainerConfig):
        data_cfg = dict(config.data_config)
        adapter_name = data_cfg.pop("adapter", "hf_datasets")
        max_samples = int(data_cfg.pop("max_samples", 0) or 0)
        data_cfg.pop("image_root", None)
        if adapter_name != "hf_datasets":
            raise ValueError(
                f"Unsupported data adapter '{adapter_name}'. Only 'hf_datasets' is supported."
            )

        if "dataset_name" not in data_cfg:
            if "data_path" in data_cfg:
                data_cfg["dataset_name"] = data_cfg.pop("data_path")
            elif "dataset" in data_cfg:
                data_cfg["dataset_name"] = data_cfg.pop("dataset")

        return HFDatasetsAdapter(
            max_samples=max_samples if max_samples > 0 else None,
            **data_cfg,
        )

    def _build_tokenized_dataset(self, adapter, config: TrainerConfig):
        preprocessor = ChatTemplatePreprocessor()
        image_root = config.data_config.get("image_root", "")
        logger = lambda sample_id, exc: _emit(
            "log",
            {"message": f"Skipping sample {sample_id}: {exc}", "level": "WARNING"},
        )
        if getattr(adapter, "streaming", False):
            return _TokenizedIterableDataset(
                adapter,
                preprocessor,
                self._processor,
                image_root,
                logger,
            ), preprocessor
        return _TokenizedMapDataset(
            adapter,
            preprocessor,
            self._processor,
            image_root,
            logger,
        ), preprocessor

    def train(self, config: TrainerConfig) -> None:
        _emit("status", {"status": "loading"})

        method_obj = build_training_method(config.training_method)
        method_config = {**method_obj.default_config(), **config.method_params}

        if self._model is None:
            self._load_model(method_obj, method_config)

        _emit("log", {"message": "Loading dataset...", "level": "INFO"})
        adapter = self._build_dataset(config)
        dataset_len = len(adapter)
        if dataset_len < 0:
            raise ValueError(
                "Could not determine dataset length for training. "
                "Please provide a dataset/split with a known size or set max_samples."
            )
        _emit("log", {"message": f"{dataset_len} samples", "level": "INFO"})
        _emit(
            "log",
            {
                "message": (
                    "Dataset resolved to "
                    f"{adapter.dataset_name} split={adapter.split} "
                    f"streaming={adapter.streaming} max_samples={adapter.max_samples or 'full'}"
                ),
                "level": "INFO",
            },
        )

        _emit("log", {"message": "Preprocessing...", "level": "INFO"})
        tokenized_dataset, preprocessor = self._build_tokenized_dataset(adapter, config)
        _emit("log", {"message": f"{dataset_len} samples scheduled for lazy tokenization", "level": "INFO"})

        self._model, info_str = method_obj.prepare_model(
            self._model, self._processor, method_config,
        )
        _emit("log", {"message": info_str, "level": "INFO"})

        param_groups = method_obj.get_trainable_params(self._model)
        for param_group in param_groups:
            param_group.setdefault("lr", config.learning_rate)
        optimizer = AdamW(param_groups, weight_decay=config.weight_decay)

        loader = DataLoader(
            tokenized_dataset,
            batch_size=config.per_device_batch_size,
            shuffle=not getattr(adapter, "streaming", False),
            collate_fn=lambda samples: _safe_collate(preprocessor, samples),
            drop_last=True,
        )
        batches_per_epoch = max(1, dataset_len // config.per_device_batch_size)
        steps_per_epoch = max(1, batches_per_epoch // config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = _cosine_schedule(optimizer, warmup_steps, total_steps)
        effective_batch_size = config.per_device_batch_size * config.gradient_accumulation_steps
        _emit(
            "log",
            {
                "message": (
                    "Training plan: "
                    f"effective_batch_size={effective_batch_size}, "
                    f"batches_per_epoch~{batches_per_epoch}, "
                    f"optimizer_steps_per_epoch~{steps_per_epoch}, "
                    f"total_steps~{total_steps}, warmup_steps={warmup_steps}, "
                    f"output_dir={config.output_dir}"
                ),
                "level": "INFO",
            },
        )

        _emit("status", {"status": "training"})
        self._model.train()
        device = next(self._model.parameters()).device
        global_step = 0
        total_loss = 0.0
        ema_loss = None
        start_time = time.time()
        logged_first_batch = False

        for epoch in range(config.num_epochs):
            for step, batch in enumerate(loader):
                if not batch:
                    continue
                if not logged_first_batch:
                    _emit("log", {"message": _describe_batch(batch), "level": "INFO"})
                    logged_first_batch = True
                batch = _to_device(batch, device)
                batch["labels"] = method_obj.preprocess_labels(
                    batch["input_ids"], batch["labels"], batch_meta=batch,
                )

                forward_batch = {
                    key: value for key, value in batch.items() if key not in _NON_FORWARD_BATCH_KEYS
                }

                outputs = self._model(**forward_batch)
                loss, metrics = method_obj.compute_loss(self._model, batch, outputs)
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % config.gradient_accumulation_steps != 0:
                    continue

                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [param for group in param_groups for param in group["params"]],
                        config.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_loss = loss.item() * config.gradient_accumulation_steps
                total_loss += step_loss
                ema_loss = step_loss if ema_loss is None else (0.98 * ema_loss + 0.02 * step_loss)

                elapsed = time.time() - start_time
                eta = elapsed / global_step * (total_steps - global_step) if global_step > 0 else 0
                step_metrics = {
                    "step": global_step,
                    "total": total_steps,
                    "epoch": epoch,
                    "total_epochs": config.num_epochs,
                    "loss": round(step_loss, 6),
                    "avg_loss": round(total_loss / global_step, 6),
                    "ema_loss": round(ema_loss, 6),
                    "lr": scheduler.get_last_lr()[0],
                    "eta": round(eta, 1),
                    **{key: round(value, 6) for key, value in metrics.items()},
                }
                _emit("metric", step_metrics)

                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    ckpt_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    method_obj.save_checkpoint(self._model, self._processor, ckpt_path, {
                        "base_model": self.model_path,
                        "step": global_step,
                    })

        final_path = os.path.join(config.output_dir, "final")
        if self._tracker is not None:
            final_path = self._tracker.get_checkpoint_dir()

        avg_loss = round(total_loss / max(1, global_step), 6)
        method_obj.save_checkpoint(self._model, self._processor, final_path, {
            "base_model": self.model_path,
            "final_loss": avg_loss,
        })

        if self._tracker is not None:
            trainable = sum(param.numel() for param in self._model.parameters() if param.requires_grad)
            total_params = sum(param.numel() for param in self._model.parameters())
            self._tracker.log_train_summary(
                avg_loss=avg_loss,
                total_steps=global_step,
                train_time_s=time.time() - start_time,
                trainable_params=trainable,
                total_params=total_params,
            )
            self._tracker.set_checkpoint_path(final_path)

        _emit("status", {
            "status": "completed",
            "result": f"{global_step} steps, avg loss={avg_loss:.4f}",
        })
