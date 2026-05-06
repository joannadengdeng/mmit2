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
from torch.utils.data import DataLoader

from mmit2.data.adapters.hf_datasets import HFDatasetsAdapter
from mmit2.modeling import load_processor, load_vlm
from mmit2.registry import build_training_method
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
        if adapter_name != "hf_datasets":
            raise ValueError(
                f"Unsupported data adapter '{adapter_name}'. Only 'hf_datasets' is supported."
            )

        if "dataset_name" not in data_cfg:
            if "data_path" in data_cfg:
                data_cfg["dataset_name"] = data_cfg.pop("data_path")
            elif "dataset" in data_cfg:
                data_cfg["dataset_name"] = data_cfg.pop("dataset")

        adapter = HFDatasetsAdapter(**data_cfg)
        samples = list(adapter)
        if max_samples > 0:
            samples = samples[:max_samples]
        return samples

    def _preprocess_dataset(self, samples, config: TrainerConfig):
        preprocessor = ChatTemplatePreprocessor()
        processed = []
        image_root = config.data_config.get("image_root", "")
        max_length = 2048

        for sample in samples:
            try:
                tokenized = preprocessor.tokenize(
                    sample,
                    self._processor,
                    image_root=image_root,
                    max_length=max_length,
                )
                processed.append(tokenized)
            except Exception as exc:
                _emit("log", {"message": f"Skipping sample {sample.id}: {exc}", "level": "WARNING"})

        return processed, preprocessor

    def train(self, config: TrainerConfig) -> None:
        _emit("status", {"status": "loading"})

        method_obj = build_training_method(config.training_method)
        method_config = {**method_obj.default_config(), **config.method_params}

        if self._model is None:
            self._load_model(method_obj, method_config)

        _emit("log", {"message": "Loading dataset...", "level": "INFO"})
        samples = self._build_dataset(config)
        _emit("log", {"message": f"{len(samples)} samples", "level": "INFO"})

        _emit("log", {"message": "Preprocessing...", "level": "INFO"})
        processed, preprocessor = self._preprocess_dataset(samples, config)
        _emit("log", {"message": f"{len(processed)} samples tokenized", "level": "INFO"})
        if not processed:
            raise ValueError("No samples after preprocessing")

        self._model, info_str = method_obj.prepare_model(
            self._model, self._processor, method_config,
        )
        _emit("log", {"message": info_str, "level": "INFO"})

        param_groups = method_obj.get_trainable_params(self._model)
        for param_group in param_groups:
            param_group.setdefault("lr", config.learning_rate)
        optimizer = AdamW(param_groups, weight_decay=config.weight_decay)

        loader = DataLoader(
            processed,
            batch_size=config.per_device_batch_size,
            shuffle=True,
            collate_fn=preprocessor.collate,
            drop_last=True,
        )
        steps_per_epoch = max(1, len(loader) // config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = _cosine_schedule(optimizer, warmup_steps, total_steps)

        _emit("status", {"status": "training"})
        self._model.train()
        device = next(self._model.parameters()).device
        global_step = 0
        total_loss = 0.0
        start_time = time.time()

        for epoch in range(config.num_epochs):
            for step, batch in enumerate(loader):
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
                total_loss += loss.item() * config.gradient_accumulation_steps

                elapsed = time.time() - start_time
                eta = elapsed / global_step * (total_steps - global_step) if global_step > 0 else 0
                step_metrics = {
                    "step": global_step,
                    "total": total_steps,
                    "epoch": epoch,
                    "total_epochs": config.num_epochs,
                    "loss": round(loss.item() * config.gradient_accumulation_steps, 6),
                    "avg_loss": round(total_loss / global_step, 6),
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
