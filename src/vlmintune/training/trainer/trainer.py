"""Single-stage trainer for multimodal instruction tuning."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.base import load_processor, load_vlm
from vlmintune.training.trainer.helpers import (
    build_label_supervision_debug,
    build_dataset,
    build_skip_logger,
    DebugRecorder,
    cosine_schedule,
    describe_batch,
    emit,
    to_device,
)
from vlmintune.training.trainer.tokenization import build_tokenized_dataset, safe_collate

@dataclass
class TrainerConfig:
    """Configuration for a single training run."""

    data_config: Dict[str, Any] = field(default_factory=dict)
    training_method: str = "qlora"
    method_params: Dict[str, Any] = field(default_factory=dict)
    num_epochs: int = 1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    save_steps: int = 500
    output_dir: str = "output"


class Trainer:
    """Run a single training configuration end to end."""

    def __init__(self, model_name: str, experiment_tracker=None):
        self.model_name = model_name
        self.model_spec = get_model_spec(model_name)
        self.hf_model_id = self.model_spec.hf_model_id
        self.model = None
        self.processor = None
        self.tracker = experiment_tracker

    def load_model(self, method_obj, method_config: Optional[Dict[str, Any]] = None) -> None:
        emit(
            "log",
            {
                "message": f"Loading model: {self.model_name} ({self.hf_model_id})",
                "level": "INFO",
            },
        )
        self.processor = load_processor(self.hf_model_id)
        self.model = load_vlm(
            self.hf_model_id,
            quantize_4bit=method_obj.requires_quantization(method_config),
            torch_dtype=torch.bfloat16,
        )

    def train(self, config: TrainerConfig) -> None:
        emit("status", {"status": "loading"})

        # 1. Resolve the tuning method and load the base model.
        from vlmintune.training.methods.registry import build_training_method

        method_obj = build_training_method(config.training_method)
        method_config = {**method_obj.default_config(), **config.method_params}

        if self.model is None:
            self.load_model(method_obj, method_config)

        # 2. Build the dataset pipeline and lazy tokenization wrapper.
        emit("log", {"message": "Loading dataset...", "level": "INFO"})
        adapter, dataset_len = build_dataset(config)
        debug_recorder = DebugRecorder()

        emit("log", {"message": "Preprocessing...", "level": "INFO"})
        tokenized_dataset, preprocessor = build_tokenized_dataset(
            adapter=adapter,
            processor=self.processor,
            model_config=self.model.config,
            enable_instruction_supervision=config.training_method == "l2t",
            enable_mores_intervention=config.training_method == "mores",
            max_length=config.max_length,
            skip_logger=build_skip_logger(debug_recorder),
            debug_recorder=debug_recorder,
        )
        emit("log", {"message": f"{dataset_len} samples scheduled for lazy tokenization", "level": "INFO"})

        # 3. Prepare trainable parameters and optimizer state.
        self.model, info_str = method_obj.prepare_model(
            self.model, self.processor, method_config, model_spec=self.model_spec,
        )
        emit("log", {"message": info_str, "level": "INFO"})

        param_groups = method_obj.get_trainable_params(self.model)
        for param_group in param_groups:
            param_group.setdefault("lr", config.learning_rate)
        optimizer = AdamW(param_groups, weight_decay=config.weight_decay)

        loader = DataLoader(
            tokenized_dataset,
            batch_size=config.per_device_batch_size,
            shuffle=not getattr(adapter, "streaming", False),
            collate_fn=lambda samples: safe_collate(preprocessor, samples),
            drop_last=True,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
            persistent_workers=(
                config.dataloader_persistent_workers and config.dataloader_num_workers > 0
            ),
        )
        batches_per_epoch = max(1, dataset_len // config.per_device_batch_size)
        steps_per_epoch = max(1, batches_per_epoch // config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = cosine_schedule(optimizer, warmup_steps, total_steps)
        effective_batch_size = config.per_device_batch_size * config.gradient_accumulation_steps
        emit(
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

        # 4. Run the training loop and emit progress metrics.
        emit("status", {"status": "training"})
        self.model.train()
        device = next(self.model.parameters()).device
        global_step = 0
        total_loss = 0.0
        ema_loss = None
        start_time = time.time()
        logged_first_batch = False

        for epoch in range(config.num_epochs):
            for step, batch in enumerate(loader):
                if not batch:
                    continue
                batch = to_device(batch, device)
                labels_before = batch["labels"].clone() if not logged_first_batch else None
                batch["labels"] = method_obj.preprocess_labels(
                    batch["input_ids"], batch["labels"], batch_meta=batch,
                )
                if not logged_first_batch:
                    emit("log", {"message": describe_batch(batch), "level": "INFO"})
                    emit(
                        "debug",
                        build_label_supervision_debug(
                            self.processor,
                            batch["input_ids"],
                            labels_before,
                            batch["labels"],
                            batch.get("instruction_supervision_mask"),
                        ),
                    )
                    logged_first_batch = True

                forward_batch = method_obj.build_forward_batch(batch)
                outputs = self.model(**forward_batch)
                loss, metrics = method_obj.compute_loss(self.model, batch, outputs)
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
                emit("metric", step_metrics)

                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    ckpt_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    method_obj.save_checkpoint(self.model, self.processor, ckpt_path, {
                        "model_name": self.model_name,
                        "hf_model_id": self.hf_model_id,
                        "step": global_step,
                    })

        # 5. Emit debug samples, final weights, and train summary.
        debug_recorder.emit_run_log()

        final_path = os.path.join(config.output_dir, "final")
        if self.tracker is not None:
            final_path = self.tracker.get_checkpoint_dir()

        avg_loss = round(total_loss / max(1, global_step), 6)
        method_obj.save_checkpoint(self.model, self.processor, final_path, {
            "model_name": self.model_name,
            "hf_model_id": self.hf_model_id,
            "final_loss": avg_loss,
        })

        if self.tracker is not None:
            trainable = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
            total_params = sum(param.numel() for param in self.model.parameters())
            self.tracker.write_train_summary(
                {
                    "experiment_name": self.tracker.exp_name,
                    "model_name": self.model_name,
                    "hf_model_id": self.hf_model_id,
                    "training_method": config.training_method,
                    "training_params": {
                        "num_epochs": config.num_epochs,
                        "per_device_batch_size": config.per_device_batch_size,
                        "gradient_accumulation_steps": config.gradient_accumulation_steps,
                        "max_length": config.max_length,
                        "dataloader_num_workers": config.dataloader_num_workers,
                        "dataloader_pin_memory": config.dataloader_pin_memory,
                        "dataloader_persistent_workers": config.dataloader_persistent_workers,
                        "learning_rate": config.learning_rate,
                        "warmup_ratio": config.warmup_ratio,
                        "weight_decay": config.weight_decay,
                        "max_grad_norm": config.max_grad_norm,
                        "save_steps": config.save_steps,
                        "method_params": dict(config.method_params),
                    },
                    "data": dict(config.data_config),
                    "result": {
                        "status": "completed",
                        "avg_loss": avg_loss,
                        "total_steps": global_step,
                        "train_time_s": round(time.time() - start_time, 1),
                        "trainable_params": trainable,
                        "total_params": total_params,
                        "trainable_pct": round(100 * trainable / max(1, total_params), 4),
                    },
                }
            )

        emit("status", {
            "status": "completed",
            "result": f"{global_step} steps, avg loss={avg_loss:.4f}",
        })
