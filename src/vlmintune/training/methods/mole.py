"""Fixed LLaVA-MoLE recipe for single-dataset instruction tuning.

The attention projections use ordinary LoRA.  Every feed-forward projection
uses three sparse LoRA experts, while the three projections in one decoder
block share one token router and therefore make the same top-1 expert choice.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vlmintune.models.registry import get_model_spec
from vlmintune.training.methods.base import TrainingMethod, load_processor, load_vlm
from vlmintune.training.trainer.ce_loss import CrossEntropyLoss


CROSS_ENTROPY_LOSS = CrossEntropyLoss()

MOLE_MODEL_NAME = "llava15_7b"
MOLE_RANK = 32
MOLE_ALPHA = 16
MOLE_DROPOUT = 0.05
MOLE_NUM_EXPERTS = 3
MOLE_BALANCE_LOSS_WEIGHT = 1e-2
MOLE_CHECKPOINT_FORMAT = "llava_mole_single_dataset_v1"
MOLE_CHECKPOINT_FILENAME = "mole_tuned.pt"

ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _adapter_linear(in_features: int, out_features: int, device: torch.device) -> nn.Linear:
    """Create trainable LoRA/router weights in fp32 on the base layer device."""
    return nn.Linear(
        in_features,
        out_features,
        bias=False,
        device=device,
        dtype=torch.float32,
    )


class LoRAExpert(nn.Module):
    """One rank-32 LoRA branch with the fixed paper recipe."""

    def __init__(self, in_features: int, out_features: int, device: torch.device) -> None:
        super().__init__()
        self.dropout = nn.Dropout(MOLE_DROPOUT)
        self.lora_a = _adapter_linear(in_features, MOLE_RANK, device)
        self.lora_b = _adapter_linear(MOLE_RANK, out_features, device)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.lora_a.weight.dtype)
        return self.lora_b(self.lora_a(self.dropout(hidden_states)))


class LoRALinear(nn.Module):
    """Frozen linear layer plus one ordinary LoRA branch."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap torch.nn.Linear modules.")
        self.base = base
        _freeze(self.base)
        self.scaling = float(MOLE_ALPHA) / float(MOLE_RANK)
        self.expert = LoRAExpert(base.in_features, base.out_features, base.weight.device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        update = self.expert(hidden_states).to(base_output.dtype)
        return base_output + self.scaling * update


class SharedTop1Router(nn.Module):
    """One per decoder block; supplies one route to all three FFN projections."""

    def __init__(self, hidden_size: int, device: torch.device) -> None:
        super().__init__()
        self.proj = _adapter_linear(hidden_size, MOLE_NUM_EXPERTS, device)
        self._active_route: Optional[torch.Tensor] = None
        self.last_balance_loss: Optional[torch.Tensor] = None
        self.last_counts: Optional[torch.Tensor] = None
        self.last_probability_totals: Optional[torch.Tensor] = None
        self._token_mask: Optional[torch.Tensor] = None
        self._active_groups: Optional[Tuple[torch.Tensor, ...]] = None

    def set_token_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        self._token_mask = (
            attention_mask.bool() if attention_mask is not None else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self._token_mask is None:
            token_mask = torch.ones(
                hidden_states.shape[:-1],
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            sequence_length = hidden_states.shape[-2]
            token_mask = self._token_mask[..., -sequence_length:].to(
                hidden_states.device
            )
        flat_mask = token_mask.reshape(-1)
        token_indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
        routed_hidden = flat_hidden.index_select(0, token_indices)
        logits = self.proj(routed_hidden.to(self.proj.weight.dtype))
        probabilities = torch.softmax(logits, dim=-1)
        selected_experts = probabilities.argmax(dim=-1)

        route = torch.full(
            (flat_hidden.shape[0],),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        route = route.index_copy(0, token_indices, selected_experts)
        route = route.reshape(hidden_states.shape[:-1])

        flat_route = selected_experts
        flat_probabilities = probabilities.reshape(-1, MOLE_NUM_EXPERTS)
        counts = F.one_hot(flat_route, num_classes=MOLE_NUM_EXPERTS).sum(dim=0)
        counts = counts.to(flat_probabilities.dtype)
        probability_totals = flat_probabilities.sum(dim=0)

        # Eq. (7) in LLaVA-MoLE: sum_j c_j * p_j.  c_j is discrete,
        # so only p_j carries gradients into the router.
        balance_loss = (counts.detach() * probability_totals).sum()
        return route, counts, probability_totals, balance_loss

    def begin(self, hidden_states: torch.Tensor) -> None:
        route, counts, probability_totals, balance_loss = self(hidden_states)
        self._active_route = route
        flat_route = route.reshape(-1)
        self._active_groups = tuple(
            torch.nonzero(flat_route == expert_index, as_tuple=False).flatten()
            for expert_index in range(MOLE_NUM_EXPERTS)
        )
        self.last_balance_loss = balance_loss
        self.last_counts = counts.detach()
        self.last_probability_totals = probability_totals.detach()

    def route_for(self, hidden_states: torch.Tensor) -> torch.Tensor:
        route = self._active_route
        if route is None:
            raise RuntimeError(
                "MoLE FFN projection ran without an active block route. "
                "The MLP router hook was not installed or did not execute."
            )
        expected_shape = hidden_states.shape[:-1]
        if route.shape != expected_shape:
            raise RuntimeError(
                "MoLE shared route shape mismatch: "
                f"route={tuple(route.shape)}, projection_input={tuple(expected_shape)}."
            )
        return route

    def groups_for(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        self.route_for(hidden_states)
        if self._active_groups is None:
            raise RuntimeError("MoLE token groups are unavailable outside an MLP call.")
        return self._active_groups

    def end(self) -> None:
        self._active_route = None
        self._active_groups = None


class MoELoRALinear(nn.Module):
    """Frozen linear plus grouped top-1 LoRA experts.

    The router is deliberately kept as a non-registered reference.  It is
    registered exactly once on the parent MLP, so checkpoint state contains a
    single router per block rather than three aliases.
    """

    def __init__(self, base: nn.Linear, router: SharedTop1Router) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("MoELoRALinear can only wrap torch.nn.Linear modules.")
        self.base = base
        _freeze(self.base)
        self.scaling = float(MOLE_ALPHA) / float(MOLE_RANK)
        self.experts = nn.ModuleList(
            [
                LoRAExpert(base.in_features, base.out_features, base.weight.device)
                for _ in range(MOLE_NUM_EXPERTS)
            ]
        )
        object.__setattr__(self, "_shared_router", router)

    @property
    def router(self) -> SharedTop1Router:
        return object.__getattribute__(self, "_shared_router")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        token_groups = self.router.groups_for(hidden_states)

        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_update = torch.zeros(
            (flat_hidden.shape[0], base_output.shape[-1]),
            device=base_output.device,
            dtype=base_output.dtype,
        )

        # Sparse execution: a missed expert is not called at all.
        for expert, token_indices in zip(self.experts, token_groups):
            if token_indices.numel() == 0:
                continue
            expert_inputs = flat_hidden.index_select(0, token_indices)
            expert_update = expert(expert_inputs).to(base_output.dtype)
            flat_update = flat_update.index_add(0, token_indices, expert_update)

        update = flat_update.reshape(base_output.shape)
        return base_output + self.scaling * update


def _require_linear(parent: nn.Module, name: str, location: str) -> nn.Linear:
    module = getattr(parent, name, None)
    if not isinstance(module, nn.Linear):
        actual = "missing" if module is None else module.__class__.__name__
        raise ValueError(
            f"LLaVA-MoLE expected nn.Linear at {location}.{name}, found {actual}."
        )
    return module


def _router_pre_hook(router: SharedTop1Router):
    def hook(module, args, kwargs):
        del module
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError("LLaVA-MoLE could not read the MLP hidden states.")
        router.begin(hidden_states)
        return None

    return hook


def _router_post_hook(router: SharedTop1Router):
    def hook(module, args, kwargs, output):
        del module, args, kwargs
        router.end()
        return output

    return hook


def _install_block_mole(layer: nn.Module, layer_index: int) -> Tuple[SharedTop1Router, list[Any]]:
    self_attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    if not isinstance(self_attn, nn.Module) or not isinstance(mlp, nn.Module):
        raise ValueError(
            f"LLaVA-MoLE expected self_attn and mlp modules in language layer {layer_index}."
        )
    if hasattr(mlp, "mole_router"):
        raise ValueError(f"Language layer {layer_index} is already prepared for MoLE.")

    attention_linears = {
        name: _require_linear(self_attn, name, f"language_layers.{layer_index}.self_attn")
        for name in ATTENTION_PROJECTIONS
    }
    ffn_linears = {
        name: _require_linear(mlp, name, f"language_layers.{layer_index}.mlp")
        for name in FFN_PROJECTIONS
    }
    if ffn_linears["gate_proj"].in_features != ffn_linears["up_proj"].in_features:
        raise ValueError(
            f"LLaVA-MoLE layer {layer_index} gate_proj/up_proj input sizes do not match."
        )

    for name, base in attention_linears.items():
        setattr(self_attn, name, LoRALinear(base))

    router = SharedTop1Router(
        ffn_linears["gate_proj"].in_features,
        ffn_linears["gate_proj"].weight.device,
    )
    mlp.mole_router = router
    for name, base in ffn_linears.items():
        setattr(mlp, name, MoELoRALinear(base, router))

    handles = [
        mlp.register_forward_pre_hook(_router_pre_hook(router), with_kwargs=True),
        mlp.register_forward_hook(_router_post_hook(router), with_kwargs=True),
    ]
    return router, handles


def _trainable_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names:
        raise ValueError("LLaVA-MoLE model has no trainable parameters to save.")
    return {name: state_dict[name].detach().cpu() for name in names}


def _load_trainable_state(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    actual = set(state_dict)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "Invalid fixed LLaVA-MoLE checkpoint state: "
            f"missing={missing}, unexpected={unexpected}."
        )
    model.load_state_dict(state_dict, strict=False)


class MoLEMethod(TrainingMethod):
    """LLaVA-1.5-only, single-dataset, fixed LLaVA-MoLE v1 recipe."""

    name = "mole"
    display_name = "LLaVA-MoLE (fixed single-dataset v1)"

    def forward_pre_hook(self, module, args, kwargs):
        del module, args
        attention_mask = kwargs.get("attention_mask")
        for router in self.routers:
            router.set_token_mask(attention_mask)
        return None

    def prepare_model_impl(self, model, processor, model_spec):
        del processor
        if model_spec.name != MOLE_MODEL_NAME:
            raise ValueError(
                "LLaVA-MoLE v1 only supports model.name='llava15_7b'; "
                f"received '{model_spec.name}'."
            )
        _freeze(model)
        layers = list(model_spec.get_transformer_layers(model))
        if not layers:
            raise ValueError("LLaVA-MoLE found no language transformer layers.")

        routers: list[SharedTop1Router] = []
        hook_handles: list[Any] = []
        for layer_index, layer in enumerate(layers):
            router, handles = _install_block_mole(layer, layer_index)
            routers.append(router)
            hook_handles.extend(handles)

        # Routers are already registered once under each MLP.  Tuples provide
        # convenient runtime access without creating duplicate state-dict paths.
        model.mole_routers = tuple(routers)
        self.routers = tuple(routers)
        hook_handles.append(
            model.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        )
        model.mole_hook_handles = tuple(hook_handles)
        model.vlmintuneMoleMethod = self

        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        info = (
            f"LLaVA-MoLE fixed v1: backbone={MOLE_MODEL_NAME}, layers={len(layers)}, "
            f"r={MOLE_RANK}, alpha={MOLE_ALPHA}, dropout={MOLE_DROPOUT}, "
            f"experts={MOLE_NUM_EXPERTS}, top_k=1\n"
            "Attention: q/k/v/o ordinary LoRA; FFN: gate/up/down sparse experts "
            "with one shared router per block\n"
            f"Balance loss: mean_layers(sum_j(c_j*p_j)) * {MOLE_BALANCE_LOSS_WEIGHT}\n"
            f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)"
        )
        return model, info

    def compute_loss(self, model, batch, outputs):
        ce_loss, metrics = CROSS_ENTROPY_LOSS.compute(model, batch, outputs)
        routers: Sequence[SharedTop1Router] = getattr(model, "mole_routers", ())
        layer_losses = [router.last_balance_loss for router in routers]
        if not layer_losses or any(loss is None for loss in layer_losses):
            raise RuntimeError(
                "LLaVA-MoLE balance loss is unavailable; every language MLP must run "
                "before compute_loss."
            )
        balance_loss = torch.stack(
            [loss.to(device=ce_loss.device, dtype=torch.float32) for loss in layer_losses]
        ).mean()
        auxiliary_loss = MOLE_BALANCE_LOSS_WEIGHT * balance_loss
        total_loss = ce_loss + auxiliary_loss.to(ce_loss.dtype)
        metrics = {
            **metrics,
            "ce_loss": float(ce_loss.detach()),
            "mole_balance_loss": float(balance_loss.detach()),
            "mole_auxiliary_loss": float(auxiliary_loss.detach()),
        }
        return total_loss, metrics

    def get_trainable_params(self, model: nn.Module) -> List[Dict[str, Any]]:
        return [
            {
                "params": [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
            }
        ]

    def save_checkpoint(self, model, processor, path, metadata):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            "format": MOLE_CHECKPOINT_FORMAT,
            "state_dict": _trainable_state(model),
        }
        torch.save(checkpoint, os.path.join(path, MOLE_CHECKPOINT_FILENAME))
        processor.save_pretrained(path)

        saved_metadata = {
            **dict(metadata),
            "ft_method": self.name,
            "mole_checkpoint_format": MOLE_CHECKPOINT_FORMAT,
            "mole_recipe": {
                "model_name": MOLE_MODEL_NAME,
                "rank": MOLE_RANK,
                "alpha": MOLE_ALPHA,
                "dropout": MOLE_DROPOUT,
                "num_experts": MOLE_NUM_EXPERTS,
                "top_k": 1,
                "balance_loss_weight": MOLE_BALANCE_LOSS_WEIGHT,
            },
        }
        with open(os.path.join(path, "vlmintune_meta.json"), "w", encoding="utf-8") as handle:
            json.dump(saved_metadata, handle, indent=2, ensure_ascii=False)

    def load_for_inference(self, path, model_name, **kwargs):
        del kwargs
        model_spec = get_model_spec(model_name)
        if model_spec.name != MOLE_MODEL_NAME:
            raise ValueError(
                "LLaVA-MoLE v1 checkpoints only load with model_name='llava15_7b'."
            )

        with open(os.path.join(path, "vlmintune_meta.json"), "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("ft_method") != self.name or metadata.get(
            "mole_checkpoint_format"
        ) != MOLE_CHECKPOINT_FORMAT:
            raise ValueError("Checkpoint metadata is not fixed LLaVA-MoLE v1 format.")

        checkpoint = torch.load(
            os.path.join(path, MOLE_CHECKPOINT_FILENAME),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != MOLE_CHECKPOINT_FORMAT:
            raise ValueError("Checkpoint weights are not fixed LLaVA-MoLE v1 format.")
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Invalid fixed LLaVA-MoLE checkpoint: missing state_dict.")

        processor = load_processor(model_spec.hf_model_id)
        model = load_vlm(
            model_spec.hf_model_id,
            quantize_4bit=False,
            torch_dtype=torch.bfloat16,
        )
        model, _ = self.prepare_model(model, processor, model_spec)
        _load_trainable_state(model, state_dict)
        model.eval()

        adapter_name = os.path.basename(os.path.normpath(path))
        info = {
            "model_id": (
                f"{model_spec.hf_model_id} (LLaVA-MoLE fixed v1: {adapter_name})"
            )
        }
        return model, processor, info


__all__ = [
    "ATTENTION_PROJECTIONS",
    "FFN_PROJECTIONS",
    "MOLE_ALPHA",
    "MOLE_BALANCE_LOSS_WEIGHT",
    "MOLE_CHECKPOINT_FILENAME",
    "MOLE_CHECKPOINT_FORMAT",
    "MOLE_DROPOUT",
    "MOLE_MODEL_NAME",
    "MOLE_NUM_EXPERTS",
    "MOLE_RANK",
    "LoRAExpert",
    "LoRALinear",
    "MoELoRALinear",
    "MoLEMethod",
    "SharedTop1Router",
]
