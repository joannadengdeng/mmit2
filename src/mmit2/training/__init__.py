"""mmit2.training — pluggable training framework for multimodal instruction tuning.

Quick start
-----------
>>> from mmit2.training import TrainingMethod

Register a custom training method::

    from mmit2.registry import register_training_method
    from mmit2.training import TrainingMethod

    class MyMethod(TrainingMethod):
        name = "my-method"
        display_name = "My Method"
        def default_config(self): ...
        def _prepare_model_impl(self, model, processor, config): ...
        def compute_loss(self, model, batch, outputs): ...
        def get_trainable_params(self, model): ...
        def save_checkpoint(self, model, processor, path, metadata): ...
        def load_for_inference(self, path, base_model_id, **kwargs): ...

    register_training_method("my-method", MyMethod)
"""

from mmit2.training.methods import (
    DoRAMethod,
    FreezeTuningMethod,
    L2TMethod,
    LoRAMethod,
    QLoRAMethod,
)
from mmit2.training.methods.base import TrainingMethod

__all__ = [
    "TrainingMethod",
    "QLoRAMethod",
    "LoRAMethod",
    "DoRAMethod",
    "FreezeTuningMethod",
    "L2TMethod",
]
