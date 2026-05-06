"""Registry of built-in HF VQA dataset specs."""
from __future__ import annotations

from typing import Dict, Optional

from mmit2.data.datasets.base import ColumnMapping, ConfiguredVQASpec, HFDatasetSpec
from mmit2.data.datasets.textvqa import TextVQASpec
from mmit2.data.datasets.vizwiz import VizWizVQASpec
from mmit2.data.datasets.vqav2 import VQAv2Spec


_SPEC_CLASSES = (
    VQAv2Spec,
    TextVQASpec,
    VizWizVQASpec,
)

DATASET_SPECS: Dict[str, HFDatasetSpec] = {
    spec_cls.dataset_name: spec_cls()
    for spec_cls in _SPEC_CLASSES
}


def get_dataset_spec(dataset_name: str) -> Optional[HFDatasetSpec]:
    return DATASET_SPECS.get(dataset_name)


def build_configured_spec(
    dataset_name: str,
    mapping: ColumnMapping,
    *,
    prefer_streaming: bool = False,
) -> HFDatasetSpec:
    return ConfiguredVQASpec(
        dataset_name,
        mapping,
        prefer_streaming=prefer_streaming,
    )
