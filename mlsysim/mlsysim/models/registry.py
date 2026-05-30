"""Model registry — workload (model) specs.

Per-category data (params, FLOPs, training figures) lives as YAML under
``models/data/<category>.yaml`` and is loaded + validated against the
``Workload`` family at import (see ``core/loader.py``). Each entry carries a
``__type__`` selecting the concrete workload class (Transformer / CNN / SSM /
Diffusion / sparse-MoE / plain). See ``.claude/rules/mlsysim.md`` →
*Canonical organization* / *Storage format*.
"""
from pathlib import Path

from ..core.registry import Registry
from ..core.loader import load_collection
from .types import (
    Workload, TransformerWorkload, SparseTransformerWorkload,
    CNNWorkload, SSMWorkload, DiffusionWorkload,
)

_DATA = Path(__file__).parent / "data"
_WORKLOAD_TYPES = {
    "Workload": Workload,
    "TransformerWorkload": TransformerWorkload,
    "SparseTransformerWorkload": SparseTransformerWorkload,
    "CNNWorkload": CNNWorkload,
    "SSMWorkload": SSMWorkload,
    "DiffusionWorkload": DiffusionWorkload,
}


def _load(stem: str, name: str, doc: str) -> type:
    return load_collection(_DATA / f"{stem}.yaml", name=name, doc=doc, type_map=_WORKLOAD_TYPES)


LanguageModels = _load("language", "LanguageModels", "Registry namespace for LanguageModels.")
VisionModels = _load("vision", "VisionModels", "Registry namespace for VisionModels.")
TinyModels = _load("tiny", "TinyModels", "Registry namespace for TinyModels.")
RecommendationModels = _load("recommendation", "RecommendationModels", "")
StateSpaceModels = _load("statespace", "StateSpaceModels", "")
GenerativeVisionModels = _load("generativevision", "GenerativeVisionModels", "")


class Models(Registry):
    """Registry namespace for Models."""
    Language = LanguageModels
    Vision = VisionModels
    Tiny = TinyModels
    Recommendation = RecommendationModels
    StateSpace = StateSpaceModels
    GenerativeVision = GenerativeVisionModels
