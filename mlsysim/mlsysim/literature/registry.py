"""Published literature anchors used by registry-backed calculations (MFU, Chinchilla, …).

Each anchor is a provenance-carrying scalar; the values live as YAML under
``literature/data/<category>.yaml`` and are loaded via ``load_sourced_registry``
(provenance referenced by catalog key). See the project MLSysIM rules →
*Storage format*. Literature holds ONLY genuinely-cited field figures.
"""
from pathlib import Path

from ..core.registry import Registry
from ..core.loader import load_sourced_registry

_DATA = Path(__file__).parent / "data"


def _load(stem: str, name: str, doc: str = "") -> type:
    return load_sourced_registry(_DATA / f"{stem}.yaml", name=name, doc=doc)


Training = _load("training", "Training")
Chinchilla = _load("chinchilla", "Chinchilla")
Communication = _load("communication", "Communication")
BatchSize = _load("batchsize", "BatchSize", "McCandlish et al. (2018) critical batch size estimates.")


# ---------------------------------------------------------------------------
# Compute-growth trend (a SERIES, not a scalar anchor) for
# @fig-ai-training-compute-growth. The scalar-anchor YAML schema used by the
# other literature data files cannot express a multi-field series, so this
# curated milestone list lives in Python (per the mlsysim taxonomy rule:
# "composing / series -> Python"). Values are order-of-magnitude
# training-compute (FLOP) estimates, not vendor disclosures.
# ---------------------------------------------------------------------------
from ..core.provenance import Provenance, ProvenanceKind

_COMPUTE_TREND_PROVENANCE = Provenance(
    kind=ProvenanceKind.LITERATURE,
    ref="Sevilla et al. (2022), Compute Trends Across Three Eras of Machine Learning (Epoch AI)",
    url="https://arxiv.org/abs/2202.05924",
    verified="2026-05-30",
)

# (model, calendar year, training FLOP, era)
_COMPUTE_TREND_ROWS = (
    ("AlexNet", 2012, 1.2e18, "Deep Learning"),
    ("VGG-16", 2014, 3.0e19, "Deep Learning"),
    ("ResNet-50", 2015, 5.0e19, "Deep Learning"),
    ("GoogLeNet", 2014, 2.0e19, "Deep Learning"),
    ("AlphaGoZero", 2017, 3.0e22, "Deep Learning"),
    ("Transformer", 2017, 5.0e20, "Large Scale"),
    ("BERT-Large", 2018, 3.0e21, "Large Scale"),
    ("GPT-2-XL", 2019, 2.0e22, "Large Scale"),
    ("T5-11 billion", 2019, 5.0e22, "Large Scale"),
    ("GPT-3", 2020, 3.1e23, "Large Scale"),
    ("Gopher", 2021, 6.0e23, "Large Scale"),
    ("PaLM", 2022, 2.5e24, "Large Scale"),
    ("GPT-4-class", 2023, 2.0e25, "Large Scale"),
    ("Llama-3-70 billion", 2024, 8.0e24, "Large Scale"),
    ("Gemini-Ultra", 2024, 5.0e25, "Large Scale"),
    ("Llama 3.1 405B", 2024, 3.8e25, "Large Scale"),
    ("Grok-3", 2025, 3.0e26, "Large Scale"),
)


class ComputeTrend:
    """Representative training-compute milestones (FLOP) by year and era.

    Source: see ``provenance``. Consumed by @fig-ai-training-compute-growth.
    Each entry is a dict ``{"model", "year", "flops", "era"}``.
    """

    provenance = _COMPUTE_TREND_PROVENANCE
    entries = tuple(
        {"model": m, "year": y, "flops": f, "era": e}
        for (m, y, f, e) in _COMPUTE_TREND_ROWS
    )
    list = entries  # parity with the Registry `.list` convention


class Literature(Registry):
    """Registry namespace for Literature."""
    Training = Training
    Chinchilla = Chinchilla
    Communication = Communication
    BatchSize = BatchSize
    ComputeTrend = ComputeTrend
