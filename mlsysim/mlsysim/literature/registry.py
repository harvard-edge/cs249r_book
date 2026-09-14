"""Published literature anchors used by MLSysIM (MFU, Chinchilla, benchmarks, ...).

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
Benchmarks = _load("benchmarks", "Benchmarks", "Published benchmark anchors used by empirical validation tests.")
Chinchilla = _load("chinchilla", "Chinchilla")
Communication = _load("communication", "Communication")
BatchSize = _load("batchsize", "BatchSize", "McCandlish et al. (2018) critical batch size estimates.")
Fairness = _load("fairness", "Fairness", "Buolamwini & Gebru (2018) Gender Shades intersectional error rates.")
Surveys = _load("surveys", "Surveys", "Industry survey figures (CrowdFlower 2016 data-science time allocation).")
Crypto = _load("crypto", "Crypto", "Security/crypto overhead anchors (TEE, FHE, HSM) for the security_privacy chapter.")
ResponsibleAIOverhead = _load("rai_overhead", "ResponsibleAIOverhead", "Responsible-AI technique overhead ranges for the responsible_ai chapter.")


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


# ---------------------------------------------------------------------------
# Edge-inference latency/energy scatter for @fig-edge-inference-landscape. A
# multi-field series (device, latency, energy, tier), so it lives in Python like
# ComputeTrend above. Values are published MLPerf Tiny / vendor measurements.
# ---------------------------------------------------------------------------
_EDGE_INFERENCE_PROVENANCE = Provenance(
    kind=ProvenanceKind.LITERATURE,
    ref=(
        "MLPerf Tiny benchmark (Banbury et al. 2021) and vendor inference "
        "benchmarks (Syntiant, STMicroelectronics, Google Coral, NVIDIA Jetson)"
    ),
    url="https://arxiv.org/abs/2106.07597",
    verified="2026-06-11",
)

# (name, latency_ms, energy_uj, tier)  tier: 0 = MCU/ASIC, 1 = edge accelerator, 2 = edge GPU
_EDGE_INFERENCE_ROWS = (
    ("Syntiant Core 2\n(KWS, low-energy)", 4.4, 31.5, 0),
    ("Syntiant Core 2\n(KWS, high-perf)", 1.5, 43.8, 0),
    ("STM32N6 NPU\n(KWS, low-power)", 1.6, 156.5, 0),
    ("STM32N6 NPU\n(IC, high-perf)", 2.9, 443.9, 0),
    ("Syntiant Core 2\n(VWW)", 12.7, 71.7, 0),
    ("Syntiant Core 2\n(IC)", 5.1, 139.4, 0),
    ("Google Coral\nEdge TPU", 2.4, 4800, 1),
    ("Jetson AGX Orin\n(ResNet-50)", 0.64, 15000, 2),
)


class EdgeInferenceBenchmarks:
    """Published edge-inference latency (ms) and energy (uJ) by device and tier.

    Source: see ``provenance``. Consumed by @fig-edge-inference-landscape.
    Each entry is a dict ``{"name", "latency_ms", "energy_uj", "tier"}``.
    """

    provenance = _EDGE_INFERENCE_PROVENANCE
    entries = tuple(
        {"name": n, "latency_ms": l, "energy_uj": e, "tier": t}
        for (n, l, e, t) in _EDGE_INFERENCE_ROWS
    )
    list = entries  # parity with the Registry `.list` convention


# ---------------------------------------------------------------------------
# Kaplan et al. (2020) test-loss power laws for @fig-model-scaling. Each law is
# a published closed-form fit L = (X / x_c)^(-alpha) over the studied range, so
# the figure regenerates from these constants instead of a traced raster. The
# constants are the ones printed on the source figure (Kaplan et al. 2020).
# ---------------------------------------------------------------------------
_KAPLAN_SCALING_PROVENANCE = Provenance(
    kind=ProvenanceKind.LITERATURE,
    ref="Kaplan et al. (2020), Scaling Laws for Neural Language Models",
    url="https://arxiv.org/abs/2001.08361",
    verified="2026-06-13",
)

# (axis, x_c, alpha, x_min, x_max)  -- L = (X / x_c)^(-alpha) on [x_min, x_max]
_KAPLAN_SCALING_ROWS = (
    ("compute", 2.3e8, 0.050, 1e-9, 1e1),   # C_min, PF-days (non-embedding)
    ("data", 5.4e13, 0.095, 1e8, 1e10),     # D, tokens
    ("params", 8.8e13, 0.076, 1e3, 1e9),    # N, non-embedding parameters
)


class KaplanScalingLaws:
    """Kaplan et al. (2020) test-loss power laws vs compute, data, parameters.

    Source: see ``provenance``. Consumed by @fig-model-scaling. Each entry is a
    dict ``{"axis", "x_c", "alpha", "x_min", "x_max"}`` giving the published fit
    ``L = (X / x_c)^(-alpha)`` over the studied range ``[x_min, x_max]``.
    """

    provenance = _KAPLAN_SCALING_PROVENANCE
    entries = tuple(
        {"axis": k, "x_c": xc, "alpha": a, "x_min": lo, "x_max": hi}
        for (k, xc, a, lo, hi) in _KAPLAN_SCALING_ROWS
    )
    list = entries  # parity with the Registry `.list` convention


class Literature(Registry):
    """Registry namespace for Literature."""
    Training = Training
    Benchmarks = Benchmarks
    Chinchilla = Chinchilla
    Communication = Communication
    BatchSize = BatchSize
    Fairness = Fairness
    Surveys = Surveys
    Crypto = Crypto
    ResponsibleAIOverhead = ResponsibleAIOverhead
    ComputeTrend = ComputeTrend
    EdgeInferenceBenchmarks = EdgeInferenceBenchmarks
    KaplanScalingLaws = KaplanScalingLaws
