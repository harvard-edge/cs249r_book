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
TrainingRuns = _load("training_runs", "TrainingRuns", "Cited training-run field figures (GPU counts, durations, MFU, interruptions).")
FleetOperations = _load("fleet_operations", "FleetOperations", "Measured fleet-operations results (job MTTF, utilization, cluster scale).")
DatacenterElectricity = _load("datacenter_electricity", "DatacenterElectricity", "Datacenter electricity consumption and projections (LBNL, IEA).")
AcceleratorLifecycle = _load("accelerator_lifecycle", "AcceleratorLifecycle", "Accelerator life-cycle emission trends (Schneider et al. 2025).")
StorageSystems = _load("storage_systems", "StorageSystems", "Measured production storage-system results (3FS read stress test).")
# Incident investigation reports (added 2026-10-20)
IncidentReports = _load("incident_reports", "IncidentReports", "Values reported by incident investigations (NTSB Tempe, Cruise pullover).")
# Model calls and agent trajectories (added 2026-10-20)
ServingIncidents = _load("serving_incidents", "ServingIncidents", "Reported production serving-incident figures (Anthropic 2025 postmortem).")
SelfCorrection = _load("self_correction", "SelfCorrection", "Intrinsic self-correction results (Huang et al. 2024, GPT-4).")
# Added 2026-10-20: benchmark error rates and documented accuracy failures.
PredictionQuality = _load("prediction_quality", "PredictionQuality", "Cited prediction-quality results (ILSVRC-2012 errors, Google Flu Trends misses).")
GamePlaying = _load("game_playing", "GamePlaying", "Cited results of landmark game-playing systems (Deep Blue, AlphaGo Zero).")
# Deployment-tier field figures (added 2026-10-20)
DeviceMeasurements = _load("device_measurements", "DeviceMeasurements", "Cited measurements of deployed ML hardware (TPU v1 roofline, memory-wall growth, phone stress test, two-pass wake word).")
# Spaceflight anomaly inquiries (added 2026-10-20)
SpaceflightAnomalies = _load("spaceflight_anomalies", "SpaceflightAnomalies", "Cited results of spaceflight anomaly inquiries (ESA Schiaparelli 2017).")
# Diabetic retinopathy screening studies (added 2026-10-20)
ScreeningStudies = _load("screening_studies", "ScreeningStudies", "Cited results of diabetic retinopathy screening systems (Gulshan 2016, Beede 2020, FDA DEN180001).")
# Test-time compute results (added 2026-10-20)
TestTimeCompute = _load("test_time_compute", "TestTimeCompute", "Cited test-time compute results (Snell et al. 2024; PRM800K, Lightman et al. 2024).")
# Context engineering (added 2026-10-20)
ContextUse = _load("context_use", "ContextUse", "Cited results on how models use a request: evidence position, an irrelevant sentence, the RULER bar.")
AgentRetrieval = _load("agent_retrieval", "AgentRetrieval", "Cited results on how agents retrieve: SWE-agent against BM25 retrieval, PoisonedRAG's injection count.")  # added 2026-10-20
SummarizationFaithfulness = _load("summarization_faithfulness", "SummarizationFaithfulness", "Cited summarization-faithfulness results (Maynez et al. 2020, XSum).")
PrefixCaching = _load("prefix_caching", "PrefixCaching", "Cited prefix-cache results from a public serving deployment (SGLang 2024).")
# Published vision-language-action and dual-system robot policies (added 2026-10-20)
VlaSystems = _load("vla_systems", "VlaSystems", "Cited figures of published VLA and dual-system robot policies (OpenVLA, ACT, pi0, FAST, Helix, GR00T N1).")


# ---------------------------------------------------------------------------
# Compute-growth trend (a SERIES, not a scalar anchor) for
# @fig-ai-training-compute-growth. The scalar-anchor YAML schema used by the
# other literature data files cannot express a multi-field series, so this
# curated milestone list lives in Python (per the mlsysim taxonomy rule:
# "composing / series -> Python"). Values are third-party training-compute
# (FLOP) estimates from Epoch AI's Notable AI Models database, not vendor
# disclosures. Era: "Large Scale" marks runs above 10^22 FLOP.
#
# 2026-10-12: the pre-2020 rows were 1.5-67x above Epoch's estimates (e.g.
# Transformer 5e20 vs 7.4e18, AlphaGo Zero 3e22 vs 6.5e20) and ResNet-50 had no
# source; all rows re-pinned to the Epoch snapshot and ResNet-50 replaced by
# ResNet-152, the Epoch-listed ResNet. Epoch's AlphaGo Zero estimate counts
# network training only, not self-play data generation.
# ---------------------------------------------------------------------------
from ..core.provenance import Provenance, ProvenanceKind

_COMPUTE_TREND_PROVENANCE = Provenance(
    kind=ProvenanceKind.LITERATURE,
    ref=(
        "Epoch AI, Notable AI Models database (training compute, FLOP), "
        "accessed 2026-10-12; method per Sevilla et al. (2022), Compute Trends "
        "Across Three Eras of Machine Learning"
    ),
    url="https://epoch.ai/data/notable-ai-models",
    verified="2026-10-12",
)

# (model, calendar year of publication, training FLOP, era)
_COMPUTE_TREND_ROWS = (
    ("AlexNet", 2012, 4.7e17, "Deep Learning"),
    ("VGG-16", 2014, 1.23e19, "Deep Learning"),
    ("GoogLeNet", 2014, 1.51e18, "Deep Learning"),
    ("ResNet-152", 2015, 1.04e19, "Deep Learning"),
    ("Transformer", 2017, 7.42e18, "Deep Learning"),
    ("AlphaGoZero", 2017, 6.49e20, "Deep Learning"),
    ("BERT-Large", 2018, 2.85e20, "Deep Learning"),
    ("GPT-2-XL", 2019, 1.92e21, "Deep Learning"),
    ("T5-11 billion", 2019, 3.3e22, "Large Scale"),
    ("GPT-3", 2020, 3.14e23, "Large Scale"),
    ("Gopher", 2021, 6.31e23, "Large Scale"),
    ("PaLM", 2022, 2.53e24, "Large Scale"),
    ("GPT-4-class", 2023, 2.1e25, "Large Scale"),
    ("Gemini-Ultra", 2023, 5.0e25, "Large Scale"),
    ("Llama-3-70 billion", 2024, 7.86e24, "Large Scale"),
    ("Llama 3.1 405B", 2024, 3.8e25, "Large Scale"),
    ("Grok-3", 2025, 3.5e26, "Large Scale"),
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
# Algorithmic-efficiency series for @fig-algo-efficiency (added 2026-10-12).
# Training compute each ImageNet architecture needed to reach AlexNet-level
# accuracy (79.1% top-5), measured by Hernandez and Brown with fixed hardware
# held out. The efficiency factor of a model is AlexNet's compute divided by
# its own; the book computes that ratio, so only the measured compute lives here.
# Years are first-publication dates (fractional, for plotting order).
# ---------------------------------------------------------------------------
_ALGO_EFFICIENCY_PROVENANCE = Provenance(
    kind=ProvenanceKind.LITERATURE,
    ref=(
        "Hernandez and Brown (2020), Measuring the Algorithmic Efficiency of "
        "Neural Networks, Appendix B, Table 3 (teraflop/s-days to AlexNet-level accuracy)"
    ),
    url="https://arxiv.org/abs/2005.04305",
    verified="2026-10-12",
)

# (model, publication year, teraflop/s-days to reach AlexNet-level accuracy)
_ALGO_EFFICIENCY_ROWS = (
    ("AlexNet", 2012.5, 266.1),
    ("VGG-11", 2014.7, 367.7),
    ("GoogLeNet", 2014.75, 61.4),
    ("ResNet-18", 2015.9, 97.9),
    ("DenseNet-121", 2016.7, 82.9),
    ("MobileNet-v1", 2017.3, 24.0),
    ("ShuffleNet-v1", 2017.5, 12.9),
    ("MobileNet-v2", 2018.1, 20.2),
    ("ShuffleNet-v2", 2018.5, 10.8),
    ("EfficientNet-B0", 2019.4, 6.0),
)


class AlgorithmicEfficiency:
    """Training compute to AlexNet-level ImageNet accuracy, by architecture.

    Source: see ``provenance``. Consumed by @fig-algo-efficiency and the
    efficiency-versus-scale prose in the introduction. Each entry is a dict
    ``{"model", "year", "tfs_days"}``; ``baseline_model`` names the reference.
    """

    provenance = _ALGO_EFFICIENCY_PROVENANCE
    baseline_model = "AlexNet"
    entries = tuple(
        {"model": m, "year": y, "tfs_days": c} for (m, y, c) in _ALGO_EFFICIENCY_ROWS
    )

    @classmethod
    def factor(cls, model):
        """AlexNet compute divided by ``model``'s compute (higher is more efficient)."""
        by_model = {e["model"]: e["tfs_days"] for e in cls.entries}
        return by_model[cls.baseline_model] / by_model[model]


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
    ref="Kaplan et al. (2020), Scaling Laws for Neural Language Models (Figure 1; Tables 2, 3, 5, 6)",
    url="https://arxiv.org/abs/2001.08361",
    verified="2026-10-15",
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

    # -- Added 2026-10-15: the joint fits and the compute-efficient frontier, so
    # learning-curve and allocation figures regenerate from the published laws.
    # Values verified against arXiv:2001.08361v1. ``entries`` above keep the
    # constants printed on Figure 1 (C_min scale 2.3e8 PF-days); Table 5 states
    # the same law's scale as 3.1e8 PF-days (``single_variable_fits`` below).

    # Table 5 (Appendix A): single-variable power laws L = (X_c / X)^alpha.
    # Units: params (non-embedding), tokens, PF-days, tokens, steps.
    single_variable_fits = {
        "params": {"alpha": 0.076, "scale": 8.8e13},
        "data": {"alpha": 0.095, "scale": 5.4e13},
        "compute_naive": {"alpha": 0.057, "scale": 1.6e7},
        "compute_min": {"alpha": 0.050, "scale": 3.1e8},
        "critical_batch": {"alpha": 0.21, "scale": 2.1e8},
        "steps": {"alpha": 0.76, "scale": 2.1e3},
    }

    # Table 2 (Section 4.2): L(N, D) = [(N_c/N)^(alpha_N/alpha_D) + D_c/D]^alpha_D,
    # Eq. 1.5, fit with early stopping. N in non-embedding params, D in tokens.
    loss_vs_params_and_data = {"alpha_n": 0.076, "alpha_d": 0.103, "n_c": 6.4e13, "d_c": 1.8e13}

    # Table 3 (Section 5.2): L(N, S_min) = (N_c/N)^alpha_N + (S_c/S_min)^alpha_S,
    # Eq. 1.6 / 5.6, infinite-data learning curves. S_min = S / (1 + B_crit(L)/B)
    # (Eq. 5.4), with B_crit(L) = B_star / L^(1/alpha_B) (Eq. 5.3, Table 5 values).
    loss_vs_params_and_steps = {"alpha_n": 0.077, "alpha_s": 0.76, "n_c": 6.5e13, "s_c": 2.1e3}

    # Table 6 (Appendix A): compute-efficient frontier X = X_e * C_min^p,
    # C_min in PF-days. Units: params, tokens, steps, tokens.
    compute_efficient_frontier = {
        "params": {"p": 0.73, "scale": 1.3e9},
        "batch_tokens": {"p": 0.24, "scale": 2.0e6},
        "steps_min": {"p": 0.03, "scale": 5.4e3},
        "tokens": {"p": 0.27, "scale": 2e10},
    }

    # Section 2.2: default run, 2.5e5 steps of 512 sequences x 1024 tokens
    # (B = 2^19 tokens). Compute is C = 6NBS (Eq. 5.5; the 6 is
    # Literature.Chinchilla.ComputeConstant); a PF-day is 1e15 FLOP/s for one
    # day (Section 1.2), so build it with pint units rather than a constant here.
    default_steps = 2.5e5
    default_batch_sequences = 512
    default_context_tokens = 1024


from .series import Series  # noqa: E402  (dataset series behind trend figures)
from . import series as _series


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
    TrainingRuns = TrainingRuns
    FleetOperations = FleetOperations
    DatacenterElectricity = DatacenterElectricity
    AcceleratorLifecycle = AcceleratorLifecycle
    StorageSystems = StorageSystems
    IncidentReports = IncidentReports
    ServingIncidents = ServingIncidents
    SelfCorrection = SelfCorrection
    LlmInferenceTraceCode2023 = _series.LlmInferenceTraceCode2023
    PredictionQuality = PredictionQuality  # added 2026-10-20
    GamePlaying = GamePlaying  # added 2026-10-20
    DeviceMeasurements = DeviceMeasurements  # added 2026-10-20
    SpaceflightAnomalies = SpaceflightAnomalies  # added 2026-10-20
    ScreeningStudies = ScreeningStudies  # added 2026-10-20
    TestTimeCompute = TestTimeCompute  # added 2026-10-20
    ContextUse = ContextUse  # added 2026-10-20
    AgentRetrieval = AgentRetrieval  # added 2026-10-20
    SummarizationFaithfulness = SummarizationFaithfulness  # added 2026-10-20
    PrefixCaching = PrefixCaching  # added 2026-10-20
    VlaSystems = VlaSystems  # added 2026-10-20
    ComputeTrend = ComputeTrend
    AlgorithmicEfficiency = AlgorithmicEfficiency
    EdgeInferenceBenchmarks = EdgeInferenceBenchmarks
    KaplanScalingLaws = KaplanScalingLaws
    Series = Series
    AgentTimeHorizon = _series.AgentTimeHorizon
    LowestPriceAtCapability = _series.LowestPriceAtCapability
    TerminalBenchHarness = _series.TerminalBenchHarness
    LongContextComprehension = _series.LongContextComprehension
    RulerEffectiveLength = _series.RulerEffectiveLength  # added 2026-10-20
    AnswerPositionAccuracy = _series.AnswerPositionAccuracy  # added 2026-10-20
    SweBenchBm25Context = _series.SweBenchBm25Context  # added 2026-10-20
    PoisonedRagBlackBox = _series.PoisonedRagBlackBox  # added 2026-10-20
    SweBenchVerifiedRuns = _series.SweBenchVerifiedRuns
    CybenchSolveRate = _series.CybenchSolveRate
    OSWorldStepBudget = _series.OSWorldStepBudget
    FrontierSweTrials = _series.FrontierSweTrials
    SweBenchTrajectories = _series.SweBenchTrajectories
    SweBenchMiniSonnet4Run = _series.SweBenchMiniSonnet4Run  # added 2026-10-20
    RepeatedSamplingCoverage = _series.RepeatedSamplingCoverage
    RepeatedSamplingMajority = _series.RepeatedSamplingMajority
    InjectionDefenseResults = _series.InjectionDefenseResults
    ModelsByDomain = _series.ModelsByDomain
    OpenXEmbodimentComposition = _series.OpenXEmbodimentComposition
    LeRobotHubMonthly = _series.LeRobotHubMonthly
    LeRobotEpisodeHistogram = _series.LeRobotEpisodeHistogram
    EdgeInferenceOrin = _series.EdgeInferenceOrin
    KittiOdometry = _series.KittiOdometry
    CaliforniaDisengagements = _series.CaliforniaDisengagements
    WaymoSafetyImpact = _series.WaymoSafetyImpact
