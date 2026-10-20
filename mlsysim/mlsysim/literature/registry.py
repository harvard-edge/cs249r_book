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
IncidentReports = _load("incident_reports", "IncidentReports", "Values reported by incident investigations (NTSB Tempe, Cruise pullover, the SEC Knight Capital order, the CrowdStrike PIR).")
# Model calls and agent trajectories (added 2026-10-20)
ServingIncidents = _load("serving_incidents", "ServingIncidents", "Reported production serving-incident figures (Anthropic 2025 postmortem).")
MultiAgentFailures = _load("multi_agent_failures", "MultiAgentFailures", "Cited multi-agent failure-taxonomy results (MAST, Cemri et al. 2025).")  # added 2026-10-20
AgentEvaluation = _load("agent_evaluation", "AgentEvaluation", "Cited agent-evaluation results (tau-bench consistency bounds, METR reward-hacking counts, the HumanEval cost study's sample).")  # added 2026-10-20
# Coordinating several agents (added 2026-10-20)
MultiAgentSystems = _load("multi_agent_systems", "MultiAgentSystems", "Cited results of deployed multi-agent systems (Hadfield et al. 2025 research system).")
CorrelatedModelErrors = _load("correlated_model_errors", "CorrelatedModelErrors", "Cited results on correlated errors across language models (Kim et al. 2025, Helm).")
FanoutTailLatency = _load("fanout_tail_latency", "FanoutTailLatency", "Measured fan-out tail latency and hedged requests (Dean and Barroso 2013).")
SelfCorrection = _load("self_correction", "SelfCorrection", "Intrinsic self-correction results (Huang et al. 2024, GPT-4).")
# Added 2026-10-20: benchmark error rates and documented accuracy failures.
PredictionQuality = _load("prediction_quality", "PredictionQuality", "Cited prediction-quality results (ILSVRC-2012 errors, Google Flu Trends misses).")
GamePlaying = _load("game_playing", "GamePlaying", "Cited results of landmark game-playing systems (Deep Blue, AlphaGo Zero).")
# Deployment-tier field figures (added 2026-10-20)
DeviceMeasurements = _load("device_measurements", "DeviceMeasurements", "Cited measurements of deployed ML hardware (TPU v1 roofline, memory-wall growth, phone stress test, two-pass wake word).")
# Spaceflight anomaly inquiries (added 2026-10-20)
WakeWordBenchmarks = _load("wake_word_benchmarks", "WakeWordBenchmarks", "Published wake-word detector benchmark results (Picovoice wake-word benchmark).")  # added 2026-10-20
AnnotationTimes = _load("annotation_times", "AnnotationTimes", "Measured human annotation times (ImageNet boxes, Cityscapes masks).")  # added 2026-10-20
DataPipelineResults = _load("data_pipeline_results", "DataPipelineResults", "Measured input-pipeline results (data stalls, FFCV, a profiler's step breakdown).")  # added 2026-10-20
DataFileMeasurements = _load("data_file_measurements", "DataFileMeasurements", "Metadata of real public data files (a Parquet file, a WebDataset shard).")  # added 2026-10-20
SpeechCommandsReleases = _load("speech_commands_releases", "SpeechCommandsReleases", "Speech Commands v1 and v2 sizes and cross-release accuracy (Warden 2018).")  # added 2026-10-20
CompilerResults = _load("compiler_results", "CompilerResults", "Cited results of ML compilers and fused kernels (PyTorch 2, FlashAttention).")  # added 2026-10-20
# Fitting a training step on one accelerator (added 2026-10-20)
TrainingMemory = _load("training_memory", "TrainingMemory", "Cited results on fitting a training step in memory (loss scaling, recomputation, checkpointing, framework defaults).")
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
# Attacks on and defenses of learned systems (added 2026-10-20)
ExtractionAttacks = _load("extraction_attacks", "ExtractionAttacks", "Cited extraction results through a model's query interface (Carlini et al. 2024 production-API extraction; Carlini et al. 2021 GPT-2 memorization).")
DataPoisoning = _load("data_poisoning", "DataPoisoning", "Cited training-data poisoning results (Nightshade doses; web-scale poisoning cost).")
DifferentialPrivacy = _load("differential_privacy", "DifferentialPrivacy", "Cited differentially private training results (Abadi et al. 2016 accountant comparison; MNIST and CIFAR-10 utility).")
SecurityIncidents = _load("security_incidents", "SecurityIncidents", "Cited figures of documented security incidents (Stuxnet, Mirai).")
PromptCacheAudit = _load("prompt_cache_audit", "PromptCacheAudit", "Cited results of a timing audit of prompt caching in model APIs (Gu et al. 2025).")  # added 2026-10-20
MlOperations = _load("ml_operations", "MlOperations", "Measured ML-operations results (data validation anomalies, model-load latency, interleaving).")  # added 2026-10-20
ToolInterception = _load("tool_interception", "ToolInterception", "Cited tool-interception results for augmented LLM serving (InferCept 2024).")  # added 2026-10-20
SandboxIsolation = _load("sandbox_isolation", "SandboxIsolation", "Cited results on containing agent code: Cybench task count, gVisor system-call latency.")  # added 2026-10-20
DistributedTracing = _load("distributed_tracing", "DistributedTracing", "Cited results of production distributed tracing (Dapper sampling).")  # added 2026-10-20
# Published vision-language-action and dual-system robot policies (added 2026-10-20)
VlaSystems = _load("vla_systems", "VlaSystems", "Cited figures of published VLA and dual-system robot policies (OpenVLA, ACT, pi0, FAST, Helix, GR00T N1).")
# Robot demonstration data and event data recorders (added 2026-10-20)
RobotDataSystems = _load("robot_data_systems", "RobotDataSystems", "Cited figures of dated robot demonstration-data systems and interactive imitation results (DROID, Open X-Embodiment, AgiBot World, ThriftyDAgger).")
OnDeviceLearning = _load("on_device_learning", "OnDeviceLearning", "Cited results of on-device and federated learning systems (Gboard, Google FL system, TinyTL).")  # added 2026-10-20
RlTrainingDynamics = _load("rl_training_dynamics", "RlTrainingDynamics", "Training-curve readings from RL of language models on verifiable rewards (DeepSeek-R1-Zero length, DAPO entropy).")  # added 2026-10-20
EventDataRecorders = _load("event_data_recorders", "EventDataRecorders", "Pre-crash recording windows required of automotive event data recorders (49 CFR 563).")
# Published vision-language-action and dual-system robot policies (added 2026-10-20)
# Robot policy training: sim-to-real, parallel simulation, on-robot RL (added 2026-10-20)
RobotLearning = _load("robot_learning", "RobotLearning", "Cited figures of published robot-learning runs (Dactyl, legged_gym, HIL-SERL, QT-Opt, a grasping farm, Diffusion Policy, OpenVLA pretraining).")
RobotPerception = _load("robot_perception", "RobotPerception", "Cited figures of robot odometry, mapping, and tracking systems (UMBmark, nvblox Table II, SORT defaults).")  # added 2026-10-20
CompressionResults = _load("compression_results", "CompressionResults", "Cited model-compression results (Gale 2019, DistilBERT, sparse GPU kernels, 2:4 sparse tensor cores, Krishnamoorthi 2018, Deep Compression, Nagel 2019, LLM.int8(), GPTQ, AWQ, llama.cpp block formats).")  # added 2026-10-20
# Road-vehicle safety demonstrated from driven exposure (added 2026-10-20)
RobotGrounding = _load("robot_grounding", "RobotGrounding", "Cited results of grounding learned proposals on real robots (AutoRT task filter, OK-Robot failure analysis, KnowNo).")  # added 2026-10-20
HumanMotion = _load("human_motion", "HumanMotion", "Cited measurements of how fast people walk (Bohannon 1997).")  # added 2026-10-20
ContactForceMeasurements = _load("contact_force_measurements", "ContactForceMeasurements", "Measured transient contact forces of collaborative arms (Kirschner et al. 2022).")  # added 2026-10-20
RoadVehicleSafety = _load("road_vehicle_safety", "RoadVehicleSafety", "Cited miles-to-demonstrate requirements (RAND RR-1478) and a dated driverless-fleet exposure total (Kusano et al. 2025).")
ContaminatedSurfaceFriction = _load("contaminated_surface_friction", "ContaminatedSurfaceFriction", "Measured locked-wheel friction on contaminated surfaces (TRL PPR073, Lambourn and Viner 2006).")  # added 2026-10-20
RobotDeployments = _load("robot_deployments", "RobotDeployments", "Vendor-reported figures of commercial robot deployments (Agility Digit totes).")  # added 2026-10-20
# Network architectures (added 2026-10-20)
NetflixPrize = _load("netflix_prize", "NetflixPrize", "Cited figures of the Netflix Prize and what Netflix shipped from it (Johnston 2012; Amatriain and Basilico 2012).")
KerasApplications = _load("keras_applications", "KerasApplications", "Keras Applications model table: top-1 accuracy and time per inference step on a CPU and an A100.")
ContextWindows = _load("context_windows", "ContextWindows", "Announced context windows of language models outside the Models registry (GPT-4 Turbo, Claude 2.1, Gemini 1.5 Pro).")  # added 2026-10-20
RecommendationSystems = _load("recommendation_systems", "RecommendationSystems", "Cited figures of production recommendation systems (Gupta et al. 2020; Mudigere et al. 2022).")  # added 2026-10-20
ServingMemoryLayout = _load("serving_memory_layout", "ServingMemoryLayout", "Memory layout of a 13B model served on one A100 40 GB (Kwon et al. 2023, Figure 1).")  # added 2026-10-20
RobustnessStudies = _load("robustness_studies", "RobustnessStudies", "Cited results of robustness studies (Eykholt 2018 stickers, Goodfellow 2015 FGSM, Guo 2017 calibration, Zou 2023 GCG, Gu 2017 BadNets, Brown 2017 patch, Robey 2025 RoboPAIR).")  # added 2026-10-20
VerificationStudies = _load("verification_studies", "VerificationStudies", "Cited results of verification studies of driving stacks (DriveFI fault injection, Jha et al. 2019; cross-entropy rare-event simulation, O'Kelly et al. 2018).")  # added 2026-10-20
ServingResults = _load("serving_results", "ServingResults", "Cited model-serving results and documented defaults (ServerlessLLM loading, the AI Tax frame breakdown, Triton perf_analyzer and dynamic-batcher examples, the MLPerf Inference ResNet-50 Server bound; LLM serving results from Splitwise, DistServe, Orca, Sarathi-Serve, PagedAttention, speculative decoding, and vLLM logs).")  # model serving (added 2026-10-20)
InferenceDeterminism = _load("inference_determinism", "InferenceDeterminism", "Cited results on run-to-run nondeterminism of LLM inference (He and Thinking Machines Lab 2025).")  # added 2026-10-20
DataSelectionResults = _load("data_selection_results", "DataSelectionResults", "Cited data-selection results (model collapse, scored pruning, selection via proxy, echoing, deduplication, FineWeb, FixMatch, augmentation, an input-bound profile).")  # added 2026-10-20
TrajectoryGeneration = _load("trajectory_generation", "TrajectoryGeneration", "Cited figures of published online trajectory generators (Ruckig calculation times, Berscheid and Kroger 2021).")  # added 2026-10-20
SharedHardwareInterference = _load("shared_hardware_interference", "SharedHardwareInterference", "Measured victim slowdowns from co-runners on shared multicore hardware (Valsan et al. 2016, Bechtel and Yun 2019).")  # added 2026-10-20
PlatformConfigurations = _load("platform_configurations", "PlatformConfigurations", "Values read from published platform configurations (a Bao hypervisor demo, the Jetson AGX Orin device tree).")  # added 2026-10-20
RobotOffload = _load("robot_offload", "RobotOffload", "Measured results of offloading robot computation (FogROS2, Ichnowski et al. 2023).")  # added 2026-10-20
HumanTakeover = _load("human_takeover", "HumanTakeover", "Cited figures on human takeover: filed disengagement records, Sirius, Merat et al. 2014, UN R157 limits, ISO 10218-1 reduced speed.")  # added 2026-10-20


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
    InferenceDeterminism = InferenceDeterminism  # added 2026-10-20
    SelfCorrection = SelfCorrection
    LlmInferenceTraceCode2023 = _series.LlmInferenceTraceCode2023
    LlmInferenceTraceConversation2023 = _series.LlmInferenceTraceConversation2023  # added 2026-10-20
    LlmInferenceTraceConversation2024TenMinute = _series.LlmInferenceTraceConversation2024TenMinute  # added 2026-10-20
    PredictionQuality = PredictionQuality  # added 2026-10-20
    GamePlaying = GamePlaying  # added 2026-10-20
    DeviceMeasurements = DeviceMeasurements  # added 2026-10-20
    SpaceflightAnomalies = SpaceflightAnomalies  # added 2026-10-20
    CompilerResults = CompilerResults  # added 2026-10-20
    TrainingMemory = TrainingMemory  # added 2026-10-20
    ScreeningStudies = ScreeningStudies  # added 2026-10-20
    TestTimeCompute = TestTimeCompute  # added 2026-10-20
    ContextUse = ContextUse  # added 2026-10-20
    AgentRetrieval = AgentRetrieval  # added 2026-10-20
    SummarizationFaithfulness = SummarizationFaithfulness  # added 2026-10-20
    PrefixCaching = PrefixCaching  # added 2026-10-20
    PromptCacheAudit = PromptCacheAudit  # added 2026-10-20
    ToolInterception = ToolInterception  # added 2026-10-20
    SandboxIsolation = SandboxIsolation  # added 2026-10-20
    DistributedTracing = DistributedTracing  # added 2026-10-20
    MlOperations = MlOperations  # added 2026-10-20
    ExtractionAttacks = ExtractionAttacks  # added 2026-10-20
    DataPoisoning = DataPoisoning  # added 2026-10-20
    DifferentialPrivacy = DifferentialPrivacy  # added 2026-10-20
    SecurityIncidents = SecurityIncidents  # added 2026-10-20
    VlaSystems = VlaSystems  # added 2026-10-20
    WakeWordBenchmarks = WakeWordBenchmarks  # added 2026-10-20
    AnnotationTimes = AnnotationTimes  # added 2026-10-20
    DataPipelineResults = DataPipelineResults  # added 2026-10-20
    DataFileMeasurements = DataFileMeasurements  # added 2026-10-20
    SpeechCommandsReleases = SpeechCommandsReleases  # added 2026-10-20
    RobotDataSystems = RobotDataSystems  # added 2026-10-20
    EventDataRecorders = EventDataRecorders  # added 2026-10-20
    OnDeviceLearning = OnDeviceLearning  # added 2026-10-20
    RobotLearning = RobotLearning  # added 2026-10-20
    RobotPerception = RobotPerception  # added 2026-10-20
    CompressionResults = CompressionResults  # added 2026-10-20
    RoadVehicleSafety = RoadVehicleSafety  # added 2026-10-20
    ContaminatedSurfaceFriction = ContaminatedSurfaceFriction  # added 2026-10-20
    RobotDeployments = RobotDeployments  # added 2026-10-20
    RobotGrounding = RobotGrounding  # added 2026-10-20
    HumanMotion = HumanMotion  # added 2026-10-20
    ContactForceMeasurements = ContactForceMeasurements  # added 2026-10-20
    NetflixPrize = NetflixPrize  # network architectures (added 2026-10-20)
    KerasApplications = KerasApplications  # network architectures (added 2026-10-20)
    ContextWindows = ContextWindows  # network architectures (added 2026-10-20)
    RecommendationSystems = RecommendationSystems  # network architectures (added 2026-10-20)
    ServingMemoryLayout = ServingMemoryLayout  # network architectures (added 2026-10-20)
    FlashAttentionBenchmark = _series.FlashAttentionBenchmark  # network architectures (added 2026-10-20)
    NvidiaGemmTileQuantizationA100 = _series.NvidiaGemmTileQuantizationA100  # added 2026-10-20
    SsdGradientHistogram = _series.SsdGradientHistogram  # added 2026-10-20
    RobustnessStudies = RobustnessStudies  # added 2026-10-20
    VerificationStudies = VerificationStudies  # added 2026-10-20
    VnnComp2025ScoredBenchmarks = _series.VnnComp2025ScoredBenchmarks  # added 2026-10-20
    OKelly2018RareEventTable1 = _series.OKelly2018RareEventTable1  # added 2026-10-20
    Wang2025OpenVlaLiberoUntargeted = _series.Wang2025OpenVlaLiberoUntargeted  # added 2026-10-20
    ChallengerORingDistressRecord = _series.ChallengerORingDistressRecord  # added 2026-10-20
    DataSelectionResults = DataSelectionResults  # added 2026-10-20
    TrajectoryGeneration = TrajectoryGeneration  # added 2026-10-20
    SharedHardwareInterference = SharedHardwareInterference  # added 2026-10-20
    PlatformConfigurations = PlatformConfigurations  # added 2026-10-20
    ServingResults = ServingResults  # model serving (added 2026-10-20)
    RobotOffload = RobotOffload  # added 2026-10-20
    HumanTakeover = HumanTakeover  # added 2026-10-20
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
    PagedAttentionMemoryBreakdown = _series.PagedAttentionMemoryBreakdown  # added 2026-10-20
    ToolInterceptionTimes = _series.ToolInterceptionTimes  # added 2026-10-20
    MetastableFailures = _series.MetastableFailures  # added 2026-10-20
    DapperSamplingOverhead = _series.DapperSamplingOverhead  # added 2026-10-20
    MultiAgentFailures = MultiAgentFailures  # added 2026-10-20
    AttentionSinkPerplexity = _series.AttentionSinkPerplexity  # added 2026-10-20
    SweBenchVerifiedRuns = _series.SweBenchVerifiedRuns
    AgentEvaluation = AgentEvaluation  # added 2026-10-20
    MultiAgentSystems = MultiAgentSystems  # added 2026-10-20
    CorrelatedModelErrors = CorrelatedModelErrors  # added 2026-10-20
    FanoutTailLatency = FanoutTailLatency  # added 2026-10-20
    JudgePositionBias = _series.JudgePositionBias  # added 2026-10-20
    HumanEvalAgentCost = _series.HumanEvalAgentCost  # added 2026-10-20
    CybenchSolveRate = _series.CybenchSolveRate
    FirecrackerBootPreconfigured = _series.FirecrackerBootPreconfigured  # added 2026-10-20
    OSWorldStepBudget = _series.OSWorldStepBudget
    FrontierSweTrials = _series.FrontierSweTrials
    SweBenchTrajectories = _series.SweBenchTrajectories
    SweBenchMiniSonnet4Run = _series.SweBenchMiniSonnet4Run  # added 2026-10-20
    SweBenchVerifiedLeaderboard = _series.SweBenchVerifiedLeaderboard  # added 2026-10-20
    SweBenchMiniSonnet4Django12050Calls = _series.SweBenchMiniSonnet4Django12050Calls  # added 2026-10-20
    ImitationSuperTuxKart = _series.ImitationSuperTuxKart  # added 2026-10-20
    GrpoPassAtKSteps = _series.GrpoPassAtKSteps  # added 2026-10-20
    RlTrainingDynamics = RlTrainingDynamics  # added 2026-10-20
    RepeatedSamplingCoverage = _series.RepeatedSamplingCoverage
    RepeatedSamplingMajority = _series.RepeatedSamplingMajority
    InjectionDefenseResults = _series.InjectionDefenseResults
    ModelsByDomain = _series.ModelsByDomain
    OpenXEmbodimentComposition = _series.OpenXEmbodimentComposition
    LeRobotHubMonthly = _series.LeRobotHubMonthly
    LeRobotEpisodeHistogram = _series.LeRobotEpisodeHistogram
    DactylPhysicalTrials = _series.DactylPhysicalTrials  # added 2026-10-20
    EdgeInferenceOrin = _series.EdgeInferenceOrin
    KittiOdometry = _series.KittiOdometry
    CaliforniaDisengagements = _series.CaliforniaDisengagements
    TakeoverTimeMeta = _series.TakeoverTimeMeta  # added 2026-10-20
    PedestrianAp75BySkinType = _series.PedestrianAp75BySkinType  # added 2026-10-20
    PedestrianMissRateByAge = _series.PedestrianMissRateByAge  # added 2026-10-20
    WaymoSafetyImpact = _series.WaymoSafetyImpact
