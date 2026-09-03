"""Shared provenance records (stable ids, single definition)."""

from __future__ import annotations

from .provenance import Provenance, ProvenanceKind


def _ds(
    id: str,
    ref: str,
    url: str,
    *,
    verified: str = "2026-03-06",
    notes: str | None = None,
) -> Provenance:
    """Creates a Provenance object for a hardware datasheet or technical specification."""
    return Provenance(
        id=id,
        kind=ProvenanceKind.DATASHEET,
        ref=ref,
        url=url,
        verified=verified,
        notes=notes,
    )


def _lit(
    id: str,
    ref: str,
    *,
    url: str | None = None,
    verified: str = "2026-03-06",
    notes: str | None = None,
) -> Provenance:
    """Creates a Provenance object for peer-reviewed literature or academic whitepapers."""
    return Provenance(
        id=id,
        kind=ProvenanceKind.LITERATURE,
        ref=ref,
        url=url,
        verified=verified,
        notes=notes,
    )


def _est(
    id: str,
    ref: str,
    notes: str,
    *,
    url: str | None = None,
    verified: str = "2026-03-06",
) -> Provenance:
    """Creates a Provenance object for expert estimates or rules of thumb."""
    return Provenance(
        id=id,
        kind=ProvenanceKind.ESTIMATE,
        ref=ref,
        url=url,
        verified=verified,
        notes=notes,
    )


def _conv(id: str, ref: str, *, notes: str | None = None) -> Provenance:
    """Creates a Provenance object for widely accepted industry conventions."""
    return Provenance(
        id=id,
        kind=ProvenanceKind.CONVENTION,
        ref=ref,
        verified="2026-03-06",
        notes=notes,
    )


# --- Grid and reference anchors ---
IEA_WEO_2023 = Provenance(
    id="prov:iea-weo-2023-carbon",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="IEA World Energy Outlook 2023 (rounded gCO2/kWh)",
    url="https://www.iea.org/reports/world-energy-outlook-2023",
    verified="2026-03-06",
)

UPTIME_PUE_2022 = Provenance(
    id="prov:uptime-pue-survey-2022",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="Uptime Institute Global Data Center Survey 2022",
    url="https://uptimeinstitute.com/resources/research-and-reports/uptime-institute-global-data-center-survey-2022",
    verified="2026-03-06",
)

CLUSTER_TIER_CONVENTIONS = _conv(
    "prov:cluster-tier-convention",
    "MLSysIM reference cluster tiers (256 / 2k / 8k / 100k GPUs)",
)

KEMPNER_AI_CLUSTER_H100 = _ds(
    "prov:kempner-ai-cluster-h100",
    "Kempner Institute Computing Handbook, Overview of Cluster - H100 partition specs",
    "https://handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/kempner_cluster/overview_of_kempner_cluster.html",
    verified="2026-06-01",
    notes="H100 partition: 384 H100 80GB GPUs, 24 servers per rack, 4 GPUs per server, four H100 racks.",
)

# --- Real-world case-study / workload scale anchors (Scenarios registry) ---
REFERENCE_WORKLOAD_SCALE = Provenance(
    id="prov:reference-workload-scale",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative real-world scale anchors (Gmail volume, Google searches, Waymo sensor rate) for order-of-magnitude intuition",
    verified="2026-03-06",
)
TINYML_ANOMALY_CASE = Provenance(
    id="prov:tinyml-anomaly-case",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="TinyML anomaly-detection case study (latency / AUC / energy) used as a benchmarking example",
    verified="2026-03-06",
)
CLINICAL_IMAGING_WORKFLOW_ANCHORS = Provenance(
    id="prov:clinical-imaging-workflow-anchors",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Clinical imaging workflow anchors for rural-clinic bandwidth and edge-deployment examples",
    verified="2026-06-01",
)
OURA_SLEEP_STAGE_STUDY = _lit(
    "prov:oura-sleep-stage-study",
    "Altini and Kinnunen (2021), \"The Promise of Sleep: A Multi-Sensor Approach for Accurate Sleep Stage Detection Using the Oura Ring\"",
    url="https://doi.org/10.3390/s21134302",
    verified="2026-06-03",
)
ENERGY_SCALE_ANCHORS = Provenance(
    id="prov:energy-scale-anchors",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Everyday energy-scale comparison anchors (smartphone charge ~40 kJ, boiling a cup of water ~100 kJ, US household electricity ~10.7 MWh/year) for order-of-magnitude intuition about ML energy",
    verified="2026-03-06",
)
MOBILE_DEVICE_ANCHORS = Provenance(
    id="prov:mobile-device-anchors",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Mobile/edge device reference figures (flagship phone battery ~15 Wh / 3000 mAh @ 3.7 V, mobile NPU power 3-4 W, object-detector ~2 W) for on-device ML intuition",
    verified="2026-03-06",
)
EDGE_DEVICE_SPECTRUM_ANCHORS = _est(
    "prov:edge-device-spectrum-anchors",
    "Representative edge-device spectrum anchors for on-device learning heterogeneity examples",
    notes="Pedagogical class endpoints spanning microcontroller/tiny-sensor and flagship-mobile envelopes; not a single SKU specification.",
    verified="2026-06-04",
)
EDGE_ADAPTATION_TIER_PROFILE = _est(
    "prov:edge-adaptation-tier-profile",
    "Reference edge adaptation tier profile",
    notes=(
        "Reusable device-tier, wearable, and fleet-mix assumptions for on-device "
        "learning examples. These are pedagogical deployment-profile anchors, "
        "not a claim about one production fleet."
    ),
    verified="2026-06-04",
)

# --- Hardware technology-class facts (Hardware.Tech) ---
HOROWITZ_ENERGY = _lit(
    "prov:horowitz-2014",
    "Horowitz (2014), \"Computing's Energy Problem (and what we can do about it)\", ISSCC — 45 nm per-operation/per-byte energies",
    url="https://ieeexplore.ieee.org/document/6757323",
)
ON_CHIP_MEMORY_ACCESS_ENERGY_ANCHORS = Provenance(
    id="prov:on-chip-memory-access-energy-anchors",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="MLSysIM illustrative register, L1, and L2 access-energy anchors",
    verified="2026-08-11",
    notes="Pedagogical hierarchy anchors, not values reported by Horowitz (2014).",
)
MEMORY_LATENCY_HIERARCHY = _conv(
    "prov:memory-latency-hierarchy",
    "MLSysIM memory/interconnect access-latency hierarchy (order-of-magnitude class figures)",
)
STORAGE_TIER_CONVENTIONS = _conv(
    "prov:storage-tier-conventions",
    "Generic storage/memory bandwidth tiers (NVMe Gen3/4/5, DDR, host DRAM) from vendor datasheet ranges",
)
CXL_PCIE_GEN5_BW = Provenance(
    id="prov:cxl-pcie-gen5-x16-bw-derived",
    kind=ProvenanceKind.DERIVED,
    ref="Reference CXL memory tier over PCIe Gen5 x16 -> 64 GB/s",
    notes="Teaching anchor for a CXL 3.x memory-expansion link using the same byte-rate convention as the H100 PCIe Gen5 x16 profile.",
    verified="2026-06-03",
)
STORAGE_ACCESS_PATH_REFERENCE = _conv(
    "prov:storage-access-path-reference",
    "Reference GPU-storage access-path latencies for traditional CPU-mediated I/O and GPU Direct Storage bypass paths",
)
FRAMEWORK_RUNTIME_OVERHEAD_REFERENCE = _conv(
    "prov:framework-runtime-overhead-reference",
    "Reference framework/runtime overhead latencies for dispatch, kernel launch, and tiny memory-access operations",
)

RELIABILITY_MTTF_LITERATURE = _lit(
    "prov:reliability-mttf-literature",
    "Kokolis et al. (2025, HPCA); Zu et al. (2024, NSDI); Barroso et al. (2018) — order-of-magnitude steady-state MTTF",
    url="https://doi.org/10.1109/hpca61900.2025.00096",
)

ILLUSTRATIVE_IOWA_CARBON = Provenance(
    id="prov:illustrative-iowa-carbon",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative high-carbon US grid contrast (not IEA country average)",
    verified="2026-03-06",
)

HYDRO_QUEBEC_GRID = _ds(
    "prov:hydro-quebec-grid",
    "Hydro-Québec electricity mix and carbon intensity",
    "https://www.hydroquebec.com/about/our-energy.html",
)

# --- Cloud accelerators ---
NVIDIA_H200 = _ds(
    "prov:nvidia-h200-datasheet",
    "NVIDIA H200 Tensor Core GPU product documentation",
    "https://www.nvidia.com/en-us/data-center/h200/",
)

NVIDIA_GB200_NVL72 = _ds(
    "prov:nvidia-gb200-nvl72-datasheet",
    "NVIDIA GB200 NVL72 rack-scale system documentation",
    "https://www.nvidia.com/en-us/data-center/gb200-nvl72/",
)

AMD_MI250X = _ds(
    "prov:amd-mi250x-datasheet",
    "AMD Instinct MI250X product documentation",
    "https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html",
)

INTEL_GAUDI2 = _ds(
    "prov:intel-gaudi2-datasheet",
    "Intel Gaudi 2 AI accelerator product documentation",
    "https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi2.html",
)

INTEL_GAUDI3 = _ds(
    "prov:intel-gaudi3-datasheet",
    "Intel Gaudi 3 AI accelerator product documentation",
    "https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi3.html",
)

AWS_TRAINIUM2 = _ds(
    "prov:aws-trainium2-datasheet",
    "AWS Trainium2 accelerator (EC2 Trn2) product documentation",
    "https://aws.amazon.com/ai/machine-learning/trainium/",
)

GOOGLE_TPU_V1 = _lit(
    "prov:google-tpu-v1",
    "Jouppi et al. (2017), In-Datacenter Performance Analysis of a Tensor Processing Unit",
    url="https://arxiv.org/abs/1704.04760",
)

GOOGLE_TPU_V2_V3 = _lit(
    "prov:google-tpu-v2-v3",
    "Jouppi et al. (2020), A Domain-Specific Supercomputer for Training Deep Neural Networks",
    url="https://doi.org/10.1145/3360307",
)

GOOGLE_TPU_V4 = _lit(
    "prov:google-tpu-v4",
    "Jouppi et al. (2023), TPU v4: An Optically Reconfigurable Supercomputer for Training Deep Neural Networks",
    url="https://arxiv.org/abs/2304.01433",
)

GOOGLE_TPU_V5P = _ds(
    "prov:google-tpu-v5p",
    "Google Cloud TPU v5p documentation",
    "https://cloud.google.com/tpu/docs/v5p",
)
GOOGLE_TPU_V5P_WITH_VMEM = Provenance(
    id="prov:google-tpu-v5p-with-vmem",
    kind=ProvenanceKind.DERIVED,
    ref="Google Cloud TPU v5p specifications and JAX TPU hardware reference",
    url="https://docs.jax.dev/en/latest/pallas/tpu/hardware.html",
    verified="2026-08-11",
    notes="The 128 MiB VMEM capacity is derived from two TensorCores per chip and 64 MiB of VMEM per TensorCore; other fields follow the Google Cloud v5p specification.",
)

GOOGLE_TPU_V6 = _ds(
    "prov:google-tpu-v6-trillium",
    "Google Cloud TPU v6e (Trillium) documentation",
    "https://cloud.google.com/tpu/docs/v6e",
)

CEREBRAS_CS3 = _ds(
    "prov:cerebras-cs3-datasheet",
    "Cerebras CS-3 system product documentation",
    "https://www.cerebras.net/product-system/",
)

INTEL_SGX = _ds(
    "prov:intel-sgx-datasheet",
    "Intel Software Guard Extensions (SGX) developer guide",
    "https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html",
)

REFERENCE_DESKTOP_CPU = _conv(
    "prov:reference-desktop-cpu",
    "Reference 1 TFLOP/s FP32 desktop CPU for pedagogy (order-of-magnitude)",
)

# --- Workstation / mobile / edge / tiny ---
NVIDIA_DGX_SPARK = _ds(
    "prov:nvidia-dgx-spark-gb10",
    "NVIDIA DGX Spark (GB10 Grace Blackwell) product page",
    "https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
)

APPLE_M3_MAX = _est(
    "prov:apple-m3-max-estimate",
    "Apple M3 Max technical specifications (GPU core count × peak FLOP/core, rounded)",
    notes="Peak TFLOP/s is an MLSysIM rounded estimate of Apple-published core counts, not a sustained ML benchmark.",
    url="https://www.apple.com/macbook-pro/specs/",
)

MOBILE_SOC_ESTIMATE = _est(
    "prov:mobile-npu-peak-estimate",
    "Smartphone SoC NPU peak TOPS from vendor product briefs (marketing peak)",
    notes="Used for order-of-magnitude edge inference examples; not MLPerf-submitted sustained throughput.",
)

NVIDIA_JETSON_AGX_ORIN = _ds(
    "prov:nvidia-jetson-agx-orin",
    "NVIDIA Jetson AGX Orin technical brief",
    "https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/",
)

NVIDIA_JETSON_ORIN_NX = _ds(
    "prov:nvidia-jetson-orin-nx",
    "NVIDIA Jetson Orin NX series documentation",
    "https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/",
)

GOOGLE_CORAL = _ds(
    "prov:google-coral-edge-tpu",
    "Google Coral Edge TPU product documentation",
    "https://coral.ai/products/",
)

INTEL_NUC_MOVIDIUS = _est(
    "prov:intel-nuc-movidius-estimate",
    "Intel NUC + Movidius VPU reference kit (peak TOPS from Intel Neural Compute Stick spec)",
    notes="Composite edge reference node for tutorials; not a single-SKU datasheet.",
    url="https://www.intel.com/content/www/us/en/products/details/processors/neural-processing-unit.html",
)

REFERENCE_EDGE_SERVER = _conv(
    "prov:reference-edge-server",
    "Reference edge server (1 TFLOP/s, 128 GB) for pedagogy",
)

ESP32_S3 = _ds(
    "prov:esp32-s3-datasheet",
    "Espressif ESP32-S3 technical reference manual",
    "https://www.espressif.com/en/products/socs/esp32-s3",
)

NORDIC_NRF52840 = _ds(
    "prov:nordic-nrf52840-datasheet",
    "Nordic nRF52840 product specification (MLPerf Tiny reference MCU)",
    "https://www.nordicsemi.com/Products/Development-hardware/Other/nRF52840",
)

HIMAX_WE1 = _ds(
    "prov:himax-we1-plus",
    "Himax WE-I Plus Edge AI platform documentation",
    "https://www.himax.com.tw/products/edge-ai-platform/we-plus/",
)

# --- Models ---
RADFOR_GPT2 = _lit(
    "prov:radford-gpt2-2019",
    "Radford et al. (2019), Language Models are Unsupervised Multitask Learners",
    url="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
)

DEVLIN_BERT = _lit(
    "prov:devlin-bert-2019",
    "Devlin et al. (2019), BERT: Pre-training of Deep Bidirectional Transformers",
    url="https://arxiv.org/abs/1810.04805",
)

META_LLAMA = _ds(
    "prov:meta-llama-model-card",
    "Meta Llama model card / Hugging Face model documentation",
    "https://github.com/meta-llama/llama-models",
)

HE_RESNET = _lit(
    "prov:he-resnet-2016",
    "He et al. (2016), Deep Residual Learning for Image Recognition",
    url="https://arxiv.org/abs/1512.03385",
)

SANDLER_MOBILENETV2 = _lit(
    "prov:sandler-mobilenetv2-2018",
    "Sandler et al. (2018), MobileNetV2: Inverted Residuals and Linear Bottlenecks",
    url="https://arxiv.org/abs/1801.04381",
)
MOBILENETV2_WITH_ENERGY_ANCHOR = Provenance(
    id="prov:mobilenetv2-with-energy-anchor",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Sandler et al. (2018) MobileNetV2 architecture with an illustrative book energy anchor",
    url="https://arxiv.org/abs/1801.04381",
    verified="2026-08-11",
    notes="Alpha=1.0, 1000-class ImageNet classifier. The paper reports about 300M multiply-adds, represented here as 600 MFLOP under the book's convention that one multiply-accumulate is two FLOPs. The 0.1 mJ inference-energy value is an illustrative book anchor, not a measurement reported by Sandler et al. (2018).",
)

YOLOV8 = _ds(
    "prov:ultralytics-yolov8",
    "Ultralytics YOLOv8 documentation (nano variant parameter count)",
    "https://docs.ultralytics.com/models/yolov8/",
)

Krizhevsky_ALEXNET = _lit(
    "prov:krizhevsky-alexnet-2012",
    "Krizhevsky et al. (2012), ImageNet Classification with Deep Convolutional Neural Networks",
    url="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks",
)

MLPERF_TINY_KWS = _lit(
    "prov:mlperf-tiny-kws",
    "MLPerf Tiny keyword spotting reference (DS-CNN profile)",
    url="https://github.com/mlcommons/tiny",
    notes="Parameter and FLOP counts aligned to TinyMLPerf KWS reference model scale.",
)

MLPERF_TRAINING_V30_RESNET50_A100 = Provenance(
    id="prov:mlperf-training-v30-resnet50-a100",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="MLPerf Training v3.0 ResNet-50 results, A100 class systems",
    url="https://mlcommons.org/benchmarks/training/",
    verified="2026-05-31",
    notes="Used as a broad single-accelerator throughput sanity anchor for ResNet-50/A100 teaching examples.",
)

MLPERF_TRAINING_V31_RESNET50_H100 = Provenance(
    id="prov:mlperf-training-v31-resnet50-h100",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="MLPerf Training v3.1 ResNet-50 results, H100 class systems",
    url="https://mlcommons.org/benchmarks/training/",
    verified="2026-05-31",
    notes="Used as a broad single-accelerator throughput sanity anchor for ResNet-50/H100 teaching examples.",
)

NVIDIA_NIM_LLAMA3_8B_H100 = Provenance(
    id="prov:nvidia-nim-llama3-8b-h100",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="NVIDIA NIM LLM benchmarking results for Llama 3.x 8B on H100",
    url="https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html",
    verified="2026-05-31",
    notes="Used as a broad H100 Llama-family ITL sanity range; exact serving latency depends on engine, prompt shape, batching, and precision.",
)

BROWN_GPT3_2020 = _lit(
    "prov:gpt3-brown-2020",
    "Brown et al. (2020), Language Models are Few-Shot Learners",
    url="https://arxiv.org/abs/2005.14165",
)

IMAGENET_DATASET = _lit(
    "prov:imagenet-ilsvrc-2015",
    "Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge",
    url="https://arxiv.org/abs/1409.0575",
)

MSWC_DATASET = _lit(
    "prov:mswc-2021",
    "Mazumder et al. (2021), Multilingual Spoken Words Corpus",
    url="https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/fe131d7f5a6b38b23cc967316c13dae2-Abstract-round2.html",
)

CIFAR10_DATASET = _lit(
    "prov:cifar10-2009",
    "Krizhevsky (2009), Learning Multiple Layers of Features from Tiny Images",
    url="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf",
)

MNIST_DATASET = _lit(
    "prov:mnist-1998",
    "LeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition",
    url="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf",
)

DEPLOYMENT_ENVELOPES = _est(
    "prov:mlsysim-deployment-envelopes",
    "MLSysIM deployment envelope defaults for cloud, edge, mobile, and TinyML systems",
    notes="Pedagogical order-of-magnitude envelopes used for first-pass deployment reasoning; not vendor SLA targets.",
)

MLOPS_DRIFT_THRESHOLDS = _conv(
    "prov:mlops-drift-threshold-conventions",
    "Common PSI and two-sample Kolmogorov-Smirnov drift-threshold conventions",
    notes="PSI bands and KS coefficient are common monitoring defaults; production thresholds should be calibrated per application.",
)

MEMORY_SOFT_ERROR_RATE = _est(
    "prov:memory-soft-error-rate-estimate",
    "Order-of-magnitude memory soft-error bit rate for reliability and monitoring examples",
    notes="Used as a teaching-scale sanity anchor rather than a device-specific FIT rate.",
)

HBM_SOFT_ERROR_FIT_PER_MBIT = _est(
    "prov:hbm-soft-error-fit-per-mbit",
    "Unprotected HBM soft-error rate, low end of the published 200-5000 FIT/Mbit DRAM range",
    notes=(
        "Teaching-scale figure (250 FIT/Mbit) at the low end of the 200-5000 FIT/Mbit "
        "DRAM soft-error range reported in the soft-error literature (Tezzaron, 'Soft "
        "Errors in Electronic Memory'; en.wikipedia.org/wiki/Soft_error). Motivates why "
        "unprotected HBM at fleet scale mandates ECC; not a device-specific datasheet value."
    ),
    url="https://tezzaron.com/media/soft_errors_1_1_secure.pdf",
)

FHE_OVERHEAD = _lit(
    "prov:fhe-overhead-slowdown",
    "Fully homomorphic encryption compute overhead, 2-6 orders of magnitude vs plaintext",
    url="https://www.math-lock.com/benchmarks.html",
    notes=(
        "General-purpose FHE libraries (HElib, PALISADE) run 1e4-1e6x slower than "
        "plaintext; the book uses 1e4x (10,000x) as a conservative low-end teaching figure."
    ),
)

TEE_HARDWARE_SPECS = _est(
    "prov:tee-hardware-specs",
    "Trusted-execution-environment overheads (Intel SGX, ARM TrustZone) from vendor documentation",
    notes=(
        "SGX enclave page cache ~128 MB with ~100x paging penalty on overflow and 15-30 us "
        "enclave transitions; TrustZone world switch ~300-1000 cycles and 15-30% secure-mode "
        "power; mTLS handshake 15-30 ms. Order-of-magnitude figures from Intel SGX / ARM "
        "TrustZone documentation and security-engineering literature."
    ),
)

HSM_GPU_CRYPTO = _est(
    "prov:hsm-gpu-crypto-throughput",
    "HSM vs GPU RSA-2048 throughput and unit cost from security-engineering practice",
    notes=(
        "Enterprise HSMs ~10,000 RSA-2048 ops/s at $20k-$100k/unit; general-purpose GPUs "
        "~100,000 ops/s at ~$1k; the ~10x throughput gap is the tamper-resistance tax."
    ),
)

RESPONSIBLE_AI_OVERHEAD = _est(
    "prov:responsible-ai-overhead-benchmarks",
    "Responsible-AI technique overheads (accuracy, training, inference, memory) across published benchmarks",
    notes=(
        "Synthesis of reported overheads for DP-SGD, fairness-aware training, SHAP/LIME "
        "explainability, adversarial training, and federated learning. Ranges are "
        "order-of-magnitude empirical findings across the responsible-AI efficiency "
        "literature, not vendor specifications."
    ),
)

ORCHESTRATION_ASSUMPTIONS = _est(
    "prov:mlsysim-orchestration-assumptions",
    "MLSysIM cluster orchestration defaults for utilization, queue discipline, and job duration",
    notes="First-pass scheduler assumptions for analytical examples; production clusters should calibrate from trace data.",
)

FLEET_EVOLUTION_HEURISTIC = Provenance(
    id="prov:fleet-evolution-heuristic",
    kind=ProvenanceKind.HEURISTIC,
    ref="MLSysIM fleet-evolution heuristic for conclusion synthesis",
    notes=(
        "Pedagogical multiplicative gain profile used to reason about how "
        "hardware, algorithmic, and orchestration improvements combine. The "
        "numbers are scenario anchors, not forecasts."
    ),
    verified="2026-06-06",
)

CRITICAL_BATCH_SIZE_ESTIMATES = _lit(
    "prov:mccandlish-critical-batch-2018",
    "McCandlish et al. (2018), An Empirical Model of Large-Batch Training",
    url="https://arxiv.org/abs/1812.06162",
    notes="Order-of-magnitude critical batch size anchors; model-specific entries are rounded for analytical examples.",
)

WAKE_VISION = _ds(
    "prov:wake-vision-dataset",
    "Wake Vision / doorbell-classifier TinyML reference",
    "https://github.com/harvard-edge/Wake_Vision",
)

REFERENCE_ANOMALY_MLP = _conv(
    "prov:reference-anomaly-mlp",
    "TinyML anomaly-detector reference MLP (~270k parameters)",
)

NAUMOV_DLRM = _lit(
    "prov:naumov-dlrm-2019",
    "Naumov et al. (2019), Deep Learning Recommendation Model (DLRM)",
    url="https://arxiv.org/abs/1906.00091",
)

GU_GUARDRAILS_MAMBA = _lit(
    "prov:gu-mamba-2023",
    "Gu & Dao (2023), Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
    url="https://arxiv.org/abs/2312.00752",
)

ROMBACH_STABLE_DIFFUSION = _lit(
    "prov:rombach-stable-diffusion-2022",
    "Rombach et al. (2022), High-Resolution Image Synthesis with Latent Diffusion Models",
    url="https://arxiv.org/abs/2112.10752",
)

CHINCHILLA = _lit(
    "prov:hoffmann-chinchilla-2022",
    "Hoffmann et al. (2022), Training Compute-Optimal Large Language Models",
    url="https://arxiv.org/abs/2203.15556",
)

PALM_MFU = _lit(
    "prov:chowdhery-palm-2022",
    "Chowdhery et al. (2022), PaLM: Scaling Language Modeling with Pathways",
    url="https://arxiv.org/abs/2204.02311",
)

POPE_INFERENCE = _lit(
    "prov:pope-inference-2023",
    "Pope et al. (2023), Efficiently Scaling Transformer Inference",
    url="https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html",
)

MEGASCALE = _lit(
    "prov:jiang-megascale-2024",
    "Jiang et al. (2024), MegaScale: Scaling Large Language Model Training",
    url="https://arxiv.org/abs/2402.15627",
)

GENDER_SHADES = _lit(
    "prov:buolamwini-gendershades-2018",
    "Buolamwini & Gebru (2018), Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification",
    url="https://proceedings.mlr.press/v81/buolamwini18a.html",
)

CROWDFLOWER_2016 = _lit(
    "prov:crowdflower-data-science-report-2016",
    "CrowdFlower (2016), Data Science Report — 'What data scientists spend the most time doing'",
    url="https://visit.figure-eight.com/data-science-report.html",
)

GPT2_TRAINING_COST_EST = _est(
    "prov:gpt2-training-cost-estimate",
    "Community estimate of GPT-2 (1.5B) cloud training cost in 2019 (~$50K)",
    notes="Order-of-magnitude anchor for the training-vs-inference cost asymmetry; OpenAI did not disclose a figure. (Added 2026-06-07 when the cost moved out of a chapter-local literal.)",
)

GPT4_TRAINING_COST_EST = _est(
    "prov:gpt4-training-cost-estimate",
    "Industry-reported nine-figure estimate for GPT-4-class training (Altman: 'more than $100 million', Wired, 2023)",
    url="https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/",
    notes="Pedagogical anchor; OpenAI has not disclosed the exact training cost. (Added 2026-06-07 when the cost moved out of a chapter-local literal.)",
)

MEGATRON_OVERLAP = _lit(
    "prov:shoeybi-megatron-2019",
    "Shoeybi et al. (2019), Megatron-LM: Training Multi-Billion Parameter Language Models",
    url="https://arxiv.org/abs/1909.08053",
)

SCALING_EFFICIENCY_RULE_OF_THUMB = _conv(
    "prov:scaling-efficiency-rule-of-thumb",
    "Common industry rule-of-thumb (~90% parallel efficiency on well-tuned clusters)",
)

DGX_GPUS_PER_HOST = _conv(
    "prov:dgx-gpus-per-host",
    "NVIDIA DGX H100/H200 node envelope (8 GPUs per host)",
    notes="Used for cluster tier node counts in fleet appendices.",
)

GIBIANSKY_ALLREDUCE = _lit(
    "prov:gibiansky-allreduce-factor",
    "Gibiansky (2017), Ring AllReduce communication identity (2× factor)",
    url="https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/",
)

INFINIBAND_SPEC = _ds(
    "prov:infiniband-trade-association-spec",
    "InfiniBand Trade Association link-speed specifications",
    "https://www.infinibandta.org/",
)

INFINIBAND_NDR_GBS = Provenance(
    id="prov:infiniband-ndr-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="InfiniBand NDR 400 Gbps per port → 50 GB/s",
    url="https://www.infinibandta.org/",
    notes="Byte rate = line rate ÷ 8.",
    verified="2026-03-06",
)

INFINIBAND_HDR_GBS = Provenance(
    id="prov:infiniband-hdr-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="InfiniBand HDR 200 Gbps per port → 25 GB/s",
    url="https://www.infinibandta.org/",
    notes="Byte rate = line rate ÷ 8.",
    verified="2026-03-06",
)

INFINIBAND_XDR_GBS = Provenance(
    id="prov:infiniband-xdr-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="InfiniBand XDR 800 Gbps per port → 100 GB/s",
    url="https://www.infinibandta.org/",
    notes="Byte rate = line rate ÷ 8 (2025 generation).",
    verified="2026-03-06",
)

ETHERNET_400G_GBS = Provenance(
    id="prov:ethernet-400g-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="400 GbE → 50 GB/s",
    notes="Byte rate = 400 Gb/s ÷ 8.",
    verified="2026-03-06",
)

ETHERNET_800G_GBS = Provenance(
    id="prov:ethernet-800g-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="800 GbE → 100 GB/s",
    notes="Byte rate = 800 Gb/s ÷ 8.",
    verified="2026-03-06",
)

ROCE_100G_GBS = Provenance(
    id="prov:roce-100g-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="100 GbE RoCE → 12.5 GB/s",
    notes="Byte rate = 100 Gb/s ÷ 8.",
    verified="2026-03-06",
)

FABRIC_LATENCY_ASSUMPTIONS = _conv(
    "prov:fabric-latency-assumptions",
    "MLSysIM α-model one-way latency anchors (InfiniBand NDR/HDR, RoCE, TCP)",
    notes="Order-of-magnitude μs values for napkin math; not vendor QoS guarantees.",
)

SWITCH_OPTICS_REFERENCE = _conv(
    "prov:switch-optics-reference",
    "Datacenter switch-ASIC capacity (51.2T/102.4T) and 400G optics power (pluggable/CPO) reference figures",
    notes="2025-26 datacenter-switching reference points for network-fabric sizing analyses.",
)

NVIDIA_QUANTUM2_QM97XX_SWITCH = _ds(
    "prov:nvidia-quantum2-qm97xx-switch",
    "NVIDIA QM97XX 1U NDR 400Gbps InfiniBand Switch Systems User Manual",
    "https://docs.nvidia.com/networking/display/qm97x0um/introduction",
    verified="2026-06-01",
    notes="QM9700/QM9701/QM9790-class switches: 64 NDR 400 Gb/s ports and 51.2 Tb/s aggregate bidirectional throughput.",
)

NDR_LEAF_SPINE_64_PORT_SPLIT = _conv(
    "prov:ndr-leaf-spine-64-port-split",
    "Reference non-oversubscribed two-tier leaf-spine split for a 64-port NDR switch",
    notes="Uses 32 endpoint downlinks and 32 spine uplinks per leaf switch.",
)

NETWORK_ENERGY_ANCHORS = _conv(
    "prov:network-energy-anchors",
    "Network data-transfer energy anchors (5G per-MB, generic per-KB)",
    notes="Order-of-magnitude transfer-energy figures for intuition; not measured device values.",
)

RECOVERY_TIME_ASSUMPTIONS = _conv(
    "prov:recovery-time-assumptions",
    "Fleet recovery design assumptions (heartbeat, reschedule, checkpoint BW)",
    notes="Engineering targets for reliability analyses; calibrate from cluster traces when available.",
)

OVERHEAD_BUDGETS = _conv(
    "prov:overhead-budgets",
    "Combined overhead budgets (pipeline, checkpoint, failure, maintenance)",
    notes="Fractions of wall time for 10k+ GPU training scenarios.",
)

SCALING_EFFICIENCY_TIERS = _conv(
    "prov:scaling-efficiency-tiers",
    "Illustrative scaling efficiency vs GPU count (32→1024 GPUs)",
    notes="8192-GPU tier uses MEGASCALE literature anchor separately.",
)

ENERGY_HIERARCHY_CONVENTIONS = _conv(
    "prov:energy-hierarchy-conventions",
    "Simplified energy hierarchy: architecture-class effective pJ/FLOP (CPU→ASIC) and per-byte data-movement cost (register→network)",
    notes="Order-of-magnitude teaching figures; effective system-level energy, consistent with the Horowitz (2014) energy trend.",
)

MEMORY_INTERFACE_BANDWIDTH_TIERS = _conv(
    "prov:memory-interface-bandwidth-tiers",
    "Representative memory-interface bandwidth tiers",
    notes=(
        "Technology-class bandwidth anchors for DDR4-3200, HBM2, HBM3, and "
        "GDDR6X used in memory-protection and hierarchy examples."
    ),
)

MEMORY_PROTECTION_OVERHEADS = _conv(
    "prov:memory-protection-overheads",
    "Memory protection overhead conventions",
    notes=(
        "SECDED-style ECC reserve is represented as 12.5 percent parity overhead; "
        "no-ECC profile is included for comparative GDDR-style examples."
    ),
)

WUE_ANCHORS = _conv(
    "prov:wue-anchors",
    "Water-usage effectiveness (WUE) tiers for sustainability examples",
)

RACK_POWER_TIERS = _conv(
    "prov:rack-power-tiers",
    "Rack power tiers (traditional vs AI cluster, air-cooling limit)",
)

DGX_H100_RACK_REFERENCE = _conv(
    "prov:dgx-h100-rack-reference",
    "Reference DGX H100 rack profile",
    notes=(
        "Four DGX H100 nodes per rack, yielding 32 H100 GPUs per rack for rack-level power "
        "and cooling models; representative non-accelerator rack support load is 11.1 kW "
        "(host CPUs/DRAM, NVSwitch, InfiniBand, power conversion, and cooling overhead)."
    ),
)

STORAGE_TRAINING_CORPUS_REFERENCE = _conv(
    "prov:storage-training-corpus-reference",
    "Reference 175B-model storage running example",
    notes=(
        "Chapter-level storage scenario anchor: 1.5T training tokens, 3 TB compressed "
        "source corpus, 4-byte token IDs, and 14 bytes/parameter resumable mixed-precision "
        "Adam checkpoint storage (FP16 weights, FP32 master weights, and two FP32 moments)."
    ),
)

MODEL_LOADING_SCENARIO_ASSUMPTIONS = _est(
    "prov:model-loading-scenario-assumptions",
    "Reference model-loading scenario assumptions",
    notes=(
        "Representative cold-start loading anchors for serialized model checkpoints; "
        "values are teaching-scale assumptions and should be calibrated from fleet "
        "measurements for production sizing."
    ),
)

LLM_SERVING_PRECISION_DIVIDEND_PROFILE = _est(
    "prov:llm-serving-precision-dividend-profile",
    "Reference LLM serving precision-dividend profile",
    notes=(
        "Reusable deployment-shape assumptions for the performance-engineering "
        "70B LLM KV-cache and batch-size example."
    ),
)

HETEROGENEOUS_ROUTING_SCENARIO = _est(
    "prov:heterogeneous-routing-scenario",
    "Reference heterogeneous GPU routing scenario",
    notes=(
        "Reusable H100/A100 server counts, per-server service rates, and traffic "
        "target for the inference weighted-routing example."
    ),
)

CIRCUIT_BREAKER_SERVING_PROFILE = _est(
    "prov:circuit-breaker-serving-profile",
    "Reference GPU inference circuit-breaker profile",
    notes=(
        "Reusable threshold and recovery-window assumptions for teaching "
        "GPU inference circuit-breaker behavior; production thresholds should "
        "be calibrated from service SLOs and fleet telemetry."
    ),
)

CHECKPOINT_ARCHETYPE_SCENARIO_ASSUMPTIONS = _est(
    "prov:checkpoint-archetype-scenario-assumptions",
    "Reference checkpoint-size archetype assumptions",
    notes=(
        "Representative checkpoint footprints and bytes-per-parameter policy used "
        "to compare fault-tolerance overhead across model families."
    ),
)

CLOUD_PRICING_2024 = Provenance(
    id="prov:cloud-pricing-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative US cloud list prices (2024–2025 order of magnitude)",
    notes="GPU-hour, egress, and electricity rate anchors; not a specific vendor quote.",
    verified="2026-03-06",
)

STORAGE_PRICING_2024 = Provenance(
    id="prov:storage-pricing-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative cloud/object-storage list prices (2024 order of magnitude)",
    notes="S3, Glacier, and NVMe tier rate anchors for data-engineering scenarios.",
    verified="2026-03-06",
)

LABELING_PRICING_2024 = Provenance(
    id="prov:labeling-pricing-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative data-labeling cost ranges (2024 estimates)",
    notes="Crowd, bounding-box, and medical labeling tiers for workflow examples.",
    verified="2026-03-06",
)

FLEET_ECONOMICS_2024 = Provenance(
    id="prov:fleet-economics-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative internal GPU-hour and chargeback rates (2024)",
    notes="On-demand, spot, and internal chargeback references for fleet orchestration examples.",
    verified="2026-03-06",
)

BARROSO_DATACENTER_ECONOMICS = _lit(
    "prov:barroso-datacenter-economics",
    "Barroso et al. (2018), The Datacenter as a Computer (3rd ed.)",
    url="https://doi.org/10.1007/978-3-031-01761-2",
)

CAPACITY_LEAD_TIMES = _est(
    "prov:capacity-lead-times",
    "Illustrative datacenter build-out lead times",
    notes="Order-of-magnitude planning anchors for compute-infrastructure analyses.",
)

CARBON_PER_GPU_HR = _est(
    "prov:carbon-per-gpu-hr",
    "Illustrative per-GPU-hour carbon proxy for responsible-AI examples",
    notes="0.16 kg/GPU-hr order-of-magnitude; not a grid-specific intensity calculation.",
)

LIT_TRANSATLANTIC_ROUND_TRIP_CO2 = _est(
    "prov:lit-transatlantic-round-trip-co2",
    "Aviation CO2e factors for long-haul economy passenger travel (DEFRA-class)",
    notes=(
        "Rounded to 1000 kg CO2e for the NY-London round-trip reference anchor; "
        "agency calculators typically report about 900-1100 kg CO2e per economy passenger."
    ),
    verified="2026-05-31",
)

MFU_INFERENCE_BATCHED_LIT = _lit(
    "prov:mfu-inference-batched",
    "Pope et al. (2023); batched inference MFU upper illustrative bound",
    url="https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html",
    notes="0.40 is an upper illustrative bound for large-batch inference, not batch-1.",
)
