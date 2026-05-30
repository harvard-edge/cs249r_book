"""Shared provenance records (stable ids, single definition)."""

from __future__ import annotations

from .provenance import Provenance, ProvenanceKind


def _ds(
    id: str,
    ref: str,
    url: str,
    *,
    verified: str = "2025-03-06",
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
    verified: str = "2025-03-06",
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
    verified: str = "2025-03-06",
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
        verified="2025-03-06",
        notes=notes,
    )


# --- Book / grid anchors (existing) ---
IEA_WEO_2023 = Provenance(
    id="prov:iea-weo-2023-carbon",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="IEA World Energy Outlook 2023 (rounded gCO2/kWh)",
    url="https://www.iea.org/reports/world-energy-outlook-2023",
    verified="2025-03-06",
)

UPTIME_PUE_2022 = Provenance(
    id="prov:uptime-pue-survey-2022",
    kind=ProvenanceKind.INDUSTRY_REPORT,
    ref="Uptime Institute Global Data Center Survey 2022",
    verified="2025-03-06",
)

BOOK_CLUSTER_TIERS = _conv(
    "prov:book-cluster-tier-convention",
    "MLSysBook editorial cluster tiers (256 / 2k / 8k / 100k GPUs)",
)

# --- Real-world case-study / workload scale anchors (Scenarios registry) ---
BOOK_WORKLOAD_SCALE = Provenance(
    id="prov:book-workload-scale",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative real-world scale anchors (Gmail volume, Google searches, Waymo sensor rate) for order-of-magnitude intuition",
    verified="2025-03-06",
)
BOOK_ANOMALY_CASE = Provenance(
    id="prov:book-anomaly-tinyml-case",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="TinyML anomaly-detection case study (latency / AUC / energy) used as a benchmarking example",
    verified="2025-03-06",
)
BOOK_ENERGY_ANCHORS = Provenance(
    id="prov:book-energy-anchors",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Everyday energy-scale comparison anchors (smartphone charge ~40 kJ, boiling 1 L water ~100 kJ) for order-of-magnitude intuition about ML energy",
    verified="2025-03-06",
)
BOOK_DEVICE_ANCHORS = Provenance(
    id="prov:book-device-anchors",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Mobile/edge device reference figures (flagship phone battery ~15 Wh / 3000 mAh @ 3.7 V, mobile NPU power 3-4 W, object-detector ~2 W) for on-device ML intuition",
    verified="2025-03-06",
)

# --- Hardware technology-class facts (Hardware.Tech) ---
HOROWITZ_ENERGY = _lit(
    "prov:horowitz-2014",
    "Horowitz (2014), \"Computing's Energy Problem (and what we can do about it)\", ISSCC — 45 nm per-operation/per-byte energies",
)
BOOK_LATENCY_HIERARCHY = _conv(
    "prov:book-latency-hierarchy",
    "MLSysBook memory/interconnect access-latency hierarchy (order-of-magnitude class figures)",
)
BOOK_STORAGE_TIERS = _conv(
    "prov:book-storage-tiers",
    "Generic storage/memory bandwidth tiers (NVMe Gen3/4/5, DDR, host DRAM) from vendor datasheet ranges",
)

RELIABILITY_MTTF_LITERATURE = _lit(
    "prov:reliability-mttf-literature",
    "Kokolis et al. (2025, HPCA); Zu et al. (2024, NSDI); Barroso et al. (2019) — order-of-magnitude steady-state MTTF",
    url="https://doi.org/10.1109/hpca61900.2025.00096",
)

BOOK_ILLUSTRATIVE_IOWA_CARBON = Provenance(
    id="prov:book-illustrative-iowa-carbon",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="MLSysBook illustrative high-carbon US grid contrast (not IEA country average)",
    verified="2025-03-06",
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
    url="https://arxiv.org/abs/2007.13828",
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

BOOK_REFERENCE_CPU = _conv(
    "prov:book-reference-desktop-cpu",
    "MLSysBook reference 1 TFLOP/s FP32 desktop CPU for pedagogy (order-of-magnitude)",
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
    notes="Peak TFLOP/s is an MLSysBook rounding of Apple-published core counts, not a sustained ML benchmark.",
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

BOOK_EDGE_SERVER = _conv(
    "prov:book-edge-server-reference",
    "MLSysBook reference edge server (1 TFLOP/s, 128 GB) for pedagogy",
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

WAKE_VISION = _ds(
    "prov:wake-vision-dataset",
    "Wake Vision / doorbell-classifier TinyML reference",
    "https://github.com/TI-malaria/wake-vision",
)

BOOK_ANOMALY_MLP = _conv(
    "prov:book-tiny-anomaly-mlp",
    "MLSysBook TinyML anomaly-detector reference MLP (~270k parameters)",
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
    url="https://arxiv.org/abs/2404.02054",
)

MEGATRON_OVERLAP = _lit(
    "prov:shoeybi-megatron-2019",
    "Shoeybi et al. (2019), Megatron-LM: Training Multi-Billion Parameter Language Models",
    url="https://arxiv.org/abs/1909.08053",
)

BOOK_SCALING_RULE_OF_THUMB = _conv(
    "prov:book-scaling-efficiency-rule-of-thumb",
    "Common industry rule-of-thumb (~90% parallel efficiency on well-tuned clusters)",
)

BOOK_DGX_GPUS_PER_HOST = _conv(
    "prov:book-dgx-gpus-per-host",
    "NVIDIA DGX H100/H200 node envelope (8 GPUs per host)",
    notes="Used for cluster tier node counts in fleet appendices.",
)

GIBIANSKY_ALLREDUCE = _lit(
    "prov:gibiansky-allreduce-factor",
    "Gibiansky (2017), Ring AllReduce communication identity (2× factor)",
    url="https://arxiv.org/abs/1707.05077",
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
    verified="2025-03-06",
)

INFINIBAND_HDR_GBS = Provenance(
    id="prov:infiniband-hdr-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="InfiniBand HDR 200 Gbps per port → 25 GB/s",
    url="https://www.infinibandta.org/",
    notes="Byte rate = line rate ÷ 8.",
    verified="2025-03-06",
)

INFINIBAND_XDR_GBS = Provenance(
    id="prov:infiniband-xdr-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="InfiniBand XDR 800 Gbps per port → 100 GB/s",
    url="https://www.infinibandta.org/",
    notes="Byte rate = line rate ÷ 8 (2025 generation).",
    verified="2025-03-06",
)

ETHERNET_400G_GBS = Provenance(
    id="prov:ethernet-400g-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="400 GbE → 50 GB/s",
    notes="Byte rate = 400 Gb/s ÷ 8.",
    verified="2025-03-06",
)

ETHERNET_800G_GBS = Provenance(
    id="prov:ethernet-800g-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="800 GbE → 100 GB/s",
    notes="Byte rate = 800 Gb/s ÷ 8.",
    verified="2025-03-06",
)

ROCE_100G_GBS = Provenance(
    id="prov:roce-100g-gbs-derived",
    kind=ProvenanceKind.DERIVED,
    ref="100 GbE RoCE → 12.5 GB/s",
    notes="Byte rate = 100 Gb/s ÷ 8.",
    verified="2025-03-06",
)

BOOK_FABRIC_LATENCY = _conv(
    "prov:book-fabric-latency-assumptions",
    "MLSysBook α-model one-way latency anchors (InfiniBand NDR/HDR, RoCE, TCP)",
    notes="Order-of-magnitude μs values for napkin math; not vendor QoS guarantees.",
)

BOOK_SWITCH_OPTICS = _conv(
    "prov:book-switch-optics",
    "MLSysBook switch-ASIC capacity (51.2T/102.4T) and 400G optics power (pluggable/CPO) reference figures",
    notes="2025-26 datacenter-switching reference points for the network-fabrics worked examples.",
)

BOOK_NETWORK_ENERGY = _conv(
    "prov:book-network-energy",
    "MLSysBook network data-transfer energy anchors (5G per-MB, generic per-KB)",
    notes="Order-of-magnitude transfer-energy figures for intuition; not measured device values.",
)

BOOK_RECOVERY_ASSUMPTIONS = _conv(
    "prov:book-recovery-time-assumptions",
    "MLSysBook fleet recovery design assumptions (heartbeat, reschedule, checkpoint BW)",
    notes="Engineering targets for appendix reliability tables; see Young/Daly in book prose.",
)

BOOK_OVERHEAD_BUDGETS = _conv(
    "prov:book-overhead-budgets",
    "MLSysBook combined overhead budgets (pipeline, checkpoint, failure, maintenance)",
    notes="Fractions of wall time for 10k+ GPU training scenarios.",
)

BOOK_SCALING_EFFICIENCY_TIERS = _conv(
    "prov:book-scaling-efficiency-tiers",
    "MLSysBook illustrative scaling efficiency vs GPU count (32→1024 GPUs)",
    notes="8192-GPU tier uses MEGASCALE literature anchor separately.",
)

BOOK_ENERGY_HIERARCHY = _conv(
    "prov:book-energy-hierarchy",
    "MLSysBook simplified energy hierarchy: architecture-class effective pJ/FLOP (CPU→ASIC) and per-byte data-movement cost (register→network)",
    notes="Order-of-magnitude teaching figures; effective system-level energy, consistent with the Horowitz (2014) energy trend.",
)

BOOK_WUE_ANCHORS = _conv(
    "prov:book-wue-anchors",
    "MLSysBook water-usage effectiveness (WUE) tiers for sustainability examples",
)

BOOK_RACK_POWER = _conv(
    "prov:book-rack-power-tiers",
    "MLSysBook rack power tiers (traditional vs AI cluster, air-cooling limit)",
)

BOOK_CLOUD_PRICING_2024 = Provenance(
    id="prov:book-cloud-pricing-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative US cloud list prices (2024–2025 order of magnitude)",
    notes="GPU-hour, egress, and electricity rates for worked examples—not a specific vendor quote.",
    verified="2025-03-06",
)

BOOK_STORAGE_PRICING_2024 = Provenance(
    id="prov:book-storage-pricing-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative cloud/object-storage list prices (2024 order of magnitude)",
    notes="S3, Glacier, NVMe tier rates for data-engineering worked examples.",
    verified="2025-03-06",
)

BOOK_LABELING_PRICING_2024 = Provenance(
    id="prov:book-labeling-pricing-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative data-labeling cost ranges (2024 estimates)",
    notes="Crowd, bounding-box, and medical labeling tiers for workflow examples.",
    verified="2025-03-06",
)

BOOK_FLEET_ECONOMICS_2024 = Provenance(
    id="prov:book-fleet-economics-2024",
    kind=ProvenanceKind.ILLUSTRATIVE,
    ref="Illustrative internal GPU-hour and chargeback rates (2024)",
    notes="On-demand, spot, and internal chargeback references for fleet orchestration examples.",
    verified="2025-03-06",
)

BARROSO_DATACENTER_ECONOMICS = _lit(
    "prov:barroso-datacenter-economics",
    "Barroso et al. (2018), The Datacenter as a Computer",
    url="https://doi.org/10.1201/9781351066146",
)

BOOK_CAPACITY_LEAD_TIMES = _est(
    "prov:book-capacity-lead-times",
    "MLSysBook illustrative datacenter build-out lead times",
    notes="Order-of-magnitude planning anchors for compute-infrastructure chapters.",
)

BOOK_CARBON_PER_GPU_HR = _est(
    "prov:book-carbon-per-gpu-hr",
    "Illustrative per-GPU-hour carbon proxy for responsible-AI examples",
    notes="0.16 kg/GPU-hr order-of-magnitude; not a grid-specific intensity calculation.",
)

MFU_INFERENCE_BATCHED_LIT = _lit(
    "prov:mfu-inference-batched",
    "Pope et al. (2023); batched inference MFU upper illustrative bound",
    url="https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html",
    notes="0.40 is an upper illustrative bound for large-batch inference, not batch-1.",
)
