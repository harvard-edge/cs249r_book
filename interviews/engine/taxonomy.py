"""
DEPRECATED — Use taxonomy.json as the single source of truth.

This file contains the legacy tag-based taxonomy (~50 tags across 12 areas).
It has been superseded by taxonomy.json (549 concepts with prerequisite graph).

The legacy importers (vault_loop.py, vault_fill.py) still reference this file.
All new code should use taxonomy.json via vault.py commands:
    vault.py taxonomy-check   — diagnose issues
    vault.py taxonomy-sync    — export to staffml app
    vault.py taxonomy-improve — Gemini-powered improvement

Original description:
The canonical tag vocabulary for all StaffML questions. Every question gets
1-3 tags from this controlled set. No freestyling.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# The taxonomy: 12 competency areas → specific tags
#
# Revised based on expert review:
# - Chip Huyen: add experimentation, cost, LLM ops, expand data
# - Jeff Dean: promote parallelism + networking to first-class areas
# - Vijay Reddi: add edge architectures, sensing, embedded tags
# - All three: remove vendor-specific tags (tensorrt, kubernetes)
# ---------------------------------------------------------------------------

TAXONOMY: dict[str, dict] = {
    "compute": {
        "description": "Reasoning about compute-bound vs memory-bound workloads",
        "tags": [
            "roofline",
            "arithmetic-intensity",
            "compute-bound",
            "memory-bound",
            "tensor-cores",
            "simd",
            "heterogeneous-compute",  # CPU/GPU/NPU scheduling
        ],
    },
    "memory": {
        "description": "Accounting for where every byte lives and moves",
        "tags": [
            "memory-hierarchy",       # Covers L1/L2/HBM/SRAM/VRAM as sub-concepts
            "kv-cache",
            "activation-memory",
            "memory-bandwidth",
            "dma",
            "persistent-storage",     # Flash, NVMe, SD (renamed from flash-storage per Dean)
        ],
    },
    "precision": {
        "description": "Numerical formats, quantization, and their system effects",
        "tags": [
            "quantization",
            "mixed-precision",
            "calibration",
            "overflow",
        ],
    },
    "architecture": {
        "description": "Mapping model architecture choices to resource consumption",
        "tags": [
            "scaling-laws",
            "attention",
            "transformers",
            "cnn",
            "depthwise-separable",    # Edge architecture (per Reddi)
            "neural-architecture-search",  # NAS for MCUs (per Reddi)
            "early-exit",             # Edge/mobile inference (per Reddi)
            "moe",
            "model-cost",
        ],
    },
    "latency": {
        "description": "Decomposing end-to-end latency and identifying bottlenecks",
        "tags": [
            "latency",
            "throughput",
            "ttft",
            "tpot",
            "batching",
            "queueing",
            "real-time",
        ],
    },
    "power": {
        "description": "Energy as a first-class constraint",
        "tags": [
            "power",
            "thermal",
            "tops-w",
            "duty-cycle",
            "battery",
            "cooling",
            "energy-harvesting",      # TinyML (per Reddi)
            "sustainable-ai",
        ],
    },
    "optimization": {
        "description": "Making models smaller/faster without destroying accuracy",
        "tags": [
            "pruning",
            "distillation",
            "operator-fusion",
            "flash-attention",        # Kept — becoming a lasting principle (Dean, Reddi agree)
            "speculative-decoding",
            "graph-optimization",     # Replaced tensorrt — principle-based (all 3 experts)
            "operator-scheduling",    # MCU memory planning (per Reddi)
            "compilation",            # Graph/model compilation for target hardware (moved from deployment)
        ],
    },
    "parallelism": {
        "description": "Distributing work across devices — the physics of keeping GPUs fed",
        # Promoted from cross-cutting per Dean + Huyen
        "tags": [
            "data-parallelism",
            "tensor-parallelism",
            "pipeline-parallelism",
            "expert-parallelism",     # MoE routing
            "fsdp",                   # Fully sharded data parallel / ZeRO
            "collective-communication",  # AllReduce, AllGather, etc.
            "gradient-compression",
        ],
    },
    "networking": {
        "description": "Interconnects that determine which parallelism strategies are viable",
        # Promoted from cross-cutting per Dean + Huyen
        "tags": [
            "interconnect",           # NVLink, InfiniBand, PCIe, CXL
            "network-topology",       # Fat-tree, torus, rail-optimized
            "congestion",
            "rdma",
            "bus-protocol",           # I2C, SPI, UART for TinyML (per Reddi)
            "wireless",              # BLE, LoRaWAN, cellular for edge (per Reddi)
        ],
    },
    "deployment": {
        "description": "Getting models into production and keeping them running",
        "tags": [
            "serving",
            "deployment",             # Getting model to production (distinct from serving)
            "container-orchestration", # Principle-based (per Dean)
            "rollout",
            "model-registry",
            "ota",
            "firmware",
            "rag",                    # Retrieval-augmented generation (per Huyen)
            "guardrails",             # LLM safety/filtering (per Huyen)
        ],
    },
    "reliability": {
        "description": "Detecting silent failures and designing for graceful degradation",
        "tags": [
            "monitoring",
            "drift",
            "fault-tolerance",
            "watchdog",
            "incident-response",
            "checkpoint",
            "observability",
            "straggler-mitigation",   # Per Dean
        ],
    },
    "data": {
        "description": "Data pipelines, quality, and the feedback loop from deployment back to training",
        # Expanded per Huyen — "data problems cause 80% of production ML failures"
        "tags": [
            "data-pipeline",
            "feature-store",
            "data-quality",
            "training-serving-skew",
            "labeling",
            "data-versioning",
            "streaming-data",         # Per Huyen
            "data-debugging",         # Per Huyen
            "sensor-pipeline",        # Promoted from cross-cutting (per Reddi)
        ],
    },
}

# Cross-cutting tags (span multiple competency areas)
CROSS_CUTTING_TAGS: list[str] = [
    # Security & privacy
    "security",
    "privacy",
    "federated",
    "adversarial",
    # Economics & evaluation (per Huyen)
    "economics",
    "tco",                   # Total cost of ownership
    "cost-per-query",
    "ab-testing",            # Experimentation (per Huyen)
    "offline-evaluation",    # Metric design (per Huyen)
    # Feedback loops (per Huyen + Reddi)
    "feedback-loops",
    "retraining-triggers",
    "continual-learning",    # On-device learning (per Reddi)
    # Benchmarking (per Reddi)
    "benchmark-methodology",
]

# ---------------------------------------------------------------------------
# Derived flat sets for validation
# ---------------------------------------------------------------------------

def get_all_tags() -> set[str]:
    """Get the complete set of canonical tags."""
    tags = set()
    for area in TAXONOMY.values():
        tags.update(area["tags"])
    tags.update(CROSS_CUTTING_TAGS)
    return tags


def get_area_for_tag(tag: str) -> str | None:
    """Find which competency area a tag belongs to."""
    for area_name, area in TAXONOMY.items():
        if tag in area["tags"]:
            return area_name
    if tag in CROSS_CUTTING_TAGS:
        return "cross-cutting"
    return None


# Flat set for fast membership checks
ALL_TAGS: set[str] = get_all_tags()


# ---------------------------------------------------------------------------
# Normalization map: old freeform tags → canonical tags
# ---------------------------------------------------------------------------

NORMALIZE_MAP: dict[str, str] = {
    # -----------------------------------------------------------------------
    # Base renames: old taxonomy → canonical tags
    # -----------------------------------------------------------------------
    "tensorrt": "graph-optimization",
    "kubernetes": "container-orchestration",
    "flash-storage": "persistent-storage",
    "vram": "memory-hierarchy",
    "hbm": "memory-hierarchy",
    "sram": "memory-hierarchy",
    "int8": "quantization",
    "fp16": "mixed-precision",
    "parallelism": "data-parallelism",
    "interconnect": "interconnect",
    "sensor-pipeline": "sensor-pipeline",
    "model-size": "model-cost",
    "training": "data-parallelism",
    "distributed": "data-parallelism",
    "mlops": "deployment",
    "ml-ops": "deployment",
    "frameworks": "compilation",
    "compiler-runtime": "compilation",
    "compiler": "compilation",
    "npu-compiler": "graph-optimization",
    "storage": "persistent-storage",
    "storage-io": "persistent-storage",
    "flash-memory": "persistent-storage",
    "os": "deployment",
    "mobile-os": "deployment",
    "mobile-frameworks": "compilation",
    "compute": "roofline",
    "cpu-gpu-arch": "heterogeneous-compute",
    "hardware-mac": "simd",
    "mflops": "roofline",
    "memory": "memory-hierarchy",
    "kv-cache-memory": "kv-cache",
    "kv-cache-management": "kv-cache",
    "memory-layout": "memory-hierarchy",
    "memory-management": "memory-hierarchy",
    "precision": "mixed-precision",
    "quantization-memory": "quantization",
    "quantization-npu": "quantization",
    "tinyml-quantization": "quantization",
    "architecture": "model-cost",
    "model-architecture": "model-cost",
    "attention-mechanisms": "attention",
    "serving": "serving",
    "continuous-batching": "batching",
    "queueing-theory": "queueing",
    "power-thermal": "power",
    "thermal-management": "thermal",
    "power-management": "power",
    "energy": "power",
    "model-compression": "pruning",
    "compression": "pruning",
    "optimization": "operator-fusion",
    "model-deployment": "deployment",
    "canary-deployment": "rollout",
    "model-monitoring": "monitoring",
    "reliability": "fault-tolerance",
    "functional-safety": "fault-tolerance",
    "incident-response": "incident-response",
    "data-engineering": "data-pipeline",
    "data-versioning": "data-versioning",
    "data-management": "data-pipeline",
    "data-locality": "data-pipeline",
    "data-quality": "data-quality",
    "network-fabric": "interconnect",
    "networking": "interconnect",
    "network": "interconnect",
    "cloud-networking": "interconnect",
    "soc-interconnect": "interconnect",
    "pipeline": "data-pipeline",
    "sensors": "sensor-pipeline",
    "sensor": "sensor-pipeline",
    "sensor-fusion": "sensor-pipeline",
    "sensor-io": "sensor-pipeline",
    "hardware": "model-cost",
    "edge-hardware": "model-cost",
    "cpu-architecture": "heterogeneous-compute",
    "tpu-architecture": "heterogeneous-compute",
    "npu-architecture": "heterogeneous-compute",
    "soc-architecture": "heterogeneous-compute",
    "hardware-acceleration": "roofline",
    "hw-acceleration": "roofline",
    "rtos-scheduling": "real-time",
    "scheduling": "real-time",
    "edge": "deployment",
    "benchmarking": "roofline",
    "debug-interface": "observability",
    "cost-optimization": "economics",
    "cost-attribution": "economics",
    "npu-delegation": "heterogeneous-compute",
    "numerical-precision": "mixed-precision",
    "distributed-training": "data-parallelism",
    "collective-communication": "collective-communication",
    "network-topology": "interconnect",
    "concurrency": "data-parallelism",
    "multi-sensor-fusion": "sensor-pipeline",
    "thermal-throttling": "thermal",
    "dynamic-memory-allocation": "memory-hierarchy",
    "ota-updates": "ota",
    "adversarial-robustness": "adversarial",
    "resource-contention": "memory-bound",
    "battery-impact": "battery",
    "power-consumption": "power",
    "gpu": "roofline",
    "memory-pressure": "memory-hierarchy",
    "power-energy": "power",
    "instruction-set": "simd",
    "serverless-inference": "serving",
    "cluster-scheduling": "container-orchestration",
    "os-scheduling": "real-time",
    "thread": "real-time",
    "instruction-cache": "memory-hierarchy",
    "cache-performance": "memory-hierarchy",
    "edge-performance": "roofline",
    "display-pipeline": "latency",
    "model-versioning": "model-registry",
    "fleet-management": "deployment",
    "resource-management": "deployment",
    "network-management": "interconnect",
    "memory-management-oom-android": "memory-hierarchy",
    "memory-management-latency-spikes-android": "memory-hierarchy",
    "Distributed Training & DDP": "data-parallelism",
    "Distributed Training & 3D Parallelism": "data-parallelism",
    "npu": "heterogeneous-compute",
    "fpga": "heterogeneous-compute",
    "accelerator": "heterogeneous-compute",
    "app-lifecycle": "deployment",
    "model-loading": "serving",
    "inference": "serving",
    "real-time-os": "real-time",
    "interrupt": "real-time",
    "dma-transfer": "dma",
    "i2c": "sensor-pipeline",
    "spi": "sensor-pipeline",
    "uart": "sensor-pipeline",
    "ble": "sensor-pipeline",
    "wifi": "interconnect",
    "cellular": "interconnect",
    "5g": "interconnect",
    "cache-hierarchy": "memory-hierarchy",
    "cache-coherence": "memory-hierarchy",
    "memory-technologies": "memory-hierarchy",
    "virtual-memory": "memory-hierarchy",
    "memory-alignment": "memory-hierarchy",
    "memory-architecture": "memory-hierarchy",
    "memory-capacity": "memory-hierarchy",
    "memory-footprint": "memory-hierarchy",
    "memory-power": "memory-hierarchy",
    "memory-mapped-io": "dma",
    "on-chip-memory": "memory-hierarchy",
    "scratchpad-memory": "memory-hierarchy",
    "fragmentation": "memory-hierarchy",
    "shared-bandwidth": "memory-bandwidth",
    "cpu-cache-memory-access": "memory-hierarchy",
    "numa-multicore": "memory-hierarchy",
    "NUMA": "memory-hierarchy",
    "dma-transfers": "dma",
    "mmio": "dma",
    "sparsity-memory": "memory-hierarchy",
    "compute-intensity": "arithmetic-intensity",
    "compute-overhead": "roofline",
    "compute-memory-bandwidth": "memory-bound",
    "branch-prediction": "roofline",
    "algorithms": "roofline",
    "math": "roofline",
    "dynamic-quantization": "quantization",
    "Extreme Quantization": "quantization",
    "quantization-hardware": "quantization",
    "quantization-robustness": "quantization",
    "quantization-low-precision-hardware-support": "quantization",
    "quantization-memory-deployment": "quantization",
    "cold-start-latency": "latency",
    "latency-budgets": "latency",
    "real-time-latency": "real-time",
    "real-time-systems": "real-time",
    "wcet": "real-time",
    "deterministic-timing": "real-time",
    "rtos": "real-time",
    "rtos-deterministic": "real-time",
    "timing": "real-time",
    "interrupt-vs-polling": "real-time",
    "low-power": "power",
    "ultra-low-power": "power",
    "power-gating": "power",
    "voltage-scaling": "power",
    "dvfs": "power",
    "energy-harvesting": "duty-cycle",
    "thermal-design": "thermal",
    "thermal-power-sustained-perf": "thermal",
    "carbon-aware": "sustainable-ai",
    "power-efficiency-frameworks-android-ios": "power",
    "power-management-adaptive": "power",
    "apple-silicon": "heterogeneous-compute",
    "mobile-gpu": "heterogeneous-compute",
    "micro-npu": "heterogeneous-compute",
    "soc": "heterogeneous-compute",
    "system-on-chip": "heterogeneous-compute",
    "tinyml-hardware": "heterogeneous-compute",
    "hardware-topology": "interconnect",
    "bus-arbitration": "interconnect",
    "clock-tree": "heterogeneous-compute",
    "hardware-lifecycle": "deployment",
    "custom-ops-heterogeneous-compute-vendor-sdk": "heterogeneous-compute",
    "heterogeneous-compute-scheduling-latency-power": "heterogeneous-compute",
    "heterogeneous-memory-coherence": "memory-hierarchy",
    "containerization": "container-orchestration",
    "spot-instances": "economics",
    "model-delivery": "deployment",
    "model-formats": "deployment",
    "model-lifecycle": "deployment",
    "model-update": "ota",
    "model-optimization": "pruning",
    "platform": "deployment",
    "health-checks": "monitoring",
    "long-term-reliability": "fault-tolerance",
    "testing": "monitoring",
    "profiling": "observability",
    "boot-sequence": "firmware",
    "boot-time": "firmware",
    "secure-boot": "firmware",
    "peripheral-timer": "real-time",
    "data-collection": "data-pipeline",
    "data-loading": "data-pipeline",
    "data-privacy": "privacy",
    "data-provenance": "data-versioning",
    "reproducibility": "data-versioning",
    "Differential Privacy": "privacy",
    "fairness": "adversarial",
    "robustness": "adversarial",
    "on-device-training": "continual-learning",
    "multi-model-inference": "serving",
    "multi-tenant": "serving",
    "cross-region": "interconnect",
    "edge-cloud-sync": "interconnect",
    "zero-copy-pipeline": "dma",
    "sensor-calibration": "sensor-pipeline",
    "sensor-physics": "sensor-pipeline",
    "system-design": "model-cost",
    "Ad Click Aggregation & Streaming Data": "data-pipeline",
    "Real-Time Ranking & Retrieval": "serving",
    "Serving & Global Infrastructure": "serving",
    "LLM Inference & Memory Management": "kv-cache",
    "Multimodal Agentic OS": "deployment",
    "collective-design": "data-parallelism",

    # -----------------------------------------------------------------------
    # Corpus-wide normalization (855 freeform topic strings → canonical tags)
    # Generated 2026-03-23 by analyzing all unique topic strings in corpus.json
    # -----------------------------------------------------------------------

    # --- ab-testing ---
    "mobile-ab-testing": "ab-testing",

    # --- activation-memory ---
    "adam-optimizer-memory": "activation-memory",
    "gradient-memory-tax": "activation-memory",

    # --- adversarial ---
    "adversarial-patch-physical": "adversarial",

    # --- arithmetic-intensity ---
    "arithmetic-intensity-batching": "arithmetic-intensity",
    "tinyml-arithmetic-intensity": "arithmetic-intensity",

    # --- attention ---
    "attention-vs-decoding-optimization": "attention",
    "edge-vision-transformer": "attention",
    "vision-transformer-bottleneck": "attention",
    "vlm-optimization-strategy": "attention",
    "vision-transformer-roofline-analysis": "attention",

    # --- batching ---
    "autoscaling-batching-tradeoff": "batching",
    "continuous-batching-bottlenecks": "batching",
    "continuous-batching-economics": "batching",
    "continuous-batching-latency": "batching",
    "continuous-batching-latency-throughput": "batching",
    "continuous-batching-padding": "batching",
    "continuous-batching-queueing": "batching",
    "continuous-batching-queueing-theory": "batching",
    "continuous-batching-scheduling": "batching",
    "continuous-batching-tail-latency": "batching",
    "continuous-batching-throughput": "batching",
    "continuous-batching-tpot": "batching",
    "continuous-batching-tradeoffs": "batching",
    "continuous-batching-ttft": "batching",
    "continuous-batching-utilization": "batching",
    "continuous-batching-vs-static": "batching",
    "llm-continuous-batching-ttft": "batching",
    "static-batching-vs-latency": "batching",
    "static-vs-continuous-batching": "batching",
    "tpot-economics-continuous-batching": "batching",
    "ttft-static-batching": "batching",
    "ttft-tpot-continuous-batching-queueing": "batching",
    "ttft-tpot-continuous-batching-queueing-theory": "batching",

    # --- battery ---
    "battery-drain": "battery",
    "battery-drain-calculation": "battery",
    "battery-drain-duty-cycle": "battery",
    "battery-drain-energy-harvesting": "battery",
    "battery-duty-cycle": "battery",
    "battery-life-estimation": "battery",
    "duty-cycling-and-battery-drain": "battery",
    "duty-cycling-battery-drain": "battery",
    "energy-harvesting-battery-drain": "battery",
    "mobile-battery-drain": "battery",
    "mobile-battery-duty-cycle": "battery",
    "tinyml-duty-cycle-battery": "battery",

    # --- bus-protocol ---
    "bus-protocol-fallacy": "bus-protocol",
    "bus-protocols": "bus-protocol",
    "tinyml-bus-protocol": "bus-protocol",
    "interconnect-topology-bus-protocols": "bus-protocol",

    # --- checkpoint ---
    "checkpointing-flash-memory": "checkpoint",

    # --- cnn ---
    "cnn-architecture-efficiency": "cnn",
    "cnn-optimization": "cnn",
    "cnn-optimization-edge": "cnn",
    "cnn-scaling-edge": "cnn",
    "cnn-vs-transformer": "cnn",
    "cnn-vs-transformer-edge": "cnn",
    "cnn-vs-transformer-edge-scaling": "cnn",
    "cnn-vs-transformer-real-time-vision": "cnn",
    "cnn-vs-transformer-scaling": "cnn",
    "convolutional-architectures": "cnn",
    "inverted-residuals": "cnn",
    "mobile-cnn-transformer-tradeoff": "cnn",
    "mobile-cnn-vs-transformer": "cnn",
    "mobile-transformer-cnn-tradeoff": "cnn",
    "mobilenet-architectures": "cnn",
    "cnn-vs-transformer-roofline": "cnn",

    # --- collective-communication ---
    "allreduce-bandwidth": "collective-communication",
    "federated-learning-allreduce": "collective-communication",
    "distributed-communication": "collective-communication",

    # --- cooling ---
    "cloud-cooling-economics": "cooling",
    "cloud-cooling-power": "cooling",
    "cloud-cooling-pue": "cooling",
    "cloud-cooling-throttling": "cooling",
    "cloud-power-and-cooling": "cooling",
    "cloud-power-cooling": "cooling",
    "cooling-efficiency": "cooling",
    "power-and-cooling": "cooling",

    # --- data-parallelism ---
    "data-parallelism-allreduce": "data-parallelism",
    "distributed-inference-edge": "data-parallelism",
    "distributed-parallelism-edge": "data-parallelism",
    "distributed-systems-edge": "data-parallelism",
    "distributed-training-bottleneck": "data-parallelism",
    "distributed-training-bottlenecks": "data-parallelism",
    "distributed-training-edge": "data-parallelism",
    "edge-parallelism": "data-parallelism",
    "on-device-parallelism": "data-parallelism",
    "parallelism-edge": "data-parallelism",
    "parallelism-edge-auto": "data-parallelism",
    "parallelism-on-mobile": "data-parallelism",
    "training-time-estimation": "data-parallelism",

    # --- data-pipeline ---
    "data-ingress-bottleneck": "data-pipeline",
    "data-loading-bottleneck": "data-pipeline",
    "data-pipelines": "data-pipeline",

    # --- data-quality ---
    "data-quality-skew": "data-quality",

    # --- deployment ---
    "app-store-delivery": "deployment",
    "app-store-delivery-on-demand-model-download-a-b-testing": "deployment",
    "auto": "deployment",
    "cloud-deployment-footprint": "deployment",
    "mobile-deployment-auto": "deployment",
    "mobile-deployment-constraints": "deployment",
    "model-deployment-mobile": "deployment",
    "on-demand-model-download": "deployment",
    "tinyml-mlops-adaptation": "deployment",

    # --- depthwise-separable ---
    "cnn-optimization-depthwise-separable": "depthwise-separable",
    "depthwise-convolution-cost": "depthwise-separable",
    "depthwise-convolution-efficiency": "depthwise-separable",
    "depthwise-convolution-memory": "depthwise-separable",
    "depthwise-separable-cnn": "depthwise-separable",
    "depthwise-separable-computation": "depthwise-separable",
    "depthwise-separable-convolution": "depthwise-separable",
    "depthwise-separable-convolutions": "depthwise-separable",
    "efficient-convolutions": "depthwise-separable",
    "tinyml-depthwise-separable": "depthwise-separable",
    "scaling-laws-cnn-depthwise": "depthwise-separable",

    # --- distillation ---
    "distillation-economics": "distillation",
    "distillation-fusion-attention": "distillation",
    "knowledge-distillation": "distillation",
    "model-distillation": "distillation",
    "operator-fusion-distillation-memory-bound": "distillation",
    "speculative-decoding-distillation": "distillation",
    "distillation-pruning-fusion": "distillation",

    # --- dma ---
    "dma-bandwidth-latency": "dma",
    "dma-bottleneck": "dma",
    "dma-cpu-offload": "dma",
    "dma-cpu-tradeoff": "dma",
    "dma-double-buffering": "dma",
    "dma-energy-power-compute-analysis": "dma",
    "dma-memory-hierarchy": "dma",
    "dma-memory-transfer": "dma",
    "dma-offload": "dma",
    "dma-race-condition": "dma",
    "dma-transfer-time": "dma",
    "dma-vs-cpu": "dma",
    "dma-vs-cpu-copy": "dma",
    "pcie-dma-zero-copy": "dma",
    "tinyml-dma-memory-bus": "dma",
    "tinyml-dma-sram-contention": "dma",

    # --- drift ---
    "data-drift-monitoring": "drift",
    "data-drift-monitoring-power": "drift",
    "data-drift-monitoring-tinyml": "drift",
    "monitoring-data-drift": "drift",
    "on-device-drift-monitoring": "drift",
    "sensor-drift-regression": "drift",
    "silent-accuracy-loss": "drift",
    "tinyml-monitoring-drift": "drift",

    # --- duty-cycle ---
    "duty-cycle-energy": "duty-cycle",
    "duty-cycle-power": "duty-cycle",
    "duty-cycling": "duty-cycle",
    "duty-cycling-and-power": "duty-cycle",
    "duty-cycling-energy": "duty-cycle",
    "duty-cycling-energy-consumption": "duty-cycle",
    "duty-cycling-fundamentals": "duty-cycle",
    "duty-cycling-power": "duty-cycle",
    "power-duty-cycle": "duty-cycle",
    "real-time-duty-cycle": "duty-cycle",
    "thermal-duty-cycle-mobile": "duty-cycle",
    "thermal-management-duty-cycle-auto": "duty-cycle",
    "thermal-throttling-duty-cycling": "duty-cycle",
    "tinyml-duty-cycle": "duty-cycle",
    "tinyml-duty-cycle-average-power": "duty-cycle",
    "tinyml-duty-cycle-lifetime": "duty-cycle",
    "tinyml-duty-cycle-power": "duty-cycle",
    "tinyml-duty-cycling": "duty-cycle",

    # --- economics ---
    "cloud-economics-tco": "economics",
    "container-rollout-cost": "economics",
    "economics-ab-testing": "economics",
    "economics-federated-learning-tco": "economics",
    "economics-power-ratio": "economics",
    "economics-power-tco": "economics",
    "economics-privacy-federated-learning": "economics",
    "economics-privacy-fl": "economics",
    "economics-privacy-tco": "economics",
    "economics-serving-cost": "economics",
    "economics-tco": "economics",
    "economics-tco-ab-testing": "economics",
    "economics-tco-cloud": "economics",
    "economics-tco-cloud-auto": "economics",
    "economics-tco-federated-learning": "economics",
    "economics-tco-lifecycle": "economics",
    "economics-tco-privacy": "economics",
    "economics-tco-tinyml": "economics",
    "edge-economics-power": "economics",
    "edge-economics-tco": "economics",
    "federated-learning-economics": "economics",
    "federated-learning-tco": "economics",
    "federated-learning-tco-energy": "economics",
    "federated-learning-tco-privacy": "economics",
    "federated-learning-tco-security": "economics",
    "fleet-economics-tco": "economics",
    "mobile-economics-tco": "economics",
    "mobile-federated-economics": "economics",
    "mobile-federated-learning-economics": "economics",
    "mobile-federated-learning-tco": "economics",
    "mobile-memory-economics": "economics",
    "mobile-privacy-economics": "economics",
    "mobile-privacy-economics-ab-testing": "economics",
    "mobile-privacy-economics-federated": "economics",
    "mobile-privacy-federated-learning-economics": "economics",
    "mobile-privacy-federated-learning-tco": "economics",
    "privacy-economics": "economics",
    "security-privacy-federated-learning-economics": "economics",
    "tco-ab-testing-fleet": "economics",
    "tco-economics": "economics",
    "tco-edge-vs-cloud": "economics",
    "tco-federated-learning": "economics",
    "tco-federated-learning-economics": "economics",
    "tinyml-economics": "economics",
    "tinyml-economics-federated-learning": "economics",
    "tinyml-economics-power": "economics",
    "tinyml-economics-privacy": "economics",
    "tinyml-economics-tco": "economics",
    "tinyml-federated-economics": "economics",
    "tinyml-federated-learning-economics": "economics",
    "tinyml-federated-learning-tco": "economics",
    "tinyml-federated-tco": "economics",
    "tinyml-privacy-economics": "economics",
    "tinyml-privacy-federated-learning-tco": "economics",
    "tinyml-security-federated-learning-economics": "economics",
    "tinyml-tco-ab-testing": "economics",
    "tinyml-tco-economics": "economics",
    "tinyml-tco-federated-learning": "economics",

    # --- fault-tolerance ---
    "checkpointing-fault-tolerance-flash": "fault-tolerance",
    "data-drift-fault-tolerance-monitoring": "fault-tolerance",
    "fault-tolerance-checkpointing": "fault-tolerance",
    "fault-tolerance-checkpointing-flash": "fault-tolerance",
    "fault-tolerance-checkpointing-power": "fault-tolerance",
    "fault-tolerance-checkpointing-power-management": "fault-tolerance",
    "fault-tolerance-checkpointing-tinyml": "fault-tolerance",
    "fault-tolerance-checkpointing-watchdog-timers": "fault-tolerance",
    "fault-tolerance-drift": "fault-tolerance",
    "fault-tolerance-drift-monitoring": "fault-tolerance",
    "fault-tolerance-drift-monitoring-tinyml": "fault-tolerance",
    "fault-tolerance-drift-tinyml": "fault-tolerance",
    "fault-tolerance-monitoring-drift": "fault-tolerance",
    "fault-tolerance-monitoring-hardware-failure": "fault-tolerance",
    "fault-tolerance-watchdog-checkpointing": "fault-tolerance",
    "fault-tolerance-watchdog-timers": "fault-tolerance",
    "mobile-fault-tolerance": "fault-tolerance",
    "monitoring-data-drift-fault-tolerance": "fault-tolerance",
    "tinyml-fault-tolerance": "fault-tolerance",
    "tinyml-fault-tolerance-drift": "fault-tolerance",
    "tinyml-monitoring-drift-fault-tolerance": "fault-tolerance",

    # --- federated ---
    "federated-analytics": "federated",
    "federated-learning": "federated",
    "federated-learning-communication": "federated",
    "federated-learning-privacy": "federated",
    "federated-learning-thermals": "federated",

    # --- firmware ---
    "firmware-convergence": "firmware",
    "flash-programming": "firmware",

    # --- graph-optimization ---
    "automated-model-optimization": "graph-optimization",
    "automotive-model-optimization": "graph-optimization",
    "coreml-tflite-optimization": "graph-optimization",
    "edge-auto-optimization": "graph-optimization",
    "edge-auto-optimization-synthesis": "graph-optimization",
    "edge-architectural-optimization": "graph-optimization",
    "inference-optimization-automation": "graph-optimization",
    "inference-optimization-stack": "graph-optimization",
    "inference-optimization-strategy": "graph-optimization",
    "mobile-auto-compression-architecture": "graph-optimization",
    "mobile-optimization-systems-design": "graph-optimization",
    "mobile-transformer-optimization": "graph-optimization",
    "model-optimization-platform": "graph-optimization",
    "model-optimization-stack": "graph-optimization",
    "model-optimization-strategy": "graph-optimization",
    "model-optimization-tradeoffs": "graph-optimization",
    "multi-model-optimization-edge": "graph-optimization",
    "optimization-traps": "graph-optimization",
    "tensorrt-optimization": "graph-optimization",
    "tflite-delegate-partitioning": "graph-optimization",
    "structured-sparsity-tensorrt": "graph-optimization",
    "tensorrt-optimization-structured-pruning-for-edge": "graph-optimization",

    # --- guardrails ---
    "guardrail-deployment-memory": "guardrails",

    # --- heterogeneous-compute ---
    "mobile-backend-architecture": "heterogeneous-compute",
    "mobile-operator-delegation": "heterogeneous-compute",
    "npu-delegation-bottleneck": "heterogeneous-compute",
    "npu-delegation-power": "heterogeneous-compute",

    # --- interconnect ---
    "cloud-interconnect-bottleneck": "interconnect",
    "cloud-interconnect-topology": "interconnect",
    "edge-interconnect-bottleneck": "interconnect",
    "edge-interconnect-design": "interconnect",
    "edge-interconnect-topology": "interconnect",
    "gpu-interconnect-bottleneck": "interconnect",
    "gpu-interconnect-topology": "interconnect",
    "gpu-interconnects": "interconnect",
    "infiniband-vs-ethernet": "interconnect",
    "inter-node-topology": "interconnect",
    "interconnect-bottleneck": "interconnect",
    "interconnect-bottleneck-edge": "interconnect",
    "interconnect-bottlenecks": "interconnect",
    "interconnect-hierarchy": "interconnect",
    "interconnect-latency": "interconnect",
    "interconnect-latency-comparison": "interconnect",
    "interconnect-latency-topology": "interconnect",
    "interconnect-mismatch": "interconnect",
    "interconnect-protocol-fallacy": "interconnect",
    "interconnect-protocol-mismatch": "interconnect",
    "interconnect-protocols-tinyml": "interconnect",
    "interconnect-scale-mismatch": "interconnect",
    "interconnect-topology": "interconnect",
    "interconnect-tradeoffs": "interconnect",
    "intra-node-comms": "interconnect",
    "mobile-backend-interconnect": "interconnect",
    "mobile-cloud-interconnect": "interconnect",
    "mobile-cloud-interconnects": "interconnect",
    "mobile-interconnect-bottleneck": "interconnect",
    "mobile-interconnect-fallacy": "interconnect",
    "mobile-interconnects": "interconnect",
    "nvlink-vs-infiniband": "interconnect",
    "nvlink-vs-infiniband-latency": "interconnect",
    "nvlink-vs-infiniband-topology": "interconnect",
    "nvlink-vs-pcie": "interconnect",
    "nvlink-vs-pcie-latency": "interconnect",
    "pci-contention-vs-nvlink": "interconnect",
    "pcie-interconnect-bottleneck": "interconnect",
    "pcie-latency-overhead": "interconnect",
    "pcie-latency-vs-bandwidth": "interconnect",
    "protocol-mismatch": "interconnect",
    "tinyml-interconnects": "interconnect",
    "tensor-parallelism-interconnect": "interconnect",

    # --- kv-cache ---
    "continuous-batching-kv-cache": "kv-cache",
    "kv-cache-cost": "kv-cache",
    "kv-cache-memory-edge": "kv-cache",
    "kv-cache-sizing": "kv-cache",
    "kv-cache-transformer-audio-dma": "kv-cache",
    "kv-cache-vram": "kv-cache",
    "kv-cache-vram-accounting": "kv-cache",
    "paged-attention-memory": "kv-cache",
    "paged-attention-memory-fragmentation": "kv-cache",
    "speculative-decoding-kv-cache-memory": "kv-cache",
    "tinyml-kv-cache-dma": "kv-cache",
    "tinyml-memory-hierarchy-kv-cache": "kv-cache",

    # --- latency ---
    "data-latency": "latency",
    "delegation-overhead-latency": "latency",
    "mobile-cloud-latency": "latency",
    "mobile-generative-ai-latency": "latency",
    "mobile-generative-latency": "latency",
    "mobile-generative-latency-auto": "latency",
    "mobile-latency": "latency",
    "mobile-latency-batching": "latency",
    "mobile-latency-bottleneck": "latency",
    "mobile-latency-budget": "latency",
    "mobile-latency-deadline": "latency",
    "mobile-latency-deadlines": "latency",
    "mobile-llm-latency": "latency",
    "mobile-on-device-inference-latency": "latency",
    "mobile-streaming-latency": "latency",
    "on-device-gen-ai-latency": "latency",
    "on-device-inference-latency": "latency",
    "on-device-latency": "latency",
    "on-device-llm-feasibility": "latency",
    "on-device-llm-jank": "latency",
    "on-device-llm-latency": "latency",
    "on-device-llm-scheduling": "latency",
    "perceived-latency": "latency",

    # --- memory-bandwidth ---
    "hbm-bandwidth": "memory-bandwidth",

    # --- memory-bound ---
    "compute-vs-memory-bound": "memory-bound",

    # --- memory-hierarchy ---
    "app-memory-eviction": "memory-hierarchy",
    "audio-pipeline-memory": "memory-hierarchy",
    "dram-budget-sharing": "memory-hierarchy",
    "flash-memory-budget": "memory-hierarchy",
    "inference-memory": "memory-hierarchy",
    "inference-memory-footprint": "memory-hierarchy",
    "memory-budget": "memory-hierarchy",
    "memory-hierarchy-dma": "memory-hierarchy",
    "memory-hierarchy-edge": "memory-hierarchy",
    "memory-hierarchy-kv-cache-dma": "memory-hierarchy",
    "memory-hierarchy-kv-cache-vram-accounting-sram-tensor-arena-dma": "memory-hierarchy",
    "memory-hierarchy-ota": "memory-hierarchy",
    "memory-partitioning-tinyml": "memory-hierarchy",
    "memory-systems": "memory-hierarchy",
    "mobile-memory-bottleneck": "memory-hierarchy",
    "mobile-memory-budget": "memory-hierarchy",
    "mobile-memory-constraints": "memory-hierarchy",
    "mobile-memory-wall": "memory-hierarchy",
    "mobile-unified-memory": "memory-hierarchy",
    "ota-memory-budget": "memory-hierarchy",
    "quantization-memory-constraint": "memory-hierarchy",
    "shared-ram-eviction": "memory-hierarchy",
    "sram-partitioning-tensor-arena-sizing": "memory-hierarchy",
    "sram-tensor-arena": "memory-hierarchy",
    "tensor-arena-fragmentation": "memory-hierarchy",
    "tensor-arena-memory": "memory-hierarchy",
    "tensor-arena-planning": "memory-hierarchy",
    "tensor-arena-sizing": "memory-hierarchy",
    "tensor-arena-sram": "memory-hierarchy",
    "tensor-arena-sram-flash-xip": "memory-hierarchy",
    "tinyml-memory-arena": "memory-hierarchy",
    "tinyml-memory-budget": "memory-hierarchy",
    "tinyml-memory-hierarchy": "memory-hierarchy",
    "tinyml-memory-management": "memory-hierarchy",
    "tinyml-memory-planning": "memory-hierarchy",
    "tinyml-sram-arena": "memory-hierarchy",
    "tinyml-tensor-arena": "memory-hierarchy",
    "unified-memory-architecture": "memory-hierarchy",
    "vram-accounting": "memory-hierarchy",
    "vram-accounting-edge": "memory-hierarchy",

    # --- mixed-precision ---
    "fp16-model-footprint": "mixed-precision",
    "mixed-precision-memory": "mixed-precision",
    "mixed-precision-memory-management": "mixed-precision",
    "mixed-precision-power": "mixed-precision",
    "mixed-precision-quantization": "mixed-precision",
    "mixed-precision-training": "mixed-precision",
    "numerical-representation": "mixed-precision",
    "tinyml-mixed-precision": "mixed-precision",
    "tinyml-mixed-precision-power": "mixed-precision",

    # --- model-cost ---
    "architectural-bottleneck": "model-cost",
    "architectural-tradeoffs-edge": "model-cost",
    "architecture-selection-tinyml": "model-cost",
    "architecture-tradeoff-edge": "model-cost",
    "auto-perception-stack-design": "model-cost",
    "edge-architecture-choice": "model-cost",
    "edge-architecture-scaling": "model-cost",
    "edge-architecture-scaling-auto": "model-cost",
    "edge-architecture-tradeoff": "model-cost",
    "edge-architecture-tradeoffs": "model-cost",
    "edge-architectures-auto": "model-cost",
    "edge-auto-architecture-scaling": "model-cost",
    "efficient-architectures": "model-cost",
    "efficient-architectures-edge": "model-cost",
    "mobile-architecture-choice": "model-cost",
    "mobile-architecture-selection": "model-cost",
    "mobile-architecture-tradeoff": "model-cost",
    "mobile-architecture-tradeoffs": "model-cost",
    "mobile-architectures": "model-cost",
    "resource-constraints": "model-cost",
    "system-constraints": "model-cost",
    "system-scale-mismatch": "model-cost",
    "systems-design-edge-auto": "model-cost",
    "tinyml-architecture-choice": "model-cost",
    "tinyml-architecture-efficiency": "model-cost",
    "tinyml-architecture-selection": "model-cost",
    "tinyml-architecture-tradeoffs": "model-cost",
    "tinyml-constraints": "model-cost",

    # --- monitoring ---
    "mobile-monitoring": "monitoring",
    "monitoring-reliability": "monitoring",

    # --- network-topology ---
    "network-topology-diagnosis": "network-topology",
    "network-topology-rdma": "network-topology",

    # --- neural-architecture-search ---
    "edge-architecture-cnn-vs-transformer-nas-moe": "neural-architecture-search",
    "edge-architecture-nas-moe": "neural-architecture-search",
    "hardware-aware-nas": "neural-architecture-search",
    "hardware-aware-nas-edge": "neural-architecture-search",
    "hybrid-architecture-nas": "neural-architecture-search",
    "mobile-architecture-nas": "neural-architecture-search",
    "nas-for-mcus": "neural-architecture-search",
    "nas-for-power": "neural-architecture-search",
    "nas-power-budget": "neural-architecture-search",
    "nas-roofline": "neural-architecture-search",
    "nas-vs-moe": "neural-architecture-search",
    "tinyml-nas-memory": "neural-architecture-search",
    "tinyml-nas-operators": "neural-architecture-search",
    "tinyml-neural-architecture-search": "neural-architecture-search",

    # --- operator-fusion ---
    "operator-fusion-optimization": "operator-fusion",
    "operator-fusion-pruning": "operator-fusion",

    # --- ota ---
    "ab-testing-ota-risk": "ota",
    "auto-ota-budget": "ota",
    "edge-ota-bandwidth": "ota",
    "edge-ota-orchestration": "ota",
    "edge-ota-rollout": "ota",
    "edge-ota-storage": "ota",
    "edge-ota-update": "ota",
    "fota-flash-budget": "ota",
    "mobile-ota-constraints": "ota",
    "mobile-ota-cost": "ota",
    "mobile-ota-deployment": "ota",
    "mobile-ota-rag": "ota",
    "mobile-ota-rollout": "ota",
    "mobile-ota-size": "ota",
    "mobile-ota-storage": "ota",
    "mobile-ota-updates": "ota",
    "model-serving-ota": "ota",
    "ota-bandwidth-constraint": "ota",
    "ota-bandwidth-planning": "ota",
    "ota-data-cost": "ota",
    "ota-firmware-updates": "ota",
    "ota-firmware-updates-ab-partitioning": "ota",
    "ota-flash-budget": "ota",
    "ota-fleet-bandwidth": "ota",
    "ota-fleet-update": "ota",
    "ota-memory-footprint": "ota",
    "ota-rag-tradeoff": "ota",
    "ota-rag-update": "ota",
    "ota-storage-budget": "ota",
    "ota-storage-management": "ota",
    "ota-update-analysis": "ota",
    "ota-update-bandwidth": "ota",
    "ota-update-bottleneck": "ota",
    "ota-update-cost": "ota",
    "ota-update-economics": "ota",
    "ota-update-fleet": "ota",
    "ota-update-fleet-scale": "ota",
    "ota-update-rollout": "ota",
    "ota-update-size": "ota",
    "ota-update-storage-bottleneck": "ota",
    "rag-ota-memory": "ota",
    "rag-ota-update": "ota",
    "tinyml-ota-memory-fragmentation": "ota",

    # --- persistent-storage ---
    "flash-vs-sram": "persistent-storage",
    "tinyml-flash-budget": "persistent-storage",

    # --- power ---
    "batch-size-energy-math": "power",
    "cloud-power-budgeting": "power",
    "cloud-power-density": "power",
    "datacenter-power": "power",
    "energy-harvesting-power-budget": "power",
    "mobile-energy-consumption": "power",
    "mobile-energy-cost": "power",
    "mobile-monitoring-power": "power",
    "mobile-power-economics": "power",
    "mobile-transformer-power": "power",
    "npu-fallback-energy": "power",
    "power-brown-out-diagnosis": "power",
    "power-budget-architecture": "power",
    "power-constrained-compute": "power",
    "power-economics": "power",
    "power-management-auto": "power",
    "speculative-decoding-power": "power",
    "tinyml-power-analysis": "power",
    "tinyml-power-economics": "power",

    # --- privacy ---
    "ab-testing-privacy-power": "privacy",
    "mobile-energy-privacy": "privacy",
    "on-device-differential-privacy": "privacy",

    # --- pruning ---
    "pruning-distillation": "pruning",
    "pruning-distillation-fusion": "pruning",
    "pruning-distillation-fusion-attention": "pruning",
    "pruning-distillation-fusion-flash-attention-speculative-decoding": "pruning",
    "pruning-distillation-fusion-speculation": "pruning",
    "pruning-distillation-fusion-speculative-decoding": "pruning",
    "pruning-fusion-edge": "pruning",
    "pruning-unstructured-sparsity": "pruning",
    "pruning-vs-distillation": "pruning",
    "structured-pruning-for-edge": "pruning",
    "structured-vs-unstructured-pruning": "pruning",
    "tensorrt-pruning": "pruning",

    # --- quantization ---
    "edge-quantization-memory": "quantization",
    "int8-quantization": "quantization",
    "mobile-quantization-calibration": "quantization",
    "mobile-quantization-memory": "quantization",
    "mobile-quantization-overflow": "quantization",
    "model-quantization-energy": "quantization",
    "quantization-activation-memory": "quantization",
    "quantization-arithmetic": "quantization",
    "quantization-calibration": "quantization",
    "quantization-calibration-auto": "quantization",
    "quantization-calibration-drift": "quantization",
    "quantization-calibration-error": "quantization",
    "quantization-calibration-failure": "quantization",
    "quantization-calibration-overflow": "quantization",
    "quantization-energy": "quantization",
    "quantization-energy-cost": "quantization",
    "quantization-energy-edge": "quantization",
    "quantization-energy-mobile": "quantization",
    "quantization-energy-ratio": "quantization",
    "quantization-energy-savings": "quantization",
    "quantization-energy-tinyml": "quantization",
    "quantization-energy-tradeoff": "quantization",
    "quantization-failure": "quantization",
    "quantization-failure-modes": "quantization",
    "quantization-fleet-automation": "quantization",
    "quantization-memory-bandwidth": "quantization",
    "quantization-memory-footprint": "quantization",
    "quantization-memory-savings": "quantization",
    "quantization-memory-tradeoff": "quantization",
    "quantization-mixed-precision-overflow": "quantization",
    "quantization-mixtured-of-experts-overflow": "quantization",
    "quantization-overflow": "quantization",
    "quantization-overflow-automotive": "quantization",
    "quantization-overflow-calibration": "quantization",
    "quantization-overflow-calibration-auto": "quantization",
    "quantization-overflow-mixed-precision": "quantization",
    "quantization-overflow-mobile": "quantization",
    "quantization-overflow-safety": "quantization",
    "quantization-overflow-tinyml": "quantization",
    "quantization-performance": "quantization",
    "quantization-safety-critical": "quantization",
    "requantization-arithmetic": "quantization",
    "tinyml-quantization-calibration": "quantization",
    "tinyml-quantization-overflow": "quantization",

    # --- queueing ---
    "edge-real-time-queueing": "queueing",
    "head-of-line-blocking-qos": "queueing",
    "inference-queueing-theory": "queueing",
    "llm-serving-architecture-queueing": "queueing",
    "llm-serving-queueing": "queueing",
    "llm-serving-queueing-theory": "queueing",
    "mobile-llm-latency-queueing": "queueing",
    "mobile-llm-queueing": "queueing",
    "mobile-queueing-theory-real-time": "queueing",
    "mobile-realtime-queueing": "queueing",
    "on-device-latency-queueing": "queueing",
    "preemption-qos-scheduling": "queueing",
    "queueing-rate-limiting": "queueing",
    "queueing-theory-and-slos": "queueing",
    "queueing-theory-deadlines": "queueing",
    "queueing-theory-deadlines-load-shedding": "queueing",
    "queueing-theory-inference": "queueing",
    "queueing-theory-latency": "queueing",
    "queueing-theory-littles-law": "queueing",
    "queueing-theory-mobile": "queueing",
    "queueing-theory-preemption-sla": "queueing",
    "queueing-theory-real-time": "queueing",
    "queueing-theory-realtime": "queueing",
    "queueing-theory-sla": "queueing",
    "queueing-theory-slo": "queueing",
    "queueing-theory-stability": "queueing",
    "queueing-theory-tail-latency": "queueing",
    "real-time-inference-queueing": "queueing",
    "real-time-queueing": "queueing",
    "real-time-queueing-theory": "queueing",
    "real-time-scheduling-queueing-theory": "queueing",
    "serving-architecture-queueing-theory": "queueing",
    "serving-latency-queueing": "queueing",
    "serving-queueing-theory-sla": "queueing",
    "throughput-queueing": "queueing",
    "tinyml-real-time-queueing": "queueing",
    "tinyml-realtime-queueing": "queueing",

    # --- rag ---
    "latency-rag-network": "rag",
    "rag-latency": "rag",
    "rag-latency-bottleneck": "rag",
    "rag-latency-guardrails": "rag",
    "rag-latency-serving": "rag",
    "rag-memory-calculation": "rag",
    "rag-memory-footprint": "rag",
    "rag-operations": "rag",
    "rag-retrieval-latency": "rag",
    "rag-rollout-memory": "rag",
    "rag-rollout-storage": "rag",
    "rag-update-cost": "rag",
    "rag-update-memory": "rag",
    "retrieval-augmented-generation": "rag",

    # --- rdma ---
    "infiniband-rdma": "rdma",
    "interconnect-arbitration-rdma": "rdma",
    "kernel-bypass-rdma": "rdma",
    "nvlink-vs-infiniband-pcie-network-topology-rdma-bus-protocols": "rdma",
    "pcie-rdma-bottleneck": "rdma",

    # --- real-time ---
    "anr-timeout": "real-time",
    "anr-timeout-analysis": "real-time",
    "edge-real-time-batching": "real-time",
    "frame-deadline-calculation": "real-time",
    "jank-budget-analysis": "real-time",
    "mobile-jank-budget": "real-time",
    "mobile-llm-realtime-deadline": "real-time",
    "mobile-real-time-batching": "real-time",
    "mobile-real-time-deadline": "real-time",
    "mobile-realtime-batching": "real-time",
    "mobile-realtime-latency": "real-time",
    "mobile-realtime-llm": "real-time",
    "mobile-realtime-llm-scheduling": "real-time",
    "mobile-realtime-throughput": "real-time",
    "real-time-batching": "real-time",
    "real-time-deadline": "real-time",
    "real-time-deadline-calculation": "real-time",
    "real-time-deadline-contention": "real-time",
    "real-time-deadline-jitter": "real-time",
    "real-time-deadlines": "real-time",
    "real-time-deadlines-batching": "real-time",
    "real-time-deadlines-prefill": "real-time",
    "real-time-frame-budget": "real-time",
    "real-time-inference": "real-time",
    "real-time-inference-batching": "real-time",
    "real-time-inference-deadline": "real-time",
    "real-time-inference-scheduling": "real-time",
    "real-time-interrupt-latency": "real-time",
    "real-time-interrupts": "real-time",
    "real-time-llm-metrics": "real-time",
    "real-time-llm-scheduling": "real-time",
    "real-time-pipeline-saturation": "real-time",
    "real-time-scheduling": "real-time",
    "real-time-scheduling-batching": "real-time",
    "real-time-scheduling-fifo-edf": "real-time",
    "real-time-scheduling-tinyml": "real-time",
    "rtos-priority-inversion-watchdog": "real-time",
    "tinyml-latency-budget": "real-time",
    "ui-jank-budget": "real-time",
    "watchdog-realtime-selftest": "real-time",
    "worst-case-execution-time": "real-time",

    # --- rollout ---
    "model-rollout-orchestration": "rollout",
    "rollout-resource-planning": "rollout",
    "model-serving-rollout": "rollout",

    # --- roofline ---
    "accelerator-efficiency": "roofline",
    "compute-analysis": "roofline",
    "compute-efficiency": "roofline",
    "compute-efficiency-topsw": "roofline",
    "compute-throughput-analysis": "roofline",
    "edge-compute-efficiency": "roofline",
    "edge-compute-limits": "roofline",
    "edge-efficiency-metric": "roofline",
    "edge-roofline": "roofline",
    "edge-roofline-analysis": "roofline",
    "edge-roofline-bottleneck": "roofline",
    "edge-roofline-diagnosis": "roofline",
    "edge-roofline-efficiency": "roofline",
    "edge-roofline-power-thermal": "roofline",
    "gpu-roofline": "roofline",
    "gpu-roofline-analysis": "roofline",
    "gpu-roofline-architecture": "roofline",
    "gpu-roofline-arithmetic-intensity": "roofline",
    "gpu-roofline-arithmetic-intensity-compute-bound-vs-memory-bound-tops-w": "roofline",
    "gpu-roofline-arithmetic-intensity-compute-bound-vs-memory-bound-topsw": "roofline",
    "gpu-roofline-arithmetic-intensity-edge-compute": "roofline",
    "gpu-roofline-arithmetic-intensity-edge-power": "roofline",
    "gpu-roofline-arithmetic-intensity-power": "roofline",
    "gpu-roofline-basics": "roofline",
    "gpu-roofline-batching": "roofline",
    "gpu-roofline-bottleneck": "roofline",
    "gpu-roofline-economics": "roofline",
    "gpu-roofline-edge": "roofline",
    "gpu-roofline-efficiency": "roofline",
    "gpu-roofline-heterogeneous-compute": "roofline",
    "gpu-roofline-intensity": "roofline",
    "gpu-roofline-memory-bound": "roofline",
    "gpu-roofline-model": "roofline",
    "gpu-roofline-optimization": "roofline",
    "gpu-roofline-power-efficiency": "roofline",
    "gpu-roofline-ridge-point": "roofline",
    "gpu-roofline-scheduling": "roofline",
    "gpu-roofline-tco": "roofline",
    "hardware-throughput": "roofline",
    "hardware-utilization-cnn-vs-vit": "roofline",
    "mcu-roofline": "roofline",
    "mcu-roofline-analysis": "roofline",
    "mobile-roofline-analysis": "roofline",
    "mobile-roofline-arithmetic-intensity": "roofline",
    "mobile-roofline-intensity": "roofline",
    "real-time-compute": "roofline",
    "real-time-compute-analysis": "roofline",
    "ridge-point-calculation": "roofline",
    "roofline-analysis": "roofline",
    "roofline-analysis-automotive": "roofline",
    "roofline-analysis-edge": "roofline",
    "roofline-analysis-power-efficiency": "roofline",
    "roofline-analysis-procurement": "roofline",
    "roofline-and-arithmetic-intensity": "roofline",
    "roofline-architecture-power": "roofline",
    "roofline-arithmetic-intensity": "roofline",
    "roofline-arithmetic-intensity-edge": "roofline",
    "roofline-co-design": "roofline",
    "roofline-edge-efficiency": "roofline",
    "roofline-intensity-tinyml": "roofline",
    "roofline-memory-bound": "roofline",
    "roofline-memory-bound-edge": "roofline",
    "roofline-model": "roofline",
    "roofline-model-edge": "roofline",
    "roofline-model-edge-power": "roofline",
    "roofline-model-tinyml": "roofline",
    "roofline-on-edge-accelerators": "roofline",
    "roofline-thermal-throttling-edge": "roofline",
    "roofline-tinyml": "roofline",
    "roofline-tinyml-bound": "roofline",
    "thermal-throttling-ridge-point": "roofline",
    "tinyml-ridge-point": "roofline",
    "tinyml-roofline": "roofline",
    "tinyml-roofline-analysis": "roofline",
    "tinyml-roofline-bottleneck": "roofline",
    "tinyml-roofline-intensity": "roofline",
    "tinyml-roofline-limit": "roofline",
    "tinyml-roofline-model": "roofline",
    "ttft-tpot-tradeoff-roofline": "roofline",

    # --- scaling-laws ---
    "architectural-scaling-laws": "scaling-laws",
    "cnn-scaling-laws": "scaling-laws",
    "mobile-scaling-laws": "scaling-laws",
    "moe-scaling-laws": "scaling-laws",
    "scaling-laws-cnn-transformer": "scaling-laws",
    "scaling-laws-cnn-transformer-moe-nas": "scaling-laws",
    "scaling-laws-cnn-vs-transformer": "scaling-laws",
    "scaling-laws-cnn-vs-transformer-depthwise-separable-moe-nas": "scaling-laws",
    "scaling-laws-transformer-architecture-cost": "scaling-laws",

    # --- security ---
    "ab-testing-security-tinyml": "security",
    "physical-tampering": "security",
    "physical-tampering-defense": "security",
    "side-channel-attacks": "security",

    # --- sensor-pipeline ---
    "sensor-bandwidth": "sensor-pipeline",
    "sensor-pipeline-bandwidth": "sensor-pipeline",
    "sensor-pipeline-bottleneck": "sensor-pipeline",
    "sensor-pipeline-skew": "sensor-pipeline",

    # --- serving ---
    "edge-generative-ai-stack": "serving",
    "inference-serving-latency": "serving",
    "inference-serving-tradeoffs": "serving",
    "inference-sla-prioritization": "serving",
    "latency-serving": "serving",
    "llm-inference-acceleration": "serving",
    "llm-inference-avionics": "serving",
    "llm-inference-latency": "serving",
    "llm-inference-serving": "serving",
    "llm-optimization-stack": "serving",
    "llm-serving-bottlenecks": "serving",
    "llm-serving-latency": "serving",
    "llm-serving-latency-throughput": "serving",
    "llm-serving-memory": "serving",
    "llm-serving-throughput": "serving",
    "mobile-llm-optimization": "serving",
    "mobile-serving-contention": "serving",
    "model-serving": "serving",
    "model-serving-economics": "serving",
    "model-serving-footprint": "serving",
    "model-serving-latency": "serving",
    "model-serving-memory": "serving",
    "on-device-serving-latency": "serving",
    "serving-architecture": "serving",
    "serving-architecture-sla": "serving",
    "serving-batching-latency": "serving",
    "serving-latency-batching": "serving",
    "serving-latency-tradeoffs": "serving",
    "serving-optimization-strategy": "serving",
    "serving-queue-slos": "serving",
    "serving-rollout-capacity": "serving",
    "unified-serving-architecture": "serving",

    # --- simd ---
    "cmsis-nn-simd": "simd",
    "cmsis-nn-simd-utilization": "simd",
    "integer-simd-speedup": "simd",
    "mcu-mac-budget": "simd",

    # --- speculative-decoding ---
    "flashattention-speculative-decoding": "speculative-decoding",
    "speculative-decoding-attention": "speculative-decoding",
    "speculative-decoding-fusion": "speculative-decoding",
    "speculative-decoding-overhead": "speculative-decoding",
    "speculative-decoding-throughput-collapse": "speculative-decoding",
    "speculative-decoding-tradeoffs": "speculative-decoding",

    # --- thermal ---
    "edge-thermal-limit": "thermal",
    "gpu-power-thermal": "thermal",
    "power-thermal-throttling": "thermal",
    "sustained-performance-vs-peak": "thermal",
    "thermal-and-power": "thermal",
    "thermal-cooling-power": "thermal",
    "thermal-power-management": "thermal",
    "thermal-throttling-energy-harvesting": "thermal",
    "thermal-throttling-impact": "thermal",
    "tinyml-thermal-leakage": "thermal",

    # --- throughput ---
    "latency-throughput": "throughput",
    "latency-throughput-tradeoff": "throughput",
    "mobile-latency-throughput-tradeoff": "throughput",
    "mobile-llm-throughput": "throughput",
    "throughput-vs-goodput": "throughput",

    # --- tops-w ---
    "edge-power-efficiency": "tops-w",
    "power-efficiency-tops-w": "tops-w",
    "tops-per-watt": "tops-w",
    "tops-per-watt-cluster-design": "tops-w",
    "tops-per-watt-efficiency": "tops-w",
    "tops-per-watt-tco": "tops-w",
    "tops-vs-efficiency": "tops-w",

    # --- tpot ---
    "real-time-tpot-deadline": "tpot",
    "tpot-memory-bound": "tpot",
    "tpot-memory-bound-mobile": "tpot",
    "tpot-vs-ttft-batching": "tpot",

    # --- ttft ---
    "llm-inference-ttft": "ttft",
    "mobile-ttft-latency": "ttft",
    "mobile-ttft-vs-tpot": "ttft",
    "queueing-theory-tpot-ttft": "ttft",
    "queueing-theory-ttft": "ttft",
    "ttft-memory-bound": "ttft",
    "ttft-prefill-bottleneck": "ttft",
    "ttft-tpot-queueing": "ttft",
    "ttft-tpot-tradeoff": "ttft",
    "ttft-vs-tpot": "ttft",

    # --- watchdog ---
    "degradation-ladder-watchdog": "watchdog",
    "monitoring-data-drift-watchdog": "watchdog",
    "watchdog-timer-fusa": "watchdog",
    "watchdog-timers": "watchdog",
}


# Build case-insensitive lookup
_NORMALIZE_MAP_LOWER: dict[str, str] = {k.lower(): v for k, v in NORMALIZE_MAP.items()}


def normalize_tag(tag: str) -> str:
    """Normalize a freeform tag to its canonical form.

    Returns the tag unchanged if it's already canonical,
    maps it if there's a known normalization, or returns
    the closest canonical tag. Case-insensitive.
    """
    tag = tag.lower().strip()

    # Already canonical
    if tag in ALL_TAGS:
        return tag

    # Known mapping (case-insensitive)
    if tag in _NORMALIZE_MAP_LOWER:
        return _NORMALIZE_MAP_LOWER[tag]

    # Unknown — return as-is (will be flagged by validation)
    return tag


def validate_tags(tags: list[str]) -> tuple[list[str], list[str]]:
    """Validate a list of tags against the taxonomy.

    Returns (valid_tags, invalid_tags).
    """
    valid = []
    invalid = []
    for tag in tags:
        normalized = normalize_tag(tag)
        if normalized in ALL_TAGS:
            valid.append(normalized)
        else:
            invalid.append(tag)
    return valid, invalid


# ---------------------------------------------------------------------------
# Weighted target matrix: how many questions each cell deserves
# ---------------------------------------------------------------------------
# Not all cells are equal. cloud/compute/L5 is a core Staff interview topic;
# tinyml/parallelism/L6+ is a question that doesn't exist in the real world.
#
# Weights: 5 = core, 3 = important, 1 = nice-to-have, 0 = skip
# Applied per level: {competency: [L1, L2, L3, L4, L5, L6+]}

_CORE = [2, 2, 5, 5, 5, 3]       # Core competency for this track
_IMPORTANT = [1, 1, 3, 3, 3, 2]  # Important but not the defining topic
_MINOR = [1, 1, 2, 2, 1, 1]      # Present but not heavily tested
_SPARSE = [0, 0, 1, 1, 1, 0]     # Barely applies to this track
_SKIP = [0, 0, 0, 0, 0, 0]       # Doesn't apply

CELL_TARGETS: dict[str, dict[str, list[int]]] = {
    "cloud": {
        "compute": _CORE,           # Roofline is THE cloud interview question
        "memory": _CORE,            # VRAM accounting, KV-cache
        "precision": _IMPORTANT,    # Mixed precision, FP8
        "architecture": _IMPORTANT, # Scaling laws, transformers
        "latency": _CORE,           # TTFT/TPOT, continuous batching
        "power": _MINOR,            # TDP, PUE — important but less tested
        "optimization": _IMPORTANT, # FlashAttention, speculative decoding
        "parallelism": _CORE,       # DP/TP/PP — core Staff topic
        "networking": _IMPORTANT,   # InfiniBand, NVLink
        "deployment": _IMPORTANT,   # Serving, K8s, rollout
        "reliability": _IMPORTANT,  # Checkpointing, fault tolerance
        "data": _IMPORTANT,         # Feature stores, pipelines (bumped per Huyen review)
        "cross-cutting": _MINOR,    # Security, economics
    },
    "edge": {
        "compute": _CORE,           # TOPS/W, roofline on edge
        "memory": _CORE,            # DRAM budgets, DMA
        "precision": _IMPORTANT,    # INT8, calibration
        "architecture": _IMPORTANT, # CNN vs ViT, model selection
        "latency": _CORE,           # Real-time deadlines, WCET
        "power": _CORE,             # Thermal envelope — THE edge constraint
        "optimization": _IMPORTANT, # TensorRT, pruning
        "parallelism": _SPARSE,     # Barely applies on edge
        "networking": _SPARSE,      # Bus protocols, not InfiniBand
        "deployment": _IMPORTANT,   # OTA, firmware, fleet
        "reliability": _IMPORTANT,  # Watchdog, degradation ladders
        "data": _IMPORTANT,         # Sensor pipeline, calibration
        "cross-cutting": _MINOR,    # Security, adversarial
    },
    "mobile": {
        "compute": _IMPORTANT,      # NPU delegation, heterogeneous
        "memory": _CORE,            # Shared RAM, app eviction — core mobile
        "precision": _IMPORTANT,    # Float16/INT8 on NPU
        "architecture": _IMPORTANT, # MobileNet, on-device LLM
        "latency": _CORE,           # 16ms jank budget, ANR
        "power": _CORE,             # Battery drain — THE mobile constraint
        "optimization": _IMPORTANT, # CoreML, TFLite fusion
        "parallelism": _SPARSE,     # Barely applies on mobile
        "networking": _SPARSE,      # Cellular download, not fabric
        "deployment": _CORE,        # App store, model download, A/B test
        "reliability": _MINOR,      # Crash reporting, silent degradation
        "data": _MINOR,             # Federated analytics
        "cross-cutting": _IMPORTANT, # Privacy, federated learning
    },
    "tinyml": {
        "compute": _CORE,           # CMSIS-NN, SIMD, MAC budgets
        "memory": _CORE,            # SRAM, tensor arena — THE tinyml topic
        "precision": _IMPORTANT,    # INT8 zero-point, requantization
        "architecture": _IMPORTANT, # Depthwise separable, NAS
        "latency": _IMPORTANT,      # Microsecond inference, interrupts
        "power": _CORE,             # Duty cycling, energy harvesting
        "optimization": _MINOR,     # Operator scheduling
        "parallelism": _SKIP,       # Doesn't exist on MCUs
        "networking": _SPARSE,      # BLE, LoRaWAN — lightweight
        "deployment": _IMPORTANT,   # FOTA, flash programming
        "reliability": _IMPORTANT,  # Watchdog, hard real-time
        "data": _IMPORTANT,         # Sensor pipeline, signal processing
        "cross-cutting": _MINOR,    # Side-channel, model extraction
    },
}

LEVELS_LIST = ["L1", "L2", "L3", "L4", "L5", "L6+"]


def get_cell_target(track: str, competency: str, level: str) -> int:
    """Get the target question count for a specific 3D cell."""
    track_targets = CELL_TARGETS.get(track, {})
    level_targets = track_targets.get(competency, _SKIP)
    level_idx = LEVELS_LIST.index(level) if level in LEVELS_LIST else 0
    return level_targets[level_idx]
