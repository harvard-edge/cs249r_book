"""
ML Systems Interview Question Taxonomy

The canonical tag vocabulary for all StaffML questions. Every question gets
1-3 tags from this controlled set. No freestyling.

Structure:
- 10 primary competency areas (from TOPIC_MAP.md)
- Each area has 3-8 specific tags
- ~50 total canonical tags
- Tags are kebab-case, lowercase

This taxonomy reflects the enduring structure of ML systems engineering.
The competency areas are physics — they don't change with framework trends.
What changes across tracks is how each area manifests (GPU vs MCU vs NPU),
but the underlying questions are the same: where does the data live, how
fast can you move it, what's the bottleneck?
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
    # Old taxonomy → new taxonomy renames
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
    "os": "container-orchestration",
    "mobile-os": "container-orchestration",
    "mobile-frameworks": "compilation",
    "kubernetes": "container-orchestration",
    "tensorrt": "graph-optimization",

    # Compute
    "compute": "roofline",
    "cpu-gpu-arch": "heterogeneous-compute",
    "hardware-mac": "simd",
    "mflops": "roofline",

    # Memory
    "memory": "memory-hierarchy",
    "kv-cache-memory": "kv-cache",
    "kv-cache-management": "kv-cache",
    "memory-layout": "memory-hierarchy",
    "memory-management": "memory-hierarchy",
    "storage": "persistent-storage",
    "storage-io": "persistent-storage",
    "flash-memory": "persistent-storage",

    # Precision
    "precision": "mixed-precision",
    "quantization-memory": "quantization",
    "quantization-npu": "quantization",
    "tinyml-quantization": "quantization",

    # Architecture
    "architecture": "model-cost",
    "model-architecture": "model-cost",
    "attention-mechanisms": "attention",

    # Latency
    "serving": "serving",
    "continuous-batching": "batching",
    "queueing-theory": "queueing",

    # Power
    "power-thermal": "power",
    "thermal-management": "thermal",
    "power-management": "power",
    "energy": "power",

    # Optimization
    "model-compression": "pruning",
    "compression": "pruning",
    "compiler-runtime": "compilation",
    "compiler": "compilation",
    "frameworks": "compilation",
    "optimization": "operator-fusion",
    "npu-compiler": "compilation",

    # Deployment
    "mlops": "deployment",
    "ml-ops": "deployment",
    "model-deployment": "deployment",
    "canary-deployment": "rollout",

    # Reliability
    "model-monitoring": "monitoring",
    "reliability": "fault-tolerance",
    "functional-safety": "fault-tolerance",
    "incident-response": "incident-response",

    # Data
    "data-engineering": "data-pipeline",
    "data-versioning": "data-versioning",
    "data-management": "data-pipeline",
    "data-locality": "data-pipeline",
    "data-quality": "data-quality",

    # Cross-cutting
    "network-fabric": "interconnect",
    "networking": "interconnect",
    "network": "interconnect",
    "cloud-networking": "interconnect",
    "soc-interconnect": "interconnect",
    "distributed": "data-parallelism",
    "training": "data-parallelism",
    "pipeline": "data-pipeline",
    "sensors": "sensor-pipeline",
    "sensor": "sensor-pipeline",
    "sensor-fusion": "sensor-pipeline",
    "sensor-io": "sensor-pipeline",

    # Hardware-specific → nearest canonical
    "hardware": "model-cost",
    "edge-hardware": "model-cost",
    "cpu-architecture": "heterogeneous-compute",
    "tpu-architecture": "heterogeneous-compute",
    "npu-architecture": "heterogeneous-compute",
    "soc-architecture": "heterogeneous-compute",
    "hardware-acceleration": "roofline",
    "hw-acceleration": "roofline",

    # Misc
    "os": "deployment",
    "rtos-scheduling": "real-time",
    "scheduling": "real-time",
    "edge": "deployment",
    "benchmarking": "roofline",
    "debug-interface": "observability",
    "cost-optimization": "economics",
    "cost-attribution": "economics",

    # Remaining unmapped tags (from corpus audit)
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
    "model-size": "model-size",
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

    # Bulk normalization of single-use tags (109 remaining from corpus audit)
    # Memory variants
    "cache-hierarchy": "memory-hierarchy", "cache-coherence": "memory-hierarchy",
    "memory-technologies": "memory-hierarchy", "virtual-memory": "memory-hierarchy",
    "memory-alignment": "memory-hierarchy", "memory-architecture": "memory-hierarchy",
    "memory-capacity": "memory-hierarchy", "memory-footprint": "memory-hierarchy",
    "memory-power": "memory-hierarchy", "memory-mapped-io": "dma",
    "on-chip-memory": "memory-hierarchy", "scratchpad-memory": "memory-hierarchy",
    "fragmentation": "memory-hierarchy", "shared-bandwidth": "memory-bandwidth",
    "cpu-cache-memory-access": "memory-hierarchy", "numa-multicore": "memory-hierarchy",
    "NUMA": "memory-hierarchy", "dma-transfers": "dma", "mmio": "dma",
    "sparsity-memory": "memory-hierarchy",

    # Compute variants
    "compute-intensity": "arithmetic-intensity", "compute-overhead": "roofline",
    "compute-memory-bandwidth": "memory-bound", "branch-prediction": "roofline",
    "algorithms": "roofline", "math": "roofline",

    # Precision variants
    "dynamic-quantization": "quantization", "Extreme Quantization": "quantization",
    "quantization-hardware": "quantization", "quantization-robustness": "quantization",
    "quantization-low-precision-hardware-support": "quantization",
    "quantization-memory-deployment": "quantization",

    # Latency variants
    "cold-start-latency": "latency", "latency-budgets": "latency",
    "real-time-latency": "real-time", "real-time-systems": "real-time",
    "wcet": "real-time", "deterministic-timing": "real-time",
    "rtos": "real-time", "rtos-deterministic": "real-time",
    "timing": "real-time", "interrupt-vs-polling": "real-time",

    # Power variants
    "low-power": "power", "ultra-low-power": "power",
    "power-gating": "power", "voltage-scaling": "power",
    "dvfs": "power", "energy-harvesting": "duty-cycle",
    "thermal-design": "thermal", "thermal-power-sustained-perf": "thermal",
    "carbon-aware": "sustainable-ai",
    "power-efficiency-frameworks-android-ios": "power",
    "power-management-adaptive": "power",

    # Architecture / hardware variants
    "apple-silicon": "heterogeneous-compute", "mobile-gpu": "heterogeneous-compute",
    "micro-npu": "heterogeneous-compute", "soc": "heterogeneous-compute",
    "system-on-chip": "heterogeneous-compute", "tinyml-hardware": "heterogeneous-compute",
    "hardware-topology": "interconnect", "bus-arbitration": "interconnect",
    "clock-tree": "heterogeneous-compute", "hardware-lifecycle": "deployment",
    "custom-ops-heterogeneous-compute-vendor-sdk": "heterogeneous-compute",
    "heterogeneous-compute-scheduling-latency-power": "heterogeneous-compute",
    "heterogeneous-memory-coherence": "memory-hierarchy",

    # Deployment variants
    "containerization": "container-orchestration", "spot-instances": "economics",
    "model-delivery": "deployment", "model-formats": "deployment",
    "model-lifecycle": "deployment", "model-update": "ota",
    "model-optimization": "pruning", "mobile-frameworks": "compilation",
    "mobile-os": "deployment", "platform": "deployment",
    "health-checks": "monitoring", "long-term-reliability": "fault-tolerance",
    "testing": "monitoring", "profiling": "observability",
    "boot-sequence": "firmware", "boot-time": "firmware",
    "secure-boot": "firmware", "peripheral-timer": "real-time",

    # Data variants
    "data-collection": "data-pipeline", "data-loading": "data-pipeline",
    "data-privacy": "privacy", "data-provenance": "data-versioning",
    "reproducibility": "data-versioning",

    # Cross-cutting
    "Differential Privacy": "privacy", "fairness": "adversarial",
    "robustness": "adversarial", "on-device-training": "continual-learning",
    "multi-model-inference": "serving", "multi-tenant": "serving",
    "cross-region": "interconnect", "edge-cloud-sync": "interconnect",
    "zero-copy-pipeline": "dma", "sensor-calibration": "sensor-pipeline",
    "sensor-physics": "sensor-pipeline", "system-design": "model-cost",

    # Full-sentence tags from legacy data
    "Ad Click Aggregation & Streaming Data": "data-pipeline",
    "Real-Time Ranking & Retrieval": "serving",
    "Serving & Global Infrastructure": "serving",
    "LLM Inference & Memory Management": "kv-cache",
    "Multimodal Agentic OS": "deployment",
    "collective-design": "data-parallelism",
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
