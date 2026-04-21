"""Single source of truth for Python enum values.

These values mirror ``question_schema.yaml`` (LinkML). That file is the
canonical schema; this module exists only to provide Python-importable
constants for validators and typed models that can't read LinkML directly.

Any change here MUST be mirrored in question_schema.yaml and vice versa.
A CI drift check verifies the two stay in sync (see tools/check_schema_sync.py).

Schema version: 1.0
"""

from __future__ import annotations

# ─── 4-axis classification ──────────────────────────────────────────────────

VALID_TRACKS: frozenset[str] = frozenset({
    "tinyml", "edge", "mobile", "cloud", "global",
})

VALID_LEVELS: frozenset[str] = frozenset({
    "L1", "L2", "L3", "L4", "L5", "L6+",
})

VALID_ZONES: frozenset[str] = frozenset({
    # Pure zones (single skill)
    "recall", "analyze", "design", "implement",
    # Compound zones (two skills)
    "fluency", "diagnosis", "specification",
    "optimization", "evaluation", "realization",
    # Mastery (all four skills)
    "mastery",
})

VALID_BLOOM_LEVELS: frozenset[str] = frozenset({
    "remember", "understand", "apply", "analyze", "evaluate", "create",
})

VALID_PHASES: frozenset[str] = frozenset({
    "training", "inference", "both",
})

VALID_STATUSES: frozenset[str] = frozenset({
    "draft",       # authored but not yet ready for users
    "published",   # live in the corpus
    "flagged",     # under review; surface to authors not users
    "archived",    # retired but content preserved for history
    "deleted",     # soft-deleted; paired with deletion_reason field
})

VALID_PROVENANCES: frozenset[str] = frozenset({
    "human", "llm-draft", "llm-then-human-edited", "imported",
})

VALID_HUMAN_REVIEW_STATUSES: frozenset[str] = frozenset({
    "not-reviewed", "verified", "flagged", "needs-rework",
})

# ─── Competency areas (paper §4) ────────────────────────────────────────────

VALID_COMPETENCY_AREAS: frozenset[str] = frozenset({
    "compute", "memory", "latency", "precision", "power",
    "architecture", "optimization", "parallelism", "networking",
    "deployment", "reliability", "data", "cross-cutting",
})

# ─── Curated topics (paper §4; 87 as of v1.0) ───────────────────────────────
# The original paper specified 79; v1.0 adds 8 topics that already had
# substantive corpus coverage but were missing from the curated list:
# autograd-computational-graphs, chiplet-architecture,
# communication-computation-overlap, disaggregated-serving,
# model-adaptation-systems, recommendation-systems-engineering,
# software-portability, sustainability-carbon-accounting.

VALID_TOPICS: frozenset[str] = frozenset({
    # compute (6)
    "roofline-analysis", "gpu-compute-architecture", "accelerator-comparison",
    "mcu-compute-constraints", "systolic-dataflow", "compute-cost-estimation",
    # memory (8)
    "vram-budgeting", "kv-cache-management", "memory-hierarchy-design",
    "activation-memory", "memory-mapped-inference", "tensor-arena-planning",
    "dma-data-movement", "memory-pressure-management",
    # latency (6)
    "latency-decomposition", "batching-strategies", "tail-latency",
    "real-time-deadlines", "profiling-bottleneck-analysis", "queueing-theory",
    # precision (3)
    "quantization-fundamentals", "mixed-precision-training", "extreme-quantization",
    # power (5)
    "power-budgeting", "thermal-management", "energy-per-operation",
    "duty-cycling", "datacenter-efficiency",
    # architecture (7)
    "transformer-systems-cost", "cnn-efficient-design", "attention-scaling",
    "mixture-of-experts", "model-size-estimation", "neural-architecture-search",
    "encoder-decoder-tradeoffs",
    # optimization (7)
    "pruning-sparsity", "knowledge-distillation", "kernel-fusion",
    "graph-compilation", "operator-scheduling", "flash-attention",
    "speculative-decoding",
    # parallelism (6)
    "data-parallelism", "model-tensor-parallelism", "pipeline-parallelism",
    "3d-parallelism", "gradient-synchronization", "scheduling-resource-management",
    # networking (6)
    "collective-communication", "interconnect-topology",
    "network-bandwidth-bottlenecks", "rdma-transport", "load-balancing",
    "congestion-control",
    # deployment (7)
    "model-serving-infrastructure", "mlops-lifecycle", "ota-firmware-updates",
    "container-orchestration", "model-format-conversion", "ab-rollout-strategies",
    "compound-ai-systems",
    # reliability (6)
    "fault-tolerance-checkpointing", "distribution-drift-detection",
    "graceful-degradation", "safety-certification", "adversarial-robustness",
    "monitoring-observability",
    # data (7)
    "data-pipeline-engineering", "feature-store-management",
    "data-quality-validation", "dataset-curation", "streaming-ingestion",
    "storage-format-selection", "data-efficiency-selection",
    # cross-cutting (5)
    "federated-learning", "differential-privacy", "fairness-evaluation",
    "responsible-ai", "tco-cost-modeling",
    # v1.0 additions (8)
    "autograd-computational-graphs", "chiplet-architecture",
    "communication-computation-overlap", "disaggregated-serving",
    "model-adaptation-systems", "recommendation-systems-engineering",
    "software-portability", "sustainability-carbon-accounting",
})


# ─── Zone-level affinity (paper §3.3 Table 2) ───────────────────────────────
# Each zone has a natural level range. A question outside this range is not
# necessarily wrong but should be reviewed. Used by `vault lint` to emit
# soft-constraint warnings (paper line 397).

ZONE_LEVEL_AFFINITY: dict[str, set[str]] = {
    "recall":         {"L1", "L2"},
    "fluency":        {"L2", "L3"},
    "analyze":        {"L3", "L4"},
    "diagnosis":      {"L3", "L4"},
    "design":         {"L4", "L5"},
    "specification":  {"L4", "L5"},
    "optimization":   {"L4", "L5"},
    "evaluation":     {"L5", "L6+"},
    "realization":    {"L5", "L6+"},
    "mastery":        {"L6+"},
    "implement":      {"L2", "L3", "L4"},  # paper treats this as broadly-ranged
}


__all__ = [
    "VALID_TRACKS",
    "VALID_LEVELS",
    "VALID_ZONES",
    "VALID_BLOOM_LEVELS",
    "VALID_PHASES",
    "VALID_STATUSES",
    "VALID_PROVENANCES",
    "VALID_HUMAN_REVIEW_STATUSES",
    "VALID_COMPETENCY_AREAS",
    "VALID_TOPICS",
    "ZONE_LEVEL_AFFINITY",
]
