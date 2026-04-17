"""Pydantic schema for StaffML interview question corpus.

Validates corpus.json against the gold-standard classification system:
  - 79 curated topics (WHAT concept is tested)
  - 11 ikigai zones (HOW the concept is tested)
  - 5 tracks (WHERE it's deployed)
  - 6 levels (HOW HARD it is)
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

VALID_TRACKS = {"cloud", "edge", "mobile", "tinyml", "global"}
VALID_LEVELS = {"L1", "L2", "L3", "L4", "L5", "L6+"}
VALID_AREAS = {
    "compute", "memory", "latency", "precision", "power",
    "architecture", "optimization", "parallelism", "networking",
    "deployment", "reliability", "data", "cross-cutting",
}
VALID_BLOOM = {"remember", "understand", "apply", "analyze", "evaluate", "create", ""}
VALID_STATUS = {"published", "draft", "archived", "flagged"}

# --- v6.0 Gold Standard Classification ---

VALID_ZONES = {
    # Pure zones (single skill)
    "recall", "analyze", "design", "implement",
    # Compound zones (two skills)
    "diagnosis", "specification", "fluency",
    "evaluation", "realization", "optimization",
    # Mastery (all four skills)
    "mastery",
}

VALID_TOPICS = {
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
}

VALID_REASONING_MODES = {
    "concept-recall",
    "napkin-math",
    "symptom-to-cause",
    "tradeoff-analysis",
    "requirements-to-architecture",
    "optimization-task",
    "failure-to-root-cause",
}


class Resource(BaseModel):
    """Author-curated external reference attached to a question.

    Replaces the singular deep_dive_title/deep_dive_url pair. Questions may
    carry zero, one, or many resources in Details.resources (display order =
    list order). The `name` is author-written prose so the UI labels each
    link without hostname-based classification.
    """

    name: str
    url: str

    @field_validator("name")
    @classmethod
    def name_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Resource.name must be non-empty")
        if len(v) > 200:
            raise ValueError(f"Resource.name too long ({len(v)} chars, max 200)")
        return v

    @field_validator("url")
    @classmethod
    def url_is_https(cls, v: str) -> str:
        # XSS defense (REVIEWS.md H-6): allowlist https:// only.
        # Rejects javascript:, data:, http:, relative paths.
        if not v.startswith("https://"):
            raise ValueError(f"Resource.url must start with https:// (got: {v[:40]!r})")
        return v


class QuestionDetails(BaseModel):
    common_mistake: str
    realistic_solution: str
    napkin_math: str = ""
    resources: list[Resource] = []
    options: Optional[list[str]] = None
    correct_index: Optional[int] = None

    @field_validator("common_mistake")
    @classmethod
    def common_mistake_min_length(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError(f"common_mistake too short ({len(v)} chars, min 10)")
        return v

    @field_validator("realistic_solution")
    @classmethod
    def realistic_solution_min_length(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError(f"realistic_solution too short ({len(v)} chars, min 5)")
        return v

    @field_validator("napkin_math")
    @classmethod
    def napkin_math_has_substance(cls, v: str) -> str:
        # Only flag if napkin_math is present but suspiciously empty
        # Multi-line extractions may have prose intros without digits — that's OK
        if v and 0 < len(v.strip()) < 5:
            raise ValueError(f"napkin_math too short ({len(v)} chars)")
        return v

    @model_validator(mode="after")
    def mcq_consistency(self) -> "QuestionDetails":
        if self.options is not None:
            if len(self.options) != 4:
                raise ValueError(f"MCQ must have exactly 4 options, got {len(self.options)}")
            if self.correct_index is None:
                raise ValueError("MCQ has options but missing correct_index")
            if not (0 <= self.correct_index <= 3):
                raise ValueError(f"correct_index must be 0-3, got {self.correct_index}")
        return self


class Question(BaseModel):
    # Identity
    id: str
    track: str
    scope: str = ""
    level: str
    title: str

    # Gold standard classification (v6.0)
    topic: str                                    # One of 79 curated topic IDs
    zone: str                                     # One of 11 ikigai zones
    competency_area: str                          # One of 13 canonical areas
    bloom_level: str = ""

    # Content
    scenario: str
    details: QuestionDetails

    # Validation (stamped by Gemini review)
    validated: Optional[bool] = None
    validation_status: Optional[str] = None
    validation_issues: Optional[list[str]] = None
    validation_model: Optional[str] = None
    validation_date: Optional[str] = None

    # Chains
    chain_ids: Optional[list[str]] = None
    chain_positions: Optional[dict[str, int]] = None

    @field_validator("track")
    @classmethod
    def valid_track(cls, v: str) -> str:
        if v not in VALID_TRACKS:
            raise ValueError(f"Invalid track '{v}', must be one of {VALID_TRACKS}")
        return v

    @field_validator("level")
    @classmethod
    def valid_level(cls, v: str) -> str:
        if v not in VALID_LEVELS:
            raise ValueError(f"Invalid level '{v}', must be one of {VALID_LEVELS}")
        return v

    @field_validator("topic")
    @classmethod
    def valid_topic(cls, v: str) -> str:
        if v not in VALID_TOPICS:
            raise ValueError(f"Invalid topic '{v}', must be one of VALID_TOPICS ({len(VALID_TOPICS)} topics)")
        return v

    @field_validator("zone")
    @classmethod
    def valid_zone(cls, v: str) -> str:
        if v not in VALID_ZONES:
            raise ValueError(f"Invalid zone '{v}', must be one of {VALID_ZONES}")
        return v

    @field_validator("competency_area")
    @classmethod
    def valid_area(cls, v: str) -> str:
        if v not in VALID_AREAS:
            raise ValueError(f"Invalid competency_area '{v}', must be one of {VALID_AREAS}")
        return v

    @field_validator("bloom_level")
    @classmethod
    def valid_bloom(cls, v: str) -> str:
        if v and v not in VALID_BLOOM:
            raise ValueError(f"Invalid bloom_level '{v}', must be one of {VALID_BLOOM}")
        return v

    @field_validator("title")
    @classmethod
    def title_min_length(cls, v: str) -> str:
        if len(v.strip()) < 3:
            raise ValueError(f"title too short ({len(v)} chars, min 3)")
        return v

    @field_validator("scenario")
    @classmethod
    def scenario_quality(cls, v: str) -> str:
        if len(v.strip()) < 30:
            raise ValueError(f"scenario too short ({len(v)} chars, min 30)")
        if v.strip().startswith('"') or v.strip().endswith('"'):
            raise ValueError("scenario has stray quotes — should be clean text")
        return v


def validate_corpus(questions: list[dict]) -> tuple[list["Question"], list[str], list[str]]:
    """Validate a list of question dicts against the schema.

    Returns (valid_questions, errors, warnings).
    """
    valid = []
    errors = []

    for i, q_dict in enumerate(questions):
        try:
            q = Question(**q_dict)
            valid.append(q)
        except Exception as e:
            qid = q_dict.get("id", f"index-{i}")
            errors.append(f"[{qid}] {e}")

    # Cross-question checks
    id_counts: dict[str, int] = {}
    for q in valid:
        id_counts[q.id] = id_counts.get(q.id, 0) + 1
    for qid, count in id_counts.items():
        if count > 1:
            errors.append(f"Duplicate ID: '{qid}' appears {count} times")

    # Duplicate titles — warnings only
    seen_titles: dict[tuple, str] = {}
    warnings = []
    for q in valid:
        key = (q.track, q.level, q.title)
        if key in seen_titles:
            warnings.append(
                f"Duplicate title: '{q.title}' in {q.track}/{q.level} "
                f"(IDs: {seen_titles[key]}, {q.id})"
            )
        else:
            seen_titles[key] = q.id

    return valid, errors, warnings
