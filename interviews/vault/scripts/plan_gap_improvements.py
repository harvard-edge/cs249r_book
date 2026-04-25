#!/usr/bin/env python3
"""Build StaffML gap-analysis and generation-planning artifacts.

This script implements the release-oriented improvement plan:

- canonical v1 coverage cube from YAML source,
- repair backlog for metadata/content drift,
- 50-question pilot pack,
- validation gates for generated items,
- scaled 250-500 item generation queue.

It does not generate questions. It produces deterministic planning artifacts
under ``_validation_results/gap_plan/`` that can be reviewed before generation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = VAULT_DIR.parents[1]
QUESTIONS_DIR = VAULT_DIR / "questions"
TAXONOMY_DATA = VAULT_DIR / "schema" / "taxonomy_data.yaml"
CHAINS_PATH = VAULT_DIR / "chains.json"
OUTPUT_DIR = VAULT_DIR / "_validation_results" / "gap_plan"

import sys

SCHEMA_DIR = VAULT_DIR / "schema"
if str(SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEMA_DIR))

from enums import (  # noqa: E402
    VALID_COMPETENCY_AREAS,
    VALID_LEVELS,
    VALID_PHASES,
    VALID_TOPICS,
    VALID_TRACKS,
    VALID_ZONES,
    ZONE_LEVEL_AFFINITY,
)

TRACKS = ["cloud", "edge", "mobile", "tinyml", "global"]
LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6+"]
PHASES = ["training", "inference", "both"]

PRIORITY_ZONES = ["design", "specification", "evaluation", "realization", "mastery"]
STAFF_LEVEL_BY_ZONE = {
    "design": "L5",
    "specification": "L5",
    "evaluation": "L5",
    "realization": "L5",
    "mastery": "L6+",
}

MLSYSIM_WALLS: dict[str, dict[str, Any]] = {
    "compute": {"topics": ["roofline-analysis", "gpu-compute-architecture"], "area": "compute"},
    "memory": {"topics": ["vram-budgeting", "memory-hierarchy-design"], "area": "memory"},
    "software": {"topics": ["kernel-fusion", "graph-compilation"], "area": "optimization"},
    "serving": {"topics": ["model-serving-infrastructure", "latency-decomposition"], "area": "deployment"},
    "batching": {"topics": ["batching-strategies", "kv-cache-management"], "area": "latency"},
    "streaming": {"topics": ["streaming-ingestion", "memory-mapped-inference"], "area": "data"},
    "tail-latency": {"topics": ["tail-latency", "queueing-theory"], "area": "latency"},
    "ingestion": {"topics": ["data-pipeline-engineering", "streaming-ingestion"], "area": "data"},
    "transformation": {"topics": ["data-pipeline-engineering", "operator-scheduling"], "area": "data"},
    "locality": {"topics": ["interconnect-topology", "network-bandwidth-bottlenecks"], "area": "networking"},
    "complexity": {"topics": ["compute-cost-estimation", "transformer-systems-cost"], "area": "compute"},
    "reasoning": {"topics": ["speculative-decoding", "transformer-systems-cost"], "area": "architecture"},
    "fidelity": {"topics": ["quantization-fundamentals", "pruning-sparsity"], "area": "precision"},
    "communication": {"topics": ["collective-communication", "gradient-synchronization"], "area": "networking"},
    "fragility": {"topics": ["fault-tolerance-checkpointing", "graceful-degradation"], "area": "reliability"},
    "multi-tenant": {"topics": ["scheduling-resource-management", "container-orchestration"], "area": "deployment"},
    "capital": {"topics": ["tco-cost-modeling", "compute-cost-estimation"], "area": "cross-cutting"},
    "sustainability": {"topics": ["sustainability-carbon-accounting", "datacenter-efficiency"], "area": "power"},
    "checkpoint": {"topics": ["fault-tolerance-checkpointing"], "area": "reliability"},
    "safety": {"topics": ["differential-privacy", "fairness-evaluation", "responsible-ai"], "area": "cross-cutting"},
    "sensitivity": {"topics": ["profiling-bottleneck-analysis", "roofline-analysis"], "area": "compute"},
    "synthesis": {"topics": ["latency-decomposition", "model-serving-infrastructure"], "area": "latency"},
}


@dataclass(frozen=True)
class Question:
    path: str
    id: str
    track: str
    level: str
    zone: str
    topic: str
    competency_area: str
    bloom_level: str
    phase: str
    title: str
    scenario: str
    question: str
    solution: str
    napkin_math: str
    status: str
    chain_ids: tuple[str, ...]
    human_review_status: str
    has_visual: bool
    has_options: bool
    has_napkin: bool
    is_incomplete_information: bool


def load_questions() -> list[Question]:
    questions: list[Question] = []
    for path in sorted(QUESTIONS_DIR.glob("*/*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        details = data.get("details") or {}
        tags = data.get("tags") or []
        chains = data.get("chains") or []
        chain_ids = tuple(c.get("id") for c in chains if isinstance(c, dict) and c.get("id"))
        human_review = data.get("human_reviewed") or {}
        questions.append(
            Question(
                path=str(path.relative_to(ROOT_DIR)),
                id=data.get("id", path.stem),
                track=data.get("track", ""),
                level=data.get("level", ""),
                zone=data.get("zone", ""),
                topic=data.get("topic", ""),
                competency_area=data.get("competency_area", ""),
                bloom_level=data.get("bloom_level", ""),
                phase=data.get("phase") or "both",
                title=data.get("title", ""),
                scenario=str(data.get("scenario") or ""),
                question=str(data.get("question") or ""),
                solution=str(details.get("realistic_solution") or ""),
                napkin_math=str(details.get("napkin_math") or ""),
                status=data.get("status", ""),
                chain_ids=chain_ids,
                human_review_status=human_review.get("status", "not-reviewed")
                if isinstance(human_review, dict)
                else "not-reviewed",
                has_visual=bool(data.get("visual")),
                has_options=bool(details.get("options")),
                has_napkin=bool((details.get("napkin_math") or "").strip()),
                is_incomplete_information="incomplete-information" in tags,
            )
        )
    return questions


def load_taxonomy_applicability() -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], str]]:
    topics: dict[str, dict[str, Any]] = {}
    applicable: dict[tuple[str, str], str] = {}
    data = yaml.safe_load(TAXONOMY_DATA.read_text(encoding="utf-8")) or {}
    for topic in data.get("topics", []):
        topic_id = topic["id"]
        topics[topic_id] = topic
        for track in topic.get("tracks", []):
            applicable[(topic_id, track)] = "taxonomy_data"
    return topics, applicable


def count_by(questions: list[Question], *attrs: str) -> Counter[tuple[str, ...]]:
    return Counter(tuple(str(getattr(q, attr)) for attr in attrs) for q in questions)


def counter_to_dict(counter: Counter[Any], sep: str = ":") -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in counter.items():
        if isinstance(key, tuple):
            out[sep.join(str(part) for part in key)] = value
        else:
            out[str(key)] = value
    return dict(sorted(out.items()))


def gini(values: list[int]) -> float:
    """Return Gini coefficient for non-negative integer counts."""
    if not values:
        return 0.0
    sorted_values = sorted(v for v in values if v >= 0)
    total = sum(sorted_values)
    if total == 0:
        return 0.0
    n = len(sorted_values)
    weighted = sum((i + 1) * value for i, value in enumerate(sorted_values))
    return (2 * weighted) / (n * total) - (n + 1) / n


def coefficient_of_variation(values: list[int]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5 / mean


def summarize_counts(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"cells": 0, "min": 0, "max": 0, "mean": 0.0, "cv": 0.0, "gini": 0.0, "zero_cells": 0}
    return {
        "cells": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 3),
        "cv": round(coefficient_of_variation(values), 3),
        "gini": round(gini(values), 3),
        "zero_cells": sum(1 for value in values if value == 0),
    }


def topic_area_map(questions: list[Question], taxonomy_topics: dict[str, dict[str, Any]]) -> dict[str, str]:
    areas = {tid: t.get("area", "") for tid, t in taxonomy_topics.items()}
    observed: dict[str, Counter[str]] = defaultdict(Counter)
    for q in questions:
        observed[q.topic][q.competency_area] += 1
    for topic, counts in observed.items():
        if topic not in areas or not areas[topic]:
            areas[topic] = counts.most_common(1)[0][0]
    return areas


def classify_topic_track(
    topic: str,
    track: str,
    count: int,
    applicable: dict[tuple[str, str], str],
) -> str:
    if track == "global":
        return "strategic" if count > 0 else "thin"
    if (topic, track) not in applicable:
        return "suspect" if count > 0 else "invalid"
    if count == 0:
        return "thin"
    if count < 10:
        return "thin"
    return "healthy"


def chain_depths(chains: list[dict[str, Any]]) -> dict[str, int]:
    depths: dict[str, int] = defaultdict(int)
    for chain in chains:
        for entry in chain.get("questions", []):
            qid = entry.get("id") if isinstance(entry, dict) else entry
            if qid:
                depths[qid] = max(depths[qid], len(chain.get("questions", [])))
    return depths


def build_coverage(questions: list[Question], chains: list[dict[str, Any]]) -> dict[str, Any]:
    taxonomy_topics, applicable = load_taxonomy_applicability()
    all_topics = sorted(VALID_TOPICS | {q.topic for q in questions})
    topic_areas = topic_area_map(questions, taxonomy_topics)
    qid_depths = chain_depths(chains)
    topic_track_counts = count_by(questions, "topic", "track")

    topic_track: list[dict[str, Any]] = []
    for topic in all_topics:
        tracks = set(TRACKS)
        for track in sorted(tracks):
            cnt = topic_track_counts.get((topic, track), 0)
            topic_track.append(
                {
                    "topic": topic,
                    "track": track,
                    "competency_area": topic_areas.get(topic, ""),
                    "count": cnt,
                    "classification": classify_topic_track(topic, track, cnt, applicable),
                    "applicability_source": applicable.get((topic, track), "global_or_observed"),
                }
            )

    track_area = Counter((q.track, q.competency_area) for q in questions)
    area_zone = count_by(questions, "competency_area", "zone")
    track_area_level = count_by(questions, "track", "competency_area", "level")
    track_area_zone = count_by(questions, "track", "competency_area", "zone")
    track_topic_level = count_by(questions, "track", "topic", "level")
    track_topic_zone = count_by(questions, "track", "topic", "zone")
    chain_by_track = Counter(q.track for q in questions if q.chain_ids)
    topic_chain = Counter()
    for q in questions:
        if qid_depths.get(q.id, 0) >= 3:
            topic_chain[q.topic] += 1

    full_track_area_level = [
        track_area_level.get((track, area, level), 0)
        for track in TRACKS
        for area in sorted(VALID_COMPETENCY_AREAS)
        for level in LEVELS
    ]
    full_track_area_zone = [
        track_area_zone.get((track, area, zone), 0)
        for track in TRACKS
        for area in sorted(VALID_COMPETENCY_AREAS)
        for zone in sorted(VALID_ZONES)
    ]
    full_area_zone = [
        area_zone.get((area, zone), 0)
        for area in sorted(VALID_COMPETENCY_AREAS)
        for zone in sorted(VALID_ZONES)
    ]
    format_coverage = {
        "overall": {
            "napkin": sum(1 for q in questions if q.has_napkin),
            "visual": sum(1 for q in questions if q.has_visual),
            "mcq": sum(1 for q in questions if q.has_options),
            "chain": sum(1 for q in questions if q.chain_ids),
            "incomplete_information": sum(1 for q in questions if q.is_incomplete_information),
        },
        "by_track": {},
    }
    for track in TRACKS:
        track_questions = [q for q in questions if q.track == track]
        format_coverage["by_track"][track] = {
            "total": len(track_questions),
            "napkin": sum(1 for q in track_questions if q.has_napkin),
            "visual": sum(1 for q in track_questions if q.has_visual),
            "mcq": sum(1 for q in track_questions if q.has_options),
            "chain": sum(1 for q in track_questions if q.chain_ids),
            "incomplete_information": sum(1 for q in track_questions if q.is_incomplete_information),
        }
    convergence_metrics = {
        "track_area_level": summarize_counts(full_track_area_level),
        "track_area_zone": summarize_counts(full_track_area_zone),
        "competency_area_zone": summarize_counts(full_area_zone),
        "observed_track_topic_level": summarize_counts(list(track_topic_level.values())),
        "observed_track_topic_zone": summarize_counts(list(track_topic_zone.values())),
    }

    return {
        "total_questions": len(questions),
        "track_counts": counter_to_dict(Counter(q.track for q in questions)),
        "level_counts": counter_to_dict(Counter(q.level for q in questions)),
        "zone_counts": counter_to_dict(Counter(q.zone for q in questions)),
        "phase_counts": counter_to_dict(Counter(q.phase for q in questions)),
        "track_level": counter_to_dict(count_by(questions, "track", "level")),
        "track_zone": counter_to_dict(count_by(questions, "track", "zone")),
        "track_phase": counter_to_dict(count_by(questions, "track", "phase")),
        "track_area_level": counter_to_dict(track_area_level),
        "track_area_zone": counter_to_dict(track_area_zone),
        "track_area": {f"{k[0]}:{k[1]}": v for k, v in sorted(track_area.items())},
        "competency_area_zone": counter_to_dict(area_zone),
        "topic_track": topic_track,
        "track_topic_zone": counter_to_dict(track_topic_zone),
        "track_topic_level": counter_to_dict(track_topic_level),
        "topic_track_zone": counter_to_dict(count_by(questions, "topic", "track", "zone")),
        "topic_track_level": counter_to_dict(count_by(questions, "topic", "track", "level")),
        "format_coverage": format_coverage,
        "convergence_metrics": convergence_metrics,
        "chain_counts": {
            "total_chains": len(chains),
            "questions_in_chains": sum(1 for q in questions if q.chain_ids),
            "by_track": counter_to_dict(chain_by_track),
            "topic_chain_question_counts": counter_to_dict(topic_chain),
        },
        "taxonomy": {
            "taxonomy_data_topics": len(taxonomy_topics),
            "schema_topics": len(VALID_TOPICS),
            "missing_from_taxonomy_data": sorted(set(VALID_TOPICS) - set(taxonomy_topics)),
            "extra_in_taxonomy_data": sorted(set(taxonomy_topics) - set(VALID_TOPICS)),
            "global_track_in_taxonomy_data": any(track == "global" for _, track in applicable),
        },
    }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9-]{2,}", text.lower()) if t not in STOPWORDS}


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "you",
    "are",
    "has",
    "have",
    "using",
    "model",
    "system",
    "question",
    "scenario",
}


def build_repair_backlog(
    questions: list[Question],
    coverage: dict[str, Any],
    chains: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    backlog: list[dict[str, Any]] = []

    def add(priority: str, kind: str, title: str, detail: str, path: str | None = None) -> None:
        backlog.append(
            {
                "priority": priority,
                "kind": kind,
                "title": title,
                "detail": detail,
                "path": path,
            }
        )

    if coverage["taxonomy"]["missing_from_taxonomy_data"]:
        add(
            "P0",
            "taxonomy",
            "taxonomy_data.yaml is missing schema v1 topics",
            ", ".join(coverage["taxonomy"]["missing_from_taxonomy_data"]),
            "interviews/vault/schema/taxonomy_data.yaml",
        )
    if not coverage["taxonomy"]["global_track_in_taxonomy_data"]:
        add(
            "P0",
            "taxonomy",
            "global track has no taxonomy applicability model",
            "Global is a first-class track in schema and corpus but absent from taxonomy_data tracks.",
            "interviews/vault/schema/taxonomy_data.yaml",
        )

    schema_text = (VAULT_DIR / "schema" / "question_schema.yaml").read_text(encoding="utf-8")
    if re.search(r"\n\s+question:\n", schema_text) is None:
        add(
            "P0",
            "schema",
            "LinkML schema does not declare question slot",
            "YAML corpus contains question fields, but question_schema.yaml lacks the slot.",
            "interviews/vault/schema/question_schema.yaml",
        )

    for q in questions:
        if q.id == "mobile-cell-11084":
            add(
                "P0",
                "content",
                "Scenario/solution mismatch in mobile-cell-11084",
                "Scenario is about checkpoint storage, while solution and napkin math discuss NPU latency units.",
                q.path,
            )
            break

    for chain in chains:
        chain_id = chain.get("chain_id", "")
        if chain_id == "cloud-chain-004":
            add(
                "P1",
                "chain",
                "cloud-chain-004 topic/content mismatch",
                "Chain topic is pruning-sparsity but titles are adversarial debiasing/fairness.",
                "interviews/vault/chains.json",
            )
            break

    q_by_id = {q.id: q for q in questions}
    chain_topic_mismatch_count = 0
    for chain in chains:
        topic = chain.get("topic", "")
        entries = chain.get("questions", [])
        mismatches = []
        for entry in entries:
            qid = entry.get("id") if isinstance(entry, dict) else entry
            q = q_by_id.get(qid)
            if q and q.topic != topic:
                mismatches.append(f"{qid}:{q.topic}")
        if mismatches:
            chain_topic_mismatch_count += 1
            if chain_topic_mismatch_count > 80:
                continue
            add(
                "P2",
                "chain",
                f"{chain.get('chain_id')} has question topics outside chain topic",
                ", ".join(mismatches[:8]),
                "interviews/vault/chains.json",
            )

    for q in questions:
        if q.question and normalize_text(q.question) in normalize_text(q.scenario):
            add(
                "P2",
                "ux",
                "question duplicates scenario ask",
                f"{q.id}: duplicate question should be UI-suppressed or scenario-cleaned later.",
                q.path,
            )
            if sum(1 for item in backlog if item["kind"] == "ux") >= 25:
                break

    low_overlap = []
    for q in questions:
        if len(q.solution) < 80:
            continue
        scenario_tokens = token_set(q.scenario + " " + q.question)
        solution_tokens = token_set(q.solution)
        if not scenario_tokens or not solution_tokens:
            continue
        overlap = len(scenario_tokens & solution_tokens) / max(1, len(solution_tokens))
        if overlap < 0.05:
            low_overlap.append((overlap, q))
    for overlap, q in sorted(low_overlap, key=lambda x: x[0])[:40]:
        add(
            "P2",
            "content-audit",
            "low scenario/question vs solution lexical overlap",
            f"{q.id}: overlap={overlap:.3f}; review for possible mismatch or overly generic solution.",
            q.path,
        )

    return backlog


def target_spec(
    pack: str,
    priority: int,
    track: str,
    topic: str,
    area: str,
    zone: str,
    level: str,
    phase: str,
    rationale: str,
    wall: str | None = None,
    format_hint: str = "standard",
    chain_strategy: str = "standalone-first; assign chain metadata only after review",
) -> dict[str, Any]:
    return {
        "pack": pack,
        "priority": priority,
        "track": track,
        "topic": topic,
        "competency_area": area,
        "zone": zone,
        "level": level,
        "phase": phase,
        "mlsysim_wall": wall,
        "format": format_hint,
        "chain_strategy": chain_strategy,
        "rationale": rationale,
    }


def observed_area(topic: str, topic_areas: dict[str, str]) -> str:
    return topic_areas.get(topic) or "cross-cutting"


def build_generation_queue(questions: list[Question], coverage: dict[str, Any]) -> list[dict[str, Any]]:
    taxonomy_topics, _ = load_taxonomy_applicability()
    topic_areas = topic_area_map(questions, taxonomy_topics)
    queue: list[dict[str, Any]] = []
    existing = Counter((q.track, q.topic, q.zone, q.level, q.phase) for q in questions)
    topic_counts = Counter(q.topic for q in questions)

    global_topics = [
        "roofline-analysis",
        "memory-hierarchy-design",
        "latency-decomposition",
        "queueing-theory",
        "tco-cost-modeling",
        "sustainability-carbon-accounting",
        "fairness-evaluation",
        "fault-tolerance-checkpointing",
        "data-pipeline-engineering",
        "model-serving-infrastructure",
        "profiling-bottleneck-analysis",
        "compute-cost-estimation",
    ]
    for topic in sorted(global_topics, key=lambda t: topic_counts.get(t, 0)):
        for zone in PRIORITY_ZONES:
            queue.append(
                target_spec(
                    "global-cross-track",
                    10,
                    "global",
                    topic,
                    observed_area(topic, topic_areas),
                    zone,
                    STAFF_LEVEL_BY_ZONE[zone],
                    "both",
                    "Global track needs cross-regime, non-vendor-specific Staff+ reasoning.",
                    format_hint="cross-track comparison",
                    chain_strategy="candidate for new global cross-track chain once 3+ related items exist",
                )
            )

    for track in TRACKS:
        candidate_topics = [
            row["topic"]
            for row in coverage["topic_track"]
            if row["track"] == track and row["classification"] in {"healthy", "thin", "strategic"}
        ]
        for topic in sorted(set(candidate_topics), key=lambda t: topic_counts.get(t, 0))[:28]:
            for zone in ["realization", "specification"]:
                level = "L5" if zone != "mastery" else "L6+"
                if existing.get((track, topic, zone, level, "both"), 0) < 2:
                    queue.append(
                        target_spec(
                            "realization-and-specification",
                            20,
                            track,
                            topic,
                            observed_area(topic, topic_areas),
                            zone,
                            level,
                            "both",
                            "Underfilled design+quantify/specification cells need concrete sizing questions.",
                            format_hint="sizing or architecture-to-numbers",
                            chain_strategy="prefer L5/L6 capstone for an existing topic chain; otherwise standalone",
                        )
                    )

    training_topics = [
        "fault-tolerance-checkpointing",
        "gradient-synchronization",
        "collective-communication",
        "pipeline-parallelism",
        "data-pipeline-engineering",
        "streaming-ingestion",
        "differential-privacy",
        "fairness-evaluation",
        "compute-cost-estimation",
        "mixed-precision-training",
    ]
    for track in ["cloud", "edge", "global"]:
        for topic in training_topics:
            for zone in ["diagnosis", "optimization", "evaluation"]:
                queue.append(
                    target_spec(
                        "training-lifecycle",
                        30,
                        track,
                        topic,
                        observed_area(topic, topic_areas),
                        zone,
                        "L4" if zone != "evaluation" else "L5",
                        "training",
                        "Training-only scenarios are underrepresented relative to lifecycle claims.",
                        format_hint="training incident or capacity planning",
                        chain_strategy="candidate for L3-L5 training lifecycle chain within same topic",
                    )
                )

    weak_wall_names = [
        "tail-latency",
        "ingestion",
        "transformation",
        "locality",
        "fragility",
        "multi-tenant",
        "sustainability",
        "safety",
        "sensitivity",
        "synthesis",
    ]
    wall_tracks = {
        "tail-latency": ["cloud", "edge", "mobile"],
        "ingestion": ["cloud", "edge", "tinyml"],
        "transformation": ["cloud", "mobile", "tinyml"],
        "locality": ["cloud", "edge"],
        "fragility": ["cloud", "global"],
        "multi-tenant": ["cloud", "global"],
        "sustainability": ["cloud", "mobile", "tinyml", "global"],
        "safety": ["edge", "mobile", "tinyml", "global"],
        "sensitivity": ["cloud", "edge", "global"],
        "synthesis": ["cloud", "edge", "mobile", "global"],
    }
    for wall in weak_wall_names:
        info = MLSYSIM_WALLS[wall]
        for track in wall_tracks[wall]:
            for topic in info["topics"]:
                queue.append(
                    target_spec(
                        "mlsysim-weak-wall",
                        40,
                        track,
                        topic,
                        observed_area(topic, topic_areas),
                        "diagnosis" if wall != "synthesis" else "specification",
                        "L4" if wall != "synthesis" else "L5",
                        "both",
                        f"Exercise MLSysIM {wall} wall with a concrete binding-constraint question.",
                        wall=wall,
                        format_hint="binding wall diagnosis",
                        chain_strategy="group by MLSysIM wall first; add chain only when topic progression is coherent",
                    )
                )

    incomplete_topics = [
        "model-serving-infrastructure",
        "latency-decomposition",
        "tco-cost-modeling",
        "power-budgeting",
        "fault-tolerance-checkpointing",
        "data-pipeline-engineering",
        "federated-learning",
        "memory-pressure-management",
        "network-bandwidth-bottlenecks",
        "sustainability-carbon-accounting",
    ]
    for track in TRACKS:
        for topic in incomplete_topics:
            queue.append(
                target_spec(
                    "incomplete-information",
                    50,
                    track,
                    topic,
                    observed_area(topic, topic_areas),
                    "mastery",
                    "L6+",
                    "both",
                    "Future-work item type: candidate must identify missing inputs before solving.",
                    format_hint="incomplete information",
                    chain_strategy="L6+ capstone candidate; do not chain until manually reviewed",
                )
            )

    seen = set()
    deduped_by_pack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(queue, key=lambda x: (x["priority"], x["pack"], x["track"], x["topic"])):
        key = (
            item["pack"],
            item["track"],
            item["topic"],
            item["zone"],
            item["level"],
            item["phase"],
            item.get("mlsysim_wall"),
        )
        if key in seen:
            continue
        seen.add(key)
        item = dict(item)
        deduped_by_pack[item["pack"]].append(item)

    pack_limits = {
        "global-cross-track": 60,
        "realization-and-specification": 120,
        "training-lifecycle": 90,
        "mlsysim-weak-wall": 80,
        "incomplete-information": 50,
    }
    final: list[dict[str, Any]] = []
    for pack, limit in pack_limits.items():
        final.extend(deduped_by_pack.get(pack, [])[:limit])
    for idx, item in enumerate(final, 1):
        item["target_id"] = f"target-{idx:04d}"
    return final


def build_pilot_pack(queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = {
        "global-cross-track": 10,
        "realization-and-specification": 10,
        "training-lifecycle": 10,
        "mlsysim-weak-wall": 10,
        "incomplete-information": 10,
    }
    pilot: list[dict[str, Any]] = []
    used = set()
    for pack, n in buckets.items():
        choices = [q for q in queue if q["pack"] == pack]
        for item in choices:
            diversity_key = (item["track"], item["topic"])
            if diversity_key in used and len([p for p in pilot if p["pack"] == pack]) < n - 2:
                continue
            pilot.append(item)
            used.add(diversity_key)
            if len([p for p in pilot if p["pack"] == pack]) >= n:
                break
    return pilot


def validation_gates() -> list[dict[str, Any]]:
    return [
        {"gate": "schema", "command": "python3 interviews/vault/scripts/validate_generation_gates.py", "blocks": True},
        {"gate": "question-shape", "rule": "non-empty, one sentence, <=200 chars, ends with ?", "blocks": True},
        {"gate": "scenario-question-duplication", "rule": "flag exact duplicate ask for UI/editorial cleanup", "blocks": False},
        {"gate": "topic-track-applicability", "rule": "must be taxonomy-applicable unless track=global or explicitly justified", "blocks": True},
        {"gate": "zone-level-affinity", "rule": "outside ZONE_LEVEL_AFFINITY requires human review note", "blocks": False},
        {"gate": "napkin-math", "command": "python3 interviews/vault/scripts/verify_math.py", "blocks": True},
        {"gate": "dedup", "rule": "scenario/question near-duplicates checked before publish", "blocks": True},
        {"gate": "chain-integrity", "rule": "chain IDs exist and levels progress monotonically", "blocks": True},
        {"gate": "visual-assets", "rule": "visual asset exists, alt text is non-empty, and visual questions reference the diagram", "blocks": True},
        {"gate": "provenance", "rule": "report provenance and human_reviewed.status counts before release", "blocks": False},
    ]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=list) + "\n", encoding="utf-8")


def table_rows(items: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    rows = [header, sep]
    for item in items:
        rows.append("| " + " | ".join(str(item.get(f, "")) for f in fields) + " |")
    return "\n".join(rows)


def write_markdown_reports(
    coverage: dict[str, Any],
    backlog: list[dict[str, Any]],
    pilot: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    gates: list[dict[str, Any]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    track_counts = coverage["track_counts"]
    zone_counts = coverage["zone_counts"]
    phase_counts = coverage["phase_counts"]
    thin_pairs = [r for r in coverage["topic_track"] if r["classification"] == "thin"]
    suspect_pairs = [r for r in coverage["topic_track"] if r["classification"] == "suspect"]
    low_track_area_level = [
        {"cell": k, "count": v}
        for k, v in sorted(coverage["track_area_level"].items(), key=lambda item: item[1])[:80]
    ]
    low_track_area_zone = [
        {"cell": k, "count": v}
        for k, v in sorted(coverage["track_area_zone"].items(), key=lambda item: item[1])[:80]
    ]
    low_track_topic_level = [
        {"cell": k, "count": v}
        for k, v in sorted(coverage["track_topic_level"].items(), key=lambda item: item[1])[:80]
    ]
    format_rows = [
        {"track": track, **values}
        for track, values in sorted(coverage["format_coverage"]["by_track"].items())
    ]
    convergence_rows = [
        {"metric": metric, **values}
        for metric, values in coverage["convergence_metrics"].items()
    ]

    (OUTPUT_DIR / "coverage_report.md").write_text(
        "\n".join(
            [
                "# StaffML Coverage Report",
                "",
                f"Total YAML questions: {coverage['total_questions']}",
                "",
                "## Track Counts",
                "",
                table_rows([{"track": k, "count": v} for k, v in sorted(track_counts.items())], ["track", "count"]),
                "",
                "## Zone Counts",
                "",
                table_rows([{"zone": k, "count": v} for k, v in sorted(zone_counts.items())], ["zone", "count"]),
                "",
                "## Phase Counts",
                "",
                table_rows([{"phase": k, "count": v} for k, v in sorted(phase_counts.items())], ["phase", "count"]),
                "",
                "## Taxonomy Drift",
                "",
                f"- Schema topics: {coverage['taxonomy']['schema_topics']}",
                f"- taxonomy_data topics: {coverage['taxonomy']['taxonomy_data_topics']}",
                f"- Missing from taxonomy_data: {', '.join(coverage['taxonomy']['missing_from_taxonomy_data']) or 'none'}",
                f"- Global modeled in taxonomy_data: {coverage['taxonomy']['global_track_in_taxonomy_data']}",
                "",
                "## Format Coverage",
                "",
                table_rows(format_rows, ["track", "total", "napkin", "visual", "mcq", "chain", "incomplete_information"]),
                "",
                "## Convergence Metrics",
                "",
                table_rows(convergence_rows, ["metric", "cells", "min", "max", "mean", "cv", "gini", "zero_cells"]),
                "",
                "## Thin Topic-Track Pairs",
                "",
                table_rows(thin_pairs[:80], ["topic", "track", "competency_area", "count", "classification"]),
                "",
                "## Lowest Track-Area-Level Cells",
                "",
                table_rows(low_track_area_level, ["cell", "count"]),
                "",
                "## Lowest Track-Area-Zone Cells",
                "",
                table_rows(low_track_area_zone, ["cell", "count"]),
                "",
                "## Lowest Observed Track-Topic-Level Cells",
                "",
                table_rows(low_track_topic_level, ["cell", "count"]),
                "",
                "## Suspect Topic-Track Pairs",
                "",
                table_rows(suspect_pairs[:80], ["topic", "track", "competency_area", "count", "classification"]),
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "repair_backlog.md").write_text(
        "\n".join(
            [
                "# StaffML Repair Backlog",
                "",
                table_rows(backlog, ["priority", "kind", "title", "path", "detail"]),
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "pilot_pack.md").write_text(
        "\n".join(
            [
                "# StaffML 50-Question Pilot Pack",
                "",
                table_rows(
                    pilot,
                    [
                        "target_id",
                        "pack",
                        "track",
                        "topic",
                        "zone",
                        "level",
                        "phase",
                        "mlsysim_wall",
                        "format",
                        "chain_strategy",
                    ],
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "generation_queue.md").write_text(
        "\n".join(
            [
                "# StaffML Targeted Generation Queue",
                "",
                f"Targets: {len(queue)}",
                "",
                table_rows(
                    queue[:300],
                    [
                        "target_id",
                        "priority",
                        "pack",
                        "track",
                        "topic",
                        "competency_area",
                        "zone",
                        "level",
                        "phase",
                        "mlsysim_wall",
                        "format",
                        "chain_strategy",
                    ],
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "validation_gates.md").write_text(
        "\n".join(
            [
                "# StaffML Generation Validation Gates",
                "",
                table_rows(gates, ["gate", "command", "rule", "blocks"]),
                "",
            ]
        ),
        encoding="utf-8",
    )

    top_backlog = [b for b in backlog if b["priority"] in {"P0", "P1"}]
    (OUTPUT_DIR / "release_readiness_note.md").write_text(
        "\n".join(
            [
                "# StaffML Release Readiness Note",
                "",
                "## Summary",
                "",
                f"- Coverage cube built from {coverage['total_questions']} YAML questions.",
                f"- Generation queue contains {len(queue)} targeted candidate specs.",
                f"- Pilot pack contains {len(pilot)} manually reviewable specs.",
                f"- P0/P1 repair items: {len(top_backlog)}.",
                "",
                "## Do Not Overclaim",
                "",
                "- Treat `validated` and `math_verified` as automated/LLM-backed checks unless human review status says otherwise.",
                "- Report `human_reviewed.status` counts before public claims about human verification.",
                "- Keep invalid physics cells empty or explicitly justified rather than filling every combinatorial gap.",
                "",
                "## Immediate P0/P1 Repairs",
                "",
                table_rows(top_backlog, ["priority", "kind", "title", "path"]),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)

    questions = load_questions()
    chains = json.loads(CHAINS_PATH.read_text(encoding="utf-8"))
    coverage = build_coverage(questions, chains)
    backlog = build_repair_backlog(questions, coverage, chains)
    queue = build_generation_queue(questions, coverage)
    pilot = build_pilot_pack(queue)
    gates = validation_gates()

    write_json(OUTPUT_DIR / "coverage_cube.json", coverage)
    write_json(OUTPUT_DIR / "track_area_level.json", coverage["track_area_level"])
    write_json(OUTPUT_DIR / "track_area_zone.json", coverage["track_area_zone"])
    write_json(OUTPUT_DIR / "track_topic_level.json", coverage["track_topic_level"])
    write_json(OUTPUT_DIR / "track_topic_zone.json", coverage["track_topic_zone"])
    write_json(OUTPUT_DIR / "format_coverage.json", coverage["format_coverage"])
    write_json(OUTPUT_DIR / "convergence_metrics.json", coverage["convergence_metrics"])
    write_json(OUTPUT_DIR / "repair_backlog.json", backlog)
    write_json(OUTPUT_DIR / "generation_queue.json", queue)
    write_json(OUTPUT_DIR / "pilot_pack.json", pilot)
    write_json(OUTPUT_DIR / "validation_gates.json", gates)
    write_markdown_reports(coverage, backlog, pilot, queue, gates)

    print(f"Wrote StaffML gap-plan artifacts to {OUTPUT_DIR}")
    print(f"Questions: {len(questions)}")
    print(f"Repair items: {len(backlog)}")
    print(f"Pilot targets: {len(pilot)}")
    print(f"Generation targets: {len(queue)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
