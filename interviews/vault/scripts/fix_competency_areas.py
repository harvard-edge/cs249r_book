#!/usr/bin/env python3
"""Phase 0 cleanup: remap malformed `competency_area` values to canonical.

The schema's `competency_area` field is documented as "one of 13 canonical
areas" but is enforced as a free-form string at the LinkML layer. This
left a hole that Gemini-generated questions slipped through: when asked
to populate the field, the model sometimes used the topic name
(`data-pipeline-engineering`) or the zone name (`evaluation`) instead.

This script fixes existing data. The companion change adds a LinkML
permissible_values enum so the same drift cannot reoccur.

Usage:
    python3 fix_competency_areas.py --dry-run   # preview
    python3 fix_competency_areas.py             # apply
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"

CANONICAL = {
    "deployment", "parallelism", "networking", "latency", "memory",
    "compute", "data", "power", "precision", "reliability",
    "optimization", "architecture", "cross-cutting",
}

# Best-effort remap from observed malformed values to canonical.
# Built from the audit on 2026-04-25; see GUI screenshot for the
# malformed values that surfaced as area-filter buttons.
REMAP = {
    # Topic-name-as-area
    "data-pipeline-engineering": "data",
    "duty-cycling": "power",
    "kv-cache-management": "architecture",
    "memory-hierarchy-design": "memory",
    "interconnect-topology": "networking",
    "network-bandwidth-bottlenecks": "networking",
    "pipeline-parallelism": "parallelism",
    "queueing-theory": "latency",
    "fault-tolerance-checkpointing": "reliability",
    "quantization-fundamentals": "precision",
    "communication-computation-overlap": "optimization",
    "compute-cost-estimation": "compute",
    "collective-communication": "networking",
    # Zone-name-as-area (mistaken)
    "diagnosis": "cross-cutting",
    "evaluation": "cross-cutting",
    "specification": "cross-cutting",
    "realization": "cross-cutting",
    # Slash-form (Gemini occasionally output `<track> / <topic>`)
    # We remap by topic since the track is always the question's track field.
    "edge / data-pipeline-engineering": "data",
    "edge data-pipeline-engineering": "data",
    "edge / network-bandwidth-bottlenecks": "networking",
    "edge network-bandwidth-bottlenecks": "networking",
    "edge / pipeline-parallelism": "parallelism",
    "edge pipeline-parallelism": "parallelism",
    "edge / queueing-theory": "latency",
    "edge queueing-theory": "latency",
    "mobile / memory-hierarchy-design": "memory",
    "mobile memory-hierarchy-design": "memory",
    "mobile / quantization-fundamentals": "precision",
    "mobile quantization-fundamentals": "precision",
    "tinyml / communication-computation-overlap": "optimization",
    "tinyml communication-computation-overlap": "optimization",
    "tinyml / compute-cost-estimation": "compute",
    "tinyml compute-cost-estimation": "compute",
    "tinyml / queueing-theory": "latency",
    "tinyml queueing-theory": "latency",
    "tinyml / memory-hierarchy-design": "memory",
    "tinyml memory-hierarchy-design": "memory",
    "mobile / duty-cycling": "power",
    "mobile duty-cycling": "power",
    "mobile / communication-computation-overlap": "optimization",
    "mobile communication-computation-overlap": "optimization",
    "edge / model-serving-infrastructure": "deployment",
    "edge model-serving-infrastructure": "deployment",
    # Standalone topic-name-as-area (more)
    "model-serving-infrastructure": "deployment",
    "distributed-training-economics": "cross-cutting",
    "fluency": "cross-cutting",  # zone leaked into area field
    "mastery": "cross-cutting",  # zone leaked into area field
    # Bloom-verb-as-area (mistaken — bloom_level field leaked into area)
    "implement": "cross-cutting",
    "analyze": "cross-cutting",
    "design": "cross-cutting",
    # Underscore-form hallucinations (Gemini invented snake_case "areas")
    "fault_tolerance": "reliability",
    "power_management": "power",
    "network_architecture": "networking",
    "memory_management": "memory",
    "memory_architecture": "memory",
    "distributed_systems": "parallelism",
    "performance_modeling": "optimization",
    # Track-prefixed slash-form (more) — extend the existing slash-form table
    "tinyml/queueing-theory": "latency",
    "tinyml/fault-tolerance-checkpointing": "reliability",
    "mobile/queueing-theory": "latency",
    "mobile/quantization-fundamentals": "precision",
    "mobile/fault-tolerance-checkpointing": "reliability",
    "mobile/duty-cycling": "power",
    "mobile/data-pipeline-engineering": "data",
    "mobile/model-serving-infrastructure": "deployment",
    "edge/queueing-theory": "latency",
    "edge/compute-cost-estimation": "compute",
    "edge/data-pipeline-engineering": "data",
    "edge/communication-computation-overlap": "optimization",
    "edge/fault-tolerance-checkpointing": "reliability",
    # Track-prefixed dash-form ("<track> - <topic>") — new variant this run
    "tinyml - duty-cycling": "power",
    "tinyml - communication-computation-overlap": "optimization",
    "tinyml - quantization-fundamentals": "precision",
    "tinyml - model-serving-infrastructure": "deployment",
    "tinyml - memory-hierarchy-design": "memory",
    "edge - communication-computation-overlap": "optimization",
    "edge - quantization-fundamentals": "precision",
    "edge - queueing-theory": "latency",
    "edge - memory-hierarchy-design": "memory",
    "edge - fault-tolerance-checkpointing": "reliability",
    "edge - model-serving-infrastructure": "deployment",
    "mobile - duty-cycling": "power",
    "mobile - data-pipeline-engineering": "data",
    "cloud - queueing-theory": "latency",
    "cloud - duty-cycling": "power",
    "cloud - quantization-fundamentals": "precision",
    "cloud - compute-cost-estimation": "compute",
    # Capitalised / spaced variants (just in case)
    "Data pipeline engineering": "data",
    "Duty cycling": "power",
    "Memory hierarchy design": "memory",
}


def survey() -> dict[str, list[Path]]:
    """Return malformed_value -> list of files using it."""
    bad = defaultdict(list)
    for p in QUESTIONS_DIR.glob("*/*.yaml"):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d:
            continue
        ca = d.get("competency_area")
        if ca and ca not in CANONICAL:
            bad[ca].append(p)
    return bad


def apply_fix(bad: dict[str, list[Path]], dry_run: bool) -> tuple[int, list[str]]:
    """Apply REMAP. Returns (n_fixed, unmapped_values)."""
    n_fixed = 0
    unmapped = []
    for malformed, paths in bad.items():
        replacement = REMAP.get(malformed)
        if not replacement:
            # Try lower-case + dash form
            normalized = malformed.lower().replace(" ", "-").replace("/", " / ").strip()
            replacement = REMAP.get(normalized)
        if not replacement:
            unmapped.append(malformed)
            continue
        for p in paths:
            if dry_run:
                n_fixed += 1
                continue
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
            d["competency_area"] = replacement
            p.write_text(
                yaml.safe_dump(d, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            n_fixed += 1
    return n_fixed, unmapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bad = survey()
    print(f"Malformed `competency_area` values found: {len(bad)} distinct, "
          f"{sum(len(ps) for ps in bad.values())} files affected.")
    print()
    print(f"{'Malformed value':<55} {'→ canonical':<15} {'count':>5}")
    print("-" * 75)
    for v in sorted(bad, key=lambda k: -len(bad[k])):
        rep = REMAP.get(v) or REMAP.get(v.lower().replace(" ", "-").replace("/", " / ").strip()) or "?"
        marker = "→" if rep != "?" else "✗"
        print(f"  {v:<53} {marker} {rep:<13} {len(bad[v]):>5}")

    n, unmapped = apply_fix(bad, args.dry_run)
    print()
    print(f"{'Would fix' if args.dry_run else 'Fixed'}: {n} files")
    if unmapped:
        print(f"\n!! UNMAPPED values (no remap rule): {len(unmapped)}")
        for v in unmapped:
            print(f"    {v!r}")
        print("Add these to REMAP and re-run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
