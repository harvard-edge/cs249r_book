"""Pydantic schema for the canonical topic taxonomy.

Validates topics.json: unique IDs, kebab-case format, valid areas,
prerequisite existence, and DAG acyclicity.

Usage:
    python3 topic_schema.py                # Validate topics.json
    python3 topic_schema.py --stats        # Print topology stats
    python3 topic_schema.py --dot          # Output Graphviz DOT for visualization
    python3 topic_schema.py --literal      # Print a Literal type for use in schema.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

VALID_AREAS = {
    "compute", "memory", "latency", "precision", "power",
    "architecture", "optimization", "parallelism", "networking",
    "deployment", "reliability", "data", "cross-cutting",
}

KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class Topic(BaseModel):
    id: str
    name: str
    area: str
    prerequisites: list[str]
    description: str

    @field_validator("id")
    @classmethod
    def id_is_kebab(cls, v: str) -> str:
        if not KEBAB_RE.match(v):
            raise ValueError(f"ID '{v}' must be kebab-case (lowercase, hyphens only)")
        return v

    @field_validator("area")
    @classmethod
    def area_is_valid(cls, v: str) -> str:
        if v not in VALID_AREAS:
            raise ValueError(f"Area '{v}' not in {sorted(VALID_AREAS)}")
        return v

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Description must be at least 10 characters")
        return v


class TopicTaxonomy(BaseModel):
    version: str
    description: str
    last_updated: str
    areas: list[str]
    topics: list[Topic]

    @model_validator(mode="after")
    def validate_taxonomy(self) -> "TopicTaxonomy":
        errors = []
        topic_ids = {t.id for t in self.topics}

        # 1. No duplicate IDs
        seen = set()
        for t in self.topics:
            if t.id in seen:
                errors.append(f"Duplicate topic ID: '{t.id}'")
            seen.add(t.id)

        # 2. All prerequisites exist
        for t in self.topics:
            for prereq in t.prerequisites:
                if prereq not in topic_ids:
                    errors.append(
                        f"Topic '{t.id}' requires '{prereq}' which doesn't exist"
                    )

        # 3. No cycles (DFS)
        adj = defaultdict(list)
        for t in self.topics:
            for prereq in t.prerequisites:
                adj[prereq].append(t.id)

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in topic_ids}

        def dfs(node, path):
            color[node] = GRAY
            for neighbor in adj.get(node, []):
                if color.get(neighbor) == GRAY:
                    cycle_start = path.index(neighbor)
                    cycle = " → ".join(path[cycle_start:] + [neighbor])
                    errors.append(f"Cycle detected: {cycle}")
                elif color.get(neighbor) == WHITE:
                    dfs(neighbor, path + [neighbor])
            color[node] = BLACK

        for tid in topic_ids:
            if color[tid] == WHITE:
                dfs(tid, [tid])

        # 4. Areas list matches VALID_AREAS
        if set(self.areas) != VALID_AREAS:
            missing = VALID_AREAS - set(self.areas)
            extra = set(self.areas) - VALID_AREAS
            if missing:
                errors.append(f"Missing areas: {sorted(missing)}")
            if extra:
                errors.append(f"Unknown areas: {sorted(extra)}")

        if errors:
            raise ValueError(
                f"{len(errors)} taxonomy errors:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        return self


def load_and_validate(path: str | Path = None) -> TopicTaxonomy:
    """Load and validate topics.json, raise on errors."""
    if path is None:
        path = Path(__file__).parent / "topics.json"
    with open(path) as f:
        data = json.load(f)
    return TopicTaxonomy(**data)


def get_valid_topic_ids(path: str | Path = None) -> set[str]:
    """Return the set of valid topic IDs (for use in other validators)."""
    taxonomy = load_and_validate(path)
    return {t.id for t in taxonomy.topics}


def print_stats(taxonomy: TopicTaxonomy):
    """Print topology statistics."""
    topics = taxonomy.topics
    areas = defaultdict(list)
    for t in topics:
        areas[t.area].append(t)

    print(f"Topics: {len(topics)}")
    print(f"Areas: {len(areas)}")
    print()

    # Per-area breakdown
    for area in sorted(areas):
        area_topics = areas[area]
        print(f"  {area} ({len(area_topics)} topics):")
        for t in area_topics:
            prereqs = f" ← {', '.join(t.prerequisites)}" if t.prerequisites else ""
            print(f"    {t.id}{prereqs}")

    # Graph stats
    roots = [t for t in topics if not t.prerequisites]
    all_prereqs = set()
    for t in topics:
        all_prereqs.update(t.prerequisites)
    leaves = [t for t in topics if t.id not in all_prereqs]

    print(f"\nRoots (no prerequisites): {len(roots)}")
    print(f"Leaves (never a prerequisite): {len(leaves)}")

    # Depth calculation
    topic_map = {t.id: t for t in topics}
    depths = {}

    def get_depth(tid):
        if tid in depths:
            return depths[tid]
        t = topic_map.get(tid)
        if not t or not t.prerequisites:
            depths[tid] = 0
            return 0
        d = 1 + max(get_depth(p) for p in t.prerequisites)
        depths[tid] = d
        return d

    for t in topics:
        get_depth(t.id)

    max_depth = max(depths.values()) if depths else 0
    avg_depth = sum(depths.values()) / len(depths) if depths else 0
    print(f"Max depth: {max_depth}")
    print(f"Mean depth: {avg_depth:.1f}")


def print_dot(taxonomy: TopicTaxonomy):
    """Print Graphviz DOT representation."""
    print("digraph topics {")
    print('  rankdir=LR;')
    print('  node [shape=box, style=rounded, fontsize=10];')

    # Color by area
    area_colors = {
        "compute": "#cfe2f3", "memory": "#d4edda", "latency": "#fdebd0",
        "precision": "#e8d5f5", "power": "#f9d6d5", "architecture": "#d5e8d4",
        "optimization": "#fff2cc", "parallelism": "#dae8fc",
        "networking": "#e1d5e7", "deployment": "#f8cecc",
        "reliability": "#d5e8d4", "data": "#cfe2f3", "cross-cutting": "#f7f7f7",
    }

    for t in taxonomy.topics:
        color = area_colors.get(t.area, "#ffffff")
        print(f'  "{t.id}" [label="{t.name}\\n({t.area})", fillcolor="{color}", style="filled,rounded"];')

    for t in taxonomy.topics:
        for prereq in t.prerequisites:
            print(f'  "{prereq}" -> "{t.id}";')

    print("}")


def print_literal(taxonomy: TopicTaxonomy):
    """Print a Python Literal type for embedding in schema.py."""
    ids = sorted(t.id for t in taxonomy.topics)
    print("TopicID = Literal[")
    for tid in ids:
        print(f'    "{tid}",')
    print("]")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate topic taxonomy")
    parser.add_argument("--stats", action="store_true", help="Print topology stats")
    parser.add_argument("--dot", action="store_true", help="Print Graphviz DOT")
    parser.add_argument("--literal", action="store_true", help="Print Literal type")
    parser.add_argument("--path", default=None, help="Path to topics.json")
    args = parser.parse_args()

    try:
        taxonomy = load_and_validate(args.path)
        print(f"✓ topics.json is valid ({len(taxonomy.topics)} topics)")
    except Exception as e:
        print(f"✗ Validation failed:\n{e}")
        sys.exit(1)

    if args.stats:
        print()
        print_stats(taxonomy)
    elif args.dot:
        print_dot(taxonomy)
    elif args.literal:
        print_literal(taxonomy)


if __name__ == "__main__":
    main()
