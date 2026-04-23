"""Ikigai competency zones — the mapping between zones and skills.

The four fundamental skills:
    recall    — facts, definitions, specifications
    analyze   — tradeoffs, reasoning, root cause analysis
    design    — architecture decisions, requirements-to-system
    implement — napkin math, optimization, concrete building

The 11 zones:
    4 pure     (single skill)
    6 compound (two skills intersecting)
    1 mastery  (all four skills)

Usage:
    from zones import zone_skills, skills_to_zone, ZONE_DESCRIPTIONS

    zone_skills("diagnosis")        # → {"recall", "analyze"}
    skills_to_zone({"recall"})      # → "recall"
    skills_to_zone({"analyze", "design"})  # → "evaluation"
"""

from __future__ import annotations

# ── Zone → Skills mapping ────────────────────────────────────

ZONE_SKILLS: dict[str, frozenset[str]] = {
    # Pure zones
    "recall":        frozenset({"recall"}),
    "analyze":       frozenset({"analyze"}),
    "design":        frozenset({"design"}),
    "implement":     frozenset({"implement"}),
    # Compound zones (two skills)
    "diagnosis":     frozenset({"recall", "analyze"}),
    "specification": frozenset({"recall", "design"}),
    "fluency":       frozenset({"recall", "implement"}),
    "evaluation":    frozenset({"analyze", "design"}),
    "realization":   frozenset({"design", "implement"}),
    "optimization":  frozenset({"analyze", "implement"}),
    # Mastery (all four)
    "mastery":       frozenset({"recall", "analyze", "design", "implement"}),
}

ZONE_DESCRIPTIONS: dict[str, str] = {
    "recall":        "Facts & definitions",
    "analyze":       "Tradeoffs & reasoning",
    "design":        "Architecture decisions",
    "implement":     "Napkin math & building",
    "diagnosis":     "Recall + Analyze — identify and explain failures",
    "specification": "Recall + Design — constraints to architecture",
    "fluency":       "Recall + Implement — math from memory",
    "evaluation":    "Analyze + Design — compare architectures by reasoning",
    "realization":   "Design + Implement — architecture to concrete numbers",
    "optimization":  "Analyze + Implement — diagnose and fix bottlenecks",
    "mastery":       "All four — full Staff-level synthesis",
}

# ── Reverse mapping ──────────────────────────────────────────

_SKILLS_TO_ZONE: dict[frozenset[str], str] = {v: k for k, v in ZONE_SKILLS.items()}

ALL_SKILLS = {"recall", "analyze", "design", "implement"}
ALL_ZONES = set(ZONE_SKILLS.keys())


def zone_skills(zone: str) -> frozenset[str]:
    """Return the skills exercised by a zone."""
    if zone not in ZONE_SKILLS:
        raise ValueError(f"Unknown zone '{zone}'. Valid: {sorted(ALL_ZONES)}")
    return ZONE_SKILLS[zone]


def skills_to_zone(skills: set[str]) -> str:
    """Return the zone matching a set of skills."""
    key = frozenset(skills)
    if key not in _SKILLS_TO_ZONE:
        raise ValueError(
            f"No zone for skills {sorted(skills)}. "
            f"Valid combinations: {sorted(str(sorted(k)) for k in _SKILLS_TO_ZONE)}"
        )
    return _SKILLS_TO_ZONE[key]


# ── Suggested zone-level mappings ────────────────────────────

ZONE_LEVEL_AFFINITY: dict[str, list[str]] = {
    # Pure zones tend toward lower levels
    "recall":        ["L1", "L2"],
    "implement":     ["L2", "L3"],
    "analyze":       ["L3", "L4"],
    "design":        ["L4", "L5"],
    # Compound zones span mid-to-senior
    "fluency":       ["L2", "L3"],
    "diagnosis":     ["L3", "L4"],
    "specification": ["L4", "L5"],
    "optimization":  ["L4", "L5"],
    "evaluation":    ["L5", "L6+"],
    "realization":   ["L5", "L6+"],
    # Mastery is staff+
    "mastery":       ["L6+"],
}

# ── Mapping from old reasoning_mode to zones ─────────────────

REASONING_MODE_TO_ZONE: dict[str, str] = {
    "concept-recall":              "recall",
    "concept-explanation":         "recall",
    "napkin-math":                 "fluency",
    "symptom-to-cause":            "diagnosis",
    "tradeoff-analysis":           "evaluation",
    "trade-off-analysis":          "evaluation",
    "requirements-to-architecture": "specification",
    "optimization-task":           "optimization",
    "failure-to-root-cause":       "diagnosis",
}


def migrate_reasoning_mode(mode: str) -> str:
    """Convert old reasoning_mode to new zone."""
    if mode in REASONING_MODE_TO_ZONE:
        return REASONING_MODE_TO_ZONE[mode]
    return "recall"  # safe default for unknown modes


if __name__ == "__main__":
    print("StaffML Ikigai Competency Zones")
    print("=" * 50)
    print()
    for zone, skills in sorted(ZONE_SKILLS.items(), key=lambda x: len(x[1])):
        desc = ZONE_DESCRIPTIONS[zone]
        levels = ", ".join(ZONE_LEVEL_AFFINITY[zone])
        print(f"  {zone:15s}  {sorted(skills)!s:45s}  {levels:10s}  {desc}")
