#!/usr/bin/env python3
"""Verify enums.py stays in sync with question_schema.yaml.

Runs in CI as a drift check. Exits non-zero if the hand-maintained Python
enum constants disagree with the authoritative LinkML schema.

Usage:
    python3 interviews/vault-cli/scripts/check_schema_sync.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
LINKML = REPO / "vault" / "schema" / "question_schema.yaml"
ENUMS_PY = REPO / "vault" / "schema" / "enums.py"

# Map LinkML enum name -> Python constant name in enums.py.
ENUM_MAPPING = {
    "Track":             "VALID_TRACKS",
    "Level":             "VALID_LEVELS",
    "Zone":              "VALID_ZONES",
    "BloomLevel":        "VALID_BLOOM_LEVELS",
    "Phase":             "VALID_PHASES",
    "Status":            "VALID_STATUSES",
    "Provenance":        "VALID_PROVENANCES",
    "HumanReviewStatus": "VALID_HUMAN_REVIEW_STATUSES",
}


def load_linkml_enums() -> dict[str, set[str]]:
    with LINKML.open() as fh:
        data = yaml.safe_load(fh)
    out = {}
    for name, spec in (data.get("enums") or {}).items():
        values = set((spec or {}).get("permissible_values") or {})
        out[name] = values
    return out


def load_python_enums() -> dict[str, set[str]]:
    # Import the module directly; avoids running the validator machinery.
    spec_dir = ENUMS_PY.parent
    if str(spec_dir) not in sys.path:
        sys.path.insert(0, str(spec_dir))
    import enums  # type: ignore[import-not-found]

    out: dict[str, set[str]] = {}
    for linkml_name, py_name in ENUM_MAPPING.items():
        out[linkml_name] = set(getattr(enums, py_name))
    return out


def main() -> int:
    linkml = load_linkml_enums()
    py = load_python_enums()

    drift = False
    for enum_name in ENUM_MAPPING:
        linkml_vals = linkml.get(enum_name, set())
        py_vals = py.get(enum_name, set())
        if linkml_vals != py_vals:
            drift = True
            print(f"[drift] {enum_name}:")
            only_linkml = linkml_vals - py_vals
            only_py = py_vals - linkml_vals
            if only_linkml:
                print(f"  in LinkML only: {sorted(only_linkml)}")
            if only_py:
                print(f"  in enums.py only: {sorted(only_py)}")
    if drift:
        print()
        print("FAIL: schema/enums.py disagrees with schema/question_schema.yaml.")
        print("Update enums.py to match the LinkML schema, then re-run.")
        return 1
    print(f"OK: {len(ENUM_MAPPING)} enums in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
