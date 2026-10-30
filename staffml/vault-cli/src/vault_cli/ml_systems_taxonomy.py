"""Read the ML Systems Taxonomy that question tags refer to.

The taxonomy lives in the repository at ``taxonomy/`` (two levels above
``staffml/vault``), maintained and validated by ``./binder/binder concepts``.
It is not a vault file, so it is read with the standard safe loader rather than
the per-question hardened loader and its size cap; the binder build already
refuses YAML aliases in these files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Taxonomy:
    framework: dict[str, Any]
    field: dict[str, Any]

    @property
    def subjects(self) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        """Subject id → (subject row, its area)."""
        return {row["id"]: (row, area) for area in self.framework.get("areas", []) for row in area.get("rows", [])}

    @property
    def concepts(self) -> dict[str, dict[str, Any]]:
        return {c["id"]: c for c in self.field.get("concepts", [])}

    @property
    def retired(self) -> dict[str, str]:
        """Retired concept id → the concept that replaces it."""
        return {r["id"]: r.get("replaced_by", "") for r in self.field.get("retired", []) or []}

    @property
    def home(self) -> dict[str, str]:
        """Concept id → the subject id that is its home."""
        return {c: rid for rid, (row, _) in self.subjects.items() for c in row.get("concepts", [])}


def load_taxonomy(vault_dir: Path) -> Taxonomy | None:
    """Return the repo's taxonomy, or None in a vault-only checkout."""
    root = vault_dir.parent.parent / "taxonomy"
    unified_path = root / "ml-systems-taxonomy.yml"
    if not unified_path.exists():
        return None
    raw = yaml.safe_load(unified_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "areas" not in raw or "concepts" not in raw:
        raise ValueError(f"{unified_path}: malformed taxonomy file (requires 'areas' and 'concepts')")
    by_home: dict[str, list[str]] = {}
    field_concepts: list[dict[str, Any]] = []
    for c in raw.get("concepts", []):
        if isinstance(c, dict):
            if c.get("id") and c.get("home"):
                by_home.setdefault(c["home"], []).append(c["id"])
            field_concepts.append({k: v for k, v in c.items() if k != "home"})
    resolved_areas = []
    for area in raw.get("areas", []):
        rows = []
        for row in area.get("rows", []):
            if "concepts" in row:
                raise ValueError(f"{unified_path}: subject row {row.get('id')} must not hand-list concepts")
            rows.append({**row, "concepts": list(by_home.get(row.get("id", ""), []))})
        resolved_areas.append({**area, "rows": rows})
    framework = {**raw, "areas": resolved_areas}
    field = {
        "concepts": field_concepts,
        "retired": raw.get("retired", []),
    }
    return Taxonomy(framework, field)
