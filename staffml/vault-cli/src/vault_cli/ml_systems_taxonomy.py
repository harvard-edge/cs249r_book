"""Read the ML Systems Taxonomy that question tags refer to.

The taxonomy lives in the repository at ``shared/concepts/`` (two levels above
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
    root = vault_dir.parent.parent / "shared" / "concepts"
    framework_path, field_path = root / "ml-systems-framework.yml", root / "ml-systems-field.yml"
    if not (framework_path.exists() and field_path.exists()):
        return None
    return Taxonomy(yaml.safe_load(framework_path.read_text(encoding="utf-8")),
                    yaml.safe_load(field_path.read_text(encoding="utf-8")))
