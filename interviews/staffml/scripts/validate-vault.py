#!/usr/bin/env python3
"""Sparse vault sanity check for the StaffML deploy.

Validates the small committed metadata files that ship in the repo:
``taxonomy.json`` and ``vault-manifest.json``. Confirms taxonomy has
concepts, manifest has a question count, and track distributions add up.

Per-question deep validation (schema, chain integrity, math, etc.) is
covered by ``vault check --strict`` (run in CI via
``staffml-validate-vault.yml``), which validates directly against the
YAML source files in ``interviews/vault/`` rather than a generated JSON
artifact. This script is the cheap pre-deploy gate; ``vault check`` is
the authoritative one.

Exit code 0 = all checks pass, 1 = errors found.

Usage: python3 interviews/staffml/scripts/validate-vault.py
"""

import json
import sys
from pathlib import Path

STAFFML_DATA = Path(__file__).parent.parent / "src" / "data"

errors: list[str] = []
warnings: list[str] = []


def error(msg: str) -> None:
    errors.append(msg)
    print(f"  ❌ {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  ⚠️  {msg}")


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def main() -> int:
    taxonomy_path = STAFFML_DATA / "taxonomy.json"
    manifest_path = STAFFML_DATA / "vault-manifest.json"

    print("\n🔍 Sparse vault check (committed metadata only)")
    print(
        "   Per-question deep validation lives in `vault check --strict` "
        "(staffml-validate-vault.yml).\n"
    )

    if not taxonomy_path.exists():
        error(f"taxonomy.json not found at {taxonomy_path}")
        return 1
    if not manifest_path.exists():
        error(f"vault-manifest.json not found at {manifest_path}")
        return 1

    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy = json.load(f)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    concepts = taxonomy.get("concepts", [])
    if not concepts:
        error("taxonomy has no concepts")
    else:
        ok(f"taxonomy: {len(concepts)} concepts")

    qc = manifest.get("questionCount")
    if qc is None or not isinstance(qc, int) or qc < 1:
        error("manifest.questionCount missing or invalid")
    else:
        ok(f"manifest: questionCount = {qc}")

    td = manifest.get("trackDistribution") or {}
    if isinstance(td, dict) and td:
        s = sum(int(v) for v in td.values() if isinstance(v, int))
        if s != qc:
            warn(
                f"trackDistribution sum ({s}) != questionCount ({qc}) — check manifest"
            )
        else:
            ok("trackDistribution sums to questionCount")

    ver = manifest.get("version", "?")
    h = manifest.get("contentHash", "?")
    ok(f"Vault v{ver} — hash {h}")

    print(f"\n{'=' * 50}")
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"{'=' * 50}")

    if errors:
        print("\n❌ Sparse validation failed")
        return 1
    print(
        "\n🎯 Sparse checks passed — for deep per-question validation run "
        "`vault check --strict` (or rely on staffml-validate-vault in CI)."
    )
    if warnings:
        print(f"   ({len(warnings)} warnings — review recommended)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
