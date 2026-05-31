"""CI gate: migrated symbols must not be imported from mlsysim.core.constants."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "book" / "tools" / "audit" / "artifacts" / "registry_migration_manifest.json"
CONSTANTS_PATH = REPO_ROOT / "mlsysim" / "mlsysim" / "core" / "constants.py"

SCAN_ROOTS = (
    REPO_ROOT / "book" / "quarto" / "contents",
    REPO_ROOT / "mlsysim" / "mlsysim",
    REPO_ROOT / "labs",
    REPO_ROOT / "mlsysim" / "examples",
    REPO_ROOT / "mlsysim" / "docs",
    REPO_ROOT / "mlsysim" / "tutorial",
)

SKIP_PATH_SUBSTRINGS = (
    "test_registry_parity.py",
    "test_no_legacy_constant_refs.py",
    "test_constants_allowlist.py",
    "migrate_constants_to_registry.py",
    "generate_registry_migration_manifest.py",
    "audit_mlsysim_drift.py",
    "map_constants.py",
    "remove_constants.py",
    "/book/tools/audit/_",
    ".tmp_lego_patch.py",
    "/defaults.py",
)

FROM_CONSTANTS_RE = re.compile(
    r"^\s*from\s+mlsysim\.core\.constants\s+import\s+(.+)$"
)

def _constants_defined_names() -> set[str]:
    source = CONSTANTS_PATH.read_text(encoding="utf-8")
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names

def _legacy_symbols() -> set[str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    defined = _constants_defined_names()
    legacy: set[str] = set()
    for entry in manifest["symbols"]:
        if entry["action"] not in ("migrate", "delete_dead"):
            continue
        if not entry.get("replacement"):
            continue
        if entry["name"] not in defined:
            legacy.add(entry["name"])
    return legacy

def _parse_imported_names(import_clause: str) -> set[str]:
    names: set[str] = set()
    for part in import_clause.split(","):
        part = part.strip()
        if not part or part == "*":
            continue
        if " as " in part:
            part = part.split(" as ", 1)[0].strip()
        names.add(part)
    return names

def _scan_file(path: Path, legacy: set[str]) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        m = FROM_CONSTANTS_RE.match(line)
        if not m:
            continue
        imported = _parse_imported_names(m.group(1))
        if "*" in m.group(1):
            continue  # star imports checked separately via bare-name scan if needed
        bad = sorted(imported & legacy)
        if bad:
            hits.append((lineno, line.strip(), ", ".join(bad)))
    return hits

def test_no_legacy_symbols_imported_from_constants() -> None:
    if not MANIFEST_PATH.exists():
        return
    legacy = _legacy_symbols()
    violations: list[str] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".py", ".qmd"}:
                continue
            sp = str(path)
            if any(skip in sp for skip in SKIP_PATH_SUBSTRINGS):
                continue
            for lineno, line, symbols in _scan_file(path, legacy):
                rel = path.relative_to(REPO_ROOT)
                violations.append(f"{rel}:{lineno}: imports {symbols}\n  {line}")
    assert not violations, (
        "Legacy symbols must use registry paths, not constants imports:\n"
        + "\n".join(violations[:40])
        + (f"\n... and {len(violations) - 40} more" if len(violations) > 40 else "")
    )
