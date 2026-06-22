#!/usr/bin/env python3
"""Validate MLSysBook concept-map files and QMD frontmatter wiring."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
CONTENTS_DIR = REPO_ROOT / "book" / "quarto" / "contents"

LIST_FIELDS = {
    "primary_concepts",
    "secondary_concepts",
    "technical_terms",
    "methodologies",
    "formulas",
    "lighthouse_models",
    "applications",
}
REQUIRED_CONCEPT_MAP_FIELDS = {"source", "primary_concepts", "secondary_concepts"}


@dataclass
class Finding:
    severity: str
    path: str
    message: str


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    try:
        _, raw, _ = text.split("---", 2)
    except ValueError:
        return {}
    data = yaml.safe_load(raw) or {}
    return data if isinstance(data, dict) else {}


def normalized_items(values: list[Any]) -> list[str]:
    return [str(value).strip().casefold() for value in values if str(value).strip()]


def validate_list(path: Path, name: str, value: Any, findings: list[Finding]) -> None:
    if not isinstance(value, list):
        findings.append(Finding("error", rel(path), f"concept_map.{name} must be a list"))
        return
    seen: set[str] = set()
    for item in normalized_items(value):
        if item in seen:
            findings.append(Finding("error", rel(path), f"concept_map.{name} contains duplicate item {item!r}"))
        seen.add(item)


def validate_concept_map_file(path: Path, qmd_frontmatter: dict[Path, dict[str, Any]], findings: list[Finding]) -> None:
    data = read_yaml(path)
    if not isinstance(data, dict) or "concept_map" not in data:
        findings.append(Finding("error", rel(path), "missing top-level concept_map object"))
        return
    concept_map = data["concept_map"]
    if not isinstance(concept_map, dict):
        findings.append(Finding("error", rel(path), "concept_map must be a mapping"))
        return

    missing = sorted(REQUIRED_CONCEPT_MAP_FIELDS - set(concept_map))
    for field in missing:
        findings.append(Finding("error", rel(path), f"concept_map.{field} is required"))

    source = concept_map.get("source")
    if not isinstance(source, str) or not source.strip():
        findings.append(Finding("error", rel(path), "concept_map.source must be a non-empty string"))
        source_path = None
    else:
        source_path = path.parent / source
        if not source_path.exists():
            findings.append(Finding("error", rel(path), f"concept_map.source does not exist: {source}"))
        elif source_path.suffix != ".qmd":
            findings.append(Finding("error", rel(path), f"concept_map.source is not a QMD file: {source}"))

    for field in LIST_FIELDS:
        if field in concept_map:
            validate_list(path, field, concept_map[field], findings)

    for field in ("keywords", "topics_covered"):
        if field in data and not isinstance(data[field], list):
            findings.append(Finding("error", rel(path), f"top-level {field} must be a list"))

    if not source_path or not source_path.exists():
        return

    source_meta = qmd_frontmatter.get(source_path)
    if source_meta is None:
        findings.append(Finding("error", rel(path), f"source QMD was not scanned: {rel(source_path)}"))
        return

    concepts_ref = source_meta.get("concepts")
    if not concepts_ref:
        findings.append(Finding("error", rel(source_path), f"source QMD does not reference {path.name} in concepts"))
        return
    if not isinstance(concepts_ref, str):
        findings.append(Finding("error", rel(source_path), "frontmatter concepts value must be a string filename"))
        return
    referenced_path = (source_path.parent / concepts_ref).resolve()
    if referenced_path != path.resolve():
        findings.append(
            Finding(
                "error",
                rel(source_path),
                f"frontmatter concepts points to {concepts_ref}, not {path.name}",
            )
        )


def validate(contents_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    qmd_files = sorted(contents_dir.glob("**/*.qmd"))
    qmd_frontmatter = {path: frontmatter(path) for path in qmd_files}

    referenced_maps: dict[Path, Path] = {}
    for qmd_path, meta in qmd_frontmatter.items():
        concepts_ref = meta.get("concepts")
        if not concepts_ref:
            continue
        if not isinstance(concepts_ref, str):
            findings.append(Finding("error", rel(qmd_path), "frontmatter concepts value must be a string filename"))
            continue
        concept_path = qmd_path.parent / concepts_ref
        if not concept_path.exists():
            findings.append(Finding("error", rel(qmd_path), f"frontmatter concepts file does not exist: {concepts_ref}"))
            continue
        if concept_path.suffix not in {".yml", ".yaml"}:
            findings.append(Finding("error", rel(qmd_path), f"frontmatter concepts file is not YAML: {concepts_ref}"))
        referenced_maps[concept_path.resolve()] = qmd_path

    for concept_path in sorted(contents_dir.glob("**/*_concepts.y*ml")):
        validate_concept_map_file(concept_path, qmd_frontmatter, findings)
        if concept_path.resolve() not in referenced_maps:
            findings.append(Finding("error", rel(concept_path), "concept map is not referenced by any QMD frontmatter"))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contents-dir", type=Path, default=CONTENTS_DIR)
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    findings = validate(args.contents_dir.resolve())
    report = {
        "contents_dir": str(args.contents_dir.resolve()),
        "status": "clear" if not findings else "issues_found",
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if findings:
        for finding in findings:
            print(f"{finding.severity}: {finding.path}: {finding.message}")
        return 1
    print("Concept map validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
