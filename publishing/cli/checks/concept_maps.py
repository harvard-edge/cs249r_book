"""Validate MLSysBook concept-map files and QMD frontmatter wiring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import yaml

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
class ConceptMapIssue:
    severity: str
    path: str
    message: str


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _frontmatter(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    if not text.startswith("---\n"):
        return {}
    try:
        _, raw, _ = text.split("---", 2)
    except ValueError:
        return {}
    data = yaml.safe_load(raw) or {}
    return data if isinstance(data, dict) else {}


def _normalized_items(values: list[Any]) -> list[str]:
    return [str(value).strip().casefold() for value in values if str(value).strip()]


def _validate_list(path: Path, name: str, value: Any, root: Path, findings: list[ConceptMapIssue]) -> None:
    rel_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    if not isinstance(value, list):
        findings.append(ConceptMapIssue("error", rel_path, f"concept_map.{name} must be a list"))
        return
    seen: set[str] = set()
    for item in _normalized_items(value):
        if item in seen:
            findings.append(ConceptMapIssue("error", rel_path, f"concept_map.{name} contains duplicate item {item!r}"))
        seen.add(item)


def _validate_concept_map_file(
    path: Path,
    qmd_frontmatter: dict[Path, dict[str, Any]],
    root: Path,
    findings: list[ConceptMapIssue],
) -> None:
    rel_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    data = _read_yaml(path)
    if not isinstance(data, dict) or "concept_map" not in data:
        findings.append(ConceptMapIssue("error", rel_path, "missing top-level concept_map object"))
        return
    concept_map = data["concept_map"]
    if not isinstance(concept_map, dict):
        findings.append(ConceptMapIssue("error", rel_path, "concept_map must be a mapping"))
        return

    missing = sorted(REQUIRED_CONCEPT_MAP_FIELDS - set(concept_map))
    for field in missing:
        findings.append(ConceptMapIssue("error", rel_path, f"concept_map.{field} is required"))

    source = concept_map.get("source")
    if not isinstance(source, str) or not source.strip():
        findings.append(ConceptMapIssue("error", rel_path, "concept_map.source must be a non-empty string"))
        source_path = None
    else:
        source_path = path.parent / source
        if not source_path.exists():
            findings.append(ConceptMapIssue("error", rel_path, f"concept_map.source does not exist: {source}"))
        elif source_path.suffix != ".qmd":
            findings.append(ConceptMapIssue("error", rel_path, f"concept_map.source is not a QMD file: {source}"))

    for field in LIST_FIELDS:
        if field in concept_map:
            _validate_list(path, field, concept_map[field], root, findings)

    for field in ("keywords", "topics_covered"):
        if field in data and not isinstance(data[field], list):
            findings.append(ConceptMapIssue("error", rel_path, f"top-level {field} must be a list"))

    if not source_path or not source_path.exists():
        return

    source_meta = qmd_frontmatter.get(source_path)
    source_rel = str(source_path.relative_to(root)) if source_path.is_relative_to(root) else str(source_path)
    if source_meta is None:
        findings.append(ConceptMapIssue("error", rel_path, f"source QMD was not scanned: {source_rel}"))
        return

    concepts_ref = source_meta.get("concepts")
    if not concepts_ref:
        findings.append(ConceptMapIssue("error", source_rel, f"source QMD does not reference {path.name} in concepts"))
        return
    if not isinstance(concepts_ref, str):
        findings.append(ConceptMapIssue("error", source_rel, "frontmatter concepts value must be a string filename"))
        return
    referenced_path = (source_path.parent / concepts_ref).resolve()
    if referenced_path != path.resolve():
        findings.append(
            ConceptMapIssue(
                "error",
                source_rel,
                f"frontmatter concepts points to {concepts_ref}, not {path.name}",
            )
        )


def check_concept_maps(contents_dir: Path, repo_root: Path) -> List[ConceptMapIssue]:
    """Validate all concept maps in contents_dir."""
    findings: List[ConceptMapIssue] = []
    qmd_files = sorted(contents_dir.glob("**/*.qmd"))
    qmd_frontmatter = {path: _frontmatter(path) for path in qmd_files}

    referenced_maps: dict[Path, Path] = {}
    for qmd_path, meta in qmd_frontmatter.items():
        qmd_rel = str(qmd_path.relative_to(repo_root)) if qmd_path.is_relative_to(repo_root) else str(qmd_path)
        concepts_ref = meta.get("concepts")
        if not concepts_ref:
            continue
        if not isinstance(concepts_ref, str):
            findings.append(ConceptMapIssue("error", qmd_rel, "frontmatter concepts value must be a string filename"))
            continue
        concept_path = qmd_path.parent / concepts_ref
        if not concept_path.exists():
            findings.append(ConceptMapIssue("error", qmd_rel, f"frontmatter concepts file does not exist: {concepts_ref}"))
            continue
        if concept_path.suffix not in {".yml", ".yaml"}:
            findings.append(ConceptMapIssue("error", qmd_rel, f"frontmatter concepts file is not YAML: {concepts_ref}"))
        referenced_maps[concept_path.resolve()] = qmd_path

    for concept_path in sorted(contents_dir.glob("**/*_concepts.y*ml")):
        _validate_concept_map_file(concept_path, qmd_frontmatter, repo_root, findings)
        if concept_path.resolve() not in referenced_maps:
            concept_rel = str(concept_path.relative_to(repo_root)) if concept_path.is_relative_to(repo_root) else str(concept_path)
            findings.append(ConceptMapIssue("error", concept_rel, "concept map is not referenced by any QMD frontmatter"))

    return findings
