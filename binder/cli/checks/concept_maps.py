"""Validate MLSysBook concept-map files and QMD frontmatter wiring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import yaml

try:
    from cli.checks.generated_paths import is_generated
except ImportError:  # pragma: no cover - package-relative import path
    from binder.cli.checks.generated_paths import is_generated

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

# 2026-10-12: schema v2 makes the concept map an ownership contract. A chapter
# OWNS first-order (primary) and second-order (secondary) concepts, states one
# usable claim per owned concept, and names the owning chapter folder for every
# concept it only touches or previews. The contract lives in the AIConfigs rule
# concept-maps.md; files without schema_version 2 keep the legacy checks only.
V2_REQUIRED_FIELDS = {"claims"}
V2_POINTER_FIELDS = ("previews", "touches")
V2_SPLIT_FIELDS = ("concept", "here", "with", "there")
V2_MAX_PRIMARY = 8


@dataclass
class ConceptMapIssue:
    """One concept-map or frontmatter wiring problem, keyed by a repo-relative path."""

    severity: str
    path: str
    message: str


def _read_yaml(path: Path) -> Any:
    """Load a YAML file, returning ``{}`` if it is empty, unreadable, or invalid."""
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _frontmatter(path: Path) -> dict[str, Any]:
    """Return a QMD file's YAML frontmatter as a dict.

    Returns ``{}`` if the file cannot be read, does not start with ``---``,
    or its frontmatter is not a mapping. Invalid YAML is not caught here and
    propagates to the caller.
    """
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
    """Return non-blank items stringified, stripped, and casefolded for comparison."""
    return [str(value).strip().casefold() for value in values if str(value).strip()]


def _validate_list(path: Path, name: str, value: Any, root: Path, findings: list[ConceptMapIssue]) -> None:
    """Append an error if ``concept_map.<name>`` is not a list or has case-insensitive duplicates."""
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
    """Validate one ``*_concepts.yml`` file and its link back to the source QMD.

    Appends errors to ``findings`` for a missing or malformed ``concept_map``
    object, missing required fields, a ``source`` that is absent or not a
    ``.qmd``, non-list or duplicate-bearing list fields, and non-list
    top-level ``keywords``/``topics_covered``. When the source exists, it
    also checks that the source QMD's frontmatter ``concepts`` value resolves
    back to this file.
    """
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

    if concept_map.get("schema_version") == 2:
        _validate_v2_file(concept_map, rel_path, findings)

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


def _owned_names(concept_map: dict[str, Any]) -> list[str]:
    """Return the concepts a v2 map owns (first-order then second-order), as written."""
    names: list[str] = []
    for field in ("primary_concepts", "secondary_concepts"):
        value = concept_map.get(field)
        if isinstance(value, list):
            names.extend(str(v).strip() for v in value if str(v).strip())
    return names


def _validate_v2_file(concept_map: dict[str, Any], rel_path: str, findings: list[ConceptMapIssue]) -> None:
    """Check one schema-v2 map: claims cover exactly the owned concepts; pointer and split entries are well formed."""
    for field in sorted(V2_REQUIRED_FIELDS - set(concept_map)):
        findings.append(ConceptMapIssue("error", rel_path, f"concept_map.{field} is required in schema v2"))

    owned = _owned_names(concept_map)
    primary = concept_map.get("primary_concepts")
    if isinstance(primary, list) and len(primary) > V2_MAX_PRIMARY:
        findings.append(ConceptMapIssue(
            "warning", rel_path,
            f"{len(primary)} first-order concepts (more than {V2_MAX_PRIMARY}); the chapter may be two chapters"))
    overlap = {n.casefold() for n in concept_map.get("primary_concepts") or []} & {
        n.casefold() for n in concept_map.get("secondary_concepts") or []}
    for name in sorted(overlap):
        findings.append(ConceptMapIssue("error", rel_path, f"concept {name!r} is both first-order and second-order"))

    claims = concept_map.get("claims")
    if "claims" in concept_map:
        if not isinstance(claims, dict):
            findings.append(ConceptMapIssue("error", rel_path, "concept_map.claims must be a mapping of concept -> claim"))
        else:
            for name in owned:
                claim = claims.get(name)
                if not isinstance(claim, str) or not claim.strip():
                    findings.append(ConceptMapIssue("error", rel_path, f"owned concept {name!r} has no claim"))
            for name in claims:
                if name not in owned:
                    findings.append(ConceptMapIssue("error", rel_path, f"claim for {name!r}, which this chapter does not own"))

    for field in V2_POINTER_FIELDS:
        entries = concept_map.get(field)
        if entries is None:
            continue
        if not isinstance(entries, list):
            findings.append(ConceptMapIssue("error", rel_path, f"concept_map.{field} must be a list"))
            continue
        for entry in entries:
            if not (isinstance(entry, dict) and str(entry.get("concept") or "").strip()
                    and str(entry.get("owner") or "").strip()):
                findings.append(ConceptMapIssue("error", rel_path, f"concept_map.{field} entries need concept and owner"))
            elif str(entry["concept"]).strip() in owned:
                findings.append(ConceptMapIssue(
                    "error", rel_path, f"{field} lists {entry['concept']!r}, which this chapter owns"))

    splits = concept_map.get("splits")
    if splits is not None:
        if not isinstance(splits, list) or not all(
                isinstance(s, dict) and all(str(s.get(k) or "").strip() for k in V2_SPLIT_FIELDS) for s in splits):
            findings.append(ConceptMapIssue(
                "error", rel_path, f"concept_map.splits entries need {', '.join(V2_SPLIT_FIELDS)}"))


def _chapter_rank(folder: str) -> int:
    """Order chapter folders by their numeric prefix (``04_data_engineering`` -> 4); unnumbered sort last."""
    head = folder.split("_", 1)[0]
    return int(head) if head.isdigit() else 10_000


def _validate_v2_volume(
    maps: list[tuple[Path, dict[str, Any]]], root: Path, findings: list[ConceptMapIssue], include_order: bool
) -> None:
    """Cross-chapter ownership checks over the schema-v2 maps of one volume.

    One owner per concept; every preview or touch names a concept its stated
    owner actually owns; a preview's owner comes later in the volume. With
    ``include_order``, a touch of a later chapter's concept is reported as a
    use before teaching (warning): a sculpting worklist, not part of the gate.
    """
    owner_of: dict[str, str] = {}
    for path, cm in maps:
        folder = path.parent.name
        rel_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        for name in _owned_names(cm):
            key = name.casefold()
            if key in owner_of and owner_of[key] != folder:
                findings.append(ConceptMapIssue(
                    "error", rel_path, f"concept {name!r} is also owned by {owner_of[key]}; a concept has one owner"))
            owner_of.setdefault(key, folder)

    folders = {path.parent.name for path, _ in maps}
    for path, cm in maps:
        folder = path.parent.name
        rel_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        for field in V2_POINTER_FIELDS:
            for entry in cm.get(field) or []:
                if not isinstance(entry, dict):
                    continue
                name, owner = str(entry.get("concept") or "").strip(), str(entry.get("owner") or "").strip()
                if not name or not owner:
                    continue
                if owner not in folders:
                    findings.append(ConceptMapIssue("error", rel_path, f"{field}: owner {owner!r} is not a chapter folder in this volume"))
                    continue
                actual = owner_of.get(name.casefold())
                if actual != owner:
                    findings.append(ConceptMapIssue(
                        "error", rel_path,
                        f"{field}: {name!r} -> {owner}, but {'no chapter owns it' if actual is None else f'its owner is {actual}'}"))
                    continue
                later = _chapter_rank(owner) > _chapter_rank(folder)
                if field == "previews" and not later:
                    findings.append(ConceptMapIssue(
                        "error", rel_path, f"previews: {name!r} is owned by earlier chapter {owner}; list it under touches"))
                if field == "touches" and later and include_order:
                    findings.append(ConceptMapIssue(
                        "warning", rel_path,
                        f"touches: {name!r} is owned by later chapter {owner}; used before it is taught (declare a preview or move it)"))


def check_concept_maps(contents_dir: Path, repo_root: Path, include_order: bool = False) -> List[ConceptMapIssue]:
    """Validate all concept maps in contents_dir; ``include_order`` adds forward-touch warnings."""
    findings: List[ConceptMapIssue] = []
    # 2026-09-12: stale _build copies of chapters failed the push hook.
    qmd_files = sorted(p for p in contents_dir.glob("**/*.qmd") if not is_generated(p, contents_dir))
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

    v2_by_volume: dict[Path, list[tuple[Path, dict[str, Any]]]] = {}
    for concept_path in sorted(
        p for p in contents_dir.glob("**/*_concepts.y*ml") if not is_generated(p, contents_dir)
    ):
        _validate_concept_map_file(concept_path, qmd_frontmatter, repo_root, findings)
        if concept_path.resolve() not in referenced_maps:
            concept_rel = str(concept_path.relative_to(repo_root)) if concept_path.is_relative_to(repo_root) else str(concept_path)
            findings.append(ConceptMapIssue("error", concept_rel, "concept map is not referenced by any QMD frontmatter"))
        data = _read_yaml(concept_path)
        cm = data.get("concept_map") if isinstance(data, dict) else None
        if isinstance(cm, dict) and cm.get("schema_version") == 2:
            # A chapter folder sits directly under its volume directory.
            v2_by_volume.setdefault(concept_path.parent.parent, []).append((concept_path, cm))

    for maps in v2_by_volume.values():
        _validate_v2_volume(maps, repo_root, findings, include_order)

    return findings


def render_volume_map(volume_dir: Path, config_path: Path | None = None) -> str:
    """Render one volume's schema-v2 concept maps as a Markdown ownership map.

    Chapters follow the volume's PDF config order (parts included) when
    ``config_path`` exists, otherwise their folder numbers. Each chapter lists
    the concepts it owns with their claims, then what it previews and touches.
    """
    maps: dict[str, dict[str, Any]] = {}
    for path in sorted(volume_dir.glob("*/*_concepts.y*ml")):
        data = _read_yaml(path)
        cm = data.get("concept_map") if isinstance(data, dict) else None
        if isinstance(cm, dict) and cm.get("schema_version") == 2:
            maps[path.parent.name] = cm

    order: list[tuple[str, str]] = []  # (part heading or "", chapter folder)
    if config_path and config_path.exists():
        part = ""
        for line in config_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("# Part "):
                part = stripped[2:].strip()
                continue
            for folder in maps:
                if stripped.startswith("- ") and f"/{folder}/" in stripped and folder not in {f for _, f in order}:
                    order.append((part, folder))
    seen = {f for _, f in order}
    order.extend(("", f) for f in sorted(maps, key=_chapter_rank) if f not in seen)

    first = sum(len(cm.get("primary_concepts") or []) for cm in maps.values())
    second = sum(len(cm.get("secondary_concepts") or []) for cm in maps.values())
    out = [f"# Concept map: {volume_dir.name}", "",
           f"{len(maps)} chapters own {first} first-order and {second} second-order concepts. "
           "Generated from the chapters' `*_concepts.yml` files by `./binder/binder info concept-map`; "
           "edit those files, not this page.", ""]
    current_part = None
    for part, folder in order:
        cm = maps[folder]
        if part and part != current_part:
            out += [f"## {part}", ""]
            current_part = part
        claims = cm.get("claims") or {}
        out += [f"### {folder}", ""]
        for label, field in (("First-order", "primary_concepts"), ("Second-order", "secondary_concepts")):
            names = cm.get(field) or []
            if names:
                out.append(f"**{label}**")
                out.append("")
                out += [f"- **{n}**: {claims.get(n, '').strip()}" for n in names]
                out.append("")
        for label, field in (("Previews", "previews"), ("Touches", "touches")):
            entries = [e for e in cm.get(field) or [] if isinstance(e, dict)]
            if entries:
                by_owner: dict[str, list[str]] = {}
                for e in entries:
                    by_owner.setdefault(str(e.get("owner")), []).append(str(e.get("concept")))
                parts = "; ".join(f"{o}: {', '.join(c)}" for o, c in sorted(by_owner.items(), key=lambda kv: _chapter_rank(kv[0])))
                out += [f"*{label}* ({len(entries)}): {parts}", ""]
    return "\n".join(out).rstrip() + "\n"
