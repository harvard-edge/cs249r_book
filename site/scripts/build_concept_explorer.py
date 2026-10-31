#!/usr/bin/env python3
"""Quarto hook: compile the root taxonomy/ folder and stage it into site/taxonomy/."""

import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_DIR = ROOT / "taxonomy"
SITE_TAXONOMY_DIR = ROOT / "site" / "taxonomy"
SITE_BUILD_TAXONOMY_DIR = ROOT / "site" / "_build" / "taxonomy"

# Authoring, schema, benchmark, and documentation files that must NEVER be staged into site/taxonomy/.
INTERNAL_ONLY = {
    "README.md",
    "CHANGELOG.md",
    "requirements.txt",
    "ml-systems-taxonomy.yml",
    "mlsysbook-alignment.yml",
    "coverage-ledger.yml",
    "coverage",
    "agreement",
    "schema",
}

# Explicit allow-list of published Quarto pages, scripts, styles, and compiled outputs staged into site/taxonomy/.
PUBLISH_ITEMS = {
    "_workspace.qmd",
    "atlas.html",
    "atlas.js",
    "atlas.yml",
    "catalog.json",
    "downloads",
    "explorer.css",
    "explorer.js",
    "field-data.json",
    "field-focus",
    "field-preview.html",
    "field.css",
    "field.js",
    "field.qmd",
    "framework-atlas.svg",
    "framework-data.json",
    "framework-layers.svg",
    "framework-matrix.svg",
    "framework-reference.md",
    "framework-table.csv",
    "index.qmd",
    "maps",
    "study.css",
    "study.js",
    "taxonomy-reference.md",
    "taxonomy-study.qmd",
    "taxonomy.ttl",
    "vendor",
    "volume-maps.qmd",
    "volumes",
}


def _expected_field_diagram_names() -> set[str]:
    """Derive exact allowed root field-*.svg and field-*.dot names from compiled field-data.json."""
    names = {"field-overview.svg", "field-overview.dot"}
    field_data_path = TAXONOMY_DIR / "field-data.json"
    if field_data_path.is_file():
        field_data = json.loads(field_data_path.read_text(encoding="utf-8"))
        for view in field_data.get("views", []):
            if isinstance(view, dict) and isinstance(view.get("id"), str):
                vid = view["id"]
                names.add(f"field-{vid}.svg")
                names.add(f"field-{vid}.dot")
    return names


def _is_publishable(name: str, allowed_field_diagrams: set[str] | None = None) -> bool:
    if name in PUBLISH_ITEMS:
        return True
    diagrams = allowed_field_diagrams if allowed_field_diagrams is not None else _expected_field_diagram_names()
    return name in diagrams


def stage_site_taxonomy() -> None:
    allowed_diagrams = _expected_field_diagram_names()
    if SITE_TAXONOMY_DIR.exists():
        shutil.rmtree(SITE_TAXONOMY_DIR)
    SITE_TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)
    for item in sorted(TAXONOMY_DIR.iterdir()):
        if item.name.startswith(".") or item.name in INTERNAL_ONLY:
            continue
        if not _is_publishable(item.name, allowed_diagrams):
            raise RuntimeError(
                f"Unclassified item in taxonomy/: {item.name!r}; add to PUBLISH_ITEMS or INTERNAL_ONLY in build_concept_explorer.py"
            )
        dest = SITE_TAXONOMY_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def verify_staged_site_taxonomy(require_built: bool = False) -> None:
    """Verify staged site/taxonomy/ (or post-render site/_build/taxonomy/ when cleaned) against taxonomy/."""
    allowed_diagrams = _expected_field_diagram_names()
    if require_built and SITE_TAXONOMY_DIR.exists():
        raise RuntimeError(
            "Staged directory site/taxonomy/ still exists; run 'build_concept_explorer.py --clean' before '--check-staged'"
        )
    if not require_built and SITE_TAXONOMY_DIR.is_dir():
        for item in sorted(SITE_TAXONOMY_DIR.iterdir()):
            if item.name in INTERNAL_ONLY or not _is_publishable(item.name, allowed_diagrams):
                raise RuntimeError(f"Forbidden or unclassified item staged in site/taxonomy/: {item.name}")
        expected_rel_files: set[Path] = set()
        expected_rel_dirs: set[Path] = set()
        for src in sorted(TAXONOMY_DIR.rglob("*")):
            rel = src.relative_to(TAXONOMY_DIR)
            if any(part.startswith(".") for part in rel.parts):
                continue
            top = rel.parts[0]
            if top in INTERNAL_ONLY:
                continue
            if not _is_publishable(top, allowed_diagrams):
                raise RuntimeError(f"Unclassified item in taxonomy/: {top}")
            dst = SITE_TAXONOMY_DIR / rel
            if src.is_dir():
                expected_rel_dirs.add(rel)
                if not dst.is_dir():
                    raise RuntimeError(f"Missing staged directory: site/taxonomy/{rel}")
            elif src.is_file():
                expected_rel_files.add(rel)
                if not dst.is_file() or dst.read_bytes() != src.read_bytes():
                    raise RuntimeError(f"Staged file out of sync: site/taxonomy/{rel}")
        for dst in sorted(SITE_TAXONOMY_DIR.rglob("*")):
            rel = dst.relative_to(SITE_TAXONOMY_DIR)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if dst.is_file() and rel not in expected_rel_files:
                raise RuntimeError(f"Orphaned staged file in site/taxonomy/{rel}")
            if dst.is_dir() and rel not in expected_rel_dirs:
                raise RuntimeError(f"Orphaned staged directory in site/taxonomy/{rel}")
        return

    if SITE_BUILD_TAXONOMY_DIR.is_dir():
        quarto_generated_top = {"index.html", "search.json", "site_libs", "index_files"}
        expected_build_files: set[Path] = set()
        expected_html_files: set[Path] = {Path("index.html")}
        expected_build_dirs: set[Path] = set()
        for src in sorted(TAXONOMY_DIR.rglob("*")):
            rel = src.relative_to(TAXONOMY_DIR)
            if any(part.startswith(".") for part in rel.parts):
                continue
            top = rel.parts[0]
            if top in INTERNAL_ONLY:
                continue
            if not _is_publishable(top, allowed_diagrams):
                raise RuntimeError(f"Unclassified item in taxonomy/: {top}")
            dst = SITE_BUILD_TAXONOMY_DIR / rel
            if src.is_dir():
                expected_build_dirs.add(rel)
                if not dst.is_dir():
                    raise RuntimeError(f"Missing rendered taxonomy directory: site/_build/taxonomy/{rel}")
            elif src.is_file() and src.suffix == ".qmd":
                if not src.name.startswith("_"):
                    expected_html_files.add(rel.with_suffix(".html"))
            elif src.is_file() and src.suffix != ".qmd":
                expected_build_files.add(rel)
                if src.suffix == ".md" and not src.name.startswith("_"):
                    expected_html_files.add(rel.with_suffix(".html"))
                if not dst.is_file():
                    raise RuntimeError(f"Missing rendered file: site/_build/taxonomy/{rel}")
                dst_bytes = dst.read_bytes()
                if src.suffix == ".html":
                    dst_bytes = re.sub(rb"\?v=[0-9a-f]{8}(?=[\"'])", b"", dst_bytes)
                if dst_bytes != src.read_bytes():
                    raise RuntimeError(f"Rendered file out of sync: site/_build/taxonomy/{rel}")
        allowed_root_html = {p.name for p in expected_html_files if len(p.parts) == 1}
        for item in sorted(SITE_BUILD_TAXONOMY_DIR.iterdir()):
            if item.name in INTERNAL_ONLY:
                raise RuntimeError(f"Internal-only taxonomy item leaked into site/_build/taxonomy/: {item.name}")
            if (
                item.name not in quarto_generated_top
                and item.name not in allowed_root_html
                and not _is_publishable(item.name, allowed_diagrams)
            ):
                raise RuntimeError(f"Forbidden or unclassified item in site/_build/taxonomy/: {item.name}")
        for html_rel in sorted(expected_html_files):
            if not (SITE_BUILD_TAXONOMY_DIR / html_rel).is_file():
                raise RuntimeError(f"Missing rendered site/_build/taxonomy/{html_rel}")
        allowed_all_files = expected_build_files | expected_html_files
        for dst in sorted(SITE_BUILD_TAXONOMY_DIR.rglob("*")):
            rel = dst.relative_to(SITE_BUILD_TAXONOMY_DIR)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if rel.parts[0] in quarto_generated_top:
                continue
            if dst.is_file() and rel not in allowed_all_files:
                raise RuntimeError(f"Orphaned rendered file in site/_build/taxonomy/{rel}")
            if dst.is_dir() and not dst.name.endswith("_files") and rel not in expected_build_dirs:
                raise RuntimeError(f"Orphaned rendered directory in site/_build/taxonomy/{rel}")
        return

    raise RuntimeError("Neither site/taxonomy/ nor site/_build/taxonomy/ exists; run build_concept_explorer.py first")


def clean_site_taxonomy() -> None:
    if SITE_BUILD_TAXONOMY_DIR.is_dir():
        allowed_diagrams = _expected_field_diagram_names()
        for src in sorted(TAXONOMY_DIR.rglob("*")):
            rel = src.relative_to(TAXONOMY_DIR)
            if any(part.startswith(".") for part in rel.parts) or rel.parts[0] in INTERNAL_ONLY:
                continue
            if not _is_publishable(rel.parts[0], allowed_diagrams):
                continue
            dst = SITE_BUILD_TAXONOMY_DIR / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif src.is_file() and src.suffix != ".qmd":
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    if SITE_TAXONOMY_DIR.exists():
        shutil.rmtree(SITE_TAXONOMY_DIR)


if __name__ == "__main__":
    if "--clean" in sys.argv[1:]:
        # Only a whole-project render may remove the staged copy. A single-page render
        # (`quarto render taxonomy/index.qmd`) stats its input after post-render, so
        # cleaning then fails the render (2026-10-28). Quarto sets this for full renders.
        if os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1":
            clean_site_taxonomy()
        raise SystemExit(0)
    if "--check-staged" in sys.argv[1:]:
        verify_staged_site_taxonomy(require_built=True)
        raise SystemExit(0)
    if os.environ.get("TAXONOMY_PRESTAGED") == "1" and SITE_TAXONOMY_DIR.is_dir():
        verify_staged_site_taxonomy(require_built=False)
        raise SystemExit(0)
    sys.path.insert(0, str(ROOT / "binder"))
    from cli.commands.concepts import run

    if not run(ROOT, ["build"]):
        raise SystemExit(1)
    stage_site_taxonomy()
    verify_staged_site_taxonomy()

