"""Enforce the shared content skeleton across vol1-vol4.

Added 2026-09-07 after an audit found vol3 carrying 24 orphaned outline stubs,
vol4 carrying 4 orphaned figure directories, and vol4 using a different skeleton
(chapters/, front-matter/, appendix/, figures/) from the other three volumes.

These checks are cheap and catch the drift class that produced that state:
a chapter dir whose .qmd does not match its name, a config pointing at a file
that does not exist, a file on disk that no config renders, and a volume whose
top-level layout diverges from the shared skeleton.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
QUARTO = REPO / "publishing"  / "books"
CONTENTS = QUARTO
CONFIG = QUARTO / "config"

VOLUMES = ("vol1", "vol2", "vol3", "vol4")
RESERVED = {"frontmatter", "backmatter", "parts"}

QMD_REF = re.compile(r"contents/[A-Za-z0-9_./-]+\.qmd")
INCLUDE = re.compile(r"\{\{<\s*include\s+([^>}]+?)\s*>\}\}")


def config_refs() -> set[str]:
    """Every contents/*.qmd path any Quarto config mentions."""
    refs: set[str] = set()
    for cfg in CONFIG.rglob("*.yml"):
        refs.update(QMD_REF.findall(cfg.read_text(encoding="utf-8")))
    return refs


def include_refs() -> set[str]:
    """Every contents/*.qmd pulled in via a Quarto {{< include >}} shortcode."""
    refs: set[str] = set()
    for qmd in CONTENTS.rglob("*.qmd"):
        for raw in INCLUDE.findall(qmd.read_text(encoding="utf-8", errors="ignore")):
            target = (qmd.parent / raw.strip()).resolve()
            try:
                refs.add(str(target.relative_to(QUARTO)))
            except ValueError:
                pass
    return refs


@pytest.mark.parametrize("vol", VOLUMES)
def test_chapter_dir_matches_its_qmd(vol: str) -> None:
    """Every chapter directory holds <slug>/<slug>.qmd.

    publishing/cli/commands/build.py builds chapter paths from this invariant.
    """
    bad = []
    for d in sorted((CONTENTS / vol).iterdir()):
        if not d.is_dir() or d.name in RESERVED:
            continue
        if not (d / f"{d.name}.qmd").exists():
            bad.append(f"{vol}/{d.name} has no {d.name}.qmd")
    assert not bad, "Chapter dir/file name mismatch:\n  " + "\n  ".join(bad)


def test_every_config_path_exists() -> None:
    """No Quarto config points at a .qmd that is not on disk."""
    missing = sorted(r for r in config_refs() if not (QUARTO / r).exists())
    assert not missing, "Config references missing files:\n  " + "\n  ".join(missing)


def test_no_orphan_qmd_in_volumes() -> None:
    """Every .qmd inside a volume is rendered by some config or included by another .qmd.

    An orphan is a file no build path reaches. That is how vol3 accumulated 24
    superseded outline stubs and vol4 accumulated 4 dead figure directories.
    """
    live = config_refs() | include_refs()
    orphans = []
    for vol in VOLUMES:
        for qmd in sorted((CONTENTS / vol).rglob("*.qmd")):
            rel = str(qmd.relative_to(QUARTO))
            if rel not in live:
                orphans.append(rel)
    assert not orphans, "Orphaned .qmd files (rendered by nothing):\n  " + "\n  ".join(orphans)


@pytest.mark.parametrize("vol", VOLUMES)
def test_shared_skeleton(vol: str) -> None:
    """Each volume carries the same top-level layout."""
    root = CONTENTS / vol
    problems = []
    for required in ("index.qmd", "README.md"):
        if not (root / required).exists():
            problems.append(f"{vol}/ is missing {required}")
    for required in ("frontmatter", "parts", "backmatter"):
        if not (root / required).is_dir():
            problems.append(f"{vol}/ is missing the {required}/ directory")
    if not (root / "backmatter" / "references.qmd").exists():
        problems.append(f"{vol}/backmatter/ is missing references.qmd")
    # Chapter assets live in images/, never figures/.
    for figures in root.rglob("figures"):
        if figures.is_dir():
            problems.append(f"{figures.relative_to(CONTENTS)} should be named images/")
    # No nesting layer between the volume root and its chapters.
    if (root / "chapters").exists():
        problems.append(f"{vol}/chapters/ should not exist; chapters sit at the volume root")
    assert not problems, "Skeleton drift:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("vol", VOLUMES)
def test_no_stray_build_artifacts(vol: str) -> None:
    """Quarto build artifacts are never committed into the content tree."""
    strays = [
        str(p.relative_to(CONTENTS))
        for p in sorted((CONTENTS / vol).rglob("*"))
        if p.is_file() and p.suffix in {".quarto_ipynb", ".ipynb_checkpoints"}
    ]
    assert not strays, "Build artifacts in content tree:\n  " + "\n  ".join(strays)


def test_structure_manifest_is_current() -> None:
    """contents/STRUCTURE.md matches what the PDF configs say."""
    import subprocess

    gen = REPO / "publishing" / "tools" / "scripts" / "structure" / "gen_structure.py"
    r = subprocess.run(["python3", str(gen), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr or r.stdout
