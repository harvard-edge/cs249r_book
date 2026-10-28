#!/usr/bin/env python3
"""Quarto hook: compile the root taxonomy/ folder and stage it into site/taxonomy/."""

import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_DIR = ROOT / "taxonomy"
SITE_TAXONOMY_DIR = ROOT / "site" / "taxonomy"
SKIP = {
    "README.md",
    "CHANGELOG.md",
    "requirements.txt",
    "ml-systems-field.yml",
    "ml-systems-framework.yml",
    "ml-systems-guides.yml",
    "ml-systems-taxonomy.yml",
    "mlsysbook-alignment.yml",
    "coverage-ledger.yml",
    "coverage",
    "agreement",
}


def stage_site_taxonomy() -> None:
    if SITE_TAXONOMY_DIR.exists():
        shutil.rmtree(SITE_TAXONOMY_DIR)
    SITE_TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)
    for item in TAXONOMY_DIR.iterdir():
        if item.name in SKIP or item.name.startswith("."):
            continue
        dest = SITE_TAXONOMY_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def clean_site_taxonomy() -> None:
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
    sys.path.insert(0, str(ROOT / "binder"))
    from cli.commands.concepts import run

    if not run(ROOT, ["build"]):
        raise SystemExit(1)
    stage_site_taxonomy()

