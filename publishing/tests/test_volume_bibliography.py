"""Each new volume's bibliography is self-contained and current.

Volumes I and II share contents/references.bib. Volumes III and IV each carry a
dedicated bibliography so a reader can check the sources a new volume rests on
without 1,500 unrelated entries in the way (2026-09-07).

The invariant runs both directions: no citation resolves to nothing, and no
entry sits in the file uncited. The first direction is what broke the vol3 HTML
build (six citation keys were typos of real entries); the second is what makes
the file readable.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONTENTS = REPO / "publishing" / "quarto" / "contents"
CONFIG = REPO / "publishing" / "quarto" / "config"
BUILDER = REPO / "publishing" / "tools" / "scripts" / "structure" / "build_volume_bib.py"

DEDICATED = ("vol3", "vol4")
SHARED = ("vol1", "vol2")

sys.path.insert(0, str(BUILDER.parent))
import build_volume_bib as bvb  # noqa: E402


@pytest.mark.parametrize("vol", DEDICATED)
def test_every_citation_resolves(vol: str) -> None:
    _, missing = bvb.render(vol)
    assert not missing, (
        f"{vol} cites keys that exist in no bibliography:\n  " + "\n  ".join(missing)
    )


@pytest.mark.parametrize("vol", DEDICATED)
def test_no_uncited_entries(vol: str) -> None:
    bib = CONTENTS / f"references-{vol}.bib"
    entries = set(bvb.parse_entries(bib))
    cited = set(bvb.cited_keys(vol))
    unused = sorted(entries - cited)
    assert not unused, (
        f"{bib.name} carries {len(unused)} entries no chapter cites; move them to "
        f"references-{vol}-staged.bib:\n  " + "\n  ".join(unused[:20])
    )


@pytest.mark.parametrize("vol", DEDICATED)
def test_bibliography_is_current(vol: str) -> None:
    r = subprocess.run(
        ["python3", str(BUILDER), vol, "--check"], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr or r.stdout


@pytest.mark.parametrize("vol", DEDICATED + SHARED)
def test_config_points_at_the_right_bibliography(vol: str) -> None:
    """vol1/vol2 use the shared bib; vol3/vol4 use only their own."""
    expected = "references.bib" if vol in SHARED else f"references-{vol}.bib"
    problems = []
    for fmt in ("html", "pdf", "epub"):
        cfg = CONFIG / f"_quarto-{fmt}-vol{vol[-1]}.yml"
        if not cfg.exists():
            continue
        text = cfg.read_text(encoding="utf-8")
        m = re.search(r"^bibliography:(.*?)(?=^\S)", text, re.M | re.S)
        assert m, f"{cfg.name} declares no bibliography"
        found = re.findall(r"contents/(references[a-z0-9-]*\.bib)", m.group(1))
        if found != [expected]:
            problems.append(f"{cfg.name}: {found} (expected ['{expected}'])")
    assert not problems, "Bibliography wiring drift:\n  " + "\n  ".join(problems)


def test_staged_bib_is_not_wired_into_any_build() -> None:
    """The staging file holds uncited sources and must stay out of the builds."""
    staged = CONTENTS / "references-vol4-staged.bib"
    if not staged.exists():
        pytest.skip("no staged bibliography")
    wired = [
        c.name
        for c in CONFIG.rglob("*.yml")
        if "references-vol4-staged.bib" in c.read_text(encoding="utf-8")
    ]
    assert not wired, f"staged bibliography referenced by: {wired}"
