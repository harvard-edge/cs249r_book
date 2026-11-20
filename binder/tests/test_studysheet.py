"""Chapter study sheets: each sheet's numbers are computed, every \\val is set, and colors come from its book.

Runs without LaTeX. The full build and layout check is books/shared/studysheet/build.py.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD = REPO_ROOT / "books/shared/studysheet/build.py"
SHEETS = sorted((REPO_ROOT / "books").glob("vol[1-4]/*/*_studysheet.tex"))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load(BUILD, "studysheet_build")


def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


@pytest.mark.parametrize("volume", [1, 2, 3, 4])
def test_accent_comes_from_the_volume_theme(volume):
    assert re.fullmatch(r"[0-9A-F]{6}", build.accent(volume))


def test_there_is_at_least_one_sheet():
    assert SHEETS


@pytest.mark.parametrize("tex", SHEETS, ids=lambda p: str(p.relative_to(REPO_ROOT / "books")))
def test_sheet_declares_its_own_volume(tex):
    assert build.volume_of(tex) == int(tex.parts[-3].removeprefix("vol"))


@pytest.mark.parametrize("tex", SHEETS, ids=lambda p: str(p.relative_to(REPO_ROOT / "books")))
def test_every_value_is_computed(tex):
    script = tex.with_name(tex.stem + "_numbers.py")
    assert script.exists(), f"{script.name} is missing"
    sys.path.insert(0, str(REPO_ROOT / "mlsysim"))
    try:
        values = _load(script, f"numbers_{tex.stem}").numbers()
    finally:
        sys.path.remove(str(REPO_ROOT / "mlsysim"))
    used = set(re.findall(r"\\val\{([^}]+)\}", _strip_comments(tex.read_text())))
    assert used, "a sheet prints its numbers through \\val"
    assert not used - set(values), f"\\val keys with no computed value: {sorted(used - set(values))}"
    bad = {k: v for k, v in values.items() if re.search(r"\b(nan|inf|None)\b", str(v))}
    assert not bad, f"values that did not compute: {bad}"


@pytest.mark.parametrize("tex", SHEETS, ids=lambda p: str(p.relative_to(REPO_ROOT / "books")))
def test_no_digits_typed_into_answers(tex):
    """Answers restate computed results, so every number in them must come through \\val."""
    body = _strip_comments(tex.read_text())
    answers = re.search(r"\\begin\{answers\}(.*?)\\end\{answers\}", body, flags=re.S)
    assert answers, "the back has an answers block"
    text = re.sub(r"\\val\{[^}]*\}|\\ans\{[^}]*\}|\\[A-Za-z]+|\$\\?[A-Za-z]+\$", "", answers.group(1))
    text = re.sub(r"\d+-bit", "", text)                              # "8-bit" names a format, not a result
    text = re.sub(r"\b[A-Za-z][\w-]*\d[\w-]*", "", text)                # names such as ResNet-50, GPT-2
    text = re.sub(r"C\{3\}|C\$\^\{?3\}?\$", "", text)               # the C³ taxonomy name
    text = re.sub(r"CO\$?_\{?2\}?\$?|CO\{2\}|CO₂", "", text)                 # CO2, a chemical name
    text = re.sub(r"\$?\^\{?[23]\}?\$?", "", text)                       # unit exponents: m/s^2, m^3
    assert not re.findall(r"\d", re.sub(r"\(\w\)", "", text)), f"typed digits in answers: {text.strip()[:200]}"


@pytest.mark.parametrize("tex", SHEETS, ids=lambda p: str(p.relative_to(REPO_ROOT / "books")))
def test_check_yourself_has_three_or_four_questions(tex):
    """studysheet.md: Check yourself holds 3-4 questions (added 2026-11-19 after two sheets grew a fifth)."""
    body = _strip_comments(tex.read_text())
    qs = re.search(r"\\begin\{qs\}(.*?)\\end\{qs\}", body, flags=re.S)
    assert qs, "the back has a Check yourself list"
    n = len(re.findall(r"\\item\b", qs.group(1)))
    assert 3 <= n <= 4, f"Check yourself has {n} questions; the rule allows 3-4"


# Publishing (added 2026-11-20 with the studysheets-*.yml workflows).

@pytest.mark.parametrize("vol", build.VOLUMES)
def test_every_sheet_has_a_place_in_book_order(vol):
    """--stage combines a volume in _quarto-pdf-volN.yml order; a sheet outside it stops the release."""
    assert sorted(build.book_order(vol)) == build.sheets(vol)


def test_live_volumes_link_to_the_live_release_and_others_to_dev():
    for tex in SHEETS:
        vol = tex.parents[1].name
        tag = build.LIVE_TAG if vol in build.LIVE_VOLUMES else build.DEV_TAG
        assert build.sheet_url(tex).endswith(f"/{tag}/{vol}_{tex.stem}.pdf")
    assert set(build.LIVE_VOLUMES) <= set(build.VOLUMES)


def test_release_asset_names_are_unique():
    """GitHub releases are flat: two volumes' sheets must never share a file name."""
    names = [build.asset_name(t) for t in SHEETS]
    assert len(names) == len(set(names))


def test_live_volumes_match_the_slides_release():
    """Study sheets go live with the same volumes as the slides (slides-publish-live.yml)."""
    wf = (REPO_ROOT / ".github/workflows/slides-publish-live.yml").read_text()
    slides_live = re.search(r'LIVE_VOLUMES:\s*"([^"]*)"', wf).group(1).split()
    assert tuple(slides_live) == build.LIVE_VOLUMES


# The Study Sheets page (added 2026-11-20): one per volume, in the HTML render list and the sidebar.

@pytest.mark.parametrize("vol", build.VOLUMES)
def test_study_sheets_page_is_in_the_volume_site(vol):
    page = f"{vol}/backmatter/studysheets.qmd"
    assert (REPO_ROOT / "books" / page).exists()
    assert "{#studysheet-list}" in (REPO_ROOT / "books" / page).read_text()
    config = (REPO_ROOT / f"books/config/_quarto-html-{vol}.yml").read_text()
    assert config.count(page) == 2, f"{page} must be in both render: and the sidebar"


@pytest.mark.parametrize("vol", build.VOLUMES)
def test_volume_list_covers_every_sheet_in_book_order(vol):
    sheets = build.book_order(vol)
    parts = build.book_parts(vol)
    assert [t.parent.name for t in sheets] == [t.parent.name for t in build.book_order(vol)]
    assert all(build.chapter_title(t) and "{" not in build.chapter_title(t) for t in sheets)
    assert set(parts) >= {t.parent.name for t in sheets}, "every sheet's chapter is in the sidebar"
    assert build.volume_url(vol).endswith(f"MLSysBook-StudySheets-Vol{vol[3:]}.pdf")


# The header QR code (added 2026-11-20) links each sheet to its chapter online.

@pytest.mark.parametrize("tex", SHEETS, ids=lambda p: p.stem)
def test_sheet_links_to_a_chapter_the_book_renders(tex):
    vol, chapter = tex.parents[1].name, tex.parent.name
    url = build.chapter_url(tex)
    assert url.startswith("https://") and url.endswith(f"/{chapter}/{chapter}.html")
    config = (REPO_ROOT / f"books/config/_quarto-html-{vol}.yml").read_text()
    assert f"{vol}/{chapter}/{chapter}.qmd" in config, f"{chapter} is not in the {vol} HTML render list"
