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
    text = re.sub(r"\$?\^\{?[23]\}?\$?", "", text)                       # unit exponents: m/s^2, m^3
    assert not re.findall(r"\d", re.sub(r"\(\w\)", "", text)), f"typed digits in answers: {text.strip()[:200]}"
