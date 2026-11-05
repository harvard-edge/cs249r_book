from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from binder.cli.core.config import write_active_config

SCRIPT = Path(__file__).resolve().parents[2] / "books" / "shared" / "scripts" / "save_latex_log.py"
SPEC = importlib.util.spec_from_file_location("save_latex_log", SCRIPT)
assert SPEC and SPEC.loader
save_latex_log = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(save_latex_log)


def test_active_volume_reads_binder_config_source(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    volume_config = config_dir / "_quarto-pdf-vol2.yml"
    volume_config.write_text("project: {}\n", encoding="utf-8")
    write_active_config(tmp_path / "_quarto.yml", volume_config, tmp_path)

    assert save_latex_log._active_volume(tmp_path) == "vol2"


@pytest.mark.parametrize("volume", ["vol1", "vol2", "vol3", "vol4"])
def test_active_volume_reads_direct_container_config(tmp_path, volume):
    (tmp_path / "_quarto.yml").write_text(f"project:\n  output-dir: _build/pdf-{volume}\n")
    assert save_latex_log._active_volume(tmp_path) == volume


def test_html_config_is_not_treated_as_pdf(tmp_path):
    (tmp_path / "_quarto.yml").write_text("# binder: config/_quarto-html-vol4.yml\nproject:\n  output-dir: _build/html-vol4\n")
    assert save_latex_log._active_volume(tmp_path) is None


def test_postrender_finds_books_root_and_current_index_intermediates(tmp_path, monkeypatch):
    books = tmp_path / "books"
    script = books / "shared/scripts/save_latex_log.py"
    script.parent.mkdir(parents=True)
    monkeypatch.setattr(save_latex_log, "__file__", str(script))
    (books / "_quarto.yml").write_text("project:\n  output-dir: _build/pdf-vol4\n")
    (books / "index.tex").write_text("Current render source")
    (books / "index.aux").write_text("Current cross-reference metadata")
    (books / "index.log").write_text("Current LaTeX build log")
    def should_not_regenerate(*args):
        raise AssertionError("Current build intermediates must be reused")
    monkeypatch.setattr(save_latex_log, "_regenerate_auxiliary_files", should_not_regenerate)
    save_latex_log.main()
    output = books / "_build/pdf-vol4"
    assert (output / "latex-build.log").read_text() == "Current LaTeX build log"
    assert (output / "Machine-Learning-Systems-Vol4.aux").read_text() == "Current cross-reference metadata"


def test_find_intermediate_accepts_quarto_index_stem(tmp_path: Path) -> None:
    aux = tmp_path / "index.aux"
    aux.write_text("\\newlabel{sec-test}{{1}{1}}\n", encoding="utf-8")

    assert save_latex_log._find_intermediate(tmp_path, "Book-Vol2", ".aux") == aux


def test_current_index_is_preferred_over_stale_retained_stem(tmp_path):
    old = tmp_path / "Book-Vol2.tex"
    old.write_text("Stale previous build")
    current = tmp_path / "index.tex"
    current.write_text("Current Quarto render")
    os.utime(old, (1_700_000_000, 1_700_000_000))
    os.utime(current, (1_700_000_002, 1_700_000_002))
    assert save_latex_log._find_intermediate(tmp_path, "Book-Vol2", ".tex") == current


@pytest.mark.parametrize("age_seconds,expected", [(-2, False), (0, True), (2, True)])
def test_intermediate_must_be_at_least_as_new_as_retained_tex(
    tmp_path: Path, age_seconds: int, expected: bool
) -> None:
    tex = tmp_path / "Book-Vol2.tex"
    aux = tmp_path / "Book-Vol2.aux"
    tex.write_text("new source", encoding="utf-8")
    aux.write_text("stale metadata", encoding="utf-8")
    # Container overlay filesystems can give rapid writes identical mtimes.
    # Construct the required ordering rather than depending on clock resolution.
    timestamp = 1_700_000_000
    os.utime(tex, (timestamp, timestamp))
    os.utime(aux, (timestamp + age_seconds, timestamp + age_seconds))

    assert save_latex_log._is_current_for_tex(aux, tex) is expected


def test_regenerate_auxiliary_files_uses_draft_mode(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "Book-Vol2.tex").write_text("document", encoding="utf-8")
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(save_latex_log.subprocess, "run", fake_run)

    assert save_latex_log._regenerate_auxiliary_files(tmp_path, "Book-Vol2")
    assert len(observed) == 3
    assert all("-draftmode" in command for command, _ in observed)
    assert all(command[-1] == "Book-Vol2.tex" for command, _ in observed)
    assert all(kwargs["cwd"] == tmp_path for _, kwargs in observed)


def test_regenerate_auxiliary_files_stops_after_failed_pass(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "Book-Vol2.tex").write_text("document", encoding="utf-8")
    returncodes = iter((0, 1, 0))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type("Completed", (), {"returncode": next(returncodes)})()

    monkeypatch.setattr(save_latex_log.subprocess, "run", fake_run)

    assert not save_latex_log._regenerate_auxiliary_files(tmp_path, "Book-Vol2")
    assert len(calls) == 2
