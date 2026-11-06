"""Publisher metadata must use the current build's labels and folios."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "books/shared/scripts/mit_press/generate_figure_list.py"
spec = importlib.util.spec_from_file_location("figure_list_export", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def fixture_project(tmp_path, monkeypatch):
    books = tmp_path / "books"
    script = books / "shared/scripts/mit_press/generate_figure_list.py"
    script.parent.mkdir(parents=True)
    monkeypatch.setattr(exporter, "__file__", str(script))
    (books / "_quarto.yml").write_text("book:\n  chapters: [chapter.qmd]\n")
    (books / "index.tex").write_text("Current TeX")
    (books / "chapter.qmd").write_text(
        '# Chapter\n\n::: {#fig-second fig-cap="Second source figure" fig-alt="Second"}\n:::\n\n'
        '::: {#fig-first fig-cap="First source figure" fig-alt="First"}\n:::\n'
    )
    output = books / "_build/pdf-vol1"
    monkeypatch.setattr(exporter.sys, "argv", ["generate_figure_list.py", "--output", str(output)])
    return books, output


def test_export_resolves_books_root_and_matches_labels_not_positions(tmp_path, monkeypatch):
    books, output = fixture_project(tmp_path, monkeypatch)
    (books / "index.aux").write_text(
        r"\newlabel{fig-first}{{2.3}{17}{Caption}{figure.1}{}}" + "\n"
        + r"\newlabel{fig-second}{{P.1}{xx}{Caption}{figure.2}{}}" + "\n"
    )
    exporter.main()
    result = (output / "FIGURE_LIST.txt").read_text()
    assert "Figure P.1 (Page xx)\nLabel: fig-second" in result
    assert "Figure 2.3 (Page 17)\nLabel: fig-first" in result
    assert "?" not in result


def test_missing_aux_cannot_produce_a_successful_empty_export(tmp_path, monkeypatch):
    fixture_project(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="Current PDF AUX"):
        exporter.main()


def test_numbered_frontmatter_and_part_figures_are_included(tmp_path):
    part = tmp_path / "vol1/parts/principles.qmd"
    part.parent.mkdir(parents=True)
    (tmp_path / "_quarto.yml").write_text("book:\n  chapters: [index.qmd, vol1/parts/principles.qmd]\n")
    (tmp_path / "index.qmd").write_text('# Preface {.unnumbered}\n\n::: {#fig-dependencies fig-cap="Dependencies"}\n:::\n')
    part.write_text('# Principles {.unnumbered}\n\n::: {#fig-chain fig-cap="Chain"}\n:::\n')
    figures = exporter.extract_qmd_figures(tmp_path)
    assert {row["label"] for row in figures} == {"fig-dependencies", "fig-chain"}
    assert {row["chapter"] for row in figures} == {"Preface", "Principles"}


def test_chapter_export_ignores_other_target_and_stale_aux(tmp_path):
    import os

    (tmp_path / '_quarto.yml').write_text('book:\n  output-file: appendix_control\n')
    tex = tmp_path / 'appendix_control.tex'
    tex.write_text('Current chapter')
    os.utime(tex, (1_700_000_002, 1_700_000_002))
    stale = tmp_path / 'appendix_control.aux'
    stale.write_text('Old chapter labels')
    os.utime(stale, (1_700_000_000, 1_700_000_000))
    (tmp_path / 'Machine-Learning-Systems-Vol4.aux').write_text('Unrelated full volume')
    output = tmp_path / 'output'
    output.mkdir()
    assert exporter.find_figure_aux(tmp_path, output) is None
    current = output / 'appendix_control.aux'
    current.write_text('Current labels')
    assert exporter.find_figure_aux(tmp_path, output) == current


def test_chapter_export_rejects_newer_unrelated_index_aux(tmp_path):
    (tmp_path / '_quarto.yml').write_text('book:\n  output-file: appendix_control\n')
    (tmp_path / 'appendix_control.tex').write_text('Current chapter')
    (tmp_path / 'index.aux').write_text('Newer unrelated metadata')
    output = tmp_path / 'output'
    output.mkdir()
    assert exporter.find_figure_aux(tmp_path, output) is None


def test_index_tex_still_supports_renamed_preserved_target_aux(tmp_path):
    (tmp_path / '_quarto.yml').write_text('book:\n  output-file: Book\n')
    (tmp_path / 'index.tex').write_text('Current Quarto source')
    aux = tmp_path / 'index.aux'
    aux.write_text('Current metadata')
    output = tmp_path / 'output'
    output.mkdir()
    assert exporter.find_figure_aux(tmp_path, output) == aux
    preserved = output / 'Book.aux'
    preserved.write_text(aux.read_text())
    aux.unlink()
    assert exporter.find_figure_aux(tmp_path, output) == preserved
