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
