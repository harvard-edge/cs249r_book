"""Tests for mapped isolated chapter layout builds."""

from pathlib import Path

from book.cli.commands.layout_chapter import (
    H1_RE,
    _appendix_counter_hook,
    _correct_custom_callout_tex,
    _fragment_owner,
    _inject_folio_after_h1,
    _mainmatter_counter_hook,
    _plain_toc_title,
    _volume_citekeys,
    mapped_source,
    parse_aux,
    parse_toc_chapters,
    roman_to_int,
)


def test_h1_parser_accepts_attributes_after_identifier():
    match = H1_RE.search(
        '# Algorithm Foundations {#sec-appdx-algorithm-foundations dropcap="chapter-opening"}'
    )
    assert match is not None
    assert match.group(1) == "sec-appdx-algorithm-foundations"


def test_mainmatter_counter_hook_restores_numeric_chapter_and_folio():
    text = _mainmatter_counter_hook(11, 543)["text"]
    assert r"\setcounter{chapter}{10}" in text
    assert r"\setcounter{page}{543}" in text
    assert r"\renewcommand{\mainmatter}" in text


def test_appendix_counter_hook_restores_letter_position_and_folio():
    text = _appendix_counter_hook(3, 879)["text"]
    assert r"\setcounter{chapter}{2}" in text
    assert r"\setcounter{page}{879}" in text
    assert r"\renewcommand{\appendix}" in text


def test_roman_folio_conversion():
    assert roman_to_int("xiii") == 13
    assert roman_to_int("xxix") == 29


def test_parse_toc_chapters_handles_numbered_and_unnumbered(tmp_path: Path):
    aux = tmp_path / "book.aux"
    aux.write_text(
        "\\@writefile{toc}{\\contentsline {chapter}{Foreword}{xiii}{chapter*.2}\\protected@file@percent }\n"
        "\\@writefile{toc}{\\contentsline {chapter}{\\numberline {11}Hardware Acceleration}{529}{chapter.11}\\protected@file@percent }\n",
        encoding="utf-8",
    )
    records = parse_toc_chapters(aux)
    assert records[0] == {
        "title": "Foreword",
        "page": "xiii",
        "anchor": "chapter*.2",
    }
    assert _plain_toc_title(records[1]["title"]) == "Hardware Acceleration"


def test_fragment_owner_routes_include_only_sources(tmp_path: Path):
    volume_root = tmp_path / "contents" / "vol1"
    assert _fragment_owner(volume_root / "frontmatter" / "_conventions.qmd", volume_root) == volume_root / "frontmatter" / "about.qmd"
    assert _fragment_owner(volume_root / "frontmatter" / "_notation_body.qmd", volume_root) == volume_root / "frontmatter" / "notation.qmd"
    assert _fragment_owner(volume_root / "chapter.qmd", volume_root) is None


def test_inject_folio_after_unnumbered_h1():
    mapped = _inject_folio_after_h1("# Foreword {.unnumbered}\n\nText.\n", 13, roman=True)
    assert "# Foreword {.unnumbered}\n\n```{=latex}" in mapped
    assert r"\pagenumbering{roman}" in mapped
    assert r"\setcounter{page}{13}" in mapped


def test_volume_citekeys_follows_includes_and_excludes_xrefs(tmp_path: Path):
    chapter = tmp_path / "chapter.qmd"
    fragment = tmp_path / "_fragment.qmd"
    chapter.write_text(
        "See [@source] and @sec-local.\n{{< include _fragment.qmd >}}\n",
        encoding="utf-8",
    )
    fragment.write_text("Also [@included].\n", encoding="utf-8")
    assert _volume_citekeys(tmp_path, ["chapter.qmd"]) == ["included", "source"]


def test_parse_aux_extracts_number_page_and_anchor(tmp_path: Path):
    aux = tmp_path / "book.aux"
    aux.write_text(
        "\\newlabel{sec-example}{{2.3}{47}{Example}{section.2.3}{}}\n"
        "\\newlabel{fig-device}{{2.4}{49}{Device}{figure.caption.19}{}}\n",
        encoding="utf-8",
    )
    labels = parse_aux(aux)
    assert labels["sec-example"] == {
        "number": "2.3",
        "page": "47",
        "anchor": "section.2.3",
    }
    assert labels["fig-device"]["number"] == "2.4"


def test_mapped_source_maps_only_external_prose_references():
    source = """# Workflow {#sec-workflow}

See @sec-workflow, @Sec-external, and @fig-device.
`@fig-device` remains literal.
<!-- @fig-device remains literal. -->

```python
token = "@fig-device"
```
"""
    labels = {
        "sec-external": {"number": "1.3", "page": "12", "anchor": "section.1.3"},
        "fig-device": {"number": "4.2", "page": "99", "anchor": "figure.caption.8"},
    }
    mapped, count, missing = mapped_source(source, labels)
    assert "@sec-workflow" in mapped
    assert "Section\u00a01.3" in mapped
    assert "figure\u00a04.2" in mapped
    assert "`@fig-device`" in mapped
    assert "<!-- @fig-device remains literal. -->" in mapped
    assert 'token = "@fig-device"' in mapped
    assert count == 2
    assert missing == []


def test_mapped_source_recognizes_cell_labels_and_reports_missing():
    source = """```{python}
#| label: fig-local
```

Compare @fig-local with @tbl-not-in-aux.
"""
    mapped, count, missing = mapped_source(source, {})
    assert mapped == source
    assert count == 0
    assert missing == ["tbl-not-in-aux"]


def test_correct_custom_callout_tex_updates_heading_and_matching_reference():
    tex = r"""
\protect\phantomsection\label{nbk-example-cost}
\begin{fbxSimple}{callout-notebook}{napkin math 1.3:}{Cost estimate}
\phantomsection\label{nbk-example-cost}
See napkin math \hyperref[nbk-example-cost]{1.3}.
"""
    corrected, headings, references = _correct_custom_callout_tex(tex, 9)
    assert "{napkin math 9.3:}" in corrected
    assert r"\hyperref[nbk-example-cost]{9.3}" in corrected
    assert headings == 1
    assert references == 1


def test_correct_custom_callout_tex_accepts_appendix_letter():
    tex = r"""
\protect\phantomsection\label{nbk-example-cost}
\begin{fbxSimple}{callout-notebook}{notebook A.3:}{Cost estimate}
\phantomsection\label{nbk-example-cost}
See notebook \hyperref[nbk-example-cost]{A.3}.
"""
    corrected, headings, references = _correct_custom_callout_tex(tex, "C")
    assert "{notebook C.3:}" in corrected
    assert r"\hyperref[nbk-example-cost]{C.3}" in corrected
    assert headings == 1
    assert references == 1


def test_correct_custom_callout_tex_accepts_roman_part_prefix():
    tex = r"""
\protect\phantomsection\label{pri-example-cost}
\begin{fbxSimple}{callout-principle}{principle I.2:}{Cost estimate}
\phantomsection\label{pri-example-cost}
See principle \hyperref[pri-example-cost]{I.2}.
"""
    corrected, headings, references = _correct_custom_callout_tex(tex, "III")
    assert "{principle III.2:}" in corrected
    assert r"\hyperref[pri-example-cost]{III.2}" in corrected
    assert headings == 1
    assert references == 1


def test_correct_custom_callout_tex_expands_local_part_number():
    tex = r"""
\protect\phantomsection\label{pri-example-cost}
\begin{fbxSimple}{callout-principle}{Principle 1:}{Cost estimate}
\phantomsection\label{pri-example-cost}
See principle \hyperref[pri-example-cost]{1}.
"""
    corrected, headings, references = _correct_custom_callout_tex(tex, "III")
    assert "{Principle III.1:}" in corrected
    assert r"\hyperref[pri-example-cost]{III.1}" in corrected
    assert headings == 1
    assert references == 1


def test_correct_custom_callout_tex_leaves_external_reference_unchanged():
    tex = r"See \hyperref[nbk-other-chapter-cost]{4.2}."
    corrected, headings, references = _correct_custom_callout_tex(tex, 9)
    assert corrected == tex
    assert headings == 0
    assert references == 0
