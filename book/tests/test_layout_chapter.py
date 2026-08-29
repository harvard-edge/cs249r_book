"""Tests for mapped isolated chapter layout builds."""

from pathlib import Path

from book.cli.commands.layout_chapter import (
    _correct_custom_callout_tex,
    mapped_source,
    parse_aux,
)


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
\begin{fbxSimple}{callout-notebook}{Napkin Math 1.3:}{Cost estimate}
\phantomsection\label{nbk-example-cost}
See napkin math \hyperref[nbk-example-cost]{1.3}.
"""
    corrected, headings, references = _correct_custom_callout_tex(tex, 9)
    assert "{Napkin Math 9.3:}" in corrected
    assert r"\hyperref[nbk-example-cost]{9.3}" in corrected
    assert headings == 1
    assert references == 1


def test_correct_custom_callout_tex_leaves_external_reference_unchanged():
    tex = r"See \hyperref[nbk-other-chapter-cost]{4.2}."
    corrected, headings, references = _correct_custom_callout_tex(tex, 9)
    assert corrected == tex
    assert headings == 0
    assert references == 0
