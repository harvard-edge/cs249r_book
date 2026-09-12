"""Unit tests for check_bold_definitions ensuring emphasis-bold.md invariants."""

from pathlib import Path
from binder.cli.commands._index_checks import check_bold_definitions


def test_bold_definitions_clean_case(tmp_path: Path):
    ch1 = tmp_path / "01_intro.qmd"
    ch1.write_text(
        "The **dual mandate**\\index{Dual mandate!definition} requires that systems balance accuracy.\n"
        "Here is a regular term with normal index: runtime\\index{Runtime!execution}.\n"
    )
    issues = check_bold_definitions(tmp_path)
    assert issues == []


def test_bold_definitions_missing_bold(tmp_path: Path):
    ch1 = tmp_path / "01_intro.qmd"
    ch1.write_text(
        "The dual mandate\\index{Dual mandate!definition} requires balance.\n"
    )
    issues = check_bold_definitions(tmp_path)
    assert len(issues) == 1
    assert issues[0].code == "def_missing_bold"


def test_bold_definitions_bold_index_not_def(tmp_path: Path):
    ch1 = tmp_path / "01_intro.qmd"
    ch1.write_text(
        "We consider **Apache Arrow**\\index{Apache Arrow!in-memory format} in this section.\n"
    )
    issues = check_bold_definitions(tmp_path)
    assert len(issues) == 1
    assert issues[0].code == "bold_index_not_def"


def test_bold_definitions_duplicate_definition(tmp_path: Path):
    ch1 = tmp_path / "01_intro.qmd"
    ch1.write_text(
        "The **ridge point**\\index{Ridge point!definition} is the roofline boundary.\n"
    )
    ch2 = tmp_path / "02_hardware.qmd"
    ch2.write_text(
        "Recall that the **ridge point**\\index{Ridge point!definition} is important.\n"
    )
    issues = check_bold_definitions(tmp_path)
    dup_issues = [i for i in issues if i.code == "dup_definition"]
    assert len(dup_issues) == 1
    assert "Duplicate formal definition of 'Ridge point'" in dup_issues[0].message


def test_bold_definitions_structural_labels_ignored(tmp_path: Path):
    ch1 = tmp_path / "01_intro.qmd"
    ch1.write_text(
        "**Context**:\\index{Context!callout} The system setup.\n"
        "1.  **Rule**: The primary invariant.\n"
        "[^fn1]: **Term (Acronym)**\\index{Term!footnote}: Definition in note.\n"
        ": **Table Title**: Caption text.\n"
    )
    issues = check_bold_definitions(tmp_path)
    assert issues == []
