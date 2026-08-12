from types import SimpleNamespace

from book.cli.commands.validate import ValidateCommand


def _command(tmp_path):
    return ValidateCommand(
        config_manager=SimpleNamespace(book_dir=tmp_path),
        chapter_discovery=None,
    )


def test_body_footnote_ref_with_colon_is_not_a_definition(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """Body prose introduces a term[^fn-example-local]: the inline gloss continues.

[^fn-example-local]: **Example**: The real footnote definition appears here.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_footnote_cross_chapter(tmp_path)

    assert result.issues == []


def test_cross_chapter_duplicate_footnote_definitions_are_flagged(tmp_path):
    chapter_a = tmp_path / "chapter_a.qmd"
    chapter_a.write_text(
        """A sentence with a footnote[^fn-shared-context].

[^fn-shared-context]: **Shared**: First definition.
""",
        encoding="utf-8",
    )
    chapter_b = tmp_path / "chapter_b.qmd"
    chapter_b.write_text(
        """Another sentence with a footnote[^fn-shared-context].

[^fn-shared-context]: **Shared**: Second definition.
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_footnote_cross_chapter(tmp_path)

    assert [issue.code for issue in result.issues] == [
        "cross_chapter_footnote",
        "cross_chapter_footnote",
    ]
