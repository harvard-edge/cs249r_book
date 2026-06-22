from types import SimpleNamespace

from book.cli.commands.validate import ValidateCommand


def _command(tmp_path):
    return ValidateCommand(
        config_manager=SimpleNamespace(book_dir=tmp_path),
        chapter_discovery=None,
    )


def test_pipe_table_caption_below_table_passes(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """| **Dimension** | **What it measures** |
|:--------------|:---------------------|
| Throughput    | Tokens per second    |

: **Efficiency Frontier Dimensions**: Performance trade-offs. {#tbl-frontier}
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_table_caption_position(qmd)

    assert result.issues == []


def test_pipe_table_caption_above_table_fails(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """: **Efficiency Frontier Dimensions**: Performance trade-offs. {#tbl-frontier}

| **Dimension** | **What it measures** |
|:--------------|:---------------------|
| Throughput    | Tokens per second    |
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_table_caption_position(qmd)

    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.code == "table_caption_before_pipe_table"
    assert issue.line == 1
    assert "table header starts at line 3" in issue.message


def test_caption_between_two_tables_is_not_treated_as_top_caption(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """| A | B |
|---|---|
| 1 | 2 |

: **First Table**: Caption for the first table. {#tbl-first}

| C | D |
|---|---|
| 3 | 4 |

: **Second Table**: Caption for the second table. {#tbl-second}
""",
        encoding="utf-8",
    )

    result = _command(tmp_path)._run_table_caption_position(qmd)

    assert result.issues == []
