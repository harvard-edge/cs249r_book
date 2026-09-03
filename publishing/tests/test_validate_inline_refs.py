from types import SimpleNamespace

from book.cli.commands.validate import ValidateCommand


def test_markdown_table_caption_inline_python_is_allowed(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        """```{python}
class CaptionCalc:
    value_str = "1.5"
```

| Metric | Value |
|:-------|------:|
| Speedup | `{python} CaptionCalc.value_str` |

: **Computed Caption**: The computed speedup is `{python} CaptionCalc.value_str` times. {#tbl-computed-caption}
""",
        encoding="utf-8",
    )

    command = ValidateCommand(
        config_manager=SimpleNamespace(book_dir=tmp_path),
        chapter_discovery=None,
    )

    result = command._run_inline_refs(qmd, check_patterns=True)

    assert not [
        issue
        for issue in result.issues
        if issue.code == "caption_inline_python"
    ]
