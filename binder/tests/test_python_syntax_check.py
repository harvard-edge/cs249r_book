from types import SimpleNamespace

from binder.cli.commands.validate import ValidateCommand


def test_python_syntax_reports_qmd_line_after_chunk_directive(tmp_path):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(
        "```{python}\n"
        "#| echo: false\n"
        "value = 1\n"
        "plain prose inside code\n"
        "```\n",
        encoding="utf-8",
    )

    command = ValidateCommand(
        config_manager=SimpleNamespace(book_dir=tmp_path),
        chapter_discovery=None,
    )

    result = command._run_python_syntax(qmd)

    assert [(issue.code, issue.line) for issue in result.issues] == [
        ("python_syntax", 4)
    ]
