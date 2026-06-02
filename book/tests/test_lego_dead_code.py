from book.cli.checks.lego_dead_code import audit_file


def _write(tmp_path, body, prose=""):
    chapter = tmp_path / "chapter.qmd"
    chapter.write_text(
        "\n```{python}\n# LEGO\n"
        + body
        + "\n```\n\n"
        + prose
        + "\n",
        encoding="utf-8",
    )
    return chapter


def test_ignores_wrapped_keyword_arguments(tmp_path):
    chapter = _write(
        tmp_path,
        "class Example:\n"
        "    n = 1_000_000\n"
        "    # ┌── 4. OUTPUT (Formatting) ─────────────────────────────────\n"
        "    value_str = fmt_count(\n"
        "        n,\n"
        "        scale='M',\n"
        "        precision=0,\n"
        "        scale_style='word',\n"
        "    )\n",
        "Value: `{python} Example.value_str`.",
    )

    assert audit_file(chapter) == []


def test_flags_unused_output_variable(tmp_path):
    chapter = _write(
        tmp_path,
        "class Example:\n"
        "    # ┌── 4. OUTPUT (Formatting) ─────────────────────────────────\n"
        "    value_str = fmt(1)\n",
    )

    violations = audit_file(chapter)
    assert [v.code for v in violations] == ["lego_dead_code"]
    assert violations[0].message == (
        "Exported LEGO variable `Example.value_str` is never used in prose."
    )
