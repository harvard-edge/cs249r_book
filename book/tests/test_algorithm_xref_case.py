from types import SimpleNamespace

from book.cli.commands.validate import ValidateCommand


def _run_xref_case(tmp_path, body):
    qmd = tmp_path / "chapter.qmd"
    qmd.write_text(body, encoding="utf-8")
    command = ValidateCommand(
        config_manager=SimpleNamespace(book_dir=tmp_path),
        chapter_discovery=None,
    )
    return command._run_xref_sentence_start_case(tmp_path)


def test_algorithm_refs_follow_sentence_position(tmp_path):
    result = _run_xref_case(
        tmp_path,
        "\n".join(
            [
                "@Alg-start states the loop.",
                "The implementation follows @alg-mid.",
                "The transition is mechanical: @Alg-after-colon shows it.",
            ]
        ),
    )

    assert not result.issues


def test_algorithm_ref_case_violations_are_flagged(tmp_path):
    result = _run_xref_case(
        tmp_path,
        "\n".join(
            [
                "@alg-start states the loop.",
                "The implementation follows @Alg-mid.",
            ]
        ),
    )

    assert [issue.code for issue in result.issues] == [
        "xref_sentence_start_case",
        "xref_midsentence_case",
    ]


def test_algorithm_bracket_refs_are_flagged(tmp_path):
    result = _run_xref_case(
        tmp_path,
        "\n".join(
            [
                "[Algorithm @alg-start] states the loop.",
                "The implementation follows [algorithm @alg-mid].",
            ]
        ),
    )

    assert [issue.code for issue in result.issues] == [
        "algorithm_ref_bracketed_prefix",
        "algorithm_ref_bracketed_prefix",
    ]


def test_bare_algorithm_refs_are_flagged(tmp_path):
    result = _run_xref_case(
        tmp_path,
        "\n".join(
            [
                "Algorithm @alg-start states the loop.",
                "The implementation follows algorithm @alg-mid.",
            ]
        ),
    )

    assert [issue.code for issue in result.issues] == [
        "algorithm_ref_bare_prefix",
        "algorithm_ref_bare_prefix",
    ]


def test_bare_algorithm_refs_catch_capitalized_target(tmp_path):
    result = _run_xref_case(
        tmp_path,
        "The implementation follows algorithm @Alg-mid.",
    )

    assert "algorithm_ref_bare_prefix" in [issue.code for issue in result.issues]
