"""Coverage for the redundant-noun guard on native Quarto cross-references.

Quarto renders the noun itself ("figure 3.1"), so a prose noun in front of a
native ref ships doubled ("Figure figure 3.1"). The guard has to fire for every
referenceable type in the book (@sec-/@fig-/@tbl-/@eq-/@lst-/@algo-), in both
prefix casings, and through the spacing and emphasis variants authors actually
type -- while staying silent on the custom-callout \ref{} form, where the prose
noun is required.
"""

from types import SimpleNamespace

import pytest

from binder.cli.commands.validate import ValidateCommand


def _run(tmp_path, body):
    (tmp_path / "chapter.qmd").write_text(body, encoding="utf-8")
    command = ValidateCommand(
        config_manager=SimpleNamespace(book_dir=tmp_path),
        chapter_discovery=None,
    )
    return command._run_redundant_xref_prefix(tmp_path)


# Every noun/type pair the book can produce, in both prefix casings.
FLAGGED = [
    "Table @tbl-results quantifies it.",
    "The comparison in table @Tbl-results is clear.",
    "Tbl. @tbl-results and Tab. @tbl-other.",
    "Figure @fig-roofline shows it.",
    "As figure @Fig-roofline shows.",
    "Fig. @fig-roofline and Figs. @fig-other.",
    "Section @sec-training establishes it.",
    "Chapter @sec-training establishes it.",
    "Appendix @sec-refresher derives it.",
    "Appendices @sec-refresher derive it.",
    "§ @sec-training establishes it.",
    "Equation @eq-latency formalizes it.",
    "Eq. @Eq-latency and Eqn. @eq-other.",
    "Listing @lst-config defines it.",
    "Lst. @Lst-config defines it.",
    "Algorithm @algo-ring arranges it.",
    "Alg. @Algo-ring arranges it.",
]


@pytest.mark.parametrize("line", FLAGGED)
def test_every_noun_and_type_is_flagged(tmp_path, line):
    """Each line carries one violation per '@' it contains."""
    result = _run(tmp_path, line)
    expected = ["redundant_xref_prefix"] * line.count("@")
    assert [issue.code for issue in result.issues] == expected, line


# Spacing / emphasis variants that a plain `\bNoun\s+@` pattern misses.
VARIANTS = [
    "Table~@tbl-results quantifies it.",            # LaTeX tie
    "Table @tbl-results quantifies it.",       # NBSP
    "Table @tbl-results quantifies it.",       # narrow NBSP
    "Table&nbsp;@tbl-results quantifies it.",       # HTML entity
    "**Table** @tbl-results quantifies it.",        # bold noun
    "*Figure* @fig-roofline shows it.",             # italic noun
    "_Listing_ @lst-config defines it.",            # underscore emphasis
    "[Table @tbl-results] quantifies it.",          # bracketed lead-in
    "Table [@tbl-results] quantifies it.",          # bracketed ref
]


@pytest.mark.parametrize("line", VARIANTS)
def test_spacing_and_emphasis_variants_are_flagged(tmp_path, line):
    result = _run(tmp_path, line)
    assert [issue.code for issue in result.issues] == ["redundant_xref_prefix"], line


def test_line_wrapped_noun_is_flagged(tmp_path):
    result = _run(
        tmp_path,
        "\n".join(
            [
                "The measured throughput appears in Table",
                "@tbl-results, which the next section reads.",
            ]
        ),
    )
    assert [issue.code for issue in result.issues] == ["redundant_xref_prefix"]
    assert result.issues[0].line == 1


CLEAN = [
    "@Tbl-results quantifies it, and @tbl-other confirms.",
    "The comparison table (@tbl-results) is clear.",
    "The table in question, @tbl-results, is fine.",
    # Custom callouts render a bare number, so the noun is REQUIRED here.
    "The iron law (principle \\ref{pri-iron-law}) governs it.",
    "The worked example in Notebook \\ref{nbk-utility-bill} shows it.",
    "As example \\ref{exmp-hidden-debt} and definition \\ref{dfn-latency} show.",
    # `[-@...]` suppresses the rendered noun, so a prose noun is correct.
    "Table [-@tbl-results] quantifies it.",
    # Ordinary bracketed citations after a topic noun are correct prose.
    "Uber adopted the ring-allreduce algorithm [@gibiansky2017baidu] for this.",
    "The DAgger algorithm [@ross2011reduction] eliminates the drift.",
    "The governing equations [@lynch2017modern] project the dynamics.",
    "A table of contents follows.",
]


@pytest.mark.parametrize("line", CLEAN)
def test_correct_forms_are_not_flagged(tmp_path, line):
    result = _run(tmp_path, line)
    assert not result.issues, (line, [i.message for i in result.issues])


def test_code_blocks_are_ignored(tmp_path):
    result = _run(
        tmp_path,
        "\n".join(
            [
                "```python",
                "# Context: Table @tbl-results is the source",
                "```",
                "#| fig-cap: Table @tbl-results",
            ]
        ),
    )
    assert not result.issues


def test_line_wrapped_variants_are_flagged(tmp_path):
    """A tie left at the line break, and a capitalized bracketed ref, still double."""
    result = _run(
        tmp_path,
        "\n".join(
            [
                "The measured throughput appears in Table~",
                "@tbl-results, which the next section reads.",
                "",
                "The layout is drawn in Figure",
                "[@Fig-roofline] on the facing page.",
            ]
        ),
    )
    assert [issue.line for issue in result.issues] == [1, 4]
