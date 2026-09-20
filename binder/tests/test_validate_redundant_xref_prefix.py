"""Coverage for the redundant-noun guard on native Quarto cross-references.

Quarto renders the noun itself ("figure 3.1"), so a prose noun in front of a
native ref ships doubled ("Figure figure 3.1"). Everything below is driven off
one table of (noun forms x prefix) so the guard is exercised identically for
every referenceable type in the book -- section / chapter / appendix, table,
figure, equation, listing, algorithm -- in both prefix casings. A type that
gains a noun or an abbreviation is covered by adding it to TYPES; nothing else
in this file is type-specific.

The guard must stay silent on the two forms that do NOT render a noun: the
custom-callout `\\ref{}` (bare number, so the prose noun is required) and the
suppressed `[-@ref]` form.
"""

from types import SimpleNamespace

import pytest

from binder.cli.commands.validate import ValidateCommand

# (base noun, every accepted written form, native prefix)
TYPES = [
    ("section", ["Section", "section", "Sections", "Sec.", "§"], "sec"),
    ("chapter", ["Chapter", "chapter", "Chapters", "Chap.", "Ch."], "sec"),
    ("appendix", ["Appendix", "appendix", "Appendices", "App."], "sec"),
    ("table", ["Table", "table", "Tables", "Tbl.", "Tab."], "tbl"),
    ("figure", ["Figure", "figure", "Figures", "Fig.", "Figs."], "fig"),
    ("equation", ["Equation", "equation", "Equations", "Eq.", "Eqn."], "eq"),
    ("listing", ["Listing", "listing", "Listings", "Lst."], "lst"),
    ("algorithm", ["Algorithm", "algorithm", "Algorithms", "Alg.", "Algo."], "algo"),
]

# Both source casings render a noun, so both double after a prose noun.
CASINGS = [str.lower, str.capitalize]


def _run(tmp_path, body):
    (tmp_path / "chapter.qmd").write_text(body, encoding="utf-8")
    command = ValidateCommand(
        config_manager=SimpleNamespace(book_dir=tmp_path),
        chapter_discovery=None,
    )
    return command._run_redundant_xref_prefix(tmp_path)


def _cases(template):
    """Expand `template` over every noun form, type, and prefix casing."""
    return [
        pytest.param(
            template.format(noun=noun, ref=f"{casing(prefix)}-target"),
            id=f"{base}-{noun}-{casing(prefix)}",
        )
        for base, nouns, prefix in TYPES
        for noun in nouns
        for casing in CASINGS
    ]


# ---------------------------------------------------------------------------
# Flagged: the noun renders twice.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line", _cases("{noun} @{ref} carries the point."))
def test_noun_before_bare_ref_is_flagged(tmp_path, line):
    result = _run(tmp_path, line)
    assert [issue.code for issue in result.issues] == ["redundant_xref_prefix"], line


@pytest.mark.parametrize("line", _cases("{noun} [@{ref}] carries the point."))
def test_noun_before_bracketed_ref_is_flagged(tmp_path, line):
    """`[@ref]` renders "(Table 3.1)" -- still doubled after a prose noun."""
    result = _run(tmp_path, line)
    assert [issue.code for issue in result.issues] == ["redundant_xref_prefix"], line


# Spacing and emphasis variants a plain `\bNoun\s+@` pattern misses. Applied to
# every type, not just the one that happened to surface the bug.
VARIANT_TEMPLATES = [
    "{noun}~@{ref} carries the point.",             # LaTeX tie
    "{noun} @{ref} carries the point.",        # NBSP
    "{noun} @{ref} carries the point.",        # narrow NBSP
    "{noun}&nbsp;@{ref} carries the point.",        # HTML entity
    "**{noun}** @{ref} carries the point.",         # bold noun
    "*{noun}* @{ref} carries the point.",           # italic noun
    "_{noun}_ @{ref} carries the point.",           # underscore emphasis
    "[{noun} @{ref}] carries the point.",           # bracketed lead-in
]


@pytest.mark.parametrize(
    "line", [c for t in VARIANT_TEMPLATES for c in _cases(t)]
)
def test_spacing_and_emphasis_variants_are_flagged(tmp_path, line):
    result = _run(tmp_path, line)
    assert [issue.code for issue in result.issues] == ["redundant_xref_prefix"], line


@pytest.mark.parametrize(
    "noun,prefix",
    [(nouns[0], prefix) for _, nouns, prefix in TYPES],
)
@pytest.mark.parametrize("tie", ["", "~"])
@pytest.mark.parametrize("bracket", [False, True])
def test_line_wrapped_noun_is_flagged(tmp_path, noun, prefix, tie, bracket):
    """The noun can end one source line with the ref opening the next."""
    ref = f"@{prefix.capitalize()}-target"
    ref = f"[{ref}]" if bracket else ref
    result = _run(
        tmp_path,
        "\n".join([f"The result appears in {noun}{tie}", f"{ref}, as discussed."]),
    )
    assert [issue.code for issue in result.issues] == ["redundant_xref_prefix"]
    assert result.issues[0].line == 1


# ---------------------------------------------------------------------------
# Clean: these forms render no noun of their own, or none at all.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line", _cases("{noun} [-@{ref}] carries the point."))
def test_suppressed_ref_after_noun_is_clean(tmp_path, line):
    """`[-@ref]` suppresses the rendered noun, so the prose noun is correct."""
    result = _run(tmp_path, line)
    assert not result.issues, (line, [i.message for i in result.issues])


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(
            f"The measurement (@{casing(prefix)}-target) is clear.",
            id=f"{base}-parenthetical-{casing(prefix)}",
        )
        for base, _nouns, prefix in TYPES
        for casing in CASINGS
    ],
)
def test_parenthetical_ref_is_clean(tmp_path, line):
    """A noun separated from the ref by punctuation is ordinary prose."""
    result = _run(tmp_path, line)
    assert not result.issues, (line, [i.message for i in result.issues])


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(
            template.format(noun=nouns[0].lower(), ref=f"{casing(prefix)}-target"),
            id=f"{base}-{idx}-{casing(prefix)}",
        )
        for base, nouns, prefix in TYPES
        for casing in CASINGS
        for idx, template in enumerate(
            [
                "@{ref} carries the point.",                      # bare ref, no noun
                "The result, @{ref}, carries the point.",         # comma-separated
                "The {noun} in question (@{ref}) is clear.",      # parenthesized
                "A {noun} of contents follows this chapter.",     # noun, no ref
            ]
        )
    ],
)
def test_correct_forms_are_clean(tmp_path, line):
    result = _run(tmp_path, line)
    assert not result.issues, (line, [i.message for i in result.issues])


# Custom numbered callouts render a BARE number via \ref{}, so the prose noun
# in front of them is required. Prefixes come from custom-numbered-blocks.yml.
CALLOUTS = [
    ("principle", "pri-iron-law"),
    ("napkin math", "nbk-utility-bill"),
    ("example", "exmp-hidden-debt"),
    ("definition", "dfn-latency"),
    ("lighthouse", "lhs-kws-journey"),
    ("systems perspective", "psp-sparsity-trap"),
    ("theorem", "thrm-littles-law"),
    ("checkpoint", "chk-scaling"),
    ("case study", "cs-vllm"),
    ("war story", "ws-outage"),
]


@pytest.mark.parametrize("noun,target", CALLOUTS)
def test_callout_ref_form_keeps_its_noun(tmp_path, noun, target):
    result = _run(tmp_path, f"The point holds ({noun} \\ref{{{target}}}) throughout.")
    assert not result.issues, [i.message for i in result.issues]


# Ordinary bracketed citations sit behind a topic noun all the time. Restricting
# the bracketed branch to native crossref prefixes is what keeps these clean.
CITATIONS = [
    "Uber adopted the ring-allreduce algorithm [@gibiansky2017baidu] for this.",
    "The DAgger algorithm [@ross2011reduction] eliminates compounding drift.",
    "The governing equations [@lynch2017modern] project the dynamics.",
    "The synthetic-data table [@goncalves2020] reports the leakage rate.",
    "Following Algorithm 1 of [@dao2022], the block sizes follow from SRAM.",
]


@pytest.mark.parametrize("line", CITATIONS)
def test_citations_after_a_topic_noun_are_clean(tmp_path, line):
    result = _run(tmp_path, line)
    assert not result.issues, (line, [i.message for i in result.issues])


@pytest.mark.parametrize(
    "line",
    [f"# Context: {nouns[0]} @{prefix}-target" for _base, nouns, prefix in TYPES],
)
def test_code_blocks_and_cell_options_are_ignored(tmp_path, line):
    result = _run(tmp_path, "\n".join(["```python", line, "```", f"#| fig-cap: {line}"]))
    assert not result.issues
