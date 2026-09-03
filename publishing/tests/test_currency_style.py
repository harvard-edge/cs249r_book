from pathlib import Path

from book.cli.checks.currency_style import (
    NOTATION_DEFINITION,
    audit,
    audit_rendered_html,
)


def test_currency_style_allows_single_notation_definition(tmp_path):
    notation = tmp_path / "book/quarto/contents/vol1/frontmatter/_notation_body.qmd"
    notation.parent.mkdir(parents=True)
    notation.write_text(NOTATION_DEFINITION + "\n", encoding="utf-8")

    assert audit([tmp_path / "book/quarto/contents"]) == []


def test_currency_style_flags_usd_in_content_sources(tmp_path):
    content = tmp_path / "book/quarto/contents"
    chapter = content / "vol1/ml_ops/ml_ops.qmd"
    quiz = content / "vol2/ops_scale/ops_scale_quizzes.json"
    svg = content / "vol2/sustainable_ai/images/svg/carbon-tco.svg"
    chapter.parent.mkdir(parents=True)
    quiz.parent.mkdir(parents=True)
    svg.parent.mkdir(parents=True)

    chapter.write_text("Training cost: USD 30K/year\n", encoding="utf-8")
    quiz.write_text('{"answer": "USD 10K"}\n', encoding="utf-8")
    svg.write_text("<text>Cumulative TCO (USD)</text>\n", encoding="utf-8")

    violations = audit([content])

    assert [violation.code for violation in violations] == [
        "currency_usd_literal",
        "currency_usd_literal",
        "currency_usd_literal",
    ]
    assert {Path(violation.file).suffix for violation in violations} == {
        ".json",
        ".qmd",
        ".svg",
    }


def test_currency_style_flags_fmt_prefix_and_suffix_currency(tmp_path):
    content = tmp_path / "book/quarto/contents"
    chapter = content / "vol1/ml_ops/ml_ops.qmd"
    chapter.parent.mkdir(parents=True)

    # Both the unescaped and escaped prefix forms, plus a `$` in suffix, are
    # forbidden: currency must go through fmt_usd().
    chapter.write_text(
        'a_str = fmt(cost, precision=0, prefix="$", suffix="K")\n'
        'b_str = fmt(cost, precision=0, prefix="\\\\$", suffix="K")\n'
        'c_str = fmt(cost, precision=0, suffix="$/GB")\n'
        'good_str = fmt_usd(cost, suffix="K")\n',
        encoding="utf-8",
    )

    violations = audit([content])

    assert [violation.code for violation in violations] == [
        "currency_fmt_prefix_suffix",
        "currency_fmt_prefix_suffix",
        "currency_fmt_prefix_suffix",
    ]


def test_currency_style_allows_fmt_usd(tmp_path):
    content = tmp_path / "book/quarto/contents"
    chapter = content / "vol1/ml_ops/ml_ops.qmd"
    chapter.parent.mkdir(parents=True)

    chapter.write_text(
        'a_str = fmt_usd(cost, precision=0)\n'
        'b_str = fmt_usd(rate, precision=2, commas=False, suffix="/GB")\n'
        'c_str = fmt_usd(total, approx=True, suffix="/year")\n',
        encoding="utf-8",
    )

    assert audit([content]) == []


def test_rendered_currency_flags_visible_artifacts_but_skips_math_and_code(tmp_path):
    html = tmp_path / "chapter.html"
    html.write_text(
        """
        <main>
          <p>Cost: $$50M and USD 50M.</p>
          <span class="math display">\\[\\text{ROI}=\\frac{1}{2}\\]</span>
          <pre><code>USD $$ \\frac{1}{2}</code></pre>
        </main>
        """,
        encoding="utf-8",
    )

    violations = audit_rendered_html([tmp_path])

    assert [violation.code for violation in violations] == [
        "rendered_usd_literal",
        "rendered_double_dollar",
    ]


def test_rendered_currency_flags_currency_text_swallowed_by_math(tmp_path):
    html = tmp_path / "chapter.html"
    html.write_text(
        """
        <main>
          <p>from $1M to <span class="math inline">\\(10K), per-task cost\\)</span></p>
        </main>
        """,
        encoding="utf-8",
    )

    violations = audit_rendered_html([tmp_path])

    assert [violation.code for violation in violations] == [
        "rendered_currency_math_span",
    ]


def test_rendered_currency_allows_notation_definition(tmp_path):
    html = tmp_path / "notation.html"
    html.write_text(
        "<main><p>unless otherwise noted, dollar-denominated costs are U.S. dollars (USD).</p></main>",
        encoding="utf-8",
    )

    assert audit_rendered_html([tmp_path]) == []
