"""Regression tests for `binder check math --scope latex-balance`.

Added 2026-08-13. A corpus-wide cleanup script silently corrupted 38 math sites
across both volumes: it stripped the backslash from `\\}` and the opening `$`
from `$<`. Every one passed `check math`, `check all`, and a full pre-commit
run; LaTeX would have failed at PDF build time, far from the edit.

Each POSITIVE case below is a real corruption observed in that incident. Each
NEGATIVE case is a real construct from the corpus that must NOT be flagged.
"""

from pathlib import Path

import pytest

from book.cli.commands.validate import ValidateCommand


class _StubConfig:
    """`_relative_file` is the only instance state the check touches."""

    def __init__(self, book_dir: Path):
        self.book_dir = book_dir


def _run(tmp_path: Path, body: str):
    chapter = tmp_path / "chapter.qmd"
    chapter.write_text(body, encoding="utf-8")
    cmd = ValidateCommand.__new__(ValidateCommand)
    cmd.config_manager = _StubConfig(tmp_path)
    return cmd._run_latex_balance(tmp_path).issues


def _codes(issues):
    return sorted(i.code for i in issues)


# ── POSITIVE: must be caught ────────────────────────────────────────────────

@pytest.mark.parametrize("body,code", [
    # `\right\}` -> `\right}` (model_serving.qmd, distributed_training.qmd)
    (
        r"$$B = \max\left\{B : T(B) \leq L\right}$$",
        "latex_bare_delimiter",
    ),
    # `\{1,\dots,N_L\}` -> `\{1,\dots,N_L}` (training.qmd)
    (
        r"Checkpoint set $\mathcal{C} \subseteq \{1,\dots,N_L}$ layers.",
        "latex_unbalanced_escaped_brace",
    ),
    # `$<1$` -> `< 1$` (model_serving.qmd caption)
    (
        "Components seem small (< 1$ ms), so they compound.",
        "latex_unbalanced_dollar",
    ),
    # display math spanning the corruption
    (
        "$$\\text{Shard}_i = \\{e_j : \\text{hash}(j) = i}$$",
        "latex_unbalanced_escaped_brace",
    ),
    # a closing set brace with no opener
    (
        r"The set $x \in a,b\}$ is malformed.",
        "latex_unbalanced_escaped_brace",
    ),
])
def test_catches_real_corruptions(tmp_path, body, code):
    issues = _run(tmp_path, body)
    assert code in _codes(issues), f"{code} not raised for: {body}"


def test_catches_math_inside_tikz_blocks(tmp_path):
    """TikZ node labels carry real math. The 2026-08-13 incident broke two of
    them, and they sit inside a fenced block, so the check must look there."""
    body = (
        "```{.tikz}\n"
        "\\node[above=2pt of B3]{< 1$};\n"
        "```\n"
    )
    assert "latex_unbalanced_dollar" in _codes(_run(tmp_path, body))


# ── NEGATIVE: must NOT be flagged ───────────────────────────────────────────

def test_balanced_math_is_clean(tmp_path):
    body = (
        r"$$B = \max\left\{B : T(B) \leq L\right\}$$" "\n\n"
        r"Checkpoint set $\mathcal{C} \subseteq \{1,\dots,N_L\}$ layers." "\n\n"
        r"Components seem small ($<1$ ms), so they compound." "\n"
    )
    assert _run(tmp_path, body) == []


def test_escaped_currency_is_not_a_delimiter(tmp_path):
    """`\\$` is a literal dollar. The book uses it throughout for prices."""
    body = "Training cost roughly \\$4.6M, or \\$0.02 per query.\n"
    assert _run(tmp_path, body) == []


def test_doubled_escape_in_caption_attribute(tmp_path):
    """Quarto parses attribute strings once before Pandoc, so a literal dollar
    is written `\\\\$` there. Real corpus line from network_fabrics.qmd."""
    body = (
        '::: {#fig-x fig-cap="**Topologies**: Fat-Tree (25.6, \\\\$3.0/Gb/s), '
        'Dragonfly (20.0, \\\\$4.0), Torus 3D (8.0, \\\\$1.5)."}\n'
        "![](x.png)\n:::\n"
    )
    assert _run(tmp_path, body) == []


def test_tikz_linebreak_before_math_is_not_an_escape(tmp_path):
    """In TikZ, `\\\\$` is a line break followed by a real delimiter, the
    opposite of the caption case. Real corpus line from security_privacy.qmd."""
    body = (
        "```{.tikz}\n"
        "\\node[eval] (Eval) {Compute Loss\\\\$\\mathcal{L}(f, \\mathcal{S})$};\n"
        "```\n"
    )
    assert _run(tmp_path, body) == []


def test_code_fences_are_skipped(tmp_path):
    body = (
        "```python\n"
        "cost = f\"${x}\"  # a lone dollar in code is not math\n"
        "```\n"
    )
    assert _run(tmp_path, body) == []


def test_inline_code_spans_are_skipped(tmp_path):
    body = "Set the variable `price=$5` and move on.\n"
    assert _run(tmp_path, body) == []


def test_braces_may_span_a_display_block(tmp_path):
    """A `\\{` opened on one line and closed on the next inside one `$$` block
    is legal; the check accumulates the block before judging."""
    body = (
        "$$\n"
        r"\mathcal{C} = \{ x : x \in S," "\n"
        r"\quad x > 0 \}" "\n"
        "$$\n"
    )
    assert _run(tmp_path, body) == []
