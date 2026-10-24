"""Tests for the notation_macros audit check."""
from pathlib import Path
from binder.tools.audit.checks.notation_macros import check


def test_notation_macros_catches_raw_latex_in_prose():
    content = "The data volume moved across bus is $D_{\\text{vol}}$ and bandwidth is $\\text{BW}$."
    issues = check(Path("test.qmd"), content)
    assert len(issues) >= 1
    macros_suggested = [issue.reason for issue in issues]
    assert any("\\DataVolume" in m for m in macros_suggested)


def test_notation_macros_ignores_python_cells():
    content = """```{python}
ax.set_xlabel("$D_{\\text{vol}}$")
```
"""
    issues = check(Path("test.qmd"), content)
    assert len(issues) == 0


def test_notation_macros_passes_clean_macros():
    content = "The data volume is $\\DataVolume$ and bandwidth is $\\Bandwidth$."
    issues = check(Path("test.qmd"), content)
    assert len(issues) == 0


def test_notation_macros_ignores_generic_fenced_code_blocks():
    content = """```{.tikz}
\\node {$D_{\\text{vol}}$};
```
```bash
echo "$D_{\\text{vol}}$"
```
"""
    issues = check(Path("test.qmd"), content)
    assert len(issues) == 0


def test_notation_macros_catches_macro_outside_math():
    content = "The data volume is \\DataVolume in the memory path."
    issues = check(Path("test.qmd"), content)
    assert len(issues) >= 1
    assert any("outside math delimiters" in issue.reason for issue in issues)


def test_notation_macros_allows_macros_in_multiline_display_math():
    content = """
$$
\\DataVolume = \\Bandwidth \\times T
$$
"""
    issues = check(Path("test.qmd"), content)
    assert len(issues) == 0


def test_notation_macros_allows_macros_in_latex_math_envs():
    content = """
\\begin{gather*}
\\DataVolume = \\Bandwidth \\times T \\\\
\\Operations = \\PeakRate \\times \\EfficiencyHw
\\end{gather*}
"""
    issues = check(Path("test.qmd"), content)
    assert len(issues) == 0


def test_notation_macros_catches_raw_latex_in_multiline_display_math():
    content = """
$$
D_{\\text{vol}} = \\Bandwidth \\times T
$$
"""
    issues = check(Path("test.qmd"), content)
    assert len(issues) >= 1
    assert any("Found raw notation 'D_{\\text{vol}}'" in issue.reason for issue in issues)


def test_notation_macros_ignores_frontmatter_files():
    content = "Table of notations: $D_{\\text{vol}}$ is raw."
    issues = check(Path("books/vol1/frontmatter/_notation_degradation.qmd"), content)
    assert len(issues) == 0


