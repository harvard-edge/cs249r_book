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

