"""Regression tests for layout-safe QMD Python formatting."""

from book.tools.scripts.content.format_python_in_qmd import format_python_blocks


def test_display_block_is_preserved_by_default() -> None:
    source = """```{.python}
def first():
    return 1

def second():
    # This intentionally long displayed comment must remain on one source line for layout.
    return 2
```
"""

    assert format_python_blocks(source) == source


def test_executable_cell_is_formatted_by_default() -> None:
    source = """```{python}
#| echo: false
value=1
```
"""

    formatted = format_python_blocks(source)

    assert "#| echo: false" in formatted
    assert "value = 1" in formatted


def test_display_block_formatting_requires_explicit_opt_in() -> None:
    source = """```{.python}
value=1
```
"""

    formatted = format_python_blocks(source, include_display=True)

    assert "value = 1" in formatted
