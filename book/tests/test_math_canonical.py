from book.cli.checks.math_canonical import audit


def test_math_canonical_flags_manual_fmt_string_assembly(tmp_path):
    chapter = tmp_path / "chapter.qmd"
    chapter.write_text(
        """
```{python}
from mlsysim.fmt import fmt, fmt_int

class BadFormatting:
    cost_str = "\\\\$" + str(fmt(10, precision=0))
    size_str = fmt(10, precision=0) + "K"
    rate_str = "$" + fmt_int(10)
```
""",
        encoding="utf-8",
    )

    issues = audit([chapter])

    manual_issues = [
        issue for issue in issues if issue.code == "manual_fmt_string_assembly"
    ]
    assert len(manual_issues) == 3


def test_math_canonical_allows_fmt_prefix_suffix(tmp_path):
    chapter = tmp_path / "chapter.qmd"
    chapter.write_text(
        """
```{python}
from mlsysim.fmt import fmt, fmt_int

class GoodFormatting:
    cost_str = fmt(10, precision=0, prefix="$", suffix="K")
    size_str = fmt(10, precision=0, suffix="K")
    count_str = fmt_int(10, prefix="$")
```
""",
        encoding="utf-8",
    )

    issues = audit([chapter])

    assert not [
        issue for issue in issues if issue.code == "manual_fmt_string_assembly"
    ]


def test_math_canonical_allows_typed_range_helpers(tmp_path):
    chapter = tmp_path / "chapter.qmd"
    chapter.write_text(
        """
```{python}
from mlsysim.fmt import fmt_usd_range

class GoodFormatting:
    cost_str = fmt_usd_range(25000, 30000, repeat_symbol=False)
```
""",
        encoding="utf-8",
    )

    issues = audit([chapter])

    assert not [
        issue for issue in issues if issue.code == "noncanonical_str_assign"
    ]
