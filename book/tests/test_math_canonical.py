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


def test_math_canonical_flags_manual_markdownstr_math(tmp_path):
    chapter = tmp_path / "chapter.qmd"
    chapter.write_text(
        """
```{python}
from mlsysim.fmt import MarkdownStr

class BadMath:
    inline_math = MarkdownStr(f"$x={1}$")
    display_eq = MarkdownStr(f"$$x={1}$$")
```
""",
        encoding="utf-8",
    )

    issues = audit([chapter])

    math_issues = [
        issue for issue in issues if issue.code == "manual_markdownstr_math"
    ]
    assert len(math_issues) == 2
    assert "fmt_math" in math_issues[0].message
    assert "fmt_display_math" in math_issues[1].message


def test_math_canonical_allows_fmt_math_helpers(tmp_path):
    chapter = tmp_path / "chapter.qmd"
    chapter.write_text(
        """
```{python}
from mlsysim.fmt import fmt_math, fmt_display_math

class GoodMath:
    inline_math = fmt_math(f"x={1}")
    display_eq = fmt_display_math(f"x={1}")
```
""",
        encoding="utf-8",
    )

    issues = audit([chapter])

    assert not [
        issue for issue in issues if issue.code == "manual_markdownstr_math"
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


def test_math_canonical_allows_rounded_quantity_helper(tmp_path):
    chapter = tmp_path / "chapter.qmd"
    chapter.write_text(
        """
```{python}
from mlsysim.fmt import fmt_qty_int

class GoodFormatting:
    memory_str = fmt_qty_int(memory, GB, commas=False)
```
""",
        encoding="utf-8",
    )

    issues = audit([chapter])

    assert not [
        issue for issue in issues if issue.code == "noncanonical_str_assign"
    ]


def test_math_canonical_allows_domain_formatters(tmp_path):
    chapter = tmp_path / "chapter.qmd"
    chapter.write_text(
        """
```{python}
from mlsysim.fmt import fmt_area, fmt_flop_rate, fmt_heat_flux, fmt_params, fmt_water_rate

class GoodFormatting:
    peak_str = fmt_flop_rate(peak, unit=TFLOP / second, precision=0)
    params_str = fmt_params(params, scale="B", precision=1)
    water_str = fmt_water_rate(water, precision=0)
    area_str = fmt_area(area, unit=ureg.millimeter**2)
    heat_flux_str = fmt_heat_flux(flux, unit=watt / (ureg.centimeter**2))
```
""",
        encoding="utf-8",
    )

    issues = audit([chapter])

    assert not [
        issue for issue in issues if issue.code == "noncanonical_str_assign"
    ]
