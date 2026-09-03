"""Tests for closed LEGO export ↔ prose unit duplication checker."""
from pathlib import Path

from book.cli.checks.lego_prose_units import check_file


def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "chapter.qmd"
    p.write_text(body, encoding="utf-8")
    return p


CELL = """```{{python}}
#| echo: false
class C:
    {assigns}
```

{prose}
"""


def test_closed_arithmetic_intensity_dup_flop_per_byte(tmp_path):
    p = _write(
        tmp_path,
        CELL.format(
            assigns=(
                "ridge_str = fmt_arithmetic_intensity("
                "100 * flop / byte, unit=flop / byte, precision=0, commas=False)"
            ),
            prose="ridge `{python} C.ridge_str` FLOP/byte is high",
        ),
    )
    issues = check_file(p)
    assert issues, "expected duplicate FLOP/byte after closed intensity export"


def test_open_fmt_intensity_allows_flop_per_byte_in_prose(tmp_path):
    p = _write(
        tmp_path,
        CELL.format(
            assigns="ai_str = fmt(0.125, precision=3, commas=False)",
            prose="ReLU AI = `{python} C.ai_str` FLOP/byte (memory bound)",
        ),
    )
    assert check_file(p) == []


def test_closed_carbon_intensity_dup_g_per_kwh(tmp_path):
    p = _write(
        tmp_path,
        CELL.format(
            assigns=(
                "ci_str = fmt_carbon_intensity("
                "5 * gram / kWh, unit=gram / kWh, precision=0, commas=False)"
            ),
            prose="intensity `{python} C.ci_str` g/kWh in Quebec",
        ),
    )
    issues = check_file(p)
    assert issues


def test_closed_area_dup_mm2(tmp_path):
    p = _write(
        tmp_path,
        CELL.format(
            assigns=(
                "area_mm2_str = fmt_area("
                "814 * ureg.millimeter**2, unit=ureg.millimeter**2, commas=False)"
            ),
            prose="die area `{python} C.area_mm2_str` mm^2 is large",
        ),
    )
    issues = check_file(p)
    assert issues, "expected duplicate mm^2 after closed area export"


def test_closed_heat_flux_dup_w_per_cm2(tmp_path):
    p = _write(
        tmp_path,
        CELL.format(
            assigns=(
                "flux_w_per_cm2_str = fmt_heat_flux("
                "86 * watt / (ureg.centimeter**2), unit=watt / (ureg.centimeter**2), commas=False)"
            ),
            prose="heat flux `{python} C.flux_w_per_cm2_str` W/cm^2 is high",
        ),
    )
    issues = check_file(p)
    assert issues, "expected duplicate W/cm^2 after closed heat-flux export"


def test_open_byte_scalar_allows_bytes_in_prose(tmp_path):
    p = _write(
        tmp_path,
        CELL.format(
            assigns="bytes_value_str = fmt(16, precision=0, commas=False)",
            prose="FP16 uses `{python} C.bytes_value_str` bytes per element",
        ),
    )
    assert check_file(p) == []


def test_latex_dollar_after_ref_not_currency_dup(tmp_path):
    p = _write(
        tmp_path,
        CELL.format(
            assigns="n_str = fmt(4096, precision=0, commas=False)",
            prose="$n=$ `{python} C.n_str` in the GEMM",
        ),
    )
    assert check_file(p) == []
