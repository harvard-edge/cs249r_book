"""Tests for closed LEGO export ↔ prose unit duplication checker."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "book" / "tools" / "audit"))

from book_check_lego_prose_units import check_file  # noqa: E402


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
