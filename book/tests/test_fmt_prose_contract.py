"""Tests for the OUTPUT-formatter ↔ prose glyph contract checker."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/audit/fmt"))
from fmt_prose_contract import check_file, build_formatter_map  # noqa: E402
from audit_fmt_usage import extract_python_cells  # noqa: E402


def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "c.qmd"
    p.write_text(body, encoding="utf-8")
    return p


CELL = """```{{python}}
#| echo: false
class C:
    {assigns}
```

{prose}
"""


def test_formatter_map_picks_up_typed_calls(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="cost_str = fmt_usd(1500)\n    rate_str = fmt_percent(0.5, style='symbol')",
        prose="x"))
    cells = "\n".join(src for _, src in extract_python_cells(p.read_text()))
    fmap = build_formatter_map(cells)
    assert fmap["cost_str"][0] == "fmt_usd"
    assert fmap["rate_str"] == ("fmt_percent", {"style": "symbol"})


def test_percent_symbol_dup_flagged(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="rate_str = fmt_percent(0.5, style='symbol')",
        prose="improves by `{python} C.rate_str`% overall"))
    v = check_file(p)
    assert any(x.code == "percent_dup" for x in v)


def test_percent_number_style_not_flagged(tmp_path):
    # bare-number style: prose is *expected* to supply 'percent'
    p = _write(tmp_path, CELL.format(
        assigns="rate_str = fmt_percent(0.5, style='number')",
        prose="improves by `{python} C.rate_str` percent overall"))
    assert check_file(p) == []


def test_usd_prefix_dup_flagged(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="cost_str = fmt_usd(1500)",
        prose="costs \\$`{python} C.cost_str` total"))
    assert any(x.code == "usd_dup" for x in check_file(p))


def test_usd_no_false_positive_on_math_delimiter(tmp_path):
    # '$\\times$ `{python} ...`' — the '$' is a math close, not currency
    p = _write(tmp_path, CELL.format(
        assigns="cost_str = fmt_usd(1500)",
        prose="3 $\\times$ `{python} C.cost_str` per node"))
    assert check_file(p) == []


def test_multiple_requires_times_glyph(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="speedup_str = fmt_multiple(6)",
        prose="is `{python} C.speedup_str` faster than baseline"))
    assert any(x.code == "mult_missing_glyph" for x in check_file(p))


def test_multiple_with_times_glyph_ok(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="speedup_str = fmt_multiple(6)",
        prose="is `{python} C.speedup_str`$\\times$ faster than baseline"))
    assert check_file(p) == []


def test_multiple_literal_x_flagged(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="speedup_str = fmt_multiple(6)",
        prose="is `{python} C.speedup_str`× faster than baseline"))
    assert any(x.code == "mult_literal_x" for x in check_file(p))
