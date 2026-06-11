"""Tests for the OUTPUT-formatter ↔ prose glyph contract checker."""
from pathlib import Path

from book.cli.checks.fmt_prose_contract import (
    build_formatter_map,
    check_file,
    extract_python_cells,
)


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
    # keyed by qualified Class.attr now
    assert fmap["C.cost_str"][0] == "fmt_usd"
    assert fmap["C.rate_str"] == ("fmt_percent", {"style": "symbol"})


def test_same_name_two_classes_no_false_positive(tmp_path):
    # the usd_dup false positive: Plain.cost_str=fmt (needs $ in prose),
    # Money.cost_str=fmt_usd (owns $). A qualified ref must use ITS class's rule.
    body = (
        "```{python}\n#| echo: false\n"
        "class Plain:\n    cost_str = fmt(5)\n"
        "class Money:\n    cost_str = fmt_usd(5)\n"
        "```\n\n"
        "plain \\$`{python} Plain.cost_str` and money `{python} Money.cost_str` end\n"
    )
    p = _write(tmp_path, body)
    viol = check_file(p)
    # Plain.cost_str correctly carries a prose '$' (fmt has none) -> NOT flagged;
    # Money.cost_str has no prose '$' -> NOT flagged. Zero violations.
    assert [v.code for v in viol] == []


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


def test_multiple_export_requires_mult_suffix(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="speedup_str = fmt_multiple(6)",
        prose="is `{python} C.speedup_str` faster than baseline"))
    assert any(x.code == "mult_suffix" for x in check_file(p))


def test_multiple_without_extra_times_ok(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="speedup_mult_str = fmt_multiple(6)",
        prose="is `{python} C.speedup_mult_str` faster than baseline"))
    assert check_file(p) == []


def test_multiple_extra_times_flagged(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="speedup_mult_str = fmt_multiple(6)",
        prose="is `{python} C.speedup_mult_str`$\\times$ faster than baseline"))
    assert any(x.code == "mult_double_glyph" for x in check_file(p))


def test_multiple_literal_x_flagged_as_double_glyph(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="speedup_mult_str = fmt_multiple(6)",
        prose="is `{python} C.speedup_mult_str`× faster than baseline"))
    assert any(x.code == "mult_double_glyph" for x in check_file(p))


def test_generic_fmt_with_compact_times_flagged(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="speedup_str = fmt(6)",
        prose="is `{python} C.speedup_str`$\\times$ faster than baseline"))
    assert any(x.code == "mult_wrong_formatter" for x in check_file(p))


def test_generic_fmt_arithmetic_product_not_flagged(tmp_path):
    p = _write(tmp_path, CELL.format(
        assigns="nodes_str = fmt(6)\n    cost_str = fmt_usd(1500)",
        prose="cost is `{python} C.nodes_str` $\\times$ `{python} C.cost_str`"))
    assert check_file(p) == []


def test_hardware_count_idiom_not_flagged(tmp_path):
    # "N× H100" / "N× A100 node" is a count of accelerators, not a multiplier:
    # a generic fmt_int GPU count followed by a compact $\times$ + device name
    # is the established house idiom and must not be flagged.
    p = _write(tmp_path, CELL.format(
        assigns="node_gpus_str = fmt_int(8)",
        prose="on an `{python} C.node_gpus_str`$\\times$ H100 node (640 GB HBM)"))
    assert check_file(p) == []


def test_hardware_idiom_does_not_mask_real_multiplier(tmp_path):
    # the exemption must be narrow: "× improvement" is still a real multiplier
    # bug even though "× H100" next to it would be exempt.
    p = _write(tmp_path, CELL.format(
        assigns="imp_str = fmt_int(4)",
        prose="a `{python} C.imp_str`$\\times$ improvement over the baseline"))
    assert any(x.code == "mult_wrong_formatter" for x in check_file(p))
