"""Tests for the paired multiplier codemod (provable lane + queue lane)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/audit/fmt"))
import ast  # noqa: E402

from codemod_fmt import (  # noqa: E402
    _rewrite_call_to_multiple, _patch_prose, scan_file,
    _rewrite_percent, scan_percent,
    _rewrite_scale, scan_scale,
    _rewrite_scale_style, scan_scale_style,
    _rewrite_unit, scan_unit,
    _rewrite_time, scan_time,
)


def _percent(expr: str):
    """Parse a single fmt(...) expression and run the percent rewriter on it."""
    call = ast.parse(expr, mode="eval").body
    return _rewrite_percent(call, expr)


def _scale(expr: str):
    call = ast.parse(expr, mode="eval").body
    return _rewrite_scale(call, expr)


def _scale_style(expr: str):
    call = ast.parse(expr, mode="eval").body
    return _rewrite_scale_style(call, expr)


def _unit(expr: str):
    call = ast.parse(expr, mode="eval").body
    return _rewrite_unit(call, expr)


def _time(expr: str):
    call = ast.parse(expr, mode="eval").body
    return _rewrite_time(call, expr)


# --- the provable cell rewrite -------------------------------------------------

def test_fmt_multiplier_rewrite_drops_suffix_keeps_precision():
    out = _rewrite_call_to_multiple("fmt(adam_mult, precision=0, commas=False, suffix='×')")
    assert out == "fmt_multiple(adam_mult, precision=0, commas=False)"


def test_fmt_int_is_not_auto_rewritten():
    # fmt_int rounds; not provably equal to fmt_multiple -> declined here
    assert _rewrite_call_to_multiple("fmt_int(round(x), commas=False, suffix='×')") is None


def test_multiline_call_declined():
    assert _rewrite_call_to_multiple("fmt(x,\n    suffix='×')") is None


def test_commas_true_injected_when_omitted():
    # fmt defaults commas=True; fmt_multiple defaults commas=False -> must pin
    assert _rewrite_call_to_multiple("fmt(x, suffix='×')") == "fmt_multiple(x, commas=True)"


def test_explicit_commas_false_preserved():
    out = _rewrite_call_to_multiple("fmt(x, commas=False, suffix='×')")
    assert out == "fmt_multiple(x, commas=False)"


def test_prefix_call_declined():
    # fmt_multiple has no prefix= -> not a clean rewrite
    assert _rewrite_call_to_multiple("fmt(x, prefix='~', suffix='×')") is None


# --- the prose patch (position-based, handles repeats) -------------------------

def test_prose_patch_single_ref():
    text = "speedup is `{python} C.s_str` over baseline\n"
    new, patches = _patch_prose(text, {"C.s_str"})
    assert "`{python} C.s_str`$\\times$ over baseline" in new
    assert len(patches) == 1


def test_prose_patch_repeated_ref_on_one_line_each_once():
    # the bug that produced '16××' / '1.75×××' must not recur
    text = "| h | `{python} C.s_str` | `{python} C.s_str` |\n"
    new, patches = _patch_prose(text, {"C.s_str"})
    assert new.count("$\\times$") == 2
    assert "$\\times$$\\times$" not in new
    assert len(patches) == 2


def test_prose_patch_skips_when_glyph_already_present():
    text = "ratio `{python} C.s_str`$\\times$ already\n"
    new, patches = _patch_prose(text, {"C.s_str"})
    assert patches == []
    assert new.count("$\\times$") == 1


def test_prose_patch_ignores_cell_bodies():
    text = "```{python}\ns_str = fmt_multiple(6)\n```\nvalue `{python} C.s_str` here\n"
    new, patches = _patch_prose(text, {"C.s_str"})
    # only the prose ref is patched, not the assignment in the cell
    assert "fmt_multiple(6)$\\times$" not in new
    assert len(patches) == 1


def test_prose_patch_ignores_unrelated_refs():
    text = "cost `{python} C.cost_str` total\n"
    new, patches = _patch_prose(text, {"C.s_str"})
    assert patches == []
    assert new == text


# --- queue lane ----------------------------------------------------------------

CELL = """```{{python}}
#| echo: false
class C:
    {assigns}
```

{prose}
"""


def test_percent_and_scale_go_to_queue_not_rewritten(tmp_path):
    p = tmp_path / "c.qmd"
    p.write_text(CELL.format(
        assigns=("pct_str = fmt(savings, suffix='%')\n"
                 "    big_str = fmt(count / 1e6, suffix='M')\n"
                 "    sp_str = fmt(speedup, suffix='×')"),
        prose="x `{python} C.sp_str` y"), encoding="utf-8")
    edits, mult_vars, queue = scan_file(p)
    kinds = {q.kind for q in queue}
    assert "percent" in kinds and "scale" in kinds
    # only the multiplier is auto-rewritten
    assert [e.var for e in edits] == ["sp_str"]
    assert mult_vars == {"C.sp_str"}


def test_cross_class_same_name_only_rewritten_one_is_patched(tmp_path):
    # the data_engineering bug: two classes export 'ratio_str'; only Auto's is
    # rewritten (fmt …×), Kept's uses fmt_int (queued). Prose must patch ONLY
    # Auto.ratio_str, never Kept.ratio_str (which still carries × in its string).
    body = (
        "```{python}\n#| echo: false\n"
        "class Auto:\n    ratio_str = fmt(a, precision=1, commas=False, suffix='×')\n"
        "class Kept:\n    ratio_str = fmt_int(round(b), commas=False, suffix='×')\n"
        "```\n\n"
        "auto `{python} Auto.ratio_str` and kept `{python} Kept.ratio_str` end\n"
    )
    p = tmp_path / "c.qmd"
    p.write_text(body, encoding="utf-8")
    edits, mult_vars, queue = scan_file(p)
    assert mult_vars == {"Auto.ratio_str"}
    new, patches = _patch_prose(body, mult_vars)
    assert "`{python} Auto.ratio_str`$\\times$" in new
    assert "`{python} Kept.ratio_str`$\\times$" not in new
    assert [r for _, r in patches] == ["Auto.ratio_str"]


def test_literal_x_and_space_glyph_go_to_queue_not_auto(tmp_path):
    p = tmp_path / "c.qmd"
    p.write_text(CELL.format(
        assigns=("xx_str = fmt(ratio, suffix='x')\n"
                 "    sp_str = fmt(speedup, suffix=' ×')"),
        prose="x"), encoding="utf-8")
    edits, mult_vars, queue = scan_file(p)
    assert edits == []  # neither auto-rewritten
    assert all(q.kind == "multiplier-variant" for q in queue)
    assert len(queue) == 2


# --- the byte-identical percent lane -------------------------------------------

def test_percent_symbol_standalone_divides_by_100():
    out = _percent("fmt(savings_pct, precision=1, commas=False, suffix='%')")
    assert out == "fmt_percent(savings_pct/100, precision=1, commas=False, style='symbol')"


def test_percent_prose_word_uses_prose_style():
    out = _percent("fmt(share, precision=0, suffix=' percent')")
    assert out == "fmt_percent(share/100, precision=0, style='prose')"


def test_percent_strips_ratio_times_100():
    # the clean win: fmt(ratio*100, '%') -> fmt_percent(ratio) (no *100/100)
    assert _percent("fmt(mfu * 100, precision=1, suffix='%')") == \
        "fmt_percent(mfu, precision=1, style='symbol')"
    assert _percent("fmt(100 * util, suffix='%')") == "fmt_percent(util, style='symbol')"


def test_percent_parenthesises_binop_before_dividing():
    # a bare expression must be parenthesised so /100 binds to the whole value
    assert _percent("fmt(a + b, suffix='%')") == "fmt_percent((a + b)/100, style='symbol')"


def test_percent_declines_prefix_and_nonexact_suffix():
    assert _percent("fmt(x, prefix='~', suffix='%')") is None
    assert _percent("fmt(x, suffix=' %')") is None      # spacing variant -> queue
    assert _percent("fmt(x, suffix='percent')") is None  # no leading space -> queue


def test_scan_percent_migrates_fmt_int_exact_queues_variants(tmp_path):
    p = tmp_path / "c.qmd"
    p.write_text(CELL.format(
        assigns=("ok_str = fmt(acc, suffix='%')\n"
                 "    int_str = fmt_int(rate, suffix='%')\n"
                 "    sp_str = fmt(x, suffix=' %')"),
        prose="x"), encoding="utf-8")
    edits, queue = scan_percent(p)
    # exact fmt('%') and fmt_int('%') both migrate (the latter via round()/100)
    assert sorted(e.var for e in edits) == ["int_str", "ok_str"]
    int_edit = next(e for e in edits if e.var == "int_str")
    assert int_edit.new == "fmt_percent(round(rate)/100, precision=0, commas=True, style='symbol')"
    # the spacing variant ' %' still goes to the adjudication queue
    assert [q.var for q in queue] == ["sp_str"]
    assert queue[0].kind == "percent"


def test_scale_strips_named_divisor_matching_glyph():
    assert _scale("fmt(x / MILLION, suffix='M')") == \
        "fmt_count(x, scale='M', precision=1)"
    assert _scale("fmt(toks / c.TRILLION, precision=1, commas=False, suffix='T')") == \
        "fmt_count(toks, scale='T', precision=1, commas=False)"


def test_scale_strips_numeric_literal_divisor():
    assert _scale("fmt(n / 1e9, precision=2, suffix='B')") == \
        "fmt_count(n, scale='B', precision=2)"


def test_scale_declines_when_divisor_does_not_match_glyph():
    # divisor magnitude (1e3) != glyph 'M' (1e6): a latent bug, leave it
    assert _scale("fmt(x / THOUSAND, suffix='M')") is None


def test_scale_declines_prescaled_and_prefix_and_lowercase():
    assert _scale("fmt(params_b, suffix='B')") is None        # no division
    assert _scale("fmt(x / THOUSAND, suffix='k')") is None     # lowercase glyph
    assert _scale("fmt(x / MILLION, prefix='~', suffix='M')") is None


def test_scan_scale_queues_prescaled_migrates_division(tmp_path):
    p = tmp_path / "c.qmd"
    p.write_text(CELL.format(
        assigns=("a_str = fmt(toks / MILLION, suffix='M')\n"
                 "    b_str = fmt(params_b, suffix='B')\n"
                 "    c_str = fmt_int(n / THOUSAND, suffix='K')"),
        prose="x"), encoding="utf-8")
    edits, queue = scan_scale(p)
    assert [e.var for e in edits] == ["a_str"]            # only the clean division
    assert sorted(q.var for q in queue) == ["b_str", "c_str"]
    assert all(q.kind == "scale" for q in queue)


def test_scale_style_prescaled_value_reconstructs_raw_count():
    assert _scale_style("fmt(params_b, precision=0, commas=False, suffix=' B')") == \
        "fmt_count(params_b * BILLION, scale='B', precision=0, commas=False)"


def test_scale_style_lowercase_k_division_becomes_uppercase_scale():
    assert _scale_style("fmt(n / THOUSAND, precision=0, commas=False, suffix='k')") == \
        "fmt_count(n, scale='K', precision=0, commas=False)"


def test_scale_style_param_quantity_uses_raw_param_count():
    assert _scale_style("fmt(model.parameters.m_as(Bparam), precision=0, commas=False, suffix=' B')") == \
        "fmt_count(model.parameters.m_as('param'), scale='B', precision=0, commas=False)"


def test_scale_style_fmt_int_round_division():
    assert _scale_style("fmt_int(round(n_params / BILLION), commas=False, suffix='B')") == \
        "fmt_count(round(n_params / BILLION) * BILLION, scale='B', precision=0, commas=False)"


def test_scan_scale_style_qualifies_class_exports(tmp_path):
    p = tmp_path / "c.qmd"
    p.write_text(CELL.format(
        assigns=("a_str = fmt(params_b, precision=0, commas=False, suffix=' B')\n"
                 "    b_str = fmt(n / THOUSAND, precision=0, commas=False, suffix='k')"),
        prose="x"), encoding="utf-8")
    edits, qualified, queue = scan_scale_style(p)
    assert sorted(e.var for e in edits) == ["a_str", "b_str"]
    assert qualified == {"C.a_str", "C.b_str"}
    assert queue == []


def test_scan_percent_queues_multiline_calls_not_silently_dropped(tmp_path):
    # a multiline fmt(...) can't be spliced by the single-line applier; it must
    # land in the queue, never be emitted as an edit that won't apply.
    p = tmp_path / "c.qmd"
    p.write_text(
        "```{python}\n#| echo: false\nclass C:\n"
        "    total_str = fmt(\n        (a + b) * 100,\n        precision=0,\n"
        "        suffix=' percent',\n    )\n```\n\nx `{python} C.total_str` y\n",
        encoding="utf-8")
    edits, queue = scan_percent(p)
    assert edits == []
    assert len(queue) == 1 and queue[0].kind == "percent"


def test_unit_rewrite_quantity_m_as_to_fmt_qty():
    assert _unit("fmt(d_vol.m_as('GB'), precision=0, commas=False, suffix=' GB')") == \
        "fmt_qty(d_vol, GB, precision=0, commas=False)"
    assert _unit("fmt(bw.m_as(TB/second), precision=2, commas=False, suffix=' TB/s')") == \
        "fmt_qty(bw, TB/second, precision=2, commas=False)"


def test_unit_rewrite_singularizes_flop_rate_suffix():
    assert _unit("fmt(gpu.compute.peak_flops.m_as('TFLOPs/s'), precision=0, commas=False, suffix=' TFLOP/s')") == \
        "fmt_qty(gpu.compute.peak_flops, TFLOP/second, precision=0, commas=False)"


def test_unit_rewrite_pins_default_commas():
    assert _unit("fmt(mem.m_as(GB), suffix=' GB')") == "fmt_qty(mem, GB, commas=True)"


def test_unit_rewrite_declines_plain_float_and_noncanonical_words():
    assert _unit("fmt(weights_gb, precision=1, suffix=' GB')") is None
    assert _unit("fmt(cluster_mtbf.m_as(ureg.hour), precision=1, suffix=' hours')") is None


def test_scan_unit_migrates_quantity_backed_and_queues_plain_float(tmp_path):
    p = tmp_path / "c.qmd"
    p.write_text(CELL.format(
        assigns=("ok_str = fmt(mem.m_as(GB), precision=0, commas=False, suffix=' GB')\n"
                 "    float_str = fmt(mem_gb, precision=0, suffix=' GB')"),
        prose="x"), encoding="utf-8")
    edits, queue = scan_unit(p)
    assert [e.var for e in edits] == ["ok_str"]
    assert edits[0].new == "fmt_qty(mem, GB, precision=0, commas=False)"
    assert [q.var for q in queue] == ["float_str"]
    assert queue[0].kind == "unit"


def test_time_rewrite_uses_full_unit_names_for_symbol_style():
    assert _time("fmt(latency_ms, precision=0, commas=False, suffix=' ms')") == \
        "fmt_time(latency_ms, 'millisecond', precision=0, commas=False)"
    assert _time("fmt(duration_s, precision=1, suffix=' s')") == \
        "fmt_time(duration_s, 'second', precision=1, commas=True)"


def test_time_rewrite_word_style_and_fmt_int_rounding():
    assert _time("fmt(elapsed, precision=0, commas=False, suffix=' seconds')") == \
        "fmt_time(elapsed, 'second', precision=0, commas=False, style='word')"
    assert _time("fmt_int(hours, commas=False, suffix=' hours')") == \
        "fmt_time(round(hours), 'hour', precision=0, commas=False, style='word')"


def test_time_rewrite_preserves_markers():
    assert _time("fmt(duration, precision=0, commas=False, approx=True, suffix=' s')") == \
        "fmt_time(duration, 'second', precision=0, commas=False, approx=True)"


def test_scan_time_migrates_exact_time_suffixes(tmp_path):
    p = tmp_path / "c.qmd"
    p.write_text(CELL.format(
        assigns=("latency_str = fmt(latency, precision=0, commas=False, suffix=' ms')\n"
                 "    duration_str = fmt_int(duration, commas=True, suffix=' seconds')\n"
                 "    other_str = fmt(size, precision=0, suffix=' GB')"),
        prose="x"), encoding="utf-8")
    edits, queue = scan_time(p)
    assert [e.var for e in edits] == ["latency_str", "duration_str"]
    assert queue == []
