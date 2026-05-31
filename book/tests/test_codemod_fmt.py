"""Tests for the paired multiplier codemod (provable lane + queue lane)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/audit/fmt"))
import ast  # noqa: E402

from codemod_fmt import (  # noqa: E402
    _rewrite_call_to_multiple, _patch_prose, scan_file,
    _rewrite_percent, scan_percent,
)


def _percent(expr: str):
    """Parse a single fmt(...) expression and run the percent rewriter on it."""
    call = ast.parse(expr, mode="eval").body
    return _rewrite_percent(call, expr)


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


def test_scan_percent_routes_fmt_int_and_variants_to_queue(tmp_path):
    p = tmp_path / "c.qmd"
    p.write_text(CELL.format(
        assigns=("ok_str = fmt(acc, suffix='%')\n"
                 "    int_str = fmt_int(rate, suffix='%')\n"
                 "    sp_str = fmt(x, suffix=' %')"),
        prose="x"), encoding="utf-8")
    edits, queue = scan_percent(p)
    assert [e.var for e in edits] == ["ok_str"]      # only the exact fmt('%')
    assert all(q.kind == "percent" for q in queue)
    assert len(queue) == 2                            # fmt_int + ' %' variant


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
