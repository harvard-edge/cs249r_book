"""Unit tests for the strict notation check (binder/cli/checks/notation_strict.py).

A small fake registry keeps them independent of mlsysim's data. They pin what
counts as a raw symbol, the role-based exemptions, decorated macros, scope,
the KaTeX rendering check, and the deterministic applier.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "binder" / "cli" / "checks"))
from notation_strict import (  # noqa: E402
    Registry, RegistryEntry, apply_assignments, extract_fmt_math, extract_math,
    extract_quiz_math, judge, verify_rendering,
)

BUILTINS = {"frac", "sum", "prod", "log", "cdot", "times", "left", "right", "sqrt", "le", "approx",
            "max", "min", "in", "Pr", "Big", "quad", ",", "top", "nabla", "partial", "to"}
REG = Registry([
    RegistryEntry("NumberOfDevices", "N", "number of devices", "count"),
    RegistryEntry("LinkBandwidth", r"\beta", "bandwidth of one link"),
    RegistryEntry("DecodeInterval", r"T_{\text{decode}}", "decode interval"),
    RegistryEntry("GeneralizedCoordinates", r"\mathbf{q}", "generalized coordinates", "vector"),
    RegistryEntry("Bandwidth", r"\text{BW}", "bandwidth"),
    RegistryEntry("Time", "T", "time"),
], {"OldBandwidth": "Bandwidth"}, BUILTINS)


def raws(math, scope=None):
    return [(f.raw, f.kind) for f in judge(math, REG, scope)]


# --- what is a raw symbol -------------------------------------------------

def test_letters_and_greek():
    assert raws(r"N + \alpha") == [("N", "letter"), (r"\alpha", "greek")]


def test_registered_macros_pass():
    assert raws(r"\NumberOfDevices \cdot \LinkBandwidth + \Bandwidth") == []


def test_index_letters_only_as_indices():
    assert raws(r"\sum_{i=1}^{N} x_j") == [("N", "letter"), ("x", "letter")]
    assert raws(r"\text{top-}k") == [("k", "letter")]          # k outside a subscript is a symbol


def test_differential_euler_transpose_by_role():
    assert raws(r"d\Time + e^{2} + \mathbf{q}^T") == [(r"\mathbf{q}", "form")]
    assert raws(r"d + e") == [("d", "letter"), ("e", "letter")]   # not in their roles
    assert raws(r"T^{T}") == [("T", "letter")]                    # the base T is a symbol


def test_text_operatorname_and_single_letter_text():
    # the variable under arg max is a symbol like any other
    assert raws(r"\text{peak} + \operatorname*{arg\,max}_{x} z") == [("x", "letter"), ("z", "letter")]
    # single upright letters are units or labels (5\,\text{s}, group \text{A}); italic is a symbol
    assert raws(r"5\,\text{s} + G=\text{A}") == [("G", "letter")]


def test_label_subscripts_belong_to_the_symbol():
    assert raws(r"(A_{\text{component}})^N") == [(r"A_{\text{component}}", "letter"), ("N", "letter")]
    assert raws(r"x_i + \alpha_{\text{lr}} + \mathbf{q}_{\text{des}}") == [
        ("x", "letter"), (r"\alpha_{\text{lr}}", "greek"), (r"\mathbf{q}_{\text{des}}", "form")]


def test_forms():
    assert raws(r"\mathbf{q} + \hat{x} + \mathsf{y}") == [(r"\mathbf{q}", "form"), (r"\hat{x}", "form"), (r"\mathsf{y}", "form")]
    assert raws(r"\mathbb{E}[\NumberOfDevices] \in \mathbb{R}") == []


def test_decorated_macros():
    assert raws(r"\Bandwidth_{\text{bisect}}") == [(r"\Bandwidth_{\text{bisect}}", "decorated")]
    assert raws(r"\Time'") == [(r"\Time'", "decorated")]
    assert raws(r"\Time^2 + \NumberOfDevices^{-1} + \GeneralizedCoordinates^\top + \Time_i") == []
    assert raws(r"\Time_{k+1} + \Time^{(i)}") == []                 # index expressions


def test_unknown_alias_scope():
    assert raws(r"\NotRegistered + \foo") == [(r"\NotRegistered", "unknown"), (r"\foo", "unknown")]
    assert raws(r"\OldBandwidth") == [(r"\OldBandwidth", "alias")]
    assert raws(r"\Time + \LinkBandwidth", scope={"Time"}) == [(r"\LinkBandwidth", "out-of-scope")]


def test_layout_commands_are_not_symbols():
    assert raws(r"\begin{array}{cc} \Time & \Time \\[4pt] \end{array} \mkern-3mu") == []


# --- extraction --------------------------------------------------------------

def test_extraction_skips_code_and_escaped_dollars():
    src = "Cost \\$5 and $N$ here.\n\n```{python}\nx = '$y$'\n```\n\n`$z$` and $$\\beta$$\n"
    assert [s.text for s in extract_math(src)] == ["N", "\\beta"]


def test_closer_followed_by_digit_is_not_math():
    assert extract_math("from $5 to $10 dollars") == []


def test_captions_are_read():
    src = '::: {#fig-x fig-cap="Speed $v$ at $t$."}\n:::\n'
    assert [(s.text, s.source) for s in extract_math(src)] == [("v", "caption"), ("t", "caption")]


def test_fmt_math_and_quiz():
    cell = 'x = fmt_math(r"\\alpha + N")\ny = fmt_math(f"{a} \\\\cdot \\\\beta")\n'
    assert [s.text for s in extract_fmt_math(cell)] == ["\\alpha + N", "1 \\cdot \\beta"]
    nested = 'z = fmt_math(f"2^{{-{alpha:.3f}}} \\\\approx {x}")'
    assert [s.text for s in extract_fmt_math(nested)] == ["2^{-1} \\approx 1"]
    assert [s.text for s in extract_quiz_math('{"q": "Find $p$ when $N=4$"}')] == ["p", "N=4"]


# --- rendering check (needs node + katex) ------------------------------------

node = shutil.which("node") and (ROOT / "binder/tools/notation/node_modules/katex").exists()
MACROS = {"\\NumberOfDevices": "N", "\\LinkBandwidth": "\\beta", "\\DecodeInterval": "T_{\\text{decode}}",
          "\\Bandwidth": "\\text{BW}"}


@pytest.mark.skipif(not node, reason="node or katex missing")
def test_rendering_equal_after_macro_substitution():
    before = r"Ring $2(N-1)\beta$ and $T_{\text{decode}}$ and $x^\beta$."
    after = r"Ring $2(\NumberOfDevices-1)\LinkBandwidth$ and $\DecodeInterval$ and $x^\LinkBandwidth$."
    assert verify_rendering(before, after, "f.qmd", MACROS, MACROS) == []


@pytest.mark.skipif(not node, reason="node or katex missing")
@pytest.mark.parametrize("before,after", [
    (r"$\frac{a}{b}$", r"$\frac{ab}{}$"),
    (r"$x^{10}$", r"$x^10$"),
    (r"$\text{a b}$", r"$\text{ab}$"),
    (r"$2(N-1)$", r"$2(\NumberOfDevices+1)$"),
    (r"$x_{t}$", r"$x$"),
])
def test_rendering_change_is_caught(before, after):
    assert verify_rendering(before, after, "f.qmd", MACROS, MACROS)


@pytest.mark.skipif(not node, reason="node or katex missing")
def test_registry_latex_change_is_caught():
    text = r"$\LinkBandwidth$"
    changed = dict(MACROS, **{"\\LinkBandwidth": "\\gamma"})
    assert verify_rendering(text, text, "f.qmd", MACROS, changed)


# --- deterministic applier ----------------------------------------------------

def _site(src, raw):
    span = extract_math(src)[0]
    for f in judge(span.text, REG):
        if f.raw == raw:
            return {"line": span.line, "col": span.col, "token_start": f.start, "token_end": f.end, "raw": raw}
    raise AssertionError(raw)


def test_apply_replaces_exact_tokens():
    src = "Cost $2(N-1)\\beta$ here.\n"
    a = [dict(_site(src, "N"), macro="NumberOfDevices"), dict(_site(src, "\\beta"), macro="LinkBandwidth")]
    new, refused = apply_assignments(src, a)
    assert refused == [] and new == "Cost $2(\\NumberOfDevices-1)\\LinkBandwidth$ here.\n"


def test_apply_keeps_a_space_before_letters():
    src = "$N x$ and $Nx$\n"
    span2 = extract_math(src)[1]
    a = [{"line": 1, "col": span2.col, "token_start": 0, "token_end": 1, "raw": "N", "macro": "NumberOfDevices"}]
    new, _ = apply_assignments(src, a)
    assert "$\\NumberOfDevices x$" in new


def test_apply_refuses_when_text_moved():
    src = "Cost $2(N-1)$.\n"
    a = [dict(_site(src, "N"), raw="M", macro="NumberOfDevices")]
    new, refused = apply_assignments(src, a)
    assert new == src and refused


def _alias_package():
    site = {"line": 3, "col": 5, "token_start": 0, "token_end": 13, "raw": "\\OldBandwidth",
            "kind": "alias", "source": "body"}
    return {"sections": [{"heading": "Intro", "sites": [site]}]}


def test_alias_without_a_row_gets_its_canonical_macro():
    from notation_strict import assign_from_glossary
    res = assign_from_glossary(_alias_package(), {"rows": []}, REG)
    assert [(a["macro"], a["via"]) for a in res["assignments"]] == [("Bandwidth", "alias")]


def test_glossary_row_overrides_an_alias_used_with_another_meaning():
    from notation_strict import assign_from_glossary
    row = {"symbol": "\\OldBandwidth", "macro": "LinkBandwidth", "lines": [3]}
    res = assign_from_glossary(_alias_package(), {"rows": [row]}, REG)
    assert [(a["macro"], a["via"]) for a in res["assignments"]] == [("LinkBandwidth", "glossary")]


def test_apply_replaces_the_whole_span_a_macro_renders():
    src = "Drift $D(P_t, P_0)$ grows.\n"
    span = extract_math(src)[0]
    toks = __import__("notation_strict").tokenize(span.text)
    k = toks.index("P")
    a = [{"line": 1, "col": span.col, "token_start": k, "token_end": k + 1, "raw": "P", "macro": "CurrentDistribution"}]
    new, refused = apply_assignments(src, a, {"\\CurrentDistribution": "P_{t}"})
    assert refused == [] and new == "Drift $D(\\CurrentDistribution, P_0)$ grows.\n"


def test_apply_keeps_head_only_when_latex_is_one_token():
    src = "Cost $2N_t$.\n"
    span = extract_math(src)[0]
    a = [{"line": 1, "col": span.col, "token_start": 1, "token_end": 2, "raw": "N", "macro": "NumberOfDevices"}]
    new, _ = apply_assignments(src, a, {"\\NumberOfDevices": "N"})
    assert new == "Cost $2\\NumberOfDevices_t$.\n"


def test_glyph_mentions_are_wrapped_and_exempt():
    src = "The book writes $B$ for batch size.\n"
    span = extract_math(src)[0]
    a = [{"line": 1, "col": span.col, "token_start": 0, "token_end": 1, "raw": "B", "macro": "Glyph"}]
    new, refused = apply_assignments(src, a, {"\\Glyph": "#1"})
    assert refused == [] and new == "The book writes $\\Glyph{B}$ for batch size.\n"
    assert raws("\\Glyph{B}") == []


def test_inline_python_inside_math_is_a_value():
    assert raws("\\frac{`{python} Cost.saved_str` \\times N}{`{python} Cost.total_str`}") == [("N", "letter")]


def test_macro_decorations_that_keep_the_symbol():
    reg = Registry([
        RegistryEntry("WeightMatrix", r"\mathbf{W}", "weights", "matrix"),
        RegistryEntry("LayerIndex", "l", "layer index", "index"),
        RegistryEntry("DatasetSize", "D", "dataset size", "scalar"),
        RegistryEntry("DataScalingExponent", r"\beta", "exponent", "scalar"),
        RegistryEntry("ActivationFunction", r"\sigma", "activation", "function"),
        RegistryEntry("Time", "T", "time"),
    ], {}, BUILTINS)
    ok = [r"\WeightMatrix^{(\LayerIndex)}", r"\WeightMatrix^{(\LayerIndex-1)}", r"\WeightMatrix^{(1)}",
          r"\DatasetSize^{\DataScalingExponent}", r"\DatasetSize^{-\DataScalingExponent}", r"\Time^*",
          r"\ActivationFunction'", r"\WeightMatrix_{\LayerIndex}", r"\Time_0"]
    for m in ok:
        assert [f.kind for f in judge(m, reg)] == [], m
    bad = [r"\Time^{\text{new}}", r"\Time'", r"\Time_{\text{disk}}", r"\WeightMatrix^{(\DatasetSize)}"]
    for m in bad:
        assert [f.kind for f in judge(m, reg)] == ["decorated"], m


def test_matrix_entry_positions_are_indices_and_are_kept():
    assert raws("A_{i0}B_{0j}") == [("A", "letter"), ("B", "letter")]
    src = "Entry $A_{i0}$ here.\n"
    span = extract_math(src)[0]
    a = [{"line": 1, "col": span.col, "token_start": 0, "token_end": 6, "raw": "A_{i0}", "macro": "LeftEntry"}]
    new, refused = apply_assignments(src, a, {"\\LeftEntry": "A"})
    assert refused == [] and new == "Entry $\\LeftEntry_{i0}$ here.\n"


def test_glyph_collisions_within_a_chapter():
    from notation_strict import glyph_collisions
    reg = Registry([
        RegistryEntry("BatchSize", "B", "batch size", "count"),
        RegistryEntry("GemmRightOperand", "B", "right operand", "matrix"),
        RegistryEntry("Time", "T", "time"),
    ], {}, BUILTINS)
    assert glyph_collisions("Use $\\BatchSize$ and $\\Time$.\n", reg) == []
    col = glyph_collisions("## One\n\nUse $\\BatchSize$.\n\nThen $\\GemmRightOperand$.\n", reg)
    assert col == [{"section": "One", "latex": "B", "macros": ["BatchSize", "GemmRightOperand"], "lines": [3, 5]}]
    # the same two meanings in different top-level sections do not collide
    assert glyph_collisions("## One\n\n$\\BatchSize$\n\n## Two\n\n$\\GemmRightOperand$\n", reg) == []


def test_apply_drops_a_label_that_belonged_to_the_replaced_symbol():
    src = "Weights $\\mathbf{A}_{\\text{attn}}$ here.\n"
    span = extract_math(src)[0]
    toks = __import__("notation_strict").tokenize(span.text)
    a = [{"line": 1, "col": span.col, "token_start": 0, "token_end": len(toks), "raw": span.text, "macro": "AttentionWeightMatrix"}]
    new, refused = apply_assignments(src, a, {"\\AttentionWeightMatrix": "\\mathbf{A}"})
    assert refused == [] and new == "Weights $\\AttentionWeightMatrix$ here.\n"
