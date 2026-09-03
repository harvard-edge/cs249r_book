from book.cli.commands.validate import (
    _inline_python_latex_operator_warnings,
    _inline_python_math_spans,
)


def test_inline_python_between_math_spans_is_not_inside_math():
    line = "Training distribution $P_0$: `{python} KLDrift.P0_vec_str`. Serving distribution $P_t$."
    assert _inline_python_math_spans(line) == []


def test_inline_python_inside_dollar_math_is_flagged():
    line = "The value is $x = `{python} Example.x_str`$."
    assert _inline_python_math_spans(line) == ["$x = `{python} Example.x_str`$"]


def test_inline_python_inside_paren_math_is_flagged():
    line = r"The value is \(x = `{python} Example.x_str`\)."
    assert _inline_python_math_spans(line) == [r"\(x = `{python} Example.x_str`\)"]


def test_qualified_str_ref_adjacent_to_latex_operator_is_not_raw():
    line = r"`{python} Example.x_str` $\times$ `{python} Example.y_str`"
    assert _inline_python_latex_operator_warnings(line) == []


def test_noncanonical_ref_adjacent_to_latex_operator_is_warned():
    line = r"`{python} Example.x` $\times$ `{python} Example.y_str`"
    assert _inline_python_latex_operator_warnings(line) == ["`{python} Example.x`"]
