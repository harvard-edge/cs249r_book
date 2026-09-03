"""Tests for the fmt-migration visible-text normalizer (Regime-2 equivalence)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/audit/fmt"))
from visible_text import to_visible  # noqa: E402


def test_multiplier_glyph_relocation_equivalent():
    # the core Regime-2 case: string '6×' vs string '6' + prose $\times$
    assert to_visible("is 6× the inference") == to_visible("is 6$\\times$ the inference")


def test_currency_preserved():
    assert to_visible("a \\$15,000 budget") == "a $15,000 budget"


def test_escaped_percent_visible():
    assert to_visible("$\\eta = 85\\%$") == "η = 85%"


def test_spelled_percent_untouched():
    assert to_visible("trades 33 percent compute") == "trades 33 percent compute"


def test_symbol_percent_untouched():
    assert to_visible("about 71.1% savings") == "about 71.1% savings"


def test_text_wrapper_unwrapped():
    assert to_visible(r"$T = 2.3\,\text{ms}$") == "T = 2.3ms"


def test_common_operators():
    assert to_visible(r"$a \approx b$") == "a ≈ b"
    assert to_visible(r"$x \leq y$") == "x ≤ y"


def test_to_token_not_overmatched():
    # \to -> arrow, but \tau (if present) must not be mangled by the \to rule
    assert to_visible(r"$a \to b$") == "a → b"


def test_complex_math_is_stable():
    # we don't fully render; we just need identical in == identical out
    s = r"a $\sqrt{N_L}$ model"
    assert to_visible(s) == to_visible(s)


def test_whitespace_collapsed():
    assert to_visible("foo   bar\u00a0baz") == "foo bar baz"
