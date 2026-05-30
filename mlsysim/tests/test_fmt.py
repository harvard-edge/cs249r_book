"""Tests for mlsysim.fmt formatting guards."""

import pytest

from mlsysim.core.constants import ureg
from mlsysim.core.units import USD
from mlsysim.fmt import MarkdownStr, fmt, fmt_int, fmt_percent, fmt_qty, fmt_usd


class TestFmtPrecisionGuards:
    def test_precision_zero_accepts_whole_numbers(self):
        assert fmt(989, precision=0, commas=False) == "989"
        assert fmt(10.0, precision=0, commas=False) == "10"
        assert fmt(175.0, precision=0, commas=False) == "175"

    def test_precision_zero_rejects_small_fractions_that_display_as_zero(self):
        with pytest.raises(ValueError, match="formatted as '0'"):
            fmt(0.1, precision=0, commas=False)
        with pytest.raises(ValueError, match="formatted as '0'"):
            fmt(0.4, precision=0, commas=False)

    def test_precision_zero_rejects_non_integer_values(self):
        with pytest.raises(ValueError, match="not integer-like"):
            fmt(10.7, precision=0, commas=False)
        with pytest.raises(ValueError, match="not integer-like"):
            fmt(84.7, precision=0, commas=False)
        with pytest.raises(ValueError, match="not integer-like"):
            fmt(4.1, precision=0, commas=False)

    def test_precision_zero_accepts_explicit_round(self):
        assert fmt(round(10.7), precision=0, commas=False) == "11"
        assert fmt(round(362507.545), precision=0, commas=True) == "362,508"

    def test_precision_one_preserves_fractions(self):
        assert fmt(10.7, precision=1, commas=False) == "10.7"
        assert fmt(8.5, precision=1, commas=False) == "8.5"

    def test_precision_one_rejects_spurious_trailing_zeros_on_integers(self):
        with pytest.raises(ValueError, match="spurious trailing zeros"):
            fmt(512.0, precision=1, commas=False)
        with pytest.raises(ValueError, match="spurious trailing zeros"):
            fmt(989, precision=1, commas=False)

    def test_returns_markdown_str(self):
        out = fmt(42, precision=0, commas=False)
        assert isinstance(out, MarkdownStr)
        assert out._repr_markdown_() == "42"


class TestFmtInt:
    def test_rounds_computed_values_explicitly(self):
        assert fmt_int(120.28, commas=False) == "120"
        assert fmt_int(10.7, commas=False) == "11"
        assert fmt_int(362507.545, commas=True) == "362,508"

    def test_accepts_prefix_and_suffix(self):
        assert fmt_int(175, commas=False, suffix=" billion") == "175 billion"


class TestFmtQty:
    def test_mj_over_ms_to_mw(self):
        energy = 66 * ureg.millijoule
        time = 1000 * ureg.millisecond
        power = energy / time
        out = fmt_qty(power, ureg.mW, precision=0, commas=False)
        assert out == "66 mW"

    def test_gb_display(self):
        mem = 140 * ureg.GB
        out = fmt_qty(mem, ureg.GB, precision=0, commas=False)
        assert out == "140 GB"

    def test_currency_is_refused(self):
        # Currency must go through fmt_usd, not fmt_qty: fmt_qty cannot emit the
        # Pandoc-safe escaped "\\$" and would leak a literal " USD" suffix.
        price = 2.5 * USD
        with pytest.raises(ValueError, match="fmt_usd"):
            fmt_qty(price, USD, precision=2, commas=False)

    def test_returns_markdown_str(self):
        out = fmt_qty(5 * ureg.millisecond, ureg.millisecond, precision=0, commas=False)
        assert isinstance(out, MarkdownStr)
        assert out == "5 ms"


class TestFmtUsd:
    def test_basic_dollar_is_escaped(self):
        # The escaped "\\$" is mandatory so prose never enters math mode.
        assert fmt_usd(15000) == "\\$15,000"
        assert fmt_usd(10, commas=False) == "\\$10"

    def test_rounds_to_whole_dollars_at_precision_zero(self):
        assert fmt_usd(12345.6) == "\\$12,346"
        assert fmt_usd(999.4) == "\\$999"

    def test_precision_preserves_cents(self):
        assert fmt_usd(0.09, precision=2, commas=False, suffix="/GB") == "\\$0.09/GB"
        assert fmt_usd(4.6, precision=1, suffix="M") == "\\$4.6M"

    def test_approx_prepends_tilde(self):
        assert fmt_usd(1234.7, approx=True, suffix="/year") == "~\\$1,235/year"

    def test_accepts_pure_dollar_quantity(self):
        assert fmt_usd(2500 * USD) == "\\$2,500"

    def test_never_emits_literal_usd(self):
        for out in (fmt_usd(5), fmt_usd(5 * USD, suffix="/hr"), fmt_usd(5, approx=True)):
            assert "USD" not in out

    def test_returns_markdown_str(self):
        assert isinstance(fmt_usd(100), MarkdownStr)


class TestFmtPercentGuards:
    def test_precision_zero_accepts_whole_percentages(self):
        assert fmt_percent(0.45, precision=0, commas=False) == "45"

    def test_precision_zero_rejects_fractional_percentages(self):
        with pytest.raises(ValueError, match="not integer-like"):
            fmt_percent(0.456, precision=0, commas=False)
