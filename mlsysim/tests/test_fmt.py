"""Tests for mlsysim.fmt formatting guards."""

import pytest

from mlsysim.core.constants import ureg
from mlsysim.core.units import USD
import math

from mlsysim.fmt import (
    MarkdownStr,
    fmt,
    fmt_count,
    fmt_count_range,
    fmt_int,
    fmt_multiple,
    fmt_percent,
    fmt_pp,
    fmt_qty,
    fmt_qty_range,
    fmt_range,
    fmt_rate,
    fmt_ratio,
    fmt_sci,
    fmt_time,
    fmt_time_range,
    fmt_usd,
    fmt_usd_range,
    fmt_val,
)


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

    def test_named_display_markers_replace_raw_prefixes(self):
        assert fmt(100, precision=0, commas=False, approx=True, suffix=" MB/s") == "~100 MB/s"
        assert fmt(1000, precision=0, lower_bound=True, suffix=" MB/s") == "> 1,000 MB/s"

    def test_rejects_conflicting_display_markers(self):
        with pytest.raises(ValueError, match="both approximate and a lower bound"):
            fmt(100, precision=0, approx=True, lower_bound=True)
        with pytest.raises(ValueError, match="Use either prefix="):
            fmt(100, precision=0, prefix="~", approx=True)


class TestFmtInt:
    def test_rounds_computed_values_explicitly(self):
        assert fmt_int(120.28, commas=False) == "120"
        assert fmt_int(10.7, commas=False) == "11"
        assert fmt_int(362507.545, commas=True) == "362,508"

    def test_accepts_prefix_and_suffix(self):
        assert fmt_int(175, commas=False, suffix=" billion") == "175 billion"

    def test_accepts_named_approx_marker(self):
        assert fmt_int(80, commas=False, approx=True, suffix=" GB") == "~80 GB"


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

    def test_accepts_named_marker(self):
        mem = 140 * ureg.GB
        out = fmt_qty(mem, ureg.GB, precision=0, commas=False, approx=True)
        assert out == "~140 GB"

    def test_currency_is_refused(self):
        # Currency must go through fmt_usd, not fmt_qty: fmt_qty cannot emit the
        # Pandoc-safe escaped "\\$" and would leak a literal " USD" suffix.
        price = 2.5 * USD
        with pytest.raises(ValueError, match="fmt_usd"):
            fmt_qty(price, USD, precision=2, commas=False)

    def test_plain_number_is_refused(self):
        with pytest.raises(TypeError, match="requires a Pint Quantity"):
            fmt_qty(5, ureg.GB, precision=0, commas=False)

    def test_returns_markdown_str(self):
        out = fmt_qty(5 * ureg.millisecond, ureg.millisecond, precision=0, commas=False)
        assert isinstance(out, MarkdownStr)
        assert out == "5 ms"

    def test_structured_denominator(self):
        energy = 0.1 * ureg.millijoule
        out = fmt_qty(
            energy,
            ureg.millijoule,
            precision=1,
            commas=False,
            per="inference",
        )
        assert out == "0.1 mJ/inference"

    def test_rejects_legacy_and_structured_denominator_mix(self):
        with pytest.raises(ValueError, match="extra_suffix="):
            fmt_qty(
                1 * ureg.millijoule,
                ureg.millijoule,
                precision=0,
                extra_suffix="/token",
                per="inference",
            )


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

    def test_structured_scale_and_denominator(self):
        assert fmt_usd(4_600_000, precision=1, commas=False, scale="M") == "\\$4.6M"
        assert fmt_usd(0.09, precision=2, commas=False, per="GB") == "\\$0.09/GB"
        assert fmt_usd(12_000, commas=False, scale="K", per="year") == "\\$12K/year"
        assert fmt_usd(8000, approx=True, marker="*") == "~\\$8,000*"

    def test_rejects_legacy_suffix_with_structured_parts(self):
        with pytest.raises(ValueError, match="suffix="):
            fmt_usd(1000, scale="K", suffix="K")
        with pytest.raises(ValueError, match="suffix="):
            fmt_usd(1000, marker="*", suffix="*")

    def test_rejects_bad_denominator(self):
        with pytest.raises(ValueError, match="omit the leading"):
            fmt_usd(0.09, precision=2, per="/GB")
        with pytest.raises(ValueError, match="fmt_usd per must be"):
            fmt_usd(0.09, precision=2, per="widgets")

    def test_rejects_bad_marker(self):
        with pytest.raises(ValueError, match="fmt_usd marker"):
            fmt_usd(1000, marker="note")

    def test_approx_prepends_tilde(self):
        assert fmt_usd(1234.7, approx=True, suffix="/year") == "~\\$1,235/year"

    def test_accepts_pure_dollar_quantity(self):
        assert fmt_usd(2500 * USD) == "\\$2,500"

    def test_never_emits_literal_usd(self):
        for out in (fmt_usd(5), fmt_usd(5 * USD, suffix="/hr"), fmt_usd(5, approx=True)):
            assert "USD" not in out

    def test_returns_markdown_str(self):
        assert isinstance(fmt_usd(100), MarkdownStr)


class TestFmtCountLegacy:
    def test_accepts_named_marker(self):
        assert fmt_count(1024, suffix=" GPUs", approx=True) == "~1,024 GPUs"


class TestFmtRate:
    def test_formats_allowlisted_service_rates(self):
        assert fmt_rate(2500, "QPS") == "2,500 QPS"
        assert fmt_rate(1200, "tokens/s") == "1,200 tokens/s"
        assert fmt_rate(2500, "QPS", commas=False) == "2500 QPS"
        assert fmt_rate(60, "FPS") == "60 FPS"

    def test_rejects_unknown_rate_unit(self):
        with pytest.raises(ValueError, match="fmt_rate unit must be"):
            fmt_rate(10, "GB/s")

    def test_rejects_negative_by_default(self):
        with pytest.raises(ValueError, match="non-negative rate"):
            fmt_rate(-1, "QPS")
        assert fmt_rate(-1, "QPS", allow_negative=True) == "-1 QPS"

    def test_returns_markdown_str(self):
        assert isinstance(fmt_rate(1, "QPS"), MarkdownStr)


class TestFmtTime:
    def test_symbol_style_accepts_quantities_and_plain_numbers(self):
        assert fmt_time(1500 * ureg.millisecond, ureg.second) == "1.5 s"
        assert fmt_time(35, ureg.second, precision=0) == "35 s"
        assert fmt_time(35, "second", precision=0) == "35 s"
        assert fmt_time(35, "s", precision=0) == "35 s"
        assert fmt_time(12, "millisecond", precision=0) == "12 ms"
        assert fmt_time(12, "ms", precision=0) == "12 ms"
        assert fmt_time(5, "microsecond", precision=0) == "5 μs"
        assert fmt_time(5, "µs", precision=0) == "5 μs"
        assert fmt_time(5, "μs", precision=0) == "5 μs"

    def test_word_style_pluralizes(self):
        assert fmt_time(1, ureg.second, precision=0, style="word") == "1 second"
        assert fmt_time(2, ureg.second, precision=0, style="word") == "2 seconds"
        assert fmt_time(2, "year", precision=0, style="word") == "2 years"

    def test_attributive_word_style_is_hyphenated_singular(self):
        assert (
            fmt_time(1, "hour", precision=0, style="word", attributive=True)
            == "1-hour"
        )
        assert (
            fmt_time(24, "hour", precision=0, style="word", attributive=True)
            == "24-hour"
        )
        assert (
            fmt_time(15, "minute", precision=0, style="word", attributive=True)
            == "15-minute"
        )

    def test_attributive_rejects_symbol_style_and_per(self):
        with pytest.raises(ValueError, match="attributive"):
            fmt_time(5, "hour", precision=0, attributive=True)
        with pytest.raises(ValueError, match="per="):
            fmt_time(
                5,
                "hour",
                precision=0,
                style="word",
                attributive=True,
                per="day",
            )

    def test_checked_symbol_marker(self):
        assert fmt_time(100, "millisecond", precision=0, marker="+") == "100 ms+"
        with pytest.raises(ValueError, match="marker"):
            fmt_time(100, "millisecond", precision=0, marker="ish")
        with pytest.raises(ValueError, match="symbol"):
            fmt_time(100, "millisecond", precision=0, style="word", marker="+")
        with pytest.raises(ValueError, match="cannot be combined"):
            fmt_time(100, "millisecond", precision=0, per="step", marker="+")

    def test_rejects_non_time_unit(self):
        with pytest.raises(ValueError, match="time unit"):
            fmt_time(5, ureg.GB)

    def test_rejects_negative_by_default(self):
        with pytest.raises(ValueError, match="non-negative duration"):
            fmt_time(-1, ureg.second, precision=0)
        assert fmt_time(-1, ureg.second, precision=0, allow_negative=True) == "-1 s"

    def test_returns_markdown_str(self):
        assert isinstance(fmt_time(1, ureg.second, precision=0), MarkdownStr)


class TestFmtPercentGuards:
    def test_precision_zero_accepts_whole_percentages(self):
        assert fmt_percent(0.45, precision=0, commas=False) == "45"

    def test_precision_zero_rejects_fractional_percentages(self):
        with pytest.raises(ValueError, match="not integer-like"):
            fmt_percent(0.456, precision=0, commas=False)

    def test_default_style_is_bare_number_backward_compatible(self):
        # The 11 pre-existing call sites rely on the bare-number default.
        assert fmt_percent(0.85, precision=0) == "85"
        assert fmt_percent(0.123, precision=1) == "12.3"

    def test_prose_style_owns_the_word_percent(self):
        assert fmt_percent(0.85, precision=0, style="prose") == "85 percent"

    def test_symbol_style_owns_the_glyph(self):
        assert fmt_percent(0.85, precision=0, style="symbol") == "85%"

    def test_rejects_already_scaled_value_the_10000_percent_bug(self):
        with pytest.raises(ValueError, match="0-1 ratio"):
            fmt_percent(85)
        with pytest.raises(ValueError, match="0-1 ratio"):
            fmt_percent(45, precision=0)

    def test_allows_over_100_percent_only_with_explicit_max_ratio(self):
        with pytest.raises(ValueError, match="0-1 ratio"):
            fmt_percent(2.0, precision=0)
        assert fmt_percent(2.0, precision=0, max_ratio=3) == "200"

    def test_rejects_negative_ratio_by_default(self):
        # a bounded proportion (accuracy, utilization) is never negative
        with pytest.raises(ValueError, match="0-1 ratio"):
            fmt_percent(-0.05, precision=0)

    def test_allow_negative_widens_domain_for_signed_change(self):
        # ROI / cost-delta: signed, may exceed 100% -> opt in explicitly
        assert fmt_percent(-0.818, precision=1, style="symbol",
                           allow_negative=True) == "-81.8%"
        assert fmt_percent(8.089, precision=1, style="symbol",
                           allow_negative=True, max_ratio=9) == "808.9%"

    def test_allow_negative_still_bounds_magnitude(self):
        # the guard still catches a true 100x blunder even when signed
        with pytest.raises(ValueError, match="0-1 ratio"):
            fmt_percent(-85, allow_negative=True)

    def test_rejects_unknown_style(self):
        with pytest.raises(ValueError, match="style must be"):
            fmt_percent(0.5, style="pct")

    def test_returns_markdown_str(self):
        assert isinstance(fmt_percent(0.5, precision=0, style="prose"), MarkdownStr)


class TestFmtPp:
    def test_prose_default(self):
        assert fmt_pp(7.0, precision=0) == "7 percentage points"

    def test_symbol_style(self):
        assert fmt_pp(7.0, precision=0, style="symbol") == "7 pp"

    def test_not_multiplied_by_100(self):
        # pp is already on the 0-100 point scale; 7 stays 7, not 700.
        assert fmt_pp(7.0, precision=0).startswith("7 ")

    def test_returns_markdown_str(self):
        assert isinstance(fmt_pp(3.0, precision=0), MarkdownStr)

    def test_singular_when_value_is_one(self):
        assert fmt_pp(1.0, precision=0) == "1 percentage point"

    def test_plural_for_fractional_near_one(self):
        assert fmt_pp(0.9, precision=1) == "0.9 percentage points"
        assert fmt_pp(1.5, precision=1) == "1.5 percentage points"

    def test_pluralization_follows_rendered_number(self):
        # any integer != 1 is plural
        assert fmt_pp(2.0, precision=0) == "2 percentage points"
        # a fractional value is always plural, even just below 1
        assert fmt_pp(0.5, precision=1) == "0.5 percentage points"

    def test_attributive_is_hyphenated_singular_word(self):
        assert fmt_pp(5.0, precision=0, attributive=True) == "5 percentage-point"
        # attributive stays singular/hyphenated regardless of magnitude
        assert fmt_pp(12.0, precision=0, attributive=True) == "12 percentage-point"

    def test_attributive_rejected_for_symbol_style(self):
        with pytest.raises(ValueError, match="attributive"):
            fmt_pp(5.0, style="symbol", attributive=True)


class TestFmtMultiple:
    def test_number_only_no_glyph(self):
        assert fmt_multiple(3.2) == "3.2"
        assert fmt_multiple(10, precision=0) == "10"

    def test_inherits_fmt_precision_guard(self):
        # An integer-like factor at precision=1 would render "2.0" — the
        # shared fmt() guard rejects that; the caller picks precision=0.
        with pytest.raises(ValueError, match="spurious trailing zeros"):
            fmt_multiple(2.0, precision=1)
        assert fmt_multiple(2.0, precision=0) == "2"

    def test_rejects_negative_factor(self):
        with pytest.raises(ValueError, match="non-negative factor"):
            fmt_multiple(-3)

    def test_returns_markdown_str(self):
        assert isinstance(fmt_multiple(2.5), MarkdownStr)


class TestFiniteGuard:
    """The divide-by-zero / non-finite last line of defense, on every helper."""

    INF = float("inf")
    NAN = float("nan")

    def test_fmt_rejects_inf_and_nan(self):
        with pytest.raises(ValueError, match="Non-finite"):
            fmt(self.INF, precision=1)
        with pytest.raises(ValueError, match="Non-finite"):
            fmt(self.NAN, precision=1)

    def test_message_names_divide_by_zero(self):
        with pytest.raises(ValueError, match="divide-by-zero"):
            fmt(self.INF, precision=0)

    def test_guard_on_every_numeric_helper(self):
        for call in (
            lambda: fmt_int(self.INF),
            lambda: fmt_usd(self.INF),
            lambda: fmt_percent(self.NAN),
            lambda: fmt_multiple(self.INF),
            lambda: fmt_count(self.INF),
            lambda: fmt_ratio(self.INF),
            lambda: fmt_val(self.NAN),
            lambda: fmt_sci(self.INF),
            lambda: fmt_qty(self.INF * ureg.millisecond, ureg.millisecond),
            lambda: fmt_time(self.INF, ureg.second),
            lambda: fmt_rate(self.INF, "QPS"),
        ):
            with pytest.raises(ValueError, match="Non-finite"):
                call()

    def test_finite_values_pass(self):
        assert fmt(3.0, precision=0) == "3"
        assert fmt_ratio(5.3, precision=1) == "5.3"


class TestFmtRatio:
    def test_bare_number_no_decoration(self):
        assert fmt_ratio(5.3, precision=1) == "5.3"
        assert fmt_ratio(3.2) == "3.2"
        assert fmt_ratio(5.0, precision=0) == "5"

    def test_rejects_negative_by_default(self):
        with pytest.raises(ValueError, match="non-negative ratio"):
            fmt_ratio(-2.0)

    def test_allows_signed_ratio_with_flag(self):
        assert fmt_ratio(-2.0, precision=0, allow_negative=True) == "-2"

    def test_returns_markdown_str(self):
        assert isinstance(fmt_ratio(1.5), MarkdownStr)


class TestFmtCount:
    def test_no_scale_uses_commas(self):
        assert fmt_count(8192) == "8,192"
        assert fmt_count(1024, suffix=" GPUs") == "1,024 GPUs"

    def test_structured_label_pluralizes_from_raw_count(self):
        assert fmt_count(1, label="GPU") == "1 GPU"
        assert fmt_count(2, label="GPU") == "2 GPUs"
        assert fmt_count(1024, label="GPU") == "1,024 GPUs"
        assert fmt_count(2, label="query") == "2 queries"
        assert fmt_count(2, label="batch", plural_label="batches") == "2 batches"

    def test_scale_glyphs(self):
        assert fmt_count(5_000_000, scale="M") == "5M"
        assert fmt_count(5_300_000, scale="M", precision=1) == "5.3M"
        assert fmt_count(70e9, scale="B") == "70B"
        assert fmt_count(70e9, scale="B", label="parameter") == "70B parameters"

    def test_scale_words(self):
        assert fmt_count(1_000_000, scale="M", scale_style="word") == "1 million"
        assert (
            fmt_count(
                60_000_000,
                scale="M",
                scale_style="word",
                label="parameter",
            )
            == "60 million parameters"
        )
        assert (
            fmt_count(
                70_000_000_000,
                scale="B",
                scale_style="word",
                precision=0,
                commas=False,
                label="parameter",
            )
            == "70 billion parameters"
        )

    def test_scale_inherits_precision_guard(self):
        # 5.3M at precision=0 would silently hide the .3 — guard refuses.
        with pytest.raises(ValueError, match="not integer-like"):
            fmt_count(5_300_000, scale="M", precision=0)

    def test_rejects_negative_count(self):
        with pytest.raises(ValueError, match="non-negative count"):
            fmt_count(-5)

    def test_rejects_fractional_count_by_default(self):
        with pytest.raises(ValueError, match="whole-number count"):
            fmt_count(1.5, label="GPU", precision=1)
        assert (
            fmt_count(1.5, label="GPU", precision=1, allow_fractional=True)
            == "1.5 GPUs"
        )

    def test_rejects_unknown_scale(self):
        with pytest.raises(ValueError, match="scale must be"):
            fmt_count(1000, scale="G")

    def test_rejects_unknown_scale_style(self):
        with pytest.raises(ValueError, match="scale_style"):
            fmt_count(1000, scale="K", scale_style="long")
        with pytest.raises(ValueError, match="requires scale"):
            fmt_count(1000, scale_style="word")

    def test_rejects_label_suffix_conflicts_and_unit_like_labels(self):
        with pytest.raises(ValueError, match="structured label"):
            fmt_count(2, label="GPU", suffix=" GPUs")
        with pytest.raises(ValueError, match="looks like a unit"):
            fmt_count(2, label="QPS")

    def test_returns_markdown_str(self):
        assert isinstance(fmt_count(1000, scale="K"), MarkdownStr)


class TestFmtRange:
    def test_uses_en_dash_not_hyphen(self):
        out = fmt_range(5, 10, precision=0)
        assert out == "5\u201310"
        assert "-" not in out  # never an ASCII hyphen

    def test_endpoints_written_in_full(self):
        # MIT: 1992-1993, never 1992-93
        assert fmt_range(1992, 1993, precision=0, commas=False) == "1992\u20131993"

    def test_unit_appended_once(self):
        assert fmt_range(5, 10, precision=0, unit="GB") == "5\u201310 GB"
        assert fmt_range(2, 4, precision=0, unit="percent") == "2\u20134 percent"

    def test_usd_kind_each_endpoint_carries_dollar(self):
        assert fmt_range(0.10, 0.50, kind="usd", precision=2) == "\\$0.10\u2013\\$0.50"

    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="hi >= lo"):
            fmt_range(10, 5, precision=0)

    def test_rejects_non_finite_endpoint(self):
        with pytest.raises(ValueError, match="finite|infinite"):
            fmt_range(float("inf"), 5)

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="kind must be"):
            fmt_range(5, 10, precision=0, kind="percent")

    def test_returns_markdown_str(self):
        assert isinstance(fmt_range(5, 10, precision=0), MarkdownStr)


class TestTypedRanges:
    def test_quantity_range_appends_unit_once(self):
        out = fmt_qty_range(
            1 * ureg.GB,
            2 * ureg.GB,
            ureg.GB,
            precision=0,
            commas=False,
        )
        assert out == "1\u20132 GB"

    def test_time_range_symbol_and_word_styles(self):
        assert (
            fmt_time_range(5, 20, ureg.millisecond, precision=0, commas=False)
            == "5\u201320 ms"
        )
        assert (
            fmt_time_range(1, 2, ureg.second, precision=0, style="word",
                           commas=False)
            == "1\u20132 seconds"
        )

    def test_count_range_pluralizes_from_raw_count(self):
        assert fmt_count_range(1, 2, label="GPU", commas=False) == "1\u20132 GPUs"
        assert (
            fmt_count_range(
                1_000_000,
                2_000_000,
                scale="M",
                label="parameter",
                commas=False,
            )
            == "1M\u20132M parameters"
        )

    def test_usd_range_supports_scale_and_denominator(self):
        assert (
            fmt_usd_range(10_000, 30_000, scale="K", commas=False)
            == "\\$10K\u2013\\$30K"
        )
        assert (
            fmt_usd_range(25000, 30000, approx=True, repeat_symbol=False)
            == "~\\$25,000\u201330,000"
        )
        assert (
            fmt_usd_range(0.10, 0.50, precision=2, commas=False, per="GB")
            == "\\$0.10\u2013\\$0.50/GB"
        )

    def test_ranges_reject_inverted_endpoints(self):
        with pytest.raises(ValueError, match="hi >= lo"):
            fmt_qty_range(2 * ureg.GB, 1 * ureg.GB, ureg.GB, precision=0)
        with pytest.raises(ValueError, match="hi >= lo"):
            fmt_count_range(2, 1)
        with pytest.raises(ValueError, match="hi >= lo"):
            fmt_usd_range(2, 1)
