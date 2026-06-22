"""Tests for mlsysim.fmt formatting guards."""

import pytest

from mlsysim.core.units import ureg
from mlsysim.core.units import GB, J, K, MB, TB, USD, hour, kg, kWh, second
import math

from mlsysim.fmt import (
    MarkdownStr,
    fmt,
    fmt_arithmetic_intensity,
    fmt_area,
    fmt_count,
    fmt_count_range,
    fmt_compute_efficiency,
    fmt_carbon_intensity,
    fmt_decibel,
    fmt_display_math,
    fmt_emissions,
    fmt_eur,
    fmt_int,
    fmt_energy_per_bit,
    fmt_energy_per_byte,
    fmt_energy_per_flop,
    fmt_energy_per_op,
    fmt_heat_flux,
    fmt_illuminance,
    fmt_multiple,
    fmt_multiple_range,
    fmt_memory_capacity,
    fmt_percent,
    fmt_percent_range,
    fmt_pp,
    fmt_params,
    fmt_power,
    fmt_qty,
    fmt_qty_int,
    fmt_qty_range,
    fmt_magnitude,
    fmt_bandwidth,
    fmt_flop_rate,
    fmt_flops,
    fmt_length,
    fmt_ops_rate,
    fmt_water,
    fmt_water_rate,
    fmt_water_intensity,
    fmt_range,
    fmt_rate,
    fmt_fps,
    fmt_ratio,
    fmt_sci,
    fmt_sci_flops,
    fmt_sci_qty,
    fmt_specific_heat,
    fmt_temperature,
    fmt_temperature_rate,
    fmt_text,
    fmt_time,
    fmt_time_range,
    fmt_tokens,
    fmt_unit,
    fmt_usd,
    fmt_usd_range,
    fmt_val,
)


class TestFmtComposition:
    def test_fmt_text_preserves_markdown_rendering(self):
        out = fmt_text(
            fmt_usd(400_000),
            " $\\times$ ",
            fmt(0.4, precision=2),
            " = ",
            fmt_usd(160_000),
        )

        assert out == "\\$400,000 $\\times$ 0.40 = \\$160,000"
        assert isinstance(out, MarkdownStr)
        assert out._repr_markdown_() == str(out)

    def test_fmt_display_math_wraps_block_math(self):
        out = fmt_display_math(r"x = 1")

        assert out == r"$$x = 1$$"
        assert isinstance(out, MarkdownStr)


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

    def test_fmt_precision_none_auto_trims_integer_like_values(self):
        assert fmt(153.0, precision=None, commas=False) == "153"
        assert fmt(153.016, precision=None, commas=False) == "153"
        assert fmt(2.04, precision=None, commas=False) == "2.04"
        assert fmt(0.2, precision=None, commas=False) == "0.2"

    def test_auto_precision_preserves_large_fractions(self):
        assert (
            fmt_arithmetic_intensity(
                153.016 * ureg.flop / ureg.byte,
                commas=False,
            )
            == "153 FLOP/byte"
        )

    def test_auto_precision_keeps_meaningful_small_decimals(self):
        assert fmt_bandwidth(0.2 * MB / second, unit=MB / second, commas=False) == "0.2 MB/s"
        assert fmt_bandwidth(2.04 * TB / second, unit=TB / second, commas=False) == "2.04 TB/s"

    def test_auto_precision_trim_preserves_commas(self):
        assert (
            fmt_bandwidth(1000.01 * MB / second, unit=MB / second, commas=True)
            == "1,000 MB/s"
        )

    def test_explicit_precision_preserves_requested_decimal(self):
        assert (
            fmt_arithmetic_intensity(
                153.016 * ureg.flop / ureg.byte,
                precision=1,
                commas=False,
            )
            == "153.0 FLOP/byte"
        )

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
        with pytest.raises(ValueError, match="only one display marker"):
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

    def test_accepts_unit_keyword(self):
        mem = 140 * ureg.GB
        assert fmt_qty(mem, unit=ureg.GB, precision=0, commas=False) == "140 GB"
        with pytest.raises(TypeError, match="either a positional unit or unit="):
            fmt_qty(mem, ureg.GB, unit=ureg.GB, precision=0)

    def test_fmt_magnitude_keeps_unit_conversion_checked(self):
        q = 2048 * ureg.MB
        assert fmt_magnitude(q, unit=ureg.GB, precision=3, commas=False) == "2.048"

    def test_fmt_magnitude_requires_quantity(self):
        with pytest.raises(TypeError, match="requires a Pint Quantity"):
            fmt_magnitude(2048, unit=ureg.GB, precision=0)

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

    def test_unit_label_preserves_dimension_check(self):
        q = 5 * ureg.flop / ureg.byte
        assert (
            fmt_qty(q, ureg.flop / ureg.byte, precision=0, unit_label="FLOP/byte") == "5 FLOP/byte"
        )
        with pytest.raises(Exception):
            fmt_qty(5 * ureg.second, ureg.flop / ureg.byte, unit_label="FLOP/byte")

    def test_unit_label_rejects_value_kind_glyphs(self):
        with pytest.raises(ValueError, match="currency, percent, or multiplier"):
            fmt_qty(5 * ureg.GB, ureg.GB, unit_label="GB%")

    def test_rejects_legacy_and_structured_denominator_mix(self):
        with pytest.raises(ValueError, match="extra_suffix="):
            fmt_qty(
                1 * ureg.millijoule,
                ureg.millijoule,
                precision=0,
                extra_suffix="/token",
                per="inference",
            )


class TestFmtQtyInt:
    def test_rounds_quantity_after_unit_conversion(self):
        assert fmt_qty_int(100 * ureg.GiB, ureg.GB, commas=False) == "107 GB"

    def test_supports_structured_quantity_suffixes(self):
        out = fmt_qty_int(
            1250 * ureg.MB / ureg.second,
            ureg.GB / ureg.second,
            commas=False,
        )
        assert out == "1 GB/s"

    def test_accepts_unit_keyword(self):
        assert fmt_qty_int(100 * ureg.GB, unit=ureg.GB, commas=False) == "100 GB"

    def test_plain_number_is_refused(self):
        with pytest.raises(TypeError, match="requires a Pint Quantity"):
            fmt_qty_int(5, ureg.GB, commas=False)

    def test_returns_markdown_str(self):
        assert isinstance(fmt_qty_int(5 * ureg.GB, ureg.GB), MarkdownStr)


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
        assert fmt_usd(4_750_000, precision=2, commas=False, scale="million") == "\\$4.75 million"
        assert fmt_usd(4_750_000, precision=2, commas=False, scale="Million") == "\\$4.75 Million"
        assert fmt_usd(0.09, precision=2, commas=False, per="GB") == "\\$0.09/GB"
        assert fmt_usd(0.50, precision=2, commas=False, per="click") == "\\$0.50/click"
        assert fmt_usd(10_000, per="run") == "\\$10,000/run"
        assert fmt_usd(10, commas=False, per="million queries") == "\\$10/million queries"
        assert (
            fmt_usd(0.06, precision=2, commas=False, per="1K inferences") == "\\$0.06/1K inferences"
        )
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

    def test_accepts_dollar_rate_quantity_with_physical_denominator(self):
        assert fmt_usd(0.09 * USD / GB, precision=2, commas=False, per="GB") == "\\$0.09/GB"
        assert fmt_usd(5 * USD / hour, commas=False, per="hr") == "\\$5/hr"
        assert (
            fmt_usd(23 * USD / (TB * ureg.month), commas=False, per="TB/month") == "\\$23/TB/month"
        )
        assert fmt_usd(0.12 * USD / kWh, precision=2, commas=False, per="kWh") == "\\$0.12/kWh"

    def test_rejects_non_currency_quantities(self):
        with pytest.raises(ValueError, match="dollar Quantity"):
            fmt_usd(3 * GB)

    def test_rejects_rate_quantity_without_denominator(self):
        with pytest.raises(ValueError, match="no per="):
            fmt_usd(0.09 * USD / GB, precision=2)

    def test_never_emits_literal_usd(self):
        for out in (fmt_usd(5), fmt_usd(5 * USD, suffix="/hr"), fmt_usd(5, approx=True)):
            assert "USD" not in out

    def test_returns_markdown_str(self):
        assert isinstance(fmt_usd(100), MarkdownStr)


class TestFmtEur:
    def test_basic_eur_code_style(self):
        assert fmt_eur(210_000_000, scale="M", precision=0, commas=False) == "EUR 210M"


class TestFmtCountLegacy:
    def test_accepts_named_marker(self):
        assert fmt_count(1024, suffix=" GPUs", approx=True) == "~1,024 GPUs"


class TestFmtRate:
    def test_formats_allowlisted_service_rates(self):
        assert fmt_rate(2500, "QPS") == "2,500 QPS"
        assert fmt_rate(1200, "tokens/s") == "1,200 tokens/s"
        assert fmt_rate(500_000, "tokens/s", scale="K", commas=False) == "500K tokens/s"
        assert (
            fmt_rate(
                45_200_000,
                "tokens/hour",
                scale="million",
                precision=1,
                commas=False,
            )
            == "45.2 million tokens/hour"
        )
        assert fmt_rate(2500, "QPS", commas=False) == "2500 QPS"
        assert fmt_rate(60, "FPS") == "60 FPS"
        assert fmt_rate(5000, "requests/s") == "5,000 requests/s"
        assert fmt_rate(1000, "samples/hour", commas=False) == "1000 samples/hour"

    def test_rejects_unknown_rate_unit(self):
        with pytest.raises(ValueError, match="fmt_rate unit must be"):
            fmt_rate(10, "GB/s")

    def test_rejects_negative_by_default(self):
        with pytest.raises(ValueError, match="non-negative rate"):
            fmt_rate(-1, "QPS")
        assert fmt_rate(-1, "QPS", allow_negative=True) == "-1 QPS"

    def test_returns_markdown_str(self):
        assert isinstance(fmt_rate(1, "QPS"), MarkdownStr)


class TestFmtFps:
    def test_formats_frame_rate(self):
        assert fmt_fps(30) == "30 FPS"
        assert fmt_fps(30.3) == "30 FPS"
        assert fmt_fps(29.5, precision=1) == "29.5 FPS"
        assert fmt_fps(120, commas=True) == "120 FPS"

    def test_rejects_negative_by_default(self):
        with pytest.raises(ValueError, match="non-negative rate"):
            fmt_fps(-1)
        assert fmt_fps(-1, allow_negative=True) == "-1 FPS"

    def test_returns_markdown_str(self):
        assert isinstance(fmt_fps(30), MarkdownStr)


class TestFmtTime:
    def test_symbol_style_accepts_quantities_and_plain_numbers(self):
        assert fmt_time(1500 * ureg.millisecond, ureg.second) == "1.5 s"
        assert fmt_time(1500 * ureg.millisecond, unit=ureg.second) == "1.5 s"
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
        assert fmt_time(1, "hour", precision=0, style="word", attributive=True) == "1-hour"
        assert fmt_time(24, "hour", precision=0, style="word", attributive=True) == "24-hour"
        assert fmt_time(15, "minute", precision=0, style="word", attributive=True) == "15-minute"

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
        # Default precision=None auto-resolves integer-like percentages.
        assert fmt_percent(0.85) == "85"
        assert fmt_percent(0.123, precision=1) == "12.3"

    def test_default_precision_auto_resolves_integer_like_percentages(self):
        assert fmt_percent(0.85, style="prose") == "85 percent"
        assert fmt_percent(0.90, style="symbol") == "90%"
        assert fmt_percent_range(0.85, 0.90, style="prose") == "85\u201390 percent"
        assert fmt_pp(7.0) == "7 percentage points"

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
        assert fmt_percent(-0.818, precision=1, style="symbol", allow_negative=True) == "-81.8%"
        assert (
            fmt_percent(8.089, precision=1, style="symbol", allow_negative=True, max_ratio=9)
            == "808.9%"
        )

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
    def test_owns_times_glyph(self):
        assert fmt_multiple(3.2) == "3.2×"
        assert fmt_multiple(10, precision=0) == "10×"

    def test_owns_display_markers(self):
        assert fmt_multiple(7.8, precision=1, approx=True) == "~7.8×"
        assert fmt_multiple(10, precision=0, lower_bound=True) == "> 10×"
        assert fmt_multiple(2, precision=0, upper_bound=True) == "< 2×"
        with pytest.raises(ValueError, match="only one display marker"):
            fmt_multiple(2, approx=True, upper_bound=True)

    def test_inherits_fmt_precision_guard(self):
        # An integer-like factor at precision=1 would render "2.0" — the
        # shared fmt() guard rejects that; explicit precision=1 still errors.
        with pytest.raises(ValueError, match="spurious trailing zeros"):
            fmt_multiple(2.0, precision=1)
        assert fmt_multiple(2.0, precision=0) == "2×"
        assert fmt_multiple(2) == "2×"
        assert fmt_multiple(2.0) == "2×"

    def test_rejects_negative_factor(self):
        with pytest.raises(ValueError, match="non-negative factor"):
            fmt_multiple(-3)

    def test_rejects_dimensioned_quantity(self):
        with pytest.raises(ValueError, match="dimensionless scalar"):
            fmt_multiple(3 * GB)

    def test_returns_markdown_str(self):
        assert isinstance(fmt_multiple(2.5), MarkdownStr)


class TestFmtSpecificHeat:
    def test_uses_textbook_unit_label(self):
        assert fmt_specific_heat(1005 * J / (kg * K), precision=0, commas=True) == "1,005 J/kg/K"

    def test_rejects_wrong_dimension(self):
        with pytest.raises(ValueError, match="energy/mass/temperature"):
            fmt_specific_heat(1 * J, precision=0)


class TestFmtTimeAutoPrecision:
    def test_auto_precision_trims_integer_like_display(self):
        assert fmt_time(1.0 * second, second, precision=None, commas=False) == "1 s"
        assert fmt_time(37.0 * second, second, precision=None, commas=False) == "37 s"

    def test_auto_precision_preserves_sub_unit_fraction(self):
        assert fmt_time(0.98 * second, second, precision=None, commas=False) == "0.98 s"


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

    def test_rejects_dimensioned_quantity(self):
        with pytest.raises(ValueError, match="dimensionless scalar"):
            fmt_ratio(5 * GB)

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
        assert fmt_count(1_000_000, scale="million") == "1 million"
        assert fmt_count(1_000_000, scale="Million") == "1 Million"
        assert (
            fmt_count(
                60_000_000,
                scale="million",
                label="parameter",
            )
            == "60 million parameters"
        )
        assert (
            fmt_count(
                70_000_000_000,
                scale="billion",
                precision=0,
                commas=False,
                label="parameter",
            )
            == "70 billion parameters"
        )

    def test_scale_word_attributive_modifier(self):
        assert (
            fmt_count(
                7_000_000_000,
                scale="billion",
                precision=0,
                commas=False,
                attributive=True,
            )
            == "7-billion"
        )
        with pytest.raises(ValueError, match="requires scale"):
            fmt_count(7, attributive=True)
        with pytest.raises(ValueError, match="only the scaled modifier"):
            fmt_count(
                7_000_000_000,
                scale="billion",
                label="parameter",
                attributive=True,
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
        assert fmt_count(1.5, label="GPU", precision=1, allow_fractional=True) == "1.5 GPUs"

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
        assert (
            fmt_qty_range(1 * ureg.GB, 2 * ureg.GB, unit=ureg.GB, precision=0, commas=False)
            == "1\u20132 GB"
        )

    def test_quantity_range_accepts_checked_display_label(self):
        out = fmt_qty_range(
            1 * ureg.kilojoule,
            2 * ureg.kilojoule,
            ureg.kilojoule,
            precision=0,
            commas=False,
            unit_label="kJ per hour",
        )
        assert out == "1\u20132 kJ per hour"

    def test_percent_range_uses_ratio_domain_and_one_suffix(self):
        assert (
            fmt_percent_range(
                0.846,
                0.88,
                precision=(1, 0),
                commas=False,
                style="prose",
            )
            == "84.6\u201388 percent"
        )
        assert (
            fmt_percent_range(0.90, 0.95, precision=0, commas=False, style="symbol")
            == "90\u201395%"
        )
        with pytest.raises(ValueError, match="0-1 ratio"):
            fmt_percent_range(84.6, 88, precision=1)

    def test_multiple_range_owns_times_glyph_and_is_checked(self):
        assert fmt_multiple_range(3, 7.8, precision=(0, 1), commas=False) == "3\u20137.8×"
        with pytest.raises(ValueError, match="non-negative"):
            fmt_multiple_range(-1, 2)
        with pytest.raises(ValueError, match="hi >= lo"):
            fmt_multiple_range(3, 2)

    def test_time_range_symbol_and_word_styles(self):
        assert fmt_time_range(5, 20, ureg.millisecond, precision=0, commas=False) == "5\u201320 ms"
        assert fmt_time_range(5, 20, ureg.millisecond, commas=False) == "5\u201320 ms"
        assert fmt_time_range(5, 20, unit=ureg.millisecond, commas=False) == "5\u201320 ms"
        assert (
            fmt_time_range(1, 2, ureg.second, precision=0, style="word", commas=False)
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
        assert (
            fmt_count_range(
                1_000_000,
                2_000_000,
                scale="million",
                label="parameter",
                commas=False,
            )
            == "1 million\u20132 million parameters"
        )

    def test_usd_range_supports_scale_and_denominator(self):
        assert fmt_usd_range(10_000, 30_000, scale="K", commas=False) == "\\$10K\u2013\\$30K"
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


class TestFmtSciQty:
    def test_scientific_quantity_appends_checked_unit(self):
        out = fmt_sci_qty(
            4.1e17 * ureg.flop,
            ureg.flop,
            precision=2,
            unit_label="FLOPs",
        )
        assert out == "4.10 × 10¹⁷ FLOPs"
        assert isinstance(out, MarkdownStr)
        assert (
            fmt_sci_qty(4.1e17 * ureg.flop, unit=ureg.flop, precision=2, unit_label="FLOPs")
            == "4.10 × 10¹⁷ FLOPs"
        )

    def test_scientific_quantity_converts_before_formatting(self):
        out = fmt_sci_qty(
            4.1e8 * ureg.GFLOPs,
            ureg.flop,
            precision=2,
            unit_label="FLOPs",
        )
        assert out == "4.10 × 10¹⁷ FLOPs"

    def test_scientific_quantity_requires_quantity(self):
        with pytest.raises(TypeError, match="requires a Pint Quantity"):
            fmt_sci_qty(4.1e17, ureg.flop)


class TestDomainFormatters:
    def test_fmt_power_scales_to_mw(self):
        from mlsysim.fmt import fmt_power
        from mlsysim.core.units import watt

        out = fmt_power(17.5e6 * watt, precision=1, commas=False)
        assert out == "17.5 MW"

    def test_fmt_power_scales_to_gw(self):
        from mlsysim.fmt import fmt_power
        from mlsysim.core.units import GW

        out = fmt_power(2.6 * GW, precision=1, commas=False)
        assert out == "2.6 GW"

    def test_fmt_energy_scales_to_kwh(self):
        from mlsysim.fmt import fmt_energy
        from mlsysim.core.units import kWh, watt, hour

        out = fmt_energy(700 * watt * hour, precision=0, commas=False)
        assert out == "700 Wh"

    def test_fmt_bandwidth_scales(self):
        from mlsysim.fmt import fmt_bandwidth
        from mlsysim.core.units import GB, TB, second

        out = fmt_bandwidth(3.35 * TB / second, precision=2, commas=False)
        assert out == "3.35 TB/s"

    def test_fmt_area_owns_square_unit_label(self):
        assert (
            fmt_area(814 * ureg.millimeter**2, unit=ureg.millimeter**2, commas=False)
            == "814 mm²"
        )
        with pytest.raises(ValueError, match="area unit"):
            fmt_area(814 * ureg.millimeter**2, unit=ureg.watt)

    def test_fmt_heat_flux_owns_power_density_label(self):
        assert (
            fmt_heat_flux(
                86.0 * ureg.watt / (ureg.centimeter**2),
                unit=ureg.watt / (ureg.centimeter**2),
                commas=False,
            )
            == "86 W/cm²"
        )
        with pytest.raises(ValueError, match="power/area"):
            fmt_heat_flux(86 * ureg.watt / (ureg.centimeter**2), unit=ureg.watt)

    def test_fmt_params_auto_precision(self):
        from mlsysim.core.units import Bparam, Mparam

        assert fmt_params(175 * Bparam) == "175B"
        assert fmt_params(1.5 * Bparam) == "1.5B"
        assert fmt_params(150 * Mparam) == "150M"
        assert fmt_params(150 * Mparam, scale="B") == "0.15B"

    def test_fmt_tokens_scales_checked_counts(self):
        from mlsysim.core.units import count

        assert fmt_tokens(300e9 * count, scale="B", commas=False) == "300B"
        assert fmt_tokens(1.5e12 * count, scale="T", commas=False) == "1.5T"

    def test_fmt_flop_rate_scales_to_tflop_per_second(self):
        from mlsysim.core.units import TFLOP, second

        out = fmt_flop_rate(989 * TFLOP / second, precision=0, commas=False)
        assert out == "989 TFLOP/s"

    def test_fmt_flop_rate_scales_to_pflop_per_second(self):
        from mlsysim.core.units import PFLOP, TFLOP, second

        out = fmt_flop_rate(1_979 * TFLOP / second, precision=3, commas=False)
        assert out == "1.979 PFLOP/s"
        assert fmt_flop_rate(2 * PFLOP / second, precision=0, commas=False) == "2 PFLOP/s"

    def test_fmt_flop_rate_requires_quantity(self):
        with pytest.raises(TypeError, match="requires a Pint Quantity"):
            fmt_flop_rate(989)

    def test_fmt_flops_scales_to_canonical_singular_units(self):
        from mlsysim.core.units import GFLOP, MFLOP, TFLOP

        assert fmt_flops(569 * MFLOP, precision=0, commas=False) == "569 MFLOP"
        assert fmt_flops(4.1 * GFLOP, precision=1, commas=False) == "4.1 GFLOP"
        assert fmt_flops(350 * TFLOP, precision=0, commas=False) == "350 TFLOP"
        assert (
            fmt_flops(4.1 * GFLOP, precision=1, commas=False, per="inference")
            == "4.1 GFLOP/inference"
        )

    def test_fmt_sci_flops_owns_plural_scientific_label(self):
        from mlsysim.core.units import GFLOP, flop

        assert fmt_sci_flops(4.1e17 * flop, precision=2) == "4.10 × 10¹⁷ FLOPs"
        assert fmt_sci_flops(4.1e8 * GFLOP, precision=2) == "4.10 × 10¹⁷ FLOPs"

    def test_fmt_flops_requires_quantity(self):
        with pytest.raises(TypeError, match="requires a Pint Quantity"):
            fmt_flops(4.1e9)
        with pytest.raises(TypeError, match="requires a Pint Quantity"):
            fmt_sci_flops(4.1e9)

    def test_fmt_arithmetic_intensity_owns_flop_per_byte_label(self):
        from mlsysim.core.units import GB, TFLOP, byte, flop

        assert fmt_arithmetic_intensity(5 * flop / byte, precision=0, commas=False) == "5 FLOP/byte"
        assert (
            fmt_arithmetic_intensity(1 * TFLOP / GB, precision=0, commas=False) == "1000 FLOP/byte"
        )

    def test_fmt_energy_density_helpers_own_readable_labels(self):
        from mlsysim.core.units import bit, byte, flop, pJ

        assert fmt_energy_per_byte(160 * pJ / byte, precision=0, commas=False) == "160 pJ/byte"
        assert fmt_energy_per_bit(5 * pJ / bit, precision=0, commas=False) == "5 pJ/bit"
        assert fmt_energy_per_flop(10 * pJ / flop, precision=0, commas=False) == "10 pJ/FLOP"
        assert fmt_energy_per_op(0.9 * pJ / flop, commas=False) == "0.9 pJ/op"
        assert fmt_energy_per_op(0.03 * pJ / flop, commas=False) == "0.03 pJ/op"
        assert fmt_energy_per_op(0.03 * pJ / flop, precision=2, commas=False) == "0.03 pJ/op"
        assert fmt_qty(160 * pJ / byte, pJ / byte, precision=0, commas=False) == "160 pJ/byte"

    def test_fmt_energy_density_helpers_require_quantities(self):
        with pytest.raises(TypeError, match="requires a Pint Quantity"):
            fmt_energy_per_byte(160)
        with pytest.raises(TypeError, match="requires a Pint Quantity"):
            fmt_energy_per_bit(5)
        with pytest.raises(TypeError, match="requires a Pint Quantity"):
            fmt_energy_per_flop(10)
        with pytest.raises(TypeError, match="requires a Pint Quantity"):
            fmt_energy_per_op(10)
        with pytest.raises(ValueError, match="energy/operation"):
            fmt_energy_per_op(10 * ureg.picojoule, unit=ureg.picojoule)
        with pytest.raises(ValueError, match="non-negative energy"):
            fmt_energy_per_op(-1 * ureg.picojoule / ureg.flop)

    def test_fmt_ops_rate_scales_integer_ops(self):
        from mlsysim.core.units import TOPS

        assert fmt_ops_rate(2 * TOPS, precision=0, commas=False) == "2 TOPS"
        assert fmt_ops_rate(0.002 * TOPS, precision=0, commas=False) == "2 GOPS"

    def test_fmt_rate_allows_iops(self):
        assert fmt_rate(150, "IOPS", precision=0, commas=False) == "150 IOPS"
        assert fmt_rate(6_000, "IOPS", scale="K", precision=0, commas=False) == "6K IOPS"

    def test_fmt_rate_allows_macs_per_cycle(self):
        assert fmt_rate(16_384, "MACs/cycle", precision=0, commas=True) == "16,384 MACs/cycle"

    def test_fmt_rate_allows_event_and_inference_rates(self):
        assert fmt_rate(20, "events/s", precision=0, commas=False) == "20 events/s"
        assert fmt_rate(300, "inferences/s", precision=0, commas=False) == "300 inferences/s"

    def test_fmt_rate_allows_book_service_rates(self):
        assert fmt_rate(12.5, "failures/day", precision=1, commas=False) == "12.5 failures/day"
        assert fmt_rate(90, "queries/hour", precision=0, commas=False) == "90 queries/hour"
        assert fmt_rate(5000, "queries/class", precision=0, commas=True) == "5,000 queries/class"
        assert fmt_rate(4, "deploys/week", precision=0, commas=False) == "4 deploys/week"
        assert (
            fmt_rate(40, "utterances/speaker", precision=0, commas=False) == "40 utterances/speaker"
        )
        assert fmt_rate(12, "cameras/store", precision=0, commas=False) == "12 cameras/store"
        assert fmt_rate(3, "boards/store", precision=0, commas=False) == "3 boards/store"
        assert fmt_rate(1000, "cases/day", precision=0, commas=True) == "1,000 cases/day"
        assert fmt_rate(120, "km/h", precision=0, commas=False) == "120 km/h"

    def test_fmt_named_markers_include_upper_bound(self):
        from mlsysim.core.units import GB, TOPS, TFLOP, milliwatt, second

        assert fmt(100, precision=0, lower_bound=True, suffix=" MB/s") == "> 100 MB/s"
        assert fmt(100, precision=0, upper_bound=True, suffix=" MB/s") == "< 100 MB/s"
        assert (
            fmt_flop_rate(
                100 * TFLOP / second,
                unit=TFLOP / second,
                precision=0,
                commas=False,
                lower_bound=True,
            )
            == "> 100 TFLOP/s"
        )
        assert (
            fmt_bandwidth(
                500 * GB / second,
                unit=GB / second,
                precision=0,
                commas=False,
                lower_bound=True,
            )
            == "> 500 GB/s"
        )
        assert (
            fmt_ops_rate(1 * TOPS, unit=TOPS, precision=0, commas=False, upper_bound=True)
            == "< 1 TOPS"
        )
        assert (
            fmt_power(10 * milliwatt, unit=milliwatt, precision=0, commas=False, upper_bound=True)
            == "< 10 mW"
        )
        with pytest.raises(ValueError, match="only one display marker"):
            fmt(100, precision=0, approx=True, upper_bound=True)

    def test_small_physical_label_helpers(self):
        from mlsysim.core.units import second

        assert fmt_decibel(ureg.Quantity(20, ureg.decibel), precision=0, commas=False) == "20 dB"
        assert fmt_illuminance(100 * ureg.lux, precision=0, commas=False) == "100 lux"
        assert fmt_temperature(ureg.Quantity(80, ureg.degC), precision=0, commas=False) == "80 °C"
        assert (
            fmt_temperature_rate(
                1 * ureg.delta_degC / second,
                precision=0,
                commas=False,
            )
            == "1 °C/s"
        )

    def test_fmt_compute_efficiency_renders_full_unit(self):
        from mlsysim.core.units import TFLOP, second, watt

        out = fmt_compute_efficiency(
            (989 * TFLOP / second) / (700 * watt),
            precision=2,
            commas=False,
        )
        assert out == "1.41 TFLOP/s/W"

    def test_fmt_memory_scales(self):
        from mlsysim.fmt import fmt_memory
        from mlsysim.core.units import GB, byte

        out = fmt_memory(14 * GB, precision=0, commas=False)
        assert out == "14 GB"
        assert fmt_memory(4 * byte, unit=byte, precision=0, commas=False) == "4 bytes"

    def test_fmt_memory_capacity_preserves_binary_magnitude_with_vendor_label(self):
        from mlsysim.core.units import GB, GiB, KiB, MiB
        from mlsysim.fmt import fmt_memory

        assert fmt_memory_capacity(80 * GiB, unit=GiB, precision=0, commas=False) == "80 GB"
        assert fmt_memory_capacity(33 * MiB, unit=MiB, precision=0, commas=False) == "33 MB"
        assert fmt_memory_capacity(256 * KiB, unit=KiB, precision=0, commas=False) == "256 KB"
        assert fmt_memory(80 * GiB, unit=GB, precision=1, commas=False) == "85.9 GB"

    def test_fmt_carbon_intensity_defaults_to_grams_per_kwh(self):
        from mlsysim.core.units import gram, kilogram, kWh

        assert fmt_carbon_intensity(429 * gram / kWh, precision=0, commas=False) == "429 g/kWh"
        assert (
            fmt_carbon_intensity(0.429 * kilogram / kWh, precision=0, commas=False) == "429 g/kWh"
        )

    def test_fmt_emissions_supports_display_markers(self):
        from mlsysim.core.units import kilogram, metric_ton

        assert (
            fmt_emissions(
                85_000 * kilogram,
                unit=metric_ton,
                precision=0,
                commas=False,
                approx=True,
            )
            == "~85 t"
        )

    def test_fmt_water_helpers_use_book_liter_labels(self):
        from mlsysim.core.units import L, day, hour, kWh

        assert fmt_water(18_000_000 * L, precision=0) == "18,000,000 L"
        assert fmt_water_rate(12_000 * L / hour, unit=L / hour, precision=0) == "12,000 L/h"
        assert fmt_water_rate(12_000 * L / hour, unit=L / day, precision=0) == "288,000 L/day"
        assert fmt_water_intensity(1.8 * L / kWh, precision=1) == "1.8 L/kWh"

    def test_fmt_water_helpers_reject_wrong_dimensions(self):
        from mlsysim.core.units import L, watt

        with pytest.raises(ValueError, match="volume unit"):
            fmt_water(1 * L, unit=watt)
        with pytest.raises(ValueError, match="volume/time"):
            fmt_water_rate(1 * L, unit=L)
        with pytest.raises(ValueError, match="volume/energy"):
            fmt_water_intensity(1 * L, unit=L)

    def test_usd_range_repeat_symbol_false_with_scale(self):
        assert (
            fmt_usd_range(10_000, 30_000, scale="K", commas=False, repeat_symbol=False)
            == "\\$10K\u201330K"
        )


class TestAuditRepairs:
    """Focused regressions for the 2026-06 verified audit findings.

    Five findings: fmt_int silent zero + banker's rounding, the auto-scale
    1000-mantissa boundary, fmt_unit/fmt_qty label divergence, fmt_pp
    singular grammar on negatives, and fmt_usd negative sign placement.
    """

    # --- Finding 1: fmt_int silent zero + round-half-away-from-zero --------

    def test_fmt_int_rejects_nonzero_values_that_round_to_zero(self):
        with pytest.raises(ValueError, match="rounds to '0'"):
            fmt_int(0.4)
        with pytest.raises(ValueError, match="rounds to '0'"):
            fmt_int(-0.4)
        with pytest.raises(ValueError, match="rounds to '0'"):
            fmt_int(0.49999)

    def test_fmt_int_true_zero_still_renders(self):
        assert fmt_int(0) == "0"
        assert fmt_int(0.0) == "0"

    def test_fmt_int_rounds_half_away_from_zero(self):
        # Python round() is banker's (2.5 -> 2, 3.5 -> 4); fmt_int must round
        # adjacent halves consistently away from zero.
        assert fmt_int(2.5, commas=False) == "3"
        assert fmt_int(3.5, commas=False) == "4"
        assert fmt_int(-2.5, commas=False) == "-3"
        assert fmt_int(-3.5, commas=False) == "-4"
        assert fmt_int(0.5, commas=False) == "1"
        assert fmt_int(-0.5, commas=False) == "-1"

    def test_fmt_int_still_absorbs_float_noise(self):
        assert fmt_int(7.9999999999, commas=False) == "8"
        assert fmt_int(362507.545, commas=True) == "362,508"

    def test_fmt_int_keeps_commas_and_markers(self):
        assert fmt_int(1500.5, commas=True) == "1,501"
        assert fmt_int(80.2, commas=False, approx=True, suffix=" GB") == "~80 GB"

    # --- Finding 2: auto-scale boundary promotion (M -> B -> T) ------------

    def test_auto_scale_promotes_rounded_mantissa_at_boundary(self):
        # 999,999,999 picks "M" from the raw value but ROUNDS to a 1000.0
        # mantissa at display precision; it must promote to "1B".
        assert fmt_params(999_999_999) == "1B"
        assert fmt_params(999_999_999_999) == "1T"

    def test_fmt_tokens_auto_scale_boundary(self):
        from mlsysim.core.units import count

        assert fmt_tokens(999_999_999 * count, scale="auto") == "1B"

    def test_auto_scale_exact_boundaries_unchanged(self):
        assert fmt_params(1_000_000_000) == "1B"
        assert fmt_params(1_000_000_000_000) == "1T"

    def test_auto_scale_below_boundary_does_not_promote(self):
        assert fmt_params(999_400_000_000) == "999.4B"

    def test_auto_scale_top_scale_keeps_value_with_commas(self):
        # No scale above T: keep the T mantissa and group digits instead.
        assert fmt_params(2_000_000_000_000_000) == "2,000T"

    # --- Finding 3: fmt_unit routed through shared normalization -----------

    def test_fmt_unit_normalizes_all_flops_rate_labels(self):
        from mlsysim.core.units import Q_

        assert fmt_unit(Q_(1, "EFLOPs/second")) == "EFLOP/s"
        assert fmt_unit(Q_(1, "KFLOPs/second")) == "KFLOP/s"
        assert fmt_unit(Q_(1, "TFLOPs/second")) == "TFLOP/s"

    def test_fmt_unit_normalizes_bare_work_units(self):
        from mlsysim.core.units import Q_

        assert fmt_unit(Q_(1, "TFLOPs")) == "TFLOP"
        assert fmt_unit(Q_(1, "EFLOPs")) == "EFLOP"

    def test_fmt_unit_matches_fmt_qty_suffix_exactly(self):
        from mlsysim.core.units import Q_

        q = Q_(2, "EFLOPs/second")
        qty_suffix = str(fmt_qty(q, q.units, precision=0, commas=False)).split(" ", 1)[1]
        assert str(fmt_unit(q)) == qty_suffix

    def test_fmt_unit_default_and_plain_units(self):
        from mlsysim.core.units import Q_

        assert fmt_unit(80) == "-"
        assert fmt_unit(Q_(1, "GB")) == "GB"
        assert isinstance(fmt_unit(Q_(1, "GB")), MarkdownStr)

    # --- Finding 4: fmt_pp singular grammar on negatives -------------------

    def test_fmt_pp_negative_one_is_singular(self):
        assert fmt_pp(-1) == "-1 percentage point"
        assert fmt_pp(-1.0, precision=0) == "-1 percentage point"

    def test_fmt_pp_singular_plural_agreement_unchanged(self):
        assert fmt_pp(1.0, precision=0) == "1 percentage point"
        assert fmt_pp(-2.0, precision=0) == "-2 percentage points"
        assert fmt_pp(7.0) == "7 percentage points"
        assert fmt_pp(0.9, precision=1) == "0.9 percentage points"

    # --- Finding 5: fmt_usd negative sign-first rendering -------------------

    def test_fmt_usd_negative_renders_sign_first(self):
        assert fmt_usd(-500) == "-\\$500"
        assert fmt_usd(-12345.6) == "-\\$12,346"

    def test_fmt_usd_negative_with_approx(self):
        assert fmt_usd(-500, approx=True) == "~-\\$500"

    def test_fmt_usd_negative_with_scale_and_per(self):
        assert fmt_usd(-4_600_000, precision=1, scale="M") == "-\\$4.6M"
        assert fmt_usd(-0.09, precision=2, commas=False, per="GB") == "-\\$0.09/GB"
        assert fmt_usd(-12_000, commas=False, scale="K", per="year") == "-\\$12K/year"

    def test_fmt_usd_positive_unchanged(self):
        assert fmt_usd(15000) == "\\$15,000"
        assert fmt_usd(8000, approx=True, marker="*") == "~\\$8,000*"


class TestFmtLength:
    def test_fmt_length_meters_small_no_commas(self):
        from mlsysim.core.units import meter

        assert fmt_length(2.8 * meter, unit=meter, precision=1) == "2.8 m"

    def test_fmt_length_kilometers_commas(self):
        from mlsysim.core.units import kilometer

        assert fmt_length(3600 * kilometer, unit=kilometer, precision=0) == "3,600 km"

    def test_fmt_rate_kmh(self):
        assert fmt_rate(120, "km/h", precision=0, commas=False) == "120 km/h"
