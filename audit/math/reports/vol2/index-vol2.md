# Math Audit Report: `book/quarto/index-vol2.qmd`

## Checked scope

Audited `book/quarto/index-vol2.qmd` for mathematical statements, equations, numeric calculations, unit conversions, complexity claims, and prose-equation consistency. The file contains no equations, formal complexity notation, unit conversions, or worked numeric derivations. Quantitative/prose claims checked include:

- Line 24: "tens of thousands of accelerators," "megawatts of heat," and a "cluster of ten thousand devices" with a hardware failure "roughly every two hours."
- Line 26: "thousands of accelerators" and AllReduce completing in "milliseconds, not seconds."
- Line 28: a "ninety-day training run" and "day forty-seven."
- Line 32: "a *thousand*" ML systems.
- Line 47: "models too large for single GPUs" and services "spanning continents."
- Lines 63-68: "four parts" followed by four listed parts.
- Line 54: publisher year "2027."
- Line 87: "2026 Goal" to help "100,000 students."

## Findings

No mathematical correctness issues found.

The only count claim with a list is internally consistent: line 63 introduces four parts, and lines 65-68 list exactly four parts. The failure-rate prose on line 24 is also internally plausible: a cluster-level failure every two hours across 10,000 devices implies an average per-device failure interval of about 20,000 hours, or about 2.3 years, under an independent constant-rate approximation. The remaining numeric statements are scale examples, dates, goals, or qualitative latency comparisons without accompanying calculations to verify.
