# FMT audit ledger

This ledger records formatter migration validation. Keep it current while
working. The goal is to make each touched LEGO cell and final inline prose audit
reviewable without rediscovery.

## Ledger rules

- Add an entry for every LEGO cell touched on this branch.
- Validate every exported `_str`, `_math`, `_eq`, and `_frac` from that cell.
- Record whether output is byte-identical or intentionally changed.
- Read the rendered sentence after inline substitution.
- Record exceptions instead of leaving them implicit.

## Entry template

```text
Chapter:
File:
Cell anchor / line:
Change type:
Exports checked:
Before values:
After values:
Equivalence:
Rendered prose checked:
Spacing/unit/glyph result:
Manual prose read:
HTML evidence:
PDF/tex evidence:
Notes / exceptions:
Status:
```

## Current session notes

- Plan of record added before continuing implementation.
- Existing branch history already includes multiple WS4/A1/A2/render
  verifications summarized in `NIGHT_RESUME.md`.
- Structured formatter API batch added:
  - `fmt_usd(scale=..., per=...)`
  - `fmt_count(label=..., plural_label=...)`
  - `fmt_rate(...)`
  - `fmt_time(..., style="symbol"|"word")`
  - `fmt_qty_range`, `fmt_time_range`, `fmt_count_range`, `fmt_usd_range`
- No QMD LEGO cells were touched in this API batch, so there are no per-cell
  rendered prose entries yet.
- API batch verification:
  - `python3 -m py_compile mlsysim/mlsysim/fmt.py book/tools/audit/fmt/audit_fmt_usage.py` PASS
  - `git diff --check` PASS
  - `PYTHONPATH=mlsysim python3 -m pytest mlsysim/tests/test_fmt.py book/tests/test_codemod_fmt.py book/tests/test_fmt_prose_contract.py book/tests/test_audit_prose_semantics.py book/tests/test_visible_text.py -q -o addopts=''` PASS, 157 tests
  - `PYTHONPATH=mlsysim python3 book/tools/audit/fmt/fmt_prose_contract.py --root book/quarto/contents` PASS, 0 violations
  - `PYTHONPATH=mlsysim python3 book/tools/audit/fmt/codemod_fmt.py queue --root book/quarto/contents` PASS, `by kind: {}`
  - `PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose_semantics.py --root book/quarto/contents` PASS, 0 findings

## 2026-05-31 — Currency denominator relocation

Change type: byte-identical formatter relocation. Replaced 91
`fmt_usd(..., suffix="/...")` chapter call sites with structured
`fmt_usd(..., per="...")`. This keeps the rendered denominator text unchanged
while moving denominator validation into `fmt_usd`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/data_engineering/data_engineering.qmd` | 26 | 203 | 99 | identical values + prose |
| `vol1/hw_acceleration/hw_acceleration.qmd` | 5 | 255 | 140 | identical values + prose |
| `vol1/ml_ops/ml_ops.qmd` | 8 | 132 | 75 | identical values + prose |
| `vol1/ml_systems/ml_systems.qmd` | 15 | 296 | 143 | identical values + prose |
| `vol1/ml_workflow/ml_workflow.qmd` | 8 | 77 | 36 | identical values + prose |
| `vol1/responsible_engr/responsible_engr.qmd` | 3 | 168 | 79 | identical values + prose |
| `vol1/training/training.qmd` | 6 | 453 | 214 | identical values + prose |
| `vol2/distributed_training/distributed_training.qmd` | 1 | 261 | 142 | identical values + prose |
| `vol2/ops_scale/ops_scale.qmd` | 16 | 195 | 95 | identical values + prose |
| `vol2/sustainable_ai/sustainable_ai.qmd` | 3 | 197 | 120 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_currency_per_assess`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` confirms the `rate_denominator` `suffix=` bucket is now
  gone; only the pre-existing single `extra_suffix=` denominator remains.
- Render evidence: deferred to the chapter render sweep; this batch is a pure
  byte-identical relocation with executed value/prose proof.

## 2026-05-31 — Currency scale/range relocation

Change type: structured currency migration. Replaced the remaining 78
chapter-level `fmt_usd(..., suffix=...)` call sites with `scale=`, `per=`,
`marker=`, or `fmt_usd_range(...)`. QMD now has zero `fmt_usd(..., suffix=...)`
calls.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/data_engineering/data_engineering.qmd` | 13 | 203 | 99 | identical values + prose |
| `vol1/data_selection/data_selection.qmd` | 5 | 250 | 128 | identical values + prose |
| `vol1/hw_acceleration/hw_acceleration.qmd` | 2 | 255 | 140 | intentional range dash change |
| `vol1/ml_ops/ml_ops.qmd` | 6 | 132 | 75 | identical values + prose |
| `vol1/ml_systems/ml_systems.qmd` | 2 | 296 | 143 | identical values + prose |
| `vol1/responsible_engr/responsible_engr.qmd` | 19 | 168 | 79 | identical values + prose |
| `vol1/training/training.qmd` | 4 | 453 | 214 | identical values + prose |
| `vol2/inference/inference.qmd` | 1 | 208 | 119 | identical values + prose |
| `vol2/network_fabrics/network_fabrics.qmd` | 3 | 106 | 62 | identical values + prose |
| `vol2/ops_scale/ops_scale.qmd` | 17 | 195 | 95 | identical values + prose |
| `vol2/responsible_ai/responsible_ai.qmd` | 3 | 51 | 25 | identical values + prose |
| `vol2/sustainable_ai/sustainable_ai.qmd` | 3 | 197 | 120 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_currency_scale_marker_assess`.
- Every scale relocation was byte-identical. The only value/prose diff was
  `AcceleratorEconomics.h100_price_str`: `~\$25,000-30,000` became
  `~\$25,000–30,000`, intentionally moving the price range through
  `fmt_usd_range(..., repeat_symbol=False)` so the range dash is formatter-owned.
- The TPU footnote marker moved from legacy `suffix="*"` to checked
  `marker="*"` with byte-identical output.
- `audit_fmt_usage.py` confirms `fmt_usd` has zero remaining `suffix=` call
  sites and the `scale_glyph` `suffix=` bucket is gone. The only scale suffixes
  left are 8 non-currency `fmt(..., suffix="million")`-style sites for later
  count/quantity review.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 161 tests;
  `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue` PASS,
  `by kind: {}`; `audit_prose_semantics.py` PASS, 0 findings across 81 files.
- Render evidence: deferred to the chapter render sweep; this batch has executed
  value/prose proof plus one documented intentional range-dash normalization.

## 2026-05-31 — QPS rate relocation

Change type: byte-identical formatter relocation. Replaced 26
`fmt(..., suffix=" QPS")` / `fmt_int(..., suffix=" QPS")` chapter call sites
with `fmt_rate(..., "QPS")`. `fmt_rate` now defaults to `commas=True`, matching
the old `fmt`/`fmt_int` default; compact table sites keep `commas=False`
explicitly.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/benchmarking/benchmarking.qmd` | 3 | 240 | 102 | identical values + prose |
| `vol1/ml_ops/ml_ops.qmd` | 1 | 132 | 75 | identical values + prose |
| `vol1/model_serving/model_serving.qmd` | 10 | 365 | 172 | identical values + prose |
| `vol2/fleet_orchestration/fleet_orchestration.qmd` | 5 | 160 | 60 | identical values + prose |
| `vol2/inference/inference.qmd` | 7 | 208 | 119 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_qps_rate_assess`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` now reports 26 `fmt_rate` calls and the `count_label`
  suffix bucket dropped from 144 to 118.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 161 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Benchmarking millisecond time suffixes

Change type: byte-identical formatter relocation. Replaced 25
`fmt(..., suffix=" ms")` sites in `vol1/benchmarking/benchmarking.qmd` with
`fmt_time(..., "ms")`. Explicit `commas=False` was preserved for the migration;
redundant comma overrides are deferred to the formatter-default cleanup pass.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/benchmarking/benchmarking.qmd` | 25 | 240 | 102 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_time_benchmarking_ms`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose`.
- `audit_fmt_usage.py` now reports 25 `fmt_time` calls and the `time_unit`
  suffix bucket dropped from 648 to 623.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 161 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Benchmarking remaining time suffixes

Change type: byte-identical formatter relocation. Replaced the 7 remaining
time-label suffix sites in `vol1/benchmarking/benchmarking.qmd` with
`fmt_time(...)`: compact seconds use symbol style, and prose labels
(`hours`, `seconds`, `minutes`) use `style="word"` so singular/plural agreement
is checked in the formatter. Explicit `commas=False` was preserved for the
migration; redundant comma overrides are deferred to the formatter-default
cleanup pass.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/benchmarking/benchmarking.qmd` | 7 | 240 | 102 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_time_benchmarking_remaining`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose`.
- `audit_fmt_usage.py` now reports 32 `fmt_time` calls and the `time_unit`
  suffix bucket dropped from 623 to 616. `benchmarking.qmd` has no remaining
  `time_unit` suffix sites.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 161 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — ML systems time suffixes

Change type: byte-identical formatter relocation plus formatter hardening.
Replaced all 24 `time_unit` suffix sites in
`vol1/ml_systems/ml_systems.qmd` with `fmt_time(...)`. Symbol units (`ms`,
`s`, `h`) use compact style; prose units (`months`, `days`, `years`, `hours`)
use `style="word"` so pluralization is owned by the formatter. The negative
cloud-latency headroom site uses `allow_negative=True`, and the one grouped
millisecond value keeps an explicit `commas=True` override. `fmt_time` now also
recognizes Pint's canonical `year` alias (`a`) for word-style formatting.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/ml_systems/ml_systems.qmd` | 24 | 296 | 143 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_time_ml_systems`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose`.
- `audit_fmt_usage.py` now reports 56 `fmt_time` calls and the `time_unit`
  suffix bucket dropped from 616 to 592. `ml_systems.qmd` has no remaining
  `time_unit` suffix sites.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 161 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Introduction time suffixes

Change type: byte-identical formatter relocation. Replaced all 15 `time_unit`
suffix sites in `vol1/introduction/introduction.qmd` with `fmt_time(...)`.
Old `fmt_int(..., suffix=" days"/" months")` sites now round explicitly before
calling `fmt_time`, preserving the rendered integer wording while keeping the
precision guard intact. Millisecond sites use compact symbol style; prose
`days` and `months` labels use `style="word"`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/introduction/introduction.qmd` | 15 | 91 | 45 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_time_introduction`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose`.
- `audit_fmt_usage.py` now reports 71 `fmt_time` calls and the `time_unit`
  suffix bucket dropped from 592 to 577. `introduction.qmd` has no remaining
  `time_unit` suffix sites.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 161 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Hardware acceleration millisecond suffixes

Change type: byte-identical formatter relocation. Replaced all 6 `time_unit`
suffix sites in `vol1/hw_acceleration/hw_acceleration.qmd` with
`fmt_time(..., "ms")`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/hw_acceleration/hw_acceleration.qmd` | 6 | 255 | 140 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_time_hw_acceleration`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose`.
- `audit_fmt_usage.py` now reports 77 `fmt_time` calls and the `time_unit`
  suffix bucket dropped from 577 to 571. `hw_acceleration.qmd` has no remaining
  `time_unit` suffix sites.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 161 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Structured quantity denominator cleanup

Change type: byte-identical formatter API cleanup. Replaced the one remaining
QMD `fmt_qty(..., extra_suffix="/inference")` site in
`vol1/ml_systems/ml_systems.qmd` with structured `per="inference"`. This
removes the `extra_suffix=` bucket from chapter formatter usage.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/ml_systems/ml_systems.qmd` | 1 | 296 | 143 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_qty_per_ml_systems`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose`.
- `audit_fmt_usage.py` now reports no `extra_suffix=` calls in QMD.

## 2026-05-31 — Remaining direct count labels

Change type: byte-identical formatter relocation. Replaced 40 hard-coded direct
count noun suffixes (`tokens`, `nodes`, `layers`, `queries`, `images`) with
`fmt_count(..., label=...)`. For old `fmt_int` query count sites whose values
were fractional before rounding, the rounding is now explicit via `round(...)`
before `fmt_count`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/benchmarking/benchmarking.qmd` | 1 | 240 | 102 | identical values + prose |
| `vol1/frameworks/frameworks.qmd` | 1 | 131 | 70 | identical values + prose |
| `vol1/ml_systems/ml_systems.qmd` | 4 | 296 | 143 | identical values + prose |
| `vol1/model_serving/model_serving.qmd` | 4 | 365 | 172 | identical values + prose |
| `vol1/responsible_engr/responsible_engr.qmd` | 2 | 168 | 79 | identical values + prose |
| `vol1/training/training.qmd` | 6 | 453 | 214 | identical values + prose |
| `vol2/collective_communication/collective_communication.qmd` | 1 | 104 | 60 | identical values + prose |
| `vol2/compute_infrastructure/compute_infrastructure.qmd` | 1 | 320 | 183 | identical values + prose |
| `vol2/data_storage/data_storage.qmd` | 5 | 180 | 99 | identical values + prose |
| `vol2/distributed_training/distributed_training.qmd` | 3 | 261 | 142 | identical values + prose |
| `vol2/fault_tolerance/fault_tolerance.qmd` | 1 | 129 | 70 | identical values + prose |
| `vol2/inference/inference.qmd` | 5 | 208 | 119 | identical values + prose |
| `vol2/network_fabrics/network_fabrics.qmd` | 2 | 106 | 62 | identical values + prose |
| `vol2/security_privacy/security_privacy.qmd` | 3 | 69 | 38 | identical values + prose |
| `vol2/sustainable_ai/sustainable_ai.qmd` | 1 | 197 | 120 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_remaining_count_assess`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` reports `fmt_count` calls increased to 203. The only
  remaining `count_label` suffix is the documented `GPU-days` compound.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 161 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — GPU count label relocation

Change type: byte-identical formatter relocation. Replaced 77 hard-coded
`suffix=" GPUs"` sites with `fmt_count(..., label="GPU")`. The singular
`suffix=" GPU"` site in `model_compression.qmd` was intentionally left alone
because it renders the compound `GPU-days`, not a standalone GPU count.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/benchmarking/benchmarking.qmd` | 2 | 240 | 102 | identical values + prose |
| `vol1/introduction/introduction.qmd` | 1 | 91 | 45 | identical values + prose |
| `vol1/ml_systems/ml_systems.qmd` | 1 | 296 | 143 | identical values + prose |
| `vol1/responsible_engr/responsible_engr.qmd` | 1 | 168 | 79 | identical values + prose |
| `vol1/training/training.qmd` | 8 | 453 | 214 | identical values + prose |
| `vol2/backmatter/appendix_assumptions.qmd` | 3 | 275 | 122 | identical values + prose |
| `vol2/backmatter/appendix_c3.qmd` | 1 | 46 | 20 | identical values + prose |
| `vol2/backmatter/appendix_communication.qmd` | 1 | 56 | 33 | identical values + prose |
| `vol2/collective_communication/collective_communication.qmd` | 3 | 104 | 60 | identical values + prose |
| `vol2/compute_infrastructure/compute_infrastructure.qmd` | 1 | 320 | 183 | identical values + prose |
| `vol2/data_storage/data_storage.qmd` | 1 | 180 | 99 | identical values + prose |
| `vol2/distributed_training/distributed_training.qmd` | 14 | 261 | 142 | identical values + prose |
| `vol2/fault_tolerance/fault_tolerance.qmd` | 2 | 129 | 70 | identical values + prose |
| `vol2/fleet_orchestration/fleet_orchestration.qmd` | 18 | 160 | 60 | identical values + prose |
| `vol2/inference/inference.qmd` | 1 | 208 | 119 | identical values + prose |
| `vol2/introduction/introduction.qmd` | 1 | 37 | 20 | identical values + prose |
| `vol2/network_fabrics/network_fabrics.qmd` | 11 | 106 | 62 | identical values + prose |
| `vol2/ops_scale/ops_scale.qmd` | 4 | 195 | 95 | identical values + prose |
| `vol2/performance_engineering/performance_engineering.qmd` | 1 | 109 | 58 | identical values + prose |
| `vol2/sustainable_ai/sustainable_ai.qmd` | 2 | 197 | 120 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_gpu_count_assess`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` reports `fmt_count` calls increased to 163 and the
  `count_label` suffix bucket dropped from 118 to 41.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 161 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.
