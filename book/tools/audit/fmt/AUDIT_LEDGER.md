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

## 2026-05-31 — Time denominator suffix relocation

Change type: byte-identical formatter relocation. Replaced four
denominator-style time suffixes (`μs/op`, `s/hr`, `ms/step`, `hours/day`) with
`fmt_time(..., per=...)`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/data_engineering/data_engineering.qmd` | 1 | 203 | 99 | identical values + prose |
| `vol1/frameworks/frameworks.qmd` | 1 | 131 | 70 | identical values + prose |
| `vol1/introduction/introduction.qmd` | 1 | 91 | 45 | identical values + prose |
| `vol1/model_serving/model_serving.qmd` | 1 | 365 | 172 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_time_den_*`.
- `audit_fmt_usage.py` reports `fmt_time` calls increased to 657 and
  `time_compound` suffixes dropped to 4.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/tools/audit/fmt/audit_fmt_usage.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 176 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Remaining suffix bucket split and epoch label

Change type: audit improvement plus byte-identical formatter relocation.
`audit_fmt_usage.py` now separates remaining suffixes into physical units,
resource-time labels, unit rates/denominators, compound scale phrases, operation
counts, and time compounds. The lone remaining plain count noun found in that
split, `epochs`, moved to `fmt_count(..., label="epoch",
plural_label="epochs")` in `vol1/data_engineering/data_engineering.qmd`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/data_engineering/data_engineering.qmd` | 1 | 203 | 99 | identical values + prose |

Validation details:

- `audit_fmt_usage.py` reports remaining suffix buckets as: `physical_unit`
  1,126; `resource_time` 19; `unit_rate_or_denominator` 16;
  `compound_scale` 14; `op_count` 12; `time_compound` 8.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  `data_engineering`.
- Verification: `python3 -m py_compile book/tools/audit/fmt/audit_fmt_usage.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS;
  `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue` PASS,
  `by kind: {}`; `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — Word-form micro/millisecond suffix cleanup

Change type: byte-identical formatter relocation. Added word-form
`microsecond(s)` and `millisecond(s)` support to the time codemod and migrated
the three remaining exact word-form time suffixes to `fmt_time(...,
style="word")`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/model_compression/model_compression.qmd` | 2 | 162 | 74 | identical values + prose |
| `vol1/responsible_engr/responsible_engr.qmd` | 1 | 168 | 79 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_word_time_*`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  both touched chapters.
- `audit_fmt_usage.py` reports `fmt_time` calls increased to 653 and direct
  suffix calls dropped to 1,196.
- Verification: `python3 -m py_compile book/tools/audit/fmt/codemod_fmt.py book/tools/audit/fmt/audit_fmt_usage.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 176 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Residual plain count-label suffix relocation

Change type: byte-identical formatter relocation. Replaced 17 remaining plain
count-noun suffixes (`errors`, `steps`, `photos`, `requests`, `servers`,
`workers`, `stages`, `V100s`, `V100 GPUs`, etc.) with structured
`fmt_count(..., label=...)`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/benchmarking/benchmarking.qmd` | 4 | 240 | 102 | identical values + prose |
| `vol1/data_engineering/data_engineering.qmd` | 1 | 203 | 99 | identical values + prose |
| `vol1/ml_systems/ml_systems.qmd` | 1 | 296 | 143 | identical values + prose |
| `vol1/model_serving/model_serving.qmd` | 3 | 365 | 172 | identical values + prose |
| `vol1/training/training.qmd` | 4 | 453 | 214 | identical values + prose |
| `vol2/conclusion/conclusion.qmd` | 1 | 19 | 18 | identical values + prose |
| `vol2/data_storage/data_storage.qmd` | 1 | 180 | 99 | identical values + prose |
| `vol2/distributed_training/distributed_training.qmd` | 1 | 261 | 142 | identical values + prose |
| `vol2/network_fabrics/network_fabrics.qmd` | 1 | 106 | 62 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_count_labels_*`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` reports `fmt_count` calls increased to 230 and direct
  suffix calls dropped to 1,199, all under the remaining physical/compound
  suffix bucket.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/fmt_semantic_suffix.py book/tools/audit/fmt/audit_fmt_usage.py book/cli/checks/lego_dead_code.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS; focused pytest suite
  PASS, 176 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Exact scale-word suffix relocation

Change type: byte-identical formatter relocation. Added
`fmt_count(..., scale_style="word")` and migrated the exact `suffix=" million"`
and `suffix=" billion"` sites to a typed count formatter. The semantic suffix
gate now flags future exact scale-word suffixes on generic `fmt`/`fmt_int`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/data_selection/data_selection.qmd` | 1 | 250 | 128 | identical values + prose |
| `vol1/introduction/introduction.qmd` | 2 | 91 | 45 | identical values + prose |
| `vol1/ml_ops/ml_ops.qmd` | 1 | 132 | 75 | identical values + prose |
| `vol1/training/training.qmd` | 1 | 453 | 214 | identical values + prose |
| `vol2/network_fabrics/network_fabrics.qmd` | 2 | 106 | 62 | identical values + prose |
| `vol2/sustainable_ai/sustainable_ai.qmd` | 1 | 197 | 120 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_scale_word_*`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` reports no remaining `scale_word` suffix bucket;
  `fmt_count` calls increased to 213 and physical-unit suffixes remain 1,216.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/fmt_semantic_suffix.py book/tools/audit/fmt/audit_fmt_usage.py book/cli/checks/lego_dead_code.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 176 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.
- Open API note: the user flagged `scale="B", scale_style="word"` as unclear.
  Before final API lock, discuss a clearer spelling such as `scale="billion"`
  for word-scale output while keeping `scale="B"` for compact glyph output.

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

## 2026-05-31 — ML ops pp suffix gap

Change type: byte-identical formatter relocation plus checker hardening.
Replaced the 3 remaining `suffix=" pp"` sites in
`vol1/ml_ops/ml_ops.qmd` with `fmt_pp(..., style="symbol")`. The semantic
suffix checker now treats `pp` as a percentage-point suffix, and the audit
inventory classifies percentage-point suffixes separately from percent-share
suffixes.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/ml_ops/ml_ops.qmd` | 3 | 132 | 75 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_pp_ml_ops`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose`.
- `audit_fmt_usage.py` now reports `fmt_pp` calls increased to 21 and physical
  suffix calls dropped by 3.
- `book/tests/test_fmt_semantic_suffix.py` now locks `suffix=" pp"` as a
  `pp_in_suffix` violation.

## 2026-05-31 — ML ops time suffixes

Change type: byte-identical formatter relocation. Replaced all 19 `time_unit`
suffix sites in `vol1/ml_ops/ml_ops.qmd` with `fmt_time(...)`. Symbol units
(`ms`, `s`) use compact style; prose labels (`hours`, `weeks`, `year`,
`seconds`, `minutes`) use `style="word"`. Old `fmt_int` duration sites retain
intentional integer display through `precision=0`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/ml_ops/ml_ops.qmd` | 19 | 132 | 75 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_time_ml_ops`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose`.
- `audit_fmt_usage.py` now reports 96 `fmt_time` calls and the `time_unit`
  suffix bucket dropped from 571 to 552. `ml_ops.qmd` has no remaining
  `time_unit` suffix sites.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py book/cli/checks/fmt_semantic_suffix.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 167 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Data engineering time suffixes

Change type: byte-identical formatter relocation. Replaced all 21 `time_unit`
suffix sites in `vol1/data_engineering/data_engineering.qmd` with
`fmt_time(...)`. Symbol units (`h`, `ms`, `s`) use compact style; prose labels
(`seconds`, `days`, `months`, `minutes`, `hours`) use `style="word"`. Old
`fmt_int` duration sites now round explicitly before `fmt_time` where needed.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/data_engineering/data_engineering.qmd` | 21 | 203 | 99 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_time_data_engineering`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose`.
- `audit_fmt_usage.py` now reports 116 `fmt_time` calls and the `time_unit`
  suffix bucket dropped from 552 to 532. `data_engineering.qmd` has no
  remaining `time_unit` suffix sites.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py book/cli/checks/fmt_semantic_suffix.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 167 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — ML workflow time suffixes

Change type: byte-identical formatter relocation. Replaced all 10 `time_unit`
suffix sites in `vol1/ml_workflow/ml_workflow.qmd` with `fmt_time(...)`.
Symbol units (`ms`, `s`, `h`) use compact style; prose labels (`weeks`) use
`style="word"`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/ml_workflow/ml_workflow.qmd` | 10 | 77 | 36 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_time_ml_workflow`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose`.
- `audit_fmt_usage.py` now reports 126 `fmt_time` calls and the `time_unit`
  suffix bucket dropped from 532 to 522. `ml_workflow.qmd` has no remaining
  `time_unit` suffix sites.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/cli/checks/math_canonical.py book/cli/checks/fmt_semantic_suffix.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 167 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Corpus time-unit suffix lane

Change type: byte-identical formatter relocation plus formatter/source
normalization. Added `run_time_lane.py` for exact time suffixes and migrated the
remaining 522 `time_unit` suffix sites to `fmt_time(...)`. The lane uses full
unit-name strings in source (`"millisecond"`, `"second"`, `"hour"`,
`"microsecond"`, etc.) and keeps `style="word"` only for prose unit labels.
Old `fmt_int(..., suffix=...)` duration sites now round explicitly before
`fmt_time(..., precision=0)`.

Touched chapters and equivalence:

| Result | Chapter files | Calls | Equivalence |
|---|---|---:|---|
| pass | 30 chapters | 505 | byte-identical values + prose by `run_time_lane.py` |
| pass after microsecond formatter normalization | `vol1/frameworks`, `vol1/model_serving`, `vol2/data_storage`, `vol2/network_fabrics` | 17 | byte-identical values + prose by `run_time_lane.py` |

Validation details:

- Initial lane pass migrated 505/522 sites; the 17 queued sites differed only by
  `μs` (Greek mu, dominant in book source) vs Pint's `µs` micro sign.
- `fmt_time`/`fmt_qty` compact unit rendering now normalizes microsecond output
  to `μs`, centralizing that display decision in the formatter instead of
  encoding glyph variants at call sites.
- After that formatter normalization, the 17 queued microsecond sites migrated
  byte-identically.
- A source-only cleanup converted earlier QMD `fmt_time(..., "ms"/"s"/"h")`
  calls to full unit-name strings. Rendering remains controlled by `style` and
  is unchanged.
- Targeted equivalence check for `vol1/introduction/introduction.qmd` found the
  only intentional visible diff from formatter-owned microsecond normalization:
  `13.1 µs`/`27.6 µs` now render as `13.1 μs`/`27.6 μs`. The surrounding
  sentence was read and remains semantically correct.
- `audit_fmt_usage.py` now reports `fmt_time` calls at 650 and no remaining
  `time_unit` suffix bucket.
- Verification: `git diff --check` PASS; py_compile PASS for `fmt.py`,
  `codemod_fmt.py`, `run_time_lane.py`, audit and binder check modules;
  focused pytest suite PASS, 171 tests; `fmt_prose_contract.py` PASS, 0
  violations; `codemod_fmt.py queue` PASS, `by kind: {}`; `./book/binder check
  math` PASS; `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — WS4 clean Quantity-backed unit batch 2

Change type: byte-identical formatter relocation plus lane hardening. Ran
`run_unit_lane.py --write --all` after the time suffix lane and migrated 99
additional clean Quantity-backed physical-unit suffix sites to `fmt_qty(...)`.
The lane accepts a rewrite only when every exported value and substituted prose
line is byte-identical.

Touched chapters and equivalence:

| Chapter file | Calls migrated | Result |
|---|---:|---|
| `vol1/data_engineering/data_engineering.qmd` | 2 | byte-identical |
| `vol1/frameworks/frameworks.qmd` | 6 | byte-identical |
| `vol1/hw_acceleration/hw_acceleration.qmd` | 11 | byte-identical |
| `vol1/introduction/introduction.qmd` | 9 | byte-identical |
| `vol1/ml_ops/ml_ops.qmd` | 1 | byte-identical |
| `vol1/model_compression/model_compression.qmd` | 6 | byte-identical |
| `vol1/model_serving/model_serving.qmd` | 4 | byte-identical |
| `vol1/nn_computation/nn_computation.qmd` | 4 | byte-identical |
| `vol1/training/training.qmd` | 3 | byte-identical |
| `vol2/backmatter/appendix_assumptions.qmd` | 2 | byte-identical |
| `vol2/compute_infrastructure/compute_infrastructure.qmd` | 1 | byte-identical |
| `vol2/data_storage/data_storage.qmd` | 6 | byte-identical |
| `vol2/fault_tolerance/fault_tolerance.qmd` | 8 | byte-identical |
| `vol2/network_fabrics/network_fabrics.qmd` | 11 | byte-identical |
| `vol2/ops_scale/ops_scale.qmd` | 14 | byte-identical |
| `vol2/performance_engineering/performance_engineering.qmd` | 11 | byte-identical |

Validation details:

- `fmt_qty` calls increased from 164 to 263.
- `physical_unit` suffix calls dropped from 1,357 to 1,258.
- The unit lane now emits structured `per="token"` instead of
  `extra_suffix="/token"` for suffixes like `J/token`.
- `audit_fmt_usage.py` confirms QMD has zero `extra_suffix=` calls.
- Twenty Quantity-backed candidates remain queued because canonical formatter
  output would visibly change the unit label. Main buckets: binary memory
  capacity (`80 GB`→`80 GiB`) and missing bandwidth denominators
  (`TB`→`TB/s`, `GB`→`GB/s`). These require correctness/prose decisions.
- Verification: `git diff --check` PASS; py_compile PASS for formatter,
  codemod, audit, and binder check modules; focused pytest suite PASS, 171
  tests; `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue`
  PASS, `by kind: {}`; `./book/binder check math` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — Service-rate suffix lane

Change type: byte-identical formatter relocation plus checker hardening. Added
`run_rate_lane.py` and migrated exact non-physical service-rate suffixes to
`fmt_rate(...)`. Physical rates (`GB/s`, `TFLOP/s`, `W`) remain `fmt_qty` work;
this lane is only for counted service throughputs.

Touched chapters and equivalence:

| Chapter file | Calls migrated | Result |
|---|---:|---|
| `vol1/benchmarking/benchmarking.qmd` | 1 | byte-identical |
| `vol1/data_engineering/data_engineering.qmd` | 4 | byte-identical |
| `vol1/frameworks/frameworks.qmd` | 2 | byte-identical |
| `vol1/ml_systems/ml_systems.qmd` | 4 | byte-identical |
| `vol1/model_compression/model_compression.qmd` | 2 | byte-identical |
| `vol1/model_serving/model_serving.qmd` | 19 | byte-identical |
| `vol1/training/training.qmd` | 4 | byte-identical |
| `vol2/data_storage/data_storage.qmd` | 3 | byte-identical |
| `vol2/performance_engineering/performance_engineering.qmd` | 2 | byte-identical |

Validation details:

- `fmt_rate` calls increased from 26 to 67.
- `physical_unit` suffix calls dropped from 1,258 to 1,217.
- `audit_fmt_usage.py` now classifies exact service-rate suffixes as
  `service_rate`; none remain in QMD.
- `fmt_semantic_suffix` now reports exact service-rate suffixes as
  `rate_in_suffix`, with regression coverage.
- Verification: `git diff --check` PASS; py_compile PASS for formatter,
  codemod, audit, and binder check modules; focused pytest suite PASS, 174
  tests; `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue`
  PASS, `by kind: {}`; `./book/binder check math` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — Compound GPU-day count

Change type: formatter relocation with visible prose preserved. Replaced the
last split count-label construction in `vol1/model_compression/model_compression.qmd`:
`gpu_days_str` no longer renders `22,400 GPU` with prose appending `-days`;
it renders `22,400 GPU-days` directly via `fmt_count(..., label="GPU-day",
plural_label="GPU-days")`. The matching GPU-hour compound now also uses
`fmt_count(..., label="GPU-hour", plural_label="GPU-hours")`.

Validation details:

- `assess_equiv.py` reported the expected value diff for
  `NASCostCalc.gpu_days_str`: `22,400 GPU` → `22,400 GPU-days`.
- `assess_equiv.py` reported identical substituted prose for the footnote after
  dropping the literal `-days` outside the inline reference.
- `audit_fmt_usage.py` now reports no `count_label` suffix bucket.
- Verification after the service-rate and GPU-day count batch: `git diff
  --check` PASS; py_compile PASS; focused pytest suite PASS, 174 tests;
  `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue` PASS,
  `by kind: {}`; `./book/binder check math` PASS; `audit_prose_semantics.py`
  PASS, 0 findings across 81 files.

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
