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

## 2026-05-31 — `vol1/responsible_engr` physical-unit cleanup

Change type: byte-identical formatter relocation. Replaced all 14 remaining
physical-unit suffix sites in `responsible_engr` with typed quantity formatters.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/responsible_engr/responsible_engr.qmd` | 14 | 168 | 79 | identical values + prose |

Validation details:

- Migrated GPU power, training/inference carbon mass, metric-ton carbon
  summaries, and GPT-3-scale training energy in MWh/kWh.
- Carbon masses now use `ureg.kilogram`; metric-ton displays use
  `ureg.metric_ton` plus `unit_label="tons"` to preserve existing prose.
- `assess_equiv.py` baseline/snapshot/diff reported `IDENTICAL values` and
  `IDENTICAL prose` after preserving comma-sensitive kilogram displays.
- `responsible_engr` now has zero `suffix=` calls.
- `audit_fmt_usage.py` reports `physical_unit` suffixes dropped 870 -> 856,
  `fmt_qty` at 532, and `fmt_qty_int` at 31.
- Verification: `git diff --check` PASS; py_compile PASS for formatter,
  math-canonical, and fmt audit modules; focused pytest suite PASS, 190 tests;
  `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue` PASS,
  `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.
- HTML/PDF render evidence: not run yet. Full rendering is intentionally
  deferred for the separate render/prose audit checkpoint.

Status: non-render verified; render audit pending.

## 2026-05-31 — `vol1/backmatter/appendix_algorithm` physical-unit cleanup

Change type: byte-identical formatter relocation. Replaced all 11 remaining
physical-unit suffix sites in `appendix_algorithm` with typed quantity
formatters.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/backmatter/appendix_algorithm.qmd` | 11 | 39 | 18 | identical values + prose |

Validation details:

- Migrated sparse embedding storage and the GPT-2 XL training-memory breakdown:
  weights, gradients, optimizer state, model state, accelerator memory,
  remaining memory, small-batch total, large-batch activations, and the 80 GB
  accelerator class reference.
- `assess_equiv.py` baseline/snapshot/diff reported `IDENTICAL values` and
  `IDENTICAL prose`.
- `appendix_algorithm` now has zero `suffix=` calls.
- `audit_fmt_usage.py` reports `physical_unit` suffixes dropped 881 -> 870,
  `fmt_qty` at 518, and `fmt_qty_int` at 31.
- Verification: `git diff --check` PASS; py_compile PASS for formatter,
  math-canonical, and fmt audit modules; focused pytest suite PASS, 190 tests;
  `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue` PASS,
  `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.
- HTML/PDF render evidence: not run yet. Full rendering is intentionally
  deferred for the separate render/prose audit checkpoint.

Status: non-render verified; render audit pending.

## 2026-05-31 — `vol2/introduction` physical-unit cleanup

Change type: byte-identical formatter relocation. Replaced all 8 remaining
physical-unit suffix sites in `vol2/introduction` with typed quantity
formatters.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol2/introduction/introduction.qmd` | 8 | 37 | 20 | identical values + prose |

Validation details:

- Migrated GPT-3 gradient sizes, InfiniBand HDR/NDR bandwidth recaps, NVLink
  H100 bandwidth, and A100 FP16 throughput.
- Split-rate bandwidth values are now checked as `Gb/s` or `GB/s` through
  `fmt_qty(...)`, while `unit_label="Gb"|"GB"` preserves the existing value
  string because prose supplies `/s`.
- `assess_equiv.py` baseline/snapshot/diff reported `IDENTICAL values` and
  `IDENTICAL prose`.
- `vol2/introduction` now has zero `suffix=` calls.
- `audit_fmt_usage.py` reports `physical_unit` suffixes dropped 889 -> 881,
  `fmt_qty` at 507, and `fmt_qty_int` at 31.
- Verification: `git diff --check` PASS; py_compile PASS for formatter,
  math-canonical, and fmt audit modules; focused pytest suite PASS, 190 tests;
  `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue` PASS,
  `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.
- HTML/PDF render evidence: not run yet. Full rendering is intentionally
  deferred for the separate render/prose audit checkpoint.

Status: non-render verified; render audit pending.

## 2026-05-31 — `vol1/ml_workflow` physical-unit cleanup

Change type: byte-identical formatter relocation. Replaced all 8 remaining
physical-unit suffix sites in `ml_workflow` with typed quantity formatters.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/ml_workflow/ml_workflow.qmd` | 8 | 77 | 36 | identical values + prose |

Validation details:

- Migrated MobileNet model size, stage-interface memory constraints,
  bandwidth-vs-compute daily data volume/uplink rates, and deployment image
  size.
- Uplink bandwidth is now represented as `ureg.megabit/second` and converted to
  `MB/second` for the paired display.
- `assess_equiv.py` baseline/snapshot/diff reported `IDENTICAL values` and
  `IDENTICAL prose` after preserving the comma-sensitive `7,500 MB` display.
- `ml_workflow` now has zero `suffix=` calls.
- `audit_fmt_usage.py` reports `physical_unit` suffixes dropped 897 -> 889,
  `fmt_qty` at 499, and `fmt_qty_int` at 31.
- Verification: `git diff --check` PASS; py_compile PASS for formatter,
  math-canonical, and fmt audit modules; focused pytest suite PASS, 190 tests;
  `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue` PASS,
  `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.
- HTML/PDF render evidence: not run yet. Full rendering is intentionally
  deferred for the separate render/prose audit checkpoint.

Status: non-render verified; render audit pending.

## 2026-05-31 — Vol1 introduction/data-selection physical-unit cleanup

Change type: byte-identical formatter relocation. Replaced 8 remaining
physical-unit suffix sites with typed quantity formatters across two Vol1
chapters.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/introduction/introduction.qmd` | 2 | 91 | 45 | identical values + prose |
| `vol1/data_selection/data_selection.qmd` | 6 | 250 | 128 | identical values + prose |

Validation details:

- Migrated GPT-3 training energy, A100 peak FLOP/s, and the random-access
  storage throughput table.
- The storage throughput table remains a split-rate display: each value is
  checked as `MB/second` through `fmt_qty(...)`, while `unit_label="MB"`
  preserves the existing value string because the table literal appends `/s`.
- `assess_equiv.py` baseline/snapshot/diff reported `IDENTICAL values` and
  `IDENTICAL prose` for both touched files.
- Both touched files now have zero `suffix=` calls.
- `audit_fmt_usage.py` reports `physical_unit` suffixes dropped 905 -> 897,
  `fmt_qty` at 491, and `fmt_qty_int` at 31.
- Verification: `git diff --check` PASS; py_compile PASS for formatter,
  math-canonical, and fmt audit modules; focused pytest suite PASS, 190 tests;
  `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue` PASS,
  `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.
- HTML/PDF render evidence: not run yet. Full rendering is intentionally
  deferred for the separate render/prose audit checkpoint.

Status: non-render verified; render audit pending.

## 2026-05-31 — Assumptions appendix physical-unit cleanup

Change type: byte-identical formatter relocation. Replaced 10 remaining
physical-unit suffix sites with typed quantity formatters across the Vol1 and
Vol2 assumptions appendices.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/backmatter/appendix_assumptions.qmd` | 5 | 257 | 124 | identical values + prose |
| `vol2/backmatter/appendix_assumptions.qmd` | 5 | 275 | 122 | identical values + prose |

Validation details:

- Migrated H100 peak FLOP/s, memory bandwidth, training memory, H100 memory
  capacity, A100 TDP, Llama gradient size, facility power, WUE example power,
  and AI rack power displays.
- `fmt_qty(..., unit_label="TFLOP/s")` preserves the established singular FLOP
  display instead of Pint's default `TFLOPs/s`.
- `assess_equiv.py` baseline/snapshot/diff reported `IDENTICAL values` and
  `IDENTICAL prose` for both touched files.
- Both touched files now have zero `suffix=` calls.
- `audit_fmt_usage.py` reports `physical_unit` suffixes dropped 915 -> 905,
  `fmt_qty` at 483, and `fmt_qty_int` at 31.
- Verification: `git diff --check` PASS; py_compile PASS for formatter,
  math-canonical, and fmt audit modules; focused pytest suite PASS, 190 tests;
  `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue` PASS,
  `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.
- HTML/PDF render evidence: not run yet. Full rendering is intentionally
  deferred for the separate render/prose audit checkpoint.

Status: non-render verified; render audit pending.

## 2026-05-31 — Small chapter physical-unit cleanup

Change type: byte-identical formatter relocation. Replaced 17 remaining
physical-unit suffix sites with typed quantity formatters across three small
chapters.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/ml_ops/ml_ops.qmd` | 5 | 132 | 75 | identical values + prose |
| `vol1/backmatter/appendix_data.qmd` | 6 | 36 | 28 | identical values + prose |
| `vol2/security_privacy/security_privacy.qmd` | 6 | 69 | 38 | identical values + prose |

Validation details:

- Migrated KV-cache capacity, observability ingest rates, monitoring storage
  volume, serialization throughput, data-algebra sizes, and TEE/model memory
  displays.
- The rejected automatic `ml_ops` unit-lane candidates were manually corrected:
  split-rate ingest values now attach `/second` before `fmt_qty(...,
  GB/second|MB/second)`.
- `assess_equiv.py` baseline/snapshot/diff reported `IDENTICAL values` and
  `IDENTICAL prose` for all three touched files after preserving the
  comma-sensitive `> 1,000 MB/s` display.
- All three touched files now have zero `suffix=` calls.
- `audit_fmt_usage.py` reports `physical_unit` suffixes dropped 932 -> 915,
  `fmt_qty` at 473, and `fmt_qty_int` at 31.
- Verification: `git diff --check` PASS; py_compile PASS for formatter,
  math-canonical, and fmt audit modules; focused pytest suite PASS, 190 tests;
  `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue` PASS,
  `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.
- HTML/PDF render evidence: not run yet. Full rendering is intentionally
  deferred for the separate render/prose audit checkpoint.

Status: non-render verified; render audit pending.

## 2026-05-31 — Tiny Vol2 physical-unit tail

Change type: byte-identical formatter relocation. Replaced 7 remaining
physical-unit suffix sites with typed quantity formatters across four small Vol2
files.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol2/backmatter/appendix_c3.qmd` | 1 | 46 | 20 | identical values + prose |
| `vol2/robust_ai/robust_ai.qmd` | 1 | 36 | 19 | identical values + prose |
| `vol2/conclusion/conclusion.qmd` | 2 | 19 | 18 | identical values + prose |
| `vol2/backmatter/appendix_reliability.qmd` | 3 | 51 | 25 | identical values + prose |

Validation details:

- Migrated aggregate PFLOP/s, V100 memory bandwidth, machine power, H100
  TFLOP/s, checkpoint size, and checkpoint write bandwidth displays.
- `assess_equiv.py` baseline/snapshot/diff reported `IDENTICAL values` and
  `IDENTICAL prose` for all four touched files after preserving comma behavior
  on the two comma-sensitive displays.
- All four touched files now have zero `suffix=` calls.
- `audit_fmt_usage.py` reports `physical_unit` suffixes dropped 939 -> 932,
  `fmt_qty` at 456, and `fmt_qty_int` at 31.
- Verification: `git diff --check` PASS; py_compile PASS for formatter,
  math-canonical, and fmt audit modules; focused pytest suite PASS, 190 tests;
  `fmt_prose_contract.py` PASS, 0 violations; `codemod_fmt.py queue` PASS,
  `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.
- HTML/PDF render evidence: not run yet. Full rendering is intentionally
  deferred for the separate render/prose audit checkpoint.

Status: non-render verified; render audit pending.

## 2026-05-31 — Compound-scale suffix lane

Change type: typed formatter relocation with three deliberate adjudicated
visible/value changes. Cleared all 14 `compound_scale` suffix sites:
word-scale counts moved to direct `fmt_count(..., scale="million"|"billion")`,
scaled count rates moved to `fmt_rate(..., scale=...)`, the `7-billion`
modifier moved to `fmt_count(..., attributive=True)`, and word-scale FLOP prose
now uses `fmt_qty(..., unit_label="billion FLOPs")` / `"trillion FLOPs"` so Pint
still validates the value.

Touched chapters and equivalence:

| Chapter file | Result |
|---|---|
| `vol1/data_selection/data_selection.qmd` | `ActiveLearningRoi.cost_saving_str` value changed from `4.75 Million` to `\$4.75 Million`; substituted prose stayed identical because the external `\$` moved into `fmt_usd`. |
| `vol1/ml_systems/ml_systems.qmd` | Word-scale parameter/query values stayed identical except the user-approved compact scale style changed `200 K parameters` to `200K parameters`; surrounding sentence was read and remains correct. |
| `vol1/model_serving/model_serving.qmd` | `2.2 million tokens` and `45.2 million tokens/hour` stayed byte-identical. |
| `vol1/nn_computation/nn_computation.qmd` | `7-billion` stayed byte-identical through `fmt_count(..., attributive=True)`. |
| `vol1/training/training.qmd` | FLOP phrases stayed byte-identical; `32K tokens` changed to `32.8K tokens` because the exact batch count is 32,768 and `fmt_count` correctly refused to hide that at precision 0. |

Validation details:

- `audit_fmt_usage.py` reports no remaining `compound_scale` suffix bucket;
  remaining suffix bucket is `physical_unit` 1,126.
- `assess_equiv.py baseline --ref HEAD` / `snapshot` checked touched chapters:
  `data_selection` 250 values + 128 prose lines; `ml_systems` 296 + 143;
  `model_serving` 365 + 172; `nn_computation` 200 + 107; `training` 453 + 214.
- Verification: `git diff --check` PASS; `python3 -m py_compile
  mlsysim/mlsysim/fmt.py book/tools/audit/fmt/audit_fmt_usage.py` PASS;
  focused pytest suite PASS, 182 tests; `fmt_prose_contract.py` PASS, 0
  violations; `codemod_fmt.py queue` PASS, `by kind: {}`; `./book/binder check
  math` PASS; `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — Appendix bandwidth denominator relocation

Change type: substituted-prose-identical unit relocation. Moved bandwidth `/s`
denominators from prose/table literals into `fmt_qty(...)` so the formatter owns
the full Pint unit (`GB/second` or `TB/second`).

Touched chapters and equivalence:

| Chapter file | Result |
|---|---|
| `vol2/backmatter/appendix_assumptions.qmd` | `FleetQuickCalc.ib_bw_str`, `H100Recap.bw_tb_str`, and `UnitConstants.ib_ndr_gbs_from_gbps_str` values changed from `GB`/`TB` strings to `GB/s`/`TB/s`; external `/s` text was removed, and substituted prose stayed byte-identical across 122 inline-ref lines. |
| `vol2/backmatter/appendix_fleet.qmd` | 13 bandwidth strings moved to `fmt_qty(..., GB/second|TB/second)`; table `/s` text was removed. Substituted prose stayed byte-identical across 99 inline-ref lines, including comma-preserved `1,200 GB/s` and `1,800 GB/s`. |

Validation details:

- `audit_fmt_usage.py` physical-unit suffix count dropped from 1,126 to 1,110.
- Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
  PASS, 182 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — Network `Gb/s` suffix lane

Change type: byte-identical formatter relocation. Replaced all 20
`suffix=" Gb/s"` sites with `fmt_qty(..., Gbps, unit_label="Gb/s")`, preserving
the book's visible `Gb/s` spelling while carrying Pint units into the formatter.

Touched chapters and equivalence:

| Chapter file | Values/prose checked | Result |
|---|---:|---|
| `vol1/backmatter/appendix_machine.qmd` | 111 values / 70 prose lines | identical |
| `vol1/data_engineering/data_engineering.qmd` | 203 / 99 | identical |
| `vol1/hw_acceleration/hw_acceleration.qmd` | 255 / 140 | identical |
| `vol2/compute_infrastructure/compute_infrastructure.qmd` | 320 / 183 | identical |
| `vol2/distributed_training/distributed_training.qmd` | 261 / 142 | identical |
| `vol2/fleet_orchestration/fleet_orchestration.qmd` | 160 / 60 | identical |
| `vol2/network_fabrics/network_fabrics.qmd` | 106 / 62 | identical |

Validation details:

- `audit_fmt_usage.py` reports no remaining `suffix=" Gb/s"` sites; physical
  unit suffix count dropped from 1,110 to 1,090.
- Follow-up TODO: replace repeated `unit_label="Gb/s"` with a central stock unit
  display-label registry so `fmt_qty(..., Gbps)` renders `Gb/s` automatically.
- Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
  PASS, 182 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — GiB-backed memory capacity lane

Change type: byte-identical formatter relocation. Replaced the remaining
`m_as(GiB)` + `suffix=" GB"` memory-capacity sites with
`fmt_qty(..., GiB, unit_label="GB")`. This preserves the book's visible `GB`
house style while keeping the binary-capacity source unit attached through the
formatter.

Touched chapters and equivalence:

| Chapter file | Values/prose checked | Result |
|---|---:|---|
| `vol1/model_serving/model_serving.qmd` | 365 values / 172 prose lines | identical |
| `vol2/backmatter/appendix_fleet.qmd` | 173 / 99 | identical |
| `vol2/compute_infrastructure/compute_infrastructure.qmd` | 320 / 183 | identical |
| `vol2/distributed_training/distributed_training.qmd` | 261 / 142 | identical |
| `vol2/fleet_orchestration/fleet_orchestration.qmd` | 160 / 60 | identical |
| `vol2/inference/inference.qmd` | 208 / 119 | identical |
| `vol2/performance_engineering/performance_engineering.qmd` | 109 / 58 | identical |

Validation details:

- `audit_fmt_usage.py` physical-unit suffix count dropped from 1,090 to 1,071.
- Follow-up TODO: hardware/model display accessors should own recurring spec
  display policies for memory capacity instead of requiring QMD call sites to
  know `GiB` + house-label `GB`.
- Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
  PASS, 182 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — Direct Quantity physical-unit lane

Change type: byte-identical formatter relocation plus formatter API hardening.
Added `fmt_qty_int(...)` for the narrow case where a Pint Quantity should remain
unit-checked but the displayed magnitude is intentionally rounded to an integer.
Kept `fmt_qty(..., precision=0)` strict so it still rejects hidden fractional
unit conversions.

Touched chapters and equivalence:

| Chapter file | Values/prose checked | Result |
|---|---:|---|
| `vol1/introduction/introduction.qmd` | 91 values / 45 prose lines | identical |
| `vol1/model_compression/model_compression.qmd` | 162 / 74 | identical |
| `vol1/data_engineering/data_engineering.qmd` | 203 / 99 | identical |
| `vol1/ml_systems/ml_systems.qmd` | 296 / 143 | identical |
| `vol1/model_serving/model_serving.qmd` | 365 / 172 | identical |
| `vol2/backmatter/appendix_fleet.qmd` | 173 / 99 | identical |
| `vol2/data_storage/data_storage.qmd` | 180 / 99 | identical |
| `vol2/performance_engineering/performance_engineering.qmd` | 109 / 58 | identical |

Validation details:

- `audit_fmt_usage.py` physical-unit suffix count dropped from 1,071 to 1,042.
- `fmt_qty_int` call count is 20; `fmt_qty` call count is 357.
- Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
  PASS, 190 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — `vol1/ml_systems` physical-unit lane

Change type: byte-identical chapter cleanup. Migrated all 30 remaining
physical-unit suffix sites in `vol1/ml_systems/ml_systems.qmd` to typed
quantity formatters. This included data-size, bandwidth, TOPS, power, energy,
temperature, and battery-capacity displays. One-off editorial labels such as
`TOPS peak`, `TOPS derated`, `Mb/s`, and `KB of detection summaries` remain
checked `unit_label=` values rather than raw suffixes.

Touched chapter and equivalence:

| Chapter file | Values/prose checked | Result |
|---|---:|---|
| `vol1/ml_systems/ml_systems.qmd` | 296 values / 143 prose lines | identical |

Validation details:

- `audit_fmt_usage.py` physical-unit suffix count dropped from 1,042 to 1,012.
- `vol1/ml_systems/ml_systems.qmd` now has 0 physical-unit suffix sites.
- `fmt_qty` call count is 384; `fmt_qty_int` call count is 23.
- Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
  PASS, 190 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — `vol2/backmatter/appendix_fleet` physical-unit lane

Change type: byte-identical chapter cleanup. Migrated all 28 remaining
physical-unit suffix sites in `vol2/backmatter/appendix_fleet.qmd` to typed
quantity formatters. This covered checkpoint sizes, network bandwidth, HBM
bandwidth, BF16 peak FLOP/s, model weight footprint, rack/IT/facility power,
and PUE overhead displays.

Touched chapter and equivalence:

| Chapter file | Values/prose checked | Result |
|---|---:|---|
| `vol2/backmatter/appendix_fleet.qmd` | 173 values / 99 prose lines | identical |

Validation details:

- `audit_fmt_usage.py` physical-unit suffix count dropped from 1,012 to 984.
- `vol2/backmatter/appendix_fleet.qmd` now has 0 `suffix=` calls.
- `fmt_qty` call count is 406; `fmt_qty_int` call count is 29.
- The checkpoint write-bandwidth export remains byte-identical as `100 GB`
  with prose adding `/s`, but the value is now checked as `GB/second` through
  `fmt_qty(..., unit_label="GB")`. This split-rate rendering is recorded as a
  unit-label/prose-bound output cleanup follow-up.
- Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
  PASS, 190 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — `vol2/collective_communication` physical-unit lane

Change type: byte-identical chapter cleanup. Migrated all 26 remaining
physical-unit suffix sites in `vol2/collective_communication/collective_communication.qmd`
to typed quantity formatters. This covered gradient/message sizes, critical
message size, MoE token transfer sizes, Ring-vs-Tree crossover examples,
NVLink/InfiniBand bandwidth recaps, hierarchical AllReduce volumes, and overlap
bucket sizes.

Touched chapter and equivalence:

| Chapter file | Values/prose checked | Result |
|---|---:|---|
| `vol2/collective_communication/collective_communication.qmd` | 104 values / 60 prose lines | identical |

Validation details:

- `audit_fmt_usage.py` physical-unit suffix count dropped from 984 to 958.
- `vol2/collective_communication/collective_communication.qmd` now has 0
  `suffix=` calls.
- `fmt_qty` call count is 432; `fmt_qty_int` call count remains 29.
- Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
  PASS, 190 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — `vol2/fault_tolerance` physical-unit lane

Change type: byte-identical chapter cleanup. Migrated all 10 remaining
physical-unit suffix sites in `vol2/fault_tolerance/fault_tolerance.qmd` to
typed quantity formatters. This covered checkpoint component sizes, total
checkpoint size, per-node storage throughput, local NVMe bandwidth, GPT-3 shard
size, and recovery read bandwidth.

Touched chapter and equivalence:

| Chapter file | Values/prose checked | Result |
|---|---:|---|
| `vol2/fault_tolerance/fault_tolerance.qmd` | 129 values / 70 prose lines | identical |

Validation details:

- `audit_fmt_usage.py` physical-unit suffix count dropped from 958 to 948.
- `vol2/fault_tolerance/fault_tolerance.qmd` now has 0 `suffix=` calls.
- `fmt_qty` call count is 441; `fmt_qty_int` call count is 30.
- `CheckpointDebug.per_node_mbs_str` uses `fmt_qty_int(...)` because the old
  value intentionally rounded `19.5 MB/s` to `20 MB/s`.
- Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
  PASS, 190 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.

## 2026-05-31 — `vol2/backmatter/appendix_communication` physical-unit lane

Change type: byte-identical chapter cleanup. Migrated all 9 remaining
physical-unit suffix sites in `vol2/backmatter/appendix_communication.qmd` to
typed quantity formatters. Most of these are split-rate bandwidth displays that
export `GB` while prose/table text appends `/s`; the formatter now checks them
as `GB/second` with `unit_label="GB"` to preserve the byte-identical value.

Touched chapter and equivalence:

| Chapter file | Values/prose checked | Result |
|---|---:|---|
| `vol2/backmatter/appendix_communication.qmd` | 56 values / 33 prose lines | identical |

Validation details:

- `audit_fmt_usage.py` physical-unit suffix count dropped from 948 to 939.
- `vol2/backmatter/appendix_communication.qmd` now has 0 `suffix=` calls.
- `fmt_qty` call count is 449; `fmt_qty_int` call count is 31.
- Verification: `git diff --check` PASS; py_compile PASS; focused pytest suite
  PASS, 190 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `./book/binder check math` PASS;
  `./book/binder check code --scope lego-dead-code` PASS;
  `audit_prose_semantics.py` PASS, 0 findings across 81 files.

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

## 2026-05-31 — Resource-time count-label relocation

Change type: byte-identical formatter relocation. Replaced 15 straightforward
resource-time suffixes with strict `fmt_count(..., label=..., plural_label=...)`
calls: `PFLOP-days`, `TPUv4-hours`, `person-years`, `instance-seconds`,
`GPU-hours`, and `GPU-hr`.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/data_engineering/data_engineering.qmd` | 2 | 203 | 99 | identical values + prose |
| `vol1/ml_systems/ml_systems.qmd` | 1 | 296 | 143 | identical values + prose |
| `vol1/data_selection/data_selection.qmd` | 5 | 250 | 128 | identical values + prose |
| `vol1/responsible_engr/responsible_engr.qmd` | 4 | 168 | 79 | identical values + prose |
| `vol1/model_serving/model_serving.qmd` | 1 | 365 | 172 | identical values + prose |
| `vol2/inference/inference.qmd` | 1 | 208 | 119 | identical values + prose |
| `vol2/ops_scale/ops_scale.qmd` | 1 | 195 | 95 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_resource_time_*`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` reports `fmt_count` calls increased to 246. Remaining
  suffix buckets: `physical_unit` 1,126; `unit_rate_or_denominator` 16;
  `compound_scale` 14; `op_count` 12; `resource_time` 4; `time_compound` 4.
- The four remaining `resource_time` sites are hyphenated attributive forms
  (`-hour`, `-minute`) and are intentionally deferred to a prose/API decision.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/tools/audit/fmt/audit_fmt_usage.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 176 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Exact FLOP-count suffix relocation

Change type: byte-identical formatter relocation. Replaced 12 exact FLOP-count
suffixes (`GFLOPs`, `MFLOPs`, `KFLOPs`, `PFLOPs`) with `fmt_qty(...)` and the
existing Pint FLOP units. No `fmt_ops` wrapper was added; Pint already provides
the unit check for these exact-unit FLOP counts.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/conclusion/conclusion.qmd` | 1 | 25 | 15 | identical values + prose |
| `vol1/frameworks/frameworks.qmd` | 6 | 131 | 70 | identical values + prose |
| `vol1/ml_systems/ml_systems.qmd` | 3 | 296 | 143 | identical values + prose |
| `vol1/ml_workflow/ml_workflow.qmd` | 1 | 77 | 36 | identical values + prose |
| `vol1/training/training.qmd` | 1 | 453 | 214 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_op_count_*`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` reports no remaining `op_count` suffix bucket and
  `fmt_qty` calls increased to 275.
- Word-scale FLOP phrases such as `billion FLOPs` and `trillion FLOPs` remain in
  `compound_scale`; migrating those to `GFLOPs`/`TFLOPs` would be a visible prose
  change and needs a separate decision.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/tools/audit/fmt/audit_fmt_usage.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 176 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Attributive time/resource-time suffix relocation

Change type: small formatter API addition plus byte-identical relocation. Added
`fmt_time(..., style="word", attributive=True)` for hyphenated singular time
noun modifiers and replaced the four remaining `resource_time` suffix sites
(`-hour`, `-minute`).

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/data_engineering/data_engineering.qmd` | 1 | 203 | 99 | identical values + prose |
| `vol1/data_selection/data_selection.qmd` | 1 | 250 | 128 | identical values + prose |
| `vol2/distributed_training/distributed_training.qmd` | 2 | 261 | 142 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_time_attr_*`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` reports no remaining `resource_time` suffix bucket and
  `fmt_time` calls increased to 661.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/tools/audit/fmt/audit_fmt_usage.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 178 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Time-compound suffix relocation

Change type: small formatter API addition plus byte-identical relocation.
Added checked `fmt_time(..., marker="+")` for compact trailing-plus time
notation and replaced the four remaining `time_compound` suffix sites.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/introduction/introduction.qmd` | 2 | 91 | 45 | identical values + prose |
| `vol1/ml_systems/ml_systems.qmd` | 1 | 296 | 143 | identical values + prose |
| `vol1/ml_workflow/ml_workflow.qmd` | 1 | 77 | 36 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_time_compound_*`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` reports no remaining `time_compound` suffix bucket and
  `fmt_time` calls increased to 665.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/tools/audit/fmt/audit_fmt_usage.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 179 tests; `fmt_prose_contract.py` PASS, 0 violations;
  `codemod_fmt.py queue` PASS, `by kind: {}`; `audit_prose_semantics.py` PASS,
  0 findings across 81 files.

## 2026-05-31 — Unit-rate/denominator suffix relocation

Change type: small formatter API addition plus byte-identical relocation. Added
checked `fmt_qty(..., unit_label=...)` for house-style unit labels where Pint's
compact label would not be byte-identical, then migrated all 16
`unit_rate_or_denominator` suffix sites.

Touched chapters and equivalence:

| Chapter file | Calls | Value exports checked | Inline prose lines checked | Result |
|---|---:|---:|---:|---|
| `vol1/hw_acceleration/hw_acceleration.qmd` | 3 | 255 | 140 | identical values + prose |
| `vol1/introduction/introduction.qmd` | 3 | 91 | 45 | identical values + prose |
| `vol1/ml_systems/ml_systems.qmd` | 2 | 296 | 143 | identical values + prose |
| `vol1/ml_workflow/ml_workflow.qmd` | 3 | 77 | 36 | identical values + prose |
| `vol1/nn_computation/nn_computation.qmd` | 1 | 200 | 107 | identical values + prose |
| `vol1/training/training.qmd` | 3 | 453 | 214 | identical values + prose |
| `vol2/edge_intelligence/edge_intelligence.qmd` | 1 | 75 | 39 | identical values + prose |

Validation details:

- Before/after snapshots were generated with `assess_equiv.py baseline --ref HEAD`
  and `assess_equiv.py snapshot` under `/tmp/fmt_den_*`.
- `assess_equiv.py diff` reported `IDENTICAL values` and `IDENTICAL prose` for
  every touched chapter.
- `audit_fmt_usage.py` reports no remaining `unit_rate_or_denominator` suffix
  bucket and `fmt_qty` calls increased to 291.
- Verification: `python3 -m py_compile mlsysim/mlsysim/fmt.py book/tools/audit/fmt/audit_fmt_usage.py` PASS;
  `git diff --check` PASS; `./book/binder check math` PASS; focused pytest
  suite PASS, 181 tests; `fmt_prose_contract.py` PASS, 0 violations;
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
