# MLSysBook LEGO Unit Hardening - Agent Execution Plan

**Written:** 2026-05-31
**Worktree:** `/Users/VJ/GitHub/MLSysBook-fmt-fix`
**Branch:** `fmt-fix`
**Purpose:** Finish the LEGO unit hardening work by making every LEGO cell
follow a single source-of-truth, Pint-safe, formatter-safe contract.

This document is the coordinator plan for the overnight pass. It complements
`mlsysim-lego-unit-hardening-plan.md` and
`mlsysim-lego-unit-hardening-PROGRESS.md`; it does not replace the live progress
tracker.

**Current status (2026-06-01):** the overnight source/render pass is complete
for the current worktree. Static LEGO source gates, focused tests, full
`mlsysim` tests, full pre-commit, Vol I/II PDFs, full Vol I/II HTML, xref scans,
and rendered LEGO substitution verification are green. Remaining work is Phase
10 commit/merge verification.

---

## 1. Non-negotiable Doctrine

### 1.1 MLSysIM is the source of truth

Every measurable fact used by the book must trace to `mlsysim`.

This includes hardware specs, memory capacity, memory bandwidth, compute peak,
power, latency, energy, carbon intensity, prices, parameter counts, token
counts, model specs, grid assumptions, datacenter assumptions, and any reusable
teaching anchor.

Chapter-local literals are allowed only when the value is genuinely local to a
scenario and has no canonical home. Such values must be visibly loaded in the
LEGO `LOAD` stage and named as assumptions, not hidden in `EXECUTE` or
`OUTPUT`.

### 1.2 Pint quantities stay attached

Pint units are not display decoration. They are the type system for physical
values.

The preferred path is:

```text
mlsysim registry Quantity
  -> LEGO LOAD
  -> EXECUTE as Quantity
  -> GUARD as Quantity or explicit scalar check
  -> OUTPUT via fmt_qty or a domain formatter
  -> prose reference
```

Scalar extraction with `.to(unit).magnitude` is permitted only at true scalar
boundaries: plotting, algorithms that require floats, dimensionless ratios,
guard comparisons, or intentionally value/unit-split tables.

Passing a scalar magnitude back into `fmt_qty` or reattaching units after an
avoidable scalar extraction is a defect.

### 1.3 OUTPUT owns rendered units

The formatter chosen in `OUTPUT` determines what prose is allowed to write.

- `fmt_qty`, domain quantity formatters, `fmt_usd`, `fmt_percent` in prose or
  symbol style, `fmt_pp`, and `fmt_count` produce closed strings. Prose uses the
  inline Python reference bare.
- `fmt`, `fmt_int`, `fmt_ratio`, and `fmt_percent(style="number")` produce open
  strings. Prose supplies the unit or semantic word.
- `fmt_multiple` currently produces the number only; prose supplies
  `$\times$`. A future formatter improvement may move glyph ownership inside the
  formatter, but that must be done as one coordinated migration with a prose
  contract update and tests.

No LEGO cell should manually assemble a numeric string with f-strings,
concatenation, ad hoc suffixes, or `MarkdownStr` unless the export is genuinely
non-numeric, such as a label or sequence.

### 1.4 Improvements bubble upward

Do not solve repeated problems locally. The coordinator decides the proper home:

| Discovery | Durable home |
|---|---|
| Missing unit alias | `mlsysim/mlsysim/core/units.py` or unit definition file, plus tests |
| Missing hardware/model/grid/fabric spec | MLSysIM registry/schema/data with provenance |
| Repeated formula | `mlsysim/mlsysim/physics/quantities.py` or a domain module, plus tests |
| Repeated display policy | `mlsysim/mlsysim/fmt.py` domain formatter, plus tests |
| Repeated lintable mistake | `book/tools/scripts/lint_lego_units.py`, baseline/update tests |
| Prose contract drift | prose-unit checker or fmt prose contract checker |

Chapter agents may identify these gaps, but they must not invent shared
abstractions independently.

---

## 2. Why The Work Is Staged

The book currently mixes several states: some cells are already migrated to
domain formatters, some have legacy open `fmt(...)` calls, some use local
scenario assumptions, and some rely on registry values. A single all-at-once
rewrite would confuse context and make review hard.

The safe strategy is staged bubbling:

1. Normalize what each chapter exports and how prose consumes it.
2. Preserve dimensional correctness through `EXECUTE`.
3. Audit and migrate `LOAD` to the registry/source-of-truth layer.
4. Promote repeated local patterns into `mlsysim`.
5. Tighten lint gates only after the corpus is clean or honestly baselined.

This keeps each change reviewable and prevents deadlock: chapter agents can
work on OUTPUT and local quantity flow while the coordinator batches shared
registry and formatter work.

---

## 3. Chapter State Machine

Every QMD with LEGO cells moves through the same states. A chapter cannot be
called finished until it reaches S5.

| State | Name | What must be true |
|---|---|---|
| S0 | Inventory | Every LEGO cell, export, inline prose ref, local literal, and unit conversion site is listed. |
| S1 | OUTPUT-safe | Exports use typed/domain formatters; prose follows closed/open contract; no duplicated units. |
| S2 | Quantity-safe | Physical calculations keep Pint quantities; scalar extraction is explicit and justified. |
| S3 | Source-safe | Real facts load from MLSysIM; remaining local assumptions are scenario-local and documented. |
| S4 | Guard-safe | Important values have dimensional/range checks; formatter calls cannot silently print nonsense. |
| S5 | Render-safe | Static lint, headless execution, and chapter render checks pass; chapter report is complete. |

Important rule: S1 may be completed before S3 if needed. That allows a chapter
agent to standardize outputs using the current local variables, while a later
coordinator wave moves those variables into MLSysIM. The final chapter status is
not complete until S3 is also done.

---

## 4. Work Lanes

### Lane A - OUTPUT normalization

Goal: every prose-facing export is owned by one typed formatter.

Tasks:

- Replace closed-name `fmt(...)` uses with `fmt_qty` or the correct domain
  formatter.
- Replace suffix-owned physical units with Pint display units.
- Replace percent, currency, count, and multiplier ad hoc formatting with typed
  helpers.
- Remove prose unit duplication for closed exports.
- Preserve visible prose unless the old value was wrong or the new formatter
  intentionally changes style.

Exit criteria:

- L014 baseline entries for the chapter are burned down or reclassified.
- Prose-unit checker is clean for the chapter.
- No numeric `MarkdownStr` escape hatches remain unless justified.

### Lane B - Quantity-flow audit

Goal: calculations remain dimensional until the last responsible boundary.

Tasks:

- Review `.to(...).magnitude` sites and classify each scalar boundary.
- Replace avoidable scalar arithmetic with Pint arithmetic.
- Fix cross-unit ratios by converting to a common unit before magnitude
  extraction, or by using `.to("")` for dimensionless quantities.
- Preserve compound units such as `TFLOP / second / watt`,
  `joule / token`, `g / kWh`, and `MWh / year`.
- Use parenthesized unit expressions for readability:
  `1.9 * (TB / second)`.

Exit criteria:

- No `fmt_qty(<scalar>, unit)` pattern.
- No magnitude extraction followed by reattached units unless documented.
- Rate quantities keep their full dimensions.

### Lane C - LOAD source-of-truth audit

Goal: real facts come from MLSysIM.

Tasks:

- Identify chapter-local literals in LOAD, EXECUTE, tables, and helper data
  structures.
- Classify each literal:
  - registry fact;
  - technology-class fact;
  - literature/cited field figure;
  - scenario-local assumption;
  - pure pedagogical toy value.
- Move registry facts into MLSysIM with provenance.
- Keep scenario-local assumptions in LOAD with explicit names and comments.
- Add schema fields rather than hiding missing fields as local constants.

Exit criteria:

- Every real-world spec or reusable anchor has a registry path.
- Remaining local assumptions are named, documented, and intentionally local.

### Lane D - Domain formatter and helper promotion

Goal: repeated chapter patterns become shared code.

Tasks:

- Promote repeated display decisions to `fmt.py`.
- Promote repeated physics/math to `physics/quantities.py` or a domain module.
- Add tests before broad chapter migration when a helper becomes shared.
- Prefer narrow helpers with strong units over broad string utilities.

Exit criteria:

- No repeated local formatter wrappers across chapters.
- Shared helpers have focused tests.

### Lane E - Verification and gates

Goal: mistakes found once become automated checks.

Tasks:

- Run lint and formatter tests after each wave.
- Add lint rules for recurring unsafe patterns.
- Keep baselines honest: a baseline entry is a queue item, not a success.
- Render representative chapters during migration and all chapters at the end.

Exit criteria:

- Chapter report includes command results.
- Global gates pass before merge.

---

## 5. Coordinator / Chapter-Agent Model

### 5.0 Pre-launch foundation pass

Before any chapter agents are launched, the coordinator must settle the shared
surface that agents are allowed to use. This is a short central pass, not
chapter work.

1. **Rule consistency check.** Search `.claude/rules` for stale instructions
   that contradict the current unit hardening policy, especially `.m_as(`,
   legacy `mlsys.*` imports, `fmt(..., suffix=" GB/s")`, `suffix="%"`,
   `prefix="$"`, and examples that tell prose to repeat closed units. Patch the
   rule files first so agents see one path.
2. **Allowed formatter matrix.** Confirm the formatter surface in `fmt.py` and
   tell agents to use only these value-kind routes:
   - physical quantity: `fmt_qty` or domain formatter;
   - memory: `fmt_memory`;
   - bandwidth: `fmt_bandwidth`;
   - FLOP throughput: `fmt_flop_rate`;
   - FLOP throughput per watt: `fmt_compute_efficiency`;
   - power: `fmt_power`;
   - energy: `fmt_energy`;
   - emissions/carbon mass: `fmt_emissions`;
   - grid carbon intensity: `fmt_carbon_intensity`;
   - water volume/rates/intensity: `fmt_water`, `fmt_water_rate`,
     `fmt_water_intensity`;
   - latency/duration: `fmt_latency` or `fmt_time`;
   - currency: `fmt_usd`;
   - percent/share: `fmt_percent`;
   - percentage-point delta: `fmt_pp`;
   - multiplier: `fmt_multiple` plus prose `$\times$` under current policy;
   - parameters/tokens: `fmt_params` / `fmt_tokens`;
   - other counts/requests: `fmt_count`;
   - non-physical event rates: `fmt_rate`;
   - dimensionless ratios: `fmt_ratio`;
   - equations/scientific math: `fmt_math(sci_latex(...))`.
3. **Unit alias policy.** Prefer exported unit aliases from `mlsysim`
   (`J`, `kWh`, `MW`, `kg`, `GB`, `L`, `second`, etc.) over `ureg.*` in QMD files,
   unless a unit has no exported alias yet. If a missing alias appears in more
   than one chapter, the coordinator adds it centrally instead of letting agents
   use mixed styles.
4. **Domain gap list.** Run a dry inventory for repeated patterns not covered
   by current helpers, such as `TFLOP/s/W`, `J/token`, `kWh/token`, `g/kWh`,
   `tokens/s`, and parameter/token scaled counts. Decide whether each remains
   `fmt_qty`/`fmt_rate` or needs a new domain helper before broad migration.
   Current central helpers cover FLOP rates, compute efficiency, parameters,
   tokens, carbon intensity, and water display (`L`, `L/h`, `L/day`,
   `L/kWh`). Event throughputs such as `tokens/s` or `QPS` remain `fmt_rate`.
5. **Source-of-truth queue.** Generate the initial list of chapter-local
   hardware/model/grid/fabric/literature values. Agents may classify and report
   these, but only the coordinator moves them into MLSysIM.
6. **Linter gap queue.** Any mistake found twice becomes a candidate lint rule
   before agents repeat the same review manually across 44 chapters.

Chapter agents start only after this pass is done or after the coordinator has
explicitly marked an item as deferred.

### 5.1 Coordinator responsibilities

The coordinator owns shared state and integration. It must:

- maintain the chapter ledger;
- assign disjoint QMD files to agents;
- run global audits before and after each wave;
- collect source-of-truth gaps from agents;
- batch MLSysIM registry/schema/helper changes;
- review every agent patch before accepting it;
- keep `PROGRESS.md` current;
- prevent shared-file edit conflicts.

Only the coordinator edits shared files such as:

- `mlsysim/**`;
- `book/tools/**`;
- `.claude/**`;
- `mlsysim-lego-unit-hardening-*.md`;
- global audit baselines.

### 5.2 Chapter-agent responsibilities

Each chapter agent owns exactly one QMD file at a time.

The agent may:

- inspect the assigned QMD;
- inspect relevant `mlsysim` APIs and rules;
- edit the assigned QMD only;
- produce a structured report.

The agent must not:

- edit `mlsysim`;
- edit shared lint/test/rule files;
- edit another chapter;
- commit;
- mark a source-of-truth gap as resolved without a registry path;
- suppress or delete a calculation because it is inconvenient.

### 5.3 No-deadlock protocol

When an agent finds a missing shared abstraction:

1. Finish all QMD-local work that does not depend on the abstraction.
2. Leave a clear TODO marker only in the agent report, not in prose.
3. Report the missing abstraction using the schema in section 10.
4. Continue to the next independent cell.

The coordinator later batches shared fixes and sends affected chapters back
through S1-S5.

This prevents a chapter from blocking the whole wave because one registry field
or formatter does not exist yet.

---

## 6. Chapter Execution Algorithm

For each chapter, run this sequence exactly.

### Step 1 - Inventory

Collect:

- all `{python}` cells;
- LEGO class names;
- `LOAD`, `EXECUTE`, `GUARD`, `OUTPUT` sections;
- every `_str`, `_math`, `_eq`, `_frac` export;
- every inline prose reference;
- every `fmt*` call;
- every `.to(...).magnitude` call;
- every local numeric literal with a physical or domain meaning;
- every `Q_("...")` or `number * unit` local quantity;
- every table/list/dict carrying numeric values.

### Step 2 - Classify each export

For each prose-facing export, classify:

- physical quantity;
- time/duration;
- memory/capacity/bandwidth;
- compute throughput;
- power/energy/carbon;
- cost/currency;
- percent/share;
- percentage-point delta;
- multiplier;
- count/parameters/tokens/requests;
- ratio;
- mathematical expression;
- label/sequence.

Then choose the formatter from `fmt.md` and `lego-units.md`.

### Step 3 - Fix OUTPUT and prose together

For each export:

1. Keep the computed value as a Quantity if it is physical.
2. Use the typed formatter.
3. Rename the export if closed/open semantics require it.
4. Update all prose references in the same chapter.
5. Re-read the surrounding sentence for duplicated units, missing units, and
   grammar after substitution.

### Step 4 - Audit calculations

For each calculation:

1. Check dimensional intent.
2. Preserve compound rates.
3. Convert to display units only at formatter/plot/table boundaries.
4. Add or update `check(...)` guards when an invariant is obvious.

### Step 5 - Audit LOAD

For each local input:

1. Ask whether changing the value globally should update this chapter.
2. If yes, it belongs in MLSysIM.
3. If no, keep it local but name it as a scenario assumption.
4. If uncertain, report it as a source-of-truth candidate.

### Step 6 - Verify

Run chapter-scoped checks where possible, then record results:

```bash
python3 book/tools/scripts/lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json
python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents
pytest mlsysim/tests/test_fmt.py mlsysim/tests/test_quantity_formulas.py mlsysim/tests/test_lego_unit_invariants.py book/tools/tests/test_lint_lego_units.py -o addopts=
```

Render verification comes after waves or whenever a chapter has nontrivial
prose/export edits.

---

## 7. Wave Plan

### Current execution snapshot - 2026-05-31 late-night pass

The broad pass is now audit-driven rather than render-driven. The coordinator
has already landed shared formatter/unit improvements and many chapter fixes.
Current advisory corpus counts from
`book_check_lego_quantity_flow.py book/quarto/contents --summary`:

| Scope | QF001 | QF002 | QF003 | QF004 | QF005 | ST001 |
|---|---:|---:|---:|---:|---:|---:|
| Default LEGO cells | 0 | 0 | 0 | 0 | 0 | 0 |
| All Python cells | 0 | 0 | 0 | 0 | 0 | 0 |

Current remaining default queue:

None. The default quantity-flow/source audit is clean.

Current all-cell-only queue after the default queue:

None. The all-cell quantity-flow/source audit is clean.

Completed coordinator/agent chapter passes already include
`vol1/data_engineering`, `vol1/hw_acceleration`, `vol1/ml_ops`,
`vol2/introduction`, `vol2/compute_infrastructure`,
`vol2/performance_engineering`, `vol2/backmatter/appendix_assumptions`,
`vol2/backmatter/appendix_c3`, `vol2/backmatter/appendix_dam`,
`vol2/backmatter/appendix_reliability`, `vol1/introduction`,
`vol2/backmatter/appendix_inference`, `vol2/conclusion`,
`vol2/data_storage`, `vol2/network_fabrics`, and `vol2/sustainable_ai`.

The final cleanup wave also promoted
`Scenarios.EnergyAnchors.USHouseholdAnnualElectricity` as the MLSysIM source of
truth for the GPT-3 household-year comparison, kept Little's Law latency as a
Pint duration until the request-count scalar boundary, and removed the last
all-cell scalar reattachment/quantity suffix advisories.

Completed source-discipline wave:

`book_check_registry_sources.py`, `book_check_lego_prose_literals.py`,
global LEGO lint, global prose-units, and the quantity-flow audit are clean.
`book_check_lego_load_pint.py` is now also clean across all 81 QMD files.
That wave burned down 234 legacy `*_value` bare-scalar assignments across eight
Vol I files by converting true physical values to Pint quantities or registry
paths, and by renaming dimensionless false positives so the audit no longer
confuses counts/ratios/percent values with physical measurements.

Completed `book_check_lego_load_pint.py` queue:

| File | Findings |
|---|---:|
| `vol1/model_serving/model_serving.qmd` | 118 -> 0 |
| `vol1/data_selection/data_selection.qmd` | 56 -> 0 |
| `vol1/ml_systems/ml_systems.qmd` | 17 -> 0 |
| `vol1/nn_computation/nn_computation.qmd` | 16 -> 0 |
| `vol1/backmatter/appendix_machine.qmd` | 13 -> 0 |
| `vol1/responsible_engr/responsible_engr.qmd` | 7 -> 0 |
| `vol1/hw_acceleration/hw_acceleration.qmd` | 5 -> 0 |
| `vol1/benchmarking/benchmarking.qmd` | 2 -> 0 |

Current verification state before Phase 9B/9C:

- `book_check_lego_quantity_flow.py book/quarto/contents --summary`: 0 findings.
- `book_check_lego_quantity_flow.py book/quarto/contents --summary --all-cells`: 0 findings.
- `book_check_lego_load_pint.py book/quarto/contents`: 81/81 OK.
- `book_check_registry_sources.py book/quarto/contents`: 81/81 OK.
- `book_check_lego_prose_literals.py book/quarto/contents`: 81/81 OK.
- `lint_lego_units.py --fail-on warning --baseline ...`: 0 new warnings.
- `book_check_lego_prose_units.py book/quarto/contents`: 81/81 OK.
- Headless execution: 44 QMD files / 1,099 Python cells OK.
- Focused pytest: 173 passed.

Completed chapter passes must be rechecked with both the default audit and
`--all-cells`. A chapter is considered locally clean only when these pass:

```bash
python3 book/tools/audit/book_check_lego_quantity_flow.py <qmd> --all-cells
python3 book/tools/scripts/lint_lego_units.py <qmd> --fail-on warning --baseline book/tools/audit/lego_units_baseline.json
python3 book/tools/audit/book_check_lego_prose_units.py <qmd>
git diff --check -- <qmd>
PYTHONPATH=mlsysim MPLBACKEND=Agg python3 book/tools/audit/fmt/audit_prose.py <qmd> --json
PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose_semantics.py <qmd>
```

The coordinator may launch workers on disjoint QMD files from this queue, but
workers may edit only their assigned QMD. Shared MLSysIM, audit, lint, test,
and plan files remain coordinator-owned. If a worker finds a missing source of
truth or helper, it must finish local independent fixes and report the shared
gap; it must not patch shared code.

### Wave 0 - Coordinator baseline

Before assigning agents:

1. Produce a fresh inventory of all LEGO QMDs.
2. Produce a current L014 baseline burn-down list.
3. Produce a local literal/source-candidate list.
4. Produce a magnitude-boundary list.
5. Produce a `fmt_qty` scalar misuse list.
6. Produce a prose duplicate-unit list.

The baseline lists become the coordinator ledger. They should be stored as
working artifacts under `book/tools/audit/artifacts/` only if they are useful,
and they should not be committed unless deliberately promoted.

### Wave 1 - Pilot chapters

Use chapters that exercise the hardest unit domains:

| Chapter | Why |
|---|---|
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd` | MWh/kWh, carbon, emissions, energy, grid intensity |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd` | H100 specs, TFLOP/s, bandwidth, power, rate per watt |
| `book/quarto/contents/vol1/training/training.qmd` | throughput, batch/token counts, cost/time patterns |
| `book/quarto/contents/vol2/inference/inference.qmd` | TTFT, tokens, latency, serving rates |

Do these first to validate the rules before broad parallelization.

### Wave 2 - Vol I content chapters

Process in book order:

1. `vol1/introduction/introduction.qmd`
2. `vol1/ml_systems/ml_systems.qmd`
3. `vol1/ml_workflow/ml_workflow.qmd`
4. `vol1/data_engineering/data_engineering.qmd`
5. `vol1/nn_computation/nn_computation.qmd`
6. `vol1/nn_architectures/nn_architectures.qmd`
7. `vol1/frameworks/frameworks.qmd`
8. `vol1/training/training.qmd`
9. `vol1/data_selection/data_selection.qmd`
10. `vol1/model_compression/model_compression.qmd`
11. `vol1/hw_acceleration/hw_acceleration.qmd`
12. `vol1/benchmarking/benchmarking.qmd`
13. `vol1/model_serving/model_serving.qmd`
14. `vol1/ml_ops/ml_ops.qmd`
15. `vol1/responsible_engr/responsible_engr.qmd`
16. `vol1/conclusion/conclusion.qmd`

### Wave 3 - Vol II content chapters

Process in book order:

1. `vol2/introduction/introduction.qmd`
2. `vol2/compute_infrastructure/compute_infrastructure.qmd`
3. `vol2/network_fabrics/network_fabrics.qmd`
4. `vol2/data_storage/data_storage.qmd`
5. `vol2/distributed_training/distributed_training.qmd`
6. `vol2/collective_communication/collective_communication.qmd`
7. `vol2/fault_tolerance/fault_tolerance.qmd`
8. `vol2/fleet_orchestration/fleet_orchestration.qmd`
9. `vol2/performance_engineering/performance_engineering.qmd`
10. `vol2/inference/inference.qmd`
11. `vol2/edge_intelligence/edge_intelligence.qmd`
12. `vol2/ops_scale/ops_scale.qmd`
13. `vol2/security_privacy/security_privacy.qmd`
14. `vol2/robust_ai/robust_ai.qmd`
15. `vol2/sustainable_ai/sustainable_ai.qmd`
16. `vol2/responsible_ai/responsible_ai.qmd`
17. `vol2/conclusion/conclusion.qmd`

### Wave 4 - Appendices with LEGO

Process after content chapters because appendices often reuse patterns surfaced
in the main text.

Vol I:

1. `vol1/appendix_data/appendix_data.qmd`
2. `vol1/appendix_algorithm/appendix_algorithm.qmd`
3. `vol1/appendix_machine/appendix_machine.qmd`
4. `vol1/appendix_assumptions/appendix_assumptions.qmd`

Vol II:

1. `vol2/appendix_dam/appendix_dam.qmd`
2. `vol2/appendix_c3/appendix_c3.qmd`
3. `vol2/appendix_fleet/appendix_fleet.qmd`
4. `vol2/appendix_communication/appendix_communication.qmd`
5. `vol2/appendix_reliability/appendix_reliability.qmd`
6. `vol2/appendix_inference/appendix_inference.qmd`
7. `vol2/appendix_assumptions/appendix_assumptions.qmd`

### Wave 5 - Global hardening

After all chapters reach S5:

1. Burn down or justify every L014 baseline entry.
2. Promote recurring source gaps into MLSysIM.
3. Add tests for new units, registries, formatters, and formulas.
4. Tighten lint gates.
5. Run full renders and merge gates.

---

## 8. Audit Commands

These commands are starting points. The coordinator may replace them with
better scripts, but the same classes must be covered.

### 8.1 Existing unit and formatter gates

```bash
python3 book/tools/scripts/lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json
python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents
pytest mlsysim/tests/test_fmt.py mlsysim/tests/test_quantity_formulas.py mlsysim/tests/test_lego_unit_invariants.py book/tools/tests/test_lint_lego_units.py -o addopts=
```

### 8.2 Magnitude extraction inventory

```bash
rg -n '\.to\([^)]*\)\.magnitude|\.m_as\(' book/quarto/contents --glob '*.qmd'
```

### 8.3 Suspicious scalar-to-quantity formatting

```bash
rg -n 'fmt_qty\([^,\n]*(\.magnitude|_gb|_gbs|_mb|_mib|_ms|_s|_w|_kw|_mw|_j|_kwh|_mwh)' book/quarto/contents --glob '*.qmd'
```

### 8.4 Manual unit suffixes in OUTPUT

```bash
rg -n 'fmt\([^#\n]*(suffix|prefix|unit=)|MarkdownStr\(f?["'\''].*(GB|GiB|MB|TB|ms|s|W|kW|MW|J|kWh|MWh|kg|g|CO|FLOP|token|parameter|request|percent|%)' book/quarto/contents --glob '*.qmd'
```

### 8.5 Local source-of-truth candidates

```bash
rg -n '=\s*[0-9][0-9_.,]*(e[+-]?[0-9]+)?\s*(\*|/)?\s*(GB|GiB|MB|MiB|TB|ms|second|minute|hour|watt|W|kW|MW|joule|J|kWh|MWh|TFLOP|FLOP|token|param|parameter|kg|gram|dollar|USD|Q_\()' book/quarto/contents --glob '*.qmd'
```

### 8.6 Possible duplicated units in prose

```bash
python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents
```

### 8.7 Full final gates

```bash
pre-commit run --all-files
pytest mlsysim/ book/tools/tests/test_lint_lego_units.py -o addopts=
./book/binder check lego-units
```

Render gates remain as listed in `mlsysim-lego-unit-hardening-CODEX-HANDOFF.md`.

---

## 9. Formatter Policy Decisions To Resolve Deliberately

These are not blockers for chapter-local cleanup, but they should be decided by
the coordinator before final gate tightening.

### 9.1 `fmt_multiple` glyph ownership

Current policy from `fmt.md`: `fmt_multiple(3.2)` renders `3.2`, and prose adds
`$\times$`.

Concern: this splits ownership across OUTPUT and prose. A stronger policy may
be to make a closed multiplier formatter, for example:

```python
speedup_str = fmt_multiple(3.2, style="closed")
```

which renders `3.2$\times$` or the agreed math-safe equivalent. If adopted,
the change must be corpus-wide and paired with a prose-contract lint update.

### 9.2 Pint pretty formatting

Pint supports format specifications such as pretty and LaTeX forms. The book
should not blindly switch to Pint's native prose formatting because MLSysBook
has stricter editorial contracts than Pint: MIT prose style, glyph ownership,
domain labels, compact units, carbon notation, tokens, parameters, and rendered
Quarto behavior.

Recommended stance:

- keep Pint as the unit algebra and conversion authority;
- keep `fmt_qty` and domain formatters as the book's rendering authority;
- selectively use Pint formatting internally only if it reduces formatter code
  without weakening the book's prose contract.

### 9.3 Count scaling

Counts such as parameters and tokens are dimensionless in Pint but semantic in
the book. Do not rely on Pint alone to choose `K`, `M`, `B`, or `T`.

Policy should live in `fmt_count` or domain helpers:

- forced `scale="B"` must not silently render `150M` as `0B`;
- automatic scale should choose a readable scale by magnitude;
- precision should default safely, not hide nonzero values.

### 9.4 Domain-specific rate units

FLOP/s, TFLOP/s/W, tokens/s, requests/s, J/token, kWh/token, g/kWh, and
MWh/year should have tested domain helpers when the pattern appears repeatedly.

Do not encode these as local suffixes. Use Pint compound units plus formatter
policy.

---

## 10. Agent Report Schema

Every chapter agent returns this report.

```markdown
## Chapter Report: <path>

### Summary
- State reached: S0/S1/S2/S3/S4/S5
- Files edited: <assigned qmd only>
- Visible prose changed: yes/no

### LEGO cells inspected
| Class | Lines | Exports | Status | Notes |
|---|---:|---|---|---|

### OUTPUT fixes
| Export | Old pattern | New pattern | Prose updated |
|---|---|---|---|

### Quantity-flow findings
| Site | Classification | Action |
|---|---|---|

### Source-of-truth findings
| Value | Current location | Recommended MLSysIM home | Blocking? |
|---|---|---|---|

### Shared-abstraction requests
| Need | Proposed home | Chapters affected |
|---|---|---|

### Verification
- lint_lego_units: pass/fail/not run
- prose_units: pass/fail/not run
- headless exec/render: pass/fail/not run
- tests: pass/fail/not run

### Residual risk
- <short bullets>
```

---

## 11. Chapter-Agent Prompt Template

Use this prompt when spawning chapter agents.

```text
You are auditing and hardening one MLSysBook QMD chapter for LEGO unit
discipline.

Worktree: /Users/VJ/GitHub/MLSysBook-fmt-fix
Branch: fmt-fix
Assigned file: <QMD path>

Read:
- mlsysim-lego-unit-hardening-agent-execution-plan.md
- .claude/rules/lego-units.md
- .claude/rules/fmt.md
- .claude/rules/mlsysim.md

Your job:
1. Move the assigned chapter through S0-S5 as far as possible.
2. Edit only the assigned QMD file.
3. Keep Pint quantities attached through calculations.
4. Use typed/domain formatters in OUTPUT.
5. Keep real specs sourced from MLSysIM when already available.
6. Report missing MLSysIM registry/helper/formatter needs instead of editing
   shared files.
7. Prefer exported unit aliases from mlsysim over ureg.* in QMD code.
8. Do not introduce .m_as(), fmt(..., suffix=physical-unit), prefix="$",
   suffix="%", or scale suffixes.
9. Do not commit.

Return the Agent Report Schema from section 10.
```

---

## 12. Coordinator Acceptance Criteria

The work is not complete until all are true:

- Every LEGO QMD has an S5 chapter report.
- Every L014 baseline entry is either removed or explicitly justified.
- No new `.m_as(` exists in QMD LEGO cells.
- No physical `fmt_qty` call receives a scalar magnitude.
- No avoidable magnitude extraction is followed by unit reattachment.
- Real specs and reusable anchors come from MLSysIM.
- Remaining local literals are scenario assumptions and documented in LOAD.
- Sustainable AI and Responsible Engineering energy/carbon units are checked
  explicitly, including MWh, kWh, g/kWh, kg, metric tons, and CO2e prose.
- Token, parameter, request, and count outputs use `fmt_count` or a domain
  helper rather than ad hoc scale suffixes.
- `fmt_multiple` policy is either kept as-is with a clean prose contract, or
  migrated corpus-wide to a closed formatter.
- Static gates pass.
- Headless execution passes.
- HTML/PDF render gates pass.
- `PROGRESS.md` records the final state.

---

## 13. Immediate Overnight Execution Order

1. Run Wave 0 audits and create the coordinator ledger.
2. Run Wave 1 pilot chapters manually under coordinator review.
3. Decide any shared formatter/helper additions surfaced by Wave 1.
4. Spawn chapter agents for Wave 2 in small batches.
5. Integrate Wave 2 reports and patches; run gates.
6. Spawn chapter agents for Wave 3 in small batches.
7. Integrate Wave 3 reports and patches; run gates.
8. Process Wave 4 appendices.
9. Burn down L014 and source-of-truth queues.
10. Add or tighten lint/tests for mistakes found during the pass.
11. Run full verification, including render gates.
12. Update `PROGRESS.md` and handoff docs.

The coordinator should not stop because a chapter has a missing registry field
or helper. Record it, continue independent work, then batch shared fixes.

---

## 14. Improvement Notice Policy

The user has explicitly authorized maintainability improvements when they are
clearly the right long-term approach for the book. The operating rule is:

1. Announce the improvement briefly before making it.
2. Make the improvement if it supports the single source-of-truth doctrine,
   reduces repeated local logic, or adds a durable guardrail.
3. Keep the scope tight.
4. Add tests or lint when the improvement changes shared behavior.
5. Record the reason in `PROGRESS.md`.

Examples that qualify:

- moving a repeated local rate calculation into a tested helper;
- adding a missing unit alias used by multiple chapters;
- adding a domain formatter for repeated token/energy/carbon output;
- converting a repeated source literal into an MLSysIM registry field with
  provenance;
- adding a lint rule for a mistake found twice.

Examples that do not qualify:

- stylistic rewrites unrelated to units;
- broad registry refactors not required by current chapters;
- changing visible prose merely because it could be worded differently;
- replacing a working policy with a new one without corpus-wide migration.

---

## 15. Pre-launch Audit Snapshot

**Run date:** 2026-05-31

This snapshot was taken before launching chapter agents. It tells the
coordinator what to settle centrally and what to let agents classify chapter by
chapter.

### 15.1 Current green gates

These are not the main risk right now:

```bash
python3 book/tools/scripts/lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json --format json
# []

python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents
# OK LEGO prose units (81 QMD files checked)

rg -n '\.m_as\(' book/quarto/contents --glob '*.qmd'
# 0 matches
```

Interpretation: the existing hard gates are useful, but they do not prove the
corpus is source-of-truth clean or quantity-flow clean.

### 15.2 Active queues

| Queue | Current size | Meaning |
|---|---:|---|
| L014 baseline | 81 | Honest burn-down queue for closed-name `fmt(...)` debt. Not "done." |
| `fmt_qty(<scalar> * unit, ...)` candidates | 498 | Many are likely scalar reattachment or local value/unit splits. Agents must classify; coordinator promotes repeated fixes. |
| `ureg.*` in QMD | 38 advisory alias fallbacks | Mostly consistency debt. Prefer exported aliases; add aliases centrally if missing. |
| domain formatter calls | ~120 | Good start, but uneven rollout. Most are in `sustainable_ai`; use pilots to decide broader replacement policy. |
| `fmt_count(... scale="B", precision=0 ...)` candidates | 49 | Prefer `fmt_params` / `fmt_tokens` for model/tokens; review other count scales. |
| formatted string reused in arithmetic | 1 found, 1 fixed | QF007 catches `fmt(...)` output converted back to `float(...)`; keep numeric/Quantity state separate from prose strings. |

### 15.3 Pre-launch improvements already made

The rule files in `/Users/VJ/GitHub/AIConfigs/projects/MLSysBook/.claude`
had conflicting older guidance. They now point agents toward the current
hardening policy:

- `qmd-patterns.md`: added the 2026-05 unit-hardening override, replaced
  `.m_as()` and suffix-based examples with `fmt_qty`/domain formatter examples,
  and updated imports to `from mlsysim import *`.
- `numbers-and-math-in-prose.md`: updated LEGO examples from legacy `mlsys`
  imports to the `mlsysim` public API.
- `math.md`: added a prominent override that `fmt.md` and `lego-units.md` win
  for unit work, and changed helper tables away from physical/percent suffixes.
- `fmt.md`: changed quantity examples to prefer exported aliases (`kW`, `mJ`)
  and structured denominator `per=`.
- `lego-units.md`: clarified that `fmt_params`, `fmt_tokens`,
  `fmt_flop_rate`, `fmt_compute_efficiency`, and `fmt_carbon_intensity` are now
  available for the repeated domains found in pilot chapters.

### 15.4 Coordinator decisions before broad launch

1. **Use the new domain wrappers when touching matching cells.** The pilot
   audits showed enough repeated policy to add `fmt_params`, `fmt_tokens`,
   `fmt_flop_rate`, `fmt_compute_efficiency`, and `fmt_carbon_intensity`.
   Agents should not hand-roll these with `fmt(...)` or scalar division.
2. **Do not make agents choose between `fmt_qty` and domain helpers freely.**
   Agents classify; the coordinator decides when a chapter-wide pattern should
   become `fmt_power`, `fmt_energy`, `fmt_bandwidth`, `fmt_memory`,
   `fmt_emissions`, or `fmt_latency`.
3. **Treat scalar reattachment as the main overnight queue.** A site like
   `fmt_qty(foo_gb * GB, GB, ...)` may be harmless if `foo_gb` is an intentional
   table scalar, but it is suspicious if the upstream value was already a Pint
   quantity. Agents must report the lineage, not just rewrite the line.
4. **Use the advisory quantity-flow checker before and after chapter edits.**
   `book_check_lego_quantity_flow.py` reports QF001 scalar reattachment, QF002
   unit-suffixed scalar boundaries, QF004 alias fallbacks, QF005 count-scale
   boilerplate, QF007 formatted strings reused as floats, and ST001 source
   candidates. It is a work queue, not yet a blocking gate.
5. **Keep render work secondary until source discipline is clear.** Renders
   still matter, but the hard problem is source-of-truth and quantity-flow
   consistency.
