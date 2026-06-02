# MLSysIM Full Audit, Consumer Update, and Render Verification Plan

**Status:** Execution and audit record. Commit in logical chunks; do not push without final sign-off.
**Working tree:** `/Users/VJ/GitHub/MLSysBook-fmt-fix` on branch `fmt-fix`
**Created:** 2026-06-01
**Primary objective:** Make every MLSysIM-backed value, every LEGO cell, every corresponding prose reference, and every downstream consumer use a clean single-source-of-truth path, then verify the book with full HTML and PDF renders before any push.

This plan supersedes ad hoc cleanup ordering for the remaining work. It does not replace the narrower unit-hardening and SSOT plans; it coordinates them into one execution path.

---

## Latest Validation Checkpoint - 2026-06-01

The current pass established these additional invariants:

- Full focal LEGO verification is clean across all LEGO-bearing QMDs: every cell executed by `lego_focal_verify.py` reports `cross=0` and `issues=0`.
- QMD code no longer uses `unit_label=`. Non-Pint display-label policies now belong in typed helpers such as `fmt_memory_capacity`, `fmt_sci_flops`, `fmt_decibel`, `fmt_illuminance`, and temperature helpers.
- Shared values needed by more than one LEGO cell must move to MLSysIM. Example fixed in this pass: `Scenarios.ClinicalImaging.RetinalPhotoSize` replaces a cross-cell dependency between `BandwidthCompute` and `DeploymentEconomics`.
- The static LEGO gates passed after the cleanup:
  - `book_check_lego_quantity_flow.py`
  - `book_check_lego_load_pint.py`
  - `book_check_lego_prose_units.py`
  - `lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json`
- MLSysIM provenance audit passed with `--scope all --strict`.
- Focused tests passed: `mlsysim/tests/test_fmt.py`, `test_ops_registry.py`, `test_system_registry.py`, and `test_units_registry.py` (`178 passed`).

Remaining before final push/merge readiness:

- commit completed MLSysIM, LEGO, docs, slides, and rule updates in logical chunks;
- merge local `dev` into `fmt-fix` and resolve conflicts from a clean branch baseline;
- rerun LEGO/prose/static checks after the merge;
- render full HTML/PDF after the merge;
- push only after final sign-off.

---

## 0. Current State and Guardrails

### 0.1 Commit and push policy

The user has approved committing completed work before merging local `dev` into
`fmt-fix`. Commit only coherent chunks, stage explicit paths, and keep unrelated
dirty files out of the commit.

Never push to `origin` until all acceptance gates in this document are green and the user signs off.

### 0.2 Worktree policy

Work only in:

```bash
/Users/VJ/GitHub/MLSysBook-fmt-fix
```

The protected main checkout is:

```bash
/Users/VJ/GitHub/MLSysBook
```

Use the main checkout only as the local `dev` reference and merge source. Do not make risky edits there. Before editing, committing, merging, or retiring anything, verify:

```bash
pwd
git rev-parse --show-toplevel
git branch --show-current
git worktree list
```

### 0.3 Historical dirty files at plan creation

These files were the initial dirty state when this plan was drafted. Do not use
this list as current status; use `git status --short` before any commit or
merge.

```text
M  book/tools/audit/fmt/audit_fmt_usage.py
M  book/tools/audit/fmt/fmt_prose_contract.py
M  mlsysim/mlsysim/core/loader.py
M  mlsysim/mlsysim/core/provenance_catalog.py
M  mlsysim/mlsysim/literature/data/batchsize.yaml
```

Original interpretation:

- `book/tools/audit/fmt/audit_fmt_usage.py` and `book/tools/audit/fmt/fmt_prose_contract.py` began as fmt-thread WIP and are now part of the semantic formatting/audit hardening once validated.
- `mlsysim/mlsysim/core/loader.py`, `mlsysim/mlsysim/core/provenance_catalog.py`, and `mlsysim/mlsysim/literature/data/batchsize.yaml` are drafted taxonomy/provenance hardening edits. They should be validated before keeping:
  - `Literature.BatchSize` now has explicit provenance.
  - sourced registries reject bare scalars.
  - `MCCANDLISH_LARGE_BATCH_TRAINING` was added to `provenance_catalog.py`.

### 0.4 Merge timing

Do not merge `dev` first. The current branch already contains major local unit/prose/registry work. Finish the source-of-truth cleanup and consumer updates on `fmt-fix`, then merge `dev` once the branch has a clean internal baseline. After the merge, rerun every LEGO/prose/render gate because conflicts can reintroduce stale values or prose.

---

## 1. Target Standard

### 1.1 MLSysIM is the single source of truth

Every reusable measurable fact must live in the correct MLSysIM semantic registry:

- hardware specs in `Hardware.*`
- model specs in `Models.*`
- infrastructure facts in `Infrastructure.*`
- systems and composed objects in `Systems.*`
- operational policies, thresholds, and run-overhead profiles in `Ops.*`
- scenario bundles or comparison anchors in `Scenarios.*`
- directly literature-sourced field figures in `Literature.*`

Book use is not a category. Do not encode chapter names, `BOOK_*`, `MLSysBook`, `Volume I`, `Volume II`, or "worked example" into MLSysIM registry names, provenance identifiers, or descriptions.

### 1.2 Provenance is metadata, not a semantic home

Keep the split:

- `core.provenance` defines the provenance data model and source contract.
- `core.provenance_catalog` stores reusable source records.
- semantic registries hold values.
- `Literature.*` is a semantic registry only for values whose category is a cited paper/report field figure.

Do not collapse `Provenance` and `Literature`. A value can be sourced from a paper while still belonging in `Hardware`, `Systems`, `Infrastructure`, or `Scenarios`.

### 1.3 LEGO stage contract

Every LEGO calculation should follow the same contract:

| Stage | Requirement |
|-------|-------------|
| L - Load | Pull reusable specs/scenarios from MLSysIM. Local literals are allowed only for truly local assumptions and must be labeled as scenario/illustrative/budget assumptions. |
| E - Execute | Keep Pint quantities attached. Use formulas/helpers where available. Avoid early `.magnitude` extraction except at explicit guard or output boundaries. |
| G - Guard | Add checks that make the prose hard to break: dimensions, representative values, ratios, and closed output expectations. |
| O - Output | Export typed strings only. Unit-bearing prose strings use domain formatters and names that assert the rendered unit. |

### 1.4 Output names are assertions

For fixed-unit output strings, the suffix must identify the value and rendered unit:

```python
facility_energy_kwh_str = fmt_energy(facility_energy, unit=kWh)
h100_tdp_w_str = fmt_power(h100.tdp, unit=watt)
peak_flops_tflop_s_str = fmt_flop_rate(peak_flops, unit=TFLOP / second)
params_b_str = fmt_params(params, scale="B")
decode_tokens_s_str = fmt_rate(decode_rate, "tokens/s")
```

If the formatter auto-scales, the name must not claim a fixed unit:

```python
facility_energy_str = fmt_energy(facility_energy)
```

Closed strings should be referenced bare in prose. Do not repeat the unit manually:

```markdown
`{python} CarbonEstimate.total_tonnes_str`
```

not:

```markdown
`{python} CarbonEstimate.total_tonnes_str` t CO2
```

### 1.5 Domain formatter policy

Anything unit-bearing should use a domain formatter where one exists or should get one if the domain recurs:

- memory: `fmt_memory`
- bandwidth: `fmt_bandwidth`
- energy: `fmt_energy`
- power: `fmt_power`
- emissions: `fmt_emissions`
- carbon intensity: `fmt_carbon_intensity`
- latency/time durations: `fmt_time`, `fmt_latency`, or a clarified domain-specific helper
- FLOPs and FLOP rates: `fmt_flops`, `fmt_flop_rate`
- arithmetic intensity: `fmt_arithmetic_intensity`
- compute efficiency: `fmt_compute_efficiency`
- parameters: `fmt_params`
- tokens: `fmt_tokens`
- rates such as tokens/s, samples/s, QPS: `fmt_rate`
- multipliers/speedups/ratios: `fmt_multiple` or a new `fmt_ratio` if audit shows recurring ambiguity

Plain `fmt()` is acceptable for dimensionless scalars such as PUE, counts that do not need semantic suffixes, or local values whose semantics are clear. If a value has a unit or a recurring semantic type, prefer a typed formatter.

---

## 2. Phase A - Freeze and Inventory

Goal: know exactly what will be touched before changing more files.

### A1. Confirm branch and dirty state

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
pwd
git rev-parse --show-toplevel
git branch --show-current
git status --short
git log --oneline -5
```

Record dirty files in this plan or a progress note before edits.

### A2. Inventory all MLSysIM consumers

Build a complete file list for surfaces that may import or copy MLSysIM values:

```bash
rg -n "mlsysim|from mlsysim|import mlsysim|Hardware\\.|Models\\.|Systems\\.|Infrastructure\\.|Literature\\.|Scenarios\\." \
  book mlsysim slides labs .claude \
  --glob '!**/_build/**' \
  --glob '!**/.quarto/**' \
  --glob '!**/__pycache__/**'
```

Expected surfaces:

- `book/quarto/contents/vol1/**/*.qmd`
- `book/quarto/contents/vol2/**/*.qmd`
- `book/quarto/**` support scripts and rendered config helpers
- `slides/**`
- `labs/**`
- `mlsysim/docs/**`
- `mlsysim/tutorials/**` or examples, if present
- `mlsysim/mlsysim/**`
- `mlsysim/tests/**`
- `.claude/rules/**`
- `.claude/docs/**`

### A3. Inventory every LEGO cell

Generate a machine-readable list of every QMD with LEGO cells and every exported string:

```bash
python3 book/tools/audit/fmt/audit_lego_cells.py book/quarto/contents --json \
  > /private/tmp/mlsysbook-lego-cells.json
```

If the existing audit tool is incomplete or unstable, write a small read-only scanner first. The output must include:

- file
- class/cell name
- line number range
- imports
- `LOAD` bindings
- `EXECUTE` assignments
- `GUARD` checks
- `OUTPUT` assignments ending in `_str`, `_math`, `_val`, `_tbl`, etc.
- inline prose references to those outputs

Do not start broad rewrites until this inventory exists.

---

## 3. Phase B - MLSysIM Taxonomy and Provenance Hardening

Goal: make the source registry honest before updating prose and downstream consumers.

### B1. Validate the drafted batch-size provenance change

Current draft:

- `Literature.BatchSize` gets explicit `MCCANDLISH_LARGE_BATCH_TRAINING` provenance.
- `load_sourced_registry` rejects bare scalars.

Validation:

```bash
PYTHONPATH=mlsysim pytest mlsysim/tests -o addopts=
PYTHONPATH=mlsysim python -m mlsysim.tools.audit_provenance --scope all --strict
```

Decision:

- Keep if all sourced registries can comply with `{value, provenance}`.
- If failures show legitimate non-sourced registries are using the sourced loader, split the loader contract rather than weakening sourced registry requirements.

### B2. Audit `Literature.*` for non-literature values

Run an introspection script to list every `Literature` leaf with provenance kind and description.

Known suspected migrations:

| Current path | Issue | Target direction |
|--------------|-------|------------------|
| `Literature.Scaling.*` | mixed: some values are convention tiers, one is a cited/empirical 8192-GPU anchor | do not move to a vague `Systems.Scaling` bucket. Either derive scaling from `Systems.Fleet` + engine, or create explicit scale-profile records such as `Scenarios.TrainingScaleProfiles.*` for scenario assumptions and keep truly cited field measurements in `Literature.*` only if they are used as literature facts |
| `Literature.Overheads.*` | operational goodput budget fractions, not physical systems | `Ops.TrainingRunOverheads.*` or a named `Ops.TrainingRunProfiles.*` object if reused as an operational policy profile; if the values are one scenario only, put the complete bundle in `Scenarios.*` |
| `Literature.Energy.*` | architecture-class effective pJ/FLOP and per-byte movement hierarchy; these describe technology behavior, not literature as a category | `Hardware.Tech.*` when they are technology-class facts, preferably as typed quantities; use `Scenarios.*` only for explicit comparison bundles that are not asserted as hardware/technology facts |
| `Literature.Sustainability.TransatlanticRoundTripCo2Kg` | sustainability comparison anchor, not a literature domain | `Scenarios.EmissionsAnchors.*` or `Scenarios.SustainabilityAnchors.*`, parallel to existing `Scenarios.EnergyAnchors.*` |

The decision rule:

- If the value describes a part, generation, operation, memory movement, or process technology, put it under `Hardware.Tech.*`.
- If the value describes a composed physical/logical system, cluster, rack, node, fabric, topology, or storage path, put it under `Systems.*`.
- If the value describes an operational policy, maintenance/recovery budget, monitoring threshold, or goodput loss profile, put it under `Ops.*`.
- If the value is a reusable scenario or comparison bundle, put it under `Scenarios.*`.
- If the value is directly a cited field figure from a paper/report, keep it under `Literature.*`.
- If the value can be derived from a system configuration and an engine model, prefer deriving it instead of storing another scalar.

### B3. Create neutral semantic homes

Add the smallest set of neutral registries needed. Do not add book-specific names.

Candidate additions:

- `Ops.TrainingRunOverheads` or `Ops.TrainingRunProfiles` for reusable operational goodput-loss fractions such as pipeline bubbles, checkpointing, recovery, and maintenance. Do not put these under `Systems` unless they are fields of a concrete fleet/system profile.
- `Scenarios.TrainingScaleProfiles` only for explicit scenario assumptions such as "32-GPU near-linear tier" or "1024-GPU teaching tier." Prefer an engine-derived value when possible. Keep true paper/report measurements in `Literature` only when they are used as cited literature facts, not as generic defaults.
- `Hardware.Tech.EffectiveOpEnergy` / `Hardware.Tech.MovementEnergy` or equivalent typed technology-class homes for effective pJ/FLOP and pJ/byte hierarchy values. Avoid `Scenarios.ComputeEnergyHierarchy` unless the value is explicitly an illustrative comparison rather than a hardware/technology fact.
- `Scenarios.EmissionsAnchors` or `Scenarios.SustainabilityAnchors` for reusable emissions comparison anchors, parallel to `Scenarios.EnergyAnchors`.
- `Systems.Clusters` entries for reusable cluster configurations that currently exist as local "dummy" or "frontier" bundles in LEGO cells, but only when the object is a composed fleet/node/fabric system. Scenario bundles that include workload, grid, utilization, or amortization should live in `Scenarios.*` and reference `Systems.Clusters.*`.
- existing storage homes should be used before adding a new namespace:
  - `Hardware.Tech.Storage` for generic storage technology bandwidth/latency tiers
  - `Systems.StorageSubsystem`, `NodeStorageConfig`, and `CheckpointStoragePath` for composed storage systems
  - `Infrastructure.Pricing.*` for storage prices and billing rates
  Add a new `Systems.Storage` registry only if the audit shows repeated reusable storage subsystems that cannot cleanly live in the existing storage types.

Avoid alias shims if all consumers are in-repo and can be migrated atomically. If a compatibility alias is temporarily necessary, mark it deprecated and add a grep/audit rule that forbids new uses.

### B4. Add source/invariant checks backwards from mistakes

For every mistake found, add a check before broad migration or in the same commit:

- no bare scalars in sourced registries
- no `BOOK_*`, `prov:book`, `MLSysBook`, `Volume I`, `Volume II`, or "worked example" inside `mlsysim/mlsysim`
- no non-literature convention values under `Literature.*`, except explicit allowlist during migration
- no old migrated paths such as `Literature.Scaling`, `Literature.Overheads`, `Literature.Energy`, `Literature.Sustainability` across book/slides/labs/docs after migration
- no local hardware/model/infrastructure specs in LEGO `LOAD`
- no quantity-to-float-to-quantity reattachment patterns in LEGO cells
- no unit-bearing `_str` outputs using plain `fmt()` unless explicitly allowed
- no closed-output prose that repeats the unit
- no `_kwh_str`, `_gb_s_str`, `_tflop_s_str`, etc. names whose formatter does not force that unit

These checks should be pre-commit capable where fast. Slower whole-book audits can be binder or CI gates first.

---

## 4. Phase C - Rules Update Before Broad Editing

Goal: encode the lessons that should guide every future agent before broad mechanical work starts.

Update local `.claude/rules` only with durable rules, not one-off project status.

### C1. `.claude/rules/mlsysim.md`

Add or tighten:

- provenance is metadata, not a semantic home
- `Literature` is only for directly cited field figures
- conventions, estimates, scenarios, hardware facts, and system compositions must live in their semantic homes even when sourced from a paper
- sourced registries require provenance-bearing records
- no book-specific registry/provenance naming inside MLSysIM
- if a value is reusable across QMD/slides/labs/docs, it belongs in MLSysIM first

### C2. `.claude/rules/lego-units.md`

Add or tighten:

- output names are assertions of rendered unit/scale
- fixed-unit output names must force the same unit in the formatter
- auto-scaling output names must stay generic
- unit-bearing outputs should use domain formatters
- `fmt()` is only for dimensionless values or values intentionally formatted without a unit
- local literals in `LOAD` must be marked as scenario/illustrative/budget assumptions
- any recurring local scenario should be promoted to `Scenarios` or `Systems`

### C3. `.claude/rules/slides.md`

Add:

- slides must not copy stale constants from book prose
- if a slide uses a measurable book claim, source it from MLSysIM or from generated artifacts derived from MLSysIM
- slide examples should use the current semantic registry path, not migrated/deprecated paths

### C4. `.claude/rules/labs.md`

Add:

- labs should import current MLSysIM registries rather than retyping hardware/model/system values
- lab constants should follow the same semantic-home distinction as the book
- lab output numbers should use MLSysIM formatters or lab-specific wrappers built on them

### C5. `.claude/rules/margin-figures.md`

Defer detailed margin-figure rules until after the margin-figure audit, but add one durable placeholder:

- margin figures that encode quantitative values must either source those values from MLSysIM or be regenerated from MLSysIM-backed data.

---

## 5. Phase D - Whole-QMD LEGO and Prose Audit

Goal: every QMD cell and nearby prose follows the same source, quantity, guard, and output convention.

### D1. Process order

Audit in book order:

Vol I:

1. `introduction`
2. `ml_systems`
3. `ml_workflow`
4. `data_engineering`
5. `nn_computation`
6. `nn_architectures`
7. `frameworks`
8. `training`
9. `data_selection`
10. `model_compression`
11. `hw_acceleration`
12. `benchmarking`
13. `model_serving`
14. `ml_ops`
15. `responsible_engr`
16. `conclusion`
17. appendices with LEGO

Vol II:

1. `introduction`
2. `compute_infrastructure`
3. `network_fabrics`
4. `data_storage`
5. `distributed_training`
6. `collective_communication`
7. `fault_tolerance`
8. `fleet_orchestration`
9. `performance_engineering`
10. `inference`
11. `edge_intelligence`
12. `ops_scale`
13. `security_privacy`
14. `robust_ai`
15. `sustainable_ai`
16. `responsible_ai`
17. `conclusion`
18. appendices with LEGO

### D2. Per-chapter checklist

For each QMD:

1. List all LEGO cells and classes.
2. Confirm `LOAD` sources:
   - hardware/model/grid/system/storage/fabric/price facts come from MLSysIM
   - local literals are only local assumptions and are marked
   - repeated assumptions are promoted to `Scenarios` or `Systems`
3. Confirm `EXECUTE` discipline:
   - Pint quantities remain quantities until output
   - use `Q_`, units, or registry quantities consistently
   - parenthesize compound unit construction, e.g. `1.9 * (TB / second)`
   - avoid ambiguous `R.value * GB/second`; prefer `R.value * (GB / second)` when the source is scalar
4. Confirm `GUARD` checks:
   - unit/dimension checks for physical results
   - value checks for critical textbook numbers
   - ratio/multiple checks where prose depends on comparison
   - prose-facing expectations for fixed-unit output names
5. Confirm `OUTPUT` naming:
   - every unit-bearing output uses a domain formatter
   - fixed-unit names force the matching formatter unit
   - auto-scaled names stay generic
   - no stale manual suffix pattern remains
6. Confirm prose:
   - closed strings are referenced bare
   - prose does not duplicate units
   - approximate/equality spacing is typographically clean
   - equations avoid awkward display lines where inline prose is clearer
   - repeated table/figure references are intentional
   - single-sentence floating paragraphs are either merged, expanded, or justified
7. Confirm no old path usage after registry migration.
8. Run the chapter through the headless LEGO executor.

### D3. Specific known patterns to include

Include these in the audit backlog:

- `fmt(hours, ...)` used for durations such as chargeback hours should become `fmt_time` or a duration-specific helper if it needs unit validation. Fixed-unit names such as `_hours_str` should force a time unit.
- `fmt(ratio, ...)` for speedups/ratios should be reviewed. If ratios recur, use `fmt_multiple` for `x`-style outputs and consider `fmt_ratio` for unitless ratios that should not imply speedup.
- Avoid `7x $\times$ speedup`; a formatter that emits `7x` should not be followed by a manual `\times`.
- Decide whether `fmt_multiple` should own the LaTeX multiplication symbol for math contexts or whether prose should consistently use `7x` without extra symbols.
- `samples/s`, `tokens/s`, QPS, requests/s should use a semantic rate formatter rather than raw `fmt()` plus a manual suffix.
- QMD code should not pass `unit_label=`. Add or use a semantic formatter helper in `mlsysim.fmt` for every deliberate label policy.
- LEGO code should not reference another LEGO class in Python. Promote the shared value to `Hardware`, `Models`, `Systems`, `Infrastructure`, `Ops`, `Scenarios`, or a helper, then have both cells load from that source.
- Parenthesize compound units in code for readability: `(USD / GB)`, `(GB / second)`, `(TFLOP / second)`.
- Replace awkward display equations like `T_load = 10 GB / 32 GB/s approx 312.5 ms` with either a proper aligned equation block or an inline sentence, depending on layout.
- Review table/figure references that repeat in adjacent sentences and simplify where the second reference does not add clarity.
- Review generic "see @sec-glossary" prose. If the target is too broad or not useful in context, replace with a more specific reference or remove it.
- Review HTML/PDF conditional text blocks that duplicate the same sentence unless the format-specific split materially improves layout.

### D4. Parallelization model

Parallelize review by chapter only after Phase B and C are complete. Central MLSysIM changes stay with one owner.

Each chapter agent gets:

- this plan
- `.claude/rules/mlsysim.md`
- `.claude/rules/lego-units.md`
- the generated LEGO inventory for that chapter
- the current allowed formatter list
- explicit instruction not to modify MLSysIM

Each chapter agent returns:

- patch proposal or direct file edits
- list of MLSysIM values they think are missing
- list of formatter gaps
- list of prose/layout concerns
- local validation commands and outputs

The central owner then:

- adds MLSysIM registry entries
- adds/updates domain formatters
- resolves cross-chapter naming
- runs whole-book checks

---

## 6. Phase E - Update Non-QMD Consumers

Goal: no stale MLSysIM path, copied constant, or old teaching number survives outside book chapters.

### E1. Slides

Audit:

```bash
rg -n "Hardware\\.|Models\\.|Systems\\.|Infrastructure\\.|Literature\\.|Scenarios\\.|mlsysim|GB/s|TFLOP|kWh|MWh|PUE|H100|A100|BatchSize|Scaling|Overheads|Energy" slides
```

Fix:

- migrated registry paths
- copied hardware/model/system values that should come from generated MLSysIM data
- stale units or suffixes
- deck speaker notes that cite old values

Validate:

```bash
cd slides
make check
```

If full slide builds are too slow, build touched decks first, then run `make check` before sign-off.

### E2. Labs

Audit:

```bash
rg -n "Hardware\\.|Models\\.|Systems\\.|Infrastructure\\.|Literature\\.|Scenarios\\.|mlsysim|GB/s|TFLOP|kWh|MWh|PUE|H100|A100|BatchSize|Scaling|Overheads|Energy" labs mlsysim/mlsysim/labs
```

Fix:

- stale imports
- hardcoded specs
- lab-specific constants that should be promoted to MLSysIM
- output formatting that bypasses MLSysIM formatters where consistency matters

Validate:

```bash
pytest labs/tests/test_static.py -v
```

### E3. MLSysIM docs/site/examples

Audit:

```bash
rg -n "Hardware\\.|Models\\.|Systems\\.|Infrastructure\\.|Literature\\.|Scenarios\\.|mlsysim|BatchSize|Scaling|Overheads|Energy|BOOK_|prov:book|MLSysBook|Volume I|Volume II|textbook|worked example" \
  mlsysim/docs mlsysim/examples mlsysim/tutorials mlsysim/mlsysim \
  --glob '!**/__pycache__/**'
```

Fix:

- stale registry paths
- stale API docs
- examples that teach local constants instead of registry imports
- docs that describe `Literature` as a catch-all source domain

If generated API docs exist, regenerate them after code changes rather than hand-editing generated stale output.

### E4. Book support files and tools

Audit:

```bash
rg -n "Literature\\.Scaling|Literature\\.Overheads|Literature\\.Energy|Literature\\.Sustainability|BOOK_|prov:book|MLSysBook|Volume I|Volume II|worked example" \
  book mlsysim .claude \
  --glob '!book/quarto/_build/**'
```

Fix tools/tests so new checks are enforced by local validation.

---

## 7. Phase F - Validation Before Merging Dev

Goal: establish that `fmt-fix` is internally clean before conflict resolution.

Run:

```bash
PYTHONPATH=mlsysim pytest mlsysim/tests -o addopts=
PYTHONPATH=mlsysim python -m mlsysim.tools.audit_provenance --scope all --strict
python3 book/tools/audit/book_check_lego_quantity_flow.py book/quarto/contents --summary
python3 book/tools/audit/book_check_lego_load_pint.py book/quarto/contents
python3 book/tools/scripts/lint_lego_units.py --fail-on warning --baseline book/tools/audit/lego_units_baseline.json
python3 book/tools/audit/book_check_lego_scenario_inputs.py book/quarto/contents --summary
pre-commit run --all-files
```

Run headless LEGO execution across all QMD files. If the existing command is unstable, use the established audit harness from the unit-hardening effort and record the exact command in the progress note.

Expected result:

- no high-severity scenario-input findings
- no stale migrated registry paths
- no unit/prose duplication warnings except explicit baseline items
- all QMD LEGO cells execute

---

## 8. Phase G - Merge Dev Into `fmt-fix`

Goal: reconcile with current local `dev` after the branch is internally clean.

### G1. Inspect main checkout

```bash
cd /Users/VJ/GitHub/MLSysBook
git status --short
git branch --show-current
git pull --ff-only
```

If the main checkout is dirty, stop and report. Do not overwrite it.

### G2. Merge into task branch

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
git fetch origin dev
git merge --no-ff origin/dev
```

Conflict rules:

- prefer unit-hardened, Pint-safe LEGO cells
- preserve current MLSysIM semantic-home decisions
- preserve user/editorial changes from `dev` unless they reintroduce stale constants or broken prose
- do not resolve conflicts by deleting checks
- do not revive deprecated registry paths

### G3. Post-merge re-audit

After conflict resolution, rerun all Phase F checks. Treat every failure as a possible merge regression, not as expected noise.

---

## 9. Phase H - Full HTML and PDF Verification

Goal: build the real artifacts and prove rendered output is clean.

### H1. Environment

Use the task worktree and force the correct Jupyter kernel:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
python3 -c "import mlsysim; print(mlsysim.__file__)"
```

The printed path must point into `/Users/VJ/GitHub/MLSysBook-fmt-fix/mlsysim/...`.

If not:

```bash
pip install -e /Users/VJ/GitHub/MLSysBook-fmt-fix/mlsysim
```

Always render with:

```bash
-M jupyter:python3
```

### H2. Full HTML renders

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix/book/quarto

ln -sf config/_quarto-html-vol1.yml _quarto.yml
MPLBACKEND=Agg quarto render --to html -M jupyter:python3

ln -sf config/_quarto-html-vol2.yml _quarto.yml
MPLBACKEND=Agg quarto render --to html -M jupyter:python3
```

Post-checks:

```bash
rg '\\{python\\}' _build/html-vol1 _build/html-vol2 --glob '*.html'
python3 scripts/verify_rendered_xrefs.py
rg -n '\\?@|Traceback|ImportError|NameError' _build/html-vol1 _build/html-vol2 --glob '*.html'
```

Any raw `{python}` leak, unresolved `?@`, traceback, import error, or name error blocks sign-off.

### H3. Full PDF renders

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix/book/quarto

ln -sf config/_quarto-pdf-vol1.yml _quarto.yml
ln -sf index-vol1.qmd index.qmd
MPLBACKEND=Agg quarto render --to titlepage-pdf -M jupyter:python3

ln -sf config/_quarto-pdf-vol2.yml _quarto.yml
ln -sf index-vol2.qmd index.qmd
MPLBACKEND=Agg quarto render --to titlepage-pdf -M jupyter:python3
```

Do not use `--to pdf`; use `--to titlepage-pdf` so the PDF header includes are loaded.

Post-checks:

```bash
ls -lh _build/pdf-vol1/*.pdf _build/pdf-vol2/*.pdf
rg '^!|Undefined control sequence|LaTeX Error|Traceback|ImportError|NameError' index.log _build/pdf-vol1 _build/pdf-vol2
```

If Quarto saves TeX logs elsewhere, scan those logs too.

### H4. Rendered prose spot checks

Spot-check rendered HTML and PDF for:

- no duplicated units such as `W W`, `kWh kWh`, `t CO2 CO2`
- no awkward `7x x` or `7x \times` speedup phrasing
- no raw Pint reprs
- no fake unit labels after scalar conversion
- no unintentional display equations that should be prose
- no table/figure reference loops in adjacent sentences
- no stale glossary or section links

Spot checks must include:

- `sustainable_ai`
- `responsible_engr`
- `compute_infrastructure`
- `data_storage`
- `distributed_training`
- `model_serving`
- `benchmarking`
- any chapter touched by taxonomy migration

---

## 10. Phase I - Margin-Figure Audit Plan

Goal: after source/prose/rendering is stable and `dev` is merged, audit margin figures as a separate design/layout effort.

Do not start this before Phase H is green.

### I1. Inventory

List all margin figures, margin notes, and side graphics:

```bash
rg -n "marginfigure|marginnote|margin-|column-margin|aside|fig-margin|layout-ncol|layout-valign|includegraphics|\\.svg|\\.png" \
  book/quarto/contents \
  --glob '*.qmd'
```

Capture:

- file and line
- figure asset path
- page/section after PDF render
- whether it contains quantitative values
- whether values are sourced from MLSysIM or static drawing text

### I2. Automatic layout checks

Use LaTeX logs and PDF artifacts to find likely problems:

- overfull hbox/vbox warnings
- underfull warnings near margin content
- float placement warnings
- pages with dense footnotes and margin notes
- pages with multiple nearby side figures

Potential commands:

```bash
rg -n "Overfull|Underfull|marginpar|Float|too large|rerun|LaTeX Warning" book/quarto/*.log book/quarto/_build/pdf-vol*/**
```

This will not catch everything, but it provides a triage list.

### I3. Visual review

Review PDF spreads for:

- cramped margin notes
- overlapping margin figures and footnotes
- side graphics that fight main text
- labels too small to read
- diagrams whose style no longer matches the book
- quantitative labels not sourced from MLSysIM
- redundant figures where prose/table already carries the idea

### I4. Improvement candidates

Possible actions:

- move some dense footnotes into endnotes or prose
- move some margin notes earlier/later in the paragraph
- convert a marginal graphic into an inline figure when it needs inspection
- simplify a margin figure to one idea
- regenerate quantitative graphics from MLSysIM data
- add new margin figures only where they clarify a central systems idea

Do not treat margin-figure polish as part of numeric correctness. It is a separate final pass after the book is already correct.

---

## 11. Commit Plan After Review

When the user approves execution, commit in small stages. Do not use `git add -A`.

Candidate commit boundaries:

1. provenance/source-loader hardening
2. Literature taxonomy migration and consumer path updates
3. MLSysIM Systems/Scenarios additions for reusable cluster/storage/fleet assumptions
4. formatter/check improvements
5. Vol I QMD LEGO/prose cleanup
6. Vol II QMD LEGO/prose cleanup
7. slides/labs/docs consumer updates
8. `.claude/rules` updates
9. dev merge conflict resolution
10. render/preflight fixes

Each commit should include only files relevant to that stage. Keep unrelated fmt-thread WIP unstaged unless explicitly included.

---

## 12. Acceptance Criteria

Do not sign off until all are true:

- all reusable measurable values used by QMDs/slides/labs/docs come from MLSysIM
- no book-specific names remain inside `mlsysim/mlsysim`
- `Literature.*` contains only true literature/report field figures, or any remaining exceptions are documented and scheduled
- every sourced registry record has provenance
- every LEGO cell in Vol I and Vol II follows Load, Execute, Guard, Output discipline
- every unit-bearing output string uses a domain formatter or a documented exception
- fixed-unit output names force the matching formatter unit
- rendered prose does not duplicate units
- no stale migrated registry paths remain in book/slides/labs/docs
- all MLSysIM tests pass
- provenance audit passes strict mode
- LEGO quantity/load/lint/scenario audits pass
- headless LEGO execution passes all QMD files
- pre-commit passes
- `dev` has been merged into `fmt-fix`
- all checks pass again after the merge
- full Vol I and Vol II HTML render cleanly
- full Vol I and Vol II PDF render cleanly
- no raw `{python}`, unresolved `?@`, or render tracebacks remain
- PDF logs have no LaTeX errors
- margin-figure audit plan is ready to execute after correctness work
- no push has happened before final user sign-off

---

## 13. Immediate Next Step After Review

If this plan is approved, start with Phase A and B:

1. preserve current dirty-state notes
2. validate or adjust the drafted batch-size provenance/source-loader edits
3. generate the whole-repo MLSysIM consumer inventory
4. generate the LEGO-cell/output/prose inventory
5. update the durable `.claude/rules` before broad QMD edits

Only after those are stable should broad chapter-by-chapter work begin.
