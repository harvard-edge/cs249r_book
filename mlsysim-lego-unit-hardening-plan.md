# MLSysBook LEGO Unit Hardening Plan

This is a handoff plan for an implementation agent. The goal is to stop recurring LEGO-cell unit bugs by adding durable unit, formula, formatting, lint, and test guardrails around `mlsysim`, without redesigning the book's computation model.

---

## As-built status (2026-05-31, revised) — read this first

**Live tracker:** [`mlsysim-lego-unit-hardening-PROGRESS.md`](mlsysim-lego-unit-hardening-PROGRESS.md) — systematic checklist, gates, work queue.

**Current position:** Layer A/B/C + `.m_as()` migration **done**. **Phase 8½ gate hardening NOT done** — do not treat branch as merge-ready. Phase 9 renders **not started**.

**Branch / worktree (actual):** `/Users/VJ/GitHub/MLSysBook-fmt-fix` on `fmt-fix`.

### Checklist snapshot

| Phase | Status |
|-------|--------|
| Layer A (Steps 1–10) — mlsysim infra | **DONE** |
| Layer A′ — LOAD registry-first | **NOT DONE** (deferred) |
| Layer B (Steps 11–13) — lint + hooks wired | **DONE** |
| Layer C (Steps 14+) — `.m_as()` migration | **DONE** (bulk) |
| Phase 8½ — trustworthy gates + OUTPUT/prose cleanup | **IN PROGRESS** ← **YOU ARE HERE** |
| Phase 9A–9C — Quarto HTML/PDF renders | **NOT STARTED** |
| Phase 10 — sync dev, re-verify, promote | **NOT STARTED** |

### Merge-ready gates (all must pass)

| # | Gate | Status | Blocker |
|---|------|--------|---------|
| G1 | **L014 linter trustworthy** | **FAIL** | `lint_lego_units.py:144` checks `"= fmt("` after space-stripping → never matches `=fmt(`; ~85+ closed-name `fmt()` assignments undetected |
| G2 | **`lego-units` lint re-baselined** | **BLOCKED on G1** | Empty baseline is a false all-clear |
| G3 | **`book_check_lego_prose_units.py` clean** | **FAIL** | 17 files with duplicated units / math-span violations |
| G4 | **Rate quantities stay dimensional** | **PARTIAL** | e.g. `compute_infrastructure.qmd:1815` — TFLOP/s÷W reattached as TFLOP/s only |
| G5 | **fmt precision defaults ergonomic** | **OPEN** | `fmt_percent(0.85)`, `fmt_*_range(...)` default `precision=1` fights spurious-zero guard |
| G6 | **Headless cell exec** | **PASS** | 81/81 files |
| G7 | **Phase 9 renders green** | **NOT STARTED** | HTML/PDF per chapter + full volume |
| G8 | **No accidental artifact commits** | **WATCH** | `lego_cells_verify_report.json` unstaged partial regen — do not commit |

### Lint rollout (actual vs plan below)

| Plan said | Actual (2026-05-31) |
|-----------|---------------------|
| Separate feature branch → merge into `fmt-fix` | Work committed directly on `fmt-fix` |
| `default=True` at Phase 9 prep | **Done** at closure (`89c287556f`) |
| L014–L017 block after Phase 9 | L019 blocks `.m_as()` only; **L017 retired**; **L014 silently broken** |
| Baseline 0 warnings | **Not trustworthy** until G1 fixed and re-run |

### Codex review lessons (2026-05-31)

1. **Exec clean ≠ render clean** — sustainable_ai cells execute; render still has L015/math-span issues and unrelated xref failures.
2. **Two linters, one story** — `lint_lego_units.py` (L014–L015) and `book_check_lego_prose_units.py` must both be green.
3. **Closed formatter contract** — if `fmt_emissions`/`fmt_power` owns the unit, prose must not repeat it; if value goes in `$...$`, use math-safe atoms not `_str` exports.
4. **Quantity-first means through OUTPUT** — dividing TFLOP/s by W then storing TFLOP/s loses `/W`; use `(flops / tdp).to(TFLOP/second/watt)` or rename to open export.

### Next action (strict order)

See **Phase 8½** below and PROGRESS.md work queue. Do **not** start Phase 9A until G1–G3 are addressed (G4–G5 can overlap with Phase 9 pilot chapters).

The sections below are the **original spec**. Where they conflict with this box or PROGRESS.md, trust the tracker.

---

## Current Worktree and Branch Strategy

> **Note (2026-05-31):** This section describes the *planned* isolation model. Execution used `fmt-fix` directly — see as-built status above. Phase 10C is now: merge `fmt-fix` → `dev` after Phase 9 green.

### Integration target

Unit hardening is a **feature branch off `fmt-fix`**, not direct work on `fmt-fix`
and not direct promotion to `dev`. When the work is green, merge **into
`fmt-fix`**. The broader fmt thread (`fmt-fix` → `dev`) stays separate.

```text
fmt-fix  ──branch──►  feat/lego-unit-hardening  ──merge (10C)──►  fmt-fix  ──later──►  dev
```

### Where implementation runs

**Preferred:** a **dedicated worktree** so `MLSysBook-fmt-fix` can stay on
`fmt-fix` for any parallel fmt work.

| Role | Path | Branch |
|------|------|--------|
| Parent integration (fmt thread) | `/Users/VJ/GitHub/MLSysBook-fmt-fix` | `fmt-fix` (do not switch away casually) |
| **Unit hardening (this plan)** | `/Users/VJ/GitHub/MLSysBook-lego-units` (create) | `feat/lego-unit-hardening` |
| Main reference checkout | `/Users/VJ/GitHub/MLSysBook` | usually `dev` — **do not edit** |

**Acceptable alternative:** implement in `MLSysBook-fmt-fix` on
`feat/lego-unit-hardening` if no fmt WIP remains on that checkout — but a
separate worktree is cleaner for GitKraken and parallel work.

### Step 0 — Create branch and worktree (before Step 1)

Run from the `fmt-fix` tip (user or agent, once, at execution start):

```bash
# Ensure fmt-fix tip is current
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
git status --short
git branch --show-current   # expect fmt-fix

# Create feature branch at fmt-fix tip (if it does not exist)
git branch feat/lego-unit-hardening fmt-fix

# Dedicated worktree (recommended)
git worktree add /Users/VJ/GitHub/MLSysBook-lego-units feat/lego-unit-hardening

# All Steps 1–10C happen here
cd /Users/VJ/GitHub/MLSysBook-lego-units
git branch --show-current   # expect feat/lego-unit-hardening
```

**Rules:**

- Do **not** edit, delete, or retire `/Users/VJ/GitHub/MLSysBook`.
- Do **not** force-push `fmt-fix` or `dev`.
- All commits for this plan go to `feat/lego-unit-hardening`.
- Record worktree path and branch in PROGRESS.md Step 0/1.

**After Phase 10C (success):** merge `feat/lego-unit-hardening` → `fmt-fix`
with `--no-ff` from the `fmt-fix` checkout; optionally remove the worktree:

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
git merge --no-ff feat/lego-unit-hardening
git worktree remove /Users/VJ/GitHub/MLSysBook-lego-units   # when user confirms
git branch -d feat/lego-unit-hardening                       # after merge
```

### Agent capability

Yes — when execution is approved, the agent **can** create the branch and
worktree (Step 0), do all work in `MLSysBook-lego-units`, and leave
`MLSysBook-fmt-fix` on `fmt-fix` untouched.

### Related docs

- This document is planning only. The existing broader formatting plan is:
  - `mlsysim-domain-formatting-plan.md`
- **Live progress tracker (create at Step 1, update after every step):**
  - [`mlsysim-lego-unit-hardening-PROGRESS.md`](mlsysim-lego-unit-hardening-PROGRESS.md)

## Progress Ledger and Engineering Integrity

### Progress file (required)

Create [`mlsysim-lego-unit-hardening-PROGRESS.md`](mlsysim-lego-unit-hardening-PROGRESS.md) at **Step 1** before any
other implementation. Update it **after every completed step** — same commit
or immediately following commit — so the user can track status in GitKraken and
any agent can resume without rediscovery.

The progress file is the **single source of truth for "where we are."** The plan
doc is the spec; the progress file is the build log.

**Top of file — always-current checklist:**

> **Use [`mlsysim-lego-unit-hardening-PROGRESS.md`](mlsysim-lego-unit-hardening-PROGRESS.md)** for the live checklist. Template below is the original shape; do not maintain duplicate state here.

```markdown
## Step checklist (see PROGRESS.md for live state)
- [x] Steps 1–10 — Layer A mlsysim foundation
- [ ] Layer A′ — LOAD registry-first (deferred)
- [x] Steps 11–13 — lint + pre-commit wired
- [x] Steps 14+ — .m_as() migration (bulk)
- [x] Phase 8½-A — Fix L014 + re-baseline (G1–G2)
- [ ] Phase 8½-B — prose-units clean, 17 files (G3)
- [ ] Phase 8½-C — rate-quantity integrity audit (G4)
- [ ] Phase 8½-D — fmt precision defaults (G5)
- [ ] Phase 9A — HTML per chapter
- [ ] Phase 9B — PDF per chapter
- [ ] Phase 9C — Full volume builds
- [ ] Phase 10A — Merge dev → fmt-fix
- [ ] Phase 10B — Re-verify
- [ ] Phase 10C — Merge fmt-fix → dev

**Current step:** Phase 8½-B — prose-units pilot (`sustainable_ai.qmd`)
**Last commit:** *(pending — Phase 8½-A)*
**Next action:** Fix `sustainable_ai.qmd` prose-units findings; then remaining 16 files
```

**Per-step log entry (append after each finished step):**

```markdown
### Step N — YYYY-MM-DD — `<short commit subject>`

- **Commit:** `<full sha>`
- **Files:** list of paths touched
- **Gate:** pytest / pre-commit / render — pass/fail
- **Work done:** 2–5 bullets
- **Discoveries:** anything found (missing unit, formatter gap, prose drift)
- **Follow-ups:** registry additions, lint rule, deferred cells
- **Status:** DONE | BLOCKED (reason)
```

**For each Layer C cell, also append a row to a migration table:**

```markdown
| Step | Chapter | Class | Closed/open | Render OK | Commit |
|------|---------|-------|-------------|-----------|--------|
| 14 | vol1/introduction/introduction.qmd | ScenarioFoo | closed-fixed | yes | abc1234 |
```

Mirror the fmt migration pattern in
[`book/tools/audit/fmt/AUDIT_LEDGER.md`](book/tools/audit/fmt/AUDIT_LEDGER.md)
but scoped to **unit hardening** — do not mix fmt suffix cleanup entries here.

### Discovery workflow (no shortcuts)

When migration or tests reveal a gap, **stop the cell**, fix the foundation,
log it, then continue. Do not patch around missing infrastructure.

| Discovery | Proper fix (in order) | Do not |
|-----------|----------------------|--------|
| Missing Pint unit or alias | Add to `mlsysbook_units.txt` / `units.py` + test in `test_units_registry.py` | Hardcode in QMD; `ureg.define` only in chapter |
| Missing hardware/model spec | Add to MLSysIM registry with provenance | Chapter-local literal |
| Repeated formula | Add to `physics/quantities.py` + test | Copy-paste `.m_as` math in cell |
| Missing domain formatter policy | Add to `fmt.py` + golden test | Raw `fmt(..., suffix=" kWh")` |
| New failure class | Add lint rule + pre-commit scope + note in progress file | One-off cell exception |
| Pre-commit blocks legacy corpus | Log baseline; fix queue-ordered cells; promote hook only when ready | `--no-verify`; global allowlist without plan |

**Integrity rules (no reward hacking):**

1. **Do not mark a step DONE** until its gate actually passed (pytest, pre-commit,
   render — whichever the step requires).
2. **Do not skip a step** because a later step seems more interesting.
3. **Do not close a cell** without OUTPUT + prose aligned and noted in the progress file.
4. **Do not declare Phase 9 complete** after spot-checking — every chapter in
   manifest order must be built and checked.
5. **Do not merge `dev` early** (Phase 10 waits for Phase 9).
6. When a step exposes a plan gap, **update the progress file discovery section
   and the plan if the spec was wrong** — then implement the proper fix.

### Resume protocol

Any new session starts by reading, in order:

1. [`mlsysim-lego-unit-hardening-PROGRESS.md`](mlsysim-lego-unit-hardening-PROGRESS.md) — current step + checklist
2. [`mlsysim-lego-unit-hardening-plan.md`](mlsysim-lego-unit-hardening-plan.md) — spec
3. Last 3 entries in the step log

Then continue from **Next action** only.

## Core Diagnosis

The registry is already mostly unit-safe:

- `mlsysim/mlsysim/core/units.py` creates one Pint registry.
- `ureg.formatter.default_format = "~P"` is already set.
- `pint.set_application_registry(ureg)` makes the registry canonical for the package.
- Registry data such as `Hardware.Cloud.H100.memory.capacity`, `Hardware.Cloud.H100.memory.bandwidth`, and `Hardware.Cloud.H100.compute.peak_flops` enters through `Quantity` validation in `mlsysim/mlsysim/core/types.py`.
- `fmt_qty()` requires a Pint `Quantity` and converts through `.to(display_unit)`, so dimensional errors are caught before prose rendering.

The recurring failures are not mainly because Pint is missing. They come from gaps around Pint:

- LEGO cells often strip quantities to floats using `.m_as(...)`.
- Some later calculations reattach units manually.
- Generic formatters require repeated per-cell choices for unit, scale, precision, label, and suffix.
- Dimensionless quantities are ambiguous: `count`, `param`, `token`, `FLOP`, `request`, and `GPU-hour` can all look like scalars.
- Energy and carbon math uses a mix of raw floats, manual scale factors, and partially defined units.
- Formatting policy is book-specific: Pint can format units, but it does not know MLSysBook editorial policy.

The stable architecture should be:

```text
registry Quantity -> domain formula helper -> Quantity result -> domain formatter -> prose
```

The implementation should make this path easier than ad hoc conversions.

## Non-Goals

Do not do these in this unit-hardening pass:

- Do not rewrite all LEGO cells at once.
- Do not remove all `.m_as(...)` usage blindly. Some scalar extraction is legitimate at boundaries.
- Do not replace `fmt_qty`, `fmt_time`, `fmt_count`, `fmt_usd`, or `fmt_percent`.
- Do not install a global custom Pint formatter unless there is a narrowly proven need.
- Do not introduce Pint contexts for carbon accounting as the first implementation.
- Do not change numerical claims unless a test exposes a real bug.
- Do not combine this with the separate suffix-cleanup thread unless the user explicitly requests it.

## Guiding Principles

1. Keep quantities attached as long as possible.
2. Extract magnitudes only at display boundaries, charting boundaries, or intentionally scalar algorithm boundaries.
3. Put repeated equations in helper functions.
4. Put repeated display policy in domain formatters.
5. Make unit mistakes fail at calculation time when possible.
6. Make display mistakes fail at formatter time when possible.
7. Make recurring bad patterns fail in lint before rendering.
8. Add golden tests for the book's most common calculations and rendered strings.
9. Encode the **display unit** in export names (`_{unit_token}_str`) so the
   formatter contract and prose contract are visible at a glance — and enforce
   that contract in lint.

## Pint Capabilities To Use

Use Pint for these things:

- `UnitRegistry` as the single source of units.
- `Quantity` construction through `Q_`.
- `.to(...)` for unit conversion when the result remains a `Quantity`.
- `.m_as(...)` only when a plain scalar is intentionally needed.
- `.check(...)` for dimension validation inside formula helpers and formatters.
- `ureg.wraps(...)` for formula helpers where function signatures are stable enough.
- `~P` compact pretty formatting as the raw basis for unit labels.
- A Pint definition file for custom book units and aliases.

### Quantity Construction Cleanup Rule

This expression is correct Pint usage:

```python
kv_cache_bandwidth = 1.9 * TB / second
```

It creates a Pint `Quantity` with units `TB/s`. It is not a division-only scalar mistake, because `TB` and `second` are Pint unit objects, and multiplying a number by a Pint unit produces a Pint quantity.

For readability in LEGO cells, prefer one of these equivalent forms:

```python
kv_cache_bandwidth = 1.9 * (TB / second)
```

or:

```python
kv_cache_bandwidth = Q_("1.9 TB/s")
```

The parenthesized form makes it visually obvious that the unit is `TB/s`. The `Q_("1.9 TB/s")` form is also clear and is useful when the value is naturally read from prose, a registry, or a table.

Recommended LEGO pattern:

```python
kv_cache_bandwidth = 1.9 * (TB / second)
kv_cache_bandwidth_tbs = kv_cache_bandwidth.m_as(TB / second)
kv_cache_bandwidth_tbs_str = fmt_qty(
    kv_cache_bandwidth,
    TB / second,
    precision=1,
    commas=False,
)
```

The important rule is that `fmt_qty()` receives the original Pint quantity, not the scalar extracted with `.m_as(...)`.

Do not rely on Pint alone for these things:

- When to show `GB` vs `GiB`.
- When to switch `150M` to `0.15B` or keep `150M`.
- Whether `TFLOPs/s` should render as `TFLOP/s`.
- Whether carbon should render as `kg CO2e` or `tonnes CO2e`.
- Whether body prose should spell out time units.
- Whether table cells should use compact symbols.

Those are MLSysBook formatting policies, so they belong in `mlsysim.fmt` domain helpers.

## Phase 0: Baseline Before Editing

Before making implementation changes, the agent should record current state.

Run from `/Users/VJ/GitHub/MLSysBook-fmt-fix`:

```bash
pwd
git rev-parse --show-toplevel
git branch --show-current
git status --short
```

Expected:

- Top-level path is `/Users/VJ/GitHub/MLSysBook-fmt-fix`.
- Branch is `fmt-fix`.
- There may already be an untracked `mlsysim-domain-formatting-plan.md`.
- This new plan file may also be untracked.

Then inspect the relevant files:

```bash
sed -n '1,260p' mlsysim/mlsysim/core/units.py
sed -n '1,120p' mlsysim/mlsysim/core/types.py
sed -n '1,360p' mlsysim/mlsysim/fmt.py
sed -n '1080,1360p' mlsysim/mlsysim/fmt.py
```

Confirm these facts before implementation:

- `ureg = pint.UnitRegistry()` is in `core/units.py`.
- `Q_ = ureg.Quantity`.
- Custom units are currently created with `ureg.define(...)`.
- `fmt_qty()` rejects non-Pint quantities.
- `fmt_qty()` converts with `quantity.to(display_unit)`.
- `fmt_time()` validates that display units have time dimensions.
- `_compact_unit_suffix()` derives a suffix from Pint formatting.

## Phase 1: Add A Custom Pint Definition File

### Objective

Make MLSysBook custom units reviewable as a single, testable unit vocabulary instead of scattering many `ureg.define(...)` calls throughout Python.

### Proposed File

Add:

```text
mlsysim/mlsysim/core/mlsysbook_units.txt
```

### Proposed Contents

Start conservative. Move or duplicate only stable custom units first:

```text
# Data units used by MLSysBook.
KB = 1e3 * byte
MB = 1e6 * byte
GB = 1e9 * byte
TB = 1e12 * byte
PB = 1e15 * byte

KiB = 1024 * byte
MiB = 1048576 * byte
GiB = 1073741824 * byte
TiB = 1099511627776 * byte

# ML operation units.
flop = 1 * count
KFLOP = 1e3 * flop
MFLOP = 1e6 * flop
GFLOP = 1e9 * flop
TFLOP = 1e12 * flop
PFLOP = 1e15 * flop
EFLOP = 1e18 * flop
ZFLOP = 1e21 * flop

KFLOPs = KFLOP
MFLOPs = MFLOP
GFLOPs = GFLOP
TFLOPs = TFLOP
PFLOPs = PFLOP
EFLOPs = EFLOP
ZFLOPs = ZFLOP

# Integer operation rate units.
OPS = count / second
KOPS = 1e3 * OPS
MOPS = 1e6 * OPS
GOPS = 1e9 * OPS
TOPS = 1e12 * OPS

# Network shorthand used in prose and registries.
Gbps = 1e9 * bit / second

# Currency placeholders. Display must still go through fmt_usd().
dollar = 1 * count
USD = dollar
EUR = dollar

# Model parameter counts.
param = 1 * count
Kparam = 1e3 * param
Mparam = 1e6 * param
Bparam = 1e9 * param
Tparam = 1e12 * param
```

### Energy And Carbon Units

Add these if not already available from Pint under the names the book wants to export:

```text
Wh = watt * hour
kWh = kilowatt * hour
MWh = megawatt * hour
GWh = gigawatt * hour
```

Also verify built-in Pint names for:

- `gram`
- `kilogram`
- `metric_ton`
- `tonne`

If built-ins work cleanly, export aliases from `units.py` without redefining them.

### Time Alias Warning

The current code defines:

```python
ureg.define('MS = 1e-3 * second')
ureg.define('US = 1e-6 * second')
ureg.define('NS = 1e-9 * second')
```

This is risky because uppercase `MS` looks like an SI-prefixed unit, not standard milliseconds. Do not break existing chapters immediately. Instead:

- Keep `MS`, `US`, and `NS` as backward-compatible aliases for now.
- Prefer `ms`, `microsecond`, `nanosecond`, or string units like `"ms"` in new code.
- Add tests that display formatting normalizes these to `ms`, `us`, and `ns` or the chosen book spelling.
- Add a later migration ticket to remove uppercase aliases from chapter code.

### Implementation Notes

In `mlsysim/mlsysim/core/units.py`:

- Create `ureg = pint.UnitRegistry()` as today.
- Load `mlsysbook_units.txt` immediately after registry creation.
- Keep exported Python variables like `GB = ureg.GB`.
- Remove direct `ureg.define(...)` calls only after equivalent definitions are covered by tests.

Use `importlib.resources` or a simple path relative to `__file__` to load the definition file. Prefer a robust package-data approach if the package is distributed.

### Tests

Add or update unit tests:

```text
mlsysim/tests/test_units_registry.py
```

Tests should assert:

- `Q_("1 GB").to("byte").magnitude == 1e9`
- `Q_("1 GiB").to("byte").magnitude == 1073741824`
- `Q_("1 TFLOP/s").check("[count] / [time]")` or equivalent
- `Q_("1 TOPS").check("[count] / [time]")`
- `Q_("1 Gbps").to("GB/s").magnitude == 0.125`
- `Q_("1 kWh").to("J").magnitude == 3.6e6`
- `Q_("1 MWh").to("kWh").magnitude == 1000`
- `Q_("1 Bparam").to("param").magnitude == 1e9`

Acceptance:

- Existing registry YAML strings still parse.
- Existing tests still pass.
- Unit definitions are visible in one text file.

## Phase 1A: Normalize Exported Unit Aliases

### Objective

Make unit usage in LEGO cells consistent. Today some QMD cells use exported
MLSysIM aliases such as `MB`, `GB`, `TB`, `second`, and `joule`, while others
reach directly into the Pint registry with `ureg.joule`, `ureg.megawatt`,
`ureg.millijoule`, `ureg.kilowatt_hour`, and similar names.

Both forms are valid Pint. The inconsistency is authoring friction: an agent
has to remember which units are exported aliases and which require `ureg.*`.
The cleanup goal is not mathematical correctness; it is readability and a
single book-facing style.

### Current Scan

A Vol 1/Vol 2 QMD scan found direct `ureg.*` usage concentrated in these units:

| Unit accessed as `ureg.*` | Count |
|---|---:|
| `picojoule` | 52 |
| `megawatt` | 33 |
| `byte` | 31 |
| `millijoule` | 30 |
| `hour` | 24 |
| `kilogram` | 21 |
| `millisecond` | 15 |
| `minute` | 14 |
| `second` | 11 |
| `kilowatt_hour` | 11 |
| `megawatt_hour` | 10 |
| `count` | 10 |
| `microsecond` | 8 |
| `microjoule` | 7 |
| `metric_ton` | 6 |
| `mJ` | 6 |
| `flop` | 6 |
| `Wh` | 6 |
| `watt_hour` | 5 |
| `ms` | 5 |
| `megabit` | 4 |
| `km` | 4 |
| `joule` | 4 |

Representative inconsistency:

```python
fp32_size_mb_str = fmt_qty(fp32_size, MB, precision=1, commas=False)
int8_size_mb_str = fmt_qty(int8_size, MB, precision=1, commas=False)
fp32_energy_j_str = fmt_qty(fp32_energy, ureg.joule, precision=2, commas=False)
int8_energy_j_str = fmt_qty(int8_energy, ureg.joule, precision=2, commas=False)
```

`MB` and `ureg.joule` are both Pint units. The more consistent form is:

```python
fp32_size_mb_str = fmt_qty(fp32_size, MB, precision=1, commas=False)
int8_size_mb_str = fmt_qty(int8_size, MB, precision=1, commas=False)
fp32_energy_j_str = fmt_qty(fp32_energy, joule, precision=2, commas=False)
int8_energy_j_str = fmt_qty(int8_energy, joule, precision=2, commas=False)
```

because `joule` is already exported from `mlsysim.core.units`.

### Policy

Use exported MLSysIM aliases for common book-facing units. Use `ureg.*` only
when the unit is obscure, special, not yet exported, or intentionally being
accessed from the registry.

Recommended style:

```python
fmt_qty(size, MB)
fmt_qty(bandwidth, GB / second)
fmt_qty(energy, joule)
fmt_qty(power, watt)
fmt_time(latency, "millisecond")
```

Allowed direct registry style:

```python
fmt_qty(signal_to_noise, ureg.decibel)
fmt_qty(temp_rise, ureg.delta_degC / second, unit_label="degC/s")
```

The direct registry style should become rare in QMD after the common aliases
below are exported.

### Alias Inventory To Add Or Confirm

`mlsysim/mlsysim/core/units.py` already exports several common aliases:

- `byte`, `bit`, `second`, `joule`, `watt`, `kilowatt`, `milliwatt`
- `meter`, `hour`, `day`, `count`
- `KB`, `MB`, `GB`, `TB`, `PB`, `KiB`, `MiB`, `GiB`, `TiB`
- `ms`, `microsecond`, `millisecond`, `nanosecond`
- `flop`, `TFLOP`, `TFLOPs`, `PFLOPs`, `OPS`, `TOPS`, `Gbps`
- `param`, `Mparam`, `Bparam`, `Tparam`

Add or confirm these book-facing aliases:

Energy:

```python
J = ureg.joule
mJ = ureg.millijoule
uJ = ureg.microjoule
pJ = ureg.picojoule
Wh = ureg.watt_hour
kWh = ureg.kilowatt_hour
MWh = ureg.megawatt_hour
GWh = ureg.gigawatt_hour
```

Power:

```python
MW = ureg.megawatt
kW = ureg.kilowatt
mW = ureg.milliwatt
megawatt = ureg.megawatt
```

Time:

```python
minute = ureg.minute
week = ureg.week
year = ureg.year
```

Mass and carbon:

```python
gram = ureg.gram
kilogram = ureg.kilogram
kg = ureg.kilogram
metric_ton = ureg.metric_ton
tonne = ureg.metric_ton
```

Length:

```python
kilometer = ureg.kilometer
km = ureg.kilometer
```

Network and bit rates:

```python
megabit = ureg.megabit
gigabit = ureg.gigabit
terabit = ureg.terabit
```

Special units to consider, but handle carefully:

```python
degC = ureg.degC
delta_degC = ureg.delta_degC
decibel = ureg.decibel
volt = ureg.volt
mAh = ureg.mAh
```

Temperature offset units and logarithmic units can be surprising in Pint, so
export them only with focused tests.

### Migration Rules

Do not mass-rewrite all `ureg.*` sites immediately. Use this sequence:

1. Add aliases in `core/units.py` and `__all__`.
2. Add tests proving the aliases are the same Pint units as the registry names.
3. Add a warning-only lint rule for direct `ureg.*` use where an exported alias exists.
4. Opportunistically update touched LEGO cells.
5. Later, run a scoped cleanup pass over common units.

Acceptable before-and-after examples:

```python
# Before
fmt_qty(fp32_energy, ureg.joule, precision=2)
fmt_qty(power, ureg.megawatt, precision=1)
fmt_qty(total_energy, ureg.kilowatt_hour, precision=0)
fmt_qty(carbon, ureg.kilogram, precision=1)

# After
fmt_qty(fp32_energy, joule, precision=2)
fmt_qty(power, MW, precision=1)
fmt_qty(total_energy, kWh, precision=0)
fmt_qty(carbon, kilogram, precision=1)
```

### Tests

Extend `mlsysim/tests/test_units_registry.py`:

```python
assert joule == ureg.joule
assert mJ == ureg.millijoule
assert pJ == ureg.picojoule
assert MW == ureg.megawatt
assert kWh == ureg.kilowatt_hour
assert MWh == ureg.megawatt_hour
assert kilogram == ureg.kilogram
assert metric_ton == ureg.metric_ton
assert kilometer == ureg.kilometer
```

Add display tests for representative aliases:

```python
assert str(fmt_qty(Q_("1 J"), joule, precision=0, commas=False)) == "1 J"
assert str(fmt_qty(Q_("1 mJ"), mJ, precision=0, commas=False)) == "1 mJ"
assert str(fmt_qty(Q_("1 kWh"), kWh, precision=0, commas=False)) == "1 kWh"
assert str(fmt_qty(Q_("1 MW"), MW, precision=0, commas=False)) == "1 MW"
```

Acceptance:

- Common QMD units no longer require `ureg.*`.
- Direct `ureg.*` remains available for obscure units and registry internals.
- Formatter examples use one consistent style.
- The linter can tell whether a direct registry unit has a preferred alias.

## Phase 2: Add Formula Helpers For Recurring LEGO Calculations

### Objective

Reduce ad hoc `.m_as(...)` calculations in chapters. Repeated formulas should live in one tested module and return Pint quantities.

### Proposed File

Add:

```text
mlsysim/mlsysim/physics/quantities.py
```

or, if the existing codebase has a better local pattern, add to:

```text
mlsysim/mlsysim/physics/common.py
```

Do not put these in `fmt.py`. Formatting is presentation; formula helpers are computation.

### Helper Functions

Implement these first:

```python
def transfer_time(payload, bandwidth):
    """Return duration for moving payload bytes over bandwidth."""

def compute_time(work, throughput):
    """Return duration for work divided by operation rate."""

def energy_from_power(power, duration):
    """Return energy for power over duration."""

def carbon_from_energy(energy, carbon_intensity):
    """Return mass of carbon equivalent from energy and grid intensity."""

def memory_from_params(parameters, bytes_per_param):
    """Return memory footprint from parameter count and bytes per parameter."""

def token_throughput(tokens, duration):
    """Return token rate from token count and duration."""
```

### Validation Rules

Each helper should reject plain floats for physical inputs unless the function is intentionally for scalar counts.

Examples:

```python
if not isinstance(payload, ureg.Quantity):
    raise TypeError("transfer_time payload must be a Pint Quantity.")
if not payload.check("[length] ** 3") and not payload.check("[mass]"):
    ...
```

Pint dimensions for bytes may not be intuitive because `byte` is not a physical length dimension. Test the exact `.check(...)` strings locally before committing. If `.check(...)` is awkward for `byte`, validate by attempting `.to(byte)` or `.to(byte / second)` and let Pint raise `DimensionalityError`.

Recommended pattern:

```python
payload = payload.to(byte)
bandwidth = bandwidth.to(byte / second)
return (payload / bandwidth).to(second)
```

This uses Pint conversion as the validation mechanism and is simple.

### Formula Semantics

Use explicit semantic names:

- `transfer_time(payload, bandwidth)` not `duration(a, b)`.
- `compute_time(work, throughput)` not `latency(flops, flops_per_second)`.
- `carbon_from_energy(energy, carbon_intensity)` not `.to("kg")`, because carbon conversion depends on policy data.

### Carbon Intensity Unit

Support carbon intensity as:

```text
gram / kWh
kilogram / kWh
```

The helper should return `kilogram` or base mass quantity by default, not a string.

Example:

```python
energy = Q_("1287 MWh")
intensity = Q_("429 gram / kWh")
carbon = carbon_from_energy(energy, intensity)
assert round(carbon.to("metric_ton").magnitude) == 552
```

### Tests

Add:

```text
mlsysim/tests/test_quantity_formulas.py
```

Test cases:

- `transfer_time(Q_("16 GB"), Q_("3.35 TB/s")).to("ms")` is about `4.78 ms`.
- `compute_time(Q_("989 TFLOP"), Q_("989 TFLOP/s")).to("s") == 1`.
- `energy_from_power(Q_("700 W"), Q_("1 hour")).to("kWh") == 0.7`.
- `carbon_from_energy(Q_("1287 MWh"), Q_("429 g/kWh")).to("metric_ton")` is about `552`.
- `memory_from_params(Q_("7 Bparam"), Q_("2 byte / param")).to("GB") == 14`.
- Wrong dimensions raise Pint errors:
  - `transfer_time(Q_("1 second"), Q_("1 GB/s"))`
  - `compute_time(Q_("1 GB"), Q_("1 TFLOP/s"))`
  - `energy_from_power(Q_("1 GB"), Q_("1 hour"))`

Acceptance:

- Helpers return Pint quantities, not strings.
- Helpers do not format.
- Helpers catch wrong dimensions before prose rendering.

## Cross-Cutting Rule: Unit Ownership By LEGO Stage

The current branch diff shows many correct-but-fragile conversions such as:

```python
tdp_w = Hardware.Cloud.H100.tdp.m_as(watt)
cluster_accel_mw = cluster_gpus * tdp_w / MILLION
cluster_accel_mw_str = fmt_qty(cluster_accel_mw * ureg.megawatt, ureg.megawatt, precision=1)
```

That works numerically, but it drops the unit in EXECUTE and reattaches it in
OUTPUT. This is the pattern that keeps sending the book back to unit cleanup.

Standardize the stages as follows.

### LOAD

LOAD should acquire authoritative values and attach units to local scalar
assumptions immediately.

**Registry-first rule (2026-05-31):** If a value is a cited hardware/model/grid
fact or a reusable comparison anchor, it must come from MLSysIM — not a chapter
literal. Hypothetical scenario inputs (e.g. “10 GWh training run” in a notebook)
may stay in LOAD with an explicit comment.

| LOAD kind | Source | Example |
|-----------|--------|---------|
| Hardware / model / grid / dataset | `Hardware.*`, `Models.*`, `Infrastructure.*`, `Datasets.*` | `Hardware.Cloud.H100.tdp`, `Models.Language.GPT3.training_energy`, `Infrastructure.Grids.US_Avg` |
| Cited field / comparison anchors | `Literature.*` | Transatlantic flight CO₂e, MFU anchors |
| Pedagogical scenarios only | Local in LOAD, commented | `energy = 10_000 * MWh  # hypothetical notebook scenario` |
| Never | Literals duplicating registry | `energy_mwh = 1287` when `Models.Language.GPT3.training_energy_mwh` exists in YAML |

**Exemplar gap — `CarbonCostGPT3` (sustainable_ai, first cell in chapter):**

- `energy_mwh = 1287` → already in `models/data/language.yaml` as
  `GPT3.training_energy_mwh`; cell should pull from `Models.Language.GPT3` once
  promoted to `Quantity`.
- `Infrastructure.Grids.US_Avg` / `Quebec` — already correct.
- `kg_per_flight = 1000` → add `Literature` sustainability anchor with
  provenance (rounded pedagogical ~1 t CO₂e per transatlantic round trip unless
  a tighter citation is chosen).

If the registry is missing a value, **add to MLSysIM first** (Layer A′), then
wire LOAD — same rule as unit hardening, no chapter-local competing source.

Preferred:

```python
tdp = Hardware.Cloud.H100.tdp
n_gpus = 25_000
```

For a local assumption:

```python
host_overhead = 10.3 * kilowatt
training_duration = 14 * day
```

Avoid this in LOAD unless a third-party API genuinely requires a scalar:

```python
tdp_w = Hardware.Cloud.H100.tdp.m_as(watt)
```

If an external scalar API is unavoidable, name the scalar explicitly with a
`_val` suffix (never `_str`):

```python
tdp_w_val = Hardware.Cloud.H100.tdp.m_as(watt)
```

Do not format from that scalar later. Convert it back into a quantity at the
boundary where the scalar API returns. See **Export Naming Convention** below.

### EXECUTE

EXECUTE should perform physics with Pint quantities and semantic formula
helpers.

Preferred:

```python
accelerator_power = n_gpus * Hardware.Cloud.H100.tdp
facility_power = accelerator_power * pue
facility_energy = energy_from_power(facility_power, training_duration)
```

If a canonical internal unit improves readability, use `.to(...)` and keep the
result as a quantity:

```python
facility_power_kw = facility_power.to(kilowatt)
```

Avoid scalar canonicalization:

```python
facility_power_kw = facility_power.m_as(kilowatt)
```

The scalar form is acceptable only for algorithms, plotting libraries, or checks
that cannot consume Pint quantities. In those cases, keep the `_value`, `_kw`,
or `_w` name narrow and do not pass it to `fmt_qty`.

### GUARD

GUARD should compare quantities with units still attached. Prefer a small
assertion helper rather than repeated scalar extraction.

Proposed helper:

```python
assert_qty_close(actual, expected, display_unit, *, rel=1e-9, abs=None)
```

Example:

```python
assert_qty_close(facility_power, Q_("17.5 MW"), MW, rel=1e-3)
```

Until that helper exists, scalar extraction is acceptable inside `check(...)`
only:

```python
check(abs(facility_power.to(MW).magnitude - 17.5) < 0.1, "unexpected power")
```

### OUTPUT

OUTPUT is the only stage that should choose presentation units and precision.
The value passed to a physical formatter should still be a Pint quantity.

Preferred:

```python
cluster_accel_mw_str = fmt_power(accelerator_power, unit=MW)
facility_energy_kwh_str = fmt_energy(facility_energy, unit=kWh)
```

Until domain helpers exist:

```python
cluster_accel_mw_str = fmt_qty(accelerator_power, MW, precision=1)
facility_energy_kwh_str = fmt_qty(facility_energy, kWh, precision=1)
```

Avoid:

```python
cluster_accel_mw_str = fmt_qty(cluster_accel_mw * MW, MW, precision=1)
facility_energy_kwh_str = fmt_qty(facility_energy_kwh * kWh, kWh, precision=1)
```

This "reattach units in OUTPUT" pattern is the exact cleanup target.

### Stage Ownership Acceptance

- Registry and local physical values enter as Pint quantities in LOAD.
- Physical formulas in EXECUTE return Pint quantities.
- Unit conversion for display happens in OUTPUT, not earlier.
- `.m_as(...)` is limited to scalar-only consumers and checks.
- `fmt_qty` and domain formatters receive quantities, not reconstituted scalar
  magnitudes.
- Physical exports use the display-unit token naming convention below; prose
  does not repeat a unit that the export name already promises.

## Cross-Cutting Rule: Export Naming — Closed vs Open Strings

### The actual binary

Every prose-facing `_str` export is either:

| Class | Name pattern | Formatter output | Prose |
|---|---|---|---|
| **Closed** | `*_{unit_token}_str` | number **and** physical unit (or rate) | bare `` `{python} ref` `` |
| **Open** | `*_str` (no unit token) | number **only** (or kind-specific glyph via typed formatter) | prose supplies unit/word/glyph |

There is no third state for **physical** Pint quantities. Either the string
carries the unit (closed) or it does not (open). Mixing them — closed name with
open formatter, open name with closed formatter, closed string with prose-side
unit — is always a bug.

Scalars for math/plots/checks use `*_{unit_token}_val` or plain `*_val`; they
are never prose-facing.

### Why name alone is not enough (and what to add)

Encoding `_w` in the name helps readability, but mistakes are **symmetric**:

| Error | Cell | Prose | Rendered | Today |
|---|---|---|---|---|
| **A. Closed name, open formatter** | `tdp_w_str = fmt(tdp_w, …)` | `` … `tdp_w_str` W `` | `700 W` ✓ (accidentally OK) | common legacy |
| **B. Closed name, open formatter, bare prose** | `tdp_w_str = fmt(tdp_w, …)` | `` … `tdp_w_str` per chip `` | `700 per chip` ✗ | uncaught |
| **C. Closed formatter, closed name, prose repeats unit** | `tdp_w_str = fmt_qty(tdp, watt, …)` | `` … `tdp_w_str` W `` | `700 W W` ✗ | uncaught |
| **D. Closed formatter, open name** | `power_str = fmt_qty(power, MW, …)` | `` … `power_str` `` | `17.5 MW` ✓ | uncaught — name lied |
| **E. Open formatter, open name, bare prose** | `latency_str = fmt(latency_ms, …)` | `` … `latency_str` `` | `4.8` ✗ missing ms | uncaught |
| **F. Open formatter, open name, prose adds unit** | `latency_str = fmt(latency_ms, …)` | `` … `latency_str` ms `` | `4.8 ms` ✓ | intended open pattern |

The convention only pays off if lint enforces **both directions**:

1. **`_*_{unit_token}_str` ⇒ closed** — must be `fmt_qty` / domain formatter /
   `fmt_time`; prose must not append the same unit token.
2. **Plain `*_str` ⇒ not closed-physical** — must not be assigned from
   `fmt_qty` or a domain formatter on a Pint quantity. If you need a physical
   unit in the string, rename to include the display unit token.
3. **Plain `*_str` + `fmt()` / `fmt_ratio()` ⇒ open** — prose must supply the
   unit word (L017, harder; start as advisory on pilot chapters).

Error **D** is the one a name-only convention misses without rule 2. Error **B**
is why legacy `_w_str` + `fmt()` is dangerous: it looks hardened but fails the
moment prose drops the manual ` W`.

### Convention (strict)

```text
{concept}_{display_unit_token}_str   CLOSED — unit is IN the formatted string
{concept}_{display_unit_token}_val   scalar magnitude only; never in prose
{concept}_str                        OPEN — no physical unit in the string
```

**Closed rules (`_*_{unit_token}_str`):**

1. Built only with `fmt_qty`, a domain formatter, or `fmt_time` on a Pint
   `Quantity` — never `fmt()` on a scalar.
2. Display unit token is the **committed editorial unit** (`_w_str`, `_mw_str`,
   `_kwh_str`, `_gb_str`, `_gib_str`, `_ms_str`, `_g_per_kwh_str`, …) — not
   Pint registry spellings (`_watt_str`, `_ureg_joule_str`).
3. Prose uses a **bare** inline ref. Do not append ` W`, ` MW`, ` kWh`, etc.

**Open rules (plain `*_str`, no unit token):**

1. Built with `fmt`, `fmt_int`, `fmt_ratio`, `fmt_multiple(style="number")`, or
   typed closed-kind formatters that are not Pint-physical (see below).
2. Prose **must** supply the physical unit word or glyph after the ref when the
   value is physical (`… `{python} latency_str` ms ``, `… `{python} ai_str`
   FLOP/byte `).
3. **Forbidden:** `power_str = fmt_qty(power, MW, …)` — rename to `power_mw_str`
   or downgrade to open: `power_mw_val = power.m_as(MW)` and
   `power_str = fmt(power_mw_val, …)` with prose ` MW`.

**`_val` rules:**

- Never referenced in prose.
- Only for plotting, external APIs, or GUARD checks until `assert_qty_close`
  exists.

### Parallel track: typed closed strings (not Pint units)

`fmt_usd`, `fmt_percent(style="prose"|"symbol")`, `fmt_pp`, and `fmt_count`
with a label suffix produce **closed** strings, but the closure is by **value
kind**, not by a pint unit token. Do not invent `_usd_str` / `_pct_str` in the
name unless the team later standardizes kind tokens. For now:

- Use plain `cost_str`, `mfu_str`, `n_gpus_str`.
- The **formatter name** (`fmt_usd`, not `fmt_qty`) is the contract.
- Lint already covered by existing fmt migration rules; do not overload physical
  unit tokens for currency or percent.

### Honest exceptions (do not pretend they are closed-physical)

**Auto-scaling domain formatters** (`fmt_energy`, `fmt_memory`, `fmt_emissions`)
may emit a different display unit than any single token would promise. For
these, use a **semantic open-class name without a unit token**:

```python
facility_energy_str = fmt_energy(facility_energy)   # may be "1,287 MWh" or "1.29 GWh"
```

The string is still closed (unit inside), but the name correctly signals *do not
lint me as `_mwh_str`*. Tag these in lint as **closed-auto** (formatter-owned
unit, no fixed token) — distinct from closed-fixed (`_mw_str`) and open
(`latency_str` + prose ` ms`).

Only use `_{unit_token}_str` when the display unit is **fixed** for that
export.

### Display unit token vocabulary (closed-fixed only)

| Token | Committed display unit | Example |
|---|---|---|
| `_w_str` | watt | `"700 W"` |
| `_kw_str` | kilowatt | `"10.3 kW"` |
| `_mw_str` | megawatt | `"17.5 MW"` |
| `_kwh_str` | kilowatt_hour | `"1,287 kWh"` |
| `_mwh_str` | megawatt_hour | `"1287 MWh"` |
| `_j_str` | joule | `"265 J"` |
| `_gb_str` | GB | `"140 GB"` |
| `_gib_str` | GiB | `"80 GiB"` |
| `_ms_str` | millisecond | `"4.8 ms"` |
| `_g_per_kwh_str` | g/kWh | `"429 g/kWh"` |
| `_kg_str` | kilogram | `"552 kg"` |

Compound rates: `_g_per_kwh_str`, not `_g/kwh_str`. If a domain formatter
auto-scales within a family (e.g. bandwidth `_gb_s_str` might render as
`3.35 TB/s`), either pin the display unit in the formatter call or use
closed-auto naming instead.

### Worked example: `ArchetypeATdp` (sustainable_ai)

**Before — mixed closed/open (errors A and C both possible):**

```python
tdp_w = tdp.m_as(watt)
h_h100_tdp_w_str = fmt(tdp_w, precision=0, commas=False)          # closed name, open formatter
cluster_accel_mw_str = fmt_qty(cluster_accel_mw * MW, MW, ...)      # closed OK
```

Prose: `` `{python} h_h100_tdp_w_str` W per chip `` — works only because prose
repairs the open formatter; rename or drop ` W` and the sentence breaks.

**After — both physical exports closed-fixed:**

```python
tdp = Hardware.Cloud.H100.tdp
accelerator_power = cluster_gpus * tdp
h100_tdp_w_str = fmt_qty(tdp, watt, precision=0, commas=False)
cluster_accel_mw_str = fmt_qty(accelerator_power, MW, precision=1, commas=False)
```

Prose: `` `{python} h100_tdp_w_str` per chip `` — bare refs; units live in the
strings.

### Migration notes

- When touching a cell, migrate **name + formatter class + prose** together.
- Legacy `_w_str` + `fmt()` is **closed name / open formatter** — high-priority
  lint warning; pick one side:
  - **Upgrade to closed:** `fmt_qty` + bare prose (preferred for physical).
  - **Downgrade to open:** rename to `tdp_str` or `h100_tdp_str`, keep `fmt()`,
    keep prose ` W`.
- Do not mass-rename. Do not add unit tokens to names unless the formatter is
  upgraded to closed in the same commit.
- Document in `.claude/rules/lego-units.md`; cross-reference from
  `.claude/rules/fmt.md` §6–§7.

### Naming acceptance

- Every physical prose export is explicitly closed-fixed, closed-auto, or open
  — never ambiguous.
- Linter enforces closed-fixed ⇔ `_*_{unit_token}_str` ⇔ `fmt_qty`/domain (L014,
  L016) and prose non-duplication (L015).
- Linter enforces no `fmt_qty` on plain `*_str` (L017).
- Scalars use `_val`, never `_str`.

## Phase 3: Add Domain Formatters

### Objective

Stop making each LEGO cell decide units, precision, scale, and labels from scratch.

### Proposed Location

Add to:

```text
mlsysim/mlsysim/fmt.py
```

Keep generic helpers intact. Domain helpers should call existing helpers, especially `fmt_qty`, `fmt_time`, `fmt_count`, and `fmt_rate`.

### First Helper Set

Implement these first:

```python
def fmt_memory(quantity, *, binary=False, precision=None, commas=False):
    ...

def fmt_bandwidth(quantity, *, precision=None, commas=False):
    ...

def fmt_flop_rate(quantity, *, precision=None, commas=False):
    ...

def fmt_flops(quantity, *, precision=None, commas=False):
    ...

def fmt_latency(duration, *, precision=None, style="symbol"):
    ...

def fmt_duration(duration, *, precision=None, style="word"):
    ...

def fmt_params(value, *, precision=None, scale=None, label=True):
    ...

def fmt_tokens(value, *, precision=None, scale=None, label=True):
    ...

def fmt_token_rate(value, *, precision=None, scale=None):
    ...

def fmt_power(quantity, *, precision=None, commas=False):
    ...

def fmt_energy(quantity, *, precision=None, commas=False):
    ...

def fmt_emissions(quantity, *, precision=None, commas=False):
    ...

def fmt_carbon_intensity(quantity, *, precision=None, commas=False):
    ...
```

### Auto-Scaling Policies

Start with deterministic, conservative thresholds.

Memory:

- `< 1 MB`: use `KB`
- `< 1 GB`: use `MB`
- `< 1 TB`: use `GB`
- otherwise use `TB`
- If `binary=True`, use `KiB`, `MiB`, `GiB`, `TiB`.

Bandwidth:

- `< 1 GB/s`: use `MB/s`
- `< 1 TB/s`: use `GB/s`
- otherwise use `TB/s`

FLOP rate:

- `< 1 TFLOP/s`: use `GFLOP/s`
- `< 1 PFLOP/s`: use `TFLOP/s`
- otherwise use `PFLOP/s`

FLOPs:

- `< 1 TFLOP`: use `GFLOP`
- `< 1 PFLOP`: use `TFLOP`
- `< 1 EFLOP`: use `PFLOP`
- otherwise use `EFLOP`

Latency:

- `< 1 us`: use `ns`
- `< 1 ms`: use `us`
- `< 1 s`: use `ms`
- `< 60 s`: use `s`
- `< 60 min`: use `min`
- otherwise use `hour`

Duration:

- `< 60 s`: use `s`
- `< 60 min`: use `min`
- `< 48 h`: use `hour`
- otherwise use `day`

Parameters:

- `< 1e3`: raw count
- `< 1e6`: `K`
- `< 1e9`: `M`
- `< 1e12`: `B`
- otherwise `T`

Tokens:

- Same K/M/B/T policy as parameters unless a chapter requires raw token counts.

Energy:

- `< 1 Wh`: use `J`
- `< 1 kWh`: use `Wh`
- `< 1 MWh`: use `kWh`
- `< 1 GWh`: use `MWh`
- otherwise use `GWh`

Emissions:

- `< 1 kg`: use `g`
- `< 1000 kg`: use `kg`
- otherwise use `metric_ton`

Carbon intensity:

- Default display unit: `g/kWh`.
- Precision default: `0` for whole-number grid intensity unless fractional source data is present.

### Precision Policy

Default precision should be set by rendered magnitude, not by raw value.

Recommended defaults:

- If displayed magnitude is integer-like, use `precision=0`.
- If displayed magnitude is `>= 100`, use `precision=0`.
- If displayed magnitude is `>= 10`, use `precision=1`.
- If displayed magnitude is `< 10`, use `precision=2` only when needed to avoid hidden nonzero values.

Do not bypass existing `fmt()` precision guards. If a domain helper picks a bad precision, the test should expose it.

### Forced Scale Policy

For `fmt_params(150e6, scale="B")`, do not silently emit `0B`.

Choose one of these policies and document it:

1. Recommended: if forced scale plus default precision would hide the value, automatically raise precision.
   - `fmt_params(150e6, scale="B")` -> `0.15B`
2. Alternative: refuse hidden forced scale and ask caller to pick precision.
   - raises `ValueError`
3. Alternative: ignore forced scale when it would hide the value.
   - `fmt_params(150e6, scale="B")` -> `150M`

Recommended policy is option 1 because explicit `scale="B"` probably means the author wants billions, and the formatter can safely preserve nonzero magnitude.

### Multiplier Formatting Policy

`fmt_multiple` is conceptually the right helper because a speedup/reduction
factor is dimensionless. Pint's `P` formatting is not the right layer for the
`times` glyph: `×` is not a unit, and a multiplier is not a Pint quantity.

The current number-only API still creates drift risk because every prose
reference must remember to add `$\times$` separately:

```markdown
`{python} speedup_str`$\times$ faster
```

Evolve `fmt_multiple` to own the glyph policy while staying backward-compatible.

Proposed API:

```python
def fmt_multiple(factor, *, precision=None, commas=False, style="number"):
    ...
```

Style policy:

- `style="number"`: current behavior, returns `3.2`.
- `style="symbol"`: returns `3.2$\times$` as `MarkdownStr` for prose/Pandoc.
- `style="word"`: returns `3.2 times`.

Use `precision=None` or an equivalent auto policy so integer factors do not
surprise authors:

```python
fmt_multiple(2)                 # "2", not a spurious-zero error
fmt_multiple(2, style="symbol") # "2$\times$"
fmt_multiple(3.2)               # "3.2"
```

Migration policy:

- Keep `style="number"` as the initial default for compatibility.
- Prefer `style="symbol"` in new prose once render tests confirm HTML/PDF output.
- Convert old prose from `` `{python} speedup_str`$\times$ faster `` to
  `` `{python} speedup_str` faster `` only when the exported value uses
  `style="symbol"`.
- Never use `fmt_qty`, Pint formatting, or `unit_label` for multipliers.

Tests:

- Number, symbol, and word styles.
- Integer and non-integer factors with automatic precision.
- Negative factors still raise.
- HTML/PDF render of `MarkdownStr("3.2$\\times$")` in inline Python.
- A duplicate-glyph guard catches `style="symbol"` followed by prose-side
  `$\times$`.

### Range Formatting Policy

`fmt_range(..., unit="GB")` is a useful generic helper, but it can become a
free-text physical-unit backdoor. For physical quantities, prefer typed range
helpers:

```python
fmt_qty_range(lo_memory, hi_memory, GB)
fmt_time_range(lo_latency, hi_latency, millisecond)
fmt_usd_range(lo_cost, hi_cost, scale="K")
```

Reserve generic `fmt_range(unit=...)` for non-physical prose labels where no
dimension check is possible. Add a linter warning for physical-looking units in
`fmt_range(unit=...)` once the typed range helper inventory is complete.

### Tests

Add:

```text
mlsysim/tests/test_domain_formatters.py
```

Test examples:

- `fmt_memory(Q_("80 GiB"), binary=True)` -> `80 GiB`
- `fmt_memory(Q_("140 GB"))` -> `140 GB`
- `fmt_bandwidth(Q_("3350 GB/s"))` -> `3.35 TB/s` or chosen policy.
- `fmt_flop_rate(Q_("989 TFLOP/s"))` -> `989 TFLOP/s`
- `fmt_latency(Q_("0.0000048 s"))` -> `4.8 us` or chosen policy.
- `fmt_energy(Q_("1287 MWh"))` -> `1,287 MWh` or chosen policy.
- `fmt_emissions(Q_("552000 kg"))` -> `552 tonnes CO2e` or chosen label policy.
- `fmt_carbon_intensity(Q_("429 g/kWh"))` -> `429 g/kWh`.
- `fmt_params(Q_("150 Mparam"))` -> `150M parameters` or chosen label policy.
- `fmt_multiple(2, style="symbol")` -> `2$\times$` or the final chosen
  Markdown-safe symbol form.
- `fmt_usd_range(10_000, 30_000, scale="K", repeat_symbol=False)` does not
  reference removed private scale constants.
- `fmt_params(Q_("150 Mparam"), scale="B")` -> `0.15B parameters` if recommended policy is chosen.

Acceptance:

- Domain helpers use Pint quantities where relevant.
- Domain helpers avoid raw suffix strings.
- Domain helpers make common LEGO cells shorter and safer.

## Phase 4: Improve Generic Unit Labeling Safely

### Objective

Use Pint formatting as the base, then apply small MLSysBook label normalization.

### Current Code

`_compact_unit_suffix(display_unit)` currently formats `1 * display_unit` with `~P`, then splits the result.

This is the right basic idea.

### Proposed Normalization Map

Add a small internal map:

```python
_UNIT_LABEL_NORMALIZATION = {
    "KFLOPs": "KFLOP",
    "MFLOPs": "MFLOP",
    "GFLOPs": "GFLOP",
    "TFLOPs": "TFLOP",
    "PFLOPs": "PFLOP",
    "EFLOPs": "EFLOP",
    "ZFLOPs": "ZFLOP",
    "KFLOPs/s": "KFLOP/s",
    "MFLOPs/s": "MFLOP/s",
    "GFLOPs/s": "GFLOP/s",
    "TFLOPs/s": "TFLOP/s",
    "PFLOPs/s": "PFLOP/s",
    "EFLOPs/s": "EFLOP/s",
    "ZFLOPs/s": "ZFLOP/s",
    "Gbps": "Gb/s",
    "US": "us",
    "MS": "ms",
    "NS": "ns",
}
```

Apply this after Pint produces the label. Keep the map small and tested.

### `unit_label` Escape Hatch

Do not remove `unit_label` yet, but add tests around common misuse.

Current suspicious pattern:

```python
fmt_qty(x, GB/second, unit_label="GB")
```

This can be intentional in some communication math where beta is shown as a coefficient, but it can also hide `/s`.

Action:

- Document `unit_label` as a legacy escape hatch.
- Prefer domain helpers over `unit_label`.
- Add a lint warning for `fmt_qty(..., unit_label=...)` except allowlisted files or specific comments.

Acceptance:

- Existing rendered strings do not regress unexpectedly.
- FLOP labels become consistent.
- Time unit aliases render consistently.

## Phase 5: Add A LEGO Unit Linter

### Objective

Catch bad patterns in QMD files before they become rendered prose mistakes.
Include name/formatter/prose mismatches for display-unit token exports (rules
L014–L016).

### Proposed File

Add:

```text
book/tools/scripts/lint_lego_units.py
```

If the repo already has a linter framework for QMD checks, integrate there instead.

### Scope

Scan:

```text
book/quarto/contents/vol1/**/*.qmd
book/quarto/contents/vol2/**/*.qmd
```

Only inspect code cells and inline Python if practical. A first pass can scan the full QMD text with regexes and report line numbers.

### Lint Rules

Start with warnings, then promote high-confidence rules to errors.

#### Rule L001: Do Not Pass Scalar Magnitudes To `fmt_qty`

Flag:

```python
fmt_qty(x.m_as(GB), GB)
```

Message:

```text
fmt_qty requires a Pint Quantity. Pass x, not x.m_as(...).
```

#### Rule L002: Avoid Raw Physical Unit Suffixes In `fmt`

Flag:

```python
fmt(value, suffix=" GB/s")
fmt(value, suffix=" MWh")
fmt(value, suffix=" kWh")
fmt(value, suffix=" TFLOP/s")
```

Message:

```text
Use fmt_qty or a domain formatter for physical units.
```

Do not flag:

- `fmt_percent`
- `fmt_usd`
- already structured `fmt_count`
- prose-only strings outside code cells

#### Rule L003: Avoid Reattaching Units After `.m_as(...)`

Flag patterns like:

```python
x_gb = x.m_as(GB)
y = x_gb * GB
```

This will be approximate in regex. It is acceptable as a warning.

Message:

```text
Keep the original Pint Quantity when possible. Reattaching units after .m_as() can hide conversion errors.
```

#### Rule L004: Avoid Raw Energy And Carbon Math

Flag:

```python
energy_mwh * 1000
carbon_intensity_g_kwh
kg_co2 = ...
tonnes = kg / 1000
```

Message:

```text
Use energy_from_power, carbon_from_energy, fmt_energy, or fmt_emissions.
```

#### Rule L005: Forced Count Scale With Hidden Precision

Flag:

```python
fmt_count(..., scale="B", precision=0)
```

When the input is visibly a smaller expression such as `150 * MILLION`, this is suspicious.

Message:

```text
Forced scale with precision=0 can hide nonzero values. Use fmt_params/fmt_tokens or choose precision explicitly.
```

#### Rule L006: Direct `unit_label=`

Flag:

```python
fmt_qty(..., unit_label=...)
```

Message:

```text
Prefer a domain formatter. unit_label is an escape hatch and should be reviewed.
```

This should start as warning only.

#### Rule L007: Uppercase Time Aliases

Flag new usage of:

```python
MS
US
NS
```

Message:

```text
Prefer ms, microsecond, nanosecond, or string units like "ms".
```

This should start as warning only because existing code uses uppercase aliases.

#### Rule L008: Prefer Clear Unit Construction For Rates

Flag unparenthesized scalar-times-rate expressions as cleanup warnings, not errors:

```python
kv_cache_bandwidth = 1.9 * TB / second
gemm_bandwidth = 2.8 * TB / second
achieved_flops = 15 * TFLOPs / second
```

Message:

```text
This is valid Pint usage, but prefer 1.9 * (TB / second) or Q_("1.9 TB/s") so the rate unit is visually explicit.
```

Do not flag already clear forms:

```python
kv_cache_bandwidth = 1.9 * (TB / second)
kv_cache_bandwidth = Q_("1.9 TB/s")
```

This rule is about readability and future maintenance, not mathematical correctness.

#### Rule L009: Prefer Exported Aliases For Common Units

After Phase 1A exports common book-facing aliases, flag direct `ureg.*` usage
when a preferred alias exists.

Flag as warning:

```python
fmt_qty(fp32_energy, ureg.joule, precision=2)
fmt_qty(power, ureg.megawatt, precision=1)
fmt_qty(total_energy, ureg.kilowatt_hour, precision=0)
fmt_qty(carbon, ureg.kilogram, precision=1)
latency.m_as(ureg.millisecond)
```

Preferred forms:

```python
fmt_qty(fp32_energy, joule, precision=2)
fmt_qty(power, MW, precision=1)
fmt_qty(total_energy, kWh, precision=0)
fmt_qty(carbon, kilogram, precision=1)
latency.m_as(millisecond)
```

Message:

```text
Use the MLSysIM exported unit alias for common book-facing units. Reserve ureg.* for obscure or special units.
```

Keep this warning-only until the alias inventory is added and tested. Do not
flag units that intentionally remain registry-only, such as special logarithmic
or offset-temperature units, unless an alias has been explicitly exported.

#### Rule L010: Multiplier Glyph Ownership

Once `fmt_multiple(..., style=...)` exists, the linter should understand which
layer owns the `times` glyph.

Flag missing glyph during the compatibility period:

```python
speedup_str = fmt_multiple(6, style="number")
```

with prose:

```markdown
`{python} C.speedup_str` faster
```

Message:

```text
fmt_multiple(..., style="number") returns a number only. Add $\times$ in prose or use style="symbol".
```

Flag duplicate glyph after migration:

```python
speedup_str = fmt_multiple(6, style="symbol")
```

with prose:

```markdown
`{python} C.speedup_str`$\times$ faster
```

Message:

```text
fmt_multiple(..., style="symbol") already owns the times glyph. Remove the prose-side $\times$.
```

Also flag migration candidates where a multiplier-like variable is still plain
`fmt(...)`:

```python
speedup_str = fmt(speedup, precision=1)
reduction_str = fmt(reduction, precision=1)
improvement_str = fmt(improvement, precision=1)
```

Message:

```text
This name looks like a multiplier. Use fmt_multiple so the domain guard and glyph policy apply.
```

#### Rule L011: Unit Drop/Reattach Stage Boundary

Flag scalar extraction followed by unit reattachment for the same conceptual
value:

```python
gpu_power_w = Hardware.Cloud.A100.tdp.m_as(watt)
gpu_energy = n_gpus * gpu_power_w * watt * training_hours * hour
```

and:

```python
facility_energy_kwh = gpu_energy.to(kWh).magnitude
facility_energy_kwh_str = fmt_qty(facility_energy_kwh * kWh, kWh)
```

Message:

```text
Keep physical quantities attached through EXECUTE and convert only at OUTPUT. Reattaching units after .m_as() can hide unit mistakes.
```

Allow:

- `_value` scalars used only by plotting or external APIs.
- scalar extraction inside `check(...)` while `assert_qty_close` is not yet available.
- scalar extraction immediately followed by non-physical `fmt(...)` for a
  deliberately dimensionless ratio.

This rule directly enforces the LOAD/EXECUTE/GUARD/OUTPUT unit ownership policy.

#### Rule L012: Physical Units In Generic Range Formatter

Flag:

```python
fmt_range(5, 10, unit="GB")
fmt_range(2, 4, unit="kWh")
fmt_range(1, 3, unit="ms")
```

Message:

```text
Use fmt_qty_range or fmt_time_range for physical ranges so dimensions are checked.
```

Allow non-physical labels such as `unit="percent"` only if they remain covered
by separate percent-range policy.

#### Rule L013: Scientific Notation Unit Choice

`sci_latex` is a TeX atom helper, not a quantity formatter. It strips Pint
quantities to magnitudes, so the unit must be selected before calling it.

Flag:

```python
sci_latex(quantity)
```

when `quantity` is visibly a registry quantity or a variable name that does not
show the display unit.

Preferred:

```python
sci_latex(work.to(flop))
sci_latex(rate.to(flop / second))
```

Future safer helper:

```python
sci_qty_latex(work, flop)
sci_qty_latex(rate, flop / second)
```

Message:

```text
sci_latex emits only a LaTeX magnitude. Convert to an explicit display unit first, or use the quantity-aware scientific-notation helper.
```

#### Rule L014: Display Unit Token Requires Quantity Formatter

When an OUTPUT export matches `*_{unit_token}_str` for a known physical display
unit token (`_w_str`, `_mw_str`, `_kwh_str`, `_mwh_str`, `_gb_str`, `_gib_str`,
`_ms_str`, `_gb_s_str`, `_j_str`, `_kg_str`, `_g_per_kwh_str`, …), the RHS
must be `fmt_qty(...)`, a domain formatter (`fmt_power`, `fmt_energy`, …), or
`fmt_time(...)` — not plain `fmt(...)` on a scalar.

Flag:

```python
h_h100_tdp_w_str = fmt(tdp_w, precision=0, commas=False)
gpu_energy_kwh_str = fmt(gpu_energy_kwh, precision=1, commas=True)
```

Preferred:

```python
h100_tdp_w_str = fmt_qty(tdp, watt, precision=0, commas=False)
gpu_energy_kwh_str = fmt_qty(gpu_energy, kWh, precision=1, commas=True)
```

Message:

```text
Export name promises a physical display unit in the formatted string. Use fmt_qty or a domain formatter, not fmt() on a scalar.
```

Start as warning only; many legacy cells still use `fmt()` with `_w_str` names.

#### Rule L015: Prose Must Not Repeat A Promised Display Unit

When prose references an export whose name ends in a known physical display unit
token (`_w_str`, `_mw_str`, `_kwh_str`, …), flag prose that repeats the same
unit word or symbol immediately after the inline ref.

Flag prose near:

```python
h_h100_tdp_w_str = fmt_qty(tdp, watt, ...)
```

with markdown:

```markdown
`{python} ArchetypeATdp.h_h100_tdp_w_str` W per chip
```

Message:

```text
Export name already promises the display unit in the formatted string. Use a bare `{python} …_w_str` ref; do not add " W" in prose.
```

Also flag the inconsistent legacy pattern where `_str` is magnitude-only but
prose adds the unit — this is a paired migration hint with L014:

```markdown
`{python} ArchetypeATdp.h_h100_tdp_w_str` W per chip
```

when the cell still has `h_h100_tdp_w_str = fmt(tdp_w, ...)`.

Implementation note: L015 requires scanning prose within a bounded window of the
inline ref (same paragraph or callout). Start warning-only.

#### Rule L016: Scalars Must Use `_val`, Not `_str`

Flag OUTPUT exports that assign plain scalars from `.m_as(...)` to a name ending
in a physical unit token plus `_str`:

```python
tdp_w_str = fmt(tdp.m_as(watt), ...)
```

Preferred:

```python
tdp_w_val = tdp.m_as(watt)                    # scalar boundary only
tdp_w_str = fmt_qty(tdp, watt, ...)           # prose-facing
```

Message:

```text
Scalars extracted with .m_as() belong in a _val export. Prose-facing physical values use _str with fmt_qty or a domain formatter on the original Quantity.
```

#### Rule L017: Plain `_str` Must Not Use Closed-Physical Formatters

If an OUTPUT export matches `*_str` but does **not** contain a known physical
display unit token before `_str`, the RHS must not be `fmt_qty(...)`, a domain
formatter on a Pint quantity, or `fmt_time(...)` on a duration quantity.

Flag:

```python
power_str = fmt_qty(accelerator_power, MW, precision=1)
latency_str = fmt_qty(latency, millisecond, precision=1)
facility_energy_str = fmt_energy(facility_energy)   # only if treating as closed-fixed; see closed-auto note
```

Preferred (closed-fixed — rename):

```python
power_mw_str = fmt_qty(accelerator_power, MW, precision=1)
latency_ms_str = fmt_qty(latency, millisecond, precision=1)
```

Preferred (open — keep plain name, prose owns unit):

```python
latency_ms_val = latency.m_as(millisecond)
latency_str = fmt(latency_ms_val, precision=1, commas=False)
```

Prose: `` `{python} latency_str` ms ``

Message:

```text
fmt_qty/domain formatter produces a closed string (unit included). Rename to *_{unit_token}_str or downgrade to open fmt() with prose-side unit.
```

Exception: **closed-auto** exports (`facility_energy_str = fmt_energy(...)`,
`mem_str = fmt_memory(...)`) intentionally have no unit token because the
display unit is not fixed. Maintain an allowlist of formatter names permitted
for closed-auto plain `_str` exports. Do not assign closed-auto formatters to
names that already contain a unit token (that would over-promise).

#### Rule L018: Open Physical Export Missing Prose Unit (Advisory)

If an export is open (`*_str` without unit token) and assigned from `fmt(...)` on
a variable whose name ends in `_ms`, `_w`, `_gb`, etc., flag prose references
that use a bare inline ref with no adjacent unit word in the same sentence.

Flag prose:

```markdown
latency is `{python} Cell.latency_str`
```

when `latency_str = fmt(latency_ms, ...)` and `latency_ms` is clearly a
physical scalar.

Message:

```text
Open export (no unit token in name). Prose must supply the unit word after the ref, e.g. `{python} latency_str` ms.
```

Start advisory only — natural language is hard to parse reliably.

#### Rule L019: LOAD Literal Duplicates Registry (Warning)

Flag numeric literals in LOAD that match a known registry field for the same
semantic (e.g. `energy_mwh = 1287` when `Models.Language.GPT3.training_energy_mwh`
exists in YAML).

Message:

```text
LOAD literal duplicates registry. Pull from Models.* / Infrastructure.* / Literature.* instead.
```

#### Rule L020: Float Property When Quantity Exists (Advisory)

Flag `grid.carbon_intensity_kg_kwh` or similar float accessors in EXECUTE/OUTPUT
when a Quantity-based grid intensity helper exists.

Message:

```text
Prefer Quantity carbon intensity at LOAD (gram/kWh) and carbon_from_energy in EXECUTE.
```

### CLI Behavior

Support:

```bash
python3 book/tools/scripts/lint_lego_units.py
python3 book/tools/scripts/lint_lego_units.py --fail-on error
python3 book/tools/scripts/lint_lego_units.py --fail-on warning
python3 book/tools/scripts/lint_lego_units.py --format text
```

Output should include:

- File path
- Line number
- Rule id
- Severity
- Short message
- Offending line excerpt

### Tests

Add:

```text
book/tools/tests/test_lint_lego_units.py
```

or the existing test location for book scripts.

Test:

- Each lint rule catches a small fixture.
- Allowlisted patterns are not flagged.
- Exit status respects `--fail-on`.
- L014 catches `h_h100_tdp_w_str = fmt(tdp_w, ...)`.
- L015 catches prose `` `{python} …_w_str` W `` when the export uses `fmt_qty`.
- L016 catches `tdp_w_str = fmt(tdp.m_as(watt), ...)`.

Acceptance:

- Linter can run locally without rendering Quarto.
- Linter output is concise enough for an author to fix.
- Initial integration can be advisory if there are many existing warnings.

## Phase 6: Add Golden LEGO Calculation Tests

### Objective

Protect the book's recurring physical relationships with tests that cover both numeric results and rendered strings.

### Proposed Test File

Add:

```text
mlsysim/tests/test_lego_unit_invariants.py
```

### Required Test Categories

#### Bandwidth To Time

```python
payload = Q_("16 GB")
bw = Q_("3.35 TB/s")
latency = transfer_time(payload, bw)
assert latency.to("ms").magnitude == pytest.approx(4.776, rel=1e-3)
assert str(fmt_latency(latency)) == "4.8 ms"
```

#### Compute To Time

```python
work = Q_("989 TFLOP")
rate = Q_("989 TFLOP/s")
duration = compute_time(work, rate)
assert duration.to("s").magnitude == pytest.approx(1.0)
```

#### Params To Memory

```python
params = Q_("7 Bparam")
bytes_per_param = Q_("2 byte / param")
memory = memory_from_params(params, bytes_per_param)
assert memory.to("GB").magnitude == pytest.approx(14)
```

#### Energy From Power

```python
energy = energy_from_power(Q_("700 W"), Q_("1 hour"))
assert energy.to("kWh").magnitude == pytest.approx(0.7)
```

#### Carbon From Energy

```python
carbon = carbon_from_energy(Q_("1287 MWh"), Q_("429 g/kWh"))
assert carbon.to("metric_ton").magnitude == pytest.approx(552.123, rel=1e-3)
```

#### Network Bit/Byte Conversion

```python
assert Q_("400 Gbps").to("GB/s").magnitude == pytest.approx(50)
```

#### H100 Registry Invariants

Use actual registry values:

```python
h = Hardware.Cloud.H100
assert h.memory.capacity.to("GiB").magnitude == pytest.approx(80)
assert h.memory.bandwidth.to("TB/s").magnitude == pytest.approx(3.35)
assert h.compute.peak_flops.to("TFLOP/s").magnitude == pytest.approx(989)
```

#### Wrong-Dimension Failures

Assert failures:

```python
with pytest.raises(Exception):
    transfer_time(Q_("1 second"), Q_("1 GB/s"))

with pytest.raises(Exception):
    compute_time(Q_("1 GB"), Q_("1 TFLOP/s"))

with pytest.raises(Exception):
    fmt_qty(Q_("1 GB"), second)
```

Acceptance:

- These tests run quickly.
- They fail if a future edit breaks key unit semantics.
- They cover the exact classes of mistakes that previously caused LEGO churn.

## Phase 7: Book-Order QMD Migration (Volume 1, then Volume 2)

### Objective

Migrate LEGO cells **in the order chapters appear in the book**, not by perceived
risk. Complete **all eligible cells in Volume 1** (front to back), then **all
eligible cells in Volume 2** (front to back). Within each chapter, migrate cells
**top-to-bottom** in document order.

### Migration order (authoritative source)

Chapter sequence comes from the Quarto render manifests:

- Volume I: [`book/quarto/config/_quarto-html-vol1.yml`](book/quarto/config/_quarto-html-vol1.yml)
- Volume II: [`book/quarto/config/_quarto-html-vol2.yml`](book/quarto/config/_quarto-html-vol2.yml)

**Volume I — process `.qmd` files in this order** (skip files with no LEGO cells
after lint scan):

```text
contents/vol1/introduction/introduction.qmd
contents/vol1/ml_systems/ml_systems.qmd
contents/vol1/ml_workflow/ml_workflow.qmd
contents/vol1/data_engineering/data_engineering.qmd
contents/vol1/nn_computation/nn_computation.qmd
contents/vol1/nn_architectures/nn_architectures.qmd
contents/vol1/frameworks/frameworks.qmd
contents/vol1/training/training.qmd
contents/vol1/data_selection/data_selection.qmd
contents/vol1/model_compression/model_compression.qmd
contents/vol1/hw_acceleration/hw_acceleration.qmd
contents/vol1/benchmarking/benchmarking.qmd
contents/vol1/model_serving/model_serving.qmd
contents/vol1/ml_ops/ml_ops.qmd
contents/vol1/responsible_engr/responsible_engr.qmd
contents/vol1/conclusion/conclusion.qmd
contents/vol1/backmatter/appendix_dam.qmd
contents/vol1/backmatter/appendix_data.qmd
contents/vol1/backmatter/appendix_algorithm.qmd
contents/vol1/backmatter/appendix_machine.qmd
contents/vol1/backmatter/appendix_assumptions.qmd
contents/vol1/backmatter/glossary/glossary.qmd
```

Skip unless LEGO found: frontmatter, part dividers (`parts/*.qmd`), `references.qmd`,
`socratiq.qmd`.

**Volume II — same rule, after Vol I is complete:**

```text
contents/vol2/introduction/introduction.qmd
contents/vol2/compute_infrastructure/compute_infrastructure.qmd
contents/vol2/network_fabrics/network_fabrics.qmd
contents/vol2/data_storage/data_storage.qmd
contents/vol2/distributed_training/distributed_training.qmd
contents/vol2/collective_communication/collective_communication.qmd
contents/vol2/fault_tolerance/fault_tolerance.qmd
contents/vol2/fleet_orchestration/fleet_orchestration.qmd
contents/vol2/performance_engineering/performance_engineering.qmd
contents/vol2/inference/inference.qmd
contents/vol2/edge_intelligence/edge_intelligence.qmd
contents/vol2/ops_scale/ops_scale.qmd
contents/vol2/security_privacy/security_privacy.qmd
contents/vol2/robust_ai/robust_ai.qmd
contents/vol2/sustainable_ai/sustainable_ai.qmd
contents/vol2/responsible_ai/responsible_ai.qmd
contents/vol2/conclusion/conclusion.qmd
contents/vol2/backmatter/appendix_dam.qmd
contents/vol2/backmatter/appendix_c3.qmd
contents/vol2/backmatter/appendix_fleet.qmd
contents/vol2/backmatter/appendix_communication.qmd
contents/vol2/backmatter/appendix_reliability.qmd
contents/vol2/backmatter/appendix_inference.qmd
contents/vol2/backmatter/appendix_assumptions.qmd
contents/vol2/backmatter/glossary/glossary.qmd
```

Do **not** jump to high-risk chapters (e.g. `sustainable_ai.qmd`) early; they
arrive in queue when Vol II reaches them (~15th content chapter).

### Within-chapter cell order

For each `.qmd` file:

1. List LEGO classes **top-to-bottom** (document order): grep for
   `# ┌── LEGO` blocks and `class Name:` in `{python}` cells.
2. Migrate **one class per step** (one commit): E → G → O + prose locality window.
3. Do not skip ahead to a later cell in the same chapter.
4. Record progress in a migration log (chapter path, class name, step id).

**Step 14 (first QMD step)** = first LEGO class in
`contents/vol1/introduction/introduction.qmd` (not `ArchetypeATdp` in
`sustainable_ai.qmd`). `ArchetypeATdp` remains a **reference example** in this
plan but is migrated when the queue reaches `sustainable_ai.qmd`.

### Migration pattern (per cell)

1. Replace repeated formula logic with helpers where applicable.
2. Replace generic `fmt_qty` with domain formatters where policy is clear.
3. Align export name + formatter class + prose in the same edit (closed/open).
4. Keep numeric checks in GUARD.
5. Run `lint_lego_units.py` on the touched file.
6. Render the hosting chapter before moving to the **next chapter** in the queue;
   during the **first Vol I chapter**, render after each cell until the pattern
   is stable.

### What not to migrate yet

Leave a cell alone if:

- `.m_as(...)` feeds plotting code or a scalar-only algorithm.
- Change would alter prose scope beyond unit-hardening.
- The suffix-cleanup thread is already touching that exact call site.

### Render command (template)

From `book/quarto/`:

```bash
PYTHONPATH=../..:../../mlsysim MPLBACKEND=Agg \
  quarto render contents/vol1/introduction/introduction.qmd \
  --to html --output-dir /private/tmp/mlsysbook-unit-hardening
```

Swap the path for the chapter currently in the queue.

Acceptance:

- Every content chapter in Vol I then Vol II processed in manifest order.
- Key displayed values unchanged unless a test exposes a bug.
- Linter warning count trends down corpus-wide as the queue advances.

## Phase 8: Pre-Commit, Binder, and CI Integration

### Objective

Make unit regressions **block commits locally** via pre-commit (same pattern as
`book-check-math`, `book-check-code`, `book-check-registry-sources`). The linter
is not optional tooling — it becomes part of the repo guardrails.

### Wiring (follow existing binder pattern)

Per [`.pre-commit-config.yaml`](.pre-commit-config.yaml) and
[`book/docs/BINDER.md`](book/docs/BINDER.md):

1. Implement [`book/tools/scripts/lint_lego_units.py`](book/tools/scripts/lint_lego_units.py).
2. Register a binder command, e.g. `./book/binder check lego-units` (or add a
   `lego-units` scope under `check code` if that group is a better fit — pick
   one dispatch path, document in BINDER.md).
3. Add a pre-commit hook:

```yaml
- id: book-check-lego-units
  name: "Book: check LEGO unit discipline (closed/open, fmt_qty, prose contract)"
  entry: ./book/binder check lego-units
  language: system
  files: ^book/quarto/contents/.*\.qmd$|^mlsysim/
```

4. Add hook tests in `book/tools/tests/test_lint_lego_units.py` when that file exists.

**Rollout (do not flip to blocking on day one):**

| Phase | Pre-commit behavior | As-built (2026-05-31) |
|-------|---------------------|------------------------|
| Step 12 | Linter exists; manual CLI only | **Done** |
| Step 13 | Hook installed; `default=False` or `--fail-on warning` with baseline allowlist | **Done** — hook + empty baseline |
| After Vol I queue ~50% clean | Promote high-confidence rules to `--fail-on error` on touched files | L019 only (`.m_as()`) |
| After full migration (Phase 9 prep) | Hook `default=True`; errors block commit | **`default=True` done**; warnings still `--fail-on warning` |
| After Phase 9 renders green | All L014–L017 rules block; L018 stays advisory | **Not yet** — L017 retired |

Also keep existing hooks that overlap:

- `book-check-units` — mlsysim physics unit **tests** (keep)
- `book-check-registry-sources` — hardcoded spec drift (keep)
- `book-check-math` — fmt canonical / suffix discipline (keep; complementary)

### Commands (local + CI)

```bash
pytest mlsysim/tests/test_units_registry.py
pytest mlsysim/tests/test_quantity_formulas.py
pytest mlsysim/tests/test_domain_formatters.py
pytest mlsysim/tests/test_lego_unit_invariants.py
./book/binder check lego-units
pre-commit run book-check-lego-units --all-files   # before closure
```

CI: the validate-dev workflow already runs binder default scopes; set
`default=True` on the new scope so push validation picks it up automatically
(per CLAUDE.md cumulative-check policy).

Acceptance:

- `pre-commit run book-check-lego-units` passes at closure (or only allowlisted legacy warnings documented).
- A contributor cannot commit a new closed-name/open-formatter mismatch unnoticed.

---

## Phase 8½: Gate Hardening (before Phase 9 — added 2026-05-31)

### Why this phase exists

Codex review (2026-05-31) showed that **exec-clean + lint 0 warnings is not sufficient** for merge:

- **L014 is silently broken** in `lint_lego_units.py` (line ~144): space-stripping turns `= fmt(` into `=fmt(` but the check looks for `"= fmt("`.
- **`book_check_lego_prose_units.py`** still fails **17 files** (duplicate units, closed exports in math spans).
- **OUTPUT/prose contract** incomplete: closed formatters + repeated prose units; prose `_str` inside `$...$`.
- **Scalar reattachment** still loses dimensions (e.g. TFLOP/s per W stored as TFLOP/s).
- **`fmt_percent` / range helpers** default `precision=1` conflicts with spurious-zero guard.

Phase 9 renders will surface these; fix gates first so renders are diagnostic, not whack-a-mole.

### Step 8½-A — Fix L014 and re-baseline (G1, G2) — **DONE 2026-05-31**

| Do | Gate | Status |
|----|------|--------|
| Fix L014 match: `L014_CLOSED_FMT` regex on assignment line | Unit test: `energy_kwh_str = fmt(...)` → L014 | ✓ |
| Add regression in `test_lint_lego_units.py` | pytest 7 passed | ✓ |
| Re-run linter; refresh baseline with **real** L014 counts | 81 L014 in `lego_units_baseline.json` | ✓ |
| Do **not** promote L014 to error until baseline queue has a burn-down plan | Allowlist + defer burn-down to 8½-B | ✓ |

### Step 8½-B — Prose-unit contract (G3)

Tool: `python3 book/tools/audit/book_check_lego_prose_units.py book/quarto/contents`

| Category | Fix pattern | Example |
|----------|-------------|---------|
| **L015 duplicate unit** | Remove prose unit after closed export | `cf_quebec_tonnes_str` + `tonnes CO₂` → bare ref |
| **Math span + closed export** | Move ref outside `$...$` or use open/`_math` atom | `gpu_tdp_w_str` (`700 W`) inside `$...$` |
| **Misnamed export** | Rename or pin `unit=` on formatter | `*_kg_str` that auto-scales to tonnes |

**Queue:** fix **17 files** reported by prose-units checker; pilot on `sustainable_ai.qmd` first (Codex already diagnosed).

Wire into pre-commit only after clean (or baseline + burn-down).

### Step 8½-C — Quantity integrity audit (G4)

Scan for anti-patterns:

```python
# BAD: loses /W
x = (flops_mag / tdp_mag) * TFLOP / second

# GOOD: keep rate dimensions
efficiency = (peak_flops / tdp).to(TFLOP / second / watt)
efficiency_str = fmt_qty(efficiency, TFLOP / second / watt, ...)
```

Queue: grep for `.magnitude/` followed by `* TFLOP` without `/ watt`; fix per cell; add L011/L003 lint if pattern is stable.

**Known instance:** `compute_infrastructure.qmd` `GpuEfficiencyTrajectoryRecap` (~1815).

### Step 8½-D — fmt precision defaults (G5)

| Helper | Issue | Fix direction |
|--------|-------|---------------|
| `fmt_percent` | `precision=1` default → `0.85` → `0.9` or guard error | Default `precision=None` → auto (like `fmt_multiple`) |
| `fmt_percent_range`, `fmt_qty_range`, `fmt_time_range` | Same | Same |

Add tests; do not change rendered values in chapters until defaults settled.

### Step 8½-E — Hygiene

- **Do not commit** `book/tools/audit/artifacts/lego_cells_verify_report.json` partial regen.
- Keep fmt-thread WIP (`audit_fmt_usage.py`, etc.) out of unit-hardening commits.

### Phase 8½ acceptance

- G1–G3 green (L014 detects known cases; prose-units 0 files).
- G4 queue started with compute_infrastructure pilot fix.
- G5 decision documented (defaults changed or documented author rule).
- PROGRESS.md merge-ready table updated.

**Only then** proceed to Phase 9A.

---

## Phase 9: Final Render Verification (after Phase 8½ and migration steps)

### Objective

After **every** Layer C cell in Vol I and Vol II is migrated, prove the book
**renders** — not just that pytest and lint pass. Build **one chapter at a
time**, then full volumes. This is the dress rehearsal before merging into
`fmt-fix` (Phase 10C).

### Order (same as book manifest)

Process chapters in `_quarto-html-vol1.yml` order, then `_quarto-html-vol2.yml`.
Skip frontmatter/part dividers with no LEGO unless a render fails.

### Step 9A — HTML, one chapter at a time

From `book/quarto/`:

```bash
PYTHONPATH=../..:../../mlsysim MPLBACKEND=Agg \
  quarto render contents/vol1/introduction/introduction.qmd \
  --to html --output-dir /private/tmp/mlsysbook-unit-hardening/html-vol1
```

Repeat for **every** content chapter in Vol I, then Vol II. After each render:

- No traceback in log
- Grep HTML for literal `{python}` (must be zero)
- Spot-check migrated callouts: unit strings read correctly (no duplicate/missing units)
- Optional: run rendered-output scans from preflight (`audit_lego_html.py` if available)

**Gate:** all chapters HTML-green before starting PDF pass.

### Step 9B — PDF, one chapter at a time

Same chapter order; render each to PDF:

```bash
PYTHONPATH=../..:../../mlsysim MPLBACKEND=Agg \
  quarto render contents/vol1/introduction/introduction.qmd \
  --to pdf --output-dir /private/tmp/mlsysbook-unit-hardening/pdf-vol1
```

After each PDF:

- LaTeX log: no errors, no undefined citations/refs in that chapter
- Spot-check printed unit formatting (closed exports, math symbols)

**Gate:** all chapters PDF-green before full-volume builds.

### Step 9C — Full volume builds

Build complete books (both volumes, HTML + PDF):

```bash
# From book/quarto — use project configs for full vol renders
quarto render --to html   # vol1 config / vol2 config per existing CI pattern
quarto render --to pdf
```

Run full preflight-style scans on complete outputs:

- No `{python}` literals in HTML
- No unresolved `?@` xrefs
- Margin overflow / LaTeX error scan on PDF logs
- `pytest mlsysim` + `pre-commit run --all-files` green

**Gate:** both volumes HTML + PDF build with zero errors. Only then proceed to
Phase 10.

---

## Phase 10: Sync, Re-Verify, and Merge into `fmt-fix`

### Objective

Integration happens **after** Phase 9 is green. Promotion target is **`fmt-fix`**
(not `dev` directly). Sync with `dev` on the feature branch first so the merge
into `fmt-fix` is not stale.

### Step 10A — Merge `dev` into `feat/lego-unit-hardening`

In `/Users/VJ/GitHub/MLSysBook-lego-units` (or the active unit-hardening worktree):

```bash
git fetch origin dev   # if needed
git merge --no-ff dev  # resolve conflicts here; prefer unit-hardened LEGO cells
```

If `fmt-fix` has moved ahead of the feature branch base, also merge `fmt-fix`:

```bash
git merge --no-ff fmt-fix
```

### Step 10B — Full re-verification after merge

On the merged `feat/lego-unit-hardening` tip:

1. `pre-commit run --all-files`
2. `pytest mlsysim`
3. `./book/binder check lego-units` (and promoted code scopes)
4. Phase 9C full volume HTML + PDF (minimum; re-render conflicted chapters)
5. Fix regressions — **one commit per fix**; update PROGRESS.md

### Step 10C — Promote into `fmt-fix`

Only when 10B is green, from the **`fmt-fix` checkout** (typically
`MLSysBook-fmt-fix`):

```bash
cd /Users/VJ/GitHub/MLSysBook-fmt-fix
git merge --no-ff feat/lego-unit-hardening
```

- User validates in GitKraken; push `fmt-fix` when ready (triggers validate workflow).
- **Do not** merge `fmt-fix` → `dev` as part of this plan — that stays the fmt
  thread’s separate step.
- After merge: optional `git worktree remove` + branch delete per user confirmation.

---

## Git Commit Discipline (every atomic step)

### Rule

**One atomic step = one commit.** The user requires a clean, linear record in
GitKraken — no batching Layer A + Layer C, no WIP commits spanning multiple
steps.

### Before each commit

```bash
git add <specific-files-only>   # never git add -A
pre-commit run --files <those files>   # or full pre-commit at closure milestones
pytest mlsysim                  # Layer A/B steps
```

Update [`mlsysim-lego-unit-hardening-PROGRESS.md`](mlsysim-lego-unit-hardening-PROGRESS.md):
check off completed step, append log entry, set **Current step** / **Next action**.

### Commit message shape

```text
<area>: <what> (unit-hardening step N)

Examples:
mlsysim: add unit registry characterization tests (step 2)
mlsysim: load mlsysbook_units.txt (step 3)
book: wire lego-units pre-commit hook warning-only (step 13)
vol1/introduction: harden ScenarioFoo LEGO cell (step 14)
vol1/training: harden BarScenario LEGO cell (step 892)
book: promote lego-units lint to pre-commit error (closure)
```

Use imperative mood. No `Co-Authored-By`. No vendor footers.

### When to commit

| Layer | Commit after | Progress file |
|-------|----------------|---------------|
| A (mlsysim) | Each step 2–10 | Check step; log discoveries |
| B (lint/docs) | Each step 11–13 | Check step; note baseline counts |
| C (QMD) | **Each LEGO class** | Row in migration table + step log |
| Closure | Phase 9/10 milestones | Check Phase boxes; log render evidence |

### Do not

- `--no-verify` unless user explicitly requests
- Amend commits that failed pre-commit (new commit instead)
- Combine unrelated cells or chapters in one commit
- Merge `dev` mid-migration (wait for Phase 10 on the feature branch)

---

## Existing Diff Issues To Carry Forward

These came from inspecting the current `fmt-fix` branch diff against local
`dev`. Do not fix them in this planning pass, but include them in the next
implementation or review pass.

### `fmt_usd_range` Removed Constant Bug

Current behavior:

```python
fmt_usd_range(10_000, 30_000, scale="K", repeat_symbol=False)
```

raises `NameError` because `fmt_usd_range` still references `_USD_SCALES`, but
the newer structured scale implementation uses `_resolve_decimal_scale(...)` and
`_DECIMAL_SCALE_FACTORS`.

Cleanup:

- Replace the stale `_USD_SCALES` branch with `_resolve_decimal_scale(...)`.
- Add a regression test for `repeat_symbol=False` plus each supported scale.
- Add a test that word scales either work intentionally or are rejected with a
  clear error in this branch.

### `fmt_multiple` Default Precision Surprise

Current behavior:

```python
fmt_multiple(2.0)
```

raises the spurious-zero precision guard because `fmt_multiple` defaults to
`precision=1`. That is mathematically defensible but poor author ergonomics for
common speedups like `2x`.

Cleanup:

- Change `fmt_multiple` to use automatic precision when `precision=None`.
- Keep explicit `precision=1` strict.
- Add tests for `fmt_multiple(2)`, `fmt_multiple(2.0)`, and `fmt_multiple(3.2)`.

### `sci_latex` Is Useful But Too Low-Level

Why it exists:

- `fmt_sci` emits Unicode display text such as `4.10 × 10⁹`.
- `sci_latex` emits a TeX atom such as `4.10 \times 10^{9}` for use inside
  `fmt_math(...)`, `fmt_frac(...)`, and equation strings.

Problem:

```python
sci_latex(quantity)
```

uses only `quantity.magnitude`, so unit choice has already been decided by the
caller, whether intentionally or accidentally.

Cleanup:

- Keep `sci_latex` but document it as a low-level TeX atom helper.
- Consider renaming or aliasing it to `sci_latex_atom`.
- Add `sci_qty_latex(quantity, display_unit, precision=2)` so scientific
  notation for quantities has the same unit discipline as `fmt_qty`.
- Add linter rule L013 for direct `sci_latex(quantity)` without an explicit
  `.to(...)`.

### `mlsysim.__init__` Formatter Export Drift

`math_canonical.py` treats many `fmt` helpers as available from `from mlsysim
import *`, but `mlsysim/__init__.py` currently exports only a subset. Chapters
usually compensate with direct `from mlsysim.fmt import ...` imports, but the
surface is inconsistent.

Cleanup:

- Decide whether all book-facing formatters should be exported from
  `mlsysim.__init__`.
- If yes, export `fmt_qty_int`, `fmt_val`, `fmt_unit`, `fmt_sci`, `fmt_frac`,
  `sci_latex`, and any new domain formatters.
- Add an import-surface test that `math_canonical.MLSYSIM_STAR_FMT_NAMES` and
  `mlsysim.__all__` do not drift.

### Generic Range Physical Unit Backdoor

`fmt_range(..., unit="GB")` appends a free-text unit with no Pint dimension
check. That is acceptable for generic text ranges, but it should not become the
preferred path for physical quantities.

Cleanup:

- Prefer `fmt_qty_range` and `fmt_time_range` for physical ranges.
- Add linter rule L012 for physical-looking `fmt_range(unit=...)`.
- Add domain range helpers only where recurring prose needs them.

## LEGO Stages and Unit Hardening — Keep the Four Stages

**Do not rename or replace LOAD → EXECUTE → GUARD → OUTPUT.** The original
structure is still correct. Unit hardening tightens *what belongs in each stage*,
not the stage names.

The new closed/open naming convention lives almost entirely inside **OUTPUT**.
LOAD and EXECUTE become quantity-first; GUARD stays invariant checks; OUTPUT
becomes the only place that decides closed vs open strings.

| Stage | Unit-hardening role | Holds quantities? | Prose-facing exports? |
|---|---|---|---|
| **LOAD** | Registry values, scenario inputs; attach units immediately | yes | no |
| **EXECUTE** | Physics via Pint + formula helpers; no formatting | yes | no |
| **GUARD** | `check()` / `assert_qty_close()`; may use `_val` scalars | optional `_val` | no |
| **OUTPUT** | One typed formatter per export; closed vs open naming | input is `Quantity` | yes (`*_str`) |

Optional comment upgrade in migrated cells (cosmetic, not required):

```python
# ┌── 4. OUTPUT (Formatting — closed / open) ─────────────────────────────
```

Do **not** add a fifth stage (e.g. "FORMAT" vs "OUTPUT"). Do **not** split
OUTPUT into sub-stages in the cell header — closed vs open is a naming/formatter
choice within the existing OUTPUT block.

### Per-stage checklist (use when migrating one cell)

**LOAD**

- Registry specs stay as `Quantity` objects.
- Local assumptions: `14 * day`, `10.3 * kilowatt`, not bare floats with manual
  units later.
- Scalars only for external APIs → `*_{unit}_val`, never `*_str`.

**EXECUTE**

- All physical math on `Quantity` objects or formula helpers returning
  `Quantity`.
- No `fmt_*`, no `_str` exports, no `.m_as()` unless feeding plot/API/check
  (`*_val`).

**GUARD**

- Prefer comparing quantities (`assert_qty_close`) over bare floats.
- Acceptable interim: `.to(unit).magnitude` inside `check()` only.

**OUTPUT**

- Pick export class per value: **closed-fixed** (`tdp_w_str`), **closed-auto**
  (`facility_energy_str`), **open** (`latency_str` + prose ` ms`), or **typed**
  (`cost_str` via `fmt_usd`).
- One formatter per export; physical closed-fixed passes `Quantity` to
  `fmt_qty`/domain formatter — never `fmt()` on a scalar.
- Align **name + formatter + prose** in the same edit when changing class.

### One-cell migration order (within a single LEGO cell)

When hardening one cell, touch stages in this order so renders stay green after
each micro-commit if desired:

```text
1. EXECUTE — quantity-first physics (prose unchanged; old OUTPUT still works)
2. GUARD   — update checks if magnitudes shifted slightly
3. OUTPUT  — closed/open exports + prose fix in the same commit
4. Render  — verify the callout that references this class
```

Step 1 alone should not break prose if OUTPUT exports are unchanged. Step 3 must
include prose when switching open → closed-fixed.

---

## Atomic Execution Steps (one concern per step)

**Operating rules**

- **One step = one merge-worthy unit of work = one git commit.**
- **After every step:** update [`mlsysim-lego-unit-hardening-PROGRESS.md`](mlsysim-lego-unit-hardening-PROGRESS.md) (checklist, log entry, next action).
- **Three layers never mix in one step:**
  - **Layer A** — `mlsysim/` + tests only
  - **Layer B** — linter, rule docs, CI hooks
  - **Layer C** — one LEGO cell (+ its prose), or one chapter at most after
    pilots prove the pattern
- **No QMD edits until Step 12** (first pilot cell). Infrastructure first.
- After Step 12, every step is **one cell or one callout block**, not a whole
  chapter sweep.

### Layer A — mlsysim foundation (Steps 1–10, no QMD)

| Step | Do exactly this | Gate before next step |
|---:|---|---|
| **0** | Create `feat/lego-unit-hardening` + worktree `MLSysBook-lego-units` from `fmt-fix` tip | `git worktree list` shows new path; **commit** plan/progress bootstrap if any |
| **1** | Create PROGRESS.md + baseline SHA, pytest, regressions | progress file created |
| **2** | Add `test_units_registry.py` for **current** registry behavior (characterization tests) | pytest pass |
| **3** | Add `mlsysbook_units.txt`; load from `units.py`; keep existing `ureg.define` until tests prove equivalence | pytest pass; import mlsysim works |
| **4** | Add exported unit aliases (`mJ`, `MW`, `kWh`, `kilogram`, …) + alias equality tests | pytest pass |
| **5** | Add `physics/quantities.py` with `transfer_time`, `compute_time` only | pytest pass |
| **6** | Add `energy_from_power`, `carbon_from_energy`, `memory_from_params` + `test_quantity_formulas.py` | pytest pass |
| **7** | Fix `fmt_usd_range` / `_USD_SCALES` regression + test | pytest pass |
| **8** | Fix `fmt_multiple` default precision + test | pytest pass |
| **9** | Add `fmt_power`, `fmt_energy` + tests (first domain batch) | pytest pass |
| **10** | Add remaining domain formatters in small batches (bandwidth, memory, emissions, …) one helper per commit if needed | pytest pass per helper |

### Layer A′ — LOAD registry-first (parallel to A; zero QMD until wired)

| Step | Do exactly this | Gate before next step |
|---:|---|---|
| **A′-1** | Audit QMD LOAD for numeric literals that duplicate registry YAML | inventory in PROGRESS |
| **A′-2** | Promote model training-energy fields (`training_energy_mwh`, …) from `float` → `Quantity` in `models/types.py` + loader tests | pytest pass |
| **A′-3** | Add missing comparison anchors to `Literature` (e.g. transatlantic flight CO₂e) with provenance + tests | pytest pass |
| **A′-4** | Optional: export grid carbon intensity as `Quantity` (gram/kWh) so LOAD stops using float `*_kg_kwh` properties | pytest pass |

### Layer B — contract and lint (Steps 11–13, still no QMD)

| Step | Do exactly this | Gate before next step |
|---:|---|---|
| **11** | Document closed vs open naming in `.claude/rules/lego-units.md`; cross-ref `fmt.md` §6–§7 | doc review |
| **12** | Add `lint_lego_units.py` warning-only; run on vol1+vol2; commit **baseline warning counts** | linter runs; no errors |
| **13** | Wire `./book/binder check lego-units` + `book-check-lego-units` pre-commit hook (warning-only + baseline) | hook runs; commit |

### Layer C — QMD migration (Step 14+, book order, one cell per step)

**Queue rule:** Volume I chapters in [`_quarto-html-vol1.yml`](book/quarto/config/_quarto-html-vol1.yml)
render order, then Volume II in [`_quarto-html-vol2.yml`](book/quarto/config/_quarto-html-vol2.yml)
render order. Within each `.qmd`, cells top-to-bottom. See Phase 7 for the full
chapter lists.

| Step | Do exactly this | Gate before next step |
|---:|---|---|
| **14** | First LEGO class in `vol1/introduction/introduction.qmd` (document order) | pre-commit on touched files; lint; render chapter; **commit** |
| **15+** | Next LEGO class in same chapter; when chapter done, advance to next `.qmd` in Vol I queue | same; **commit per class** |
| **…** | Continue through all Vol I content chapters | Vol I queue complete |
| **…** | Start Vol II at `vol2/introduction/introduction.qmd`; same one-class-per-commit rule | Vol II queue complete |
| **…** | Promote `book-check-lego-units` to `--fail-on error` incrementally | pre-commit blocks regressions |
| **9A** | HTML render **every** chapter Vol I then Vol II, one at a time; check each output | all chapters HTML-green; commit baseline log if needed |
| **9B** | PDF render **every** chapter, same order; check LaTeX logs | all chapters PDF-green |
| **9C** | Full volume HTML + PDF both vols; pre-commit + pytest full suite | closure commit |
| **10A** | Merge `dev` (and `fmt-fix` if needed) into `feat/lego-unit-hardening` | merge commit |
| **10B** | Full re-verify on feature branch | fix commits as needed |
| **10C** | Merge `feat/lego-unit-hardening` → `fmt-fix` (`--no-ff`) | user validates; optional worktree cleanup |

Do **not** skip to `sustainable_ai.qmd` or other high-risk chapters early.

### What each pilot cell step contains (never split across steps)

For one cell in one step:

0. **LOAD (registry-first):** replace chapter literals with registry pulls; add
   missing specs to MLSysIM in a prior A′ commit if needed
1. EXECUTE: quantities + formula helpers where applicable
2. GUARD: update if needed
3. OUTPUT: closed/open exports with correct names
4. Prose: fix only `{python}` refs in the paragraph(s) immediately below that cell
5. Render that chapter (or callout grep for the exported strings)

Do **not** in the same step: alias cleanup across the chapter, suffix-thread
formatters, unrelated cells, or rule-doc edits.

### Prose scope — what this plan checks and cleans up

Unit hardening **does** cover prose where OUTPUT values appear, but only the
**unit/glyph contract** around inline `` `{python} ClassName.export_str` `` refs —
not general editorial prose rewrites.

**In scope (checked and fixed during Layer C migration):**

| Check | How | Example |
|---|---|---|
| Duplicate unit on closed export | L015 + manual fix in same commit as OUTPUT | `` `{python} tdp_w_str` W `` → bare ref when string is `"700 W"` |
| Missing unit on open export | L018 (advisory) + manual fix | `` `{python} latency_str` ms `` when export is magnitude-only |
| Duplicate `$\times$` on multipliers | L010 | `` `{python} speedup_str`$\times$ `` when `style="symbol"` already includes glyph |
| Typed formatter glyph duplication | existing fmt migration / `fmt.md` §7 | do not add `$` after `fmt_usd`; do not add `%` after `fmt_percent(style="symbol")` |
| Locality of refs | existing `validate_inline_refs.py` | ref must appear after defining cell |
| Rendered ground truth | quarto render of pilot callout/chapter | grep exported strings in HTML; no `{python}` literals left |

**Migration rule:** when OUTPUT changes closed ↔ open, **prose in the same
callout/paragraph block** that references that cell's exports must be updated in
the **same step** — never change the cell alone and leave prose carrying a unit
the formatter now owns (or vice versa).

**Out of scope for this plan (other tracks):**

- Voice, tone, sentence craft, footnotes, cross-refs
- Full-chapter prose audit unrelated to unit contract
- Table/figure YAML captions with `{python}` (forbidden pattern; separate fix)
- Suffix-cleanup / full `fmt_*` migration ledger (`book/tools/audit/fmt/`) unless
  the same callout is already being touched for units

**Three-layer prose verification (use at Step 14+):**

```text
1. Static — lint_lego_units.py pairs cell exports with nearby prose (L015, L018, L010)
2. Migration — human/agent edits prose in the locality window below the cell
3. Render — quarto HTML/PDF shows correct combined string (700 W per chip, not 700 W W)
```

L015 scans prose **within the same paragraph or callout** as the inline ref, not
the whole chapter. A corpus-wide prose cleanup happens **one cell at a time** as
Layer C steps, not as a separate bulk prose pass.

### When to promote linter warnings to errors

One rule per step, only after migrated cells in the queue pass that rule:

1. L001, L003, L011 (physics/scalar mistakes)
2. L014, L016, L017 (closed/open naming)
3. L015 (prose duplicate unit)
4. L009, L018 (style / advisory)

---

## Suggested Implementation Order (summary)

The Atomic Execution Steps table above is authoritative. Short form:

1. Characterization tests for current units (Step 2).
2. Pint definition file + load (Step 3).
3. Exported aliases + tests (Step 4).
4. Formula helpers + tests (Steps 5–6).
5. Fix fmt regressions on this branch (Steps 7–8).
6. Domain formatters one batch at a time (Steps 9–10).
7. Document closed/open contract (Step 11).
8. Warning-only linter + baseline (Steps 12–13).
9. Book-order QMD migration: **Vol I manifest order**, then **Vol II**; one LEGO
   class per **commit** (Step 14+).
10. Promote `book-check-lego-units` pre-commit to blocking incrementally.
11. Phase 9: HTML per chapter → PDF per chapter → full volume builds.
12. Phase 10: sync dev on feature branch, re-verify, merge into `fmt-fix`.

Do not start by editing all QMD files. Do not combine Layer A and Layer C in one
step. Do not skip chapter order to hit high-risk files early.

## Detailed Acceptance Criteria

The hardening pass is done when all of these are true:

- Custom MLSysBook units are centralized in a Pint definition file or an equivalent documented unit registry module.
- `kWh`, `MWh`, and `GWh` are supported and exported cleanly.
- Common book units have exported aliases, so QMD cells do not need direct `ureg.*` for `joule`, `mJ`, `pJ`, `MW`, `kWh`, `MWh`, `kilogram`, `metric_ton`, `kilometer`, and similar common units.
- Carbon intensity can be represented as a Pint quantity such as `429 g/kWh`.
- Carbon output can be represented as a mass quantity and formatted consistently.
- Formula helpers exist for transfer time, compute time, energy, carbon, and parameter memory.
- Domain formatters exist for memory, bandwidth, FLOP rate, latency, duration, params, tokens, power, energy, emissions, and carbon intensity.
- Tests prove the H100 registry quantities convert correctly.
- Tests prove common dimensional mistakes raise.
- Tests prove common displayed strings are stable.
- A LEGO unit linter exists and reports at least the high-confidence bad patterns.
- The linter warns when direct `ureg.*` usage has a preferred exported alias.
- The linter warns when `.m_as(...)` scalar extraction is followed by
  unit reattachment for display or downstream physics.
- `fmt_multiple` has a documented glyph ownership policy and tests cover missing
  and duplicate `times` glyphs.
- Scientific notation helpers make explicit whether they emit prose text,
  TeX atoms, or quantity-aware TeX.
- All Vol I then Vol II content chapters processed in Quarto manifest order; each
  renders after its cells are migrated.
- `book-check-lego-units` pre-commit hook blocks new unit-discipline regressions
  at closure (`default=True`, `--fail-on error`).
- Phase 9 complete: every chapter HTML + PDF green; full volume HTML + PDF green.
- Phase 10 complete: `feat/lego-unit-hardening` merged into `fmt-fix` with full
  re-verification green.
- The implementation does not require authors to remember raw suffix strings for common physical units.
- Display-unit naming is documented as a **closed vs open** binary for physical
  exports, plus a separate typed closed track for USD/percent/count.
- The linter enforces both directions: closed name requires closed formatter
  (L014, L016); closed formatter on plain `_str` requires rename (L017); prose
  must not duplicate closed units (L015).

## Risk Register

### Risk: Unit Definition File Breaks Package Loading

Mitigation:

- Add import tests.
- Use package-data-safe loading.
- Keep old `ureg.define(...)` calls until the definition file is proven.

### Risk: Time Alias Changes Break Existing QMD

Mitigation:

- Do not remove `MS`, `US`, `NS` immediately.
- Normalize display labels.
- Warn in linter before migrating.

### Risk: Alias Cleanup Becomes Cosmetic Churn

Mitigation:

- Add aliases and tests first.
- Keep direct `ureg.*` valid for obscure units.
- Make the linter warning-only at first.
- Do not mass-rewrite QMD only for alias style while another formatter cleanup is in flight.
- Update aliases opportunistically in cells already being touched.

### Risk: Auto-Scaling Changes Prose Unexpectedly

Mitigation:

- Start domain helpers in new code and pilots only.
- Do not swap all `fmt_qty` calls globally.
- Add golden display tests.

### Risk: Token And Param Units Become Too Clever

Mitigation:

- Use `fmt_params` and `fmt_tokens` first.
- Only add Pint `token` if there is a clear formula need.
- Keep `param` because it is already in the registry and used in models.

### Risk: Carbon Contexts Hide Policy

Mitigation:

- Use explicit `carbon_from_energy(energy, intensity)`.
- Do not use Pint contexts for carbon in the initial implementation.

### Risk: Linter Produces Too Many Warnings

Mitigation:

- Start warning-only.
- Add allowlist comments if needed.
- Promote only high-confidence rules to errors.

### Risk: Legacy `_w_str` Names Imply Unit-In-String Before Migration

Mitigation:

- Treat L014/L015 as warning-only until pilot chapters are migrated.
- When touching a cell, fix name + formatter + prose together (ArchetypeATdp
  pattern).
- Do not mass-rename exports for cosmetic consistency alone.

## Design Decisions To Ask The User Before Finalizing

Ask only if implementing, not while merely planning.

1. Should time display use `us`, `μs`, or `microseconds` in compact prose?
2. Should emissions display include `CO2e` inside the formatter or should prose add it?
3. Should `fmt_energy(Q_("1287 MWh"))` stay `1,287 MWh` or auto-scale to `1.29 GWh`?
4. Should `fmt_params(150e6, scale="B")` render `0.15B` or refuse the forced scale?
5. Should memory default to decimal `GB` for model sizes and binary `GiB` only for hardware capacity?
6. Should `token` become a Pint unit or remain a `fmt_count`/`fmt_rate` domain concept?
7. Should `fmt_multiple(style="symbol")` emit `3.2$\times$` or a literal
   Unicode `3.2×`?
8. Should `sci_latex` stay as a public helper, be renamed to
   `sci_latex_atom`, or be wrapped by a quantity-aware `sci_qty_latex` and
   treated as low-level?

Recommended defaults:

- Compact time: `us`, `ms`, `ns` in symbols; word style for prose when requested.
- Emissions formatter should include mass unit only; prose can say `CO2e` unless a dedicated `fmt_co2e` helper is added.
- Energy auto-scaling should be conservative: use `MWh` up to `9999 MWh`, then `GWh`.
- Forced param scale should preserve nonzero magnitude by raising precision.
- Hardware memory capacity can use `GiB`; model memory can use decimal `GB`.
- Keep tokens as count/rate helpers first; add Pint token later only if formula helpers need it.
- Prefer `fmt_multiple(style="symbol")` with the LaTeX-safe `$\times$` form for
  Quarto prose unless render tests prove Unicode `×` is equally safe across
  HTML and PDF.
- Keep `sci_latex` for compatibility, but add `sci_qty_latex(quantity,
  display_unit)` and document `sci_latex` as a low-level TeX atom helper.

## Suggested Agent Prompt

The next agent can be given this prompt:

```text
You are working on branch `feat/lego-unit-hardening` in worktree
`/Users/VJ/GitHub/MLSysBook-lego-units` (create via Step 0 if missing).
Parent integration branch is `fmt-fix` in `/Users/VJ/GitHub/MLSysBook-fmt-fix`.
Do not edit `/Users/VJ/GitHub/MLSysBook`. Merge target when done: `fmt-fix`
(Phase 10C), not `dev`.
Implement the unit-hardening plan in mlsysim-lego-unit-hardening-plan.md.
Follow **Atomic Execution Steps** one step at a time; **one git commit per step**.
Maintain [`mlsysim-lego-unit-hardening-PROGRESS.md`](mlsysim-lego-unit-hardening-PROGRESS.md)
after every step (checklist + log + discoveries). No shortcuts: missing units
→ MLSysIM registry first; missing formulas → physics/quantities.py; no QMD
literals. Do not combine Layer A/B/C. Wire pre-commit via Phase 8. Phase 9
HTML/PDF per chapter then full volumes; Phase 10 merge dev and re-verify.
Start at Step 1 (create progress file) unless resuming from PROGRESS.md.
Book order Vol I → Vol II; one LEGO class per commit. Run pre-commit before
each commit.
```

## Appendix: Codebase Audit — Do Once, Extend Don't Duplicate

Scan findings (2026-05) — add these to the plan so work is not repeated or
forked.

### 1. Existing binder checks — extend, do not rewrite

Several unit-adjacent checks **already exist** but are **`default=False`**
(opt-in). The new work should **add scopes to `./book/binder check code`**, not
a parallel standalone linter unless a single orchestrator wraps these:

| Existing module | Binder scope | What it catches | Gap vs plan |
|-----------------|--------------|-----------------|-------------|
| [`book_check_lego_prose_units.py`](book/tools/audit/book_check_lego_prose_units.py) | `lego-prose-units` | Unit/currency after `` `{python} *_str` `` | Assumes `fmt(..., suffix=)`; must upgrade for **closed-fixed `fmt_qty`** (L015) |
| [`book_check_lego_load_pint.py`](book/tools/audit/book_check_lego_load_pint.py) | `lego-load-pint` | Bare floats on physical `*_value` in LOAD | Uses `*_value` not `*_val`; align naming convention |
| [`book_check_lego_equations.py`](book/tools/audit/book_check_lego_equations.py) | `lego-equations` | A/B=C prose vs computed values | Keep; run per chapter at render gate |
| [`book_check_registry_sources.py`](book/tools/audit/book_check_registry_sources.py) | `registry --scope sources` | Hardcoded specs, legacy flat constants | Already pre-commit; keep |
| [`math_canonical.py`](book/cli/checks/math_canonical.py) | `check math` | fmt family, suffix discipline, double-wrap | Add rules for `fmt_qty(.m_as())`, closed/open name (L001, L014, L017) |
| [`fmt_semantic_suffix.py`](book/cli/checks/fmt_semantic_suffix.py) | `suffix-semantics` | Typed formatters vs raw `suffix=` | **Fmt migration thread** — flip `default=True` only at closure, not mid-unit pass |
| [`book-check-units`](.pre-commit-config.yaml) | `check units` | mlsysim physics unit **tests** | Complements; not LEGO static lint |

**Revised Step 12–13:** implement missing rules (L001, L003, L011, L014–L017) as
**new binder scopes** under `check code` (e.g. `lego-unit-discipline`) that
reuse one QMD cell parser; wire **one** pre-commit hook. Deprecate the plan's
standalone `book/tools/scripts/lint_lego_units.py` path unless it is a thin CLI
over the binder scopes.

**Promotion schedule:** as PROGRESS checklist advances, flip scopes to
`default=True` in [`validate.py`](book/cli/commands/validate.py) one at a time
(record baseline counts in PROGRESS.md first).

### 2. Refactor existing physics helpers — don't fork `quantities.py` blindly

[`physics/memory.py`](mlsysim/mlsysim/physics/memory.py) already has
`model_memory()` but it returns a **float magnitude**, not a `Quantity` — the
old pattern this plan replaces. Before adding `physics/quantities.py`:

- Audit [`physics/performance.py`](mlsysim/mlsysim/physics/performance.py)
  (`dTime`, `calc_bottleneck`), [`physics/memory.py`](mlsysim/mlsysim/physics/memory.py),
  [`physics/_units.py`](mlsysim/mlsysim/physics/_units.py).
- Prefer **Quantity-returning v2 helpers** (or refactor in place with tests) over
  a parallel module that duplicates names.
- Migrate chapters to v2 helpers during Layer C; do not leave two APIs forever.

There is **no** `core/formulas.py` on this branch — do not assume it exists.

### 3. Add `assert_qty_close` early (Step 6 or 7)

Not in codebase yet. GUARD blocks still use bare float comparisons. Add to
[`fmt.py`](mlsysim/mlsysim/fmt.py) before heavy Layer C migration; export from
[`mlsysim/__init__.py`](mlsysim/mlsysim/__init__.py).

### 4. Fix `mlsysim.__init__` export surface (Step 8)

[`__init__.py`](mlsysim/mlsysim/__init__.py) omits `fmt_val`, `fmt_unit`,
`fmt_sci`, `sci_latex`, and future domain formatters that chapters import ad hoc.
Add an **import-surface test** matching `math_canonical.MLSYSIM_STAR_FMT_NAMES` so
star-import drift cannot recur.

New unit aliases must export through the path chapters use (`from mlsysim import *`
→ [`core/constants.py`](mlsysim/mlsysim/core/constants.py) star export chain).

### 5. Step 14 prep — cell inventory automation

Before first QMD migration, generate the ordered migration queue into
PROGRESS.md:

- Walk Vol I + Vol II manifest order.
- List every LEGO class (grep `# ┌── LEGO` / `class Name:` in `{python}` cells).
- ~1,500+ `.m_as(` sites corpus-wide; ~36 chapters affected — inventory prevents
  skipped cells.

Reuse patterns from [`lego_focal_verify.py`](book/tools/audit/lego_focal_verify.py).

### 6. High-density chapters (expect longer queues)

Corpus `.m_as(` counts (indicative workload):

| Chapter | ~`.m_as(` hits |
|---------|----------------|
| `compute_infrastructure.qmd` | 157 |
| `hw_acceleration.qmd` | 117 |
| `training.qmd` | 100 |
| `data_storage.qmd` | 74 |
| `ml_systems.qmd` | 78 |
| `sustainable_ai.qmd` | 34 |

Plan time accordingly; do not rush these chapters.

### 7. Energy/carbon `/ THOUSAND` pattern

Many cells use `facility_energy_kwh * grid_ci_g_per_kwh / THOUSAND` instead of
Pint carbon math — especially `sustainable_ai`, `compute_infrastructure`,
`responsible_engr`. **`carbon_from_energy` + `fmt_emissions`** are the highest-leverage
helpers; prioritize in Steps 5–10.

### 8. Fmt migration coordination (avoid doing twice)

Active parallel thread: [`book/tools/audit/fmt/`](book/tools/audit/fmt/) (suffix
cleanup, typed formatters). Rules:

- Unit hardening owns **Pint discipline + closed/open physical exports**.
- Fmt migration owns **value-kind suffix cleanup** (`fmt_percent` vs `suffix=`).
- If a cell is open in fmt AUDIT_LEDGER, **skip or merge** in one visit — log in
  both PROGRESS.md and AUDIT_LEDGER when both apply.
- Flip `suffix-semantics` to pre-commit **only at global closure** (after Phase 9).

### 9. Phase 9 — enable render-truth checks

Turn on existing opt-in scopes during closure:

- `rendered-python-leak` (`check code --scope rendered-python-leak`)
- `math --scope render-audit` (expensive; run at Phase 9C not per cell)

### 10. `lego-prose-units` upgrade checklist

When migrating to closed-fixed exports, update
[`book_check_lego_prose_units.py`](book/tools/audit/book_check_lego_prose_units.py) to:

- Detect `fmt_qty` / domain formatter exports with `_*_{unit}_str` names.
- Flag prose unit tokens after those refs (L015).
- Flag **missing** prose units for open exports (L018 partial — today only catches duplicate).

Do this in Step 12 **before** Layer C, so migration gets immediate feedback.

---

## Minimal First PR Scope

If the next agent needs a smaller first chunk, use this scope:

1. Add `mlsysim/mlsysim/core/mlsysbook_units.txt`.
2. Load it from `units.py` while preserving existing exports.
3. Add common exported aliases for book units such as `mJ`, `pJ`, `MW`, `Wh`, `kWh`, `MWh`, `kilogram`, `metric_ton`, and `kilometer`.
4. Add `test_units_registry.py`.
5. Add Quantity-returning helpers (audit `physics/memory.py` first; extend or add
   `physics/quantities.py`):
   - `transfer_time`
   - `compute_time`
   - `energy_from_power`
   - `carbon_from_energy`
   - `memory_from_params`
6. Add `assert_qty_close` to `fmt.py` + test.
7. Add `test_quantity_formulas.py`.
8. Do not touch QMD files yet.

That first PR gives the book a safer foundation without creating chapter churn.
