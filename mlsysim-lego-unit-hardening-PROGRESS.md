# MLSysBook LEGO Unit Hardening — Progress

**Branch:** `fmt-fix`
**Worktree:** `/Users/VJ/GitHub/MLSysBook-fmt-fix`
**Started:** 2026-05-31

## Checklist

| Step | Status | Notes |
|------|--------|-------|
| 1 — Baseline + PROGRESS | DONE | SHA `34e4f12ace80`; pytest green; regressions documented |
| 2 — test_units_registry.py | DONE | Characterization tests |
| 3 — mlsysbook_units.txt | DONE | Loaded from units.py |
| 4 — Book-facing aliases | DONE | mJ, MW, kWh, kg, km, … |
| 5–6 — physics/quantities.py | DONE | Formula helpers + assert_qty_close |
| 7–8 — fmt_usd_range, fmt_multiple | DONE | _USD_SCALES fix; precision=None auto |
| 9–10 — Domain formatters | DONE | fmt_power, fmt_energy, … |
| 11–13 — Docs + lint + CI | DONE | lego-units.md; lint_lego_units.py; binder hook |
| A′ — LOAD registry-first | IN PROGRESS | Schema + Literature anchors; see below |
| 14+ — QMD migration | IN PROGRESS | Pilots: ArchetypeATdp, CarbonFrontier; CarbonCostGPT3 next |

## Current

- **Last step:** 15 — CarbonFrontier pilot (unit discipline)
- **Next:** A′-2/A′-3 registry prep, then **CarbonCostGPT3** (first cell in file — LOAD + E + O + prose in one visit)

## LOAD registry-first (Layer A′)

**Principle:** LOAD pulls authoritative facts from MLSysIM; EXECUTE never re-types them.

| LOAD kind | Source | Example |
|-----------|--------|---------|
| Hardware / model / grid specs | `Hardware.*`, `Models.*`, `Infrastructure.*` | H100 TDP, GPT-3 training energy, US grid CI |
| Cited field / comparison anchors | `Literature.*` | Transatlantic flight CO₂e (~1000 kg, provenance TBD) |
| Pedagogical scenarios | Local in LOAD, **commented** as scenario-only | `CarbonFrontier` 10 GWh hypothetical run |
| Never | Bare literals for specs that exist (or should exist) in registry | `energy_mwh = 1287` when `Models.Language.GPT3.training_energy_mwh` exists |

| Step | Status | Deliverable |
|------|--------|-------------|
| A′-1 | pending | Audit QMD LOAD literals that duplicate registry YAML |
| A′-2 | pending | Promote `training_energy_mwh` (and peers) from `float` → `Quantity` in `models/types.py` |
| A′-3 | pending | Add `Literature` sustainability anchors (flight CO₂e) + provenance + tests |
| A′-4 | pending | Grid carbon intensity as `Quantity` at LOAD (gram/kWh or kg/kWh) |

**Per-cell rule (Layer C):** registry addition (if needed) → LOAD pull → E → G → O+prose → lint → render → commit.

### CarbonCostGPT3 — pending migration checklist

| Issue | Today | Target |
|-------|-------|--------|
| LOAD `energy_mwh = 1287` | Chapter literal | `Models.Language.GPT3.training_energy` (Quantity) |
| LOAD `kg_per_flight = 1000` | Chapter literal | `Literature.*` flight anchor |
| EXECUTE | `energy_kwh * grid_ci_us_kg` floats | `carbon_from_energy(energy, intensity)` |
| OUTPUT | Mixed closed names + `fmt()` | See fmt audit below |
| Prose | Repeats `kg`, `kg/kWh` after closed-name exports | Bare refs once exports are closed |

### CarbonCostGPT3 OUTPUT fmt audit (lines 156–164, **not yet migrated**)

| Export | Formatter today | Contract | Verdict |
|--------|-----------------|----------|---------|
| `training_mwh_str` | `fmt_qty(..., MWh)` | Closed-fixed | Formatter OK; E stage still reattaches unit from float — fix with registry `Quantity` + `fmt_energy` |
| `energy_kwh_str` | `fmt_qty(..., kWh)` | Closed-fixed | Same |
| `total_emissions_kg_str` | `fmt()` | Name is closed (`_kg_str`) | **Wrong** — use `fmt_emissions()`; drop prose `kg` |
| `grid_ci_us_kg_str`, `grid_ci_low_kg_str` | `fmt()` | Closed names | **Wrong** — use `fmt_qty(..., kg/kWh)` or open `grid_ci_us_str` + prose ` kg/kWh` |
| `kg_per_flight_str` | `fmt()` | Closed name | **Wrong** — use `fmt_emissions()` or `fmt_qty(..., kilogram)` |
| `flight_ratio_str`, `emissions_ratio_str` | `fmt()` | Dimensionless | **OK** (open); prose owns `×` |
| `low_carbon_flights_str` | `fmt()` | Count-like | Open OK; or `fmt_count(..., label="passenger round trip")` |

**Staged by design:** only `ArchetypeATdp` and `CarbonFrontier` are migrated so far. `CarbonCostGPT3` is still pre-hardening legacy — lint baseline allows it until Step 16.

## Baseline (Step 1)

- **SHA:** `34e4f12ace80e2d877b55bc6abf07ecffee179a4`
- **pytest:** `python3 -m pytest mlsysim/tests -q` — all pass (114 fmt tests)
- **Known regressions (pre-fix):**
  - `fmt_usd_range(..., repeat_symbol=False, scale="K")` → `NameError: _USD_SCALES`
  - `fmt_multiple(2)` → spurious-zero precision error at default precision=1

## Migration table (Layer C)

| Chapter | Class | Step | Status |
|---------|-------|------|--------|
| vol2/sustainable_ai | ArchetypeATdp | 14 | DONE |
| vol2/sustainable_ai | CarbonFrontier | 15 | DONE |
| vol2/sustainable_ai | CarbonCostGPT3 | 16 | pending |

## Step log

### Step 1 — 2026-05-31

- Created PROGRESS ledger
- Documented baseline SHA and fmt regressions
