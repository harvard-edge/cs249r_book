# LEGO Audit Sign-Off — 2026-06-06 (updated)

## Scope

Both volumes; P0–P2 fixes applied; **P6 arithmetic intensity + P5 length/speed** applied in follow-up pass.

## Automated gates (post-fix)

| Gate | Result |
|------|--------|
| Category A (closed name + open fmt) | **0** remaining |
| Category C (_b/_m + bare fmt) | **0** remaining |
| fmt_prose_contract | **0** violations |
| lego-prose-units (corpus) | **PASS** |
| math canonical (corpus) | **PASS** |
| validate_inline_refs (corpus) | **PASS** |
| lego-dead-code (corpus) | **0** violations |
| mlsysim pytest | **PASS** |

## Files changed (15)

**vol1:** hw_acceleration, introduction, ml_ops, ml_systems, model_compression, model_serving, nn_architectures, training

**vol2:** compute_infrastructure, fault_tolerance, fleet_orchestration, inference, introduction, performance_engineering, security_privacy

## HTML render verify (changed chapters)

| Chapter | Build | audit_html | Notes |
|---------|-------|------------|-------|
| compute_infrastructure | OK | CLEAN | tau_opt_s_str, power_delta_kw_str fixes verified |
| inference | OK | FAIL (pre-existing) | Algorithm LaTeX in HTML; ranking cascade renders |
| ml_systems | OK | CLEAN | ww_devices_b_str fix verified |
| hw_acceleration | OK | FAIL (pre-existing) | Algorithm listing LaTeX leak |
| introduction | FAIL | — | Pre-existing EnergyMovementRatios cell (flop/count) |
| model_serving | FAIL | — | Pre-existing MobileServingCalc cell |
| nn_architectures | FAIL | — | Pre-existing EnergyConsumptionAnalysis cell |

## Chapter sign-off (touched files)

| Chapter | Ph1 gates | P0–P2 fixes | audit_prose | HTML |
|---------|-----------|-------------|-------------|------|
| compute_infrastructure | PASS | DONE | PASS | OK |
| inference | PASS | DONE | PASS | OK |
| model_serving | PASS | DONE | blocked* | blocked* |
| introduction | PASS | DONE | blocked* | blocked* |
| ml_systems | PASS | DONE | PASS | OK |
| hw_acceleration | PASS | DONE | PASS | OK† |
| nn_architectures | PASS | DONE | blocked* | blocked* |
| training | PASS | DONE | blocked* | — |
| ml_ops | PASS | DONE | PASS | — |
| model_compression | PASS | DONE | blocked* | — |
| fault_tolerance | PASS | DONE | 3 flags‡ | — |
| fleet_orchestration | PASS | DONE | PASS | — |
| performance_engineering | PASS | DONE | PASS | — |
| security_privacy | PASS | DONE | blocked* | — |
| vol2 introduction | PASS | DONE | PASS | — |

\* `audit_prose` exec fails on unrelated pre-existing cells (`No module named 'book'`, dimension errors).

† Build OK; audit_html flags algorithm environment (not LEGO naming).

‡ Spurious `.0` on unrelated exports; not introduced by this pass.

## Deferred backlog (P3–P4)

### P3 — Locality / architecture

- `edge_intelligence.qmd`: EdgeDeviceSpectrum gap:738, cross_cell
- `compute_infrastructure.qmd`: ~108 *Recap classes (multi-section span)
- `data_engineering.qmd`: KWSProblemTargets span:678
- `nn_architectures.qmd`: CNNLighthouseProfile span:3599
- `conclusion.qmd` (vol2): ConclusionScaleFacts, FleetEvolution
- `sustainable_ai.qmd`: Gpt3HouseholdEnergyAnchor span:2394
- `performance_engineering.qmd`: KVCacheAnalysis span:1064
- `fleet_orchestration.qmd`: ClusterEconomicsMigRecap dead export

### P4 — Polish

- Generic USD export names (~21 corpus-wide)
- Header slimming (drop redundant Imports/Exports)
- 4 multi-class cells: responsible_engr, training, distributed_training ×2

### P5 — Distance / length — DONE (2026-06-06 follow-up)

- Added `fmt_length()` in `mlsysim/fmt.py` (auto m/km; default comma policy for small meters)
- Pilot migrations: `LightLatency.distance_km_str`, `BrakingDistance.distance_m_str`, `EdgeLatencyDistance.distance_m_str`
- Prose: bare refs (removed trailing “meters” / “km/h” where closed)

### P6 — Typed formatters — DONE (partial)

**Arithmetic intensity:** ~35+ exports → `fmt_arithmetic_intensity`; duplicate prose ` FLOP/byte` stripped corpus-wide (vol1 + vol2).

**Speed:** `"km/h"` added to `fmt_rate`; `speed_kmh_str` closed in braking/edge cells.

**Still queued:** `fmt_temperature` audit; hw_acceleration open intensity cells (`qkv_ai_str`, etc.); binder rules for intensity/length lanes; P3–P4 items below.

### P5/P6 — Original planning notes (archive)


**Known inconsistent pattern today** — same physical quantity, three styles:

| Location | Export | Formatter | Prose unit |
|----------|--------|-----------|------------|
| `ml_systems.qmd` `LightLatency` | `distance_str` | `fmt_qty(..., km, commas=True)` | bare (closed km) |
| `ml_systems.qmd` `BrakingDistance` | `distance_str` | `fmt(distance_m, precision=1, commas=False)` | `` `{python} ref` meters `` |
| `vol2/introduction.qmd` `EdgeLatencyDistance` | `distance_str` | `fmt(distance_m, precision=2, commas=False)` | `` `{python} ref` meters `` |

**Questions to resolve before codifying:**

1. **Closed vs open** — Should sub-100 m prose use `distance_m_str = fmt_length(..., unit=meter)` with bare ref, or keep open `distance_str` + prose “meters”?
2. **Comma policy** — `commas=False` for small magnitudes (2.8, 3.33) is correct; when distance ≥ 1,000 m or when using km, use `commas=True`? Document in `fmt.md` / `lego-units.md`.
3. **Auto-scale** — Like `fmt_memory`, should long straight-line distances auto-select m vs km (`fmt_length` closed-auto name `distance_str`) vs pinned `distance_km_str`?
4. **Naming** — Rename open exports to drop misleading unit tokens (`distance_str` OK for open; avoid `distance_m_str` unless closed).
5. **Binder check** — Optional lint: flag `distance_*_str` + bare `fmt()` without prose-unit pairing audit (similar to Category A).

**Corpus grep to seed the pass:** `distance_str`, `distance_m_str`, `distance_km_str`, `fmt(.*_m`, length/scaling in braking and latency cells.

**If `fmt_length` is added:** pin unit or auto-scale policy in `fmt.py`, add to `audit_fmt_usage.py` inventory, one-line note in `lego-units.md`, and migrate the three cells above as pilot.

### P6 — Typed `fmt_*` gap analysis (queue; prose-unit lock-in)

**Principle:** Every typed formatter closes the unit in OUTPUT. Prose uses a **bare** `` `{python} ref` ``; `lego-prose-units` catches duplicate glyphs. Prefer **migrating to existing helpers** before inventing new ones.

| Priority | Helper | Status | Corpus signal | Action when we return |
|----------|--------|--------|---------------|------------------------|
| **1** | `fmt_arithmetic_intensity` | **Exists, underused** | ~47 prose hits of `` `{python} …` FLOP/byte ``; many `*_ridge_str = fmt_int(...)` / `*_intensity_str = fmt(...)` in benchmarking, hw_acceleration, compute_infrastructure, nn_computation, appendix_fleet/dam | Migrate to closed `*_flop_per_byte_str = fmt_arithmetic_intensity(..., unit=flop/byte)`; strip prose ` FLOP/byte`. **No new helper.** |
| **2** | `fmt_length` | **Missing** | 2× open `distance_str` + prose “meters”; 1× closed `fmt_qty(..., km)` in `LightLatency` | Add helper; pilot BrakingDistance, EdgeLatencyDistance, LightLatency (see P5). |
| **3** | `fmt_rate` or `fmt_speed` | **Partial** (`fmt_rate` exists) | 3× open `speed_kmh_str` / `speed_str` + prose “km/h” (ml_systems, vol2 introduction) | Extend `fmt_rate(..., "km/h")` or thin `fmt_speed(q, unit=km/hour)`; rename `speed_kmh_str`, bare ref. |
| **4** | `fmt_temperature` | **Exists, rare** | 1 use in ml_systems; thermal prose elsewhere may duplicate °C | Audit thermal cells; migrate where prose repeats unit. |
| **5** | `fmt_latency` vs `fmt_time` | **Both exist** | Clarify split: wall-clock narrative → `fmt_time`; SLA/latency budget labels → `fmt_latency`? | Document in `fmt.md`; no new helper unless audit finds gap. |
| **6** | `fmt_count` / `fmt_params` | **Exists** | Was Category C (_b/_m + bare fmt); mostly fixed | Spot-check remaining open dimension exports (`n_gpus_str`, batch sizes) — often **intentionally open** when prose says “GPUs” / “layers”. |
| **7** | Clock **cycles** (not time) | **No helper** | `t_cycles_str`, `exp_cycles_*_value_str` — dimensionless, not seconds | Keep open `fmt()` + `_value_str` for equation slots; **do not** use `fmt_time`. |

**Do not add** (low ROI or wrong domain):

- Generic `fmt_distance` alias if `fmt_length` covers it.
- New helper for every roofline variable — `fmt_arithmetic_intensity` + naming (`*_flop_per_byte_str`) is enough.
- Comma logic baked into helpers globally — document per-domain in `fmt.md` (length: no commas &lt;1000 m; intensity: often `commas=False` for ridge points ~156).

**Sanity checks to add with each migration lane:**

1. `lego-prose-units` — no unit word after closed `*_str` ref.
2. Binder rule: `*_flop_per_byte_str` / `*_ridge_*` must use `fmt_arithmetic_intensity`, not `fmt`/`fmt_int`.
3. Binder rule: `*_m_str` / `*_km_str` length exports must use `fmt_length` or `fmt_qty` with pinned unit, not bare `fmt`.

**Corpus grep seeds:** `FLOP/byte`, `km/h`, `meters`, `*_ridge_str = fmt`, `*_intensity_str = fmt`, `speed_kmh_str`.

### Pre-existing render blockers (not this pass)

- `introduction.qmd`: EnergyMovementRatios (flop/count dimension)
- `model_serving.qmd`: MobileServingCalc import path
- `nn_architectures.qmd`: EnergyConsumptionAnalysis (picojoule/flop → picojoule)

## PDF / full pipeline

Full vol1+vol2 PDF builds not run in this pass (multi-hour). Recommend `/precheck` before push to origin.

Gate-only pipeline: `./book/tools/audit/verify_lego_pipeline.sh --skip-render --skip-pdf --skip-llm` — fails on focal-verify P3 chapters only.
