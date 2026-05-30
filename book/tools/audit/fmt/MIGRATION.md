# FMT Migration Map

> **Goal.** Every prose value in the book flows from a LEGO OUTPUT cell through a
> **typed, guarded formatter** (`fmt_usd`, `fmt_percent`, `fmt_pp`, `fmt_multiple`,
> `fmt_count`, `fmt_qty`, `fmt_ratio`, `fmt_range`) — never a raw `fmt(prefix=/suffix=)`
> carrying semantic meaning. The migration is **one-time, global, and must ship zero
> wrong numbers.** Authoritative rules: `.claude/rules/fmt.md`.

## Why this exists

`fmt(x, suffix=' percent')` and friends let a 0–1 ratio silently render as `8500 percent`.
Typed formatters move the unit *into the function*, where a guard can reject the
impossible (e.g. `fmt_percent` rejects a ratio outside 0–`max_ratio`). The danger is
**semantic** suffixes (percent / × / scale glyphs) that change the *meaning* of a number.
Unit suffixes (`GB`, `ms`, `W`) are only cosmetic — they get migrated too (full purity,
per decision), but **after** the dangerous ones and with far lower risk.

## Baseline (post-`dev`-sync audit — `/tmp/fmt_audit.json`)

| Metric | Count |
|---|---|
| Total fmt-family call sites | 6,611 |
| **Dangerous-semantic `suffix=`** (migrate first) | **870** |
| &nbsp;&nbsp;• percent (`%`,`percent`) → `fmt_percent` | 612 |
| &nbsp;&nbsp;• scale (`K`/`M`/`B`) → `fmt_count` | 152 |
| &nbsp;&nbsp;• multiplier (`×`,`x`) → `fmt_multiple` | 106 |
| Unit `suffix=` (`GB`,`ms`,`W`,…) → `fmt_qty`/`fmt_unit` (Phase 2) | 2,412 |
| Raw `MarkdownStr` (unguarded) → `fmt_range`/typed | 337 |
| `fmt_percent` **used today** | **10** |

The 612-vs-10 gap is the project. Regenerate with:
`python3 book/tools/audit/fmt/audit_fmt_usage.py --root book/quarto/contents --json /tmp/fmt_audit.json`

---

## Phases

### Phase 0 — Foundation
- [x] Commit typed formatters + `fmt_semantic_suffix` checker + audit tooling
- [x] Merge `dev` (the suffix-consolidation pass) into `fmt-fix` — clean
- [x] Re-audit → true baseline (above)
- [x] This ledger (you are here)
- [ ] Keep `fmt_semantic_suffix` **opt-in** during migration (run per-chapter in step 5); flip to a **global pre-commit blocker only at the end** (Phase 2) once the board is all DONE — flipping early would block every chapter commit.
- [ ] **Production AST codemod** — auto-rewrites only *provable* cases; queues ambiguous ones
- [ ] **`fmt_range`** typed/guarded helper + tests
- [ ] **Prose-unit duplication checker** — flags a unit/glyph typed after a ref that already owns it

### Phase 1 — Dangerous-811 rollout (per-chapter loop, hardest first)
Walk the board top-down: `training → data_selection → model_serving → benchmarking → …`

### Phase 2 — Unit-suffix purity + final sweep
Codemod 2,412 unit suffixes → `fmt_qty`/`fmt_unit`; full-book render; corpus guard sweep;
**flip `fmt_semantic_suffix` to a global pre-commit blocker** (the regression lock).

---

## The per-chapter loop (run identically for every chapter)

1. **Baseline render** the chapter to HTML; capture rendered numbers as the "before" truth.
2. **Codemod provable sites** (`x*100, suffix=' percent'` → `fmt_percent(x)`; `×`→`fmt_multiple`; `K/M/B`→`fmt_count`). Ambiguous percent sites are **flagged, never auto-touched**.
3. **Human-resolve the queue** — read each ambiguous cell's compute; normalize the value **to a 0–1 ratio at the source** so `fmt_percent` owns the ×100. *(Only judgment step; where wrong numbers hide.)*
4. **Fix the prose** — for each rewritten cell, find its `` {python} *_str `` refs and strip now-duplicated glyphs/units; fix wording.
5. **Gates green** — `binder check math` (canonical + suffix-semantics), `fmt_semantic_suffix`, dead-LEGO, prose-unit checker.
6. **Re-render** the chapter.
7. **Render-diff** before↔after — the *only* allowed visible change is glyph/format, **never a magnitude**. A changed number = stop & investigate.
8. **Sign off** in the board below; commit that one chapter.

## Auditability (working back from "we caught the error")
1. **Source guards** — `fmt_percent` throws on out-of-range ratio at render time.
2. **Static gates** — suffix-semantics + canonical block old patterns at commit.
3. **Render-diff** — an unexplained magnitude change is the tripwire.
4. **This board** — the single source of truth for where every chapter stands.

---

## Per-chapter board

Status legend: `pending` → `codemod` → `queue-resolved` → `prose-fixed` → `gated` → `rendered` → `diff-clean` → **`DONE`**

Columns from baseline audit: **DGR** = dangerous suffixes (pct+mult+scale), **unit** = unit suffixes (Phase 2), **mds** = raw MarkdownStr.

| DGR | pct | mlt | scl | unit | mds | Chapter | Status |
|---:|---:|---:|---:|---:|---:|---|---|
| 86 | 43 | 23 | 20 | 205 | 21 | vol1/training | pending |
| 66 | 46 | 0 | 20 | 57 | 10 | vol1/data_selection | pending |
| 63 | 44 | 19 | 0 | 199 | 13 | vol1/model_serving | pending |
| 59 | 40 | 14 | 5 | 88 | 13 | vol1/benchmarking | pending |
| 58 | 21 | 16 | 21 | 96 | 5 | vol1/data_engineering | pending |
| 48 | 27 | 1 | 20 | 50 | 16 | vol1/responsible_engr | pending |
| 41 | 33 | 2 | 6 | 133 | 10 | vol2/distributed_training | pending |
| 36 | 19 | 0 | 17 | 70 | 6 | vol2/ops_scale | pending |
| 33 | 28 | 0 | 5 | 39 | 4 | vol1/ml_ops | pending |
| 32 | 31 | 0 | 1 | 65 | 1 | vol2/backmatter/appendix_fleet | pending |
| 30 | 30 | 0 | 0 | 3 | 1 | vol2/backmatter/appendix_c3 | pending |
| 23 | 21 | 0 | 2 | 56 | 5 | vol2/fleet_orchestration | pending |
| 19 | 16 | 0 | 3 | 6 | 0 | vol2/responsible_ai | pending |
| 18 | 17 | 1 | 0 | 33 | 4 | vol1/ml_workflow | pending |
| 18 | 17 | 0 | 1 | 47 | 13 | vol2/fault_tolerance | pending |
| 17 | 17 | 0 | 0 | 53 | 2 | vol2/performance_engineering | pending |
| 16 | 12 | 0 | 4 | 54 | 15 | vol2/sustainable_ai | pending |
| 15 | 14 | 0 | 1 | 75 | 7 | vol2/inference | pending |
| 15 | 8 | 0 | 7 | 19 | 1 | vol2/security_privacy | pending |
| 14 | 8 | 6 | 0 | 41 | 5 | vol1/introduction | pending |
| 13 | 1 | 12 | 0 | 8 | 2 | vol1/conclusion | pending |
| 12 | 11 | 0 | 1 | 16 | 22 | vol2/backmatter/appendix_assumptions | pending |
| 12 | 5 | 4 | 3 | 129 | 69 | vol1/ml_systems | pending |
| 11 | 9 | 2 | 0 | 94 | 4 | vol1/hw_acceleration | pending |
| 11 | 8 | 2 | 1 | 112 | 8 | vol2/data_storage | pending |
| 10 | 10 | 0 | 0 | 60 | 12 | vol1/model_compression | pending |
| 10 | 10 | 0 | 0 | 16 | 1 | vol2/backmatter/appendix_communication | pending |
| 10 | 7 | 0 | 3 | 74 | 4 | vol2/network_fabrics | pending |
| 10 | 6 | 0 | 4 | 52 | 1 | vol1/nn_architectures | pending |
| 10 | 5 | 4 | 1 | 74 | 1 | vol1/frameworks | pending |
| 8 | 8 | 0 | 0 | 174 | 2 | vol2/compute_infrastructure | pending |
| 7 | 5 | 0 | 2 | 22 | 14 | vol1/backmatter/appendix_machine | pending |
| 6 | 6 | 0 | 0 | 14 | 3 | vol2/introduction | pending |
| 6 | 6 | 0 | 0 | 2 | 2 | vol2/robust_ai | pending |
| 5 | 5 | 0 | 0 | 22 | 3 | vol2/backmatter/appendix_reliability | pending |
| 5 | 2 | 0 | 3 | 45 | 8 | vol1/nn_computation | pending |
| 4 | 4 | 0 | 0 | 11 | 2 | vol1/backmatter/appendix_algorithm | pending |
| 3 | 3 | 0 | 0 | 53 | 1 | vol2/collective_communication | pending |
| 3 | 3 | 0 | 0 | 4 | 0 | vol2/conclusion | pending |
| 2 | 2 | 0 | 0 | 27 | 6 | vol2/edge_intelligence | pending |
| 2 | 2 | 0 | 0 | 6 | 17 | vol1/backmatter/appendix_data | pending |
| 2 | 2 | 0 | 0 | 3 | 0 | vol2/backmatter/appendix_inference | pending |
| 1 | 0 | 0 | 1 | 5 | 3 | vol1/backmatter/appendix_assumptions | pending |

**Totals:** 43 files · 870 dangerous · 2,412 unit · 337 MarkdownStr.
