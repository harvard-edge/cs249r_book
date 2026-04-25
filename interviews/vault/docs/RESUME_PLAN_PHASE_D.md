# Resume Plan — Phase D/E/F (Priority Gap Closure + Generator Leverage)

**Purpose:** hand the next Claude session everything it needs to close
the parallelism + global L4-L6+ gaps that have remained open across
two prior multi-phase pushes, plus three high-leverage generator
improvements that pay for themselves on every future run.

**Companion docs (same branch):**
- `RESUME_PLAN_2026-04-25.md` — Phase 1-7 (committed at `ece6eccf2`)
- `RESUME_PLAN_RELEASE.md` — Phase A (committed at `542aaf95d`)
- this doc — Phase D/E/F

---

## Current state

| | |
|---|---|
| **Worktree** | `/Users/VJ/GitHub/MLSysBook-massive-build` |
| **Branch** | `feat/massive-build-2026-04-25-run` |
| **HEAD** | `e7cd3b24c feat(vault): Phase B + C — 144 PASS items added (B.5: 110, C.4: 34)` |
| **Bundle** | 9,688 published (was 9,224 at branch cut, +464 net) |
| **All gates** | green (`vault check --strict`, lint, doctor, codegen, validate-vault, render) |

---

## What's already done (do NOT redo)

### Phase A (commit `542aaf95d`)
- 3 structural Pydantic validators added: `visual.path-resolves`, `_zone_bloom_compatible`, `disk-coverage`
- Lint calibration via 4-expert consensus (1,308 → 0 warnings)
- Registry repaired (5,269 IDs appended), doctor split into `disk-coverage` (HARD) + `registry-history` (INFO)
- Chain integrity full pass (0 errors / 0 warnings)
- Practice page zoom modal + 9th Playwright test

### Phase B (in commit `e7cd3b24c`)
- Generator hardened: `bloom_for_zone_level()` respects ZONE_BLOOM_AFFINITY, prompt requires `bloom_level` field, lists 13 canonical competency_areas inline, demands L5/L6+ depth (no trivial division framings).
- **Validate-at-write**: every Gemini-emitted YAML round-trips through `Question.model_validate()` before disk write.
- B.5 loop saturated at iter 4 on `DROP rate 38.3% > 35%` (judge tightening on L6+ depth, not budget). Yield: 110 PASS in 26 calls.

### Phase C (in commit `e7cd3b24c`)
- 120 NEEDS_FIX items from prior session re-edited via fix-agent (92 edited, 28 already-resolved).
- Re-judge: 67 of 92 judged → 34 PASS / 13 NEEDS_FIX / 20 DROP. 34 PASS promoted.

### Saturation reasons (carry-forward signal)
- B.5: `DROP rate 38.3% exceeds 35% — likely hallucination`. Judge rejects nearly half of L6+ depth items even with the strengthened prompt. Adding more API calls won't help; deeper prompt scaffolding will.
- C.3: 25 of 92 items unjudged (max-calls=5 chunk cap).

---

## What's still open (Phase D/E/F)

### Three priority gaps that remain
| Gap | Current | Expected | Status |
|---|---|---|---|
| `tinyml/parallelism` (area-level) | 1 | ~95 | **never closed** |
| `mobile/parallelism` (area-level) | 0 | ~134 | **never closed** |
| `edge/parallelism` (area-level) | 13 | ~159 | barely moved |
| `global/realization/L4-L6+` | 0 | ~14 | empty |
| `global/specification/L6+` | 0 | ~5 | empty |
| `global/mastery/L5` | 0 | ~5 | empty |

**Why prior runs didn't close them**: the analyzer's recommended_plan
picks **topic-level** cells (queueing-theory, memory-hierarchy-design,
etc.) by priority, but the parallelism gap aggregates across multiple
parallelism-flavored topics (pipeline-parallelism,
collective-communication, kv-cache-management, interconnect-topology).
None of those individual topic cells crack the top-100 priority list, so
the loop never targets them. Closing the area-level gap requires
**hand-built topic targets**, bypassing the analyzer.

### Three carry-forwards from C.3
- 25 unjudged items — max-calls cap left them on the table
- 13 still-NEEDS_FIX after one fix attempt — second fix pass possible
- 20 DROP items — could be salvaged with a deeper rewrite

---

## Phases D + E + F

### Phase D — Priority gap closure (THE mission, finally)

| ID | Task | Acceptance | Effort |
|---|---|---|---|
| D.1 | Hand-author **~50 parallelism targets** as `track:topic:zone:level` strings. Topics: `pipeline-parallelism`, `collective-communication`, `kv-cache-management`, `interconnect-topology`. Tracks: edge/mobile/tinyml at L4-L6+. Skip cloud (already dense). Save to `tools/phase_d/parallelism_targets.txt`. | File written, ≥40 cells, all 4 topics represented | 30 min |
| D.2 | Author a **parallelism-specific prompt variant** in the generator. Adds these rules: (a) forbid bandwidth-division framings (`payload / bandwidth`); (b) require concrete topology (NVLink/IB/PCIe/RoCE/LoRa) appropriate to the track; (c) require a synchronization or bubble cost in the question; (d) require non-trivial system integration. Toggle via `--prompt-variant parallelism` CLI flag. | Manual test: feed 5 cells, judge ≥3 of 5 PASS at high confidence | 1.5 hr |
| D.2' | **REVIEW CHECKPOINT** — surface prompt + 5 sample drafts for user review before D.3 burns API budget | User signs off | — |
| D.3 | Run focused loop (15-20 API calls, batch_size 30) targeting D.1's hand-built cells with `--prompt-variant parallelism` | Loop summary: ≥20 PASS items in parallelism cells | 2 hr wall clock |
| D.4 | Spot-read all PASS items from D.3 (~30-50); reject any that read as bandwidth-math (manual edit to set `status: archived` or rewrite). Promote the rest. | All promoted items have non-trivial framings | 30 min |
| D.5 | Same mechanism for **global L4-L6+**: hand-author ~20 cells, run focused loop with **standard prompt** (global cells aren't parallelism-flavored, just under-filled). | ≥10 global L4-L6+ PASS items | 2 hr wall clock |
| D.6 | Promote, rebuild bundle, regen paper artifacts | `vault check --strict` clean; published count up by 30-60 | 30 min |

**Phase D total**: ~7 hr work, ~5 hr wall clock, ~30-40 API calls.

### Phase E — Generator efficiency (compounding leverage)

| ID | Task | Acceptance | Effort | Saves |
|---|---|---|---|---|
| E.1 | **Retry-on-validation-fail** in `gemini_cli_generate_questions.py`. If `Question.model_validate()` rejects, single retry with prompt suffix `"your previous JSON had these violations: <list>. Re-emit only the failed items, fixed."` Second failure logs structured error and skips. | Unit test: feed bad dict → script retries once, recovers | 45 min | ~50% of API calls (B.5's iter 1 + iter 3 lost 8 of 26 = 31%) |
| E.2 | **Auto-update vault-manifest.json from `vault build`**. Currently maintained by hand; pre-commit caught the gap twice this session. | `vault build --legacy-json` writes a fresh manifest with current counts + hash | 30 min | Manifest-stale failures eliminated |
| E.3 | **Tighten the analyzer**: add `--include-areas parallelism,networking` flag so the recommended_plan can include cells weighted by track×area gap (not only track×topic gap). Solves the structural issue that drove D.1's hand-authoring. | Run with `--include-areas parallelism` returns plan with ≥10 parallelism-topic cells | 1 hr | Future runs don't need D.1's hand-build step |

**Phase E total**: ~2.5 hr.

### Phase F — Residual cleanup (completeness)

| ID | Task | Acceptance | Effort |
|---|---|---|---|
| F.1 | **Re-judge the 25 unjudged items** from C.3. Use the same fix-agent-edited paths from `tools/phase_c/needs_fix_manifest.json`. | 25 items judged; promote any flipped to PASS | 20 min |
| F.2 | **Second-pass fix-agent** on remaining 13 NEEDS_FIX + 20 DROP from C.3. Spawn `general-purpose` agent with the C.3 judge's verdicts as input. | Each item edited; re-judged; promote flipped | 1 hr |
| F.3 | **Spot-read 20 PASS items** stratified across this push's promotions (Phase B + C combined = 144 items). Rejection bar: shallow framings, math errors, hardware-spec inaccuracies. | Reviewed list saved; rejection rate ≤ 10% | 1 hr |

**Phase F total**: ~2.5 hr.

---

## Parallelism map (what can run concurrently)

The cleanest interleaving:

```
Stage 1 — sequential prep (no API)              ~3 hr
  D.1 (hand-build targets)
    └── D.2 (parallelism prompt)
          └── E.1 (retry-on-validate-fail)
                └── E.2 (auto-manifest)
                      └── E.3 (analyzer flag)
                            └── (D.2' user review)

Stage 2 — parallel execution                    ~2 hr wall clock
  D.3 (parallelism loop, 15-20 calls)  ━┓
                                          ┣━ both write disjoint IDs
  F.2 (fix-agent on 33 items)          ━┛   no race risk

Stage 3 — parallel execution                    ~2 hr wall clock
  D.5 (global loop, 10-15 calls)       ━┓
                                          ┣━ all disjoint
  F.1 (re-judge 25 unjudged)           ━┫
                                          ┃
  F.3 (spot-read first 10 of 20)       ━┛  read-only

Stage 4 — sequential finalize                   ~1 hr
  D.4 (parallelism spot-read + promote)
    └── D.6 (rebuild bundle, regen paper)
          └── F.3 (finish spot-read second 10)
                └── final commit
```

**Total wall clock**: ~8 hr (vs ~10-12 hr serial).

**API budget**: ~30-40 calls expected (Gemini cap is 250/day; today used ~76, so ~174 remaining).

### Parallelism safety rules

1. **No two generation loops concurrent** — both call `next_id_for_track()` which is filesystem-stat-based; concurrent calls can race on the next ID. D.3 must finish before D.5 starts.
2. **Generation loop + fix-agent OK** — disjoint ID ranges (loop writes new, agent edits existing).
3. **Generation loop + judge OK** — judge reads files, doesn't write to questions/.
4. **No schema changes during loops** — schema changes invalidate validate-at-write contract mid-stream.

---

## Locked decisions (do NOT relitigate)

| Decision | Choice |
|---|---|
| **Release tag** | One stable dev branch, no mid-stream release tag (per prior plan) |
| **Bloom canonical** | When zone-bloom conflict, trust bloom; reclassify zone via `BLOOM_CANONICAL_ZONE` |
| **Validate-at-write severity** | ERROR (Pydantic hard-rejects), not WARN |
| **D.2 prompt authorship** | Claude drafts, user reviews at D.2' |
| **Test-first for E.x** | Unit tests before real API calls (cheaper failure mode) |

---

## Review checkpoints

1. **D.2'** — surface parallelism prompt + 5 sample drafts for user review before D.3 fires the loop.
2. **D.4** — surface PASS items for spot-read; user can flag any that read shallow.
3. **Final** — surface all gates green + commit summary.

---

## Common saturation outcomes for D.3 / D.5

If D.3 stops early:

| Reason | Meaning | What to do |
|---|---|---|
| `DROP rate > 35%` | Judge rejecting parallelism items as too shallow | Inspect the latest iter's `judge_summary.json` — if rejections are about "trivial topology" framings, tighten D.2 prompt further. If about correctness errors, accept the saturation. |
| `same top cell two iters` | Generator can't fill | Hit budget cap; move on, document as ceiling |
| `max-calls reached` | Burned through API budget | Stop. Commit what we have. |
| `0 drafts produced` | Validate-at-write rejected entire batch | E.1's retry should have prevented this; if it persists, dump the prompt and inspect Gemini's raw output |

---

## What NOT to do

- ❌ Don't merge to `dev` until all gates green AND user explicitly OKs.
- ❌ Don't push to remote without explicit user OK.
- ❌ Don't run two generation loops concurrently (next-id race).
- ❌ Don't add `Co-Authored-By` lines or automated attribution footers.
- ❌ Don't change ZONE_BLOOM_AFFINITY or schema enum values without explicit user direction.
- ❌ Don't auto-promote NEEDS_FIX without re-judge.
- ❌ Don't suppress lint warnings or skip pre-commit hooks (`--no-verify` forbidden).
- ❌ Don't auto-cut a release tag (`v0.1.2`) — single stable commit is the goal.
- ❌ Don't navigate to or modify files in sibling worktrees.

---

## Files of interest

| File | Why |
|---|---|
| `interviews/vault/docs/RESUME_PLAN_2026-04-25.md` | Phase 1-7 history |
| `interviews/vault/docs/RESUME_PLAN_RELEASE.md` | Phase A history |
| `interviews/vault/docs/MASSIVE_BUILD_RUNBOOK.md` | Methodology document |
| `interviews/vault/_validation_results/coverage_loop/20260425_192956/` | Most recent loop output (B.5) — judge_summary.json per iter, NEEDS_FIX details with fix_suggestions |
| `interviews/vault/_validation_results/phase_c_rejudge/judge_summary.json/20260425_201121/summary.json` | C.3 re-judge verdicts |
| `tools/phase_c/needs_fix_manifest.json` | The 120-item NEEDS_FIX queue (the 13 still-pending + 20 DROP go here for F.2) |
| `tools/phase_b/cell_triage.json` | The 14 L6+/L5-deep cells (a subset of what D.2's prompt should target) |
| `interviews/vault/scripts/gemini_cli_generate_questions.py` | **D.2 + E.1 edit here.** |
| `interviews/vault/scripts/analyze_coverage_gaps.py` | **E.3 edits here.** |
| `interviews/vault-cli/src/vault_cli/commands/build.py` (or equivalent) | **E.2 edits here** to write the manifest. |
| `interviews/vault/schema/enums.py` | ZONE_BLOOM_AFFINITY + BLOOM_CANONICAL_ZONE + widened ZONE_LEVEL_AFFINITY (do not edit lightly) |

---

## One-liner status check (run first in next session)

```bash
cd /Users/VJ/GitHub/MLSysBook-massive-build && \
  git log --oneline -3 && echo "---" && \
  git status --short | head -5 && echo "---" && \
  PYTHONPATH=interviews/vault-cli/src \
    python3 -m vault_cli.main check --strict 2>&1 | tail -2 && \
  echo "---" && \
  PYTHONPATH=interviews/vault-cli/src \
    python3 -m vault_cli.main lint interviews/vault/questions/ 2>&1 | tail -2 && \
  echo "---" && \
  PYTHONPATH=interviews/vault-cli/src \
    python3 -m vault_cli.main doctor 2>&1 | grep -cE "fail" | xargs -I{} echo "doctor fails: {}" && \
  echo "---" && \
  python3 -c "
import json
c = json.load(open('interviews/staffml/src/data/corpus.json'))
print(f'published: {len(c)}')
"
```

If output shows commit `e7cd3b24c`, clean tree, vault check passes,
0 lint warnings, 0 doctor fails, 9,688 published — the resume state
matches this plan's starting assumptions. **Proceed to D.1.**

If anything differs, **stop and reconcile** before any code edits.

---

## Pacing

This is a ~12-15 hour push compressed to ~8 hr wall clock by the
parallelism map. Plausibly two focused sessions, or one long one.

The biggest risk is D.3 saturating at low yield (<10 parallelism PASS
items). If that happens, D.5 becomes the only material content gain
of this push, and the parallelism gap stays open as a documented
limitation rather than a closed mission. That is acceptable — the
branch was already StaffML-day-ready before Phase D started.

The smallest budget commitment is Phase E (no API calls; pure
generator infra). If only one phase fits, do E — it compounds for
every future generation run, while D is a one-time content gain.

Three explicit user-review checkpoints (D.2', D.4, final). Wait for
sign-off at each before continuing.
