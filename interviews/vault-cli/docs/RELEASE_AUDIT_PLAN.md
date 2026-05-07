# StaffML Vault — Release Audit Plan

**Status:** ready to execute (handoff doc)
**Branch:** `yaml-audit` at `a74c98576` (sync'd with origin/dev)
**Worktree:** `/Users/VJ/GitHub/MLSysBook-yaml-audit`
**Daily Gemini cap:** 250 calls

---

## What this is

A scoped plan for taking the corpus from "post-Phase-3 working state" to
"ready to release." Designed to fit inside the 250 Gemini-calls/day cap.
Built from a corpus-wide health survey done 2026-05-02 (numbers below).

This document is the resume reference for the **new session** that will
execute the audit. Read this whole thing top-to-bottom before starting.

---

## Current corpus health (snapshot 2026-05-02)

Survey ran on 9,446 published questions:

### What's clean

- **Schema:** 100% pass (all 9,446 satisfy the Pydantic Question model)
- **Format compliance** (Pitfall/Rationale/Consequence + Assumptions/Calculations/Conclusion): **90.9% pass** (8,585 of 9,446)
- **Napkin-math style:** 95.3% use the canonical bold-marker form (8,999 of 9,446)
- **Chains:** 843 chains, 9,446 published questions, releaseHash `5a4783e62d…`

### What needs cleanup before release

| # | issue | count | affected | fix |
|---|---|---:|---|---|
| 1 | Placeholder titles (`Global New 0006`, etc.) | **134** | all `global-*` track | rewrite each with a real, descriptive title |
| 2 | `provenance: None` (string) | **407** | scattered across tracks | flip to `imported` (the historical default) |
| 3 | Napkin-math missing one or more bold markers | **447** | scattered | add the markers — prose is usually fine, just labels missing |
| 4 | `common_mistake` missing one or more markers | **414** of 861 format-fails | scattered | same pattern: add labels |
| 5 | Solutions that read like rubrics (1./2./Step …) | **42** | scattered | review case-by-case; may be intentional |
| 6 | Scenario < 30 chars | **1** | (`global-0253`) | rewrite a fuller scenario |

Full data: `interviews/vault/_pipeline/format-audit-full.json` (gitignored,
regenerable).

### What's UNAUDITED (the new session's job)

- **Math correctness in napkin_math** — Gemini hasn't independently
  re-derived the calculations on any large slice of the corpus.
  Phase 3 batch ran audit_math on 9 drafts (100% pass) and audit run
  on 4 pilot drafts. That's <0.2% of the corpus.
- **Cognitive-load fit** (`level_fit` gate from `validate_drafts.py`) —
  ran only on Phase 3 drafts.
- **Coherence + physical realism + vendor fabrication** — ran only on
  Phase 3 drafts and the 506 secondary-tier chain audit (which doesn't
  per-question check coherence).

The 90.9% format pass is regex-cheap. The other gates need Gemini, and
that's where the 250/day cap bites.

---

## The plan: stratified sample, not full corpus

A full audit is mathematically infeasible inside the cap (98 days at
250/day for the full ~25K-call sweep — see calculation in §"Why
sampling" below). A stratified sample gets 95% of the value at 5% of
the cost.

### Sample design

**1,000 questions, stratified by (track × level)**, RNG-seeded for
reproducibility. Per-cell counts:

| | L1 | L2 | L3 | L4 | L5 | L6+ | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| cloud  | 33 | 33 | 33 | 33 | 33 | 33 | 200 |
| edge   | 33 | 33 | 33 | 33 | 33 | 33 | 200 |
| mobile | 33 | 33 | 33 | 33 | 33 | 33 | 200 |
| tinyml | 33 | 33 | 33 | 33 | 33 | 33 | 200 |
| global | 33 | 33 | 33 | 33 | 33 | 33 | 200 |

This gives 33 questions per cell — enough to detect any failure mode
that occurs at >10% rate with reasonable confidence, while keeping
the call budget bounded.

### Gates per sampled question

For each sampled question, run all 4 Gemini-judge gates:

1. **`level_fit`** — does the question's cognitive load match its level field?
2. **`coherence`** — physical realism, vendor fabrication, scenario/Q/solution mismatch, arithmetic errors
3. **`bridge`** — only for chained questions (skip when not applicable)
4. **`audit_math`** — independent re-derivation of napkin_math arithmetic (skip when no napkin_math)

Plus the regex `format_compliance` gate (free).

### Call budget

| gate | calls/q | applicable to | sample calls |
|---|---:|---|---:|
| format | 0 | all 1,000 | 0 |
| level_fit | 1 | all 1,000 | 1,000 |
| coherence | 1 | all 1,000 | 1,000 |
| bridge | 1 | ~30% chained | ~300 |
| audit_math | 1 | ~60% have napkin_math | ~600 |
| **total** | | | **~2,900 calls** |

At **250/day cap**: ~12 days
At **600/day** (if higher quota available): ~5 days
With aggressive 4-way parallelism within daily budget: same days, faster wall clock per day

### Daily run shape

Each day:
1. Resume from previous run's progress file
2. Pull next ~200 questions (4 gates × 50 questions = 200 calls)
3. Run with 4-way parallelism (`audit_math.py` style)
4. Persist progress + per-row results
5. Stop when daily quota exhausted; queue resumes tomorrow

Total calendar time: **2 weeks** at the safe pace.

---

## Tooling we have vs. need

### Have

- `scripts/audit_math.py` — math gate, ThreadPoolExecutor, `--workers N`, `--files`, `--sample-track`, `--sample-size`, `--seed`. Already parallel-safe.
- `scripts/validate_drafts.py` — has all 5 gates (schema, format, originality, level_fit, coherence, bridge). But targets `*.yaml.draft` only.
- `scripts/audit_chains_with_gemini.py` — chain-level audit, batched, serial, has `--seed`.

### Need (build before launch)

A new script (or extension of existing) with:
- **Multi-day quota tracking** — knows daily cap, stops at N calls, resumes tomorrow
- **Persistent progress** — writes `interviews/vault/_pipeline/audit-progress.json` after each call so a Ctrl-C / kill / overnight pause picks back up cleanly
- **Stratified sample selection** — driven by a config that says "33 per (track, level)"
- **Multi-gate per question** — runs format + level_fit + coherence + bridge + math in one pass per question
- **`--published` mode** — applies to `*.yaml`, not just `*.yaml.draft`
- **Parallel within daily budget** — 4-8 worker threads

Recommended approach: **extend `audit_math.py`** rather than write fresh,
since its parallel skeleton is solid. Rename to `audit_corpus.py` (or
similar). Add the multi-gate + multi-day + resume layers.

Estimated build time: 2-3 hours of focused work in the new session.

---

## Why sampling, not full

| target | math calls | coherence calls | level_fit calls | bridge calls | total | days @ 250/day |
|---|---:|---:|---:|---:|---:|---:|
| Sample (1,000) | ~600 | 1,000 | 1,000 | ~300 | **~2,900** | **~12** |
| Full (9,446) | ~5,700 | 9,446 | 9,446 | ~2,800 | **~27,400** | **~110** |
| Sample fraction of full cost | | | | | | **~11%** |

Sampling captures systemic patterns (anything occurring at >5-10% rate is
detectable) without burning four months of quota. If sampling reveals
specific buckets / levels with high failure rates, *targeted* full-coverage
audits on those slices are still cheap.

---

## Cleanup tasks the new session can do without Gemini calls

Before/alongside the audit, these mechanical fixes have **zero Gemini cost**:

### 1. Fix placeholder titles (134 questions)

All are in `global/`. Pattern: `"Global New NNNN"`. The new session
should **not auto-rewrite** these — they need real titles authored
from the question content. One option: feed the scenario+question to
Gemini and ask for a 3-8 word title. That's 134 Gemini calls but
they're cheap and one-shot. Could be done as a side-batch.

**Suggested approach:** batch them at 5/call (~27 calls) asking Gemini
to propose titles for each, then human review.

### 2. Fix `provenance: None` (407 questions)

This is mechanical — change the value to `imported` (the historical
default for the pre-2026-04 corpus). One-line script. Verify after
with `vault check --strict`.

### 3. Add missing format markers (447 napkin_math + 414 common_mistake)

Many of these have the structured prose but lost the bold labels. A
careful sed-style script could detect "Assumptions" prose at the top
of napkin_math and prepend the bold marker. Risk: false positives.

**Conservative approach:** flag them in a report, fix interactively
or with explicit user review.

### 4. Author a real `AUTHORING.md` template

`vault new` scaffolds only `scenario: <TODO>` and `realistic_solution: <TODO>`
— it doesn't include the `common_mistake` and `napkin_math` template
blocks, so authors have to know the convention from elsewhere
(currently only documented in the generation prompt and the format-
compliance regex check).

**Action:** create `interviews/vault/AUTHORING.md` that codifies:
- Required field list with shapes
- The Pitfall/Rationale/Consequence template for common_mistake
- The Assumptions/Calculations/Conclusion template for napkin_math
- Title conventions (≤120 chars, descriptive, no trailing period)
- Examples (2-3 well-formed reference questions per level)
- Gotchas: don't index symbols in titles, don't use mathmode in titles, etc.

Also extend `vault new`'s scaffold to include the template stubs.

---

## Resume instructions for the new session

### What the new session will see

- Branch: `yaml-audit` at `a74c98576` (or wherever it is at start)
- This document at `interviews/vault-cli/docs/RELEASE_AUDIT_PLAN.md`
- All tooling under `interviews/vault-cli/scripts/`
- Format-audit findings already cached at
  `interviews/vault/_pipeline/format-audit-full.json` (rerun if stale)

### Before doing anything, the new session must

1. Confirm worktree + branch:
   ```bash
   pwd          # /Users/VJ/GitHub/MLSysBook-yaml-audit
   git branch   # * yaml-audit
   git log --oneline -5
   ```
2. Run baseline validators:
   ```bash
   vault check --strict          # expect: 10,711 loaded, 0 invariant failures
   vault build --local-json      # expect: 9,446 published, 843 chains
   pytest interviews/vault-cli/tests/ -q  # expect: 74/74 pass
   ruff check interviews/vault-cli         # expect: clean
   ```
3. **Read** this entire document, then `interviews/vault/README.md`
   §"Pipeline artifacts" for the `_pipeline/` convention.
4. **Read** `interviews/vault-cli/docs/CHAIN_ROADMAP.md` Progress Log
   bottom-up for the latest audit findings — they inform the failure
   modes to watch for.

### Execution order (recommended)

1. **Build the audit tool** (~2-3 hours). Extend `audit_math.py` or
   write `audit_corpus.py` with: multi-gate, `--published` mode,
   stratified sample, daily cap, resume-from-progress.
2. **Run mechanical cleanups** (no Gemini cost):
   a. Flip `provenance: None` → `provenance: imported` (407 questions, 1 commit)
   b. Add missing format markers where prose is unambiguous (~half of the 447+414, 1 commit)
   c. Write `interviews/vault/AUTHORING.md` (1 commit)
   d. Extend `vault new` scaffold to include template stubs (1 commit)
3. **Run the placeholder-title fix** (134 questions × 1 Gemini call/5 = ~27 calls).
   Generate proposed titles, present for review, apply accepted.
4. **Launch the daily audit** (~2,900 Gemini calls over ~12 days).
   Run once per day; commit per-day progress JSON to `_pipeline/`
   (gitignored).
5. **Synthesise findings** at end of audit. Surface failure rates
   per category × cell. Bulk-fix targeted issues.
6. **Update `paper.tech`** with audit-pass numbers, post-cleanup corpus
   stats, etc.

### Stopping rules

- If `vault check --strict` ever fails → stop, investigate
- If pytest drops below 74/74 → stop, fix
- If audit failure rate exceeds 30% in any cell → stop, that bucket
  needs investigation before continuing the sample
- Daily Gemini cap exhausted → stop, queue resumes tomorrow

### Cost ledger format

Each daily run should append a row to
`interviews/vault/_pipeline/audit-cost-ledger.json`:

```json
{
  "date": "2026-05-03",
  "calls_made": 247,
  "calls_remaining_today": 3,
  "questions_audited": 49,
  "passes": 41,
  "fails": 8,
  "errors": 0
}
```

So we can see the ramp at any time.

---

## After the audit completes

1. **Compile a corpus-quality report** with per-cell failure rates,
   common failure modes, recommended bulk fixes
2. **Apply targeted fixes** — most likely a mechanical sweep on
   identifiable patterns (e.g., "all questions with X bug pattern")
3. **Update `paper.tech`** — corpus size, chain count, audit pass
   rates, methodology description
4. **Re-run validators** end-to-end, regenerate corpus.json,
   regenerate vault-manifest, push
5. **Tag a release** if everything is green

---

## What this plan deliberately doesn't include

- **Full-corpus audit** — too expensive for the 250/day cap (98+ days)
- **Re-authoring failed Phase 3 drafts** — `edge-2543` content is
  unrecoverable (was never committed); the rest are decisions for
  later
- **Cross-encoder reranking experiment** (Phase 4.5) — still OOM on
  16GB; out of scope until better hardware
- **`vault chains suggest`** post-write hook (Phase 4.3) — depends on
  Phase 3 stabilising; defer

---

## Summary in one paragraph

Pre-audit corpus is 90.9% structurally clean by regex; needs 134 title
rewrites, 407 provenance fixes, ~447 format-marker patches, plus an
AUTHORING.md template doc. The Gemini audit (math + coherence + level
fit) is feasible only as a 1,000-question stratified sample at ~2,900
calls / ~12 days. New session should build a quota-aware multi-day
audit tool first, then run the mechanical cleanups in parallel with
the daily audit cycle, then synthesise findings and update
`paper.tech`. Total wall-clock: **~2 weeks** end-to-end. Total Gemini
spend: **~3,000 calls**.
