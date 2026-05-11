# Massive-Build Runbook

The repeatable pipeline for adding hundreds of questions in a single day.
Use this for any "let's grow the corpus" session, not just the
2026-04-25 build that motivated it.

The runbook is opinionated about three things:

1. **Pre-flight is non-negotiable.** Cleaning malformed values in the
   existing corpus before generating new content prevents
   amplification: bad data in begets more bad data out.
2. **Maximize each Gemini call.** The 250-call/day cap is the binding
   constraint. Use batch sizes of 30-40 cells for text-only and 15-20
   for visual-bearing cells. Anything smaller wastes context.
3. **Iterate, don't bulk-generate.** Run the iterative coverage loop
   with a generous budget and let saturation auto-stop the work. This
   prevents over-filling cells the analyzer no longer flags.

---

## Phase 0 — Pre-flight cleanup (one-time, 10 minutes)

Run before any new generation in a session. Intent: get the corpus to
a clean baseline so new content is authored against valid data.

### Tasks

1. **Survey malformed `competency_area` values** in the existing
   corpus. Look for values that aren't one of the 13 canonical areas:
   `deployment`, `parallelism`, `networking`, `latency`, `memory`,
   `compute`, `data`, `power`, `precision`, `reliability`,
   `optimization`, `architecture`, `cross-cutting`.

2. **Write a remap table** from each malformed value to its closest
   canonical area:

   | Malformed seen | Canonical |
   |---|---|
   | `data-pipeline-engineering` | `data` |
   | `duty-cycling` | `power` |
   | `kv-cache-management` | `architecture` |
   | `memory-hierarchy-design` | `memory` |
   | `interconnect-topology` | `networking` |
   | `network-bandwidth-bottlenecks` | `networking` |
   | `pipeline-parallelism` | `parallelism` |
   | `queueing-theory` | `latency` |
   | `fault-tolerance-checkpointing` | `reliability` |
   | `quantization-fundamentals` | `precision` |
   | `communication-computation-overlap` | `optimization` |
   | `compute-cost-estimation` | `compute` |
   | `diagnosis` / `evaluation` (zone names!) | `cross-cutting` |
   | `<track> / <topic>` slash-form | by topic mapping above |

3. **Apply the fix** in-place. One commit, atomic.

4. **Add the LinkML enum constraint** so `competency_area` rejects
   anything outside the 13 canonical values at validation time.
   Codegen Pydantic from the updated LinkML schema.

5. **Rebuild bundle** (`vault build --local-json`) and verify the
   GUI's area filter shows exactly 13 entries plus "All".

### Acceptance

- `vault check --strict` passes.
- GUI area filter has 14 entries (`All` + 13 canonical), no
  topic-shaped strings.
- Pre-commit `vault-schema-drift` hook passes.

---

## Phase 1 — Establish the day's targets (5 min)

Run the analyzer to surface today's gap priorities:

```bash
python3 interviews/vault/scripts/analyze_coverage_gaps.py \
  --total 100 --published-only
```

The output `report.md` ranks every weak cell by priority. Use the top
40-60 cells as Phase 2's batch targets.

Categorize the recommendations into three target families:

| Family | Source of cells | Why we care |
|---|---|---|
| **A. Track × area dead zones** | track_area_gaps with priority > 1.0 | Whole quadrants of the corpus are empty (e.g., TinyML/parallelism) |
| **B. Global track L4-L6+** | track_zone_level_gaps for `track == global` | Global track is at ~13% of expected density; cross-track concept questions are rare |
| **C. Visual archetype gaps** | visual_topic_counts < 3 | Visual filter pool is the user-facing "wow" — needs density |

---

## Phase 2 — Iterative generation (rest of the day, ~80 calls)

The single command. Run it, then walk away (or watch the logs).

```bash
python3 interviews/vault/scripts/iterate_coverage_loop.py \
  --max-iters 30 \
  --max-calls 80 \
  --gen-batch-size 30 \
  --gen-calls-per-iter 4 \
  --judge-chunk-size 25 \
  --visual-each-iter \
  --gap-threshold 0.8 \
  --max-drop-rate 0.35
```

**What it does each iteration**:

1. Re-runs the analyzer (gap priorities update as new cells fill).
2. Picks `30 × 4 = 120` top-priority cells.
3. Issues 4 Gemini calls of 30 cells each (each call returns a JSON
   array of 30 questions in one shot — this is the
   "make-effective-use-of-context" rule).
4. Renders any DOT/matplotlib visuals.
5. Runs LLM-as-judge in chunks of 25 to validate the new drafts.
6. Drops `verdict: DROP` items, keeps `PASS` and `NEEDS_FIX` as drafts.
7. Records iteration to `_validation_results/coverage_loop/<ts>/`.
8. Halts on saturation (drop rate > 35%, gap priority < 0.8, or two
   consecutive iters target the same top cell).

**Why these numbers**:

- `batch-size 30` — Gemini's context easily handles 30 cells × ~600 tokens
  question + ~1500 tokens constants reference. ~75K input tokens per
  call, well under the 1M context limit.
- `calls-per-iter 4` — caps each iter at 120 questions before judging.
  4 generation + 1 judge = 5 calls per iter.
- `max-iters 30 / max-calls 80` — gives the loop room to keep pushing
  if the corpus keeps yielding healthy drops; budget caps prevent
  runaway.
- `gap-threshold 0.8` — stop when the worst remaining gap has priority
  below 0.8 (≈ 20% of expected, which is "good enough" for any cell).
- `max-drop-rate 0.35` — 35% drop rate means Gemini is starting to
  hallucinate; stop before we waste budget on garbage.

**Expected output (based on yesterday's loop runs)**:

- ~600-1,200 generated drafts
- Pass rate ~65-75% via judge → ~400-900 promotable
- 8-15 iterations before saturation auto-stop fires

---

## Phase 3 — Quality gate (15 min)

Before promoting, eyeball the generated content:

1. **Visual rendering check** via Playwright on 5 random visual drafts.
   Confirm SVGs render inline at column width without overflow, alt
   text is present, layout is clean.

2. **MCQ/non-MCQ ratio check.** If the corpus's MCQ proportion drops
   below 12% from current 17% (drafts are generally non-MCQ), we're
   skewing toward open-ended. Decide whether to enrich with MCQ in a
   follow-up or accept the drift.

3. **Spot-read 3 random drafts per track.** Quick sanity on scenario
   realism, math correctness, hardware citations.

---

## Phase 4 — Promotion + bundle rebuild (5 min)

```bash
python3 interviews/vault/scripts/promote_validated.py
PYTHONPATH=interviews/vault-cli/src \
  python3 -m vault_cli.main build --local-json
PYTHONPATH=interviews/vault-cli/src \
  python3 -m vault_cli.main check --strict
```

Acceptance: `vault check --strict` passes, manifest matches corpus,
no orphan chains.

---

## Phase 5 — Paper refresh (10 min)

```bash
cd interviews/paper
python3 scripts/analyze_corpus.py
python3 scripts/generate_figures.py
PYTHONPATH=../vault-cli/src python3 scripts/generate_macros.py
pdflatex -interaction=nonstopmode paper.tex
```

Acceptance: paper builds clean, page count stable, zone-count prose
in §sec:coverage updated to match the new top-3 zones.

Hardcoded zone counts in `paper.tex` need a manual edit if the order
or values changed materially (the script doesn't rewrite prose, only
macros).

---

## Phase 6 — Verify the GUI (5 min)

```bash
cd interviews/staffml
npx playwright test tests/practice-smoke.spec.ts --reporter=list
```

All 8 tests must pass:

1-3: existing layout + safeguard
4: visual filter at L5 returns non-empty pool with inline SVG
5: chained-only filter reduces but stays non-empty
6: `?q=<known>` deep-link surfaces the right question
7: `?q=<unknown>` shows the not-found banner
8: `?q=<legacy-cohort-id>` resolves through the redirect map

Then a manual eyeball pass on `/practice` itself: open the area filter,
confirm exactly 13 canonical entries.

---

## Phase 7 — Atomic commit + handoff (5 min)

```bash
git add -A interviews/
git commit -m "feat(vault): massive build — N drafts generated, M promoted"
```

Commit message names the phase counts (Phase 1 N, Phase 2 N, etc.) and
the saturation reason that stopped the loop. This becomes the audit
trail for the day.

---

## Re-running the runbook

The whole sequence is designed to be re-runnable. To execute another
massive build session next week:

1. Pre-flight cleanup (typically a no-op if the schema enum is in
   place).
2. Phase 1 analyzer.
3. Phase 2 loop with the same flags.
4. Phases 3-7 as above.

Each session adds ~600-1,200 questions; saturation increasingly fires
earlier as the corpus matures.

---

## Open questions

- **MCQ enrichment**: today's drafts are uniformly open-ended. The MCQ
  proportion in published is 17%; left untouched, that drops over
  time. A separate "MCQ-only" mode for the generator might be worth
  building.
- **Visual archetype expansion**: the 10 archetypes in
  `audit_visual_questions.py` are the high-confidence pedagogical
  visuals. Expanding to 15-18 archetypes (attention compute graph,
  MoE shuffle, etc.) is a separate authoring exercise — the generator
  can produce them once the catalog lists them.
- **Cross-model judge**: today both generation and judging use
  `gemini-3.1-pro-preview`. Adding a Claude-side judge call (via
  Anthropic API) would give true cross-model agreement. Worth doing
  for production-grade releases.

---

## Saturation playbook

If the loop's auto-stop fires on **drop rate > 35%**:

- Inspect the most recent `_validation_results/coverage_loop/.../iter_NN/judge_summary.json`.
- Common causes: (a) the analyzer is recommending cells where the
  topic doesn't apply to the track (TinyML / pipeline-parallelism is
  nonsensical); (b) the prompt's hardware reference is missing
  constants for an unusual cell.
- Fix: update `TRACK_TOPIC_BLOCKLIST` in
  `analyze_coverage_gaps.py` if it's the first cause, or extend the
  hardware reference block in the generator if the second.

If auto-stop fires on **gap priority < 0.8**:

- This is success. The corpus is balanced enough that the analyzer
  can't identify a critical gap.
- Re-run the analyzer the next day — new corpus state may surface
  different priorities (especially after promotion).

If auto-stop fires on **same top cell two iters in a row**:

- The analyzer keeps recommending a cell the generator can't fill
  (usually visual cells where the source artifact fails to render).
- Inspect the rendered SVG output for the cells in that iter; check
  matplotlib script syntax.
