# Resume Plan — Massive Build Session (2026-04-25)

**Purpose:** hand the next Claude session everything it needs to pick up
the day's massive question-generation work without re-discovering state.

**Current branch:** `feat/massive-build-2026-04-25` (off
`audit/vault-schema-folder` ← off `dev`)
**Worktree:** `/Users/VJ/GitHub/MLSysBook-vault-audit`
**Last commit:** `24d3269c7 feat(vault): Phase 0 — competency_area cleanup + closed-enum hardening`

---

## What's already done (do NOT redo)

### From the audit branch (parent)

- 4,754 cohort-tagged IDs renamed to clean `<track>-NNNN` form
  (commit `8a5c3ff3c`).
- Redirect map at `interviews/vault/docs/id-renames-2026-04-25.yaml` +
  `interviews/staffml/src/data/id-redirects.json` — preserves shared
  links to renamed IDs. Wired into the practice page's `?q=` handler.
- 8 Playwright tests passing.
- `vault check --strict` clean.

### From this session (commit `24d3269c7`)

- **Phase 0 cleanup**: 41 malformed `competency_area` values fixed (e.g.,
  `data-pipeline-engineering` → `data`, `evaluation` → `cross-cutting`,
  `tinyml / queueing-theory` → `latency`).
- **LinkML schema**: added `CompetencyArea` closed enum. `competency_area`
  field now references it. Future malformed values fail validation.
- **Pydantic validator**: `_area()` field_validator on `Question` rejects
  anything outside `VALID_COMPETENCY_AREAS`.
- **Generator defaults raised**: `batch_size` 12 → 30, `total` 12 → 30,
  `max_calls` 10 → 20. Gemini's 1M context easily handles 30 cells/call;
  the 250/day cap rewards bigger batches.
- **`MASSIVE_BUILD_RUNBOOK.md`**: the methodology document — read this
  first if you don't know what to do next.

### Verified

- Bundle: 9,224 published, **13 canonical competency areas, 0
  malformed**.
- All 8 Playwright tests pass.
- `vault check --strict` clean.

---

## Current corpus state

```
Published: 9,224
  cloud:  4,131  (44.8%)
  edge:   1,976  (21.4%)
  mobile: 1,644  (17.8%)
  tinyml: 1,168  (12.7%)
  global:   305  ( 3.3%)

Drafts (status:draft):    275
Deleted (dedup archive):  458
Total YAMLs:              9,982

Visual-eligible (published): 17 across 8 of 10 archetypes
  Missing: collective-communication (0), kv-cache-management (0)

Top track-area gaps:
  TinyML/parallelism:    0 of ~90 expected
  Mobile/parallelism:    0 of ~127 expected
  Edge/parallelism:     11 of ~152 expected
  TinyML/networking:     2 of ~90 expected
  Global L4-L6+:    ~13% of expected density
```

---

## API budget

- **Gemini cap**: 250 calls/day
- **Used today (estimate)**: ~30 calls (audit + Phase 0 dry-runs)
- **Available**: ~220 calls
- **Plan budget**: ~80 calls (40 generation + 30 judge + 10 buffer)
- **Headroom remaining**: 140 calls for retries

---

## What to do next — execute these phases in order

Each phase is a single command (or short sequence). Stop after Phase 7
or earlier if anything looks wrong.

### Phase 1 — Run the analyzer (1 minute)

```bash
cd /Users/VJ/GitHub/MLSysBook-vault-audit
python3 interviews/vault/scripts/analyze_coverage_gaps.py \
  --total 100 --published-only
```

Output goes to `interviews/vault/_validation_results/coverage_gaps/<ts>/`.
Look at `report.md` for the priority gap ranking. Top cells should be
the TinyML/Mobile/Edge parallelism rows and Global L4-L6+ cells.

### Phase 2 — Bump loop defaults, then run (2-4 hours wall clock; 80 API calls)

First, bump the loop defaults. **Edit** `interviews/vault/scripts/iterate_coverage_loop.py`:

| Flag | Current default | New default |
|---|---|---|
| `--max-iters` | 20 | 30 |
| `--max-calls` | 60 | 80 |
| `--gen-batch-size` | 12 | 30 |
| `--gen-calls-per-iter` | 3 | 4 |
| `--judge-chunk-size` | 15 | 25 |

Specifically lines 220-226 of `iterate_coverage_loop.py`. Update both the
`default=N` values AND the help text comments.

Then run the loop:

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

Each iteration:
- 4 generation calls × 30 cells = 120 questions
- 1-2 judge calls
- ~5 minutes wall clock

The loop self-paces and stops on saturation (drop rate > 35%, gap
priority < 0.8, or convergence on the same top cell two iters in a
row).

**Expected output**: 600-1,200 generated drafts, 70-75% pass rate via
judge, 8-15 iterations before auto-stop.

### Phase 3 — Quality gate (10 min)

Spot-read 3 generated drafts per track:

```bash
ls -t interviews/vault/questions/cloud/*.yaml | head -3 | xargs -I{} cat {}
ls -t interviews/vault/questions/tinyml/*.yaml | head -3 | xargs -I{} cat {}
# etc.
```

Check the visual quality on 2 random visual drafts via Playwright by
deep-linking. Open `/practice?q=<id>` for an SVG visual that was just
rendered, eyeball whether it fits the column at 720px width without
overflow, alt text reads clean, no horizontal scroll.

### Phase 4 — Promote PASS items + rebuild bundle (5 min)

```bash
python3 interviews/vault/scripts/promote_validated.py
PYTHONPATH=interviews/vault-cli/src \
  python3 -m vault_cli.main build --legacy-json
PYTHONPATH=interviews/vault-cli/src \
  python3 -m vault_cli.main check --strict
```

Acceptance: `vault check --strict` returns exit 0, no orphan chains,
`published_count` is up by ~600-900.

### Phase 5 — Refresh paper artifacts (10 min)

```bash
# vault build re-emits corpus.json to staffml/. Mirror it to vault/:
cp interviews/staffml/src/data/corpus.json interviews/vault/corpus.json

# Then the paper-side regen sequence:
cd interviews/paper
python3 scripts/analyze_corpus.py     # legacy schema corpus_stats.json
python3 scripts/generate_figures.py    # 4 data figures
PYTHONPATH=../vault-cli/src python3 scripts/generate_macros.py
                                       # macros.tex + corpus_stats.json (overwrites legacy)

# Update hardcoded zone counts in paper.tex if shifted:
# Line ~867: "diagnosis (1{,}583), fluency (1{,}227), and evaluation (1{,}113)"
# Replace with new values from current corpus_stats.json by_zone.

pdflatex -interaction=nonstopmode paper.tex
```

Acceptance: `Output written on paper.pdf (N pages, ...)` with no
"undefined citation" errors in the output (citation warnings are pre-
existing and unrelated).

### Phase 6 — GUI verification (5 min)

```bash
# Restart dev server fresh:
pkill -f "next-server\|next dev"; sleep 1
cd /Users/VJ/GitHub/MLSysBook-vault-audit/interviews/staffml
(npx next dev > /tmp/staffml-dev.log 2>&1 &)
sleep 8
curl -sI http://localhost:3000/practice 2>&1 | head -1   # expect 200

npx playwright test tests/practice-smoke.spec.ts --reporter=list
```

Acceptance: all 8 tests pass.

Then a manual eyeball: open `http://localhost:3000/practice` in a
browser, click the area filter, confirm exactly 13 canonical entries
plus "All". This is the user-facing fix that motivated Phase 0.

### Phase 7 — Atomic commit (3 min)

```bash
cd /Users/VJ/GitHub/MLSysBook-vault-audit
git status --short  # should show vault/questions/ changes + paper artifacts

git add interviews/vault/questions/ \
        interviews/staffml/src/data/corpus.json \
        interviews/staffml/src/data/corpus-summary.json \
        interviews/staffml/src/data/vault-manifest.json \
        interviews/paper/macros.tex \
        interviews/paper/corpus_stats.json \
        interviews/paper/figures/ \
        interviews/paper/paper.tex \
        interviews/vault/_validation_results/

git commit -m "feat(vault): massive build — N drafts generated, M promoted

Phase 1 (analyzer):  top priority cells were tinyml/parallelism (0/90),
                     mobile/parallelism (0/127), edge/parallelism (11/152).
Phase 2 (loop):      <ITERS> iterations, <CALLS> API calls, <GEN> generated.
                     Auto-stop fired on: <SATURATION REASON>.
Phase 3 (quality):   spot-read 15 drafts; <Y/N> needed manual edits.
Phase 4 (promote):   <K> PASS items promoted; bundle now <P> published.
Phase 5 (paper):     macros bumped to <P>, figures rebuilt, zone-count
                     prose updated.
Phase 6 (GUI):       all 8 Playwright tests pass; area filter shows 13
                     canonical entries.

The runbook (vault/docs/MASSIVE_BUILD_RUNBOOK.md) is the methodology
this session followed; it can be re-run on any future generation day."
```

If the corpus.json hand-edit warning fires, add the trailer:
```
Vault-Override: corpus-json-hand-edit: regenerated via vault build
```

---

## Common saturation outcomes

If Phase 2's loop stops early, the auto-stop reason will be one of:

| Reason | Meaning | What to do |
|---|---|---|
| `top priority gap < 0.8` | Corpus is balanced enough that no cell is desperately empty | This is success. Move to Phase 3. |
| `DROP rate > 35%` | Gemini is hallucinating; cells we're targeting are nonsensical for some tracks | Inspect the latest iter's `judge_summary.json` to see which cells failed. Add to `TRACK_TOPIC_BLOCKLIST` in `analyze_coverage_gaps.py`. |
| `same top cell two iters in a row` | Generator can't fill the cell (likely matplotlib script crashing) | Check `_validation_results/gemini_generation/<latest>/raw_*.txt` for the source code Gemini generated; render manually with `python3 render_visuals.py --id <id>` to see the error. |
| `max-iters reached` | Hit the iteration cap before saturation | Re-run with higher `--max-iters 50` if budget allows. |
| `max-calls reached` | Burned through the API budget | Stop. We're done for the day. |

---

## What NOT to do

These are settled decisions; don't relitigate without explicit user
direction:

- ❌ Don't add `<track>/<topic>/` subdirs (ARCHITECTURE.md §3.3 — flat
  is correct).
- ❌ Don't rename more legacy IDs (already done: 4,754 renamed in commit
  `8a5c3ff3c`).
- ❌ Don't merge to dev without explicit user OK.
- ❌ Don't push to remote without explicit user OK.
- ❌ Don't change schema enum values (CompetencyArea, Track, Level, Zone,
  Status, Provenance) — those are the canonical 4-axis taxonomy.
- ❌ Don't auto-promote NEEDS_FIX items; only PASS verdicts go to
  published.
- ❌ Don't skip the Pydantic validator pass (`vault check --strict`)
  before commit.

---

## Files of interest (for context)

| File | Why |
|---|---|
| `interviews/vault/docs/MASSIVE_BUILD_RUNBOOK.md` | The full day's methodology. Read first. |
| `interviews/vault/audit/2026-04-25-schema-folder-audit.md` | Why the schema/folder is shaped the way it is. |
| `interviews/vault/CHANGELOG.md` | History of the v0.1 → v1.0 migration and what it fixed. |
| `interviews/vault/ARCHITECTURE.md` §3.3 | Why path-as-classification was rejected. |
| `interviews/vault/docs/ID_SCHEMES.md` | Why IDs are `<track>-NNNN`. |
| `interviews/vault/docs/id-renames-2026-04-25.yaml` | The 4,754 cohort→clean rename map. |
| `interviews/vault/scripts/iterate_coverage_loop.py` | The day's main driver. |
| `interviews/vault/scripts/analyze_coverage_gaps.py` | Priority ranking. |
| `interviews/vault/scripts/gemini_cli_generate_questions.py` | Batched Gemini generation. |
| `interviews/vault/scripts/gemini_cli_llm_judge.py` | Multi-criteria validator. |
| `interviews/vault/scripts/promote_validated.py` | Lifecycle flip. |
| `interviews/vault/scripts/render_visuals.py` | DOT/matplotlib → SVG. |
| `interviews/vault/scripts/fix_competency_areas.py` | Phase 0 cleanup script (one-time, can re-run safely). |

---

## One-liner status check (run first in next session)

```bash
cd /Users/VJ/GitHub/MLSysBook-vault-audit && \
  git log --oneline -5 && echo "---" && \
  git status --short | head -10 && echo "---" && \
  PYTHONPATH=interviews/vault-cli/src \
    python3 -m vault_cli.main check --strict 2>&1 | tail -3 && \
  echo "---" && \
  python3 -c "
import json
c = json.load(open('interviews/staffml/src/data/corpus.json'))
print(f'published: {len(c)}')
visuals = [q for q in c if q.get('visual')]
print(f'with visuals: {len(visuals)}')
from collections import Counter
print('areas:', sorted(set(q['competency_area'] for q in c)))
"
```

If the output shows commit `24d3269c7`, clean tree, `vault check`
passes, and 13 canonical areas — the resume state is healthy. Proceed
to Phase 1.
