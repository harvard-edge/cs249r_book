# Phase 4 audit handoff — resume guide for the next session

**Status as of 2026-05-03 (updated):** Phases 0-4 + Phase 4 backfill +
Phase 8 (CLI + cron) complete. **Ready for Phase 5 (interactive
review).**
**Branch:** `yaml-audit` (97 commits ahead of origin/dev, 0 behind — merged into
local dev when ready)
**Worktree:** `/Users/VJ/GitHub/MLSysBook-yaml-audit`
**Active workplan:** `interviews/vault-cli/docs/CORPUS_HARDENING_PLAN.md`

## Update appended 2026-05-03

The handoff doc below was written before the Phase 4 backfill ran. Key
deltas to know about:

- **Phase 4 backfill is done.** Cloud + edge failures that were
  audit-only got `--propose-fixes` passes. The merged dataset now has
  **2,757 questions with suggested_corrections** (up from 1,767),
  spanning all 5 tracks. **0 error rows** (all retried).
- **6 cloud questions migrated.** `cloud-{0048,0273,0291,0336,0418,0454}`
  had stray top-level `options`/`correct_index` (MCQ data) — moved
  into `details:` per the schema. Phase 6's `Details extra='forbid'`
  flip is now safe with no further corpus migrations.
- **Phase 8 CLI subcommand shipped.** `vault audit run / review /
  summarize / merge` wraps the underlying scripts. Cron workflow
  was already in place.

So the new session can **skip Step 1 (backfill)** in the doc below
and go straight to **Step 2 (Phase 5 interactive review)**.

The merged audit dataset for Phase 5 is at:
`interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json`

---

---

## What's done

```diff
+ Phase 0  Cleanup deprecated scripts + dead-end audit_corpus.py     ✅
+ Phase 1  Backfilled provenance: imported on 407 published YAMLs    ✅
+ Phase 2  AUTHORING.md + vault new scaffold with format-marker stubs ✅
+ Phase 3  Built audit_corpus_batched.py + _judges.py + _batching.py  ✅
+ Phase 4  Full-corpus audit run                                      ✅ (9,446 / 9,446)
```

**Phase 4 outputs (all in `interviews/vault/_pipeline/runs/`, gitignored):**

```
full-corpus-20260503/         main run dir (cloud audit-only + edge mixed + global + 140 mobile)
full-corpus-20260503-mobile/  parallel mobile run (1,824 rows with fixes)
full-corpus-20260503-tinyml/  parallel tinyml run (1,202 rows with fixes)
full-corpus-20260503-merged/  ✦ canonical merged dataset — start here
```

The merged dataset has all 9,446 questions with 1,767 already carrying
`suggested_corrections` from the `--propose-fixes` invocations.

**Phase 4 findings doc (committed):**
`interviews/vault-cli/docs/AUDIT_FINDINGS_2026-05-03.md`

---

## Final corpus state (Phase 4 results)

| gate | fail | rate |
|---|---:|---:|
| format_compliance | ~960 | 10.2% |
| level_fit | ~1,580 | 16.7% |
| coherence | ~480 | 5.1% |
| math_correct | ~330 | 3.5% |
| title_quality | ~250 placeholder + ~25 malformed | 2.9% |
| errors (need retry) | 20 (all global) | 0.2% |

Per-track failure rates: tinyml has the highest level-inflation rate
(21.4%); cloud has the most absolute math errors. Edge has higher
coherence-fail rate than other tracks (7%). See AUDIT_FINDINGS for
details and qid lists.

---

## What's left

```yaml
remaining:
  Phase 4 finalization:
    - retry 20 errored rows in global (1 invocation, ~1 batch)
    - backfill --propose-fixes on cloud's ~1,344 failures (no fixes today)
    - backfill --propose-fixes on edge's ~253 failures missing fixes
    - re-merge into full-corpus-20260503-merged/

  Phase 5  Walk ~3,300 corrections interactively                  ~6h human review
  Phase 6  Schema tightening + lift format gate                   ~2h
  Phase 7  Title-quality pass (~250 placeholders)                 ~30 calls + review
  Phase 8  Add `vault audit` CLI subcommand                       ~30 min
  Phase 9  Update paper.tech, vault publish 1.0.0, tag            ~1h
```

Cron workflow (`staffml-audit-corpus-monthly.yml`) is already shipped.

---

## How to resume

### Step 0 — sanity check the worktree

```bash
cd /Users/VJ/GitHub/MLSysBook-yaml-audit
git status                       # should be clean
git log --oneline -10            # confirm Phase 0-4 commits present
git branch                       # * yaml-audit
vault check --strict             # 10,711 loaded, 0 invariant failures
pytest interviews/vault-cli/tests/ -q  # 84 passed
ruff check interviews/vault-cli  # clean
```

### Step 1 — finish Phase 4 backfill (~5 invocations of Gemini)

The cloud track was audited with audit-only in invocations 1-7
(yesterday). To get suggested_corrections for cloud's failures, run a
propose-fixes pass against the failure-only subset.

#### Step 1a — list cloud failure qids

```bash
python3 -c "
import json
d = json.loads(open('interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json').read())
fails = [
    r['qid'] for r in d['rows']
    if r['qid'].startswith('cloud-')
    and not r.get('suggested_corrections')
    and any(r.get(g) == 'fail' for g in ('format_compliance','level_fit','coherence','math_correct'))
]
print(','.join(fails))
" > /tmp/cloud-fails-to-fix.txt

wc -w /tmp/cloud-fails-to-fix.txt   # ~1,344 qids
```

#### Step 1b — run propose-fixes on those qids

```bash
QIDS=$(cat /tmp/cloud-fails-to-fix.txt)

# Run multiple invocations until quota or qids exhausted.
# Each invocation does ~18 batches of 20 questions = ~360 qids.
# Cloud needs ~4 invocations to clear all failures.

for i in 1 2 3 4; do
  python3 interviews/vault-cli/scripts/audit_corpus_batched.py \
    --qids "$QIDS" \
    --propose-fixes \
    --workers 8 \
    --max-calls 18 \
    --output interviews/vault/_pipeline/runs/full-corpus-20260503-cloud-backfill
  # The script auto-resumes — already-fixed qids skip on each iteration.
done
```

#### Step 1c — same for edge's 253 missing-fix failures

```bash
python3 -c "
import json
d = json.loads(open('interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json').read())
fails = [
    r['qid'] for r in d['rows']
    if r['qid'].startswith('edge-')
    and not r.get('suggested_corrections')
    and any(r.get(g) == 'fail' for g in ('format_compliance','level_fit','coherence','math_correct'))
]
print(','.join(fails))
" > /tmp/edge-fails-to-fix.txt

QIDS=$(cat /tmp/edge-fails-to-fix.txt)
python3 interviews/vault-cli/scripts/audit_corpus_batched.py \
  --qids "$QIDS" \
  --propose-fixes \
  --workers 8 \
  --max-calls 18 \
  --output interviews/vault/_pipeline/runs/full-corpus-20260503-edge-backfill
```

#### Step 1d — retry the 20 errored global rows

These will resume automatically on any --tracks global run:

```bash
python3 interviews/vault-cli/scripts/audit_corpus_batched.py \
  --tracks global \
  --propose-fixes \
  --workers 8 \
  --max-calls 18 \
  --output interviews/vault/_pipeline/runs/full-corpus-20260503    # main dir, will retry errors
```

#### Step 1e — re-merge

```bash
python3 interviews/vault-cli/scripts/merge_audit_runs.py \
  --inputs interviews/vault/_pipeline/runs/full-corpus-20260503 \
           interviews/vault/_pipeline/runs/full-corpus-20260503-mobile \
           interviews/vault/_pipeline/runs/full-corpus-20260503-tinyml \
           interviews/vault/_pipeline/runs/full-corpus-20260503-cloud-backfill \
           interviews/vault/_pipeline/runs/full-corpus-20260503-edge-backfill \
  --output interviews/vault/_pipeline/runs/full-corpus-20260503-merged
```

#### Step 1f — refresh the findings doc

```bash
python3 interviews/vault-cli/scripts/summarize_audit.py \
  --input interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json \
  --output interviews/vault-cli/docs/AUDIT_FINDINGS_2026-05-04.md   # date the new file
```

Commit the new findings doc; leave the 2026-05-03 one in place as a
historical baseline.

**Estimated Gemini cost for Step 1: ~80 calls (~30% of one day's quota).**
**Estimated wall time: 1-2 hours of audit runs.**

---

### Step 2 — Phase 5: walk corrections interactively

After Step 1 there will be ~3,300 rows with `suggested_corrections`.
Use `apply_corrections.py` to walk them.

#### Step 2a — start with the safest auto-acceptable batch

Format-marker-only corrections are mechanical — auto-acceptable:

```bash
python3 interviews/vault-cli/scripts/apply_corrections.py \
  --input interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json \
  --filter-gate format_compliance \
  --auto-accept-format
```

Expect this to clear ~600-800 format-only rows automatically.

#### Step 2b — math errors (highest priority, manual review)

```bash
python3 interviews/vault-cli/scripts/apply_corrections.py \
  --input interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json \
  --filter-gate math_correct \
  --limit 50    # cap each session to ~50 to avoid review fatigue
```

Per CORPUS_HARDENING_PLAN.md §10 Q2, when math is wrong, accept napkin_math AND realistic_solution as a unit. Reject if Gemini's proposed fix changes meaning.

#### Step 2c — coherence failures by failure mode

Vendor fabrication (1 instance: cloud-0560) — likely scenario rewrite.
Physical absurdity (~70 instances) — usually a number adjustment.
Scenario/solution mismatch (~80 instances) — review case-by-case.

```bash
python3 interviews/vault-cli/scripts/apply_corrections.py \
  --input interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json \
  --filter-gate coherence \
  --limit 50
```

#### Step 2d — level-fit (relabel down per CORPUS_HARDENING_PLAN.md §10 Q3)

```bash
python3 interviews/vault-cli/scripts/apply_corrections.py \
  --input interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json \
  --filter-gate level_fit \
  --limit 100
```

The default disposition is to relabel the question to the actual level.
Reject if you want to rewrite the question UP (separate authoring task,
not Phase 5).

#### Step 2e — placeholder titles (Phase 7)

```bash
python3 interviews/vault-cli/scripts/apply_corrections.py \
  --input interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json \
  --filter-gate title_quality
```

#### Validation after each apply session

```bash
vault check --strict     # must stay green
pytest interviews/vault-cli/tests/ -q
git status               # review the YAML diffs before committing
```

Commit each disposition session as one logical commit:
- `fix(vault): format markers — N questions auto-accepted from Phase 5 review`
- `fix(vault): math errors — N questions reviewed (with notes for any non-trivial)`
- etc.

---

### Step 3 — Phase 6: schema tightening + lift format gate

Once the corpus is clean (format-failure rate ≈ 0), make the cleanliness load-bearing.

```yaml
files to edit:
  interviews/vault/schema/question_schema.yaml:
    - Add pattern constraint to Details.common_mistake
      pattern: '(?s).*\*\*The Pitfall:\*\*.*\*\*The Rationale:\*\*.*\*\*The Consequence:\*\*.*'
    - Add pattern constraint to Details.napkin_math
      pattern: '(?s).*\*\*Assumptions.*\*\*Calculations:\*\*.*\*\*Conclusion.*'
    - Make provenance required: true (no default fallback at YAML load)

  interviews/vault-cli/src/vault_cli/models.py:
    - Flip Details `model_config = ConfigDict(extra="allow")` → `extra="forbid"`
      (Pre-checked: 0 unknown extra keys across 9,446 published YAMLs)
    - Add explicit attributes for any audit-stamp fields if needed

  interviews/vault-cli/src/vault_cli/validator.py:
    - Add _format_compliance() to structural_tier (lift gate_format from
      validate_drafts.py into a published-corpus invariant)

run:
  vault codegen          # regenerate Pydantic / SQL DDL / TS types
  pytest                 # add tests covering the new invariant
  vault check --strict   # 0 failures
```

**Also fix 6 cloud questions with stray top-level `options`/`correct_index`:**
- `cloud-0048`, `cloud-0273`, `cloud-0291`, `cloud-0336`, `cloud-0418`, `cloud-0454`
- Move those fields from top-level into `details:`

---

### Step 4 — Phase 7: any remaining title-quality fixes

After Phase 5's title-quality session, re-audit just the questions that
were placeholders to verify the fixes landed:

```bash
python3 interviews/vault-cli/scripts/audit_corpus_batched.py \
  --qids <the-original-placeholder-qids> \
  --propose-fixes \
  --output interviews/vault/_pipeline/runs/full-corpus-20260503-titles-verify
```

Expect title_quality to be `good` for all of them now.

---

### Step 5 — Phase 8 second half: `vault audit` CLI subcommand

The cron workflow is already shipped. The CLI integration isn't yet.

```yaml
add new file:
  interviews/vault-cli/src/vault_cli/commands/audit.py
    - vault audit run [--all|--tracks|--qids] [--propose-fixes] ...
      → wraps audit_corpus_batched.py
    - vault audit review <run-dir> [--filter-gate ...]
      → wraps apply_corrections.py
    - vault audit summarize <run-dir> [--output ...]
      → wraps summarize_audit.py

register in:
  interviews/vault-cli/src/vault_cli/main.py
    add: from vault_cli.commands import audit; audit.register(app)
```

---

### Step 6 — Phase 9: paper update + release

```bash
# Update paper.tech with post-audit corpus stats:
#   - 9,446 published, audit pass rates per gate, per-track tables
#   - Methodology paragraph naming gemini-3.1-pro-preview as audit model
#   - Citation of audit_corpus_batched.py + AUDIT_FINDINGS_<date>.md

vault export-paper
vault build --local-json    # release_hash should roll
vault publish 1.0.0
vault verify 1.0.0 --git-ref v1.0.0   # citation-grade round-trip

git tag vault-1.0.0
```

---

## Tooling reference

All scripts live under `interviews/vault-cli/scripts/`:

| Script | What it does |
|---|---|
| `audit_corpus_batched.py` | Batched corpus audit. `--workers N --propose-fixes` |
| `apply_corrections.py` | Interactive accept/reject for proposed corrections |
| `summarize_audit.py` | Generate AUDIT_FINDINGS markdown from 01_audit.json |
| `merge_audit_runs.py` | Combine multiple per-track output dirs |
| `backfill_provenance.py` | Phase 1 helper (already run) |
| `_judges.py` | Shared prompts + Gemini-call wrapper |
| `_batching.py` | Generic char-budgeted batcher |
| `validate_drafts.py` | Single-draft multi-gate (per-question) |
| `audit_math.py` | Single-question math spot-check |
| `audit_chains_with_gemini.py` | Chain-level audit (existing, separate concern) |

Preserved-for-adaptation in `interviews/vault/scripts/` (see DEPRECATED.md):
- `gemini_backfill_question.py`, `gpt_backfill_question.py`
- `gemini_cli_generate_questions.py`, `generate.py`
- `gemini_fix_errors.py`, `deep_verify.py`

---

## Open questions that may still need answers

From `CORPUS_HARDENING_PLAN.md §10`:
1. `extra="forbid"` on `Question` (recommended: keep lenient on Question, strict on Details — needed for Phase 6)
4. Cron cadence (recommended: monthly — already implemented)
5. Per-track audit floor (recommended: introduce post-Phase-5 cleanup)
6. `audit_math.py` deprecation timing (recommended: keep one quarter)
7. AUTHORING.md maintenance hook (recommended: pre-commit field-name check)
8. Sample size for cron (recommended: full monthly — already configured)

Q2 + Q3 already answered (math errors as a unit; level relabel down).

---

## Commit log highlights from this session

```
d2621cc9e  feat(vault-cli): merge_audit_runs.py + Phase 4 findings doc
2d9330da6  fix(vault-cli): isolate gemini CLI scratch files in temp dir
e7a2a27bf  feat(ci): staffml-audit-corpus-monthly.yml — recurring corpus audit workflow
3eaac3ca9  feat(vault-cli): summarize_audit.py — Phase 4 finalization helper
1722133fa  feat(vault-cli): apply_corrections.py — interactive accept/reject
1b58a9c50  feat(vault-cli): parallel audit_corpus_batched.py with submit-stagger
12032f700  fix(vault-cli): audit_corpus_batched.py reliability fixes from canary
03031dc38  test(vault-cli): smoke tests for audit_corpus_batched batching
69cf6f0a5  feat(vault-cli): audit_corpus_batched.py — full-corpus batched audit
dd71c66ca  feat(vault-cli): _judges.py + _batching.py — shared infra
f691d6c14  feat(vault-cli): vault new scaffolds full Pitfall/Rationale/Consequence stubs
7500b9281  docs(vault): AUTHORING.md — single-source authoring reference
e8f0faa83  chore(vault): explicit provenance: imported on 407 published questions
39d567f26  feat(vault-cli): backfill_provenance.py — Phase 1 helper
3f0773706  chore(vault): restore 6 unique-capability scripts as preserved-for-adaptation
56d3ed155  chore(vault): remove 18 deprecated scripts per CORPUS_HARDENING_PLAN.md Phase 0
36f2ef592  docs(vault-cli): CORPUS_HARDENING_PLAN.md — supersedes RELEASE_AUDIT_PLAN.md
```

---

## Estimated remaining time to ship 1.0.0

```
Step 1 (Phase 4 backfill):        ~2h Gemini  (~80 calls today's quota)
Step 2 (Phase 5 review):          ~6h human   (the big sink)
Step 3 (Phase 6 schema):          ~2h
Step 4 (Phase 7 verify):          ~30 min
Step 5 (Phase 8 CLI subcommand):  ~30 min
Step 6 (Phase 9 release):         ~1h
─────────────────────────────────────────────
Total                             ~12h, mostly Phase 5 review
```

Spread over 2-3 working days.

---

## Troubleshooting

**Q: Gemini calls fail with rate-limit errors.**
A: Drop `--workers` to 4 and re-run. The script's resume picks up where it left off.

**Q: `vault check --strict` fails after applying corrections.**
A: A correction's edited YAML failed Pydantic. The script logs the qid;
investigate the diff. `apply_corrections.py` validates BEFORE writing,
so this only happens if the corpus had a stale validation issue.

**Q: pre-commit hook codespell rejects a finding doc.**
A: `summarize_audit.py:truncate_words` already addresses mid-word
truncations. If new typos slip in, lengthen the truncation budget or
add a baseline allow.

**Q: Worktree shows untracked gemini scratch files.**
A: My fix in `2d9330da6` isolates new gemini CLI scratch to a temp dir.
Old scratch from before that commit lingers — safe to `/bin/rm` (do
not use `rm` since it's aliased to `trash`).
