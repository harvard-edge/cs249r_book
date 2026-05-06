# Phase 5 — corrections that need human review

**Generated:** 2026-05-03 after the autonomous Phase 5 mass-apply +
math-verify passes. **Updated 2026-05-04** after two follow-up slices:
math-skip-level apply (15 fixes) and math-finish queue drain (66 fixes).

Autonomous Phase 5 work has now applied **2,372 of 2,757** Gemini-proposed
corrections (86.0%). **385 residual** corrections still differ from the
YAML and need human attention; these are accepted as known-deferred
ahead of Phase 6.

---

## Summary by skip reason (post-2026-05-04 drain)

| reason | count | next step |
|---|---:|---|
| Level-only diff (relabel-up or chain-monotonicity-block) | 253 | Authoring rewrite OR chain restructuring per §10 Q3; not pure Phase 5 |
| Math nm/rs diff (Gemini-2 said "no" + nm-only edge cases) | 77 | Read rationale in `03_math_verification.json`; manual rewrite per case |
| Level + math combo blocked | 35 | Math already applied via `apply_math_skip_level.py`; level still pending |
| Misc small combos (cm, title, multi-field) | 20 | Case-by-case |
| **Total residual** | **385** | |

### What changed since the original 2026-05-03 doc

- Original doc claimed **70 already-applied no-ops** — actually those were
  **70 unverified math candidates** that `verify_math_corrections.py`
  skipped because its "already-applied" guard checked only
  `realistic_solution`. The guard was widened on 2026-05-04 to also
  consider `napkin_math` / `common_mistake`, and the queue was drained:
  68 verdicts were `yes` (66 applied + 2 level-blocked → math fields
  applied via `apply_math_skip_level.py`), 2 verdicts were `no`.
- 13 level-block math fixes from the original doc's "Math fix has
  level-relabel that breaks a chain" bucket got their math fields
  applied via `apply_math_skip_level.py` (level relabel still deferred).
- Original 70 "already-applied (no-op)" was a doc accounting error;
  no such bucket exists in the disposition logs.

---

## Where the disposition logs live

```
interviews/vault/_pipeline/runs/full-corpus-20260503-merged/
├── 01_audit.json              full audit dataset (input)
├── 02_mass_apply.json          per-qid result for the low-risk auto-apply pass
├── 03_math_verification.json   per-qid Gemini verification verdict (yes/no/unclear)
└── 04_math_applied.json        per-qid result for the math auto-apply pass
```

These are gitignored (regenerable from the audit JSON), so they live
on the worktree disk only. To inspect:

```bash
# All "no" verdicts on math:
python3 -c "
import json
d = json.loads(open('interviews/vault/_pipeline/runs/full-corpus-20260503-merged/03_math_verification.json').read())
for v in d['verdicts']:
    if v['verdict'] == 'no':
        print(f'{v[\"qid\"]}: {v[\"rationale\"]}')
"

# All chain-monotonicity blocks from low-risk pass:
python3 -c "
import json
d = json.loads(open('interviews/vault/_pipeline/runs/full-corpus-20260503-merged/02_mass_apply.json').read())
for d in d['dispositions']:
    if d['result'] == 'chain-monotonicity-block':
        print(f'{d[\"qid\"]}: {d.get(\"error\", \"\")}')"
```

---

## Recommended review workflow

### A. Math "no" verdicts (75 questions) — highest priority

These are questions where Gemini's first proposal was rejected by an
independent second-eye Gemini call. Two interpretations:

1. The first proposal was genuinely wrong — Gemini hallucinated the fix.
2. The second pass was overly strict — the fix is fine.

Either way, a human needs to pick. Read the original audit's
`level_fit_rationale` + `coherence_rationale` + `math_errors` to
understand the original failure, then look at Gemini's proposed
correction in `01_audit.json`'s `suggested_corrections.napkin_math`
field, then decide.

For the questions where the original audit found genuine math errors
(arithmetic-off-by-X, contradiction between scenario and napkin), a
manual rewrite is usually quicker than another Gemini round-trip.

### B. Relabel-up blocks (168 questions)

Per CORPUS_HARDENING_PLAN.md §10 Q3, when level inflation is the
diagnosis (claimed L4 but reads as L2), default policy is to RELABEL
DOWN — move the level field to match what the question actually
demands.

The 168 in this category are the OPPOSITE: Gemini said "this question
deserves a HIGHER level than claimed." That means either:

- Relabel-up policy was wrong here (the question is genuinely bigger
  than its label) — accept the relabel-up
- OR the question needs to be rewritten DOWN to actually match the
  claimed level — separate authoring task

Triage in batches by track and topic. Some fraction is genuine
under-stamping; rest is mislabel that needs re-authoring.

### C. Chain-monotonicity blocks (138 + 13 = 151 questions)

A level relabel was blocked because applying it would break the
chains.json invariant that a chain's question levels are
non-decreasing along position.

For each blocked qid:

1. Identify which chain(s) it's in (`02_mass_apply.json`'s `error`
   field names the chain).
2. Walk the chain — if the BLOCKED relabel makes pedagogical sense
   for THIS question, it might mean the chain ordering is wrong (the
   question should be at a different position) or the chain needs a
   member added/removed.
3. Two paths forward:
   a. Move the question OUT of the chain (`vault chain unlink ...`),
      then apply the relabel.
   b. Restructure the chain — possibly merge two chains, drop a
      member, reorder.

These are the chain-restructuring decisions from
CHAIN_ROADMAP.md territory. Don't auto-do.

### D. Math fix + level-block (13 questions)

Subset of (C) where a math fix was ready but its accompanying level
relabel was chain-blocked. Two options:

1. Apply just the math fix (skip the level part) — easy. The math
   correction is independently valid.
2. Restructure the chain first, then apply both.

### E. The 70 "already applied" entries — no action needed

These are questions where the YAML's current state matches the
proposed correction. Pre-existed somehow.

---

## Re-running this analysis

Disposition logs persist in `_pipeline/runs/`, so re-running the same
mass-apply + verify scripts produces the same results
(idempotent — already-applied corrections skip).

To target a specific category for re-review:

```bash
# Just the math 'no' verdicts:
vault audit review \
    --input interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json \
    --filter-gate math_correct \
    --limit 25
```

Use `vault audit review` (which wraps `apply_corrections.py`) to walk
each one interactively. The resumable disposition sidecar tracks
your progress.

---

## Cumulative Phase 5 status (post-2026-05-04 drain)

```
2,757  proposed corrections
2,372  auto-applied + validated     (86.0%)
  385  known-deferred                (14.0%)
   253  level-only (relabel-up + chain-monotonicity-block)
    77  math nm/rs (Gemini-2 'no' + nm-only divergence cases)
    35  level + math combo (math applied; level still deferred)
    20  misc small combos
```

Phase 6 (schema tightening + lift format gate) proceeds with these 385
accepted as known-deferred. The cron audit workflow
(`.github/workflows/staffml-audit-corpus-monthly.yml`) will resurface
them on the next run, at which point per-category review can resume.

### Sidecar pipeline dirs (post-drain, gitignored)

```
interviews/vault/_pipeline/runs/
├── full-corpus-20260503-merged/        original autonomous Phase 5 outputs
│   └── 05_math_skip_level.json         13 math-only applies for level-blocked
└── full-corpus-20260503-mathfinish/    2026-05-04 queue-drain run (70 unverified)
    ├── 01_audit.json                    filtered audit (70 rows only)
    ├── 03_math_verification.json        70 verdicts (68 yes, 2 no)
    ├── 04_math_applied.json             66 applied, 2 level-block
    └── 05_math_skip_level.json          2 math-only applies for level-blocked
```
