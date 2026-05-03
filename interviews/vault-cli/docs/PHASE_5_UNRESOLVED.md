# Phase 5 — corrections that need human review

**Generated:** 2026-05-03 after the autonomous Phase 5 mass-apply +
math-verify passes.

The autonomous Phase 5 work applied **2,279 of 2,757** Gemini-proposed
corrections (82.6%). The remaining **478** were deliberately not
applied because they fail one of three safety checks. They live in
the disposition logs and need human attention to close out.

---

## Summary by skip reason

| reason | count | next step |
|---|---:|---|
| Math verification said "no" | 75 | Read rationale in `03_math_verification.json`; re-derive yourself; either re-prompt Gemini for a different fix or rewrite manually |
| Math verification said "unclear" | 14 | Same as above; case-by-case |
| Math fix has level-relabel that breaks a chain | 13 | Either relabel chain neighbors too OR keep math fix without level change |
| Level relabel is upward (against §10 Q3 policy) | 168 | Each requires a question REWRITE to match the higher level — separate authoring task; not a Phase 5 correction |
| Level relabel-down would break chain monotonicity | 138 | Same chain-restructuring decision as #3, but no math fix accompanying |
| Already-applied (no-op) | 70 | No action needed |
| **Total skipped** | **478** | |

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

## Cumulative Phase 5 status

```
2,757  proposed corrections
2,279  auto-applied + validated     (82.6%)
  478  awaiting human review        (17.4%)
   75  math 'no'              ← highest priority
   14  math 'unclear'
   13  math + level-block
  168  relabel-up
  138  chain-block (level-only)
   70  no-op / already-applied
```

The corpus is now in much better shape than it was at audit time.
Phase 6 (schema tightening + lift format gate) is now safe to attempt
once these 478 are dispositioned (or accepted as known-deferred).
