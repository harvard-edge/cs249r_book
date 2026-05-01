# Phase 3 Draft Review Guide

This is the human review handoff for LLM-authored questions produced by
[`scripts/generate_question_for_gap.py`](../scripts/generate_question_for_gap.py)
and validated by
[`scripts/validate_drafts.py`](../scripts/validate_drafts.py). It tells
you what to actually look at, in what order, and where the LLM gates can
fool you.

> Background: Phase 3 of [`CHAIN_ROADMAP.md`](CHAIN_ROADMAP.md). The
> pilot landed 4 promoted drafts and 1 fail (edge-2535) in the corpus
> on 2026-05-01.

---

## TL;DR — the review pipeline

```
.yaml.draft files (in tree)
        ↓
validate_drafts.py            ← 5 LLM-judged gates → scorecard JSON
        ↓
human review (this guide)     ← you, ~10-15 min for the pilot
        ↓
promote_drafts.py             ← rename, registry, status flip
        ↓
vault check && vault build
        ↓
build_chains_with_gemini.py   ← absorbs new questions into chains
```

---

## What the gates catch (and what they miss)

The five gates in `validate_drafts.py`:

| gate          | catches                                             | misses |
|---------------|-----------------------------------------------------|--------|
| schema        | malformed YAML, wrong enum values, missing fields   | bad content with valid shape |
| originality   | near-duplicates of existing in-bucket questions (cosine ≥ 0.92) | semantic duplicates worded differently |
| level_fit     | obvious level mismatch (L2 prose at L5 slot)        | subtle cognitive-load drift |
| coherence     | scenario / question / solution contradicting each other | factually wrong but internally consistent answers |
| bridge        | candidate doesn't pedagogically chain between anchors | candidate that *does* chain but is pedagogically weak |

**The gates are triage triggers, not authoritative.** A pass means
"worth reading"; a fail means "almost certainly skip." Reverse is not
true: passes routinely include subtly-bad questions.

---

## What to read, in what order

For each candidate draft (typical: 2-3 minutes per draft):

### 1. Read the scenario aloud. Does it sound like a real ML systems situation?

Watch for:
- **Magic numbers** that don't relate to the question. ("A 13.7B-parameter model" — fine. "An accelerator running at 4.2 TFLOPS" — pause: is 4.2 actually anything that exists?)
- **Vendor-name confabulation.** Gemini will sometimes invent hardware ("the Coral Edge TPU XL", "Snapdragon 9-Gen Plus") that don't exist. Cross-check.
- **Fabricated benchmarks.** If the scenario claims "MobileBERT achieves 67.3 ms on …", treat the number as suspect unless it's standard.

### 2. Read the question. Is the cognitive demand right for the level?

Bloom mapping for reference:

| level | typical verb                  | typical task |
|-------|-------------------------------|--------------|
| L1 | identify, name                | "Which of these is the …?" |
| L2 | explain, describe             | "Why does the system …?" |
| L3 | compute, apply                | "Calculate the …" |
| L4 | analyze, decompose            | "Decompose the bottleneck" |
| L5 | evaluate, judge, recommend    | "Which approach minimizes …?" |
| L6+ | design, architect, synthesize | "Architect a system that …" |

The most common Gemini failure mode: **stamping a higher level than the question deserves**. An L4 candidate that's actually a definition recall is worse than a clean L2 — it pollutes chains.

### 3. Read the realistic_solution. Is it the actual answer, not adjacent?

Watch for:
- **Restating the question** instead of answering it
- **Generic systems advice** ("profile your workload, identify the bottleneck") that doesn't engage with the specific scenario
- **Answers a slightly different question** than what was asked

### 4. Read common_mistake. Is the pitfall a real one a candidate would actually make?

The format is **`Pitfall / Rationale / Consequence`**. Watch for:
- Pitfalls that no one would actually make
- Pitfalls that contradict the realistic_solution
- "Consequence" that's just a restatement of the pitfall

### 5. Read napkin_math (when present). Does the arithmetic check out?

Walk the calculation. Gemini is competent but not infallible at arithmetic. If the answer hinges on a number, verify it.

### 6. (Bridge questions only) Compare to the two between-questions

Open `<bucket>/edge-XXXX.yaml` (the lower anchor) and `<bucket>/edge-YYYY.yaml`
(the higher anchor) side-by-side with the candidate. Three checks:

- **Same scenario thread?** The bridge should feel like a step in the same
  setup, not a fresh system.
- **Strict +1 progression?** Lower → bridge → higher should each be one
  Bloom step, not a +2 skip.
- **No content overlap?** The bridge shouldn't restate either anchor's
  premise.

The `_authoring.gap` block in each draft tells you exactly what gap it
was built for. (After `promote_drafts.py`, that's a `gap-bridge:<from>-<to>`
tag.)

---

## Fast-path checklist (for the impatient)

For each draft, ask yourself in order:

1. **Would I have written this as an interview author?** If yes → flag for promotion. If "kind of" → keep reading. If no → reject.
2. **Does the scenario reference anything that doesn't exist or is mis-attributed?** If yes → reject.
3. **Does the level match the cognitive demand?** If no → reject (don't try to "fix the level field"; the body is wrong).
4. **Is the answer correct?** If no → reject.

If a draft passes all four, it's safe to promote.

---

## What to do with each verdict

### Promote with publish

```bash
python3 interviews/vault-cli/scripts/promote_drafts.py \
  --qids edge-2536,mobile-2146 \
  --publish \
  --reviewed-by <your-handle>
```

This sets `status: published`, `human_reviewed.status: verified`. The
question enters the release set. Re-run the chain build to absorb it:

```bash
vault build --local-json
python3 interviews/vault-cli/scripts/build_chains_with_gemini.py --all \
  --output interviews/vault/chains.proposed.json
python3 interviews/vault-cli/scripts/apply_proposed_chains.py
vault check --strict
```

### Promote as draft (not yet ready for release)

Leaves `status: draft` so the question stays out of the release set
while you iterate on it:

```bash
python3 interviews/vault-cli/scripts/promote_drafts.py --qids edge-2536
```

Useful when the bones are right but the answer needs an edit pass.

### Reject

Just delete the `.yaml.draft` file:

```bash
git rm interviews/vault/questions/edge/latency/edge-2535.yaml.draft
```

The id slot stays unallocated; the next batch run will reuse it
naturally (the allocator scans corpus + existing drafts, not the
registry).

### Edit then retry

Open the `.yaml.draft`, fix the issue, then re-run the validator on
just that file:

```bash
python3 interviews/vault-cli/scripts/validate_drafts.py \
  --scope interviews/vault/questions/edge/latency/ \
  --output /tmp/retry-scorecard.json
```

If it now passes, promote.

---

## The 4 pilot drafts (as of 2026-05-01)

| qid | gap | scorecard verdict | top-neighbour cosine | level_fit rationale |
|---|---|---|---|---|
| edge-2536 | edge/pruning-sparsity L4 | pass | TODO (see scorecard) | TODO |
| edge-2537 | edge/tco-cost-modeling L3 | pass | TODO | TODO |
| mobile-2146 | mobile/duty-cycling L3 | pass | TODO | TODO |
| mobile-2147 | mobile/model-format-conversion L2 | pass | TODO | TODO |
| **edge-2535** | edge/latency-decomposition L3 | **fail** (cos=0.933 vs edge-1883) | — | — |

Open `interviews/vault/draft-validation-scorecard.json` for the full
per-row detail (originality cosine, judge rationales).

---

## When the LLM gates are wrong

If you find yourself rejecting a "pass" draft, the value is in the
**why**. Common reasons surfacing now:

- **Pretty prose hides a wrong answer.** The coherence judge treats
  internal consistency as truth.
- **Scenario invents hardware or benchmarks.** No gate currently
  checks for vendor-name fabrication.
- **Gap rationale itself was off.** Sometimes Gemini's gap detection in
  Phase 1.4 hallucinated a missing rung that doesn't actually exist.
  In that case the bridge will feel forced — fix the gap, not the draft.

If a pattern emerges across multiple drafts, file an issue against the
prompt in `generate_question_for_gap.py` rather than fixing each draft
by hand. The roadmap calls for prompt iteration as a Phase 3 follow-up.

---

## Once you're done

1. Commit the promotion (`promote_drafts.py` does the rename + registry,
   you do `git add -A && git commit`).
2. Re-run the chain builder so the new questions get absorbed.
3. Push `yaml-audit`.
4. Update [`CHAIN_ROADMAP.md`](CHAIN_ROADMAP.md) Progress Log with the
   review outcome (which qids accepted, which rejected, and any
   prompt-improvement notes).

---

## Cost reminder

Each draft costs ~4 Gemini calls (1 generation + 3 judge calls).
A 30-gap batch is ~120 calls (under the 250/day Pro cap). Budget the
day's call count if you're scaling up.
