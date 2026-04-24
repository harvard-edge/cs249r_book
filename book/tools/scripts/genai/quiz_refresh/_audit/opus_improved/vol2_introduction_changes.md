# vol2/introduction — Quiz Improvement Change Log

**Chapter position:** 17 of 33 (Vol 2 opener)
**Source:** `book/quarto/contents/vol2/introduction/introduction.qmd`
**Prior grade (gpt-5.4 audit):** B
**Target grade:** A (per §16 canonical gold-standard patterns)
**Date:** 2026-04-24
**Model:** claude-opus-4-7
**Improved by:** opus-subagent-phase2

---

## Summary counts

| Metric | Count |
|---|---|
| Sections | 9 |
| Questions in original | 40 |
| Questions in improved | 41 |
| Rewritten (substantial upgrade) | 35 |
| Lightly revised (wording only) | 3 |
| Kept verbatim | 0 |
| Deleted | 0 |
| Added | 1 |

**Type mix (new):** MCQ 19 (46%), SHORT 13 (32%), TF 5 (12%), FILL 3 (7%), ORDER 1 (2%). Within §4 targets.

**MCQ letter distribution (new):** A=5, B=5, C=4, D=5. Balanced at generation time per §10.

**Anti-shuffle violations (§10):** 0 — every MCQ refutes distractors by content, never by letter.

**Validator status:** `python3 validate_quiz_json.py ... introduction.qmd` → OK, exit 0.

---

## Three issue patterns fixed across the chapter

1. **Trivia-fill → mechanism-inferred FILL.** The flagged `communication intensity` and `Measure Everything` fills were guessable from adjacent phrasing. Replaced with:
   - A `bisection`-bandwidth FILL in `#sec-vol2-introduction-scale-moment` where the blank is reasoned from the described worst-case cut, not from a synonym in the sentence.
   - A `Chinchilla`-frontier FILL in `#sec-vol2-intro-ai-scaling-laws-a043` where the reader infers the 2022-paper name from the described co-scaling behavior.
   - A `Reliability` Gap FILL in `#sec-vol2-introduction-constraints-scale` inferred from exponential fleet-availability decay, replacing the flagged trivia fill on "communication intensity."
   - The `Measure`-everything fill (flagged build_up_violation) became a SHORT on distributed observability instrumentation, defending against `build_up_violation` by asking what *concretely* changes at fleet scale.

2. **Build-up violations → prior-vocabulary-as-context.** Several questions reintroduced Vol 1 concepts (memory wall, roofline, arithmetic intensity, Amdahl) as if first-time. Rewrote to treat them as assumed vocabulary and test their *application* at fleet scale — e.g., §5 now derives the 5x Amdahl ceiling from a 20-percent communication share (§5 Q6) and §5 Q1 asks the reader to distinguish Communication (wire) from Coordination (logic) given a concrete profile, a move only possible when the reader already holds Vol 1's iron-law vocabulary.

3. **Easy TFs / recall-only ORDERs → misconception-targeted items.** The flagged recall-only ORDER in §5 (Fleet Law terms) was replaced by two diagnostic MCQs: one that classifies NCCL/barrier time into Communication vs Coordination, and one that uses the Fleet Law to select an intervention from a numeric breakdown. The easy "governance = False" TF in §4 was sharpened into a runtime-invariant TF in the summary, and new TFs (e.g., the 10-percent-straggler-amplification TF in §1, the distributed-failure-modes TF in §2, the Conservation of Overhead TF in §5) target real practitioner misconceptions and justify with explicit mechanisms.

---

## One substantial-rework section: `#sec-vol2-introduction-c-cube` (The C³ Taxonomy)

This section had the most consequential issues and received the deepest rewrite (6 questions, nearly all new):

- **Original Q1 (MCQ on Coordination):** choice-count of 4 single-word options with no scenario. **Now:** scenario MCQ giving 12% NCCL + 18% barrier/fault-recovery timings and asking the reader to map them to C₂ vs C₃ — tests the exact distinction the section makes, with a concrete profile.
- **Original Q4 (ORDER of Fleet Law terms):** flagged by gpt-5.4 audit as recall_only — the answer even says reordering would not change the sum. **Now:** replaced by a diagnostic MCQ on a numeric Fleet Law breakdown (T_Compute 0.9, T_Comm 0.8, T_Coord 0.4) that forces the reader to apply Amdahl-style reasoning to pick the bottleneck.
- **Original Q5 (MCQ on T_overlap):** kept the mechanism but rebuilt as a quantitative calculation — reader plugs 1000/1000 + 0.5 - T_overlap = 1.3 to derive T_overlap = 0.2 s and interpret what it represents.
- **Added Q6 (SHORT on Amdahl's 5x bound):** makes the section's hand-wave ("no amount of faster GPUs can make it more than 5x faster") into a derivation question — the reader must compute 1/0.20 = 5 and identify Communication as the un-accelerable fraction.

---

## Audit-flagged issues addressed

| Section | Original flag | Resolution |
|---|---|---|
| `#sec-vol2-introduction-c-cube` | q4 recall_only ORDER | Deleted ORDER; added diagnostic MCQ on Fleet Law breakdown + SHORT on Amdahl derivation |
| `#sec-vol2-introduction-constraints-scale` | q2 trivia_fill on "communication intensity" | Replaced with a mechanism-inferred FILL on "Reliability Gap"; CI-ratio concept moved to a scenario MCQ that asks the reader to compute CI for two workloads and rank their scaling behavior |
| `#sec-vol2-introduction-foundational-concepts` | q4 build_up_violation on "Measure everything" FILL | Replaced with a SHORT asking what specific instrumentation (clock synchronization, telemetry scalability, NCCL/barrier/scheduler tracing) changes when observability itself becomes distributed |

---

## Metadata updated

```json
{
  "generated_on": "2026-04-24",
  "model": "claude-opus-4-7",
  "improved_by": "opus-subagent-phase2"
}
```

Counts (`total_sections`, `sections_with_quizzes`, `sections_without_quizzes`) preserved at 9/9/0 per the validator's consistency check.
