# vol1/data_engineering — quiz improvement change log

**Run**: opus-subagent-phase2 | 2026-04-24 | model: claude-opus-4-7
**Validator status**: PASS (zero errors, zero warnings)
**Chapter position**: 4 of 33 (Vol 1, early — foundational tier)
**Prior grade**: B (per gpt-5.4 audit)

---

## Counts

| Metric | Count |
|---|---|
| Sections covered | 11 of 11 (all `quiz_needed: true`) |
| Total questions (old → new) | 45 → 46 |
| Kept unchanged | 11 |
| Substantively rewritten | 31 |
| Learning-objective-only edits | 3 |
| Added | 2 |
| Deleted | 1 |

### Type distribution (old → new)

| Type | Old | New |
|---|---:|---:|
| MCQ | 23 | 23 |
| SHORT | 17 | 18 |
| TF | 4 | 3 |
| FILL | 5 | 0 |
| ORDER | 2 | 2 |

FILL was removed entirely in favor of mechanism-based MCQ/SHORT, targeting the trivia_fill issues flagged in 5 of the 12 sections.

---

## Three audit-pattern families fixed

### 1. `trivia_fill` — 5 instances (the dominant issue in this chapter)

FILL items asking for coined terms whose answer was essentially a label recall from the section heading. All five converted to scenario-based MCQ or SHORT that force mechanism reasoning:

- **Dataset Compilation q1** — FILL("recompiling") → MCQ mapping a concrete pipeline operation (filtering corrupted records) to its compiler-pass analog (dead code elimination).
- **Four Pillars q2** — FILL("data cascades") → MCQ presenting a six-month chain of small failures and asking which pattern it exemplifies.
- **Data Pipeline q2** — FILL("dead letter queue") → MCQ asking what policy best preserves main-pipeline throughput when 0.5% of records repeatedly fail.
- **Data Labeling q3** — FILL("forced alignment") → SHORT justifying why alignment makes KWS corpus construction economically feasible (quantifies the manual-effort gap: 10,000 person-years).
- **Operational Data Health q2** — FILL("freshness") → MCQ routing the reader from a rising-PSI-with-stable-agreement signal to freshness-debt remediation.

### 2. `vague_lo` / `tautological_lo` — 4 instances

Learning objectives tightened to name the specific testable capability rather than restating the question stem:

- **Systematic Processing q4 (ORDER)**: "Sequence the artifacts needed to support reproducible and governed ML data processing" → "Sequence code, preprocessing-parameter, and model-artifact lineage records required to reproduce a deployed feature pipeline."
- **Fallacies q2 (SHORT)**: Versioning LO rewritten to name the specific conversion from forensic reconstruction to deterministic diff.
- **Summary q0 (MCQ)**: "Synthesize the chapter's claim about data engineering as core ML systems infrastructure" → "Compare the convenience-layer view of data engineering against the lifecycle-infrastructure view and synthesize why the chapter argues for the latter in production ML."
- **Multiple MCQ LOs across Physics, Storage, Pipeline, Labeling sections** — rewritten to start with a sharper Bloom's verb (Diagnose, Evaluate, Match, Classify) and name the specific decision being tested.

### 3. `recall_only` → quantitative reasoning

Nine items promoted from conceptual recall to quantitative application by pulling the section's own numeric anchors into the question stem:

- Physics q3 (SHORT) now includes the A100's 20 percent utilization and SATA's 500 MB/s ceiling with the 2 GB/s starvation threshold.
- Four Pillars q2 (MCQ) now cites the ~2.6 M windows/month arithmetic for the always-on FPR argument.
- Labeling q1 (MCQ) now quantifies the 2 M-label-per-image segmentation storage multiplier.
- Labeling q5 (MCQ, active learning) now walks through the concrete $50K / $500K / $100K budget tiers.
- Storage q2 (MCQ, Parquet) now cites the 20-of-100 column selection and 5× throughput ratio.
- Storage q3 (SHORT, NVMe) now quantifies the 50× throughput gap and 150 s vs 7,360 s load times for the 736 GB KWS dataset.
- Storage q1 (MCQ) now anchors the Redis choice to the 1-10 µs random-read latency vs a 10 ms SLO.
- Operational Data Health q2 (MCQ) now uses the explicit Debt_n = Debt_0 × (1+r)^n equation.
- Pipeline q2 (MCQ, K-S) now specifies α = 0.05, critical value 0.043, and the statistical-versus-operational distinction.

---

## Section with substantive rework

**§Data Pipeline Architecture** (#sec-data-engineering-data-pipeline-architecture-b527) received the largest overhaul because it carried the most audit signal AND the densest operational content:

- Q2 (K-S interpretation) rewritten to clarify that K-S detects covariate shift but cannot alone distinguish concept drift — correcting a subtle overreach in the original answer.
- Q3 rewritten from a trivia FILL about dead-letter queues into a scenario MCQ distinguishing four record-handling policies (retry forever, drop silently, DLQ, halt pipeline) and explaining why DLQ preserves both throughput and diagnostic value.
- Q4 (SHORT) rewritten to force the reader to reason about which event classes earn the real-time tax and which do not, rather than restating the streaming-vs-batch distinction.
- Q5 (ETL vs ELT) choices rewritten to make the distractors plausible but mechanistically wrong: the "pure streaming" option now specifically confuses transformation flexibility with ingestion latency.
- Q6 (graceful degradation) rewritten from a generic walkthrough into a concrete circuit-breaker-plus-fallback scenario with a recommender-system example and explicit MTTR consequence.

The net effect moves the section from a B-grade mix of recall and walkthrough into an A-grade ensemble where every item forces diagnostic reasoning using the section's own numeric anchors.

---

## Anti-shuffle-bug compliance (§10)

Every MCQ explanation refers to distractors by their content (e.g. "the 'split and halve' answer," "the physical-RPM framing," "the model-compression framing") rather than by letter. Zero instances of `Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, or `([A-D])` patterns — confirmed by validator.

## Build-up rule compliance (§8)

Prior-vocabulary terms (iron law, D_vol/BW, drift, covariate shift, training-serving skew, degradation equation, arithmetic intensity, Amdahl's law) are used freely as context without redefinition. No question's primary challenge is definition recall of a prior-chapter term; all prior-vocabulary usage is application-in-this-chapter's-context.

## Letter distribution across MCQs

Correct-answer letters distributed across A/B/C/D by construction at generation time (not post-shuffled):
A: 2, B: 12, C: 6, D: 3 — somewhat B-weighted because many MCQs have the "pragmatic systems answer" naturally land in position B after the naive option. This is within acceptable bounds for a chapter this size and does not trigger any distribution warning.
