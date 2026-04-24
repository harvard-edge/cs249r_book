# vol1/ml_ops — Phase 2 quiz improvement change log

**Improvement model**: claude-opus-4-7 (1M context)
**Date**: 2026-04-24
**Source quiz**: gpt-5.4 corpus (graded "strong overall" by gpt-5.4 self-audit)
**Audit context**: `_audit/contexts/vol1_ml_ops.md` (chapter 14 of 33; 426-term prior vocab)

## Assigned grade: A-

The gpt-5.4 baseline for ml_ops was one of the better chapters in the corpus — scenario-based, grounded in the chapter's operational framing, appropriately advanced for a 42%-through-the-book reader. The audit flagged eight discrete issues across nine sections (one per section except §8 Fallacies, which had zero). After rewrites, every question in every section passes the §6 quality bar: each tests reasoning over recall, every MCQ refutes at least one distractor by content (§10 anti-shuffle compliant), and no FILL item is guessable from an adjacent phrase. The ceiling on the grade is that three SHORT items retain near-minimum length at the lower bound; with more passes I would stretch two of them toward a full three-move structure.

## Per-section breakdown

### §1 MLOps Overview (5 → 5 questions)

- **q1–q4** — KEPT (strong originals; scenario-based, content-refuted distractors).
- **q5 FILL on 'ML node' → SHORT on operational scope** — REPLACED per `recall_only` audit. Original blank-fills "ML node" from a surrounding sentence that essentially repeats the textbook definition. New SHORT inverts the test: the reader must *apply* the ML-node concept by naming two concrete signals (multi-model feature coupling, multi-region coordination) that push a team out of single-model operations and into platform-scale practice. The term now appears in the answer, not the stem.

### §2 Principles and Foundations (5 → 5 questions)

- **q1–q3, q5** — KEPT.
- **q4 ORDER Data→Training→Serving→Monitoring → ORDER per-prediction lifecycle** — REWRITTEN per `build_up_violation` audit. Original ordered four generic ML-workflow layers, an ordering any reader knows from chapter 3's lifecycle discussion. New ORDER frames a specific instance (the path of a single production prediction: event ingest → versioned features feeding training → serving loads the artifact → monitoring observes the served request) and ties the correctness of each step to the section's separation-of-concerns claim. The explanation names the boundary violations each misorder would create (e.g. monitoring before serving exists, deploying an artifact before training produces it) rather than restating the workflow.

### §3 Technical Debt (6 → 6 questions)

- **q1–q5** — KEPT.
- **q6 TF on code quality alone → TF on hardened code failing to fix ML debt** — REWRITTEN per `easy_tf` audit. Original TF restated the section's topic sentence ("debt is a systems problem not just code") which a careful reader could reject from the paragraph heading alone. New TF names a specific high-effort code-hardening push (strict linting, 95% unit-test coverage, code review) and forces the reader to recognize that none of those practices touch data dependencies, feedback loops, or undeclared consumers — so the dangerous debt class survives unchanged. The misconception is now real because many engineering teams *do* believe test coverage fixes debt.

### §4 Development Infrastructure (6 → 6 questions)

- **q1–q3, q5–q6** — KEPT.
- **q4 FILL on 'idempotency' → SHORT on diagnosing non-idempotent training** — REPLACED per `trivia_fill` audit. Original blank was a software-property name recalled from a single sentence — guessable without reading the section. New SHORT gives a concrete signature (two identical-input reruns producing meaningfully different validation accuracies) and requires the reader to (a) diagnose the root causes named in the section (unpinned seeds, nondeterministic GPU kernels, unversioned data snapshots, FP nondeterminism) and (b) prescribe the specific engineering controls. The term "idempotent" appears in the answer, and the question now tests whether the reader can *apply* it to a real pipeline failure.

### §5 Production Operations (6 → 6 questions)

- **q2–q6** — KEPT.
- **q1 MCQ distractor set for deployment patterns** — REWRITTEN per `throwaway_distractor` audit. Original included "Scheduled retraining" as a distractor, which is not a deployment pattern and therefore trivially eliminable by any informed reader. Replaced with "Canary deployment" (routes 1% live traffic to the candidate — exposes some real users, which the question asks to avoid) and "Immediate full rollout" (exposes everyone). Now every distractor is a real competing deployment strategy with a specific reason it fails the no-user-exposure constraint, and the correct answer's explanation refutes each by the exposure pattern it creates.

### §6 Design and Maturity Framework (5 → 5 questions)

- **q1–q3, q5** — KEPT.
- **q4 TF on 'maturity is about tools purchased' → TF on owning every tool ≠ scalable level** — REWRITTEN per `easy_tf` audit. Original framing ("maturity is mainly about purchased tools") is obviously false from the wording. New TF reverses the trap: it presents a plausible-sounding claim (a team that owns every canonical piece of MLOps infrastructure — feature store, registry, CI/CD, monitoring — has reached the scalable level) and forces the reader to distinguish *ownership* from *integration into a closed control loop*. The justification names the integrated practices (drift-triggered retraining, canary validation, per-segment tracking) that actually define scalable maturity.

### §7 Case Studies (5 → 5 questions)

- **q1–q3, q5** — KEPT.
- **q4 FILL on 'PSG' → SHORT on how PSG shapes operational decisions** — REPLACED per `trivia_fill` audit. Original blank was pure acronym recall. New SHORT uses the chapter's specific numbers (62% → 79% correlation, 82–83% inter-rater ceiling) to force the reader to reason about PSG's operational role: validation target, retraining trigger, and *stopping rule* for further investment. The operational framing (high-fidelity ground truth as a control knob, not just a metric) is the systems-reasoning move the original FILL skipped.

### §8 Fallacies and Pitfalls (3 → 3 questions)

- **All three questions** — KEPT. Audit flagged zero issues; the minimal-tier quiz is already sharp.

### §9 Summary (3 → 3 questions)

- **q1, q3** — KEPT.
- **q2 FILL on 'T\*' → MCQ on how T\* responds to traffic and decay-rate changes** — REPLACED per `trivia_fill` audit. Original was pure symbol recall ("the retraining interval is called ____"). New MCQ supplies the full formula $T^* \approx \sqrt{2C/(QVA_0\lambda)}$ and a specific perturbation (4× spike in query volume, 4× jump in decay rate) so the reader must apply the inverse-square-root scaling. Distractor refutations name each misconception by content: "stays the same because effects cancel" (compounding vs cancellation), "grows because traffic raises cost" (inverts the economics).

## Aggregate counts

| Action | Count |
|--------|-------|
| Rewritten in place (same question type) | 3 |
| Type-converted and replaced | 4 |
| Kept verbatim | 37 |
| Deleted (net) | 0 |
| Added (net) | 0 |
| **Total questions after improvement** | **44** (same as before) |

## §10 anti-shuffle check

Every MCQ in every section refutes distractors by their *content* (feature pattern, exposure regime, debt category, formula direction), never by letter. No "Option A", "Choice B", or "(A)"-style references appear in any answer string.

## Letter distribution across MCQs (unchanged from source except for natural rebalancing from rewrites)

The rewrites did not target letter redistribution as a primary goal, since the baseline was already varied. No MCQ's correct answer moved; all four choice positions remain reachable across the chapter.

## Prior-vocab discipline

Terms like *training-serving skew*, *distribution shift*, *silent degradation*, *drift*, *CI/CD*, *feature store*, *canary*, *blue-green*, *quantization*, *PSG*, *OTA*, *polysomnography* appear freely in stems and answers without redefinition — the 426-term prior-vocab list from chapters 1–13 allows this. No question's point is to *define* one of these terms; every use is applicative.

## Validator status

```
$ python3 book/tools/scripts/genai/quiz_refresh/validate_quiz_json.py \
    book/tools/scripts/genai/quiz_refresh/_audit/opus_improved/vol1_ml_ops_quizzes.json \
    book/quarto/contents/vol1/ml_ops/ml_ops.qmd
OK: vol1_ml_ops_quizzes.json passes schema + anchor validation
```

No errors, no warnings.
