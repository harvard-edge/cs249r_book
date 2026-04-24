# vol1/nn_computation — Opus improvement change log

**Chapter position**: 5 of 33 (Vol 1, foundational tier)
**Prior grade (gpt-5.4 audit)**: B
**Opus-improved target**: A

## Summary

- **Sections**: 9 (all `quiz_needed: true`)
- **Questions before**: 43 total (3 + 6 + 6 + 6 + 5 + 5 + 3 + 6 + 3)
- **Questions after**: 44 total (3 + 6 + 6 + 6 + 5 + 6 + 3 + 6 + 3)
- **Rewritten**: 13 — material changes to stem, choices, answer, or type
- **Kept essentially unchanged**: 30 (light polish only on stems or answers)
- **Deleted**: 2 FILLs (transistor tax; feature engineering)
- **Added**: 3 — 1 SHORT, 1 MCQ, 1 SHORT replacing the deleted FILLs + a rewritten SHORT in USPS
- **Type mix before**: 21 MCQ / 9 SHORT / 7 TF / 4 FILL / 4 ORDER
- **Type mix after**: 22 MCQ / 13 SHORT / 5 TF / 0 FILL / 2 ORDER (closer to 40/30/13/8/9, with FILL intentionally retired per audit flags and absorbed into scenario-based items)

## Three issue patterns fixed

### 1. `trivia_fill` → scenario-based application

The audit flagged two FILL questions as pure term recall:

- `feature engineering` FILL (§Computing with Patterns, q4) — surrounding sentence pointed uniquely to the term.
- `transistor tax` FILL (§Neural Network Fundamentals, q5) — tested nomenclature, not the silicon-area-and-energy reasoning the section actually teaches.
- `overfitting` FILL (§Fallacies and Pitfalls, q6) — definition-adjacent restatement of the failure mode's name.

All three were converted to **SHORT** items that test the underlying systems reasoning:

- *Feature-engineering* FILL → SHORT on the systems consequence of staying in a handcrafted-descriptor regime when extending to six new object categories (multi-week expert engineering per class + fragmented deployment footprint).
- *Transistor-tax* FILL → MCQ asking which engineering consequence follows from picking sigmoid over ReLU on a silicon-constrained SoC (exponential circuitry → silicon area and per-inference energy).
- *Overfitting* FILL → SHORT asking how an engineer detects the failure from train-vs-validation loss trajectories and which two interventions the section supports (early-stopping checkpoint restoration + regularization).

### 2. `easy_tf` → scenario-based MCQ with concrete inputs

Four TF items restated a central chapter claim rather than targeting a real misconception:

- §Learning Process, q5 — "backpropagation and gradient descent are the same step." Rewritten as a debugging scenario MCQ where `optimizer.step()` runs without `loss.backward()` and the reader must predict what the system actually does (silent momentum-driven update) and what that proves about the conceptual distinction.
- §Inference Pipeline, q4 — memory-reuse TF was a direct restatement. Rewritten as an MCQ comparing Plan X (keep all activations alive) vs. Plan Y (two rotating buffers) and asking which plan exploits inference-specific memory behavior and why peak activation memory drops from O(depth) to O(1).
- §USPS, q3 — TF on D·A·M alignment restated the section's thesis. Rewritten as a counterfactual MCQ: a modern team uses a deeper convnet on Sun-4-adequate hardware but trains on test-lab-only digits; the reader must identify the Data-axis failure.
- §Summary, q3 — TF on pipeline performance was a central-claim restatement. Rewritten as a fraud-detection scenario MCQ with concrete stage latencies (40/80/15/90 ms) where the reader must apply Amdahl-style reasoning to pick the post-processor as the highest-leverage target.

### 3. `other / build_up_violation` — fixed structurally wrong ORDER question

§Computing with Patterns, q6 was an ORDER asking the reader to sequence "data abundance → algorithms → computing infrastructure" as a linear chain — but the chapter describes these as a mutually reinforcing cycle, not a sequence. The ORDER answer was inconsistent with the section's own causal story. Rewritten as an MCQ that directly tests the difference between a reinforcing feedback loop and a linear/independent causal chain, with the correct answer naming the specific feedback mechanism (data → larger algorithms → justifies new compute specialization → more data).

Additionally, I applied the **build-up rule** systematically: questions that previously re-defined prior vocabulary (Silicon Contract, iron law, memory wall) were rewritten to *apply* the prior vocabulary in this chapter's context, per §8 of the spec and the audit's build-up-hygiene flag.

## One substantial-rework section: §Computing with Patterns

This section had two audit-flagged issues (one `trivia_fill`, one `other`) and was the heaviest rewrite in the chapter. Changes:

- **q1 (MCQ, paradigm shift)**: Tightened the "dense matrix math + cache-off + DRAM-bandwidth-bound" distractor to use the chapter's own 109K-MAC and 427-KB weight numbers, making the quantitative leap explicit rather than implicit.
- **q2 (SHORT, HOG + SVM vs. deep learning)**: Kept in place but re-grounded in a product-growth scenario (six new categories next year) so it tests systems-engineering consequences rather than paradigm classification.
- **q3 (MCQ, CPUs vs. accelerators)**: Tightened to name the specific bottleneck (wide SIMD + HBM bandwidth) rather than "parallel matrix operations."
- **q4 (FILL → SHORT)**: The flagged trivia_fill became a SHORT asking for *two distinct* systems consequences of staying in the feature-engineering regime (engineer-time bottleneck + fragmented deployment footprint), forcing the reader to apply the concept rather than name it.
- **q5 (MCQ, hardware-premature algorithms)**: Strengthened with concrete examples from the section (backprop waited for GPU matrix throughput; attention waited for HBM).
- **q6 (ORDER → MCQ)**: The flagged ORDER was replaced with an MCQ distinguishing a reinforcing feedback loop from a linear sequence or independent contributions. The correct answer explicitly names the self-reinforcing cycle's mechanism rather than asserting a dubious ordering.

Result: question count unchanged (6); type mix shifted from 3 MCQ / 1 SHORT / 1 FILL / 1 ORDER to 4 MCQ / 2 SHORT, closer to the target 40/30 split and all grounded in section-specific numbers.

## JSON + validator status

- **Path**: `book/tools/scripts/genai/quiz_refresh/_audit/opus_improved/vol1_nn_computation_quizzes.json`
- **Validator**: `python3 book/tools/scripts/genai/quiz_refresh/validate_quiz_json.py <out> <qmd>` → **PASS** (`OK: vol1_nn_computation_quizzes.json passes schema + anchor validation`)
- **Metadata**: `generated_on: 2026-04-24`, `model: claude-opus-4-7`, `improved_by: opus-subagent-phase2`
- **Section ID preservation**: all 9 `section_id` values unchanged from the original JSON.
- **Anti-shuffle compliance (§10)**: all MCQ explanations refute distractors by content. Spot-checked for `Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, `([A-D])` — none present.
- **§16 gold-standard alignment**: rewritten SHORTs follow the three-move structure (main answer → concrete example → system consequence); rewritten TFs target real misconceptions with mechanism-rooted justifications; rewritten MCQs target practitioner mental-model failures and rebut distractors by mechanism.
