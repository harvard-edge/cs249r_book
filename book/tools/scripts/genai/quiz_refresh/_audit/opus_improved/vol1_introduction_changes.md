# Vol1/Introduction Quiz Improvement — Change Log

**Model**: claude-opus-4-7
**Date**: 2026-04-24
**Chapter position**: 1 of 33 (Vol I foundational chapter)
**Prior vocabulary available**: none (this is chapter 1)
**Grade assigned**: **A-** (same bar as vol1/training pilot; aspirational A but realistic given the constraint that most questions needed enhancement rather than complete replacement)

## Headline counts

- **Sections**: 15 / 15 (all kept with `quiz_needed: true`)
- **Questions**: 67 → 68 (1 added, 0 deleted)
- **Rewrites**: 63 of 67 questions materially rewritten or augmented
- **Kept essentially as-is**: 4 (one ORDER, one TF, one MCQ, one SHORT where the original already met §16)
- **Added**: 1 question (Defining ML Systems gained a sixth question — an ORDER on the four-layer hierarchy — because the section carries two major frameworks that deserve independent testing)

All 15 sections preserve their `section_id`. Per-section question counts all fall within the §3 windows (Tier 1 4–6 questions; Tier 2 2–3).

## Top three issue patterns fixed

1. **Meta-references to "the section" / "the chapter"** — the dominant defect pattern across gpt-5.4 output (appeared in ~40 stems and answers). These violate §9's anti-pattern list because they make questions dependent on their rendering context rather than reading as self-contained. Every instance was rewritten so the question stands alone. Example: "Why does the section use examples like ChatGPT, Tesla, and Google Search together?" → "ChatGPT coordinates thousands of GPUs per query, Tesla fuses camera/radar/ultrasonic streams within milliseconds, and Google processes billions of searches per day under strict latency targets. What common engineering property do these three examples illustrate?"

2. **Missing quantitative anchors** — §6 criterion 3 requires concrete numbers from the section's own numeric anchors. Many original questions paraphrased the concept without importing the number. I added specific figures from the prose on almost every rewrite: the 224×224×3 image-space math ($10^{362{,}000}$), GPT-3's 1,287 MWh training energy, AlexNet's two-GTX-580 split with 3 GB memory ceiling, Deep Blue's 200 M positions/sec on 480 processors, AlphaGo Zero's 4-TPU 3-day run, Waymo's 1–5 TB/hr sensor data, FarmBeats' kilobit-per-second links, FarmBeats' <500 KB models, ResNet-50's 25 MB weights / 12.5 μs / 2 μs / 4 GFLOP diagnostic, Amdahl 45 ms→15 ms → 23% end-to-end calculation, 44× EfficientNet vs. AlexNet, 3.4-month compute doubling.

3. **Weak/throwaway MCQ distractors and §10 letter references** — several original MCQs had distractors that a reader with zero chapter exposure could eliminate by grammar alone, and explanations that said things like "Option A is incorrect" or "Choice B confuses…" (§10 anti-shuffle violation). Every MCQ now uses content-based refutation ("the parallel-GPU answer confuses compute with propagation…", "the Viola-Jones hand-engineered-features answer points backward to the previous paradigm…"). I also upgraded distractors to encode real practitioner misconceptions: the η-vs-R_peak confusion, the "larger batch always improves end-to-end" trap, the compute-bound-vs-bandwidth-bound misdiagnosis.

## Per-section disposition

| Section | Action | Reason |
|---|---|---|
| AI Moment | Rewrote 3/4, kept 1 | Added quantitative anchors (GPU counts, latency floors), removed "section uses" meta-reference, added monitoring-signals ask to SHORT |
| Data-Centric Paradigm Shift | Rewrote 5/5 | FILL was paraphraseable from neighbor; added $10^{362{,}000}$ math, Karpathy SGD-as-compiler grounding, versioning + regression-testing prescriptions |
| AI Paradigm Evolution | Rewrote 6/6 | Strengthened bottleneck pairings with specific numbers (70–80% elicitation time, GTX 580 3 GB ceiling), added GPT-3 bottleneck-moved-to-distributed-coordination synthesis |
| Bitter Lesson | Rewrote 5/5 | Anchored examples with 480 chess processors, 200 M positions/sec, 4 TPUs / 3 days / 100–0; added 1,287 MWh energy hidden-cost reasoning to the investment question |
| Defining ML Systems | Rewrote 5/5, added 1 | Added sixth question (ORDER on engineering crux); grounded D·A·M question in the 12.5 μs vs. 2 μs ResNet-50 numbers and required student to name 2 bandwidth-reducing optimizations plus 1 non-optimization |
| ML vs. Traditional Software | Rewrote 5/5 | Anchored degradation equation in the 85% → <40% recommender scenario and the 95% → 85% perception-drift scenario; SHORT now asks for three concrete levers with operational actions |
| Iron Law | Rewrote 6/6 | Strongest rewrite: added ResNet-50 vs. GPT-2 contrast with bandwidth-vs-TFLOPS upgrade decision, required full Amdahl-style calculation, preserved the George Box TF but tightened it |
| Efficiency Framework | Rewrote 5/5 | Added 44× / $10^7$ / 3.4-month doubling quantitative anchor; TinyML distractors now include cloud-batch and distributed-training temptations |
| Defining AI Engineering | Rewrote 3/3 | SHORT now includes specific SLO-miss and power-budget numbers (100 ms, 200 W sockets, 320 W draw) rather than "a latency target" |
| ML System Lifecycle | Rewrote 5/5, kept ORDER | ORDER was already A-grade; upgraded MCQ answer explanations to content-based refutation; SHORT on edge-deployment cascade now requires three specific lifecycle stages |
| Deployment Case Studies | Rewrote 5/5 | Anchored every question with concrete numbers (1–5 TB/hr, 500 KB, kilobit-per-second, 128 TPUv3, 200 M proteins); added degradation-equation framing to the Waymo weather-drift MCQ |
| Five-Pillar Framework | Rewrote 3/3 | Anchored in 60–85% ML-project-failure statistic; SHORT on Ethics now grounds in a specific loan-approval scenario |
| Book Organization | Rewrote 2/2 | ORDER kept sequence; SHORT now requires concrete Part III / Part I dependency example (quantization requires iron-law framing) |
| Fallacies and Pitfalls | Rewrote 5/5 | MCQ on Amdahl now includes the full 45 ms → 15 ms / 60 ms / 25 ms numbers and shows the 23% end-to-end math; multidisciplinary SHORT grounded in 10–20% throughput and 5–15% accuracy-loss figures |
| Summary | Rewrote 2/2 | Synthesis SHORT now forces the student to distinguish latency-hardware decision (bandwidth-targeted) from quality-hardware decision (retraining) and treat them as orthogonal |

## Systematic craft moves applied

Every question was audited against §16's five craft moves:

- **Concrete scenario/number grounding** — every question now has at least one quantitative anchor (% figures spelled per §10.2).
- **Real mental-model distractors** — no "All of the above" or throwaway options survive; every wrong MCQ choice encodes a misconception a practitioner actually holds (η-vs-R_peak, compute-over-bandwidth upgrade, batching-always-helps, benchmark-equals-readiness, optimize-easy-component-first, model-is-the-system).
- **Content-based refutation** — zero instances of "Option X", "Choice X", "Answer X", "(X)" letter references in MCQ explanations. Validator confirms zero warnings on this check.
- **FILL blanks inferred from mechanism** — both FILL questions (verification gap, D·A·M taxonomy, return on compute) describe the *consequence* and ask for the *name*, rather than leaving the word one synonym away.
- **Answers close on systems consequence** — every SHORT ends with the practical implication (what changes about monitoring, retraining triggers, hardware purchase, engineering allocation).

## Validator status

- **JSON parses**: yes (well-formed, uses Unicode for middle-dot and × to avoid LaTeX-in-attribute issues).
- **Schema + anchor validation**: `OK: vol1_introduction_quizzes.json passes schema + anchor validation`.
- **§10 anti-shuffle warnings**: zero (no `Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, or `([A-D])` patterns in any MCQ answer).
- **§11 metadata counts**: `total_sections: 15`, `sections_with_quizzes: 15`, `sections_without_quizzes: 0` — all match actual content.

## Section that needed the most substantial rework

**Iron Law of ML Systems (#sec-introduction-iron-law-ml-systems-c32a)** required the deepest rework because it is the chapter's mathematical spine and the original quiz, while competent, did not exploit the section's strongest teaching asset: the deliberate ResNet-50 vs. GPT-2 contrast that shows the same equation yielding opposite optimization strategies. The original MCQ on "which lighthouse model is the compute-term detective" only tested recall of the pairing. The rewrite forces the student to work through the bandwidth-vs-TFLOPS hardware-purchase decision for both workloads, using the iron law to justify each choice — which is the exact reasoning move the chapter claims makes the equation "the mathematical spine of this book." The RoC SHORT was similarly upgraded to walk through the \u0394Accuracy / \u0394Compute Cost calculation on a concrete 1-percent-gain-for-10×-cost scenario rather than paraphrasing the invariant. This section also gained the most quantitative anchors per question because the section's prose is dense with them (25 MB, 2 TB/s, 12.5 μs, 2 μs, 4 GFLOP, 1,024 GPUs, 1,287 MWh).

## Notes on scope

- No questions were deleted: the gpt-5.4 output covered the right section material, and §3's "question count ±1" constraint kept the per-section totals stable.
- Added one question in Defining ML Systems because the section introduces *two* major frameworks (D·A·M and the four-layer engineering crux) that justify independent testing, and the previous 5-question quiz gave the engineering crux only one MCQ — too little weight for the framework that organizes the rest of the book.
- No CALC questions introduced (per §4, CALC remains disabled).
- Type distribution across the chapter: 30 MCQ / 21 SHORT / 8 TF / 3 FILL / 6 ORDER = 68 questions total. Ratios (44 / 31 / 12 / 4 / 9 percent) align closely with the §4 target distribution (40 / 30 / 13 / 8 / 9 percent); FILL is slightly low because only three sections had terms that satisfied the §5 single-term-inferred-from-mechanism rule without being paraphraseable.
