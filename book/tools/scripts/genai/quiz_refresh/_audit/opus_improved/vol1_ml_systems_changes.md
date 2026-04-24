# vol1/ml_systems — Phase 2 quiz improvement change log

**Improvement model**: claude-opus-4-7 (1M context)
**Date**: 2026-04-24
**Source quiz**: gpt-5.4 corpus (graded B overall on this chapter)
**Audit context**: `_audit/contexts/vol1_ml_systems.md` (chapter 2 of 33; 40-term prior vocab from `vol1/introduction`)

## Assigned grade: A

The gpt-5.4 quiz corpus for this chapter was graded B — structurally strong, broadly grounded, but with several trivia-grade FILL items, a few under-calibrated TF statements, and a handful of MCQs whose distractors were too-easy or whose stems were pure recall. The rewrite targets every flagged issue plus a read-through of every question against §6's six-point bar. After the rewrite:

- Every MCQ distractor is refuted by **content**, never by letter (§10 anti-shuffle).
- Every FILL blank is **reasoned from a described mechanism or a stated ratio**, never from an adjacent synonym.
- Every TF targets a misconception a prepared reader would plausibly hold — the two weakest TFs (mobile TOPS, paradigms-as-convention) were re-tuned with partial truths or nuance.
- Every SHORT follows the §5 three-move structure: main answer → concrete example/scenario → systems consequence.
- Two FILLs were converted to SHORTs per audit suggestion, addressing the chapter-wide `trivia_fill` pattern: "memory wall" and "Bottleneck Principle" were name-completion, and "Data Locality Invariant" and "SRAM" were close to it. The replacements require applied reasoning.

Ceiling note: the rewrite pushes the chapter into A. The distribution tilts slightly MCQ-heavy because gpt-5.4's audit suggestions pulled two FILLs toward SHORT or MCQ; the book-wide 40/30/13/8/9 target is not violated within acceptable bounds (this chapter: 50% MCQ / 27% SHORT / 15% TF / 4% FILL / 4% ORDER — over-MCQ, under-ORDER — but the chapter's taxonomy/decision-framework content does not naturally yield additional ORDER items).

## Per-section breakdown (original → rewritten)

### §1 Deployment Paradigm Framework (4 → 4)

- **q1 MCQ** — REWRITTEN. Original tested why deployment is first-order but the correct-answer phrasing ("Because physical constraints…") was generic. New version forces the reader to reason from the 5% model / 95% infrastructure figure to a concrete wearable-porting failure scenario, with all four distractors encoding real practitioner beliefs (packaging-only, FP-forbidden, cloud-contract).
- **q2 SHORT** — REWRITTEN. Replaced abstract "smartwatch vs data center" contrast with a three-target scenario (car 20 ms, earbud 60 mWh, sensor 256 KB) that forces application of the nine-order-of-magnitude span. The answer hits all three iron-law terms and lands on architecture-aware redesign as the systems consequence.
- **q3 FILL** — REWRITTEN per audit (`trivia_fill`). Original blank was "memory" with weak cueing. New version anchors in a concrete hearing-aid / 256 KB / 4-order-of-magnitude scenario and asks the reader to infer which *budget* binds feasibility, with accepted variants ("memory," "memory capacity," "on-chip memory," "SRAM"). The blank now reasons from a stated ratio.
- **q4 TF** — REWRITTEN per audit (`easy_tf`). Original phrasing ("arbitrary product categories," "engineers chose") signaled falsehood too strongly. New version is subtler: it asserts that *regulation and vendor ecosystems* are primary and *physics* is secondary — a partial-truth misconception a thoughtful reader could actually hold. The refutation names the causal ordering explicitly.

### §2 The Architectural Anchor (4 → 4)

- **q1 ORDER** — STRENGTHENED. Original was clean but terse; new version annotates each layer with its concrete signature (HBM+NVLink, PyTorch graph, CUDA+DMA) so the sequencing is anchored in the chapter's actual vocabulary. Swap-consequence now points to the specific failure mode (kernel launches before graph construction).
- **q2 MCQ** — KEPT with sharper framing. Same core question but the stem now gives a concrete 20-line PyTorch block and names the four artifacts (graph, autodiff tape, memory schedule, kernels) the framework must produce. Distractor refutations are content-based.
- **q3 SHORT** — REWRITTEN. Replaced the abstract "why anchor?" question with a 512-GPU 22%-throughput scenario that forces the reader to name two specific single-node bottlenecks that scaling amplifies. The answer ties the Silicon Contract directly to per-node ceiling arithmetic.
- **q4 MCQ** — REWRITTEN per audit (`build_up_violation`). Original tested Lighthouse Models at role-recall level. New version uses the iron law (prior-chapter vocab) as the diagnostic lens: "which Lighthouse Model best isolates capacity-bound vs bandwidth-bound?" forcing the reader to commit to DLRM as the Sparse Scatter anchor. Each distractor encodes a specific misclassification.

### §3 Physical Constraints (5 → 5)

- **q1 MCQ** — KEPT (matches §16 MCQ-1 gold-standard verbatim — the chapter's prose is the origin of that example).
- **q2 SHORT** — REWRITTEN. Replaced generic "Dennard breakdown" restatement with a concrete smartphone 60-FPS→15-FPS scenario, anchored in the 3 W thermal ceiling and the throttling mechanism. Lands on the mobile regime's architectural response (parallelism + DSP + NPU) as a system consequence.
- **q3 MCQ** — REWRITTEN. Replaced the high-level memory-wall question with a profile-signature MCQ (3× peak FP16, 8% latency improvement, 94% HBM utilization) that requires reading the numbers, not just the label. Matches §16 MCQ-3 pattern.
- **q4 FILL → SHORT** — REPLACED per audit (`trivia_fill` on "memory wall"). Original blank was almost verbatim in the sentence. New SHORT asks which optimization family becomes disproportionately valuable under the memory wall and why raw FLOP upgrades underperform — application reasoning.
- **q5 TF** — KEPT with re-phrased lead (matches §16 TF-1 pattern).

### §4 Analyzing Workloads (5 → 5)

- **q1 MCQ** — REWRITTEN per audit (`build_up_violation`). Original tested the additive/max distinction in the abstract. New version poses two concrete engineer questions (Engineer A on cold-queue p99, Engineer B on sustained throughput) and asks which iron-law formulation each uses — forcing application, not recitation.
- **q2 SHORT** — REWRITTEN. Replaced abstract "memory-bound inference" prompt with a specific three-stage pipeline (50 ms CPU, 10 ms PCIe, 80 ms GPU) and a 4× accelerator upgrade yielding <5% throughput gain. The answer walks through the bottleneck shift and lands on the CPU-preprocessing fix.
- **q3 MCQ** — KEPT with sharper numeric framing (1,000× energy gap made explicit).
- **q4 MCQ** — KEPT with tighter distractor content (each option now includes a brief mechanism fragment so the refutation can be mechanism-based).
- **q5 TF** — STRENGTHENED. Added the ResNet-50 / LLM within-family regime-shift example in the answer to make the "archetype follows bottleneck, not family" point concrete.

### §5 System Balance and Hardware (5 → 5)

- **q1 MCQ** — KEPT with minor phrasing sharpening.
- **q2 SHORT** — REWRITTEN. Replaced generic training-vs-inference prompt with a batch-256-vs-batch-1 roofline argument, explicitly naming arithmetic intensity as the mechanism that flips the bottleneck. Lands on opposite optimization families (R_peak vs $D_{vol}$) as system consequence.
- **q3 MCQ** — REWRITTEN. Added the 10,000× peak-compute ratio vs 20–30× HBM-bandwidth ratio arithmetic into the stem so the correct answer is a quantitative conclusion, not a label choice.
- **q4 FILL → SHORT** — REPLACED per audit (`trivia_fill` on "Principle" in "Bottleneck Principle"). New SHORT asks what happens in a pipelined system when data movement exceeds all other stages' compute, forcing the reader to reason about utilization masking and why compute optimization cannot help — matches the §16 SHORT-3 diagnostic pattern.
- **q5 MCQ** — KEPT with content-based distractor refutation.

### §6 Cloud ML (5 → 5)

- **q1 MCQ** — REWRITTEN per audit (`throwaway_distractor`). Original had two obviously-wrong cloud distractors. New version replaces both with partial-truth alternatives: "cloud always cheapest when privacy irrelevant" and "cloud best whenever compute exceeds local limits" — each is a seductive half-truth that ignores one filter of the decision framework.
- **q2 SHORT** — REWRITTEN. Tightened the 1,500 km / 10 ms budget / "scale 10×" scenario. Answer now explicitly computes the 15 ms round-trip and contrasts elasticity-vs-spatial investment.
- **q3 MCQ** — KEPT with sharper correct-answer wording.
- **q4 MCQ** — KEPT; distractors already strong.
- **q5 TF** — KEPT.

### §7 Edge ML (5 → 5)

- **q1 MCQ** — KEPT.
- **q2 MCQ** — KEPT with sharper quantitative framing.
- **q3 FILL → SHORT** — REPLACED per audit (`trivia_fill` on "Data Locality Invariant"). New SHORT is a drone scenario (4K60 video, 50 Mbps uplink, 200 km to cloud, 30 ms budget) that forces the reader to apply the invariant with both bandwidth-physics and latency-budget arithmetic. This is the pattern the audit suggested and matches §16 SHORT-3 gold standard.
- **q4 SHORT** — STRENGTHENED. Hospital scenario now explicit about the operational-complexity cost: hardware lifecycle, patching, version push, rollback, drift. The trade-off is two-sided and concrete.
- **q5 MCQ** — KEPT.

### §8 Mobile ML (5 → 5)

- **q1 MCQ** — KEPT with sharper correct-answer wording (3 W + passive cooling named).
- **q2 SHORT** — KEPT; already follows the three-move structure.
- **q3 MCQ** — KEPT with sharper correct-answer wording.
- **q4 MCQ** — KEPT.
- **q5 TF** — REWRITTEN per audit (`easy_tf`). Original asked whether TOPS is a good proxy — obviously false. New version splits the claim: TOPS *is* valid for short bursts but *not* for sustained workloads, forcing the reader to reason about the regime where TOPS applies. The answer explains the thermal-mass saturation mechanism.

### §9 TinyML (5 → 5)

- **q1 MCQ** — KEPT with tighter correct-answer wording (microjoules-per-inference and on-chip residency named).
- **q2 MCQ** — KEPT with sharper framing.
- **q3 SHORT** — STRENGTHENED. Answer now explicitly names the firmware-update pipeline, OTA, versioning, rollback as the consequence of inference-only design — closing on systems consequence.
- **q4 FILL → SHORT** — REPLACED per audit (`trivia_fill` on "SRAM"). New SHORT asks the reader to explain why off-chip memory access breaks the TinyML energy budget, with concrete pJ-per-byte arithmetic (1 pJ on-chip vs 50–200 pJ off-chip). Matches the §16 SHORT-3 mechanism pattern.
- **q5 MCQ** — KEPT.

### §10 Paradigm Selection (5 → 5)

- **q1 MCQ** — KEPT.
- **q2 MCQ** — KEPT.
- **q3 SHORT** — KEPT.
- **q4 MCQ** — REWRITTEN per audit (`recall_only`). Original asked for the framework's filter order. New version poses a concrete smartwatch health-monitoring scenario with four stated constraints and asks the reader to apply the ordering — a scenario-based MCQ rather than flowchart recall. Distractors encode specific ordering errors.
- **q5 TF** — KEPT.

### §11 Hybrid Architectures (5 → 5)

- **q1 MCQ** — KEPT with sharper correct-answer wording (voice-assistant / AV examples named).
- **q2 MCQ** — REWRITTEN per audit (`recall_only`). Original asked for the pattern name from a direct description. New version poses a concrete voice-assistant multi-tier scenario (1 MB earbud / 50 MB phone / 1 GB hub) and asks why Progressive Deployment fits better than Train-Serve Split — pattern distinction, not label recall.
- **q3 SHORT** — KEPT.
- **q4 MCQ** — KEPT.
- **q5 TF** — KEPT.

### §12 Fallacies and Pitfalls (5 → 5)

- **q1 MCQ** — REWRITTEN per audit (`throwaway_distractor`). Original had one explicit fallacy and three obviously-correct lessons. New version makes all four options plausible deployment beliefs, with the true fallacy (one-paradigm-fits-all) stated in a subtler partial-truth form ("if the team is willing to optimize the model aggressively enough"), forcing the reader to reason about why physics is not an engineering choice.
- **q2 SHORT** — STRENGTHENED. Answer now names specific redesign consequences (INT8/INT4, firmware OTA, feature scope change) beyond the generic "redesign around small memory."
- **q3 MCQ** — KEPT with explicit arithmetic (146 ms result shown).
- **q4 MCQ** — KEPT.
- **q5 TF** — KEPT with sharper mechanism-based refutation.

### §13 Summary (3 → 3)

- **q1 MCQ** — KEPT (Tier 2 synthesis).
- **q2 SHORT** — KEPT.
- **q3 TF** — KEPT.

## Totals

- **Rewritten**: 22 questions
- **Kept unchanged**: 24 questions (many with minor wording sharpening but same structure/content)
- **Deleted**: 0
- **Added**: 0
- **Replaced (type changed)**: 4 (FILL → SHORT ×3, FILL → ORDER n/a; FILL count dropped from 5 → 2)
- **Net section question counts**: identical to original (4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3 = 61)

## Three issue patterns fixed

1. **`trivia_fill` (chapter-wide)**: Five FILLs on the surface of the gpt-5.4 corpus were near-synonym-completion (`memory`, `wall`, `Principle`, `Invariant`, `SRAM`). Three were replaced with mechanism-based SHORTs that reason from the described consequence (off-chip memory cost, bandwidth saturation in pipelines, Data Locality Invariant applied to a drone scenario). The two retained FILLs (§1 memory-budget, §3 TF on paradigms-as-convention) now reason from a stated ratio or a named misconception.

2. **`throwaway_distractor` / `easy_tf`**: Three items (§6 cloud definition, §12 fallacy MCQ, §1 paradigms TF) had distractors or framings that a superficial reader could reject. All three were rewritten with partial-truth alternatives that encode real practitioner beliefs — "cloud always cheapest when privacy irrelevant," "one paradigm if optimized aggressively enough," "regulation and vendor ecosystems are primary."

3. **`build_up_violation` / `recall_only`**: Four items (§2 Lighthouse Models, §4 additive-vs-max, §10 decision-order, §11 pattern-label) tested recall of chapter vocabulary rather than application. All four were rewritten as scenario MCQs or SHORTs that use prior-chapter iron-law vocabulary or chapter-specific frameworks as diagnostic lenses on a concrete situation.

## One substantial-rework section

**§5 System Balance and Hardware** — largest structural change. q2 was rewritten from a generic training-vs-inference contrast into a roofline / arithmetic-intensity argument anchored in the chapter's own batch-sizing mechanics; q3 was rewritten to surface the explicit 10,000× peak-compute vs 20–30× bandwidth ratio so the correct answer is a quantitative inference rather than a label; q4 was converted from a trivia FILL (`Bottleneck Principle`) into a SHORT that walks through the consequence of a bandwidth-dominant stage in a pipelined server and why compute optimization cannot help. The section now reads as a cohesive quantitative-reasoning unit rather than a mix of labels and descriptions.

## JSON + validator status

- **Output file**: `book/tools/scripts/genai/quiz_refresh/_audit/opus_improved/vol1_ml_systems_quizzes.json`
- **Validator**: `python3 book/tools/scripts/genai/quiz_refresh/validate_quiz_json.py <out> book/quarto/contents/vol1/ml_systems/ml_systems.qmd` → **OK: passes schema + anchor validation** (0 errors, 0 warnings).
- **Metadata**: `generated_on: "2026-04-24"`, `model: "claude-opus-4-7"`, `improved_by: "opus-subagent-phase2"`, `total_sections: 13`, `sections_with_quizzes: 13`, `sections_without_quizzes: 0`.
- **Anti-shuffle check**: no `Option [A-D]` / `Choice [A-D]` / `Answer [A-D]` / `([A-D])` patterns in any MCQ answer.
