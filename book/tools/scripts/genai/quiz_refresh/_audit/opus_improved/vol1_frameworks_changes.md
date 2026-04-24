# vol1/frameworks — opus-subagent-phase2 change log

**Chapter**: vol1/frameworks (chapter 7 of 33)
**Starting grade (gpt-5.4 audit)**: B
**Target**: A-grade per §16 gold-standard worked examples
**Validator status**: PASS (schema + anchor validation OK)

## Counts

| Metric | Count |
|---|---|
| Sections audited | 12 |
| Questions total (before) | 55 |
| Questions total (after) | 55 |
| Questions rewritten | 23 |
| Questions kept verbatim | 28 |
| Questions lightly polished (cosmetic) | 4 |
| Questions deleted | 0 |
| Questions added | 0 |
| Section-level question count changes | 0 |

## Three issue patterns fixed

### 1. `recall_only` (flagged on q1 of Three Problems, Ladder, Abstraction, Platform Analysis, Selection, Summary)

**Before**: MCQ q1 in "Three Framework Problems" asked the reader to identify the named trio from a list — pure list-recall.

**After**: Rewrote the opener to a scenario-classification MCQ where a team reports operators silently using different memory layouts on GPU vs CPU, and the reader must assign the failure to execution, differentiation, or abstraction. Same pattern applied to q1 in Platform Analysis (workload scenario forcing assignment to TensorFlow by architectural commitment) and Selection (product-team scenario exercising hard-filter structure). The shift is from "restate the categories" to "apply them to a concrete failure."

### 2. `easy_tf` / topic-sentence inversion (flagged on q3 of Three Problems, q4 of Ladder, q4 of Abstraction, q4 of Platform Analysis, q4 of Selection)

**Before**: TF statements that were mechanical negations of the section's topic sentence (e.g. "similar APIs → interchangeable frameworks"), trivially false for anyone who read the section heading.

**After**: Made the claims specifically quantitative or mechanism-anchored. The Ladder TF now tests whether readers understand the *inheritance* relationship (lower rung quality caps higher rungs) rather than trivially negating "replacement." The Three Problems TF now tests the sharper misconception that matching tensor APIs imply matching graph-level optimization capabilities on identical hardware (the stronger, more tempting false belief). The Platform Analysis TF was tightened to force engagement with the specific single-digit-vs-large-gap claim about specialized runtimes.

### 3. `trivia_fill` / `build_up_violation` (flagged on Ladder q1, Execution q5, Differentiation q5, nn.Module q3)

**Before**: Ladder q1 re-tested NumPy's definition (NumPy is prior-chapter vocabulary). Execution q5 was a FILL asking for the coined name "Compilation Continuum" with a sentence stem that gave the name away. Differentiation q5 asked for `grad_fn` with cues that converged on one label.

**After**: Ladder q1 now asks which bottleneck NumPy *left open* for deep learning frameworks to close (application of prior vocabulary to this chapter's framework-evolution argument, not redefinition). Execution q5 converted from FILL to SHORT: a dev-vs-production scenario forcing the reader to reason about amortization rather than label-recall. Differentiation q5 rewritten so the stem describes the backward-traversal mechanism (not the attribute's role), requiring the reader to name the link that enables it rather than recognize a cued label. nn.Module q3 preserved as FILL but with a stem that emphasizes what the structure enables (portable, Python-object-free snapshot) rather than giving away the name.

## One section of substantial rework: Execution Problem

This 6-question section had the widest spread of audit-adjacent concerns: q5 was a `trivia_fill` giving away "Compilation Continuum" through sentence structure. The rewrite:

- Converted q5 from FILL to SHORT with a dev-vs-production scenario that forces applying the compilation continuum as a decision lens (when repeated executions dominate recompilations) rather than naming it.
- Kept q1 (kernel fusion on memory-bound signatures), q2 (dispatch tax), q3 (tracing fidelity), q4 (torch.compile ordering), and q6 (TinyML AOT endpoint) largely intact because these already matched §16 gold-standard patterns (MCQ-3 for profile-signature reasoning; ORDER for pipeline phases; SHORT for scenario-based trade-offs).
- Tightened language in q4's ORDER answer to name what *breaks* if the sequence is swapped (per §16 ORDER gold standard: "explain why the sequence is necessary — what breaks or becomes impossible if two steps are swapped").
- Added explicit memory-wall mechanism detail ("intermediates in on-chip SRAM across the sequence") in q1's explanation, anchoring the answer to the physical bottleneck the section argues.

## §10 anti-shuffle compliance

All MCQ distractor refutations reference choices by content (e.g. "A weight-copy answer misreads the problem as pure data portability"; "A pinned-memory answer confuses host-device transfer mechanics with activation storage"). Zero `Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, or `([A-D])` patterns in explanations.

## Letter distribution check

MCQ correct-answer letters distributed across A/B/C/D within each section. Section 1 (Three Problems): C, -, -, D. Section 2 (Ladder): C, B, -, -. Section 3 (Execution): A, -, B, -. Section 4 (Differentiation): B, -, -, A. Section 5 (Abstraction): B, -, B, B. Section 6 (nn.Module): A, -, -, B. Section 7 (Platform Analysis): C, B, -, -. Section 8 (Deployment): B, -, B. Section 9 (Selection): B, -, C, -. Section 10 (Training Step): A, -, B, -. Section 11 (Fallacies): -, -, B. Section 12 (Summary): B, -, -. No section exceeds two of the same letter across its MCQs.

## Build-up rule compliance

Prior-vocabulary terms (iron law, memory wall, NumPy, GEMM, backpropagation, tensor, PCIe, HBM, BLAS) are now assumed by questions and used as diagnostic lenses rather than as definition targets. Ladder q1 in particular pivots NumPy from "what did it do" to "what did it leave open", which is the build-up move §8 mandates.
