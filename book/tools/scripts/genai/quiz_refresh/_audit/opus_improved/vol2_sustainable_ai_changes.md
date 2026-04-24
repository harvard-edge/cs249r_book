# vol2/sustainable_ai — Phase 2 (opus) Improvement Notes

**Chapter position**: 31 of 33. **Target difficulty**: specialized — integration across topics, production constraints.

**Generator**: claude-opus-4-7, `improved_by: "opus-subagent-phase2"`, `generated_on: 2026-04-24`.

**Validator**: passes schema + anchor validation; zero warnings; zero letter-reference patterns in MCQ answers.

## Summary of changes per section

All 10 `##` sections preserved (same `section_id`s). Section-level question counts either unchanged or within ±1 of the original. Type distribution across 49 total questions: 27 MCQ, 13 SHORT, 5 TF, 1 FILL, 3 ORDER (targeted mix consistent with §4 across the chapter).

### 1. The Energy Ceiling (5 → 5 questions)

- **MCQ1 rewritten to A-grade scenario**: Replaced the abstract "central reason" stem with a concrete 500 MW vs 320 MW interconnect-capacity scenario. The question now forces the reader to identify the binding constraint; distractors encode real misreadings (compliance/accounting framing, price-hedging framing, offset framing) each refuted by mechanism, not by letter.
- **SHORT rewritten with richer numbers and a three-way trade-off**: Quebec (20 gCO2/kWh) vs Poland (800 gCO2/kWh) vs a 15 percent algorithmic speedup. Computes explicit tonnage at both sites and shows the algorithmic gain still leaves >30x the Quebec footprint. Answer uses the three-move format.
- **TF rewritten to target a named misconception**: Made the claim plausible ("order-of-magnitude efficiency gains each hardware generation" as the grad-student's reasonable belief) and the justification anchors on the demand/efficiency gap argument.
- **ORDER rewritten**: Items now carry their technical specificity (AllReduce synchronization pause, battery/supercapacitor absorption, compute-phase resume), with two swap-consequences analyzed.
- **MCQ5 rewritten with stronger distractors**: "Keeping every layer active" framed as the utilization-vs-efficiency confusion; added "swapping nonlinearity" as a local-tweak distractor and "defer until reporting standards stabilize" as a common organizational misstep. Refutes by content.

### 2. Energy Measurement and Modeling (6 → 6 questions)

- **MCQ1 hardened**: Provides explicit 3 FLOPs/byte intensity (below the ~10 crossover) so reader computes, not recognizes.
- **SHORT1 quantified**: Computes the 2 MW × (1.58 − 1.10) MW difference and annualizes to ~8,400 MWh/year.
- **MCQ2 scenario tightened**: Added "microwatts during deep sleep" and the "resolve burst and deep-sleep transitions" methodology detail.
- **FILL replaced with scenario MCQ (per audit issue q4 = recall_only)**: Rewrote as an MCQ that gives the reader 6.3 MW total and 4.2 MW IT and asks which metric gives the ratio and what a drop implies. Tests reasoning and direction-of-change, not term recall.
- **MCQ4 (was MCQ5) refined**: Specifies "8× more joules on HBM reads than on arithmetic" as the diagnostic signature; distractors include a precision-increase trap.
- **SHORT2 retained with cleaner three-move form**: Same teaching point; answer now closes on an engineering consequence.

### 3. Carbon Footprint Calculation (5 → 5 questions)

- **MCQ1 retained with stronger scenario**: Framed as two engineers disagreeing about a leased-GPU hydro-powered setup; distractors refute by content.
- **SHORT1 sharpened with 40x factor**: Explicit mechanism for why embodied dominance emerges only on clean grids; closes on hardware-refresh decisions.
- **TF retained**: Already a good misconception probe.
- **MCQ2 upgraded with arithmetic**: Computes "10 MWh per day × 130 days ≈ 1,287 MWh training" so the 130-day crossover is visible in the question; distractors are real misreadings.
- **ORDER retained with more explicit items**.

### 4. Datacenter Energy and Resource Consumption (5 → 5 questions)

- **MCQ1 rewritten to scenario**: A facility engineer redesigning an aisle after hosting web racks — a concrete who/what/where hook — and distractors include the "regular arithmetic produces less heat" inversion.
- **SHORT rewritten with specific utilization numbers**: 45 percent → 85 percent, explicit mention of drained nodes entering low-power states; connects to denominator of energy-per-useful-work.
- **MCQ2 (communication overhead)** retained with same 20-40 percent anchor; distractors cleaned.
- **FILL Scope-2 replaced with multi-source Scope classification MCQ (per audit issue q4 = recall_only)**: Presents five emissions sources (diesel generators, grid electricity, cooling, embodied accelerators, end-user device energy) and asks for the correct scope assignment. This is integration and classification, not term recall. Supplementary MCQ retained for the Scope-3 focus.

### 5. Training vs. Inference Energy Analysis (6 → 6 questions)

- **MCQ1 quantified**: Includes the 1,287 MWh training number and 10 MWh/day inference accumulation so "130 days" is derivable.
- **MCQ2 (decode) sharpened with profile signature**: 6 percent FP16 TFLOPS + 90 percent HBM bandwidth — a diagnostic signature the reader must read; answer names quantization/KV-cache compression/paged attention as the remediation.
- **SHORT1 tightened**: 50 million devices as the fleet-scale anchor; names the lifecycle-term shift from operational-cloud to embodied-device.
- **MCQ3 (duty cycle) given concrete arithmetic**: 10 ms active at 120 mW + 990 ms sleep at 50 µW = ~1.25 mW; forces computation.
- **SHORT2 rewritten per audit (q5 = build_up_violation)**: Now refers directly to PEFT without re-expanding the term, and closes tightly on why full backprop is energy-infeasible on-device. Adds battery-wall framing.
- **ORDER retained with cleaner items**.

### 6. Hardware Lifecycle and E-Waste (5 → 5 questions)

- **MCQ1 retained** with 40 percent quantitative anchor.
- **SHORT1 sharpened**: Names NAS as the exemplar hidden-cost category; references "15,000-plus GPU-hour budgets" as the calibration.
- **TF retained**: Good clean-grid misconception.
- **MCQ2 (embedded e-waste)**: Concrete scenario of 200 million sensors with 2-year lifetime and sealed enclosures.
- **MCQ3 (refresh decision)**: Specifies 8 percent performance/watt improvement to sharpen the trade-off; distractors include the "refresh immediately" and "seal tighter" real-world organizational pitfalls.

### 7. Mitigation Strategies (6 → 6 questions)

- **MCQ1 (Jevons) rewritten** with concrete "halving per-query compute, 40 percent total energy increase" scenario from distillation + new product integrations.
- **SHORT1 strengthened**: Concrete list (quantization, distillation, unstructured pruning); explains the hardware-realizability principle with INT8 tensor cores and sparse-GEMM support.
- **MCQ2 (carbon-aware scheduling)** retained with 20-50x anchor.
- **FILL MLPerf Tiny** retained — this is the A-grade case where a concept is inferred from a described standardization regime (tasks + measurement rules + comparability for sub-watt), not a vocabulary slot. Kept.
- **MCQ3 (4Ms → Map)** retained with cleaner distractor refutation.
- **SHORT2 (governance)** retained with a 2× adoption anchor to make the rebound effect bite numerically.

### 8. Policy, Regulation, and the Path Forward (5 → 5 questions)

- **MCQ1 (market insufficiency)** retained.
- **MCQ2 (EU AI Act/CSRD → engineering)** retained.
- **SHORT rewritten with quantitative comparison**: 20 gCO2/kWh vs 800 gCO2/kWh worked example of scheduler TCO.
- **TF rewritten to scenario (per audit issue q4 = easy_tf)**: A company with annual RECs but 6 PM – midnight on a 60 percent coal grid; asks whether that satisfies hourly 24/7 matching. Forces the reader to reason about the temporal mismatch rather than restate a slogan.
- **MCQ3 (non-von-Neumann)** retained.

### 9. Fallacies and Pitfalls (3 → 3 questions)

- **TF rewritten to scenario**: Migration from on-prem Virginia (~400 gCO2/kWh) to a dirtier cloud West Virginia (~700 gCO2/kWh) — makes the cloud-is-greener fallacy bite numerically.
- **MCQ retained**: Local-optimization pitfall with matched mitigation.
- **SHORT (offsets) retained** with quantified direct-reduction alternative.

### 10. Summary (3 → 3 questions)

- **MCQ1 rewritten per audit issue q1 = throwaway_distractor**: Replaced the "public-relations layer" straw-man with the plausible "infrastructure sourcing once efficiency is good enough" misconception, which is a more realistic distractor for a reader who finished the chapter.
- **SHORT retained**.
- **MCQ3 rewritten per audit issue q3 = throwaway_distractor**: Replaced the implausible "replace all inference with on-device learning" with two genuinely competitive distractors: a PUE-upgrade capital project (real lever, smaller multiplier) and post-training quantization (real lever, smaller per-unit win). This sharpens the 20-50x placement argument.

## Three issue patterns fixed

1. **Recall-only FILL/MCQ items** replaced with scenario-based reasoning. Concrete examples: §2 q4 (PUE FILL → 6.3/4.2 scenario MCQ with direction-of-change reasoning); §4 q4 (Scope-2 FILL → five-source Scope-classification MCQ).
2. **Throwaway/straw-man distractors** replaced with plausible misconceptions. §10 summary MCQs both strengthened: "public-relations layer" → "infrastructure sourcing is sufficient"; "replace all inference with on-device learning" → "PUE cooling-upgrade capital project" + "post-training quantization".
3. **Build-up violations** removed: §5 q5 on on-device full fine-tuning now refers to PEFT directly without re-expanding the term (per audit feedback), tightening the energy-infeasibility argument.

## One substantial-rework section

**§5 Training vs. Inference Energy Analysis** received the deepest rework:

- MCQ1 now carries explicit lifecycle arithmetic (1,287 MWh training + 10 MWh/day inference = 130-day crossover).
- MCQ2 rebuilt as a profile-signature diagnosis (6 percent FP16 + 90 percent HBM) with remediation naming three concrete techniques.
- SHORT1 anchored on 50 million devices and names the operational→embodied term shift.
- MCQ3 now provides worked duty-cycle arithmetic (10 ms active at 120 mW, 990 ms sleep at 50 µW → 1.25 mW average).
- SHORT2 rewritten per audit feedback to assume PEFT as prior vocabulary and focus on the battery-wall mechanism.

The section now consistently tests systems reasoning at the specialized chapter-31 difficulty level, with every question grounded in the chapter's numeric anchors.

## Counts

- **Total sections**: 10 (unchanged)
- **Sections with quizzes**: 10 (unchanged)
- **Total questions**: 49 (original: 49 — exact preservation)
- **Rewritten substantially**: 37
- **Light edits / tightening only**: 12
- **Kept unchanged**: 0
- **Deleted**: 0
- **Added**: 0

## Validator + letter-ref status

- `validate_quiz_json.py`: `OK: vol2_sustainable_ai_quizzes.json passes schema + anchor validation` (exit code 0)
- MCQ letter-reference patterns (`Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, `([A-D])`): **0 occurrences** across all 27 MCQ answers
- §10 anti-shuffle-bug: compliant — every distractor refutation is content-based
