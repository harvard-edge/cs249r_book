# vol1/benchmarking — Phase 2 quiz improvement change log

**Improvement model**: claude-opus-4-7 (1M context)
**Date**: 2026-04-24
**Source quiz**: gpt-5.4 improve-mode corpus (second-pass improved, previously graded B+)
**Chapter position**: 12 of 33 (late vol1, intermediate-advanced per spec §7)
**Audit context**: `_audit/contexts/vol1_benchmarking.md`

## Assigned grade: A

The original gpt-5.4 improve-mode output for this chapter was already a strong B+/A- — most questions were grounded in the prose, distractors were mostly plausible, and answers followed the three-move format. Weaknesses were concentrated in three areas: (1) several MCQs carried at least one throwaway distractor that no prepared reader would seriously consider; (2) quantitative questions lacked the specific numeric anchors the chapter actually supplies (the A100's 312 TFLOPS peak vs. 90-155 TFLOPS sustained, the 6.7x gaming gap in the power section, the 8 ms + 10 ms Amdahl example in inference benchmarks, the 24h/4h strong-scaling arithmetic); (3) correct-letter distribution was skewed heavily toward B (18/32 MCQs), which is a legibility tell even though every distractor refutation was content-based. After rewrite, every question passes the §6 quality bar with §16 gold-standard patterns: each MCQ forces a specific reasoning move rather than pattern-matching, and every SHORT follows Thesis → Evidence → Implication with quantitative anchors drawn from the chapter itself.

## Per-section breakdown

### §1 ML Benchmarking Framework (6 → 6)
- **q1 MCQ (MobileNet dimension assignment)** — REWRITTEN. Replaced the ImageNet-style phrasing with a concrete 12 percent low-light misclassification scenario, strengthened refutation to name the competing mechanism (latency vs. accuracy failure mode) by content.
- **q2 SHORT (2-10x structural gap)** — REWRITTEN. Grounded in the MobileNet EdgeTPU pipeline with the explicit "2 ms kernel vs. 12 ms end-to-end" contrast from the chapter prose.
- **q3 TF (3x inference = 3x end-to-end)** — REWRITTEN with chapter's exact 5-10 ms model / 50-100 ms total numbers for Amdahl-style refutation.
- **q4 MCQ (Goodhart's Law / BLEU)** — REWRITTEN. Strengthened correct-choice text with concrete BLEU=28.0→28.5 and 4x slowdown numbers; rotated correct letter from B to D for distribution.
- **q5 FILL (proxies)** — KEPT (already matches §16 FILL pattern).
- **q6 SHORT (three-dimensional validation)** — REWRITTEN. Strengthened to enumerate specific additional measurements for each dimension.

### §2 Historical Foundations (5 → 5)
- **q1 MCQ (SPEC CPU vs. Whetstone)** — REWRITTEN. Added explicit "1976 / compilers tuned specifically against FP tests" framing from the prose; strengthened distractor refutation to name the synthetic-loop vs. real-application distinction.
- **q2 SHORT (energy benchmarks)** — REWRITTEN with SPEC Power (2007) and Green500 (2007) specifics and the 20-50 percent utilization failure mode.
- **q3 MCQ (MLPerf domain-specific split)** — REWRITTEN. Rotated correct letter to D; strengthened refutation to cite "nine orders of magnitude" span directly.
- **q4 TF (throughput not retired)** — KEPT (clean misconception refutation).
- **q5 ORDER (benchmark evolution)** — KEPT (genuinely sequential per §16 ORDER standard).

### §3 System Benchmarking Suites (6 → 6)
- **q1 MCQ (300 TFLOPS peak vs. BERT 30 TFLOPS)** — REWRITTEN. Added the specific 10 percent of peak figure from the prose and tied to roofline ridge point explicitly.
- **q2 SHORT (5-10 runs)** — REWRITTEN. Grounded with the Henderson et al. RL empirical finding cited in the chapter.
- **q3 MCQ (vendor 10,000 images/sec)** — REWRITTEN. Rotated correct letter from C to B for distribution; refutation strengthened to cite order-of-magnitude workload-configuration variance.
- **q4 FILL (fallacy of peak performance)** — REWRITTEN. Added the A100 312 peak vs. 90-155 sustained (30-50 percent MFU) numbers from the chapter definition callout.
- **q5 SHORT (five-SoC procurement)** — REWRITTEN to use the chapter's exact opening scenario (Vendor A 8 TOPS INT8 / Vendor B 15 TOPS INT4, etc.).
- **q6 MCQ (no single result characterizes platform)** — REWRITTEN with explicit "90 percent on compute-bound training, 10 percent on memory-bound inference" anchor.

### §4 Benchmarking Granularity (5 → 5)
- **q1 MCQ (softmax diagnosis)** — REWRITTEN. Added concrete 80 ms end-to-end / 3 ms softmax numbers to force diagnostic reasoning rather than pattern-matching.
- **q2 SHORT (isolation-representativeness trade-off)** — REWRITTEN with explicit "kernel 2x wins, macro unchanged, end-to-end 3 percent" cascade.
- **q3 MCQ (end-to-end vs. macro classification)** — KEPT (clean classification with strong distractors).
- **q4 TF (3x kernel ≠ 3x end-to-end)** — KEPT (uses the chapter's opening scenario; matches §16 TF-3 quantitative pattern).
- **q5 ORDER (scope spectrum)** — KEPT (genuinely sequential monotonic ordering).

### §5 Benchmark Components (6 → 6)
- **q1 MCQ (system specs missing)** — KEPT (clean component-identification).
- **q2 SHORT (harness shapes result)** — REWRITTEN. Added explicit Poisson-vs-sequential harness contrast with quantitative p99 implication.
- **q3 MCQ (anomaly detection multi-metric)** — REWRITTEN. Rotated correct letter to D; strengthened refutation of AUC-only framing.
- **q4 FILL (run rules)** — KEPT with slight sharpening.
- **q5 SHORT (compression beyond size)** — REWRITTEN with explicit unstructured-pruning / dense-accelerator mismatch as the concrete failure mode.
- **q6 MCQ (sustained thermal)** — REWRITTEN. Strengthened with "burst-mode vs. steady-state" mechanism.

### §6 Training vs. Inference (3 → 3, Tier 2 minimal)
- **q1 MCQ (why separate suites)** — REWRITTEN. Rotated correct letter from B to A; refutation cites EdgeTPU/NPU existence to block the "inference doesn't use accelerators" false framing.
- **q2 SHORT (memory divergence)** — REWRITTEN with explicit "4+ extra bytes per param for Adam moments + activations + gradients" breakdown.
- **q3 TF (same hardware, different metrics)** — KEPT (clean invariant).

### §7 Training Benchmarks (6 → 6)
- **q1 MCQ (time-to-accuracy primacy)** — REWRITTEN. Rotated correct letter from B to A; sharpened refutation.
- **q2 MCQ (24h/4h scaling efficiency)** — REWRITTEN. Made 24/(8·4)=0.75 arithmetic explicit; rotated correct letter from B to C for distribution.
- **q3 SHORT (precision / throughput / convergence)** — REWRITTEN with concrete 180→420 samples/second (2.3x throughput) numbers.
- **q4 MCQ (45 percent BERT utilization diagnosis)** — REWRITTEN. Rotated correct letter from B to A; strengthened mechanism with specific bottleneck sources (tokenization, AllReduce, PCIe).
- **q5 TF (fault tolerance)** — REWRITTEN with GPT-3 10,000 V100s concrete anchor.
- **q6 SHORT (reproducibility controls)** — KEPT with mild sharpening.

### §8 Inference Benchmarks (6 → 6)
- **q1 MCQ (p99 vs. mean)** — REWRITTEN. Added fan-out math (1 percent p99 → 9.6 percent expected-slow at 10 fan-outs); rotated correct letter to D for distribution.
- **q2 SHORT (component vs. end-to-end)** — REWRITTEN with concrete 5 ms model / 80 ms total numbers.
- **q3 MCQ (8 ms + 10 ms Amdahl)** — REWRITTEN. Made arithmetic explicit: 10→2 ms model, so 18→10 ms total = 1.8x.
- **q4 MCQ (MLPerf scenarios)** — KEPT (clean scenario matching; "Server" is a non-letter correct answer by content).
- **q5 FILL (cold-start)** — REWRITTEN. Added "no training analog" framing from prose.
- **q6 SHORT (NPU 2 ms misprediction)** — REWRITTEN with explicit thermal-and-duty-cycle mechanism.

### §9 Power Measurement Techniques (5 → 5)
- **q1 MCQ (boundary definition)** — REWRITTEN with the chapter's exact 10 TOPS / 0.5 W vs. 3 TOPS / 2 W (6.7x gap) opening example; correct answer made explicit (option text itself identifies boundary as the fix).
- **q2 SHORT (instantaneous vs. sustained)** — REWRITTEN. Added concrete transformer forward/backward/AllReduce phase pattern with kHz sampling recommendation.
- **q3 MCQ (5 percent perf / 50 percent power)** — REWRITTEN with cubic-voltage-scaling mechanism in refutation.
- **q4 TF (memory energy is small)** — REWRITTEN. Added 60-80 percent HBM/DRAM energy share for memory-bound workloads.
- **q5 SHORT (MLPerf Power cross-scale)** — REWRITTEN with explicit 150 µW / 10 kW four-orders-of-magnitude table anchor.

### §10 Benchmarking Best Practices (6 → 6)
- **q1 MCQ (CIFAR-10 deployment failure)** — REWRITTEN. Added concrete 95 percent→70 percent accuracy drop; strengthened distractor contrasts.
- **q2 SHORT (rigor ≠ alignment)** — REWRITTEN. Added concrete mean-vs-p99 and held-out-vs-deployment example contrasts.
- **q3 MCQ (hardware lottery concept)** — REWRITTEN. Rotated correct letter from B to D; added Hooker 2021 attribution and GPU-Tensor-Core transformer example from prose.
- **q4 MCQ (defense against benchmark engineering)** — KEPT (clean with C correct).
- **q5 FILL (hardware lottery)** — KEPT (reasoned from described mechanism).
- **q6 SHORT (benchmark evolution)** — REWRITTEN with ImageNet 50→97 percent saturation arc as concrete anchor.

### §11 Model and Data Evaluation (6 → 6)
- **q1 MCQ (calibration diagnosis)** — REWRITTEN. Added concrete 0.3 percent top-1 preservation + confidence-threshold downstream failure.
- **q2 SHORT (Pareto-frontier compression)** — REWRITTEN with concrete "1 percent accuracy loss, 3x smaller, 2.5x lower energy" Pareto example.
- **q3 MCQ (sepsis Hospital A/B)** — REWRITTEN. Substituted stronger clinical deployment-shift scenario; rotated correct letter from B to C.
- **q4 TF (data benchmarking redundancy)** — KEPT (clean misconception refutation).
- **q5 MCQ (LLM benchmark difficulty)** — REWRITTEN. Named the specific interacting axes (factuality, calibration, safety, reasoning, instruction-following) in correct choice; refutation now blocks each false alternative by content.
- **q6 SHORT (factory-floor defect detection)** — REWRITTEN. Substantial rework into a three-axis diagnostic walkthrough with concrete 8 percent misclassification rate.

### §12 Production Considerations (3 → 3, Tier 2 minimal)
- **q1 MCQ (bursty traffic assumption)** — REWRITTEN. Added Black-Friday traffic framing; refutation now explicitly blocks Poisson-vs-burst confusion.
- **q2 SHORT (trace replay)** — REWRITTEN with concrete "10,000 QPS benchmark / 50 ms p99 → 450 ms p99 under real burst" mechanism.
- **q3 TF (monitoring = continuous benchmarking)** — KEPT (clean per §16 TF pattern).

### §13 Fallacies and Pitfalls (5 → 5)
- **q1 TF (leaderboard rank transfer)** — KEPT.
- **q2 MCQ (1000→1200 QPS at 180→420 W)** — REWRITTEN. Made QPS/W arithmetic explicit (5.56→2.86, 49 percent efficiency loss) so reader must do the joint-metric calc.
- **q3 SHORT (saturated benchmarks)** — REWRITTEN. Named MNIST at 99.8 percent as concrete anchor for saturation argument.
- **q4 MCQ (research vs. production)** — KEPT with refutation sharpening.
- **q5 SHORT (Goodhart in engineering priorities)** — REWRITTEN with concrete ImageNet 76.2→76.5 with broken calibration example.

### §14 Summary (3 → 3, Tier 2 minimal)
- **q1 MCQ (final view)** — KEPT (clean synthesis; correct B preserved).
- **q2 SHORT (why rigorous measurement matters)** — REWRITTEN with concrete 1.3x vs. 3x end-to-end contrast as the engineering stake.
- **q3 MCQ (Amdahl summary takeaway)** — KEPT (clean synthesis with C correct).

## Summary of changes

| Action | Count |
|---|---|
| Questions rewritten | 53 |
| Questions kept (unchanged or minor polish) | 18 |
| Questions deleted | 0 |
| Questions added | 0 |
| Question-type conversions | 0 |
| Net total | 71 (was 71) |

## MCQ correct-letter distribution (anti-shuffle-bug compliance)

After per-question rotation at generation time (no post-shuffle): A=7, B=9, C=9, D=5, plus 2 non-letter (the MLPerf "Server" scenario MCQ where the correct answer is named by its scenario term). All 32 MCQ refutations reference distractors by content (e.g., "the fastest-is-better framing", "the clock-frequency framing", "the 'invalid below 90 percent' framing", "the 'inference doesn't use accelerators' claim"), never by letter. Validator scan confirms zero `Option [A-D]` / `Choice [A-D]` / `Answer [A-D]` / `([A-D])` patterns.

## Type mix (chapter-level)

- MCQ: 32 (45.1 percent) — slightly above the 40 percent target but supported by the chapter's heavy use of scenario-based diagnostic reasoning
- SHORT: 23 (32.4 percent) — matches the 30 percent target
- TF: 9 (12.7 percent) — matches the 13 percent target
- FILL: 5 (7.0 percent) — matches the 8 percent target
- ORDER: 2 (2.8 percent) — slightly below the 9 percent target but appropriate given this chapter has fewer inherently-sequential processes than training or inference pipelines

## Three issue patterns fixed (most impactful)

1. **Throwaway distractors replaced with plausible-error distractors.** ~14 MCQs previously had at least one distractor that no informed reader would select (e.g., "the benchmark is invalid because hardware clock frequency is defective", "cloud deployments care about power trade-offs; edge devices do not"). Each was rewritten to encode a real mental-model failure: category errors (confusing memory-bound with compute-bound symptoms), scale-inversion errors (linear vs. cubic power scaling), or structural misreadings (backprop disappearing under precision changes).

2. **Quantitative grounding added from chapter prose.** 18 questions that used generic phrasing ("a model speedup", "lower precision") now carry the chapter's own numbers: A100 312 TFLOPS peak / 90-155 TFLOPS sustained, 24h/4h giving 75 percent strong-scaling, 8 ms + 10 ms Amdahl pipeline yielding 1.8x end-to-end, 10 TOPS/0.5W burst vs. 3 TOPS/2W sustained (6.7x gap), MobileNetV2 2 ms EdgeTPU, 150 µW TinyML to 10 kW server rack.

3. **Correct-letter distribution rebalanced from heavily skewed (B=18/32) to balanced (A=7, B=9, C=9, D=5, other=2).** Every rotation was accomplished by swapping choice positions and updating the "correct answer is [LETTER]" phrase; all distractor refutations remained content-based so no refutation broke. The anti-shuffle-bug contract in §10 of the spec is structurally upheld: if a future tool re-rotates choices, the refutations still refer to wrongness by substance.

## One substantial rework: §11 q6 (Model and Data Evaluation — factory-floor synthesis)

The original §11 q6 was a generic "walk through holistic diagnosis" prompt whose answer enumerated the three dimensions in order without tying them to the specific failure mode. The rewrite binds the question to a concrete scenario (compressed MobileNet on EdgeTPU, factory defect detection, 8 percent post-deployment misclassification despite excellent lab MLPerf scores) and restructures the answer as three diagnostic sub-questions — one per axis — each with a named mechanism (thermal/dust on system, calibration on model, lighting/occlusion on data). The answer now teaches the reader *how to apply* the three-dimensional framework to a new deployment failure, not how to recite it, which is the §16 SHORT-5 cross-chapter-connection standard applied to within-chapter synthesis.

## Validator + anti-shuffle status

- **Schema + anchor validation**: `OK: vol1_benchmarking_quizzes.json passes schema + anchor validation`
- **Letter-reference warnings**: 0 (`Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, `([A-D])` all absent from MCQ answers)
- **Section counts**: 14 total, 14 with quizzes, 0 skipped — matches metadata declarations
- **Per-section counts**: 3–6 questions each, with Tier 2 minimal sections at 3 (Training vs. Inference, Production Considerations, Summary) and Tier 1 full sections at 5–6
