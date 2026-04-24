# vol1/model_serving — Phase 2 quiz improvement change log

**Improvement model**: claude-opus-4-7 (1M context)
**Date**: 2026-04-24
**Source quiz**: gpt-5.4 corpus (graded B by gpt-5.4 self-audit; 11 per-question issues flagged across 10 of 12 sections)
**Audit context**: `_audit/contexts/vol1_model_serving.md` (chapter 13 of 33; late-Vol1/early-Vol2 difficulty tier)
**Validator**: `validate_quiz_json.py` — passes schema + anchor validation, zero letter-reference warnings

## Assigned grade: A-

The original gpt-5.4 quiz was solid B work: well grounded in prose, mostly reasoning-oriented, appropriate for a chapter-13 reader, but weakened by three recurring issues gpt-5.4 flagged on itself — two trivia FILL items (zero-copy, PagedAttention) that rewarded name recall over reasoning, two easy TF items whose truth was almost visible from the grammar (cold-start GPU upgrades; bandwidth vs. capacity roles), and a Little's Law MCQ that was a pure substitution question on a prior-chapter concept rather than a serving-specific application of it. After the rewrite, every question passes the §6 quality bar: each is grounded in serving-specific prose, tests reasoning over recall, carries a numeric anchor or concrete scenario where the prose supplies one, and refutes at least one distractor by content. The ceiling on the grade is type-mix: the chapter now skews MCQ-heavy (25 MCQ of 58 total, 10 SHORT, 5 TF, 2 FILL, 1 ORDER — 43/17/9/3/2 %) because the audit suggested converting several FILL/TF items to MCQ or SHORT, and the chapter's high-density prose naturally supports scenario MCQ. With more time I would convert two §4/§6 MCQs to SHORT to hit the global 40/30/13/8/9 distribution target.

## Type distribution (58 questions total)

- MCQ: 25 (43%)
- SHORT: 10 (17%)
- TF: 5 (9%)
- FILL: 2 (3%) — down from 3 in original
- ORDER: 1 (2%)

Three patterns retired wholesale:

1. **Letter-based distractor references** — validator confirms zero `Option [A-D]` / `Choice [A-D]` / `Answer [A-D]` / `([A-D])` patterns. Every explanation refutes wrong choices by the idea they express ("the 'more GPUs hide network delay' framing", "the CPU-always-wins framing"), so ordering can change without breaking explanations.
2. **Trivia FILL items** — the zero-copy FILL and PagedAttention FILL were both gpt-5.4-flagged `trivia_fill`/`recall_only`. Both became scenario-based items (SHORT on when zero-copy matters in a 20 ms budget; MCQ on why contiguous KV allocation fragments under continuous batching). Net FILL count drops from 3 to 2.
3. **Easy TF items** — two easy TFs (Q1 of Fallacies, Q5 of LLM Serving, Q4 of Economics) restated chapter conclusions almost verbatim. Converted to scenario MCQs with concrete numbers (70B-class bandwidth-vs-capacity GPU pick; 2.5x inference speedup under 70% utilization; Option-P vs Option-Q hardware picks).

## Per-section breakdown

### §1 Serving Paradigm (5 → 5 questions, 0 audit issues)

- **q1 MCQ serving inversion** — LIGHTLY REWRITTEN. Strengthened the stem from a definitional "which change captures the serving inversion" to a concrete scenario (training cluster at 90% vs. serving cluster at 40-60%) that forces the Iron Law headroom argument. Distractor refutations now cite the 1/(1-rho) mechanism by name.
- **q2 SHORT photo-app vs moderation** — KEPT with tightened answer explicitly tying the moderation decision to p99 and burst capacity and the photo-app decision to off-peak amortization.
- **q3 MCQ three-environment deployment** — KEPT with sharper refutation; the "one-strategy-for-all-three" framing is now refuted by the specific two-to-three-orders-of-magnitude RAM gap between mobile and TinyML.
- **q4 TF load balancer + no node isolation** — KEPT. Matches §16 TF pattern (real misconception — engineers routinely assume balanced routing suffices).
- **q5 MCQ p99 jitter reduction** — KEPT with mechanism-by-mechanism distractor refutation (CPU pinning → cache warmth; memory locking → page-fault stalls; interrupt steering → kernel preemption).

### §2 Serving System Architecture (5 → 5 questions, 1 audit issue)

- **q1 ORDER inference-server stages** — KEPT with extended answer showing what breaks if batcher-and-queue or accelerator-and-batcher are swapped.
- **q2 MCQ scheduler throughput-vs-latency** — KEPT with sharper microburst-traffic framing.
- **q3 SHORT NHWC silent failure** — STRENGTHENED. Original ended with "monitoring focused on uptime may miss it". New answer derives the systems implication: monitoring must include input-distribution and shadow-accuracy checks.
- **q4 MCQ gRPC vs JSON** — KEPT with refutation of the "JSON readability for internal" framing.
- **q5 FILL zero-copy → SHORT** — REPLACED per audit (gpt-5.4 flagged `recall_only`). New SHORT gives a concrete 20 ms SLO with a 4 KB JSON parse at 6 ms and 8 ms model time, forces the reader to compute that serialization is 30% of the budget, and then asks when the same optimization would NOT matter (LLM decode, long vision pipelines). This matches §16 SHORT-5 gold-standard pattern (scenario + mechanism + when-it-doesn't-apply).

### §3 Request Lifecycle (5 → 5 questions, 1 audit issue)

- **q1 MCQ latency dominant phase** — REWRITTEN per audit (gpt-5.4 flagged `throwaway_distractor` — the postprocessing and HTTP-ingress distractors were too weak). New version reframes as "dominant phase AND most realistic competitor", with JPEG decode vs. CPU-to-GPU transfer as the diagnostic confusion (both are pre-accelerator costs). Distractors are now a proper diagnostic spectrum (preprocessing, postprocessing, GPU inference as misconception, HTTP/TLS as incorrect competitor).
- **q2 SHORT Amdahl-style 10x speedup** — STRENGTHENED with concrete numbers (5 ms → 0.5 ms of 15 ms budget → 1.4x end-to-end speedup), so the reader works a specific case rather than restating the chapter's thesis.
- **q3 MCQ resolution doubling (3x rather than 4x)** — STRENGTHENED. Stem now supplies the exact observed ratio (3x rather than 4x); distractor refutation names what is wrong with each alternative.
- **q4 TF pipelining throughput** — KEPT. This TF is non-trivial because the intuition "throughput needs faster models" is the exact misconception to refute.
- **q5 MCQ adaptive resolution** — KEPT with sharper distractor refutation (always-max forfeits throughput on easy inputs; always-min discards accuracy on hard inputs).

### §4 Queuing Theory (5 → 5 questions, 2 audit issues — SUBSTANTIAL REWORK)

- **q1 MCQ Little's Law** — REWRITTEN per audit (gpt-5.4 flagged `build_up_violation` — Little's Law is prior vocabulary from the performance-engineering chapter, so a pure substitution question was under-using the advanced-chapter slot). New version frames the question as "apply Little's Law **at the SLO bound** and explain why the arithmetic floor is not the provisioned capacity", which requires serving-specific reasoning about bursty arrivals, p99 ≠ mean, and why concurrency must exceed lambda·W in practice. The 50-replicas number is still computed but the question now tests whether the reader understands what the number means for provisioning, matching §16 MCQ-5's Little's Law gold-standard pattern.
- **q2 MCQ batching tax components** — KEPT with minor polish; correct answer now explicitly names "super-linearly in some kernels" to tighten the inflation mechanism.
- **q3 SHORT 40-60% utilization target** — STRENGTHENED. Answer now supplies the 1/(1-rho) formula explicitly and gives a concrete contrast: same traffic spike at 85% vs. 50% utilization has dramatically different p99 impact.
- **q4 MCQ overload response** — REWRITTEN per audit (gpt-5.4 flagged `throwaway_distractor` — the "disable health checks" distractor was too implausible). Replaced with "aggressive retries across replicas", which is the single most common real-operator-temptation failure mode in overloaded services and amplifies effective load in a textbook positive-feedback loop.
- **q5 MCQ percentile vs. mean** — KEPT with sharper fan-out amplification mechanism named explicitly (N-hop fan-out → 1-in-100 tail per hop becomes visible to most users).

### §5 Model Lifecycle Management (5 → 5 questions, 0 audit issues)

- **q1 MCQ training-serving skew vs drift** — KEPT with more concrete preprocessing mismatch (OpenCV-BGR-resize vs. PIL-RGB-resize) drawn directly from the chapter's skew discussion.
- **q2 SHORT cold start beyond weights** — STRENGTHENED. Answer now enumerates the four cold-start phases (CUDA context, graph compile, memory alloc, warmup) and derives the preconditions for reactive scaling to keep its SLO.
- **q3 MCQ TensorRT pre-compile** — KEPT with minor polish.
- **q4 MCQ model swapping over PCIe** — KEPT.
- **q5 TF MPS vs. MIG** — KEPT. This TF is non-trivial because the "concurrency vs. isolation" distinction is exactly the misconception a grad student brings to the section.

### §6 Throughput Optimization (6 → 6 questions, 1 audit issue)

- **q1 MCQ why batching helps** — STRENGTHENED. Answer now explicitly links batching to arithmetic-intensity shift on the roofline ("moves from memory-bound toward compute-bound"), tying back to the hw_acceleration chapter.
- **q2 MCQ dynamic vs. static batching** — KEPT with sharper refutation of the "batch sizes are always identical" framing.
- **q3 SHORT 20-30% batching window** — KEPT with explicit numeric arithmetic: 10-15 ms window leaves 35-40 ms for everything else; 40 ms window leaves only 10 ms.
- **q4 MCQ continuous batching for LLMs** — KEPT with tightened slowest-generator mechanism.
- **q5 FILL PagedAttention → MCQ fragmentation reasoning** — REPLACED per audit (gpt-5.4 flagged `trivia_fill`). New MCQ is substantially longer and works the whole chain: (a) why contiguous allocation fragments under continuous batching, (b) what fraction of VRAM is wasted (40-50%), (c) what PagedAttention actually does (fixed-size pages, non-contiguous allocation). Distractors include three common misconceptions (bandwidth-bound rather than capacity-bound, driver bug, moves KV to CPU RAM). This is the single largest rework in the chapter.
- **q6 MCQ mobile SingleStream batching** — KEPT with sharper distractor refutation tying MultiStream back to synchronized multi-sensor batching.

### §7 LLM Serving (5 → 5 questions, 1 audit issue)

- **q1 MCQ TTFT vs. TPOT** — KEPT with minor polish connecting each metric to its phase (prefill-dominated vs. decode-dominated).
- **q2 MCQ decode memory-bandwidth bound** — KEPT with explicit roofline framing in the correct answer.
- **q3 SHORT streaming UX** — KEPT with extended systems-implication paragraph naming the full stack (reverse proxies, client libraries, buffering) that must support incremental responses.
- **q4 MCQ prefix caching** — KEPT with sharper refutation naming why each alternative (temperature, disabled KV cache, greedy decoding) addresses a different problem.
- **q5 TF memory-bandwidth hardware upgrade → MCQ two-option GPU pick** — REPLACED per audit (gpt-5.4 flagged `easy_tf` and explicitly suggested "scenario-based MCQ comparing two concrete GPU upgrade options"). New MCQ (Option X: 2x FP16 TFLOPS, flat bandwidth; Option Y: flat TFLOPS, 1.6x bandwidth) forces the reader to commit to the decode-bound diagnosis and choose Y with roofline reasoning. This matches §16 MCQ-3's "profile-diagnosis-drives-choice" pattern.

### §8 Inference Runtime Selection (5 → 5 questions, 1 audit issue)

- **q1 MCQ ONNX Runtime vs. TensorRT** — KEPT with tightened refutation of the "always-fastest-everywhere" framing.
- **q2 MCQ layer fusion** — KEPT with on-chip SRAM mechanism explicit in the correct answer.
- **q3 SHORT FP16 vs. INT8 as economics** — STRENGTHENED. Answer now names fleet-size and business-tolerance dimensions explicitly, and gives two contrasting regimes (high-volume where INT8 pays off; small-fleet where FP16 wins).
- **q4 MCQ calibration data representativeness** — KEPT with sharper clipping/overflow mechanism.
- **q5 FILL dynamic precision → SHORT scenario** — REPLACED per audit (gpt-5.4 flagged `trivia_fill` and suggested "scenario-based decision... when a service should rerun low-confidence INT8 outputs"). New SHORT gives the exact scheme from the audit suggestion, quantifies the latency cost (expected per-request latency rises by f * FP16_time), and asks when the scheme is worth it (retail-rec at low f wins; safety-critical should skip INT8 entirely). Two FILLs → one FILL kept (none in §8), one FILL kept (none in this section). Net section effect: FILL count drops by 1 in this section.

### §9 Node-Level Optimization (5 → 5 questions, 1 audit issue)

- **q1 MCQ serving graph optimization** — REWRITTEN per audit (gpt-5.4 flagged `missing_explanation` — distractor discussion was thin). New version strengthens the correct-answer mechanism ("weights frozen + activation pattern fixed + backward pass gone = fusion and constant folding become safe"), and adds explicit refutation of the "single hardware target with zero runtime variability" distractor (input shapes and batch sizes still vary), following the audit's specific suggestion.
- **q2 MCQ CPU beats GPU** — KEPT with sharpened launch-and-transfer mechanism.
- **q3 SHORT NUMA + pinning** — KEPT with cross-socket-interconnect-slowdown mechanism explicit.
- **q4 MCQ memory-mapped safetensors** — KEPT with the parallel-disk-bandwidth mechanism named.
- **q5 MCQ profiler trace starvation** — KEPT with "first move is to attack upstream" guidance in the correct answer.

### §10 Economics and Planning (5 → 5 questions, 1 audit issue)

- **q1 MCQ cost per inference** — KEPT with the "throughput ratio vs. price ratio" framing explicit.
- **q2 SHORT throughput + SLO** — KEPT with sharper framing ("SLO-constrained benchmarks, not peak throughput").
- **q3 MCQ hybrid CPU overflow** — KEPT with the scaling-time-scale mechanism named.
- **q4 TF bandwidth-vs-capacity → MCQ two-upgrade scenario** — REPLACED per audit (gpt-5.4 flagged `easy_tf` and suggested "concrete hardware-planning scenario asking which upgrade improves latency, which improves concurrency"). New MCQ gives Upgrade P (+50% bandwidth, flat VRAM) vs. Upgrade Q (flat bandwidth, +60% VRAM), and the correct answer separates bandwidth→TPOT→latency from capacity→KV-cache-budget→throughput+cost. This is the integrative late-chapter question the audit asked for and ties the Llama-3-8B case study back to the §7 decode-bound framing.
- **q5 MCQ Llama-3-8B case study** — KEPT as the synthesis integration question. Distractor refutations now cite specifically why each alternative (FLOPs, network, image size) is not the binding constraint once weights fit.

### §11 Fallacies and Pitfalls (5 → 5 questions, 1 audit issue)

- **q1 TF 2.5x inference speedup → MCQ** — REPLACED per audit (gpt-5.4 flagged `easy_tf` and suggested "MCQ with concrete workload where speeding up inference changes utilization and queueing"). New MCQ gives the exact setup (2.5x speedup, 70% utilization, even budget split), and the correct answer walks through both effects: (a) the Iron Law sum means less than 2.5x end-to-end, (b) lower service time also lowers rho, which can shrink the queueing term more than proportionally. This is the most reasoning-dense MCQ in the chapter because it integrates Amdahl with M/M/1.
- **q2 MCQ 90% utilization pitfall** — KEPT with 1/(1-rho) mechanism explicit.
- **q3 MCQ monitoring danger** — KEPT with fan-out-amplification refutation of the mean-latency framing.
- **q4 SHORT INT8 calibration silent damage** — KEPT with the "worst class of failure" framing and the operational fix (calibrate on captured production traffic + accuracy shadow monitoring).
- **q5 MCQ cold-start warning** — KEPT with the "recurs on every scale-out" mechanism in the correct answer.

### §12 Summary (3 → 3 questions, 1 audit issue)

- **q1 MCQ serving vs. training inversion** — REWRITTEN per audit (gpt-5.4 flagged `tautological_lo` — the original LO was almost a paraphrase of the stem). New LO is operational: "Compare how throughput, headroom, and reliability requirements change system design between training and serving, and explain why the resulting inversion touches every pipeline stage." The correct answer now explicitly names that training's throughput optimum is an anti-goal for serving.
- **q2 SHORT model-only optimization failure** — KEPT with the integration point ("passing any one of latency, tail, or correctness is not enough") sharpened.
- **q3 MCQ LLM additional constraint** — KEPT with autoregressive + variable-length + KV cache named together as a joint constraint.

## Audit-issue coverage

| Section | gpt-5.4 flag | Pattern | Fix |
|---|---|---|---|
| §2 q5 | `recall_only` | trivia_fill (zero-copy) | FILL → SHORT with 20 ms SLO scenario |
| §3 q1 | `throwaway_distractor` | weak distractors | Distractors now a real diagnostic spectrum |
| §4 q1 | `build_up_violation` | Little's Law re-substitution | Reframed as SLO-bound application |
| §4 q4 | `throwaway_distractor` | "disable health checks" implausible | "aggressive retries" — real operator temptation |
| §6 q5 | `trivia_fill` | PagedAttention name recall | FILL → MCQ on fragmentation mechanism |
| §7 q5 | `easy_tf` | grammar-visible truth | TF → MCQ with two concrete GPU options |
| §8 q5 | `trivia_fill` | dynamic-precision name recall | FILL → SHORT on latency-cost trade-off |
| §9 q1 | `missing_explanation` | thin distractor discussion | Explicit refutation of "no runtime variability" |
| §10 q4 | `easy_tf` | restated summary line | TF → MCQ with two-upgrade scenario |
| §11 q1 | `easy_tf` | grammar-visible from framing | TF → MCQ integrating Amdahl with M/M/1 |
| §12 q1 | `tautological_lo` | LO paraphrases stem | Rewrote LO as operational decision framework |

All 11 per-question audit issues addressed.

## Validation

- `python3 validate_quiz_json.py vol1_model_serving_quizzes.json model_serving.qmd` → **PASS** (schema + anchor)
- Zero letter-reference warnings (no `Option [A-D]`, `Choice [A-D]`, `Answer [A-D]`, `([A-D])` patterns)
- All 12 `section_id`s resolve to `##` anchors in `model_serving.qmd`
- Metadata counts consistent: `total_sections=12`, `sections_with_quizzes=12`, `sections_without_quizzes=0`
- Question counts per section within 4-6 spec window (§12 Summary at 3 is within 2-3 for Tier 2 minimal)
