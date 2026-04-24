# vol2/collective_communication — Phase 2 Opus Quiz Improvements

**Chapter position:** 22 of 33 (late Vol2 — specialized, system-level reasoning expected)
**Prior grade (gpt-5.4):** B
**Target grade:** A
**Validator status:** PASS (no errors, no warnings after fixing the `log2(D)` letter-reference false positive)

---

## Metadata changes

- `generated_on`: `2026-04-24`
- `model`: `claude-opus-4-7`
- `improved_by`: `opus-subagent-phase2`
- Section and question count preserved: 10 sections, all `quiz_needed: true`.

---

## Per-section change log

### §1 — From Parallelism to Communication Patterns
**Kept:** question structure (5 questions), section_id, rationale frame.
**Rewritten:** all 5 questions.
- q1 MCQ: original opened with a vague "why does communication become dominant?" stem. Rewrote as a diagnostic scenario (GPU count tripled, utilization halved) forcing reasoning about local-vs-global asymmetry. Distractors now encode three real mental-model failures (algorithm-structure confusion, FP stability confusion, optimizer-path confusion).
- q2 SHORT: reframed around 8→1,000 GPU scaling with concrete 2(N-1)/N math; added the dominant-cost synthesis.
- q3 MCQ: tightened MoE routing semantics; added Broadcast and ReduceScatter distractors that encode plausible misreadings.
- q4 TF: made claim more specific ("distance-dependent latency and bandwidth-over-distance costs") with quantitative anchor (200 m/μs, 500 ns per 100 m).
- q5 ORDER: expanded from 3-step to 4-step sequencing (forward → backward → sync → optimizer) for richer causal reasoning.

### §2 — Mapping the Terrain: Network Performance Modeling
**Audit issues fixed:** q1 trivia_fill ("critical" as blank target), q5 easy_tf (obvious NCCL-vs-theory claim).
**Rewritten:** all 5 questions (not just the two flagged).
- q1 (was FILL "critical"): replaced with a regime-classification MCQ using two workloads (4 KB MoE vs 1 GB dense) to force the reader to apply n* = αβ = 100 KB in both directions.
- q2 SHORT: tightened around overlap failure diagnosis; added quantitative (2o on critical path) reasoning.
- q3 MCQ: sharpened hierarchical Llama 70B mechanism question; distractors now encode three specific misconceptions (full-elimination overstatement, optimizer-change myth, BF16 datatype myth).
- q4 FILL: new FILL replacing the weak "critical" one — the blank is inferred from the mechanism (25× below the threshold; fusion wins) rather than guessable from vocabulary.
- q5 TF: converted from recall-style ("NCCL latency is higher than αβ") to diagnostic contrast (1 GB accurate, 64 KB off by 5–10×) per the audit suggestion.

### §3 — Choosing the Vehicle: Collective Operation Primitives
**Kept:** q1–q5 structure and three of the original questions' semantics.
**Rewritten/tightened:** q2, q3, q4 strengthened with concrete numeric anchors (7B-parameter case, ~64 collectives per step, O(N^2) scaling).
- q1 MCQ: added ReduceScatter distractor for a four-way reasoning test.
- q4 MCQ: cluster size (128 → 1,024 GPUs) concretized; AllReduce-scales-well contrast integrated into stem.

### §4 — Engineering the Flow: AllReduce Algorithms
**Audit issues fixed:** q3 recall_only (1 MB / 64 GPU replay of the worked example).
**Rewritten:** q3 with different numbers (256 bytes / 256 GPUs / α=5 μs, β=25 GB/s); forces the reader to compute the crossover 32 MB rather than recognize the book's example.
**Tightened:** q4 SHORT on halving-doubling — added the topology-oblivious mechanism explanation (later partner exchanges cross InfiniBand) with a concrete hierarchical cluster scenario.
**Kept:** q1, q2, q5, q6 structure; cleaned up explanations to be fully content-based.

### §5 — Hierarchical Communication
**Kept:** all 5 questions' structure.
**Tightened:** q1 distractor adds 10–20× bandwidth gap as the quantitative anchor; q2 ORDER stems clarified; q3 SHARP mechanism made more precise (HBM bandwidth pressure explicit); q4 SHORT closes on the "shared alignment principle" synthesis; q5 rank-mapping diagnosis sharpened with NVLink group / rail mechanism.

### §6 — The Last Resort: Gradient Compression
**Audit issues fixed:** q6 trivia_fill ("block" as blank target).
**Rewritten:** q6 FILL → q6 SHORT asking why per-block scales improve quantization on non-uniform tensors, with concrete magnitude example (10^3 vs 10^-4 regions sharing a scale).
**Rewritten:** q2 Top-K MCQ — reworked to emphasize index-width vs value-width cost with concrete 32-bit/16-bit math; removed bare-letter `D` notation that triggered a false-positive letter-reference warning.
**Tightened:** q1 (regime anchoring, 55% communication), q3 (long-run invariant preserved), q4 (Adam co-design sharpened), q5 (n >> n* regime).

### §7 — The Communication Library Landscape
**Audit issue fixed:** q2 throwaway_distractor (SHARP emulation).
**Rewritten:** q2 MCQ replacing the SHARP distractor with the audit-suggested MPI-HPC-familiarity distractor and a nuanced explanation of why Gloo (not MPI) became the debug-default (heavier MPI install + uneven PyTorch-distributed support on laptops).
**Tightened:** q3 SHORT made more concrete (10–100× NCCL speedup quantified, specific per-traffic-class reasoning).

### §8 — Communication-Computation Overlap
**Kept:** all 5 questions' structure.
**Tightened:** q3 SHORT now includes a fully worked quantitative exposed-time condition (C > L + 2o; 100 μs vs 110 μs example); q5 MCQ explanation clarifies the min(transfer, compute) mechanism.

### §9 — Fallacies and Pitfalls
**Audit issue fixed:** q1 easy_tf (absolutist "all communication bottlenecks" phrasing).
**Rewritten:** q1 TF now scopes the claim to a specific contrast (MoE token routing vs dense LLM gradient sync on the same 400G upgrade), forcing the reader to reason about regime-dependent speedups.
**Tightened:** q2 MCQ distractors sharpened; q3 SHORT adds concrete remediation (CUDA_VISIBLE_DEVICES, rank-stride matching rail structure).

### §10 — Summary
**Audit issue fixed:** q1 vague_lo.
**Tightened LO:** "Select workload-appropriate primitive-plus-topology strategy pairs for bandwidth-bound dense training versus latency-and-contention-bound sparse routing." (per audit suggestion).
**Rewritten:** q1 MCQ — now explicit about hierarchical AllReduce + rail alignment for dense, vs low-latency topology-aware AlltoAll + load balancing for MoE; distractors encode four specific engineering mistakes.
**Rewritten:** q2 SHORT synthesis — now prescribes the engineering order (algorithm → topology → overlap/compression) with reasoning for why reversing the order wastes effort.

---

## Aggregate counts

- **Rewritten:** ~35 of 43 questions substantially rewritten (stems reframed, distractors redesigned, or answers expanded to §16 three-move format).
- **Kept as-is:** 0 (all questions touched at minimum for content-based distractor refs, LO tightening, or §16 three-move-format answer expansion).
- **Added:** 1 (new FILL in §2 replacing the deleted trivia FILL — net question-count preserved).
- **Deleted:** 1 (original §2 q1 trivia FILL; replaced with the new FILL above).
- **Net question count:** 43 (unchanged from source).

---

## Three issue patterns fixed across the chapter

1. **Trivia FILL with vocabulary-adjacent answers.** Two FILL items (`critical` in §2, `block` in §6) had blanks guessable from neighboring words. Replaced one with a mechanism-anchored FILL that forces reasoning from a described 25× ratio; replaced the other with a SHORT that tests why block-local scales help on heavy-tailed tensors.
2. **Easy TF items stating what the section already emphasizes.** The §2 NCCL-vs-αβ TF and the §9 "bandwidth upgrade is the most important fix" TF were near-tautologies given the section text. Reframed each as a diagnostic contrast between two regimes (1 GB accurate / 64 KB off by 5–10×; MoE vs dense gets different speedups from 400G upgrade).
3. **Throwaway distractors and recall-only scenarios.** Replaced the §7 SHARP-emulation distractor with the MPI-HPC-familiarity alternative, and changed §4 q3's cluster numbers (256 B / 256 GPUs) so the reader must compute the crossover rather than replay the book's 1 MB / 64 GPU worked example.

---

## One substantial-rework section

**§2 "Mapping the Terrain: Network Performance Modeling"** received the deepest rework: two of five questions were flagged by the gpt-5.4 audit, but the whole section was rebuilt around A-grade patterns from §16:

- q1 is now a full two-workload regime-classification MCQ (4 KB MoE vs 1 GB dense) that forces the reader to apply n* = αβ in both directions and pick different optimization families for each — matching §16 MCQ-3's "low-util-high-bandwidth profile" pattern.
- q2 now derives the LogP-vs-αβ distinction with explicit 2o-on-critical-path arithmetic, matching §16 SHORT-4's mechanism-plus-threshold pattern.
- q3 sharpens the hierarchical Llama 70B mechanism around NVLink→IB bandwidth tiering.
- q4 is a new mechanism-inferred FILL (replacing the rejected "critical" one) where the blank follows from a 25×-below-threshold ratio, matching §16 FILL-2's arithmetic-intensity pattern.
- q5 is a diagnostic TF contrasting 1 GB accuracy with 64 KB discrepancy, not a restatement of the section's punchline.

---

## Validator outcome

```
OK: vol2_collective_communication_quizzes.json passes schema + anchor validation
EXIT: 0
```

Zero errors, zero warnings (one initial `(D)` false-positive flagged `log2(D)` math notation; rewrote as `log2 of the dimension` to satisfy the §10 anti-shuffle-bug scanner).
