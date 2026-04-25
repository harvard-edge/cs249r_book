# Cloud Parallelism Cleanup Report

Date: 2026-04-25
Manifest: `tools/phase_g/cleanup_manifest.json` (47 items)
Output: edits applied in place to `interviews/vault/questions/cloud/*.yaml`

## Totals

| Disposition | Count |
|---|---:|
| Rewritten (kept, status: published) | 31 |
| Archived (status: archived, deletion_reason set) | 16 |
| Errors / unfixable | 0 |
| **Total** | **47** |

All edits preserve `track`, `level`, `zone`, `topic`, `competency_area`, `bloom_level`,
`id`, `chains`, and any existing `visual` block. No `visual` blocks were added.
No `status: draft → published` promotions were made.

## Rewrites (31)

### cloud-1088 — NEEDS_FIX (math imprecision)
- **Verdict:** rewritten
- **Change:** clarified scenario uses *hierarchical* AllReduce; rewrote napkin_math to derive 6 GB inter-node bus traffic from N=4 nodes (the inter-node ring), not flat N=32; updated common_mistake to flag flat-vs-hierarchical confusion.
- **Key concern:** the math now matches the stated topology and the 0.48 s exposed time.

### cloud-0687 — DROP (no diagram, but salvageable)
- **Verdict:** rewritten
- **Change:** removed "Based on the diagram" from scenario and question; reframed as direct optimizer-explosion sizing problem.
- **Key concern:** Q is now self-contained; no visual block needed.

### cloud-2799 — DROP (cell-fit / scenario error)
- **Verdict:** rewritten
- **Change:** replaced Cortex-M4 federated edge premise with cross-region cloud DP aggregation across 192 H100s with secure multi-party-computation cross-region averaging; added concrete bandwidth numbers (NVLink 900 GB/s bidirectional, IB NDR 400 Gbps, 200 Gbps WAN) and ring-AllReduce 2*(N-1)/N math.
- **Key concern:** kept (track=cloud, level=L6+, zone=mastery, topic=data-parallelism, bloom=create) intact while replacing the broken edge premise.

### cloud-2701 — DROP (deprecated SSP/ASP for LLM training)
- **Verdict:** rewritten
- **Change:** replaced BSP/SSP/ASP framing with synchronous FSDP + backup-worker + node-remediation comparison vs bounded-staleness async; added p50/p99 numbers (720 ms vs 1.4 s) and quantified throughput cost.
- **Key concern:** modern practice (synchronous FSDP) is now the correct answer; common_mistake explicitly calls out SSP/ASP as deprecated for foundation-model training.

### cloud-1609 — NEEDS_FIX (artificial pure-MP setup)
- **Verdict:** rewritten
- **Change:** reframed as an explicit theoretical sizing exercise that motivates ZeRO/FSDP, with parenthetical noting "production teams use FSDP/ZeRO; this exercise isolates the sizing math that motivates them."
- **Key concern:** the math (16 bytes/param * 30B / 32 GB usable = 15 → 16 GPUs) is unchanged; the framing now signals that pure-MP is not recommended.

### cloud-0823 — DROP (math error: bidirectional vs unidirectional NVLink)
- **Verdict:** rewritten
- **Change:** corrected divisor from 900 GB/s bidirectional to 450 GB/s unidirectional → time = 109 ms not 54 ms; updated MCQ options (5/54/109/1090 ms) and correct_index = 2.
- **Key concern:** common_mistake now explicitly names the bidirectional/unidirectional confusion as the trap.

### cloud-0899 — NEEDS_FIX (NVLink unidirectional)
- **Verdict:** rewritten (napkin_math only)
- **Change:** updated NVLink to 450 GB/s unidirectional (0.88 ms) and IB NDR to 25 GB/s unidirectional (~16 ms per layer); cumulative inter-node TP penalty restated as ~1.5 s.
- **Key concern:** consistency with PARALLELISM_RULES on unidirectional ring AllReduce bandwidth.

### cloud-0694 — DROP (no diagram, but salvageable)
- **Verdict:** rewritten
- **Change:** removed "Based on the router diagram" reference; added concrete profile signature (8x compute on expert 2, 7 GPUs at near-zero) so the diagnosis is grounded in measurable data.
- **Key concern:** Q is now self-contained; same MoE expert-imbalance teaching preserved.

### cloud-3927 — DROP (math error: GPipe ≠ 1F1B bubble fraction)
- **Verdict:** rewritten
- **Change:** corrected the central claim — GPipe and 1F1B share the *same* bubble fraction (k-1)/(m+k-1); reframed Q to ask what 1F1B *actually* saves (peak activation memory O(m) → O(k)).
- **Key concern:** common_mistake explicitly flags the (formerly canonical) "1F1B reduces bubble" misstatement.

### cloud-2807 — NEEDS_FIX (topic mismatch)
- **Verdict:** rewritten
- **Change:** replaced activation-recompute scenario with heterogeneous-bandwidth DP AllReduce architecture across an IB island + Ethernet half (128 A100s, 16 nodes, 2 gateway switches); added 3-level hierarchical AllReduce design with concrete numbers.
- **Key concern:** topic is now genuinely DP-centric; the bubble-of-bandwidth-asymmetry is the load-bearing teaching.

### cloud-1748 — NEEDS_FIX (topic mismatch: TP not DP)
- **Verdict:** rewritten
- **Change:** kept the "missing process group" diagnosis but re-anchored on the *DP* gradient AllReduce defaulting to `dist.group.WORLD` instead of TP AllReduce; concrete numbers (175B / TP*PP shard, 8-rank IB ring, 1024-rank fallback).
- **Key concern:** scenario now matches the data-parallelism topic.

### cloud-3429 — NEEDS_FIX (LLM filler)
- **Verdict:** rewritten
- **Change:** replaced "diagnose and propose" boilerplate with a concrete heterogeneous-stage scenario (stages 1-7 = 100 ms, stage 8 = 175 ms LM head), forcing the candidate to compute the bubble fraction and identify both the imbalance and the m-too-small problem.
- **Key concern:** uniqueness gain comes from the asymmetric stage timing.

### cloud-3427 — NEEDS_FIX (LLM buzzwords, trivial math)
- **Verdict:** rewritten
- **Change:** replaced "colossal/boasting/frontier" buzzwords with explicit 80 GB H100 budget (60 GB consumed by non-activation state) and 1F1B activation-budget arithmetic; the answer requires choosing m=64 and verifying the 2.5 GB per-micro-batch activation budget.
- **Key concern:** the math is now multi-step (memory budget + bubble target) instead of a trivial division.

### cloud-3449 — NEEDS_FIX (5-bullet listicle)
- **Verdict:** rewritten
- **Change:** collapsed five generic bullets into a single concrete TPU v5e contention scenario: inference SLO 80 ms vs training's 9 ms ICI dwell window, with quantified slice-isolation cost (12.5% training throughput).
- **Key concern:** uniqueness gain via specific ICI-ring contention numbers.

### cloud-3438 — DROP (200B FP16 cannot fit on 8xA100)
- **Verdict:** archived (see Archives section).

### cloud-3431 — NEEDS_FIX (LLM tone, trivial math)
- **Verdict:** rewritten
- **Change:** replaced descriptive "what does each parallelism do" prose with concrete sizing math — 70B Adam state = 1120 GB; per-GPU budget 64 GB; min(TP*PP) = 18 → choose TP=8, PP=4. Each axis maps to a different memory-component reduction.
- **Key concern:** Q now has a definite numerical answer.

### cloud-2711 — NEEDS_FIX (topic mismatch: MoE not DP)
- **Verdict:** rewritten
- **Change:** kept topic=data-parallelism by reframing as "what does outer DP add to an already expert-parallel MoE", with concrete BF16 AllReduce sizing on an IB NDR ring.
- **Key concern:** scenario stays in the DP cell while leveraging the chain context (cloud-chain-520).

### cloud-0712 — DROP (math error: 280 vs 28 ms)
- **Verdict:** rewritten
- **Change:** restated NVLink as 450 GB/s unidirectional, IB NDR as 25 GB/s unidirectional; redid every throughput row with the corrected divisors; explicitly derived the 1.75 GB per-node shard for hierarchical inter-node AllReduce; updated math_status to CORRECTED with 2026-04-25 note.
- **Key concern:** the cliff-at-GPU-9 narrative is preserved with consistent numbers.

### cloud-3448 — NEEDS_FIX (generic boilerplate)
- **Verdict:** rewritten
- **Change:** replaced "How would you diagnose..." with a specific decomposed tail (150 ms HBM contention + 50 ms launch serialization = 240 ms p99 tail) and asked which of MIG vs MPS is the cleaner fix.
- **Key concern:** the candidate must know that MIG hard-partitions HBM bandwidth while MPS only multiplexes contexts.

### cloud-0774 — NEEDS_FIX (topic mismatch: single-GPU not 3D)
- **Verdict:** rewritten
- **Change:** replaced single-GPU 7B fine-tune OOM walkthrough with 70B Adam-trained model on a 3D parallel layout; computes min(TP*PP)=18→32 and explains which axis shrinks which Adam component.
- **Key concern:** content is now genuinely 3d-parallelism.

### cloud-3440 — NEEDS_FIX (generic boilerplate)
- **Verdict:** rewritten
- **Change:** sharpened to a concrete PowerSGD r=8 vs FP8 gradients tradeoff on 32 MI300X over 200 Gbps RoCE; computed AllReduce time and convergence overhead for each.
- **Key concern:** common_mistake explicitly flags the trivial-bandwidth-division pattern banned by PARALLELISM_RULES.

### cloud-2215 — NEEDS_FIX (topic mismatch: scheduling not DP)
- **Verdict:** rewritten
- **Change:** replaced queueing-theory cluster-scheduler scenario with a DP replica-count optimization: T(N) = compute/N + 2.08*(N-1)/N, with the asymptotic AllReduce saturation as the binding insight.
- **Key concern:** topic stays data-parallelism; the optimum-N analysis exercises strong-scaling vs AllReduce saturation.

### cloud-4087 — NEEDS_FIX (canonical Megatron)
- **Verdict:** rewritten
- **Change:** added asymmetric fabric (NVLink pair + PCIe pair) so the column-vs-row choice now turns on which collective traverses the slow link the fewest times; included specific bus-traffic computations.
- **Key concern:** uniqueness gained via the asymmetric topology constraint.

### cloud-4047 — DROP (math error: 2*(N-1)/N applied to AllGather/ReduceScatter)
- **Verdict:** rewritten
- **Change:** corrected per-collective traffic — AllGather and ReduceScatter each move (N-1)/N, not 2*(N-1)/N; updated total per-iteration to 3 * (N-1)/N * size = 177.2 GB rather than the doubly-counted 354 GB; common_mistake now explicitly explains why AllReduce = ReduceScatter + AllGather.
- **Key concern:** the comm-volume comparison vs flat AllReduce is now ~2.85x rather than the wrongly-stated 6x.

### cloud-3930 — NEEDS_FIX (missing 2x AllReduce factor)
- **Verdict:** rewritten (napkin_math only)
- **Change:** added the missing 2*(N-1)/N factor for the inter-node 32-node AllReduce step; corrected from 13 ms to ~26 ms; restated total hierarchical AllReduce as ~26 ms, still feasible against 200 ms backward.
- **Key concern:** the "26 ms vs 104 ms flat" win still holds but the math is now self-consistent.

### cloud-1751 — NEEDS_FIX (topic warn: DP-only focus)
- **Verdict:** rewritten
- **Change:** narrowed Q from "all 192 process groups" to "the 64 DP groups specifically" and walked through which IB links each DP group's ring traverses; added concrete shard-size and AllReduce-time computation.
- **Key concern:** topic is now strictly DP-centric.

### cloud-3962 — NEEDS_FIX (math: unexplained 100 ms overlap window)
- **Verdict:** rewritten (napkin_math only)
- **Change:** explained the 100 ms cap as DDP's reverse-layer bucket fill rule — the first ~100 ms of backward computes activation gradients that produce buckets, AllReduce can only start once a bucket is full.
- **Key concern:** the 100 ms figure now follows from a stated mechanism.

### cloud-1578 — NEEDS_FIX (math: 71.6 GB barely fits 80 GB)
- **Verdict:** rewritten (napkin_math only)
- **Change:** sharpened the OOM by raising token skew to 95% (498k tokens on GPU 2) and adding the 8-15 GB PyTorch workspace overhead — combined with weights, the spike now definitively exceeds 80 GB.
- **Key concern:** the OOM claim is now mathematically forced, not borderline.

### cloud-1086 — NEEDS_FIX (300M model on 1024 GPUs unrealistic)
- **Verdict:** rewritten
- **Change:** scaled to 7B / 16 nodes / 128 GPUs; updated all numbers (FP16 14 GB gradient, 26.25 GB inter-node bus traffic) and showed that the IB upgrade alone is *not* sufficient — BF16 + bucketed overlap is also needed.
- **Key concern:** scale is now realistic; the reasoning chain is tighter (upgrade necessary but not sufficient).

### cloud-1059 — NEEDS_FIX (uniqueness duplicate of cloud-1698)
- **Verdict:** rewritten
- **Change:** pivoted from H2D/compute/D2H pipelining to NCCL AllReduce + backward overlap on a 4-GPU T4 box without NVLink (PCIe ring); added the diminishing-returns analysis on bucket size.
- **Key concern:** no longer duplicates cloud-1698's compute-pipeline framing.

### cloud-3054 — NEEDS_FIX (confusing memory math)
- **Verdict:** rewritten (napkin_math only)
- **Change:** decomposed memory cleanly — 8B params/stage × 16 bytes Adam = 128 GB raw, sharded by ZeRO/FSDP to ~24 GB; activation budget = 56 GB; per-micro-batch activation 3.5 GB; GPipe peak = 56 GB (just over) vs PipeDream-Flush = 14 GB (clear headroom).
- **Key concern:** the OOM-vs-fit comparison now traces from clear arithmetic.

### cloud-3931 — NEEDS_FIX (math: payload doesn't match dimensions)
- **Verdict:** rewritten (napkin_math only)
- **Change:** introduced batch=4 to make the 134 MB activation tensor traceable to stated dimensions; updated NVLink to 450 GB/s unidirectional and recomputed total TP comm at ~100 ms / step.
- **Key concern:** payload now derives from explicit dimensions.

## Archives (16)

Each archived item sets `status: archived` and adds a `deletion_reason` field
documenting why a rewrite was not viable. Archive rationale is item-specific
and quoted in each YAML.

1. **cloud-3438** — 200B FP16 model (400 GB) cannot fit + train on 8xA100 (640 GB). Premise broken.
2. **cloud-3059** — Question about H100 TFLOPS spec is disconnected from a pipeline-load-balancing scenario.
3. **cloud-0769** — Topic 3d-parallelism but content is single-replica loss-spike debugging; better placed elsewhere.
4. **cloud-3441** — Generic LLM template + duplicates rewritten cloud-3440 content.
5. **cloud-3444** — Generic LLM template + invents nonexistent MI300X partitioning capability.
6. **cloud-3436** — Math uses 2*G instead of 1.75*G; cell-fit warn; LLM template; trivial bandwidth division.
7. **cloud-2357** — Topic data-parallelism but scenario is exclusively MoE NVSwitch routing skew.
8. **cloud-2712** — Topic data-parallelism but content is exclusively expert-parallel all-to-all dispatch.
9. **cloud-0770** — Topic 3d-parallelism but scenario is single-host dataloader deadlock.
10. **cloud-1845** — Topic data-parallelism but scenario is single-GPU LSTM serialization.
11. **cloud-3934** — Claims 175B backward on 512 A100s takes 80 ms (real-world ≈ 600-1200 ms).
12. **cloud-3418** — FLOP-count error (missing 2 FLOPs/MAC factor) inverts the realistic_solution claim.
13. **cloud-3430** — Uses HBM3 bandwidth (5.3 TB/s) for inter-device communication, off by 1-2 orders of magnitude.
14. **cloud-3662** — Claims TP AllReduces 512 MB weight shards (TP AllReduces ~16 KB activations, not weights). Off by ~30,000x.
15. **cloud-3935** — Direct duplicate of cloud-3758.
16. **cloud-3422** — Uses flat NVLink 900 GB/s for a 128-GPU ring (NVLink only spans 8 GPUs intra-node).

## Validation contract

All 31 rewritten YAMLs preserve the original `track`, `level`, `zone`, `topic`,
`competency_area`, `bloom_level`, and `id`. None added `visual` blocks. None
promoted `status: draft` to `status: published`. Each archived YAML carries a
`deletion_reason` explaining why a rewrite was not viable.

PARALLELISM_RULES applied throughout: trivial `payload / bandwidth` divisions
were replaced with multi-step ring AllReduce arguments (with the 2*(N-1)/N
factor and unidirectional NVLink/IB bandwidths), concrete topology specs
(NVLink 450 GB/s unidirectional, IB NDR 25 GB/s unidirectional, RoCE v2 22
GB/s, PCIe Gen3 12 GB/s, PCIe Gen4 25 GB/s), and explicit synchronization
costs (bubble fraction (k-1)/(m+k-1), exposed AllReduce, expert-imbalance
memory pressure). The forbidden "Llama-70B KV cache on 4×H100" framing was
avoided in all rewrites.
