# Float-explanation worklist — model_serving.qmd (vol1)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 7 | 7 | 0 | 0 |
| table | 21 | 21 | 0 | 0 |
| listing | 2 | 2 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 18 | 18 | 0 | 0 |
| **total** | **48** | **48** | **0** | **0** |

No under-explained floats found; all references explained in-neighborhood.

---

## Notes on scanner discrepancies

The scanner reported "(none found)" for captions on the majority of tables. All of these tables use Quarto's pipe-table footer caption syntax (`: **Bold Title**: explanation {#tbl-id}`), which the scanner does not detect. Every table was confirmed to have a proper caption by direct source inspection. Caption quality was factored into the neighborhood check for each table.

## Floats confirmed ✅ (all 48)

All floats were verified by reading the full local neighborhood: ref sentence, prev paragraph, next paragraph, caption, and payoff paragraph after the float definition. Explanation is present and substantive for each.

**Equations (18 — all ✅)**

- `eq-resolution-throughput` (L1864): prev ¶ establishes compute-bound vs. memory-bound distinction; ref sentence introduces the formula; payoff ¶ gives concrete quadratic slowdown numbers and the implication (serving costs are latency + activation pressure, not bandwidth).
- `eq-system-efficiency` (L2074): ref sentence defines and motivates the metric; payoff ¶ gives ResNet-50 50% efficiency example with pipelining fix; second ref (L4418) applies it to profiler interpretation.
- `eq-littles-law` (L2174): ref sentence explains all three quantities and the "stable system" condition; payoff ¶ gives the worked capacity-planning translation (QPS + SLO → concurrent request slots).
- `eq-batching-tax` (L2315): prev ¶ defines the two latency components; ref sentence frames the Pareto frontier; payoff ¶ gives concrete 500 QPS numbers and the engineer's use (batch size as regulator of economic efficiency).
- `eq-mm1-wait` (L2322): defined inline in a paragraph that explains service rate and utilization; next sentence (L2327) gives the nonlinear divergence implication; refs at L5032 and L5087 apply it with concrete utilization numbers and system-level speedup analysis.
- `eq-p99-latency` (L2356): payoff ¶ (L2358) explains why 70% utilization yields ≈15× service time at p99 and why average latency hides tail experience; ref at L2470 applies it in a worked capacity planning example.
- `eq-batching-latency` (L2943): ref sentence decomposes wait + compute with inline variable definitions; batching window bounds the wait time constraint; payoff transitions to quantitative analysis.
- `eq-avg-wait` (L2949): ref sentence derives the half-window result from uniform arrivals; payoff ¶ (L2987) gives concrete budget impact (average wait as fraction of SLO).
- `eq-batch-distribution` (L2992): ref sentence explains Poisson distribution with mean λT_window; next sentence and Tbl-batch-variability quantify the variance implications.
- `eq-batch-throughput` (L3008): payoff paragraph (= "this ¶" at L3010) explains λ, μ(B), sub-linear service time, and traffic-regime behavior; next ¶ (L3012) gives ResNet-50 formula with numbers.
- `eq-compute-time` (L3117): ref sentence frames the iron-law connection in a callout; explains fixed overhead amortization and queue wait as separate costs.
- `eq-latency-constrained-batch` (L3133): ref sentence explains the optimization problem (max batch satisfying SLO); payoff ¶ immediately gives a 50 ms SLO worked example.
- `eq-p99-batch-latency` (L3289): ref sentence explains why batch variability causes SLO violations (worst-case wait + worst-case batch size); inline worked numbers follow; payoff ¶ gives p99/mean ratio.
- `eq-poisson-batch` (L3522): ref sentence explains Poisson arrivals for cloud APIs and derives expected batch size; footnote payoff explains variance = mean and the empty-window consequence.
- `eq-optimal-window` (L3529): ref sentence frames the heuristic, explains square-root scaling law, names the counterintuitive result (shorter window as traffic rises); Tbl-traffic-adaptive demonstrates across four traffic levels; second ref (L3623) is in the table caption.
- `eq-precision-throughput` (L4186): ref sentence explains memory-bandwidth-bound operations and proportional throughput gain from reduced data movement; payoff ¶ gives 2.5–3.5× practical achieved speedup and Tensor Core alignment constraints.
- `eq-quant-error` (L4279): ref sentence explains layer sensitivity proportionality with inline parameter definitions; explains observed patterns for first/middle/final layers; payoff ¶ adds calibration dataset constraint.
- `eq-token-generation-time` (L4820): second ref (L4819) explains the memory wall for GenAI (arithmetic intensity ≈ 1 FLOP/byte, bandwidth-limited); payoff ¶ (L4822) gives concrete A100 token-generation time and the engineering implication (faster memory changes the bound, not more compute cores).

**Figures (7 — all ✅)**

- `fig-tail-latency-explosion` (L125): 5 substantive refs; payoff explains 70% threshold, why production systems run at 40–60%, and the queueing-knee collapse mechanism.
- `fig-intelligence-deflation` (L233): ref (L231) explains log-scale price trajectory and the each-order-of-magnitude-drop framing; refs at L4468 and L5327 place it in economic context.
- `fig-serving-inference-pipeline` (L411): ref sentence (L409) walks through all stages left-to-right, names the bottleneck ambiguity, and defers quantification to @sec-model-serving-latency-budget-ef40.
- `fig-server-anatomy` (L1074): ref sentence walks through six stages; payoff ¶ (L1368) explains concurrency management, request transformation, and tensor memory layout in depth.
- `fig-serving-pipeline-timing` (L2029): ref sentence explains both diagrams (serial idle gaps vs. pipelined overlap); caption explains throughput impact; payoff ¶ (L2069) explains the async I/O mechanism.
- `fig-throughput-latency-knee` (L2841): ref sentence (L2924) names the optimal operating point; explains the batching-beyond-knee cost; notes representative values.
- `fig-kv-cache-growth` (L4605): ref sentence explains dominant cost (KV-cache memory), linear growth with context length and batch size; payoff ¶ (L4702) states the hard throughput/context trade-off.

**Listings (2 — all ✅)**

- `lst-resnet-postprocessing` (L2086): ref sentence (L2121) explains logits→probabilities→top-k→API-response pipeline, timing annotations, and 0.1 ms total cost.
- `lst-adaptive-batching` (L3318): prev sentence frames the why (fixed window wastes budget at high traffic); ref sentence names the adaptive mechanism; next ¶ (L3314) gives 27% latency reduction with concrete QPS numbers; caption also states the reduction.

**Tables (21 — all ✅)**

- `tbl-serving-spectrum` (L911): prev ¶ (L897) details all three deployment contexts; ref sentence says it summarizes the design impact; caption explains the physical-wall origin of each constraint set.
- `tbl-resnet-serving-spectrum` (L1028): ref sentence (L1014) explains the three-tier contrast with specific model/hardware/precision differences.
- `tbl-model-serving-resnet50-latency-budget` (L1665): caption (L1665) states preprocessing and data transfer rival the forward pass; payoff ¶ (L1667) gives percentage with TensorRT counterfactual.
- `tbl-model-serving-dlrm-latency-budget` (L1729): ref sentence (L1719) contrasts CNN compute vs. embedding-table bandwidth; systems insight block (L1731) names the MLP fraction; payoff ¶ (L1735) generalizes the failure mode via Amdahl's Law.
- `tbl-serving-tax` (L1804): ref sentence (L1794) frames the "tax" concept; caption (L1804) explains 50% budget consumption for 5 ms service; second ref (L1855) applies it to the killer-microseconds analysis.
- `tbl-resolution-bottleneck` (L1978): preceding paragraph (L1903) establishes the quadratic physics; ref sentence introduces the data; caption explains the ridge-point result and the compute-bound implication.
- `tbl-utilization-latency` (L2339): prev sentence (L2327) gives the nonlinear divergence framing; ref sentence provides context; caption (L2339) explains the 3.3×/10× rule; payoff ¶ (L2341) discusses M/M/1 vs. M/D/1 practical implications.
- `tbl-model-serving-resnet-coldstart` (L2645): ref sentence traces per-phase durations; caption (L2645) states where dominant cost lives; payoff ¶ (L2651) explains CUDA context creation cost in depth.
- `tbl-model-serving-batching-throughput-tradeoff` (L2825): ref sentence (L2810) says it illustrates the throughput-latency trade-off; caption explains 6.4× throughput growth; systems insight block (L2833) quantifies latency with batching window; payoff (L2837) gives the latency-SLO trade-off decision.
- `tbl-batch-variability` (L3003): ref sentence (L2994) names the 10 ms fixed window and traffic levels; prev equation establishes the Poisson distribution; table data quantifies empty-window and burst probabilities.
- `tbl-batching-throughput` (L3109): ref sentence (L3012) explains it extends the pure-inference sweep by adding window wait; caption states throughput increase and latency more-than-doubling; second ref (L3111) connects to iron-law amortization; third ref (L3232) applies it in optimization comparison.
- `tbl-pareto-batching` (L3362): ref sentence (L3352) frames the Pareto frontier; second ref (L3366) explains the knee and the principled configuration approach; payoff ¶ names the T_max formula and p95-arrival-rate guidance.
- `tbl-practical-batching-config` (L3459): ref sentence (L3445) explains it turns SLO/arrival-rate into two deployable knobs (window + max batch size); table rows labeled with engineering roles.
- `tbl-traffic-adaptive` (L3623): ref sentence (L3530) names the counterintuitive result (shorter optimal window at higher traffic); caption explains it was computed from eq-optimal-window with specific assumptions; second ref is in caption.
- `tbl-model-serving-multicamera-timeline` (L3652): ref sentence (L3635) gives the six-camera 30-FPS scenario; caption explains 7 ms arrival spread and 12 ms jitter tolerance within 33 ms deadline; key-constraints bullets below the table name the timeout policy; payoff ¶ (L3665) gives the synchronization-policy implication.
- `tbl-model-serving-mobile-pipeline-breakdown` (L3767): ref sentence decomposes latency and energy per phase; caption explains JPEG decode dominates energy even though NPU carries compute; systems insight block (L3780) gives NPU utilization comparison; payoff ¶ (L3786) lists three mobile-specific constraints.
- `tbl-traffic-patterns-summary` (L3807): ref sentence (L3798) maps MLPerf scenarios to deployment contexts; caption repeats the mapping; payoff ¶ (L3809) identifies each scenario by name.
- `tbl-model-serving-runtime-comparison` (L4163): ref sentence (L4147) names the axes (latency, speedup, runtimes, V100, batch-1); caption (L4163) states the optimization-compatibility trade-off; systems insight block (L4169) quantifies INT8 speedup with cost; payoff ¶ (L4173) gives the choice criteria.
- `tbl-model-serving-precision-tradeoffs` (L4266): ref sentence (L4257) names all four dimensions (latency, memory, accuracy, Tensor Core util) and four precision paths; payoff leads into eq-quant-error for layer sensitivity.
- `tbl-optimization-impact` (L4447): ref sentence (L4437) frames it as a decision aid ("choose the technique whose target metric matches the measured bottleneck"); caption (L4447) explains the high-impact/high-cost vs. low-effort/specific-metric pattern.
- `tbl-model-serving-resnet-cloud-cost` (L4537): ref sentence (L4529) names the axes (hourly cost, throughput, cost per 1M images); caption (L4537) explains how higher hourly rate can yield lower cost-per-inference; systems insight block (L4539) identifies the T4 as lowest cost-per-inference.
