# Float Exposition Audit — `model_serving.qmd` (vol1)

Graded against FLOAT_EXPOSITION_STANDARD.md. Caption, fig-alt, in-figure labels, and code comments do not count toward the prose's job; only running body prose is evaluated.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| Equation | 🔴 | 18 | 14 | 4 | 0 |
| Figure | 🟠 | 7 | 6 | 1 | 0 |
| Listing | 🟡 | 2 | 1 | 1 | 0 |
| Table | 🟠 | 21 | 14 | 7 | 0 |
| **Total** | | **48** | **35** | **13** | **0** |

---

## Passing floats (✅)

**Equations:** `eq-resolution-throughput`, `eq-system-efficiency`, `eq-littles-law`, `eq-batching-tax`, `eq-mm1-wait`, `eq-p99-latency`, `eq-batching-latency`, `eq-avg-wait`, `eq-batch-throughput`, `eq-compute-time`, `eq-latency-constrained-batch`, `eq-p99-batch-latency`, `eq-precision-throughput`, `eq-token-generation-time`

**Figures:** `fig-tail-latency-explosion`, `fig-intelligence-deflation`, `fig-serving-inference-pipeline`, `fig-server-anatomy`, `fig-serving-pipeline-timing`, `fig-throughput-latency-knee`

**Listings:** `lst-resnet-postprocessing`

**Tables:** `tbl-resnet-serving-spectrum`, `tbl-model-serving-resnet50-latency-budget`, `tbl-model-serving-dlrm-latency-budget`, `tbl-serving-tax`, `tbl-resolution-bottleneck`, `tbl-utilization-latency`, `tbl-model-serving-resnet-coldstart`, `tbl-model-serving-batching-throughput-tradeoff`, `tbl-batch-variability`, `tbl-batching-throughput`, `tbl-pareto-batching`, `tbl-model-serving-runtime-comparison`, `tbl-model-serving-precision-tradeoffs`

---

## Findings (⚠️ only — 13 floats)

---

### F01 — `eq-batch-distribution` (🔴 equation) — def L2992

**Ref sentence (L2991):** "@Eq-batch-distribution formalizes this relationship:"

**What is missing:** Lead-out/payoff. The citation paragraph names the distribution and writes the equation but never states the consequence or implication for serving engineers. The prose immediately after the float pivots directly to citing a table ("@Tbl-batch-variability quantifies this variability"), bypassing the interpret move entirely. The takeaway lives only in the table caption.

**Where the takeaway currently lives:** Caption of `tbl-batch-variability` and the payoff paragraph after `tbl-batch-variability`.

**Rule:** 🔴 Equation — prose must state what the equation expresses in words, the meaning of key terms, and the consequence it implies. "Formalizes this relationship" with no stated implication fails the interpret move.

**Suggested rewrite (replace the citation sentence and its immediate follow-on):**

> The number of requests collected during window $T_{\text{window}}$ follows a Poisson distribution with mean $\lambda T_{\text{window}}$. @Eq-batch-distribution captures this formally, but the engineering consequence is more important than the formula: because Poisson variance equals the mean, batch sizes fluctuate wildly at moderate traffic. At 200 QPS with a 10 ms window, expected batch size is two, yet roughly 14 percent of windows collect zero requests, wasting GPU cycles, while others spike to four or more. @Tbl-batch-variability quantifies this variability across traffic levels with a fixed 10 ms window.

---

### F02 — `eq-poisson-batch` (🔴 equation) — def L3522

**Ref sentence (L3521):** "@Eq-poisson-batch expresses the expected batch size for Poisson arrivals with rate $\lambda$ and batching window $T_{\text{window}}$:"

**What is missing:** Lead-out/payoff. The follow-on prose (L3526) is a footnote that repeats the same content from the footnote body verbatim. The body paragraph after the equation restates the variance-equals-mean property and a numeric example, which is adequate orientation, but it never states the engineering consequence: what the serving engineer should do with this equation when tuning. The interpretation stops at description and does not cross to implication.

**Where the takeaway currently lives:** Partially in L3526 (numeric example) but the actionable consequence (use this as an input to window sizing) is deferred to `eq-optimal-window` prose.

**Rule:** 🔴 Equation — prose must state the implication or regime the equation governs.

**Suggested rewrite (add one sentence after the existing L3526 paragraph):**

> This means the window is the single tuning knob that controls expected batch size: doubling the window doubles the expected batch, but also doubles average wait time. The heuristic in @Eq-optimal-window uses this linear relationship to derive the window that balances both costs.

---

### F03 — `eq-optimal-window` (🔴 equation) — def L3529

**Ref sentence (L3528):** "A useful heuristic for the batching window balances waiting cost against throughput benefit. @Eq-optimal-window expresses one such rule:"

**What is missing:** Lead-out on the second citation (L3623 inside the table caption). The first citation (L3528) has solid prose including symbol definitions, economic analogy, and the counterintuitive traffic observation, so that cite is fine. However, the second reference to this equation (at L3623) appears only inside the table caption. No body prose in the second context states what the equation delivers. This is a caption-only takeaway for the second reference context.

**Where the takeaway currently lives:** In the original citation context (L3528) the prose is complete. For the second reference, the caption carries the interpretation.

**Rule:** 🟠 Table (for that reference) — the second cite context is `tbl-traffic-adaptive`, where the lead-out for what the table demonstrates lives only in the caption.

**Suggested rewrite (add body prose after `tbl-traffic-adaptive`, currently L3631 onward):**

> The table confirms the counterintuitive consequence of @eq-optimal-window: higher traffic shrinks the optimal window because the square-root term falls faster than the arrival-rate term rises, while the batch size grows with $\lambda T_{\text{window}}$. A system serving 5,000 QPS achieves larger batches with a shorter window than one serving 100 QPS uses with a longer window, delivering more throughput with less latency budget consumed.

---

### F04 — `fig-kv-cache-growth` (🟠 figure) — def L4605

**Ref sentence (L4603):** "@fig-kv-cache-growth shows why."

**What is missing:** The prose at L4603 is partially adequate (it names the mechanism: KV-cache memory grows linearly with context and batch) but the lead-out payoff is thin. The sentence states "the cache grows linearly with context length and batch size until long contexts push even an H100 into its out-of-memory zone" but then pivots immediately to the Llama 3 case study without naming the concrete engineering trade-off the figure demonstrates. The interpret move stops at description ("shows why") rather than delivering the figure's story: the batch-size vs. context-length trade-off and its quantitative regime.

**Where the takeaway currently lives:** The payoff prose at L4702 finally states the trade-off ("to support longer contexts, we must reduce batch size, which in turn kills throughput efficiency") but that is 97 lines after the figure and separated by a case-study section boundary.

**Rule:** 🟠 Figure — prose must state what the figure demonstrates and why it matters, not merely describe what it shows.

**Suggested rewrite (extend L4603):**

> The dominant cost in serving a large language model is not compute but KV-cache memory, and @fig-kv-cache-growth reveals the mechanism. Cache memory grows linearly with both context length and batch size, so the two levers that engineers control to improve throughput (longer context for capability, larger batch for utilization) pull directly against the GPU's memory wall. At batch size 32, the 70-billion-parameter-scale cache exhausts an H100's VRAM at just 8k context, forcing a hard choice between context window (capability) and batch size (throughput). The 8-billion-parameter Llama 3 profile analyzed below obeys the same physics with more headroom, making it a workload an engineer can fit on one GPU and reason about end to end.

---

### F05 — `lst-adaptive-batching` (🟡 listing) — def L3318

**Ref sentence (L3312):** "@Lst-adaptive-batching demonstrates how adaptive strategies adjust the window based on queue depth."

**What is missing:** The citation is a plain announcer with no mechanism or design choice called out. The reader does not know what to look at in the code before reading it: the key observation (that the window shrinks when the queue is long, not grows) is the counterintuitive mechanism, and the prose does not name it. The payoff paragraph (L3314) names the latency improvement numbers but still does not identify the mechanism or the specific design choice embodied in the listing.

**Where the takeaway currently lives:** Embedded in the code itself (via comments and logic) and partially in L3314 (the numbers), but the mechanism "window shrinks under high load because the queue is already full enough to form a large batch without waiting" is never stated in body prose.

**Rule:** 🟡 Listing — prose must deliver the mechanism the code embodies and what the reader should notice.

**Suggested rewrite (replace L3312):**

> Fixed batching windows waste latency budget during high traffic when large batches form quickly. @Lst-adaptive-batching demonstrates the core mechanism: when queue depth is large, the window shortens (the queue itself provides the batch) while when queue depth is small, the window extends (arrivals need more time to accumulate). The key design choice is that window length and queue depth are inversely coupled, so the adaptive controller trades latency budget only when the trade is actually needed.

---

### F06 — `tbl-serving-spectrum` (🟠 table) — def L911

**Ref sentence (L899):** "@Tbl-serving-spectrum summarizes how these deployment contexts shape serving system design."

**What is missing:** Lead-out/interpret move. The cite is a bare summary announcer. No body prose before or after the table states the load-bearing contrast the table encodes: specifically, that the three contexts do not merely differ quantitatively but represent qualitatively different engineering problems where the failure mode, monitoring capability, and optimization target are categorically distinct. The insight that TinyML cannot even retry on failure while cloud can failover is the "so what," but it lives only in the table cells.

**Where the takeaway currently lives:** Distributed across the table cells and the caption. The post-table prose (L913) pivots immediately to a ResNet example without delivering the takeaway.

**Rule:** 🟠 Table — prose must deliver the conclusion the table encodes, not merely announce it.

**Suggested rewrite (replace L899 and add a lead-out sentence after the table):**

> The spectrum in @Tbl-serving-spectrum exposes the key discontinuity: the three deployment contexts do not just differ by scale but by failure mode and recovery capability. Cloud systems can retry and failover; mobile systems degrade gracefully; TinyML systems silently reset or produce wrong outputs with no telemetry to detect the failure. This shapes every optimization decision downstream.

> *(table follows)*

> The monitoring row encodes this discontinuity most sharply: an engineer debugging a cloud serving regression has full telemetry; a TinyML engineer debugging the same class of bug has only a heartbeat signal. To make these architectural differences concrete, consider how a single model must adapt to each deployment context.

---

### F07 — `tbl-model-serving-resnet50-latency-budget` (🟠 table) — def L1665

**Ref sentence (L1653):** "@Tbl-model-serving-resnet50-latency-budget breaks a typical ResNet-50 serving request down per phase:"

**What is missing:** The citation is inside a callout and the cite sentence is a bare "breaks down" announcer. The key insight (preprocessing consumes more than inference; with TensorRT, preprocessing would dominate at a specific percentage) lives in the payoff sentence at L1667, but that sentence is *also inside the callout*, making it a callout-scoped payoff rather than body prose. The standard requires body prose to carry the takeaway. The body prose after the callout (L1671) does reference the ResNet bottleneck but does so indirectly without citing the table's specific numbers.

**Where the takeaway currently lives:** L1667 inside the callout (not body prose by the standard's definition, since it is a notebook callout interior). The payoff sentence is effectively caption-adjacent.

**Note:** The cite context is inside a `.callout-notebook`. This evaluator treats callout-internal prose as non-body prose per the standard ("only running body prose counts"). The surrounding section body does not restate the table's key finding.

**Rule:** 🟠 Table — body prose (outside the callout) must deliver the takeaway the table encodes.

**Suggested rewrite (add to the body paragraph at L1671, which currently begins "The ResNet example represents compute-bound inference..."):**

> The ResNet example in @Tbl-model-serving-resnet50-latency-budget reveals the central irony of optimized serving: preprocessing consumes the majority of total request latency despite model inference being the computationally intensive phase, and TensorRT optimization sharpens this imbalance further. The ResNet example represents compute-bound inference where the forward-pass arithmetic dominates the latency budget...

---

### F08 — `tbl-model-serving-resnet-coldstart` (🟠 table) — def L2645

**Ref sentence (L2632):** "@Tbl-model-serving-resnet-coldstart traces the per-phase duration of a ResNet-50 cold start:"

**What is missing:** Same pattern as F07. The cite and table are inside a `.callout-notebook`. The body prose after the callout (L2651 onward) discusses CUDA context and MPS in detail but never states the load-bearing takeaway the table encodes: that TensorRT compilation is the single dominant phase (roughly two orders of magnitude longer than weight loading from SSD) and that pre-compiling the engine essentially eliminates that phase. This conclusion lives only in the callout's "Systems insight" box, not in body prose.

**Where the takeaway currently lives:** Inside the callout (L2647 "Systems insight" sentence).

**Rule:** 🟠 Table — body prose must deliver the conclusion the table encodes.

**Suggested rewrite (add one sentence at the start of the body section after the callout, L2651):**

> The cold start timeline in @Tbl-model-serving-resnet-coldstart shows that TensorRT compilation dominates first-deploy cost by two orders of magnitude compared to weight loading from SSD. Precompiling the engine and caching it reduces cold start from tens of seconds to under two seconds. The CUDA context is the first cost in the cold start timeline...

---

### F09 — `tbl-practical-batching-config` (🟠 table) — def L3459

**Ref sentence (L3445):** "The calculation turns the SLO and arrival-rate assumptions into two deployable knobs: the batching window and maximum batch size. @Tbl-practical-batching-config summarizes the resulting configuration and the predicted operating point."

**What is missing:** The cite introduces what the table contains but delivers no interpret move. The payoff paragraph (L3463) jumps immediately to autoregressive models, completely skipping any lead-out on what the table demonstrates. The load-bearing conclusion from the table is that the predicted p99 latency stays within the SLO with the chosen configuration, validating the backward-calculation method. That result is implicit in the table rows but is never stated in prose.

**Where the takeaway currently lives:** Implicit in the table cells (the predicted p99 row), not stated in body prose.

**Rule:** 🟠 Table — prose must state the takeaway, specifically what the table confirms or demonstrates.

**Suggested rewrite (add a sentence between L3445 and the table, or immediately after the table before L3463):**

> @Tbl-practical-batching-config summarizes the resulting configuration and the predicted operating point. The key result is in the final two rows: backward calculation from a 50 ms SLO produces a predicted p99 latency that lands comfortably within the SLO, confirming that the analytic framework closes. The next section examines what happens when fixed-length output assumptions break down.

---

### F10 — `tbl-model-serving-multicamera-timeline` (🟠 table) — def L3652

**Ref sentence (L3635):** "@Tbl-model-serving-multicamera-timeline traces the per-event timeline for one synchronized frame set:"

**What is missing:** The cite is inside a `.callout-notebook`. The body prose after the callout (L3665) states "unlike Poisson traffic... streaming traffic requires synchronization policies that handle sensor jitter while meeting hard deadlines," which is a correct observation but does not draw the specific conclusion the timeline table encodes: that jitter tolerance (12 ms of 33 ms budget) is the governing constraint and that a single late camera forces the system to either wait (consuming the jitter budget) or substitute the prior frame (accuracy cost). The takeaway that the synchronization budget is a first-class design parameter consuming 36 percent of the hard deadline lives only in the callout bullet list.

**Where the takeaway currently lives:** Inside the callout "Key constraints" bullets.

**Rule:** 🟠 Table — body prose must state the load-bearing insight.

**Suggested rewrite (replace L3665):**

> The timeline in @Tbl-model-serving-multicamera-timeline makes the budget arithmetic concrete: jitter tolerance consumes 12 of the 33 ms hard deadline (36 percent) before inference begins. Unlike Poisson traffic where dynamic batching optimizes throughput, streaming traffic requires synchronization policies that trade this budget explicitly against accuracy: wait for the late frame (consuming jitter budget) or substitute the prior frame (introducing staleness). The hard deadline leaves no room for queuing delay.

---

### F11 — `tbl-model-serving-mobile-pipeline-breakdown` (🟠 table) — def L3767

**Ref sentence (L3751):** "@Tbl-model-serving-mobile-pipeline-breakdown decomposes per-phase latency and energy for a single-user mobile vision inference:"

**What is missing:** Same callout-interior pattern. The cite is inside a `.callout-notebook`. The body prose after the callout (L3786) discusses three constraints but never draws the conclusion the table encodes: that JPEG decode on the CPU consumes the largest share of energy even though the NPU inference stage carries the compute, which means optimization pressure lands on preprocessing, not the model. The body prose lists energy, thermal throttling, and memory constraints as separate items but does not connect them to the table's specific cross-cutting result.

**Where the takeaway currently lives:** In the caption and the callout's "Systems insight" box.

**Rule:** 🟠 Table — body prose must carry the takeaway.

**Suggested rewrite (add one sentence at L3786 before "Unlike cloud serving..."):**

> The mobile pipeline in @Tbl-model-serving-mobile-pipeline-breakdown reveals that JPEG decode on the CPU dominates energy even though NPU inference carries the arithmetic work. This inversion means that mobile energy optimization targets preprocessing, not the model — the same structural insight as the cloud ResNet latency budget, but with energy as the binding metric rather than time. Unlike cloud serving where cost dominates, mobile serving faces three related constraints that shape optimization strategy.

---

### F12 — `tbl-traffic-patterns-summary` (🟠 table) — def L3807

**Ref sentence (L3798):** "@Tbl-traffic-patterns-summary maps the four MLPerf scenarios to their deployment contexts and optimal batching strategies, providing a decision framework for serving system design."

**What is missing:** The cite names what the table provides but the interpret move ("the decision framework") is not stated. The payoff paragraph (L3809) restates which scenario maps to which context (cloud/synchronized/batch) using exactly the language in the table cells, which is narrating the table rather than interpreting it. The load-bearing conclusion is which scenarios are latency-constrained vs. throughput-constrained and how that drives the selection of strategy, but that framing is absent.

**Where the takeaway currently lives:** Distributed across the table's Focus column and the caption.

**Rule:** 🟠 Table — prose must deliver the conclusion, not restate the cells.

**Suggested rewrite (replace L3809):**

> The critical distinction in @Tbl-traffic-patterns-summary is the Focus column: Server and MultiStream scenarios are governed by latency constraints (window tuning against an SLO, synchronization budget against a hard deadline), while Offline is governed purely by throughput. SingleStream sits apart from all three: with no batching possible, optimization shifts entirely to preprocessing and power efficiency rather than scheduling policy. Matching the batching strategy to this focus column is the first decision in serving system design.

---

### F13 — `tbl-model-serving-resnet-cloud-cost` (🟠 table) — def L4537

**Ref sentence (L4529):** "@Tbl-model-serving-resnet-cloud-cost compares hourly cost, throughput, and per-million-image cost for serving ResNet-50 on AWS (US-East, on-demand pricing in 2026):"

**What is missing:** The cite is inside a `.callout-notebook`. The body prose after the callout (L4545) begins "In the worked AWS example above, GPU instances cost more per hour but deliver much higher parallel throughput" which is a partial lead-out, but it reads as a restatement of the obvious rather than delivering the table's specific conclusion. The load-bearing result is that the T4 is the cost-optimal choice at moderate traffic while the V100 requires specific high-sustained-traffic conditions to justify its price multiple. The specific crossover logic is in the callout's "Systems insight" box, not in body prose.

**Where the takeaway currently lives:** Inside the callout "Systems insight" bullet.

**Rule:** 🟠 Table — body prose must state the decision the table drives.

**Suggested rewrite (replace L4545 first sentence):**

> The unit-economics comparison in @Tbl-model-serving-resnet-cloud-cost shows that the T4 achieves the lowest cost per inference despite a higher hourly rate than the CPU instance, because GPU throughput more than compensates for the rate premium. The V100 crosses into cost-optimal territory only at sustained high traffic where its throughput multiplier justifies the hourly premium over the T4. The crossover point depends on model characteristics and latency requirements.

---

## Notes

- Two dangling references exist (`@tbl-energy-movement-cost` at L4815 and L4816) with no matching definition in this chapter. These are flagged by the scanner as orphan refs. This is a cross-reference integrity issue, not a float exposition issue; it is out of scope for this audit.
- All 18 equations have symbol definitions in prose. The 14 passing equations are well-treated, with worked numeric instances for the most critical (batching-tax, MM1, p99-latency, token-generation-time).
- The dominant pattern for findings is callout-interior prose counted as non-body, creating a systematic gap where notebook callouts carry the takeaway but the surrounding body prose does not restate the key conclusion. Seven of the 13 findings have this structure.
