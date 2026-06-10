# Float Exposition Worklist — `fault_tolerance.qmd` (vol2)

Graded against the Float Exposition Standard. Caption, fig-alt, in-figure labels, code
comments, and callout interiors do not count toward the prose's job. Only running body
prose was evaluated.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| Equation | 🔴 strict | 11 | 6 | 5 | 0 |
| Figure | 🟠 high | 33 | 19 | 14 | 0 |
| Listing | 🟡 medium | 4 | 3 | 1 | 0 |
| Table | 🟠 high | 19 | 11 | 8 | 0 |
| **Total** | | **67** | **39** | **28** | **0** |

---

## Findings (⚠️ only — no 🛑 found)

---

### EQUATIONS

---

#### `eq-system-reliability-product` (🔴 strict) — def L156

**Verbatim ref sentence (L154):**
> "When multiple independent components operate in a system where any single failure causes system failure, @eq-system-reliability-product formalizes how system reliability becomes the product of individual component reliabilities:"

**Missing move:** The lead-in sets up the equation and the payoff (L158) bridges immediately to the simplified N-component form. No prose sentence states what this product rule *implies* — namely that reliability degrades multiplicatively and even modest per-component failure rates compound catastrophically at scale. The consequence lives implicitly in the chain of equations that follow, not in an explicit interpret-move sentence.

**Where the takeaway currently lives:** Implicit; the chain of simplifications (L158-164) carries it but no single prose sentence names the degradation consequence of the product rule.

**Rule-compliant rewrite (insert after the equation at L156, before L158):**

> The product structure has a decisive consequence: even components with high individual reliability (99.99 percent survival over one year) combine to produce a system whose reliability falls below any single component's rate. For two components each at 99 percent, system reliability drops to 98 percent; for 10,000, it approaches zero for any meaningful time horizon.

---

#### `eq-system-reliability-n-components` (🔴 strict) — def L160

**Verbatim ref sentence (L690):**
> "Correlated failures violate the independence assumption underlying @eq-system-reliability-n-components. When failures are correlated, the actual system reliability is lower than the formula predicts."

**Missing move:** This equation has no dedicated interpret-move prose at its definition site (L160). The only body-prose reference uses it to state what the equation *fails* to capture, not what it *delivers*. The equation simplifies from the product form to the $e^{-N\lambda t}$ form — the prose never states the insight this simplification reveals: that all N components can be collapsed into a single effective rate $N\lambda$, which is the foundation of the MTBF scaling argument.

**Where the takeaway currently lives:** Nowhere in body prose — the definition at L160 has no adjacent lead-out.

**Rule-compliant rewrite (insert after the equation at L160, before L162):**

> Combining N identical components collapses the product into a single exponential with rate $N\lambda$: the cluster behaves as one device whose failure rate scales linearly with the number of components. This linearity is the engine of the MTBF scaling argument that follows.

---

#### `eq-ft-total-cost` (🔴 strict) — def L825

**Verbatim ref sentence (L823):**
> "@Eq-ft-total-cost presents a simplified economic model for expected cost per training run:"

**Missing move:** The payoff at L827 names all symbols ("where $C_{\text{compute}}$ is the base compute cost…"), which satisfies the symbol-naming requirement. The equation's *implication* is never stated: that $C_{\text{ft}}$ is additive and appears alongside $E[N_{\text{failures}}] \times C_{\text{per-failure}}$, so the decision question is whether increasing $C_{\text{ft}}$ reduces the middle term by more than it costs. The payoff pivots immediately to the next equation without stating this marginal logic.

**Where the takeaway currently lives:** The marginal logic is stated only in the follow-on equation's payoff (L833: "investing in fault tolerance until the marginal cost…exceeds the marginal reduction"). It belongs adjacent to this equation as well.

**Rule-compliant rewrite (insert between L827 and L829, replacing or augmenting the bridge):**

> The model's operational reading is that the three terms compete: reducing $E[N_{\text{failures}}] \times C_{\text{per-failure}}$ through better fault tolerance increases $C_{\text{ft}}$, so the investment is only justified when the reduction in expected failure cost exceeds the added prevention cost. At small scale where failures are infrequent, $C_{\text{ft}}$ dominates; at large scale where failures are daily, the middle term dominates and fault tolerance investment yields positive return.

---

#### `eq-checkpoint-overhead` (🔴 strict) — def L2272

**Verbatim ref sentence (L2270):**
> "@Eq-checkpoint-overhead quantifies the wasted time due to checkpointing:"

**Missing move:** The "where" clause at L2274 names $T_{\text{pause}}$ but does not explain what the equation as a whole *says*. $f_{\text{ckpt}}$ is the fraction of training time wasted, and the equation's key structural insight — that both terms share $\tau_{\text{ckpt}}$ in the denominator, so longer intervals reduce overhead but increase expected rework — is never stated in prose. No worked numeric instance appears adjacent to the equation.

**Where the takeaway currently lives:** The financial consequence appears in L2278 ("severe…\$24,000 per day") but that paragraph does not interpret the equation's structural insight.

**Rule-compliant rewrite (insert after the "where" clause at L2274, before L2276):**

> Because both terms share $\tau_{\text{ckpt}}$ in the denominator, longer checkpoint intervals reduce $f_{\text{ckpt}}$ from the write term but increase expected rework (captured separately in the Young-Daly analysis). For a cluster writing a 2-minute checkpoint every hour, $f_{\text{ckpt}} \approx 2/60 \approx 3.3$ percent write overhead alone, before accounting for any pipeline stall beyond the write time.

---

#### `eq-lr-scaling` (🔴 strict) — def L3028

**Verbatim ref sentence (L3026):**
> "@Eq-lr-scaling expresses an alternative square root scaling law that provides more conservative adjustment during recovery:"

**Missing move:** The lead-in names the square root law and contrasts it with linear scaling (from @goyal2017accurate), but never states the *consequence* of the difference: the square root rule underscales the learning rate relative to linear scaling, which reduces the risk of instability during elastic recovery but also slows convergence after a resize. The payoff (L3032) pivots entirely to gradient accumulation as an alternative and never returns to interpret the equation.

**Where the takeaway currently lives:** Not stated in body prose — the consequence of choosing square root versus linear scaling is left implicit.

**Rule-compliant rewrite (insert after the equation at L3028, before L3032):**

> For a 50-percent worker loss ($N_{\text{new}}/N_{\text{base}} = 0.5$), the square root rule reduces the learning rate by a factor of $\sqrt{0.5} \approx 0.71$, compared to the 0.5 factor the linear rule would prescribe. The smaller adjustment preserves momentum in the remaining workers at the cost of a temporarily oversized learning rate relative to the reduced batch, trading theoretical optimality for empirical stability during the volatile period immediately following a failure.

---

### FIGURES

---

#### `fig-fault-tolerance-failure-spectrum` (🟠 high) — def L124

**Verbatim ref sentence (L122):**
> "The challenges span the full failure spectrum, from transient bit flips through intermittent aging-related errors to permanent component failures, as @fig-fault-tolerance-failure-spectrum illustrates."

**Missing move:** The cite sentence names the three categories but the interpret move — *what* the spectrum implies for system design — lives only in the caption ("Each category demands different detection latency and recovery strategy"). The payoff paragraph (L128) discusses the chapter structure, not what the figure teaches.

**Where the takeaway currently lives:** Caption only.

**Rule-compliant rewrite (replace the reference clause in L122):**

> The challenges span the full failure spectrum: transient bit flips demand fast detection and retry, intermittent aging-related errors demand statistical evidence collection before quarantine, and permanent component failures demand immediate isolation and replacement, as @fig-fault-tolerance-failure-spectrum maps across the three temporal categories. This span of detection latency requirements is why a single fault-tolerance policy cannot cover all three.

---

#### `fig-checkpoint-tax` (🟠 high) — def L250

**Verbatim ref sentence (L248):**
> "@Fig-checkpoint-tax decomposes that same U-curve into its two competing components, the view the Checkpointing section uses when it works the formula through on a concrete cluster."

**Missing move:** "Decomposes…the view the Checkpointing section uses" is a forward pointer, not an interpret move. The actual lesson of the decomposition — that save overhead is hyperbolic (falls rapidly with interval) while rework is linear (rises steadily), so the optimal sits at a specific crossover — lives only in the caption.

**Where the takeaway currently lives:** Caption only.

**Rule-compliant rewrite:**

> @Fig-checkpoint-tax separates the two forces driving that U-curve: save overhead falls hyperbolically as intervals lengthen (a 2x longer interval roughly halves the write fraction), while expected rework rises linearly with the probability of a failure between checkpoints. Their crossover is the Young-Daly optimum, and the figure makes the asymmetry visible — small increases in interval dramatically reduce write overhead while only modestly increasing rework risk.

---

#### `fig-sdc-jeffdean` (🟠 high) — def L639

**Verbatim ref sentence (L637):**
> "@Fig-sdc-jeffdean shows corrupted data blocks accumulating in a shuffle and merge database at Google, where even a small fraction of corrupted blocks can cascade into significant data quality degradation."

**Missing move:** The sentence is a float-announcer followed by a consequence claim. "Even a small fraction…cascade" is the takeaway but it is not explained: *why* does a small fraction cascade? The mechanism (corrupted blocks propagate through downstream joins and aggregations, each inheriting the error silently) lives only in the caption.

**Where the takeaway currently lives:** Partially in the reference sentence as a bare assertion; the propagation mechanism is caption-only.

**Rule-compliant rewrite:**

> Google's production evidence, collected by Jeff Dean and shown in @fig-sdc-jeffdean, demonstrates why even a small fraction of corrupted blocks in a shuffle-and-merge database becomes a fleet-wide reliability threat: corrupted records propagate through every downstream join and aggregation without triggering an exception, so a single corrupted block multiplies its error surface with each pipeline stage it passes through.

---

#### `fig-fault-temporal-categories` (🟠 high) — def L863

**Verbatim ref sentence (L861):**
> "@Fig-fault-temporal-categories summarizes the three categories that matter operationally."

**Missing move:** "Summarizes" is a pointer verb. The interpret move — what each temporal category implies for the recovery decision — is stated partially in the lead-in ("A one-time corruption asks for detection and rollback…") but is not delivered as a prose lead-out from the figure. The payoff (L875) says only "The three categories differ by what the recovery system should infer" without stating what that inference is.

**Where the takeaway currently lives:** Partially in the lead-in (before the figure), not in a dedicated lead-out; payoff is itself a pointer.

**Rule-compliant rewrite (replace the ref sentence and payoff):**

> @Fig-fault-temporal-categories maps the three temporal categories to the evidence each produces. A transient fault leaves no damaged component to find — rollback is the right response. A permanent fault produces repeatable errors across every job that routes through the affected hardware — quarantine and replacement are required. An intermittent fault reappears only under specific thermal or voltage conditions — evidence collection across multiple occurrences is needed before the node is condemned or cleared.

---

#### `fig-bit-flip` (🟠 high) — def L885

**Verbatim ref sentence (L883):**
> "Transient faults are the most common category, and @fig-bit-flip illustrates the basic mechanism: a bit-flip error occurs when a single bit in memory unexpectedly changes state, potentially altering critical data or computations in ways that cascade through neural network layers."

**Missing move:** The sentence names the mechanism (state change) and gestures at cascade but does not name the ML-specific consequence adjacent to the figure. "Potentially altering…in ways that cascade" is vague. The actual implication (a single exponent bit flip in a gradient tensor can produce numerically catastrophic updates) is not stated until L942, well after the figure.

**Where the takeaway currently lives:** L942 (payoff paragraph), too far removed from the figure.

**Rule-compliant rewrite (add a lead-out sentence after the figure):**

> A single bit flip in a gradient tensor's exponent field can shift a value by many orders of magnitude: a $10^{-5}$ gradient becomes $10^{20}$, injecting a catastrophic update that accumulates through every parameter touched by the backward pass. This is why transient faults, despite their ephemeral nature, rank as high-priority targets for detection and correction in distributed training.

---

#### `fig-transient-fault` (🟠 high) — def L1027

**Verbatim ref sentence (L1025):**
> "@Fig-transient-fault shows the same charge-disturbance mechanism the failure taxonomy introduced, now at the device level: a cosmic ray strikes a memory cell or transistor and the induced charge alters stored or transmitted data. What this pass adds is the downstream effect on the model rather than the physics."

**Missing move:** The sentence promises "the downstream effect on the model" but does not deliver it — that effect is deferred to the payoff paragraph (L1033). Adjacent to the figure, the prose gives only the physical mechanism.

**Where the takeaway currently lives:** L1033 (payoff), several lines after the figure.

**Rule-compliant rewrite (add a lead-out sentence after the figure, before L1033):**

> At the model level, this charge disturbance manifests as a corrupted tensor value in the exact memory address that the induced charge struck — an effect invisible to the training process, which continues accumulating gradients from the corrupted value into every downstream parameter update until a validation check or loss anomaly surfaces the problem.

---

#### `fig-permanent-fault` (🟠 high) — def L1051

**Verbatim ref sentence (L1049):**
> "The Intel Pentium FDIV bug, discovered in 1994, provides the canonical illustration of this failure mode in a general-purpose processor. An error in the lookup table used by the Pentium processor's division unit caused incorrect results for specific operations (@fig-permanent-fault)."

**Missing move:** The figure is cited parenthetically. The lead-in paragraph develops the analogy before the float, and the payoff (L1057) immediately pivots to the stuck-fault figure. No prose sentence appears *after* the figure that draws the ML implication of what was just shown. The Pentium FDIV lesson (a small fifth-digit error that compounded across operations) and its ML equivalent (stuck-at fault corrupting every matrix-multiply through the affected lane) is stated only in the lead-in, not as a conclusion from the figure.

**Where the takeaway currently lives:** Lead-in only (L1049); payoff is a pointer to the next figure.

**Rule-compliant rewrite (insert after the figure float, before L1057):**

> The FDIV bug illustrates the permanent fault's defining property: the error is small, repeatable, and operation-specific. In ML accelerators, a stuck-at fault in a Tensor Core lane produces the same biased output on every matrix-multiply that routes through that lane — meaning every forward and backward pass accumulates the same systematic error, which training loss cannot distinguish from a difficult batch until the accumulated bias diverges the model.

---

#### `fig-intermittent-fault` (🟠 high) — def L1141

**Verbatim ref sentence (L1139):**
> "Physical degradation (cracks in solder joints, aging ball grid arrays, residue-induced electrical connections) creates those load-dependent conditions (@fig-intermittent-fault)."

**Missing move:** The figure is cited parenthetically within a sentence that is itself about the conditions, not about what the figure teaches. No prose sentence appears after the figure before the pivot to `fig-intermittent-fault-dram`.

**Where the takeaway currently lives:** Nowhere in body prose — the physical mechanism is named in the lead-in; no interpret move exists.

**Rule-compliant rewrite (insert after the figure, before L1147):**

> The crack between the copper bump and solder joint is the key detail: resistance at that junction increases with thermal expansion during high-utilization training and decreases during idle cool-down, so the fault appears only under sustained load. This load-sensitivity is what makes intermittent faults so destructive — a node that passes a reboot-and-ping health check can still fail repeatedly during the next multi-GPU training run.

---

#### `fig-intermittent-fault-dram` (🟠 high) — def L1149

**Verbatim ref sentence (L1147):**
> "@Fig-intermittent-fault-dram reveals how residue-induced intermittent faults in DRAM chips create unreliable electrical connections that lead to sporadic failures."

**Missing move:** Float-announcer ("reveals how…create…lead to"). The sentence describes the figure's subject but does not state the ML implication or the engineering takeaway. Why does DRAM residue matter for ML training specifically?

**Where the takeaway currently lives:** L1155 (payoff) gives the ML response but does not connect back to the DRAM-specific mechanism shown in the figure.

**Rule-compliant rewrite:**

> DRAM residue faults are particularly damaging in ML training because weight and optimizer state tensors reside in exactly the high-density DRAM banks where residue accumulates over time (@fig-intermittent-fault-dram). An intermittent DRAM error during a weight read injects a corrupted value into the forward pass without invalidating the allocation — the process continues running, gradient accumulation continues, and the corrupted weight silently biases every subsequent update until the anomaly surfaces in the loss curve.

---

#### `fig-tesla-dmr` (🟠 high) — def L1281

**Verbatim ref sentence (L1275):**
> "Tesla's Full Self-Driving computer uses DMR across two independent system on chip (SoC) units (@fig-tesla-dmr), while the Boeing 777 uses TMR in its primary flight computer for safety-critical aviation control."

**Missing move:** Parenthetical citation. The figure's content (the DMR comparator architecture detecting mismatches before actuator commands) is not described in adjacent prose. The interpret move — what DMR buys at the silicon level and why it stops at detection rather than correction — is in a footnote, not in running body prose.

**Where the takeaway currently lives:** Footnote fn-dmr-detection, not body prose.

**Rule-compliant rewrite (replace the parenthetical with an explicit reference):**

> Tesla's Full Self-Driving computer demonstrates how hardware redundancy is implemented at the SoC level: two independent chips run the same computation in parallel, and a hardware comparator blocks any actuator command on which they disagree (@fig-tesla-dmr). The design pays 100 percent silicon overhead for detection-without-correction — the right trade for a safety context where halting on disagreement is preferable to masking an error silently.

---

#### `fig-regression-testing-ft` (🟠 high) — def L1354

**Verbatim ref sentence (L1352):**
> "regression tests preserve representative prompts and edge cases before they enter distributed execution (@fig-regression-testing-ft)."

**Missing move:** Parenthetical citation with no dedicated interpret move. The broader sentence is about what a CI/CD gate must do; the figure's content (a regression-test flow diagram) is never interpreted. What should the reader notice when looking at the figure?

**Where the takeaway currently lives:** L1352 explains the requirement in general terms but not what the figure specifically shows about regression test flow.

**Rule-compliant rewrite (add a sentence after the parenthetical citation):**

> The flow in @fig-regression-testing-ft places the regression suite at the gate between a code commit and distributed execution: every change to the tokenizer, data collation, or model architecture must pass the representative-prompt assertions before the modified artifact can run on the training cluster.

---

#### `fig-error-masking-ft` (🟠 high) — def L1900

**Verbatim ref sentence (L1898):**
> "This masking phenomenon can cause faults to be filtered out before they propagate to higher levels (@fig-error-masking-ft)."

**Missing move:** Parenthetical citation. The figure shows a decision tree with four fault outcomes (logical masking, microarchitectural masking, SDC, DUE). The prose does not name what these outcomes are or why the masking branches matter for software-based fault injection tools.

**Where the takeaway currently lives:** Context established in L1898 but as assertion; the four-outcome structure and its implication for software tools are caption-only.

**Rule-compliant rewrite:**

> The outcome tree in @fig-error-masking-ft exposes why software-based fault injection underestimates real hardware resilience: two of the four outcome branches (logical masking and microarchitectural masking) filter the fault before it becomes a visible bit flip, while software simulators inject at the architectural register level and therefore see only the SDC and DUE outcomes, missing the majority of events that hardware silently absorbs.

---

#### `fig-distributed-checkpoint-architecture` (🟠 high) — def L2616

**Verbatim ref sentence (L2614):**
> "In distributed checkpointing, each worker writes its portion of the checkpoint to a shared filesystem or object storage, as @fig-distributed-checkpoint-architecture contrasts with the centralized approach."

**Missing move:** The cite sentence describes what the figure contrasts but does not state the *reason* the contrast matters. The payoff (L2620) moves immediately to the six-step protocol without naming the bandwidth advantage of distributed writes.

**Where the takeaway currently lives:** Caption only ("aggregating bandwidth across the storage fabric and minimizing the training pause").

**Rule-compliant rewrite:**

> In distributed checkpointing, each worker writes its shard directly to the parallel filesystem at its own network interface bandwidth rather than serializing through a single coordinator (@fig-distributed-checkpoint-architecture). With 1,000 workers each writing 10 GB/s, the aggregate write bandwidth is 10 TB/s — a centralized approach on a single network path cannot approach that rate, so the distributed pattern is the only viable design for checkpointing at fleet scale.

---

#### `fig-observability-three-pillars` (🟠 high) — def L3365

**Verbatim ref sentence (L3363):**
> "@Fig-observability-three-pillars summarizes those three evidence types, but the operating rule is correlation: no signal is sufficient unless it can be joined to the same request, model version, feature version, and deployment event."

**Missing move:** "Summarizes" is a pointer verb. The correlation rule is stated well in the same sentence, but the figure's own content (a Venn diagram showing the three overlapping pillars) is never interpreted. What does the overlap *mean* for fault diagnosis?

**Where the takeaway currently lives:** The correlation principle is stated in prose, but the visual argument of the Venn structure (overlap enables joint diagnosis) is not interpreted.

**Rule-compliant rewrite:**

> The overlap in @fig-observability-three-pillars is the point: metrics show that p99 latency spiked at 14:32, traces reveal which service span consumed the budget, and logs supply the GPU OOM error from that exact span. No single pillar diagnoses the incident — correlation across all three identifies which model version, deployed at what time, caused the memory spike that started the cascade.

---

### LISTINGS

---

#### `lst-nan-detection` (🟡 medium) — def L3424

**Verbatim ref sentence (L3420):**
> "@Lst-nan-detection shows a minimal check that catches corruption before it reaches users."

**Missing move:** "Shows a minimal check" does not name the mechanism the code embodies. What does the reader need to notice before reading the listing?

**Where the takeaway currently lives:** L3436 (payoff) discusses gradient statistics generally but does not return to the specific NaN-detection mechanism.

**Rule-compliant rewrite:**

> @Lst-nan-detection implements the first defense against silent numerical corruption: it validates model outputs for NaN values and logs the input hash before the result reaches any downstream consumer. The key design choice is the input hash — it makes the exact corrupting input reproducible during diagnosis, converting a transient numerical anomaly into a reproducible test case.

---

### TABLES

---

#### `tbl-memory-bandwidth-protection` (🟠 high) — def L1019

**Verbatim ref sentence (L952):**
> "@Tbl-memory-bandwidth-protection quantifies this cost across memory technologies:"

**Missing move:** "Quantifies this cost" is a bare pointer. The key row-level finding — HBM's superior bandwidth comes with 10× higher error rates that make ECC mandatory — is not stated in prose adjacent to the table.

**Where the takeaway currently lives:** L1021 (payoff) states the HBM finding, but no prose at the cite site names the architectural implication of the bandwidth-reliability tradeoff.

**Rule-compliant rewrite (expand the citation sentence):**

> @Tbl-memory-bandwidth-protection quantifies the protection tax across memory technologies, and the key finding is architectural: HBM's 3D stacking that delivers 10× higher bandwidth also produces 10× higher error rates compared to planar DRAM, making ECC not an option but a requirement for any HBM-based accelerator in a production ML cluster.

---

#### `tbl-software-faults-summary-ft` (🟠 high) — def L1350

**Verbatim ref sentence (L1338):**
> "@Tbl-software-faults-summary-ft organizes these gates by the kind of corruption they are designed to catch."

**Missing move:** "Organizes" is a pointer verb. The table's decision-forcing insight — that each gate must be assigned a specific corruption class rather than acting as a general quality check — is in the lead-in paragraph but not as a conclusion drawn from the table.

**Where the takeaway currently lives:** Lead-in L1338 explains the assignment logic; no dedicated lead-out names what the table's structure reveals.

**Rule-compliant rewrite (add a sentence after the table, before L1352):**

> The table's columns reveal the assignment logic: testing catches corruption that is present at development time and reproducible, monitoring catches corruption that emerges at runtime or at scale, and fault-tolerant design provides the fallback when both miss. Each gate defends a distinct window — no single technique covers all three windows, so the layered strategy is a necessity, not a preference.

---

#### `tbl-fault-tolerance-detection-latencies` (🟠 high) — def L2749

**Verbatim ref sentence (L2738):**
> "Production experience shows that failure detection takes significantly longer than theoretical heartbeat timeouts suggest. The core challenge is distinguishing failures from stragglers, as @tbl-fault-tolerance-detection-latencies records."

**Missing move:** "Records" is a pointer. The key finding from the table — that SDC detection latency is two to three orders of magnitude longer than a process crash — is not stated in prose adjacent to the table.

**Where the takeaway currently lives:** The multi-stage detection rule is in the lead-in; the specific latency spread is only in the table cells.

**Rule-compliant rewrite (replace "records" with a takeaway):**

> Production experience shows that failure detection takes significantly longer than theoretical heartbeat timeouts suggest. The core challenge is distinguishing failures from stragglers: @tbl-fault-tolerance-detection-latencies shows that detection latency spans roughly three orders of magnitude from a process crash (detectable in tens of seconds) to silent data corruption (detectable only after hours of anomalous training metrics), and a single timeout threshold cannot serve both ends of that range.

---

#### `tbl-cold-vs-warm-restart` (🟠 high) — def L2886

**Verbatim ref sentence (L2876):**
> "@Tbl-cold-vs-warm-restart contrasts the two strategies:"

**Missing move:** "Contrasts" is a pointer. The table's decision rule — warm restart as the fast path for single-node failures, cold restart as the safety net for correlated or SDC events — is stated in the lead-in paragraph but not as a prose lead-out derived from the table.

**Where the takeaway currently lives:** Lead-in L2876; no dedicated lead-out.

**Rule-compliant rewrite (add a sentence after the table, before L2890):**

> The table's bottom rows make the decision rule concrete: warm restart wins on recovery time (30-90 seconds vs. 4-10 minutes) but requires falling back to cold restart on any failure *during* the warm restart itself. A mature deployment therefore treats warm restart as the default path for the most common failure class (single-node GPU errors) and cold restart as the guaranteed-clean fallback for correlated failures and SDC events, where surviving worker state cannot be trusted.

---

#### `tbl-elastic-training-comparison` (🟠 high) — def L3073

**Verbatim ref sentence (L3064):**
> "The elastic training framework support summarized in @tbl-elastic-training-comparison compares the recovery automation and state management approaches across these frameworks."

**Missing move:** "Compares" is a pointer with no stated conclusion. No prose sentence names which framework dimension matters most for a given use case, or what the key differentiator across the four frameworks is.

**Where the takeaway currently lives:** Nowhere in adjacent body prose — the payoff (L3077) pivots to model-specific fault tolerance without deriving a conclusion from the comparison.

**Rule-compliant rewrite (add a sentence after the table, before L3077):**

> The most consequential column is state resharding: PyTorch Elastic and Horovod Elastic require manual resharding, meaning the application code must handle the remapping of model shards to new ranks, while Ray Train automates resharding at the cost of tighter coupling to the Ray cluster scheduler. The right choice follows from whether the training codebase can absorb manual resharding logic or whether operational simplicity at the cost of scheduler lock-in is acceptable.

---

#### `tbl-stateful-failover` (🟠 high) — def L3191

**Verbatim ref sentence (L3182):**
> "The choice depends on state size, update frequency, and the quality impact of state loss; @tbl-stateful-failover summarizes the trade-offs across the four common approaches:"

**Missing move:** "Summarizes the trade-offs" is a pointer. The decision criterion is stated before the table (in the lead-in), but the prose does not name *which row* the table's structure favors under what conditions. The lead-out is missing.

**Where the takeaway currently lives:** Decision factors in lead-in; no prose derives a conclusion from the table.

**Rule-compliant rewrite (add a sentence after the table, before L3199):**

> The table's fast/medium latency column drives the primary decision: for KV cache in LLM serving, where regeneration takes seconds and user session continuity is the product requirement, synchronous replication's fast recovery at high operational cost is the appropriate choice. For lower-value session state, the distributed-state approach (fast recovery, configurable consistency) avoids both the quality hit of state loss and the operational burden of synchronous replication.

---

#### `tbl-serving-recovery-state-invariants` (🟠 high) — def L3207

**Verbatim ref sentence (L3199):**
> "@Tbl-serving-recovery-state-invariants maps that question across the three common regimes."

**Missing move:** "Maps" is a pointer. No prose sentence derives the table's common pattern or names a decision rule from its rows.

**Where the takeaway currently lives:** L3209 discusses the KV cache cost but does not derive a general principle from the table.

**Rule-compliant rewrite (add a sentence after the table, before L3209):**

> The table's right column reveals the common structure: each serving workload has one state class where loss triggers user-visible quality degradation (context for LLMs, feature freshness for recommendation, accelerator health for vision), and the fault-tolerance policy must protect that class first while accepting graceful degradation in the others.

---

#### `tbl-serving-failure-response-map` (🟠 high) — def L3502

**Verbatim ref sentence (L3493):**
> "@Tbl-serving-failure-response-map separates the common symptoms by what they reveal and what the serving system must do first."

**Missing move:** "Separates" is a pointer. The table's key ranking — which symptom class is most urgent to address — is not stated in prose.

**Where the takeaway currently lives:** No dedicated lead-out; payoff (L3504) discusses cascade failures as one case.

**Rule-compliant rewrite (add a sentence after the table, before L3504):**

> Rising error rates and cascade failures are the two rows that demand immediate action: error rates compound across dependent services within seconds, and cascade failures propagate to the full fleet if circuit breakers are not tripped before isolation boundaries are lost. Latency spikes and silent quality degradation, while serious, allow a diagnostic window that the error-rate and cascade rows do not.

---

*End of worklist. 28 findings, all ⚠️ (partial). No 🛑 (fails) found.*
