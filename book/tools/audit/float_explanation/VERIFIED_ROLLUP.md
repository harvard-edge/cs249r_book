# Float-Explanation Audit — VERIFIED survivors (post-refutation)

Raw first-pass findings adversarially re-tested with table captions now visible. A finding SURVIVES only if NO neighborhood element (ref sentence, prev/next paragraph, caption, or payoff) explains what the float shows.

## vol1 · frameworks  (1)

### ⚠️ `eq-execution-continuum` — def L1768  (punctuation-only)
- Ref: "The execution models form a continuum from maximum flexibility to maximum optimization, visualized in @eq-execution-continuum:"
- Why it survives: The explanation is fully present in the neighborhood. The preceding paragraph (L1762) frames the quantitative problem, and the payoff at L1770 states "Each step rightward sacrifices flexibility for performance" and immediately introduces @eq-compilation-benefit to operationalize the principle. The float's content and significance are clear. The surviving issue is punctuation only: the ref sentence terminates with a colon before the float block, violating the book's prose rule that float references use a period rather than a colon as terminal punctuation. No substantive rewrite is needed.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - The execution models form a continuum from maximum flexibility to maximum optimization, visualized in @eq-execution-continuum:
  + The execution models form a continuum from maximum flexibility to maximum optimization. @Eq-execution-continuum maps the four positions on that axis, with each arrow labeling the mechanism that moves a project one step rightward.
  ```


## vol1 · hw_acceleration  (2)

### ⚠️ `lst-nonlinear_layer` — def L1343
- Ref: "To see why dedicated hardware matters, consider a typical layer sequence. @Lst-nonlinear_layer combines linear transformations with nonlinear activations—operations that appear simple in Python but reveal substantial computational complexity at the hardware level."
- Why it survives: The ref frames the purpose ("to see why dedicated hardware matters") but names nothing specific. The payoff (L1354) says only "introduces multiple nonlinear transformations that extend beyond simple matrix operations" — it acknowledges complexity without naming what type of hardware demand each operation creates. The caption ("enhances model expressiveness and learning capabilities") is pure ML-speak with no hardware content. No prose in the direct neighborhood of this listing names the three qualitatively different hardware demands: `nn.Linear` maps to matmul (tensor cores), `nn.ReLU` maps to a conditional max (special function unit), and `nn.BatchNorm1d` requires two reduction passes for mean and variance. The explanation appears only after `lst-nonlinear_math` closes (L1371), which is outside this listing's neighborhood.
- Suggested rewrite (rule-compliant; no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - This sequence introduces multiple nonlinear transformations that extend beyond simple matrix operations. @Lst-nonlinear_math breaks down these operations into their mathematical components, exposing the computational complexity that hardware must address.
  + The three operations in this four-line sequence each place a different demand on the accelerator: the linear layer maps to matrix multiplication (the tensor core's native operation), ReLU maps to a per-element conditional selection (a special function unit), and BatchNorm requires two reduction passes (mean, then variance) before the normalization itself. @Lst-nonlinear_math breaks down these operations into their mathematical components to make those demands explicit.
  ```

---

### ⚠️ `lst-arm_sve_vector` — def L1461
- Ref: "The Arm Scalable Vector Extension (SVE) provides a representative example of how modern architectures implement scalable SIMD operations efficiently. @Lst-arm_sve_vector demonstrates this approach."
- Why it survives: The ref is a bare announcement ("demonstrates this approach") that adds nothing beyond naming the listing. The preceding paragraph (L1457) explains SIMD in general (512 scalar ops become 32–64 instructions) but says nothing specific to SVE. The payoff (L1473) is a generic bridge sentence: "Processor architectures continue to expand SIMD capabilities to accommodate increasing computational demands." The caption ("Vector multiplication and addition operations enable efficient parallel processing in machine learning models") could describe any vector listing. The distinctive feature of SVE visible in the code, `ptrue p0.s`, is never explained: SVE's scalable-width design queries the hardware's native vector length at runtime, so the same binary runs correctly on cores with 128-bit through 2048-bit vector units without recompilation. No prose in the neighborhood names this property or contrasts it with fixed-width SIMD.
- Suggested rewrite (rule-compliant; no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - Processor architectures continue to expand SIMD capabilities to accommodate increasing computational demands. Intel's Advanced Matrix Extensions (AMX) [@intel2021amx] and Arm's SVE architecture [@stephens2017] provide flexible execution models, enabling software to scale across different hardware implementations.
  + The `ptrue p0.s` predicate at the head of this sequence captures SVE's key property: the instruction queries the hardware's native vector width at runtime, so the same binary executes correctly on cores with 128-bit to 2048-bit vector units without recompilation. A model kernel built once can achieve full vector utilization on a mobile core and a server core. Intel's Advanced Matrix Extensions (AMX) [@intel2021amx] and Arm's SVE [@stephens2017] represent successive steps toward flexible, width-agnostic execution models that eliminate the fixed-width rewrite cycle traditional SIMD required.
  ```

---


## vol1 · introduction  (1)

### ⚠️ `tbl-software-1-vs-2` — def L128
- Ref: "@Tbl-software-1-vs-2 summarizes this paradigm shift."
- Why it survives: The ref sentence is a bare pointer ("summarizes this paradigm shift") with no inline takeaway. The preceding paragraph names the 1.0/2.0 framing but describes no row content. The next paragraph (L130) pivots immediately to Google's technical-debt paper without unpacking the table. The caption carries the real explanation (debugging moves upstream from code to data; the compiler analogy is stochastic), so the reader is not stranded, but the caption is the *only* place the table's significance is stated. No body-prose sentence tells the reader why the failure-mode row (loud crash vs. silent metric degradation) is the consequential distinction that motivates the entire chapter. The adversarial standard requires that explanation live in the neighborhood, not solely in the caption.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - Andrej Karpathy[^fn-karpathy-sw2] formalized this distinction as the shift from **Software 1.0**\index{Software 1.0} to **Software 2.0**\index{Software 2.0} [@karpathy2017software], a framing that captures *why* ML systems require entirely new engineering approaches. @Tbl-software-1-vs-2 summarizes this paradigm shift.
  + Andrej Karpathy[^fn-karpathy-sw2] formalized this distinction as the shift from **Software 1.0**\index{Software 1.0} to **Software 2.0**\index{Software 2.0} [@karpathy2017software], a framing that captures *why* ML systems require entirely new engineering approaches. @Tbl-software-1-vs-2 maps the shift term by term. The row that drives the rest of this chapter is the failure mode: Software 1.0 fails loudly with a crash, while Software 2.0 fails silently through metric degradation, making the failure invisible until a monitoring system catches it.
  ```


## vol1 · training  (1)

### ⚠️ `tbl-scaling-decision` — def L6522
- Ref: "@Tbl-scaling-decision provides quantitative guidance for scaling decisions across different model and data scales."
- Why it survives: The ref sentence is a pure pointer with no content. The prev paragraph (L6506-6511) lists four single-machine optimizations to exhaust first but names no thresholds. The caption states the organizing principle but does not unpack any specific value. The payoff paragraph (L6524+) explains three hard limits (memory exhaustion, wall-clock time, dataset scale) in narrative terms but never translates the table's specific numeric brackets into prose — the reader is not told that 1-10B fits on a single multi-GPU node and why ("Model parallelism within node avoids network"), nor that sub-1B fits on a single GPU, nor what drives the >10 TB dataset threshold. The table rows carry all of the substantive guidance; nothing in the surrounding text primes the reader on what to look for or what the threshold values mean.
- Suggested rewrite (rule-compliant; no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - @Tbl-scaling-decision provides quantitative guidance for scaling decisions across different model and data scales.
  + @Tbl-scaling-decision translates these limits into a practical lookup: models below one billion parameters fit on a single GPU with the optimizations above, models in the 1-10 billion range fit on a single multi-GPU node (keeping high-speed intra-node interconnect rather than the slower inter-node fabric), and only models above 10 billion parameters or datasets above 10 TB justify multi-node distributed complexity.
  ```

---


## vol2 · distributed_training  (2)

### ⚠️ `fig-comm-convergence-tradeoff` — def L1704
- Ref: "The fundamental trade-off in distributed training is between communication efficiency and convergence quality. @Fig-comm-convergence-tradeoff visualizes this trade-off space."
- Why it survives: The ref sentence is a pure announcer with no geometric reading of the figure. The prev context is only a section heading (L1700). The payoff paragraph (L1708) reads "Several techniques occupy different positions on this trade-off curve" and then pivots directly into describing individual techniques (gradient compression, Local SGD, decentralized SGD) without ever reading a conclusion from the Pareto geometry. The caption names the positions of BSP/ASP/SSP/gradient-compression on the frontier but does not state the practical design consequence of where the gaps between them fall. No neighborhood element tells the reader what the frontier's shape means for actual system choices.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - The fundamental trade-off in distributed training is between communication efficiency and
  - convergence quality. @Fig-comm-convergence-tradeoff visualizes this trade-off space.
  + The fundamental trade-off in distributed training is between communication efficiency and
  + convergence quality. @Fig-comm-convergence-tradeoff plots each strategy on that frontier:
  + BSP occupies the high-convergence corner at the cost of throughput; ASP maximizes
  + throughput at the cost of convergence; gradient compression and SSP sit in the middle,
  + where most production systems operate when bandwidth cost outweighs the convergence
  + penalty of moderate staleness or lossy updates.
  ```

---

### ⚠️ `tbl-convergence-comparison` — def L1425
- Ref: "@Tbl-convergence-comparison summarizes the convergence properties of each synchronization model."
- Why it survives: The preceding paragraphs (L1392–L1415) develop each convergence bound individually, but no sentence synthesizes what the side-by-side comparison reveals. The ref sentence is a bare pointer. The payoff paragraph (L1427) opens a new section on learning-rate scaling rules and makes no contact with the table's content. The caption correctly states per-model take-aways but does not surface the cross-model comparison that makes the table worth a read: BSP achieves $\mathcal{O}(1/\sqrt{NbK})$ with full $1/N$ variance reduction; SSP adds an $s^2$ staleness-penalty term; ASP loses variance reduction entirely so its dominant term scales with average staleness squared rather than worker count. No neighborhood element draws that comparison.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - @Tbl-convergence-comparison summarizes the convergence properties of each synchronization model.
  + @Tbl-convergence-comparison places the three convergence rates side by side. BSP achieves
  + $\mathcal{O}(1/\sqrt{NbK})$ with full $1/N$ variance reduction; SSP adds a staleness-penalty
  + term that grows as $s^2$; ASP loses variance reduction entirely, so its dominant term scales
  + with average staleness squared rather than worker count. The practical consequence is that
  + increasing $N$ only improves ASP convergence if staleness is actively controlled.
  ```

---


## vol2 · fault_tolerance  (1)

### ⚠️ `fig-intermittent-fault-dram` — def L1149
- **Ref:** "@Fig-intermittent-fault-dram reveals how residue-induced intermittent faults in DRAM chips create unreliable electrical connections that lead to sporadic failures." (L1147)
- **Why it survives:** Every neighborhood element repeats the same claim without adding a DRAM-specific argument. The section intro at L1139 already lists "residue-induced electrical connections" as one instance of physical degradation. The preceding figure (fig-intermittent-fault, L1141) shows the solder-crack mechanism with its own caption. The ref sentence at L1147 is a pure announcer: it restates the mechanism already named in the section intro and mirrors the caption verbatim. The caption itself says only that residue causes unreliable connections and that this "highlights the need for fault-tolerant system design and hardware testing" — the same generic takeaway the section gives for every intermittent-fault type. The payoff at L1155 covers intermittent faults as a category and gives ML-specific advice (treat as suspect, use runtime monitoring, adaptive resource management) but makes no DRAM-specific point: the advice applies equally to solder-crack or any other intermittent mechanism. No neighborhood element answers why the reader needs a second figure showing DRAM residue after the first figure showed solder cracks, or what is distinct about DRAM intermittent faults for ML systems (e.g., load-dependent bandwidth failures that pass manufacturing test, different screening difficulty, different detection signal). This is a genuine dead-end where the second figure adds visual content not connected by any prose claim.
- **Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):**
  ```diff
  - @Fig-intermittent-fault-dram reveals how residue-induced intermittent faults in DRAM
  -  chips create unreliable electrical connections that lead to sporadic failures.
  + DRAM is particularly susceptible to this failure class: residue contamination between
  +  memory-cell contacts (@fig-intermittent-fault-dram) creates a load-dependent resistance
  +  path that passes manufacturing test under light access patterns yet fails under the
  +  sustained bandwidth demands of a training run. Unlike the solder-crack mechanism
  +  shown above, DRAM residue faults are not exposed by thermal cycling alone, making
  +  them harder to screen before deployment and more likely to appear mid-job when
  +  gradient tensors stress memory bandwidth continuously.
  ```


## vol2 · inference  (2)

### ⚠️ `fig-serving-hierarchy` — def L589
- Ref: "A related deployment stack appears in @fig-serving-hierarchy, showing how requests pass through edge, routing, and model-serving infrastructure in production."
- Why it survives: The ref sentence is a bare announcer. The prev paragraph (L585) establishes a four-level conceptual hierarchy (request, replica, service, platform). The figure shows a three-tier physical deployment topology (CDN/Edge Cache, Gateway/Router, Model Serving Cluster). No element in the neighborhood — not the ref sentence, not the caption, not the payoff (L593, which pivots immediately to the table) — ever states the relationship between the four conceptual levels and the three physical tiers, nor why the figure reinforces the hierarchy argument made in the preceding paragraph. The caption describes what each tier does (SLA numbers per tier) but does not connect the physical tiers to the conceptual levels. A reader cannot tell whether Tier 1 maps to the "service level," the "platform level," or neither. The figure reads as a digression from the hierarchy argument rather than its physical embodiment.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - The hierarchy matters because each level changes a different metric and fails at a different boundary. A related deployment stack appears in @fig-serving-hierarchy, showing how requests pass through edge, routing, and model-serving infrastructure in production.
  + The hierarchy matters because each level changes a different metric and fails at a different boundary. @Fig-serving-hierarchy shows the physical deployment stack that hosts these four levels: the CDN tier absorbs request-level traffic before it reaches the serving infrastructure, the gateway tier routes and rate-limits at the service level, and the model-serving cluster is where replica-level and platform-level optimizations operate. Each tier boundary is a latency checkpoint with its own SLA budget, and a failure at any boundary manifests as a different metric violation.
  ```

---

### ⚠️ `lst-metric-based-scaling` — def L4709
- Ref: "@Lst-metric-based-scaling shows a typical metric-based scaling configuration."
- Why it survives: The ref is a bare announcer with no substance. The YAML listing contains three distinct design choices that embody tradeoffs: asymmetric thresholds (80 percent scale-up, 50 percent scale-down), a specific cooldown duration (300 seconds), and the choice of cpu_utilization as the primary metric. None of these choices is explained anywhere in the neighborhood. The prev paragraph (L4705) discusses cold-start latency, not threshold design. The caption (L4709) names the cooldown period and says it prevents oscillation, but does not explain why 300 seconds or why the scale-down threshold is set 30 points below the scale-up threshold. The payoff paragraph (L4722) moves immediately to a queue-depth alternative without explaining what the listing's asymmetric thresholds achieve. A student reading this listing has no way to understand why the thresholds differ or what would happen with a symmetric design.
- Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):
  ```diff
  - @Lst-metric-based-scaling shows a typical metric-based scaling configuration.
  + @Lst-metric-based-scaling shows a typical threshold-based autoscaler. The asymmetric thresholds (80 percent to scale up, 50 percent to scale down) create a hysteresis band that prevents oscillation between scale-up and scale-down decisions when utilization hovers near a single threshold. The 300-second cooldown reinforces this stability by blocking further scaling actions until newly provisioned replicas have fully warmed up and begun absorbing traffic, so the system does not interpret transient underutilization during warm-up as a signal to scale back down.
  ```

---


## vol2 · introduction  (1)

### ⚠️ `fig-loss-vs-n-d` — def L1101
- **Ref:** "These predictions find strong empirical support across multiple model configurations. @Fig-loss-vs-n-d shows *how* early-stopped test loss varies predictably with both dataset size and model size, confirming that learning curves across configurations align through appropriate parameterization."
- **Why it survives:** Every neighborhood element was checked. The ref sentence (L1081) is a pure float-announcer that restates the caption in methodological terms ("learning curves align through appropriate parameterization") without stating any systems implication. The prev paragraph is a callout close marker (:::). The next paragraph (L1083) opens a new subsection ("Resource-constrained scaling regimes") with no look-back at the figure. The caption names the behavior ("all curves exhibit diminishing returns at high token counts") but does not land the engineering implication. The payoff paragraph (L1172) is distant and does not reference this figure. No neighborhood element states the key takeaway: that all model sizes plateau as token volume grows, and that larger models reach a lower plateau rather than avoiding it, meaning model capacity does not rescue data starvation but only lowers the floor that data volume determines.
- **Suggested rewrite (ref sentence, L1081):**
  ```diff
  - These predictions find strong empirical support across multiple model configurations. @Fig-loss-vs-n-d shows *how* early-stopped test loss\index{Scaling Laws!loss curves} varies predictably with both dataset size and model size, confirming that learning curves across configurations align through appropriate parameterization.
  + These predictions find strong empirical support across multiple model configurations. @Fig-loss-vs-n-d shows that every model size plateaus as token volume grows, and that larger models reach a lower plateau rather than avoiding it: capacity lowers the floor that data volume determines, but does not escape the plateau entirely.
  ```

---


## vol2 · ops_scale  (1)

### ⚠️ `fig-tco-iceberg` — def L3635
- **Ref:** "As @fig-tco-iceberg illustrates, while GPU compute and storage are the visible costs, hidden operational costs often constitute fully half of the actual budget."
- **Why it survives:** The ref sentence mirrors the caption word-for-word; no neighborhood element adds explanation. The caption (L3635) restates the same claim. The preceding content (L3631) is the equation definition with no iceberg commentary. The post-figure prose (L3737) explains only the equation symbols ($C_\text{train}$, $C_\text{infer}$, etc.), not the figure's visual insight. The payoff paragraph (L3739) addresses how the dominant cost component shifts with organizational maturity — a different claim about TCO dynamics, not about the iceberg framing itself. Nowhere does any prose element explain which specific hidden categories surprise organizations, why the visible/hidden split is not obvious from the equation alone, or what operational action the two-zone breakdown implies.
- **Suggested rewrite (flag-only):**
  ```diff
  - As @fig-tco-iceberg illustrates, while GPU compute and storage are the visible costs, hidden operational costs often constitute fully half of the actual budget.
  + The distribution in @fig-tco-iceberg explains why cost-reduction efforts aimed only at GPU spend routinely disappoint: the waterline separates the two visible infrastructure categories (GPU compute at 40 percent, object storage at 10 percent) from six operational categories — engineering labor, data pipeline maintenance, retraining compute, monitoring, incident response, and compliance — that collectively match them. A team that halves GPU spend leaves the larger half of its budget untouched and gains no relief on the operational side.
  ```

---


---
**Total verified survivors: 12**
