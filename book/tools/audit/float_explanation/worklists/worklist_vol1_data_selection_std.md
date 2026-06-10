# Float Exposition Worklist — `data_selection.qmd` (vol1)

**Standard:** FLOAT_EXPOSITION_STANDARD.md
**Chapter path:** `book/quarto/contents/vol1/data_selection/data_selection.qmd`

---

## Summary Table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| eq   | 🔴    | 2      | 2  | 0  | 0  |
| fig  | 🟠    | 14     | 8  | 5  | 1  |
| lst  | 🟡    | 1      | 1  | 0  | 0  |
| tbl  | 🟠    | 21     | 13 | 8  | 0  |
| **Total** | | **38** | **24** | **13** | **1** |

---

## Findings (⚠️ and 🛑 only)

---

### F01 — `fig-data-selection-pipeline` (fig 🟠) — def L727

**Verbatim ref sentence (L725):**
> "A random batch of raw data often has low ICR: it contains redundant examples, noisy samples, or "easy" examples the model has already mastered, wasting GPU cycles on zero-information updates. High-efficiency data pipelines (@fig-data-selection-pipeline) filter, order, and synthesize data to maximize ICR, ensuring that every FLOP contributes to learning."

**Missing move:** Lead-out / payoff. The prose explains *why* random batches are wasteful and names the three pipeline operations, but never states what the three-stage sequence as a whole accomplishes — namely, that each stage attacks a different cause of low ICR (redundancy, ordering, scarcity) and that applying them in the pipeline order compounds the gains. The post-float payoff (L943) pivots immediately to a checkpoint: "With the ICR framework established, we can verify understanding of its core mechanics."

**Where the takeaway currently lives:** Implied by the stage labels (Static Pruning, Dynamic Selection, Synthetic Generation) in the figure and its caption; not present in body prose.

**Rule:** Figure 🟠 — prose must deliver what the figure *demonstrates* (the relationship or mechanism) and why it matters.

**Rewrite (lead-out, insert after the figure at L943):**

> The three stages are ordered by cost and reversibility. Static pruning runs once before training and delivers the cheapest per-FLOP savings; dynamic selection adjusts the working set each epoch and captures the evolving difficulty signal the model exposes during training; synthetic generation creates examples that no amount of curation can produce from real data alone. Together they form a cascading filter: each stage raises the ICR of the data that enters the next, so the gains compound rather than add.

---

### F02 — `fig-amortization-comparison` (fig 🟠) — def L2328

**Verbatim ref sentence (L2326):**
> "Contrast the two bar charts in @fig-amortization-comparison to see this cost structure in action. Training from scratch (left) incurs the full cost for each task independently. The foundation model approach (right) pays a large upfront pretraining cost but then fine-tunes each task at a fraction of the per-task cost."

**Missing move:** Lead-out. The prose describes what each bar chart shows but does not state the conclusion: at what task count the foundation model becomes cheaper, and what the magnitude of the per-task marginal reduction is. The crossover task count and the marginal reduction multiplier appear in the figure caption's computed variables only. The post-float payoff (L2405) pivots to comparing SSL methods on the cost frontier — it does not return to what the figure established.

**Where the takeaway currently lives:** Figure caption (`crossover_tasks_str`, `marginal_compute_reduction_mult_str` computed values).

**Rule:** Figure 🟠 — prose must tell the figure's story; the caption is for float-only readers and does not substitute for body-prose interpretation.

**Rewrite (lead-out, append to the ref paragraph at L2326 or insert as a new sentence after the figure):**

> The crossover point — where the foundation model approach becomes cheaper in aggregate — is around `{python} FoundationCostAmortization.crossover_tasks_str` tasks. Beyond that threshold the marginal cost per task drops by `{python} FoundationCostAmortization.marginal_compute_reduction_mult_str`, which is why organizations building many specialized applications favor a single large pretraining investment over repeated full training runs.

---

### F03 — `fig-technique-decision-tree` (fig 🟠) — def L2671

**Verbatim ref sentence (L2669):**
> "@Tbl-technique-selection maps individual constraints to techniques, but real projects face multiple constraints simultaneously. The decision tree in @fig-technique-decision-tree structures the selection process hierarchically: start by identifying the primary bottleneck, then follow the branches to narrow the field."

**Missing move:** Lead-out / payoff. The prose describes the tree's structure (start at bottleneck, follow branches) but not its insight — that the hierarchical ordering places data availability as the first branch because it gates whether any other technique is feasible, and that combining branches from different paths is the norm for real projects. The post-float payoff (L2739) says only "Each path requires a structured assessment," which is a pivot sentence, not a conclusion drawn from the tree.

**Where the takeaway currently lives:** In the tree's leaf nodes and branch labels themselves.

**Rule:** Figure 🟠 — prose must deliver what the figure demonstrates and why it matters.

**Rewrite (lead-out, replace L2739 or add sentence before the "Step 1" subhead):**

> The tree encodes a priority order that the flat table in @tbl-technique-selection cannot express. Data availability is the first split because it determines whether label-free techniques (self-supervised, synthetic generation) are viable at all; teams with ample unlabeled data face a fundamentally different decision than teams with none. Subsequent splits refine by budget and latency tolerance, and most real projects land on a leaf that combines two or three techniques rather than one.

---

### F04 — `fig-distributed-coreset-architecture` (fig 🟠) — def L3695 🛑

**Verbatim ref sentence (L3669):**
> "**Setup**: @Fig-distributed-coreset-architecture shows the coordinator-worker topology."

**Missing move:** Everything — lead-in, interpretation, and lead-out are all absent. The citation appears inside a callout-example as a single bare pointer with no prose narration of what the topology achieves or why the coordinator-worker split is the design choice. The post-float payoff (L3746) discusses coordination tax generally without connecting it back to the architecture the figure just showed.

**Where the takeaway currently lives:** Figure caption and the callout-example's numbered mechanism steps (L3671-3677), which are callout content, not body prose.

**Rule:** Figure 🟠 — prose must deliver what the figure demonstrates and why it matters; bare pointer with no interpretation fails.

**Rewrite (replace the bare pointer at L3669 with a full prose sentence in the callout body):**

> The coordinator-worker topology in @fig-distributed-coreset-architecture separates the global view (coordinator: FAISS index, final re-ranking, broadcast) from the parallel local work (each worker: embeddings and EL2N scores on its 150 K-image shard). This split avoids the cost of shipping raw embeddings across the full dataset while preserving global selection quality — the coordinator sees all local top-k candidates and resolves the final coreset from that representative pool rather than from every individual sample.

---

### F05 — `tbl-difficulty-scoring` (tbl 🟠) — def L1458

**Verbatim ref sentence (L1447):**
> "The difficulty scorer is a systems choice because it trades probe-compute overhead against ordering quality, as @tbl-difficulty-scoring shows."

**Missing move:** Lead-out. The prose names the trade-off but does not state the conclusion the table encodes — which scorer offers the best balance for most practitioners, and under what constraint each one wins or loses. The post-float payoff (L1524) pivots to curriculum benchmark numbers with no sentence drawing the key lesson from the scoring table.

**Where the takeaway currently lives:** Table cells (Best For column and the method names themselves).

**Rule:** Table 🟠 — prose must deliver the takeaway; "the table shows the trade-off" is not the takeaway.

**Rewrite (append to the ref sentence at L1447 or add a sentence immediately after the table):**

> Loss-based scoring is the general-purpose default: it requires a brief probe-training pass but produces the most reliable difficulty rankings across domains. Self-paced scoring eliminates the probe entirely at the cost of dynamic overhead each epoch, making it attractive when probe-training time is prohibitive. Domain heuristics are free but transfer poorly outside their design domain, so they are best treated as a fast sanity check rather than a primary scorer.

---

### F06 — `tbl-self-supervised-tasks` (tbl 🟠) — def L2224

**Verbatim ref sentence (L2214):**
> "The key insight is that labels represent just one form of supervision. Data structure itself provides rich learning signals that require no human annotation, as @tbl-self-supervised-tasks summarizes."

**Missing move:** Lead-out. The prose states the general claim (labels are not the only signal) and the post-float paragraph (L2226) explains what each pretext task *learns*, but neither sentence draws a conclusion from reading across the table's rows — e.g., that pretext task choice is modality-determined, that text tasks dominate in volume because web-scale unlabeled text is abundant, or that multi-modal alignment requires paired data that is itself a scarcer resource than single-modality corpora.

**Where the takeaway currently lives:** Caption ("each task extracts supervision from data structure").

**Rule:** Table 🟠 — prose must state the conclusion the table encodes, not just set up the concept.

**Rewrite (add a sentence between the table and L2226, or append to the L2214 ref sentence):**

> The modality column reveals the constraint: text tasks can exploit billions of documents with no curation cost, so they dominate pretraining at scale; image tasks require carefully constructed augmentation pairs or masked patches; multi-modal alignment depends on co-occurring image-text pairs, which are abundant online but require non-trivial alignment filtering to be useful. The right pretext task is therefore determined first by data availability, then by the downstream task family.

---

### F07 — `tbl-technique-selection` (tbl 🟠) — def L2667

**Verbatim ref sentence (L2656):**
> "@Tbl-technique-selection provides a decision guide for selecting techniques based on the dominant constraint."

**Missing move:** Lead-in and lead-out are both thin. The prose is a bare pointer with no prose statement of what the table encodes as its key conclusion. The second citation (L2669) is also a pointer that immediately pivots to the decision tree. The table's non-obvious finding — that privacy and large-model-small-data constraints each have their own dedicated technique (synthetic data and distillation) that nothing else covers — is never stated in body prose.

**Where the takeaway currently lives:** Table rows (Constraint and Why columns).

**Rule:** Table 🟠 — "provides a decision guide" is a float-announcer sentence; the prose must state the point.

**Rewrite (replace the bare pointer at L2656):**

> The constraint-to-technique mapping in @tbl-technique-selection has two non-obvious entries worth flagging. Privacy requirements and large-model-small-dataset regimes each have a single viable technique (synthetic data and knowledge distillation, respectively) with no substitute: synthetic data avoids processing real user records, while distillation is the only way to extract learning signal when the labeled set is too small for direct training. The remaining constraints — labeling budget, redundancy, rare classes, and slow convergence — each have multiple viable options, making them judgment calls rather than forced choices.

---

### F08 — `tbl-technique-prerequisites` (tbl 🟠) — def L2759

**Verbatim ref sentence (L2747):**
> "Each approach carries specific requirements that must be met before implementation can begin (@tbl-technique-prerequisites)."

**Missing move:** Lead-out. The prose describes the purpose of the table but does not state its conclusion — which techniques are highest-barrier and therefore the most common failure modes in practice. The post-float payoff (L2763) moves to an ROI formula without drawing the lesson from the prerequisites table.

**Where the takeaway currently lives:** Table rows (Prerequisites column).

**Rule:** Table 🟠 — prose must state the conclusion the table encodes.

**Rewrite (add a sentence after the table, before "Meeting the prerequisites is necessary but not sufficient" at L2763):**

> Active learning carries the highest combined burden: it requires a queryable oracle, a live unlabeled pool, and retraining infrastructure that must keep pace with each labeling round. Self-supervised pretraining requires the largest compute budget. Augmentation is the lowest-barrier entry point, requiring only domain knowledge of the invariances the model should learn. Practitioners who cannot meet a technique's prerequisites consistently are better served by a lower-barrier method than by attempting to build the missing infrastructure in parallel with a training run.

---

### F09 — `tbl-selection-dependencies` (tbl 🟠) — def L3577

**Verbatim ref sentence (L3563):**
> "Data selection techniques, however, introduce selection dependencies\index{Selection Dependency!distributed training} (@tbl-selection-dependencies):"

**Missing move:** Lead-out. The post-float payoff (L3583) says "The selection dependencies admit several architectural solutions, each navigating a different point in the consistency-scalability trade-off space." This pivots to solutions without naming the dominant conclusion from the table — which technique is hardest to distribute and why.

**Where the takeaway currently lives:** Table cells (Distributed Challenge column).

**Rule:** Table 🟠 — prose must state the specific conclusion the table encodes, not pivot immediately to solutions.

**Rewrite (replace or supplement the L3583 payoff sentence):**

> Curriculum learning presents the hardest distributed case because it requires a global difficulty ordering: different workers on different shards may rank the same sample differently, and resolving those rankings requires coordination that grows with cluster size. Deduplication is the most tractable because hash comparisons can be distributed across shards with only a final merge step. Coreset selection and active learning fall in between, both requiring either a coordinator bottleneck or a hierarchical approximation that trades selection quality for coordination cost.

---

### F10 — `tbl-synthetic-mix` (tbl 🟠) — def L2528

**Verbatim ref sentence (L2519):**
> "In practice, the best results often come from mixing synthetic and real data rather than relying on either source alone. @Tbl-synthetic-mix summarizes representative outcomes across different mixing ratios."

**Missing move:** Lead-out. The prose states the general conclusion ("mixing is better") but not the specific conclusion the table encodes — at what mixing ratio performance peaks, and what the slope of the degradation looks like on either side. The post-float payoff (L2614) pivots to a speech dataset scenario with no return to the mixing table.

**Where the takeaway currently lives:** Table rows (Representative Outcome column).

**Rule:** Table 🟠 — prose must deliver the specific row(s) that matter, not just restate the general claim.

**Rewrite (add a sentence after the table, before the section that follows):**

> The table shows that 50/50 mixing is the most common performance peak, but the 80/20 ratio (synthetic-heavy) is the practical starting point because it delivers most of the gain at a fraction of the real-data collection cost. Pure synthetic training consistently underperforms; the critical minimum of real data is typically in the 10-20 percent range, where it supplies enough distribution anchor to prevent domain gap from dominating the learned boundary.

---

### F11 — `fig-active-learning-multiplier` (fig 🟠) — def L1894

**Verbatim ref sentence (L1892):**
> "Compare the two curves in @fig-active-learning-multiplier: active learning shifts the learning curve to the left, achieving target accuracy with far fewer samples than random selection. The curves are illustrative to highlight the qualitative gap."

**Assessment:** The lead-in names the comparison and its direction. The payoff (L1945) quantifies the 4× fewer labeled samples claim and explains why the gap compounds across iterations. This is borderline — the ref sentence describes the relationship without the magnitude, which arrives in L1945 with quantification. The payoff does cover the "so what." ✅ (passes on payoff delivery; flagged here only for completeness — see note below.)

**Note:** L1892 hedges with "The curves are illustrative to highlight the qualitative gap" immediately after citing the figure, which slightly undercuts the lead-out. L1945 recovers with the 4× number, so this meets the standard. No finding issued; included for reviewer awareness.

---

*End of findings.*
