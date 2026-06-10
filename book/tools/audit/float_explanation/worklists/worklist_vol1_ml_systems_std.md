# Float exposition eval — ml_systems.qmd (vol1)
Standard: FLOAT_EXPOSITION_STANDARD.md (caption excluded from prose budget)

## Summary
| type | level | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|---|
| equation | 🔴 | 7 | 6 | 1 | 0 |
| algorithm | 🔴 | 0 | — | — | — |
| table | 🟠 | 14 | 8 | 6 | 0 |
| figure | 🟠 | 13 | 12 | 1 | 0 |
| listing | 🟡 | 0 | — | — | — |
| **total** | | **34** | **26** | **8** | **0** |

---

## Findings (⚠️ only)

### ⚠️ `eq-memory-wall` (equation 🔴) — def L431
- **Ref (body prose):** `@Eq-memory-wall quantifies this divergence: processors have doubled in compute capacity roughly every …`
- **Missing move:** Lead-in (the sentence introducing the equation is a float-announcer fragment: "The memory wall reflects the widening bandwidth gap:" feeds directly into the display equation with a fragment + colon, establishing no prior question or claim the equation resolves). The payoff paragraph does deliver symbol meaning and consequence, but the lead-in fails the anticipation move at the strictest equation level.
- **Suggested rewrite (no em-dash/hyphen, one colon per para, content leads):**
  ```diff
  - The memory wall\index{Memory Wall!bandwidth divergence}\index{Memory Wall!compute-memory gap} [@wulf1995] reflects the widening bandwidth[^fn-bandwidth-memory-wall] gap:
  - $$\frac{\text{Compute Growth Rate}}{\text{Memory Bandwidth Growth Rate}} \approx \frac{1.6}{1.2} \approx 1.33$$ {#eq-memory-wall}
  + The memory wall\index{Memory Wall!bandwidth divergence}\index{Memory Wall!compute-memory gap} [@wulf1995] arises because compute performance and memory bandwidth have not grown at the same pace. @Eq-memory-wall formalizes this divergence, expressing the ratio of their annual growth rates:
  + $$\frac{\text{Compute Growth Rate}}{\text{Memory Bandwidth Growth Rate}} \approx \frac{1.6}{1.2} \approx 1.33$$ {#eq-memory-wall}
  ```

---

### ⚠️ `fig-cloud-edge-TinyML-comparison` (figure 🟠) — def L71
- **Ref (body prose):** `@fig-cloud-edge-TinyML-comparison maps where each paradigm sits along that centralization axis.`
- **Missing move:** Lead-out. The citation sentence only says the figure "maps" positions — it does not state what the figure *demonstrates* about the relationship (the insight a reader should take away). The payoff at L245 immediately pivots to "@Tbl-deployment-paradigms-overview makes these trade-offs quantitative" without ever stating the figure's point in prose.
- **Suggested rewrite:**
  ```diff
  - Each paradigm functions as a distinct operating envelope, defined by how much power, memory, and network connectivity is available. Every ML application must fit within at least one of these envelopes, and that fit determines which algorithms, hardware, and engineering trade-offs apply. The envelopes span a continuous spectrum from centralized cloud infrastructure to distributed ultra-low-power devices, and @fig-cloud-edge-TinyML-comparison maps where each paradigm sits along that centralization axis.
  + Each paradigm functions as a distinct operating envelope, defined by how much power, memory, and network connectivity is available. Every ML application must fit within at least one of these envelopes, and that fit determines which algorithms, hardware, and engineering trade-offs apply. @Fig-cloud-edge-TinyML-comparison shows that moving left along the centralization axis trades unlimited compute for lower latency and tighter power budgets: cloud ML anchors the high-compute, high-latency extreme, while TinyML anchors the always-on, milliwatt extreme. No single position dominates — each paradigm is optimal for a different class of application constraint.
  ```

---

### ⚠️ `tbl-ml-systems-lighthouse-archetypes` (table 🟠) — def L639
- **Ref (body prose):** `Throughout this book, we use the five Lighthouse Models summarized in @tbl-ml-systems-lighthouse-archetypes: concrete workloads that span the deployment spectrum and isolate distinct system bottlenecks.`
- **Missing move:** Lead-out. The citation only describes what the table contains ("concrete workloads that span the deployment spectrum"). The key insight the table encodes — which archetype bottleneck each lighthouse isolates, and why those five were chosen — is left entirely in the cells. The payoff at L643 says only "we analyze these five Lighthouse Models in turn," a forward pointer with no conclusion.
- **Suggested rewrite:**
  ```diff
  - Throughout this book, we use the five Lighthouse Models summarized in @tbl-ml-systems-lighthouse-archetypes: concrete workloads that span the deployment spectrum and isolate distinct system bottlenecks. @Sec-network-architectures provides full architectural details and model biographies.
  + Throughout this book, we return to five Lighthouse Models, each chosen because it isolates a different binding constraint. @Tbl-ml-systems-lighthouse-archetypes pairs each lighthouse with its archetype: ResNet-50 is the Compute Beast (arithmetic-bound training), GPT-2/Llama is the Bandwidth Hog (weight-streaming inference), DLRM is the Sparse Scatter (irregular memory access), MobileNetV2 is the efficient Compute Beast constrained by mobile power, and Keyword Spotting is the Tiny Constraint operating at microcontroller limits. Together they span every iron-law regime, so optimizing any one of them requires a different lever. @Sec-network-architectures provides full architectural details and model biographies.
  ```

---

### ⚠️ `tbl-ml-systems-paradigm-bottlenecks` (table 🟠) — def L903
- **Ref (body prose):** `The dominant term varies by paradigm (see @tbl-ml-systems-paradigm-bottlenecks), shifting the optimization strategy entirely.`
- **Missing move:** Lead-out. The citation is a bare pointer with a weak payoff clause ("shifting the optimization strategy entirely"). Which term dominates in which paradigm, and what that means for engineering, is left entirely in the table cells. The standard requires that the "key result" sentence appear in prose.
- **Suggested rewrite:**
  ```diff
  - Here, $O$ represents total operations, $R_{\text{peak}}$ is peak compute rate, $\eta_{\text{hw}}$ is hardware utilization efficiency, $D_{\text{vol}}$ is data volume, $\text{BW}$ is memory bandwidth, $\text{BW}_{\text{IO}}$ is I/O bandwidth (storage or network), and $L_{\text{lat}}$ is fixed overhead. The equation identifies which resource (compute, memory, or I/O) limits performance. The dominant term varies by paradigm (see @tbl-ml-systems-paradigm-bottlenecks), shifting the optimization strategy entirely.
  + Here, $O$ represents total operations, $R_{\text{peak}}$ is peak compute rate, $\eta_{\text{hw}}$ is hardware utilization efficiency, $D_{\text{vol}}$ is data volume, $\text{BW}$ is memory bandwidth, $\text{BW}_{\text{IO}}$ is I/O bandwidth (storage or network), and $L_{\text{lat}}$ is fixed overhead. The equation identifies which resource limits performance. @Tbl-ml-systems-paradigm-bottlenecks maps the answer by paradigm: cloud training is compute-bound (the accelerator is the bottleneck), LLM inference is memory-bound (weight streaming from HBM dominates), and TinyML is memory-fit-constrained (the model must stay on-chip). Because the bottleneck shifts, so does the optimization lever — accelerator upgrades help cloud training but yield zero speedup for memory-bound LLM inference.
  ```

---

### ⚠️ `tbl-dam-phase` (table 🟠) — def L930
- **Ref (body prose):** `@Tbl-dam-phase shows how each component behaves differently depending on whether the system is training (learning patterns) or serving (applying them).`
- **Missing move:** Lead-out. The citation describes the table's structure ("shows how each component behaves differently") but does not state the table's conclusion. The key decision insight — that the same model creates entirely different system bottlenecks depending on the phase, so optimizing training infrastructure wastes resources during inference and vice versa — is left in the cells. The payoff at L936 moves immediately to a worked example without stating the table's point.
- **Suggested rewrite:**
  ```diff
  - This shift between training and inference is critical to understand. Recall the D·A·M taxonomy from @tbl-dam-taxonomy: every ML system comprises Data, Algorithm, and Machine. @Tbl-dam-phase shows how each component behaves differently depending on whether the system is training (learning patterns) or serving (applying them).
  + This shift between training and inference is critical to understand. Recall the D·A·M taxonomy from @tbl-dam-taxonomy: every ML system comprises Data, Algorithm, and Machine. @Tbl-dam-phase makes the practical consequence explicit: the same model optimized for training (high-bandwidth clusters, large batches, compute-throughput maximized) is over-engineered for inference (single samples, latency-critical, power-constrained devices). Infrastructure built for one phase is mismatched to the other, which is why training and serving pipelines are designed and priced independently.
  ```

---

### ⚠️ `tbl-ml-systems-cloud-tco` (table 🟠) — def L1744
- **Ref (body prose):** `For the worked comparison, @tbl-ml-systems-cloud-tco itemizes the annual GPU, network, load-balancer, and observability costs of an illustrative cloud implementation under public list pricing …`
- **Missing move:** Lead-out for the cloud TCO table specifically. The citation describes what the table itemizes but does not state what the table reveals — which cost component dominates, and what that means for the cloud-vs-edge decision. The payoff at L1768 gives the break-even result but does not name the dominant cloud cost driver in prose (that information lives only in the cells and caption).
- **Suggested rewrite:**
  ```diff
  - For the worked comparison, @tbl-ml-systems-cloud-tco itemizes the annual GPU, network, load-balancer, and observability costs of an illustrative cloud implementation under public list pricing, and @tbl-ml-systems-edge-tco itemizes the corresponding hardware, power, cooling, network, and DevOps labor costs of an on-premise NVIDIA T4 implementation.
  + For the worked comparison, @tbl-ml-systems-cloud-tco itemizes the annual costs of an illustrative cloud implementation: GPU instance hours dominate, with egress and load-balancing adding smaller but non-trivial recurring charges. @Tbl-ml-systems-edge-tco itemizes the corresponding on-premise costs: DevOps labor is the dominant line item, not hardware, because a single full-time engineer allocated to maintaining the deployment outweighs the amortized hardware cost.
  ```

---

### ⚠️ `tbl-ml-systems-edge-sizing-requirements` (table 🟠) — def L2439
- **Ref (body prose):** `@Tbl-ml-systems-edge-sizing-requirements cascades the per-store inference rate through YOLOv8-nano's per-frame FLOP count to yield the throughput target.`
- **Missing move:** Lead-out. The citation describes the table's calculation method but does not state the result — what throughput target the table produces, and whether that number is achievable on the candidate hardware. The payoff at L2445 immediately pivots to "@Tbl-ml-systems-edge-hardware-options scores three candidates against the throughput target," again without stating the target.
- **Suggested rewrite:**
  ```diff
  - @Tbl-ml-systems-edge-sizing-requirements cascades the per-store inference rate through YOLOv8-nano's per-frame FLOP count to yield the throughput target.
  + @Tbl-ml-systems-edge-sizing-requirements cascades the per-store inference rate through YOLOv8-nano's per-frame FLOP count, arriving at a sustained throughput target that a single low-power USB accelerator cannot meet on its own. That number, not the model accuracy, is the binding constraint that determines hardware selection.
  ```

---

### ⚠️ `tbl-big_vs_tiny` (table 🟠) — def L3447
- **Ref (body prose):** `@Tbl-big_vs_tiny provides this comparison across fourteen dimensions, from compute power and latency to cost and deployment speed.`
- **Missing move:** Lead-out. The citation is a bare pointer ("provides this comparison") with no stated conclusion from the table. The payoff at L3449 states a general inverse relationship (privacy vs. compute) that is drawn from the table, but does not call out the specific rows or columns a system architect would key on to make a deployment decision. A reader needs to know which dimensions most differentiate paradigms for practical selection.
- **Suggested rewrite:**
  ```diff
  - @Tbl-big_vs_tiny provides this comparison across fourteen dimensions, from compute power and latency to cost and deployment speed.
  + @Tbl-big_vs_tiny compares all four paradigms across fourteen dimensions. The rows that do the most work for deployment decisions are latency (100+ ms for cloud vs. 1–10 ms for TinyML, a 100-fold gap set by physics), privacy (cloud requires data transmission while TinyML keeps data on-chip), and cost structure (cloud charges per query at runtime, TinyML charges up-front in hardware and development). These three dimensions together eliminate paradigms faster than any other combination of constraints.
  ```
