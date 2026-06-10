# Float-explanation worklist — model_compression.qmd (vol1)

## Summary

| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 32 | 28 | 4 | 0 |
| table | 14 | 14 | 0 | 0 |
| listing | 6 | 6 | 0 | 0 |
| algorithm | 1 | 1 | 0 | 0 |
| equation | 3 | 3 | 0 | 0 |
| **total** | **56** | **52** | **4** | **0** |

---

## Findings (⚠️ only — no 🛑 dead-ends or orphans)

---

### ⚠️ `fig-kd-overview` — def L2014  (Thin — float-announcer colon)

- **Caption:** Knowledge Distillation Workflow: An input sample passes through both the teacher and the student network. The teacher produces soft labels via temperature-scaled softmax, while the student output is compared against both the soft labels (distillation loss) and the hard labels (student loss).
- **Ref(s):** L2009 `@fig-kd-overview`: "The distillation workflow, laid out in @fig-kd-overview, trains the student model to minimize two loss functions:"
- **Context checked:** ref ✗ (float-announcer colon, no payoff on the dual-path structure) · prev ¶ ✗ (inside :::, no prose) · next ¶ lists the two losses but could accompany any prose · caption ✓ (names the two paths) · payoff ✗ (footnote about KL divergence, not about what the figure's two-path layout reveals)
- **Problem:** The colon at the end of the ref sentence turns the figure into a bullet-list introducer. The reader never learns *why* the two-path structure matters — that the distillation loss and student loss serve different roles and must be balanced by a mixing coefficient. The figure carries information (separate loss heads, temperature-scaled softmax in one path) that the prose never explains.
- **Suggested rewrite (flag-only):**
  ```diff
  - The distillation workflow, laid out in @fig-kd-overview, trains the student model to minimize two loss functions:
  + The dual-path structure in @fig-kd-overview makes visible why training requires two separate loss terms rather than one. The teacher path applies temperature-scaled softmax before producing soft labels, while the student path feeds into both a distillation loss against those soft labels and a standard loss against the ground-truth hard labels. Balancing these two terms controls how much of the teacher's dark knowledge the student absorbs versus how tightly it tracks the labeled training set.
  ```

---

### ⚠️ `fig-quantization-roadmap` — def L3799  (Thin — caption restatement)

- **Caption:** Quantization Complexity Roadmap: Three progressive tiers of quantization techniques, from foundational approaches suitable for quick deployment to research frontier methods for extreme resource constraints, reflecting increasing implementation effort, resource requirements, and potential accuracy trade-offs.
- **Ref(s):** L3797 `@Fig-quantization-roadmap`: "@Fig-quantization-roadmap maps quantization techniques into three progressive tiers based on implementation complexity, resource requirements, and target use cases."
- **Context checked:** ref ✗ (identical to caption, no practitioner takeaway) · prev ¶ ✓ (names PTQ/QAT/mixed-precision in one line each) · next ¶ ✗ (figure definition) · caption ✗ (restates tiers without guidance) · payoff ✗ (jumps to PTQ section prose at L3992, far downstream, no explicit tie-back)
- **Problem:** The ref sentence is a one-for-one restatement of the caption. Neither the ref nor the payoff tells the reader what the roadmap is *for*: that the tiers encode a starting-point heuristic (begin at PTQ, escalate to QAT only if accuracy loss is unacceptable, treat INT4/binary as last resort). The figure communicates a decision tree; the prose treats it as a taxonomy list.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Fig-quantization-roadmap maps quantization techniques into three progressive tiers based on implementation complexity, resource requirements, and target use cases.
  + @Fig-quantization-roadmap organizes quantization techniques as a decision ladder. The foundation tier (PTQ, INT8) is the right starting point for most deployments: no retraining, minutes to apply, and accuracy loss is typically within 1 percent. Teams escalate to the production tier (QAT, mixed precision) only when PTQ's accuracy budget is exhausted and training time is available. The research frontier tier (INT4, binary, ternary) applies only under the most extreme resource constraints and carries the highest risk of accuracy degradation.
  ```

---

### ⚠️ `fig-color-mapping` — def L7432  (Thin — what to do with it)

- **Caption:** Convolutional Kernel Weights: Color mapping reveals learned feature patterns in convolutional filters. First-layer filters learn oriented edges, color blobs, and frequency patterns; analyzing weight distributions helps diagnose issues like dead or saturated filters.
- **Ref(s):** L7430 `@Fig-color-mapping`: "@Fig-color-mapping shows color-mapped first-layer convolutional kernels grouped by the pattern type each filter has learned, functioning as a sparsity heat map for the learned filters."
- **Context checked:** ref ✗ (describes what it shows but gives no diagnostic action) · prev ¶ lists three visualization types without connecting any to a specific compression failure mode · next ¶ ✗ (names tools: TensorFlow Quantization Debugger, etc.) · caption ✓ (mentions dead/saturated filters) · payoff ✗ (L7721 points to `fig-sparse-heat-map`, unrelated)
- **Problem:** The figure's diagnostic value — that a row of uniform (dead) filters signals over-pruning of a layer that should not have been a compression target — is never stated. The prose treats the figure as an illustration of a visualization type rather than an explanation of what a practitioner should *conclude* from a specific pattern.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Fig-color-mapping shows color-mapped first-layer convolutional kernels grouped by the pattern type each filter has learned, functioning as a sparsity heat map for the learned filters.
  + @Fig-color-mapping groups first-layer convolutional kernels by the pattern each has learned. The bottom row of uniform filters is the diagnostic signal: a filter that has learned nothing (dead filter) is a pruning candidate, while a filter that was alive before compression and dead after it signals that the pruning threshold was set too aggressively for that layer. Structured pruning decisions should be validated against this kind of visualization, not only against aggregate accuracy metrics.
  ```

---

### ⚠️ `fig-sparse-heat-map` — def L7723  (Thin — no implication stated)

- **Caption:** Sparsity Distribution: Darker shades indicate higher sparsity where more weights were removed. The heatmap reveals how pruning affects different layers nonuniformly, with later layers typically exhibiting higher sparsity than early feature-extraction layers.
- **Ref(s):** L7721 `@fig-sparse-heat-map`: "Sparsity heat maps show sparsity distribution across layers (@fig-sparse-heat-map). Darker regions indicate higher sparsity."
- **Context checked:** ref ✗ (two sentences; repeats caption's color key without implication) · prev ¶ ✗ (ends the `fig-color-mapping` payoff; unrelated) · next ¶ ✗ (figure definition) · caption ✓ (notes nonuniform distribution across layers) · payoff ✗ (L8001 is about sequential application of techniques, no tie-back)
- **Problem:** The ref says what dark means but not what a practitioner *learns* from the nonuniform distribution. The implication — that if early feature-extraction layers show high sparsity, the model has almost certainly been over-pruned in a way that will hurt accuracy on inputs requiring low-level features — is never stated. The heatmap's job is to reveal layer-level pruning mistakes; that diagnostic role is absent.
- **Suggested rewrite (flag-only):**
  ```diff
  - Sparsity heat maps show sparsity distribution across layers (@fig-sparse-heat-map). Darker regions indicate higher sparsity.
  + @fig-sparse-heat-map reveals the layer-level distribution of sparsity across a pruned network. Uniform darkness across all layers indicates that the pruning schedule treated every layer equivalently, which is rarely optimal: early feature-extraction layers typically have fewer redundant weights than later classification layers. Concentrated darkness in early layers, or a sudden shift from light to dark within a single layer block, signals that the per-layer pruning budget should be rebalanced before deployment.
  ```
