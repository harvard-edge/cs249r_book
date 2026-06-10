# Float Exposition Worklist — `model_compression.qmd` (vol1)

Graded against FLOAT_EXPOSITION_STANDARD.md. Caption, fig-alt, in-figure labels, and code comments do not count toward the prose's job; only running body prose is assessed.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| Equation | 🔴 strict | 3 | 3 | 0 | 0 |
| Algorithm | 🔴 strict | 1 | 1 | 0 | 0 |
| Figure | 🟠 high | 32 | 27 | 5 | 0 |
| Listing | 🟡 medium | 6 | 4 | 2 | 0 |
| Table | 🟠 high | 14 | 10 | 4 | 0 |
| **Total** | | **56** | **45** | **11** | **0** |

---

## Findings (⚠️ only — no 🛑)

---

### Figure findings

---

#### `fig-kd-overview` (figure 🟠) — def L2014

**Ref sentence (L2009):**
> The distillation workflow, laid out in @fig-kd-overview, trains the student model to minimize two loss functions:

**Problem — missing lead-out / interpret move.** The citation at L2009 is purely anticipatory: it points to the figure and then immediately pivots to the loss-function enumeration without ever stating what the figure *demonstrates* or what the reader should take away from seeing the workflow diagram. The key insight — that the dual-loss structure forces the student to simultaneously match teacher uncertainty and ground-truth labels, which is *why* distilled models generalize better than models trained on hard labels alone — lives only in the individual bullet items and the caption. The prose never synthesizes it as a single payoff sentence. Removability test: deleting the figure would leave the two-bullet enumeration intact; the prose would lose nothing pedagogically because the bullets do all the work.

**Missing move:** lead-out / interpret after the float (or strengthening the cite sentence to carry the "so what").

**Where takeaway currently lives:** partly in the enumerated bullets that follow the citation; fully synthesized only in the caption.

**Rule-compliant rewrite (cite sentence only):**

```diff
- The distillation workflow, laid out in @fig-kd-overview, trains the student model to minimize two loss functions:
+ The distillation workflow in @fig-kd-overview shows why the dual-loss objective is necessary: the teacher's soft labels carry inter-class uncertainty that cross-entropy against hard labels would discard, so the student is trained to match both simultaneously.
```

Then keep the enumerated bullets that follow.

---

#### `fig-quantization-roadmap` (figure 🟠) — def L3799

**Ref sentence (L3797):**
> @Fig-quantization-roadmap maps quantization techniques into three progressive tiers based on implementation complexity, resource requirements, and target use cases.

**Problem — bare-pointer citation, no interpret move.** The cite sentence is pure label: it names the tiers in the abstract but delivers no lead-out. The prose paragraph immediately before (L3795) identifies PTQ, QAT, and mixed precision and assigns them rough positions, but it does not state *what the tier structure means for practitioners* — the critical decision rule (reach for the higher tier only when accuracy loss at the lower tier is unacceptable). The payoff paragraph after the figure (L3992) simply re-opens the PTQ section without referencing the roadmap's conclusion. Removability test: removing the figure would not change what the prose teaches.

**Missing move:** interpret / lead-out after the float — the decision rule embedded in the tier structure.

**Where takeaway currently lives:** implied by the tier labels inside the figure; not stated in any surrounding body prose.

**Rule-compliant rewrite (add a lead-out sentence after the figure closes):**

```diff
+ Each tier sets a minimum accuracy budget: PTQ is the starting point, QAT is the fallback when PTQ leaves more than roughly one percent accuracy on the table, and the frontier tier applies only when the deployment environment permits the engineering and accuracy trade-offs those methods require.
```

---

#### `fig-3float` (figure 🟠) — def L3666

**Ref sentence (L3664):**
> Compare the three bit layouts in @fig-3float to see exactly where the bits go—and why the trade-off between precision and numerical range differs so sharply across formats.

**Problem — thin interpret move; payoff lands in wrong float's section.** The cite sentence correctly orients the reader ("see exactly where the bits go") but the promised "so what" — why the trade-off differs — is never stated in body prose near this figure. The preceding paragraph (L3662) explains BF16 vs FP16 in depth, but its argument is complete before the citation appears; the citation is therefore additive decoration rather than essential interpretation. The payoff paragraph after the figure (L3738) pivots abruptly to INT8 without ever synthesizing the FP16 vs BF16 insight that the figure is supposed to confirm. Removability test: removing the figure would leave the preceding explanation fully intact.

**Missing move:** interpret sentence after the figure that names the specific bit-layout consequence the comparison reveals (the exponent-width difference that separates BF16's training safety from FP16's inference-only use).

**Where takeaway currently lives:** in the preceding paragraph (L3662), but that paragraph precedes rather than follows the figure.

**Rule-compliant rewrite (add after figure closes, before the INT8 paragraph):**

```diff
+ The layouts make the training-safety distinction concrete: BF16's 8-bit exponent matches FP32's, so it survives the large gradient magnitudes common early in training, while FP16's narrower 5-bit exponent makes overflow likely at those magnitudes, which is why FP16 is restricted to inference or carefully loss-scaled training.
```

---

#### `fig-color-mapping` (figure 🟠) — def L7432

**Ref sentence (L7430):**
> @Fig-color-mapping shows color-mapped first-layer convolutional kernels grouped by the pattern type each filter has learned, functioning as a sparsity heat map for the learned filters.

**Problem — cite-only; no diagnostic takeaway in body prose.** The sentence names what the figure shows but does not state what the reader should *do with* or *conclude from* it. The surrounding paragraph (L7430) is an enumeration of tool types (error histograms, activation visualizations, color maps) rather than an argument. The key diagnostic point — that a "dead filters" row signals over-pruning, not just sparsity — lives only in the figure's alt-text row label, not in prose. Removability test: the three named tool types would stand without the figure.

**Missing move:** interpret sentence identifying the diagnostic signal the color map reveals and what practitioners should act on when they see it.

**Where takeaway currently lives:** implied by the row labels inside the figure alt-text.

**Rule-compliant rewrite (add at end of the cite sentence's paragraph):**

```diff
- TensorFlow's Quantization Debugger, PyTorch's FX Graph Mode, and TensorRT Inspector provide these capabilities.
+ TensorFlow's Quantization Debugger, PyTorch's FX Graph Mode, and TensorRT Inspector provide these capabilities. The critical diagnostic signal in the color map is not sparsity itself but pattern: uniform gray filters in the final row indicate dead units whose weights never exceeded the pruning threshold, which means the pruning target was too aggressive for those layers and accuracy recovery will require lowering the sparsity budget or widening the fine-tuning schedule.
```

---

#### `fig-sparse-heat-map` (figure 🟠) — def L7723

**Ref sentence (L7721):**
> Sparsity heat maps show sparsity distribution across layers (@fig-sparse-heat-map). Darker regions indicate higher sparsity.

**Problem — float-announcer sentence with no interpret move.** The citing sentence is minimal: it names the visualization type and restates the color legend. No body prose explains *what the pattern in the heatmap reveals* or *what a practitioner should infer from it* (the key result: later layers tolerate more sparsity than early feature-extraction layers, which informs per-layer pruning budget allocation). The payoff paragraph after the figure (L8001) discusses sequential compression in general without referencing the heatmap's specific finding. Removability test: the enumeration of tools at L7721 loses nothing diagnostic without the figure.

**Missing move:** interpret sentence naming the structural finding the heatmap encodes and why it matters for budget allocation.

**Where takeaway currently lives:** only in the caption.

**Rule-compliant rewrite (replace the cite sentence):**

```diff
- Sparsity heat maps show sparsity distribution across layers (@fig-sparse-heat-map). Darker regions indicate higher sparsity.
+ The sparsity heatmap in @fig-sparse-heat-map reveals a consistent architectural pattern: later layers sustain higher sparsity than early feature-extraction layers, which means per-layer pruning budgets should increase with depth rather than be applied uniformly. Trend plots track how this distribution evolves across pruning iterations.
```

---

### Listing findings

---

#### `lst-conv-bn-relu-fusion` (listing 🟡) — def L5126

**Ref sentence (L5124):**
> This ubiquitous Conv-BN-ReLU fusion pattern, illustrated in @lst-conv-bn-relu-fusion, appears in nearly every modern CNN architecture and reduces three memory round-trips to a single kernel launch.

**Problem — mechanism named but design choice unstated.** The cite sentence identifies the pattern and the quantitative saving (three round-trips to one kernel launch), which satisfies the medium threshold for mechanism. However, it does not name what in the code the reader should look at to understand *why* fusion achieves this: specifically, that the intermediate tensors written after conv and batch-norm in the unfused path are never materialized to memory in the fused path. The payoff paragraph at L5199 delivers the numbers ("6 transfers to 2") but is positioned well after the code, leaving the reader to make the connection alone. The cite sentence reads as a float-announcer; the design choice (intermediate-tensor elimination) is absent.

**Missing move:** orient the reader toward the key structural feature in the code (the suppressed intermediate writes).

**Where takeaway currently lives:** in the payoff paragraph (L5199), but that paragraph provides the quantitative result rather than the structural explanation of the mechanism.

**Rule-compliant rewrite:**

```diff
- This ubiquitous Conv-BN-ReLU fusion pattern, illustrated in @lst-conv-bn-relu-fusion, appears in nearly every modern CNN architecture and reduces three memory round-trips to a single kernel launch.
+ The Conv-BN-ReLU fusion pattern in @lst-conv-bn-relu-fusion eliminates the intermediate tensor writes that separate layers produce in the unfused path: compare the three-launch unfused block against the fused kernel below it, where conv output feeds directly into the batch-norm fold without being written back to memory. This suppression of intermediate writes is the mechanism behind the 6-to-2 reduction in memory transfers.
```

---

#### `lst-pytorch_pruning` (listing 🟡) — def L7401

**Ref sentence (L7399):**
> The same framework-level pattern applies to pruning. The API owns the weight tensor, applies a mask or structured removal rule, and records which parameters were removed so that subsequent fine-tuning and export operate on the intended model. @Lst-pytorch_pruning illustrates both unstructured and structured pruning.

**Problem — no framing of the mechanism or design choice to look for.** The cite sentence names what the listing illustrates (both types of pruning) but does not tell the reader what to look for or what design choice the code demonstrates. The standard for listings requires that the reader know "the mechanism it embodies and what the reader should notice." The key structural contrast in the code — that unstructured pruning applies a floating-point mask leaving a dense tensor, while structured pruning physically removes filter rows — is the mechanism, but it is not named in prose. The payoff at L7418 discusses "repeatability" as the engineering value, which is a process claim rather than a code-mechanism claim.

**Missing move:** orient the reader toward the structural contrast between the two pruning calls in the code.

**Where takeaway currently lives:** only deducible by reading the code itself.

**Rule-compliant rewrite (replace the cite sentence):**

```diff
- @Lst-pytorch_pruning illustrates both unstructured and structured pruning.
+ @Lst-pytorch_pruning contrasts the two approaches in a single code block: the unstructured call applies a magnitude mask, leaving the weight tensor at its original shape with zeros in place of pruned values, while the structured call removes filter rows entirely, producing a smaller dense weight matrix. That shape difference is why structured pruning accelerates inference on standard hardware while unstructured pruning does not.
```

---

### Table findings

---

#### `tbl-deployment-scenarios` (table 🟠) — def L303

**Ref sentence (L295):**
> @Tbl-deployment-scenarios summarizes the key constraints across deployment environments.

**Problem — "summarizes" pointer; no conclusion from the cells delivered in body prose.** The cite sentence is a bare pointer with no takeaway. The payoff paragraph (L307) does deliver useful follow-through for cloud and mobile/edge separately, which partially compensates. However, the critical row-level conclusion the table encodes — that TinyML's power envelope (milliwatt scale) demands energy-first optimization in a way that cloud and mobile do not, making size and energy the dominant criteria rather than latency — is not stated anywhere in prose. The L295 sentence adds nothing beyond pointing; the standard requires at minimum naming the load-bearing contrast.

**Missing move:** the cite sentence should name the decision-driving contrast across rows rather than just pointing.

**Where takeaway currently lives:** partially in L307 (cloud and mobile), but the TinyML row's distinctive constraint is caption-only.

**Rule-compliant rewrite:**

```diff
- @Tbl-deployment-scenarios summarizes the key constraints across deployment environments.
+ @Tbl-deployment-scenarios shows that the binding constraint shifts as the deployment tier shrinks: cloud prioritizes throughput and cost, mobile must fit device memory within real-time latency budgets, and TinyML operates at milliwatt power levels where energy per inference dominates both latency and size concerns.
```

---

#### `tbl-pruning` (table 🟠) — def L1237

**Ref sentence (L1225):**
> @Tbl-pruning formalizes these comparisons across the dimensions that matter most for deployment.

**Problem — pointer with no extracted conclusion.** The preceding paragraph (L1225) does deliver the key contrasts verbally (irregular sparsity, structured run-efficiency, dynamic flexibility) and is strong lead-in. The cite sentence, however, is purely "see table" and does not name the conclusion the table drives — which is the hardware-compatibility row: unstructured pruning's irregular sparsity cannot be exploited without specialized sparse kernels, and that hardware dependency is the reason structured pruning dominates production despite lower compression ratios. The payoff paragraph (L1241) pivots to iterative vs. one-shot strategies without extracting the table's decision signal.

**Missing move:** a lead-out sentence that extracts the table's decision-driving row (hardware compatibility) rather than just pointing.

**Where takeaway currently lives:** stated implicitly in the preceding paragraph but not reinforced as a conclusion drawn from the table.

**Rule-compliant rewrite:**

```diff
- @Tbl-pruning formalizes these comparisons across the dimensions that matter most for deployment.
+ @Tbl-pruning formalizes those comparisons and makes the hardware-compatibility row the decisive one: unstructured pruning cannot deliver inference speedup on standard hardware because accelerators designed for dense matrix operations cannot skip individual zero-valued multiplications without specialized sparse execution kernels.
```

---

#### `tbl-nas-strategies` (table 🟠) — def L2711

**Ref sentence (L2703):**
> @Tbl-nas-strategies compares the trade-offs between search cost, architectural diversity, and optimality guarantees for each approach.

**Problem — lists the comparison dimensions without delivering the conclusion.** The cite sentence names what the table compares but does not state what a practitioner should conclude from the comparison. The key result in the cells is that gradient-based search (DARTS) reduces search cost by two to three orders of magnitude versus RL-based NAS (1 to 4 GPU-days vs. 400 to 1,000), which changes NAS from a hyperscaler tool into a production-accessible technique. That conclusion is the "so what" the table encodes, but it is never stated in body prose: the payoff paragraph (L2713) describes RL-based NAS mechanics without referencing the comparative cost result.

**Missing move:** lead-out sentence naming the magnitude of the RL-to-gradient-based cost gap and its deployment implication.

**Where takeaway currently lives:** in the table cells; not in prose.

**Rule-compliant rewrite:**

```diff
- @Tbl-nas-strategies compares the trade-offs between search cost, architectural diversity, and optimality guarantees for each approach.
+ @Tbl-nas-strategies shows that the choice of search strategy is primarily a compute budget decision: gradient-based methods (1 to 4 GPU-days) are two to three orders of magnitude cheaper than reinforcement learning (400 to 1,000 GPU-days), which is what moved NAS from a hyperscaler-only tool into a practical option for teams with limited infrastructure.
```

---

#### `tbl-quantization_methods` (table 🟠) — def L4708

**Ref sentence (L4700):**
> @Tbl-quantization_methods compares post-training quantization, quantization-aware training, and dynamic quantization, each offering distinct strengths and trade-offs for different deployment scenarios.

**Problem — "compares" pointer with no conclusion extracted.** The cite sentence names the rows but delivers no conclusion. The table's actual insight — the "reach for it when" column makes the decision rule explicit and the key result is that PTQ is almost always the right starting point because its cost is negligible and accuracy loss is acceptable in the majority of production scenarios, with QAT reserved for the minority needing sub-one-percent degradation — is not stated anywhere in surrounding body prose. The payoff paragraph (L4710) says "highlights the diverse strategies available," which is a summary-of-contents statement, not a conclusion.

**Missing move:** lead-out sentence naming the decision rule the "reach for it when" column encodes.

**Where takeaway currently lives:** in the table cells ("reach for it when" column); not in prose.

**Rule-compliant rewrite:**

```diff
- @Tbl-quantization_methods compares post-training quantization, quantization-aware training, and dynamic quantization, each offering distinct strengths and trade-offs for different deployment scenarios.
+ @Tbl-quantization_methods places the three approaches on a cost-accuracy ladder: PTQ costs nothing in training time and is the correct starting point for any deployment; QAT is the fallback when PTQ leaves more than roughly one percent accuracy on the table and the training budget allows it; dynamic quantization applies when activation ranges vary too widely across inputs for a fixed calibration range to capture.
```

---

*End of findings.*
