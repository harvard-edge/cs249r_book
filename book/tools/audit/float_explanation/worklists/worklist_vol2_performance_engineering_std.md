# Float Exposition Audit — `performance_engineering.qmd` (vol2)

Graded against FLOAT_EXPOSITION_STANDARD.md. Caption, fig-alt, in-figure labels, code comments, and callout interiors are excluded from the prose's job. Only running body prose counts.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|------|-------|--------|----|----|-----|
| Equation | 🔴 strict | 3 | 0 | 1 | 2 |
| Algorithm | 🔴 strict | 1 | 1 | 0 | 0 |
| Figure | 🟠 high | 8 | 5 | 3 | 0 |
| Table | 🟠 high | 4 | 3 | 1 | 0 |
| **Total** | | **16** | **9** | **5** | **2** |

---

## Findings (⚠️ and 🛑 only)

---

### `eq-iron-law-perf` (🔴 Equation) — def L88

**Grade: 🛑 Fails**

**Ref sentence (L119, inside callout only):**
> Which term in @eq-iron-law-perf are you reducing?

The only prose citations of this equation appear inside a `.callout-checkpoint` block (L119–122), which is a callout interior and does not count as body prose. The equation is introduced at L88 with a one-sentence lead-in:

> The iron law of ML system performance decomposes execution time into three competing costs:

There is no where-clause in body prose: $D_{\text{vol}}$, BW, $O$, $R_{\text{peak}}$, $\eta_{\text{hw}}$, and $L_{\text{lat}}$ are never defined before the equation in body prose. Symbol meanings appear only at L96, eight lines after the equation, inside a dense paragraph that never names the equation by cross-reference. That paragraph describes the terms informally ("compute fraction," "data fraction," "overhead term") but never links them back to the display equation's symbols or to `@eq-iron-law-perf`. No stated consequence or regime implication ("if the data term dominates, increasing FLOP/s yields nothing") appears in body prose adjacent to the equation.

**What is missing:** A where-clause naming every symbol immediately after the display equation, and a follow-on sentence stating the consequence (which term dominates determines which optimization applies).

**Where the takeaway currently lives:** Partially in L96 body prose (informal description of terms) and in callout checkboxes (L119–122), neither of which counts.

**Rule-compliant diff rewrite (insert after L88, before "In overlapped execution…"):**

```diff
+Here $D_{\text{vol}}$ is the total volume of data transferred from memory (bytes), $\text{BW}$ is
+the available memory bandwidth (bytes per second), $O$ is the total floating-point operation count,
+$R_{\text{peak}}$ is peak hardware throughput (FLOP/s), $\eta_{\text{hw}}$ is the hardware
+efficiency factor (a value between 0 and 1 capturing utilization), and $L_{\text{lat}}$ captures
+all remaining overhead: kernel launch gaps, synchronization, and communication. The three-term
+decomposition is actionable because each term has a different remedy. A workload where the data
+term $D_{\text{vol}}/\text{BW}$ dominates is memory-bound, and increasing $R_{\text{peak}}$
+(buying faster compute) yields no improvement until that term shrinks. A workload where the compute
+term dominates is compute-bound, and memory optimizations yield diminishing returns. The overhead
+term is the silent third cost: tens of microseconds of Python dispatch per kernel launch accumulate
+into a large fraction of wall-clock time when thousands of short operations run per training step.
```

---

### `eq-performance-roofline` (🔴 Equation) — def L332

**Grade: 🛑 Fails**

**Ref sentence (L328):**
> For a given accelerator with peak compute $R_{\text{peak}}$ (in FLOP/s) and peak memory bandwidth $\text{BW}$ (in bytes/s), @eq-performance-roofline gives the achievable performance of a workload with arithmetic intensity $I$ (in FLOP/byte):

The cite sentence names all three symbols with units — that satisfies the where-clause. However, the equation is a display of `min(R_peak, BW × I)`, and body prose never states what the min() operation *means*: that below the ridge point, performance scales linearly with arithmetic intensity (bandwidth-limited), while above it, performance is flat at peak compute (compute-limited). This interpretation appears only in the payoff for `@Eq-ridge-point` at L340, which covers that behavior *after* the ridge-point equation is introduced. There is no body-prose interpretation sentence immediately after or adjacent to `@eq-performance-roofline` itself that states what the two cases of the min() represent.

**What is missing:** A one-sentence interpretation of what the two cases of the min() mean in terms of workload behavior, placed in body prose between the equation and the ridge-point equation that follows.

**Where the takeaway currently lives:** Implicit, deferred to L340 in the lead-out for `@Eq-ridge-point`.

**Rule-compliant diff rewrite (insert after L332, before "@Eq-ridge-point locates…"):**

```diff
+Below the ridge, where $\text{BW} \times I < R_{\text{peak}}$, performance scales linearly with
+arithmetic intensity because the workload is exhausting memory bandwidth before saturating compute.
+Above the ridge, the min() clamps at $R_{\text{peak}}$: the arithmetic units are the bottleneck
+and loading data faster would not improve throughput.
```

---

### `eq-ridge-point` (🔴 Equation) — def L338

**Grade: ⚠️ Partial**

**Ref sentence (L334):**
> @Eq-ridge-point locates the ridge point where these two limits intersect:

Pure float-announcer: the sentence names what the equation does ("locates the ridge point") but offers no prior framing of why the ridge point matters before the equation appears. The payoff paragraph (L340) delivers the full interpretation (memory-bound vs compute-bound regimes, H100 worked example at L346) in body prose. The announce-then-interpret structure means the reader hits the equation before understanding its significance.

**What is missing:** A lead-in sentence that establishes why the ridge point matters before presenting the formula, not after.

**Where the takeaway currently lives:** L340 payoff paragraph, which follows the equation.

**Rule-compliant diff rewrite (replace L334):**

```diff
-@Eq-ridge-point locates the ridge point where these two limits intersect:
+The point at which those two regimes meet determines the hardware's fundamental efficiency boundary.
+At that crossing, $R_{\text{peak}} = \text{BW} \times I_{\text{ridge}}$, so the ridge arithmetic
+intensity is:
```

---

### `fig-iron-law-flowchart` (🟠 Figure) — def L128

**Grade: ⚠️ Partial**

**Ref sentence (L134):**
> The central lesson of @fig-iron-law-flowchart is that profiling must precede optimization: applying operator fusion to a compute-bound workload, or precision engineering to an overhead-bound one, yields zero improvement regardless of implementation quality.

The payoff sentence at L134 is strong and delivers the takeaway. However, the pre-float lead-in (L126) is thin:

> The same diagnostic process can be codified as a decision flowchart, mapping each bottleneck to its corresponding optimization technique.

This sentence only describes the figure's content, not the *reason* the reader should care about studying it before they see it. The standard requires the prose to establish the *question or tension* the figure resolves before it appears. The tension here (how to avoid misdiagnosis when multiple bottleneck types coexist) is established nowhere in the pre-float prose.

**What is missing:** A pre-float sentence that names the diagnostic risk (misdiagnosis wastes effort and can cause outages), so the figure appears as the resolution of an established problem rather than an incidental illustration.

**Where the takeaway currently lives:** L134 payoff paragraph (after the float).

**Rule-compliant diff rewrite (replace L126):**

```diff
-The same diagnostic process can be codified as a decision flowchart, mapping each bottleneck to its corresponding optimization technique.
+A profiling workflow that branches incorrectly wastes engineering effort and can ship a latency
+regression into production. The following flowchart makes the diagnostic path explicit: starting
+from the observed symptoms, it routes through the right sequence of checks to reach the correct
+bottleneck classification before any optimization is applied.
```

---

### `fig-roofline-model` (🟠 Figure) — def L342

**Grade: ⚠️ Partial**

**Ref sentence (L340):**
> Workloads with $I < I_{\text{ridge}}$ are memory-bound: their performance is limited by how fast data can be loaded, not how fast it can be processed. Workloads with $I > I_{\text{ridge}}$ are compute-bound: the arithmetic units are the bottleneck. @Fig-roofline-model illustrates this relationship graphically.

The cite is placed at the end of a sentence that has already stated the key relationship, so the figure reference is a pure float-announcer ("illustrates this relationship graphically"). Body prose establishes the relationship correctly, but the sentence citing the figure adds no demonstration claim: it does not state what the figure *shows specifically* (the log-log structure, where specific ML operations fall on the plot, or why the visual representation is more informative than the prose statement). The standard requires the prose to tell the figure's story, not just point at it.

**What is missing:** A sentence after the relationship statement that describes what the figure *demonstrates* beyond restating the prose (for example, where specific workload types fall on the plot and what that means for practice).

**Where the takeaway currently lives:** Largely in the caption (which lists where LLM decode, element-wise ops, and GEMMs fall) — caption content does not count toward the prose's job.

**Rule-compliant diff rewrite (replace the citation clause at the end of L340):**

```diff
-Workloads with $I < I_{\text{ridge}}$ are memory-bound\index{Memory-Bound}: their performance is
-limited by how fast data can be loaded, not how fast it can be processed. Workloads with
-$I > I_{\text{ridge}}$ are compute-bound: the arithmetic units are the bottleneck.
-@Fig-roofline-model illustrates this relationship graphically.
+Workloads with $I < I_{\text{ridge}}$ are memory-bound\index{Memory-Bound}: their performance is
+limited by how fast data can be loaded, not how fast it can be processed. Workloads with
+$I > I_{\text{ridge}}$ are compute-bound: the arithmetic units are the bottleneck.
+@Fig-roofline-model plots this on a log-log scale, showing that transformer decode and element-wise
+operations cluster well to the left of the ridge while large batched GEMMs sit to the right. That
+clustering is not incidental: most inference operations are memory-bound not because they lack
+arithmetic, but because they process small sequences or batch sizes that prevent the arithmetic
+units from saturating before the next memory load is needed.
```

---

### `fig-optimization-hierarchy` (🟠 Figure) — def L1948

**Grade: ⚠️ Partial**

**Ref sentence (L1946):**
> ML system profiling operates at four levels, each providing different granularity and targeting different bottleneck categories. @Fig-optimization-hierarchy makes this drill-down explicit, starting from application-level symptoms and descending toward kernel and hardware-counter evidence.

"Makes this drill-down explicit" is near-announcer phrasing. The cite names the figure's structure but does not state what the figure *demonstrates*: that starting at the top and skipping levels leads to optimizing the wrong bottleneck, or that the pyramid shape encodes both the diagnostic order and the granularity available at each level. The payoff paragraph (L1952) is excellent and carries the full interpretation, but it appears *after* the float, not as the cite sentence.

**What is missing:** A statement in the cite sentence of the key insight the figure encodes (the direction of drill-down and the cost of skipping levels), so the reader approaches the figure already knowing what to look for.

**Rule-compliant diff rewrite (replace the citation clause in L1946):**

```diff
-ML system profiling operates at four levels, each providing different granularity and targeting
-different bottleneck categories. @Fig-optimization-hierarchy makes this drill-down explicit,
-starting from application-level symptoms and descending toward kernel and hardware-counter evidence.
+ML system profiling operates at four levels, each providing different granularity and targeting
+different bottleneck categories. @Fig-optimization-hierarchy shows why level-skipping fails:
+application-level symptoms point toward the right hardware tier to investigate, but they cannot
+distinguish a memory-bound kernel from a launch-overhead problem — that distinction requires
+descending to kernel and hardware-counter evidence, in the order the pyramid shows.
```

---

### `tbl-performance-engineering-bottleneck-paths` (🟠 Table) — def L2351

**Grade: ⚠️ Partial**

**Ref sentence (L2343):**
> @tbl-performance-engineering-bottleneck-paths turns that decision into a compact map: identify the binding resource, use the typical setting as a sanity check, and try the interventions in order until a reprofiled trace shows the bottleneck has moved.

The cite describes how to *use* the table (the procedure) but does not state the table's load-bearing contrast: that the three bottleneck classes require qualitatively different optimization paths and that applying the wrong path (for example, chasing FLOP/s on a communication-bound workload) yields no improvement. The payoff (L2353) adds only "apply and verify" and does not extract the table's key insight. The standard requires the prose to deliver the conclusion the table encodes, not just the table's procedural use.

**What is missing:** A sentence stating that the three bottleneck classes drive entirely different remedies and that a misclassification sends the engineer down a path that cannot move the bottleneck, regardless of implementation quality.

**Rule-compliant diff rewrite (extend the cite sentence at L2343 and add a payoff after L2351):**

```diff
-3. **Selecting the primary bottleneck**\index{Bottleneck!primary selection}: The primary
-bottleneck determines which remedy should be tried first. For memory-bound
-\index{Memory-Bound!optimization path}, compute-bound\index{Compute-Bound!optimization path},
-and communication-bound\index{Communication-Bound!optimization path} workloads,
-@tbl-performance-engineering-bottleneck-paths turns that decision into a compact map: identify
-the binding resource, use the typical setting as a sanity check, and try the interventions in
-order until a reprofiled trace shows the bottleneck has moved.
+3. **Selecting the primary bottleneck**\index{Bottleneck!primary selection}: The three bottleneck
+classes require qualitatively different remedies. A memory-bound workload needs fewer bytes moved
+per useful result (precision reduction, fusion, algorithmic change). A compute-bound workload needs
+higher arithmetic utilization (Tensor Cores, graph compilation, FP8). A communication-bound
+workload needs overlap and topology-aware placement. Applying the memory-bound path to a
+communication-bound system yields no improvement because the binding resource does not change.
+@tbl-performance-engineering-bottleneck-paths\index{Memory-Bound!optimization path}\index{Compute-Bound!optimization path}\index{Communication-Bound!optimization path}
+encodes these distinctions as a compact map: identify the binding resource, use the typical
+setting as a sanity check, and try the interventions in the listed order until a reprofiled trace
+confirms the bottleneck has moved.
```

---

## Dangling reference

`@fig-fleet-stack` (L72) — referenced in a callout perspective block but has no matching definition in this chapter. Not graded as a float finding (float is defined elsewhere), but flagged for completeness.
