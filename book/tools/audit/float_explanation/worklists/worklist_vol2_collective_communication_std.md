# Float Exposition Audit — `collective_communication.qmd` (vol2)

Graded against FLOAT_EXPOSITION_STANDARD.md. Caption, fig-alt, in-figure labels, code comments,
and boxed callout interiors do not count toward the prose's job. Only running body prose counts.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| Algorithm | 🔴 strict | 1 | 1 | 0 | 0 |
| Figure | 🟠 high | 9 | 7 | 2 | 0 |
| Table | 🟠 high | 15 | 10 | 2 | 3 |
| **Total** | | **25** | **18** | **4** | **3** |

**Dangling ref (no def):** `@fig-fleet-stack` at L50 — never defined in this file.

---

## Findings (⚠️ and 🛑 only)

---

### ⚠️ `fig-comm-compute-overlap` (figure 🟠) — def L508

**Ref sentence (L502):**
> The network latency is hidden, but the processor overhead remains exposed. @Fig-comm-compute-overlap shows this overlap visually.

**Missing move:** The citation is a bare pointer ("shows this overlap visually"). The figure's visual job is to reveal why the non-overlappable overhead $o$ bounds pipelined performance even when $L_\text{lat}$ is hidden. That insight is stated numerically in the bullet list before the cite, but no sentence after the figure draws the implication — the payoff paragraph pivots to model-selection guidance, not to a conclusion about what the figure demonstrates. A reader who skips the figure learns the same numbers from the bullets; a reader who looks only at the figure gets no prose takeaway.

**Where the takeaway currently lives:** The numeric calculation is in the bullet list (L496–500) inside the callout; the systems-level implication is in the **Systems insight** line inside the same callout — both are callout-interior prose, which does not count.

**Rule-compliant rewrite** (replace the citation sentence at L502 end):

> The processor overhead $o$ remains exposed regardless of how well communication is pipelined — it is the irreducible floor on pipelined step time, visible in @Fig-comm-compute-overlap as the gap between the effective timeline and the fully hidden ideal.

---

### ⚠️ `fig-torus-reduction` (figure 🟠) — def L515 (image), cited L1519

**Ref sentence (L1519):**
> @fig-torus-reduction depicts the same dimension-ordered idea as a 2D simplification.

**Missing move:** "Depicts … as a 2D simplification" is a pure pointer. The figure's pedagogical point is that dimension-ordered reduction achieves global AllReduce by decomposing into independent per-dimension rings, keeping each link busy in exactly one direction at a time and avoiding contention. That mechanism is explained in the surrounding prose, but the citation sentence itself adds nothing and fails the removability test: deleting "@fig-torus-reduction depicts … as a 2D simplification" leaves the argument unchanged.

**Where the takeaway currently lives:** The payoff after the float (same paragraph, continuing sentences) explains the dimension-ordered steps and states the bandwidth cost. The figure's specific contribution — showing why the torus layout makes contention-free dimension routing possible — is never stated in body prose.

**Rule-compliant rewrite** (replace the citation sentence fragment):

> @fig-torus-reduction shows this as a 2D cross-section: each axis-aligned ring operates independently, so every link carries traffic in one direction at a time, and no two rings contend for the same path.

---

### ⚠️ `tbl-collective-selection` (table 🟠) — def L801

**Ref sentence (L789):**
> @Tbl-collective-selection maps these primitives to the parallelism strategies introduced in @sec-distributed-training-systems and the Lighthouse architectures defined in the Introduction.

**Missing move:** The cite is a setup pointer ("maps these primitives"). It names what the table contains but does not state the conclusion the table encodes. The H&P standard for tables requires a "the key result is" sentence that carries the load-bearing contrast in prose so the reader could skip the cells. The payoff paragraph (L811) pivots immediately to FSDP's specific communication pattern without stating what the table as a whole teaches — namely that bandwidth-bound vs. latency-bound bottleneck type is the axis that determines which collective, and that this axis cuts cleanly across the table.

**Where the takeaway currently lives:** Partially in the per-cell bottleneck column; partially scattered across the preceding MoE discussion and the following FSDP discussion. No single body-prose sentence draws the table's cross-cutting conclusion.

**Rule-compliant rewrite** (add after the existing cite sentence at L789):

> The table's key axis is the bandwidth-vs.-latency divide: data-parallel and FSDP workloads are bandwidth-bound and tolerate ring-based algorithms, while tensor- and pipeline-parallel workloads are latency-bound and demand the lower-overhead collectives or point-to-point paths that only operate well inside a single node.

---

### ⚠️ `tbl-allreduce-comparison` (table 🟠) — def L1058

**Ref sentence (L1049):**
> @Tbl-allreduce-comparison summarizes the AllReduce algorithm comparison and the performance characteristics of the four algorithms.

**Missing move:** "Summarizes … performance characteristics" is a float-announcer. The table adds two specific insights the surrounding prose does not state: (1) Butterfly achieves the same bandwidth optimality as Ring but with logarithmic latency — making it strictly better when $N = 2^k$; (2) Double Binary Tree is the practical pick because it is near-optimal on both axes with no power-of-two constraint. Neither of these cell-level conclusions is stated in body prose near the table. The payoff paragraph immediately following the table belongs to the section on algorithm crossover and discusses the figure, not the table.

**Where the takeaway currently lives:** Butterfly's constraint ($N = 2^k$) is in the caption. Double Binary Tree's practical advantage is discussed three paragraphs earlier (L1045–1047) before the table is defined, making it hard to identify as the table's conclusion.

**Rule-compliant rewrite** (add after the float-announcer sentence at L1049):

> The table's decision column reveals that Butterfly strictly dominates Ring in the latency term whenever $N$ is a power of two, while Double Binary Tree removes that constraint at the cost of only near-optimal bandwidth — making it the default pick for production libraries that cannot assume power-of-two cluster sizes.

---

### 🛑 `tbl-interconnect-parameters` (table 🟠) — def L387

**Ref sentence (L377):**
> @Tbl-interconnect-parameters shows typical values for data center interconnects.

**Missing move:** "Shows typical values" is a bare pointer — the weakest possible cite. No lead-out sentence states what conclusion the table encodes. The payoff at L389 says "Applying the critical message size formula to a concrete workload reveals which optimization strategy matters most" — this is a transition to a calculation, not an interpretation of the table. The key insight the table encodes (InfiniBand NDR has a critical size ten times smaller than NVLink, which is why inter-node messages are latency-bound far longer) never appears in body prose. The removability test fails: the surrounding text would be unchanged if the table were deleted.

**Where the takeaway currently lives:** In the cells themselves. The critical-size column contains the decision-relevant numbers, but no prose sentence names the pattern or its consequence.

**Rule-compliant rewrite** (replace the bare pointer at L377):

> @Tbl-interconnect-parameters shows that InfiniBand NDR carries a critical-size crossover near 100 KB — ten times smaller than NVLink's 1 MB — meaning inter-node messages must be an order of magnitude larger before they exit the latency-bound regime. A typical per-layer gradient of 10–50 MB clears both crossovers easily; a single MoE routing message of 4 KB does not clear either.

---

### 🛑 `tbl-error-feedback-naive` (table 🟠) — def L1698

**Ref sentence (L1688):**
> **Without Error Feedback** (Naive Compression, @tbl-error-feedback-naive):

**Missing move:** This citation lives inside a `.callout-notebook` block. Per the standard, callout interiors do not count toward the prose's job. No body prose outside the callout cites or interprets this table. The payoff paragraph at L1727 (first body prose after the callout closes) discusses the general limitations of treating the optimizer as a black box — it does not name, cite, or draw a conclusion from the naive-compression trace. The table is invisible to body prose.

**Where the takeaway currently lives:** Inside the callout-notebook at L1700: "After 5 steps, the system has transmitted 0 but the true cumulative gradient is 1.6. The parameter never updates, and 100 percent of gradient information is lost." This is callout-interior prose only.

**Rule-compliant rewrite** — the fix requires adding a body-prose lead-out sentence after the callout closes (L1724). Example:

> The worked trace inside the callout above shows the failure mode directly: a parameter receiving gradients of 0.3–0.4 every step never updates under naive compression because no individual gradient crosses the transmission threshold, and the missing signal accumulates invisibly.

---

### 🛑 `tbl-error-feedback` (table 🟠) — def L1712

**Ref sentence (L1702):**
> **With Error Feedback** (@tbl-error-feedback):

**Missing move:** Same scope problem as `tbl-error-feedback-naive` above. The citation is inside the `.callout-notebook` and no body prose outside the callout cites or interprets the error-feedback trace. The payoff paragraph at L1727 discusses 1-bit Adam motivation rather than closing the loop on what the error-feedback trace demonstrates. The table is invisible to body prose.

**Where the takeaway currently lives:** Inside the callout-notebook at L1714–1721: the "Result" and "Systems insight" lines that explain conservation of gradient information and convergence recovery are all callout-interior prose.

**Rule-compliant rewrite** — add a body-prose lead-out after the callout closes (L1724), contiguous with the one suggested above for `tbl-error-feedback-naive`:

> The error-feedback trace in the same callout shows the repair: by accumulating residuals between steps, the compressor eventually transmits all gradient information — the parameter receives the correct total update across five steps, just with one-step delay at each threshold crossing.

---

## Notes

- `@fig-fleet-stack` (L50) is a dangling reference — the figure definition does not appear in this file. This is a cross-file reference that the scanner cannot resolve; it should be verified against the volume's other chapters.
- Tables `tbl-error-feedback-naive` and `tbl-error-feedback` are structurally tied to a callout-notebook worked example. The recommended fix (adding two body-prose sentences after the callout closes) is minimal and preserves the callout's self-contained pedagogy while satisfying the body-prose contract.
