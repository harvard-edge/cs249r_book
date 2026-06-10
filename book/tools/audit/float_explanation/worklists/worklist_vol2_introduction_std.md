# Float-Exposition Audit — vol2/introduction/introduction.qmd

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| eq | 🔴 strictest | 5 | 2 | 3 | 0 |
| fig | 🟠 high | 14 | 10 | 3 | 1 |
| tbl | 🟠 high | 4 | 0 | 4 | 0 |
| lst | 🟡 medium | 0 | — | — | — |
| **Total** | | **23** | **12** | **10** | **1** |

---

## Findings (⚠️ and 🛑 only)

---

### `eq-reliability-gap` (eq 🔴) — def L1389

**Verbatim ref sentence (L1387):**
> When a fleet coordinates 25,000 GPUs, @eq-reliability-gap multiplies the individual component probabilities to give the probability that the entire system is healthy ($R_{\text{system}}(t)$):

**Missing move:** The lead-in names $R_{\text{system}}(t)$ but never defines $\lambda$ (the per-component failure rate) in body prose. The equation contains four symbols; three are named ($R_{\text{system}}$, $R_{\text{component}}$, $N$) but $\lambda$ is left implicit. The payoff paragraph (L1433) gives correct numeric instances but never states what $\lambda$ represents or what value drives the worked examples. The symbol's meaning lives only in the reader's prior knowledge, not in the surrounding prose.

**Rule:** Eq 🔴 — prose must deliver the meaning of every symbol (a "where" clause is fine).

**Suggested rewrite — add a where-clause to the existing reference sentence or the sentence immediately following the equation:**

> When a fleet coordinates 25,000 GPUs, @eq-reliability-gap multiplies the independent component survival probabilities to give the probability that the entire system is healthy ($R_{\text{system}}(t)$). Here $R_{\text{component}}(t) = e^{-\lambda t}$ is the probability that a single node survives to time $t$, where $\lambda$ is the per-component failure rate (the reciprocal of its mean time between failures), and $N$ is the fleet size. As $N$ grows, the exponent $N\lambda t$ magnifies even tiny per-node failure rates into near-certain fleet failures over any training window.

---

### `eq-ci-ratio` (eq 🔴) — def L1449

**Verbatim ref sentence (L1447):**
> @Eq-ci-ratio defines **communication intensity** $(\text{CI})$ as the ratio of data moved across the network to the operations performed locally:

**Missing move:** The lead-in and where-clause establish the ratio correctly. But the lead-out (payoff at L1460) only describes optimization strategies (gradient sparsification, 3D parallelism) and defers the worked consequence to an appendix. No body sentence states the regime consequence: at what CI threshold does a kernel become communication-bound rather than compute-bound, or what a typical LLM training step's CI looks like. The takeaway lives only in the deferred appendix, not in body prose. A reader who skips the appendix never learns what CI value to worry about.

**Rule:** Eq 🔴 — prose must deliver the consequence or regime the equation implies. A worked numeric instance is the gold standard.

**Suggested rewrite — add a regime sentence immediately after the payoff (L1460), before the appendix forward-reference:**

> A transformer training step on InfiniBand HDR moves roughly 700 GB of gradients while performing on the order of $10^{23}$ FLOPs of compute; that ratio places a 175B-parameter AllReduce squarely in the communication-bound regime on any interconnect slower than InfiniBand NDR. The fleet law's $T_{\text{comm}}(N)$ term dominates whenever CI exceeds the hardware's bytes-per-FLOP crossover point, which is why gradient sparsification and 3D parallelism aim to lower CI rather than add more accelerators. @sec-appdx-fleet-foundations-comm-compute-ratio works through the bandwidth-saturation point numerically.

---

### `eq-energy-scale-invariant` (eq 🔴) — def L1562

**Verbatim ref sentence (L1560):**
> @Eq-energy-scale-invariant defines the **fleet energy productivity** ($\rho_{\text{energy}}$), measured in FLOP/J, as the ratio of useful work to total energy consumed:

**Missing move:** The where-clause at L1564 defines $O_{\text{useful}}$ and $E_{\text{network}}$ but never names $E_{\text{compute}}$ and $E_{\text{cooling}}$, which together with $E_{\text{network}}$ form the entire denominator. More critically, the payoff (L1566) immediately pivots: "The energy-side metric is only half of the scaling diagnosis." No body sentence states what $\rho_{\text{energy}}$ implies in practice — what a typical value is, what happens as $N$ scales, or what the Pareto frontier concretely means for a training run. The practical consequence lives only in the appendix forward-reference.

**Rule:** Eq 🔴 — prose must deliver meaning of every symbol plus the consequence or regime implied.

**Suggested rewrite — extend the existing where-clause at L1564 and add a consequence sentence:**

> where $O_{\text{useful}}$ is useful work in FLOPs, $E_{\text{compute}}$ is the energy drawn by the accelerators and memory, $E_{\text{cooling}}$ captures the thermodynamic overhead of keeping silicon at operating temperature (typically adding 30–50 percent via the Power Usage Effectiveness factor), and $E_{\text{network}}$ often becomes a nonnegligible fraction of the total budget as we move terabytes across optical fabrics. As fleet size grows, cooling and network energy grow faster than compute energy — the denominator inflates while $O_{\text{useful}}$ plateaus — so $\rho_{\text{energy}}$ degrades with scale unless the architecture is co-designed to minimize data movement. Mastery of scale requires optimizing for the Pareto frontier of both laws: minimizing $T_{\text{step}}$ while maximizing $\rho_{\text{energy}}$.

---

### `fig-compute-trends` (fig 🟠) — def L1019

**Verbatim ref sentence (L1017):**
> @Fig-compute-trends traces *how* computational demands of training large models have escalated at an unsustainable rate, growing faster than Moore's Law improvements in hardware.

**Missing move:** The lead-in frames the figure as evidence for "why scaling laws are necessary." The payoff paragraph (L1025) immediately introduces the universal scaling law principle without delivering what the figure *shows* — the 3.4-month doubling rate versus Moore's Law's 2-year cycle, and what that gap means for the systems infrastructure the book is building. The numbers appear in the caption but not in body prose. Without the lead-out, a reader who skips the caption learns only that compute grew fast, not by how much faster than hardware could keep up.

**Rule:** Fig 🟠 — prose must deliver what the figure demonstrates and why it matters; the prose tells the figure's story.

**Suggested rewrite — replace or extend L1025 to open with the figure's takeaway before pivoting to scaling laws:**

> Between 2012 and 2019, training compute doubled roughly every 3.4 months — about seven times faster than Moore's Law's two-year cadence. That gap, visible in @fig-compute-trends as the slope divergence after the AlexNet inflection, is the reason hardware efficiency cannot close the compute deficit by making chips faster: the demand curve outruns the supply curve by an order of magnitude per decade. The universal scaling law (Principle \ref{pri-universal-scaling}) provides a quantitative framework for navigating these trade-offs by making the demand curve predictable.

---

### `fig-loss-vs-n-d` (fig 🟠) — def L1101

**Verbatim ref sentence (L1081):**
> @Fig-loss-vs-n-d shows *how* early-stopped test loss varies predictably with both dataset size and model size, confirming that learning curves across configurations align through appropriate parameterization.

**Missing move:** The citation is a bare pointer ("shows how X varies"). The payoff paragraph at L1172 only pivots: "Performance improvements follow predictable patterns, but the relevant design action changes with resource availability." Neither sentence states what the figure demonstrates — specifically that larger models achieve lower loss at every dataset size, that all curves exhibit diminishing returns at high token counts, and that the curves' alignment validates the power-law parameterization used for cross-configuration prediction. The takeaway lives only in the caption.

**Rule:** Fig 🟠 — the point of the figure must be in the prose, not just in the caption.

**Suggested rewrite — add a payoff sentence after the figure (before L1083) or replace the citation sentence:**

> @Fig-loss-vs-n-d makes the parameterization concrete: every curve, regardless of model scale from 393K to 708M parameters, bends toward the same diminishing-returns shape as token count grows. The key insight is not that larger models are better — that is expected — but that the loss gap between model scales *narrows* as the training dataset grows. Once a model enters the saturation regime, adding more tokens yields less improvement than adding more parameters would, which is the empirical basis for the compute-optimal frontier.

---

### `fig-fleet-stack` (fig 🟠) — def L1704

**Verbatim ref sentence (L1702):**
> @Fig-fleet-stack organizes the complexity of this book into **The Fleet Stack**, a four-layer framework where engineering decisions at the bottom constrain possibilities at the top.

**Missing move:** The citation names the figure and labels the framework but states no mechanism. The payoff (L1738) only adds: "This layered progression structures the textbook's four parts." No body prose explains *why* bottom-layer decisions constrain top-layer possibilities — what the coupling is, what it means concretely that storage hierarchy choices at Part I constrain parallelism strategies at Part II, or that parallelism choices constrain governance options at Part IV. The mechanical argument lives only in the caption.

**Rule:** Fig 🟠 — the prose must deliver what the figure demonstrates and why it matters.

**Suggested rewrite — extend L1738 with the mechanical consequence:**

> This layered progression structures the textbook's four parts, and the arrows in @fig-fleet-stack are not decorative: a decision made at the infrastructure layer — network topology, storage bandwidth, accelerator memory capacity — propagates upward as a hard constraint on every layer above it. A fat-tree topology at Part I determines which collective communication algorithms remain feasible at Part II; the parallelism strategy chosen at Part II determines how much per-model state must be checkpointed, which shapes the fault-tolerance architecture at Part II and the multi-tenant scheduling policies at Part III. Governance at Part IV is last not because it matters least, but because it can only constrain what the layers below it make physically possible.

---

### `fig-vol2-ai-triad` (fig 🟠) — 🛑 — def L1744

**Verbatim ref sentence (L1742):**
> @Fig-vol2-ai-triad visualizes these dependencies between data, algorithms, and infrastructure, revealing the optimization landscape that ML systems engineers must address.

**Missing move:** Both the citation and the payoff are bare pointers. The citation "visualizes dependencies… revealing the optimization landscape" tells the reader to look at the figure without stating what the dependencies are or why the bidirectional arrows matter. After the figure, the text (L1860) immediately introduces the Five-Pillar Framework with no sentence returning to the AI Triad's takeaway. No body prose explains the cascade mechanism: that a data architecture choice (e.g., deduplication strategy) cascades into algorithm behavior (generalization vs. memorization) which cascades into infrastructure requirements (memory per replica, checkpoint frequency). The figure exists as an unsupported exhibit. Takeaway lives only in caption.

**Rule:** Fig 🟠 — prose must deliver what the figure demonstrates and why it matters; a float without a body-prose lead-out fails the removability test.

**Suggested rewrite — add a lead-out sentence immediately after the figure (before L1860):**

> The double-headed arrows in @fig-vol2-ai-triad encode a practical constraint: no vertex can be scaled independently without stressing the other two. Doubling training data without expanding model capacity yields diminishing returns; expanding model capacity without scaling infrastructure produces out-of-memory failures or bandwidth-starved training; and upgrading infrastructure without data quality improvements wastes the larger compute budget on noise. At GPT-4-class scale, this coupling means that a decision as seemingly local as deduplication policy in the data pipeline determines whether the model memorizes training examples or generalizes, which in turn determines the serving infrastructure needed to handle adversarial queries at deployment. The AI Triad names the vertices; the fleet stack, below, names the engineering layers that manage each edge.

---

### `tbl-training-compute-evolution` (tbl 🟠) — def L248

**Verbatim ref sentence (L250):**
> @Tbl-training-compute-evolution captures the growth in training compute, but an equally important dimension is the growth in cluster size itself.

**Missing move:** The only sentence about the table pivots immediately away from it. No body prose delivers the table's load-bearing contrast: seven orders of magnitude of compute growth (from $10^{18}$ FLOPs for AlexNet to $10^{25}$ for GPT-4-class), with training times growing from days to months and node counts growing from 2 to 25,000. The reader who skips the table cells never learns the quantitative magnitude. The insight that this growth is what makes distributed systems necessary lives only in the caption.

**Rule:** Tbl 🟠 — prose must deliver the takeaway the table encodes; a "captures X, but" pivot is not a takeaway.

**Suggested rewrite — replace the pivot sentence at L250 with a takeaway first, then the pivot:**

> @Tbl-training-compute-evolution makes the scale shift quantitative: from AlexNet's $10^{18}$ FLOPs on two GPUs to the GPT-4-class scenario's estimated $10^{25}$ FLOPs on 25,000 GPUs, training compute grew seven orders of magnitude in just over a decade — with training durations stretching from days to months. That seven-order-of-magnitude expansion is the reason distributed systems are not an optimization for large models; they are a prerequisite. An equally important dimension is the growth in cluster size itself, which @Fig-cluster-size-explosion traces.

---

### `tbl-scaling-breakdown` (tbl 🟠) — def L1360

**Verbatim ref sentence (L1350):**
> @Tbl-scaling-breakdown organizes these failure modes, mapping each breakdown type to its underlying cause and a representative scenario, so practitioners can anticipate the inefficiency before committing the budget.

**Missing move (first cite):** The citation is functional but no payoff sentence follows in body prose — the next content is a checkpoint callout ("Verify your understanding…"), which does not count as body prose. No body sentence draws the table's conclusion: which breakdown is most common, or what the shared root cause across the rows is (growth in one dimension outpacing the others).

**Missing move (second cite, L1377):** "Each of these dimensions addresses a different breakdown condition from @tbl-scaling-breakdown" is a back-reference without restating the table's conclusion.

**Rule:** Tbl 🟠 — prose must deliver the conclusion the table drives; the insight that "most breakdowns share one root cause" appears before the table (L1348) but never appears as a payoff after it.

**Suggested rewrite — add a payoff sentence after the table (at L1364 before the checkpoint callout):**

> The common thread across all five rows is dimensional imbalance: a system that doubles model parameters without proportionally expanding tokens, compute, or both will land in overfitting, diminishing returns, or bandwidth saturation. Tracking which dimension is currently lagging — the diagnosis the table supports — is the difference between intentional scaling and expensive guesswork.

---

### `tbl-framework-rosetta-stone` (tbl 🟠) — def L1874

**Verbatim ref sentence (L1862):**
> @Tbl-framework-rosetta-stone provides a cross-framework mapping as a translation aid for the primitives developed across the fleet stack, not as a tool ranking.

**Payoff sentence (L1876):**
> These primitives recur throughout the book; the table is a reference to return to as each one is introduced, not a syllabus to memorize now.

**Missing move:** Both sentences are meta-commentary about how to use the table. No body prose states what the table reveals: that the same six abstract primitives (data parallelism, sharding, tensor parallelism, pipeline parallelism, gradient accumulation, checkpointing) underlie every major distributed ML framework, and that the naming diversity across frameworks masks a shared conceptual vocabulary. A reader who skims the table without that framing sees a grid of API names, not a cross-framework unification.

**Rule:** Tbl 🟠 — prose must deliver the insight the table encodes, not just instructions for using the table.

**Suggested rewrite — replace L1876 with a content takeaway:**

> The table's key finding is that all five frameworks implement the same six abstract primitives under different names: FSDP and ZeRO-1/2/3 both solve parameter sharding, DTensor and xmap both express tensor parallelism, and torchrun and jax.distributed both manage process group initialization. That naming diversity hides a shared vocabulary — mastering the primitive in one framework transfers directly to the others. These primitives recur throughout the book; the table is a reference to return to as each one is introduced in its systems context.

---

### `tbl-vol2-lighthouse-archetypes` (tbl 🟠) — def L1908

**Verbatim ref sentence (L1900):**
> @Tbl-vol2-lighthouse-archetypes summarizes the three canonical workloads tracked throughout this book.

**Missing move:** The analytical sentence that should serve as the payoff precedes the citation rather than following it (L1898: "The LLM case asks whether dense synchronization can keep thousands of accelerators useful…"). After the table, the text immediately pivots to textbook structure (L1912). No body prose follows the table with a conclusion about what the three-way contrast reveals — specifically that the C$^3$ taxonomy partitions real-world workloads into exhaustive constraint regimes, and that a system designed for one archetype will fail at a different bottleneck if deployed for another.

**Rule:** Tbl 🟠 — the prose owes the conclusion after the table, not only before it; the payoff sentence must follow.

**Suggested rewrite — add a lead-out sentence between L1908 (table) and L1910 (structure section):**

> The diagnostic value of the table is in the C$^3$ column: each archetype exposes a different irreducible constraint, and the engineering discipline required differs accordingly. A solution that eliminates communication bottlenecks for Archetype A (tensor and pipeline parallelism) does nothing for Archetype B's coordination bottleneck (sparse all-to-all routing) and is irrelevant to Archetype C's compute bottleneck (milliwatt-scale on-device silicon). This three-way partition ensures that every principle in this book is stress-tested against the full diversity of production fleet engineering, not optimized for one regime.
