# Math Audit Report: `book/quarto/contents/vol2/parts/distributed_ml_principles.qmd`

## Checked scope

Audited `book/quarto/contents/vol2/parts/distributed_ml_principles.qmd` for distributed-training, scaling, communication, checkpointing, reliability, complexity, numeric, unit-conversion, and prose-equation consistency issues using direct reasoning only. No Gemini assistance was used.

The file contains four displayed equations and one qualitative overhead principle. There are no worked unit conversions or detailed numeric examples beyond the `10x` performance versus `100x-1000x` compute statement.

## Findings

### Medium: Part numbering is internally inconsistent

- **Lines:** 1, 3, 41-43
- **Current text:** The title says `# Part II: Distributed ML`; line 3 says "If Part I built the physical substrate, Part II establishes the Logic of Distribution"; the roadmap heading and introduction say "Part VI Roadmap" and "Part VI establishes how we make the Fleet compute."
- **Issue:** The file identifies the same part as both Part II and Part VI. This is a prose/count consistency defect that makes the roadmap numbering ambiguous.
- **Proposed correction:** Use one numbering scheme consistently. If this is the second part of Volume 2, change lines 41 and 43 to "Part II." If it is globally Part VI across both volumes, change lines 1 and 3 to use "Part VI" and update the reference to the prior part accordingly.

### Medium: The scaling-law implication does not follow from the displayed equation without specifying `\alpha`

- **Lines:** 8-11
- **Current text/equation:** `\mathcal{L}(C) \propto C^{-\alpha}` followed by "To achieve a 10x improvement in performance, you typically need a 100x-1000x increase in compute."
- **Issue:** If "10x improvement" means a 10x reduction in loss under the displayed law, then `\mathcal{L}_2/\mathcal{L}_1 = (C_2/C_1)^{-\alpha}` implies `C_2/C_1 = 10^{1/\alpha}`. A `100x-1000x` compute increase is only implied when `\alpha` is between `1/2` and `1/3`. Since the equation leaves `\alpha` unspecified and the prose shifts from loss to "performance," the numeric implication is not derivable from the stated law.
- **Proposed correction:** Either specify the assumed exponent and metric, or soften the claim. For example: "Under `\mathcal{L}(C) \propto C^{-\alpha}`, a 10x reduction in loss requires `10^{1/\alpha}` more compute; for `\alpha` between `1/3` and `1/2`, this corresponds to `100x-1000x` more compute." If "performance" is not loss, define the performance metric separately.

### Low: The universal scaling prose names `D` and `P`, but the equation only models `C`

- **Lines:** 8-9
- **Current text/equation:** The invariant says loss improves as a power-law function of compute `C`, dataset size `D`, and parameters `P`, but the displayed equation is only `\mathcal{L}(C) \propto C^{-\alpha}`.
- **Issue:** The prose presents a three-variable scaling relationship, while the equation is a one-variable compute scaling relationship. The notation is not wrong, but it is incomplete relative to the sentence immediately above it.
- **Proposed correction:** Either narrow the prose to compute-only scaling, or show a multivariate form. For example: "Holding dataset size and model size on an appropriate scaling frontier, loss often improves as a power law in compute: `\mathcal{L}(C) \propto C^{-\alpha}`." Alternatively, use a schematic multivariate expression such as `\mathcal{L}(C,D,P)`.

### Medium: Step-time equation can become physically invalid unless overlap is bounded

- **Lines:** 15-18
- **Current text/equation:** "The time to complete one training step is the sum of computation and non-overlapped communication." Then `T_{\text{step}}(N) = T_{\text{compute}}/N + T_{\text{comm}}(N) - T_{\text{overlap}}`.
- **Issue:** The prose says to add non-overlapped communication, but the equation subtracts an unconstrained overlap term. As written, if `T_{\text{overlap}} > T_{\text{comm}}(N)`, the communication contribution becomes negative, which is not physically meaningful. If `T_{\text{overlap}}` overlaps compute with communication, it should be capped by the communication time, or the equation should directly use a non-overlapped communication term.
- **Proposed correction:** Write `T_{\text{step}}(N) = T_{\text{compute}}/N + \max(0, T_{\text{comm}}(N) - T_{\text{overlap}})` or define `T_{\text{comm,nonoverlap}}(N)` and use `T_{\text{step}}(N) = T_{\text{compute}}/N + T_{\text{comm,nonoverlap}}(N)`.

### Low: Gradient accumulation is misclassified as maximizing communication overlap

- **Lines:** 18
- **Current text:** "To scale efficiently, algorithms must minimize `T_comm` or maximize `T_overlap` (for example, through Gradient Accumulation or Communication Hiding)."
- **Issue:** Communication hiding directly increases overlap between computation and communication. Gradient accumulation usually reduces communication frequency by doing multiple microsteps before synchronization; it changes amortized communication per optimizer update and may increase compute per communication event, but it is not itself an overlap mechanism.
- **Proposed correction:** Split the examples by mechanism: "minimize or amortize `T_comm` through gradient accumulation, compression, or larger buckets; maximize `T_overlap` through communication hiding."

### Low: Compression reduces message size, not the physical bandwidth parameter

- **Lines:** 21-25
- **Current text/equation:** `T(n) = \alpha + n/\beta`, with `\beta` defined as bandwidth; the implication says "compress large messages to improve `\beta`."
- **Issue:** With the equation as written, `\beta` is the link or effective bandwidth in units of data per time, while `n` is message size. Compressing a message primarily reduces `n`; it does not improve the physical bandwidth parameter unless the text explicitly redefines `\beta` as effective application-level bandwidth after compression.
- **Proposed correction:** Change the final phrase to "compress large messages to reduce `n` or improve effective throughput." If keeping "improve `\beta`," define `\beta` as effective bandwidth including compression/decompression overhead.

### Medium: The checkpoint law names frequency, but `\tau_{\text{opt}}` is an interval

- **Lines:** 28-32
- **Current text/equation:** "The optimal checkpoint frequency..." followed by `\tau_{\text{opt}} = \sqrt{2 \cdot T_{\text{write}} \cdot \text{MTBF}}`.
- **Issue:** The units of the right-hand side are `sqrt(time * time) = time`, so `\tau_{\text{opt}}` is a checkpoint interval or period, not a frequency. Frequency would have units of `1/time` and would be `1/\tau_{\text{opt}}`.
- **Proposed correction:** Change "frequency" to "interval" or "period": "The optimal checkpoint interval balances the cost of writing the checkpoint..." If frequency is desired, define `f_{\text{opt}} = 1/\tau_{\text{opt}}`.

### Low: The conservation-of-overhead statement is too absolute as a mathematical invariant

- **Lines:** 35-38
- **Current text:** "Overhead in a distributed ML system cannot be eliminated, only redistributed among Compute, Communication, and Coordination. Reducing one necessarily increases at least one other."
- **Issue:** This is a useful design heuristic but not a mathematical invariant. A hardware, compiler, scheduling, or topology improvement can reduce one overhead without necessarily increasing another for the same workload. The examples about asynchronous training and pipeline parallelism describe common trade-offs, not a conservation law.
- **Proposed correction:** Recast as a tendency rather than an invariant. For example: "In many distributed ML designs, reducing one overhead often shifts cost into compute, communication, coordination, memory, or convergence efficiency; optimizations should state which cost moved and which cost was actually reduced."

## No-Issue Checks

- **Lines 21-23:** `T(n) = \alpha + n/\beta` is dimensionally valid if `n` is data volume and `\beta` is bandwidth in data per time. The convention differs from the common form where the per-byte time is named `\beta`, but the local definition is internally consistent.
- **Lines 29-30:** Apart from the frequency-versus-interval wording, `\sqrt{2 T_{\text{write}} \text{MTBF}}` has units of time and decreases as cluster MTBF decreases, matching the prose that larger clusters require more frequent checkpoints.
- **Lines 45-48:** The roadmap lists four chapters and then provides exactly four items, so the item count is internally consistent.
