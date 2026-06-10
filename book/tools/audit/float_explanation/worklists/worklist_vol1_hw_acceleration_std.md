# Float Exposition Worklist — `vol1/hw_acceleration/hw_acceleration.qmd`

Graded against FLOAT_EXPOSITION_STANDARD.md. Caption, fig-alt, in-figure labels, and code comments do not count toward the prose's job; only running body prose is evaluated.

---

## Summary Table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| Algorithm | 🔴 strict | 1 | 1 | 0 | 0 |
| Equation | 🔴 strict | 6 | 4 | 2 | 0 |
| Figure | 🟠 high | 14 | 10 | 3 | 1 |
| Listing | 🟡 medium | 22 | 16 | 6 | 0 |
| Table | 🟠 high | 24 | 20 | 3 | 1 |
| **Total** | | **67** | **51** | **14** | **2** |

**16 findings** (14 ⚠️ + 2 🛑). Dominant pattern: listings with bare pointer lead-ins that name the mechanism but skip the design-choice payoff, and tables cited with "summarizes/highlights/contrasts" without the key row or conclusion in prose.

---

## Findings

---

### Equations

---

#### ⚠️ `eq-required-bandwidth` (🔴 Equation) — def L3835

**Ref sentence (L3834):**
> "The roofline model's memory-bound region is determined by the peak memory bandwidth. For an operation to achieve throughput $R_{\text{ops}}$ (FLOP/s, often expressed in TFLOP/s) in the memory-bound regime, @eq-required-bandwidth gives the required bandwidth:"

**Missing move:** The equation is introduced correctly with a "where" clause implicit in the inline text, and the symbols are identified. However, the consequence or regime it implies is not stated in the lead-in: the prose never says what happens when $\text{BW}_{\text{req}}$ exceeds available bandwidth, nor what an engineer should conclude from the ratio $R_{\text{ops}}/\text{AI}$. The next paragraph pivots immediately to a capping equation (@eq-attainable-throughput) without ever naming the architectural implication from this equation alone. The takeaway ("when this value exceeds the hardware bandwidth ceiling, you are memory-bound regardless of compute capacity") lives only implicitly.

**Where takeaway lives:** Implied by the transition to @eq-attainable-throughput, but never stated as a conclusion in prose.

**Rule-compliant rewrite for the lead-out (add after the equation, before the @eq-attainable-throughput sentence):**

> The ratio $R_{\text{ops}}/\text{AI}$ is the bandwidth the operation demands to stay fed. When that demand exceeds the hardware's peak bandwidth, the operation is memory bound: delivering more compute capacity cannot accelerate it further.

---

#### ⚠️ `eq-batch-ai` (🔴 Equation) — def L4190

**Ref sentence (L4189):**
> "Increasing batch size\index{Batch Size!arithmetic intensity impact} improves AI for matrix operations by amortizing weight loading. @Eq-batch-ai formalizes this relationship for a dense layer $(B{\times}M){\times}(M{\times}N)$:"

**Missing move:** The cite paragraph names the relationship ("increasing batch size improves AI by amortizing weight loading") but does not state the consequence: what AI value corresponds to a practically useful batch, what regime flip this enables, or when the approximation breaks down. The payoff paragraph is just "Example: Dense layer with M=N=2048 (FP16)," which is a code-cell label, not prose. The implication that batch size is an engineer's lever for escaping the memory-bound regime is never stated in body prose.

**Where takeaway lives:** Partial framing in the lead-in sentence but no stated conclusion or regime implication.

**Rule-compliant rewrite (add as a lead-out sentence after the equation, before the code cell):**

> For large weight matrices ($2MN \gg 2B(M+N)$), AI grows linearly with batch size, meaning that doubling the batch doubles arithmetic intensity and can push an operation from memory bound across the ridge point into compute bound. This makes batch size the cheapest engineering lever for arithmetic intensity improvement when latency requirements allow it.

---

### Figures

---

#### ⚠️ `fig-tech-s-curve` (🟠 Figure) — def L659

**Ref sentence (L657):**
> "This technology S-curve\index{Technology S-Curve!computing paradigms} pattern appears in the two overlapping curves in @fig-tech-s-curve: as a general-purpose curve saturates, domain-specific architectures can open a new efficiency curve for workloads with stable computational structure."

**Missing move:** The lead-in names what the figure shows (two S-curves, overlap at saturation) but provides no lead-out in the sentences immediately before or after the float. The payoff paragraph (L739) arrives several hundred lines later after extensive intervening prose about DSA specifics; it no longer reads as the figure's payoff. Directly after the float, the prose pivots to describing how DSAs work mechanistically, never stating the figure's conclusion: that the general-purpose curve has saturated and ML workloads are in the take-off phase of the new curve, meaning hardware investments now must track the domain-specific trajectory, not wait for the old curve to recover.

**Where takeaway lives:** Partially in L739 ("The 'easy' gains from shrinking transistors are gone") but that is many paragraphs distant and does not explicitly tie back to the figure.

**Rule-compliant rewrite (add immediately after the float definition, before the domain-specific narrative):**

> The practical consequence of the saturation point (circa 2010) is that general-purpose clock-speed gains have already been harvested. ML workloads now sit on the steep take-off segment of the domain-specific curve, where architectural choices, not process shrinks, drive the next decade of efficiency improvement.

---

#### ⚠️ `fig-ai-performance` (🟠 Figure) — def L1574

**Ref sentence (L1572):**
> "To appreciate the magnitude of this shift, trace the curve in @fig-ai-performance from left to right: over a single decade, NVIDIA GPU performance jumped roughly 1,000$\times$ as the architecture transitioned from general-purpose floating-point execution units to highly optimized tensor processing cores."

**Missing move:** The lead-in gives a quantitative description of the trend (1,000x over a decade, FP units to tensor cores). However, it contains no lead-out stating the "so what": what this gain means for model feasibility, why it enables workloads not previously viable, or how this 1,000x relates to algorithmic demands. The payoff paragraph (L1711) arrives hundreds of lines later and discusses processing elements in general rather than interpreting the figure's curve. The figure's story (sustained exponential gain driven by architectural innovation rather than process shrink) is never completed in body prose near the float.

**Where takeaway lives:** Partially implicit in the setup context; the payoff paragraph (L1711) does not interpret the figure.

**Rule-compliant rewrite (add as lead-out immediately after the float definition):**

> This 1,000-fold gain came not from Moore's Law transistor scaling but from three architectural decisions: dedicated matrix units, reduced precision (FP32 to FP8), and hardware-accelerated sparsity. Each step applied the DSA principle: give up general-purpose flexibility to serve the dominant workload's specific arithmetic pattern.

---

#### 🛑 `fig-rising-ridge` (🟠 Figure) — def L2775

**Ref sentence (L2773):**
> "The imbalance has a direct architectural consequence visible in @fig-rising-ridge: the hardware ridge point has climbed sharply and remains high, pushing sparse and low-reuse operations further into the memory-bound regime on modern accelerators."

**Missing move:** The cite-and-cite-away sentence names the consequence (rising ridge pushes low-reuse ops memory-bound) but functions as the entire treatment. There is no lead-out after the figure: the payoff paragraph (L2932) arrives over 150 lines later and discusses energy costs without connecting to the ridge figure. Removability test: delete the figure and the one-sentence citation disappears; the surrounding prose loses the quantitative trend evidence but carries forward without teaching the engineering implication. The figure's key story — that the H100 ridge point (~295 FLOP/byte) has roughly doubled from the V100 (~140 FLOP/byte), meaning a layer must now achieve twice the reuse to escape the memory-bound regime — is never stated in body prose.

**Where takeaway lives:** Only in the figure caption and alt-text. Body prose has no lead-out.

**Rule-compliant rewrite (add immediately after the float definition as a standalone lead-out paragraph):**

> The ridge point roughly doubled from V100 to H100 (approximately 140 to 295 FLOP/byte), meaning that any layer with arithmetic intensity below 295 FLOP/byte is memory bound on an H100 even though the same layer was compute bound on a V100. Attention with small batch sizes, depthwise convolutions, and normalization layers all fall in this gap. Deploying a newer, faster accelerator on these operations without algorithmic changes produces less speedup than the headline TFLOP/s ratio suggests, because the bottleneck has shifted from compute to memory traffic.

---

### Listings

---

#### ⚠️ `lst-dense_layer_def` (🟡 Listing) — def L1148

**Ref sentence (L1146):**
> "@Lst-dense_layer_def demonstrates how a dense layer decomposes at the framework level, encapsulating thousands of multiply-accumulate operations in a single high-level call."

**Missing move:** The cite sentence names what the listing shows (framework-level decomposition, thousands of MACs hidden in one call). However, it does not tell the reader what to look for in the code: specifically, that the mechanism is the abstraction boundary between the single `nn.Linear` call and the underlying weight matrix multiplication, and that this boundary is precisely where hardware acceleration engages. The payoff paragraph (L1157) immediately transitions the reader to @Lst-dense_expansion without completing the lead-out for this listing. The design choice that matters (what the API hides and why that matters for hardware) is only gestured at.

**Where takeaway lives:** Partially in the caption; the payoff paragraph (L1157) pivots rather than completing.

**Rule-compliant rewrite (extend the cite sentence to add the design-choice framing):**

> @Lst-dense_layer_def demonstrates how a dense layer decomposes at the framework level: notice that the single `nn.Linear` call hides the weight matrix shape and MAC count from the developer entirely, which is the abstraction that allows the framework to substitute hardware-specific matrix kernels without changing application code.

---

#### ⚠️ `lst-linear_matrix_hierarchy` (🟡 Listing) — def L1255

**Ref sentence (L1253):**
> "Neural network computations decompose into hierarchical matrix operations\index{Matrix Operation!hierarchical decomposition}. @Lst-linear_matrix_hierarchy captures this hierarchy through a linear layer that transforms input features into output neurons over a batch."

**Missing move:** The cite paragraph names the mechanism (hierarchical matrix operations, linear layer over a batch) but gives no framing of what to notice in the code: specifically, that the listing reveals three distinct operation types in sequence (matrix multiply, bias add, activation), each mapping to a different hardware primitive (tensor core, vector unit, SFU). Without that observation in prose, the reader sees Python code without knowing which line is the dominant cost or why the three operations require different silicon. The payoff paragraph (L1273) gives scale numbers but still does not name the three-primitive insight as the observation to draw from this listing.

**Where takeaway lives:** The three-primitive insight appears later in L1193 (payoff for `lst-loop_level_dense`), not here.

**Rule-compliant rewrite (replace the cite sentence):**

> @Lst-linear_matrix_hierarchy reveals this hierarchy directly: the `matmul` on line 3 is the dominant cost (over 95 percent of time), the bias add on line 5 is a vector operation, and the ReLU on line 7 is a special-function operation. Each maps to a different hardware primitive, which is why a single `nn.Linear` call touches three distinct silicon units.

---

#### ⚠️ `lst-matrix_unit` (🟡 Listing) — def L1307

**Ref sentence (L1305):**
> "This pervasive pattern of matrix multiplication has direct implications for hardware design\index{Matrix Operation!hardware acceleration}: accelerators need specialized units that can handle these computations at scale. @Lst-matrix_unit demonstrates how modern processors implement dedicated matrix units that process entire $16{\times}16$ blocks simultaneously, achieving 32$\times$ higher throughput than vector processing alone."

**Missing move:** The cite includes a throughput claim (32x over vector) but does not explain the mechanism that produces that gain: the key insight is that the 16x16 block bypasses 256 individual scalar fetch-multiply-accumulate cycles and replaces them with a single hardware-issued tile operation, keeping operands on chip. The payoff paragraph (L1320) gives the 256-MAC-per-instruction figure but also does not explain the on-chip register retention as the mechanism. The design-choice that matters (why blocking into tiles changes the memory access profile, not just the instruction count) is absent.

**Where takeaway lives:** Count is in payoff; mechanism is not stated in body prose.

**Rule-compliant rewrite (add to the end of the cite paragraph as a second sentence):**

> The key mechanism is that the tile remains in registers throughout the block computation: unlike 256 scalar instructions that each fetch operands from memory, the matrix unit loads the tile once and keeps it on chip, converting 256 memory round-trips into one.

---

#### ⚠️ `lst-arm_sve_vector` (🟡 Listing) — def L1461

**Ref sentence (L1459):**
> "SIMD execution applies identical operations to multiple data elements in parallel, minimizing instruction overhead while maximizing data throughput. [...] @Lst-arm_sve_vector demonstrates this approach."

**Missing move:** The cite says the listing "demonstrates this approach" but does not name the specific mechanism to observe in the code. The reader needs to know what to look for: specifically, that the predicate register (`p0`) enables scalable width (the key SVE innovation over fixed-width SIMD), and that a single `fmul`/`fadd` pair replaces N scalar multiply-add sequences. The payoff paragraph (L1473) expands SIMD capabilities in general but does not interpret the SVE listing's specific design choice. The listing is cited as a demonstration without an observation.

**Where takeaway lives:** Not stated in body prose near the float.

**Rule-compliant rewrite (extend the cite sentence):**

> @Lst-arm_sve_vector demonstrates this approach: the predicate register `p0` on line 1 is the scalable-width innovation, letting the same code run on 128-bit to 2048-bit implementations without recompilation, while the two arithmetic instructions (`fmul`, `fadd`) replace an entire scalar loop.

---

#### ⚠️ `lst-cuda_simt` (🟡 Listing) — def L1483

**Ref sentence (L1475):**
> "[...] Threads are organized into warps\index{Warp!execution unit}[^fn-warp-divergence], which are the basic execution units that enable SIMT efficiency. @Lst-cuda_simt shows this parallel processing model in action."

**Missing move:** "Shows this parallel processing model in action" is a pure pointer with no observation. The listing shows a CUDA kernel where each thread computes one output element using its `(row, col)` coordinates derived from `blockIdx` and `threadIdx`. The key design choice visible in the code is that thread-to-data mapping (the `row`/`col` indexing) is the mechanism that makes all outputs independent and therefore parallel. The payoff paragraph does not appear until a footnote at L1503, which discusses CUDA ecosystem lock-in rather than interpreting the listing.

**Where takeaway lives:** Not stated in body prose.

**Rule-compliant rewrite (replace the pointer sentence):**

> @Lst-cuda_simt shows this parallel processing model: each thread derives its unique `(row, col)` output coordinates from `blockIdx` and `threadIdx` (lines 3-4), making every output element independent and allowing the hardware to dispatch thousands of threads without synchronization. The inner loop (lines 6-9) is identical across all threads, which is the SIMT invariant that lets the SM execute one warp of 32 threads with a single instruction fetch.

---

#### ⚠️ `lst-input_stationary` (🟡 Listing) — def L4613

**Ref sentence (L4611):**
> "@Lst-input_stationary illustrates this approach, maximizing reuse by keeping input activations stationary in local memory while dynamically streaming weights."

**Missing move:** The cite sentence names the mechanism (activations stationary, weights streaming) and the payoff paragraph (L4630) correctly continues by noting that inputs are loaded once and held. However, the cite sentence does not name what the reader should look at in the code to verify this: specifically, the outer loop over input activations with the weight stream in the inner loop, and where the partial-sum write happens. More importantly, there is no stated consequence: why this is the right strategy for transformers and large-batch inference, which was set up in the lead-in but not completed in the cite sentence itself. The payoff paragraph is thin (two sentences) and does not name the transformer scenario or the conditions under which this dataflow wins over weight-stationary or output-stationary.

**Where takeaway lives:** The lead-in paragraph (L4609) names the condition; payoff (L4630) is thin.

**Rule-compliant rewrite (extend the cite sentence):**

> @Lst-input_stationary illustrates this approach: the outer loops load each input activation block once (line 3) and keep it in place while the weight stream passes through, so the same activation data serves all weight rows without a second memory fetch. This wins when the model visits each activation across many weight-matrix rows, as happens when a transformer processes the same token through multiple attention heads simultaneously.

---

### Tables

---

#### ⚠️ `tbl-roofline-operations` (🟠 Table) — def L3736

**Ref sentence (L3724-3725):**
> "Depthwise convolution\index{Depthwise Convolution!low arithmetic intensity}, embedding lookup\index{Embedding Lookup!memory-bound operation}, LayerNorm\index{LayerNorm!memory-bound operation}, and softmax\index{Softmax!memory-bound operation} are useful low-intensity reference points because they spend more time moving bytes than doing arithmetic.
> @Tbl-roofline-operations maps common neural network operations to the Roofline model."

**Missing move:** The cite is preceded by a one-sentence setup that names four memory-bound examples and then "maps common operations" — but this is a list announcement rather than a stated conclusion. The table's decision-driving insight is that the operations covering most of a transformer's wall-clock time (attention softmax at 2-5 FLOP/byte, LayerNorm at 1-2 FLOP/byte, embedding lookup at <1 FLOP/byte) all sit far below typical ridge points, and no amount of compute-side optimization can fix them. That conclusion is not stated in body prose. The payoff paragraph (L3738) says "to see how these intensity values translate..." which is a pivot, not a takeaway.

**Where takeaway lives:** Only in the table cells and caption.

**Rule-compliant rewrite (add a lead-out sentence after the @Tbl-roofline-operations citation):**

> The key result is that the operations consuming most transformer wall-clock time, attention softmax (2-5 FLOP/byte), LayerNorm (1-2 FLOP/byte), and embedding lookup (below 1 FLOP/byte), all sit far below any modern accelerator's ridge point. For these operations, upgrading to a faster GPU increases peak TFLOP/s but not actual throughput; the binding constraint is how fast bytes can be moved, not how fast arithmetic can be executed.

---

#### ⚠️ `tbl-tiling-strategies` (🟠 Table) — def L5077

**Ref sentence (L5065):**
> "@Tbl-tiling-strategies provides a comparative overview of spatial, temporal, and hybrid tiling approaches, highlighting their respective benefits and trade-offs."

**Missing move:** The cite is a pure "provides a comparative overview" pointer. The table contains the decision rule (when to use each strategy) that the reader actually needs, but the body prose never states which strategy dominates in which ML workload context, or what the critical trade-off is that makes the choice non-obvious. The payoff paragraph (L5079) says "tiling remains a critical tool" which is a restatement of what has already been said, not a conclusion drawn from the table. The lead-in (L5063-L5023) discusses register blocking and compilers generally; it does not provide the framing sentence that would make the table's "Best When" column unnecessary.

**Where takeaway lives:** In the "Best When" and "Common Use Cases" cells of the table.

**Rule-compliant rewrite (replace the cite sentence):**

> @Tbl-tiling-strategies maps these strategies to their winning conditions: spatial tiling suits large tensor workloads where the data footprint exceeds fast memory and locality along the spatial dimension is the binding constraint; temporal tiling suits iterative computations such as convolutions and RNNs where the same weights are revisited across iterations; hybrid tiling, used by production AI compilers, applies both simultaneously when neither alone saturates the available reuse.

---

#### 🛑 `tbl-runtime-comparison` (🟠 Table) — def L5350

**Ref sentence (L5337):**
> "@Tbl-runtime-comparison highlights the key differences between traditional and AI runtimes. One of the key distinctions lies in execution flow. Traditional software runtimes operate on a predictable, structured execution model where function calls and CPU threads follow a predefined control path. AI runtimes, however, execute computational graphs, requiring complex scheduling decisions that account for dependencies between tensor operations, parallel kernel execution, and efficient memory access."

**Missing move:** The cite sentence narrates cells from the table in prose form (execution flow, control path vs. computational graph). This is the operations-manual pattern — describing what the table contains rather than stating the conclusion it drives. The "key result" that the table encodes is never stated: that the structural divergence in execution model (sequential instruction stream vs. dataflow graph) means that traditional profiling tools, debugging practices, and runtime tuning assumptions are invalid when applied to AI workloads. The payoff paragraph (L5352) says "AI runtimes are inherently designed for adaptability" which continues narrating cells rather than completing the engineering argument. Removability test: delete the table and the prose still reads as a generic comparison of traditional vs. AI runtimes with no structural information lost — the prose is doing the table's job without reaching the table's conclusion.

**Where takeaway lives:** Implied by the comparison but never stated as an engineering implication.

**Rule-compliant rewrite (replace the cite sentence with a lead-in that sets up and a lead-out that concludes):**

Lead-in (keep the prior setup paragraph); replace the cite sentence with:

> @Tbl-runtime-comparison makes the divergence concrete across six dimensions. The decision-driving row is Optimization Priorities: where a traditional runtime minimizes instruction latency, an AI runtime minimizes memory stalls, which means every assumption a developer carries over from CPU profiling — optimize the hot function, reduce branch mispredictions, shrink stack allocations — points in the wrong direction for an AI workload.

---

#### ⚠️ `tbl-hw-acceleration-systolic-dataflow` (🟠 Table) — def L2293

**Ref sentence (L2281):**
> "@Tbl-hw-acceleration-systolic-dataflow previews the three stationary-operand strategies and the workloads each favors; @sec-hardware-acceleration-dataflow-optimization-strategies-ce52 develops each one in full as a general mapping decision."

**Missing move:** The cite is a forward-reference pointer ("previews ... develops later"). For a table that drives a concrete architectural decision (which data to keep stationary), a preview citation without any stated conclusion leaves the reader without orientation. The callout lead-in (L2281, "The architects' dilemma") frames the choice but the prose that follows the cite points ahead rather than stating even the preview-level conclusion. The reader cannot answer "what is the default right choice for a CNN vs. a transformer?" from body prose near this table.

**Where takeaway lives:** Only in the table cells and in the subsequent callout text that follows the table.

**Rule-compliant rewrite (extend the cite sentence to add the preview conclusion):**

> @Tbl-hw-acceleration-systolic-dataflow previews the three stationary-operand strategies and the workloads each favors: weight-stationary favors CNNs where the same filter is reused across many spatial positions, output-stationary favors fully connected layers and attention where partial sums accumulate heavily, and input-stationary favors transformers at large batch where the same activation serves many weight rows. @sec-hardware-acceleration-dataflow-optimization-strategies-ce52 develops each one in full.
