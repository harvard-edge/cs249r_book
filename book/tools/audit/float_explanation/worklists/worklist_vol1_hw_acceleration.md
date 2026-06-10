# Float-explanation worklist — hw_acceleration.qmd (vol1)

## Summary

| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 14 | 12 | 2 | 0 |
| table | 24 | 24 | 0 | 0 |
| listing | 22 | 17 | 5 | 0 |
| algorithm | 1 | 1 | 0 | 0 |
| equation | 6 | 5 | 1 | 0 |
| **total** | **67** | **59** | **8** | **0** |

---

## Findings (⚠️ only — ✅ floats tallied above, not expanded)

### ⚠️ `fig-ai-performance` — def L1574  (Thin)

- **Caption:** GPU Performance Scaling: NVIDIA single-chip inference performance increased by more than 1,000× over a decade, from 3.9 TFLOP/s (FP32) on the K20X to about 4,000 TFLOP/s (FP8 sparse tensor operations) on the H100 and roughly 9,000 TFLOP/s on the B200. Three-orders-of-magnitude gain driven by architectural innovations: tensor core acceleration, reduced precision (FP16, INT8, FP8), and hardware-accelerated structured sparsity.
- **Ref(s):** L1572 `@fig-ai-performance`: "To appreciate the magnitude of this shift, trace the curve in @fig-ai-performance from left to right: over a single decade, NVIDIA GPU performance jumped roughly 1,000× as the architecture transitioned from general-purpose floating-point execution units to highly optimized tensor processing cores."
- **Context checked:** ref ✓ (magnitude stated) · prev ¶ ✓ (notes increasing specialization) · next ¶ = figure definition · payoff ¶ (L1711) ✗ (new section on processing elements, no connection to this figure) · caption ✓ (identifies drivers)
- **Issue:** The ref sentence delivers the magnitude but not the implication: *what does 1,000× in one decade mean for how a practitioner should think about hardware selection?* The payoff paragraph is the opening of the next section and addresses a completely different topic. A one- or two-sentence conclusion after the figure — connecting the growth curve to the architectural levers that drove it and why they are not exhausted — would anchor the takeaway.
- **Suggested rewrite (flag-only):**
  ```diff
  - The increasing specialization of AI hardware has driven measurable performance improvements in deep learning workloads. To appreciate the magnitude of this shift, trace the curve in @fig-ai-performance from left to right: over a single decade, NVIDIA GPU performance jumped roughly 1,000× as the architecture transitioned from general-purpose floating-point execution units to highly optimized tensor processing cores.
  + The increasing specialization of AI hardware has driven measurable performance improvements in deep learning workloads. To appreciate the magnitude of this shift, trace the curve in @fig-ai-performance from left to right: over a single decade, NVIDIA GPU performance jumped roughly 1,000× as the architecture transitioned from general-purpose floating-point execution units to highly optimized tensor processing cores. Each inflection on that curve corresponds to a discrete architectural bet — adding Tensor Cores, cutting precision from FP32 to FP16, then to FP8, and enabling structured sparsity — rather than to transistor scaling alone. The implication is that the remaining gains on the curve will come from the same source: deeper hardware-algorithm co-design, not passive Moore's Law.
  ```

---

### ⚠️ `fig-rising-ridge` — def L2775  (Thin)

- **Caption:** The Rising Ridge: Hardware arithmetic intensity (FLOP/byte) over time using dense FP16 tensor peaks and memory bandwidth. Ridge point rises from V100 through H100 and remains high on B200. This trend explains why architectures with high data reuse flourish while low-reuse workloads face a growing hardware tax.
- **Ref(s):** L2773 `@fig-rising-ridge`: "The imbalance has a direct architectural consequence visible in @fig-rising-ridge: the hardware ridge point has climbed sharply and remains high, pushing sparse and low-reuse operations further into the memory-bound regime on modern accelerators."
- **Context checked:** ref ✓ (states the trend) · prev ¶ (L2771) = closing `:::` of a callout, no prose · next ¶ = figure definition · payoff ¶ (L2932) ✗ (pivots entirely to energy costs and DRAM, never develops the rising ridge's implications for algorithm choice or hardware selection) · caption ✓ (names the consequence: "low-reuse workloads face a growing hardware tax")
- **Issue:** The ref sentence names the trend but stops there. The natural implication — that each GPU generation raises the bar on arithmetic intensity an operation must achieve to be compute-bound, meaning operations like LayerNorm and embedding lookup are *increasingly* disadvantaged over time — is never stated. The payoff paragraph (L2932) changes subject. A single bridging sentence after the float would close this gap.
- **Suggested rewrite (flag-only):**
  ```diff
  - The imbalance has a direct architectural consequence visible in @fig-rising-ridge: the hardware ridge point has climbed sharply and remains high, pushing sparse and low-reuse operations further into the memory-bound regime on modern accelerators.
  + The imbalance has a direct architectural consequence visible in @fig-rising-ridge: the hardware ridge point has climbed sharply and remains high, pushing sparse and low-reuse operations further into the memory-bound regime on modern accelerators. Because the ridge point rose from roughly 140 FLOP/byte on the V100 to roughly 295 FLOP/byte on the H100, an operation that was compute-bound in 2017 may be memory-bound on 2022-era hardware — the target moved without any change to the algorithm. This means that operator fusion and data-reuse strategies are not optional refinements but structural requirements on modern accelerators.
  ```

---

### ⚠️ `eq-batch-ai` — def L4190  (Thin)

- **Caption:** (none)
- **Ref(s):** L4189 `@Eq-batch-ai`: "Increasing batch size improves AI for matrix operations by amortizing weight loading. @Eq-batch-ai formalizes this relationship for a dense layer (B×M)×(M×N):"
- **Context checked:** ref ✓ (states what it formalizes) · prev ¶ = callout header · next ¶ (L4192) = "Example: Dense layer with M=N=2048 (FP16)" followed by three bullets with numbers · payoff ¶ = the bullets themselves, no prose sentence · context after callout (L4202) ✓ (explains the inference-serving batching tension) — but this is *after* the callout closes
- **Issue:** Inside the callout, the equation is followed only by a label line and three bullet numbers. No sentence within the callout extracts the key insight: that AI grows approximately linearly with B at large B, meaning doubling the batch size approximately doubles the arithmetic intensity, which can shift a workload from memory-bound to compute-bound. The post-callout prose (L4202) draws the serving implications but assumes the reader already understood what the equation showed. A one-sentence payoff inside the callout would complete it.
- **Suggested rewrite (flag-only):**
  ```diff
  - Example: Dense layer with M=N=2048 (FP16)
  -
  - - Batch = 1: AI ≈ `{python} BatchAiCalc.batch1_ai_str` FLOP/byte (memory bound)
  - - Batch = 32: AI ≈ `{python} BatchAiCalc.batch32_ai_str` FLOP/byte (memory bound)
  - - Batch = 256: AI ≈ `{python} BatchAiCalc.batch256_ai_str` FLOP/byte (compute bound on A100)
  + Example: Dense layer with M=N=2048 (FP16)
  +
  + - Batch = 1: AI ≈ `{python} BatchAiCalc.batch1_ai_str` FLOP/byte (memory bound)
  + - Batch = 32: AI ≈ `{python} BatchAiCalc.batch32_ai_str` FLOP/byte (memory bound)
  + - Batch = 256: AI ≈ `{python} BatchAiCalc.batch256_ai_str` FLOP/byte (compute bound on A100)
  +
  + Because AI grows approximately linearly with B in the large-weight regime, each doubling of batch size roughly doubles the arithmetic intensity. This is why single-sample inference sits deep in the memory-bound regime while batched inference can cross into the compute-bound regime and use the tensor cores efficiently.
  ```

---

### ⚠️ `lst-dense_layer_def` — def L1148  (Thin)

- **Caption:** Dense Layer Abstraction: High-level framework APIs encapsulate thousands of MACs in a single function call, hiding the computational complexity from developers while enabling automatic hardware optimization.
- **Ref(s):** L1146 `@Lst-dense_layer_def`: "@Lst-dense_layer_def demonstrates how a dense layer decomposes at the framework level, encapsulating thousands of multiply-accumulate operations in a single high-level call."
- **Context checked:** ref ✗ (announcement only, no insight) · prev ¶ = Python code block · next ¶ = figure definition · payoff (L1157) ✗ ("This single line of code conceals the computational complexity that accelerators must handle" — restates without extracting takeaway; immediately redirects to next listing) · caption ✓ (notes the hiding of complexity)
- **Issue:** This listing opens a three-step pedagogical cascade (dense_layer_def → dense_expansion → loop_level_dense). The ref sentence and immediate payoff both only point forward. No prose within the neighborhood draws the specific insight from *this* listing: that the `Dense(512)(input_tensor)` call maps to a fixed-cost matrix operation whose parameter count is known at compile time — which is precisely what makes it possible for a compiler to plan hardware resources. The cascade's final payoff (L1193) is excellent, but this first step contributes nothing standalone.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Lst-dense_layer_def demonstrates how a dense layer decomposes at the framework level, encapsulating thousands of multiply-accumulate operations in a single high-level call.
  + @Lst-dense_layer_def shows the framework level: a single call encapsulates a fixed, statically knowable number of multiply-accumulate operations whose count is determined entirely by the declared input and output dimensions. That static structure is what allows a compiler to plan memory allocation and hardware dispatch before a single sample has been seen.
  ```

---

### ⚠️ `lst-dense_expansion` — def L1159  (Thin)

- **Caption:** Matrix Operation Expansion: Each dense layer decomposes into matrix multiplication and element-wise operations, exposing the dominant compute pattern that consumes over 95 percent of neural network execution time.
- **Ref(s):** L1157 `@Lst-dense_expansion`: "This single line of code conceals the computational complexity that accelerators must handle. @Lst-dense_expansion reveals how the framework expands this high-level call into mathematical operations."
- **Context checked:** ref ✗ (announces the reveal, does not state what the reveal shows) · prev ¶ = `:::` closing · payoff (L1174) ✗ ("The matrix multiplication dominates computation time, but this abstraction still hides the underlying loop structure" — pivots immediately to next listing) · caption ✓ (identifies the 95 percent dominance claim)
- **Issue:** The ref sentence says the listing "reveals how the framework expands this high-level call into mathematical operations" without naming *what* that expansion shows. The payoff (L1174) acknowledges that matmul dominates but treats it as a stepping stone. The reader learns the dominance claim only from the caption. A one-sentence payoff in the body prose would complete the explanation.
- **Suggested rewrite (flag-only):**
  ```diff
  - This single line of code conceals the computational complexity that accelerators must handle. @Lst-dense_expansion reveals how the framework expands this high-level call into mathematical operations.
  + This single line of code conceals the computational complexity that accelerators must handle. @Lst-dense_expansion reveals that the expansion consists of two operations with very different hardware profiles: a matmul whose cost grows as O(batch × in_dim × out_dim) and an element-wise activation whose cost grows only as O(batch × out_dim) — the asymmetry that makes the matmul the bottleneck at all realistic scales.
  ```

---

### ⚠️ `lst-nonlinear_layer` — def L1343  (Thin)

- **Caption:** Nonlinear Transformations: Neural networks process input data through a sequence of linear transformations followed by nonlinear activations to capture complex patterns. This layer sequence enhances model expressiveness and learning capabilities.
- **Ref(s):** L1341 `@Lst-nonlinear_layer`: "To see why dedicated hardware matters, consider a typical layer sequence. @Lst-nonlinear_layer combines linear transformations with nonlinear activations—operations that appear simple in Python but reveal substantial computational complexity at the hardware level."
- **Context checked:** ref ✓ (frames the purpose) · payoff (L1354) = "This sequence introduces multiple nonlinear transformations that extend beyond simple matrix operations" — partial, immediately redirects to `lst-nonlinear_math` · caption ✗ (generic, no hardware insight)
- **Issue:** The ref sentence sets up the "why dedicated hardware matters" question but the listing is never answered within its own neighborhood — the answer is deferred entirely to the math expansion listing. A sentence noting *which* operation in `nn.Sequential(Linear, ReLU, BatchNorm)` is the outlier that requires dedicated hardware would complete this listing's role.
- **Suggested rewrite (flag-only):**
  ```diff
  - This sequence introduces multiple nonlinear transformations that extend beyond simple matrix operations. @Lst-nonlinear_math breaks down these operations into their mathematical components, exposing the computational complexity that hardware must address.
  + This sequence already contains three qualitatively different hardware demands in four lines: the linear layer maps to matmul (handled by tensor cores), ReLU maps to a conditional max (handled by a special function unit), and BatchNorm requires two reduction passes for mean and variance before the normalization itself. @Lst-nonlinear_math breaks down these operations into their mathematical components to make those three demands explicit.
  ```

---

### ⚠️ `lst-arm_sve_vector` — def L1461  (Thin)

- **Caption:** Vector Operation: Vector multiplication and addition operations enable efficient parallel processing in machine learning models.
- **Ref(s):** L1459 `@Lst-arm_sve_vector`: "The Arm Scalable Vector Extension (SVE) provides a representative example of how modern architectures implement scalable SIMD operations efficiently. @Lst-arm_sve_vector demonstrates this approach."
- **Context checked:** ref ✗ (bare announcement: "demonstrates this approach") · prev ¶ ✓ (explains SIMD concept and the 512 → 32–64 instructions reduction) · payoff (L1473) ✗ ("Processor architectures continue to expand SIMD capabilities to accommodate increasing computational demands" — generic transitional sentence that could follow any SIMD listing) · caption ✗ (generic, no specifics from the code)
- **Issue:** The key differentiating feature of the SVE listing — scalable predication (the `ptrue p0.s` instruction that adapts to the hardware's native vector length without recompilation) — is never named in the surrounding prose. The payoff sentence is a generic bridge to the next topic. A reader finishes this listing knowing only "SIMD is faster" but not "what SVE specifically adds over fixed-width SIMD."
- **Suggested rewrite (flag-only):**
  ```diff
  - Processor architectures continue to expand SIMD capabilities to accommodate increasing computational demands. Intel's Advanced Matrix Extensions (AMX) [@intel2021amx] and Arm's SVE architecture [@stephens2017] provide flexible execution models, enabling software to scale across different hardware implementations.
  + The key feature in this sequence is the `ptrue p0.s` predicate: SVE allows the same binary to run across hardware with different native vector widths (128-bit to 2048-bit) by querying the hardware vector length at runtime. This scalability means a model kernel compiled once can achieve full vector utilization on a phone core and a server core without recompilation — eliminating the fixed-width rewrite cycle that traditional SIMD required. Intel's Advanced Matrix Extensions (AMX) and Arm's SVE [@stephens2017; @intel2021amx] represent successive steps in this same direction toward flexible, width-agnostic execution models.
  ```
