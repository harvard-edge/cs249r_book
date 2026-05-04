# Math Audit Report: `book/quarto/contents/vol1/parts/optimize_principles.qmd`

## Checked scope

Audited `book/quarto/contents/vol1/parts/optimize_principles.qmd` for mathematical statements, displayed equations, numeric claims, unit consistency, complexity claims, and prose-equation consistency using direct reasoning only. The file contains three displayed equations, one bottleneck table, and several qualitative optimization claims. No external model or Gemini assistance was used.

## Findings

### Low: Roofline prose says "bounded" but the equation states equality

- **Lines:** 18-19
- **Current text/equation:** Line 18 says attainable throughput `R` is "bounded by" peak compute and the bandwidth-scaled arithmetic intensity, followed by:
  `$$ R = \min(R_{\text{peak}}, I \times \text{BW}) $$`
- **Issue:** The units are consistent if `I` is operations per byte and `BW` is bytes per second, because `I \times BW` has units of operations per second. The consistency problem is that "bounded by" implies an upper bound, while the displayed equation asserts that the bound is always achieved. Real kernels can fall below this ideal because of imperfect utilization, memory access inefficiency, instruction mix, synchronization, or launch overhead. If `R` is intended to mean the ideal roofline ceiling, equality is fine, but the prose calls it "attainable throughput," which reads as observed or achievable throughput.
- **Proposed correction:** Either make the equation explicitly an upper bound:
  `$$ R_{\text{attainable}} \le \min(R_{\text{peak}}, I \times \text{BW}) $$`
  or define the displayed quantity as the ideal roofline ceiling:
  `$$ R_{\text{roof}} = \min(R_{\text{peak}}, I \times \text{BW}) $$`
  and then state that actual throughput satisfies `R \le R_{\text{roof}}`.

### Medium: Latency-bound batching claim conflates throughput latency with per-request latency

- **Lines:** 30
- **Current text/equation:** The latency-bound row lists dominant term `$L_{\text{lat}}$`, "Batching, kernel fusion, async dispatch" as optimizations that work, and "More FLOP/s or bandwidth alone" as wasted.
- **Issue:** Batching can reduce amortized overhead per item when the latency term is a fixed per-kernel, per-dispatch, or per-request setup cost. However, batching can increase the wall-clock latency seen by an individual request because it may wait for batch formation and then share execution with other items. The row is mathematically correct only if the optimized metric is throughput or per-item amortized latency, not necessarily tail latency or single-request latency.
- **Proposed correction:** Scope the batching entry. For example: "Batching for amortized per-item latency/throughput, kernel fusion, async dispatch." If the intended metric is single-request latency, replace or qualify batching with "microbatching only when queueing delay is bounded."

### Medium: DRAM energy comparison mixes bit-level data movement with operation-level arithmetic

- **Lines:** 37-38
- **Current text/equation:** Line 37 states: "Moving 1 bit of data from DRAM costs 100--1,000`$\times$` more energy than performing an arithmetic operation on it." The displayed equation is:
  `$$ E_{\text{move}} \gg E_{\text{compute}} $$`
- **Issue:** The inequality is directionally correct, but the numeric comparison is under-specified and mixes granularities. "Moving 1 bit" is a bit-level event, while "an arithmetic operation" is usually a word-level operation whose energy depends strongly on precision and operation type, such as integer add, floating-point add, multiply, or fused multiply-add. A fair comparison should either compare word movement to word arithmetic or define energy per bit and energy per operation separately. As written, the 100--1,000x range may be interpreted as universal across all arithmetic precisions and all DRAM transfers, which is too strong.
- **Proposed correction:** Define the comparison at the same granularity and make it architecture-dependent. For example: "Moving a word from DRAM can cost orders of magnitude more energy than a simple arithmetic operation on that word, with the ratio depending on precision and hardware." If retaining the numeric range, write: "For typical word-level DRAM accesses versus simple on-chip arithmetic, the energy ratio is often on the order of `10^2` to `10^3`."

### Low: Memory-bound row overstates pruning as generally effective

- **Lines:** 28
- **Current text/equation:** For memory-bound workloads with dominant term `$D_{\text{vol}}/\text{BW}$`, the table lists "Quantization, pruning, batching" as optimizations that work.
- **Issue:** Quantization directly reduces bytes moved, and batching can increase weight reuse and arithmetic intensity in some inference regimes. Pruning only reduces the memory-bound term if it actually reduces effective memory traffic without adding offsetting sparse-index overhead or irregular-access penalties. Unstructured pruning can fail to reduce `D_{\text{vol}}/\text{BW}_{\text{eff}}` on dense hardware, even when it reduces nominal parameter count.
- **Proposed correction:** Qualify pruning as structured or hardware-supported. For example: "Quantization, structured/hardware-supported pruning, batching when it improves reuse." This preserves the table's intent while matching the memory-traffic model.

## No-Issue Checks

- **Lines 5-12:** The Pareto frontier definition is mathematically consistent: a Pareto-optimal point is one where improving one objective requires worsening at least one other objective, assuming the feasible set and metrics are fixed.
- **Lines 18-19:** Aside from the equality-versus-bound issue above, the arithmetic-intensity equation is dimensionally valid: operations/byte multiplied by bytes/second gives operations/second.
- **Lines 26-32:** The bottleneck table's dominant terms have compatible dimensions when `D_{\text{vol}}` is bytes, `BW` is bytes/second, `O` is operations, `R_{\text{peak}}` is operations/second, `\eta_{\text{hw}}` is dimensionless, and `L_{\text{lat}}` is time.
- **Line 47:** Amdahl's Law is stated correctly as `1 / ((1-p) + p/s)` for accelerating fraction `p` by factor `s`.
- **Line 50:** The numeric example is correct: with `p = 0.95` and `s = 100`, total speedup is `1 / (0.05 + 0.95/100) = 16.8067`, so `~16.8x` is accurate.
- **Line 53:** The D-A-M taxonomy sentence introduces no additional mathematical, numeric, unit, or complexity claim requiring correction.
