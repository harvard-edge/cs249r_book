# Math Audit Report: `book/quarto/contents/vol1/parts/build_principles.qmd`

## Checked scope

Audited `book/quarto/contents/vol1/parts/build_principles.qmd` for mathematical statements, displayed equations, numeric claims, unit consistency, complexity claims, and prose-equation consistency using direct reasoning only. The file contains one displayed timing equation and several qualitative bottleneck claims. No explicit numeric calculations or unit conversions are present.

## Findings

### Medium: Timing equation is stated as total time but the overlap prose requires a different form

- **Lines:** 6-10
- **Current text/equation:** Lines 6-8 state that total time is governed by data movement, compute, and fixed overhead:
  `$$ T = \frac{D_{\text{vol}}}{BW} + \frac{O}{R_{\text{peak}} \cdot \eta_{\text{hw}}} + L_{\text{lat}} $$`
  Line 10 then says that when stages overlap, wall-clock time is dominated by whichever term is largest and that the practical lesson is dominance, not summation.
- **Issue:** The displayed equation is dimensionally consistent: bytes divided by bytes/second, operations divided by operations/second, and latency all have units of time. The problem is prose-equation consistency. A sum is appropriate for non-overlapped or serialized costs. If data movement and compute overlap, the lower-bound/dominant-time model is closer to a maximum over overlapped terms, with fixed non-overlapped latency added separately when applicable. As written, the invariant first defines `T` as a serial sum, then immediately says the lesson is not summation. That leaves unclear whether the equation is meant as an accounting upper bound, a non-overlapped model, or a roofline-style dominance model.
- **Proposed correction:** Scope the displayed equation to serialized execution and add an overlapped form. For example:
  `$$ T_{\text{serial}} = \frac{D_{\text{vol}}}{BW} + \frac{O}{R_{\text{peak}}\eta_{\text{hw}}} + L_{\text{lat}} $$`
  and
  `$$ T_{\text{overlap}} \approx \max\!\left(\frac{D_{\text{vol}}}{BW}, \frac{O}{R_{\text{peak}}\eta_{\text{hw}}}\right) + L_{\text{lat}} $$`
  if `L_{\text{lat}}` is not hidden by overlap. Alternatively, revise line 6 to say the equation is a simplified serial cost decomposition, then keep line 10 as the overlapped interpretation.

### Low: "Increase data movement" points to a time expression, not data volume

- **Lines:** 12
- **Current text/equation:** "Unstructured pruning reduces compute (`$O$`) but introduces irregular memory access patterns that can increase data movement (`$D_{\text{vol}}/\text{BW}$`)."
- **Issue:** `$D_{\text{vol}}/\text{BW}$` has units of time, not data volume. The prose says "increase data movement," which should refer to `D_{\text{vol}}` or to memory traffic in bytes. Irregular access can also reduce effective bandwidth, so the slowdown may come from larger effective `D_{\text{vol}}`, lower effective `BW`, or both.
- **Proposed correction:** Replace the parenthetical with either `D_{\text{vol}}` or movement time. For example: "can increase effective memory traffic (`$D_{\text{vol}}$`) and/or reduce effective bandwidth, increasing `$D_{\text{vol}}/\text{BW}_{\text{eff}}$`."

### Medium: Llama-3-8B bandwidth-bound claim is under-scoped

- **Lines:** 20-22
- **Current text/equation:** The list contrasts ResNet-50 as compute-bound, Llama-3-8B as bandwidth-bound with performance limited by `$D_{\text{vol}}/\text{BW}$`, and DLRM as capacity-bound.
- **Issue:** The Llama-3-8B statement is plausible for token-by-token inference, where repeatedly streaming weights and KV-cache state can make memory bandwidth the dominant bottleneck. It is not generally true for all Llama-3-8B operations. Training and long prefill/batched matrix-multiplication phases can be compute-bound depending on batch size, sequence length, precision, hardware, and implementation efficiency. Because the bullets are presented as architecture-level contracts, the current wording overgeneralizes a workload-dependent bottleneck.
- **Proposed correction:** Scope the example to the relevant phase. For example: "**Llama-3-8B autoregressive decoding** often assumes high-bandwidth memory access. In small-batch token generation it is frequently **Bandwidth-Bound**: performance is limited by `$D_{\text{vol}}/\text{BW}_{\text{eff}}$`." If the intended claim covers training or prefill, add the condition under which compute rather than bandwidth dominates.

## No-Issue Checks

- **Lines 7-8:** The variables in the iron-law equation have compatible dimensions if `BW` is bytes/second, `R_{\text{peak}}` is floating-point operations/second, `O` is floating-point operations, `\eta_{\text{hw}}` is dimensionless, and `L_{\text{lat}}` is seconds.
- **Lines 15-25:** The qualitative "Silicon Contract" framing is internally consistent as a bottleneck-selection principle. No numeric proof is implied.
- **Line 22:** The DLRM capacity-bound example is qualitatively reasonable for embedding-heavy workloads whose tables or working set exceed fast-memory capacity, though bandwidth and latency may still dominate after capacity is satisfied.
- **Line 27:** The closing summary does not introduce additional equations, numeric claims, or unit-sensitive statements.
