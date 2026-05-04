# Math Audit Report: `book/quarto/contents/vol2/parts/fleet_principles.qmd`

## Checked scope

Audited `book/quarto/contents/vol2/parts/fleet_principles.qmd` for mathematical statements, displayed equations, numeric claims, unit consistency, complexity claims, and prose-equation consistency using direct reasoning only. No Gemini assistance was used.

The file contains four displayed equations and several quantitative or count-based statements: a 100 kW rack heat example, model-parameter and GPU-memory growth rates, a 10,000 GPU / 1 Gbps network example, a storage-throughput utilization bound, and scaling-efficiency values.

## Findings

### Medium: Part numbering is internally inconsistent

- **Lines:** 1, 53, 55
- **Current text:** The title says `# Part I: The Fleet`, but line 53 says "Part V builds this machine," and line 55 introduces "Part V Roadmap."
- **Issue:** This is a numeric/prose consistency problem. Within the same file, the Fleet part is identified as both Part I and Part V. A reader cannot tell whether the roadmap belongs to the first part of Volume 2 or the fifth part of the broader curriculum.
- **Proposed correction:** Use one part number consistently. If this file is intended to be the first part of Volume 2, change lines 53 and 55 to "Part I." If the global curriculum numbering is intended, change line 1 to `# Part V: The Fleet`.

### Low: Roadmap block is duplicated

- **Lines:** 57-69
- **Current text:** Lines 57-63 introduce and list four chapters, then lines 64-69 repeat the same introductory sentence and the same four-item list.
- **Issue:** The duplicate block is not a mathematical error, but it is a count/prose consistency defect: the roadmap appears twice, which can make downstream references to "this part" and "four chapters" look accidental or stale.
- **Proposed correction:** Delete one copy of the duplicated roadmap block. Keep either lines 57-63 or lines 64-69, not both.

### Low: Parameter-vs-memory growth rates are stated as an invariant without scope

- **Lines:** 20-23
- **Current text:** "Model parameter growth (10$\times$/year) consistently outpaces GPU memory capacity growth (2$\times$/year)."
- **Issue:** The arithmetic implication is clear: if those rates held, the parameter-to-memory gap would grow by a factor of `10/2 = 5` per year. The problem is the word "consistently" inside an invariant. Parameter counts and GPU memory capacity are not governed by fixed universal annual multipliers; they vary by model family, market cycle, product generation, precision, sparsity, and parallelization strategy. As written, the statement presents illustrative historical rates as a stable law.
- **Proposed correction:** Recast the numbers as representative examples rather than an invariant. For example: "In recent frontier-model cycles, parameter growth has often outpaced per-GPU memory growth; illustrative rates such as 10x/year for model scale versus 2x/year for memory imply a 5x/year pressure to shard state across devices."

### Medium: Scaling-efficiency prose mixes weak-scaling and strong-scaling assumptions

- **Lines:** 46-50
- **Current text/equation:** The invariant says communication overhead grows with cluster size "while per-node computation remains constant," followed by:
  `$$ \eta_{\text{scaling}} = \frac{T_1}{N \times T_N} \leq 1 $$`
- **Issue:** The equation is the standard strong-scaling efficiency form for a fixed total workload: `T_1` is the one-node time for the same job and `T_N` is the `N`-node time. In that setting, ideal scaling gives each node about `1/N` of the original work, so per-node computation does not remain constant. "Per-node computation remains constant" describes weak scaling, where total workload grows with `N`; the matching efficiency metric would not use the same fixed-workload `T_1/(N T_N)` interpretation. The prose and equation therefore describe different scaling regimes.
- **Proposed correction:** Choose one scaling regime. For strong scaling, revise the invariant to: "Adding nodes to a fixed distributed training job yields diminishing returns because per-node computation decreases while communication and synchronization overheads do not shrink proportionally." Keep the equation. For weak scaling, replace the equation with a weak-scaling efficiency metric such as `\eta_{\text{weak}} = T_1/T_N` for fixed work per node.

## No-Issue Checks

- **Lines 7-11:** `P_{\text{limit}} = \Delta Q/\Delta t` is dimensionally valid for power as heat flow rate. The 100 kW rack example is a qualitative threshold statement and does not include a worked conversion that needs correction.
- **Lines 32-36:** `\eta_{\text{specific}} \gg \eta_{\text{general}}` is dimensionless and directionally consistent if `\eta` denotes useful work or throughput per unit power. The prose overstates the trend as a "physical law," but the equation itself is not dimensionally inconsistent.
- **Lines 39-43:** `\text{BW}_{\text{storage}}/(N_{\text{GPU}} \times R_{\text{consumption}})` is dimensionless if `R_{\text{consumption}}` is per-GPU data-consumption bandwidth. The `min(1, ...)` bound is consistent with utilization being capped at 1.
- **Lines 26-29:** The 10,000 GPU / 1 Gbps Ethernet example is qualitative but internally coherent: the claim is that low bisection bandwidth can dominate collective operations. No numeric bandwidth calculation is provided to audit.
- **Lines 48-50:** Apart from the strong-vs-weak scaling prose issue above, the efficiency equation is algebraically consistent for strong scaling, and the stated `0.85`-`0.95` efficiencies are valid dimensionless fractions.
