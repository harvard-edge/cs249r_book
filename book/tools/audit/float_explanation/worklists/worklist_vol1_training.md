# Float-explanation worklist — training.qmd (vol1)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 19 | 18 | 1 | 0 |
| table | 18 | 16 | 2 | 0 |
| listing | 7 | 7 | 0 | 0 |
| algorithm | 3 | 3 | 0 | 0 |
| equation | 18 | 18 | 0 | 0 |
| **total** | **65** | **62** | **3** | **0** |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

---

### ⚠️ `fig-data-pipeline` — def L2037  (Thin)
- **Caption:** **CPU-to-GPU Data Flow**: Three distinct zones compose the data pipeline: the storage zone houses raw data on disk, the CPU preprocessing zone handles format conversion, processing, and batching, and the GPU training zone distributes preprocessed batches across multiple accelerator workers for parallel computation.
- **Ref(s):** L2035 `@Fig-data-pipeline`: "The data pipeline running on the CPU bridges raw data storage and accelerator computation. @Fig-data-pipeline breaks down this architecture into three distinct zones."
- **Context checked:** ref ✗ (announces structure, no takeaway) · prev ¶ ✓ (identifies the data pipeline's role in utilization) · next ¶ — the figure definition · payoff ¶ at L2287 ✓ (explains why each zone can become the bottleneck) — but this payoff is ~250 lines below the figure, well outside the local neighborhood
- **Issue:** The single ref sentence tells the reader there are three zones but says nothing about what those zones reveal or why the architecture matters. The substantive payoff (any zone can become the slowest stage; format conversion and batching are throughput gates) is far below the figure. Readers encounter the figure with no guidance about what to look for.
- **Suggested rewrite (flag-only):**
  ```diff
  - The data pipeline running on the CPU bridges raw data storage and accelerator computation. @Fig-data-pipeline breaks down this architecture into three distinct zones.
  + The data pipeline running on the CPU bridges raw data storage and accelerator computation. @Fig-data-pipeline breaks down this architecture into three distinct zones — and the key insight is that each zone can become the binding constraint: storage supplies raw examples, CPU preprocessing converts and batches them, and the GPU training zone consumes the result. If any stage runs slower than its neighbors, the accelerator idles while the pipeline catches up.
  ```
  Note: The rewrite above uses an em-dash which violates house style. A rule-compliant alternative:
  ```diff
  - The data pipeline running on the CPU bridges raw data storage and accelerator computation. @Fig-data-pipeline breaks down this architecture into three distinct zones.
  + The data pipeline running on the CPU bridges raw data storage and accelerator computation. @Fig-data-pipeline breaks down this architecture into three distinct zones, each of which can become the slowest stage: storage supplies raw examples, CPU preprocessing converts and batches them, and the GPU training zone consumes the result. When any stage runs slower than its neighbors, the accelerator idles while the pipeline catches up.
  ```

---

### ⚠️ `tbl-optimization-roadmap` — def L3444  (Thin)
- **Caption:** **Optimization Technique Roadmap**: Each primary bottleneck category has targeted solutions that address specific performance constraints, matching techniques to profiling results for systematic optimization.
- **Ref(s):** L3433 `@Tbl-optimization-roadmap`: "@Tbl-optimization-roadmap extends the D·A·M-based bottleneck classification from @tbl-dam-training-bottlenecks by mapping each bottleneck to the specific optimization technique that addresses it."
- **Context checked:** ref ✗ (structural description, no takeaway) · prev ¶ ✓ (explains the utilization gap and motivation for optimization) · next ¶ ✓ (explains how bottlenecks interact and gives GPT-2 example) · caption ✓ (describes the mapping purpose) · payoff ¶ at L3446 ✓ (explains conservation of complexity) — but the ref sentence gives no guidance on what the table's mappings show
- **Issue:** The ref describes the table structurally ("extends … by mapping") without telling the reader what the mapping reveals. The payoff paragraph shifts immediately to the conservation-of-complexity principle rather than unpacking the table's specific bottleneck-to-technique assignments. The reader is not told, for instance, that this is the table that establishes which optimization to reach for first given a profiling result.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Tbl-optimization-roadmap extends the D·A·M-based bottleneck classification from @tbl-dam-training-bottlenecks by mapping each bottleneck to the specific optimization technique that addresses it.
  + @Tbl-optimization-roadmap maps each D·A·M bottleneck to its primary technique, giving the practitioner a direct lookup from profiling result to optimization action: data-bound systems need prefetching, compute-bound systems need mixed precision, and memory-capacity bottlenecks need gradient accumulation or activation checkpointing.
  ```

---

### ⚠️ `tbl-scaling-decision` — def L6522  (Thin)
- **Caption:** **Scaling Decision Guidelines**: Model size, dataset scale, and available hardware determine when distributed training complexity is justified. Single-machine optimization provides better cost-efficiency below these thresholds.
- **Ref(s):** L6513 `@Tbl-scaling-decision`: "@Tbl-scaling-decision provides quantitative guidance for scaling decisions across different model and data scales."
- **Context checked:** ref ✗ (empty pointer) · prev ¶ at L6508-6511 ✗ (lists four single-machine optimizations to exhaust first, unrelated to the table content) · next ¶ at L6524 ✗ (explains when distributed is necessary via three hard limits, but does not recap the table thresholds) · caption ✓ (names the organizing principle) · payoff ¶ at L6524 ✓ (explains conditions for going distributed) — but the specific threshold content of the table (sub-billion, 1-10B, above-10B, above-10TB) is never unpacked in prose
- **Issue:** The ref sentence is a pure pointer with no content. Neither the preceding nor following prose tells the reader what the parameter-scale thresholds are or what rationale drives them. The caption names the principle but the table's actual guidance (for example, why 1-10B fits on a single node, or what "Model parallelism within node avoids network" means in practice) is left entirely to the table rows.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Tbl-scaling-decision provides quantitative guidance for scaling decisions across different model and data scales.
  + @Tbl-scaling-decision translates these limits into a practical lookup: models below one billion parameters fit on a single GPU with the optimizations above, models in the 1-10B range fit on a single multi-GPU node (keeping high-speed intra-node interconnect rather than the slower inter-node fabric), and only models above 10B or datasets above 10 TB require multi-node distributed complexity.
  ```
