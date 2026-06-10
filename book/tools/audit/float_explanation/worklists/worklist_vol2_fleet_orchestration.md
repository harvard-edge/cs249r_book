# Float-explanation worklist — fleet_orchestration.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 6 | 6 | 0 | 0 |
| table | 7 | 6 | 1 | 0 |
| listing | 2 | 2 | 0 | 0 |
| algorithm | 1 | 1 | 0 | 0 |
| equation | 5 | 5 | 0 | 0 |
| **total** | **21** | **20** | **1** | **0** |

> Note: `fig-fleet-stack` appears at L80 and L2428 as cross-chapter references. The float is defined in `vol2/introduction/introduction.qmd`, not in this chapter, so it is out of scope for this audit.

---

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

### ⚠️ `tbl-fleet-orchestration-slurm-partitions` — def L614  (Thin)
- **Caption:** "**Slurm Partition Configuration**: Partitions organize heterogeneous accelerators into logical pools matched to workload characteristics. NVLink-connected partitions support tensor parallelism, while PCIe partitions serve workloads that rely primarily on data parallelism. Separating inference and debug partitions prevents experimental workloads from impacting production serving."
- **Ref(s):** L616 `@tbl-fleet-orchestration-slurm-partitions`: "As @tbl-fleet-orchestration-slurm-partitions shows, GPU allocation strategies significantly impact utilization, and Slurm provides several mechanisms for controlling GPU placement."
- **Context checked:** ref ✗ (misrepresents table content) · prev ¶ partial (L605 intro sentence sets up table purpose, L601-603 explain Slurm imperatives) · caption ✓ (explains partition pools and NVLink/PCIe distinction) · next ¶ (pivots immediately to `--gres` flags, unrelated to table rows) · payoff ✗
- **Issue:** The reference sentence claims the table shows that "GPU allocation strategies significantly impact utilization," but the table actually shows four partition types with their GPU hardware, interconnects, and typical uses. The mismatch between what the ref claims the table demonstrates and what it actually contains leaves the reader without a clear takeaway from the table itself. The caption and the L605 intro sentence do explain the table, but the reference sentence points the reader toward a conclusion the table does not support.
- **Suggested rewrite (flag-only):**
  ```diff
  - As @tbl-fleet-orchestration-slurm-partitions shows, GPU allocation strategies significantly impact utilization, and Slurm provides several mechanisms for controlling GPU placement.
  + @Tbl-fleet-orchestration-slurm-partitions shows why partition design matters: the dgx-a100 partition's NVLink fabric supports tensor parallelism, while the PCIe a100 and Ethernet-only partitions serve workloads that can tolerate lower interconnect bandwidth. Partition selection therefore determines which parallelism strategies are available to a job before the first scheduling decision is made.
  ```
