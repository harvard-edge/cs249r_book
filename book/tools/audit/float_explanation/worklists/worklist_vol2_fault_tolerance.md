# Float-explanation worklist — fault_tolerance.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 33 | 32 | 1 | 0 |
| table | 19 | 19 | 0 | 0 |
| listing | 4 | 4 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 11 | 11 | 0 | 0 |
| **total** | **67** | **66** | **1** | **0** |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

### ⚠️ `fig-intermittent-fault-dram` — def L1149  (Thin)

- **Caption:** **DRAM Residue Fault**: Intermittent failures in DRAM chips commonly arise from microscopic residue accumulation and create unreliable electrical connections. Physical defects can induce sporadic errors and highlight the need for fault-tolerant system design and hardware testing. Source: Constantinescu.
- **Ref(s):** L1147 `@Fig-intermittent-fault-dram`: "@Fig-intermittent-fault-dram reveals how residue-induced intermittent faults in DRAM chips create unreliable electrical connections that lead to sporadic failures."
- **Context checked:** ref ✗ (bare announcer restating the caption) · prev ¶ ✗ (float def boundary only) · next ¶ ✓ (L1155 explains ML implications of intermittent faults in general, but not what this figure specifically adds beyond the preceding `fig-intermittent-fault`) · caption ✗ (restates mechanism already shown; no ML takeaway) · payoff ✓ (L1155 general intermittent-fault advice)
- **Issue:** This is the second of two consecutive figures on the same solder/packaging physical mechanism (`fig-intermittent-fault` shows a solder-crack cross-section; this one shows DRAM residue contamination). The ref sentence and caption together give no reason why the reader needs both images or what DRAM-specific point this adds to the chapter's argument. The ML implication at L1155 applies to the category, not to the DRAM detail.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Fig-intermittent-fault-dram reveals how residue-induced intermittent faults in DRAM
  -  chips create unreliable electrical connections that lead to sporadic failures.
  + DRAM is especially vulnerable to this class of fault: residue contamination between
  +  memory-cell contacts (@fig-intermittent-fault-dram) creates a load-dependent
  +  resistance path that passes manufacturing test yet fails under the sustained bandwidth
  +  demands of a training run, making DRAM intermittent faults harder to screen out than
  +  solder-crack defects that surface under thermal cycling.
  ```
