# Verified findings — fault_tolerance.qmd (vol2)
Prior findings: 1 | Survived: 1 | Refuted: 0

## SURVIVING findings

### ⚠️ `fig-intermittent-fault-dram` — def L1149
- **Ref:** "@Fig-intermittent-fault-dram reveals how residue-induced intermittent faults in DRAM chips create unreliable electrical connections that lead to sporadic failures." (L1147)
- **Why it survives:** Every neighborhood element repeats the same claim without adding a DRAM-specific argument. The section intro at L1139 already lists "residue-induced electrical connections" as one instance of physical degradation. The preceding figure (fig-intermittent-fault, L1141) shows the solder-crack mechanism with its own caption. The ref sentence at L1147 is a pure announcer: it restates the mechanism already named in the section intro and mirrors the caption verbatim. The caption itself says only that residue causes unreliable connections and that this "highlights the need for fault-tolerant system design and hardware testing" — the same generic takeaway the section gives for every intermittent-fault type. The payoff at L1155 covers intermittent faults as a category and gives ML-specific advice (treat as suspect, use runtime monitoring, adaptive resource management) but makes no DRAM-specific point: the advice applies equally to solder-crack or any other intermittent mechanism. No neighborhood element answers why the reader needs a second figure showing DRAM residue after the first figure showed solder cracks, or what is distinct about DRAM intermittent faults for ML systems (e.g., load-dependent bandwidth failures that pass manufacturing test, different screening difficulty, different detection signal). This is a genuine dead-end where the second figure adds visual content not connected by any prose claim.
- **Suggested rewrite (no em-dash/hyphen, ≤1 colon/para):**
  ```diff
  - @Fig-intermittent-fault-dram reveals how residue-induced intermittent faults in DRAM
  -  chips create unreliable electrical connections that lead to sporadic failures.
  + DRAM is particularly susceptible to this failure class: residue contamination between
  +  memory-cell contacts (@fig-intermittent-fault-dram) creates a load-dependent resistance
  +  path that passes manufacturing test under light access patterns yet fails under the
  +  sustained bandwidth demands of a training run. Unlike the solder-crack mechanism
  +  shown above, DRAM residue faults are not exposed by thermal cycling alone, making
  +  them harder to screen before deployment and more likely to appear mid-job when
  +  gradient tensors stress memory bandwidth continuously.
  ```

## REFUTED findings

*(none)*
