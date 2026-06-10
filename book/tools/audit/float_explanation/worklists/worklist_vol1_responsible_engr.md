# Float-explanation worklist — responsible_engr.qmd (vol1)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 6 | 6 | 0 | 0 |
| table | 16 | 15 | 1 | 0 |
| listing | 1 | 1 | 0 | 0 |
| algorithm | 0 | — | — | — |
| equation | 2 | 2 | 0 | 0 |
| **total** | **25** | **24** | **1** | **0** |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

### ⚠️ `tbl-model-efficiency-comparison` — def L1677  (Thin)

- **Caption:** (none found — the table has no inline caption line; only a colon-label caption buried in the tbl- definition)
- **Ref(s):** L1668 `@tbl-model-efficiency-comparison`: "The benchmarks in @tbl-model-efficiency-comparison provide actionable guidance for efficiency optimization. Techniques that enable deployment on power-constrained platforms (quantization, pruning, and efficient architectures) directly reduce environmental impact per inference regardless of deployment context. Power savings at inference time translate directly to financial savings when aggregated across millions of requests."
- **Context checked:** ref ✓ (names optimization techniques) · prev ¶ ✗ (ends with a tbl-edge-deployment-constraints caption line, no prose setup) · next ¶ is the table rows themselves · payoff ¶ (L1688) ✗ (pivots to wearable power margin numbers that reference `tbl-edge-deployment-constraints`, not this table) · caption ✗ (absent)
- **Gap:** The ref sentence explains what a reader *should do* with the table (optimize) but never tells the reader what the table *shows*: a side-by-side comparison of MobileNetV2, EfficientNet-B0, ResNet-50, and a TinyML model across parameter count, inference power, latency, and fit for smartphone and IoT contexts. The structure and the takeaway (TinyML is the only model that fits both contexts; ResNet-50 fits neither) are left entirely implicit.
- **Suggested rewrite (flag-only):**
  ```diff
  - The benchmarks in @tbl-model-efficiency-comparison provide actionable guidance for efficiency optimization. Techniques that enable deployment on power-constrained platforms (quantization, pruning, and efficient architectures) directly reduce environmental impact per inference regardless of deployment context. Power savings at inference time translate directly to financial savings when aggregated across millions of requests.
  + @tbl-model-efficiency-comparison compares MobileNetV2, EfficientNet-B0, ResNet-50, and a TinyML model across parameter count, inference power, latency, and whether each fits the smartphone and IoT power envelopes defined above. The pattern is unambiguous: only the TinyML model fits both constrained contexts, while ResNet-50 fits neither. This comparison makes the efficiency argument concrete — selecting a smaller architecture is not a quality concession but a prerequisite for responsible deployment at the edge, where quantization and pruning are the tools that move models leftward across the fit boundary.
  ```
