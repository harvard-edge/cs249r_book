# Verified findings — responsible_engr.qmd (vol1)
Prior findings: 1 | Survived: 0 | Refuted: 1

## SURVIVING findings

*(none)*

## REFUTED findings

- `tbl-model-efficiency-comparison` — REFUTED: explanation in caption (L1677) and payoff ¶ (L1688).

  The first pass claimed the caption was absent; the scanner finds it at L1677: "**Model Efficiency Comparison**: Model selection must account for deployment constraints. Larger models provide better accuracy but require more power and time. The smallest model that meets accuracy requirements minimizes both cost and environmental impact." That caption states the principle the table demonstrates. The ref sentence at L1668 names the optimization techniques (quantization, pruning, efficient architectures) and connects them to environmental and financial impact. The payoff at L1688 closes the loop with specific numbers: the TinyML model leaves a measurable power margin on the wearable budget while MobileNetV2 exceeds it by a stated multiple. Taken together, caption + ref sentence + payoff ¶ tell the reader what the table shows (parameter count, power, latency, and fit for smartphone and IoT contexts across four models) and why it matters (only the smallest architecture fits both constrained contexts, making architecture selection a prerequisite for responsible edge deployment). The neighborhood clears the refutation bar on multiple elements.
