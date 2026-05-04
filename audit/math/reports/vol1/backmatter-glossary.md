# Math Audit: vol1/backmatter/glossary/glossary.qmd

Source audited: `book/quarto/contents/vol1/backmatter/glossary/glossary.qmd`

## Summary

The glossary is mostly prose definitions, but it includes several numeric claims, units, metric definitions, and one asymptotic-complexity statement. I found three substantive math/numeric/prose-equation consistency issues: the CSR storage bound drops the row-pointer term, tail latency is described as "worst-case" while using percentile examples, and FLOPS is defined as operations on "decimal numbers" rather than floating-point values.

No source `.qmd` files were modified.

## Findings

### 1. CSR storage complexity omits the row-pointer term

- Line 214: compressed sparse row storage is described as "reducing memory from `$O(N^2)$` (for an `$N \times N$` matrix) to `$O(K)$` where `$K$` is the number of nonzeros."
- Issue: Standard CSR storage needs values for the nonzeros, column indices for the nonzeros, and row pointers for each row. For an `$N \times N$` matrix, that is `$O(K + N)$`, not just `$O(K)$`. The stated `$O(K)$` bound is only valid under an additional assumption such as `$K = \Omega(N)$`, and that assumption is not stated.
- Suggested correction: Change the reduction to `$O(K + N)$`, where `$K$` is the number of nonzeros, or explicitly state the dense-enough assumption if `$O(K)$` is intended.

### 2. Tail latency conflates percentile latency with worst-case latency

- Line 1220: tail latency is defined as "Worst-case response times in a system, typically measured as 95th or 99th percentile latency."
- Issue: The 95th and 99th percentiles are high-tail quantiles, not worst-case response times. Worst case means the maximum or an upper bound, while p95/p99 intentionally exclude the slowest 5% or 1% of observations. This is a statistical/prose consistency error.
- Suggested correction: Define tail latency as high-percentile response time, for example "High-percentile response times in a system, typically measured as 95th or 99th percentile latency."

### 3. FLOPS definition incorrectly says floating-point operations involve decimal numbers

- Line 505: FLOPS is defined as "Floating Point Operations Per Second" and says it quantifies operations "involving decimal numbers."
- Issue: Floating-point values are not necessarily decimal numbers; in most ML hardware they are binary floating-point formats such as FP32, FP16, BF16, or FP8. The important distinction is approximate real-valued floating-point arithmetic versus integer/fixed-point arithmetic, not decimal representation.
- Suggested correction: Replace "decimal numbers" with "floating-point values" or "real-valued floating-point numbers."

## Checked But No Issue

- Line 48: AlexNet's ImageNet error-rate reduction from 26 percent to sixteen percent is a rounded historical claim and is internally consistent as a reduction in error rate.
- Line 69: arithmetic intensity as FLOPs/byte is dimensionally coherent and matches roofline usage.
- Lines 140, 517, and 520: the FP/BF precision descriptions are numerically coherent: BF16 is 16-bit with FP32-like dynamic range, FP16 is 16-bit, and FP32 is 32-bit.
- Lines 146 and 1253: binary quantization values `-1` and `+1`, and ternary values `-1`, `0`, and `+1`, are consistent with the definitions.
- Lines 178 and 1031: the CAP and query-key-value entries correctly refer to three components.
- Line 184: the Cerebras Wafer-Scale Engine numbers are numeric factual claims, but they do not create an internal math/unit inconsistency in this glossary entry.
- Line 523: converting 32-bit values to 8-bit values gives a raw storage reduction of `32/8 = 4`, so the roughly `4x` memory reduction is mathematically consistent.
- Lines 576, 840, 990, 1028, 1069, and 1268: the Green500, MFU, PUE, QPS, ridge point, and TOPS metric definitions are dimensionally coherent.
- Line 637: information-compute ratio as model-performance gain per FLOP is a coherent ratio/efficiency metric, assuming the performance-gain metric is defined elsewhere.
- Line 643: INT8 replacing 32-bit floating-point values gives a raw storage reduction of `32/8 = 4`, matching the stated roughly `4x` reduction.
- Line 655: the iron-law equation is dimensionally consistent: data volume divided by bandwidth gives time, operation count divided by effective FLOP/s gives time, and the latency term is added as time.
- Line 671: k-anonymity's "at least k-1 other records" condition is mathematically correct.
- Lines 1051, 1122, 1146, and 1226: the ReLU, sigmoid, softmax, and tanh prose-equation definitions are mathematically consistent.
- Lines 1104 and 1158: sensitivity as true positive rate and specificity as true negative rate are correct.
- Line 1164: 200,000 km/s in fiber and sub-10 ms latency are mutually plausible; a 10 ms one-way propagation budget corresponds to about 2,000 km in fiber before routing and processing overheads.
- Lines 1235 and 1337: tensor dimensional examples and the typical NVIDIA warp size of thirty-two threads are internally consistent.
