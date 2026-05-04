# Math Audit: `book/quarto/contents/vol1/data_engineering/data_engineering.qmd`

Audit method: direct reasoning over the source file only; no Gemini or external validation. Scope included equations, numeric examples, unit conversions, complexity/scaling claims, and prose-equation consistency.

## Findings

### 1. K-S critical value uses the one-sample formula for a two-sample comparison

- **Lines:** 1921, 1967-1982
- **Severity:** High
- **Issue:** The prose describes comparing training distribution `P` against serving distribution `Q`, i.e. a two-sample K-S test. The formula used is `D_crit ≈ 1.36 / sqrt(n)`, which is the one-sample form. For two samples of equal size `n = 1000`, the usual critical value is `1.36 * sqrt((n + n)/(n*n)) = 1.36 * sqrt(2/1000) ≈ 0.061`, not `0.043`.
- **Proposed correction:** Either state this is a one-sample K-S test against a fixed reference CDF, or change the two-sample formula to:
  `D_crit ≈ 1.36 * sqrt((n_P + n_Q)/(n_P n_Q))`; for `n_P = n_Q = 1000`, use `0.061`.

### 2. PSI is described as exactly measuring the degradation-equation divergence

- **Lines:** 2070, 2112, 3824
- **Severity:** Medium
- **Issue:** The text says the divergence term `D(P_t || P_0)` is "exactly what PSI and KL divergence measure." KL can match a chosen `D_KL` divergence, but PSI is not exactly KL; for binned distributions it is commonly `sum (actual - expected) * ln(actual/expected)`, closer to a symmetric/Jeffreys-style divergence over bins. It is related to distribution divergence, not identical to `D(P_t || P_0)`.
- **Proposed correction:** Replace "exactly what PSI and KL divergence measure" with "what metrics such as binned PSI and KL divergence approximate or operationalize."

### 3. KWS budget example extrapolates from a table range that excludes its computed sample count

- **Lines:** 1012, 1055-1060, 1096-1106
- **Severity:** Medium
- **Issue:** The calculation gives `90K / 0.12 = 750K` labeled examples. The text then says this falls between 1M and 10M examples and suggests a `+5-6 percent` accuracy contribution from the `1M vs. 10M` row. But `750K` is below 1M, so it does not fall within that range.
- **Proposed correction:** Say "This is just below the 1M point in the design-space table" and avoid assigning the `+5-6 percent` gain unless another scaling assumption is introduced.

### 4. KWS projected spend is not derived from the preceding budget arithmetic

- **Lines:** 1090-1097, 1120-1125
- **Severity:** Medium
- **Issue:** The example allocates the full `$150K` budget as `$90K` labeling, `$37.5K` storage/processing, and `$22.5K` governance. The final configuration then adds `2M synthetic` examples and reports `$145K spend`, but no calculation shows how synthetic data cost is included or how total spend falls from the allocated `$150K` to `$145K`.
- **Proposed correction:** Add a small cost breakdown for `750K real + 2M synthetic`, or change the projected outcome to a qualitative "within the `$150K` budget" claim.

### 5. KWS design-space table mixes units under the wrong impact columns

- **Lines:** 1008-1015
- **Severity:** Low
- **Issue:** Several entries are dimensionally misplaced. For example, `16 kHz vs. 8 kHz sampling` lists `2x storage` under **Latency Impact** and `2x processing` under **Cost Impact**. Storage is a cost/capacity impact; processing may affect latency and compute cost.
- **Proposed correction:** Rename columns to broader terms such as **Processing/Latency Impact** and **Storage/Cost Impact**, or move the row entries to the columns matching their units.

### 6. False-positive callout labels specificity as precision

- **Lines:** 987-988
- **Severity:** Low
- **Issue:** `1 - FPR` is the true negative rate/specificity, not precision. Precision is `TP / (TP + FP)` and depends on base rate and true positives, neither of which is included in this calculation.
- **Proposed correction:** Replace "**Precision Requirement**" with "**Specificity / Rejection Requirement**."

### 7. Coordination-tax distributed mean example overstates local compute speed and speedup

- **Lines:** 2755-2769
- **Severity:** Medium
- **Issue:** With `1 TB` across `100` nodes, each node processes about `10 GB`. At the stated RAM bandwidth range elsewhere in the chapter (`50-200 GB/s`, lines 1893 and 2749), local processing is about `0.05-0.2 s`, not `0.01 s`. The resulting speedup over `~101 s` is roughly `500-2,000x`, not `10,000x`.
- **Proposed correction:** Use `~0.1 s` local compute and report an order-of-magnitude speedup such as `~1,000x`, or explicitly assume a `1 TB/s` in-memory/vectorized processing rate per node.

### 8. ETL/ELT break-even conclusion is unsupported by the provided arithmetic

- **Lines:** 2573-2581, 2607-2617
- **Severity:** Medium
- **Issue:** The calculation computes storage savings but does not include monthly ETL Spark compute in the final comparison or assign a dollar value to engineering time. The statement "if feature definitions change weekly, ELT's flexibility pays for itself" and the "fewer than once per month" break-even point are therefore not derived from the numbers shown.
- **Proposed correction:** Add an engineering-hour cost variable and include monthly ETL compute, then compute break-even changes/month explicitly. Otherwise remove the break-even claim.

### 9. CAP theorem prose says Kinesis balances all three properties

- **Lines:** 2621-2625
- **Severity:** Medium
- **Issue:** The paragraph correctly states that distributed systems cannot simultaneously guarantee consistency, availability, and partition tolerance, then says Kinesis "balances all three properties." That phrasing conflicts with CAP under partition; systems can tune trade-offs in normal operation but cannot guarantee all three during a partition.
- **Proposed correction:** Change to "Kinesis exposes configuration choices that trade among these properties" or "balances operational trade-offs, but still cannot guarantee all three under partition."

### 10. Activation checkpointing description incorrectly claims increased storage I/O

- **Lines:** 3145
- **Severity:** Medium
- **Issue:** Activation/gradient checkpointing trades extra computation for reduced activation memory by discarding selected activations and recomputing them during backpropagation. It does not generally increase disk storage I/O by writing intermediate values to disk; the point is usually to avoid storing them.
- **Proposed correction:** Replace the storage-I/O sentence with: "This reduces GPU memory requirements but increases compute time because discarded activations must be recomputed during backpropagation."

### 11. ResNet-50 eight-GPU throughput example has inconsistent bandwidth arithmetic

- **Lines:** 3384
- **Severity:** Medium
- **Issue:** `40,000 images/s` at `500 MB/s` implies only `12.5 KB/image`. Earlier the chapter uses `150 KB/image` for compressed ImageNet-style JPEGs (lines 3186 and 3227-3229), which would require about `6 GB/s`; decoded FP32 `224x224x3` images would require about `24 GB/s`.
- **Proposed correction:** Change `500 MB/s` to `~6 GB/s` if using `150 KB` compressed images, or change the image rate/bytes-per-image assumption so the numbers align.

### 12. Serving random-read requirement confuses latency budget with read rate

- **Lines:** 3386
- **Severity:** Medium
- **Issue:** A service handling `10,000 requests/s` with a `10 ms` latency budget does not by itself require `100,000 random reads/s`. With one read per request, it requires `10,000 reads/s` and about `100` concurrent in-flight reads by Little's law. `100,000 reads/s` would require an additional assumption such as ten random feature lookups per request.
- **Proposed correction:** State the missing assumption: "If each request performs ten independent feature reads, this requires 100,000 random reads/s." Otherwise use `10,000 random reads/s`.

### 13. Cohen's kappa footnote undercounts chance agreement in the binary example

- **Lines:** 2092
- **Severity:** Low
- **Issue:** If two annotators both label `90 percent` of images as "not spam" and `10 percent` as "spam", chance agreement is `0.9^2 + 0.1^2 = 0.82`, not just `0.81`. The `0.81` term is only the chance agreement on the majority class.
- **Proposed correction:** Replace `81 percent` with `82 percent`, or clarify that `81 percent` is agreement on "not spam" alone.

### 14. Energy table relative-cost baseline is inconsistent with the row units

- **Lines:** 157-162, 193-195, 363-368
- **Severity:** Low
- **Issue:** The table's baseline row is a "32-bit Floating Point MAC" with energy in pJ, but the local SSD and network relative costs are computed by dividing per-bit movement energies by `energy_flop_fp32_pj / 32`. That makes the relative costs "per bit of FP32 compute energy," not relative to the displayed 32-bit MAC row.
- **Proposed correction:** Either compare all movement rows directly to the 32-bit MAC energy, or relabel the baseline/relative-cost column to make clear the ratio is normalized per bit.

## Checked Without Findings

- Data gravity unit conversions are internally consistent: `1 PB = 1,000,000 GB`; `100 Gbps = 12.5 GB/s`; transfer time is about `80,000 s`, or `22 h`. The separate `10 Gbps` preview gives about `9 days`.
- False-positive arithmetic is internally consistent for one-second windows: about `2.6M` windows/month, FPR about `3.9e-7`, and rejection about `99.99996 percent`.
- Storage loading speedup for `736 GB` at `5 GB/s` vs. `0.1 GB/s` correctly gives a `50x` ratio.
- Format-efficiency example for `20` of `100` columns correctly gives `80 percent` wasted row-format bandwidth and `5x` effective throughput improvement before compression.
- Data debt compounding equation `Debt_n ≈ Debt_0(1+r)^n` is mathematically consistent with the prose claim of superlinear growth.
