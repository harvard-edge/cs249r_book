# Math Audit: `book/quarto/contents/vol1/backmatter/appendix_data.qmd`

Audit method: direct reasoning over the source file only; no Gemini or external validation. Scope included equations, numeric examples, unit conversions, complexity/scaling claims, probability/statistics examples, and prose-equation consistency. No source `.qmd` files were modified.

## Findings

### 1. Serialization table mixes cycles-per-value and cycles-per-byte units

- **Lines:** 130-131, 142-145, 166-170
- **Severity:** Medium
- **Issue:** The code comment says the table shows "cycles-per-value," but the rendered table header says "CPU Cycles per Byte." Those are different units. If the values are interpreted literally as cycles/byte, the throughput column is not dimensionally consistent: for example, `100 cycles/byte` at a 3 GHz CPU implies about `30 MB/s`, not `~100 MB/s`, and `10 cycles/byte` implies about `300 MB/s`, not `> 1,000 MB/s`, absent parallelism or vector-width assumptions. Protobuf is also listed as `200 cycles` while having higher throughput than CSV, which is hard to reconcile under a single cycles/byte interpretation.
- **Proposed correction:** Rename the column to **Representative Decode Cost** or **CPU Cycles per Value**, or provide an explicit CPU frequency, parallelism/vectorization assumption, and byte/value size that connects cycles to MB/s.

### 2. Indexed selection cost omits output-size dependence

- **Lines:** 333-334
- **Severity:** Low
- **Issue:** The selection primitive says filtering rows is `O(log N)` if indexed. An index can make locating a point or range boundary `O(log N)`, but returning matching rows costs at least `O(k)` for `k` output rows, and range scans are usually `O(log N + k)`. The current statement can imply that filtering any number of rows remains logarithmic.
- **Proposed correction:** Change the indexed cost to `O(log N + k)` where `k` is the number of matching rows, or clarify that `O(log N)` is only the lookup/boundary-search cost.

### 3. P99 request latency is treated as a user count without a session model

- **Lines:** 361-365, 373-391
- **Severity:** Medium
- **Issue:** The calculation `1% of 1M users = 10,000 unhappy people` treats a request-level P99 as if it directly means one percent of users are slow. P99 latency usually describes requests or operations, not users. If each user makes one request, the arithmetic is fine; if each user makes many requests, the user-level probability is `1 - 0.99^N`, which is much larger. The next example correctly uses this session-level complement, so the earlier prose is inconsistent with the later model.
- **Proposed correction:** Say "with one request per user, this is 10,000 slow requests/users" or replace the user count with the session probability formulation used below.

### 4. Tail-latency equation appends a percent sign to a percent-formatted value

- **Lines:** 423-428
- **Severity:** Low
- **Issue:** `fmt_percent(p_slow, ...)` returns the numeric percentage string, so `p_slow_pct_str` is `63.4` for `1 - 0.99^100`. The equation then appends `\\%`, while the prose on line 434 appends the word "percent." This is internally workable only because `p_slow_pct_str` has no percent symbol. The variable name and two rendering contexts make the expression easy to misuse and can produce duplicate percent signs if the formatter behavior changes.
- **Proposed correction:** Rename the variable to `p_slow_pct_num_str`, or define both `p_slow_pct_num_str = "63.4"` and `p_slow_pct_label = "63.4%"` for math vs. prose contexts.

### 5. KL divergence prose says "information is lost" where the equation measures extra coding cost

- **Lines:** 471-473, 605
- **Severity:** Low
- **Issue:** The equation for `D_KL(P || Q)` is correct, and the footnote correctly describes expected extra coding cost. The prose "measures how much information is lost if we approximate P with Q" is less precise and can be misleading: KL divergence is not a bounded lost-information amount, but an expected log-likelihood/code-length penalty under `P` when using `Q`.
- **Proposed correction:** Replace with "KL divergence measures the expected extra code length, or log-loss penalty, incurred when samples from `P` are modeled using `Q`."

### 6. Logit overflow claim is too broad about floating-point range

- **Lines:** 627-629, 741
- **Severity:** Low
- **Issue:** Line 629 says `e^{100}` overflows "floating-point range" generally, but line 741 correctly states the example is representable in FP64 and overflows FP32. The broad statement is therefore inconsistent with the worked example.
- **Proposed correction:** Change line 629 to "overflows common low-precision or FP32 floating-point ranges" or "can overflow the floating-point range used for training/inference."

### 7. "All intermediate values are in [0, 1]" overstates the log-sum-exp example

- **Lines:** 669-671, 705-707, 749-753
- **Severity:** Low
- **Issue:** In the stable calculation, the exponentials and final softmax probabilities are in `[0, 1]`, but not all intermediate values are. The stable sum is about `1.503`, and `LogSumExp` is about `102.408`. The shifted logits include negative values. So the final sentence is mathematically false as written.
- **Proposed correction:** Replace with "All exponentials passed into the softmax denominator are in `[0, 1]`, and the final probabilities are in `[0, 1]`."

### 8. Summary calls PSI a symmetric derivative of KL

- **Lines:** 595-603, 797
- **Severity:** Low
- **Issue:** The worked PSI formula `sum (P_i - Q_i) log(P_i / Q_i)` is symmetric for positive binned distributions and equals `D_KL(P || Q) + D_KL(Q || P)` for the same bins. Calling PSI a "symmetric derivative" of KL is imprecise; it is better described as a symmetric KL-style divergence or Jeffreys-divergence-like binned metric.
- **Proposed correction:** Replace "KL divergence and its symmetric derivative PSI" with "KL divergence and symmetric KL-style metrics such as PSI."

## Checked Without Findings

- **Lines 59-95, 110-116:** Data-gravity transfer-time arithmetic is internally consistent under decimal-style TB/PB and bit-to-byte conversion: time scales linearly with data volume and inversely with bandwidth.
- **Lines 304-320, 336-341:** Projection waste is correctly computed as `1 - 4/1024 = 99.6%`, and the shuffle-join example's `~2 TB` network traffic for two `1 TB` tables is a reasonable approximation when both tables must be repartitioned.
- **Lines 410-428, 431-435:** The session-level P99 calculation is correct: `1 - 0.99^100 = 0.633967...`, displayed as about `63.4%`.
- **Lines 496-578, 587-599:** The KL/PSI numeric results are internally consistent for `P = [0.60, 0.30, 0.10]` and `Q = [0.45, 0.40, 0.15]`: `D_KL(P||Q) ≈ 0.046`, `D_KL(Q||P) ≈ 0.046`, and `PSI ≈ 0.092`.
- **Lines 656-730, 739-751:** The log-sum-exp arithmetic is correct: shifting `[100, 101, 102]` by `102` gives exponentials `[0.135, 0.368, 1.0]`, sum `1.503`, `logsumexp ≈ 102.408`, and softmax approximately `[0.090, 0.245, 0.665]`.
- **Lines 759-761:** The checkpoint's implied answers are mathematically coherent: `500 GB` over `10 Gbps` is about `400 s` or `6.7 min`, and fifty independent P99 opportunities gives `1 - 0.99^50 ≈ 39.5%`.
