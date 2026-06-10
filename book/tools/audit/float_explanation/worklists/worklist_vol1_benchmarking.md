# Float-explanation worklist — benchmarking.qmd (vol1)

## Summary

| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 13 | 13 | 0 | 0 |
| table | 22 | 19 | 3 | 0 |
| listing | 0 | 0 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 3 | 3 | 0 | 0 |
| **total** | **38** | **35** | **3** | **0** |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

---

### ⚠️ `tbl-benchmarking-vendor-claims` — def L536  (Thin)

- **Caption:** "**Decoding Vendor Benchmark Claims**: Four common marketing phrases and the technical caveats behind each."
- **Ref(s):** L525 `@Tbl-benchmarking-vendor-claims`: "translates common marketing phrases into the technical caveats behind each."
- **Context checked:** ref ✗ (mechanical) · prev ¶ (checklist setup — explains what to look for but not the payoff) · next ¶ (L538, pivots to hardware infrastructure without stating what the pattern reveals) · caption (descriptive but only restates function) · payoff ✗
- **Issue:** The ref sentence announces the table's mechanics but never tells the reader the take-away: that vendor claims systematically misrepresent full-pipeline costs by scoping to peak, single-operation, or optimal-precision conditions. The payoff paragraph moves on without closing the loop. A reader could see the four rows and learn nothing about the systematic pattern.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Tbl-benchmarking-vendor-claims translates common marketing phrases into the technical caveats behind each.
  + @Tbl-benchmarking-vendor-claims pairs four common marketing phrases with the technical scope limitation each conceals. The pattern is consistent: every claim optimizes one variable (batch size, compute-only timing, per-operation efficiency) while holding the rest at ideal conditions, so the number is real but the context is missing.
  ```

---

### ⚠️ `tbl-edge-vs-cloud-constraints` — def L1793  (Thin)

- **Caption:** "**Edge vs. Cloud Deployment Constraints**: The same three constraints (power, latency, accuracy) carry fundamentally different meanings across deployment contexts. Cloud systems treat power as an operational cost and latency as a UX metric, leaving accuracy as the primary optimization target; edge systems must treat power and latency as hard physical limits, leaving accuracy as the residual variable to optimize."
- **Ref(s):** L1785 `@tbl-edge-vs-cloud-constraints`: "Edge deployment requires navigating trade-offs that cloud deployments can largely ignore, summarized in @tbl-edge-vs-cloud-constraints."
- **Context checked:** ref ✗ (bare pointer) · prev ¶ ✓ (names the constraint triangle) · next ¶ ✓ (smartphone example immediately quantifies) · caption ✓ (strong) · payoff ✓ (strong)
- **Issue:** The ref sentence is a bare "see this table" with no indication of what insight the table delivers. The caption and payoff paragraph are both strong, so the float is not a dead-end, but the ref sentence itself adds nothing. The reader who skips to the ref gets no advance signal of what to look for.
- **Suggested rewrite (flag-only):**
  ```diff
  - Edge deployment requires navigating trade-offs that cloud deployments can largely ignore, summarized in @tbl-edge-vs-cloud-constraints.
  + @Tbl-edge-vs-cloud-constraints makes the distinction concrete: the same three constraints (power, latency, accuracy) that cloud systems treat as tunable cost variables become hard physical limits at the edge, inverting the optimization hierarchy.
  ```

---

### ⚠️ `tbl-benchmarking-edgetpu-validation` — def L2929  (Thin)

- **Caption:** "**EdgeTPU vs. Cortex-M7 MobileNetV2 validation**: SingleStream-scenario measurements comparing inference latency, end-to-end latency, power, and energy per inference for INT8 MobileNetV2, showing how preprocessing overhead narrows the headline accelerator speedup."
- **Ref(s):** L2920 `@Tbl-benchmarking-edgetpu-validation`: "reports the validation protocol under the SingleStream scenario."
- **Context checked:** ref ✗ (purely mechanical) · prev ¶ (L2918, states the hardware claim numbers but not what to look for in the table) · next ¶ (L2931-2933, strong payoff explaining the preprocessing-narrowing finding) · caption ✓ (states the key finding) · payoff ✓
- **Issue:** The ref sentence "reports the validation protocol" gives the reader no signal about the finding they are about to encounter. The table's central insight (preprocessing overhead narrows the headline speedup) does appear in the caption and payoff, but the ref is a mechanical pointer. A reader who glances at the ref and moves on will miss the point.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Tbl-benchmarking-edgetpu-validation reports the validation protocol under the SingleStream scenario.
  + @Tbl-benchmarking-edgetpu-validation shows what that claim looks like under the SingleStream protocol: the headline inference speedup holds, but end-to-end improvement is considerably smaller because preprocessing runs on the host CPU in both cases.
  ```
