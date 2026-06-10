# Float-explanation worklist — ops_scale.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 13 | 12 | 1 | 0 |
| table | 39 | 39 | 0 | 0 |
| listing | 10 | 9 | 1 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 20 | 20 | 0 | 0 |
| **total** | **82** | **80** | **2** | **0** |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

---

### ⚠️ `fig-tco-iceberg` — def L3635  (Thin)

- **Caption:** **The TCO Iceberg**: Total Cost of Ownership analysis for ML systems. While GPU compute and storage are the visible costs, the hidden operational costs---including engineering labor, maintenance, and compliance---often constitute fully half of the actual budget.
- **Ref(s):** L3633 `@fig-tco-iceberg`: "As @fig-tco-iceberg illustrates, while GPU compute and storage are the visible costs, hidden operational costs often constitute fully half of the actual budget."
- **Context checked:** ref ✗ (float-announcer, restates caption) · prev ¶ is the equation definition (L3631) · next ¶ (L3737) explains component symbols and maturity shift, not the figure insight · caption ✗ (same claim restated) · payoff ✗ (L3739 elaborates TCO equation, not figure)
- **Issue:** The reference sentence is a float-announcer that mirrors the caption word-for-word. It tells the reader the figure "illustrates" a fact but adds no explanation of why the iceberg framing is operationally consequential, which specific hidden categories surprise organizations, or what action the reader should take upon seeing that hidden costs dominate visible ones.
- **Suggested rewrite (flag-only):**
  ```diff
  - As @fig-tco-iceberg illustrates, while GPU compute and storage are the visible costs, hidden operational costs often constitute fully half of the actual budget.
  + The distribution in @fig-tco-iceberg explains why cost-cutting efforts aimed only at GPU spend routinely disappoint: the waterline separates the two visible infrastructure categories (GPU compute and storage, roughly 50 percent) from six operational categories — labor, pipeline maintenance, retraining, monitoring, incident response, and compliance — that collectively match or exceed them. A team that halves GPU spend leaves the larger half of its budget untouched.
  ```

---

### ⚠️ `lst-anomaly-attribution` — def L3079  (Thin)

- **Caption:** **Fleet Anomaly Attribution**: Detecting correlated anomalies across a model fleet and attributing them to shared infrastructure or data causes.
- **Ref(s):** L3077 `@Lst-anomaly-attribution`: "@Lst-anomaly-attribution shows a fleet-wide correlation detector that attributes simultaneous anomalies to shared causes."
- **Context checked:** ref ✗ (announcer, restates caption) · prev ¶ (L3073-3075) lists three system properties (attribution, deduplication, prioritization) ✓ partial setup · next ¶ (L3115) pivots immediately to drift detection ✗ · caption (restates) ✗ · payoff ✗ (no follow-through on the listing's design)
- **Issue:** The reference announces the listing without explaining the key design choice the code embodies: using a fractional threshold (>60 percent of models simultaneously anomalous) as the criterion for inferring a shared cause rather than a model-specific one. The preceding bullets name desirable properties but do not tie them to the threshold mechanism the listing implements. The payoff paragraph pivots to drift detection without any closing statement about when the correlation detector fires and what it implies for the on-call responder.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Lst-anomaly-attribution shows a fleet-wide correlation detector that attributes simultaneous anomalies to shared causes.
  + When more than 60 percent of models show simultaneous anomalies, the root cause is almost certainly shared infrastructure or data rather than any individual model. @Lst-anomaly-attribution implements that threshold rule: the detector scans each timestamp, counts the fraction of anomalous models, and calls `attribute_to_shared_cause` only when that fraction exceeds the correlation threshold, routing the alert to the platform team rather than to individual model owners.
  ```
