# Verified findings — ops_scale.qmd (vol2)
Prior findings: 2 | Survived: 1 | Refuted: 1

---

## SURVIVING findings

### ⚠️ `fig-tco-iceberg` — def L3635
- **Ref:** "As @fig-tco-iceberg illustrates, while GPU compute and storage are the visible costs, hidden operational costs often constitute fully half of the actual budget."
- **Why it survives:** The ref sentence mirrors the caption word-for-word; no neighborhood element adds explanation. The caption (L3635) restates the same claim. The preceding content (L3631) is the equation definition with no iceberg commentary. The post-figure prose (L3737) explains only the equation symbols ($C_\text{train}$, $C_\text{infer}$, etc.), not the figure's visual insight. The payoff paragraph (L3739) addresses how the dominant cost component shifts with organizational maturity — a different claim about TCO dynamics, not about the iceberg framing itself. Nowhere does any prose element explain which specific hidden categories surprise organizations, why the visible/hidden split is not obvious from the equation alone, or what operational action the two-zone breakdown implies.
- **Suggested rewrite (flag-only):**
  ```diff
  - As @fig-tco-iceberg illustrates, while GPU compute and storage are the visible costs, hidden operational costs often constitute fully half of the actual budget.
  + The distribution in @fig-tco-iceberg explains why cost-reduction efforts aimed only at GPU spend routinely disappoint: the waterline separates the two visible infrastructure categories (GPU compute at 40 percent, object storage at 10 percent) from six operational categories — engineering labor, data pipeline maintenance, retraining compute, monitoring, incident response, and compliance — that collectively match them. A team that halves GPU spend leaves the larger half of its budget untouched and gains no relief on the operational side.
  ```

---

## REFUTED findings

- `lst-anomaly-attribution` — REFUTED: explanation across preceding bullets (L3073–3075) and listing inline comments (L3097–3106). The bullets name the three efficiencies the detector enables (attribution, deduplication, prioritization). The listing's own inline comment at the threshold check reads "# Many models affected -> likely shared cause," making the 60 percent threshold criterion and its routing implication visible within the float itself. The caption (L3079) names the detection-and-attribution goal. Together these elements tell the reader what the listing does and why the threshold mechanism is the operative design choice. Under the default-REFUTED standard, the combination of bullets + code comment + caption is sufficient to clear the bar.
