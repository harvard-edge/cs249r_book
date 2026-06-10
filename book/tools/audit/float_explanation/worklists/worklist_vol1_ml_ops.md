# Float-explanation worklist — ml_ops.qmd (vol1)

## Summary

| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 12 | 12 | 0 | 0 |
| table | 31 | 28 | 3 | 0 |
| listing | 6 | 6 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 16 | 16 | 0 | 0 |
| **total** | **65** | **62** | **3** | **0** |

---

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

### ⚠️ `tbl-monitoring-cost-components` — def L2589  (Thin)

- **Caption:** "**Monitoring Cost Components**: Costs scale differently across components. Metric ingestion scales with cardinality (number of unique metric series), while storage scales with retention. Query costs scale with dashboard usage patterns."
- **Ref(s):** L2580 `@Tbl-monitoring-cost-components`: "@Tbl-monitoring-cost-components provides typical unit costs for each component:"
- **Context checked:** ref ✗ (bare pointer) · prev ¶ ✓ (partial: establishes four-category breakdown via eq) · next ¶ ✓ (payoff mentions "clarifies the real expense") · caption ✓ (explains differential scaling) · payoff ✓ (links to budget estimation)
- **Issue:** The ref sentence is a pure announcement. The caption and payoff do carry the "why," but the ref sentence itself contributes nothing beyond pointing. A reader scanning would miss that the table's organizing insight is that monitoring cost is dominated by different drivers per component — not a flat per-metric fee — which is the actionable take-away.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Tbl-monitoring-cost-components provides typical unit costs for each component:
  + The four cost components scale on different axes. @Tbl-monitoring-cost-components shows that ingestion cost grows with metric cardinality, storage cost with retention period, and alert-evaluation cost with rule count, so the dominant line item shifts as monitoring scope expands.
  ```

---

### ⚠️ `tbl-ab-test-decisions` — def L1725  (Thin)

- **Caption:** "**A/B Test Decision Matrix**: Deployment decisions should consider both primary metrics and guardrails. Improvements that come at the cost of guardrail violations require careful trade-off analysis rather than automatic deployment."
- **Ref(s):** L1716 `@Tbl-ab-test-decisions`: "@Tbl-ab-test-decisions turns those constraints into a deployment decision:"
- **Context checked:** ref ✗ (bare pointer) · prev ¶ ✓ (describes guardrails, interference, segment heterogeneity as complications) · next ¶ ✓ (payoff explains preregistration discipline) · caption ✓ (states the guardrail-vs-primary tension) · payoff ✓
- **Issue:** "Turns those constraints into a deployment decision" names what the table IS (a decision matrix) without stating what the reader should take from it. The key insight — that a significant improvement combined with guardrail failures is not an automatic ship — is absent from the ref sentence and requires the reader to find it in the caption.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Tbl-ab-test-decisions turns those constraints into a deployment decision:
  + Those complications collapse into four quadrants. @Tbl-ab-test-decisions maps each combination of primary-metric outcome and guardrail state to a decision, with the critical row being a significant improvement that fails a guardrail — that result demands investigation, not deployment.
  ```

---

### ⚠️ `tbl-technical-debt-summary` — def L3156  (Thin)

- **Caption:** "**Technical Debt Patterns**: Machine learning systems accumulate distinct forms of technical debt from data dependencies, model interactions, and evolving operational contexts. Primary debt patterns, their causes, symptoms, and recommended mitigation strategies guide practitioners in recognizing and addressing these challenges systematically."
- **Ref(s):** L3143 `@Tbl-technical-debt-summary`: "@Tbl-technical-debt-summary consolidates the debt patterns discussed throughout this chapter, providing the reference that the assessment rubric below builds on."
- **Context checked:** ref ✗ (pure summary pointer) · prev ¶ ✗ (section header only) · next ¶ ✓ (payoff transitions to assessment rubric) · caption ✓ (describes four-column taxonomy) · payoff ✓ (links to rubric)
- **Issue:** The ref sentence describes administrative purpose ("consolidates," "assessment rubric below builds on") without giving the reader a reason to actually read the table at this point. No hint of what a practitioner should look for, or which debt patterns are hardest to detect, or what the Mitigation column enables.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Tbl-technical-debt-summary consolidates the debt patterns discussed throughout this chapter, providing the reference that the assessment rubric below builds on.
  + The eight patterns share a common structure. @Tbl-technical-debt-summary maps each pattern to its primary cause, observable symptoms, and mitigation strategy — a format that makes it usable as a diagnostic checklist: when a symptom appears in production, the corresponding cause and mitigation are one row away.
  ```
