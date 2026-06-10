# Verified findings — ml_ops.qmd (vol1)
Prior findings: 3 | Survived: 0 | Refuted: 3

## SURVIVING findings

_(none)_

---

## REFUTED findings

- `tbl-monitoring-cost-components` — REFUTED: explanation is in caption (L2589) and payoff ¶ (L2591). Caption states the organizing insight directly: "Costs scale differently across components. Metric ingestion scales with cardinality (number of unique metric series), while storage scales with retention. Query costs scale with dashboard usage patterns." The payoff sentence ("Translating these unit costs into a concrete budget estimate clarifies the real expense of monitoring even a single production model.") connects the table to its downstream use. A reader scanning the caption gets the differential-scaling principle without the ref sentence.

- `tbl-ab-test-decisions` — REFUTED: explanation is in caption (L1725) and payoff ¶ (L1727). Caption states the critical take-away: "Improvements that come at the cost of guardrail violations require careful trade-off analysis rather than automatic deployment." The preceding paragraph (L1714) establishes that guardrails, segment analysis, and preregistered decisions are necessary — setting up exactly what the table resolves. The payoff adds the preregistration discipline. The ref sentence's "turns those constraints into a deployment decision" is a structural pointer, but the neighborhood fully carries the content.

- `tbl-technical-debt-summary` — REFUTED: explanation is in caption (L3156) and payoff ¶ (L3158). Caption describes the four-column taxonomy: "Primary debt patterns, their causes, symptoms, and recommended mitigation strategies guide practitioners in recognizing and addressing these challenges systematically." The payoff explicitly articulates the diagnostic value: "awareness alone is insufficient; teams need a systematic technical debt assessment rubric that transforms subjective 'is this system ready?' conversations into quantifiable evaluations." Together these tell the reader what the table contains and why to use it.
