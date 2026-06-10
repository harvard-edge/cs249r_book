# Float-explanation worklist — responsible_ai.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 14 | 13 | 1 | 0 |
| table | 7 | 7 | 0 | 0 |
| listing | 2 | 2 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 0 | 0 | 0 | 0 |
| **total** | **23** | **22** | **1** | **0** |

---

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

### ⚠️ `fig-monitoring-pipeline` — def L1883  (Thin)

- **Caption:** **Fairness Monitoring Pipeline**: End-to-end observability for deployed models. Model predictions feed subgroup metric computation across demographic segments; a threshold check identifies performance or fairness regressions; and alerts trigger automated retraining or manual review. This continuous feedback loop ensures that responsible AI properties are maintained postdeployment.
- **Ref(s):** L1881 `@fig-monitoring-pipeline`: "Implementing effective monitoring depends on robust infrastructure. Systems must log inputs, outputs, and contextual metadata in a structured and secure manner, feeding a continuous observability pipeline (@fig-monitoring-pipeline)."
- **Context checked:** ref ✗ (pure parenthetical) · prev ¶ ✓ (explains what monitoring involves — surfacing subgroup metrics, detecting distribution shifts, linking to retraining/rollback — but does not describe the pipeline's internal stages) · next ¶ ✗ (continues with telemetry details, does not describe figure mechanics) · caption ✓ (names the stages: subgroup metric computation → threshold check → alert → retrain) · payoff ✓ (L1887 discusses telemetry infrastructure)
- **Gap:** The figure shows a specific four-stage feedback loop (predictions → subgroup metrics → threshold check → alert/retrain), but no prose sentence describes those stages. The caption carries the explanation; the ref sentence is a bare parenthetical. A payoff sentence after the float would materially help readers connect the mechanics to the surrounding argument about monitoring infrastructure.
- **Suggested rewrite (flag-only):**
  ```diff
  - Implementing effective monitoring depends on robust infrastructure. Systems must log inputs, outputs, and contextual metadata in a structured and secure manner, feeding a continuous observability pipeline (@fig-monitoring-pipeline).
  + Implementing effective monitoring depends on robust infrastructure. Systems must log inputs, outputs, and contextual metadata in a structured and secure manner. @Fig-monitoring-pipeline shows how predictions feed subgroup metric computation across demographic segments, a threshold check flags performance or fairness regressions, and alerts trigger automated retraining or manual review, closing the observability loop postdeployment.
  ```

---

## Notes

- **Dangling ref at L70 (`@fig-fleet-stack`):** This figure is defined in another chapter (the fleet stack diagram). The prose self-explains the figure's content and the reference is contextually clear. Out of scope for this audit (not defined in this chapter).
- All seven tables are explained in-neighborhood, including the three captionless tables (`tbl-ml-principles-comparison`, `tbl-practitioner-decision-framework`, `tbl-autonomous-safety-case`), each of which has a substantive setup paragraph naming what each column and row represents.
- `fig-responsible-ai-architecture` (def L1061) has a large float body (TikZ block through L1349). The payoff paragraph lands at L1351, well after the float closes, but the ref sentence at L1059 describes the data flow explicitly enough to be ✅.
