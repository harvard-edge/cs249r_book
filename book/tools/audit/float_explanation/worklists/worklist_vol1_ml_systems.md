# Float-explanation worklist — ml_systems.qmd (vol1)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 13 | 13 | 0 | 0 |
| table | 14 | 14 | 0 | 0 |
| listing | 0 | 0 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 7 | 7 | 0 | 0 |
| **total** | **34** | **34** | **0** | **0** |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

No under-explained floats found; all references explained in-neighborhood.

## Notes (scanner artifacts, not findings)

- The scanner flagged `eq-latency-physics` (def L304) and `tbl-ml-systems-lighthouse-archetypes` (def L639) as orphans. Both are false positives: the reference token is immediately followed by a colon (`@eq-latency-physics:` at L303, `@tbl-ml-systems-lighthouse-archetypes:` at L629), which the scanner's def/ref matcher dropped into the "dangling refs" bucket. Both floats are genuinely referenced AND explained.
  - `eq-latency-physics`: ref L303 sets up the round-trip-time formula, L305 defines $c_{\text{fiber}}$, and the payoff at L349 (California-to-Virginia ~36 ms, sub-10 ms apps cannot use distant cloud) carries the takeaway. ✅
  - `tbl-ml-systems-lighthouse-archetypes`: ref L629 names the five lighthouse models and their purpose, the caption states each pairs an archetype with a deployment paradigm to isolate a bottleneck, and the payoff L643 frames the per-workload analysis that follows. ✅
- Three dangling refs have NO definition in this chapter and are therefore OUT OF SCOPE (cross-chapter references, not floats defined here): `@fig-ai-triad` (L59), `@tbl-dam-taxonomy` (L609, L922), `@Eq-degradation` (L4288). Confirmed via grep that none is defined in ml_systems.qmd. These belong to the auditor's "do not flag" set since the audit covers only floats DEFINED in this chapter.
