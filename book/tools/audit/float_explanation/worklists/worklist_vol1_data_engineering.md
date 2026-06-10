# Float-explanation worklist — data_engineering.qmd (vol1)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 15 | 15 | 0 | 0 |
| table | 10 | 10 | 0 | 0 |
| listing | 4 | 4 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 8 | 8 | 0 | 0 |
| **total** | 37 | 37 | 0 | 0 |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

No under-explained floats found; all references explained in-neighborhood.

## Notes on apparent orphans (scanner false positives)

The scanner reported five floats as "orphan — NONE" and listed them under
"Dangling refs (no matching def)." All five are in fact referenced; the scanner's
ref-matcher simply did not match the `@ref` because the reference carries trailing
punctuation (a period or colon) immediately after the label. Each was verified
against the source and is explained in its neighborhood:

- `eq-data-supply` (def L4166) — referenced at L4164 (`@eq-data-supply.`); defined and
  explained alongside `eq-training-throughput` (storage bandwidth times one minus
  overhead). ✅
- `eq-debt-compound` (def L4503) — referenced at L4502 (`@eq-debt-compound:`); L4502 sets
  up every variable and the superlinear/compounding behavior, payoff L4504 gives the
  10–30 percent per-period rate, and a margin figure shows the divergence. ✅
- `fig-spectrogram-example` (def L3231) — referenced at L3225 (`@fig-spectrogram-example:`)
  and richly described in the L3223 prose (waveform → spectrogram → MFCC). ✅
- `fig-labels` (def L3450) — referenced at L3448 (`@fig-labels.`); the preceding label-type
  paragraphs and L3448 explain the cost/precision progression and when each is chosen. ✅
- `lst-etl-elt-cost-comparison` (def L3062) — referenced at L3060 (`@lst-etl-elt-cost-comparison.`);
  the surrounding worked cost model (L3052 "Systems insight", L3056–3058) carries it. ✅

Cross-chapter references with no in-chapter definition (out of scope for this audit):
`fig-ds-time`, `tbl-dam-taxonomy`, `eq-degradation`, `Tbl-serialization-cost`.
