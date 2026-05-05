# Math Audit Report: `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd`

## Checked scope

Audited the chapter for equations, generated numeric examples, unit conversions, complexity/scaling claims, and prose-equation consistency using direct reasoning only. No source `.qmd` files were modified.

## Findings

### High Severity

- **Lines 1718-1724 and 1782-1788: deployment-economics scenario does not match its arithmetic.**  
  The notebook problem says the production model processes **one million images per month** (line 1782), and the code comment says the model is for **500 clinics processing 1M images/month total** (lines 1718-1719). But the displayed annual cloud-cost calculation uses `500 clinics x 50 patients/day x 365 days = 9,125,000 images/year`, which is only about **760,417 images/month**, not 1,000,000/month. At USD 0.01/image, the displayed inference total is USD 91,250/year, whereas one million images/month would be USD 120,000/year before network cost.
  - Proposed correction: Either change the problem statement to "about 760,000 images per month" / "9.125 million images per year", or change the arithmetic to use 12,000,000 images/year. For consistency with the current `500 clinics` setup, another exact option is `500 clinics x 66.7 patients/day x 365 days ≈ 12.2 million images/year`.

- **Lines 1796-1801: edge annual cost and payback omit a cost term in prose.**  
  The code includes `edge_inf_cost = 0.001` USD/image and includes it in `edge_opex_year = edge_maint_cost + total_images_year * edge_inf_cost` (lines 1735 and 1748). For 9,125,000 images/year, edge inference is **USD 9,125/year**, so edge annual operating cost is **USD 34,125/year**, not just "negligible operational, ~USD 25,000 maintenance" (line 1797) or "USD 250,000 upfront + ~USD 25,000/year" (line 1798). The reported payback of about 2 years is actually based on the larger USD 34,125/year OpEx, so the prose understates recurring edge costs.
  - Proposed correction: Change the edge annual-cost bullets to include inference electricity explicitly, e.g. "Annual cost: ~USD 25,000 maintenance + ~USD 9,125 inference electricity = ~USD 34,125/year" for the current volume. If the volume is changed to 12M/year, use USD 12,000/year inference and USD 37,000/year total edge OpEx.

### Medium Severity

- **Lines 1902, 1936-1938, and 1987: the constraint-propagation formula is only valid when the reference stage is Stage 1.**  
  The definition says late constraints at lifecycle stage `N` incur cost relative to "the stage where they should have been defined" but gives `2^(N-1)` (line 1902). That formula assumes the missed constraint should have been defined at Stage 1. If the intended baseline can be another stage `M`, the general expression is `2^(N-M)`. The checkpoint on lines 1936-1938 is consistent with Stage 1 and gives `2^(6-1)=32x`, and the summary on line 1987 is also Stage-1-specific. The definition should make that assumption explicit.
  - Proposed correction: Either write "relative to Stage 1" everywhere, or define the general form as `Correction Cost ≈ 2^(N-M) x Base Effort`, where `M` is the stage where the constraint should have been specified. Then state that the common Stage-1 case reduces to `2^(N-1)`.

- **Lines 1961 and 1902: the deployment-stage multiplier is compared to the wrong baseline.**  
  Line 1961 says a deployment-stage discovery costs `$2^{5-1}=16x$` the effort of catching it "during evaluation design." Under the chapter's formula, `16x` is the multiplier relative to Stage 1, not relative to evaluation/validation. If "evaluation design" means Stage 4, the relative multiplier from Stage 4 to Stage 5 would be `2^(5-4)=2x` under the generalized formula.
  - Proposed correction: Change "during evaluation design" to "during problem definition" if the intended multiplier is 16x. If the intended comparison is truly evaluation-stage detection, change the multiplier to 2x and use the generalized `2^(N-M)` wording.

- **Lines 64, 847-848, and 864-869: "over 100x" / "Rule of Ten" conflicts with the implemented `2^k` model.**  
  The chapter's visible principle uses powers of two: deployment discovered at Stage 5 gives `2^(5-1)=16x` (lines 864-869). However, line 64 says deployment violations cost "over 100x" more than data-understanding catches, and code comments say the helper shows "100x" and models the "Rule of Ten" (lines 847-848). A rule-of-ten model would produce `10^(5-1)=10,000x` from Stage 1 to Stage 5, while the implemented rule gives only `16x`. These are different escalation models.
  - Proposed correction: Use one model consistently. If the chapter wants the `2^(N-1)` teaching model, revise line 64 and the code comments to avoid "over 100x" / "Rule of Ten." If it wants Boehm-style 100x escalation, change the formal formula and all generated examples accordingly.

- **Lines 305-326, 357-360, and 362: the iteration-tax example mixes theoretical capacity, effective iterations, and daily rate.**  
  The small model has a 1-hour cycle, so six months contains `26 x 168 = 4,368` one-hour slots (line 358). The result calculation then uses only `100` effective iterations (lines 317 and 326). That is fine as a modeling cap, but the conclusion switches to "ten experiments/day" (line 362), which is neither the 1-hour capacity (24/day) nor the capped effective rate over six months (100/182 days ≈ 0.55/day). The example also describes a "compound effect" even though the calculation is linear addition capped at 99 percent.
  - Proposed correction: State the model explicitly as "linear improvement with a ceiling and an effective-iteration cap." Replace "ten experiments/day" with wording aligned to the setup, such as "many experiments per week" or "up to hourly experiments, subject to an effective-iteration cap."

### Low Severity

- **Lines 1862-1867 and 1872: alert margins use percent where percentage points are meant.**  
  The sensitivity target drops from 90 to 88 and specificity from 80 to 78. Those are **2 percentage-point** margins, not 2 percent relative margins. The comments say "2% margin" (lines 1864-1865), and the prose uses "percent" for threshold differences (line 1872), which can be misread as relative percent change.
  - Proposed correction: Use "percentage points" for absolute metric deltas, e.g. "alert threshold (2 percentage-point margin)" and "drops more than five percentage points below baseline" when that is the intended meaning.

- **Lines 1505, 1536-1538, and 1593: scaling-law prose overstates the plotted formulas.**  
  The manual curve is `5n e^{-0.05n}`, which rises sublinearly and peaks at `n=20`; it is not a linear curve before saturation in the strict mathematical sense. The prose also says coordination overhead grows "combinatorially" (line 1593), but the described need for every engineer to coordinate with every existing engineer is pairwise and scales like `O(n^2)` interactions, not generally combinatorial/exponential.
  - Proposed correction: Describe the manual curve as "initially increasing but saturating due to pairwise coordination overhead" and, if using complexity language, say "quadratically" or "as `O(n^2)` pairwise coordination paths."

- **Lines 1027-1029: rounded daily data hides the exact unit calculation.**  
  The bandwidth calculation is internally correct: `150 x 10 x 5 MB = 7,500 MB = 7.5 GB`; at 2 Mbps = 0.25 MB/s, upload time is `30,000 s = 8.33 h`, or 104 percent of an 8-hour day. The rendered `bw_daily_gb_str` rounds 7.5 GB to 8 GB, which is acceptable, but the exact arithmetic line jumps from inputs to a rounded output.
  - Proposed correction: Render the daily data as "7.5 GB/day" or "about 8 GB/day" to avoid making `150 x 10 x 5` appear to equal exactly 8 GB.

## Verified Correct

- Lines 201-203 and 241-246: the pie-chart percentages sum to 100 percent; model-focused categories 9 + 5 + 4 = 18 percent.
- Lines 253 and 1985: iteration-cause percentages 60 + 25 + 15 = 100 percent.
- Lines 754-759 and 763: MobileNetV2 size and FLOP calculations are internally consistent with the constants: 3.5M parameters x 4 bytes = 14 MB, and 0.3 GFLOPs = 300 MFLOPs.
- Lines 976-999 and 1027-1031: bandwidth upload time and reduction factor are arithmetically correct aside from the rounded GB display noted above: 7,500 MB / 0.25 MB/s = 30,000 s = 8.3 h; `(7,500 MB x 1000 KB/MB) / (150 x 10 KB) = 5,000x`.
- Lines 1078-1089: storage-cost conversion is consistent with decimal units: USD 100/TB/month = USD 0.10/GB/month and USD 23/TB/month = USD 0.023/GB/month.
- Lines 1436: `4^5 = 1,024` hyperparameter combinations is correct.
- Lines 1832 and 1875: the KS-test complexity claim `O(n log n)` is reasonable when sorting dominates, and `p < 0.01` is a valid threshold statement.
