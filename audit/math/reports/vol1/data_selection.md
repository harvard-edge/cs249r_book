# Math Audit: `book/quarto/contents/vol1/data_selection/data_selection.qmd`

Audit method: direct reasoning over the source file only; no Gemini or external validation. Scope included equations, derivations, numeric examples, unit conversions, complexity/scaling claims, dataset-size arithmetic, and prose-equation consistency.

## Findings

### 1. KWS parameter count is inconsistent with the claimed quantized model size

- **Lines:** 2488
- **Severity:** High
- **Issue:** The text says the DS-CNN has `200 K` parameters and a `~50 KB quantized` model size. With standard 8-bit quantization, 200K parameters require about 200 KB for weights alone, before metadata, alignment, and activation memory. A 50 KB 8-bit model would be closer to 50K parameters, not 200K.
- **Proposed correction:** Either change the model size to `~200 KB quantized weights` or change the parameter count to `~50 K parameters` if the intended deployed model is 50 KB at 8-bit precision. If using 2-bit weights or structured sharing, state that explicitly.

### 2. Clinical labeling full-dataset cost is undercounted

- **Lines:** 1977
- **Severity:** Medium
- **Issue:** The footnote says a radiologist reviews `50--80` X-rays/hour at `$150--300`/hour. Labeling all `50,000` scans therefore takes `50,000/80 = 625` to `50,000/50 = 1,000` hours. The cost range is `625 * $150 = $93,750` to `1,000 * $300 = $300,000`, not `$75,000--150,000`.
- **Proposed correction:** Replace the full-supervised cost range with approximately `$94,000--300,000`, or revise the hourly rate/throughput assumptions to support `$75,000--150,000`.

### 3. Selection-inequality figure prose mixes two incompatible examples

- **Lines:** 2763-2775, 2860-2862, 2888-2921
- **Severity:** Medium
- **Issue:** The worked example reports Option B savings from `1M * 100 * 0.01 s` baseline versus proxy selection plus subset training, which is about `90 percent` savings. Immediately after, the figure prose says the stacked bars demonstrate that same `savings_b_pct_str`, but the plotted bars use `5 + 40 = 45` versus `100`, i.e. `55 percent` savings. The figure is illustrative, but the prose ties it to the previous computed value.
- **Proposed correction:** Either change the figure arrays to match the worked example's normalized values, or change line 2860 to say the figure is a separate normalized illustration with `55 percent` savings.

### 4. Distributed coreset end-to-end speedup is overstated

- **Lines:** 3553-3561
- **Severity:** Medium
- **Issue:** The text reports `67 minutes` total selection overhead "for 10x training speedup." A ten percent coreset gives a `10x` training-only reduction before overhead. End-to-end time is `0.1 * T_full + 1.12 h`; for a 24-hour full run this is about `3.52 h`, or `6.8x`, not `10x`. The statement that overhead "pays for itself if full training takes more than twelve hours" is a ten-percent-overhead heuristic, not the true break-even: break-even is `1.12 h < 0.9 * T_full`, so `T_full > 1.24 h`.
- **Proposed correction:** Say "67 minutes overhead for a 10x smaller training set; end-to-end speedup is about 6.8x for a 24-hour full run." If retaining the twelve-hour threshold, phrase it as "keeps selection overhead under about ten percent of full training."

### 5. Active-learning break-even example omits or ambiguously double-counts the initial labeled set

- **Lines:** 3234-3248, 3270-3283
- **Severity:** Medium
- **Issue:** The prose lists an initial labeled set of `1,000` samples costing `$10,000`, but the active-learning cost formula only includes `n_active * cost_label + n_rounds * cost_inference = 2,000 * $10 + 10 * $50 = $20,500`. If the `2,000` active labels exclude the initial set, the total should be `$30,500`; if they include it, the text should say so.
- **Proposed correction:** Clarify that `2,000` is the total labeled budget including the initial `1,000`, or add the initial-set cost explicitly to the active-learning total and recompute ROI as `(50,000 - 30,500) / 30,500 ≈ 64 percent`.

### 6. Amortized ROI equation omits the percent normalization used in the table

- **Lines:** 3290-3292, 3323-3325, 3349-3354
- **Severity:** Low
- **Issue:** The displayed formula defines amortized ROI as a raw ratio, but the code multiplies by `100` and the table appends `%`. The numeric table is internally consistent with a percent ROI; the equation is not.
- **Proposed correction:** Change the displayed equation to multiply by `100%`, or remove percent signs from the table and present the values as ratios.

### 7. Data-exhaustion figure y-axis labels are shifted relative to the plotted range

- **Lines:** 185-193
- **Severity:** Low
- **Issue:** The y-axis minimum is `1e9`, but the custom `yticklabels` start at `10^8`. Since the explicit `ytick` list is commented out, the labels risk being assigned to automatically selected ticks, and the first listed label is one order of magnitude below the axis minimum. This can mislabel the log-scale dataset sizes.
- **Proposed correction:** Restore explicit ticks matching the labels, e.g. `ytick={1e9,...,1e15}` with labels `10^9` through `10^15`, or remove custom labels and let pgfplots format them.

### 8. Opening time-savings arithmetic is off by one day

- **Lines:** 34
- **Severity:** Low
- **Issue:** A run that takes a week on the full dataset and a day on a selected subset saves six calendar days, not five. The prose says "that five-day savings."
- **Proposed correction:** Change to "six-day savings," or say "roughly a workweek of savings" if the intended comparison is business days.

### 9. FixMatch label-efficiency framing mixes baselines

- **Lines:** 2010-2012, 2019-2057, 2088-2097
- **Severity:** Low
- **Issue:** The code comments and class docstring describe "200x label reduction for ~8x total cost savings." The `200x` label efficiency is measured against the full `50,000`-label CIFAR-10 baseline, while the `8x` cost reduction is computed against a `4,000`-label supervised baseline. Both calculations can be valid, but combining them in one sentence makes it sound as if the same baseline supports both ratios.
- **Proposed correction:** State the two baselines explicitly: "250 labels is 200x fewer than full supervision; compared with a 4,000-label supervised run, this cost model gives about 8x lower total cost."

### 10. PPD figure labels a vertical accuracy gap as saved compute

- **Lines:** 4182-4186
- **Severity:** Low
- **Issue:** The red arrow is vertical at a fixed dataset size, so it measures an accuracy/performance gap. "Saved compute" is represented by a horizontal gap at a fixed target accuracy. The figure caption earlier notes this, but the label in the figure remains mathematically inconsistent.
- **Proposed correction:** Rename the vertical label to "Performance Gap" or redraw a horizontal arrow between the curves at a fixed accuracy and label that "Saved Compute."

### 11. Deduplication guarantee is overstated

- **Lines:** 4447
- **Severity:** Low
- **Issue:** The takeaway says deduplication is "guaranteed zero accuracy penalty." Exact duplicate removal often has negligible risk, but deduplication is not mathematically guaranteed to preserve accuracy in all cases: duplicates can reflect deployment-frequency weighting, class imbalance, or intentional oversampling. Near-duplicate and semantic deduplication are especially threshold-dependent.
- **Proposed correction:** Replace with "lowest-risk data selection technique, often with no accuracy penalty when exact duplicates are removed carefully."

### 12. Self-supervised labeled-data multiplier conflicts with the immediately stated factors

- **Lines:** 4440, 4450
- **Severity:** Low
- **Issue:** The summary claims a `1,000x` labeled-data multiplier, then the takeaway says foundation models reduce per-task labels by `100x` and marginal compute by `20x`. The earlier table permits 100x to 10,000x label reductions depending on the endpoints, but the local numeric example uses 100x labels (`100,000` to `1,000`). The prose should avoid implying that the demonstrated label multiplier is exactly 1,000x.
- **Proposed correction:** Say "100x to 1,000x-class label reductions are possible depending on the task" or keep the worked-example wording at "100x fewer labels and 20x lower marginal compute."

## Checked Without Findings

- Scaling-asymmetry annualization is internally consistent: `10x` over 3 years is about `2.15x/year`; `2x` over 5 years is about `1.15x/year`; their annualized gap is about `1.87x`.
- The ICR coreset worked example is internally consistent: a 50 percent coreset with 90 percent of the accuracy gain gives `(4.5 / 0.5) / 5.0 = 1.8x` higher ICR.
- The data-quality multiplier arithmetic is consistent: clean `O(1/N)` versus noisy `O(1/sqrt(N))` at `epsilon = 0.01` gives `N_clean = 100`, `N_noisy = 10,000`, and a `100x` ratio.
- Curriculum speedups are correctly computed from epoch counts: `150 -> 115` is about `23 percent`, `220 -> 180` about `18 percent`, `90 -> 80` about `11 percent`, and `90 -> 70` about `22 percent`.
- Active-learning ROI arithmetic is consistent for the stated 1M-scan scenario: labeling all scans at `$5` costs `$5M`; labeling `50,000` costs `$250K`; the label-count ratio is `20x`.
- Storage random-throughput estimates are dimensionally consistent for 4 KB random reads: HDD `80 IOPS * 4 KB ≈ 0.3 MB/s`, SATA SSD `10k IOPS * 4 KB ≈ 40 MB/s`, and NVMe `500k IOPS * 4 KB ≈ 2,000 MB/s`.
- Data echoing arithmetic is internally consistent under the stated "echoed data is equally valuable" assumption: `90 * 1.28M / 300 ≈ 384,000 s ≈ 107 h`, and echo factor 2 halves that to about `53 h`.
- Chinchilla rule-of-thumb arithmetic is consistent: if `D_opt ∝ C^0.5`, doubling compute increases optimal data by `sqrt(2) ≈ 1.41`, about 40 percent.
