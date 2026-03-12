# Mission Plan: lab_15_responsible_engr

## 1. Chapter Alignment

- **Chapter:** Responsible Engineering (`@sec-responsible-engineering`)
- **Core Invariant:** The **Fairness-Accuracy Pareto Frontier**: aggregate accuracy conceals disparities across demographic groups. The Gender Shades study found a 43x disparity in error rates (0.8% for light-skinned males vs. 34.7% for dark-skinned females) while aggregate metrics showed "high accuracy." The Pareto frontier reveals that a "sweet spot" typically exists where large fairness gains (4x reduction in disparity) cost only modest accuracy loss (~3 percentage points: from 94.7% to 91.3%).
- **Central Tension:** Students believe that high aggregate accuracy implies the model works well for everyone. The chapter demonstrates the opposite: "A loan model that approves 95% of qualified majority-group applicants while rejecting 40% of equally qualified minority-group applicants meets its loss function perfectly." COMPAS achieved calibration while showing false positive rates of 44.9% for Black defendants vs. 23.5% for White defendants. Aggregate metrics are not just incomplete; they actively hide the problem.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that a model with 95% aggregate accuracy is ready for deployment. The chapter's Gender Shades data reveals that disaggregated evaluation can uncover 43x error disparities hidden beneath "high accuracy." Students predict whether a 95% accurate model can have any subgroup with >10% error rate, discover that the answer is emphatically yes, and learn that aggregate accuracy is a misleading metric for safety-critical systems.

**Act 2 (Design Challenge, 23 min):** Students navigate the Fairness-Accuracy Pareto Frontier, adjusting fairness constraints to find the "sweet spot" where substantial disparity reduction costs minimal accuracy. They discover that enforcing strict equality (zero disparity) drops accuracy significantly, but that a moderate constraint (Point B on the Pareto frontier) achieves 4x fairer outcomes at only ~3% accuracy cost. The challenge is to meet the four-fifths rule (selection rate ratio >= 0.8) while maintaining accuracy above a deployment floor.

---

## 3. Act 1: The Disparity Discovery (Calibration -- 12 minutes)

### Pedagogical Goal
Students conflate aggregate accuracy with subgroup performance. The chapter states: "Systems reporting high overall accuracy simultaneously achieved error rates as low as 0.8% for light-skinned males and as high as 34.7% for dark-skinned females." This 43x disparity is invisible in aggregate metrics. This act forces students to predict whether large subgroup disparities can coexist with high aggregate accuracy, then shows them the Gender Shades data.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "A facial recognition system reports 96% aggregate accuracy on a 10,000-image test set. When you disaggregate by demographic group, what is the highest possible error rate for the worst-performing subgroup?"

Options:
- A) About 6% -- aggregate accuracy bounds subgroup performance within a few points
- B) About 10% -- some groups may be worse, but not dramatically
- C) About 20% -- there can be significant disparities
- **D) About 35% -- aggregate accuracy can conceal 43x error rate disparities across subgroups** <-- correct

The correct answer is D. The Gender Shades study found exactly this: systems with high aggregate accuracy showed 0.8% error for the best subgroup and 34.7% for the worst. Students who pick A or B have the misconception that aggregate accuracy constrains subgroup performance. It does not, because the majority group dominates the aggregate.

### The Instrument: Disaggregated Evaluation Dashboard

A **grouped bar chart** showing error rates by demographic subgroup:

- **X-axis:** Demographic subgroups (Light-skinned Males, Light-skinned Females, Dark-skinned Males, Dark-skinned Females)
- **Y-axis:** Error rate (%), 0 to 40%
- **Bars:** One per subgroup, colored by severity: green (<5%), orange (5--15%), red (>15%)
- **Horizontal dashed line:** Aggregate error rate (appears low, ~4%)
- **Disparity ratio annotation:** "Worst/Best = 43x" displayed prominently

Controls:
- **Training data balance slider** (Balanced to Highly Imbalanced, 5 levels, default Highly Imbalanced): At "Balanced," the subgroup error rates converge. At "Highly Imbalanced" (reflecting real-world training data composition), disparities emerge. Students see how training data composition directly drives disparity.
- **Subgroup sample size slider** (10 to 10,000, step log scale, default 1000): Demonstrates the testing constraint: with 1% minority representation, a 1,000-sample test set provides only 10 samples for the minority subgroup -- insufficient for reliable evaluation. At 10 samples, the error bar is enormous.
- **Deployment context toggle** (H100 Cloud / Smartphone Mobile): Cloud serves global traffic (diverse demographics); Mobile serves individual users (may serve primarily one demographic, masking disparity).

### The Reveal
After interaction:
> "You predicted the worst subgroup error rate would be [X]%. The actual worst-performing subgroup (dark-skinned females) has an error rate of **34.7%** -- a **43x disparity** from the best-performing subgroup (0.8%). The aggregate metric of ~4% concealed this entirely. This is why disaggregated evaluation is not optional: it is the only way to see the harm."

### Reflection (Structured)
Four-option multiple choice:

> "The COMPAS recidivism algorithm was 'calibrated' (a score of 7 meant the same probability for any group). Yet it was found to be biased. Why?"
- A) Calibration is not a real fairness metric -- it was incorrectly applied
- B) The algorithm had a bug that produced different scores for different groups
- **C) Calibration and equalized odds cannot both be satisfied when base rates differ between groups (the Impossibility Theorem); COMPAS achieved calibration but violated equalized odds, with false positive rates of 44.9% vs. 23.5%** <-- correct
- D) The data was too small to produce reliable calibration

**Math Peek (collapsible):**
$$\text{Disparate Impact Ratio} = \frac{\text{Selection Rate}_{\text{protected group}}}{\text{Selection Rate}_{\text{majority group}}} \geq 0.8 \quad \text{(Four-Fifths Rule)}$$
$$\text{Error Disparity} = \frac{\text{Error Rate}_{\text{worst subgroup}}}{\text{Error Rate}_{\text{best subgroup}}} = \frac{34.7\%}{0.8\%} = 43.4\times$$

---

## 4. Act 2: The Fairness-Accuracy Trade-off (Design Challenge -- 23 minutes)

### Pedagogical Goal
Students believe that fairness and accuracy are strictly opposed: making a model fairer always makes it significantly less accurate. The chapter's Pareto frontier visualization shows that "while perfect fairness (zero disparity) often requires a significant drop in accuracy, a 'Sweet Spot' typically exists where large fairness gains can be achieved with minimal accuracy loss." Students must find Point B on the frontier and discover that responsible engineering is feasible, not prohibitively expensive.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "An unconstrained model achieves 94.7% accuracy with 18% demographic disparity (Point A on the Pareto frontier). You add a fairness constraint that reduces disparity to 5% (Point B). By how many percentage points does accuracy drop?"

Students type a number. Expected wrong answers: 8--15 points (students assume a large accuracy tax). Actual: approximately **3.4 percentage points** (from 94.7% to ~91.3%). The frontier's shape (exponential decay near zero disparity, nearly flat in the mid-range) means that the first large reduction in disparity is nearly "free."

### The Instrument: Fairness-Accuracy Pareto Explorer

**Primary chart -- Pareto Frontier Curve:**
- **X-axis:** Demographic Disparity (0.0 to 0.20, reversed so "fairer" is to the right)
- **Y-axis:** Model Accuracy (85% to 96%)
- **Curve:** Pareto frontier (BlueLine) following `accuracy = 0.85 + 0.10 * (1 - exp(-20 * disparity))`
- **Annotated points:**
  - Point A (RedLine dot): Unconstrained (18% disparity, 94.7% accuracy)
  - Point B (GreenLine dot): Sweet Spot (5% disparity, 91.3% accuracy)
  - Point C (BlueLine dot): Strict Equality (0% disparity, 85.0% accuracy)
- **Student's prediction** overlaid as a horizontal dashed line at their predicted accuracy

Controls:
- **Fairness constraint slider** (0% to 20% maximum allowed disparity, step 1%, default 20%): Moves the operating point along the frontier. At 20%, no constraint (Point A). At 0%, strict equality (Point C). Students see that most accuracy loss occurs in the last few percentage points of disparity reduction.
- **Four-fifths rule toggle** (ON/OFF): When ON, displays a vertical line at the disparity level where the four-fifths rule is violated. Students see whether their current operating point passes the legal threshold.
- **Deployment context toggle** (H100 Cloud / Smartphone Mobile): Cloud: serves millions of users across demographics; even small disparity rates affect thousands of people. Mobile: serves individual users; aggregate disparity may be less relevant, but individual fairness matters more.

**Secondary chart -- Subgroup Performance Table:**
- Live-updating table showing error rates for each demographic subgroup at the current fairness constraint level
- Columns: Subgroup, Error Rate, Selection Rate, Four-Fifths Compliance (checkmark or X)
- Color-coded: green when within threshold, red when violating

**Tertiary chart -- Cost of Fairness Bar:**
- **X-axis:** Fairness constraint level (Point A, B, C)
- **Y-axis:** Two bars per level: Accuracy (left bar) and Audit/Compliance Cost (right bar)
- Shows that the total cost of responsible engineering (accuracy loss + monitoring/audit infrastructure) is minimized at Point B, not at the extremes

### The Scaling Challenge
**"Find the fairness constraint level that satisfies the four-fifths rule (disparate impact ratio >= 0.8) while keeping accuracy above 90%. Then compare the cost across deployment contexts."**

Students discover:
- The four-fifths rule is satisfied at approximately 8% disparity, where accuracy is ~93%
- Point B (5% disparity, 91.3% accuracy) provides additional safety margin above the legal threshold
- On H100 Cloud (serving 1M queries/day): 3.4% accuracy loss = ~34,000 additional errors/day, but the fairness gain prevents harm to ~50,000 users/day who would have been affected by the disparity
- On Smartphone: individual fairness (does the model work for THIS user?) matters more than group statistics

### The Failure State
**Trigger:** `disparate_impact_ratio < 0.8` (violates four-fifths rule)

**Visual:** The four-fifths rule line turns red; non-compliant subgroup rows in the table turn red. Banner:
> "**DISPARATE IMPACT VIOLATION -- Four-fifths rule failed.** Selection rate ratio is [X] (threshold: 0.80). The worst-performing subgroup ([name]) has error rate [Y]% vs. best group at [Z]%. This would fail a regulatory audit."

**Secondary failure (accuracy floor):** When accuracy < 85%:
> "**ACCURACY FLOOR BREACHED -- Model unusable.** Accuracy has dropped to [X]% due to strict fairness constraints. Relax the constraint or improve the training data to shift the Pareto frontier."

### Structured Reflection
Four-option multiple choice:

> "The chapter states that Amazon scrapped its recruiting AI after discovering gender bias. Through the D-A-M taxonomy, where did the failure originate?"
- A) Machine axis -- the hardware was not powerful enough to process resumes fairly
- B) Algorithm axis -- the loss function was misaligned with fairness requirements
- **C) Data axis -- historical hiring data encoded gender bias, and the model faithfully reproduced it; removing explicit gender left proxy variables (college names, activity descriptions) that reconstructed the protected attribute** <-- correct
- D) All three axes equally -- the failure was systemic with no single root cause

**Math Peek:**
$$\text{Accuracy} = 0.85 + 0.10 \times (1 - e^{-20 \times \text{disparity}})$$
$$\text{Four-Fifths Rule: } \frac{\text{Selection Rate}_{\text{protected}}}{\text{Selection Rate}_{\text{majority}}} \geq 0.80$$
$$\text{At Point B: } 94.7\% - 91.3\% = 3.4\% \text{ accuracy cost for } 3.6\times \text{ fairness improvement}$$

---

## 5. Visual Layout Specification

### Act 1: Disparity Discovery
- **Primary:** Grouped bar chart -- X: Demographic subgroups (4 groups), Y: Error rate (%). Bars colored green/orange/red by severity. Horizontal dashed line at aggregate error rate. Disparity ratio annotation.
- **Secondary:** Confidence interval overlay showing how subgroup sample size affects reliability of error rate estimates.

### Act 2: Fairness-Accuracy Trade-off
- **Primary:** Pareto frontier curve -- X: Disparity (reversed), Y: Accuracy. Three annotated points (A, B, C). Student prediction overlaid. Four-fifths rule threshold as vertical line.
- **Secondary:** Subgroup performance table with live error rates, selection rates, and compliance status. Color-coded.
- **Tertiary:** Cost of fairness comparison bars at three operating points.

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---|---|---|---|---|
| **Cloud (H100)** | H100 (80 GB HBM3) | 80 GB | 700 W | Serves diverse global population; group fairness metrics (demographic parity, equalized odds) are primary; scale amplifies harm of small disparities |
| **Mobile (Smartphone)** | Mobile GPU (4--8 GB) | 4--8 GB | 5 W | Serves individual users; individual fairness (does it work for this person?) matters more; on-device inference means user data stays private but model updates are slow |

The two contexts demonstrate that "fairness" means different things at different scales. A 2% disparity across 1 million daily cloud users affects 20,000 people. The same disparity on a personal device affects one person, but that person has no recourse if the model fails for their demographic.

---

## 7. Design Ledger Output

```json
{
  "chapter": 15,
  "fairness_constraint_disparity_pct": 5,
  "accuracy_at_constraint_pct": 91.3,
  "accuracy_loss_pct": 3.4,
  "four_fifths_compliant": true,
  "worst_subgroup_error_pct": 8.2,
  "disparity_ratio": 4.1,
  "dam_failure_axis": "data | algorithm | machine"
}
```

The `fairness_constraint_disparity_pct` and `accuracy_at_constraint_pct` feed forward to:
- **Lab 16 (Conclusion):** The fairness-accuracy trade-off contributes to the multi-objective synthesis; the accuracy loss becomes one dimension of the final Pareto analysis

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Gender Shades: 0.8% error (light male) vs. 34.7% (dark female) | @sec-responsible-engineering, @tbl-gender-shades-results (line 376--381) | Error rates: 0.8%, 7.1%, 12.0%, 34.7% across four demographic groups |
| 43x error rate disparity | @sec-responsible-engineering (line 384) | "a single aggregate accuracy number can conceal 43x error rate disparities across intersectional subgroups" |
| Pareto frontier: accuracy = 0.85 + 0.10*(1-exp(-20*disparity)) | @sec-responsible-engineering, @fig-fairness-frontier code (line 280) | "accuracy = 0.85 + 0.10 * (1 - np.exp(-20 * disparity))" |
| Point A: 18% disparity, 94.7% accuracy | @sec-responsible-engineering, @fig-fairness-frontier code (line 286) | Point A plotted at (0.18, 0.947) |
| Point B: 5% disparity, 91.3% accuracy, "4x Fairer" | @sec-responsible-engineering, @fig-fairness-frontier code (line 291--292) | Point B plotted at (0.05, 0.913) with label "91% Acc, 4x Fairer" |
| Point C: strict equality, 85% accuracy | @sec-responsible-engineering, @fig-fairness-frontier code (line 294) | Point C plotted at (0.0, 0.85) |
| Four-fifths rule: selection rate ratio >= 0.8 | @sec-responsible-engineering (line 388) | "selection rate ratios of at least 0.8 relative to the highest group's rate (the four-fifths rule)" |
| COMPAS: false positive 44.9% Black vs. 23.5% White | @sec-responsible-engineering, COMPAS callout (line 93) | "Black defendants...incorrectly flagged as high-risk at nearly twice the rate of White defendants (44.9% vs. 23.5%)" |
| Amazon recruiting tool: proxy variables | @sec-responsible-engineering (line 79) | "The model reconstructed protected attributes from these proxies without ever seeing gender labels directly" |
| 1000-sample test, 1% minority = 10 samples | @sec-responsible-engineering (line 373) | "a 1,000-sample test set...provides only 10 samples for a 1% minority subgroup" |
