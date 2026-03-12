# Mission Plan: lab_16_responsible_ai (Volume 2)

## 1. Chapter Alignment

- **Chapter:** Responsible Engineering (`@sec-responsible-ai`)
- **Core Invariant:** The **Fairness Impossibility Theorem** -- it is mathematically impossible to simultaneously satisfy Demographic Parity, Equalized Odds, and Calibration when base rates differ between groups (Kleinberg et al. 2016, Chouldechova 2017). Engineers must treat fairness metrics like latency budgets: explicit trade-offs chosen by stakeholders, enforced by the system, and monitored for violation. The "fairness tax" (4% accuracy drop to enforce demographic parity in the chapter's credit model) is a permanent engineering cost, not a bug to be optimized away.
- **Central Tension:** Students believe there exists a "fair" threshold or algorithm that satisfies all fairness criteria simultaneously -- that fairness is a technical problem with a single correct solution. The chapter's impossibility theorem and worked examples prove otherwise: at every classification threshold, at least one fairness metric is substantially violated when base rates differ. Fairness is a constrained multi-objective optimization requiring explicit *policy* choices. At fleet scale, this problem compounds: a model deployed across regions with heterogeneous demographics must satisfy different fairness constraints in each region, and the overhead of real-time fairness monitoring (10--20 ms per inference) becomes a first-order system cost.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students explore the Fairness Impossibility Theorem through a loan approval classifier. They predict that there exists a single threshold that can satisfy both Demographic Parity and Equalized Odds simultaneously. The instrument sweeps the threshold and shows that at every point, at least one metric is substantially violated -- confirming the mathematical impossibility. Students discover that "fairness" is not a single number but a set of mutually exclusive constraints requiring stakeholder deliberation.

**Act 2 (Design Challenge, 23 min):** Students deploy a fairness-constrained model across two regions with different demographic base rates. They must choose which fairness metric to prioritize for each region, budget the monitoring overhead (10--20 ms latency per inference, 100--500 MB memory), and find a configuration that keeps total accuracy above 80% while satisfying the chosen fairness constraint within a 5% disparity tolerance. The failure state triggers when fairness monitoring overhead pushes latency beyond the SLA, forcing students to confront the responsible AI infrastructure cost at fleet scale.

---

## 3. Act 1: The Impossibility Wall (Calibration -- 12 minutes)

### Pedagogical Goal
Students approach fairness as a technical optimization problem: surely there exists a threshold, algorithm, or post-processing step that makes the model "fair" on all dimensions simultaneously. The chapter's Fairness Impossibility Theorem demolishes this belief: when base rates differ between groups (Group A: 60% positive, Group B: 30% positive), no single threshold can simultaneously satisfy Demographic Parity (equal approval rates), Equalized Odds (equal TPR and FPR), and Calibration (equal predictive meaning). This act forces students to see the impossibility with their own eyes by sweeping the threshold and watching all three metrics.

### The Lock (Structured Prediction)

> "A loan approval model serves two demographic groups. Group A has a 60% repayment rate (base rate). Group B has a 30% repayment rate. You can set any classification threshold between 0 and 1. Can you find a single threshold that simultaneously satisfies Demographic Parity (equal approval rates) AND Equalized Odds (equal TPR and FPR)?"

Options:
- A) Yes -- there is always a threshold that balances both metrics
- B) Yes -- but only if you use separate thresholds per group (which is legal)
- **C) No -- the impossibility theorem proves these metrics are mutually exclusive when base rates differ** (correct)
- D) No -- but only because the model is poorly trained; a better model would satisfy both

Common wrong answer: A) Yes. Students with optimization backgrounds believe any multi-objective problem has a Pareto-optimal solution. They do not yet internalize that certain fairness constraints are *logically incompatible*, not just hard to satisfy simultaneously.

Why wrong: The Chouldechova-Kleinberg result is a mathematical proof, not an empirical finding. When $P(Y=1|S=a) \neq P(Y=1|S=b)$, calibration + equal FPR + equal FNR is provably impossible. No algorithm, no threshold, no amount of data can overcome a logical impossibility.

### The Instrument: Fairness Threshold Sweep

**Primary chart: Triple-line plot -- "Fairness Metric Disagreement Across Thresholds"**
- X-axis: Classification threshold (0.15 -- 0.85)
- Y-axis: Metric disparity (absolute difference between groups), range 0.0 -- 0.6
- Three lines:
  - BlueLine: Demographic Parity disparity (|approval_rate_A - approval_rate_B|)
  - OrangeLine: Equalized Odds disparity (|TPR_A - TPR_B| + |FPR_A - FPR_B|)
  - GreenLine: Equal Opportunity disparity (|TPR_A - TPR_B|)
- Horizontal reference line at y = 0 (perfect fairness)
- A dashed vertical line tracks the student's selected threshold

Controls:
- **Threshold** slider: 0.15 -- 0.85 (step 0.01; default 0.50)
- **Group A base rate** slider: 0.30 -- 0.80 (step 0.05; default 0.60)
- **Group B base rate** slider: 0.10 -- 0.60 (step 0.05; default 0.30)

At every threshold position, the three metric lines are displayed. The student observes that when one metric approaches zero (fair), at least one other metric increases substantially. An annotation box shows the current values: "DP: [X], EO: [Y], EqOpp: [Z]".

When base rates are set equal (A = B), all three metrics converge to zero at the same threshold -- confirming that the impossibility only arises when base rates differ.

**Secondary chart: Confusion matrix panel**
- Two side-by-side confusion matrices (Group A, Group B) that update in real time as the threshold moves
- TP, FP, FN, TN counts displayed with TPR and FPR derived below each

### The Reveal
> "You predicted [answer]. At threshold 0.50 with base rates 0.60 and 0.30: Demographic Parity disparity = [X], Equalized Odds disparity = [Y], Equal Opportunity disparity = [Z]. **No threshold brings all three to zero simultaneously.** This is the Chouldechova-Kleinberg impossibility: when base rates differ, you must choose which fairness criterion to satisfy and accept violation of the others."

### Reflection (Structured)

Four-option multiple choice:

> "The chapter states that engineers must treat fairness metrics 'like latency budgets -- explicit trade-offs chosen by stakeholders.' What does this mean in practice?"

- A) Engineers should maximize accuracy and let regulators decide fairness after deployment
- B) Engineers should always prioritize Demographic Parity because it is the simplest metric
- **C) Engineers must make an explicit policy decision about which fairness criterion to prioritize for each deployment context, accepting that other criteria will be violated** (correct)
- D) Engineers should use separate models for each demographic group to avoid the impossibility entirely

### Math Peek

$$P(\hat{Y}=1 | S=a) = P(\hat{Y}=1 | S=b) \quad \text{(Demographic Parity)}$$
$$P(\hat{Y}=1 | S=a, Y=y) = P(\hat{Y}=1 | S=b, Y=y) \quad \text{(Equalized Odds)}$$

When $P(Y=1|S=a) \neq P(Y=1|S=b)$, these two conditions plus calibration ($P(Y=1|\hat{Y}=1, S=a) = P(Y=1|\hat{Y}=1, S=b)$) are mutually exclusive. Proof: Kleinberg et al. (2016), Chouldechova (2017).

---

## 4. Act 2: Fleet Fairness Under Constraint (Design Challenge -- 23 minutes)

### Pedagogical Goal
Students move from the single-model impossibility to the fleet-scale challenge: deploying a fairness-constrained model across regions with heterogeneous demographics. The chapter documents that real-time fairness monitoring adds 10--20 ms latency per inference and 100--500 MB memory, while on-demand explainability requires 50--1000x more compute than inference itself. Students must balance the fairness criterion choice (which metric to prioritize), the monitoring overhead (which eats into the latency SLA), and the accuracy cost (the "fairness tax" of ~4 percentage points for demographic parity enforcement). The design challenge is to meet a fairness SLA across both regions while staying within a 100 ms total latency budget.

### The Lock (Numeric Prediction)

> "Your fleet serves two regions. Region 1 has homogeneous demographics (base rates: 55% vs 50%). Region 2 has heterogeneous demographics (base rates: 70% vs 25%). Your baseline model has 85% accuracy. If you enforce Demographic Parity across both regions, what is your expected accuracy in Region 2 (the heterogeneous one)?"

Students enter a percentage (bounded: 60% -- 90%). Expected wrong answers: 82--84% (students expect a small drop similar to the chapter's 4% fairness tax). Actual: With base rates of 70% vs 25%, enforcing parity requires dramatically lowering the threshold for the minority group, producing an accuracy of approximately **75--78%** in Region 2, substantially worse than the 4% drop seen in the homogeneous case. The fairness tax scales with base rate divergence.

### The Instrument: Fleet Fairness Dashboard

**Primary chart: Dual-region fairness radar**
- Two radar plots side by side (Region 1, Region 2)
- Five axes: Accuracy, DP Disparity, EqOpp Disparity, Latency (ms), Memory (MB)
- The student's configuration is overlaid on the target SLA boundary

**Secondary chart: Latency budget waterfall**
- Stacked horizontal bar showing: Model inference (fixed, 30 ms) + Fairness monitoring (10--20 ms) + Explainability (0 / 20--50 ms) = Total
- SLA line at 100 ms
- Bar turns RedLine when total exceeds SLA

Controls:
- **Fairness metric** selector: Demographic Parity / Equalized Odds / Equal Opportunity / None
- **Fairness tolerance** slider: 0.01 -- 0.20 (step 0.01; default 0.05) -- maximum allowed disparity
- **Monitoring level** selector: None (0 ms, 0 MB) / Basic (10 ms, 100 MB) / Full (20 ms, 500 MB)
- **Explainability level** selector: None (0 ms) / On-demand LIME (20 ms) / Full SHAP (50 ms, +50--100 MB)
- **Deployment context** toggle: Single-region (homogeneous) vs Multi-region (heterogeneous)

Formulas:
- `Accuracy_constrained = Accuracy_baseline - Fairness_tax(metric, base_rate_gap)`
- `Fairness_tax(DP, gap) = gap * 10` (percentage points; scales with base rate divergence)
- `Fairness_tax(EqOpp, gap) = gap * 5` (less restrictive than DP)
- `Total_latency = Inference_ms + Monitoring_ms + Explainability_ms`
- `Total_memory = Model_MB + Monitoring_MB + Explainability_MB`

### The Scaling Challenge

**"Deploy a model across both regions that satisfies: (1) chosen fairness metric disparity < 0.05, (2) total latency < 100 ms, (3) accuracy > 80% in both regions."**

Students discover:
- Demographic Parity in Region 2 (heterogeneous, 70% vs 25%) drops accuracy to ~75%, failing the 80% accuracy floor
- Equalized Odds or Equal Opportunity allow higher accuracy (~81%) while maintaining disparity < 0.05
- Full SHAP explainability (50 ms) + Full monitoring (20 ms) + inference (30 ms) = 100 ms, exactly at the SLA boundary
- The solution requires: Equal Opportunity metric + Basic monitoring (10 ms) + On-demand LIME (20 ms) = 60 ms total latency with ~81% accuracy and < 0.05 disparity

### The Failure State

**Trigger condition:** `Total_latency > 100` (SLA violation from responsible AI overhead)

**Visual change:** The latency waterfall bar extends past the 100 ms SLA line, turning RedLine. The radar plot's latency axis turns red.

**Banner text:** "SLA VIOLATED -- Responsible AI overhead ({monitoring_ms} ms monitoring + {explain_ms} ms explainability) pushes total latency to {total_ms} ms, exceeding the 100 ms budget. Reduce monitoring level or disable on-demand explainability to recover SLA compliance."

The failure is reversible: reducing monitoring or explainability level immediately pulls latency below the SLA.

### Structured Reflection

Four-option multiple choice:

> "The chapter documents that responsible AI infrastructure adds 10--20 ms monitoring latency and 50--1000x compute for explainability. Given the fairness impossibility theorem, how should a fleet architect prioritize?"

- A) Deploy without fairness monitoring to preserve latency SLA, and audit fairness quarterly
- **B) Choose the fairness metric that best matches the deployment context's regulatory requirements, budget the monitoring overhead as a first-class latency cost, and accept the accuracy-fairness trade-off explicitly** (correct)
- C) Use Demographic Parity everywhere because it is the most commonly understood metric
- D) Train separate models for each region to avoid the impossibility theorem entirely

### Math Peek

$$\text{Fairness Tax}(\text{DP}) \approx |P(Y=1|S=a) - P(Y=1|S=b)| \times k$$

where $k \approx 10$ (percentage points of accuracy per unit base-rate gap). For Region 2: $|0.70 - 0.25| \times 10 = 4.5$ pp, yielding $85 - 4.5 \approx 80.5\%$ (rough approximation; actual depends on score distributions).

$$\text{Latency}_{total} = \text{Inference} + \text{Monitoring} + \text{Explainability}$$

---

## 5. Visual Layout Specification

### Act 1: Fairness Threshold Sweep
- **Primary:** Triple-line plot. X: threshold (0.15--0.85). Y: metric disparity (0--0.6). Three lines (DP/EO/EqOpp). Vertical dashed line at student's selected threshold. Reference line at y=0.
- **Secondary:** Two side-by-side confusion matrices (Group A, Group B) with TP/FP/FN/TN counts updating in real time.
- **Failure state:** N/A for Act 1.

### Act 2: Fleet Fairness Dashboard
- **Primary:** Two radar plots (Region 1, Region 2). Five axes: Accuracy (80--90%), DP Disparity (0--0.30), EqOpp Disparity (0--0.30), Latency (0--150 ms), Memory (0--1000 MB). SLA boundary polygon overlaid.
- **Secondary:** Horizontal stacked bar (latency waterfall). Segments: Inference (BlueLine, 30 ms), Monitoring (OrangeLine, 0--20 ms), Explainability (GreenLine, 0--50 ms). SLA line at 100 ms.
- **Failure state:** When total latency > 100 ms, waterfall bar extends past SLA in RedLine; banner appears.

---

## 6. Deployment Context Definitions

| Context | Region | Base Rates (A vs B) | Regulatory Regime | Key Constraint |
|---|---|---|---|---|
| **Single-region (homogeneous)** | US domestic | 55% vs 50% | ECOA, Fair Housing Act | Small base-rate gap; DP and EqOpp nearly achievable simultaneously; fairness tax is ~2% |
| **Multi-region (heterogeneous)** | US + developing market | 70% vs 25% | GDPR + local regulation | Large base-rate gap; impossibility bites hard; fairness tax is ~5--10%; metric choice is critical |

The two contexts demonstrate that fairness constraints have *deployment-dependent costs*. In homogeneous regions, the fairness tax is modest and multiple metrics can be approximately satisfied. In heterogeneous regions, the base-rate gap amplifies the impossibility theorem, forcing explicit prioritization of one fairness criterion at the expense of others and imposing a larger accuracy penalty.

---

## 7. Design Ledger Output

```json
{
  "chapter": 16,
  "fairness_metric_chosen": "demographic_parity | equalized_odds | equal_opportunity",
  "fairness_tolerance": 0.05,
  "monitoring_level": "none | basic | full",
  "explainability_level": "none | lime | shap",
  "accuracy_region1_pct": 83,
  "accuracy_region2_pct": 81,
  "total_latency_ms": 60,
  "sla_violated": false,
  "fairness_tax_experienced_pct": 4
}
```

The `fairness_metric_chosen` and `fairness_tax_experienced_pct` fields feed forward to:
- **Lab 17 (Conclusion):** The fleet synthesis radar includes Responsible Engineering as one of the 6 principle dimensions, reading `fairness_metric_chosen` and `total_latency_ms` to show the governance cost on the radar
- The `sla_violated` flag indicates whether the student successfully navigated the latency-fairness trade-off

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Impossibility theorem (DP + EO + Calibration incompatible) | `@sec-responsible-ai-fairness-machine-learning-2ba4`, fn-fairness-impossibility | "Kleinberg et al. (2016) and Chouldechova (2017) independently proved that calibration, equalized odds, and demographic parity are mutually exclusive for any classifier where base rates differ" |
| 4% fairness tax (85% -> 81%) | `@sec-responsible-ai-fairness-machine-learning-2ba4`, FairnessTaxAnalysis class | "Fairness is not free. Enforcing parity cost 4% accuracy" |
| 30 percentage point DP disparity (70% vs 40% approval) | `@sec-responsible-ai-fairness-machine-learning-2ba4`, Calculating Fairness Metrics example | "Disparity: 0.70 - 0.40 = 0.30 (30 percentage point gap)" |
| Fairness monitoring: 10--20 ms latency | `@sec-responsible-ai-introduction-responsible-ai-2724` | "real-time bias monitoring adds 10--20 ms of latency per inference" |
| Fairness monitoring: 100--500 MB memory | `@sec-responsible-ai-introduction-responsible-ai-2724` | "fairness monitoring adds 10--20 ms latency per request and requires 100-500 MB additional memory" |
| Explainability: 50--1000x compute overhead | `@sec-responsible-ai-introduction-responsible-ai-2724` | "on-demand explainability can require 50--1000x more compute than the inference itself" |
| Fairness-aware training: 5--15% more training time | `@sec-responsible-ai-introduction-responsible-ai-2724` | "fairness-aware training algorithms add 5--15% to training time" |
| COMPAS: 45% vs 23% false positive rate by race | `@sec-responsible-ai-transparency-explainability-b137`, fn-compas-bias | "Black defendants were falsely flagged at nearly twice the rate of white defendants (45% vs. 23% false positive rate)" |
| Healthcare algorithm: 200M Americans affected | `@sec-responsible-ai-fairness-machine-learning-2ba4`, fn-healthcare-algorithm-bias | "The Optum algorithm affected approximately 200 million Americans annually" |
| GDPR Article 22: explainability as legal requirement | `@sec-responsible-ai-transparency-explainability-b137`, fn-gdpr-article-22 | "any model making automated decisions with legal or significant effects must expose decision logic on demand" |
| Explanation engine: 20--50 ms additional latency | `@sec-responsible-ai-introduction-responsible-ai-2724` | "the explanation engine increases response time by 20--50 ms" |
