# Mission Plan: lab_14_ml_ops

## 1. Chapter Alignment

- **Chapter:** ML Operations (`@sec-ml-operations`)
- **Core Invariant:** The **Statistical Drift Invariant** (Principle 10): `Acc(t) ≈ Acc_0 - lambda * D(P_t || P_0)`. Models decay without code changes because the world drifts away from the training distribution. When Jensen-Shannon divergence exceeds 0.1, accuracy drops exceed 5%; when it exceeds 0.3, degradation reaches 15--30%. Infrastructure health checks (latency, uptime, error rate) remain green throughout.
- **Central Tension:** Students believe that deployment is a milestone: once a model passes benchmarks and ships, the engineering is done. The chapter's central claim demolishes this: "A model experiencing data drift continues serving predictions with full confidence while accuracy degrades week by week, triggering no alerts because every health check (latency, throughput, uptime) remains green." A recommendation system initially boosting sales by 15% silently degraded over six months, eventually reducing sales by 5%, losing an estimated $10 million before detection. The failure mode is not a crash but a silent, invisible rot.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that a healthy monitoring dashboard (green uptime, green latency, green error rate) means the model is performing well. The chapter's "operational mismatch" shows that a model can degrade from 94% to 81% accuracy while every infrastructure metric stays green. Students predict whether standard monitoring will detect a 13-percentage-point accuracy drop, discover it will not, and learn why MLOps exists: to close the gap between infrastructure health and model health.

**Act 2 (Design Challenge, 23 min):** Students must design a retraining strategy that balances compute cost against accuracy decay. The chapter's retraining economics formula (Retrain if: delta_Accuracy x Value_per_Point > Training_Cost + Deployment_Risk) and the automation ROI calculation (80 hours one-time vs. 4 hours/week manual, break-even at 20 weeks) provide the quantitative framework. Students manipulate drift rate, retraining cost, and monitoring threshold to find the optimal retraining interval, discovering that retraining frequency spans 4 orders of magnitude (daily for recommendation systems to quarterly for embedded devices) and that the "sweet spot" depends on the deployment context.

---

## 3. Act 1: The Silent Failure (Calibration -- 12 minutes)

### Pedagogical Goal
Students equate system health with model health. The chapter's opening question -- "Why can an ML system be perfectly available and perfectly wrong at the same time?" -- is the core misconception this act corrects. A ridesharing demand prediction system shows 94% accuracy at deployment, drops to 88% by week 4, and no infrastructure metric flags the degradation. Students must confront the operational mismatch: traditional monitoring answers "Is the server running?" but MLOps must answer "Is the model still accurate?"

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "A model is deployed with 94% accuracy. Over 6 months, user behavior shifts due to a competitor's new promotion. Your infrastructure dashboard shows: 100% uptime, 15 ms P99 latency, 0.01% error rate. What has happened to model accuracy?"

Options:
- A) Still ~94% -- all health metrics are green, so the model is fine
- B) Down to ~91% -- slight degradation, but infrastructure metrics would show some warning
- **C) Down to ~81% -- accuracy has degraded 13 points, but no infrastructure metric has changed** <-- correct
- D) The model has crashed -- hidden failures eventually cause system failures

The correct answer is C. The chapter describes exactly this scenario: "Week one: excellent. Week four: accuracy has dropped to 88%...Week eight: a product manager notices driver dispatch is inefficient; investigation reveals the model has not adapted to a competitor's new promotion." Students who pick A have the misconception this act is designed to correct. Students who pick B underestimate the severity of silent degradation.

### The Instrument: The Drift Radar

A **dual-panel dashboard** with synchronized time axis:

**Left panel -- Infrastructure Health (All Green):**
- **X-axis:** Weeks since deployment (0 to 26)
- **Y-axis:** Health metrics (0--100%)
- **Lines:** Uptime (solid green, flat at 99.9%), Latency P99 (dashed green, flat at 15 ms), Error rate (dotted green, flat at 0.01%)
- All lines stay flat and green regardless of time slider position

**Right panel -- Model Health (Silent Decay):**
- **X-axis:** Weeks since deployment (0 to 26), synced with left panel
- **Y-axis:** Model accuracy (70% to 100%)
- **Line:** Accuracy (RedLine) decaying according to the Degradation Equation
- **Shaded zones:** Green (>90%), Orange (80--90%), Red (<80%)
- **PSI indicator:** Population Stability Index displayed as a bar that grows over time. Threshold line at PSI = 0.1 (warning) and PSI = 0.25 (critical)

Controls:
- **Weeks since deployment slider** (0 to 26, step 1, default 0): Advances both panels simultaneously. Infrastructure stays green; accuracy decays.
- **Drift rate (lambda) slider** (0.01 to 0.20, step 0.01, default 0.10): Controls how quickly accuracy degrades. Higher lambda = faster decay. Maps to different domain types (e-commerce = high lambda ~0.15; medical imaging = low lambda ~0.03).
- **Deployment context toggle** (H100 Cloud / Jetson Orin NX Edge): Cloud model sees faster drift (dynamic user behavior); edge model sees slower drift (stable physical environment) but retraining is harder (OTA update required).

### The Reveal
After interaction:
> "You predicted accuracy would be [X]%. After 26 weeks with lambda = 0.10, accuracy has dropped to **~81%**. Meanwhile, every infrastructure metric stayed green. The model served confidently wrong predictions for months. This is the operational mismatch: infrastructure health and model health are independent signals."

### Reflection (Structured)
Four-option multiple choice:

> "The chapter states that a recommendation model lost $10 million in revenue over 6 months due to silent degradation. What monitoring would have caught this?"
- A) CPU/RAM utilization monitoring -- resource spikes indicate model problems
- B) Error rate monitoring -- degraded models produce more errors
- **C) Statistical drift monitoring (PSI/KL divergence on input features + accuracy on holdout set) -- tracks prediction quality, not infrastructure** <-- correct
- D) Latency percentile monitoring -- degraded models take longer to compute

**Math Peek (collapsible):**
$$\text{Acc}(t) \approx \text{Acc}_0 - \lambda \cdot D(P_t \| P_0)$$
where $D(P_t \| P_0)$ is the Jensen-Shannon divergence between the training distribution and the current distribution, and $\lambda$ is a domain-dependent sensitivity constant.

$$\text{PSI} = \sum_i (O_i - E_i) \ln\left(\frac{O_i}{E_i}\right) \quad \text{(Population Stability Index)}$$

---

## 4. Act 2: The Retraining ROI (Design Challenge -- 23 minutes)

### Pedagogical Goal
Students believe that retraining more frequently is always better ("just retrain every day"). The chapter's cost-aware automation principle (Principle 5) shows that retraining decisions must balance compute cost against accuracy improvement: "Retrain if: delta_Accuracy x Value_per_Point > Training_Cost + Deployment_Risk." The automation ROI calculation demonstrates that the optimal retraining interval varies by 4 orders of magnitude across domains. Students must find the retraining frequency that minimizes total cost (accuracy loss + compute cost) for their deployment context.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "An automated retraining pipeline costs 80 engineering hours to build (one-time). Manual retraining takes 4 hours per week. After how many weeks does the automation investment break even?"

Students type a number of weeks. Expected wrong answers: 5--10 weeks (students underestimate manual overhead) or 50+ weeks (students overestimate automation complexity). Actual: **20 weeks** (80/4 = 20). The deeper insight: after 1 year, manual teams spend 100% of time on maintenance while pipeline teams spend 0%.

### The Instrument: Retraining Economics Dashboard

**Primary chart -- Total Cost vs. Retraining Interval:**
- **X-axis:** Retraining interval (1 day to 180 days, log scale)
- **Y-axis:** Total annual cost (USD), 0 to $500K
- **Data series:** Compute cost (BlueLine, decreasing with longer intervals), Accuracy loss cost (RedLine, increasing with longer intervals), Total cost (GreenLine, U-shaped with clear minimum)
- **"Sweet Spot" annotation** at the minimum of the total cost curve

Controls:
- **Training cost per run slider** ($100 to $50,000, step log scale, default $1,000): Higher training cost shifts the optimal interval to the right (retrain less frequently).
- **Accuracy decay rate (lambda) slider** (0.01 to 0.20, step 0.01, default 0.10): Higher decay shifts the optimal interval to the left (retrain more frequently).
- **Revenue per accuracy point slider** ($1,000 to $100,000/year per 1% accuracy, step $1,000, default $10,000): Higher revenue impact shifts the optimal interval to the left.
- **Deployment context toggle** (H100 Cloud / Jetson Orin NX Edge): Cloud: training cost = $1,000/run, fast deployment. Edge: training cost = $5,000/run (must retrain + validate + OTA push), slower deployment cycle.

**Secondary chart -- Automation Crossover:**
- **X-axis:** Weeks (0 to 52)
- **Y-axis:** Cumulative engineering hours
- **Lines:** Manual process (RedLine, linear at 4 hrs/week), Automated pipeline (GreenLine, step at 80 hrs then flat at 0)
- **Crossover annotation** at week 20

**Tertiary chart -- Accuracy Timeline:**
- **X-axis:** Weeks (0 to 52)
- **Y-axis:** Model accuracy (70% to 100%)
- **Lines:** No retraining (RedLine, continuous decay), Periodic retraining (BlueLine, sawtooth pattern with accuracy jumps at each retrain), Threshold-triggered (GreenLine, retrains only when accuracy drops below threshold)

### The Scaling Challenge
**"Find the retraining interval that minimizes total annual cost for each deployment context. Then determine: at what lambda (drift rate) does daily retraining become cheaper than weekly retraining?"**

Students discover:
- H100 Cloud (lambda=0.10, training=$1K): Optimal interval is ~7--14 days
- Jetson Edge (lambda=0.03, training=$5K): Optimal interval is ~60--90 days
- Daily retraining beats weekly only when lambda > 0.15 AND training cost < $500
- Threshold-triggered retraining (retrain when PSI > 0.1) is almost always more cost-effective than fixed-interval retraining

### The Failure State
**Trigger:** `accuracy < 80%` (the model has degraded past the acceptable floor)

**Visual:** Accuracy line turns red; all infrastructure metrics remain green. Banner:
> "**SILENT FAILURE -- Model accuracy below 80%.** All infrastructure metrics are green. The model has been serving confidently wrong predictions for [X] weeks. Estimated revenue loss: $[Y]. Retrain immediately."

**Secondary failure (over-retraining):** When retraining cost > accuracy improvement value:
> "**NEGATIVE ROI -- Retraining costs exceed benefit.** Training cost ($[X]/run) exceeds accuracy improvement value ($[Y]/run). Extend the retraining interval or reduce training cost."

### Structured Reflection
Four-option multiple choice:

> "Monitoring strategy varies by workload archetype. The chapter states that retraining frequency spans 4 orders of magnitude. Which pairing is correct?"
- A) Recommendation system = quarterly; Embedded device = daily
- **B) Recommendation system = daily; Embedded device = quarterly** <-- correct
- C) All systems should retrain weekly as a standard practice
- D) Retraining frequency depends only on training cost, not on drift rate

**Math Peek:**
$$\text{Retrain if: } \Delta\text{Accuracy} \times \text{Value per Point} > \text{Training Cost} + \text{Deployment Risk}$$
$$\text{Break-even (automation)} = \frac{\text{Pipeline build hours}}{\text{Manual hours per week}} = \frac{80}{4} = 20 \text{ weeks}$$
$$T^* \approx \sqrt{\frac{2 \times C_{\text{train}}}{\lambda \times V_{\text{accuracy}}}} \quad \text{(Optimal retraining interval)}$$

---

## 5. Visual Layout Specification

### Act 1: Silent Failure
- **Primary (Left):** Infrastructure health dashboard -- three flat green lines (uptime, latency, error rate) vs. time. Deliberately boring.
- **Primary (Right):** Model accuracy decay curve (RedLine) with PSI bar growing. Shaded zones: green/orange/red. Student's prediction overlaid as horizontal dashed line.

### Act 2: Retraining Economics
- **Primary:** Total cost vs. retraining interval (U-shaped curve). Three components: compute cost (decreasing), accuracy loss (increasing), total (U-shaped with minimum). "Sweet spot" annotation.
- **Secondary:** Automation crossover -- cumulative hours (manual vs. automated) with crossover at 20 weeks.
- **Tertiary:** Accuracy timeline -- three lines (no retrain, periodic, threshold-triggered) showing sawtooth patterns.

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---|---|---|---|---|
| **Cloud (H100)** | H100 (80 GB HBM3) | 80 GB | 700 W | Fast retraining ($1K/run, hours), dynamic user behavior (high drift), but monitoring infrastructure is mature |
| **Edge (Jetson Orin NX)** | Jetson Orin NX (8 GB) | 8 GB | 25 W | Expensive retraining ($5K/run, OTA deployment), stable physical environment (low drift), but monitoring is constrained and rollback is slow |

The two contexts demonstrate that optimal retraining frequency depends on both drift rate and retraining cost. Cloud systems retrain frequently because drift is fast and retraining is cheap. Edge systems retrain infrequently because drift is slow and retraining is expensive.

---

## 7. Design Ledger Output

```json
{
  "chapter": 14,
  "drift_rate_lambda": 0.10,
  "optimal_retrain_interval_days": 14,
  "psi_threshold": 0.10,
  "automation_breakeven_weeks": 20,
  "accuracy_floor_pct": 80,
  "monitoring_type": "statistical_drift | infrastructure_only"
}
```

The `drift_rate_lambda` and `psi_threshold` feed forward to:
- **Lab 15 (Responsible Engineering):** The drift rate determines how quickly fairness metrics degrade across subgroups; the PSI threshold informs when to trigger a fairness audit
- **Lab 16 (Conclusion):** The retraining interval feeds the synthesis cost model

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| 94% accuracy dropping to 88% by week 4 | @sec-ml-operations (line 70) | "Week one: excellent. Week four: accuracy has dropped to 88%...the model has not adapted to a competitor's new promotion" |
| $10 million revenue loss from silent degradation | @sec-ml-operations (line 153) | "The company lost an estimated $10 million in revenue before the issue was discovered during routine quarterly analysis" |
| D_JS > 0.1 causes >5% accuracy drop | @sec-ml-operations, MLOps definition (line 147) | "at D(P_t || P_0) > 0.1 (Jensen-Shannon divergence), observed accuracy drop exceeds 5% in production systems" |
| 10--20% accuracy loss within 6 months | @sec-ml-operations, MLOps definition (line 147) | "A recommendation model deployed without drift monitoring loses roughly 10--20% absolute accuracy within 6 months" |
| Retraining frequency spans 4 orders of magnitude | @sec-ml-operations, Monitoring by Archetype callout (line 297) | "The retraining frequency spans 4 orders of magnitude: daily for recommendation systems to quarterly for embedded devices" |
| Automation: 80 hrs one-time vs. 4 hrs/week manual | @sec-ml-operations, Automation ROI callout (lines 410--411) | "Manual Retrain: 4 engineering hours per week. Pipeline Build: 80 engineering hours (one-time)" |
| Break-even at 20 weeks | @sec-ml-operations, Automation ROI callout (line 415) | "Break-even Point: 20 weeks" |
| Retrain if delta_Accuracy x Value > Training_Cost + Risk | @sec-ml-operations, @eq-retrain-decision (line 262) | "Retrain if: delta_Accuracy x Value per Point > Training Cost + Deployment Risk" |
| 1% skew error on 1M daily queries costs $365K/year | @sec-ml-operations, Skew Economics (line 243) | "annual skew cost reaches USD 365,000" |
| Infrastructure health stays green during model degradation | @sec-ml-operations, Purpose (line 31) | "A model experiencing data drift continues serving predictions with full confidence while accuracy degrades week by week, triggering no alerts" |
