# Mission Plan: lab_12_ops_scale

## 1. Chapter Alignment

- **Chapter:** ML Operations at Scale (`@sec-ops-scale`)
- **Core Invariant:** The **Complexity Explosion** -- operational complexity scales superlinearly with model count: monitoring alerts grow O(N), deployment coordination grows O(N log N), and dependency conflicts grow O(N^2). The total operational load crosses team capacity at approximately 50 models, marking the phase transition from artisanal management to platform-required operations.
- **Central Tension:** Students believe that managing 100 models is 100x the work of managing 1 model -- a linear scaling problem solvable by hiring more people. The chapter's data reveals quadratic dependency growth and a platform ROI threshold: a $2M/year platform breaks even at ~20 models and delivers 5x ROI at 100 models, while delaying platform investment until "at scale" accumulates operational debt that paralyzes the organization. The second surprise is that staged rollouts (canary deployments) are not optional safety theater -- a silent 0.5% CTR drop at 5,000 QPS costs $1,080,000 in 24 hours if undetected.
- **Target Duration:** 35-40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that operational cost scales linearly with model count and that the "right time" to build a platform is when you have hundreds of models. The chapter's complexity curves show that dependency conflicts grow quadratically: at 50 models, total operational load exceeds team capacity even with linear staffing. The platform ROI calculation reveals that a $2M platform breaks even at just 20 models. Students predict the break-even point, discover it is far earlier than expected, and learn that the cost of delay is not just inefficiency but organizational paralysis from accumulated operational debt.

**Act 2 (Design Challenge, 22 min):** Students design a staged rollout strategy for a model update serving 1M requests/hour. They must determine canary duration using the chapter's formula (t_stage = n_samples / (r_requests * p_stage)), then confront the cost of a silent failure: a 0.5% CTR regression undetected for 24 hours costs over $1M. The design challenge requires balancing rollout speed against detection sensitivity, discovering that aggressive rollouts (skipping canary stages) save hours but risk million-dollar losses, while overly cautious rollouts delay value delivery by weeks.

---

## 3. Act 1: The Complexity Explosion (Calibration -- 12 minutes)

### Pedagogical Goal
Students assume that scaling from 1 model to 100 models is a linear problem: hire 100x the ops engineers, or automate each model independently. The chapter shows this intuition fails because dependencies between models grow quadratically. At 50 models, monitoring alerts (~1,000), deployment coordination (~196), and dependency conflicts (~1,250) sum to a total load that exceeds any team's capacity without a platform. The platform ROI calculation crystallizes the economic argument: at $2M/year platform cost and $100K savings per model per year, break-even occurs at just 20 models.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "Your organization manages ML models in production. You are considering building a shared ML platform costing $2M/year. At what model count does the platform investment break even (ROI = 1.0x)?"

Options:
- A) ~5 models -- platforms pay for themselves almost immediately
- **B) ~20 models -- the platform savings per model compound quickly** (correct)
- C) ~100 models -- you need massive scale to justify platform cost
- D) ~500 models -- only hyperscalers benefit from ML platforms

The correct answer is B. The chapter shows ROI = (N_models * $100K savings) / $2M cost, yielding break-even at N=20. Students who pick C or D dramatically overestimate the threshold, which is the most common error -- organizations delay platform investment far too long.

### The Instrument: N-Models Complexity Dashboard

A **multi-line plot** showing three complexity dimensions and total load:

- **X-axis:** Number of models in production (1-500, log scale)
- **Y-axis:** Operational complexity (log scale)
- **Lines:**
  - Monitoring Alerts: O(N) -- 20*N (dotted BlueLine)
  - Deployment Coordination: O(N log N) -- N*log(N) (dashed OrangeLine)
  - Dependency Conflicts: O(N^2) -- 0.5*N^2 (dash-dot RedLine)
  - Total Operational Load: sum of all three (solid black, thick)
- **Threshold line:** Team capacity at 4,000 (horizontal crimson dashed)
- **Shaded regions:** Artisanal (1-10), Growing Pains (10-50), Platform Required (50+)

Controls:
- **Number of models slider:** 1-500 (default: 10)
- **Team size multiplier:** 1x / 2x / 5x (shifts capacity threshold up)
- **Platform toggle:** ON/OFF -- when ON, dependency conflicts drop from O(N^2) to O(N log N) and monitoring alerts are aggregated

**Secondary instrument: Platform ROI Calculator**

- **Input:** N models (linked to main slider), Platform cost ($2M or $5M toggle)
- **Output:** ROI ratio, annual savings, break-even annotation
- ROI = (N * $100K) / Platform_cost
- Break-even highlighted with green marker when ROI >= 1.0

### The Reveal
After interaction:
> "You predicted the platform breaks even at [X] models. The actual break-even is **20 models** for a $2M/year platform. At 100 models, the platform delivers **5x ROI** ($500K savings vs. $100K cost per model). The chapter's data shows that organizations with 50+ models and no platform spend 57% more on operational engineering than those with shared infrastructure."

### Reflection (Structured)
Four-option multiple choice:

> "At 100 models, which operational dimension contributes the most to total complexity?"
- A) Monitoring alerts -- 100 dashboards overwhelm the team
- B) Deployment coordination -- scheduling 100 rollouts is the bottleneck
- **C) Dependency conflicts -- O(N^2) growth means 5,000 potential conflicts dominate** (correct)
- D) All three contribute equally at scale

### Math Peek (collapsible)
$$\text{ROI}_{\text{platform}} = \frac{N_{\text{models}} \times T_{\text{saved}} \times C_{\text{engineer}}}{C_{\text{platform}}}$$
$$\text{Complexity}_{\text{total}} = \underbrace{20N}_{\text{alerts}} + \underbrace{N \log N}_{\text{deployment}} + \underbrace{0.5N^2}_{\text{dependencies}}$$

At N=50: Total = 1,000 + 196 + 1,250 = 2,446 (crosses team capacity of ~4,000 when combined with overhead).

---

## 4. Act 2: The Silent Failure Tax (Design Challenge -- 22 minutes)

### Pedagogical Goal
Students believe that deployment is a binary event (ship or do not ship) and that monitoring will catch regressions quickly. The chapter reveals that model regressions are silent: a 0.5% CTR drop produces no crashes, no errors, no latency spikes -- just lost revenue. At 5,000 QPS and $0.50 per click, 24 hours of undetected regression costs $1,080,000. The canary duration formula (t_stage = n_samples / (r_requests * p_stage)) shows that statistical detection requires a minimum observation window that depends on traffic volume and effect size. Students must design a rollout strategy that balances speed against detection sensitivity.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "A recommendation model serves 1,000,000 requests per hour. You need 10,000 samples to detect a 1% regression with 95% confidence. At a 1% canary traffic split, how many minutes does the canary stage need to run?"

Students type a number (in minutes). Expected wrong answers: 5-15 minutes (students underestimate sample requirements). Actual: **60 minutes** (10,000 / (1,000,000 * 0.01) = 1 hour). At 5% traffic, it drops to 12 minutes.

### The Instrument: Staged Rollout Designer

A **timeline visualization** showing the rollout progression:

- **X-axis:** Time since deployment start (0-48 hours)
- **Y-axis:** Traffic percentage served by new model (0-100%)
- **Step function:** Shows canary progression (1% -> 5% -> 25% -> 50% -> 100%)
- **Duration annotations:** Each stage shows computed minimum duration and actual configured duration

Controls:
- **Request rate:** 100K / 500K / 1M / 5M per hour (default: 1M)
- **Canary start percentage:** 1% / 5% / 10% (default: 5%)
- **Effect size to detect:** 0.1% / 0.5% / 1.0% / 5.0% (default: 1.0%)
- **Confidence level:** 90% / 95% / 99% (default: 95%)
- **Rollout speed:** Conservative (2x minimum duration) / Standard (1.5x) / Aggressive (1x minimum)

**Secondary instrument: Silent Failure Cost Calculator**

- **Inputs:** QPS (linked), CTR baseline (5%), CTR drop (configurable: 0.1%-5.0%), value per click ($0.50), detection time (computed from canary config or manual override)
- **Output:** Total lost revenue, displayed as a running counter that accumulates in real time as the student adjusts detection time
- Formula: Loss = QPS * 3600 * T_detection * (CTR_baseline - CTR_new) * Value_per_click

### The Scaling Challenge
**"Design a rollout strategy that detects a 0.5% CTR regression within 4 hours while deploying to 100% traffic within 8 hours."**

Students must configure the canary stages to satisfy both constraints simultaneously. Key discovery: at 1M req/hr with a 0.5% effect size, detecting the regression at 1% canary requires ~4 hours (40,000 samples needed at 0.5% effect). Increasing to 5% canary reduces detection to ~48 minutes but increases blast radius. The optimal strategy uses a fast 5% canary (1 hour) followed by staged ramp-up.

**Failure state:**
- **Trigger 1 -- Undetected regression:** Detection time exceeds 24 hours at current canary configuration.
- **Visual:** Cost calculator turns RedLine; banner appears.
- **Banner:** "SILENT FAILURE -- At current rollout speed, a 0.5% CTR regression would go undetected for [X] hours, costing $[Y]. The chapter shows this exact scenario: 5,000 QPS * 24h * 0.5% drop * $0.50/click = **$1,080,000 lost**."

- **Trigger 2 -- Rollout too slow:** Total rollout time exceeds 48 hours.
- **Visual:** Timeline extends beyond view; OrangeLine warning.
- **Banner:** "DEPLOYMENT STALL -- Your rollout takes [X] hours. While you wait, the old model continues serving stale predictions. Balance safety against value delivery."

### Structured Reflection
Four-option multiple choice:

> "A 5% canary with 1M req/hr needs 12 minutes to collect 10,000 samples for a 1% effect size. Why would an organization configure a 2-hour canary stage instead of the 12-minute minimum?"

- A) Regulatory requirements mandate minimum observation windows
- B) Statistical tests need time to stabilize -- 12 minutes is a theoretical minimum
- **C) Delayed effects (e.g., user behavior changes, downstream metric shifts) only manifest over longer windows; the 12-minute minimum detects only immediate crashes** (correct)
- D) Network propagation delays mean not all servers receive the canary version simultaneously

### Math Peek (collapsible)
$$t_{\text{stage}} = \frac{n_{\text{samples needed}}}{r_{\text{requests}} \times p_{\text{stage}}}$$
$$\text{Loss}_{\text{silent}} = \text{QPS} \times 3600 \times T_{\text{detection}} \times (\text{CTR}_{\text{base}} - \text{CTR}_{\text{new}}) \times \text{Value}_{\text{click}}$$
$$\text{Risk Reduction}_{\text{canary}} = \frac{100\%}{p_{\text{canary}}} = \frac{100\%}{5\%} = 20\times$$

---

## 5. Visual Layout Specification

### Act 1: Complexity Explosion
- **Primary:** Multi-line log-log plot. X: model count (1-500). Y: operational complexity (log scale). Four lines (alerts, coordination, dependencies, total) with capacity threshold and shaded regions.
  - Failure state: total line turns RedLine when crossing capacity threshold
- **Secondary:** Platform ROI bar -- single horizontal bar showing ROI ratio with break-even marker at 1.0x, colored green above break-even and red below.

### Act 2: Silent Failure Tax
- **Primary:** Timeline step function. X: hours since deployment (0-48). Y: traffic percentage (0-100%). Each stage annotated with duration and sample count.
  - Failure state: timeline turns RedLine if detection window exceeds 24h
- **Secondary:** Revenue loss counter -- real-time accumulating dollar figure. Starts at $0, grows as detection time increases.
  - Turns RedLine above $100K; shows $1.08M reference from chapter
- **Tertiary:** Canary duration calculator -- shows minimum stage duration for current settings.

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---------|--------|-----|--------------|----------------|
| **Kubernetes (Managed Orchestration)** | GPU cluster, auto-scaling | Elastic | Elastic | Platform automates rollouts, canary stages, and rollback; dependency tracking is centralized; monitoring is aggregated. The risk: over-reliance on automation masks silent failures. |
| **Bare Metal (Manual Scheduling)** | Fixed GPU allocation per team | Fixed quota | Fixed | No automated rollback; each team manages its own deployment scripts; dependency tracking is manual spreadsheets. The risk: at 50+ models, manual coordination collapses. |

The two contexts demonstrate why platform investment has a threshold: on Kubernetes, platform costs are higher ($5M/year) but scale sublinearly; on bare metal, initial costs are lower but operational toil grows quadratically with model count, crossing the Kubernetes cost at approximately 30 models.

---

## 7. Design Ledger Output

```json
{
  "chapter": 12,
  "n_models_managed": 50,
  "platform_roi": 1.25,
  "canary_pct": 5,
  "canary_duration_hours": 2,
  "detection_sensitivity_pct": 1.0,
  "rollout_total_hours": 4,
  "silent_failure_cost_24h": 1080000
}
```

The `canary_pct` and `detection_sensitivity_pct` fields feed forward to:
- **Lab 13 (Security & Privacy):** The canary deployment pattern is reused for privacy-preserving model updates -- differential privacy noise must not mask the regression signal that canary monitoring relies on.
- **Lab 14 (Robust AI):** Detection sensitivity determines whether adversarial drift is caught during staged rollout or after full deployment.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| O(N) monitoring alerts, O(N log N) deployment, O(N^2) dependencies | @fig-n-models-complexity, line 172 | "Monitoring alerts grow linearly...dependency conflicts grow quadratically...total operational load crosses team capacity around 50 models" |
| Team capacity threshold at ~50 models | @sec-ml-operations-scale-nmodels-problem-fcff, line 172 | "The total operational load crosses team capacity around 50 models" |
| Platform ROI formula | @eq-platform-roi, line 257 | "ROI_platform = (N_models * T_saved * C_engineer) / C_platform" |
| $2M platform breaks even at ~20 models | @fig-platform-roi-threshold, line 396 | "A $2M/year platform breaks even at approximately 20 models" |
| $5M platform breaks even at ~50 models | @fig-platform-roi-threshold, line 396 | "a more expensive $5M/year enterprise platform requires roughly 50 models" |
| 100 models: 5x ROI for $2M platform | @fig-platform-roi-threshold, line 396 | "At 100 models, the $2M platform delivers 5x return on investment" |
| 57% savings from shared infrastructure | PlatformEconomics class, line 373 | "savings_pct > 50" (check guard); "Annual savings...a 56% reduction" |
| Multi-tenant sharing: 70% idle -> 30% idle = 57% savings | MultiTenantEfficiency class, lines 131-145 | "avg_util_dedicated = 0.30...avg_util_shared = 0.70...savings_pct ≈ 57%" |
| Canary: 5% traffic = 20x risk reduction | DeploymentSafety class, lines 1333-1345 | "risk_reduction = 1.0 / canary_pct = 20.0" |
| Canary duration formula: t = n_samples / (r * p) | @eq-canary-duration, line 1384 | "t_stage = n_samples_needed / (r_requests * p_stage)" |
| 1% canary at 1M req/hr: 1 hour for 10,000 samples | Worked example, line 1393 | "t_1% = 10,000 / (1,000,000 * 0.01) = 1 hour" |
| Silent failure: 0.5% CTR drop at 5,000 QPS costs $1,080,000/day | SilentFailure class, lines 1425-1442 | "total_loss ≈ $1,080,000" |
| Staged rollout: 1% -> 5% -> 25% -> 50% -> 100% in ~4 hours | @sec-ml-operations-scale-staged-rollout-strategies-2d1f, lines 1398-1406 | "Total rollout: approximately 4 hours for a confident deployment" |
| Ensemble deployment: shadow (24-48h), canary (4-8h), staged (24-72h), soak (7-14d) | @tbl-ops-scale-ensemble-deploy, line 1008 | "shadow deployment (24-48 hours), canary (1% traffic for 4-8 hours)" |
