# Mission Plan: lab_03_ml_workflow

## 1. Chapter Alignment

- **Chapter:** ML Workflow (`@sec-ml-workflow`)
- **Core Invariant:** The **Constraint Propagation Principle** ($2^{N-1}$ cost escalation) — a constraint discovered at stage $N$ of the pipeline costs $2^{N-1}$ times more to fix than if discovered at stage 1. A model requirement discovered at deployment (stage 6) costs 32× more to fix than discovering it during requirements (stage 1). This is the ML analogue of Boehm's Law for software defects.
- **Central Tension:** Students believe that ML project success is determined by model accuracy and that more iteration always produces better results. The chapter's central case study demolishes both: a team achieves 95% accuracy at Day 90, then 96% at Day 120, then discovers at Day 150 that the deployment target (512 MB tablet) cannot hold the model (4 GB). Five months of work is discarded. The bottleneck was never the model — it was the absent constraint propagation gate. More iteration on the wrong objective produces more waste, not better outcomes.
- **Target Duration:** 35–40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that the majority of ML project time is spent on model development (architecture search, hyperparameter tuning, training). The chapter's data shows 60–80% of time goes to data activities; model development is 10–20%. This act uses the Rural Clinic case study to show how wrong assumptions about time allocation compound into project failure — and how the $2^{N-1}$ law quantifies exactly why late-stage discoveries are so expensive.

**Act 2 (Design Challenge, 22 min):** Students run the iteration velocity race from the chapter: a large, high-accuracy model (95% start, +0.15% per iteration, 1-week cycle) vs. a small, lower-accuracy model (90% start, +0.1% per iteration, 1-hour cycle). The chapter's calculation shows the small model overtakes the large one within the 26-week project window. Students must find the crossover week, then configure monitoring thresholds that catch the Rural Clinic constraint failure before Day 150.

---

## 3. Act 1: The Constraint Tax (Calibration — 12 minutes)

### Pedagogical Goal
Students dramatically underestimate the cost of late constraint discovery. The chapter's Rural Clinic case study is the pedagogical instrument: a team spends 150 days achieving 96% accuracy on diabetic retinopathy detection, then on Day 151 discovers the model requires 4 GB of memory — and the deployment target is a 512 MB tablet. All 150 days of work is discarded. The $2^{N-1}$ law quantifies why: a deployment constraint discovered at stage 6 costs 32× more than discovering it at stage 1 (the requirements phase). This act forces students to predict the cost multiplier before showing them the case study timeline.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "An ML pipeline has 6 stages: Requirements, Data, Modeling, Evaluation, Deployment, Monitoring. The Rural Clinic team discovers a memory constraint at Stage 5 (Deployment) — 150 days into the project. Compared to discovering it at Stage 1 (Requirements), how much more expensive is it to fix?"

Options:
- A) About 5× more expensive — roughly proportional to how far through the project you are
- B) About 10× more expensive — late changes always cost more but double-digit is the cap
- **C) About 16× more expensive — each stage compounds the rework required** ← correct
- D) About 100× more expensive — exponential blowup dominates by deployment

The answer is 16× because each stage's output becomes the input to the next: discarding at Stage 5 requires redoing Stage 5 and Stage 4 at minimum, plus reconciling Stage 3 decisions. The compounding factor is approximately 2× per stage crossed. Students verify this in the instrument — the formula is revealed *after* prediction, not given in the question.

### The Instrument: Constraint Propagation Timeline

A **lifecycle bar chart** showing the Rural Clinic project timeline:

- **X-axis:** Days (0 → 155)
- **Y-axis:** Cumulative Engineering Cost (person-days)
- **Bars:** One bar per stage (Requirements, Data Collection, Data Labeling, Model Development, Evaluation, Deployment Attempt)

Controls:
- **Constraint Discovery Stage slider** (Stage 1 → Stage 6): Students drag the discovery point earlier in the timeline. At each stage, the lab shows:
  - How much work must be redone
  - The cost multiplier ($2^{N-1}$)
  - The redone work highlighted in red

At Stage 6 (Deployment): "Model requires 4 GB. Tablet has 512 MB. Discard 150 days."
At Stage 5 (Evaluation): "Model too large. Discard evaluation + modeling = 60 days."
At Stage 1 (Requirements): "Tablet budget = 512 MB. Constraint recorded. Model development begins within budget = 0 days wasted."

**Secondary:** A **time allocation breakdown** pie chart showing the actual distribution from the chapter:
- Data activities (collection + cleaning + labeling): 60–80%
- Model development: 10–20%
- Deployment + monitoring: 10–20%

Students drag a slider for "What % of time do you think is spent on data?" and the pie animates to the actual answer.

### The Reveal
After interaction:
> "You predicted [X]× cost multiplier. The actual value for Stage 5 discovery is **16×** ($2^{5-1}$). The Rural Clinic team spent 150 days at an estimated [cost] before discovering the 512 MB constraint. A 1-day requirements checklist at Stage 1 would have cost 1× — the same work for 1/16th of the price."

Surface the time allocation data:
> "Students consistently overestimate time spent on modeling (typically guessing 40–60%). The chapter's data shows modeling is 10–20%. The dominant cost is data: 60–80% of total project time."

### Reflection (Structured)
Four-option multiple choice:

> "The Rural Clinic team achieved 96% accuracy. Why was the project a failure?"
- A) 96% accuracy is insufficient for medical AI — they should have aimed for 99%
- B) The team used the wrong model architecture and needed to retrain
- **C) The constraint (512 MB memory limit) was never recorded at Stage 1, so the entire modeling effort was aimed at the wrong target** ← correct
- D) The data labeling was incomplete, causing distribution shift at deployment

**Math Peek (collapsible):**
$$\text{Cost}(N) \approx 2^{N-1} \quad N = \text{stage of constraint discovery (pedagogical model)}$$
$$\text{Stage 1: } 2^0 = 1\times \quad \text{Stage 3: } 2^2 = 4\times \quad \text{Stage 6: } 2^5 = 32\times$$

> **Note for students:** The $2^{N-1}$ formula is a useful approximation, not a physical law. Real multipliers range from 3× to 1,000× depending on how deeply the constraint is embedded in prior decisions. The principle is empirically grounded — Boehm's Law in software engineering shows post-deployment defects cost 100× more to fix than requirements-phase defects (chapter footnote, line 260) — but the exact multiplier varies by system modularity.

---

## 4. Act 2: The Iteration Velocity Race (Design Challenge — 22 minutes)

### Pedagogical Goal
Students believe that starting with the highest-accuracy model is the correct strategy — "get it right the first time." The chapter's iteration calculation shows that a small model starting 5 percentage points behind but iterating 168× faster (1 hour vs. 1 week per cycle) overtakes the large model within the 26-week project window, finishing with comparable or better final accuracy because it accumulates more learning cycles. Students must find the crossover week and then design monitoring gates that would have caught the Rural Clinic constraint failure before Day 150.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "A large model starts at 95% accuracy and gains 0.15% per iteration, with a 1-week training cycle. A small model starts at 90% accuracy and gains 0.1% per iteration, with a 1-hour training cycle. In a 26-week project window, how many iterations does the small model complete compared to the large model?"

Students type a ratio (small / large). Expected wrong answers: 5–20× (students underestimate hourly cycle advantage). Actual: 26 weeks × 168 hours/week = 4,368 hours / 1 hour = **4,368 small-model iterations** vs. **26 large-model iterations** = **168× more iterations**.

### The Instrument: Iteration Velocity Comparator

A **dual-line accuracy plot** showing both models across 26 weeks:

- **X-axis:** Project week (0 → 26)
- **Y-axis:** Model accuracy (85% → 100%, ceiling at 99%)
- **Large model line** (solid, BlueLine): Starts at 95%, **saturating curve** toward 99% ceiling. Gain per iteration decays as: $\Delta a_n = \Delta a_0 \times r^n$ where $r = 0.95$ (5% decay per iteration). At 26 iterations total, this produces a realistic S-curve, not a linear ramp.
- **Small model line** (dashed, GreenLine): Starts at 90%, same saturating formula but with 100× more iterations. **Diminishing returns kicks in after ~100 useful iterations on the same data distribution** — gain/iteration drops toward zero until new data is added. This is a hard implementation requirement: the small model line must flatten, not keep climbing.

Controls:
- **Large model cycle time**: 24 hrs / 1 week / 2 weeks
- **Small model cycle time**: 1 hr / 4 hrs / 24 hrs
- **Decay rate** ($r$): 0.90 / 0.95 / 0.99 — controls how quickly gains saturate
- **Project window**: 13 / 26 / 52 weeks

A **crossover annotation** appears automatically at the week where the small model's accuracy exceeds the large model's. Students observe that the crossover week shifts based on the decay rate — if gains decay fast (r=0.90), the small model's advantage depends more on how quickly new data can be injected than on raw iteration count.

**Secondary instrument: Constraint Gate Designer**
A simplified stage checklist with toggle gates:

| Stage | Gate | Status |
|---|---|---|
| Requirements | Memory budget recorded? | ✓ / ✗ |
| Requirements | Latency budget recorded? | ✓ / ✗ |
| Data | Schema validation active? | ✓ / ✗ |
| Modeling | Memory profiling at batch=1? | ✓ / ✗ |
| Evaluation | On-device test with production hardware? | ✓ / ✗ |
| Deployment | Staged rollout with monitoring? | ✓ / ✗ |

When the "Memory budget recorded?" gate is toggled OFF:
- The timeline shows the Rural Clinic failure at Day 150
- Cost multiplier: 16× (Stage 5 discovery)

When all gates are toggled ON:
- The timeline catches the constraint at Stage 1, Day 1
- Cost multiplier: 1×
- A "Zero Wasted Days" badge appears

### The Scaling Challenge
**"Configure the iteration strategy that achieves ≥ 97% accuracy by Week 20, using either model."**

Students must adjust the sliders to find a configuration where either model reaches 97% within 20 weeks. Key discovery: with the large model (1-week cycle), reaching 97% from 95% requires ≥ 14 iterations → 14 weeks; reaching 97% from 90% with the small model requires the gain rate to be ≥ 0.042% per iteration — achievable with 1-hour cycles, which accumulate 2,352 iterations by week 20.

The catch: the small model has a **diminishing returns cap** — after ~100 iterations on the same dataset, gain per iteration drops toward zero. Students must find the data collection refill point.

**Failure state:** When the large model is selected and project deadline is Week 20:
> "🟠 **Iteration Budget Exhausted.** At 1-week cycles, you have 20 iterations in 20 weeks. At +0.15% per iteration from 95% baseline, maximum achievable accuracy = 98%. But with 0 constraint gates active, there is a [X]% probability the model fails the deployment check on Week 21."

### Structured Reflection
Sentence completion:

> "The small model (90% start, 1-hour cycle) outperforms the large model (95% start, 1-week cycle) by Week [___] because ___."

Expected answer: approximately Week 8–12, because "it accumulates 1,000+ iterations in the same time the large model completes 8–12, and each iteration provides compounding learning signal even at smaller gain-per-step."

Then select the monitoring threshold that catches the Rural Clinic failure at Stage 1:

> "Which gate, if active on Day 1, would have prevented the 150-day Rural Clinic failure?"
- A) Schema validation on training data
- B) P99 latency monitoring in production
- **C) Hardware memory budget recorded in Stage 1 requirements** ← correct
- D) Accuracy disaggregation by demographic subgroup

**Math Peek:**
$$\text{Iterations}_{small} = \frac{T_{project}}{t_{cycle}} = \frac{26 \times 168 \text{ hrs}}{1 \text{ hr}} = 4{,}368 \qquad \text{Iterations}_{large} = 26$$
$$\text{Cost}(N) = 2^{N-1} \quad \text{(Constraint Propagation Principle)}$$

---

## 5. Visual Layout Specification

### Act 1: Constraint Tax
- **Primary:** Lifecycle bar chart — each bar = one stage, colored by cost category. Red overlay appears for redone work when constraint discovery stage is slid right.
- **Secondary:** Time allocation pie chart — interactive slider for student's guess vs. actual chapter data (60–80% data activities).
- **Constraint discovery slider:** Dragging shows cost multiplier badge ($2^{N-1}$ live update) and highlights wasted stages in red.

### Act 2: Iteration Velocity Race
- **Primary:** Dual-line accuracy plot (large vs. small model) with crossover annotation and Week 20 deadline line.
- **Secondary:** Constraint Gate checklist — toggle gates, watch Rural Clinic failure appear/disappear on the timeline.
- **Tertiary:** Diminishing returns indicator — when small model accumulates >100 iterations without new data, gain/iteration drops to near-zero.
- **Failure state:** OrangeLine banner when large model selected + Week 20 deadline + no constraint gates active.

---

## 6. Deployment Context Definitions

| Context | Model Type | Cycle Time | Key Constraint |
|---|---|---|---|
| **Training Node** | Large model (4 GB, 95% start) | 1 week per iteration | Memory budget check must be Stage 1; each wasted iteration = $50K+ compute cost |
| **Edge Inference** | Small model (512 MB, 90% start) | 1 hour per iteration | Can iterate 168× faster; memory constraint is trivially satisfied; diminishing returns hit faster |

The two contexts demonstrate that "the best model" is deployment-context-dependent: the large model is optimal only if the constraint is satisfied. The small model is optimal when iteration velocity matters more than single-step quality gains — which is almost always true when the project window is the binding constraint.

---

## 7. Design Ledger Output

```json
{
  "chapter": 3,
  "model_size_chosen": "large | small",
  "iteration_cycle_hours": 1,
  "crossover_week": 10,
  "constraint_gates_active": ["memory_budget", "schema_validation", "on_device_test"],
  "constraint_discovery_stage": 1,
  "cost_multiplier": 1
}
```

The `iteration_cycle_hours` and `model_size_chosen` fields feed forward to:
- **Lab 05 (NN Computation):** The model size choice affects the memory wall visualization baseline
- **Lab 08 (Training):** The iteration velocity calculation informs the MFU optimization priority discussion

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| $2^{N-1}$ Constraint Propagation | ml_workflow.qmd, line 75 | "the Constraint Propagation Principle ($2^{N-1}$ cost escalation)" |
| 100× cost multiplier for post-deployment defects | ml_workflow.qmd, line 260 (footnote) | "defects found post-deployment cost up to 100× more to fix than those caught during requirements" |
| Rural Clinic: 95% accuracy at Day 90 | ml_workflow.qmd, line 86 | "Day 90: 95% accuracy on the test set" |
| Rural Clinic: 4 GB model, 512 MB tablet | ml_workflow.qmd, line 86 | "model requires 4 GB of memory…tablets in mobile clinics with 512 MB available" |
| Rural Clinic: 5 months discarded | ml_workflow.qmd, line 86 | "five months of work is discarded" |
| 60–80% time on data activities | ml_workflow.qmd, line 203 | "Data-related activities…consume 60–80% of total project time" |
| 10–20% time on model development | ml_workflow.qmd, line 203 | "Model development and training…typically represents only 10–20% of effort" |
| 4–8 complete iteration cycles required | ml_workflow.qmd, line 253 | "Production-ready ML systems typically require 4–8 complete iteration cycles" |
| 60% of iterations driven by data quality | ml_workflow.qmd, line 253 | "Data quality issues…drive approximately 60% of iterations" |
| Large model: 95% start, +0.15%/iter, 1-week cycle | ml_workflow.qmd, lines 312–314 | "large_start_acc = 95.0; large_gain_per_iter = 0.15; large_cycle_time_hours = 168" |
| Small model: 90% start, +0.1%/iter, 1-hour cycle | ml_workflow.qmd, lines 317–319 | "small_start_acc = 90.0; small_gain_per_iter = 0.1; small_cycle_time_hours = 1" |
| 26-week project window | ml_workflow.qmd, line 308 | "weeks_total = 26" |
| Accuracy ceiling at 99% | ml_workflow.qmd, lines 325, 329 | "min(final_acc, 99.0)" |
| 10× advantage: fast iterating model overtakes slow | ml_workflow.qmd, line 1415 | "A model starting 5% behind but iterating 10× faster overtakes the leader" |
