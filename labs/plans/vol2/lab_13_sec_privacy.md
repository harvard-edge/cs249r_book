# Mission Plan: lab_13_sec_privacy

## 1. Chapter Alignment

- **Chapter:** Security & Privacy (`@sec-security-privacy`)
- **Core Invariant:** The **Privacy-Utility Scaling Law** -- differential privacy noise scale is Sensitivity/epsilon, and the per-person error is proportional to 1/N. For small datasets (N=100), DP at epsilon=1 introduces $2,000 error per person; for large datasets (N=1,000), the same guarantee costs only $200 per person. Privacy is feasible only at scale, and the privacy budget (epsilon) is finite and non-renewable across queries.
- **Central Tension:** Students believe that privacy is a binary switch ("encrypt it or do not") and that adding noise to protect privacy has a small, constant cost. The chapter demolishes both: differential privacy kills utility for small N because the noise magnitude is independent of dataset size while the per-record error scales as 1/N. The second surprise is that secure multi-tenant GPU partitioning (MIG) costs exactly 15% throughput -- a fixed tax on every token served through an isolated enclave, not a negligible overhead.
- **Target Duration:** 35-40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that differential privacy adds a small, manageable amount of noise to model outputs. The chapter's worked example shows that for a salary query with sensitivity $200,000 and epsilon=1, the Laplace noise scale is $200,000. For N=1,000 employees, this produces a tolerable $200 per-person error. For N=100, the same mechanism produces $2,000 per-person error -- rendering the result useless. Students predict the error for a given N and epsilon, then discover that privacy utility scales with dataset size in a way that makes DP infeasible for small organizations or rare subpopulations.

**Act 2 (Design Challenge, 22 min):** Students must design a privacy architecture for a multi-tenant ML serving platform. The chapter shows that hardware isolation (MIG partitioning) costs 15% throughput, noise injection to prevent model extraction requires sigma calibration, and the total privacy budget epsilon depletes with each query. Students must configure isolation level, noise sigma, and query budget to simultaneously satisfy a privacy officer (epsilon < 1.0) and a product manager (throughput > 800 tokens/sec from a baseline of 1,000). The design challenge reveals that full isolation + strong DP can reduce effective throughput by 30-40%, forcing explicit prioritization.

---

## 3. Act 1: The Privacy Scaling Wall (Calibration -- 12 minutes)

### Pedagogical Goal
Students treat privacy as a boolean property: the data is either protected or not. The chapter introduces differential privacy as a quantitative framework where the cost of privacy is measurable in utility loss, and that cost depends critically on the dataset size N. For large N, the noise is absorbed by averaging; for small N, the noise overwhelms the signal. This act forces students to confront the 1/N scaling of DP error and understand why "privacy at scale" is not marketing language but a mathematical necessity.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "You are computing the average salary of employees with differential privacy (epsilon=1, salary range $0-$200,000). For N=1,000 employees, the per-person error is $200. For N=100 employees, what is the per-person error?"

Options:
- A) ~$200 -- the error is independent of dataset size
- B) ~$500 -- error grows modestly as N decreases
- **C) ~$2,000 -- error scales as 1/N, making DP 10x worse for small datasets** (correct)
- D) ~$20,000 -- the noise completely overwhelms the signal

The correct answer is C. Noise scale = Sensitivity/epsilon = $200,000. Per-person error = Noise_scale/N. For N=100: $200,000/100 = $2,000. Students who pick A have the most common misconception: that DP noise is independent of dataset size.

### The Instrument: Privacy-Utility Scaling Curve

A **line plot with shaded utility zones** showing per-person error vs. dataset size:

- **X-axis:** Dataset size N (10 to 100,000, log scale)
- **Y-axis:** Per-person error in dollars ($1 to $100,000, log scale)
- **Lines:** One curve per epsilon value
  - epsilon=0.1 (strongest privacy, RedLine)
  - epsilon=1.0 (standard, OrangeLine)
  - epsilon=8.0 (weak privacy, BlueLine)
- **Shaded zones:**
  - Green zone: error < $500 (utility preserved)
  - Yellow zone: $500 < error < $5,000 (marginal utility)
  - Red zone: error > $5,000 (utility destroyed)

Controls:
- **Epsilon slider:** 0.1 / 0.5 / 1.0 / 2.0 / 4.0 / 8.0 (default: 1.0)
- **Sensitivity (salary range):** $50K / $100K / $200K / $500K (default: $200K)
- **Dataset size slider:** 10 to 100,000 (default: 1,000)

**Key interaction:** As students decrease N below 100 at epsilon=1.0, the curve enters the red "utility destroyed" zone. They can "rescue" utility by either increasing epsilon (weakening privacy) or increasing N (more data). The trade-off is visceral: there is no free path to both strong privacy and good utility for small datasets.

### The Reveal
After interaction:
> "You predicted $[X] per-person error for N=100 at epsilon=1. The actual error is **$2,000** -- 10x worse than N=1,000. The chapter concludes: 'Differential Privacy kills utility for small N. It only works at scale where 1/N dampens the noise.' This is why federated learning across millions of devices is the privacy-preserving architecture of choice: only at fleet scale does the noise become tolerable."

### Reflection (Structured)
Four-option multiple choice:

> "A hospital with 50 patients wants to publish average treatment costs with epsilon=1 privacy. The per-patient error would be $4,000 on a $200K range. What is the correct engineering response?"

- A) Use a smaller epsilon (0.1) to provide stronger privacy, accepting even larger error
- B) Publish the data without DP -- the dataset is too small for meaningful privacy guarantees
- **C) Aggregate with other hospitals to increase N before applying DP -- privacy at scale is the only viable path** (correct)
- D) Use deterministic rounding instead of DP to avoid noise entirely

### Math Peek (collapsible)
$$\text{Noise Scale} = b = \frac{S}{\epsilon} = \frac{\$200{,}000}{1.0} = \$200{,}000$$
$$\text{Error per person} = \frac{b}{N} = \frac{\$200{,}000}{N}$$
$$N=1{,}000 \Rightarrow \$200 \quad N=100 \Rightarrow \$2{,}000 \quad N=50 \Rightarrow \$4{,}000$$

---

## 4. Act 2: The Defense Tax (Design Challenge -- 22 minutes)

### Pedagogical Goal
Students believe that security and privacy protections are lightweight additions to a serving system -- "just turn on encryption." The chapter shows that every defense has a quantifiable throughput cost: MIG partitioning for tenant isolation costs 15% throughput (1,000 tokens/sec drops to 850), noise injection for extraction defense costs additional latency per query, and the privacy budget epsilon depletes with each query. Students must design a serving architecture that satisfies a privacy officer (epsilon < 1.0 per user per day) and a product manager (throughput > 800 tokens/sec) simultaneously, discovering that these constraints are fundamentally in tension.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "An H100 GPU serves 1,000 tokens/sec in a dedicated (single-tenant) configuration. You enable MIG partitioning for secure multi-tenant isolation. What throughput does the isolated partition achieve?"

Students type a number (tokens/sec). Expected wrong answers: 950-990 (students assume isolation is nearly free). Actual: **850 tokens/sec** -- a 15% isolation tax, per the chapter's calculation.

### The Instrument: Defense Architecture Designer

A **waterfall chart** showing cumulative throughput cost of each defense layer:

- **X-axis:** Defense layers (left to right): Baseline -> MIG Isolation -> Noise Injection -> Output Rounding -> Rate Limiting
- **Y-axis:** Throughput (tokens/sec, 0-1,100)
- **Bars:** Each defense layer subtracts from baseline throughput
  - Baseline: 1,000 tokens/sec (BlueLine)
  - MIG Isolation: -150 tokens/sec (15% tax, OrangeLine)
  - Noise Injection (sigma=0.05): -30 tokens/sec (~3% overhead for noise computation)
  - Output Rounding: -10 tokens/sec (~1% overhead)
  - Rate Limiting: 0 tokens/sec direct cost (but limits total queries/day)
- **Threshold line:** Product manager requirement at 800 tokens/sec (GreenLine dashed)

Controls:
- **MIG toggle:** ON/OFF (default: OFF) -- enables/disables 15% isolation tax
- **Noise sigma:** 0.0 / 0.01 / 0.03 / 0.05 / 0.10 (default: 0.0)
- **Privacy budget (epsilon per day):** 0.1 / 0.5 / 1.0 / 4.0 / 8.0 (default: 8.0)
- **Queries per user per day:** 10 / 50 / 100 / 1,000 (default: 100)

**Key interaction:** When students enable MIG + noise injection (sigma=0.05), throughput drops to ~820 tokens/sec -- barely above the 800 threshold. If they then tighten epsilon to 0.5, the noise per query increases (sigma scales with 1/epsilon), dropping throughput further below 800. The privacy officer's requirement and the product manager's requirement cannot both be fully satisfied at high query volumes.

**Secondary instrument: Privacy Budget Depletion Gauge**

A **circular gauge** showing remaining privacy budget:

- Starts at epsilon_total for the day
- Each query consumes epsilon_per_query = epsilon_total / max_queries
- Gauge depletes in real-time as students adjust queries/day
- When budget is exhausted, all subsequent queries are rejected (availability drops to 0)

### The Scaling Challenge
**"Configure defenses that satisfy BOTH: (a) epsilon < 1.0 per user per day, and (b) throughput > 800 tokens/sec. The user makes 100 queries per day."**

Students must discover that with MIG ON and epsilon=1.0, each query gets epsilon_per_query = 0.01. At sigma = Sensitivity/epsilon_per_query, the noise is enormous -- but output rounding at 0.01 precision limits actual noise impact. The solution requires model sharding: keep only sensitive layers in MIG (partial isolation at ~8% tax instead of 15%), use aggressive output rounding to reduce noise computation cost, and accept that 100 queries/day at epsilon=1.0 means each query consumes only 0.01 of the budget.

### The Failure State
**Trigger 1 -- Throughput violation:** Effective throughput drops below 800 tokens/sec.
**Visual:** Waterfall bars turn RedLine; throughput number turns red.
**Banner:** "THROUGHPUT SLA VIOLATED -- Effective throughput: [X] tokens/sec. Product requirement: 800 tokens/sec. Current defense stack costs [Y]% overhead. Consider partial isolation (sensitive layers only) to reduce the MIG tax from 15% to ~8%."

**Trigger 2 -- Privacy budget exhausted:** Queries per day exceed epsilon / epsilon_per_query.
**Visual:** Privacy gauge hits zero; rejection rate indicator appears.
**Banner:** "PRIVACY BUDGET EXHAUSTED -- At [X] queries/day with epsilon=[Y], budget depletes after [Z] queries. Remaining queries are rejected (0 tokens/sec effective). Either increase epsilon (weaker privacy) or reduce query volume."

### Structured Reflection
Four-option multiple choice:

> "The chapter states that 'Privacy is a budget, not a switch.' What does this mean for a production ML API?"

- A) Once you enable DP, all queries are equally private regardless of volume
- B) Privacy can be toggled on or off per query based on user preference
- **C) Each query consumes a finite portion of the privacy budget; after enough queries, the system must stop responding to prevent cumulative information leakage** (correct)
- D) Privacy cost is paid once during model training and does not affect serving

### Math Peek (collapsible)
$$\text{Noise Scale: } b = \frac{S}{\epsilon} \qquad \text{Per-person Error: } \frac{b}{N}$$
$$\text{Isolation Tax: } T_{\text{secure}} = T_{\text{peak}} \times (1 - 0.15) = 1{,}000 \times 0.85 = 850 \text{ tokens/sec}$$
$$\text{Budget per query: } \epsilon_q = \frac{\epsilon_{\text{daily}}}{Q_{\text{daily}}} = \frac{1.0}{100} = 0.01$$

---

## 5. Visual Layout Specification

### Act 1: Privacy Scaling Wall
- **Primary:** Log-log line plot. X: dataset size N (10-100,000). Y: per-person error ($1-$100,000). Three curves for epsilon values (0.1, 1.0, 8.0). Shaded utility zones (green/yellow/red).
  - Crossover annotation when curve enters red zone
- **Secondary:** Epsilon-N trade-off heatmap -- 6x6 grid of (epsilon, N) combinations colored by utility zone.

### Act 2: Defense Tax
- **Primary:** Waterfall chart. X: defense layers (Baseline through Rate Limiting). Y: throughput (0-1,100 tokens/sec). Each bar subtracts from running total. Threshold line at 800 tokens/sec.
  - Failure state: bars turn RedLine when cumulative throughput drops below 800
- **Secondary:** Privacy budget depletion gauge -- circular dial showing remaining epsilon for the day.
  - Turns RedLine at 10% remaining budget
- **Tertiary:** Defense cost summary table -- shows each defense's overhead percentage and cumulative impact.

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---------|--------|-----|--------------|----------------|
| **On-Premise (Full Control)** | H100 cluster, MIG-enabled | 80 GB per GPU | 700 W | Full hardware isolation available; MIG partitioning costs 15% throughput; all data stays on-site; privacy budget managed locally |
| **Cloud (Shared Infrastructure)** | Multi-tenant GPU instances, no hardware isolation | Shared | Shared | No MIG available; must rely on software isolation (containers) + noise injection; higher extraction risk from co-tenants; privacy budget must account for cloud provider access |

The two contexts demonstrate the defense-depth trade-off: on-premise gets hardware isolation (15% tax) + software defenses, while cloud-only gets software defenses (less secure but no throughput tax from MIG). The chapter's isolation tax calculation (1,000 -> 850 tokens/sec) applies only to on-premise MIG; cloud deployments face a different risk profile where extraction attacks are more feasible because isolation is software-only.

---

## 7. Design Ledger Output

```json
{
  "chapter": 13,
  "epsilon_daily": 1.0,
  "noise_sigma": 0.05,
  "mig_enabled": true,
  "throughput_effective": 820,
  "isolation_tax_pct": 15,
  "queries_per_day": 100,
  "privacy_budget_per_query": 0.01
}
```

The `epsilon_daily` and `isolation_tax_pct` fields feed forward to:
- **Lab 14 (Robust AI):** Privacy noise interacts with adversarial robustness -- the noise injected for privacy can inadvertently serve as a defense against extraction attacks but may mask legitimate adversarial examples.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| DP noise scale = Sensitivity / epsilon | DPCostAnalysis class, line 167 | "noise_scale = sensitivity / epsilon" |
| Per-person error = noise_scale / N | DPCostAnalysis class, line 168 | "error_per_person = noise_scale / n_employees" |
| N=1,000: $200 error per person at epsilon=1 | DPCostAnalysis class, line 172 | "check(error_per_person == 200)" |
| N=100: $2,000 error per person at epsilon=1 | DPCostAnalysis class, line 169 | "error_small = noise_scale / n_small" (n_small=100) |
| "DP kills utility for small N" | Notebook callout, line 193 | "Differential Privacy kills utility for small N. It only works at scale where 1/N dampens the noise." |
| MIG isolation: 15% throughput overhead | MultiTenantIsolation class, lines 219-221 | "overhead_pct = 15%; peak=1,000, secure=850 tokens/sec" |
| "Security is a Capacity Drain" -- 15% GPU throughput lost | Notebook callout, line 239 | "Providing a 'Secure Enclave' for your model costs 15% of your GPU's raw throughput" |
| Privacy budget is finite: epsilon depletes across queries | @sec-security-privacy, line 1181 | "the privacy budget must be carefully managed across queries, as repeated queries on similar inputs deplete the budget" |
| "Privacy is a budget, not a switch" | @sec-security-privacy, line 1197 | "Privacy is a budget, not a switch — systems requiring strict privacy must implement Differential Privacy (DP) to quantify and cap the leakage (epsilon) per query" |
| Noise injection sigma=0.05 for sentiment classifier | @sec-security-privacy, line 1173 | "sigma = 0.05 adds sufficient uncertainty to impede extraction while rarely flipping prediction decisions" |
| Sensitivity/epsilon = $200K noise for salary range | DPCostAnalysis, line 167 | "sensitivity = salary_range = 200000; noise_scale = 200000 / 1.0 = 200,000" |
| DP epsilon range 1-8 for federated deployments, 1-5% accuracy drop | [^fn-dp-federated] in edge_intelligence.qmd, line 227 | "typical federated deployments use epsilon=1-8, where smaller values...reduce model accuracy by 1-5%" |
