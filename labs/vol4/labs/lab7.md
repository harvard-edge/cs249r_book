# Lab 7 - Robustness Testing

Status: Skeleton \
When: Week 8 \
Due: end of Week 9 \
Readings: Part II: Teaching the Machine - Closed-Loop Evaluation; Part IV: Governing the Machine - Adversarial Verification

---

## 1. Learning Objectives

By the end of this lab, you will be able to:

- Quantify the resilience of the Dual Brain deployment versus the Single Brain deployment against unexpected workloads and resource exhaustion.
- Explain how task complexity, number of predicted actions, re-evaluation frequency, and number of attention heads affect latency and, consequently, partial completion rates and collision counts.

---

## 2. Implementation Steps

1. Use the stress-ng utility to execute resource exhaustion attacks (CPU stress tests) on both the Single Brain and Dual Brain deployments (We Do - 75 mins).
2. Measure partial completion rates and collision counts under stress for the different VLA policies and scenarios provided by the instructor (You Do - 105 mins).

---

## 3. Deliverables

1. Motor updates per second, partial completion rates, and collision counts comparing the Single Brain and Dual Brain deployments under 25%, 50%, 75%, and 100% external loads.
2. Partial completion rates and collision counts for the different VLA policies and scenarios provided by the instructor.
3. A brief written analysis explaining why the metrics changed. You must explicitly address: (a) how the drop in motor updates per second under stress impacted the robot's physical performance (collisions/partial completion); (b) which deployment (Single Brain vs. Dual Brain) proved more resilient; and (c) how the specific VLA parameters (attention heads, re-evaluation frequency, etc.) influenced the outcomes.

---