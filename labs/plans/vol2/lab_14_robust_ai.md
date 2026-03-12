# Mission Plan: lab_14_robust_ai

## 1. Chapter Alignment

- **Chapter:** Robust AI (`@sec-robust-ai`)
- **Core Invariant:** The **Robustness Tax** -- adversarial training to defend against epsilon=8/255 perturbations drops ResNet-50 clean accuracy from 76% to ~50% (a 26 percentage point loss), and requires 8x more compute per epoch (PGD-7: 1 standard + 7 attack passes = 8 total). Robustness and standard accuracy are fundamentally in tension: improving worst-case performance degrades average-case performance, and this trade-off cannot be eliminated by better algorithms alone.
- **Central Tension:** Students believe that robustness is a feature that can be "turned on" after a model is trained -- a monitoring layer or input filter that catches bad inputs. The chapter demolishes this: a model's robustness properties are determined during training, not at inference. Adversarial training costs 26 percentage points of clean accuracy and 8x compute. The compound surprise: the very efficiency techniques that make deployment feasible (INT8 quantization, pruning) narrow the robustness margin, making the system more fragile to perturbations a full-precision model could absorb.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that adversarial training is a low-cost upgrade -- perhaps 2--5% accuracy drop for meaningful robustness. The chapter's data shows that adversarially training a ResNet-50 at epsilon=8/255 drops clean accuracy from 76% to 50%, a 26 percentage point loss. Students predict the accuracy cost, discover it is far larger than expected, and learn that the model must sacrifice "non-robust features" (high-frequency textures that are predictive but brittle) to gain robustness. The 8x compute multiplier per epoch transforms this from a statistical trade-off into an economic one: training one robust model costs as much as training eight standard models.

**Act 2 (Design Challenge, 22 min):** Students design a defense stack for a production system that must survive both adversarial perturbations and distribution drift within a compute budget. The chapter shows that distribution shift causes 20--40% accuracy drops on out-of-distribution inputs, that adversarial training alone costs 8--10x compute, and that certified defenses (randomized smoothing) require 100,000 forward passes per sample. Students must configure a layered defense (adversarial training, input sanitization, confidence thresholds, monitoring) that keeps accuracy above a safety floor while staying under a compute ceiling, discovering that every defense layer adds cost and that the cheapest effective strategy is detection and monitoring, not universal robustification.

---

## 3. Act 1: The Robustness Tax (Calibration -- 12 minutes)

### Pedagogical Goal

Students think of adversarial robustness as a defensive add-on with minimal cost -- analogous to adding a firewall to a server. The chapter quantifies the exact price: a ResNet-50 adversarially trained at epsilon=8/255 loses 26 percentage points of clean accuracy (76% to 50%). This is not a minor accuracy regression; it is a fundamental trade-off between average-case and worst-case performance. The model must learn to ignore "non-robust features" (high-frequency textures that are predictive on clean data but exploitable by adversaries), and these features carry significant predictive power. The 8x compute penalty (1 standard pass + 7 PGD attack steps per batch) means one robust training run costs as much as eight standard runs. This act calibrates students to understand that robustness is a budget you spend, not a switch you flip.

### The Lock (Structured Prediction)

Present a **multiple-choice prediction** before any instruments unlock:

> "A standard ResNet-50 achieves 76% top-1 accuracy on ImageNet. You adversarially train it using PGD-7 at epsilon=8/255 (an imperceptible perturbation) to make it robust against adversarial attacks. What clean accuracy does the robust model achieve on normal (non-adversarial) ImageNet images?"

Options:
- A) ~74% -- robustness costs only 2--3 percentage points, a minor tax
- B) ~65% -- significant but manageable, roughly an 11 point drop
- **C) ~50% -- a devastating 26 percentage point drop; the model sacrifices standard performance for worst-case resilience** (correct)
- D) ~76% -- adversarial training does not affect clean accuracy at all

The correct answer is C. The chapter's `RobustnessTaxAnalysis` class (line 1877) shows clean_acc=76.0, robust_acc=50.0, acc_drop=26. Students who pick A or D hold the most dangerous misconception: that robustness is cheap. Students who pick B are closer but still underestimate by 15 percentage points. The chapter explicitly states: "Robustness cannot simply be 'turned on' for free. It is a fundamental trade-off between Average-Case Performance and Worst-Case Reliability."

### The Instrument: Robustness-Accuracy Pareto Curve

A **scatter plot with annotated key points** showing the robustness-accuracy trade-off:

- **X-axis:** Defense type (categorical: None / Adversarial Training / Randomized Smoothing / Feature Squeezing)
- **Y-axis:** Accuracy (0--100%)
- **Two grouped bars per defense:**
  - Clean accuracy (BlueLine): how the model performs on normal inputs
  - Robust accuracy (OrangeLine): how the model performs under attack / with certification
- **Compute cost annotation:** Text label above each group showing the training or inference cost multiplier

Controls:
- **Defense type selector** (radio buttons): None / Adversarial Training (PGD-7) / Randomized Smoothing / Feature Squeezing
  - None: 76% clean, ~0% robust, 1x compute
  - Adversarial Training: 50% clean, ~28% robust at epsilon=8/255, 8x training compute
  - Randomized Smoothing: ~62% clean, ~49% certified at L2=0.5, 100,000x inference compute per sample
  - Feature Squeezing: ~73% clean (95%+ maintained), eliminates 70--90% of adversarial examples, ~1x inference compute
- **Perturbation budget (epsilon) slider:** 0/255 to 16/255, step 1/255, default 8/255
  - As epsilon increases, clean accuracy for adversarial training drops further; the 30--60% range from the chapter definition holds

**Secondary display:** A **compute cost comparison** bar chart with three bars:
- Standard training: 1x
- Adversarial training (PGD-7): 8x (1 + 7 PGD steps)
- Certified inference (randomized smoothing): 100,000x per sample

### The Reveal

After interaction:
> "You predicted ~[X]% clean accuracy for the robust model. The actual value is **50%** -- a **26 percentage point drop** from the standard model's 76%. The chapter states: 'Robustness cannot simply be turned on for free. It is a fundamental trade-off between Average-Case Performance and Worst-Case Reliability.' The 8x compute penalty per epoch means adversarial training costs as much as training 8 standard models. You were off by [|X - 50|] percentage points."

### Reflection (Structured)

Four-option multiple choice:

> "The robust model loses 26 percentage points of accuracy on clean images. What does it sacrifice to gain robustness?"

- A) Model capacity -- the robust model uses fewer parameters and thus has less expressive power
- **B) Non-robust features -- high-frequency textures that are predictive on clean data but exploitable by adversaries; removing them costs clean accuracy but eliminates the attack surface** (correct)
- C) Training data quality -- adversarial training corrupts the training distribution with noisy examples
- D) Inference speed -- the robust model uses slower, more careful computations that trade speed for accuracy

The chapter states: "The model must learn to ignore 'non-robust features' (like high-frequency textures) that are predictive but brittle" (line 1907).

### Math Peek (collapsible)

$$\text{Robustness Tax} = \text{Accuracy}_{\text{standard}} - \text{Accuracy}_{\text{robust}} = 76\% - 50\% = 26\%$$
$$\text{Compute Penalty (PGD-}K\text{)} = 1 + K = 1 + 7 = 8\times \text{ per epoch}$$
$$\text{Certified Radius: } R = \sigma \Phi^{-1}(p_A) \quad \text{(requires } N=100{,}000 \text{ noise samples per inference)}$$

---

## 4. Act 2: The Compound Defense Problem (Design Challenge -- 22 minutes)

### Pedagogical Goal

Students believe that deployment-time defenses (monitoring, input filters, confidence thresholds) can compensate for a non-robust model cheaply. The chapter reveals that every defense layer carries its own tax: adversarial training costs 8x compute and 26% accuracy; confidence thresholds reject 5--15% of legitimate traffic; error correction consumes 12--25% additional memory bandwidth; monitoring adds 5--15% computational overhead; and certified defenses require 100,000 forward passes per sample. Meanwhile, unmonitored distribution shifts cause 20--40% accuracy drops on OOD inputs. Students must configure a defense stack that meets a safety target (combined accuracy above 60% under attack + drift) while staying under a compute budget (total overhead below 10x), discovering that the only economically viable path for most systems is detection + monitoring rather than universal robustification -- exactly as the chapter concludes: "it is more efficient to rely on external guardrails than to train intrinsic robustness into the model weights."

### The Lock (Numeric Prediction)

Before instruments unlock:

> "A ResNet-50 achieves 76% accuracy on clean ImageNet. After 6 months of deployment without monitoring, the data distribution shifts (new lighting conditions, different cameras). Based on the chapter's data, what accuracy does the unmonitored model achieve on out-of-distribution inputs?"

Students type a percentage (bounded input: 0--100, step 1). Expected wrong answers: 70--75% (students assume small, gradual degradation). Actual: **36--56%** -- the chapter states "accuracy drops of 20--40% on out-of-distribution inputs" (line 2399). A 20--40% drop from 76% yields 36--56%. The system will show: "You predicted [X]%. The chapter reports 20--40% accuracy drops on OOD inputs, putting the actual range at 36--56%. You were off by [gap] percentage points."

### The Instrument: Defense Stack Builder

A **layered bar chart** with dual y-axes showing cumulative cost and benefit:

- **X-axis:** Defense layers applied incrementally (left to right): Baseline -> +Adversarial Training -> +Input Sanitization -> +Confidence Threshold -> +Monitoring & Retraining
- **Y-axis (left):** Accuracy (0--100%), shown as stacked bars per condition:
  - Clean accuracy (BlueLine)
  - Accuracy under adversarial attack at epsilon=8/255 (OrangeLine)
  - Accuracy under distribution shift (RedLine)
- **Y-axis (right):** Cumulative compute overhead multiplier (1x--120x, log scale line)
- **Budget line:** Horizontal dashed line at the compute budget ceiling (10x, adjustable)
- **Safety line:** Horizontal dashed line at accuracy floor (60%, adjustable)

Controls:
- **Adversarial training toggle:** ON/OFF (default: OFF)
  - ON: clean accuracy drops from 76% to 50%; robust accuracy rises from ~0% to ~28% at epsilon=8/255; compute = 8x training
  - OFF: clean accuracy 76%; robust accuracy ~0%
- **Input sanitization toggle:** ON/OFF (default: OFF)
  - Feature squeezing: eliminates 70--90% of adversarial examples; ~3% clean accuracy cost; ~1x inference compute
- **Confidence threshold slider:** 0.5 / 0.7 / 0.9 / 0.95 / 0.99 (default: 0.5)
  - Higher threshold rejects more predictions; at 0.9+, rejects 5--15% of legitimate traffic
  - Rejected predictions do not count toward accuracy (increases effective accuracy on served predictions but reduces coverage)
- **Monitoring frequency selector:** None / Monthly / Weekly / Daily (default: None)
  - None: drift accumulates silently; 20--40% accuracy degradation over 6--12 months
  - Weekly/Daily: PSI detects shifts 3--6 weeks before accuracy falls below SLA (line 2403), triggers retraining
  - Compute overhead: 5--15% for continuous monitoring (line 94)
- **Deployment context toggle:** Production (INT8) / Hardened (FP32)
  - INT8: 75% memory reduction, 2--4x inference speedup, but narrows robustness margin (reduced numerical headroom)
  - FP32: full precision, preserves robustness margin

**Secondary instrument: Silent Degradation Timeline**

A **dual-line time series** showing accuracy over deployment time:

- **X-axis:** Months since deployment (0--24)
- **Y-axis:** Model accuracy (20--100%)
- **Lines:**
  - Without monitoring (RedLine): accuracy degrades 20--40% over 6--12 months (from chapter line 2399)
  - With monitoring + retraining (GreenLine): accuracy recovers after each retraining trigger
- **Trigger annotations:** Points where PSI monitoring detects drift (3--6 weeks early per line 2403)

### The Scaling Challenge

**"Configure a defense stack that maintains combined accuracy above 60% under BOTH adversarial attack (epsilon=4/255) AND distribution shift (20% OOD), while keeping total compute overhead below 10x baseline."**

Students must discover that:
1. Adversarial training alone blows the accuracy floor on clean data (50%) and barely meets it under combined threats, while costing 8x compute
2. Feature squeezing + confidence thresholds + daily monitoring achieves comparable protection at ~1.2x compute
3. The chapter's conclusion is validated: external guardrails (detection, filtering, monitoring) are more economical than intrinsic robustification for most systems
4. The only path requiring adversarial training is when the threat model demands worst-case guarantees (safety-critical), not just average-case resilience

### The Failure State

**Trigger 1 -- Safety violation:** Combined accuracy under attack + drift drops below 60%.
- **Visual:** Accuracy bars turn RedLine; safety threshold line highlighted with pulsing animation.
- **Banner:** "SAFETY VIOLATION -- Combined accuracy: [X]%. Safety floor: 60%. The chapter warns: 'a model can achieve 95% i.i.d. test accuracy while failing completely on inputs that differ from training by amounts imperceptible to humans.' Add defense layers or increase monitoring frequency."
- **Reversible:** Toggling on defenses or increasing monitoring immediately recalculates and can restore the system above the floor.

**Trigger 2 -- Compute budget exceeded:** Total compute overhead exceeds 10x baseline.
- **Visual:** Overhead line turns RedLine; budget ceiling line highlighted.
- **Banner:** "RESOURCE BUDGET EXCEEDED -- Current defense stack costs [X]x compute (budget: 10x). Adversarial training alone costs 8x; adding certified defenses would push to 100,000x. The chapter states: 'it is more efficient to rely on external guardrails (input filtering, output verification) than to train intrinsic robustness into the model weights.' Consider detection over robustification."
- **Reversible:** Disabling adversarial training or switching from certified to empirical defenses reduces overhead.

### Structured Reflection

Four-option multiple choice:

> "The chapter concludes that 'for many applications, it is more efficient to rely on external guardrails than to train intrinsic robustness into the model weights.' Which defense strategy achieves the best accuracy-per-compute-dollar for a production image classifier facing distribution shift?"

- A) Adversarial training at epsilon=8/255 -- maximum robustness justifies the 8x compute cost
- B) Randomized smoothing -- certified guarantees are worth the 100,000x inference overhead
- **C) Feature squeezing + confidence thresholds + continuous drift monitoring -- external guardrails provide 85--95% attack elimination at less than 2x total overhead** (correct)
- D) No defenses -- the 76% clean accuracy is sufficient for production without any robustness investment

The chapter supports C: feature squeezing eliminates 70--90% of adversarial examples at 95%+ clean accuracy (line 1942), confidence thresholds add rejection-based safety (line 110), and PSI monitoring detects drift 3--6 weeks early (line 2403). Combined overhead is far below adversarial training's 8x.

### Math Peek (collapsible)

$$P(\geq 1 \text{ SDC per hour}) = 1 - (1 - p)^N \quad \text{(at } p=10^{-4}, N=10{,}000\text{: } P \approx 0.63\text{)}$$
$$\text{Adversarial Training Cost} = (1 + K_{\text{PGD}}) \times T_{\text{standard}} = 8 \times T_{\text{standard}}$$
$$\text{Accuracy}_{\text{OOD}} \approx \text{Accuracy}_{\text{clean}} \times (1 - \text{Drift}_{\%}) = 76\% \times 0.7 = 53\%$$

---

## 5. Visual Layout Specification

### Act 1: Robustness Tax

- **Primary:** Grouped bar chart. X: defense type (4 categories). Y: accuracy (0--100%). Two bars per category: clean accuracy (BlueLine), robust/certified accuracy (OrangeLine). Compute cost label above each group.
  - Key annotation at Adversarial Training showing the 26 percentage point gap between 76% and 50%
  - Failure zone: no explicit failure state in Act 1 (calibration act)
- **Secondary:** Compute cost comparison. Three horizontal bars: Standard (1x), Adversarial (8x), Certified (100,000x per sample). Log scale x-axis.
  - Emphasizes the five-order-of-magnitude jump from empirical to certified defense

### Act 2: Compound Defense

- **Primary:** Layered bar chart. X: defense layers (5 incremental stages). Left Y: accuracy under three conditions (clean, adversarial, OOD) as stacked bars. Right Y: cumulative compute overhead (log scale line).
  - Failure state 1: accuracy bars turn RedLine when combined accuracy drops below 60% safety floor
  - Failure state 2: overhead line turns RedLine when compute exceeds 10x budget ceiling
- **Secondary:** Silent degradation timeline. X: months (0--24). Y: accuracy (20--100%). Two lines: unmonitored (RedLine, degrading) vs. monitored (GreenLine, recovering at retraining triggers).
  - Annotations at PSI detection points (3--6 weeks before SLA breach)
- **Tertiary:** Deployment context comparison panel showing INT8 vs FP32 robustness margin difference.

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---------|--------|-----|--------------|----------------|
| **Production (Optimized)** | H100 cluster, INT8 quantization | 80 GB HBM3 | 700 W TDP | Maximizes throughput via INT8 (75% memory reduction, 2--4x speedup); narrow robustness margin due to reduced numerical precision; relies on external guardrails (feature squeezing, monitoring) for defense |
| **Hardened (Safety-Critical)** | H100 cluster, FP32 only | 80 GB HBM3 | 700 W TDP | Adversarially trained at 8x compute cost; no quantization (preserves full robustness margin); confidence thresholds reject 5--15% of traffic; accepts 26% clean accuracy loss for worst-case guarantees |

The two contexts demonstrate the core chapter claim: the techniques that make deployment economically feasible (INT8, pruning) are the ones that narrow robustness margins. The Production context serves 2--4x more throughput but requires external guardrails; the Hardened context pays the full robustness tax for worst-case guarantees. The chapter states: "Robustness engineering is therefore a constant negotiation with the efficiency and scalability constraints established in previous chapters" (line 303).

---

## 7. Design Ledger Output

```json
{
  "chapter": 14,
  "adversarial_training_enabled": false,
  "defense_strategy": "guardrails",
  "epsilon_budget": 8,
  "clean_accuracy_pct": 73,
  "robust_accuracy_pct": 0,
  "compute_overhead_multiplier": 1.2,
  "compression": "int8",
  "monitoring_frequency": "daily",
  "confidence_threshold": 0.9
}
```

The `defense_strategy` and `monitoring_frequency` fields feed forward to:
- **Lab 15 (Sustainable AI):** The compute overhead multiplier from the defense stack contributes to the total energy budget calculation; students see that robustness has a carbon cost
- **Lab 12 (Ops at Scale):** The monitoring frequency choice informs the operational overhead calculation for fleet-wide deployment

The `adversarial_training_enabled` field resolves a tension introduced in earlier labs: Lab 10 (Distributed Inference) establishes INT8 quantization as standard for throughput; this lab shows that INT8 narrows robustness margins, forcing a deployment-context-dependent choice.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Standard ResNet-50: 76% top-1 accuracy | `RobustnessTaxAnalysis`, line 1881 | "clean_acc = 76.0" |
| Adversarially trained ResNet-50: ~50% clean accuracy | `RobustnessTaxAnalysis`, line 1882 | "robust_acc = 50.0" |
| Robustness tax: 26 percentage point drop | `RobustnessTaxAnalysis`, line 1885--1893 | "acc_drop = clean_acc - robust_acc" = 26 |
| Adversarial training compute: 8x per epoch (PGD-7) | `AdversarialPayback`, line 592--598 | "n_pgd_steps = 7; slowdown = 1 + n_pgd_steps = 8" |
| epsilon=8/255: 30--60% accuracy reduction in non-robust models | `@sec-robust-ai` definition, line 74 | "epsilon=8/255 (a perturbation invisible to humans) typically reduces accuracy by 30--60%" |
| Distribution shift: 20--40% accuracy drop on OOD inputs | Fallacies section, line 2399 | "unmonitored distribution shifts frequently cause silent performance degradation, leading to accuracy drops of 20--40%" |
| PSI detects drift 3--6 weeks before SLA breach | Fallacies section, line 2403 | "Monitoring metrics like the Population Stability Index (PSI) detect these shifts 3--6 weeks before accuracy falls below SLA thresholds" |
| Error correction: 12--25% additional memory bandwidth | `@sec-robust-ai`, line 94 | "Error correction mechanisms consume 12--25% additional memory bandwidth" |
| Continuous monitoring: 5--15% computational overhead | `@sec-robust-ai`, line 94 | "continuous monitoring adds 5--15% computational overhead" |
| Redundant processing: 2--3x energy increase | `@sec-robust-ai`, line 94 | "redundant processing increases energy consumption by 2--3x" |
| Confidence thresholds reject 5--15% of legitimate traffic | `[^fn-failsafe-ml]`, line 110 | "aggressive confidence thresholds reject 5--15% of legitimate traffic" |
| Feature squeezing: eliminates 70--90% adversarial examples at 95%+ clean accuracy | `[^fn-feature-squeeze-defense]`, line 1942 | "Eliminating 70--90% of adversarial examples while maintaining 95%+ clean accuracy" |
| Randomized smoothing: 100,000 samples per inference, ~62% clean, ~49% certified | `@sec-robust-ai-certified-defenses`, line 1930 | "requires sampling N=100,000 noise vectors per inference...clean accuracy to ~62% and achieves ~49% certified" |
| Single bit flip: ResNet-50 accuracy from 76% to 11% | `@sec-robust-ai`, line 347 | "A single bit flip...can degrade ResNet-50 classification accuracy from 76.0% (top-1) to 11%" |
| SDC probability at 10K GPUs: P(>=1) = 0.63 per hour | `SilentErrorProbability`, lines 379--392 | "p_per_hr_meta = 1e-4; n_gpus_large = 10,000; p_at_least_one > 0.6" |
| 175B model: 8 GPU pipeline = 8x fault surface | `@sec-robust-ai`, line 303 | "175B-parameter model requires pipeline parallelism across at least 8 GPU nodes, increasing the fault surface area by 8x" |
| Data poisoning: 0.1% of training data embeds 95% success rate backdoor | Fallacies section, line 2407 | "poisoning as little as 0.1% of the training data can embed a hidden backdoor with a 95% attack success rate" |
| ECC memory: 99.9% single-bit recovery at 12.5% bandwidth overhead | `@sec-robust-ai`, line 625 | "ECC memory systems recover from single-bit errors with 99.9% success rates while adding 12.5% bandwidth overhead" |
| INT8 quantization: 75% memory reduction, 2--4x inference speedup, 1--3% accuracy trade | `@sec-robust-ai`, line 625 | "Model quantization from FP32 to INT8 reduces memory requirements by 75% and inference time by 2--4x, trading 1--3% accuracy" |
