# MLSys·im Tutorial: Pre/Post Assessment Quiz

## Administration Instructions

**Purpose:** This quiz serves as both a formative engagement tool during the tutorial
and a summative assessment instrument for education research. The identical quiz is
administered twice: once before the tutorial begins (9:00 AM) and once after the
closing (4:50 PM, during the reflection window).

**Time limit:** 5 minutes (strictly enforced both administrations).

**Format:** Paper-based or digital form (Google Forms recommended for automated scoring).
Attendees write their unique participant ID (assigned at registration) on the form.
No names are collected on the quiz itself.

**Instructions to read aloud:**

> "This is a 10-question quiz about ML systems performance reasoning. Some questions
> are multiple choice, some ask for a short numerical answer. Do your best -- there is
> no penalty for wrong answers. You will see this same quiz again at the end of the day.
> Do not discuss answers with your neighbor. You have 5 minutes."

---

## The Quiz

### Question 1: Bottleneck Identification (U1)

**An H100 GPU has 1,979 TFLOPS (FP16) and 3.35 TB/s memory bandwidth. You are
running inference on a 16 GB model at batch_size=1, where each token requires
32 GFLOP of compute and must read all 16 GB of weights from memory.**

**Is this workload compute-bound or memory-bound?**

- (a) Compute-bound, because 1,979 TFLOPS is very high
- (b) Memory-bound, because the time to read 16 GB exceeds the time to compute 32 GFLOP
- (c) Neither -- batch_size=1 means the GPU is idle most of the time
- (d) It depends on the sequence length, not the batch size

**Correct answer:** (b)

**Scoring:** 1 point for (b), 0 otherwise.

**Distractor analysis:**

| Answer | Misconception revealed |
|--------|----------------------|
| (a) | Believes high peak FLOPS guarantees compute-bound operation. Confuses capability with utilization. This is the most common pre-tutorial error -- "more FLOPS = faster." |
| (c) | Confuses "small batch" with "idle GPU." The GPU is fully busy reading memory; it is just not doing much math per byte read. |
| (d) | Partially correct intuition (sequence length does affect arithmetic intensity during prefill), but for decode at batch=1, the weights dominate regardless of sequence length. Reveals confusion between prefill and decode regimes. |

**Understanding goal:** U1 -- Every ML workload is either compute-bound or memory-bound, and the transition depends on batch size.

---

### Question 2: Roofline Reasoning (U1)

**You upgrade from an A100 (2.0 TB/s bandwidth, 624 TFLOPS FP16) to an H100
(3.35 TB/s bandwidth, 1,979 TFLOPS FP16) for LLM inference at batch_size=1.
How much faster is the H100?**

- (a) About 1.7x faster
- (b) About 3.2x faster
- (c) About 2.5x faster (the geometric mean of the bandwidth and FLOPS ratios)
- (d) About 1.0x -- they are the same because both are bottlenecked by PCIe

**Correct answer:** (a)

**Scoring:** 1 point for (a), 0 otherwise.

**Distractor analysis:**

| Answer | Misconception revealed |
|--------|----------------------|
| (b) | Uses the FLOPS ratio (1979/624 = 3.2x) as the speedup. Classic error: applying the compute-bound mental model to a memory-bound workload. |
| (c) | Attempts to average the two ratios, suggesting awareness that both metrics matter but lacking the framework to determine which one dominates. |
| (d) | Confuses HBM bandwidth with PCIe bandwidth. The weights are already in HBM; PCIe is not on the critical path for inference. |

**Understanding goal:** U1 -- The Roofline model determines whether FLOPS or bandwidth limits performance.

---

### Question 3: Iron Law Decomposition (U2)

**A training run on 64 GPUs achieves only 35% MFU (Model FLOPS Utilization). Your
colleague suggests adding 64 more GPUs to make it faster. Using the Iron Law
(`Time = FLOPs / (N * Peak * MFU * eta_scaling * Goodput)`), which is the best response?**

- (a) Good idea -- doubling N will halve the training time
- (b) Bad idea -- you should first fix MFU; doubling N with 35% MFU wastes 65% of the new GPUs too
- (c) Bad idea -- Goodput will drop to zero with 128 GPUs
- (d) It depends on the model size, not the GPU count

**Correct answer:** (b)

**Scoring:** 1 point for (b), 0 otherwise.

**Distractor analysis:**

| Answer | Misconception revealed |
|--------|----------------------|
| (a) | Treats the Iron Law terms as independent. Doubling N does not halve time if MFU remains low and eta_scaling decreases. Ignores the multiplicative structure of the equation. |
| (c) | Overstates the failure risk. Goodput concerns are real at very large scale but 128 GPUs is not inherently catastrophic. Reveals a "scaling is always dangerous" misconception without quantitative reasoning. |
| (d) | True that model size matters for parallelism strategy, but the question is about whether to invest in N or MFU. Choosing (d) avoids engaging with the Iron Law decomposition. |

**Understanding goal:** U2 -- The Iron Law decomposes performance into multiplicative terms, and every optimization maps to exactly one term.

---

### Question 4: Communication Scaling (U3)

**You are training a 7B parameter model using pure data parallelism. Each gradient
AllReduce must synchronize 14 GB of gradients. Your interconnect bandwidth is
400 Gb/s (50 GB/s) per link.**

**Approximately how long does one AllReduce take with Ring AllReduce at 64 GPUs?**

- (a) About 0.28 seconds (14 GB / 50 GB/s)
- (b) About 0.56 seconds (2 * 14 GB / 50 GB/s)
- (c) About 18 seconds (14 GB / 50 GB/s * 64 GPUs)
- (d) About 0.009 seconds (14 GB / 50 GB/s / 64 GPUs)

**Correct answer:** (b)

**Scoring:** 1 point for (b), 0 otherwise.

**Distractor analysis:**

| Answer | Misconception revealed |
|--------|----------------------|
| (a) | Forgets the factor of 2 in Ring AllReduce (scatter-reduce + allgather). Knows the bandwidth-optimal property but misses a constant factor. Partial understanding. |
| (c) | Believes AllReduce time scales linearly with N. This is the "naive broadcast" mental model. Does not understand that Ring AllReduce is bandwidth-optimal. |
| (d) | Believes AllReduce time decreases with more GPUs (like a parallel speedup). Fundamental misunderstanding: communication is overhead, not parallelizable work. |

**Understanding goal:** U3 -- Communication cost grows with scale and AllReduce time approaches 2M/BW regardless of N.

---

### Question 5: Compression as Architecture (U4)

**You need to serve Llama-3-70B (140 GB in FP16) on H100 GPUs with 80 GB HBM each.
What is the minimum number of GPUs per model replica in FP16 vs INT4?**

- (a) FP16: 2 GPUs, INT4: 1 GPU
- (b) FP16: 2 GPUs, INT4: 2 GPUs
- (c) FP16: 1 GPU, INT4: 1 GPU
- (d) FP16: 4 GPUs, INT4: 2 GPUs

**Correct answer:** (a)

**Scoring:** 1 point for (a), 0 otherwise.

**Distractor analysis:**

| Answer | Misconception revealed |
|--------|----------------------|
| (b) | Knows FP16 requires 2 GPUs but does not realize INT4 (35 GB) fits on one H100. Misses the architectural implication of quantization: it changes the parallelism requirement. |
| (c) | Does not know that 70B parameters * 2 bytes = 140 GB, which exceeds 80 GB. Lacks the habit of computing model memory. |
| (d) | Overestimates memory usage, possibly including KV-cache or optimizer states in the serving estimate. Conflates training memory with inference memory. |

**Understanding goal:** U4 -- Compression is an architectural decision that changes fleet topology, not just a latency optimization.

---

### Question 6: Fleet Impact of Quantization (U4)

**Continuing from Question 5: if you need 1,000 QPS and each INT4 replica on a single
GPU handles 25 QPS, how many GPUs does the INT4 fleet require? How does this compare
to the FP16 fleet (where each FP16 replica on 2 GPUs handles 20 QPS)?**

*Short answer -- write two numbers: INT4 fleet GPUs and FP16 fleet GPUs.*

**Correct answer:** INT4: 40 GPUs. FP16: 100 GPUs.

**Scoring rubric:**

| Score | Criteria |
|-------|----------|
| 2 | Both numbers correct (INT4 = 40, FP16 = 100) |
| 1 | One number correct, or both within 20% of correct |
| 0 | Both wrong or blank |

**Worked solution:**
- INT4: 1000 QPS / 25 QPS per GPU = 40 GPUs (1 GPU per replica)
- FP16: 1000 QPS / 20 QPS per replica = 50 replicas * 2 GPUs each = 100 GPUs

**Common errors:**
- Writing 50 for FP16 (forgetting that each replica requires 2 GPUs)
- Writing 80 for INT4 (using the FP16 throughput number with INT4)

**Understanding goal:** U4 -- Quantization changes fleet size by 2.5x in this case, not just latency.

---

### Question 7: Carbon Geography (U5)

**A training run consumes 500 MWh of electricity. The grid carbon intensity in
Virginia is 390 gCO2/kWh. In Quebec, it is 1.2 gCO2/kWh. How many tonnes of CO2
does the run produce in each location?**

*Short answer -- write two numbers: Virginia tonnes and Quebec tonnes.*

**Correct answer:** Virginia: 195 tonnes. Quebec: 0.6 tonnes.

**Scoring rubric:**

| Score | Criteria |
|-------|----------|
| 2 | Both numbers correct (within 10%: Virginia 175--215, Quebec 0.5--0.7) |
| 1 | One number correct, or both show the right method (multiply MWh * gCO2/kWh / 1000) but arithmetic error |
| 0 | Both wrong, blank, or wrong method |

**Worked solution:**
- Virginia: 500,000 kWh * 390 gCO2/kWh = 195,000,000 g = 195 tonnes
- Quebec: 500,000 kWh * 1.2 gCO2/kWh = 600,000 g = 0.6 tonnes

**Common errors:**
- Unit confusion: forgetting to convert MWh to kWh or grams to tonnes
- Writing "195,000" for Virginia (forgetting g-to-tonne conversion)
- Applying a PUE factor when none was given (over-complicating)

**Understanding goal:** U5 -- Geography is the highest-leverage sustainability decision (325x difference here).

---

### Question 8: Sustainability Reasoning (U5)

**Your company wants to reduce training carbon by 50%. Which single action achieves
this most reliably?**

- (a) Upgrade from A100 to H100 GPUs (about 30% more energy-efficient)
- (b) Move the training run from a coal-heavy grid to a hydro-powered grid
- (c) Reduce the model size by 50%
- (d) Train for half as many epochs

**Correct answer:** (b)

**Scoring:** 1 point for (b), 0 otherwise.

**Distractor analysis:**

| Answer | Misconception revealed |
|--------|----------------------|
| (a) | A 30% efficiency gain does not reach the 50% reduction target. Also conflates energy efficiency with carbon efficiency -- a more efficient GPU on a dirty grid still produces more carbon than a less efficient GPU on a clean grid. |
| (c) | Reducing model size by 50% does not reduce compute by 50% (FLOPs scale differently than parameter count for Transformers). Also, this changes the model, not just the infrastructure. |
| (d) | Halving epochs halves energy and thus carbon, which technically achieves 50%. But it changes the trained model (likely worse accuracy). The question asks about infrastructure decisions. Partial credit could be argued, but (b) achieves far more than 50% without changing the model. |

**Understanding goal:** U5 -- Grid carbon intensity dominates GPU efficiency by an order of magnitude.

---

### Question 9: Inverse Design (U6)

**Your SLA requires 40 ms per-token latency for a 16 GB (FP16) language model.
During decode at batch_size=1, the dominant cost is reading all weights from memory
once per token. What is the minimum memory bandwidth required?**

- (a) 400 GB/s
- (b) 640 GB/s
- (c) 1,600 GB/s
- (d) 3,200 GB/s

**Correct answer:** (a)

**Scoring:** 1 point for (a), 0 otherwise.

**Distractor analysis:**

| Answer | Misconception revealed |
|--------|----------------------|
| (b) | Possibly applying a safety factor or confusing bits and bytes. |
| (c) | Using 10 ms instead of 40 ms (misreading the SLA) or applying a 4x overhead factor. |
| (d) | Computing from FLOPS requirements rather than bandwidth requirements. Applying the compute-bound mental model to a memory-bound problem -- exactly the error the Roofline model corrects. |

**Worked solution:** Required bandwidth = 16 GB / 0.040 s = 400 GB/s.

**Understanding goal:** U6 -- Inverse modeling derives hardware requirements from SLAs, which is more useful than forward benchmarking.

---

### Question 10: Inverse Design Synthesis (U6)

**Given the 400 GB/s minimum from Question 9, which of the following GPUs can meet
the SLA WITHOUT tensor parallelism? (Select all that apply.)**

| GPU | HBM Bandwidth |
|-----|---------------|
| A100 | 2,039 GB/s |
| L40S | 864 GB/s |
| A10 | 600 GB/s |
| T4 | 320 GB/s |

- (a) A100, L40S, and A10
- (b) A100 and L40S only
- (c) A100 only
- (d) All four GPUs

**Correct answer:** (a)

**Scoring:** 1 point for (a), 0 otherwise.

**Distractor analysis:**

| Answer | Misconception revealed |
|--------|----------------------|
| (b) | Misses A10 (600 GB/s > 400 GB/s). May be applying a large safety margin or confusing A10 with a different GPU. |
| (c) | Only selecting the highest-bandwidth option. Reveals a "pick the best GPU" heuristic rather than computing the minimum requirement and filtering. |
| (d) | Does not check the T4 (320 GB/s < 400 GB/s). Reveals failure to apply the inverse constraint. |

**Additional consideration:** The model must also fit in GPU memory. A10 has 24 GB (fits 16 GB model), T4 has 16 GB (barely fits, but fails on bandwidth anyway). A complete answer would check both bandwidth and memory, but the question isolates bandwidth to test U6.

**Understanding goal:** U6 -- Inverse Roofline eliminates hardware options analytically, without benchmarking.

---

## Scoring Summary

| Question | Type | Points | Understanding Goal |
|----------|------|--------|--------------------|
| Q1 | Multiple choice | 1 | U1 |
| Q2 | Multiple choice | 1 | U1 |
| Q3 | Multiple choice | 1 | U2 |
| Q4 | Multiple choice | 1 | U3 |
| Q5 | Multiple choice | 1 | U4 |
| Q6 | Short answer | 2 | U4 |
| Q7 | Short answer | 2 | U5 |
| Q8 | Multiple choice | 1 | U5 |
| Q9 | Multiple choice | 1 | U6 |
| Q10 | Multiple choice | 1 | U6 |
| **Total** | | **12** | |

**Coverage:** Each understanding goal (U1--U6) is tested by exactly 2 questions, contributing
exactly 2 points each. This enables per-goal gain analysis.

---

## Expected Score Distributions

These estimates are based on the ISCA audience profile: PhD students and industry engineers
in computer architecture and systems, with varying ML systems experience.

### Pre-Tutorial Expectations

| Score Range | Expected % | Profile |
|-------------|-----------|---------|
| 0--3 | 15% | ML-focused attendees with little hardware intuition |
| 4--6 | 45% | Typical ISCA attendee: strong architecture background, partial ML systems knowledge |
| 7--9 | 30% | Experienced ML systems practitioners |
| 10--12 | 10% | Experts who already reason this way |

**Expected pre-test mean:** 5.5 +/- 2.5 (out of 12)

### Post-Tutorial Expectations

| Score Range | Expected % | Profile |
|-------------|-----------|---------|
| 0--3 | 2% | Attendees who disengaged or arrived late |
| 4--6 | 15% | Partial transfer -- grasped some but not all frameworks |
| 7--9 | 45% | Solid transfer -- internalized the Iron Law and Roofline |
| 10--12 | 38% | Full transfer -- can apply all six understandings |

**Expected post-test mean:** 8.5 +/- 2.0 (out of 12)

**Expected effect size:** Cohen's d ~ 1.2 (large), based on the 3-point mean gain with
pooled SD ~ 2.3. This is consistent with pre/post gains observed in similar hands-on
computing workshops (e.g., Software Carpentry reports d = 0.8--1.5).

---

## Distractor Summary Table

This table maps each distractor to the misconception it diagnoses, enabling aggregate
misconception analysis across the cohort.

| Misconception | Questions where it appears | Expected pre-test prevalence |
|---------------|---------------------------|------------------------------|
| "More FLOPS = faster" | Q1(a), Q2(b) | 40--50% |
| "Just add more GPUs" | Q3(a), Q4(c) | 30--40% |
| "Quantization is just a latency trick" | Q5(b), Q6 (forgetting 2-GPU requirement) | 35--45% |
| "Carbon = energy efficiency" | Q8(a) | 50--60% |
| "Benchmark first, decide later" | Q9(d), Q10(c) | 25--35% |
| "AllReduce scales linearly with N" | Q4(c) | 20--30% |

---

## IRB Considerations

### Consent Language

The following consent statement must be displayed at the top of the quiz form and
read aloud before the first administration:

> **Research Participation Notice**
>
> This quiz is part of a research study on ML systems education. Your responses
> will be used to evaluate the effectiveness of this tutorial. Participation is
> voluntary. You may skip any question or withdraw at any time without penalty.
> Your responses are identified only by a randomly assigned participant ID --
> your name is never recorded on this form.
>
> By completing this quiz, you consent to the anonymous use of your responses
> in published educational research. If you do not wish to participate in the
> research, you may still take the quiz for your own learning -- simply write
> "NO RESEARCH" next to your participant ID and your data will be excluded.
>
> Questions about this study: [PI email] | IRB Protocol #: [TBD]

### Data Handling

- **Anonymization:** Participant IDs are randomly generated 6-digit codes assigned at
  registration. The mapping from ID to name is stored separately and destroyed after
  the tutorial. Only the ID-linked quiz responses are retained for analysis.

- **Storage:** Quiz responses stored in an encrypted, access-controlled institutional
  repository. Only the research team has access.

- **Retention:** De-identified data retained indefinitely for longitudinal comparison
  across tutorial offerings. Raw forms destroyed 1 year after analysis.

- **Exempt status:** This study likely qualifies for IRB exemption under 45 CFR 46.104(d)(1)
  (research conducted in established educational settings involving normal educational
  practices). However, an IRB application must be filed before the tutorial. File at
  least 8 weeks in advance.

### ISCA-Specific Considerations

- The tutorial is an educational activity at a professional conference. Attendees are
  adults attending voluntarily. This strengthens the exemption argument.
- No compensation is offered for research participation.
- The quiz is dual-use (educational + research). Attendees benefit from the quiz
  regardless of research participation.
- Demographics (career stage, institution type, years of experience) should be collected
  on a separate voluntary form, not on the quiz itself, to maintain quiz anonymity.

---

## Digital Implementation Notes

If using Google Forms:

1. Create two identical forms: "Pre-Tutorial Assessment" and "Post-Tutorial Assessment"
2. First field: Participant ID (short text, required)
3. Multiple choice questions: radio buttons
4. Short answer questions (Q6, Q7): two separate short-text fields each
5. Enable "Collect email addresses" = OFF
6. Enable "Limit to 1 response" = OFF (some may need to resubmit)
7. Timestamp collection is automatic and provides a check on 5-minute time limit
8. Export to CSV for analysis

If using paper:

1. Print on a single double-sided sheet
2. Participant ID field at top
3. Consent statement on front
4. Collect all forms before allowing discussion
5. Enter data manually into spreadsheet (budget 2 hours for 80 forms * 2 administrations)
