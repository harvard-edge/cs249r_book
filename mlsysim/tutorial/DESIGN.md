# MLSys·im Tutorial Design Document

## First-Principles Performance Modeling for ML Systems: From Roofline to Fleet Design with mlsysim

**Venue:** ISCA 2026 (International Symposium on Computer Architecture)
**Format:** Full-day tutorial (6 hours of instruction, 9:00 AM -- 5:00 PM)
**Capacity:** 40--80 attendees
**Presenters:** [TBD]

---

## 1. Tutorial Overview

### Abstract

Every ML system hits a wall. The question is *which* wall, and *when*.

This hands-on tutorial teaches attendees to answer that question from first
principles using mlsysim, an open-source analytical modeling engine that
encodes 22 fundamental constraints ("walls") governing ML system performance.
Unlike simulation or benchmarking, mlsysim requires no GPUs -- it derives
performance predictions from physics: FLOPs, bytes, bandwidth, and latency.

Attendees will build intuition through a sequence of progressively harder
design challenges, starting with a single Roofline analysis and ending with a
multi-region fleet design that balances throughput, latency, cost, and carbon.
Every exercise produces a quantitative answer that can be checked against
hardware datasheets and published benchmarks.

The tutorial is structured around a single master equation -- the Iron Law of
ML Training -- and systematically unpacks each term through live coding,
pair exercises, and group design challenges. By the end of the day, attendees
will have internalized a mental framework for reasoning about ML system
performance that transfers directly to their own research and engineering work.

**This is not a tool demo. It is a thinking workshop that happens to use a tool.**

### Target Audience

| Profile | What they get |
|---------|---------------|
| PhD students (architecture/systems) | A quantitative framework for the "ML systems" half of their thesis |
| PhD students (ML/AI) | Intuition for why their training runs are slow and what hardware to request |
| Industry engineers (NVIDIA, Google, Meta, AMD) | A rapid prototyping tool for hardware/system design exploration |
| Faculty | A ready-made lab sequence for ML systems courses |
| Startup engineers | A way to make hardware purchasing decisions without benchmarking everything |

### Prerequisites

- Comfortable reading Python (no advanced features required)
- Laptop with Python 3.10+ and `pip install mlsysim` completed
- Familiarity with basic computer architecture concepts (caches, bandwidth, throughput)
- No GPU required. No cloud account required. Everything runs on a laptop CPU.

See `prerequisites.md` for detailed setup instructions.

---

## 2. Desired Results (Backward Design: Stage 1)

### Transfer Goals (6 months later)

What should attendees be able to do *in their own work* long after the tutorial?

1. **Back-of-envelope reasoning.** Given any ML workload and any hardware target,
   estimate whether the system will be compute-bound or memory-bound within
   60 seconds, using only a datasheet and mental arithmetic.

2. **Bottleneck diagnosis.** When a training run or serving system underperforms,
   systematically identify which of the 22 walls is binding, rather than guessing
   or blindly tuning hyperparameters.

3. **Design-space navigation.** When choosing between hardware options, parallelism
   strategies, or compression methods, frame the decision as a quantitative
   trade-off rather than a qualitative preference.

4. **Skeptical consumption of benchmarks.** Read MLPerf results, vendor claims,
   and paper benchmarks with a calibrated mental model that can distinguish
   "impressive engineering" from "easy workload on expensive hardware."

### Understanding Goals (Enduring Understandings)

These are the "big ideas" that survive after attendees forget the API syntax.

| # | Enduring Understanding | Common Misconception It Replaces |
|---|------------------------|----------------------------------|
| U1 | **Every ML workload is either compute-bound or memory-bound, and the transition depends on batch size.** The Roofline model is not just a pretty picture -- it is a decision boundary. | "More FLOPS = faster inference" (ignoring that LLM decode at batch=1 never touches peak FLOPS). |
| U2 | **The Iron Law decomposes all performance into five multiplicative terms (N, Peak, MFU, scaling efficiency, goodput). Every optimization maps to exactly one term.** | "Performance is complicated and unpredictable" (actually, it factors cleanly). |
| U3 | **Communication cost grows with scale. There is always an N beyond which adding more GPUs makes training slower, not faster.** | "Just add more GPUs" (ignoring AllReduce overhead). |
| U4 | **Compression is an architectural decision, not an optimization. INT4 vs FP16 changes the fleet size by 2x.** | "Quantization is a deployment afterthought" (it should be a design-time decision). |
| U5 | **Geography is the highest-leverage sustainability decision. The same training run produces 40x more carbon in Poland than Quebec.** | "AI carbon footprint is mainly about GPU efficiency" (grid carbon intensity dominates). |
| U6 | **Inverse modeling (given SLA, derive hardware requirements) is more useful than forward modeling (given hardware, predict performance) for system design.** | "I benchmark first, then decide" (expensive and slow). |

### Knowledge and Skill Goals (By end of day)

Attendees will be able to:

- K1: Use `Engine.solve()` to evaluate any model-hardware pair and interpret the result
- K2: Explain the Roofline model and identify the ridge point for a given workload
- K3: Calculate KV-cache memory requirements for LLM serving at a given batch size
- K4: Compute AllReduce communication time for a given cluster configuration
- K5: Use `ParallelismOptimizer` to find optimal TP/PP/DP splits
- K6: Quantify carbon footprint differences across datacenter regions
- K7: Design a multi-constraint serving fleet (throughput + latency + budget + carbon)
- K8: Use sensitivity analysis to identify which parameter to optimize next
- K9: Use inverse Roofline to derive hardware requirements from an SLA

---

## 3. Assessment Evidence (Backward Design: Stage 2)

For each understanding goal, here is the observable evidence of mastery.

### Evidence Map

| Understanding | Exercise (formative) | Observable Evidence | Common Wrong Answer (diagnostic) |
|---------------|---------------------|---------------------|----------------------------------|
| U1: Roofline transition | Ex 1: Batch-size sweep | Attendee correctly predicts *before running the code* that LLM decode at batch=1 is memory-bound, while batched CNN inference is compute-bound | "Both are compute-bound because GPUs are fast" |
| U2: Iron Law decomposition | Ex 4: "Where did the time go?" | Attendee decomposes a slow training run into the five Iron Law terms and identifies MFU as the binding factor | "The network is the bottleneck" (when actually MFU is 20%) |
| U3: Communication scaling | Ex 5: AllReduce cliff | Attendee identifies the GPU count beyond which scaling efficiency drops below 0.5 and explains why | "AllReduce is O(N)" (it is actually bandwidth-optimal but still costs 2M/BW) |
| U4: Compression as architecture | Ex 6: Fleet halving | Attendee demonstrates that INT4 quantization halves the fleet size for serving 70B models | "Quantization only helps latency" (it changes fleet economics) |
| U5: Carbon geography | Ex 7: Geography exercise | Attendee correctly estimates the 40x carbon difference and identifies grid intensity as the dominant term | "More efficient GPUs = less carbon" (ignoring grid mix) |
| U6: Inverse Roofline | Ex 8: Capstone | Attendee derives minimum hardware specs from an SLA and uses it to select hardware | "I need to benchmark each option" |

### The Five Designed "Aha Moments"

These are the specific points in the tutorial where attendees' intuition should
break and rebuild. Each is engineered through a *predict-then-reveal* structure.

1. **Aha #1 -- "The H100 is only 1.7x faster?"** (10:00 AM)
   Attendees predict that the H100 (3.2x more FLOPS than A100) will be 3.2x faster
   for LLM inference. They run the code and discover it is only ~1.7x faster because
   decode is memory-bound, and memory bandwidth only improved ~1.7x.

2. **Aha #2 -- "KV-cache eats all my memory"** (11:00 AM)
   Attendees predict they can serve 200+ concurrent LLM requests on an 80 GB GPU.
   They sweep batch sizes and discover OOM at ~64 because KV-cache grows linearly
   with batch size and dominates memory at high concurrency.

3. **Aha #3 -- "Adding GPUs made it SLOWER"** (1:30 PM)
   Attendees scale a training run from 8 to 256 GPUs and watch MFU *decrease*
   as AllReduce communication overhead grows. They find the crossover point
   where adding GPUs no longer helps.

4. **Aha #4 -- "INT4 halved my fleet"** (2:30 PM)
   Attendees realize that quantizing a 70B model from FP16 to INT4 is not just a
   latency optimization -- it changes whether the model fits on one GPU, which
   halves the number of GPUs needed for the entire serving fleet.

5. **Aha #5 -- "Geography matters more than hardware"** (3:15 PM)
   Attendees calculate that moving a training run from Poland to Quebec reduces
   carbon by 99%, while upgrading from A100 to H100 only reduces it by ~30%.
   The grid carbon intensity dominates all other factors by an order of magnitude.

### Capstone Performance Task (4:00 -- 4:45 PM)

**Design Brief:** You are the ML infrastructure lead at a startup. The CEO wants
to serve Llama-3-70B at 1,000 queries per second with the following constraints:

- Per-token latency (ITL) under 50 ms
- Annual budget under $5M
- Carbon footprint under 500 tonnes CO2/year
- Must survive a full-node failure without dropping below 800 QPS

Design the fleet. Present your recommendation as a one-slide summary with:
the hardware choice, precision, fleet size, regional split, redundancy strategy,
and the binding constraint that limits further optimization.

**Rubric for "good" capstone answer:**
- Recognizes that 70B FP16 does not fit on one H100 (forces TP=2 or INT4)
- Quantizes to INT4 to halve GPU count (or justifies FP16 + TP with cost analysis)
- Places majority of fleet in Quebec for carbon constraint
- Adds N+1 or N+2 redundancy per region
- Identifies the binding constraint (usually budget or carbon, not compute)
- Shows the numbers, not just the conclusion

---

## 4. Learning Plan (Backward Design: Stage 3)

### Design Principles

- **Predict-Code-Reflect cycle:** Every exercise starts with a prediction on paper,
  then runs code to check, then reflects on the gap between prediction and reality.
- **Pair programming by default:** Attendees work in pairs for all exercises.
  One person drives, one navigates. Switch at each exercise.
- **Energy management:** High-engagement activities after lunch. Lecture-heavy
  content in the morning when attention is fresh. Short breaks every 60--75 minutes.
- **40% hands-on minimum:** Of 360 minutes of instruction, at least 144 minutes
  are active coding, discussion, or design work.

### Hour-by-Hour Schedule

---

#### 9:00 -- 9:30 | Opening: The Hook (30 min)

**Goal:** Convince the room this is worth their attention. Establish the Iron Law.

| Time | Activity | Format | Notes |
|------|----------|--------|-------|
| 9:00 | **The $200M Question** | Lecture + live demo | Open with: "Meta spent $200M training Llama-3-405B. Before a single GPU was purchased, how would you know if it was enough? How would you know 405B was the right size? How would you know 16K H100s was the right fleet?" Run `Engine.solve(Llama3_8B, H100, batch_size=1)` live. The answer appears in 0.1 seconds, on a laptop, with no GPU. |
| 9:10 | **The Iron Law** | Lecture (whiteboard) | Write the master equation on the board. Walk through each term. "Every wall in the taxonomy maps to one of these five terms. That is the entire framework." |
| 9:20 | **Install check + hello_world** | Hands-on (solo) | Everyone runs `hello_world.py`. Debug stragglers. This is the "everyone types something" ice-breaker. |
| 9:28 | **Pair up** | Social | Ask neighbors to form pairs. "You will work with this person for the next 3 hours. Introduce yourself: name, institution, one ML system you wish was faster." |

**Slide deck:** Section 1 (8--10 slides)

---

#### 9:30 -- 10:30 | Module 1: The Roofline Model (60 min)

**Goal:** U1 (Roofline transition) + K1 (Engine.solve) + K2 (Ridge point)
**Aha moment:** #1 ("The H100 is only 1.7x faster")

| Time | Activity | Format | Notes |
|------|----------|--------|-------|
| 9:30 | **What is a Roofline?** | Lecture (15 min) | Explain arithmetic intensity. Draw the Roofline on the board. Two regions: compute-bound (flat ceiling) and memory-bound (sloped line). The ridge point is where they meet. |
| 9:45 | **Prediction exercise** | Paper (pair, 3 min) | "Before we run any code: Llama-3-8B, batch_size=1, on A100 vs H100. H100 has 3.2x more FLOPS. How much faster will it be? Write your prediction on paper." Collect a show of hands for 1x, 2x, 3x, 3.2x. |
| 9:48 | **Exercise 1: Batch-size sweep** | Hands-on (pair, 12 min) | Run Exercise 1 from `exercises.md`. Sweep batch sizes. Find the transition point. Compare A100 vs H100. |
| 10:00 | **Reveal + discussion** | Facilitated discussion (10 min) | Show the results. "Raise your hand if the H100 was less than 2x faster." Explain: decode is memory-bound, and memory bandwidth only improved 1.7x. FLOPS don't matter when you're not doing math. Draw the connection to the Roofline. |
| 10:10 | **Generalization: When does each regime apply?** | Lecture (10 min) | CNN training (compute-bound) vs LLM decode (memory-bound) vs LLM prefill (compute-bound). The batch size knob moves you along the Roofline. |
| 10:20 | **Concept check** | Quick poll (5 min) | "You have a budget for one hardware upgrade for an LLM serving system at batch_size=1. Do you buy more FLOPS or more bandwidth?" Correct answer: bandwidth. Anyone who says FLOPS gets to explain why (productive error). |
| 10:25 | **Recap + bridge** | Lecture (5 min) | "We now know how to find the bottleneck for one model on one chip. But serving is more complex -- two phases, shared memory. That is next." |

**Slide deck:** Section 2 (12--15 slides)

---

#### 10:30 -- 10:45 | Break (15 min)

---

#### 10:45 -- 11:45 | Module 2: LLM Serving and the Memory Wall (60 min)

**Goal:** U1 (deepened) + K3 (KV-cache) + Aha #2
**Aha moment:** #2 ("KV-cache eats all my memory")

| Time | Activity | Format | Notes |
|------|----------|--------|-------|
| 10:45 | **Prefill vs Decode** | Lecture (10 min) | Serving has two phases with opposite bottlenecks. Prefill = compute-bound (processing the prompt). Decode = memory-bound (generating one token at a time). Draw timeline. |
| 10:55 | **The KV-cache equation** | Lecture + whiteboard (10 min) | Derive the KV-cache formula: `2 x L x H_kv x D_head x S x B x bytes`. Walk through each term. "Each active request carries its own memory of the conversation." |
| 11:05 | **Prediction exercise** | Paper (pair, 3 min) | "Llama-3-8B on H100 (80 GB), FP16 weights = 16 GB. How many concurrent 4K-context requests can you serve? Write your prediction." Most will guess 100+. |
| 11:08 | **Exercise 2: Serving capacity sweep** | Hands-on (pair, 12 min) | Run Exercise 2. Sweep batch_size from 1 to 128 at seq_len=4096. Find where `feasible` flips to False. Record weight memory vs KV-cache memory at each point. |
| 11:20 | **Reveal: "Where did the memory go?"** | Facilitated discussion (10 min) | Show that at batch=64, KV-cache is ~64 GB while weights are only ~16 GB. KV-cache dominates. This is why PagedAttention was invented. This is why context length is expensive. |
| 11:30 | **Extension: What does quantization do to serving capacity?** | Hands-on + discussion (10 min) | "Switch from FP16 to INT8. How many more requests fit?" Both weights AND KV-cache shrink. Double benefit. Connect to U4 (compression as architecture). |
| 11:40 | **Recap + bridge** | Lecture (5 min) | "We can now model one GPU. But real systems use many GPUs. What happens when you add communication?" |

**Slide deck:** Section 3 (12--15 slides)

---

#### 11:45 -- 12:00 | Module 3: Compression as Architecture (15 min, fast)

**Goal:** U4 (compression changes fleet design) + K6 (quantify trade-offs)

| Time | Activity | Format | Notes |
|------|----------|--------|-------|
| 11:45 | **The Fidelity Wall** | Lecture (8 min) | Quantization (fewer bits), pruning (fewer weights), distillation (smaller model). Key insight: storage always shrinks, but speedup depends on whether you are compute-bound or memory-bound. Structured vs unstructured pruning. |
| 11:53 | **Exercise 3: Quantization trade-offs** | Hands-on (pair, 7 min) | Run Exercise 3. Compare FP16 vs INT4. Note compression ratio, accuracy delta, and speedup. "Is the speedup 8x or 4x? Why?" |

**Slide deck:** Section 4 (6--8 slides)

---

#### 12:00 -- 1:00 | Lunch Break (60 min)

**Lunch activity (optional):** Post a "Wall of Walls" on a whiteboard near the
lunch area. Each attendee writes on a sticky note: "The wall I hit most often
in my work is ____." Presenters cluster the notes during lunch to calibrate
afternoon emphasis.

---

#### 1:00 -- 1:15 | Energy Reset (15 min)

**Goal:** Re-engage after lunch. Low-effort but high-engagement.

| Time | Activity | Format | Notes |
|------|----------|--------|-------|
| 1:00 | **The Wall of Walls results** | Facilitated discussion (5 min) | "Here is what you told us over lunch. The top 3 walls this room hits are: ___." Validates that the afternoon content is relevant to them. |
| 1:05 | **Speed round: "Name that wall"** | Group quiz (10 min) | Show 6 real-world scenarios (one sentence each). Pairs have 30 seconds to identify which wall is binding. Examples: "Training throughput doesn't improve when we add a 9th GPU to a DGX node" (Communication Wall -- cross-node AllReduce over slower InfiniBand). Award bragging rights. |

---

#### 1:15 -- 2:15 | Module 4: Distributed Training (60 min)

**Goal:** U2 (Iron Law) + U3 (Communication scaling) + K4 (AllReduce) + K5 (ParallelismOptimizer)
**Aha moment:** #3 ("Adding GPUs made it SLOWER")

| Time | Activity | Format | Notes |
|------|----------|--------|-------|
| 1:15 | **The Iron Law, revisited** | Lecture (10 min) | Return to the master equation. Now unpack eta_scaling (communication overhead) and Goodput (failure recovery). "When you go from 1 GPU to 1000, which terms change?" N goes up (good), but eta_scaling and Goodput go down (bad). |
| 1:25 | **AllReduce: the math** | Lecture + whiteboard (10 min) | Ring AllReduce: `T = 2(N-1)/N * M/BW + 2(N-1) * alpha`. Walk through each term. Key insight: the bandwidth term approaches 2M/BW regardless of N (bandwidth-optimal), but latency grows linearly with N. |
| 1:35 | **3D Parallelism: TP, PP, DP** | Lecture (10 min) | Tensor parallelism (within node, NVLink). Pipeline parallelism (across stages, point-to-point). Data parallelism (across groups, AllReduce). Why TP stays within a node: NVLink is 18x faster than InfiniBand. The pipeline bubble. |
| 1:45 | **Prediction exercise** | Paper (pair, 3 min) | "Llama-3-70B on 64 H100s. You are sweeping from 8 GPUs to 64 GPUs with DP only (no TP/PP). At what GPU count does scaling efficiency drop below 50%? Write your prediction." |
| 1:48 | **Exercise 4: Parallelism strategy search** | Hands-on (pair, 15 min) | Run Exercise 4. Use `ParallelismOptimizer` to search over TP/PP/DP configurations. Compare pure DP (TP=1, PP=1) vs the optimizer's recommendation. Record MFU at each configuration. |
| 2:03 | **Reveal: "The communication cliff"** | Facilitated discussion (12 min) | Show the scaling curve. "At 64 GPUs with pure DP, the AllReduce time dominates the step time. The optimizer finds TP=8 (within-node NVLink) because it avoids cross-node communication for the most frequent operation." Why TP=8 is almost always optimal on DGX nodes. Connect back to Iron Law: eta_scaling was the binding term. |

**Slide deck:** Section 5 (15--18 slides)

---

#### 2:15 -- 2:30 | Break (15 min)

---

#### 2:30 -- 3:15 | Module 5: Economics, Carbon, and Fleet Design (45 min)

**Goal:** U4 (compression as architecture, deepened) + U5 (carbon geography) + K6 (carbon) + Aha #4 and #5
**Aha moments:** #4 ("INT4 halved my fleet") + #5 ("Geography > hardware for carbon")

| Time | Activity | Format | Notes |
|------|----------|--------|-------|
| 2:30 | **TCO: The Capital Wall** | Lecture (8 min) | CapEx + OpEx. A 1024-GPU H100 cluster costs ~$30M. Electricity is ~10% of TCO. The dominant cost lever is utilization -- idle GPUs cost the same as busy ones. |
| 2:38 | **The fleet-halving exercise** | Hands-on (pair, 10 min) | "Design a serving fleet for Llama-3-70B at 1000 QPS. First in FP16, then in INT4. How many GPUs do you need in each case?" Key realization: 70B FP16 = 140 GB, does not fit on one H100, requires TP=2 minimum. INT4 = 35 GB, fits on one GPU. Fleet size halves (or better). |
| 2:48 | **Reveal: Aha #4** | Discussion (5 min) | "Quantization is not an optimization. It is an architectural decision that changes the fleet design by 2x." Connect to U4. |
| 2:53 | **Carbon: The Sustainability Wall** | Lecture (7 min) | `CO2 = Energy x PUE x Carbon_Intensity`. The 100x variation in grid carbon intensity across regions. Show the map. |
| 3:00 | **Exercise 5: Carbon geography** | Hands-on (pair, 8 min) | Run Exercise 5. Compare Virginia vs Quebec for a 30-day training run. Calculate the absolute CO2 savings. |
| 3:08 | **Reveal: Aha #5** | Discussion (7 min) | "Moving from Virginia to Quebec saves 99% of carbon. Upgrading from A100 to H100 saves ~30%. Geography is the highest-leverage sustainability decision by an order of magnitude." Show the numbers side by side. |

**Slide deck:** Section 6 (12--15 slides)

---

#### 3:15 -- 3:45 | Module 6: Sensitivity Analysis and Inverse Roofline (30 min)

**Goal:** U6 (inverse modeling) + K8 (sensitivity) + K9 (inverse Roofline)

| Time | Activity | Format | Notes |
|------|----------|--------|-------|
| 3:15 | **"Which knob should I turn next?"** | Lecture (10 min) | Sensitivity analysis: perturb each parameter by 10% and measure the change in output. The parameter with the largest partial derivative is your bottleneck. This is Wall 21. |
| 3:25 | **Inverse Roofline** | Lecture (10 min) | The most powerful move: given an SLA (e.g., 50ms per token), derive the minimum hardware specs. `Required_BW = Weights / Target_Latency`. "If you need 50ms decode for a 140 GB model, you need 2.8 TB/s bandwidth. That eliminates A100 and V100 immediately." This is Wall 22. |
| 3:35 | **Quick exercise: Derive your own specs** | Hands-on (pair, 10 min) | "Your SLA is 30ms ITL for Llama-3-8B at FP16. What is the minimum memory bandwidth required? Which GPUs in the registry meet it? What if you quantize to INT4?" |

**Slide deck:** Section 7 (8--10 slides)

---

#### 3:45 -- 4:00 | Break + Capstone Briefing (15 min)

| Time | Activity | Format | Notes |
|------|----------|--------|-------|
| 3:45 | Break | -- | -- |
| 3:55 | **Capstone brief** | Lecture (5 min) | Hand out the design brief (see Section 3). Read it aloud. Clarify constraints. "You have 45 minutes. Work in your pairs. You will present a one-slide summary to the room." |

---

#### 4:00 -- 4:45 | Capstone: Fleet Design Under Constraints (45 min)

**Goal:** All understanding goals synthesized. Observable evidence of transfer.

| Time | Activity | Format | Notes |
|------|----------|--------|-------|
| 4:00 | **Work session** | Pair work (35 min) | Pairs work through the capstone. Instructors circulate, ask probing questions, give hints when stuck. Key checkpoints to watch for: (1) Do they recognize 70B FP16 doesn't fit on one GPU? (2) Do they consider INT4? (3) Do they check the carbon constraint? (4) Do they add redundancy? |
| 4:35 | **Gallery walk** | Group (10 min) | 3--4 pairs volunteer to share their design in 2 minutes each. After each, the room gives one "strength" and one "what-if" (e.g., "What if Quebec has a network outage?"). |

---

#### 4:45 -- 5:00 | Closing: The Transfer Moment (15 min)

| Time | Activity | Format | Notes |
|------|----------|--------|-------|
| 4:45 | **The framework, one last time** | Lecture (5 min) | Return to the Iron Law. "Every exercise today mapped to one term. The Roofline mapped to Peak and MFU. Serving mapped to the memory wall within Peak. Distributed training mapped to eta_scaling. Carbon mapped to the operational cost of N x Peak x Time. The capstone required all five." |
| 4:50 | **Personal transfer moment** | Solo reflection (5 min) | "Write down one system in your own work. Which wall is it hitting? Which Iron Law term is binding? What would you change first?" This is the single most important activity of the day. |
| 4:55 | **Resources and next steps** | Lecture (5 min) | Point to: mlsysim GitHub, the textbook, the cheatsheet, the 22 Walls paper. "The cheatsheet has every equation and API call from today on one page. Take a photo." |

---

### Time Budget Summary

| Category | Minutes | Percentage |
|----------|---------|------------|
| Lecture and demonstration | 168 | 47% |
| Hands-on exercises (coding) | 132 | 37% |
| Discussion and reflection | 45 | 12% |
| Breaks | 15 (not counted in 360) | -- |
| Social/logistics | 15 | 4% |
| **Total instruction time** | **360** | **100%** |
| **Active learning (hands-on + discussion)** | **177** | **49%** |

---

## 5. Slide Outline

### Section 1: The Hook (8--10 slides)

1. Title slide (tutorial title, presenters, affiliation)
2. The $200M question (Meta's Llama-3-405B training cost)
3. "What if you could answer these questions on a laptop?" (live demo screenshot)
4. The Iron Law: `Time = FLOPs / (N x Peak x MFU x eta x Goodput)` (full equation)
5. The Iron Law: term-by-term breakdown table
6. The 22 Walls taxonomy (one-page visual map organized by domain)
7. Tutorial roadmap (visual timeline of the day)
8. Setup check: `pip install mlsysim` and `hello_world.py`
9. Pair up instructions

### Section 2: The Roofline Model (12--15 slides)

1. What is arithmetic intensity? (definition + formula)
2. The Roofline diagram (classic plot with compute ceiling and memory slope)
3. The ridge point (where the two regions meet)
4. Prediction slide: "A100 vs H100 for LLM decode at batch=1"
5. Exercise 1 instructions
6. Results reveal: H100 is only 1.7x faster (actual numbers)
7. Why: memory bandwidth improvement (2.0 vs 3.35 TB/s = 1.67x)
8. Regime map: which workloads fall where on the Roofline
9. CNN training: compute-bound (high arithmetic intensity)
10. LLM decode: memory-bound (low arithmetic intensity, weights dominate)
11. LLM prefill: compute-bound (batched matmul over prompt tokens)
12. The batch-size knob: moving along the Roofline
13. Concept check: "FLOPS or bandwidth?" poll
14. Summary: one slide with the key takeaway

### Section 3: LLM Serving and Memory (12--15 slides)

1. Two-phase serving: prefill (compute) + decode (memory)
2. Timeline diagram: TTFT vs ITL
3. The KV-cache equation (all terms labeled)
4. Worked example: Llama-3-8B at 4K context, one request
5. Prediction slide: "How many concurrent requests on 80 GB?"
6. Exercise 2 instructions
7. Results reveal: OOM at ~64 (KV-cache dominates memory)
8. Memory breakdown bar chart: weights vs KV-cache at various batch sizes
9. Why this matters: PagedAttention, continuous batching
10. Extension: INT8 quantization doubles serving capacity
11. The Batching Wall: KV-cache fragmentation
12. The Tail Latency Wall: P99 grows non-linearly with utilization
13. Summary slide

### Section 4: Compression (6--8 slides)

1. Three compression methods: quantization, pruning, distillation
2. The Fidelity Wall: compression ratio vs accuracy trade-off
3. Key insight: storage always shrinks, speedup depends on bound
4. Exercise 3 instructions
5. Results: FP16 vs INT4 comparison table
6. Structured vs unstructured pruning (only structured accelerates on GPU)
7. Summary: compression is a design-time decision

### Section 5: Distributed Training (15--18 slides)

1. Why distribute? Model too big, or training too slow
2. The Iron Law revisited: which terms change at scale?
3. Data parallelism: replicate model, split data, AllReduce gradients
4. The AllReduce equation (ring algorithm, bandwidth-optimal)
5. Worked example: 1 GB gradients, 8 GPUs, NVLink
6. Worked example: same but 64 GPUs, InfiniBand
7. Tensor parallelism: shard within a layer (Megatron-style)
8. Why TP stays within a node: NVLink vs InfiniBand bandwidth gap (18x)
9. Pipeline parallelism: shard across layers
10. The pipeline bubble (micro-batch diagram)
11. 3D Parallelism: TP x PP x DP (visual grid)
12. Prediction slide: "Scaling efficiency at 64 GPUs with pure DP?"
13. Exercise 4 instructions
14. Results reveal: optimal TP=8, PP=2, DP=4
15. The communication cliff: scaling efficiency vs N
16. Goodput: failures at scale (MTBF/N)
17. The Fragility Wall: 1000 GPUs = failure every 50 hours
18. Summary: the binding term was eta_scaling

### Section 6: Economics and Sustainability (12--15 slides)

1. TCO equation: CapEx + OpEx_energy + OpEx_maintenance
2. Cost breakdown: hardware dominates (electricity is ~10%)
3. The dominant lever: utilization (idle GPUs cost the same)
4. The fleet-halving exercise: FP16 vs INT4 for 70B serving
5. Results: INT4 halves the fleet (Aha #4)
6. Carbon equation: Energy x PUE x Carbon_Intensity
7. Carbon intensity world map (1 to 800 gCO2/kWh, 100x variation)
8. Exercise 5 instructions
9. Results: Virginia vs Quebec (99% carbon reduction) (Aha #5)
10. Comparison: geography vs hardware upgrade for carbon
11. Water usage: evaporative cooling in hot climates
12. Summary: geography is the highest-leverage sustainability decision

### Section 7: Sensitivity and Inverse Roofline (8--10 slides)

1. "Which knob should I turn?" (the engineer's daily question)
2. Sensitivity analysis: perturb each parameter by 10%
3. Tornado chart: which parameter has the largest derivative?
4. Inverse Roofline: flip the equation
5. Worked example: 50ms ITL for 140 GB model requires 2.8 TB/s
6. Hardware selection by elimination (table: which GPUs qualify?)
7. Quick exercise instructions
8. Summary: inverse modeling is more useful than forward modeling

### Section 8: Capstone and Closing (6--8 slides)

1. Capstone design brief (full constraints on one slide)
2. Hints and checkpoints (for mid-exercise guidance)
3. Gallery walk rubric (what makes a good design?)
4. The Iron Law, one final time (annotated with today's exercises)
5. Personal transfer moment instructions
6. Resources: GitHub, textbook, cheatsheet, 22 Walls paper
7. Thank you + contact information

**Total slide estimate: 80--100 slides**

---

## 6. Materials Needed

### Before the Tutorial

| Material | Format | Status | Notes |
|----------|--------|--------|-------|
| Slide deck (Sections 1--8) | Keynote/PDF | To create | ~90 slides |
| Cheatsheet (1 page, double-sided) | PDF handout | Exists (`cheatsheet.md`) | Print 80 copies |
| Exercise booklet | PDF handout | Exists (`exercises.md`) | Print 80 copies |
| Prerequisites guide | PDF / web | Exists (`prerequisites.md`) | Email to attendees 2 weeks before |
| Design brief (capstone) | PDF handout | To create | One page, print 80 copies |
| Sticky notes + markers | Physical | To purchase | For "Wall of Walls" lunch activity |
| Backup USB drives | Physical | To prepare | Pre-loaded with mlsysim wheel, exercises, slides (for offline install) |

### Software and Demo Scripts

| Script | Purpose | Module |
|--------|---------|--------|
| `hello_world.py` | Install check + ice-breaker | Opening |
| `01_basic_roofline.py` | A100 vs H100 comparison | Module 1 |
| Exercise 1 code (from `exercises.md`) | Batch-size sweep | Module 1 |
| Exercise 2 code (from `exercises.md`) | Serving capacity | Module 2 |
| Exercise 3 code (from `exercises.md`) | Quantization trade-offs | Module 3 |
| Exercise 4 code (from `exercises.md`) | Parallelism optimizer | Module 4 |
| `02_carbon_geography.py` | Carbon comparison demo | Module 5 |
| Exercise 5 code (from `exercises.md`) | Carbon geography | Module 5 |
| Exercise 8 code (from `exercises.md`) | Capstone | Capstone |
| Sensitivity demo script | Tornado chart | Module 6 |
| Inverse Roofline demo script | Derive specs from SLA | Module 6 |

### To Create Before Tutorial

1. **Sensitivity analysis demo script** -- sweep parameters with partial derivatives, produce a tornado chart
2. **Inverse Roofline demo script** -- given SLA, derive minimum bandwidth, filter hardware registry
3. **Capstone design brief** -- one-page handout with all constraints, hints, and rubric
4. **"Name That Wall" quiz slides** -- 6 real-world scenarios, each mapping to a specific wall
5. **Prediction slips** -- small paper slips for the predict-then-reveal exercises (or use digital polling)

---

## 7. Facilitation Notes

### Energy Management

| Time | Energy Level | Strategy |
|------|-------------|----------|
| 9:00--10:30 | High (morning fresh) | Front-load the most conceptually demanding material (Roofline theory). Lecture-heavy is OK here. |
| 10:45--12:00 | Medium (mid-morning) | Mix lecture with hands-on. KV-cache exercise provides active engagement. |
| 1:00--1:15 | Low (post-lunch dip) | Do NOT lecture. Start with the "Wall of Walls" debrief and the "Name That Wall" quiz -- short, social, low-stakes. |
| 1:15--2:15 | Rising | Distributed training is complex but the exercises are engaging (watching scaling break). Pair work sustains attention. |
| 2:30--3:15 | Medium | Economics and carbon are lighter conceptually. The "fleet halving" and "geography" ahas land well here. |
| 3:15--3:45 | Declining | Sensitivity and inverse Roofline are meta-level tools. Keep it short and practical. |
| 4:00--4:45 | Variable | Capstone creates its own energy through challenge and social accountability (they know they will present). |
| 4:45--5:00 | Closing | The personal transfer moment is quiet and reflective. End on a thoughtful note, not a rushed one. |

### Common Pitfalls and Mitigations

| Pitfall | Mitigation |
|---------|------------|
| **Attendees can't install mlsysim** | Have USB drives with pre-built wheels. Have a Colab notebook as fallback. Test installation email 2 weeks prior. Budget 10 minutes for setup in the opening. |
| **The room is 80% industry, 20% academic (or vice versa)** | The Roofline and AllReduce content works for both. Adjust capstone emphasis: industry cares more about cost/SLA, academics care more about scaling laws. Ask during intro: "Who is trying to train? Serve? Design hardware?" |
| **Pairs are mismatched (expert + novice)** | This is actually good -- the expert teaches, the novice asks "why." Frame it: "The navigator's job is to ask 'why does that number make sense?'" |
| **An exercise takes longer than allocated** | Every exercise has a "minimum viable result" (the first print statement) and an extension ("now try INT8"). Call time on the extension, not the core. |
| **Someone asks about a wall we don't cover in depth (e.g., Tail Latency, Checkpoint)** | Acknowledge it. "That is Wall 7 / Wall 19. We don't have time today, but the cheatsheet has the equation, and the textbook chapter covers it in depth." |
| **"This is just a toy model, real systems are more complex"** | Agree, then redirect: "Absolutely. The question is whether the toy model gives you the right *direction*. If it says memory-bound and the real system is memory-bound, the model is useful. If it says the answer is 4.8 ms and the real answer is 5.2 ms, that's within 10%. The goal is not prediction -- it is reasoning." |
| **Skeptic: "I can just benchmark on real hardware"** | "Yes, for one configuration. How many configurations can you benchmark in an hour? mlsysim evaluates thousands per second. Benchmarking tells you what IS. Modeling tells you what COULD BE." |

### Instructor Preparation Checklist

- [ ] Run all 8 exercises end-to-end on a clean laptop install
- [ ] Prepare "expected output" screenshots for each exercise (for attendees who fall behind)
- [ ] Test projector resolution with Roofline plots and tables
- [ ] Prepare 3 backup discussion questions per module (in case exercises finish early)
- [ ] Print cheatsheets and exercise booklets (double-sided, stapled)
- [ ] Prepare sticky notes and markers for "Wall of Walls"
- [ ] Load USB drives with: mlsysim wheel, exercises.md, cheatsheet.md, example scripts
- [ ] Test WiFi at venue (if exercises need any network access)
- [ ] Prepare a shared document or Slack channel for attendees to ask questions after the tutorial

### Timing Flexibility

If running behind schedule:

- **Module 3 (Compression)** can be shortened to 8 minutes (lecture only, skip exercise, defer to cheatsheet)
- **Module 6 (Sensitivity/Inverse Roofline)** can be shortened to 15 minutes (lecture only, skip exercise)
- **Capstone gallery walk** can be shortened to 5 minutes (2 presentations instead of 4)
- **Do NOT cut:** Module 1 (Roofline), Module 2 (Serving), or Module 4 (Distributed). These are the core.

If running ahead of schedule:

- Extend the capstone work time (pairs always want more time)
- Add a "bring your own workload" segment: attendees model their own research system
- Deeper dive into sensitivity analysis with live parameter sweeps

---

## Appendix A: Mapping to the 22 Walls

| Wall | Covered In | Depth |
|------|-----------|-------|
| 1. Compute | Module 1 | Deep |
| 2. Memory | Module 1, 2 | Deep |
| 3. Software (MFU) | Module 1, 4 | Deep |
| 4. Serving | Module 2 | Deep |
| 5. Batching (KV-cache) | Module 2 | Deep |
| 6. Streaming | Mentioned | Light |
| 7. Tail Latency | Module 2 | Light |
| 8. Ingestion | Mentioned | Light |
| 9. Transformation | Mentioned | Light |
| 10. Locality | Module 4 | Medium |
| 11. Complexity (Chinchilla) | Opening | Medium |
| 12. Reasoning | Mentioned | Light |
| 13. Fidelity | Module 3 | Medium |
| 14. Communication | Module 4 | Deep |
| 15. Fragility | Module 4 | Medium |
| 16. Multi-tenant | Mentioned | Light |
| 17. Capital (TCO) | Module 5 | Medium |
| 18. Sustainability | Module 5 | Deep |
| 19. Checkpoint | Module 4 | Light |
| 20. Safety | Mentioned | Light |
| 21. Sensitivity | Module 6 | Medium |
| 22. Synthesis (Inverse) | Module 6 | Medium |

Deep coverage: 7 walls. Medium coverage: 6 walls. Light/mentioned: 9 walls.
This is appropriate for a single-day tutorial. The cheatsheet covers all 22.

---

## Appendix B: Assessment Rubric for Capstone

| Criterion | Excellent (3) | Adequate (2) | Needs Work (1) |
|-----------|--------------|--------------|-----------------|
| **Precision choice** | Recognizes 70B FP16 doesn't fit on 1 GPU; makes deliberate INT4 vs TP=2 decision with justification | Chooses INT4 or TP but doesn't justify the trade-off | Assumes FP16 fits on one GPU |
| **Fleet sizing** | Calculates GPUs from per-GPU throughput; shows the math | Gives a reasonable number but doesn't show derivation | Guesses a number |
| **Regional split** | Optimizes split for carbon constraint; shows Quebec advantage | Splits 50/50 without analysis | Ignores regional placement |
| **Redundancy** | Addresses failure scenario; adds N+1 or N+2 | Mentions redundancy but doesn't size it | Ignores failures |
| **Binding constraint** | Identifies which constraint is tightest and what would need to change to relax it | Meets all constraints but doesn't identify the binding one | Violates one or more constraints |
| **Presentation** | Clear one-slide summary with numbers, not just words | Readable but missing key numbers | Disorganized or incomplete |
