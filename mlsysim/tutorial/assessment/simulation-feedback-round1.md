# ISCA 2026 Tutorial Simulation Feedback: Round 1

**Tutorial:** First-Principles Performance Modeling for ML Systems: From Roofline to Fleet Design with mlsysim
**Simulated audience:** 60 attendees (mix of PhD students, industry engineers, faculty, startup CTOs)
**Date of simulation:** 2026-04-01
**Simulation method:** Walk-through of DESIGN.md, tutorial_part1.tex, tutorial_part2.tex, and exercises.md

---

## Section-by-Section Pacing Assessment

### Part 0: Welcome & Setup (9:00--9:30)

**What the presenter shows:** Title slide, the "$200M Question" hook about Meta's Llama-3-405B training cost, a live demo running `Engine.solve()` in 0.1 seconds, the five learning objectives, and the 22-Wall taxonomy overview. Then a `pip install mlsysim` check and pair-up.

**Audience energy level:** HIGH. The hook lands well. The room is fresh, people are curious, and the "$200M Question" is attention-grabbing. The live demo where a number appears instantly on a laptop creates genuine surprise.

**Where attention drops:** During the `pip install` phase. At least 5-8 people out of 60 will hit installation issues (corporate VPN blocking pip, wrong Python version, M1/M2 Mac wheel issues). The 3-minute allocation is optimistic. This will stretch to 5-7 minutes and the room gets restless while 4-5 people struggle. The presenter will be torn between helping stragglers and keeping the majority engaged.

**What questions come up:**
- "Does this work with conda? I don't use pip." (at least 2 people)
- "Is the 22-wall taxonomy published somewhere, or is this your own framework?" (a faculty member)
- "You showed Llama-3-8B in the demo, but the hook was about 405B. Can mlsysim actually model 405B?" (sharp attendee catches the bait-and-switch)

### Part 1: Iron Law & Roofline (9:30--10:30)

**What the presenter shows:** The Roofline model explained from scratch -- arithmetic intensity, the two regimes, the ridge point. Then the prediction exercise (A100 vs H100 for LLM decode), Exercise 1 (batch-size sweep), and the "Aha #1" reveal that H100 is only 1.7x faster, not 3.2x. Closes with the regime map and "FLOPS or bandwidth?" concept check.

**Audience energy level:** HIGH to MEDIUM. The prediction exercise is excellent pedagogy -- people commit to an answer on paper, then get surprised. The hands-on coding keeps energy up. Energy dips during the 10-minute "generalization" lecture after the exercise.

**Where attention drops:** The lecture segment at 10:10--10:20 ("When does each regime apply?") comes right after the exciting reveal and feels like a letdown. The presenter is explaining CNN vs LLM vs LLM-prefill regimes in lecture mode, and people want to keep coding. Also, the slide showing the full Roofline diagram with labeled regions is dense -- attendees who have never seen a Roofline plot need more time than the 15 minutes allocated for the initial explanation.

**What questions come up:**
- "What efficiency value should I use? You said 0.5 but that seems arbitrary." (at least 3 people will ask this; the slides never properly justify why efficiency=0.5)
- "Does the Roofline model account for FlashAttention? Because FlashAttention changes the arithmetic intensity." (a systems PhD student)
- "Is the 1.7x number for decode only, or also for prefill?" (good question; the answer is decode only, and the slides could be clearer)
- "What about the H200? It has the same FLOPS but more bandwidth." (an NVIDIA person, slightly trolling)

### Part 2: LLM Serving and the Memory Wall (10:45--11:45)

**What the presenter shows:** Prefill vs decode distinction, the KV-cache equation derived term by term on the whiteboard, prediction exercise on concurrent requests, Exercise 2 (serving capacity sweep), and the reveal that KV-cache dominates memory at high batch sizes.

**Audience energy level:** MEDIUM-HIGH. The KV-cache derivation is satisfying for the architecture-minded crowd. The prediction exercise works well again. The hands-on exercise is straightforward and produces a clear result.

**Where attention drops:** The KV-cache equation derivation at 10:55--11:05 goes too fast for ML students who don't think in terms of bytes-per-head-per-layer, and too slow for architecture students who already know this. The presenter has 10 minutes for a whiteboard derivation involving 7 terms -- that is tight if you want questions during the derivation.

**What questions come up:**
- "You're assuming FP16 KV-cache, but vLLM uses paged attention which fragments differently. Does mlsysim model fragmentation?" (industry engineer -- the answer is no, and this is a real gap)
- "What about GQA? Llama-3 uses grouped-query attention, which reduces KV-cache by the GQA ratio." (sharp attendee; this is listed on the v0.2.0 roadmap, meaning v0.1.0 might not model it correctly)
- "Is `feasible` just a memory check, or does it also check latency?" (several people will be confused about what `feasible` means)

### Part 3: Compression as Architecture (11:45--12:00)

**What the presenter shows:** Quick overview of quantization/pruning/distillation, the Fidelity Wall, and Exercise 3 comparing FP16 vs INT4.

**Audience energy level:** LOW-MEDIUM. This is the 15 minutes before lunch. People are hungry. The content is important but feels rushed. The exercise gets only 7 minutes, and many pairs will not finish before lunch is called.

**Where attention drops:** Right at 11:50. People are checking the time, wondering about lunch logistics. The "Is the speedup 8x or 4x?" question is genuinely interesting but gets lost in the rush. The presenter will feel pressure to wrap up and will probably skip the structured-vs-unstructured pruning explanation.

**What questions come up:**
- "Where did the 2-5% accuracy delta come from? That seems like a huge range." (fair criticism -- the model uses a rough heuristic, not actual benchmark data)
- "Can mlsysim model GPTQ vs AWQ vs different quantization methods?" (no, it treats quantization as a uniform bitwidth reduction)

### Lunch Break (12:00--1:00)

**Audience energy level:** RELIEVED. The "Wall of Walls" sticky note activity is a nice touch, but only about 15-20 of the 60 attendees will actually participate. The rest will be checking email, getting coffee, or talking to colleagues. The sticky notes that do come in will be dominated by "Memory Wall" and "Communication Wall" because those are the ones already covered.

### Part 4: Post-Lunch Energy Reset + Distributed Training (1:00--2:15)

**What the presenter shows:** Wall of Walls debrief, "Name That Wall" quiz, then the core distributed training content -- Iron Law revisited, AllReduce math, 3D parallelism (TP/PP/DP), Exercise 4 (parallelism optimizer), and the "communication cliff" reveal.

**Audience energy level:** LOW at 1:00, rising to MEDIUM by 1:15 (the quiz helps), then MEDIUM-HIGH during Exercise 4. The "Name That Wall" quiz is well-designed for post-lunch engagement -- it is social, competitive, and low-stakes.

**Where attention drops:** The AllReduce math lecture at 1:25--1:35 is the hardest 10 minutes of the day. The ring AllReduce formula `T = 2(N-1)/N * M/BW + 2(N-1) * alpha` involves multiple terms, and the post-lunch audience will struggle. At least 10-15 people will zone out here. The presenter needs to slow down and use concrete numbers, not just symbolic math.

The 3D parallelism section at 1:35--1:45 is also dense. TP, PP, and DP are three distinct concepts each requiring a mental model. Cramming all three into 10 minutes means each gets only 3 minutes of explanation, which is insufficient for attendees who have never implemented distributed training.

**What questions come up:**
- "Why is TP=8 always optimal? What if my model is small enough that TP=4 or TP=2 wastes less?" (excellent question; the answer involves NVLink bandwidth vs communication volume tradeoff)
- "The ParallelismOptimizer explored N configurations. How long does exhaustive search take? Is it always tractable?" (yes, because each evaluation is <1ms)
- "You're not modeling pipeline bubble efficiency. Isn't that a big deal for PP>1?" (the Llama-3 training report shows ~5% bubble overhead; this matters)
- "Does this work for DeepSpeed ZeRO? That's a different parallelism strategy than pure TP/PP/DP." (the answer will be "not yet" -- another gap)

### Part 5: Economics & Sustainability (2:30--3:15)

**What the presenter shows:** TCO equation, the infrastructure multiplier table showing GPUs are only 40% of cost, Exercise 5a (carbon geography), the 41x carbon gap between Poland and Quebec, embodied carbon for TinyML, and the multi-vendor TCO shootout (H100 vs MI300X vs Gaudi3).

**Audience energy level:** MEDIUM. The carbon geography result is genuinely surprising to most attendees and creates a good "aha" moment. The multi-vendor TCO shootout (Exercise 5b) is particularly interesting for the ISCA audience because it touches AMD and Intel hardware.

**Where attention drops:** The TCO equation lecture at 2:30--2:38 feels like a business school slide dropped into a systems conference. The infrastructure multiplier table is important but dry. Energy picks back up at the carbon geography exercise.

**What questions come up:**
- "Your TCO model uses an infrastructure_multiplier of 2.5. Where does that number come from? It varies wildly by datacenter." (fair -- the model is oversimplified here)
- "The MI300X has 192 GB HBM3. Does mlsysim correctly model the fact that 70B FP16 fits on a single MI300X without TP?" (this is a killer question for the ISCA audience; if yes, it changes the fleet design dramatically)
- "What about spot pricing and preemption? On-demand vs spot changes TCO by 3x on AWS." (listed as v0.2.0 feature -- not available today)
- "The embodied carbon slide cites Gupta et al. 2022, but their numbers are for older process nodes. How do you model embodied carbon for 4nm chips?" (the honest answer is "we don't model it precisely")

### Part 6: Design Space Exploration (3:15--3:45 approximately)

**What the presenter shows:** The DSE pattern (Declare-Search-Rank), Pareto fronts, live demos of `Engine.sweep` and `DSE` with constraints, and the budget-constrained design exercise.

**Audience energy level:** MEDIUM-LOW. By 3:15, the audience has been sitting for over 5 hours (with breaks). The DSE content is mechanically interesting but conceptually lighter than the morning material. People are doing the exercises but with less enthusiasm.

**Where attention drops:** The Pareto front slide references a `pareto-placeholder.pdf`, which suggests the actual visualization may not exist yet or may be a placeholder. If the Pareto plot is not compelling, this slide falls flat. The "declare, search, rank" pattern is intuitive enough that it does not need a full slide -- it feels padded.

**What questions come up:**
- "Is this just grid search? You're evaluating all combinations exhaustively?" (yes, because analytical models are fast enough -- but someone will ask about Bayesian optimization or smarter search)
- "Can I export the Pareto front data to a CSV for further analysis?" (practical question; unclear from the API)

### Part 7: TinyML to Frontier (in tutorial_part2.tex)

**What the presenter shows:** The 9-order-of-magnitude scale span from nRF52840 to H100, Flash vs SRAM memory hierarchy on MCUs, energy-per-inference table, live demo comparing keyword spotting across devices, and the IoT right-sizing exercise.

**Audience energy level:** VARIABLE. The ISCA audience has a subset that cares deeply about this (architecture researchers, edge computing people) and a larger subset that finds it tangential. The "same Roofline, different physics" message is elegant but will feel like a detour to the LLM-focused majority.

**Where attention drops:** The TinyML memory hierarchy slide (Flash vs SRAM) assumes knowledge of MCU architecture that most ML PhD students and cloud engineers simply do not have. Terms like "SRAM-only path" and "Flash bandwidth bottleneck" will confuse the LLM crowd. About 20 people will disengage during this section.

**What questions come up:**
- "This is cool but I work on datacenter GPUs. When would I use this section?" (presenter needs a bridge -- the answer is "the same equations work everywhere, and someday your model will need to run on edge")
- "The nRF52840 numbers show 500ms for DS-CNN. That's way above the 30ms SLA. Isn't this just showing the tool says 'infeasible'? What's the insight?" (valid -- the exercise proves a negative, which is less satisfying)

### Part 8: Advanced Topics (Sensitivity, Inverse Roofline, Pipeline Composition)

**What the presenter shows:** Sensitivity analysis with tornado charts, inverse Roofline (derive hardware specs from SLA), pipeline composition pattern chaining multiple solvers, and the Fallacies & Pitfalls slide.

**Audience energy level:** LOW-MEDIUM. The content is good but people are tired. The SensitivitySolver and SynthesisSolver are powerful tools, but by this point the audience is saturated with new API calls. The Fallacies & Pitfalls slide gets smiles of recognition.

**Where attention drops:** The Pipeline Composition pattern (slides 8.2--8.3) is the most abstract content in the tutorial. Chaining `ScalingModel -> DistributedModel -> EconomicsModel` is elegant for power users but confusing for beginners who have not yet mastered individual solvers. This should probably be in a "further reading" appendix, not in the main flow.

**What questions come up:**
- "The inverse Roofline is the most useful thing you've shown all day. Why is it buried at 3:30 PM?" (multiple people will think this)
- "Can I use SynthesisSolver to spec out a custom ASIC? Give it my SLA and have it output the minimum FLOPS/BW/memory?" (yes, and this is the killer use case for the architecture audience)

### Part 9: Capstone + Wrap-Up (4:00--5:00)

**What the presenter shows:** The capstone design brief ($5M budget, 1000 QPS, Llama-3-70B, two regions), 35 minutes of pair work, gallery walk presentations, the Iron Law one final time with all exercises mapped, the personal transfer moment, and resources.

**Audience energy level:** RISING during capstone (the challenge creates its own energy), then REFLECTIVE during the personal transfer moment. About 8-10 people will have already left by 4:30 (flights, other sessions). The gallery walk presentations from 2-3 volunteer pairs will be the highlight -- real designs with real numbers that the room can critique.

**Where attention drops:** During the capstone work session itself (4:00--4:35), the energy is distributed unevenly. Strong pairs are racing through it and asking for extensions. Weaker pairs are stuck on step 1 ("Wait, 70B FP16 is 140 GB?") and need instructor help. With 30 pairs and perhaps 2 instructors, coverage is thin.

**What questions come up:**
- "The capstone says 'must survive a full-node failure without dropping below 800 QPS' but we haven't covered the ReliabilityModel at all today. How do we model this?" (legitimate gap -- reliability was only mentioned briefly under "Fragility Wall")
- "Can we use B200s in the capstone? The slides showed them as an option." (yes, and this changes the answer significantly)

---

## Audience Member Feedback (Verbatim Quotes)

### Person 1: PhD student in ML (2nd year, NLP focus)

**What they liked:**

"The prediction exercises were amazing. I have never been so wrong about hardware performance in my life. I genuinely thought the H100 would be 3x faster. When I saw 1.7x, something clicked about why my advisor keeps telling me to think about memory bandwidth. The KV-cache exercise was also eye-opening -- I had no idea my 8B model was burning 4x more memory on KV-cache than on weights at batch 64. That explains why our serving costs keep going up when we increase context length."

**What confused them:**

"I got completely lost during the AllReduce math after lunch. The formula with the ring algorithm and the alpha terms -- I have never implemented distributed training myself, I just use DeepSpeed with default settings. I did not have the background to follow that derivation. I also did not understand why TP should stay within a node -- the NVLink vs InfiniBand distinction was new to me, and the slide went past it in about 30 seconds. The exercises saved me because I could just run the code and see the result, even if I did not fully understand the formula."

**What they would change:**

"I wish there had been a 'map' at the beginning showing which parts are for which audience. The TinyML section was interesting but irrelevant to my work -- I would have preferred 20 more minutes on serving optimization or the inverse Roofline. Also, the compression section before lunch felt rushed. I wanted to understand when INT4 is safe to use for NLP tasks and when it degrades quality, but the accuracy delta was just a rough estimate. I need real benchmark numbers to convince my advisor."

---

### Person 2: Industry engineer from AMD

**What they liked:**

"The multi-vendor TCO shootout in Exercise 5b was the reason I came, and it delivered. Seeing H100, MI300X, and Gaudi3 compared side-by-side using the same analytical framework -- that is exactly what the industry needs. The framework is honest about the numbers rather than cherry-picking benchmarks. I also appreciated that the presenter acknowledged limitations up front. The 'What mlsysim Does Not Model' slide was refreshing. Too many tools oversell themselves."

**What frustrated them:**

"The hardware registry has wrong numbers for the MI300X. It lists 192 GB HBM3, which is correct, but the memory bandwidth number looked off -- it should be 5.3 TB/s, not whatever was in the registry. I flagged this during the exercise and the presenter said 'we will check after the tutorial.' That is not great when the whole point is quantitative accuracy. Also, Exercise 5b uses `getattr(mlsysim.Hardware.Cloud, cluster_name)` which is not the same API pattern used everywhere else in the tutorial. It felt like that exercise was added last-minute."

"The Roofline section only used NVIDIA hardware for examples. Every single slide said 'A100' or 'H100.' I understand Harvard has a relationship with NVIDIA, but at ISCA you have AMD, Intel, Qualcomm, and startup accelerator designers in the room. The tool claims to be hardware-agnostic, but the tutorial tells a different story."

**What they would change:**

"Three things. First, fix the MI300X numbers in the hardware registry before the tutorial. Second, start the Roofline section with a multi-vendor comparison from the beginning -- show A100, H100, and MI300X side by side so the framework feels vendor-neutral. Third, add an exercise where attendees define their own custom hardware spec (not from the registry) and run the analysis. That would be far more useful for accelerator designers than running canned examples on NVIDIA hardware."

---

### Person 3: Faculty member considering adoption (teaches CS261 at Stanford)

**Whether they would adopt it:**

"Conditionally, yes. The pedagogical design is excellent -- the predict-then-reveal cycle, the pair programming, the progressive complexity from single-node to fleet. That is exactly how I want students to learn systems thinking. The backward design document (DESIGN.md) is one of the best tutorial plans I have seen at a systems conference. The transfer goals are well-articulated and measurable."

**What is missing:**

"Three blockers for course adoption. First, there are no auto-graded assignments. I teach 180 students. I cannot grade capstone designs by hand. I need exercises that produce a numeric answer that can be checked programmatically. The current exercises are open-ended, which is great for a tutorial but unusable for a course. Second, the tool is at v0.1.0 and the roadmap lists basic features like GQA support as future work. If I build my course around this and it breaks or changes API, I have a semester-long problem. Third, I need documented error messages. When a student gets a wrong answer, they need to understand why. Right now, if you pass an invalid precision string, what happens? I need graceful failure modes."

**What convinced them:**

"The Iron Law as a unifying equation is brilliant pedagogy. I have been teaching Roofline for years but never connected it to a single master equation that decomposes training time into five terms. The 22-wall taxonomy gives me a semester-long curriculum structure -- I can assign two walls per week and cover all of them. If v0.2.0 ships with GQA, MoE, and autograded exercises, I would build my fall course around this."

---

### Person 4: Startup CTO (inference startup, making hardware decisions)

**Whether the tool is useful for their work:**

"Partially. The inverse Roofline is the most directly useful feature -- I can give it our latency SLA and it tells me the minimum hardware specs. That saves us weeks of benchmarking. The fleet design capstone is also relevant -- we are literally making the 'H100 vs MI300X vs B200' decision right now, and seeing the TCO comparison in 10 seconds instead of 10 days is valuable."

"But I cannot use this for production planning yet. The serving model does not account for continuous batching, prefix caching, or speculative decoding. Our actual serving stack uses vLLM with all of those optimizations, and they change the capacity numbers by 2-3x. The mlsysim serving model gives me the physics-based floor, which is useful for sanity-checking, but not for capacity planning."

**What they would need next:**

"Three things that would make me pay for this tool. First, a 'calibration report' that shows how mlsysim predictions compare to real MLPerf results or published Llama training reports. I need to know the error bars before I trust it for a $10M hardware purchase. Second, support for speculative decoding and continuous batching in the serving model -- these are table-stakes for any inference company. Third, an API that takes a HuggingFace model card URL and auto-populates the model spec. Right now I have to manually enter parameter counts, layer counts, and head dimensions. That is error-prone and slow."

---

### Person 5: PhD student in computer architecture (accelerator design)

**What surprised them:**

"Two things genuinely surprised me. First, the sensitivity analysis showing that improving a non-binding parameter gives zero gain. I know this intellectually from Amdahl's Law, but seeing it applied to a real ML workload where doubling FLOPS gives 0% speedup because the workload is memory-bound -- that was a visceral demonstration. I have been designing an accelerator with 2x more compute units, and now I am questioning whether I should reallocate that die area to more HBM channels."

"Second, the carbon geography result. I design hardware and I have never once thought about where the datacenter is located. I optimize for performance per watt, but performance per gram of CO2 is a completely different metric, and it is dominated by the grid, not the chip."

**What they already knew:**

"The Roofline model, obviously. I have drawn hundreds of Roofline plots. The AllReduce math was also familiar -- I took the Berkeley parallel computing course. The KV-cache derivation was straightforward. The 3D parallelism taxonomy was review."

**What new insight they got:**

"The inverse Roofline -- what the tutorial calls 'Wall 22: Synthesis' -- is the tool I have wanted for my entire PhD. I design accelerators to meet a target workload. I have always done this by forward simulation: build a cycle-accurate model, run the workload, check if it meets the SLA. The inverse Roofline flips this: give me the SLA, and I will tell you the minimum specs. That eliminates 80% of my design space exploration. I am going to build a custom `HardwareNode` for my accelerator design and use mlsysim for rapid what-if analysis. If the numbers come within 20% of my cycle-accurate simulator, I will use mlsysim for the early design phase and only run the expensive simulator for the final design."

"I also appreciated the TinyML section more than most people seemed to. The fact that the same Roofline equation works for a $2 nRF52840 and a $30K H100 validates the analytical approach. It means my accelerator, which sits somewhere in between, is also modelable."

---

## Top 5 Improvements Identified

### 1. The `efficiency` parameter needs a principled explanation

The tutorial uses `efficiency=0.5` or `efficiency=0.3` throughout, but never explains where these numbers come from or how to choose them for a new workload. This is the most commonly used parameter in every exercise, and it is treated as a magic constant. The tutorial needs a slide that says: "MFU for state-of-the-art training is 0.3-0.55 (PaLM reported 0.46). For naive PyTorch code, expect 0.1-0.2. For optimized inference with FlashAttention, expect 0.5-0.7. Here is how to measure your own." Without this, every exercise result feels ungrounded.

### 2. Compression section is too rushed before lunch

Module 3 gets only 15 minutes and is placed in the worst slot (right before lunch). The "compression is architecture, not optimization" thesis (U4) is one of the six enduring understandings, but it gets less time than any other. The exercise gets 7 minutes. Recommendation: Either move compression to after lunch (combine it with the fleet-halving exercise in Module 5, where it has more impact) or extend it to 25 minutes by shortening the Roofline generalization lecture.

### 3. AllReduce math needs concrete numbers, not just symbols

The ring AllReduce formula on the whiteboard at 1:25 PM will lose half the room. The fix is simple: before showing the formula, show a concrete example. "8 GPUs, NVLink at 900 GB/s, 1 GB of gradients. How long does AllReduce take? Let us calculate: 2 * (8-1)/8 * 1 GB / 900 GB/s = 1.94 ms." Then show the formula. Numbers first, algebra second. The current tutorial does algebra first, which is backwards for a post-lunch audience.

### 4. TinyML section should be optional or restructured

The TinyML section (Part 7) is the weakest segment for the ISCA target audience. While the "same Roofline, different physics" message is elegant, it takes 20+ minutes to make a point that could be made in one slide. The architecture researchers who care about edge hardware already know this. The ML researchers do not care. Recommendation: Compress TinyML to a single 5-minute "same equation, 9 orders of magnitude" demo, and use the recovered time for: (a) a deeper inverse Roofline exercise, (b) a "bring your own workload" segment, or (c) more capstone time.

### 5. Hardware registry accuracy must be verified before the tutorial

If the MI300X bandwidth number is wrong, or the B200 specs are outdated, the entire "quantitative reasoning from first principles" thesis collapses. An ISCA audience will check every number against the datasheet. Recommendation: Create a pre-tutorial validation script that compares every hardware entry against a canonical source (vendor spec sheets). Run it the week before the tutorial. Print the hardware spec table in the cheatsheet so attendees can verify during exercises.

---

## "Would You Recommend This Tutorial?"

| Person | Score (0-10) | Would Recommend? | One-Line Rationale |
|--------|-------------|-------------------|---------------------|
| 1. ML PhD student | 8 | Yes | "The prediction exercises fundamentally changed how I think about hardware. Lost during AllReduce math but the exercises carried me." |
| 2. AMD engineer | 6 | Maybe | "Good framework, but the NVIDIA-centric examples and the wrong MI300X numbers undermine credibility at a vendor-neutral venue." |
| 3. Stanford faculty | 8 | Yes (conditional) | "Best pedagogical design I have seen in a tutorial. Will adopt if v0.2.0 ships with auto-graded exercises and GQA support." |
| 4. Startup CTO | 7 | Yes | "The inverse Roofline alone was worth the trip. Need calibration data and speculative decoding support to use in production." |
| 5. Architecture PhD | 9 | Yes | "The inverse Roofline and sensitivity analysis are tools I will use in my research immediately. The framework validates analytically what I spend weeks simulating." |

---

## Net Promoter Score Estimate

Using the standard NPS methodology (9-10 = Promoter, 7-8 = Passive, 0-6 = Detractor):

- **Promoters (9-10):** ~20% of room (architecture researchers, faculty evaluating adoption, systems PhD students)
- **Passives (7-8):** ~50% of room (ML PhD students who liked the exercises, industry engineers who found it useful but incomplete, startup people who see potential)
- **Detractors (0-6):** ~30% of room (industry engineers frustrated by hardware inaccuracies, ML researchers who found TinyML irrelevant, people who could not get pip install working and spent the morning paired with a stranger watching them code)

**Estimated NPS: -10 to +5**

This is a realistic score for a first-time tutorial at a top venue. The content quality is high (the backward design is exceptional), but first-time execution issues -- installation friction, pacing miscalculations, hardware registry inaccuracies, and uneven section depths -- drag down the score. A second delivery of this tutorial, with the five improvements above implemented, would likely score NPS +25 to +35.

**The honest summary:** The tutorial has a 9/10 design and a 6/10 execution readiness. The gap is entirely fixable with two weeks of focused preparation. The biggest risk is not content quality -- it is operational: installation failures, wrong hardware numbers, and the compression section getting squeezed by lunch.
