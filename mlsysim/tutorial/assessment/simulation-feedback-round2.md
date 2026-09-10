# ISCA 2026 Tutorial Simulation Feedback: Round 2

**Tutorial:** First-Principles Performance Modeling for ML Systems: From Roofline to Fleet Design with mlsysim
**Simulated audience:** Same 60-attendee mix as Round 1
**Date of simulation:** 2026-04-01
**Round 1 reference:** `simulation-feedback-round1.md`
**Fixes under evaluation:** 3 of 5 Round 1 issues addressed

---

## Fixes Implemented Since Round 1

| # | Round 1 Issue | Fix Applied | Slide Reference |
|---|---------------|-------------|-----------------|
| 1 | `efficiency` parameter unexplained | Added "What Is eta?" slide (1.8b) with efficiency table and CPI analogy | Slide 1.8b, lines 606--651 |
| 2 | AllReduce math loses post-lunch audience | Added "AllReduce: A Concrete Example" slide with 8-GPU numbers-first walkthrough | Slide 4.3b, lines 1723--1752 |
| 3 | Compression section too rushed, no fleet impact | Added "Compression Changes Fleet Architecture" slide with Llama-3 70B cost table | Slide 3.7b, lines 1542--1568 |
| 4 | TinyML section should be optional/restructured | **NOT FIXED** | Still 20+ min in Part 7 |
| 5 | Hardware registry accuracy unverified | **NOT FIXED** (but MI300X BW now shows 5.3 TB/s in slides) | Slides 1.6, 1.13 |

**Additional change observed:** The Roofline section (Part 1) was substantially reworked. The first prediction exercise is now "H100 vs MI300X vs Gaudi 3" (three vendors from the start), replacing what was previously an NVIDIA-only A100 vs H100 comparison. The live demo also loops over all three vendors. This directly addresses Person 2's concern about NVIDIA-centric examples, though it was not listed as one of the three "fixed" items.

---

## Persona Re-Simulations

### Person 1: ML PhD Student (2nd year, NLP focus)

**Context from Round 1:** Scored 8/10. Loved prediction exercises. Got "completely lost" during AllReduce math. Found compression rushed. Did not understand where efficiency=0.5 came from.

#### Re-experiencing the eta slide (Slide 1.8b)

The eta slide lands well for this persona. The two-column layout -- "What reduces eta" on the left, "Typical values" table on the right -- gives her exactly the mental model she was missing. The table showing "Training (PyTorch eager): 0.08--0.15" vs "Training (Megatron-LM): 0.40--0.55" is the kind of concrete anchor she needs. When the first demo uses `efficiency=0.5`, she now knows that corresponds to "optimized training" and is not an arbitrary guess.

The CPI analogy in the speaker notes ("eta is to ML systems what CPI is to CPU design") is less helpful for this persona -- she took an ML degree, not an architecture degree. She may not know what CPI is. But the table itself compensates. The bottom-of-slide message ("You do not predict eta -- you measure it once and use it for what-if analysis") is the single most important sentence for her mental model. It reframes eta from "magic constant I need to derive" to "empirical measurement I look up."

**Remaining confusion:** The table shows ranges (0.40--0.55) but the exercises use point values (0.5). She will still wonder: "Should I use 0.40 or 0.55?" The slide does not explain that the answer is "either -- you are doing what-if analysis, not prediction." A single sentence in the speaker notes addressing this would close the gap.

**New concern introduced by the fix:** The eta slide adds 3 minutes to Part 1. Part 1 was already the longest section (60 minutes). The 10-minute "generalization" lecture that Round 1 flagged as an energy dip (10:10--10:20) is still there. Adding the eta slide before it makes the lecture-heavy opening stretch even longer before students touch code. Net effect: the eta explanation is worth the 3 minutes, but the generalization lecture should be trimmed by 3 minutes to compensate. This was not done.

#### Re-experiencing the AllReduce concrete example (Slide 4.3b)

Significant improvement. The "Setup: 8 H100 GPUs, NVLink at 900 GB/s, Llama-3 8B (16 GB gradients)" framing gives her a physical picture before any math appears. The three-step enumeration (each GPU computes gradients, all must end up with the same average, ring passes chunks) is the narrative scaffolding that was missing. The code block showing `calc_ring_allreduce_time()` with named parameters lets her map each number to a physical quantity.

The punchline "35 ms to synchronize 8 GPUs. Now: what happens at 256 GPUs?" is effective because it gives her a concrete baseline. When the next slide (4.4) shows the formula and the 256-GPU number jumps to 40 ms on InfiniBand, she has the intuition to understand why: different interconnect bandwidth.

**Remaining confusion:** The concrete example uses 16 GB of gradients (Llama-3 8B in FP16), but the next slide (4.4) switches to "1 GB gradients" for the formula example. She will wonder why the numbers changed. This is a minor continuity error that creates a 5-second "wait, what?" moment. Not a dealbreaker, but avoidable.

**What she would say now:** "The concrete example before the formula made a huge difference. I could follow the math because I already knew the answer had to be in the tens-of-milliseconds range. I still do not fully understand the ring algorithm, but I understand the result. That is enough to interpret the exercises."

#### Clarity improvement: 8 --> 9 (out of 10)

The eta slide and AllReduce example together close her two biggest gaps. She goes from "lost during AllReduce and confused by eta" to "I can follow the reasoning and use the tool meaningfully." The 1-point gap from 10 is the remaining issues with range vs point values for eta and the gradient size continuity break.

#### Would she NOW recommend the tutorial?

**Yes, enthusiastically.** Moves from "Yes" to "Yes, and I would bring my labmates." The prediction exercises were already her favorite part; now the explanatory scaffolding matches the exercise quality.

---

### Person 2: Industry Engineer from AMD

**Context from Round 1:** Scored 6/10 ("Maybe" recommend). Frustrated by NVIDIA-centric examples, wrong MI300X bandwidth number, inconsistent API in Exercise 5b.

#### Re-experiencing the multi-vendor Roofline (Slides 1.6, 1.7)

This is the biggest improvement for this persona. The first prediction exercise is now explicitly "H100 vs MI300X vs Gaudi 3" with a three-column table. The MI300X bandwidth is listed as 5.3 TB/s (correct). The MI300X wins the comparison (1.6x vs H100) because of bandwidth advantage despite having fewer FLOPS than Gaudi 3. This is exactly the message AMD wants the ISCA audience to hear -- and it is backed by physics, not marketing.

The live demo (Slide 1.7) looping over `["H100", "MI300X", "Gaudi3"]` with the same API makes the framework feel genuinely vendor-neutral. The speaker note explicitly says "This is ISCA -- show that mlsysim is not an NVIDIA-only tool." Good.

The multi-vendor Roofline table (Slide 1.13) now includes B200, H100, MI300X, and Gaudi 3 with ridge points. The MI300X ridge of 247 FLOP/B vs H100's 295 FLOP/B shows that MI300X has a lower ridge (more workloads are compute-bound on MI300X), which is a nuanced and accurate insight.

**What he is satisfied with:** The framework now treats AMD hardware as a first-class citizen from the very first exercise. The numbers appear correct. The ranking (MI300X > Gaudi 3 > H100 for memory-bound decode) follows from bandwidth, and the slides make this explicit. He no longer feels like he is watching an NVIDIA advertisement.

**Remaining frustration:** Two things. First, the hardware registry accuracy issue (#5) is still unresolved as a systemic concern. The slides now show correct numbers, but the question is whether the underlying `mlsysim.Hardware.Cloud.MI300X` object also has 5.3 TB/s. If the slides say 5.3 but the code returns a different number, the live demo will contradict the slide -- which is worse than getting it wrong consistently. He will test this during the first exercise. Second, his Round 1 request for a "define your own custom hardware" exercise was not addressed. The exercises still use canned hardware from the registry. For an ISCA audience of accelerator designers, this is a missed opportunity.

**New concern introduced by the fix:** The three-vendor prediction slide (1.6) is dense. It has a 3-column table with 6 rows plus a reveal. At `\scriptsize` font in a 169 aspect ratio, this is readable but tight. Some attendees in the back rows of a large ISCA room may struggle with the numbers. A larger font with fewer rows (drop "HBM Capacity" since it is not relevant to the decode comparison) would help.

#### Clarity improvement: 6 --> 8 (out of 10)

The vendor-neutral framing is a transformative improvement. He goes from "this feels like an NVIDIA tutorial" to "this framework respects all vendors." The 2-point gap from 10 is the unresolved hardware registry verification (systemic trust issue) and the missing custom-hardware exercise.

#### Would he NOW recommend the tutorial?

**Yes, with caveats.** Moves from "Maybe" to "Yes, if the hardware registry numbers are verified before the live tutorial." He would tell AMD colleagues: "The framework is fair. The numbers check out. Go see it." But he would add: "Verify the MI300X numbers in the code match the slides before you trust the tool for real analysis."

---

### Person 3: Stanford Faculty (CS261, considering adoption)

**Context from Round 1:** Scored 8/10 ("Yes, conditional"). Loved the pedagogical design. Needed auto-graded assignments, API stability, and error messages. Found the "compression is architecture" thesis under-supported.

#### Re-experiencing the compression fleet economics slide (Slide 3.7b)

This is the slide she was waiting for. The Llama-3 70B cost table is clean and devastating:

| Precision | GPUs Needed | Annual Cost |
|-----------|------------|-------------|
| FP16      | 4 (TP=4)   | $480K       |
| INT4      | 1          | $120K       |

The extrapolation to fleet scale ("At 100 replicas: 300 fewer GPUs and $36M saved per year") makes the architectural argument concrete and teachable. She can put this exact table on a homework assignment: "Given these costs, at what annual query volume does INT4 quantization pay for itself even if you lose 2% accuracy?" That is a measurable, gradable question.

The bottom-of-slide message ("This is why quantization is a Day 1 architectural decision, not a Day 100 optimization") is the sentence she would quote in her lecture. It reframes compression from a model quality concern to an infrastructure planning decision. This is exactly the "compression as architecture" thesis that Enduring Understanding U4 claims.

**What convinces her:** The cost table is the missing quantitative backbone. Round 1's compression section said "compression is architecture" but only showed latency and model-size numbers. Latency improvement alone does not justify calling something "architecture." Fleet-level cost reduction does. She can now build a 2-week module around this: Week 1 covers the physics (why INT4 is still memory-bound), Week 2 covers the economics (fleet sizing, cost modeling, accuracy-cost Pareto frontier).

**Remaining concern:** The compression section is still in the pre-lunch slot. Slide 3.7b adds 3 minutes of content, but the total section time was already tight at 15 minutes. With the new slide, the exercise (Exercise 3) now gets even less time -- perhaps 5 minutes instead of 7. She would prefer the section to be 20--25 minutes with the fleet economics as the capstone, not squeezed into the lunch-rush slot. The original Round 1 recommendation to move compression after lunch (or extend it) was not implemented.

**Her three Round 1 blockers remain:**
1. Auto-graded assignments: still not available.
2. API stability (v0.1.0 risk): still a concern.
3. Error messages for students: still unaddressed.

None of these were in scope for the three fixes, but they remain the actual adoption blockers. The compression slide improves her assessment of the content quality, not her assessment of the tool's course-readiness.

**New concern introduced by the fix:** The cost table uses "$480K" and "$120K" annual costs. She will ask: "Where do these numbers come from? Are they AWS on-demand, reserved instances, or self-hosted? The TCO multiplier matters." If the answer is "we assumed $3/GPU-hour on-demand," that is fine but should be stated. An ungrounded cost number in a tutorial about quantitative rigor is a credibility risk, especially for a faculty member who will have 180 students asking the same question.

#### Clarity improvement: 8 --> 9 (out of 10)

The fleet economics table elevates the compression section from "rushed but interesting" to "teachable and compelling." She now has a quantitative anchor for the "compression as architecture" thesis. The 1-point gap is the timing issue (still pre-lunch, still rushed) and the missing cost assumptions.

#### Would she NOW recommend the tutorial?

**Yes, more strongly.** Her recommendation upgrades from "conditional on v0.2.0 features" to "I would attend this myself and send my TAs." She would still not build a full course around mlsysim v0.1.0, but she would assign the tutorial exercises as a 1-week supplementary module in CS261. The compression slide gives her the teaching material she was missing.

---

## Round 1 Issue Status (All 5)

| # | Issue | Status | Evidence |
|---|-------|--------|----------|
| 1 | Efficiency (eta) unexplained | **RESOLVED** | Slide 1.8b provides table of typical values, CPI analogy, and the key framing ("measure, don't predict"). Minor gap: range vs point value guidance. |
| 2 | Compression section rushed | **PARTIALLY RESOLVED** | Slide 3.7b adds the missing fleet economics argument. But the section is still 15 min pre-lunch, and the new slide compresses exercise time further. Timing was not adjusted. |
| 3 | AllReduce needs concrete numbers | **RESOLVED** | Slide 4.3b provides numbers-first example with physical setup before the formula. Minor gap: gradient size changes between concrete example (16 GB) and formula slide (1 GB). |
| 4 | TinyML section should be restructured | **NOT ADDRESSED** | Part 7 remains 20+ minutes with Flash vs SRAM details that lose the LLM-focused majority. |
| 5 | Hardware registry accuracy | **PARTIALLY RESOLVED** | Slides now show correct MI300X bandwidth (5.3 TB/s) and include multi-vendor tables. But no validation script exists to verify the code registry matches the slides. The systemic concern remains. |

---

## New Issues Introduced by Fixes

### New Issue A: Part 1 pacing is now tighter

The eta slide (3 min) was added to Part 1 without removing or shortening anything else. Part 1 was already 60 minutes. The "generalization" lecture at 10:10--10:20 that Round 1 flagged as an energy dip remains. Net effect: Part 1 is now 63 minutes, increasing time pressure on the break schedule. **Severity: Low.** The eta slide is worth more than 3 minutes of the generalization lecture. Fix: trim the generalization lecture by 3 minutes.

### New Issue B: Exercise 3 time is further compressed

The compression section gains Slide 3.7b (3 min) but does not gain total time. Exercise 3 drops from ~7 minutes to ~4--5 minutes. Many pairs will not finish. **Severity: Medium.** The fleet economics slide is more valuable than the exercise minutes it displaces, but the exercise is where learning happens. Fix: extend Part 3 to 20 minutes by shortening the Part 1 generalization lecture.

### New Issue C: Gradient size discontinuity between Slides 4.3b and 4.4

Slide 4.3b uses 16 GB gradients (Llama-3 8B). Slide 4.4 switches to 1 GB gradients for the formula derivation. This creates a minor continuity break that will confuse 5--10 people for about 10 seconds each. **Severity: Low.** Fix: use 16 GB in both slides, or add a one-line note: "Using 1 GB for cleaner arithmetic."

### New Issue D: Fleet cost table lacks assumptions

Slide 3.7b states "$480K" and "$120K" annual costs without specifying the pricing model (on-demand, reserved, self-hosted, which cloud provider). For a tutorial built on quantitative rigor, ungrounded dollar figures invite skepticism. **Severity: Medium** for the faculty audience, **Low** for others. Fix: add a footnote or speaker note stating "Assumes $3/GPU-hour on-demand (representative cloud pricing)."

---

## Updated Scoring

| Person | Round 1 Score | Round 2 Score | Change | Would Recommend? |
|--------|--------------|--------------|--------|------------------|
| 1. ML PhD student | 8 | 9 | +1 | Yes (enthusiastically) |
| 2. AMD engineer | 6 | 8 | +2 | Yes (with caveats on registry verification) |
| 3. Stanford faculty | 8 | 9 | +1 | Yes (would attend herself; still conditional for full course adoption) |

---

## Updated NPS Estimate

**Round 1 NPS: -10 to +5**

Recalculating with the fixes applied:

- **Promoters (9--10):** ~30% of room (up from ~20%). The multi-vendor framing converts AMD/Intel-sympathetic attendees from detractors to passives or promoters. The eta and AllReduce fixes reduce confusion-driven detraction.
- **Passives (7--8):** ~50% of room (stable). The ML PhD students and startup CTOs remain in this band -- the fixes help but do not transform their experience.
- **Detractors (0--6):** ~20% of room (down from ~30%). The vendor-neutral opening eliminates the "NVIDIA advertisement" detractors. Remaining detractors: pip install failures, TinyML irrelevance crowd, people who need features not yet in v0.1.0.

**Updated NPS: +10 to +20**

This is a meaningful improvement from the Round 1 estimate of -10 to +5. The single largest driver is the multi-vendor Roofline opening, which prevents the early credibility loss that would have cascaded through the rest of the day. The eta and AllReduce fixes reduce mid-tutorial confusion, and the fleet economics slide gives the compression section its missing punchline.

The remaining gap to NPS +30 (which would be excellent for a first-time ISCA tutorial) requires addressing:
1. TinyML restructuring (Issue #4) -- the easiest remaining win
2. Hardware registry validation script (Issue #5) -- the highest-risk remaining gap
3. Compression section timing -- move it or extend it
4. The four new issues (A--D) identified above

---

## Summary

The three fixes address the right problems and are well-executed. The eta table, the concrete AllReduce example, and the fleet economics slide each provide the quantitative scaffolding that was missing. The bonus multi-vendor Roofline rework is arguably the most impactful change, converting the AMD engineer from a detractor to a promoter.

The fixes do introduce minor new issues (pacing pressure in Part 1, exercise time compression in Part 3, gradient size discontinuity, ungrounded cost assumptions), but none are severe. They are the kind of second-order problems that naturally emerge when you add content without removing content -- solvable with a 30-minute editing pass.

**Overall readiness assessment:** Round 1 was "9/10 design, 6/10 execution readiness." Round 2 is **9/10 design, 7.5/10 execution readiness.** The remaining 2.5 points are: TinyML restructuring, hardware registry validation, compression timing, and the four minor new issues.
