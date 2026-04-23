# Simulated Q&A and Hallway Conversations -- ISCA 2026 Tutorial

**Purpose:** Stress-test the tutorial's claims before a live audience finds the weak spots.
**Date generated:** 2026-04-01
**Methodology:** Each question is drawn from a real archetype (the skeptic, the competitor,
the domain expert, the methodologist). Answers are the presenter's honest best response
given the current state of mlsysim v0.1.0 and the calibration data in `empirical-calibration.md`.

---

## Part 1: Tough Questions (Post-Session Q&A)

---

### Q1: "How is this different from a spreadsheet?"

**Questioner archetype:** Senior engineer, slightly bored, has seen a hundred tool demos.

**The question (verbatim):**
"I appreciate the presentation, but I could do all of this in a spreadsheet. I have
a Google Sheet where I plug in FLOPS, bandwidth, and model size and get the same
answers. What does mlsysim give me that Excel doesn't?"

**The best honest answer:**

You are right that any single equation in mlsysim can be reproduced in a spreadsheet.
People do this all the time, and for one-off calculations it works fine. The differences
show up at scale:

1. **Unit safety.** mlsysim uses Pint for dimensional analysis. Every quantity carries
   its unit. If you accidentally add GB to TFLOPS, you get a `DimensionalityError` at
   the point of the mistake, not a silently wrong number three rows down. Spreadsheets
   have no type system -- a cell is a cell. We have seen real production capacity
   planning errors that came from mixing GB and GiB in a spreadsheet, or confusing
   FLOP/s with FLOPs (rate vs count). Those errors are structurally impossible in mlsysim.

2. **Composability across 22 walls.** Your spreadsheet probably covers 3-4 constraints.
   mlsysim composes all 22 into a single `Pipeline.solve()` call. The capstone exercise
   we just did -- throughput, latency, budget, carbon, fault tolerance, all simultaneously --
   would be a nightmare in a spreadsheet because the constraints interact. Changing precision
   changes memory, which changes whether you need tensor parallelism, which changes
   communication cost, which changes fleet size, which changes carbon. That cascade is
   what the pipeline handles.

3. **Traceability.** Every hardware constant in mlsysim is a `TraceableConstant` with a
   source, date, and DOI. When your spreadsheet says "H100 bandwidth = 3350 GB/s," where
   did that number come from? Is it the datasheet peak or the measured sustained? Is it
   HBM3 or HBM3E? In mlsysim, you can audit any number back to its origin.

But I want to be clear: if you have a well-maintained spreadsheet that solves your problem,
keep using it. mlsysim is most valuable when you are doing comparative analysis across
many hardware platforms, or when you need to hand your analysis to someone else and they
need to trust the numbers.

**Answer quality: 7/10**

**What would make it stronger:** A live side-by-side demo showing a real spreadsheet error
that mlsysim catches. The unit-safety argument is strong in theory but needs a concrete
"I had a $200K capacity planning error because of a GiB/GB confusion" war story. Without
the anecdote, the senior engineer thinks "I just label my columns carefully." Also: the
22-wall composition claim is aspirational -- v0.1.0 does not actually compose all 22 in
a single solve call yet. The pipeline exists for compute/memory/communication/cost/carbon,
but walls like Tail Latency (Erlang-C), Multi-tenant (queueing), and Safety (DP-SGD
overhead) are not yet wired into the solver. Being caught overstating coverage would be
worse than admitting the gap.

---

### Q2: "Your accuracy is 2-5x off. Why wouldn't I just benchmark?"

**Questioner archetype:** Industry engineer who runs real clusters. Pragmatist.

**The question (verbatim):**
"Your own slides say the accuracy is within 2-5x of measured performance. That is
an order of magnitude. If I need to make a hardware purchasing decision, I need
numbers I can trust. Why would I use something that might be off by 5x instead of
just running a benchmark?"

**The best honest answer:**

That is a fair challenge, and I want to be precise about what "2-5x" means and
where it comes from.

First, the 2-5x number is our worst case across all configurations with default
efficiency parameters. The calibration table in our docs shows the actual spread.
For LLM decode latency -- the serving use case most people care about -- we are
within the published range, not 2-5x off, because decode is memory-bandwidth-bound
and the model correctly computes `weights / bandwidth`. There is no efficiency
parameter on the critical path. For CNN training throughput, the default eta produces
predictions that are 22% low on A100 and 54% high on H100. That is where the "2-5x"
envelope comes from.

Second, benchmarking is always better when you can do it. The question is: can you?
Benchmarking requires having the hardware, which means you have already purchased it
or negotiated cloud access. It requires having the software stack working on that
hardware, which for a new platform can take weeks. And it gives you one data point:
this model, on this hardware, at this batch size, with this framework version.

mlsysim is for the phase before benchmarking: "Should I even request time on this
cluster?" "Is it worth porting to AMD MI300X, or is it obviously bandwidth-starved
for my workload?" "How many GPUs do I need to request in my cloud allocation?"
If mlsysim tells you a workload is 3x over the memory capacity of a given GPU,
you do not need to benchmark to know it will not work.

Third, with a single calibrated measurement, accuracy improves dramatically. If you
measure eta on one hardware platform, the model matches benchmarks within 1-5%
for that platform, and gives you a reasonable starting point for others. The workflow
is: benchmark once, calibrate eta, then use mlsysim for the design space exploration.

**Answer quality: 8/10**

**What would make it stronger:** The answer is honest and the "benchmark once, explore
with mlsysim" workflow is compelling. The weak spot is that the calibration doc shows
eta does NOT transfer across GPU generations (A100 eta=0.13 vs H100 eta=0.065 for the
same ResNet-50 workload). The presenter should own this explicitly: "Cross-generation
transfer is the known weak point. We are within 2x, not 1%, when transferring eta across
architectures." Overpromising on cross-platform accuracy would get destroyed in peer
review.

---

### Q3: "The efficiency parameter is just a fudge factor, right?"

**Questioner archetype:** Architecture PhD student. Technically sharp. Wants to
understand the epistemology.

**The question (verbatim):**
"You showed that you calibrate eta per-benchmark to match published numbers.
ResNet-50 on A100 needs eta=0.13, ResNet-50 on H100 needs eta=0.065. You call
this the CPI analogy, but CPI is measured -- you back-calculate eta from the
answer you want to predict. Isn't that circular? It is a fudge factor."

**The best honest answer:**

I want to take this seriously because it is the right question.

You are correct that when we calibrate eta per-benchmark, we are fitting a
single parameter to match observations. That is definitionally a fudge factor
if the only thing we do with it is reproduce the number we already measured.

The value is not in reproducing the known number. The value is in what you can
do *after* calibration. Once you have eta=0.13 for ResNet-50 on A100, you can
ask: "What happens if I double the batch size? What if I switch to FP8? What
if I add pipeline parallelism?" The model makes predictions for those
counterfactuals that are constrained by physics -- the FLOP count changes, the
memory footprint changes, the communication volume changes -- and eta carries
forward as the empirical correction.

The CPI analogy is precise. Patterson and Hennessy measure CPI from SPEC
benchmarks. You could call CPI a fudge factor too -- it absorbs cache miss
rates, branch mispredictions, pipeline hazards, and a dozen other things into
one number. The reason it is useful is not that CPI is predictable from first
principles. It is that the performance equation `Time = Instructions x CPI x
Clock_Period` lets you reason about what happens when you change the ISA
(Instructions change), or the clock frequency (Clock_Period changes), or the
microarchitecture (CPI changes). Each term is independently variable.

Where this analogy breaks down -- and I should be honest about this -- is
transferability. CPI for a given benchmark on a given ISA transfers reasonably
well across microarchitectures within a generation. Our calibration data shows
that eta does NOT transfer well across GPU generations for the same workload.
ResNet-50 gets eta=0.13 on A100 and eta=0.065 on H100. That is a 2x difference,
and it means you cannot use eta measured on A100 to predict H100 performance
without significant error.

We think this is a real and important limitation, and we document it explicitly.
The fix is not to pretend eta transfers -- it is to build a richer efficiency
model that decomposes eta into sub-factors (kernel utilization, memory system
efficiency, framework overhead) that transfer independently. That is future work.

**Answer quality: 9/10**

**What would make it stronger:** This is the best answer in the set because it
concedes the legitimate criticism, gives the honest intellectual defense, and
names the specific failure mode. The only improvement would be showing preliminary
results from a decomposed eta model -- even a two-factor version (compute
utilization + memory utilization) that transfers better. Without that, the "future
work" claim is promissory.

---

### Q4: "Does this work for MoE models like Mixtral?"

**Questioner archetype:** Applied ML researcher working on MoE architectures.

**The question (verbatim):**
"All your examples are dense models. Mixture-of-Experts changes the arithmetic
intensity dramatically -- only 2 of 8 experts are active per token in Mixtral,
so the active parameter count is 12B but the total is 46B. The memory footprint
is 46B but the compute is 12B. Does mlsysim handle this?"

**The best honest answer:**

Not natively in v0.1.0, and this is an important gap.

You are exactly right about the analysis. MoE creates a fundamental decoupling
between memory footprint and compute that breaks the assumption of most analytical
models, including ours. For dense models, parameters and FLOPs are tightly coupled
via the `6ND` rule. For MoE, you need to track them separately: all experts must
be resident in memory (46B for Mixtral), but only the active experts contribute
FLOPs per token (roughly 12B worth).

In the current version, you could model Mixtral by manually setting the model
parameters to 46B (for memory calculations) and overriding the FLOPs to match
the active expert count. That is a workaround, not proper support.

What proper MoE support requires is: (1) separating the memory model from the
compute model in the solver, which is an architectural change; (2) modeling the
expert routing overhead -- the gating network, the all-to-all communication in
distributed MoE where different tokens route to different GPUs; and (3) modeling
the load imbalance problem, where popular experts become bottlenecks.

The expert routing communication pattern is particularly important for ISCA
audiences. Dense models use AllReduce (symmetric, bandwidth-optimal). MoE uses
All-to-All (asymmetric, sensitive to load balance). The communication wall looks
completely different.

This is on our roadmap. MoE is the single most requested feature.

**Answer quality: 6/10**

**What would make it stronger:** The answer correctly identifies the gap and shows
domain understanding, but "it is on our roadmap" is the weakest possible ending
at ISCA. A concrete timeline ("v0.2 in Q4 2026") or a branch with preliminary
MoE support would be much stronger. Even better: have the workaround as a prepared
code snippet that the questioner can try immediately. Saying "manually override the
FLOPs" without showing the 5-line code example makes it feel like a hand-wave.

---

### Q5: "Why should I use this instead of Calculon?"

**Questioner archetype:** Someone who has actually used Calculon from HPE/LLNL
for training performance modeling.

**The question (verbatim):**
"Calculon already does analytical training performance modeling. It handles 3D
parallelism, pipeline bubbles, and communication overlap. It was validated against
Megatron-LM at scale. Why should I switch?"

**The best honest answer:**

You should not switch. You should use both, for different questions.

Calculon is excellent at what it does: training performance modeling for large
language models with 3D parallelism. It was built by people who run some of the
largest training clusters in the world, and it shows. If your question is "what
is the optimal parallelism configuration for training GPT-4-scale models on
2048 H100s," Calculon is probably the better tool right now. It models pipeline
bubble fractions, communication-computation overlap, and micro-batch scheduling
in more detail than mlsysim does.

mlsysim makes different trade-offs:

1. **Breadth over depth.** Calculon covers training on NVIDIA hardware. mlsysim
   covers training, inference, serving, TinyML, cost, carbon, and sustainability
   across five vendors. If your question is "should I deploy this model on H100,
   MI300X, or Gaudi 3, and what is the carbon footprint of each option," Calculon
   cannot help you.

2. **Inference and serving.** Calculon does not model the two-phase serving regime
   (prefill/decode), KV-cache memory pressure, or tail latency under load.
   mlsysim does.

3. **Unit safety and traceability.** This is a differentiator for pedagogical and
   audit use cases. Every number in mlsysim is dimensionally typed and traceable
   to a source.

4. **Pedagogical design.** mlsysim was designed for teaching. The API is intentionally
   simple: `Engine.solve(model, hardware, batch_size)`. Calculon is designed for
   research-grade modeling, which means a steeper learning curve and more
   configuration.

Where Calculon wins cleanly: training-specific fidelity, communication overlap
modeling, validation at real scale (thousands of GPUs with real Megatron-LM
measurements). We respect that work enormously.

The honest positioning is: mlsysim is a broader, shallower tool. Calculon is a
narrower, deeper one. Use mlsysim for rapid design-space exploration across the
full stack. Use Calculon for detailed training performance prediction once you
have narrowed the design space.

**Answer quality: 8/10**

**What would make it stronger:** This answer is good because it does not trash
the competitor. The risk is that "broader but shallower" sounds like "worse at
everything Calculon does." The presenter should have a prepared example where
breadth matters: "A startup choosing between H100 and MI300X for a serving
workload cannot use Calculon at all. mlsysim gives them a quantitative answer
in under a second." The serving use case is the clearest differentiator --
lean into it hard.

---

### Q6: "All your examples are Llama and ResNet. What about diffusion models?"

**Questioner archetype:** Computer vision researcher working on generative models.

**The question (verbatim):**
"You showed Llama-3-8B and ResNet-50 in every exercise. Those are Transformer and
CNN workloads. Diffusion models like Stable Diffusion have a completely different
compute profile -- iterative denoising, U-Net backbone, cross-attention with text
embeddings, variable-length generation. Can mlsysim handle that?"

**The best honest answer:**

The honest answer is: partially, and with manual effort.

The core physics still applies. A diffusion model is ultimately a sequence of
forward passes through a neural network (the U-Net or DiT), each of which has
a known FLOP count and memory footprint. mlsysim can model each denoising step
as a forward pass and multiply by the number of steps. The Roofline analysis
applies -- each step is either compute-bound or memory-bound depending on the
batch size and model size.

What we do not model natively:

1. **The iterative structure.** Diffusion inference requires N denoising steps
   (typically 20-50). Total latency is N times the per-step latency. This is
   trivial to compute but our solver API is not designed around iterative
   generation -- you would need to multiply the single-step result by N yourself.

2. **The U-Net architecture.** Our FLOP counting assumes either a standard
   Transformer or a CNN. U-Nets have skip connections and variable resolution
   stages that make the per-layer FLOP distribution uneven. You would need to
   provide the total FLOPs manually rather than relying on our auto-counting.

3. **Cross-attention.** The text-conditioned cross-attention between the CLIP
   embeddings and the U-Net features is a different attention pattern than
   self-attention in Transformers. It has different memory and compute
   characteristics.

4. **Classifier-free guidance.** CFG doubles the forward pass cost (one
   conditioned, one unconditioned). This is easy to model (multiply by 2) but
   is not automatic.

The newer DiT (Diffusion Transformer) architectures are actually easier for us
to model because they are standard Transformers with the iterative denoising
wrapper. As the field moves from U-Net to DiT, our coverage improves.

I would say: for rough capacity planning ("can I serve Stable Diffusion XL on
this GPU at this batch size?"), mlsysim works with manual FLOP input. For
detailed latency optimization of the denoising pipeline, you need profiling
tools.

**Answer quality: 5/10**

**What would make it stronger:** This answer reveals a real coverage gap. The
presenter knows the physics but has to say "do it manually" four times. A
prepared notebook showing `DiffusionModel` support -- even if it just wraps
the manual steps into a helper function -- would turn this from a 5 to an 8.
Diffusion models are the second-largest inference workload after LLMs in 2026.
Not having native support is a significant gap for a tool claiming to cover
"22 walls." The calibration doc should include at least one diffusion model
benchmark.

---

### Q7: "The TinyML section felt like an afterthought. Is it real?"

**Questioner archetype:** Embedded systems researcher. Noticed that the hardware
zoo lists ESP32-S3 and nRF52840 but the tutorial exercises never touch them.

**The question (verbatim):**
"Your hardware table lists microcontrollers -- ESP32, nRF52840 -- and the efficiency
guide mentions TFLite Micro. But every exercise in the tutorial was about H100s and
A100s. Have you actually validated the model on microcontrollers, or is it just a
row in a table?"

**The best honest answer:**

Fair criticism. The TinyML support is real in the sense that the hardware specs
are in the registry and the solver can compute Roofline-style predictions for
them. An ESP32-S3 has known FLOPS (roughly 0.5 GOPS for INT8) and known SRAM
(512 KB). You can ask mlsysim "does a 250 KB quantized MobileNet-v2 fit in
SRAM and how long does inference take?" and get a physically grounded answer.

But you are right that it is not deeply validated. We have not run MLPerf Tiny
benchmarks against our predictions for MCUs. The efficiency parameter for TinyML
(eta=0.05-0.15) is estimated from general knowledge of interpreter overhead in
TFLite Micro, not from systematic measurement. The tutorial does not include
TinyML exercises because the ISCA audience skews toward datacenter and cloud.

What is genuinely useful for TinyML right now: the memory feasibility check.
"Does this model fit on this MCU?" is a binary question that mlsysim answers
correctly because it is just arithmetic -- model size vs SRAM capacity. That
is actually the most common question in TinyML deployment, and getting it
wrong wastes weeks of porting effort.

What needs work: per-operator latency modeling for MCUs (where there are no
tensor cores and each operator type has wildly different efficiency), flash
vs SRAM partitioning (MCUs often execute from flash, which is 10x slower),
and DMA/interrupt overhead modeling. These are real TinyML constraints that
our current single-parameter efficiency model does not capture.

If there is interest from the embedded community, I would love collaborators
who can provide calibration data from MLPerf Tiny submissions.

**Answer quality: 6/10**

**What would make it stronger:** The memory feasibility argument is strong and
honest. The weakness is the appeal for collaborators, which sounds like "we have
not done the work." A concrete plan would help: "We are running MLPerf Tiny
benchmarks on ESP32-S3 and nRF52840 this quarter and will publish calibration
tables by v0.2." Even better: have one validated MCU benchmark in the
calibration doc. One real number beats ten promises.

---

### Q8: "What's your validation methodology? You calibrate eta per-benchmark."

**Questioner archetype:** Faculty member on a program committee. This is the
question that decides whether the paper gets accepted.

**The question (verbatim):**
"Let me make sure I understand your validation. You have six calibration points.
For two of them (the CNN training cases), you set a default eta, get predictions
that are 22% and 54% off, then show that per-configuration calibration brings
error to 1%. For two more (LLM decode), the efficiency parameter does not even
appear because the workload is memory-bound. And for the last one (GPT-3 FLOPs),
it is a closed-form equation with no empirical parameter at all.

So your only genuinely predictive validation -- where eta is set in advance and
the model makes a falsifiable prediction -- is... zero cases? Every case is either
trivially correct (bandwidth division, closed-form FLOPs) or calibrated after the
fact. How is this a validated model?"

**The best honest answer:**

You have identified the central methodological weakness, and I am not going to
try to talk around it.

You are correct that in the current calibration document, there are zero cases
where we set eta before seeing the benchmark result and then made a falsifiable
prediction. The CNN cases use calibrated eta. The LLM decode cases are
efficiency-insensitive. The FLOP counting is definitional.

Here is what I think the fair assessment is:

**What is genuinely validated:** The structural claims. The model correctly
identifies that LLM decode is memory-bound (not compute-bound). It correctly
identifies that the H100 speedup over A100 for decode is ~1.7x (matching the
bandwidth ratio), not 3.2x (the FLOPS ratio). It correctly computes that
KV-cache dominates memory at high batch sizes. These qualitative predictions
are falsifiable and correct. They are also the predictions that matter most
for system design -- knowing *which* constraint binds is more useful than
knowing the exact latency.

**What is NOT validated:** Quantitative accuracy for compute-bound workloads
with a fixed eta. The model cannot currently say "ResNet-50 on H100 will
achieve X images/second" without a calibrated eta, and the calibrated eta
does not transfer across hardware generations.

**What we need for a strong validation:** A held-out test. The methodology
would be: calibrate eta on workload A (say, Llama-3-8B training on H100),
then predict workload B (Llama-3-70B training on H100) using the same eta.
If the model predicts within 20%, that is meaningful. If it is off by 2x,
that tells us eta is workload-specific, not just hardware-specific. We have
not run this experiment yet. We should, and we will before submitting the
paper.

The broader intellectual claim is not "we can predict exact performance."
It is "we can systematically identify binding constraints across 22 walls
using a common analytical framework." That claim is validated by the
structural results. But I acknowledge that the quantitative accuracy claim
is currently undersupported.

**Answer quality: 9/10**

**What would make it stronger:** This is the right answer for an academic
audience. The only improvement is having the held-out experiment done before
the tutorial. Running Llama-3-8B and Llama-3-70B on the same hardware with
a shared eta, then reporting the cross-workload transfer error, would cost
roughly $50 in cloud compute and would either validate or refute the model's
utility for quantitative prediction. Not having done this before presenting
at ISCA is a significant omission. The structural validation (correct
bottleneck identification) is the real value proposition and should be
foregrounded in the paper.

---

## Part 2: Hallway Conversations

---

### Conversation A: Two PhD Students Debating Whether to Use mlsysim

**Setting:** Coffee break, 10:35 AM, after the Roofline module.

**Characters:**
- **Priya** -- 3rd year, systems/architecture, working on communication-efficient
  distributed training. Uses ASTRA-sim for network simulation.
- **Marcus** -- 2nd year, ML/NLP, working on efficient inference for long-context
  LLMs. Has never used a performance simulator.

---

**Marcus:** That batch-size sweep exercise was actually useful. I have been fighting
with KV-cache OOM for weeks and I never sat down to do the arithmetic. It took
like 30 seconds in their tool.

**Priya:** Sure, but that is literally dividing bytes by capacity. I could do that
on a napkin.

**Marcus:** You could, but you would not. That is the point. I have been running
profilers and reading CUDA traces trying to figure out why I OOM at batch 48
on an A100. Turns out 8B parameters at FP16 is 16 GB, and at batch 48 with
4K context the KV-cache is another 48 GB. That is 64 GB on an 80 GB card.
The arithmetic was always there. I just never did it.

**Priya:** OK, fair. But my work is on communication. The AllReduce model they
showed is textbook ring AllReduce. Real systems use hierarchical AllReduce with
NVLink within the node and InfiniBand across nodes. ASTRA-sim models that at
the packet level. This tool gives me one number.

**Marcus:** Do you always need packet-level simulation though? Like, for your
paper, sure. But when you are writing your NSF proposal and you need to say
"we will need 128 GPUs for this experiment," do you fire up ASTRA-sim?

**Priya:** ...No. I use a spreadsheet.

**Marcus:** Right. So their tool is a better spreadsheet. That is the pitch. It
is not replacing ASTRA-sim for your research. It is replacing your spreadsheet
for your capacity planning.

**Priya:** The efficiency parameter bugs me though. Did you catch the calibration
numbers? ResNet-50 gets eta=0.13 on A100 and eta=0.065 on H100. That is a 2x
difference for the same workload. If I use this for my proposal and pick the
wrong eta, I am off by 2x on my GPU estimate, which means I am asking NSF for
twice too many or twice too few GPUs.

**Marcus:** Yeah, that is a real problem. For inference it is less of an issue
because decode is memory-bound and eta drops out. But for training... I do not
know. You would have to calibrate it yourself.

**Priya:** Which means I need the hardware already. Chicken and egg.

**Marcus:** I think the move is: use it for feasibility ("does the model fit?
am I compute-bound or memory-bound?") and ranking ("is MI300X better than H100
for my workload?"). Do not use it for exact throughput prediction unless you
have a calibrated eta.

**Priya:** That is a narrow use case.

**Marcus:** For you, yes. For me? Every question I have right now is a
feasibility or ranking question. "Can I serve 70B at batch 64 on one H100?"
"Should I quantize to INT4 or use tensor parallelism?" "Is it worth trying
MI300X for its 192 GB HBM?" Those are the questions keeping me up at night,
and this tool answers all of them in under a second.

**Priya:** Fine. I will install it. But if my proposal gets rejected because
the GPU count was wrong, I am blaming you.

**Marcus:** [laughs] Blame the efficiency parameter. At least you will know
which wall to hit.

**Verdict:** Marcus will use mlsysim regularly. Priya will install it, use it
twice for capacity estimates in grant proposals, and go back to ASTRA-sim for
her actual research. This is the correct adoption pattern -- the tool serves
Marcus's needs well and Priya's needs partially.

---

### Conversation B: The AMD Engineer and the Intel Engineer Comparing Notes

**Setting:** Lunch, standing near the "Wall of Walls" sticky-note board.

**Characters:**
- **Rajan** -- Senior performance architect at AMD, works on MI300X benchmarking
  and competitive analysis. Noticed mlsysim includes MI250X and MI300X.
- **Katharina** -- Software engineer at Intel, works on Gaudi accelerator software
  stack. Noticed mlsysim includes Gaudi 2 and Gaudi 3.

---

**Rajan:** Did you see they have MI300X in there? I checked their bandwidth
number -- 5.3 TB/s. That matches our datasheet.

**Katharina:** They have Gaudi 2 and Gaudi 3 as well. The FLOPS numbers look
correct. I am less sure about the memory bandwidth -- Gaudi's memory subsystem
is different from what you would infer from a single bandwidth number.

**Rajan:** That is my concern too. The roofline model assumes one bandwidth
number and one compute number. MI300X has an interesting memory hierarchy --
the HBM3 bandwidth is 5.3 TB/s, but the Infinity Fabric between the chiplets
has its own bandwidth characteristics. For workloads that fit in one chiplet's
local HBM, you get the full bandwidth. For workloads that span chiplets, you
get less.

**Katharina:** Same with Gaudi. The on-die SRAM is a critical tier that the
roofline misses entirely. Gaudi's differentiation is the large SRAM that
keeps activations close to compute. A single bandwidth number averaging across
the memory hierarchy undersells us.

**Rajan:** So both of us have the same complaint: the model is too simple for
our hardware's memory hierarchy.

**Katharina:** Yes. But I will say this -- the fact that our hardware is in
there at all is progress. Most open tools are NVIDIA-only. When a customer
asks "should I use H100 or MI300X or Gaudi 3," the default answer is "run
MLPerf." If this tool gives a directionally correct ranking with sub-second
latency, that is useful even if the absolute numbers are off.

**Rajan:** Directionally correct is the key phrase. Let me check something.
[pulls out laptop] OK, I ran Llama-3-70B inference on MI300X at batch 1.
Their model says ITL = 2.3 ms. Our internal benchmarks with ROCm show 3-5 ms
depending on the framework. So they are within 2x, on the optimistic side.

**Katharina:** Optimistic is dangerous for competitive analysis. If a customer
uses this tool and it says MI300X is faster than Gaudi 3 for their workload,
but the prediction is optimistic for MI300X and pessimistic for Gaudi, we lose
a sale based on a modeling artifact.

**Rajan:** Or vice versa. The bias matters.

**Katharina:** I think the play for both of us is to contribute calibrated
efficiency values. If they have a `TraceableConstant` system where every
number has a source, we can submit official numbers from our benchmark teams.
Then at least the hardware specs are accurate and traceable to us.

**Rajan:** That is smart. Control the inputs, and the model works in our favor.
Or at least does not work against us.

**Katharina:** The question is whether they accept vendor-submitted numbers.
There is an obvious conflict of interest.

**Rajan:** The SPEC benchmark organization handles this with disclosure rules.
You can submit your own numbers, but the methodology must be public and
reproducible. If mlsysim adopted something similar...

**Katharina:** That would actually be useful for the industry. A neutral
analytical framework with vendor-contributed, auditable hardware specs.
Like a TPC for ML hardware.

**Rajan:** That is a much bigger project than what they showed today.

**Katharina:** Agreed. But the infrastructure -- unit-safe constants with
provenance tracking -- is the right foundation. The question is whether
they execute on it.

**Rajan:** Let us talk to them after the afternoon session. I want to
understand the contribution model.

**Verdict:** Both engineers see potential value but have legitimate concerns
about accuracy bias across vendors. The most likely outcome: one or both
vendors contribute hardware specs to the project within 6 months, but only
if the contribution model is clear and the numbers are auditable. The
"neutral analytical framework" vision is compelling but requires governance
that does not exist yet.

---

### Conversation C: The Faculty Member and the Startup CTO

**Setting:** 4:50 PM, packing up after the closing. They were seated near
each other during the capstone.

**Characters:**
- **Professor Chen** -- Teaches ML Systems at a large state university. Has
  150 students per semester. Currently uses ad-hoc Jupyter notebooks for
  homework assignments.
- **Diego** -- CTO of a 40-person startup doing LLM-based document processing.
  Running inference on a mix of A100s and H100s in AWS. Attended the tutorial
  because he is about to make a $2M hardware purchasing decision.

---

**Professor Chen:** What did you think of the capstone?

**Diego:** The capstone was the best part. It was the first time today where
all the pieces came together. The individual exercises were useful but felt
like textbook problems. The capstone felt like my actual job.

**Professor Chen:** That is exactly why I am here. I want to redesign my
ML Systems course around something like this. Right now my students do
profiling labs with PyTorch and CUDA, but they never step back and think
about the system as a whole. They optimize one kernel and think they have
solved the problem.

**Diego:** So you would use mlsysim as a teaching tool?

**Professor Chen:** As the backbone of the course, potentially. The 22 walls
taxonomy is a natural syllabus. Week 1: Compute Wall. Week 2: Memory Wall.
Week 3: Software Wall. And so on. Each week, students use the tool to
explore one wall, then do a real profiling lab to validate the model.

**Diego:** The "validate the model" part is key. If students just trust
the analytical model, they learn the wrong lesson. The model is useful
precisely because it is wrong in interesting ways. The gap between the
model and reality IS the systems engineering.

**Professor Chen:** Exactly. I would have them measure eta themselves.
"Run ResNet-50 on the department A100. Measure throughput. Back-calculate
eta. Now predict what happens on H100. Next week, you will get H100 time
and check your prediction."

**Diego:** That is a great assignment. Wish I had taken that class. What
is your main concern?

**Professor Chen:** Maturity. I cannot build my course around a v0.1.0 tool
that might break or change APIs between semesters. I have 150 students and
2 TAs. If `Engine.solve()` changes its signature, that is 150 broken
notebooks and a week of debugging instead of teaching.

**Diego:** That is a real risk. What about for my use case? I need to decide
between renewing our A100 instances or migrating to H100 or Trainium2.
The cost difference over 3 years is about $1.5M.

**Professor Chen:** Did the tool help you with that?

**Diego:** Partially. The feasibility analysis is immediate -- I now know that
our 70B model in FP16 does not fit on one H100, which means tensor parallelism
or quantization regardless of the platform. That alone saved me from a mistake.
I was about to price out single-GPU instances.

**Professor Chen:** And the cost modeling?

**Diego:** Rougher. The TCO calculator gives me directional answers, but I
need exact numbers for a board presentation. I still have to get actual
quotes from AWS and run actual benchmarks. But here is the thing -- now I
know *which* benchmarks to run. Before today, I would have benchmarked every
model on every instance type at every batch size. That is a $50K benchmarking
bill. Now I know to benchmark only the three configurations that the model
says are in the right ballpark. That probably saves me $40K.

**Professor Chen:** That is the real value proposition. Not "replaces
benchmarking" but "focuses benchmarking."

**Diego:** Right. And for the carbon constraint -- my board just added a
sustainability requirement. The geography exercise was an eye-opener. I had
no idea the grid intensity variation was that large. I am going to move our
training jobs to Quebec. That is a free win.

**Professor Chen:** You and every other company that does that exercise.
Watch Quebec's grid get overloaded in two years.

**Diego:** [laughs] The tragedy of the commons, modeled analytically.

**Professor Chen:** Let me ask you something as an industry person. Is the
22-wall framework actually useful for practitioners, or is it an academic
taxonomy?

**Diego:** Some of the 22 walls matter a lot. Compute, Memory, Communication,
Capital -- those are my daily constraints. Sustainability is becoming one.
Serving, Batching -- absolutely. But "Reasoning Wall" (inference-time
compute), "Sensitivity Wall" (partial derivatives) -- those feel like
textbook walls, not engineering walls. I would never say to my team "we
are hitting the Sensitivity Wall."

**Professor Chen:** That is useful feedback. The framework might benefit
from a "practitioner's top 10" versus the full 22.

**Diego:** Or just clear tiers. Tier 1: walls that determine your architecture
(Compute, Memory, Communication, Capital). Tier 2: walls that affect your
optimization (Software, Serving, Batching, Compression). Tier 3: walls
that matter for specific contexts (Carbon, Safety, TinyML, Multi-tenant).
The full 22 is great for a course. For a startup CTO, I need the top 8.

**Professor Chen:** I am going to steal that for my syllabus. Tier 1 in
the first half of the semester, Tier 2 in the second half, Tier 3 as
optional projects.

**Diego:** Send me the syllabus. I will send you interns.

**Professor Chen:** Deal.

**Verdict:** Professor Chen will adopt mlsysim for teaching if and only if
the API stabilizes by fall 2026 and there are pre-built assignment notebooks.
Diego will use it this week to narrow his benchmarking from 20 configurations
to 3, potentially saving $40K. He will not use it for final purchasing
decisions -- those still require real benchmarks and real quotes. Both see
the 22-wall taxonomy as pedagogically strong but operationally over-broad,
and both independently converge on the idea that a tiered subset would
increase practical adoption.

---

## Summary: Strength and Weakness Assessment

### Where the tutorial is strong

| Strength | Evidence |
|----------|----------|
| The "aha moments" work | The H100-is-only-1.7x-faster exercise consistently surprises even experienced engineers. The KV-cache OOM exercise solves a real problem people have. |
| Honest positioning | The slides explicitly state "2-5x accuracy" and "not a replacement for benchmarking." This disarms skeptics. |
| The CPI analogy is intellectually sound | It is the correct framing and maps to something the ISCA audience already understands. |
| Multi-vendor coverage is a real differentiator | No other open analytical tool covers NVIDIA, AMD, Intel, Google, and Cerebras. |
| The capstone design challenge works | It synthesizes all the concepts and feels like a real engineering problem. |

### Where the tutorial is vulnerable

| Vulnerability | Severity | Mitigation needed |
|---------------|----------|-------------------|
| Zero held-out validation experiments | **Critical** | Run cross-workload eta transfer experiments before ISCA. This is a $50 experiment that determines paper acceptance. |
| MoE models not supported | **High** | At minimum, provide a documented workaround notebook. Better: native `MoEModel` class. |
| Diffusion models not supported | **High** | Add one diffusion benchmark to the calibration table. |
| TinyML claims exceed evidence | **Medium** | Run one MLPerf Tiny benchmark against predictions. |
| Eta does not transfer across GPU generations | **Medium** | Document this as a known limitation prominently, not buried in the calibration doc. Frame it as a research contribution opportunity. |
| 22 walls claim vs actual solver coverage | **Medium** | Audit which walls are actually wired into `Pipeline.solve()` vs which are standalone calculations. Be precise in slides. |
| API stability for course adoption | **Medium** | Commit to a stable v1.0 API freeze by a specific date. |
| No governance model for vendor-contributed specs | **Low** | Define a contribution policy before AMD and Intel contribute numbers. |

### The adoption decision matrix

| Persona | Will adopt? | Why / Why not |
|---------|-------------|---------------|
| PhD student (systems) | Maybe | Good for capacity planning in proposals. Not deep enough for research. |
| PhD student (ML) | Yes | Solves real "why is my training slow" questions immediately. |
| Industry engineer (NVIDIA) | No | Has internal tools that are better. |
| Industry engineer (AMD/Intel) | Interested | Wants to contribute specs to ensure fair competitive comparison. |
| Faculty (ML Systems course) | Yes, if API stabilizes | The 22-wall taxonomy is a ready-made syllabus. |
| Startup CTO | Yes, for scoping | "Focuses benchmarking" is the real value. Saves $40K on unnecessary benchmarks. |
| Startup CTO (final decision) | No | Final purchasing decisions require real benchmarks and real quotes. |
