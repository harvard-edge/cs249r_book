# Reviewer Simulation

A pre-submission read of `paper/paper.tex` from four positions: a systems
reviewer at an MLSys-class venue, an artifact evaluator, an instructor deciding
whether to adopt the suite, and a student running it for the first time.

Findings are ordered by how likely they are to change a verdict. Each names the
evidence that would answer it, because the point of this exercise is to decide
what to measure next, not to collect opinions.

Snapshot reviewed: 9 score-bearing cases (8 pass, 1 miss), 12 evidence cases
(6 repeated-timing, 6 provisional), 62.0 minutes for one score-bearing pass,
14 workloads across 7 suites, all measured on one Apple-silicon laptop.

---

## R1. The headline ratio excludes the measured failures

**Severity: high. This is the finding most likely to draw a hostile review.**

The abstract reports 8 of 9 score-bearing cases meeting their gates. Four more
workloads ran their authoritative contract and missed it: code generation
(55.49 percent against 57.30), function calling (78.52 against 82.92), image
generation (1.80 FID against 1.79), and recommendation (0.6232 against 0.635).
Counting them, the suite reproduces 8 of 13 inherited targets.

The abstract describes those four as workloads that "expose bounded functional
probes and stay ineligible for score-bearing review." That is accurate about
their promotion status and misleading about what happened. They are not probes
that were never attempted. They ran and fell short. The body of the paper says
so correctly; the abstract does not.

A reviewer who reads the body after the abstract will conclude the denominator
was chosen to flatter the result. That impression is worth more damage than the
misses themselves, because the paper's entire claim is measurement honesty.

**What answers it:** state both ratios in the abstract. The suite executed 13
inherited contracts and reproduced 8. Of the 9 admitted to score-bearing
review, 8 pass. Both numbers are true and the pair is more persuasive than
either alone, because it demonstrates the fail-closed rule doing its job rather
than hiding its consequences.

## R2. Five reproduction failures need a diagnosis, not just a record

**Severity: high.**

Recording a miss honestly is necessary but not sufficient. A reviewer will ask
whether the shortfalls indicate that laptop-scale reproduction is infeasible,
or that these particular adapters are wrong. The paper currently answers
neither.

The gaps are all small. Measured as relative shortfall against the published
target they are 0.6 percent (image generation), 0.8 percent (time-series
forecasting), 1.9 percent (recommendation), 3.2 percent (code generation), and
5.3 percent (function calling). Being on the losing side is definitional and
proves nothing; the informative part is that none exceeds six percent, which is
closer to what adapter and precision differences produce than to what a scale
limit would.

- Recommendation now has the strongest evidence: HR@10 peaks at 0.6232 on epoch
  7 and declines, so the 1.8 percent relative gap is not a budget limit. It is
  unexplained.
- Image generation misses by 0.6 percent relative on FID, within plausible
  seed variance for a three-trial procedure, but the paper reports one number
  rather than the spread.
- Code generation and function calling both use quantized or reduced-precision
  inference on Apple silicon, which is the obvious candidate explanation and is
  not tested.

**What answers it:** for at least one miss, an ablation isolating the cause.
The cheapest is precision: rerun function calling or code generation in float32
and report whether the gap closes. If it does, the finding is that laptop-class
inference precision costs a measurable and quantifiable amount of task quality,
which is a genuine systems contribution rather than an apology.

## R3. Single host invalidates every timing claim's generality

**Severity: high for an artifact evaluation, medium for the paper.**

All 12 evidence cases come from one Apple-silicon laptop. The repeatability
result, coefficients of variation between 0.31 and 3.08 percent, characterizes
that machine, not the suite. An artifact evaluator on x86 Linux with CUDA has
no basis to predict what they will see, and the 62-minute suite budget is the
number an instructor would plan a lab around.

The paper lists cross-platform replication as future work, which is honest but
leaves the accessibility claim resting on a single sample.

**What answers it:** one full pass on a second architecture, ideally x86 with
CUDA and x86 CPU-only. Even N=2 converts "62 minutes on this laptop" into a
range, and a reviewer reads a range as characterization rather than anecdote.
This is the single highest-value measurement still missing.

## R4. Quality is N=1 while timing is N=5

**Severity: medium. Largely answered by measurement; see below.**

The contract sets `acceptance_runs: 1` for quality and `outer_reference_runs: 5`
for timing. The suite therefore reports careful variance on the number that
matters least for correctness and no variance on the number the gates are
evaluated against.

For eight passing cases the margins are thin. Information retrieval matches its
target exactly at 0.6072, and text classification matches at 0.9106. A single
seed decides those verdicts. A reviewer will ask whether a different seed flips
them, and the honest answer is that no one knows.

**What answers it:** three seeds on the score-bearing set, reporting the spread
of the quality metric rather than replacing the single-run gate. This is
cheaper than it sounds for the fast workloads and would let the paper state how
close to the boundary the marginal passes actually sit.

**Answered, for inference.** A determinism study now covers 6 inference
workloads under 5 seeds and both the MPS and CPU backends, 36 executions. The
quality metric did not move at all: maximum seed spread 0.0, maximum backend
delta 0.0, every value bit-identical. The thin margins are exact reproductions
of the published numbers rather than lucky draws, because these workloads
evaluate a pinned checkpoint over a fixed set and consume no randomness. The
single-run acceptance rule is sound for them, and quality is also shown to be
backend-independent, which timing is not.

Time-series forecasting separately reproduced its recorded result exactly on
re-execution under the live contract, so same-seed determinism holds for at
least one training workload.

**Answered for training too, and the answer is worse.** Sweeping 3 seeds across
the 2 cheap training workloads, the seed spread exceeded the margin to the
effective threshold in both, and the verdict flipped in both directions:

| Workload | seed 42 | seed 1 | seed 7 | threshold | verdict |
|:---|---:|---:|---:|---:|:---|
| graph node classification | 0.72096 | 0.71148 | 0.71819 | >= 0.71450 | pass, **miss**, pass |
| time-series forecasting | 0.29239 | 0.29225 | 0.28892 | <= 0.29000 | miss, miss, **pass** |

A recorded pass becomes a miss and a recorded miss becomes a pass. Single-run
acceptance is therefore unsound for training contracts, and the honest reading
of a training row is one draw from a distribution about as wide as its distance
to the gate. The paper now says this rather than implying training behaves like
inference.

**Resolved as a design position, not a defect.** The natural reflex is to raise
`acceptance_runs` for training until the interval is narrow enough to hide the
flip. For this suite that is the wrong trade. A production submission absorbs
seed variance into a many-run protocol because its job is to publish a defensible
number. This suite's job is to teach when a comparison holds, and the cheapest
demonstration it can offer is a student watching a passing workload miss under a
different seed. Averaging that away removes the lesson and keeps only the
conclusion.

So v0.1 keeps `acceptance_runs: 1`, reports the measured spread beside every
affected verdict, and builds a classroom exercise on it. A reviewer who reads
this as sloppiness has a fair question; the answer is that the spread is
measured, published, annotated in the contracts, and load-bearing for the
pedagogy rather than unexamined.

**Still open:** whether a reviewer accepts that argument. It is a positioning
claim, not a measurement, and it is the one place where the educational purpose
and benchmark convention genuinely pull in different directions.

## R5. The suite has no comparison baseline

**Severity: medium.**

The paper argues that production suites are too heavy for a course. It never
demonstrates this. A reviewer will ask what happens if you simply run MLPerf
Inference on a laptop with a small scenario, and the paper offers no measurement
to answer it.

**What answers it:** either a measured attempt at the obvious alternative,
including where it fails, or an explicit scoping sentence stating that the
comparison is out of scope and why. The second is acceptable; silence is not.

## R6. "EDU" without an educational result

**Severity: medium, and partly by design.**

The paper claims no learning outcomes and says course pilots remain future work.
A reviewer at a systems venue will accept this. A reviewer who reads the title as
an educational claim will not.

The reframe from "A Pedagogical Evaluation Framework" to "A Laptop-Scale,
Quality-Gated Benchmark Suite" already reduces this exposure. What remains is
that the classroom material is presented without evidence that it teaches.

**What answers it:** nothing available before submission. The correct move is
the one already taken: claim the artifact, not the pedagogy, and keep the
disclaimer explicit.

## R7. Reinforcement learning is in the registry but cannot run

**Severity: low, but it invites a fairness question.**

MiniGo needs a CUDA and TensorFlow 1.x runtime on an NVIDIA system. It is
counted among the 14 registered workloads while being unrunnable on the target
platform. A reviewer may ask whether the portfolio count is inflated.

**What answers it:** the paper should state the runnable count alongside the
registered count, the same fix R1 calls for. Thirteen of fourteen workloads run
locally; one is a documented handoff.

---

## Student path, simulated

Running the suite cold surfaced problems that no reviewer would see but every
first user would.

| Observation | Status |
|:---|:---|
| Training printed nothing for tens of minutes, indistinguishable from a hang | Fixed. Progress reports epoch, metric, elapsed, and ETA |
| Recommendation took 98.5 minutes and 10.4 GB peak | Fixed. 32 minutes and 3.3 GB after the split rewrite and epoch budget |
| No environment variable was documented anywhere | Fixed. Reference page with an execution-versus-contract split |
| Troubleshooting said recommendation cannot start locally | Fixed. That stopped being true at the NCF swap |
| A dense catalog hangs the negative sampler forever | Fixed. Fails with a diagnostic |

## Instructor path, simulated

The material an instructor needs is present: profiles map to assignment
structure, labs carry rubrics, and reports are inspectable. Two gaps remain.

- No stated wall-clock budget per workload on commodity hardware, which is what
  an instructor actually schedules against. The 62-minute figure is one laptop.
- No guidance on what to do when a student's result misses a gate that the
  reference run also misses. Five of thirteen contracts currently miss, so this
  will happen in the first week of any course using them.

---

## What to measure next, in priority order

1. **A second host architecture**, full pass. Answers R3 and materially
   strengthens R4 and the instructor gap. Highest value per hour spent.
2. **A precision ablation on one miss.** Answers R2 and converts an apology
   into a finding.
3. **Three seeds on the score-bearing set.** Answers R4 and quantifies how
   close the marginal passes sit to their boundaries.
4. **Both ratios in the abstract.** Costs nothing and removes R1, the finding
   most likely to sour a review.
