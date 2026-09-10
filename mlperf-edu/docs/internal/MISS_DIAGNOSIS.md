# Why the Five Misses Miss

Five executed contracts do not reproduce their published target. The question
this answers is the one that matters before shipping: is a shortfall a broken
adapter, or a target the contract inherited badly?

**No target is adjusted to make a result pass.** Where a correction is proposed
below it is because the contract transcribed its source incorrectly, and each
one is checked against the recorded result to confirm it does not flip a
verdict.

| Workload | Observed | Target | Gap | Verdict on the verdict |
|:---|---:|---:|---:|:---|
| image-generation | 1.8016 | 1.7900 | 0.6% | likely sampling variance |
| time-series-forecasting | 0.2924 | 0.2900 | 0.8% | **contract error, now fixed** |
| recommendation | 0.6232 | 0.6350 | 1.9% | likely a real adapter gap |
| code-generation | 0.5610 | 0.5730 | 3.2% | likely a real adapter gap |
| function-calling | 0.7852 | 0.8292 | 5.3% | likely a real adapter gap |

---

## time-series forecasting — the contract was wrong, not the adapter

The PatchTST paper reports **0.290 MSE with standard deviation 0.002**. The
contract carried the mean as a hard threshold and set `tolerance: 0.0`,
discarding the spread the source states in the same sentence. The registry's own
reviewer note recorded that standard deviation while the gate ignored it.

That is inconsistent with the suite's own practice. Graph node classification
inherits the OGB ogbn-arxiv leaderboard as `target: 0.7174, tolerance: 0.0029`,
which is that leaderboard's mean **and** its reported deviation. Two contracts
inheriting a mean-plus-deviation source were treating it two different ways.

The tolerance is now 0.002, matching the paper and matching the convention.

**It does not change the verdict.** Measured across three seeds:

| seed | test MSE | distance from published mean | verdict at 0.2920 |
|---:|---:|---:|:---|
| 42 | 0.29239 | +1.20 σ | MISS |
| 1 | 0.29225 | +1.13 σ | MISS |
| 7 | 0.28892 | −0.54 σ | PASS |

The recorded result still misses. What the correction buys is a contract that
no longer holds this workload to a stricter standard than the paper it inherits
from. The seed spread straddling the published mean is the more useful finding:
this reproduction sits inside the paper's stated distribution, and the verdict
is decided by the seed.

## image generation — probably variance, not established

The target is the published minimum FID across three independent 50,000-image
trials. Our three trials returned 1.8139, 1.8016, and 1.8155, so the minimum is
1.8016 against a published 1.79.

A minimum-of-three is a biased-low statistic, and 0.6% relative is within what
sampler and seed choice plausibly move.

### RESOLVED without new compute (2026-08-04)

The comparison this section said was missing was the wrong comparison. The
published trial spread is not needed, because our own three trials already
bound the question:

| Quantity | Value |
|---|---:|
| Trials | 1.8139, 1.8016, 1.8155 |
| Best trial | 1.8016 |
| Target (lower is better) | 1.7900 |
| Margin, best trial to target | 0.0116 |
| Spread across our three trials | 0.0140 |

**The procedure's own run-to-run spread is 1.21x the distance by which it
misses.** The shortfall is therefore inside the noise of the measurement that
produced it, and a fourth trial could plausibly land under the target without
anything changing. That is established from committed evidence rather than
suspected.

The variance is in generation, not scoring. The committed
`current_packet_rescore` re-evaluates the same 150,000 images and reproduces
every trial to within 1.2e-12 FID, with all image hashes verified. So the
evaluator is deterministic and the seed block driving sampling is what moves
the number.

This is the same structural finding as the training seed sweep, in a third
place: a single accepted run does not settle a verdict whenever the procedure's
spread is comparable to its margin. Inference over a pinned checkpoint is
bit-identical; anything that samples or trains is not.

**What remains open:** whether the published 1.79 is itself a
minimum-of-three or a different statistic. If it is a minimum of three, the
comparison is like for like and the gap is variance. If it is a mean or a best
of more trials, the comparison is biased and the gap may be smaller than it
looks. That is a question about the upstream paper, not about our run.

## recommendation — a real gap, and not a budget one

HR@10 peaks at 0.6232 on epoch 7 and declines to 0.6128 by epoch 20. Training
longer makes it worse, so the 1.9% shortfall is not a budget limit. The
evaluation protocol was already checked once: a held-out-item leakage probe over
500 sampled users found zero contaminated users, and the candidate count is the
inherited 999.

That leaves the adapter. Candidates:

- ~~Learning-rate schedule.~~ **Tested and refuted, 2026-08-04. See below.**
- Negative sampling during training. Four negatives per positive, resampled
  each epoch, matches the reference in count but perhaps not in method.
- Embedding initialisation and the GMF/MLP fusion detail.

### RESULT: annealing does not help, it hurts (2026-08-04)

Running the ablation first required adding `MLPERF_EDU_NCF_LR_SCHEDULE` and
`MLPERF_EDU_NCF_LEARNING_RATE` as `pro`-envelope overrides. The runner read the
rate straight from the contract, so the hypothesis had been structurally
untestable rather than merely untested.

Controlled comparison. Same runner, same contract, same 7-epoch budget, same
5e-4 base rate, same seed. Only the schedule differs.

| Schedule | HR@10 | Gap to 0.6350 |
|---|---:|---:|
| Constant 5e-4 (contract, recorded) | 0.6232 | -0.0118 |
| Cosine 5e-4 to 2.5e-5 (ablation) | 0.6155 | -0.0195 |

Cosine per-epoch HR@10: 0.5460, 0.5874, 0.6014, 0.6110, 0.6134, **0.6155**,
0.6143. Rate per epoch: 5.00e-4, 4.75e-4, 4.06e-4, 3.06e-4, 1.94e-4, 9.41e-5,
2.48e-5.

**Annealing is 0.0077 worse than the constant rate.** It widens the gap by
about two thirds. The curve also never shows the late-training instability the
hypothesis predicted; it rises monotonically to epoch 6 and dips once at 7,
which is the shape of a model that ran out of learning rate rather than one
that overshot.

One caveat on scope. Cosine over 7 epochs conflates annealing with a reduction
in total learning, since the average rate over the run is roughly half the
constant one. What is refuted is the specific claim that annealing within the
contract's budget recovers the gap. A reference that anneals over a much longer
schedule is not tested by this, and testing it would cost proportionally more.

**Consequence:** the cheapest of the three candidates is eliminated. Negative
sampling method and the GMF/MLP fusion detail move to the front, and both are
code-inspection tasks against the v0.5 reference before they are run-cost
tasks, which makes them cheaper than this ablation was.

## code generation — a real gap, and the evaluator is not the cause

91 of 164 in the container, 92 of 164 on the host, against a 94-task gate. Two
to three tasks short.

The evaluator is exonerated. Its reference self-check passes 163 of 164
canonical solutions, which is the contract's declared expectation, and the
container and host paths agree to within one task. Whatever is missing is in
generation, not scoring.

The obvious suspect is precision. The runner declares `execution_dtype:
float32`, but the model loads through a path that may not honour it end to end
on MPS.

**What would settle it:** rerun in enforced float32 on CPU and compare. If two
or three tasks return, the finding is that laptop-class inference precision
costs a measurable amount of task quality, which is a systems result worth
reporting rather than an apology.

## function calling — the largest gap and the strongest precision suspicion

78.52% against 82.92%, the widest shortfall at 5.3% relative. The recorded
backend is `pytorch-bfloat16-mps-greedy`. bfloat16 has roughly eight bits of
mantissa; on a task scored by exact AST match, a small logit perturbation flips
a whole case.

**What would settle it:** the same float32 rerun. This workload is the most
expensive to test, at roughly 3.5 hours of generation, so code generation is the
better place to run the precision experiment first.

---

## RESULT: the precision hypothesis is refuted for code generation (2026-08-04)

The experiment was run. `MLPERF_EDU_DEVICE=cpu`, profile `max`, full 164-task
HumanEval+, backend recorded as `pytorch-cpu`, `device_executed: cpu`.

| Path | Passing | pass@1 |
|---|---:|---:|
| CPU, enforced float32 (this run) | 92 / 164 | 0.560976 |
| MPS host (recorded) | 92 / 164 | 0.560976 |
| Container (recorded) | 91 / 164 | 0.554878 |
| Gate (Qwen published) | 94 / 164 | 0.573000 |

The comparison is config-controlled. The CPU run and the committed MPS record
share `execution_dtype: float32`, `attention_implementation: eager`, greedy
decoding, the same ChatML prompt format, and the same 2,048-token cap. Only the
backend differs.

**At identical configuration, CPU and MPS agree exactly.** Not within a task:
the same 92, the same pass@1 to six decimal places. MPS was already honouring
float32 end to end, so the suspected precision loss on the committed path does
not exist and cannot be what costs the two tasks. Generation time was 851.6 s.

One caveat, and it matters. `registry/selection-ledger.yaml` records an
exploratory MPS sweep in which bfloat16 with SDPA attention and bfloat16 with
eager attention both scored 92, while a float32 variant scored **93**. That
float32 entry does not name its attention implementation, so it is most likely
the SDPA path. If so, the one recoverable task is attributable to *attention
implementation*, not to precision, and the committed eager configuration is
what leaves it on the table. That is a cheap, specific follow-up, and it is a
different experiment from the one just run.

Three consequences:

1. **The code-generation gap is real, and it is not a backend effect.** It is a
   genuine reproduction gap against Qwen's published number rather than an
   artifact of laptop execution. What remains to diagnose is narrower than
   before: attention implementation (see the caveat above), stop rules, or a
   difference between the pinned model bytes and the checkpoint behind the
   published figure.
2. **The shared-cause hypothesis is dead.** Code generation and function
   calling do not share a precision explanation, because code generation has no
   precision problem to share. Function calling still runs
   `pytorch-bfloat16-mps-greedy`, so a precision hypothesis remains live *for
   that workload alone* and is now the only reason to spend its ~3.5 hours.
3. **A positive finding for the paper.** Backend independence of task quality,
   previously measured across six deterministic inference workloads, now also
   holds for a generative greedy-decoding workload scored by test execution.
   That is a stronger statement than the determinism study alone supports.

The one real variance is host 92 vs container 91. Both are float32 greedy, so
that single-task difference is an evaluator-environment effect rather than a
model effect, and it bounds the run-to-run noise of this workload at about one
task.

## What this adds up to

One contract error, now corrected without changing a verdict. One shortfall
that is probably variance and is not yet established. Three real adapter gaps
that no longer share a common cause, one of which (code generation) is now
confirmed to be independent of execution backend.

Nothing here justifies moving a target to make a result pass, and nothing here
has been moved for that reason.
