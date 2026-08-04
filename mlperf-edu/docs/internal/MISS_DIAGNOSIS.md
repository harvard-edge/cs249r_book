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
sampler and seed choice plausibly move. But this is a suspicion, not a finding:
the published trial spread is not recorded in the contract, so there is nothing
to compare our spread against.

**What would settle it:** record the published per-trial values if they exist,
or run additional trial sets and report our own spread. Until then this stays
an honest miss with an unverified explanation.

## recommendation — a real gap, and not a budget one

HR@10 peaks at 0.6232 on epoch 7 and declines to 0.6128 by epoch 20. Training
longer makes it worse, so the 1.9% shortfall is not a budget limit. The
evaluation protocol was already checked once: a held-out-item leakage probe over
500 sampled users found zero contaminated users, and the candidate count is the
inherited 999.

That leaves the adapter. Candidates, untested:

- Learning-rate schedule. The contract fixes Adam at 5e-4 with no decay; the
  reference may anneal.
- Negative sampling during training. Four negatives per positive, resampled
  each epoch, matches the reference in count but perhaps not in method.
- Embedding initialisation and the GMF/MLP fusion detail.

**What would settle it:** an ablation over the learning-rate schedule, which is
the cheapest of the three at roughly 32 minutes per run.

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

## What this adds up to

One contract error, now corrected without changing a verdict. One shortfall
that is probably variance and is not yet established. Three that are probably
real adapter gaps sharing one plausible cause.

**The single most valuable experiment available is a float32 rerun of code
generation.** It is affordable at roughly 15 minutes, it tests the hypothesis
shared by the two largest gaps, and either outcome is publishable: the gap
closes and laptop precision has a measured quality cost, or it does not and
three workloads need individual diagnosis.

Nothing here justifies moving a target to make a result pass, and nothing here
has been moved for that reason.
