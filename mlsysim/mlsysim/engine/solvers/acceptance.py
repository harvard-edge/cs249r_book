"""Agent evaluation solvers: what a set of graded trials supports about a configuration.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20). Each
model is arithmetic on stated trial counts, verdicts, and costs: the
acceptance rate when some trials never resolve; a grader's error rates
against a reference assessment; how tasks and repeated runs share an
evaluation budget; Wald and Wilson intervals; the task counts a precision or
detection target needs; a paired candidate-against-baseline comparison
(Newcombe's method 10, McNemar's exact test, and the variance pairing
removes); the within-model spread of a leaderboard across scaffolds and the
neighboring ranks it cannot separate; repeated runs of one task against new
tasks; batch and sequential selection under one candidate cap; and the
summary of a paired repeated-run evaluation. Repeated-run consistency over
``k`` recurrences is horizon compounding (``HorizonCompoundingModel``).
"""

from __future__ import annotations

import math
import statistics
from statistics import NormalDist
from typing import Mapping, Optional, Sequence

from ..results import (
    BinomialIntervalsResult,
    CappedSelectionCostResult,
    PairedComparisonResult,
    ProportionPlanningResult,
    TaskMixConsistencyResult,
)
from ...core.types import Quantity
from ...core.units import USD
from ...physics.agents import calc_independent_success, calc_pass_at_k_unbiased, calc_pass_pow_k_unbiased
from ...physics.statistics import (
    calc_mcnemar_exact_two_sided,
    calc_newcombe_paired_difference,
    calc_paired_difference_variance,
    calc_proportion_half_width,
    calc_samples_for_half_width,
    calc_task_averaged_variance,
    calc_two_proportion_sample_size,
    calc_wald_interval,
    calc_wilson_interval,
    calc_within_task_variance,
)
from .base import ForwardModel


def z_for_confidence(confidence: float) -> float:
    """Two-sided standard normal quantile for a confidence level (1.96 at 0.95)."""
    return NormalDist().inv_cdf(0.5 + confidence / 2)








class BinomialIntervalsModel(ForwardModel):
    """
    Wald and Wilson intervals for several success counts out of ``trials``.

    Wald collapses to zero width at ``x = 0`` and leaves [0, 1] near the
    boundaries; Wilson keeps a nonzero width and stays inside, and the two
    nearly coincide near one half. The Wilson upper limit at ``x = 0`` is
    ``z^2 / (n + z^2)``.
    """
    requires = ()
    produces = BinomialIntervalsResult

    def solve(self, trials: int, successes: Sequence[int], confidence: float = 0.95) -> BinomialIntervalsResult:
        z = z_for_confidence(confidence)
        return BinomialIntervalsResult(
            z_critical=z,
            wald={x: calc_wald_interval(x, trials, z) for x in successes},
            wilson={x: calc_wilson_interval(x, trials, z) for x in successes},
        )


class ProportionPlanningModel(ForwardModel):
    """
    How many independent tasks a precision or a detection target needs.

    One rate: the normal half-width ``z sqrt(p (1 - p) / n)`` shrinks as
    ``1 / sqrt(n)``, and ``ceil(z^2 p (1 - p) / h^2)`` tasks reach a half-width
    ``h``. Two independent arms at average rate ``p``: about
    ``2 (z_alpha + z_beta)^2 p (1 - p) / delta^2`` tasks per arm detect a
    difference ``delta``, so halving ``delta`` quadruples the count. The
    detection quantiles are the rounded planning values the caller states.
    """
    requires = ()
    produces = ProportionPlanningResult

    def solve(self, rate: float, planning_tasks: int, target_half_width: float, differences: Sequence[float],
              z_alpha: float, z_beta: float, confidence: float = 0.95) -> ProportionPlanningResult:
        z = z_for_confidence(confidence)
        lo, hi = calc_wilson_interval(round(rate * planning_tasks), planning_tasks, z)
        return ProportionPlanningResult(
            z_critical=z,
            half_width=calc_proportion_half_width(rate, planning_tasks, z),
            wilson_half_width=(hi - lo) / 2,
            samples_for_target=calc_samples_for_half_width(rate, target_half_width, z),
            per_arm={d: calc_two_proportion_sample_size(rate, d, z_alpha=z_alpha, z_beta=z_beta)
                     for d in differences},
        )


class PairedComparisonModel(ForwardModel):
    """
    A candidate against a baseline, one trial of each on the same tasks.

    Only discordant pairs move the difference; Newcombe's method 10 gives its
    interval, and a noninferiority gate needs the lower limit above
    ``-tolerance``. Lower half-widths shrink as ``1 / sqrt(n)``, so the pairs
    that center the expected lower limit on the margin are
    ``n [(delta - lower) / (delta + tolerance)]^2`` (even odds of clearing),
    and a true gain of zero needs the half-width inside the tolerance itself.
    The paired outcome has variance ``d - delta^2`` per pair against
    ``p_c (1 - p_c) + p_b (1 - p_b)`` for two independent arms. McNemar's
    exact test answers the separate equality question.
    """
    requires = ()
    produces = PairedComparisonResult

    def solve(self, both: int, candidate_only: int, baseline_only: int, neither: int, tolerance: float,
              confidence: float = 0.95) -> PairedComparisonResult:
        z = z_for_confidence(confidence)
        n = both + candidate_only + baseline_only + neither
        delta, lower, upper = calc_newcombe_paired_difference(both, candidate_only, baseline_only, neither, z)
        lower_width = delta - lower
        pc, pb = (both + candidate_only) / n, (both + baseline_only) / n
        discordant = (candidate_only + baseline_only) / n
        var_paired = calc_paired_difference_variance(discordant, pc - pb)
        var_independent = pc * (1 - pc) + pb * (1 - pb)
        return PairedComparisonResult(
            pairs=n,
            candidate_rate=pc,
            baseline_rate=pb,
            difference=delta,
            lower=lower,
            upper=upper,
            noninferior=lower > -tolerance,
            superior=lower > 0,
            exact_p=calc_mcnemar_exact_two_sided(baseline_only, candidate_only),
            lower_width=lower_width,
            clear_width=delta + tolerance,
            pairs_if_gain_holds=math.ceil(n * (lower_width / (delta + tolerance)) ** 2),
            pairs_if_no_gain=math.ceil(n * (lower_width / tolerance) ** 2),
            discordant_fraction=discordant,
            paired_variance=var_paired,
            independent_variance=var_independent,
            paired_share=var_paired / var_independent,
            paired_half_width=z * math.sqrt(var_paired / n),
            independent_multiple=var_independent / var_paired,
        )






class TaskMixConsistencyModel(ForwardModel):
    """
    All-pass probability of ``k`` runs: one task repeated against ``k`` new tasks.

    Repeating one randomly drawn task passes every run with ``E[p_i^k]``;
    ``k`` independently drawn tasks all pass with ``(E[p_i])^k``, which
    Jensen's inequality places at or below it, strictly when task difficulty
    varies and ``k >= 2``. Task types are equally weighted.
    """
    requires = ()
    produces = TaskMixConsistencyResult

    def solve(self, task_pass_rates: Sequence[float], repetitions: int) -> TaskMixConsistencyResult:
        mean = sum(task_pass_rates) / len(task_pass_rates)
        same = 0.0
        for p in task_pass_rates:
            same += p ** repetitions / len(task_pass_rates)
        return TaskMixConsistencyResult(mean_rate=mean, same_task_all=same, new_tasks_all=mean ** repetitions)


class CappedSelectionCostModel(ForwardModel):
    """
    Batch against sequential selection of up to ``cap`` candidates under an exact checker.

    Both policies return a passing candidate with ``q = 1 - (1 - p)^cap``.
    The batch generates and checks all ``cap``; sequential selection stops at
    the first pass and starts ``q / p`` candidates on average. Charging every
    initiated trial to the accepted ones gives ``cap c / q`` against
    ``(c / p)``, where ``c`` is generation plus checking: sequential cost per
    accepted task does not depend on the cap, and only ``p`` lowers it.
    """
    requires = ()
    produces = CappedSelectionCostResult

    def solve(self, pass_rate: float, cap: int, generation_cost: Quantity,
              check_cost: Quantity) -> CappedSelectionCostResult:
        p = pass_rate
        q = calc_independent_success(p, cap)
        expected_n = q / p
        unit = (generation_cost + check_cost).to(USD)
        batch = cap * unit
        sequential = expected_n * unit
        batch_per = batch / q
        sequential_per = sequential / q
        return CappedSelectionCostResult(
            availability=q,
            expected_candidates=expected_n,
            batch_trial_cost=batch,
            sequential_trial_cost=sequential,
            batch_per_accepted=batch_per,
            sequential_per_accepted=sequential_per,
            batch_premium=float((batch_per / sequential_per).to("").magnitude) - 1,
            doubled_rate_per_accepted=unit / (2 * p),
        )


