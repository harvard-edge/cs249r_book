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
from statistics import NormalDist

from ..results import (
    PairedComparisonResult,
)
from ...physics.statistics import (
    calc_mcnemar_exact_two_sided,
    calc_newcombe_paired_difference,
    calc_paired_difference_variance,
)
from .base import ForwardModel


def z_for_confidence(confidence: float) -> float:
    """Two-sided standard normal quantile for a confidence level (1.96 at 0.95)."""
    return NormalDist().inv_cdf(0.5 + confidence / 2)


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
