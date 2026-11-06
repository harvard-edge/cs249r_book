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
from typing import Sequence

from ..results import (
    BinomialIntervalsResult,
    CappedSelectionCostResult,
    PairedComparisonResult,
    ProportionPlanningResult,
    TaskMixConsistencyResult,
)
from ...core.types import Quantity
from ...core.units import USD
from ...physics.agents import calc_independent_success
from ...physics.statistics import (
    calc_mcnemar_exact_two_sided,
    calc_newcombe_paired_difference,
    calc_paired_difference_variance,
    calc_proportion_half_width,
    calc_samples_for_half_width,
    calc_two_proportion_sample_size,
    calc_wald_interval,
    calc_wilson_interval,
)
from .base import ForwardModel


def z_for_confidence(confidence: float) -> float:
    """Return the standard-normal critical value for a two-sided interval.

    Parameters
    ----------
    confidence : float
        Central probability in (0, 1), dimensionless.

    Returns
    -------
    float
        Positive dimensionless normal quantile.

    Notes
    -----
    Equation: z = Phi^{-1}((1 + confidence) / 2).

    Assumptions: Equal probability is allocated to each tail of a standard normal distribution.

    Excludes: Student-t corrections and exact finite-sample binomial coverage.

    References
    ----------
    Derivation: symmetry places the upper tail at (1 - confidence) / 2;
    inverting the normal cumulative distribution gives the critical value.

    Examples
    --------
    >>> z_for_confidence(0.95) > z_for_confidence(0.90)
    True
    """
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
        """Compare Wald and Wilson intervals for specified success counts.

        Parameters
        ----------
        trials : int
            Positive number of independent Bernoulli trials shared by each count.
        successes : Sequence[int]
            Success counts from zero through trials, inclusive.
        confidence : float, optional
            Central confidence probability in (0, 1); default 0.95.

        Returns
        -------
        BinomialIntervalsResult
            Dimensionless critical value and interval endpoints keyed by success count.

        See Also
        --------
        mlsysim.physics.statistics.calc_wald_interval : Uncorrected normal interval for a binomial rate.
        mlsysim.physics.statistics.calc_wilson_interval : Score interval that handles boundary success counts.

        Notes
        -----
        Equation: Wald uses p +/- z sqrt(p(1-p)/n); Wilson inverts the binomial score test. The normal quantile z is shared by both.

        Assumptions: Each count represents independent trials with a common success probability.

        Excludes: Repeated-task dependence, multiple-comparison correction, and exact coverage guarantees for the Wald approximation.

        References
        ----------
        Wilson (1927), Probable Inference, the Law of Succession, and Statistical Inference.
        https://www.jstor.org/stable/2276774
        The solver composes the two interval functions without changing their formulas.

        Examples
        --------
        >>> result = BinomialIntervalsModel().solve(10, [0, 5])
        >>> result.wald[0] == (0.0, 0.0) and result.wilson[0][1] > 0
        True
        """
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

    def solve(
        self,
        rate: float,
        planning_tasks: int,
        target_half_width: float,
        differences: Sequence[float],
        z_alpha: float,
        z_beta: float,
        confidence: float = 0.95,
    ) -> ProportionPlanningResult:
        """Plan independent trial counts from precision and detection targets.

        Parameters
        ----------
        rate : float
            Planning success probability strictly between zero and one.
        planning_tasks : int
            Positive number of independent tasks in the proposed evaluation.
        target_half_width : float
            Positive target half-width in probability units.
        differences : Sequence[float]
            Positive differences in probabilities to detect between independent arms.
        z_alpha : float
            Positive normal critical value selected for the detection significance level.
        z_beta : float
            Positive normal quantile selected for the desired detection power.
        confidence : float, optional
            Central probability for the single-rate interval; default 0.95.

        Returns
        -------
        ProportionPlanningResult
            Dimensionless interval widths and integer sample counts, including counts per arm.

        See Also
        --------
        mlsysim.physics.statistics.calc_proportion_half_width : Normal standard-error half-width for one proportion.
        mlsysim.physics.statistics.calc_samples_for_half_width : Invert a requested one-rate interval half-width.
        mlsysim.physics.statistics.calc_two_proportion_sample_size : Equal-arm normal-approximation detection planning.

        Notes
        -----
        Equation: h = z sqrt(p(1-p)/n); n_target = ceil(z^2 p(1-p)/h^2). Two-arm planning uses ceil(2(z_alpha+z_beta)^2 p(1-p)/delta^2).

        Assumptions: Independent Bernoulli tasks; the planning rate is supplied, not estimated by this solver. The Wilson comparison rounds rate times planning_tasks to a whole success count.

        Excludes: Clustered tasks, paired-arm gains, attrition, and exact power calculations.

        References
        ----------
        Derivation: solve the normal-approximation standard error for the sample count.
        The two-arm expression adds equal independent Bernoulli variances.
        The Wilson comparison is implemented by calc_wilson_interval.

        Examples
        --------
        >>> result = ProportionPlanningModel().solve(0.5, 100, 0.1, [0.1, 0.2], 1.96, 0.84)
        >>> result.per_arm[0.1] > result.per_arm[0.2]
        True
        """
        z = z_for_confidence(confidence)
        lo, hi = calc_wilson_interval(round(rate * planning_tasks), planning_tasks, z)
        return ProportionPlanningResult(
            z_critical=z,
            half_width=calc_proportion_half_width(rate, planning_tasks, z),
            wilson_half_width=(hi - lo) / 2,
            samples_for_target=calc_samples_for_half_width(rate, target_half_width, z),
            per_arm={d: calc_two_proportion_sample_size(rate, d, z_alpha=z_alpha, z_beta=z_beta) for d in differences},
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

    def solve(
        self,
        both: int,
        candidate_only: int,
        baseline_only: int,
        neither: int,
        tolerance: float,
        confidence: float = 0.95,
    ) -> PairedComparisonResult:
        """Evaluate paired binary outcomes and approximate further sample needs.

        Parameters
        ----------
        both : int
            Nonnegative number of tasks accepted by both configurations.
        candidate_only : int
            Nonnegative number accepted only by the candidate.
        baseline_only : int
            Nonnegative number accepted only by the baseline.
        neither : int
            Nonnegative number accepted by neither; total pairs must be positive.
        tolerance : float
            Positive noninferiority margin in probability units. The observed gain plus tolerance and both variance denominators must be nonzero.
        confidence : float, optional
            Central interval probability in (0, 1); default 0.95.

        Returns
        -------
        PairedComparisonResult
            Rates, interval endpoints, p-value, variance ratios, Boolean gates, and approximate integer pair counts.

        See Also
        --------
        mlsysim.physics.statistics.calc_newcombe_paired_difference : Paired difference with the method-10 score interval.
        mlsysim.physics.statistics.calc_mcnemar_exact_two_sided : Exact equality test using discordant pair counts.

        Notes
        -----
        Equation: delta = (candidate_only - baseline_only)/n; paired variance = discordant_fraction - delta^2. Newcombe method 10 supplies the interval. Planning scales n by squared current-width/target-width ratios.

        Assumptions: Independent task pairs and stable population rates. Both methods observe the same tasks. The projected sample counts use inverse-square-root interval-width scaling.

        Excludes: Sequential testing, task clusters, zero-variance degenerate cases, and a power guarantee for the projected pair counts.

        References
        ----------
        Newcombe (1998), Improved confidence intervals for the difference between
        binomial proportions based on paired data.
        https://doi.org/10.1002/(SICI)1097-0258(19981130)17:22%3C2635::AID-SIM954%3E3.0.CO;2-C
        The additional sample projection is a solver approximation, not the interval method.

        Examples
        --------
        >>> result = PairedComparisonModel().solve(50, 20, 10, 20, 0.05)
        >>> result.candidate_rate > result.baseline_rate and result.pairs == 100
        True
        """
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
        """Compare all-pass probabilities for repeated versus newly sampled tasks.

        Parameters
        ----------
        task_pass_rates : Sequence[float]
            Nonempty sequence of per-task success probabilities in [0, 1].
        repetitions : int
            Positive number of runs in each all-pass event.

        Returns
        -------
        TaskMixConsistencyResult
            Dimensionless mean rate and all-pass probabilities under the two sampling policies.

        Notes
        -----
        Equation: same_task_all = mean(p_i^k); new_tasks_all = mean(p_i)^k.

        Assumptions: Task types have equal weight; runs are independent conditional on their task type. New tasks are sampled independently with replacement.

        Excludes: State carried across runs, correlated failures, and empirical uncertainty in task rates.

        References
        ----------
        Derivation: condition on task type before multiplying conditional pass
        probabilities; average the conditional products for repeated tasks.

        Examples
        --------
        >>> result = TaskMixConsistencyModel().solve([0.2, 0.8], 2)
        >>> result.same_task_all > result.new_tasks_all
        True
        """
        mean = sum(task_pass_rates) / len(task_pass_rates)
        same = 0.0
        for p in task_pass_rates:
            same += p**repetitions / len(task_pass_rates)
        return TaskMixConsistencyResult(mean_rate=mean, same_task_all=same, new_tasks_all=mean**repetitions)


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

    def solve(
        self, pass_rate: float, cap: int, generation_cost: Quantity, check_cost: Quantity
    ) -> CappedSelectionCostResult:
        """Compare batch and early-stopping selection under one candidate cap.

        Parameters
        ----------
        pass_rate : float
            Independent candidate acceptance probability in (0, 1].
        cap : int
            Positive maximum candidates per task.
        generation_cost : Quantity
            Nonnegative generation cost per candidate, convertible to USD.
        check_cost : Quantity
            Nonnegative exact-checker cost per candidate, convertible to USD; total unit cost must be positive.

        Returns
        -------
        CappedSelectionCostResult
            Dimensionless availability and counts; trial and accepted-task costs in USD.

        See Also
        --------
        mlsysim.physics.agents.calc_independent_success : Probability that at least one independent candidate passes.

        Notes
        -----
        Equation: q = 1 - (1-p)^cap; E[N_sequential] = q/p; batch cost per acceptance = ``cap*c/q`` and sequential cost per acceptance = c/p.

        Assumptions: Independent identically distributed candidates, an exact checker, constant generation/check costs, and immediate stopping after acceptance.

        Excludes: Parallel latency, checker errors, correlated candidates, and a feasible interpretation of the doubled-rate counterfactual when 2p exceeds one.

        References
        ----------
        Derivation: sum the truncated geometric survival probabilities for expected
        starts and divide total trial cost by probability of acceptance.

        Examples
        --------
        >>> result = CappedSelectionCostModel().solve(0.25, 4, 1 * USD, 0.1 * USD)
        >>> result.batch_per_accepted > result.sequential_per_accepted
        True
        """
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
