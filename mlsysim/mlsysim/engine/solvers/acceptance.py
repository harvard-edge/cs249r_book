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
    GraderCalibrationResult,
    LeaderboardTiesResult,
    PairedComparisonResult,
    PairedRunEvaluationResult,
    ProportionPlanningResult,
    ScaffoldSpreadResult,
    TaskMixConsistencyResult,
    TaskRepeatAllocationResult,
    VerdictBoundsResult,
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


class VerdictBoundsModel(ForwardModel):
    """
    An acceptance rate when some trials end pending or unknown.

    The default counts unresolved trials as not accepted and reports the
    bounds from resolving all of them either way. Dropping them from the
    denominator inflates the rate, and since hard tasks are the ones that
    time out, it biases the sample toward easy tasks.
    """
    requires = ()
    produces = VerdictBoundsResult

    def solve(self, passed: int, failed: int, unresolved: int) -> VerdictBoundsResult:
        n = passed + failed + unresolved
        return VerdictBoundsResult(
            trials=n,
            default_rate=passed / n,
            upper_bound=(passed + unresolved) / n,
            dropped_rate=passed / (passed + failed),
        )


class GraderCalibrationModel(ForwardModel):
    """
    A grader's error rates against an independent reference assessment.

    The false-acceptance rate conditions on reference failures and predicts
    how often the grader approves a broken result; the failure fraction among
    accepted results conditions on the grader's acceptances and moves with
    the base rate of failures. A random audit of accepted results estimates
    only the second.
    """
    requires = ()
    produces = GraderCalibrationResult

    def solve(self, reference_failures: int, accepted_failures: int, reference_passes: int,
              accepted_passes: int, confidence: float = 0.95) -> GraderCalibrationResult:
        lo, hi = calc_wilson_interval(accepted_failures, reference_failures, z_for_confidence(confidence))
        accepted = accepted_failures + accepted_passes
        return GraderCalibrationResult(
            false_accept_rate=accepted_failures / reference_failures,
            false_accept_low=lo,
            false_accept_high=hi,
            accepted=accepted,
            failure_among_accepted=accepted_failures / accepted,
            false_reject_rate=(reference_passes - accepted_passes) / reference_passes,
        )


class TaskRepeatAllocationModel(ForwardModel):
    """
    Standard error of a task-averaged acceptance estimate under one run budget.

    ``Var = sigma_task^2 / N + sigma_run^2 / (N r)``, with the within-task
    variance from the mean rate, ``m (1 - m) - sigma_task^2``. With new tasks
    free, the budget buys ``budget / r`` tasks; when a new task's fixture
    costs ``fixture_cost`` runs, it buys ``floor(budget / (fixture_cost + r))``
    whole tasks, and repeats earn a place.
    """
    requires = ()
    produces = TaskRepeatAllocationResult

    def solve(self, mean_rate: float, between_task_variance: float, budget_runs: int,
              free_runs: Sequence[int], fixture_cost_runs: int, max_runs: int) -> TaskRepeatAllocationResult:
        vb = between_task_variance
        vw = calc_within_task_variance(mean_rate, vb)
        free_tasks = {r: budget_runs // r for r in free_runs}
        free_se = {r: math.sqrt(calc_task_averaged_variance(vb, vw, n, r)) for r, n in free_tasks.items()}
        paid_tasks = {r: budget_runs // (fixture_cost_runs + r) for r in range(1, max_runs + 1)}
        paid_var = {r: calc_task_averaged_variance(vb, vw, n, r) for r, n in paid_tasks.items()}
        best = min(paid_var, key=paid_var.get)
        return TaskRepeatAllocationResult(
            within_variance=vw,
            variance_ratio=vw / vb,
            free_tasks=free_tasks,
            free_se=free_se,
            paid_tasks=paid_tasks,
            paid_se={r: math.sqrt(v) for r, v in paid_var.items()},
            best_runs=best,
            best_tasks=paid_tasks[best],
            best_se=math.sqrt(paid_var[best]),
        )


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


class ScaffoldSpreadModel(ForwardModel):
    """
    One model's benchmark score across agent scaffolds, from leaderboard rows.

    Rows carry ``model``, ``accuracy``, and ``accuracy_se``. Models with at
    least ``min_scaffolds`` rows are kept; each spread is the highest minus
    the lowest score, set against the median reported standard error.
    """
    requires = ("series",)
    produces = ScaffoldSpreadResult

    def solve(self, rows: Sequence[Mapping], min_scaffolds: int) -> ScaffoldSpreadResult:
        by_model = {}
        for r in rows:
            by_model.setdefault(r["model"], []).append(r["accuracy"])
        by_model = {m: v for m, v in by_model.items() if len(v) >= min_scaffolds}
        spreads = {m: max(v) - min(v) for m, v in by_model.items()}
        widest = max(spreads, key=spreads.get)
        return ScaffoldSpreadResult(
            models=len(by_model),
            spreads=spreads,
            widest=widest,
            widest_low=min(by_model[widest]),
            widest_high=max(by_model[widest]),
            widest_count=len(by_model[widest]),
            median_spread=statistics.median(spreads.values()),
            median_se=statistics.median([r["accuracy_se"] for r in rows if r["accuracy_se"]]),
        )


class LeaderboardTiesModel(ForwardModel):
    """
    Neighboring leaderboard ranks that an independent-samples test cannot separate.

    Rows carry ``model``, ``mean_score``, and ``stderr``. Among the ``top``
    highest scores, a neighboring pair is tied when its gap is below
    ``z sqrt(se_a^2 + se_b^2)``; with a typical standard error ``s``, two
    systems need a gap of about ``z sqrt(2) s``. Pairing on shared tasks
    would separate more, so this is conservative.
    """
    requires = ("series",)
    produces = LeaderboardTiesResult

    def solve(self, rows: Sequence[Mapping], top: int, confidence: float = 0.95) -> LeaderboardTiesResult:
        z = z_for_confidence(confidence)
        ranked = sorted(rows, key=lambda r: -r["mean_score"])
        shown = ranked[:top]
        pairs = list(zip(shown[:-1], shown[1:]))
        tied = sum(abs(a["mean_score"] - b["mean_score"]) < z * math.hypot(a["stderr"], b["stderr"])
                   for a, b in pairs)
        typical = statistics.median([r["stderr"] for r in shown])
        return LeaderboardTiesResult(
            models=len(ranked),
            top=len(shown),
            pairs=len(pairs),
            tied=int(tied),
            typical_se=typical,
            needed_gap=z * math.sqrt(2) * typical,
            top_model=shown[0]["model"],
            top_score=shown[0]["mean_score"],
            last_score=shown[-1]["mean_score"],
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


class PairedRunEvaluationModel(ForwardModel):
    """
    Two configurations run ``runs`` times on each of the same tasks, run against run.

    Rows carry ``task``, ``run``, ``config``, the sealed ``verdict`` and the
    ``visible`` check (``pass``, ``fail``, or ``unknown``), and the trial's
    ``calls``, ``input_tokens``, ``output_tokens``, and ``seconds``. Per
    configuration: acceptance with its Wilson interval, how many visible
    passes the sealed tests reject, token and time totals charged to accepted
    trials, and the unbiased pass@k and pass^k from each task's passes,
    averaged over tasks with equal weight. Paired run by run, the comparison
    uses Newcombe's method 10 and McNemar's exact test; runs of one task are
    correlated, so that interval treats them as more independent than they are.
    """
    requires = ("series",)
    produces = PairedRunEvaluationResult

    def solve(self, rows: Sequence[Mapping], task_ids: Sequence[str], runs: int, baseline: str,
              candidate: str, confidence: float = 0.95) -> PairedRunEvaluationResult:
        z = z_for_confidence(confidence)
        out = {f: {} for f in ("trials", "accepted", "rate", "wilson", "unknown", "visible_pass",
                               "visible_pass_sealed_reject", "calls", "tokens", "output_tokens", "seconds",
                               "tokens_per_accepted", "seconds_per_accepted", "per_task_passes", "pass_at",
                               "pass_pow")}
        for cfg in (baseline, candidate):
            rr = [r for r in rows if r["config"] == cfg]
            acc = sum(r["verdict"] == "pass" for r in rr)
            tok = sum(r["input_tokens"] + r["output_tokens"] for r in rr)
            sec = sum(r["seconds"] for r in rr)
            passes = {t: sum(r["verdict"] == "pass" for r in rr if r["task"] == t) for t in task_ids}
            out["trials"][cfg] = len(rr)
            out["accepted"][cfg] = acc
            out["rate"][cfg] = acc / len(rr)
            out["wilson"][cfg] = calc_wilson_interval(acc, len(rr), z)
            out["unknown"][cfg] = sum(r["verdict"] == "unknown" for r in rr)
            out["visible_pass"][cfg] = sum(r["visible"] == "pass" for r in rr)
            out["visible_pass_sealed_reject"][cfg] = sum(r["visible"] == "pass" and r["verdict"] != "pass" for r in rr)
            out["calls"][cfg] = sum(r["calls"] for r in rr)
            out["tokens"][cfg] = tok
            out["output_tokens"][cfg] = sum(r["output_tokens"] for r in rr)
            out["seconds"][cfg] = sec
            out["tokens_per_accepted"][cfg] = tok / acc
            out["seconds_per_accepted"][cfg] = sec / acc
            out["per_task_passes"][cfg] = passes
            ks = range(1, runs + 1)
            out["pass_at"][cfg] = {k: sum(calc_pass_at_k_unbiased(runs, c, k) for c in passes.values()) / len(passes)
                                   for k in ks}
            out["pass_pow"][cfg] = {k: sum(calc_pass_pow_k_unbiased(runs, c, k) for c in passes.values()) / len(passes)
                                    for k in ks}
        ok = {(r["task"], r["run"], r["config"]): r["verdict"] == "pass" for r in rows}
        both = cand = base = neither = 0
        for t in task_ids:
            for n in range(runs):
                x, y = ok[(t, n, candidate)], ok[(t, n, baseline)]
                both += x and y
                cand += x and not y
                base += y and not x
                neither += (not x) and (not y)
        delta, lower, upper = calc_newcombe_paired_difference(both, cand, base, neither, z)
        tpa = out["tokens_per_accepted"]
        return PairedRunEvaluationResult(
            runs=runs, tasks=len(task_ids), **out,
            both=both, candidate_only=cand, baseline_only=base, neither=neither,
            difference=delta, lower=lower, upper=upper,
            exact_p=calc_mcnemar_exact_two_sided(base, cand),
            tokens_per_accepted_ratio=tpa[candidate] / tpa[baseline],
        )
