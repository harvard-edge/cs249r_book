"""Agent solvers: reliability, cost, and context over a whole trajectory.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
The unit of account is the delegated task, not the model call: a task
succeeds only if every unchecked turn does, costs every attempt it takes
until one is accepted, and re-reads its own history on every turn.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from typing import Optional, Sequence

import numpy as np

from ..results import (
    AcceptedTaskCostResult,
    AgentRunSummaryResult,
    AppendOnlyContextResult,
    HorizonCompoundingResult,
    TaskHorizonTrendResult,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...core.units import Q_
from ...physics.agents import (
    calc_append_only_context,
    calc_break_even_acceptance,
    calc_cost_per_accepted,
    calc_first_horizon_below,
    calc_independent_success,
    calc_step_error_for_target,
    calc_unchecked_success,
)


class HorizonCompoundingModel(ForwardModel):
    """
    Unchecked task success over a horizon of independent steps.

    Every step is right with probability ``p``, nothing checks a step before
    the next builds on it, and nothing recovers, so a task of ``H`` turns
    succeeds with probability ``p^H``. Holding ``target`` at ``H`` turns needs
    a per-step error of ``1 - target^(1/H)``; for small error the product
    ``eps H`` approaches ``-ln(target)`` at any horizon. The first horizon whose
    success falls below ``target`` is ``floor(ln target / ln p) + 1``.
    """
    requires = ()
    produces = HorizonCompoundingResult

    def solve(self, step_success: float, horizons: Sequence[int], target_success: float,
              target_horizon: int) -> HorizonCompoundingResult:
        eps = 1 - step_success
        needed = calc_step_error_for_target(target_success, target_horizon)
        return HorizonCompoundingResult(
            step_error=eps,
            success={h: calc_unchecked_success(step_success, h) for h in horizons},
            required_step_error=needed,
            error_ratio=eps / needed,
            required_error_horizon=needed * target_horizon,
            small_error_limit=-math.log(target_success),
            first_horizon_below_target=calc_first_horizon_below(step_success, target_success),
        )


def _usd(q) -> float:
    return q.to("USD").magnitude


class AcceptedTaskCostModel(ForwardModel):
    """
    Cost per accepted task, ``C_eff = E[C_attempt] / p_acc``, and two levers on it.

    An attempt costs model tokens plus everything else (sandbox, tests,
    review); attempts repeat until one is accepted. A price cut removes a
    fraction of one component; an acceptance gain divides the whole attempt.
    The break-even acceptance is the acceptance the price cut is worth.
    Charging only the accepted attempt understates cost by ``1 / p_acc``.
    """
    requires = ()
    produces = AcceptedTaskCostResult

    def solve(self, model_cost: Quantity, other_cost: Quantity, acceptance: float,
              token_price_cut: Optional[float] = None, improved_acceptance: Optional[float] = None,
              other_cost_cut: Optional[float] = None, acceptance_gain: Optional[float] = None,
              independent_attempts: Optional[int] = None) -> AcceptedTaskCostResult:
        tokens, other = _usd(model_cost), _usd(other_cost)
        attempt = tokens + other
        share = tokens / attempt
        c_eff = calc_cost_per_accepted(attempt, acceptance)
        out = dict(attempt_cost=attempt, token_share=share, attempts_per_acceptance=1 / acceptance,
                   cost_per_accepted=c_eff, accepted_only_understatement=c_eff / attempt)
        if token_price_cut is not None:
            cheaper = tokens * (1 - token_price_cut) + other
            out.update(cheaper_attempt_cost=cheaper,
                       cost_per_accepted_cheaper_tokens=calc_cost_per_accepted(cheaper, acceptance),
                       break_even_acceptance=calc_break_even_acceptance(acceptance, token_price_cut, share))
        if improved_acceptance is not None:
            out.update(cost_per_accepted_improved=calc_cost_per_accepted(attempt, improved_acceptance))
        if other_cost_cut is not None:
            out.update(cost_per_accepted_cheaper_other=calc_cost_per_accepted(
                tokens + other * (1 - other_cost_cut), acceptance))
        if acceptance_gain is not None:
            out.update(cost_per_accepted_acceptance_gain=calc_cost_per_accepted(attempt, acceptance + acceptance_gain))
        if independent_attempts is not None:
            out.update(independent_success=calc_independent_success(acceptance, independent_attempts))
        return AcceptedTaskCostResult(**out)


class AppendOnlyContextModel(ForwardModel):
    """
    Context read by the last call, and input resent in total, by an append-only trajectory.

    Each turn appends its output and observation to the prompt and resends the
    whole history, so the last of ``H`` calls reads ``s0 + delta (H - 1)``
    tokens while the run sends ``H s0 + delta H (H - 1) / 2``: the window can
    hold the history while the resend grows quadratically with the horizon.
    """
    requires = ()
    produces = AppendOnlyContextResult

    def solve(self, initial_tokens: Quantity, tokens_per_turn: Quantity, turns: int,
              context_window: Quantity) -> AppendOnlyContextResult:
        s0 = int(initial_tokens.to("token").magnitude)
        delta = int(tokens_per_turn.to("token").magnitude)
        last, total = calc_append_only_context(s0, delta, turns)
        return AppendOnlyContextResult(
            last_call_tokens=Q_(last, "token"),
            window_fraction=last / context_window.to("token").magnitude,
            cumulative_input_tokens=Q_(total, "token"),
            reread_ratio=total / last,
        )


def _ordinal(s: str) -> int:
    return dt.date.fromisoformat(s).toordinal()


def _log2_fit(points):
    """(slope per day, intercept) of a least-squares fit of log2(horizon) on release date."""
    t = np.array([_ordinal(r["release_date"]) for r in points])
    slope, intercept = np.polyfit(t, np.log2([r["p50_min"] for r in points]), 1)
    return slope, intercept


class TaskHorizonTrendModel(ForwardModel):
    """
    Growth of the 50 percent task horizon of frontier models at release.

    Frontier rows are those marked state of the art at release. Doubling
    times come from least-squares fits of log2(horizon) on release date, over
    frontier rows at or below ``fit_cap`` (the source excludes longer, less
    certain estimates), for the whole period and from ``recent_from`` on.
    """
    requires = ("series",)
    produces = TaskHorizonTrendResult

    def solve(self, rows, fit_cap: Quantity, recent_from: str) -> TaskHorizonTrendResult:
        cap = fit_cap.to("minute").magnitude
        frontier = [r for r in rows if r["is_sota"]]
        fit = [r for r in frontier if r["p50_min"] <= cap]
        recent = [r for r in fit if r["release_date"] >= recent_from]
        slope_all, _ = _log2_fit(fit)
        slope, intercept = _log2_fit(recent)
        first, last = frontier[0], frontier[-1]
        return TaskHorizonTrendResult(
            first_frontier_minutes=first["p50_min"], last_frontier_minutes=last["p50_min"],
            first_date=first["release_date"], last_date=last["release_date"],
            growth=last["p50_min"] / first["p50_min"],
            doubling_all=Q_(1 / slope_all, "day"), doubling_recent=Q_(1 / slope, "day"),
            recent_slope_per_day=slope, recent_intercept=intercept,
        )


class AgentRunSummaryModel(ForwardModel):
    """
    Totals of one recorded agent run over a benchmark, one attempt per task.

    Acceptance is the benchmark's hidden tests; cost per accepted task is the
    run's total spend over its accepted tasks (with one attempt each, the same
    as the mean attempt over acceptance). Charging only the accepted runs
    reports their mean cost and hides the rejected spend. Tasks are also
    binned by model calls, each bin running from the previous edge (exclusive)
    to its own (inclusive), with a last bin above the final edge.
    """
    requires = ("series",)
    produces = AgentRunSummaryResult

    def solve(self, rows, call_bin_edges: Sequence[int],
              command_classes: Sequence[str] = ("read", "write", "run", "net")) -> AgentRunSummaryResult:
        n = len(rows)
        ok = [r for r in rows if r["resolved"]]
        costs = [r["cost_usd"] for r in rows]
        calls = [int(r["model_calls"]) for r in rows]
        total = sum(costs)
        mean_attempt = total / n
        per_accepted = calc_cost_per_accepted(mean_attempt, len(ok) / n)
        accepted_mean = statistics.mean(r["cost_usd"] for r in ok)
        commands = {k: int(sum(r[f"cmd_{k}"] for r in rows)) for k in command_classes}
        syntax = [r for r in rows if r["syntax_error_seen"]]
        bins, low = [], 0
        for high in list(call_bin_edges) + [None]:
            b = [r for r in rows if r["model_calls"] > low and (high is None or r["model_calls"] <= high)]
            bins.append({"low": low + 1, "high": high, "tasks": len(b), "accepted": sum(1 for r in b if r["resolved"])})
            low = high
        return AgentRunSummaryResult(
            tasks=n, accepted=len(ok), rejected=n - len(ok), acceptance=len(ok) / n,
            submitted=sum(1 for r in rows if r["exit_status"] == "Submitted"),
            total_cost=total, mean_attempt_cost=mean_attempt, cost_per_accepted=per_accepted,
            accepted_mean_cost=accepted_mean, accepted_only_understatement=per_accepted / accepted_mean,
            calls_median=statistics.median(calls), calls_min=min(calls), calls_max=max(calls),
            commands=commands, command_total=sum(commands.values()),
            observations=int(sum(r["observations"] for r in rows)),
            nonzero_exits=int(sum(r["nonzero_exit"] for r in rows)),
            truncated_outputs=int(sum(r["truncated"] for r in rows)),
            syntax_error_tasks=len(syntax), syntax_error_accepted=sum(1 for r in syntax if r["resolved"]),
            call_bins=bins,
        )
