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
    AppendOnlyContextResult,
    HorizonCompoundingResult,
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
    Growing the model cost by a fraction ``g`` raises the attempt to
    ``C' = (1 + g) C_model + C_other``; cost per accepted task holds only if
    acceptance rises to ``p_acc C' / C``.
    """
    requires = ()
    produces = AcceptedTaskCostResult

    def solve(self, model_cost: Quantity, other_cost: Quantity, acceptance: float,
              token_price_cut: Optional[float] = None, improved_acceptance: Optional[float] = None,
              other_cost_cut: Optional[float] = None, acceptance_gain: Optional[float] = None,
              independent_attempts: Optional[int] = None,
              model_cost_growth: Optional[float] = None) -> AcceptedTaskCostResult:
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
        if model_cost_growth is not None:
            # Added 2026-10-20: the same cost per accepted task at a costlier
            # attempt needs acceptance raised in proportion to the attempt cost.
            grown = tokens * (1 + model_cost_growth) + other
            out.update(grown_attempt_cost=grown, grown_attempt_rise=grown / attempt - 1,
                       grown_break_even_acceptance=acceptance * grown / attempt)
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






