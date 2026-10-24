"""Agent-harness solvers: what the runtime holds, shares, and admits (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers``. Each model is arithmetic on
stated inputs: approval escrow sized by Little's law, weighted deficit
round-robin between request classes, bounded-liability admission of the next
model call, one trajectory metered against its budget vector, and call or
step ceilings read from public agent runs. None models a particular serving
engine, provider quota, or queueing burst.
"""

from __future__ import annotations

import statistics
from typing import Dict, Mapping, Sequence

from ..results import (
    ApprovalEscrowResult,
    BoundedLiabilityAdmissionResult,
    DeficitRoundRobinResult,
    TrajectoryBudgetMeterResult,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...physics.agents import calc_bounded_liability, calc_drr_calls_per_round, calc_used_fraction
from ...physics.serving import calc_littles_law_occupancy


def _per_token_usd(price: Quantity) -> float:
    return float(price.to("USD / token").magnitude)


def _count(q: Quantity) -> float:
    return float(q.to("").magnitude)




class ApprovalEscrowModel(ForwardModel):
    """
    What approval escrow holds for a long-running workload and an urgent one.

    Little's law gives the trajectories waiting for a decision, ``L = lambda W``.
    For the repair workload each keeps its workspace lease, so escrow binds on
    retained memory; for the remediation workload fewer than one waits, and
    the decision binds on time to mitigate, ``T = diagnose + decide + apply``.
    Means only: bursts and batch review hold more for as long as they last.
    """
    requires = ()
    produces = ApprovalEscrowResult

    def solve(self, repair_rate: Quantity, repair_wait: Quantity, lease: Quantity,
              remediation_rate: Quantity, diagnosis: Quantity, remediation_wait: Quantity,
              apply_verify: Quantity) -> ApprovalEscrowResult:
        held = _count(calc_littles_law_occupancy(repair_rate, repair_wait))
        ttm = (diagnosis + remediation_wait + apply_verify).to("minute")
        return ApprovalEscrowResult(
            held=held,
            retained_memory=(held * lease).to("GiB"),
            remediation_held=_count(calc_littles_law_occupancy(remediation_rate, remediation_wait)),
            time_to_mitigate=ttm,
            approval_share=float((remediation_wait / ttm).to("").magnitude),
        )


class DeficitRoundRobinModel(ForwardModel):
    """
    Weighted deficit round-robin between backlogged request classes.

    Each round a class earns ``w q`` charge units and spends ``c`` per call
    (input tokens plus the full output allowance), carrying unspent credit
    forward. Weights fix the token-charge shares, ``w / sum(w)``; the call
    shares follow ``w q / c`` and so depend on call size. A direct simulation
    of ``rounds`` rounds checks the long-run formula.
    """
    requires = ()
    produces = DeficitRoundRobinResult

    def solve(self, quantum: float, classes: Mapping[str, Mapping[str, float]],
              rounds: int) -> DeficitRoundRobinResult:
        weight = {k: c["weight"] for k, c in classes.items()}
        charge = {k: c["input"] + c["output"] for k, c in classes.items()}
        per_round = {k: calc_drr_calls_per_round(weight[k], quantum, charge[k]) for k in classes}
        total_w, total_calls = sum(weight.values()), sum(per_round.values())
        credit, calls, spent = ({k: 0.0 for k in classes} for _ in range(3))
        for _ in range(rounds):
            for k in classes:
                credit[k] += weight[k] * quantum
                while credit[k] >= charge[k]:
                    credit[k] -= charge[k]
                    calls[k] += 1
                    spent[k] += charge[k]
        sim_calls, sim_spent = sum(calls.values()), sum(spent.values())
        return DeficitRoundRobinResult(
            credit_per_round={k: weight[k] * quantum for k in classes},
            charge_per_call=charge,
            calls_per_round=per_round,
            rounds_per_call={k: 1 / per_round[k] for k in classes},
            charge_share={k: weight[k] / total_w for k in classes},
            call_share={k: per_round[k] / total_calls for k in classes},
            call_ratio=max(per_round.values()) / min(per_round.values()),
            simulated_rounds=rounds,
            simulated_call_share={k: calls[k] / sim_calls for k in classes},
            simulated_charge_share={k: spent[k] / sim_spent for k in classes},
        )


class BoundedLiabilityAdmissionModel(ForwardModel):
    """
    Bounded-liability admission of the next model call, against a check of settled spend.

    A spend check admits when the call's liability fits ``allowance - settled``.
    The reservation test admits only when settled spend, outstanding
    reservations, the call's liability, and the protected finalization reserve
    together fit the allowance. Liability prices every input token at the
    undiscounted rate and the whole output allowance (``calc_bounded_liability``).
    Dollars throughout.
    """
    requires = ()
    produces = BoundedLiabilityAdmissionResult

    def solve(self, allowance: Quantity, settled: Quantity, reserved: Quantity,
              finalization_reserve: Quantity, input_tokens: int, output_allowance: int,
              input_price: Quantity, output_price: Quantity,
              probe: Quantity) -> BoundedLiabilityAdmissionResult:
        usd = lambda q: float(q.to("USD").magnitude)  # noqa: E731
        c_max, actual, held, fin = usd(allowance), usd(settled), usd(reserved), usd(finalization_reserve)
        p_in, p_out = _per_token_usd(input_price), _per_token_usd(output_price)
        c_in = calc_bounded_liability(input_tokens, 0, p_in, p_out)
        c_out = calc_bounded_liability(0, output_allowance, p_in, p_out)
        liability = c_in + c_out
        spend_headroom = c_max - actual
        headroom = c_max - actual - held - fin
        return BoundedLiabilityAdmissionResult(
            input_liability=c_in, output_liability=c_out, liability=liability,
            spend_headroom=spend_headroom, headroom=headroom,
            spend_check_admits=liability <= spend_headroom,
            admitted=liability <= headroom,
            finalization_shortfall=fin - (c_max - (actual + held + liability)),
            input_share=c_in / liability,
            max_input_tokens=(headroom - c_out) / p_in,
            tokens_per_probe=usd(probe) / p_in,
            reservation_share=held / c_max,
        )


class TrajectoryBudgetMeterModel(ForwardModel):
    """
    One append-only trajectory metered against a budget vector of turns, tokens, time, and dollars.

    With prefix reuse every turn's earlier prompt is billed at the cached rate
    and only the newly appended tokens (and the whole first prompt) at the
    input rate; without reuse every input token is billed uncached. The token
    meter counts cached input in full, so only the dollar meter sees a cache
    miss. Each used fraction is ``used / allowance``; the leading meter is the
    largest.
    """
    requires = ()
    produces = TrajectoryBudgetMeterResult

    def solve(self, turns: int, input_tokens: int, uncached_tokens: int, output_tokens: int,
              input_price: Quantity, cached_price: Quantity, output_price: Quantity,
              allowances: Mapping[str, float], elapsed: float) -> TrajectoryBudgetMeterResult:
        p_in, p_cache, p_out = (_per_token_usd(p) for p in (input_price, cached_price, output_price))
        cached = input_tokens - uncached_tokens
        total = input_tokens + output_tokens
        with_reuse = uncached_tokens * p_in + cached * p_cache + output_tokens * p_out
        without = input_tokens * p_in + output_tokens * p_out
        common = {"turns": calc_used_fraction(turns, allowances["turns"]),
                  "tokens": calc_used_fraction(total, allowances["tokens"]),
                  "time": calc_used_fraction(elapsed, allowances["time"])}
        used_r = dict(common, dollars=calc_used_fraction(with_reuse, allowances["dollars"]))
        used_n = dict(common, dollars=calc_used_fraction(without, allowances["dollars"]))
        return TrajectoryBudgetMeterResult(
            input_tokens=input_tokens, cached_tokens=cached, uncached_tokens=uncached_tokens,
            output_tokens=output_tokens, total_tokens=total,
            cost_with_reuse=with_reuse, cost_without_reuse=without, cost_ratio=without / with_reuse,
            used_with_reuse=used_r, used_without_reuse=used_n,
            leading_with_reuse=max(used_r, key=used_r.get),
            leading_without_reuse=max(used_n, key=used_n.get),
        )




