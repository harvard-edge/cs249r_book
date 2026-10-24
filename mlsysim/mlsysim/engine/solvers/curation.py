"""Trajectory curation solvers: what a recorded corpus may teach, and what collecting it costs.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20). Each
model is arithmetic on stated counts, rates, and service times: failures that
counterfactual probes attribute to the system, and the price of training on
all of them anyway; how often a random split of runs places one task on both
sides; the expected checking time of an admission route run in different
orders; one record class's share of a pool under three denominators; and how
fast a verifier timeout burst fills reserved staging.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

from ..results import (
    AdmissionOrderResult,
    CorpusDenominatorsResult,
    FailureTriageResult,
    RunSplitLeakageResult,
    StagingBurstResult,
)
from ...core.types import Quantity
from ...core.units import GB, MB, USD, hour, second, token
from ...physics.agents import calc_cost_per_rejection, calc_short_circuit_cost, calc_split_straddle
from ...physics.transformer import calc_transformer_training_flops
from .base import ForwardModel


class FailureTriageModel(ForwardModel):
    """
    Failures that counterfactual probes remove below the weights, and the price of training on all of them.

    ``runs x failure_rate`` failures; each probe removes its share (rounded
    to whole failures), and the rest survive every probe. Training on every
    failure for ``epochs`` epochs of ``sequence_tokens`` costs 6PD FLOPs,
    run on ``devices`` at ``peak_rate x utilization`` and billed per device-hour.
    """
    requires = ()
    produces = FailureTriageResult

    def solve(self, runs: int, failure_rate: float, probe_shares: Mapping[str, float], parameters: Quantity,
              sequence_tokens: int, epochs: int, devices: int, peak_rate: Quantity, utilization: float,
              hourly_price: Quantity) -> FailureTriageResult:
        failures = round(runs * failure_rate)
        removed = {name: round(failures * share) for name, share in probe_shares.items()}
        system = round(failures * sum(probe_shares.values()))
        residual = failures - system
        tokens_all = failures * sequence_tokens * epochs * token
        tokens_res = residual * sequence_tokens * epochs * token
        flops_all = calc_transformer_training_flops(parameters, tokens_all.to(token).magnitude)
        flops_res = calc_transformer_training_flops(parameters, tokens_res.to(token).magnitude)
        sustained = (devices * peak_rate * utilization).to("flop/s")
        wall = (flops_all / sustained).to(second)
        device_hours = (devices * wall).to(hour)
        cost = (device_hours * hourly_price).to(USD)
        residual_cost = (devices * (flops_res / sustained) * hourly_price).to(USD)
        return FailureTriageResult(
            failures=failures, removed=removed, system_failures=system, residual=residual,
            system_share=system / failures, training_tokens=tokens_all, residual_tokens=tokens_res,
            training_flops=flops_all, residual_flops=flops_res, sustained_rate=sustained,
            wall_time=wall.to(hour), device_hours=device_hours, cost=cost, residual_cost=residual_cost)


class RunSplitLeakageModel(ForwardModel):
    """
    Tasks whose runs a random split of runs places on both sides of the partition.

    Each run goes to training with probability ``f``. A task with ``k`` runs
    straddles with probability ``1 - f^k - (1 - f)^k``; a test run has a
    sibling in training unless all ``k - 1`` siblings also went to test. The
    majority count is the smallest ``k`` at which more than half the tasks
    straddle.
    """
    requires = ()
    produces = RunSplitLeakageResult

    def solve(self, attempts: int, train_fraction: float, tasks: Optional[int] = None,
              max_attempts: int = 64) -> RunSplitLeakageResult:
        f = train_fraction
        straddle = calc_split_straddle(f, attempts)
        majority = next(k for k in range(1, max_attempts + 1) if calc_split_straddle(f, k) > 0.5)
        return RunSplitLeakageResult(
            whole=1 - straddle,
            straddle=straddle,
            straddling_tasks=None if tasks is None else tasks * straddle,
            sibling_in_training=1 - (1 - f) ** (attempts - 1),
            majority_attempts=majority,
            majority_straddle=calc_split_straddle(f, majority),
            curve={k: calc_split_straddle(f, k) for k in range(1, attempts + 1)},
        )


class AdmissionOrderModel(ForwardModel):
    """
    One admission route's independent checks run in different orders, each stopping at its first failure.

    Every order admits the same records (the route yield is the product of the
    pass rates); the expected slot time per attempted record differs. Time per
    admitted record divides by the yield. The saving compares the cheapest
    order with the dearest.
    """
    requires = ()
    produces = AdmissionOrderResult

    def solve(self, checks: Mapping[str, Tuple[float, Quantity]], orders: Sequence[Sequence[str]],
              candidates: Optional[int] = None) -> AdmissionOrderResult:
        per_rejection = {name: calc_cost_per_rejection(t, p).to(second) for name, (p, t) in checks.items()}
        times, yields = [], []
        for order in orders:
            cost, reach = calc_short_circuit_cost([checks[n][0] for n in order], [checks[n][1] for n in order])
            times.append(cost.to(second))
            yields.append(reach)
        route_yield = yields[0]
        best = min(range(len(times)), key=lambda i: times[i].magnitude)
        worst = max(range(len(times)), key=lambda i: times[i].magnitude)
        return AdmissionOrderResult(
            cost_per_rejection=per_rejection,
            time_per_attempted=times,
            time_per_admitted=[t / route_yield for t in times],
            route_yield=route_yield,
            admitted=None if candidates is None else candidates * route_yield,
            best_order=best,
            saving=1 - (times[best] / times[worst]).to("").magnitude,
            saving_per_attempted=times[worst] - times[best],
        )


class CorpusDenominatorsModel(ForwardModel):
    """
    One record class's share of a retained pool counted three ways: records, stored tokens, selected targets.

    Each class gives (records, stored tokens per record, selected target tokens
    per record). The admitted-token share counts stored tokens only over the
    classes the route admits.
    """
    requires = ()
    produces = CorpusDenominatorsResult

    def solve(self, classes: Mapping[str, Tuple[int, Quantity, Quantity]], focus: str,
              admitted: Sequence[str]) -> CorpusDenominatorsResult:
        records = sum(n for n, _, _ in classes.values())
        stored = {k: (n * s).to(token) for k, (n, s, _) in classes.items()}
        targets = {k: (n * u).to(token) for k, (n, _, u) in classes.items()}
        stored_total = sum(stored.values(), 0 * token)
        admitted_total = sum((stored[k] for k in admitted), 0 * token)
        target_total = sum(targets.values(), 0 * token)
        ctx = {k: (s / u).to("").magnitude for k, (_, s, u) in classes.items() if u.to(token).magnitude > 0}
        return CorpusDenominatorsResult(
            records=records, stored_tokens=stored_total, admitted_stored_tokens=admitted_total,
            target_tokens=target_total,
            record_share=classes[focus][0] / records,
            stored_share=(stored[focus] / stored_total).to("").magnitude,
            admitted_stored_share=(stored[focus] / admitted_total).to("").magnitude,
            target_share=(targets[focus] / target_total).to("").magnitude,
            context_per_target=ctx,
        )


class StagingBurstModel(ForwardModel):
    """
    Reserved staging under a verifier timeout burst, with every record charged at its ceiling.

    Rollouts finish at ``rollouts / rollout_time``; verifiers serve
    ``slots / mean_service`` nominally and ``slots / timeout`` once every
    check runs to its timeout. The difference, times the record ceiling, fills
    the store. Stopping dispatch leaves the running rollouts and the growth
    during the controller's reaction delay, so dispatch must stop at or below
    ``1 - (in flight + reaction growth) / store``.
    """
    requires = ()
    produces = StagingBurstResult

    def solve(self, rollouts: int, rollout_time: Quantity, slots: int, mean_service: Quantity,
              timeout: Quantity, record_ceiling: Quantity, store: Quantity,
              reaction: Quantity) -> StagingBurstResult:
        arrival = (rollouts / rollout_time).to(1 / second)
        capacity = (slots / mean_service).to(1 / second)
        burst = (slots / timeout).to(1 / second)
        net = arrival - burst
        growth = (net * record_ceiling).to(MB / second)
        t_full = (store / growth).to(second)
        in_flight = (rollouts * record_ceiling).to(GB)
        reaction_growth = (reaction * growth).to(GB)
        return StagingBurstResult(
            arrival_rate=arrival, capacity=capacity, utilization=(arrival / capacity).to("").magnitude,
            burst_capacity=burst, net_growth=net, growth=growth, time_to_full=t_full,
            in_flight=in_flight, reaction_growth=reaction_growth,
            high_watermark=(1 - (in_flight + reaction_growth) / store).to("").magnitude,
        )
