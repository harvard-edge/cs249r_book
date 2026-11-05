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

    def solve(
        self,
        runs: int,
        failure_rate: float,
        probe_shares: Mapping[str, float],
        parameters: Quantity,
        sequence_tokens: int,
        epochs: int,
        devices: int,
        peak_rate: Quantity,
        utilization: float,
        hourly_price: Quantity,
    ) -> FailureTriageResult:
        """Account for training work on all failures versus probe-surviving failures.

        Parameters
        ----------
        runs : int
            Positive number of observed runs.
        failure_rate : float
            Failure fraction in (0, 1], yielding a positive rounded failure count.
        probe_shares : Mapping[str, float]
            Disjoint attributed fractions of all failures, nonnegative and summing to at most one.
        parameters : Quantity
            Model parameter count, convertible to param.
        sequence_tokens : int
            Positive training tokens per failed run.
        epochs : int
            Positive number of passes over the failure corpus.
        devices : int
            Positive number of identical training devices.
        peak_rate : Quantity
            Positive compute rate per device in flop/s.
        utilization : float
            Supplied sustained-compute fraction in (0, 1].
        hourly_price : Quantity
            Nonnegative price per device-hour, convertible to USD/hour.

        Returns
        -------
        FailureTriageResult
            Rounded failure counts; tokens and FLOPs; wall time in hours, device-hours, and costs in USD.

        See Also
        --------
        mlsysim.physics.transformer.calc_transformer_training_flops : Parameter-dominated training work per processed token.

        Notes
        -----
        Equation: F = ``round(runs*failure_rate)``; residual = F - ``round(F*sum(probe_shares))``; training work = ``6*parameters*F*sequence_tokens*epochs``; time = ``work/(devices*peak_rate*utilization)``.

        Assumptions: Probe categories are mutually exclusive. Each removed category and the sum are rounded separately, so their integer counts need not add exactly. The training FLOP convention excludes recomputation.

        Excludes: Overlapping attribution, causal identification of probe effects, communication, memory feasibility, and optimizer overhead.

        References
        ----------
        Derivation: corpus token counts feed the parameter-dominated training
        accounting in calc_transformer_training_flops; divide work by aggregate
        sustained rate and multiply device-hours by the caller-supplied price.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> result = FailureTriageModel().solve(100, 0.2, {"runtime": 0.5}, Q_("1000 param"), 10, 1, 1, Q_("1e9 flop/s"), 0.5, Q_("1 USD/hour"))
        >>> result.residual_cost < result.cost
        True
        """
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
            failures=failures,
            removed=removed,
            system_failures=system,
            residual=residual,
            system_share=system / failures,
            training_tokens=tokens_all,
            residual_tokens=tokens_res,
            training_flops=flops_all,
            residual_flops=flops_res,
            sustained_rate=sustained,
            wall_time=wall.to(hour),
            device_hours=device_hours,
            cost=cost,
            residual_cost=residual_cost,
        )


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

    def solve(
        self, attempts: int, train_fraction: float, tasks: Optional[int] = None, max_attempts: int = 64
    ) -> RunSplitLeakageResult:
        """Estimate task overlap caused by randomly splitting repeated runs.

        Parameters
        ----------
        attempts : int
            Positive number of runs per task.
        train_fraction : float
            Independent probability in (0, 1) that a run enters training.
        tasks : int, optional
            Nonnegative task count for expected overlap; None omits that output.
        max_attempts : int, optional
            Positive search cap, default 64; must include a count whose overlap probability exceeds one half.

        Returns
        -------
        RunSplitLeakageResult
            Dimensionless overlap probabilities, optional expected task count, and a majority-overlap threshold.

        See Also
        --------
        mlsysim.physics.agents.calc_split_straddle : Probability that repeated runs occupy both partitions.

        Notes
        -----
        Equation: P(straddle) = 1-f^k-(1-f)^k; P(training sibling | test run) = 1-(1-f)^(k-1).

        Assumptions: Equal run counts per task and independent split assignment for every run.

        Excludes: Task-level grouping, correlated split decisions, and actual semantic leakage detection. The majority search raises StopIteration if the cap is insufficient.

        References
        ----------
        Derivation: subtract the disjoint all-training and all-test events from
        one; a test run lacks a training sibling only if every sibling is in test.

        Examples
        --------
        >>> result = RunSplitLeakageModel().solve(4, 0.5, tasks=100)
        >>> result.straddle > result.whole
        True
        """
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

    def solve(
        self,
        checks: Mapping[str, Tuple[float, Quantity]],
        orders: Sequence[Sequence[str]],
        candidates: Optional[int] = None,
    ) -> AdmissionOrderResult:
        """Compare expected checking time for short-circuit admission orders.

        Parameters
        ----------
        checks : Mapping[str, tuple[float, Quantity]]
            Named (pass probability, positive service time) pairs; each probability is in (0, 1), times convertible to seconds.
        orders : Sequence[Sequence[str]]
            Nonempty permutations of the same check names, each used once per order.
        candidates : int, optional
            Nonnegative attempted records for an expected admitted count; None omits it.

        Returns
        -------
        AdmissionOrderResult
            Service times in seconds, route yield, expected admissions, best-order index, and fractional saving.

        See Also
        --------
        mlsysim.physics.agents.calc_short_circuit_cost : Expected reach-weighted service cost and route yield.

        Notes
        -----
        Equation: E[T] = sum_i t_i product_{j<i} p_j; yield = product_i p_i; time per admitted record = E[T]/yield.

        Assumptions: Check outcomes are independent and unchanged by order. Each route contains exactly the same checks; the implementation uses the first route yield for every order.

        Excludes: Parallel checks, dependent rejection causes, and checking whether input orders are valid permutations.

        References
        ----------
        Derivation: each check incurs cost only after all earlier checks pass;
        linearity of expectation sums those reach-weighted costs.

        Examples
        --------
        >>> result = AdmissionOrderModel().solve({"fast": (0.5, 1 * second), "slow": (0.5, 4 * second)}, [["fast", "slow"], ["slow", "fast"]])
        >>> result.best_order == 0 and result.saving > 0
        True
        """
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

    def solve(
        self, classes: Mapping[str, Tuple[int, Quantity, Quantity]], focus: str, admitted: Sequence[str]
    ) -> CorpusDenominatorsResult:
        """Compare a corpus class by record, stored-token, and target-token shares.

        Parameters
        ----------
        classes : Mapping[str, tuple[int, Quantity, Quantity]]
            Named (nonnegative record count, stored tokens per record, target tokens per record) entries. Aggregate denominators must be positive.
        focus : str
            Existing class name whose shares are reported; include it in admitted for an admitted-pool share.
        admitted : Sequence[str]
            Existing class names admitted by the route; their stored-token total must be positive.

        Returns
        -------
        CorpusDenominatorsResult
            Record counts and token quantities, dimensionless shares, and context tokens per target token.

        Notes
        -----
        Equation: Stored_i = ``n_i*s_i``; targets_i = ``n_i*u_i``; each reported share divides the focus count by its corresponding pool total.

        Assumptions: Fixed per-class record sizes. Each listed admitted class contributes once to the intended denominator; no deduplication is performed.

        Excludes: Learning-quality weighting, stochastic lengths, and automatic validation that the focus belongs to the admitted pool.

        References
        ----------
        Derivation: multiply each class count by its per-record token count, then
        normalize separately under each explicitly named denominator.

        Examples
        --------
        >>> result = CorpusDenominatorsModel().solve({"short": (2, 10 * token, 5 * token), "long": (1, 100 * token, 5 * token)}, "short", ["short", "long"])
        >>> result.record_share > result.stored_share
        True
        """
        records = sum(n for n, _, _ in classes.values())
        stored = {k: (n * s).to(token) for k, (n, s, _) in classes.items()}
        targets = {k: (n * u).to(token) for k, (n, _, u) in classes.items()}
        stored_total = sum(stored.values(), 0 * token)
        admitted_total = sum((stored[k] for k in admitted), 0 * token)
        target_total = sum(targets.values(), 0 * token)
        ctx = {k: (s / u).to("").magnitude for k, (_, s, u) in classes.items() if u.to(token).magnitude > 0}
        return CorpusDenominatorsResult(
            records=records,
            stored_tokens=stored_total,
            admitted_stored_tokens=admitted_total,
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

    def solve(
        self,
        rollouts: int,
        rollout_time: Quantity,
        slots: int,
        mean_service: Quantity,
        timeout: Quantity,
        record_ceiling: Quantity,
        store: Quantity,
        reaction: Quantity,
    ) -> StagingBurstResult:
        """Estimate staging growth while all verifier checks reach timeout.

        Parameters
        ----------
        rollouts : int
            Positive concurrent rollout count.
        rollout_time : Quantity
            Positive rollout completion time, convertible to seconds.
        slots : int
            Positive parallel verifier slots.
        mean_service : Quantity
            Positive nominal check duration, convertible to seconds.
        timeout : Quantity
            Positive burst check duration, convertible to seconds.
        record_ceiling : Quantity
            Positive worst-case bytes per completed rollout.
        store : Quantity
            Positive reserved staging capacity in bytes.
        reaction : Quantity
            Nonnegative controller reaction duration in seconds.

        Returns
        -------
        StagingBurstResult
            Rates in inverse seconds, growth in MB/s, fill time in seconds, reserves in GB, and fractional dispatch watermark.

        Notes
        -----
        Equation: arrival = rollouts/rollout_time; burst_capacity = slots/timeout; growth = ``(arrival-burst_capacity)*record_ceiling``; watermark = ``1-(in_flight+reaction*growth)/store``.

        Assumptions: A sustained burst, fixed rollout population, and positive net growth are required for the fill-time interpretation. Every record consumes its ceiling.

        Excludes: Arrival variability, partial records, spill storage, and automatic clamping of negative watermarks or nonpositive growth.

        References
        ----------
        Derivation: conservation of staged records subtracts completion service
        from arrivals; multiplying by record size converts backlog into byte growth.

        Examples
        --------
        >>> result = StagingBurstModel().solve(4, 1 * second, 1, 0.1 * second, 1 * second, 1 * MB, 1 * GB, 1 * second)
        >>> result.growth.magnitude > 0 and result.high_watermark < 1
        True
        """
        arrival = (rollouts / rollout_time).to(1 / second)
        capacity = (slots / mean_service).to(1 / second)
        burst = (slots / timeout).to(1 / second)
        net = arrival - burst
        growth = (net * record_ceiling).to(MB / second)
        t_full = (store / growth).to(second)
        in_flight = (rollouts * record_ceiling).to(GB)
        reaction_growth = (reaction * growth).to(GB)
        return StagingBurstResult(
            arrival_rate=arrival,
            capacity=capacity,
            utilization=(arrival / capacity).to("").magnitude,
            burst_capacity=burst,
            net_growth=net,
            growth=growth,
            time_to_full=t_full,
            in_flight=in_flight,
            reaction_growth=reaction_growth,
            high_watermark=(1 - (in_flight + reaction_growth) / store).to("").magnitude,
        )
