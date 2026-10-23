"""Agent cost-accounting solvers: what one accepted task costs, and what each lever buys (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers``. Each model is arithmetic on
stated inputs: the fitted fall of a fixed capability's price, cost against
score on long-horizon trials, a cohort of one-attempt tasks under a tariff,
prefix-cache bills and the payback of one compaction, retry policies under
independent attempts, a serial episode under Amdahl's law and speculative
decoding, a tiered model cascade under one gate, admission against a token
quota, Kingman's queue price, renting tokens against owning capacity, a
conserved budget across children, architectures with a human fallback, the
payback of a one-time adaptation, per-workload levers on cost per accepted
task, measured coverage against independent trials, and one server swept
across concurrency. None models a particular provider's billing system.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..results import (
    AdaptationPaybackResult,
    ArchitectureChoiceResult,
    CohortAttemptCostResult,
    CompactionPaybackResult,
    ConservedBudgetResult,
    CostVsScoreResult,
    DedicatedBreakEvenResult,
    KingmanQueueResult,
    ModelCascadeResult,
    PrefixCacheBillResult,
    PriceDeclineResult,
    RetryPolicyResult,
    SerialEpisodeResult,
    SpeculativeEpisodeResult,
    TokenPriceComparisonResult,
    TrajectoryAdmissionResult,
    WorkloadLeverResult,
    WorkloadLeverRow,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...core.units import Q_, USD as _USD, ureg
from ...physics.agents import calc_independent_success
from ...physics.serving import calc_kingman_wait_ratio, calc_speculative_round_speedup


_MTOK = 1_000_000


def _usd(q) -> float:
    return float(q.to("USD").magnitude)


def _per_mtok(price: Quantity) -> float:
    """Dollars per million tokens, as a plain number."""
    return float(price.to("USD / Mtoken").magnitude)


def _milli(q) -> int:
    return int(round(_usd(q) * 1000))


class PriceDeclineModel(ForwardModel):
    """
    Yearly decline of the lowest price that reaches each fixed capability.

    Rows are grouped by (benchmark, threshold model); each group's log10 price
    is fitted linearly on release date in years, and the decline factor is
    ``10^-slope``.
    """
    requires = ("series",)
    produces = PriceDeclineResult

    @staticmethod
    def decline_per_year(points) -> float:
        t = np.array([_dt.date.fromisoformat(p["release_date"]).toordinal() / 365.25 for p in points])
        return 10 ** -np.polyfit(t, np.log10([p["usd_per_mtok"] for p in points]), 1)[0]

    def solve(self, rows) -> PriceDeclineResult:
        series: Dict[tuple, list] = {}
        for r in rows:
            series.setdefault((r["benchmark"], r["threshold_model"]), []).append(r)
        named = {f"{b} / {m}": float(self.decline_per_year(v)) for (b, m), v in series.items()}
        rates = sorted(self.decline_per_year(v) for v in series.values())
        return PriceDeclineResult(series_count=len(series), series_rates=named,
                                  rate_low=float(rates[0]), rate_median=float(np.median(rates)),
                                  rate_high=float(rates[-1]),
                                  last_date=max(r["release_date"] for r in rows))


class CostVsScoreModel(ForwardModel):
    """
    Cost per trial against mean score, configuration by configuration.

    Compares the top-scoring configuration with the runner-up and reports the
    median best-to-worst spread across a configuration's trials.
    """
    requires = ("series",)
    produces = CostVsScoreResult

    def solve(self, rows) -> CostVsScoreResult:
        rows = sorted(rows, key=lambda r: -r["mean5"])
        top, runner_up = rows[0], rows[1]
        return CostVsScoreResult(
            configurations=len(rows),
            cost_ratio=top["usd_per_trial"] / runner_up["usd_per_trial"],
            score_gap=100 * (top["mean5"] - runner_up["mean5"]),
            cost_low=min(r["usd_per_trial"] for r in rows),
            cost_high=max(r["usd_per_trial"] for r in rows),
            hours_low=min(r["hours_per_trial"] for r in rows),
            hours_high=max(r["hours_per_trial"] for r in rows),
            median_spread=float(np.median([r["best5"] - r["worst5"] for r in rows])),
        )


class CohortAttemptCostModel(ForwardModel):
    """
    One configuration's cohort: one bounded attempt per task, all charges settled.

    The attempt costs its turns of model tokens and environment plus one
    assessment; the cohort cost per accepted task divides every attempt's
    cost by the accepted count, and triage charges each rejected attempt.
    """
    requires = ()
    produces = CohortAttemptCostResult

    def solve(self, tasks: int, accepted: int, input_price: Quantity, output_price: Quantity, turns: int,
              input_tokens_per_turn: int, output_tokens_per_turn: int, environment_per_turn: Quantity,
              assessment_per_attempt: Quantity, triage_per_rejected: Quantity) -> CohortAttemptCostResult:
        p_in, p_out = _per_mtok(input_price), _per_mtok(output_price)
        env, assess, triage = _usd(environment_per_turn), _usd(assessment_per_attempt), _usd(triage_per_rejected)
        model = turns * (input_tokens_per_turn * p_in + output_tokens_per_turn * p_out) / 1e6
        attempt = model + turns * env + assess
        total = tasks * attempt
        rejected = tasks - accepted
        total_h = total + rejected * triage
        return CohortAttemptCostResult(
            model_cost=model, environment_cost=turns * env, attempt_cost=attempt, cohort_cost=total,
            rejected=rejected, cost_per_accepted=total / accepted, rejected_per_accepted=rejected / accepted,
            cohort_cost_with_triage=total_h, cost_per_accepted_with_triage=total_h / accepted,
        )

    @staticmethod
    def compare(baseline: CohortAttemptCostResult, alternative: CohortAttemptCostResult,
                baseline_accepted: int, alternative_accepted: int) -> TokenPriceComparisonResult:
        return TokenPriceComparisonResult(
            baseline=baseline, alternative=alternative,
            cost_ratio=alternative.cost_per_accepted / baseline.cost_per_accepted,
            cost_ratio_with_triage=alternative.cost_per_accepted_with_triage / baseline.cost_per_accepted_with_triage,
            model_cut=1 - alternative.model_cost / baseline.model_cost,
            attempt_cut=1 - alternative.attempt_cost / baseline.attempt_cost,
            acceptance_cut=1 - alternative_accepted / baseline_accepted,
        )


class PrefixCacheBillModel(ForwardModel):
    """
    An append-only trajectory billed at the fresh rate, and with ideal prefix reuse.

    The logical input is ``H s0 + delta H (H - 1) / 2``; ideal reuse bills only
    ``s0 + (H - 1) delta`` at the input rate and the rest at the cached rate,
    so caching scales the quadratic term down without removing it. A miss on
    ``miss_fraction`` of the eligible prefix bills it at the full rate.
    """
    requires = ()
    produces = PrefixCacheBillResult

    def solve(self, turns: int, first_prompt: int, appended: int, output_per_turn: int,
              input_price: Quantity, cached_fraction: float, output_price: Quantity,
              miss_fraction: float) -> PrefixCacheBillResult:
        p_in, p_out = _per_mtok(input_price) * _USD, _per_mtok(output_price) * _USD
        s0, delta, k, cs = first_prompt, appended, output_per_turn, cached_fraction
        sent = turns * s0 + delta * turns * (turns - 1) // 2
        fresh = s0 + delta * (turns - 1)
        in_nocache = sent * p_in / 1_000_000
        in_cache = (fresh + cs * (sent - fresh)) * p_in / 1_000_000
        out_cost = turns * k * p_out / 1_000_000
        total_nocache, total_cache = in_nocache + out_cost, in_cache + out_cost
        cached_reads = cs * (sent - fresh) * p_in / 1_000_000
        miss = miss_fraction * (sent - fresh) * (1 - cs) * p_in / 1_000_000
        pin, pc, pout = _per_mtok(input_price) / 1e6, cs * _per_mtok(input_price) / 1e6, _per_mtok(output_price) / 1e6
        lengths = [s0 + (t - 1) * delta for t in range(1, turns + 1)]
        per_no = [L * pin + k * pout for L in lengths]
        per_re = [(s0 if t == 1 else delta) * pin + (0 if t == 1 else L - delta) * pc + k * pout
                  for t, L in enumerate(lengths, 1)]
        return PrefixCacheBillResult(
            sent_tokens=sent, fresh_tokens=fresh, input_uncached=in_nocache, input_cached=in_cache,
            output_cost=out_cost, total_uncached=total_nocache, total_cached=total_cache,
            cache_ratio=float((total_nocache / total_cache).magnitude),
            cached_reread_cost=cached_reads,
            reread_share=float((cached_reads / total_cache).to(_USD / _USD).magnitude),
            miss_cost=miss, miss_share=float((miss / total_cache).magnitude),
            per_turn_uncached=per_no, per_turn_cached=per_re,
        )


class CompactionPaybackModel(ForwardModel):
    """
    One compaction priced against keeping the transcript, under one tariff.

    Compacting at call ``n_c`` replaces ``S`` tokens of context with a
    ``S_c``-token summary that costs ``K`` output tokens and is resent fresh,
    so the compacting call costs ``p_out K + p_in S_c`` more; every later call
    reads ``S - S_c`` fewer cached tokens. Compaction has paid after
    ``n* = 1 + ceil(upfront / per-call saving)`` calls.
    """
    requires = ()
    produces = CompactionPaybackResult

    def solve(self, turns: int, first_prompt: int, appended: int, input_price: Quantity,
              cached_fraction: float, output_price: Quantity, compact_turn: int, compact_to: int,
              summary_tokens: int) -> CompactionPaybackResult:
        p_in, p_out = _per_mtok(input_price) * _USD, _per_mtok(output_price) * _USD
        s0, delta, cs = first_prompt, appended, cached_fraction
        ctx_at = s0 + (compact_turn - 1) * delta
        rest = turns - compact_turn + 1
        first_keep = (delta + (ctx_at - delta) * cs) * p_in / 1_000_000
        first_comp = compact_to * p_in / 1_000_000
        summary_cost = first_keep + summary_tokens * p_out / 1_000_000
        upfront = summary_cost + first_comp - first_keep
        saving = (ctx_at - compact_to) * cs * p_in / 1_000_000
        n_star = 1 + math.ceil((upfront / saving).magnitude)
        keep_sum = rest * ctx_at + delta * rest * (rest - 1) // 2
        keep_cost = (rest * delta + cs * (keep_sum - rest * delta)) * p_in / 1_000_000
        comp_sum = rest * compact_to + delta * rest * (rest - 1) // 2
        comp_fresh = compact_to + (rest - 1) * delta
        comp_cost = (comp_fresh + cs * (comp_sum - comp_fresh)) * p_in / 1_000_000
        net = {n: float((upfront - (n - 1) * saving).to(_USD).magnitude) for n in range(1, turns + 1)}
        return CompactionPaybackResult(
            context_at_compaction=ctx_at, remaining_calls=rest, summary_cost=summary_cost,
            upfront=upfront, per_call_saving=saving, breakeven_calls=n_star, keep_cost=keep_cost,
            compact_cost=comp_cost, compact_total=comp_cost + summary_cost,
            gap_before=upfront - (n_star - 2) * saving, saving_at=(n_star - 1) * saving - upfront,
            net_by_calls=net,
        )


class RetryPolicyModel(ForwardModel):
    """
    Bounded attempt policies under independent attempts at a fixed all-in cost.

    Sequential retry with early stopping covers ``1 - (1 - p)^k`` of tasks in
    ``coverage / p`` attempts on average, so its cost per accepted task is
    ``c / p`` whatever the cap; a batch of ``k`` pays for all ``k``.
    """
    requires = ()
    produces = RetryPolicyResult

    def solve(self, pass_probability: float, attempt_cost: Quantity, max_attempts: int,
              strong_pass_probability: float, strong_attempt_cost: Quantity) -> RetryPolicyResult:
        p, c, k = pass_probability, attempt_cost.to(_USD), max_attempts
        coverage = calc_independent_success(p, k)
        attempts = 0.0
        for j in range(k):
            attempts += (1 - p) ** j
        seq_cost, par_cost = c * attempts, k * c
        return RetryPolicyResult(
            coverage=coverage, expected_attempts=attempts, sequential_cost=seq_cost, batch_cost=par_cost,
            sequential_per_accepted=seq_cost / coverage, batch_per_accepted=par_cost / coverage,
            strong_per_accepted=strong_attempt_cost.to(_USD) / strong_pass_probability, batch_premium=k / attempts,
        )


class SerialEpisodeModel(ForwardModel):
    """
    One serial episode of disjoint intervals; a faster model shortens only its own.

    ``T' = T_other + T_model / s`` (Amdahl's law); instant model calls bound
    the episode speedup at ``1 / (1 - f)``.
    """
    requires = ()
    produces = SerialEpisodeResult

    def solve(self, model_intervals: Sequence[float], tests: float, lifecycle_intervals: Sequence[float],
              runtime_intervals: Sequence[float], model_speedup: float) -> SerialEpisodeResult:
        s = ureg.second
        model = sum(model_intervals)
        lifecycle = sum(lifecycle_intervals)
        runtime = sum(runtime_intervals)
        other = tests + lifecycle + runtime
        total = model + other
        changed = other + model / model_speedup
        return SerialEpisodeResult(
            model_time=model * s, tests_time=tests * s, lifecycle_time=lifecycle * s, runtime_time=runtime * s,
            other_time=other * s, total_time=total * s, model_fraction=model / total,
            changed_time=changed * s, saving=(total - changed) * s, speedup=total / changed,
            limiting_speedup=total / other, tests_half_saving=(tests / 2) * s,
        )


class SpeculativeEpisodeModel(ForwardModel):
    """
    Speculative decoding's decode-stream speedup, composed with a serial episode.

    Treats the episode's model time as pure decode (an optimistic bound):
    ``T' = T [(1 - f) + f / S]`` with ``S = (m + 1) / (gamma r + 1)``.
    """
    requires = ()
    produces = SpeculativeEpisodeResult

    def solve(self, total_time: float, decode_fraction: float, draft_length: int, draft_cost_ratio: float,
              accepted_high: float, accepted_low: float) -> SpeculativeEpisodeResult:
        s = ureg.second
        f = decode_fraction
        s_high = calc_speculative_round_speedup(accepted_high, draft_length, draft_cost_ratio)
        s_low = calc_speculative_round_speedup(accepted_low, draft_length, draft_cost_ratio)
        t_high = total_time * ((1 - f) + f / s_high)
        t_low = total_time * ((1 - f) + f / s_low)
        return SpeculativeEpisodeResult(
            break_even_accepted=draft_length * draft_cost_ratio, speedup_high=s_high, speedup_low=s_low,
            episode_time_high=t_high * s, episode_time_low=t_low * s, episode_speedup_high=total_time / t_high,
            episode_speedup_bound=1 / (1 - f),
        )


class ModelCascadeModel(ForwardModel):
    """
    A tiered cascade under one acceptance gate, priced over a completed cohort.

    Every reached tier pays its call and the gate; the cascade's cost per
    accepted task divides the whole spend by every tier's exits. Optionally a
    harm per false accept at the first tier, whose break-even rate erases the
    cascade's saving over the top tier alone.
    """
    requires = ()
    produces = ModelCascadeResult

    def solve(self, tasks: int, call_costs: Sequence[float], gate_cost: float, call_times: Sequence[float],
              gate_time: float, accepted_by_tier: Sequence[int], direct_accepted: int,
              false_accept_rate: Optional[float] = None,
              harm_per_false_accept: Optional[float] = None) -> ModelCascadeResult:
        n = len(call_costs)
        reached = [tasks]
        for j in range(n - 1):
            reached.append(reached[-1] - accepted_by_tier[j])
        spend, elapsed = 0.0, 0.0
        conditional, reaches = [], []
        for j in range(n):
            conditional.append(accepted_by_tier[j] / reached[j])
            reaches.append(reached[j] / tasks)
            spend += reached[j] * (call_costs[j] + gate_cost)
            elapsed += reached[j] * (call_times[j] + gate_time)
        accepted = sum(accepted_by_tier)
        per_accepted = spend / accepted
        direct_spend = tasks * (call_costs[-1] + gate_cost)
        direct_per = direct_spend / direct_accepted
        toll = 0.0
        for j in range(n - 1):
            toll += call_costs[j]
        toll += (n - 1) * gate_cost
        out = dict(
            reached=reached, conditional_acceptance=conditional, reach_shares=reaches, accepted=accepted,
            unresolved=tasks - accepted, spend=spend, per_initiated=spend / tasks, per_accepted=per_accepted,
            direct_spend=direct_spend, direct_per_accepted=direct_per,
            mean_time=elapsed / tasks, all_tiers_time=sum(call_times) + n * gate_time,
            direct_time=call_times[-1] + gate_time, direct_ratio=direct_per / per_accepted,
            lower_tier_toll=toll, saving_per_accepted=direct_per - per_accepted,
        )
        if harm_per_false_accept is not None:
            saving = direct_per - per_accepted
            out["false_accept_breakeven"] = saving * accepted / (accepted_by_tier[0] * harm_per_false_accept)
            if false_accept_rate is not None:
                harm = accepted_by_tier[0] * false_accept_rate * harm_per_false_accept
                out.update(harm_cost=harm, per_accepted_with_harm=(spend + harm) / accepted)
        return ModelCascadeResult(**out)


class TrajectoryAdmissionModel(ForwardModel):
    """
    Admitted trajectories against a token quota, in the mean.

    Little's law gives the trajectories present; the call and token rates
    scale with admissions. At fixed goodput, admissions scale as ``1 / alpha``,
    so a relative acceptance drop of ``1 - quota_share`` fills the quota.
    """
    requires = ()
    produces = TrajectoryAdmissionResult

    def solve(self, arrival: Quantity, residence: Quantity, calls: int, tokens_per_call: int, quota: Quantity,
              acceptance_drop: Optional[float] = None) -> TrajectoryAdmissionResult:
        concurrent = (arrival * residence).to("").magnitude
        call_rate = arrival * calls
        tokens = calls * tokens_per_call
        demand = arrival * tokens
        share = (demand / quota).to("").magnitude
        return TrajectoryAdmissionResult(
            concurrent=concurrent, call_rate=call_rate, tokens_per_trajectory=tokens, demand=demand,
            quota_share=share, arrival_ceiling=quota / tokens, acceptance_margin=1 - share,
            quota_share_after_drop=None if acceptance_drop is None else share / (1 - acceptance_drop),
        )


class KingmanQueueModel(ForwardModel):
    """
    Kingman's mean wait for one server at two utilizations, and its holding price.

    ``E[W] = (c_a^2 + c_s^2) / 2 * rho / (1 - rho) * E[S]``; each second of
    wait holds resources at ``holding_rate`` dollars.
    """
    requires = ()
    produces = KingmanQueueResult

    def solve(self, service: Quantity, arrival_scv: float, service_scv: float, low: float, high: float,
              holding_rate: float) -> KingmanQueueResult:
        v = (arrival_scv + service_scv) / 2
        wait_low = calc_kingman_wait_ratio(low, arrival_scv, service_scv) * service
        wait_high = calc_kingman_wait_ratio(high, arrival_scv, service_scv) * service
        return KingmanQueueResult(
            variability=v, wait_low=wait_low, wait_high=wait_high,
            holding_low=wait_low.to("second").magnitude * holding_rate,
            holding_high=wait_high.to("second").magnitude * holding_rate,
            wait_ratio=(wait_high / wait_low).to("").magnitude, throughput_gain=high / low - 1,
            unit_wait_utilization=1 / (1 + v),
        )


class DedicatedBreakEvenModel(ForwardModel):
    """
    Renting tokens from an API against owning capacity with a fixed hourly cost.

    The API's price per output token carries its input mix; owning pays a
    fixed ``F`` per hour plus ``v`` per output token. The costs tie at load
    ``u* = F / (3600 R (P_eff - v))`` of the reference rate ``R``.
    """
    requires = ()
    produces = DedicatedBreakEvenResult

    @staticmethod
    def hourly_costs(load: float, reference_rate: Quantity, api_price: float, fixed: float, variable: float):
        tokens = load * (reference_rate * ureg.hour).to("").magnitude
        return tokens * api_price / 1_000_000, fixed + tokens * variable / 1_000_000

    def solve(self, input_per_output: float, cached_fraction: float, input_price: float, cached_price: float,
              output_price: float, fixed_per_hour: float, variable_price: float, reference_rate: Quantity,
              load: float) -> DedicatedBreakEvenResult:
        weighted = (1 - cached_fraction) * input_price + cached_fraction * cached_price
        api = output_price + input_per_output * weighted
        ref = (reference_rate * ureg.hour).to("").magnitude
        delivered = load * ref
        api_hour = delivered * api / 1_000_000
        ded_hour = fixed_per_hour + delivered * variable_price / 1_000_000
        crossing = fixed_per_hour / (ref * (api - variable_price) / 1_000_000)
        api_nc = output_price + input_per_output * input_price
        return DedicatedBreakEvenResult(
            weighted_input_price=weighted, api_price=api, reference_hour_tokens=ref,
            delivered_hour_tokens=delivered, api_hour=api_hour, dedicated_hour=ded_hour, crossing=crossing,
            saving_hour=api_hour - ded_hour, api_price_no_cache=api_nc,
            crossing_no_cache=fixed_per_hour / (ref * (api_nc - variable_price) / 1_000_000),
        )


class ConservedBudgetModel(ForwardModel):
    """
    A root budget kept as settled + reserved + free across two children.

    Granting moves free to reserved; a child that closes settles its charge
    and returns the rest. A child stopped with a remote operation pending
    keeps that operation's enforceable bound reserved until its final bill
    settles, once per settlement identity however often it is delivered.
    """
    requires = ()
    produces = ConservedBudgetResult

    def solve(self, budget: int, planning: int, allocations: Sequence[int], settled_children: Sequence[int],
              pending: int, late_charge: int, deliveries: int = 2) -> ConservedBudgetResult:
        b_first, b_second = allocations
        e_first, e_second = settled_children
        f_plan = budget - planning
        r_deleg = b_first + b_second
        f_deleg = f_plan - r_deleg
        refund = b_first - e_first
        e_after = planning + e_first
        r_after = r_deleg - b_first
        f_after = f_deleg + refund
        free_at_stop = b_second - e_second - pending
        apparent = b_second - e_second
        applied: Dict[str, int] = {}
        settled_second = e_second
        for key, charge in [("final_bill", late_charge)] * deliveries:
            if key in applied:
                if applied[key] != charge:
                    raise ValueError("conflicting payload for one settlement identity")
            else:
                applied[key] = charge
                settled_second += charge
        free_settled = free_at_stop + pending - late_charge
        e_final = e_after + settled_second
        r_final = r_after - b_second
        f_final = f_after + free_settled
        states = [(planning, 0, f_plan), (planning, r_deleg, f_deleg), (e_after, r_after, f_after),
                  (e_after, r_after, f_after), (e_final, r_final, f_final)]
        for s_, r_, f_ in states:
            if s_ + r_ + f_ != budget or min(s_, r_, f_) < 0:
                raise ValueError("budget not conserved")
        return ConservedBudgetResult(
            units_per_dollar=1000, budget=budget, planning=planning, allocations=(b_first, b_second),
            settled_children=(e_first, e_second), pending=pending, late_charge=late_charge,
            free_after_planning=f_plan, reserved_delegated=r_deleg, free_after_delegation=f_deleg,
            refund_first=refund, settled_after_first=e_after, reserved_after_first=r_after,
            free_after_first=f_after, child_free_at_stop=free_at_stop, child_apparent_free=apparent,
            child_free_settled=free_settled, settled_final=e_final, reserved_final=r_final,
            free_final=f_final, states=states,
        )


class ArchitectureChoiceModel(ForwardModel):
    """
    Designs priced by automated spend alone and with a human fallback.

    A bounded loop stops at its first accepted turn, so turn ``j`` is reached
    by the share that every earlier turn rejected. Rejected tickets go to a
    person who resolves a fraction of them; cost per resolved ticket divides
    the combined spend by combined resolution.
    """
    requires = ()
    produces = ArchitectureChoiceResult

    def solve(self, tickets: int, single_costs: Mapping[str, Quantity], single_acceptance: Mapping[str, float],
              loop_name: str, turn_cost: Quantity, turn_acceptance: Sequence[float], check_cost: Quantity,
              human_cost: Quantity, human_resolution: float, point: float) -> ArchitectureChoiceResult:
        turn_cost, check_cost, human_cost = turn_cost.to(_USD), check_cost.to(_USD), human_cost.to(_USD)
        single_costs = {k: v.to(_USD) for k, v in single_costs.items()}
        reach = [1.0]
        for r in turn_acceptance[:-1]:
            reach.append(reach[-1] * (1 - r))
        p_loop = 1 - reach[-1] * (1 - turn_acceptance[-1])
        turns = 0
        for x in reach:
            turns = turns + x
        auto, accept = {}, {}
        for name, c in single_costs.items():
            auto[name] = c + check_cost
            accept[name] = single_acceptance[name]
        auto[loop_name] = turns * (turn_cost + check_cost)
        accept[loop_name] = p_loop
        per_auto = {k: auto[k] / accept[k] for k in auto}
        combined = {k: auto[k] + (1 - accept[k]) * human_cost for k in auto}
        resolved = {k: accept[k] + (1 - accept[k]) * human_resolution for k in auto}
        per_res = {k: combined[k] / resolved[k] for k in auto}
        return ArchitectureChoiceResult(
            loop_acceptance=p_loop, loop_turns=turns, reach=reach, automated=auto,
            automated_per_accepted=per_auto, combined=combined, resolved=resolved, per_resolved=per_res,
            month_spend=tickets * combined[loop_name], month_resolved=tickets * resolved[loop_name],
            point_value=point * human_cost,
        )


class AdaptationPaybackModel(ForwardModel):
    """
    Tasks until a one-time adaptation repays its data, training, and evaluation.

    Each task's model usage is steps times (input tokens x input price +
    output tokens x output price); the break-even count is the investment
    over the per-task saving, and the first whole task that crosses it.
    """
    requires = ()
    produces = AdaptationPaybackResult

    def solve(self, base: Mapping[str, float], adapted: Mapping[str, float], data_cost: Quantity,
              training_cost: Quantity, evaluation_cost: Quantity, monthly_volume: int,
              low_volume: int) -> AdaptationPaybackResult:
        def price(q):
            return _per_mtok(q) * _USD / 1_000_000

        b_pin, b_pout = price(base["input_price"]), price(base["output_price"])
        a_pin, a_pout = price(adapted["input_price"]), price(adapted["output_price"])
        base_step = base["input_tokens"] * b_pin + base["output_tokens"] * b_pout
        base_task = base["steps"] * base_step
        adapt_step = adapted["input_tokens"] * a_pin + adapted["output_tokens"] * a_pout
        adapt_task = adapted["steps"] * adapt_step
        data_cost, training_cost, evaluation_cost = (x.to(_USD) for x in (data_cost, training_cost, evaluation_cost))
        invest = data_cost + training_cost + evaluation_cost
        delta = base_task - adapt_task
        n_star = (invest / delta).to("").magnitude
        input_saving = base["steps"] * base["input_tokens"] * b_pin - adapted["steps"] * adapted["input_tokens"] * a_pin
        return AdaptationPaybackResult(
            base_step=base_step, base_task=base_task, adapted_step=adapt_step, adapted_task=adapt_task,
            investment=invest, saving_per_task=delta, breakeven_tasks=n_star, breakeven_whole=math.ceil(n_star),
            payback_months=n_star / monthly_volume, payback_months_low=n_star / low_volume,
            input_saving_share=(input_saving / delta).to("").magnitude,
            evaluation_share=(evaluation_cost / invest).to("").magnitude,
            training_share=(training_cost / invest).to("").magnitude,
        )


class WorkloadLeverModel(ForwardModel):
    """
    Cost per accepted task by workload, ``C_eff = C_attempt / p_acc``, and three levers.

    Halving the token price saves half the model share; halving the largest
    term saves half its share; raising acceptance by ``dp`` saves
    ``dp / (p + dp)`` of every term, and also removes the handoffs charged per
    rejected attempt. Acceptance beats halving tokens whenever the model share
    is below ``2 dp / (p + dp)``.
    """
    requires = ()
    produces = WorkloadLeverResult

    def solve(self, workloads: Mapping[str, Mapping[str, float]], acceptance_gain: float) -> WorkloadLeverResult:
        dp = acceptance_gain
        rows = {}
        for name, w in workloads.items():
            p = w["acceptance"]
            handoff = w.get("handoff")
            human = (1 - p) * handoff if handoff is not None else w["human"]
            terms = {"model": w["model"], "environment": w["environment"], "assessment": w["assessment"],
                     "human": human}
            a = terms["model"] + terms["environment"] + terms["assessment"] + terms["human"]
            c = a / p
            big_name = max(terms, key=terms.get)
            big = max(terms["model"], terms["environment"], terms["assessment"], terms["human"])
            p2 = p + dp
            h2 = (1 - p2) * handoff if handoff is not None else human
            rows[name] = WorkloadLeverRow(
                model=terms["model"], environment=terms["environment"], assessment=terms["assessment"],
                human=human, acceptance=p, attempt=a, cost_per_accepted=c, model_share=terms["model"] / a,
                largest_term=big_name,
                halve_tokens=1 - (a - terms["model"] / 2) / p / c,
                halve_largest=1 - (a - big / 2) / p / c,
                acceptance_gain=1 - (a - human + h2) / p2 / c,
                token_threshold=2 * dp / (p + dp),
            )
        return WorkloadLeverResult(rows=rows)




