"""Learned-proposer solvers: what one fresh evaluation costs, and how long its evidence lasts.

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). The equations live in
``physics.transformer`` (the roofline latency of one vision-language-action
evaluation) and ``physics.robotics`` (chunk streaming, the renewal schedule,
the tracked evidence horizon); these models compose them for one stated
machine or situation (added 2026-10-20).
"""

from __future__ import annotations

from ..results import (
    DecoderFamiliesResult,
    EvidenceHorizonResult,
    ProposerTrafficResult,
    VlaEvaluationResult,
)
from ...core.types import Quantity
from ...core.units import ureg
from ...physics.robotics import (
    calc_action_chunk_streaming_amortization,
    calc_chunk_renewal_schedule,
    calc_tracked_evidence_horizon,
)
from ...physics.transformer import calc_transformer_prefill_flops, calc_vla_inference_latency
from .base import ForwardModel


def _ratio(a, b) -> float:
    return float((a / b).to("").magnitude)


def _pure(q) -> float:
    return float(q.to("").magnitude)


class VlaEvaluationModel(ForwardModel):
    """
    One fresh evaluation of a decoder-only VLA backbone on one processor.

    Wraps ``calc_vla_inference_latency`` (compute-bound prefill, then
    ``action_tokens - 1`` bandwidth-bound decode sweeps, or one longer prefill
    under parallel decoding) and adds the roofline coordinates of its two
    phases: the ridge ``R_peak / BW_peak`` at the 16-bit rate, the prefill
    intensity (linear FLOPs per weight byte, about S at 16-bit weights), and
    the decode intensity (2 N FLOP per weight sweep per weight byte, 1 at
    16-bit weights).
    """
    requires = ("model", "processor")
    produces = VlaEvaluationResult

    def solve(self, model, processor, weight_bytes_per_param: Quantity, input_tokens: int, action_tokens: int,
              bus_efficiency: float, compute_utilization: float,
              parallel_decode: bool = False) -> VlaEvaluationResult:
        n = model.parameters
        weights = (n * weight_bytes_per_param).to(ureg.GB)
        peak = processor.compute.precision_flops["fp16"]
        bw = processor.memory.bandwidth
        r = calc_vla_inference_latency(
            n, weights, input_tokens, action_tokens, peak, bw,
            bus_efficiency=bus_efficiency, compute_utilization=compute_utilization,
            parallel_decode=parallel_decode, n_layers=model.layers, hidden_dim=model.hidden_dim,
        )
        linear = calc_transformer_prefill_flops(n, r["prefill_tokens"]).to(ureg.flop)
        decode_flops = (2 * n.to(ureg.count).magnitude) * ureg.flop
        ms = ureg.millisecond
        return VlaEvaluationResult(
            weight_memory=weights,
            sustained_bandwidth=(bw * bus_efficiency).to(ureg.GB / ureg.second),
            achieved_rate=(peak * compute_utilization).to("TFLOPs/s"),
            ridge_intensity=(peak / bw).to(ureg.flop / ureg.byte),
            prefill_tokens=r["prefill_tokens"],
            decode_steps=r["decode_steps"],
            prefill_linear_flops=linear,
            prefill_attention_flops=(r["prefill_flops"] - linear).to(ureg.flop),
            prefill_flops=r["prefill_flops"].to(ureg.flop),
            prefill_intensity=(linear / weights).to(ureg.flop / ureg.byte),
            decode_intensity=(decode_flops / weights).to(ureg.flop / ureg.byte),
            t_prefill_compute=r["t_prefill_compute"].to(ms),
            t_weight_sweep=r["t_weight_sweep"].to(ms),
            weight_sweep_rate=(1 / r["t_weight_sweep"]).to(ureg.hertz),
            t_prefill=r["t_prefill"].to(ms),
            prefill_bound=r["prefill_bound"],
            t_decode_step=r["t_decode_step"].to(ms),
            t_decode=r["t_decode"].to(ms),
            t_total=r["t_total"].to(ms),
            f_max=r["f_max"].to(ureg.hertz),
            prefill_share=_ratio(r["t_prefill"], r["t_total"]),
            prefill_ceiling=(1 / r["t_prefill"]).to(ureg.hertz),
        )


class DecoderFamiliesModel(ForwardModel):
    """
    One fresh H-step chunk from a VLA backbone, for each decoder family.

    Every family pays the backbone's prefill. Binned autoregressive decoding
    emits ``horizon * action_dim`` tokens one sweep at a time; FAST emits
    ``fast_tokens`` compressed tokens one at a time; a parallel head folds the
    binned tokens into one longer prefill; a flow-matching expert runs
    ``expert_steps`` refinement steps, each ``max(compute, (expert weights +
    cached prefix KV) / B_sustained)`` (Black et al. 2024; Pertsch et al.
    2025; Kim, Finn and Liang 2025).
    """
    requires = ("model", "processor")
    produces = DecoderFamiliesResult

    def solve(self, model, processor, weight_bytes_per_param: Quantity, input_tokens: int,
              bus_efficiency: float, compute_utilization: float, horizon: int, action_dim: int,
              fast_tokens: int, expert_params: Quantity, expert_steps: int,
              expert_bytes_per_param: Quantity, kv_bytes_per_value: Quantity,
              renewal_period: Quantity) -> DecoderFamiliesResult:
        ev = VlaEvaluationModel()

        def run(tokens, parallel=False):
            return ev.solve(model, processor, weight_bytes_per_param, input_tokens, tokens,
                            bus_efficiency, compute_utilization, parallel_decode=parallel)

        ms = ureg.millisecond
        binned_tokens = horizon * action_dim
        binned = run(binned_tokens)
        fast = run(fast_tokens)
        parallel = run(binned_tokens, parallel=True)
        prefill = run(1).t_total  # prefill only: the first token comes out of the prefill
        expert_weights = (expert_params * expert_bytes_per_param).to(ureg.MB)
        kv_per_token = (2 * model.layers * model.hidden_dim * kv_bytes_per_value).to(ureg.MB)
        prefix_kv = (kv_per_token * input_tokens).to(ureg.MB)
        step_bw = ((expert_weights + prefix_kv) / binned.sustained_bandwidth).to(ms)
        rate = processor.compute.precision_flops["fp16"] * compute_utilization
        step_compute = (2 * expert_params.to(ureg.param).magnitude * horizon * ureg.flop / rate).to(ms)
        step = max(step_bw, step_compute)
        expert = (prefill + expert_steps * step).to(ms)
        return DecoderFamiliesResult(
            binned_tokens=binned_tokens,
            t_prefill=prefill.to(ms),
            t_binned=binned.t_total,
            t_fast=fast.t_total,
            t_parallel=parallel.t_total,
            expert_weights=expert_weights,
            kv_per_token=kv_per_token,
            prefix_kv=prefix_kv,
            t_expert_step_bandwidth=step_bw,
            t_expert_step_compute=step_compute,
            t_expert_step=step,
            t_expert=expert,
            binned_over_expert=_ratio(binned.t_total, expert),
            prefill_over_period=_ratio(prefill, renewal_period),
        )


class ProposerTrafficModel(ForwardModel):
    """
    Weight traffic of a machine's two learned proposers on its shared memory bus.

    The intent model's 16-bit sweep ``M / (B eta)`` bounds its fresh rate;
    spread over a chunk of H setpoints it is ``t / H`` per setpoint
    (``calc_action_chunk_streaming_amortization``). One sweep per setpoint at
    the setpoint rate would demand ``f_setpoint M``; one per chunk divides that
    by H. In operation the deployed intent model sweeps its weights once per
    action token per evaluation (``D M_deployed f_intent``) and the chunk
    policy once per renewal (``M_chunk f_chunk``); their sum against the
    sustained bandwidth is the proposers' share of the bus (weight traffic only).
    """
    requires = ("machine",)
    produces = ProposerTrafficResult

    def solve(self, machine, fp16_bytes_per_param: Quantity) -> ProposerTrafficResult:
        soc = machine.brain_soc
        eta = machine.dram_efficiency
        H = machine.chunk_horizon
        gb, gbs, ms, hz = ureg.GB, ureg.GB / ureg.second, ureg.millisecond, ureg.hertz
        w16 = (machine.intent_model.parameters * fp16_bytes_per_param).to(gb)
        w_dep = machine.intent_weight_memory.to(gb)
        w_chunk = (machine.chunk_model.parameters * fp16_bytes_per_param).to(gb)
        intent = calc_action_chunk_streaming_amortization(w16, soc.memory.bandwidth, H, bus_efficiency=eta)
        deployed = calc_action_chunk_streaming_amortization(w_dep, soc.memory.bandwidth, H, bus_efficiency=eta)
        chunk = calc_action_chunk_streaming_amortization(w_chunk, soc.memory.bandwidth, H, bus_efficiency=eta)
        sus = intent["b_sustained"].to(gbs)
        setpoint_rate = (1 / machine.chunk_step).to(hz)
        demand = (setpoint_rate * w16).to(gbs)
        chunked = (demand / H).to(gbs)
        sweeps = machine.intent_model.action_dim
        intent_traffic = (sweeps * w_dep * machine.intent_rate).to(gbs)
        chunk_traffic = (w_chunk * machine.chunk_rate).to(gbs)
        total = (intent_traffic + chunk_traffic).to(gbs)
        return ProposerTrafficResult(
            sustained_bandwidth=sus,
            intent_weights_fp16=w16,
            intent_weights_deployed=w_dep,
            chunk_weights=w_chunk,
            intent_sweep_fp16=intent["t_stream"].to(ms),
            intent_sweep_rate_fp16=intent["f_single"].to(hz),
            intent_sweep_deployed=deployed["t_stream"].to(ms),
            intent_sweep_rate_deployed=deployed["f_single"].to(hz),
            amortized_sweep=intent["tau_step"].to(ms),
            chunk_sweep=chunk["t_stream"].to(ms),
            chunk_sweep_rate=chunk["f_single"].to(hz),
            chunk_sweep_share=_pure(chunk["t_stream"] * machine.chunk_rate),
            tick_over_sweep_rate=_ratio(machine.permission_rate, intent["f_single"]),
            setpoint_rate=setpoint_rate,
            single_step_demand=demand,
            single_step_overload=_ratio(demand, sus),
            chunked_demand=chunked,
            chunked_share=_ratio(chunked, sus),
            renewal_gap=_pure(machine.intent_inference_latency * machine.chunk_rate),
            sweeps_per_evaluation=sweeps,
            intent_traffic=intent_traffic,
            chunk_traffic=chunk_traffic,
            proposer_traffic=total,
            proposer_share=_ratio(total, sus),
            decode_duty=_pure(machine.intent_latency_breakdown["t_decode"] * machine.intent_rate),
        )


class EvidenceHorizonModel(ForwardModel):
    """
    How long a chunk's evidence stays true for a tracked target, against its renewal schedule.

    Untracked, a target at speed ``v`` uses up the margin ``r_tol - e_0`` in
    ``(r_tol - e_0) / v``. Tracked, only a residual acceleration remains, so
    the evidence horizon is ``sqrt(2 (r_tol - e_0) / a_slip)``
    (``calc_tracked_evidence_horizon``). The chunk policy's renewal schedule
    (``calc_chunk_renewal_schedule``) gives the evidence age at replacement
    (renewal period plus inference), the setpoints one renewal needs, and the
    span one chunk supplies; drift over an age ``T`` is ``a_slip T^2 / 2``.
    """
    requires = ("target", "schedule")
    produces = EvidenceHorizonResult

    def solve(self, tolerance: Quantity, initial_error: Quantity, target_speed: Quantity,
              slip_acceleration: Quantity, chunk_rate: Quantity, chunk_step: Quantity, chunk_horizon: int,
              inference_latency: Quantity) -> EvidenceHorizonResult:
        mm, ms = ureg.millimeter, ureg.millisecond
        margin = (tolerance - initial_error).to(mm)
        horizon = calc_tracked_evidence_horizon(tolerance, initial_error, slip_acceleration).to(ms)
        sched = calc_chunk_renewal_schedule(chunk_rate, chunk_step, chunk_horizon, inference_latency)
        age = sched["open_loop_span"].to(ms)
        span = sched["chunk_span"].to(ms)
        drift_age = (0.5 * slip_acceleration * age ** 2).to(mm)
        drift_span = (0.5 * slip_acceleration * span ** 2).to(mm)
        return EvidenceHorizonResult(
            margin=margin,
            untracked_deadline=(margin / target_speed).to(ms),
            horizon=horizon,
            renewal_period=sched["renewal_period"].to(ms),
            age_at_replacement=age,
            drift_at_replacement=drift_age,
            horizon_spare=(horizon - age).to(ms),
            fits=age < horizon,
            min_supply_setpoints=sched["min_supply_setpoints"],
            chunk_span=span,
            drift_at_chunk_end=drift_span,
            error_at_chunk_end=(initial_error + drift_span).to(mm),
            outlast=(span - horizon).to(ms),
            setpoints_within_horizon=_ratio(horizon, chunk_step),
        )
