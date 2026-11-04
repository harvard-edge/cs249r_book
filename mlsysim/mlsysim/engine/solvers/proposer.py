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
from ...physics.memory import calc_kv_cache_size
from .base import ForwardModel


def _ratio(a, b) -> float:
    return float((a / b).to("").magnitude)


def _pure(q) -> float:
    return float(q.to("").magnitude)


class VlaEvaluationModel(ForwardModel):
    """
    One fresh evaluation of a decoder-only VLA backbone on one processor.

    Wraps ``calc_vla_inference_latency`` (compute/memory bounds for prefill,
    then ``action_tokens - 1`` decode passes, or one longer prefill
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
        """Estimate one decoder-backbone evaluation and its resource bounds.

        Parameters
        ----------
        model : TransformerWorkload
            Decoder geometry with parameter count in param, layer count, and hidden
            width. The supplied model represents the language/action backbone.
        processor : HardwareNode
            Processor with positive FP16 compute throughput and memory bandwidth.
            An explicit fp16 override takes precedence over canonical peak_flops.
        weight_bytes_per_param : Quantity
            Positive stored bytes per parameter, in byte/param. Weight quantization
            changes traffic while arithmetic still uses the processor's fp16 rate.
        input_tokens : int
            Positive count of prompt and image-patch tokens entering the backbone.
        action_tokens : int
            Positive number of generated action tokens.
        bus_efficiency : float
            Caller-supplied sustained fraction of peak memory bandwidth in (0, 1].
        compute_utilization : float
            Caller-supplied sustained fraction of fp16 compute throughput in (0, 1].
        parallel_decode : bool, optional
            If True, process input and action placeholders in one bidirectional pass;
            otherwise prefill produces the first token before serial decoding.

        Returns
        -------
        VlaEvaluationResult
            Weight storage in GB, work in FLOPs, intensities in FLOP/byte, phase
            times in ms, and reciprocal evaluation rates in Hz. These are modeled
            resource bounds; this result does not check model residency or control safety.

        See Also
        --------
        mlsysim.physics.transformer.calc_vla_inference_latency : Compose compute and weight-traffic bounds for each pass.
        mlsysim.physics.transformer.calc_transformer_prefill_flops : Count parameter-linear and attention prefill work.

        Notes
        -----
        Equation: weights = parameters * weight_bytes_per_param. Each pass takes
            max(compute_work / effective_compute, weights / effective_bandwidth).
            Serial decode repeats its pass action_tokens - 1 times; parallel decode
            instead extends prefill to input_tokens + action_tokens. Prefill work
            includes the helper's attention term, while reported prefill_intensity
            intentionally uses only parameter-linear work.
        Assumptions: one full weight read per pass, compute/memory overlap within
            a pass, fp16 arithmetic, and fixed caller-provided utilization factors.
            The peak ridge uses peak rates, not efficiency-adjusted rates.
        Excludes: the vision encoder, decode attention and KV traffic, runtime
            workspaces, launch overhead, memory admission, and closed-loop dynamics.

        References
        ----------
        Derivation: compose the prefill and serial-pass bounds implemented and
            sourced in calc_vla_inference_latency; divide work by weight bytes for
            arithmetic intensity and invert modeled time for the evaluation rate.
            The decoder-family choice is an assumed execution pattern, not a
            reproduction of a published robot's latency measurement.

        Examples
        --------
        A one-token constructed request has no post-prefill decode pass.

        >>> from mlsysim import Models, Hardware, Q_
        >>> result = VlaEvaluationModel().solve(
        ...     Models.Language.Llama3_8B, Hardware.Edge.JetsonAGXOrin,
        ...     Q_(2, 'byte/param'), 8, 1, 0.5, 0.5)
        >>> result.decode_steps == 0 and result.t_total == result.t_prefill
        True
        """
        n = model.parameters
        weights = (n * weight_bytes_per_param).to(ureg.GB)
        peak = processor.compute.precision_flops.get("fp16", processor.compute.peak_flops)
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
        """Compare decoder schedules under a shared backbone and resource model.

        Parameters
        ----------
        model : TransformerWorkload
            Backbone with parameter count in param, layer count, and hidden width.
        processor : HardwareNode
            Device with positive FP16 compute throughput and memory bandwidth.
            An explicit fp16 override takes precedence over canonical peak_flops.
        weight_bytes_per_param : Quantity
            Positive backbone storage per parameter, in byte/param.
        input_tokens : int
            Positive prompt/prefix length processed before action generation.
        bus_efficiency : float
            Sustained memory-bandwidth fraction in (0, 1], supplied by the caller.
        compute_utilization : float
            Sustained fp16 compute fraction in (0, 1], supplied by the caller.
        horizon : int
            Positive number of action setpoints in one chunk.
        action_dim : int
            Positive tokens per setpoint in the binned autoregressive schedule.
        fast_tokens : int
            Positive compressed action-token count in the FAST-like schedule.
        expert_params : Quantity
            Nonnegative flow-expert parameter count, in param.
        expert_steps : int
            Nonnegative number of expert refinement passes after backbone prefill.
        expert_bytes_per_param : Quantity
            Positive stored bytes per expert parameter, in byte/param.
        kv_bytes_per_value : Quantity
            Positive bytes per cached key or value element, in bytes.
        renewal_period : Quantity
            Positive interval between requested chunks, in seconds.

        Returns
        -------
        DecoderFamiliesResult
            Modeled decoder times in ms, expert/prefix storage in MB, token counts,
            and dimensionless time ratios. No quality-equivalence or memory-capacity
            guarantee is implied by comparing these schedules.

        See Also
        --------
        VlaEvaluationModel.solve : Evaluate the shared backbone under one decoder schedule.

        Notes
        -----
        Equation: binned_tokens = horizon * action_dim. Each autoregressive schedule
            uses VlaEvaluationModel; the parallel schedule extends prefill. An expert
            pass takes max(2 * expert_params * horizon / effective_compute,
            (expert_weights + prefix_kv) / sustained_bandwidth), where prefix_kv =
            2 * layers * hidden_dim * kv_bytes_per_value * input_tokens.
            Expert total = backbone_prefill + expert_steps * expert_pass_time.
        Assumptions: shared backbone inputs and rates, full-width keys and values,
            fixed token/refinement counts, and repeated prefix-cache reads per expert
            pass. Expert work uses parameter-linear FLOPs for each horizon position.
        Excludes: expert attention/activation traffic beyond the specified prefix
            reads, launch overhead, residency checks, and decoder-quality prediction.

        References
        ----------
        Derivation: count one key and one value at each layer/hidden coordinate for
            each prefix token; combine its byte transfer with an expert weight sweep.
            The maximum of the computed resource times is the pass lower bound.
            Decoder token counts and expert iterations are scenario inputs, not
            measured performance or equal-quality guarantees.

        Examples
        --------
        A constructed chunk exposes its binned token count directly.

        >>> from mlsysim import Models, Hardware, Q_
        >>> result = DecoderFamiliesModel().solve(
        ...     Models.Language.Llama3_8B, Hardware.Edge.JetsonAGXOrin,
        ...     Q_(2, 'byte/param'), 8, 0.5, 0.5, 2, 3, 4,
        ...     Q_(1000, 'param'), 2, Q_(2, 'byte/param'), Q_(2, 'byte'),
        ...     Q_(1, 'second'))
        >>> result.binned_tokens == 2 * 3
        True
        """
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
        kv_per_token = calc_kv_cache_size(
            model.layers, 1, model.hidden_dim, 1, 1, kv_bytes_per_value
        ).to(ureg.MB)
        prefix_kv = (kv_per_token * input_tokens).to(ureg.MB)
        step_bw = ((expert_weights + prefix_kv) / binned.sustained_bandwidth).to(ms)
        rate = processor.compute.precision_flops.get("fp16", processor.compute.peak_flops) * compute_utilization
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
        """Account for two learned proposers sharing a machine's memory bus.

        Parameters
        ----------
        machine : MobileManipulatorPlatform
            Platform providing intent/chunk models, stored intent weights, processor
            bandwidth, DRAM efficiency, chunk horizon/step, proposer rates, permission
            rate, and intent latency breakdown. Rates are in Hz and times in seconds;
            bandwidth, horizon, and step must be positive and efficiency in (0, 1].
        fp16_bytes_per_param : Quantity
            Positive reference storage per parameter, in byte/param. The FP16
            comparison normally supplies two bytes; deployed intent storage comes
            separately from machine.intent_weight_memory.

        Returns
        -------
        ProposerTrafficResult
            Weight storage in GB, sweep times in ms, frequencies in Hz, traffic in
            GB/s, and dimensionless bus/duty ratios. A bus share above one indicates
            demand exceeding the supplied bandwidth model, not measured utilization.

        See Also
        --------
        mlsysim.physics.robotics.calc_action_chunk_streaming_amortization : Divide sweep time among supplied chunk setpoints.

        Notes
        -----
        Equation: sustained_bandwidth = peak_bandwidth * dram_efficiency;
            sweep_time = weights / sustained_bandwidth. Intent traffic is
            action_dim * deployed_intent_weights * intent_rate, and chunk traffic
            is chunk_weights * chunk_rate. Their sum divided by sustained_bandwidth
            is proposer_share. The single-sweep chunk comparison divides reference
            weight traffic at the setpoint rate by chunk_horizon.
        Assumptions: a full intent-model weight sweep per action dimension and a
            full chunk-model sweep per renewal. The machine supplies deployed weight
            precision and latency assumptions independently of the FP16 comparison.
        Excludes: KV/activation traffic, vision encoding, other bus clients, cache
            reuse, compute contention, scheduling jitter, and physical safety claims.

        References
        ----------
        Derivation: weight bytes per evaluation multiplied by evaluations per second
            give bandwidth demand; sum both proposer streams and divide by supplied
            sustained bandwidth. The linked chunk-streaming helper derives sweep
            amortization per setpoint. This is explicit traffic accounting, not a
            hardware benchmark or a model of application-level success.

        Examples
        --------
        The registered platform's total traffic is the sum of its two streams.

        >>> from mlsysim import Embodied, Q_
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> result = ProposerTrafficModel().solve(machine, Q_(2, 'byte/param'))
        >>> result.proposer_traffic == result.intent_traffic + result.chunk_traffic
        True
        """
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
        """Compare tracking-error growth with a chunk-renewal schedule.

        Parameters
        ----------
        tolerance : Quantity
            Positive permitted position-error magnitude, in meters.
        initial_error : Quantity
            Nonnegative current error, in meters, smaller than tolerance.
        target_speed : Quantity
            Positive untracked target speed used for the comparison deadline, in m/s.
        slip_acceleration : Quantity
            Positive residual acceleration after tracking, in meters per second squared.
        chunk_rate : Quantity
            Positive periodic chunk-renewal frequency, in Hz.
        chunk_step : Quantity
            Positive interval between chunk setpoints, in seconds.
        chunk_horizon : int
            Positive number of setpoints; must cover renewal plus inference latency.
        inference_latency : Quantity
            Positive evaluation time, in seconds, less than the renewal period.

        Returns
        -------
        EvidenceHorizonResult
            Spatial margins/drifts in mm, horizons and schedule times in ms, and
            setpoint counts. fits means replacement age is below the modeled error
            horizon; it is not an assurance of collision avoidance or tracking stability.

        Raises
        ------
        ValueError
            If the renewal period is not aligned to whole setpoint intervals, the
            inference misses its next renewal, or the chunk cannot bridge replacement.

        See Also
        --------
        mlsysim.physics.robotics.calc_tracked_evidence_horizon : Solve the constant-acceleration error horizon.
        mlsysim.physics.robotics.calc_chunk_renewal_schedule : Account for renewal alignment and replacement age.

        Notes
        -----
        Equation: margin = tolerance - initial_error. Untracked deadline =
            margin / target_speed; tracked horizon = sqrt(2 * margin / slip_acceleration).
            Replacement age = 1 / chunk_rate + inference_latency, and position drift
            at age T is slip_acceleration * ``T**2`` / 2. A chunk spans
            chunk_horizon * chunk_step; fits compares replacement age with horizon.
        Assumptions: initial velocity error has been removed by tracking, residual
            acceleration is constant, and renewal aligns with the setpoint grid.
        Excludes: jerk, measurement noise growth, missed renewals, transport/admission
            delays, obstacle motion beyond the stated target model, and control stability.

        References
        ----------
        Derivation: integrate constant residual acceleration twice to obtain
            position error ``a*T**2/2``, then solve for the remaining margin. Compose
            this horizon with the linked periodic-renewal schedule; the comparison
            inherits both models' assumptions and is not a certified safety envelope.

        Examples
        --------
        The constructed schedule's replacement age includes inference latency.

        >>> from mlsysim import Q_
        >>> result = EvidenceHorizonModel().solve(
        ...     Q_(1, 'm'), Q_(0.1, 'm'), Q_(1, 'm/s'), Q_(1, 'm/s^2'),
        ...     Q_(10, 'Hz'), Q_(0.01, 's'), 20, Q_(0.02, 's'))
        >>> result.age_at_replacement == result.renewal_period + Q_(0.02, 's')
        True
        """
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
