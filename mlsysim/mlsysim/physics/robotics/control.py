"""Action chunking, watchdog leases, real-time fieldbuses, and CBF safety."""

from __future__ import annotations

import math
import numpy as np

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_at_least, validate_nonnegative, validate_positive

CANFD_DATA_FIELD_SIZES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)



def calc_action_chunk_cadence(chunk_horizon_steps: int, control_loop_hz, brain_inference_hz):
    """
    Evaluate the freshness and execution headroom of an action chunking policy.

    Source: action chunking, predicting a sequence of actions per policy query,
    is from Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
    Low-Cost Hardware" (ACT), arXiv:2304.13705. The chunk-duration and
    headroom arithmetic (chunk steps over control rate, divided by the
    inference period) is a modeling assumption of this package, not taken
    from that paper.

    Parameters
    ----------
    chunk_horizon_steps : int
        Number of consecutive actions predicted in a single model inference chunk.
    control_loop_hz : Quantity
        Execution frequency of the low-level nervous system / motor loop (e.g. Hz).
    brain_inference_hz : Quantity
        Inference rate of the high-level neural policy (e.g. Hz).

    Returns
    -------
    dict
        Execution window duration, inference budget, and headroom factor.
    """
    validate_at_least(chunk_horizon_steps, 1, "chunk_horizon_steps")
    validate_positive(control_loop_hz, "control_loop_hz")
    validate_positive(brain_inference_hz, "brain_inference_hz")

    step_period = (1.0 / control_loop_hz).to(ureg.second)
    chunk_duration = chunk_horizon_steps * step_period
    brain_period = (1.0 / brain_inference_hz).to(ureg.second)

    headroom_factor = (chunk_duration / brain_period).to_base_units().magnitude

    return {
        "chunk_duration": chunk_duration.to(ureg.millisecond),
        "brain_period": brain_period.to(ureg.millisecond),
        "headroom_factor": headroom_factor,
    }


def calc_action_chunk_streaming_amortization(
    model_weight_memory,
    memory_bandwidth,
    chunk_horizon: int,
    bus_efficiency: float = 0.70,
):
    """
    Calculate a weight-stream lower bound and its amortized cost per waypoint.

    Under memory-bandwidth bound autoregressive weight streaming:
        t_stream = M_weights / (BW_peak * eta_bus)
        f_single = 1 / t_stream
        tau_step = t_stream / H
        f_waypoint_equivalent = 1 / tau_step = H * f_single

    The waypoint-equivalent rate is an accounting ratio, not the rate at which
    the model observes the world or generates a fresh action chunk. Decoder
    work and other inference costs are excluded. ``f_effective`` is retained
    as a legacy alias for callers of this function.

    Source: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware",
    arXiv:2304.13705; Volume IV Chapter 3.

    Parameters
    ----------
    model_weight_memory : Quantity
        Model parameter storage footprint (e.g. GB).
    memory_bandwidth : Quantity
        Peak memory bus bandwidth (e.g. GB/s).
    chunk_horizon : int
        Number of steps predicted per chunk H (>= 1).
    bus_efficiency : float, optional
        Sustained DRAM efficiency factor eta (default 0.70).

    Returns
    -------
    dict
        t_stream, f_single, tau_step, f_waypoint_equivalent,
        f_effective (legacy alias), b_sustained.
    """
    validate_at_least(chunk_horizon, 1, "chunk_horizon")
    validate_positive(model_weight_memory, "model_weight_memory")
    validate_positive(memory_bandwidth, "memory_bandwidth")

    b_sustained = memory_bandwidth * bus_efficiency
    t_stream = (model_weight_memory / b_sustained).to(ureg.second)
    f_single = (1.0 / t_stream).to(ureg.hertz)

    tau_step = (t_stream / chunk_horizon).to(ureg.second)
    f_waypoint_equivalent = (1.0 / tau_step).to(ureg.hertz)

    return {
        "t_stream": t_stream.to(ureg.millisecond),
        "f_single": f_single,
        "tau_step": tau_step.to(ureg.millisecond),
        "f_waypoint_equivalent": f_waypoint_equivalent,
        "f_effective": f_waypoint_equivalent,
        "b_sustained": b_sustained.to(ureg.GB / ureg.second),
    }


def calc_chunk_renewal_schedule(
    chunk_rate,
    chunk_step,
    chunk_horizon: int,
    inference_latency,
):
    """
    Timing of a chunk policy that renews on a whole number of setpoint steps.

    A chunk policy begins a fresh evaluation every renewal period T_r = 1/f_r
    and emits H setpoints spaced Delta t apart. With the renewal on the setpoint
    grid (T_r = n Delta t for a whole number n), every renewal instant falls on
    a setpoint boundary, so a new chunk replaces its predecessor at a defined
    setpoint index and temporal ensembling can blend chunks step by step:

        n          = T_r / Delta t                  (setpoints per renewal, integer)
        T_open     = T_r + t_infer                  (worst evidence age at replacement)
        h_min      = ceil(T_open / Delta t)         (setpoints that bridge one renewal)
        T_span     = H * Delta t                    (motion one chunk supplies)
        n_overlap  = floor(H / n)                   (chunks that cover one step)

    T_open is the worst sampling-phase wait plus inference, before transport
    and admission delays. n_overlap is the number of predictions a temporal
    ensemble blends at a step in steady state.

    Source: action chunking and temporal ensembling from Zhao et al., "Learning
    Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT), RSS 2023,
    arXiv:2304.13705, which queries its policy every step (n = 1). The
    renewal-on-the-grid requirement, the worst-case age, and the overlap count
    are arithmetic of this package for n > 1 (Volume IV, @sec-brain-action-chunking).

    Parameters
    ----------
    chunk_rate : Quantity
        Renewal rate f_r of the chunk policy (Hz).
    chunk_step : Quantity
        Setpoint spacing Delta t within a chunk (s).
    chunk_horizon : int
        Setpoints per chunk H (>= 1).
    inference_latency : Quantity
        Time from capture to an available chunk t_infer (s), for example a P99 target.

    Returns
    -------
    dict
        renewal_period, setpoints_per_renewal (int), open_loop_span,
        min_supply_setpoints (int), chunk_span, overlapping_chunks (int),
        renewal_slack (T_r - t_infer).

    Raises
    ------
    ValueError
        If the renewal period is not a whole number of setpoint steps, if the
        inference does not finish within one renewal period, or if one chunk
        cannot bridge a renewal (H < h_min).
    """
    validate_positive(chunk_rate, "chunk_rate")
    validate_positive(chunk_step, "chunk_step")
    validate_at_least(chunk_horizon, 1, "chunk_horizon")
    validate_positive(inference_latency, "inference_latency")

    period = (1.0 / chunk_rate).to(ureg.millisecond)
    step = chunk_step.to(ureg.millisecond)
    ratio = (period / step).to("dimensionless").magnitude
    n_per_renewal = round(ratio)
    if n_per_renewal < 1 or abs(ratio - n_per_renewal) > 1e-9:
        raise ValueError(
            f"renewal period {period:~P} is not a whole number of {step:~P} setpoint steps "
            f"(ratio {ratio:.4g})"
        )
    t_infer = inference_latency.to(ureg.millisecond)
    if t_infer >= period:
        raise ValueError(
            f"inference latency {t_infer:~P} does not finish within the {period:~P} renewal period"
        )

    open_loop = (period + t_infer).to(ureg.millisecond)
    h_min = math.ceil((open_loop / step).to("dimensionless").magnitude - 1e-12)
    if chunk_horizon < h_min:
        raise ValueError(
            f"chunk horizon {chunk_horizon} is shorter than the {h_min} setpoints one renewal needs"
        )
    span = (chunk_horizon * step).to(ureg.millisecond)

    return {
        "renewal_period": period,
        "setpoints_per_renewal": int(n_per_renewal),
        "open_loop_span": open_loop,
        "min_supply_setpoints": int(h_min),
        "chunk_span": span,
        "overlapping_chunks": int(chunk_horizon // n_per_renewal),
        "renewal_slack": (period - t_infer).to(ureg.millisecond),
    }


def calc_watchdog_lease_bound(
    clearance_distance,
    initial_velocity,
    max_deceleration,
):
    """
    Calculate the ideal total delay budget before emergency braking begins.

    Under constant velocity v0 during uncommanded drift, followed by emergency braking
    at constant deceleration a_max:
        d_brake = (v0^2) / (2 * a_max)
        d_drift = d_clear - d_brake
        T_lease + T_detection + T_brake_onset <= d_drift / v0

    The returned ceiling includes every pre-brake delay. A chosen watchdog
    lease must be shorter after reserving detection, command delivery, and
    brake-onset time. ``t_lease_max`` is retained as a legacy alias for the
    ideal total ceiling, not a safe standalone lease setting.

    Parameters
    ----------
    clearance_distance : Quantity
        Defended physical clearance / sensory safety field ahead of obstacle (e.g. m).
    initial_velocity : Quantity
        Cruise velocity of the vehicle / mobile base v0 (e.g. m/s).
    max_deceleration : Quantity
        Certified worst-case emergency braking deceleration a_max (e.g. m/s^2).

    Returns
    -------
    dict
        t_total_delay_max, t_lease_max (legacy alias), d_brake,
        d_drift_allowable.
    """
    validate_positive(clearance_distance, "clearance_distance")
    validate_positive(initial_velocity, "initial_velocity")
    validate_positive(max_deceleration, "max_deceleration")

    v0 = initial_velocity.to(ureg.meter / ureg.second)
    a = max_deceleration.to(ureg.meter / (ureg.second**2))
    d_clear = clearance_distance.to(ureg.meter)

    d_brake = ((v0**2) / (2.0 * a)).to(ureg.meter)
    if d_clear < d_brake:
        raise ValueError(
            f"Clearance {d_clear} is insufficient to stop even with zero lease timeout "
            f"(braking distance requires {d_brake})."
        )

    d_drift = d_clear - d_brake
    t_total_delay = (d_drift / v0).to(ureg.millisecond)

    return {
        "t_total_delay_max": t_total_delay,
        "t_lease_max": t_total_delay,
        "d_brake": d_brake.to(ureg.meter),
        "d_drift_allowable": d_drift.to(ureg.meter),
    }


def calc_watchdog_feed_window(period, completion_min, completion_max, phase_jitter):
    """
    Calculate the feed-interval range a windowed watchdog must accept from a healthy tick.

    If a controller feeds once after each successful tick, the watchdog measures
    the interval between feeds. With the completion offset from tick start in
    [c_min, c_max] and the tick-start phase jitter bounded by J_phase,

    Equation:
        spread = (c_max - c_min) + J_phase
        T_feed in [T_period - spread, T_period + spread]

    The early (T_lower) and late (T_upper) trip points are configured around this
    range; the late-trip response must still fit a separately derived fault
    budget.

    Source: interval between successive completions of a periodic task whose
    completion offsets are bounded (difference of two bounded offsets plus the
    release jitter); derived here.

    Returns
    -------
    dict
        spread, t_lower, t_upper (all in microseconds).
    """
    validate_positive(period, "period")
    validate_nonnegative(completion_min, "completion_min")
    validate_nonnegative(phase_jitter, "phase_jitter")
    if completion_max < completion_min:
        raise ValueError("completion_max must be >= completion_min.")

    spread = (completion_max - completion_min + phase_jitter).to(ureg.microsecond)
    t_lower = (period - spread).to(ureg.microsecond)
    if t_lower.magnitude <= 0:
        raise ValueError(f"Feed spread {spread} consumes the whole period {period}.")
    return {
        "spread": spread,
        "t_lower": t_lower,
        "t_upper": (period + spread).to(ureg.microsecond),
    }


def calc_phase_lag_total(
    crossover_frequency,
    sample_period,
    compute_delay=None,
    transport_delay=None,
    phase_allowance=None,
):
    """
    Total loop delay and the phase it removes at crossover (sample-and-hold,
    compute, and transport terms together).

    A zero-order hold behaves in the baseband like a pure delay of half a
    sample period. A controller that outputs its command some time after it
    samples adds that compute delay, and a command that reaches the actuator
    one bus cycle later adds the transport delay. A pure delay tau removes
    phase omega * tau at every frequency, so near crossover:

        tau_total   = T_s / 2 + tau_comp + tau_bus
        dphi_total  = 2 * pi * f_c * tau_total              (radians)

    Given a phase allowance dphi_allow, the largest total delay the loop can
    absorb, and the highest crossover a given total delay supports, are

        tau_allow   = dphi_allow / (2 * pi * f_c)
        f_c,max     = dphi_allow / (2 * pi * tau_total)

    Source: Franklin, G. F., Powell, J. D. and Workman, M. L., Digital Control
    of Dynamic Systems, 3rd ed., Addison-Wesley, 1998 (zero-order hold as a
    T/2 delay; phase of a pure delay). The half-sample approximation holds
    only well below the sampling frequency; it is a budget, not a stability
    proof.

    Parameters
    ----------
    crossover_frequency : Quantity
        Loop crossover frequency f_c (Hz).
    sample_period : Quantity
        Controller sample period T_s.
    compute_delay : Quantity, optional
        Sample-to-output compute delay tau_comp (default 0).
    transport_delay : Quantity, optional
        Command transport delay to the actuator tau_bus (default 0).
    phase_allowance : Quantity, optional
        Phase erosion the design allows at crossover (angle). When given, the
        allowed delay, the delay margin, and the highest supported crossover
        are also returned.

    Returns
    -------
    dict
        tau_zoh, tau_total (ms); phase_zoh, phase_compute, phase_transport,
        phase_total (degrees); and, with phase_allowance, tau_allowed,
        tau_margin (negative when the budget fails), f_crossover_max,
        within_allowance.
    """
    validate_positive(crossover_frequency, "crossover_frequency")
    validate_positive(sample_period, "sample_period")

    f_c = crossover_frequency.to(ureg.hertz)
    t_s = sample_period.to(ureg.millisecond)
    t_comp = (0.0 * ureg.millisecond) if compute_delay is None else compute_delay.to(ureg.millisecond)
    t_bus = (0.0 * ureg.millisecond) if transport_delay is None else transport_delay.to(ureg.millisecond)
    validate_nonnegative(t_comp, "compute_delay")
    validate_nonnegative(t_bus, "transport_delay")

    omega = 2.0 * math.pi * f_c.magnitude  # rad/s

    def _deg(tau):
        return (omega * tau.to(ureg.second).magnitude * ureg.radian).to(ureg.degree)

    tau_zoh = (t_s / 2.0).to(ureg.millisecond)
    tau_total = (tau_zoh + t_comp + t_bus).to(ureg.millisecond)

    result = {
        "tau_zoh": tau_zoh,
        "tau_total": tau_total,
        "phase_zoh": _deg(tau_zoh),
        "phase_compute": _deg(t_comp),
        "phase_transport": _deg(t_bus),
        "phase_total": _deg(tau_total),
    }

    if phase_allowance is not None:
        validate_positive(phase_allowance, "phase_allowance")
        allow_rad = phase_allowance.to(ureg.radian).magnitude
        tau_allowed = (allow_rad / omega * ureg.second).to(ureg.millisecond)
        f_max = (allow_rad / (2.0 * math.pi * tau_total.to(ureg.second).magnitude)) * ureg.hertz
        result.update(
            {
                "tau_allowed": tau_allowed,
                "tau_margin": (tau_allowed - tau_total).to(ureg.millisecond),
                "f_crossover_max": f_max,
                "within_allowance": bool(tau_total <= tau_allowed),
            }
        )

    return result


def calc_canfd_frame_bits(
    payload_bytes: int = 8,
    extended_id: bool = False,
    worst_case_stuffing: bool = True,
):
    """
    Bit count of one CAN FD data frame, split by the bit rate each bit uses.

    Frame layout (ISO 11898-1:2015 FD base format; extended format adds the
    SRR bit and the 18-bit identifier extension):

        nominal rate, before the switch:  SOF 1, ID 11, RRS 1, IDE 1, FDF 1,
                                          res 1, BRS 1          (17 bits; 36 extended)
        data rate, after BRS:             ESI 1, DLC 4, data 8n,
                                          stuff count 4, CRC 17 (n <= 16) or 21,
                                          fixed stuff bits 6 (CRC-17) or 7 (CRC-21)
        nominal rate, after the switch:   CRC delimiter 1, ACK slot 1,
                                          ACK delimiter 1, EOF 7, intermission 3
                                                                (13 bits)

    Dynamic stuff bits (one after five equal bits) apply from SOF through the
    end of the data field. The worst case over a stuffed run of L bits is
    floor((L - 1) / 4); the nominal-rate share is floor((L_nom - 1) / 4) and
    the rest fall in the data phase. The CRC field uses the fixed stuff bits
    above instead. BRS and the CRC delimiter straddle the rate switch; both
    are counted whole at the nominal rate, which slightly overstates the time.

    Source: ISO 11898-1:2015, Road vehicles -- Controller area network (CAN)
    -- Part 1: Data link layer and physical signalling (FD frame format, stuff
    count, fixed stuff bits); Robert Bosch GmbH, CAN with Flexible Data-Rate,
    Specification Version 1.0, 2012 (dual bit rate, CRC-17/21, fixed CRC stuff
    bits).

    Parameters
    ----------
    payload_bytes : int
        Application bytes to carry (0 to 64). Padded up to the next data-field
        size a DLC can encode.
    extended_id : bool
        29-bit identifier (extended format) instead of 11-bit.
    worst_case_stuffing : bool
        Include the worst-case dynamic stuff bits (the bound a response-time
        analysis needs). False gives the unstuffed length.

    Returns
    -------
    dict
        data_field_bytes, crc_bits, fixed_stuff_bits, stuff_bits_nominal,
        stuff_bits_data, bits_nominal, bits_data (all int).
    """
    validate_at_least(payload_bytes, 0, "payload_bytes")
    if payload_bytes > 64:
        raise ValueError(f"CAN FD carries at most 64 data bytes per frame, got {payload_bytes}.")

    data_field = next(s for s in CANFD_DATA_FIELD_SIZES if s >= payload_bytes)
    crc_bits = 17 if data_field <= 16 else 21
    fixed_stuff = 6 if crc_bits == 17 else 7

    arb_bits = 36 if extended_id else 17
    ctrl_data_bits = 1 + 4 + 8 * data_field  # ESI, DLC, data field
    trailer_nominal = 1 + 1 + 1 + 7 + 3  # CRC delimiter, ACK, ACK delimiter, EOF, intermission

    if worst_case_stuffing:
        stuffed_run = arb_bits + ctrl_data_bits
        stuff_total = (stuffed_run - 1) // 4
        stuff_nominal = (arb_bits - 1) // 4
        stuff_data = stuff_total - stuff_nominal
    else:
        stuff_nominal = 0
        stuff_data = 0

    bits_nominal = arb_bits + trailer_nominal + stuff_nominal
    bits_data = ctrl_data_bits + 4 + crc_bits + fixed_stuff + stuff_data

    return {
        "data_field_bytes": data_field,
        "crc_bits": crc_bits,
        "fixed_stuff_bits": fixed_stuff,
        "stuff_bits_nominal": stuff_nominal,
        "stuff_bits_data": stuff_data,
        "bits_nominal": bits_nominal,
        "bits_data": bits_data,
    }


def calc_canfd_bus_utilization(
    num_axes: int,
    payload_bytes: int = 8,
    baud_arbitration=None,
    baud_data=None,
    cycle_time=None,
    frames_per_axis: int = 2,
    extended_id: bool = False,
    worst_case_stuffing: bool = True,
):
    """
    Serialization time and bus occupancy of a cyclic CAN FD axis exchange.

    Each axis exchanges ``frames_per_axis`` frames per cycle (default 2: one
    command frame from the controller and one feedback frame from the drive;
    CAN carries them separately, unlike a summation-frame bus such as
    EtherCAT). Frame bits come from ``calc_canfd_frame_bits``:

        t_frame   = bits_nominal / R_nom + bits_data / R_data
        n_frames  = num_axes * frames_per_axis
        t_total   = n_frames * t_frame
        rho       = t_total / cycle_time

    rho <= 1 is only a necessary condition: non-preemptive blocking, competing
    frames, and error retransmissions still need a response-time analysis.

    Source: frame format per ISO 11898-1:2015 and Bosch CAN FD Specification
    1.0 (2012), via ``calc_canfd_frame_bits``.

    Parameters
    ----------
    num_axes : int
        Axes served on this bus (>= 1). Model one bus per call; parallel buses
        (for example one per limb) are separate calls.
    payload_bytes : int, optional
        Application bytes per frame (default 8: a position-mode setpoint or
        feedback record). Padded to a valid CAN FD data-field size.
    baud_arbitration : Quantity, optional
        Nominal (arbitration-phase) bit rate (default 1.0 Mbps).
    baud_data : Quantity, optional
        Data-phase bit rate (default 5.0 Mbps).
    cycle_time : Quantity, optional
        Cycle the exchange must fit (default 1.0 ms).
    frames_per_axis : int, optional
        Frames per axis per cycle (default 2, command plus feedback).
    extended_id : bool, optional
        29-bit identifiers (default False).
    worst_case_stuffing : bool, optional
        Charge worst-case dynamic stuff bits (default True).

    Returns
    -------
    dict
        t_tx (per frame), t_total_tx, n_frames, utilization, is_schedulable,
        plus the per-frame bit breakdown of ``calc_canfd_frame_bits``.
    """
    validate_at_least(num_axes, 1, "num_axes")
    validate_at_least(frames_per_axis, 1, "frames_per_axis")

    baud_arb = (1.0 * (ureg.megabit / ureg.second)) if baud_arbitration is None else baud_arbitration
    baud_dat = (5.0 * (ureg.megabit / ureg.second)) if baud_data is None else baud_data
    t_cycle = (1.0 * ureg.millisecond) if cycle_time is None else cycle_time

    frame = calc_canfd_frame_bits(
        payload_bytes=payload_bytes,
        extended_id=extended_id,
        worst_case_stuffing=worst_case_stuffing,
    )

    t_nom = ((frame["bits_nominal"] * ureg.bit) / baud_arb).to(ureg.microsecond)
    t_dat = ((frame["bits_data"] * ureg.bit) / baud_dat).to(ureg.microsecond)
    t_tx = (t_nom + t_dat).to(ureg.microsecond)
    n_frames = num_axes * frames_per_axis
    t_total_tx = (n_frames * t_tx).to(ureg.microsecond)

    rho = (t_total_tx / t_cycle).to(ureg.dimensionless).magnitude

    return {
        "t_tx": t_tx,
        "t_nominal": t_nom,
        "t_data": t_dat,
        "t_total_tx": t_total_tx,
        "n_frames": n_frames,
        "utilization": float(rho),
        "is_schedulable": bool(rho <= 1.0),
        **frame,
    }


def calc_ethercat_cycle_time(
    num_axes: int,
    bytes_per_axis: int = 32,
    baud_rate=None,
    header_bytes: int = 64,
    hop_delay=None,
    cycle_time=None,
):
    """
    Calculate on-the-fly hardware processing latency and bus utilization for EtherCAT daisy chain.

    In an EtherCAT daisy-chain ring, a single standard Ethernet frame passes through all
    slave ESCs (EtherCAT SubDevice Controllers). Each node processes and injects PDO data
    on-the-fly with hardware forwarding latency hop_delay (typically 0.6 us).

    Equation:
        total_frame_bytes = header_bytes + (num_axes * bytes_per_axis)
        t_tx = (total_frame_bytes * 8) / baud_rate
        t_forward = num_axes * hop_delay
        t_cycle_total = t_tx + t_forward
        rho = t_cycle_total / cycle_time
        t_margin = cycle_time - t_cycle_total

    Parameters
    ----------
    num_axes : int
        Number of slave joint nodes in the ring (>= 1).
    bytes_per_axis : int, optional
        Process data object (PDO) payload bytes per axis (default 32 bytes).
    baud_rate : Quantity, optional
        Fast Ethernet line rate (default 100.0 Mbps).
    header_bytes : int, optional
        Standard Ethernet preamble, SFD, MAC, EtherCAT header, FCS bytes (default 64 bytes).
    hop_delay : Quantity, optional
        Per-node ASIC hardware processing forwarding delay (default 0.6 us).
    cycle_time : Quantity, optional
        Total control cycle loop deadline (default 1.0 ms).

    Returns
    -------
    dict
        t_tx, t_forward, t_cycle_total, t_margin, utilization, is_schedulable.
    """
    validate_at_least(num_axes, 1, "num_axes")
    validate_at_least(bytes_per_axis, 1, "bytes_per_axis")

    baud = (100.0 * (ureg.megabit / ureg.second)) if baud_rate is None else baud_rate
    t_hop = (0.6 * ureg.microsecond) if hop_delay is None else hop_delay
    t_loop = (1.0 * ureg.millisecond) if cycle_time is None else cycle_time

    total_bytes = header_bytes + (num_axes * bytes_per_axis)
    bits = (total_bytes * 8) * ureg.bit

    t_tx = (bits / baud).to(ureg.microsecond)
    t_forward = (num_axes * t_hop).to(ureg.microsecond)
    t_total = t_tx + t_forward
    t_margin = (t_loop.to(ureg.microsecond) - t_total).to(ureg.microsecond)

    rho = (t_total / t_loop).to(ureg.dimensionless).magnitude

    return {
        "t_tx": t_tx.to(ureg.microsecond),
        "t_forward": t_forward.to(ureg.microsecond),
        "t_cycle_total": t_total.to(ureg.microsecond),
        "t_margin": t_margin.to(ureg.microsecond),
        "utilization": float(rho),
        "is_schedulable": bool(rho <= 1.0),
    }


def calc_cbf_safety_margin(h_val, l_f_h, l_g_h, u_cmd, alpha_coeff: float = 1.0):
    """
    Evaluate the forward-invariance safety margin of a Control Barrier Function (CBF).

    Equation:
        psi = L_f h(x) + L_g h(x) * u + alpha * h(x)

    When psi >= 0, the candidate control command u guarantees that the system state
    remains forward-invariant within the safe set C = {x : h(x) >= 0}.

    Source: Ames et al., "Control Barrier Functions: Theory and Applications",
    IEEE ECC, 2019.

    Parameters
    ----------
    h_val : float or Quantity
        Current barrier function evaluation h(x) (dimensionless or margin).
    l_f_h : float or Quantity
        Lie derivative along uncontrolled drift vector field L_f h(x).
    l_g_h : float or Quantity
        Lie derivative along control input vector field L_g h(x).
    u_cmd : float or Quantity
        Candidate actuation setpoint command proposed by the neural policy.
    alpha_coeff : float, optional
        Class-K gain coefficient alpha (default 1.0).

    Returns
    -------
    dict
        Safety margin psi, forward-invariant boolean status, and barrier value.
    """
    margin = l_f_h + (l_g_h * u_cmd) + (alpha_coeff * h_val)
    is_safe = margin.magnitude >= 0 if hasattr(margin, "magnitude") else margin >= 0
    return {
        "psi": margin,
        "is_safe": bool(is_safe),
        "h_x": h_val,
    }


def calc_cbf_qp_orthogonal_projection(u_nom, a_vec, b_scalar):
    """
    Project onto a supplied abstract half-space in Euclidean control coordinates.

    Solves the quadratic program:
        min  1/2 ||u - u_nom||^2
        s.t. a^T u <= b

    The caller must derive ``a`` and ``b`` from its plant and barrier. This
    arithmetic alone proves neither forward invariance nor a physical stop.
    """
    u_n = np.asarray(u_nom, dtype=float)
    a = np.asarray(a_vec, dtype=float)
    b = float(b_scalar)

    breach = float(np.dot(a, u_n) - b)
    if breach <= 0.0:
        return {
            "u_projected": u_n.tolist(),
            "correction_norm": 0.0,
            "lambda_star": 0.0,
            "was_modified": False,
        }

    norm_sq = float(np.dot(a, a))
    lam = breach / norm_sq
    u_star = u_n - lam * a
    correction = float(np.linalg.norm(u_star - u_n))

    return {
        "u_projected": u_star.tolist(),
        "correction_norm": correction,
        "lambda_star": lam,
        "was_modified": True,
    }


def calc_crypto_auth_deadline(
    clearance_distance,
    initial_velocity,
    emergency_deceleration,
    mechanical_brake_lag=None,
    bus_arbitration_delay=None,
):
    """
    Calculate maximum allowable cryptographic verification latency before kinematic stopping envelope breaches clearance.

    Equation:
        d_brake = v0^2 / (2 * a)
        T_auth_max = (d_clear - d_brake) / v0 - t_mech - t_bus
    """
    validate_positive(clearance_distance, "clearance_distance")
    validate_positive(initial_velocity, "initial_velocity")
    validate_positive(emergency_deceleration, "emergency_deceleration")

    t_mech = (0.0 * ureg.second) if mechanical_brake_lag is None else mechanical_brake_lag.to(ureg.second)
    t_bus = (0.0 * ureg.second) if bus_arbitration_delay is None else bus_arbitration_delay.to(ureg.second)

    v0 = initial_velocity.to(ureg.meter / ureg.second)
    a = emergency_deceleration.to(ureg.meter / (ureg.second**2))
    d_clear = clearance_distance.to(ureg.meter)

    d_brake = ((v0**2) / (2.0 * a)).to(ureg.meter)
    if d_clear < d_brake:
        raise ValueError(f"Clearance {d_clear} is insufficient even with zero auth latency (braking requires {d_brake}).")

    t_avail = (d_clear - d_brake) / v0
    t_deadline = t_avail - t_mech - t_bus
    return t_deadline.to(ureg.millisecond)


def calc_process_containment_time_to_breach(
    capacity_limit,
    current_volume,
    inflow_rate,
    outflow_rate=None,
):
    """
    Calculate time until liquid container or buffer capacity is breached under flow surge.

    Equation:
        t_breach = (V_max - V_0) / (Q_in - Q_out)
    """
    validate_positive(capacity_limit, "capacity_limit")
    validate_nonnegative(current_volume, "current_volume")
    validate_positive(inflow_rate, "inflow_rate")

    q_out = (0.0 * (ureg.liter / ureg.second)) if outflow_rate is None else outflow_rate.to(ureg.liter / ureg.second)
    v_max = capacity_limit.to(ureg.liter)
    v_0 = current_volume.to(ureg.liter)
    q_in = inflow_rate.to(ureg.liter / ureg.second)

    q_net = q_in - q_out
    if q_net.magnitude <= 0:
        return float("inf") * ureg.second

    headroom = v_max - v_0
    if headroom.magnitude <= 0:
        return 0.0 * ureg.second

    t_breach = (headroom / q_net).to(ureg.second)
    return t_breach


def calc_process_thermal_runaway_lease(
    heat_generation_rate,
    thermal_capacitance,
    max_temp_overshoot,
):
    """
    Calculate lumped thermal rise rate and an assumed overshoot horizon.

    This heat-capacity model does not predict thermal runaway or venting;
    those require chemistry, initial temperature, cooling, and validated thresholds.

    Equation:
        dT_dt = P_in / C_th
        tau = Delta_T_max / dT_dt

    Parameters
    ----------
    heat_generation_rate : Quantity
        Internal Joule/dissipative heat input rate P_in (e.g. W).
    thermal_capacitance : Quantity
        Lumped system thermal capacitance C_th (e.g. J/K).
    max_temp_overshoot : Quantity
        Permissible temperature rise Delta_T_max (e.g. K).

    Returns
    -------
    dict
        rate_of_rise and tau_lease, the latter an overshoot horizon only.
    """
    validate_positive(heat_generation_rate, "heat_generation_rate")
    validate_positive(thermal_capacitance, "thermal_capacitance")
    validate_positive(max_temp_overshoot, "max_temp_overshoot")

    p = heat_generation_rate.to(ureg.watt)
    c_th = thermal_capacitance.to(ureg.joule / ureg.kelvin)
    delta_t = max_temp_overshoot.to(ureg.kelvin)

    dt_dt = p / c_th
    tau = delta_t / dt_dt

    return {
        "rate_of_rise": dt_dt.to(ureg.kelvin / ureg.second),
        "tau_lease": tau.to(ureg.millisecond),
    }


def calc_axi_fifo_contention(
    *,
    tile_bytes: int,
    camera_bytes: int,
    telemetry_bytes: int,
    safety_read_bytes: int,
    transaction_bytes: int,
    bandwidth,
    bank_switches: int,
    bank_switch_time,
    compute_time,
):
    """Compare a constructed FIFO backlog with priority bypass at AXI granularity.

    All transfers are aligned multiples of ``transaction_bytes``. The FIFO
    scenario admits the entire tile, camera transfer, and telemetry transfer
    ahead of the safety read, then allows no new arrivals. The priority-bypass
    scenario waits only for one already in-flight transaction. These are
    controller-policy assumptions, not AXI timing guarantees. Peak bandwidth
    ignores arbitration, refresh, and protocol overhead, so times are floors.
    """
    for name, count in (
        ("tile_bytes", tile_bytes),
        ("camera_bytes", camera_bytes),
        ("telemetry_bytes", telemetry_bytes),
        ("safety_read_bytes", safety_read_bytes),
        ("transaction_bytes", transaction_bytes),
    ):
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if any(count % transaction_bytes for count in (
        tile_bytes, camera_bytes, telemetry_bytes, safety_read_bytes
    )):
        raise ValueError("all transfers must align to transaction_bytes")
    if not isinstance(bank_switches, int) or bank_switches < 0:
        raise ValueError("bank_switches must be a nonnegative integer")
    validate_positive(bandwidth, "bandwidth")
    validate_nonnegative(bank_switch_time, "bank_switch_time")
    validate_nonnegative(compute_time, "compute_time")

    byte_rate = bandwidth.to(ureg.byte / ureg.second)
    switch_time = bank_switch_time.to(ureg.second)
    compute = compute_time.to(ureg.second)
    fifo_bytes = tile_bytes + camera_bytes + telemetry_bytes
    queue_fifo = fifo_bytes * ureg.byte / byte_rate + bank_switches * switch_time
    queue_bypass = transaction_bytes * ureg.byte / byte_rate
    own_read = safety_read_bytes * ureg.byte / byte_rate
    return {
        "queued_transactions": fifo_bytes // transaction_bytes,
        "tile_transactions": tile_bytes // transaction_bytes,
        "queue_fifo": queue_fifo.to(ureg.microsecond),
        "queue_priority_bypass": queue_bypass.to(ureg.microsecond),
        "own_read": own_read.to(ureg.microsecond),
        "total_fifo": (compute + queue_fifo + own_read).to(ureg.microsecond),
        "total_priority_bypass": (compute + queue_bypass + own_read).to(ureg.microsecond),
    }
