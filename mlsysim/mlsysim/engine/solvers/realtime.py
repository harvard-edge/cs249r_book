"""Real-time solvers: the timed path between a learned proposer and the drives.

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). The equations live in
``physics.robotics.control`` (phase lag of a delay, response-time bound,
CAN FD and EtherCAT serialization, the lease ceiling) and
``physics.robotics.stopping``; these models compose them for one stated
situation (added 2026-10-20).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from ..results import (
    CanFdBusResult,
    EtherCatCycleResult,
    LatencyTailResult,
    LeaseBudgetResult,
    PhaseLagBudgetResult,
    ResponseTimeResult,
    SharedDramReadResult,
    ThermalThrottleResult,
)
from ...core.types import Quantity
from ...core.units import ureg
from ...physics.motors import calc_first_order_time_to_temperature
from ...physics.robotics.control import (
    calc_axi_fifo_contention,
    calc_canfd_bus_utilization,
    calc_ethercat_cycle_time,
    calc_phase_lag_total,
    calc_response_time_bound,
)
from .base import ForwardModel
from .embodied import StoppingBudgetModel

_MS = ureg.millisecond
_US = ureg.microsecond
_MM = ureg.millimeter
_DEG = ureg.degree


class PhaseLagBudgetModel(ForwardModel):
    """
    Phase a sampled loop spends on delay at its crossover frequency.

    The hold costs half a sample period, then the compute and transport
    delays; a pure delay tau removes ``360 f_c tau`` degrees at crossover
    (``calc_phase_lag_total``). Against a stated phase allowance the model
    also gives the largest total delay the allowance absorbs and the highest
    crossover the delay supports. A budget, not a stability proof.
    """
    requires = ("crossover", "sample_period")
    produces = PhaseLagBudgetResult

    def solve(self, crossover: Quantity, sample_period: Quantity,
              compute_delay: Optional[Quantity] = None, transport_delay: Optional[Quantity] = None,
              phase_allowance: Optional[Quantity] = None) -> PhaseLagBudgetResult:
        r = calc_phase_lag_total(crossover, sample_period, compute_delay=compute_delay,
                                 transport_delay=transport_delay, phase_allowance=phase_allowance)
        total = r["phase_total"].to(_DEG)
        out = dict(
            hold_delay=r["tau_zoh"].to(_MS), total_delay=r["tau_total"].to(_MS),
            phase_hold=r["phase_zoh"].to(_DEG), phase_compute=r["phase_compute"].to(_DEG),
            phase_transport=r["phase_transport"].to(_DEG), phase_total=total,
            phase_per_delay=(360 * crossover * _DEG).to(_DEG / _MS),
        )
        if phase_allowance is not None:
            allow = phase_allowance.to(_DEG)
            out.update(
                allowed_delay=r["tau_allowed"].to(_MS), delay_margin=r["tau_margin"].to(_MS),
                crossover_ceiling=r["f_crossover_max"].to(ureg.hertz),
                over_allowance=float((total / allow).to("").magnitude),
                fits=bool(total <= allow),
            )
        return PhaseLagBudgetResult(**out)


class ResponseTimeModel(ForwardModel):
    """
    Response-time bound of one periodic task, R = C + B + I, against its deadline.

    ``calc_response_time_bound`` sums the declared WCET, the blocking bound,
    and the interference bound; the margin and the largest blocking that still
    meets the deadline follow. With an observed unloaded run, the gap between
    it and the declared WCET is the claim that timing analysis must justify.
    """
    requires = ("wcet", "blocking", "interference", "deadline")
    produces = ResponseTimeResult

    def solve(self, wcet: Quantity, blocking: Quantity, interference: Quantity, deadline: Quantity,
              unloaded: Optional[Quantity] = None) -> ResponseTimeResult:
        r = calc_response_time_bound(wcet, blocking, interference, deadline)
        return ResponseTimeResult(
            wcet=wcet.to(_US), blocking=blocking.to(_US), interference=interference.to(_US),
            deadline=deadline.to(_US), response=r["response"], margin=r["margin"],
            meets_deadline=r["meets_deadline"], blocking_limit=r["blocking_limit"],
            wcet_claim=None if unloaded is None else (wcet - unloaded).to(_US),
            execution_ceiling=(deadline - blocking - interference).to(_US),
        )


class CanFdBusModel(ForwardModel):
    """
    Serialization occupancy of a cyclic CAN FD exchange: ``frames x t_frame`` against the cycle.

    Frame time and bits come from ``calc_canfd_bus_utilization`` (ISO
    11898-1:2015 frame format, worst-case stuffing by default). Occupancy is a
    necessary check only; blocking and retransmission need a response-time
    analysis. ``axes_capacity`` is the most axes whose frames fit one cycle.
    """
    requires = ("num_axes",)
    produces = CanFdBusResult

    def solve(self, num_axes: int, payload_bytes: int, frames_per_axis: int,
              baud_arbitration: Quantity, baud_data: Quantity, cycle: Quantity) -> CanFdBusResult:
        r = calc_canfd_bus_utilization(num_axes=num_axes, payload_bytes=payload_bytes,
                                       baud_arbitration=baud_arbitration, baud_data=baud_data,
                                       cycle_time=cycle, frames_per_axis=frames_per_axis)
        t_frame = r["t_tx"].to(_US)
        capacity = int(math.floor((cycle / (frames_per_axis * t_frame)).to("").magnitude))
        return CanFdBusResult(
            frame_time=t_frame, nominal_time=r["t_nominal"].to(_US), data_time=r["t_data"].to(_US),
            bits_nominal=r["bits_nominal"], bits_data=r["bits_data"],
            stuff_bits=r["stuff_bits_nominal"] + r["stuff_bits_data"], frames=r["n_frames"],
            occupancy=r["t_total_tx"].to(_US), utilization=r["utilization"],
            unallocated=(cycle - r["t_total_tx"]).to(_US), fits=r["is_schedulable"],
            axes_capacity=capacity,
        )


class EtherCatCycleModel(ForwardModel):
    """
    One EtherCAT summation frame through a chain of drives (``calc_ethercat_cycle_time``).

    The frame carries every axis's process data; each node forwards it with a
    fixed hop delay, so occupancy is serialization plus forwarding.
    """
    requires = ("num_axes",)
    produces = EtherCatCycleResult

    def solve(self, num_axes: int, bytes_per_axis: int, baud_rate: Quantity, header_bytes: int,
              hop_delay: Quantity, cycle: Quantity) -> EtherCatCycleResult:
        r = calc_ethercat_cycle_time(num_axes=num_axes, bytes_per_axis=bytes_per_axis,
                                     baud_rate=baud_rate, header_bytes=header_bytes,
                                     hop_delay=hop_delay, cycle_time=cycle)
        return EtherCatCycleResult(
            frame_time=r["t_tx"], forward_time=r["t_forward"], occupancy=r["t_cycle_total"],
            utilization=r["utilization"], unallocated=r["t_margin"], fits=r["is_schedulable"],
        )


class LeaseBudgetModel(ForwardModel):
    """
    A chunk lease as a term of the stopping budget, bounded by the lease rule.

    After the last admission a stalled proposer's chunk drives on for the
    lease; the tick that notices expiry and the bus cycle that carries the
    stop follow, then the brake onset. That interval is travel at ``speed``,
    added to the braking travel, the localization and margin overhead, and the
    onset travel (``StoppingBudgetModel``). The lease rule bounds the lease
    from both sides:

        T_renew + T_jitter  <  T_lease  <=  tau_max - T_tick - T_bus - T_onset - age

    where ``tau_max`` is the longest pre-brake delay whose stop fits and
    ``age`` is the evidence's age when the chunk is admitted. The stall
    timeline puts the stall after the second renewal completes.
    """
    requires = ("speed", "deceleration", "clear_distance", "lease")
    produces = LeaseBudgetResult

    def solve(self, speed: Quantity, deceleration: Quantity, clear_distance: Quantity,
              loc_margin: Quantity, physical_margin: Quantity, brake_onset: Quantity,
              lease: Quantity, tick: Quantity, bus: Quantity,
              observation_age: Quantity, admission_age: Quantity,
              renewal_period: Quantity, renewal_jitter: Quantity,
              inference_latency: Quantity) -> LeaseBudgetResult:
        path = (lease + tick + bus).to(_MS)
        to_onset = (path + brake_onset).to(_MS)
        stop = StoppingBudgetModel()
        body = stop.solve(speed, brake_onset, deceleration, clear_distance,
                          loc_margin=loc_margin, physical_margin=physical_margin)
        stall = stop.solve(speed, to_onset, deceleration, clear_distance,
                           loc_margin=loc_margin, physical_margin=physical_margin)
        ceiling = stall.delay_ceiling.to(_MS)
        lease_ceiling = (ceiling - tick - bus - brake_onset).to(_MS)
        at_admission = (lease_ceiling - admission_age).to(_MS)
        floor = (renewal_period + renewal_jitter).to(_MS)
        last = (renewal_period + inference_latency).to(_MS)
        return LeaseBudgetResult(
            lease=lease.to(_MS), lease_path=path, lease_path_travel=(speed * path).to(_MM),
            admission_to_brake_onset=to_onset,
            admission_to_brake_onset_travel=(speed * to_onset).to(_MM),
            body_terms_distance=body.required_distance.to(_MM),
            running_total=stall.required_distance.to(_MM), spare=stall.spare.to(_MM),
            delay_ceiling=ceiling, delay_spare=stall.delay_spare.to(_MS),
            lease_floor=floor, lease_ceiling=lease_ceiling,
            lease_ceiling_after_age=(lease_ceiling - observation_age).to(_MS),
            lease_ceiling_at_admission=at_admission,
            fits=bool(floor < lease <= at_admission),
            stall_last_admission=last, stall_lease_expiry=(last + lease).to(_MS),
            stall_brake_onset=(last + to_onset).to(_MS),
            renewal_lateness_limit=(lease - renewal_period).to(_MS),
        )


class LatencyTailModel(ForwardModel):
    """
    The tail of a measured latency sample: median, p99, p99.9, and maximum.

    Percentiles use linear interpolation between order statistics (numpy's
    default). A measured maximum is the largest value seen so far, never a
    bound: it can only grow as the sample grows.
    """
    requires = ("samples",)
    produces = LatencyTailResult

    def solve(self, samples: Sequence[float], unit: str = "microsecond",
              threshold: Optional[Quantity] = None) -> LatencyTailResult:
        a = np.asarray(samples, dtype=float)
        u = ureg(unit)
        pct = np.percentile(a, [50.0, 99.0, 99.9])
        over = None
        if threshold is not None:
            over = int((a > threshold.to(u).magnitude).sum())
        return LatencyTailResult(
            count=int(a.size), p50=float(pct[0]) * u, p99=float(pct[1]) * u,
            p999=float(pct[2]) * u, maximum=float(a.max()) * u, over_threshold=over,
        )


class SharedDramReadModel(ForwardModel):
    """
    A permission tick's snapshot read behind FIFO backlogs on a shared DRAM controller.

    One backlog (a weight tile, a camera transfer, and a telemetry record, at
    AXI transaction granularity) waits ``bytes / bandwidth`` plus its row
    conflicts (``calc_axi_fifo_contention``); the tick's snapshot read follows.
    Against the sense allowance, the declared WCET, and the execution ceiling
    (deadline less blocking and interference), each less the compute after
    sensing, the model counts the most whole backlogs that still fit and the
    tick's end time at each count. A FIFO controller is optimistic: a
    row-hit-first controller can keep passing the read.
    """
    requires = ("bandwidth", "sense", "compute_after_sense")
    produces = SharedDramReadResult

    def solve(self, *, tile_bytes: int, camera_bytes: int, telemetry_bytes: int, snapshot_bytes: int,
              transaction_bytes: int, bandwidth: Quantity, bank_switches: int,
              bank_switch_time: Quantity, sense: Quantity, compute_after_sense: Quantity,
              wcet: Quantity, execution_ceiling: Quantity) -> SharedDramReadResult:
        r = calc_axi_fifo_contention(
            tile_bytes=tile_bytes, camera_bytes=camera_bytes, telemetry_bytes=telemetry_bytes,
            safety_read_bytes=snapshot_bytes, transaction_bytes=transaction_bytes,
            bandwidth=bandwidth, bank_switches=bank_switches, bank_switch_time=bank_switch_time,
            compute_time=compute_after_sense)
        wait, own = r["queue_fifo"].to(_US), r["own_read"].to(_US)
        compute = compute_after_sense.to(_US)
        ceil_wcet = (wcet - compute).to(_US)
        ceil_dead = (execution_ceiling - compute).to(_US)

        def count(room):
            return int(math.floor(((room - own) / wait).to("").magnitude))

        def end(n):
            return (own + n * wait + compute).to(_US)

        n_sense, n_wcet, n_dead = count(sense), count(ceil_wcet), count(ceil_dead)
        return SharedDramReadResult(
            queued_transactions=r["queued_transactions"], tile_transactions=r["tile_transactions"],
            backlog_bytes=tile_bytes + camera_bytes + telemetry_bytes,
            backlog_wait=wait, snapshot_read=own, loaded_read=(wait + own).to(_US),
            sense_left=(sense - wait - own).to(_US),
            bypass_read=(r["queue_priority_bypass"] + own).to(_US),
            read_ceiling_wcet=ceil_wcet, read_ceiling_deadline=ceil_dead,
            backlogs_sense=n_sense, backlogs_wcet=n_wcet, backlogs_deadline=n_dead,
            unloaded_end=(sense + compute).to(_US),
            end_sense=end(n_sense), end_wcet=end(n_wcet), end_deadline=end(n_dead),
        )


class ThermalThrottleModel(ForwardModel):
    """
    A tick on a governed clock: a thermal trip halves the clock until the junction clears.

    The tick is sized to its declared WCET at the nominal clock; the throttled
    clock stretches the same cycles. The junction is a first-order lumped
    model, ``T(t) = T_inf - (T_inf - T_0) exp(-t / tau)`` with steady state
    ``T_amb + P theta``; the clock returns only at the clear threshold, so the
    hysteresis lasts ``tau ln((T_trip - T_quiet) / (T_clear - T_quiet))`` after
    the load stops (``calc_first_order_time_to_temperature``).
    """
    requires = ("wcet", "nominal_clock", "throttled_clock", "time_constant")
    produces = ThermalThrottleResult

    def solve(self, *, wcet: Quantity, blocking_and_interference: Quantity, deadline: Quantity,
              tick: Quantity, nominal_clock: Quantity, throttled_clock: Quantity,
              time_constant: Quantity, ambient: Quantity, theta: Quantity, quiet_power: Quantity,
              loaded_power: Quantity, trip: Quantity, clear: Quantity) -> ThermalThrottleResult:
        k = ureg.kelvin
        cycles = float((wcet * nominal_clock).to("").magnitude)
        ex = (cycles / nominal_clock).to(_US)
        ex_thr = (cycles / throttled_clock).to(_US)
        resp = (ex + blocking_and_interference).to(_US)
        resp_thr = (ex_thr + blocking_and_interference).to(_US)
        rise = (loaded_power * theta).to(k)
        quiet = (ambient.to(k) + (quiet_power * theta).to(k))
        loaded = (ambient.to(k) + rise)
        t_trip = calc_first_order_time_to_temperature(quiet, trip, loaded, time_constant)
        rec = calc_first_order_time_to_temperature(trip, clear, quiet, time_constant)
        rec_loaded = calc_first_order_time_to_temperature(loaded, clear, quiet, time_constant)
        overrun = (resp_thr - deadline).to(_US)
        return ThermalThrottleResult(
            cycles=cycles, execution=ex, execution_throttled=ex_thr, response=resp,
            response_throttled=resp_thr, slack=(deadline - resp).to(_US), overrun=overrun,
            overrun_ratio=float((overrun / deadline).to("").magnitude),
            junction_quiet=quiet.to(ureg.degC), junction_loaded=loaded.to(ureg.degC), loaded_rise=rise,
            time_to_trip=t_trip, recovery=rec, recovery_from_loaded=rec_loaded,
            ticks_exposed=int(math.floor((rec / tick).to("").magnitude)),
            ticks_exposed_from_loaded=int(math.floor((rec_loaded / tick).to("").magnitude)),
            heat_capacity=(time_constant / theta).to(ureg.joule / k),
            theta_never_trips=((trip.to(k) - ambient.to(k)) / loaded_power).to(k / ureg.watt),
        )
