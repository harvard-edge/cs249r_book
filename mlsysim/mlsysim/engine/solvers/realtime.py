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
from typing import Optional


from ..results import (
    CanFdBusResult,
    CommandAuthenticationResult,
    EtherCatCycleResult,
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
        """Budget the phase consumed by sampling, computation, and transport.

        Parameters
        ----------
        crossover : Quantity
            Positive loop crossover frequency, convertible to Hz.
        sample_period : Quantity
            Positive controller sampling period, convertible to seconds.
        compute_delay : Quantity, optional
            Nonnegative sample-to-output time; None contributes zero delay.
        transport_delay : Quantity, optional
            Nonnegative output-to-actuator time; None contributes zero delay.
        phase_allowance : Quantity, optional
            Positive phase budget in degrees or radians; None omits budget comparisons.

        Returns
        -------
        PhaseLagBudgetResult
            Delays in ms, phases in degrees, and optional crossover ceiling in Hz. The
            fits flag checks only the stated phase budget.

        See Also
        --------
        mlsysim.physics.robotics.control.calc_phase_lag_total : Delay and phase calculation.

        Notes
        -----
        Equation: tau = sample_period/2 + compute_delay + transport_delay; phase =
        ``360*crossover*tau`` degrees. The allowance gives tau_max =
        ``phase_allowance/(360*crossover)``.

        Assumptions: The zero-order hold is approximated by a half-sample delay well
        below the sampling frequency; compute and transport are pure delays.

        Excludes: Plant poles, controller dynamics, jitter distributions, and a
        closed-loop stability proof.

        References
        ----------
        Derivation: a pure delay has frequency response ``exp(-j*omega*tau)``, hence phase
        loss ``omega*tau``; composing delays adds their phase losses.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import PhaseLagBudgetModel
        >>> result = PhaseLagBudgetModel().solve(Q_(10, "Hz"), Q_(1, "ms"),
        ...     phase_allowance=Q_(30, "degree"))
        >>> bool(result.fits and result.total_delay > Q_(0, "ms"))
        True
        """
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
        """Compare a declared response-time bound with a deadline.

        Parameters
        ----------
        wcet : Quantity
            Nonnegative declared worst-case execution time, convertible to microseconds.
        blocking : Quantity
            Nonnegative declared blocking bound in time units.
        interference : Quantity
            Nonnegative declared higher-priority interference bound in time units.
        deadline : Quantity
            Positive relative deadline in time units.
        unloaded : Quantity, optional
            Observed unloaded execution time; None omits the WCET-minus-observation gap.

        Returns
        -------
        ResponseTimeResult
            Component times and margins in microseconds; meets_deadline is true when the
            supplied sum is no greater than deadline.

        See Also
        --------
        mlsysim.physics.robotics.control.calc_response_time_bound : Add declared timing bounds.

        Notes
        -----
        Equation: R = wcet + blocking + interference; margin = deadline - R;
        blocking_limit = deadline - wcet - interference.

        Assumptions: The caller supplies independently justified bounds that may be
        summed for the task under study.

        Excludes: Derivation of WCET, priority assignment, interference fixed-point
        iteration, and certification from an unloaded observation.

        References
        ----------
        Derivation: partition response time into execution, blocking, and interference;
        rearrange R <= deadline to solve each remaining budget.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import ResponseTimeModel
        >>> result = ResponseTimeModel().solve(Q_(20, "us"), Q_(5, "us"),
        ...     Q_(5, "us"), Q_(40, "us"))
        >>> bool(result.meets_deadline and result.response + result.margin == Q_(40, "us"))
        True
        """
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
        """Count CAN FD serialization occupancy within one control cycle.

        Parameters
        ----------
        num_axes : int
            Positive number of axes sharing this bus.
        payload_bytes : int
            Application bytes per frame, from zero through 64; padded to a legal CAN FD
            data-field size.
        frames_per_axis : int
            Positive number of command/feedback frames per axis per cycle.
        baud_arbitration : Quantity
            Positive nominal-phase bit rate, convertible to bit/s.
        baud_data : Quantity
            Positive data-phase bit rate, convertible to bit/s.
        cycle : Quantity
            Positive cycle duration in time units.

        Returns
        -------
        CanFdBusResult
            Frame bit counts, times in microseconds, utilization fraction, and
            axes_capacity. fits is a serialization-only occupancy check.

        See Also
        --------
        mlsysim.physics.robotics.control.calc_canfd_bus_utilization : CAN FD frame accounting and serialization.

        Notes
        -----
        Equation: t_frame = bits_nominal/baud_arbitration + bits_data/baud_data;
        occupancy = ``num_axes*frames_per_axis*t_frame``; axes_capacity =
        ``floor(cycle/(frames_per_axis*t_frame))``.

        Assumptions: Standard identifiers, worst-case dynamic stuffing, and the
        frame-format accounting of calc_canfd_bus_utilization; all frames have the
        supplied payload.

        Excludes: Competing traffic, nonpreemptive blocking, retransmissions, and a
        schedulability guarantee.

        References
        ----------
        Derivation: serialization time is bit count divided by bit rate, summed across
        the nominal and data phases. Frame-format assumptions are documented in the
        linked physics function.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import CanFdBusModel
        >>> result = CanFdBusModel().solve(1, 8, 2, Q_(1, "Mbit/s"),
        ...     Q_(5, "Mbit/s"), Q_(1, "ms"))
        >>> bool(result.fits and result.axes_capacity >= 1)
        True
        """
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
        """Budget one summation frame through an EtherCAT drive chain.

        Parameters
        ----------
        num_axes : int
            Positive number of nodes traversed by the frame.
        bytes_per_axis : int
            Positive process-data payload bytes contributed by each node.
        baud_rate : Quantity
            Positive link bit rate, convertible to bit/s.
        header_bytes : int
            Nonnegative total framing overhead in bytes, supplied by the caller.
        hop_delay : Quantity
            Nonnegative forwarding time per node.
        cycle : Quantity
            Positive control-cycle duration.

        Returns
        -------
        EtherCatCycleResult
            Serialization, forwarding, occupancy, and margin times; dimensionless
            utilization and fits comparing total occupancy with cycle.

        See Also
        --------
        mlsysim.physics.robotics.control.calc_ethercat_cycle_time : Summation-frame occupancy.

        Notes
        -----
        Equation: occupancy = ``8*(header_bytes`` + ``num_axes*bytes_per_axis)/baud_rate`` +
        ``num_axes*hop_delay``; utilization = occupancy/cycle.

        Assumptions: One frame carries all process data and each node adds a fixed
        forwarding delay.

        Excludes: Cable propagation unless included in hop_delay, multi-frame
        fragmentation, host jitter, and protocol error recovery.

        References
        ----------
        Derivation: transmit all frame bits at the declared line rate, then add the
        per-node forwarding budgets along the chain.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import EtherCatCycleModel
        >>> result = EtherCatCycleModel().solve(2, 32, Q_(100, "Mbit/s"),
        ...     64, Q_(1, "us"), Q_(1, "ms"))
        >>> bool(result.fits and result.unallocated > Q_(0, "us"))
        True
        """
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
        """Bound a command lease using the remaining stopping distance.

        Parameters
        ----------
        speed : Quantity
            Positive travel speed in length/time units, held constant until braking.
        deceleration : Quantity
            Positive available constant braking deceleration.
        clear_distance : Quantity
            Positive distance available before the protected boundary.
        loc_margin : Quantity
            Nonnegative localization-distance allowance.
        physical_margin : Quantity
            Nonnegative protective distance retained after stopping.
        brake_onset : Quantity
            Nonnegative delay from stop command delivery to braking onset.
        lease : Quantity
            Positive command-validity duration since admission.
        tick : Quantity
            Positive worst-case expiry-detection tick delay.
        bus : Quantity
            Nonnegative stop-command bus delay.
        observation_age : Quantity
            Nonnegative age of evidence for the reported age-adjusted ceiling.
        admission_age : Quantity
            Nonnegative evidence age at admission, used by the fits comparison.
        renewal_period : Quantity
            Positive interval between replacement requests.
        renewal_jitter : Quantity
            Nonnegative additional renewal-lateness allowance.
        inference_latency : Quantity
            Nonnegative inference time used to place the constructed stall timeline.

        Returns
        -------
        LeaseBudgetResult
            Timing budgets in ms and travel in mm. fits checks renewal_period +
            renewal_jitter < lease <= the admission-age-adjusted ceiling.

        See Also
        --------
        mlsysim.engine.solvers.embodied.StoppingBudgetModel.solve : Distance and delay ceilings.

        Notes
        -----
        Equation: stop distance = ``speed*(lease`` + tick + bus + brake_onset) +
        ``speed**2/(2*deceleration)`` + loc_margin + physical_margin. Subtract fixed delays
        from the distance-derived delay ceiling to obtain the lease ceiling.

        Assumptions: Straight-line constant speed before constant-deceleration braking;
        the constructed stall occurs after the second renewal completes.

        Excludes: Jerk ramps, changing obstacles, stochastic delays, and proof that a
        deployed watchdog or brake meets these budgets.

        References
        ----------
        Derivation: integrate constant cruise and braking motion, reserve both distance
        margins, then isolate the lease term in the stopping-distance inequality.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import LeaseBudgetModel
        >>> result = LeaseBudgetModel().solve(Q_(1, "m/s"), Q_(2, "m/s^2"),
        ...     Q_(2, "m"), Q_(0, "m"), Q_(0, "m"), Q_(10, "ms"),
        ...     Q_(100, "ms"), Q_(1, "ms"), Q_(1, "ms"), Q_(0, "ms"),
        ...     Q_(0, "ms"), Q_(50, "ms"), Q_(1, "ms"), Q_(10, "ms"))
        >>> bool(result.fits and result.spare > Q_(0, "mm"))
        True
        """
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
        """Budget a snapshot read behind a constructed FIFO memory backlog.

        Parameters
        ----------
        tile_bytes : int
            Positive weight-tile byte count, aligned to transaction_bytes.
        camera_bytes : int
            Positive camera-transfer byte count, aligned to transaction_bytes.
        telemetry_bytes : int
            Positive telemetry-transfer byte count, aligned to transaction_bytes.
        snapshot_bytes : int
            Positive snapshot-read byte count, aligned to transaction_bytes.
        transaction_bytes : int
            Positive transfer granularity in bytes.
        bandwidth : Quantity
            Positive effective transfer rate, convertible to byte/s; a peak rate yields
            a floor.
        bank_switches : int
            Nonnegative number of bank/row-switch penalties in one backlog.
        bank_switch_time : Quantity
            Nonnegative time charged for each bank switch.
        sense : Quantity
            Nonnegative sensing-time allowance, before post-sense computation.
        compute_after_sense : Quantity
            Nonnegative computation time following the snapshot read.
        wcet : Quantity
            Declared execution-time budget including sensing and computation.
        execution_ceiling : Quantity
            Deadline-derived execution budget after blocking and interference are
            removed.

        Returns
        -------
        SharedDramReadResult
            Read and completion times in microseconds, transaction counts, and
            whole-backlog counts. A negative count indicates that even the own-read
            allowance does not fit.

        See Also
        --------
        mlsysim.physics.robotics.control.calc_axi_fifo_contention : FIFO and bypass transfer accounting.

        Notes
        -----
        Equation: wait = (tile_bytes + camera_bytes + telemetry_bytes)/bandwidth +
        ``bank_switches*bank_switch_time``; own = snapshot_bytes/bandwidth; n =
        floor((read_allowance-own)/wait).

        Assumptions: All declared backlog transfers queue before the snapshot; FIFO
        prevents new arrivals from overtaking it. Priority bypass waits for one
        transaction only.

        Excludes: Refresh, arbitration inefficiency, row-hit-first starvation, cache
        behavior, and an AXI service-time guarantee.

        References
        ----------
        Derivation: sum serialized byte-transfer and row-switch times, then solve own +
        ``n*wait`` <= read_allowance for the largest integer n.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import SharedDramReadModel
        >>> result = SharedDramReadModel().solve(tile_bytes=64, camera_bytes=64,
        ...     telemetry_bytes=64, snapshot_bytes=64, transaction_bytes=64,
        ...     bandwidth=Q_(1, "GB/s"), bank_switches=0,
        ...     bank_switch_time=Q_(0, "us"), sense=Q_(10, "us"),
        ...     compute_after_sense=Q_(1, "us"), wcet=Q_(20, "us"),
        ...     execution_ceiling=Q_(30, "us"))
        >>> bool(result.loaded_read > result.snapshot_read)
        True
        """
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
    A tick on a governed clock: a thermal trip reduces the clock until the junction clears.

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
        """Translate a thermal clock reduction into task response and recovery time.

        Parameters
        ----------
        wcet : Quantity
            Positive execution time at nominal_clock, excluding blocking and
            interference.
        blocking_and_interference : Quantity
            Nonnegative combined blocking and interference time.
        deadline : Quantity
            Positive task deadline.
        tick : Quantity
            Positive period used to count complete ticks exposed during recovery.
        nominal_clock : Quantity
            Positive normal clock frequency, convertible to Hz.
        throttled_clock : Quantity
            Positive reduced clock frequency, convertible to Hz.
        time_constant : Quantity
            Positive first-order thermal time constant.
        ambient : Quantity
            Ambient absolute temperature, convertible to kelvin.
        theta : Quantity
            Positive junction-to-ambient thermal resistance in K/W.
        quiet_power : Quantity
            Nonnegative quiet-state dissipation in watts.
        loaded_power : Quantity
            Loaded dissipation in watts, greater than quiet_power.
        trip : Quantity
            Trip temperature above clear and below the loaded equilibrium.
        clear : Quantity
            Recovery temperature above the quiet equilibrium and below trip.

        Returns
        -------
        ThermalThrottleResult
            Execution and response times in microseconds, junction temperatures in
            degrees Celsius, thermal durations, and whole exposed-tick counts. Overrun
            may be negative when slack remains.

        See Also
        --------
        mlsysim.physics.motors.calc_first_order_time_to_temperature : First-order thermal threshold crossing.

        Notes
        -----
        Equation: cycles = ``wcet*nominal_clock``; throttled execution =
        cycles/throttled_clock. T_inf = ambient + ``power*theta`` and t =
        ``tau*ln((T_inf-T_start)/(T_inf-T_target))``.

        Assumptions: Fixed instruction-cycle demand, a single lumped thermal pole, step
        changes in power, and reachable trip/clear thresholds.

        Excludes: Memory-bound execution, temperature-dependent leakage, spatial thermal
        gradients, and governor transition overhead.

        References
        ----------
        Derivation: conserve execution cycles across the frequency change; solve the
        first-order exponential thermal response for each threshold crossing.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import ThermalThrottleModel
        >>> result = ThermalThrottleModel().solve(wcet=Q_(10, "us"),
        ...     blocking_and_interference=Q_(1, "us"), deadline=Q_(30, "us"),
        ...     tick=Q_(1, "ms"), nominal_clock=Q_(1, "GHz"),
        ...     throttled_clock=Q_(0.5, "GHz"), time_constant=Q_(10, "s"),
        ...     ambient=Q_(300, "K"), theta=Q_(1, "K/W"),
        ...     quiet_power=Q_(10, "W"), loaded_power=Q_(100, "W"),
        ...     trip=Q_(350, "K"), clear=Q_(330, "K"))
        >>> bool(result.execution_throttled > result.execution and result.recovery > Q_(0, "s"))
        True
        """
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


class CommandAuthenticationModel(ForwardModel):
    """
    Authentication of a remote command as one more pre-brake delay, and a flood of forgeries.

    A supplied cycle count divided by the permission processor's clock gives
    the time of one signature verification and of a MAC over a frame of
    ``mac_blocks`` cipher blocks. Verification adds to the delay before a
    remote stop can act, so each millisecond is travel at ``speed`` against
    the finished budget's spare; a replay at the edge of the freshness window
    carries the machine v x tau_fresh. Verified serially, n forged signed
    packets queued ahead of a genuine veto finish it at (n + 1) t_verify; the
    fewest that push it past the deadline is n = floor(T_latest / t_verify),
    sustained at n / ((n + 1) t_verify) packets per second (added 2026-10-20).
    """

    requires = ("clock", "verify_cycles", "speed", "spare", "deadline")
    produces = CommandAuthenticationResult

    def solve(self, clock: Quantity, verify_cycles: float, mac_block_cycles: float, mac_blocks: int,
              speed: Quantity, spare: Quantity, deadline: Quantity, deadline_after_admission: Quantity,
              freshness_window: Quantity) -> CommandAuthenticationResult:
        """Charge serial authentication work against a remote-command deadline.

        Parameters
        ----------
        clock : Quantity
            Positive verification-processor clock frequency in Hz.
        verify_cycles : float
            Positive assumed cycle count for one signature verification.
        mac_block_cycles : float
            Positive assumed cycle count for authenticating one cipher block.
        mac_blocks : int
            Positive cipher-block count per authenticated command.
        speed : Quantity
            Nonnegative machine speed held during verification and freshness delays.
        spare : Quantity
            Remaining distance allowance before adding authentication delay.
        deadline : Quantity
            Positive total command-verification deadline.
        deadline_after_admission : Quantity
            Positive verification allowance for a command already admitted.
        freshness_window : Quantity
            Nonnegative maximum age accepted by the command freshness check.

        Returns
        -------
        CommandAuthenticationResult
            Signature time in ms, MAC time in microseconds, travel in mm, and
            forged-packet counts/rate. Negative remaining spare denotes budget
            exhaustion.

        See Also
        --------
        mlsysim.physics.quantities.compute_time : Work divided by service rate.

        Notes
        -----
        Equation: t_sig = verify_cycles/clock; t_mac =
        ``mac_blocks*mac_block_cycles/clock``; travel = ``speed*time``; n =
        floor(deadline/t_sig) forged commands delay the following veto until
        ``(n+1)*t_sig``.

        Assumptions: Cycle counts are supplied, verification is serial, and each forged
        signed packet consumes a full verification before rejection.

        Excludes: Cryptographic security strength, early rejection, parallel
        verification, network admission controls, and a deployed-system timing
        guarantee.

        References
        ----------
        Derivation: convert cycles to time at a fixed clock and count serial service
        slots before the deadline; integrate constant speed over that delay.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import CommandAuthenticationModel
        >>> result = CommandAuthenticationModel().solve(Q_(1, "MHz"), 1000, 10, 1,
        ...     Q_(1, "m/s"), Q_(1, "m"), Q_(10, "ms"), Q_(5, "ms"), Q_(1, "ms"))
        >>> bool(result.veto_verified_at > result.deadline)
        True
        """
        t_sig = (verify_cycles / clock).to(_MS)
        t_mac = (mac_blocks * mac_block_cycles / clock).to(_US)
        n = math.floor((deadline / t_sig).to("").magnitude)
        admit = ((n + 1) * t_sig).to(_MS)
        fresh = (speed * freshness_window).to(_MM)
        return CommandAuthenticationResult(
            verify_time=t_sig, mac_time=t_mac, verify_travel=(speed * t_sig).to(_MM),
            spare_after_verify=(spare - speed * t_sig).to(_MM), mac_travel=(speed * t_mac).to(_MM),
            freshness_window=freshness_window.to(_MS), freshness_travel=fresh,
            spare_after_freshness=(spare - fresh).to(_MM),
            deadline=deadline.to(_MS), deadline_after_admission=deadline_after_admission.to(_MS),
            forged_to_miss=n, veto_verified_at=admit,
            forged_rate=math.ceil((n / admit).to(1 / ureg.second).magnitude),
            forged_mac_to_miss=math.floor((deadline / t_mac).to("").magnitude),
            forged_to_miss_after_admission=math.floor((deadline_after_admission / t_sig).to("").magnitude),
        )
