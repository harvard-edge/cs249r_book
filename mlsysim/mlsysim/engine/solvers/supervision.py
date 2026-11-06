"""Responsible physical AI solvers: a detection gap in meters and joules, one assistant's queue.

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). The equations live in
``physics/statistics.py`` and ``physics/serving.py``; these models compose them
for one stated machine and site, so a chapter, a slide deck, and a lab read
the same arithmetic (added 2026-10-20).
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..results import DetectionShortfallResult, DetectionShortfallRow, SupervisorQueueResult
from ...core.types import Quantity
from ...physics.serving import calc_mg1_wait_ratio, calc_queue_latency_mmc
from ...physics.statistics import calc_consecutive_detection_frames
from .base import ForwardModel


class DetectionShortfallModel(ForwardModel):
    """
    A stop sized for the best-detected group, met by groups the detector sees less often.

    Each group needs N = ceil(ln p_miss / ln(1 - p_frame)) independent
    frame opportunities to bound the probability of missing every frame. A group needing Delta N more frames than the
    reference loses Delta d = v T_frame Delta N before braking begins. The
    reference stop sheds all of v^2 just as it reaches the person, so the late
    stop arrives with v_impact = sqrt(2 a Delta d) and E = m a Delta d, valid
    while Delta d stays below the braking travel v^2 / (2 a).
    """
    requires = ("speed", "frame_period", "deceleration", "mass")
    produces = DetectionShortfallResult

    def solve(self, speed: Quantity, frame_period: Quantity, deceleration: Quantity, mass: Quantity,
              all_miss_probability: float, reference: float, groups: Sequence[float],
              delay: Optional[Quantity] = None) -> DetectionShortfallResult:
        """Translate a detection-probability gap into braking distance and energy.

        Parameters
        ----------
        speed : Quantity
            Positive initial approach speed, in meters per second.
        frame_period : Quantity
            Positive interval between detector frames, in seconds.
        deceleration : Quantity
            Positive constant braking-deceleration magnitude, in meters per second squared.
        mass : Quantity
            Positive moving mass, in kilograms.
        all_miss_probability : float
            Desired probability of missing every frame, between zero and one.
        reference : float
            Reference per-frame detection probability, between zero and one.
        groups : sequence of float
            Per-frame detection probabilities strictly between zero and one and no
            greater than reference. Higher-than-reference detection is outside
            the late-stop model.
        delay : Quantity, optional
            Nonnegative extra reaction delay, in seconds. If omitted, reference
            brake-onset and absolute stopping distances are not calculated.

        Returns
        -------
        DetectionShortfallResult
            Per-group frame counts, extra time in ms, lost distance in meters,
            impact speed in m/s, and impact energy in joules. within_braking flags
            whether each delay distance is below the reference braking travel;
            impact estimates outside that regime must not be interpreted physically.

        See Also
        --------
        mlsysim.physics.statistics.calc_consecutive_detection_frames : Independent miss-trial count.

        Notes
        -----
        Equation: N = ceil(log(all_miss_probability) / log(1 - p));
            lost_distance = speed * frame_period * (N - N_reference).
            While lost_distance < ``speed**2`` / (2 * deceleration), the residual
            impact speed is sqrt(2 * deceleration * lost_distance) and energy is
            mass * deceleration * lost_distance.
        Assumptions: independent equal-probability frame trials, a stop sized to
            reach the target at zero speed for the reference group, and constant
            deceleration after detection. Counts bound all misses, not consecutive
            successful detections.
        Excludes: correlated frames, obstacle motion, actuator saturation, collision
            mechanics, and validation of a detector's group-specific probabilities.

        References
        ----------
        Derivation: multiply independent miss probabilities to solve for frame
            count; delayed braking preserves kinetic energy ``m*a*lost_distance``
            relative to the reference trajectory under constant deceleration.

        Examples
        --------
        A constructed group identical to the reference has no extra lost distance.

        >>> from mlsysim import Q_
        >>> result = DetectionShortfallModel().solve(
        ...     Q_(1, 'm/s'), Q_(0.01, 's'), Q_(1, 'm/s^2'), Q_(10, 'kg'),
        ...     0.01, 0.9, [0.9])
        >>> result.rows[0].distance_lost == Q_(0, 'm')
        True
        """
        n_ref = calc_consecutive_detection_frames(reference, all_miss_probability)
        braking = (speed**2 / (2 * deceleration)).to("m")
        rows = []
        for p in groups:
            n = calc_consecutive_detection_frames(p, all_miss_probability)
            dn = n - n_ref
            lost = (speed * frame_period * dn).to("m")
            rows.append(DetectionShortfallRow(
                per_frame_detection=p, frames=n, extra_frames=dn,
                extra_time=(frame_period * dn).to("ms"), distance_lost=lost,
                impact_speed=((2 * deceleration * lost) ** 0.5).to("m/s"),
                impact_energy=(mass * deceleration * lost).to("J")))
        onset = stop = None
        if delay is not None:
            onset = (speed * (n_ref * frame_period + delay)).to("m")
            stop = (onset + braking).to("m")
        return DetectionShortfallResult(
            frame_period=frame_period.to("ms"), reference_frames=n_ref, braking_travel=braking,
            reference_brake_onset=onset, reference_stop=stop, rows=rows,
            within_braking=all(r.distance_lost < braking for r in rows))


class SupervisorQueueModel(ForwardModel):
    """
    One remote assistant serving a fleet, as an M/M/1 queue.

    Requests arrive at lambda = N r (N machines, r requests per machine-hour)
    and each holds the assistant for a mean s: utilization rho = lambda s, mean
    wait W_q = rho s / (1 - rho) (the M/G/1 wait ratio with exponential
    service), and the 99th-percentile wait from the Erlang C tail with one
    server.
    """
    requires = ("request_rate", "service_time", "machines")
    produces = SupervisorQueueResult

    def solve(self, request_rate: Quantity, service_time: Quantity, machines: int) -> SupervisorQueueResult:
        """Estimate a remote assistant's stationary request queue.

        Parameters
        ----------
        request_rate : Quantity
            Positive mean requests per machine per unit time, convertible to 1/hour.
        service_time : Quantity
            Positive mean time the assistant holds one request, in seconds.
        machines : int
            Positive number of independently requesting machines.

        Returns
        -------
        SupervisorQueueResult
            Fleet arrival rate in 1/hour, utilization as a fraction, mean and
            99th-percentile queue waits in seconds, and busy minutes per hour.

        Raises
        ------
        ValueError
            If fleet utilization is not between zero and one.

        See Also
        --------
        mlsysim.physics.serving.calc_mg1_wait_ratio : Mean queue-wait ratio.
        mlsysim.physics.serving.calc_queue_latency_mmc : Erlang-C queue tail.

        Notes
        -----
        Equation: lambda = machines * request_rate; rho = lambda * service_time;
            mean_wait = rho * service_time / (1 - rho). The tail comes from
            the Erlang-C waiting-time distribution with one server.
        Assumptions: independent Poisson arrivals, exponentially distributed service,
            one always-available assistant, and stationary utilization below one.
        Excludes: shifts, priority queues, correlated incidents, abandonment,
            travel time, and any guarantee on an individual intervention deadline.

        References
        ----------
        Derivation: superpose independent Poisson request streams, then apply the
            single-server M/M/1 specialization of the linked queueing functions.

        Examples
        --------
        A constructed low-load fleet yields positive finite waiting times.

        >>> from mlsysim import Q_
        >>> result = SupervisorQueueModel().solve(Q_(1, '1/hour'), Q_(1, 'minute'), 10)
        >>> 0 < result.utilization < 1 and result.mean_wait > Q_(0, 's')
        True
        """
        lam = (machines * request_rate).to("1/hour")
        rho = (lam * service_time).to("").magnitude
        if not 0.0 < rho < 1.0:
            raise ValueError(f"utilization must be in (0, 1), got {rho}")
        wait = (calc_mg1_wait_ratio(rho, 1.0) * service_time).to("s")
        _, _, p99 = calc_queue_latency_mmc(lam.to("1/s").magnitude, (1 / service_time).to("1/s").magnitude, 1)
        return SupervisorQueueResult(
            machines=machines, arrival_rate=lam, utilization=rho, mean_wait=wait, p99_wait=p99.to("s"),
            busy_per_hour=(lam * service_time).to("minute/hour"))
