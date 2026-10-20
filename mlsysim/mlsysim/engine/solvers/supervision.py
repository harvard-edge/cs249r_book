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

    Each group needs N = ceil(ln p_miss / ln(1 - p_frame)) consecutive frames
    before detection counts. A group needing Delta N more frames than the
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
        lam = (machines * request_rate).to("1/hour")
        rho = (lam * service_time).to("").magnitude
        if not 0.0 < rho < 1.0:
            raise ValueError(f"utilization must be in (0, 1), got {rho}")
        wait = (calc_mg1_wait_ratio(rho, 1.0) * service_time).to("s")
        _, _, p99 = calc_queue_latency_mmc(lam.to("1/s").magnitude, (1 / service_time).to("1/s").magnitude, 1)
        return SupervisorQueueResult(
            machines=machines, arrival_rate=lam, utilization=rho, mean_wait=wait, p99_wait=p99.to("s"),
            busy_per_hour=(lam * service_time).to("minute/hour"))
