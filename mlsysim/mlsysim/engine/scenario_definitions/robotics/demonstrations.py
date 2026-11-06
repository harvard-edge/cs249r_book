"""Physical-machine scenarios for demonstrations."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from ....core.types import Quantity
from ....core.units import Q_
from ..base import BaseScenario


class CorrelatedLogScenario(BaseScenario):
    """``duration`` of records logged at ``rate`` whose adjacent records have lag-one correlation
    ``lag1_correlation``: how few independent samples the record count holds.

    Notes
    -----
    Equation: Records=``rate*duration``; effective samples follow calc_effective_sample_size using lag-one correlation.

    Assumptions: Stationary sampling with the helper's correlation approximation.

    Excludes: General long-memory dependence and dropped or irregularly sampled records.

    References
    ----------
    Implementation references: ``mlsysim.physics.statistics.calc_effective_sample_size``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.PermissionTickTelemetryHour
    >>> round(scenario.effective_samples, 6)
    1800.90045
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rate: Quantity
    duration: Quantity
    lag1_correlation: float

    @property
    def records(self) -> float:
        """Total record count at constant rate over the stated duration; no dropped records are modeled.

        Equation: (self.rate * self.duration).to('').magnitude
        """
        return (self.rate * self.duration).to("").magnitude

    @property
    def record_interval(self) -> Quantity:
        """Time between records in ms, the reciprocal of the logging rate.

        Equation: (1 / self.rate).to('ms')
        """
        return (1 / self.rate).to("ms")

    @property
    def effective_samples(self) -> float:
        """Approximate independent sample count from ``calc_effective_sample_size`` under the declared lag-one correlation model.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ....physics.statistics import calc_effective_sample_size

        return calc_effective_sample_size(self.records, self.lag1_correlation)

    @property
    def oversampling(self) -> float:
        """Recorded observations per effective independent sample: ``records / effective_samples``.

        Equation: self.records / self.effective_samples
        """
        return self.records / self.effective_samples


class ImitationBoundScenario(BaseScenario):
    """Worst-case imitation cost over one ``episode`` at ``control_rate`` with per-step error
    ``per_step_error``: open loop, behavioral cloning, and aggregation with recoverability ``u``.

    Notes
    -----
    Equation: Cloning scales as ``horizon**2*error``; aggregation uses ``recoverability*horizon*error``; capped values cannot exceed horizon.

    Assumptions: Per-step loss is bounded by one and the supplied error/recoverability satisfy the cited helper assumptions.

    Excludes: Empirical rollout measurement or guarantees for policies outside those assumptions.

    References
    ----------
    Implementation references: ``mlsysim.physics.statistics.calc_behavior_cloning_cost_bound``, ``mlsysim.physics.statistics.calc_dagger_cost_bound``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ImitationBoundsFiveSecondEpisode
    >>> round(scenario.cloning_capped, 6)
    250.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    episode: Optional[Quantity] = None
    control_rate: Optional[Quantity] = None
    per_step_error: float
    recoverability: float = 1.0
    # Optional (added 2026-10-20): a horizon stated in decision steps (an agent's
    # turns), used when there is no episode duration and control rate.
    horizon: Optional[int] = None

    @property
    def horizon_steps(self) -> int:
        """Episode decision count; uses explicit ``horizon`` when supplied, otherwise rounds episode duration times control rate.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        if self.horizon is not None:
            return self.horizon
        return int(round((self.episode * self.control_rate).to("").magnitude))

    # Capped bounds (added 2026-10-20): with per-step cost in [0, 1], excess cost
    # never exceeds the horizon, so each bound is min(S, bound).
    @property
    def cloning_capped(self) -> float:
        """Behavior-cloning cost bound capped at the episode's maximum unit loss per step.

        Equation: min(self.horizon_steps, self.cloning)
        """
        return min(self.horizon_steps, self.cloning)

    @property
    def aggregation_capped(self) -> float:
        """Dataset-aggregation cost bound capped at the same horizon maximum.

        Equation: min(self.horizon_steps, self.aggregation)
        """
        return min(self.horizon_steps, self.aggregation)

    @property
    def capped_gap(self) -> float:
        """Dimensionless ratio of capped cloning and aggregation bounds; requires nonzero aggregation cost.

        Equation: self.cloning_capped / self.aggregation_capped
        """
        return self.cloning_capped / self.aggregation_capped

    @property
    def matching_error(self) -> float:
        """Per-step cloning error needed for its quadratic bound to equal the capped aggregation cost: ``aggregation_capped / horizon_steps**2``.

        Equation: self.aggregation_capped / self.horizon_steps ** 2
        """
        return self.aggregation_capped / self.horizon_steps**2

    @property
    def saturation_horizon(self) -> float:
        """Horizon at which the uncapped quadratic cloning bound reaches the maximum cumulative loss: reciprocal of positive per-step error.

        Equation: 1.0 / self.per_step_error
        """
        return 1.0 / self.per_step_error

    @property
    def open_loop(self) -> float:
        """Linear count of expected errors without compounding: horizon times per-step error; a comparison baseline.

        Equation: self.horizon_steps * self.per_step_error
        """
        return self.horizon_steps * self.per_step_error

    @property
    def cloning(self) -> float:
        """Uncapped quadratic imitation bound from ``calc_behavior_cloning_cost_bound``; not an empirical rollout error count.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ....physics.statistics import calc_behavior_cloning_cost_bound

        return calc_behavior_cloning_cost_bound(self.horizon_steps, self.per_step_error)

    @property
    def aggregation(self) -> float:
        """Recoverability-scaled linear bound from ``calc_dagger_cost_bound``; not a guarantee for arbitrary learned policies.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ....physics.statistics import calc_dagger_cost_bound

        return calc_dagger_cost_bound(self.horizon_steps, self.per_step_error, self.recoverability)

    @property
    def vacuous_error(self) -> float:
        """Per-step error threshold at which the quadratic bound reaches horizon loss: ``1 / horizon_steps``.

        Equation: 1.0 / self.horizon_steps
        """
        return 1.0 / self.horizon_steps

    @property
    def cloning_over_maximum(self) -> float:
        """Uncapped cloning bound divided by the maximum horizon loss; values above one indicate a vacuous bound.

        Equation: self.cloning / self.horizon_steps
        """
        return self.cloning / self.horizon_steps


class PairingLagScenario(BaseScenario):
    """An observation paired with an action ``lag`` later, against a ``clearance``, at a transit
    speed and at ``site``'s guarded and fast latch approaches.

    Notes
    -----
    Equation: Alignment displacement=``speed*lag``; permissible lag=clearance/speed.

    Assumptions: Constant speed during pairing or shutter delay and a scalar clearance budget.

    Excludes: Acceleration, rotation, deformable contact, and a safety certificate.

    References
    ----------
    Derivation: Constant-velocity kinematics gives displacement as velocity times elapsed lag. Solving that relation for lag at a supplied clearance gives the timing ceiling.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchPairingLag
    >>> round((scenario.displacement_transit).to('mm').magnitude, 6)
    17.5
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedScenario (latch approach speeds)
    lag: Quantity
    clearance: Quantity
    transit_speed: Quantity
    shutter_skew_low: Quantity  # full-frame rolling-shutter skew, fast readout
    shutter_skew_high: Quantity  # full-frame rolling-shutter skew, one frame

    @property
    def guarded_speed(self) -> Quantity:
        """Guarded approach speed copied from the site's ``v_latch_approach``, retaining velocity units.

        Equation: self.site.v_latch_approach
        """
        return self.site.v_latch_approach

    @property
    def fast_speed(self) -> Quantity:
        """Faster comparison speed copied from the site's ``v_latch_high``, retaining velocity units.

        Equation: self.site.v_latch_high
        """
        return self.site.v_latch_high

    def displacement(self, speed: Quantity, lag: Optional[Quantity] = None) -> Quantity:
        """Constant-speed travel in mm over explicit ``lag`` or the scenario's pairing lag; acceleration is excluded.

        Equation: (speed * (self.lag if lag is None else lag)).to('mm')
        """
        return (speed * (self.lag if lag is None else lag)).to("mm")

    def lag_ceiling(self, speed: Quantity) -> Quantity:
        """Maximum alignment delay in ms before constant-speed travel consumes the declared clearance.

        Equation: (self.clearance / speed).to('ms')
        """
        return (self.clearance / speed).to("ms")

    @property
    def displacement_transit(self) -> Quantity:
        """Pairing-lag travel in mm at the supplied transit speed, via ``displacement``.

        Equation: self.displacement(self.transit_speed)
        """
        return self.displacement(self.transit_speed)

    @property
    def displacement_guarded(self) -> Quantity:
        """Pairing-lag travel in mm at the guarded approach speed, via ``displacement``.

        Equation: self.displacement(self.guarded_speed)
        """
        return self.displacement(self.guarded_speed)

    @property
    def displacement_fast(self) -> Quantity:
        """Pairing-lag travel in mm at the faster approach speed, via ``displacement``.

        Equation: self.displacement(self.fast_speed)
        """
        return self.displacement(self.fast_speed)

    @property
    def transit_over_clearance(self) -> float:
        """Dimensionless fraction of clearance consumed by transit-speed pairing error.

        Equation: (self.displacement_transit / self.clearance).to('').magnitude
        """
        return (self.displacement_transit / self.clearance).to("").magnitude

    @property
    def fast_over_clearance(self) -> float:
        """Dimensionless fraction of clearance consumed by faster-approach pairing error.

        Equation: (self.displacement_fast / self.clearance).to('').magnitude
        """
        return (self.displacement_fast / self.clearance).to("").magnitude

    @property
    def lag_ceiling_guarded(self) -> Quantity:
        """Alignment-delay budget in ms at the guarded speed, via ``lag_ceiling``.

        Equation: self.lag_ceiling(self.guarded_speed)
        """
        return self.lag_ceiling(self.guarded_speed)

    @property
    def lag_ceiling_fast(self) -> Quantity:
        """Alignment-delay budget in ms at the faster speed, via ``lag_ceiling``.

        Equation: self.lag_ceiling(self.fast_speed)
        """
        return self.lag_ceiling(self.fast_speed)

    @property
    def skew_displacement_transit(self) -> Quantity:
        """Travel in mm over the high shutter-skew interval at transit speed; uses the constant-speed displacement model.

        Equation: self.displacement(self.transit_speed, self.shutter_skew_high)
        """
        return self.displacement(self.transit_speed, self.shutter_skew_high)

    @property
    def skew_displacement_fast(self) -> Quantity:
        """Travel in mm over the high shutter-skew interval at the faster approach speed.

        Equation: self.displacement(self.fast_speed, self.shutter_skew_high)
        """
        return self.displacement(self.fast_speed, self.shutter_skew_high)


class TeleopLoopScenario(BaseScenario):
    """One teleoperated correction: display, human, command transport, and mechanical response.

    Notes
    -----
    Equation: Loop delay=display+human+command+mechanical.

    Assumptions: Serial additive delays supplied for the same correction path.

    Excludes: Overlapped stages and statistical tail composition.

    References
    ----------
    Derivation: The correction path is explicitly serial, so its display, human, command, and mechanical intervals add. Dividing one interval by their sum yields its contribution to total delay.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.TeleopCorrectionLoop
    >>> round((scenario.loop).to('ms').magnitude, 6)
    340
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    display: Quantity
    human: Quantity
    command: Quantity
    mechanical: Quantity

    @property
    def loop(self) -> Quantity:
        """Complete serial correction-loop delay in ms: display, human response, command transport, and mechanical response.

        Equation: (self.display + self.human + self.command + self.mechanical).to('ms')
        """
        return (self.display + self.human + self.command + self.mechanical).to("ms")

    @property
    def command_issued(self) -> Quantity:
        """Elapsed time in ms before command transmission: display delay plus human response.

        Equation: (self.display + self.human).to('ms')
        """
        return (self.display + self.human).to("ms")

    @property
    def human_share(self) -> float:
        """Fraction of total loop delay attributable to the supplied human-response time.

        Equation: (self.human / self.loop).to('').magnitude
        """
        return (self.human / self.loop).to("").magnitude


class TeleopIngestionScenario(BaseScenario):
    """Raw telemetry of one teleoperated session on ``machine``: its navigation and wrist cameras,
    ``external_cameras`` RGB-D cameras, and the kinematic record at the permission tick.

    Notes
    -----
    Equation: Raw bandwidth sums RGB, depth, and kinematic streams; stored bytes=``bandwidth*duration``.

    Assumptions: Uncompressed fixed-rate streams and optional uniform compression ratio; camera geometry comes from the supplied device entries.

    Excludes: Codec cost, dropped frames, file framing, and storage replication.

    References
    ----------
    Implementation references: ``mlsysim.physics.robotics.perception.calc_teleop_ingestion_budget``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseTeleopIngestion
    >>> round((scenario.total_rate).to('MB/s').magnitude, 6)
    3121.6832
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # EmbodiedMobileManipulator (cameras, axes, permission rate)
    external_camera: Any  # Sensors.Camera entry (RGB and aligned depth)
    external_cameras: int
    rgb_bytes_per_pixel: int
    depth_bytes_per_pixel: int
    values_per_axis: int
    bytes_per_value: int
    timestamp_bytes: int
    shift: Quantity
    compression_ratio: float
    slow_drive_bandwidth: Quantity
    fast_drive_bandwidth: Quantity

    @property
    def record_bytes(self) -> int:
        """Bytes in one kinematic record: values for every servo axis plus its timestamp.

        Equation: self.machine.servo_axes * self.values_per_axis * self.bytes_per_value + self.timestamp_bytes
        """
        return self.machine.servo_axes * self.values_per_axis * self.bytes_per_value + self.timestamp_bytes

    @property
    def kinematic_rate(self) -> Quantity:
        """Raw kinematic telemetry bandwidth in MB/s: record bytes times permission-record frequency.

        Equation: (self.record_bytes * Q_(1, 'byte') * self.machine.permission_rate).to('MB/s')
        """
        return (self.record_bytes * Q_(1, "byte") * self.machine.permission_rate).to("MB/s")

    def _budget(self, camera, count, depth, kinematics=None):
        from ....physics.robotics.perception import calc_teleop_ingestion_budget

        fps = camera.frame_rate.to("Hz").magnitude
        extra = (
            dict(
                depth_width=camera.resolution_width,
                depth_height=camera.resolution_height,
                depth_fps=fps,
                depth_bytes_per_pixel=self.depth_bytes_per_pixel,
            )
            if depth
            else {}
        )
        return calc_teleop_ingestion_budget(
            num_cameras=count,
            rgb_width=camera.resolution_width,
            rgb_height=camera.resolution_height,
            rgb_fps=fps,
            rgb_bytes_per_pixel=self.rgb_bytes_per_pixel,
            kinematics_bytes_per_sec=kinematics,
            **extra,
        )

    @property
    def nav_budget(self) -> dict:
        """Navigation-camera and kinematic byte budget from ``_budget``/``calc_teleop_ingestion_budget``; includes one RGB stream and no depth.

        Equation: self._budget(self.machine.nav_camera, 1, False, self.kinematic_rate)
        """
        return self._budget(self.machine.nav_camera, 1, False, self.kinematic_rate)

    @property
    def wrist_budget(self) -> dict:
        """Wrist-camera budget from ``_budget``/``calc_teleop_ingestion_budget``; one RGB stream without kinematic traffic.

        Equation: self._budget(self.machine.wrist_camera, 1, False)
        """
        return self._budget(self.machine.wrist_camera, 1, False)

    @property
    def external_budget(self) -> dict:
        """External-camera budget from ``_budget``/``calc_teleop_ingestion_budget``; includes the declared number of RGB and depth streams.

        Equation: self._budget(self.external_camera, self.external_cameras, True)
        """
        return self._budget(self.external_camera, self.external_cameras, True)

    @property
    def nav_rate(self) -> Quantity:
        """Raw navigation RGB bandwidth from ``nav_budget['bw_rgb']``, in MB/s.

        Equation: self.nav_budget['bw_rgb']
        """
        return self.nav_budget["bw_rgb"]

    @property
    def wrist_rate(self) -> Quantity:
        """Raw wrist RGB bandwidth from ``wrist_budget['bw_rgb']``, in MB/s.

        Equation: self.wrist_budget['bw_rgb']
        """
        return self.wrist_budget["bw_rgb"]

    @property
    def external_rgb_rate(self) -> Quantity:
        """Combined external RGB bandwidth from the external budget, in MB/s.

        Equation: self.external_budget['bw_rgb']
        """
        return self.external_budget["bw_rgb"]

    @property
    def external_depth_rate(self) -> Quantity:
        """Combined external depth bandwidth from the external budget, in MB/s.

        Equation: self.external_budget['bw_depth']
        """
        return self.external_budget["bw_depth"]

    @property
    def external_rate(self) -> Quantity:
        """Combined external RGB-plus-depth bandwidth, in MB/s.

        Equation: self.external_budget['bw_vision']
        """
        return self.external_budget["bw_vision"]

    @property
    def total_rate(self) -> Quantity:
        """Aggregate raw bandwidth in MB/s, summing navigation, wrist, external, and kinematic components once.

        Equation: (self.nav_budget['bw_total'] + self.wrist_budget['bw_total'] + self.external_budget['bw_total']).to('MB/s')
        """
        return (self.nav_budget["bw_total"] + self.wrist_budget["bw_total"] + self.external_budget["bw_total"]).to(
            "MB/s"
        )

    @property
    def storage_per_hour(self) -> Quantity:
        """Raw storage in decimal TB for one hour, summing the three stream budgets.

        Equation: (self.nav_budget['storage_per_hour'] + self.wrist_budget['storage_per_hour'] + self.external_budget['storage_per_hour']).to('TB')
        """
        return (
            self.nav_budget["storage_per_hour"]
            + self.wrist_budget["storage_per_hour"]
            + self.external_budget["storage_per_hour"]
        ).to("TB")

    @property
    def storage_per_shift(self) -> Quantity:
        """Raw storage in decimal TB for the declared shift duration at constant ingestion rates.

        Equation: (self.storage_per_hour * self.shift.to('hour').magnitude).to('TB')
        """
        return (self.storage_per_hour * self.shift.to("hour").magnitude).to("TB")

    @property
    def nav_over_slow_drive(self) -> float:
        """Dimensionless ratio of navigation RGB bandwidth to the supplied slower drive's sustained bandwidth.

        Equation: (self.nav_rate / self.slow_drive_bandwidth).to('').magnitude
        """
        return (self.nav_rate / self.slow_drive_bandwidth).to("").magnitude

    @property
    def kinematic_share(self) -> float:
        """Fraction of aggregate raw bandwidth occupied by kinematic records.

        Equation: (self.kinematic_rate / self.total_rate).to('').magnitude
        """
        return (self.kinematic_rate / self.total_rate).to("").magnitude

    @property
    def compressed_rate(self) -> Quantity:
        """Aggregate MB/s after dividing all traffic by the supplied uniform compression ratio; excludes codec compute.

        Equation: (self.total_rate / self.compression_ratio).to('MB/s')
        """
        return (self.total_rate / self.compression_ratio).to("MB/s")

    @property
    def compressed_storage_per_hour(self) -> Quantity:
        """TB per hour under the same uniform compression ratio as ``compressed_rate``.

        Equation: (self.storage_per_hour / self.compression_ratio).to('TB')
        """
        return (self.storage_per_hour / self.compression_ratio).to("TB")


class DemonstrationCampaignScenario(BaseScenario):
    """Scheduled sessions of attempt-and-reset cycles: demonstrations kept per session, the
    campaign that keeps ``target_kept`` of them, and the wear it puts on a gear and a cable.

    Notes
    -----
    Equation: Cycles=floor(active_time/(attempt+reset)); retention=``(1-failure_rate)*(1-qa_rate)``; campaign_attempts=ceil(target/retention).

    Assumptions: QA rejects only nonfailed attempts; fixed cycle costs and constant expected rejection rates.

    Excludes: A confidence guarantee for retained count, fatigue-dependent rates, and stochastic wear.

    References
    ----------
    Derivation: Whole attempt/reset cycles are packed into available session time. Sequential filtering retains the nonfailed share times the QA-surviving share; the expected retained count and nominal cycle consumption then determine the campaign budget.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchDemonstrationCampaign
    >>> scenario.attempts_per_session
    52
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    session: Quantity  # scheduled session length
    startup: Quantity
    calibration: Quantity
    rest: Quantity
    attempt: Quantity
    reset: Quantity
    failure_rate: float  # attempts that fail in the trial
    qa_rate: float  # surviving attempts discarded by quality filtering
    target_kept: int
    gear_reversals_per_attempt: int
    cable_flexes_opening: int
    cable_flexes_return: int
    gear_rated_cycles: int
    cable_rated_cycles: int
    repeat_campaigns: int = 1  # campaigns compared against the rated lives

    @property
    def active(self) -> Quantity:
        """Available attempt-and-reset time in minutes after startup, calibration, and rest are subtracted from the session.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        s = [q.to("second") for q in (self.session, self.startup, self.calibration, self.rest)]
        return (s[0] - s[1] - s[2] - s[3]).to("minute")

    @property
    def cycle(self) -> Quantity:
        """Duration in seconds of one attempt followed by its reset.

        Equation: (self.attempt + self.reset).to('second')
        """
        return (self.attempt + self.reset).to("second")

    @property
    def attempts_per_session(self) -> int:
        """Whole attempt/reset cycles fitting the active session, rounded down.

        Equation: int(self.active.to('second').magnitude // self.cycle.to('second').magnitude)
        """
        return int(self.active.to("second").magnitude // self.cycle.to("second").magnitude)

    @property
    def idle(self) -> Quantity:
        """Unused active-session remainder in seconds after packing whole cycles.

        Equation: (self.active.to('second') - self.attempts_per_session * self.cycle.to('second')).to('second')
        """
        return (self.active.to("second") - self.attempts_per_session * self.cycle.to("second")).to("second")

    @property
    def retention(self) -> float:
        """Expected retained fraction ``(1-failure_rate)*(1-qa_rate)``; QA rejects only otherwise successful attempts.

        Equation: (1.0 - self.failure_rate) * (1.0 - self.qa_rate)
        """
        return (1.0 - self.failure_rate) * (1.0 - self.qa_rate)

    @property
    def kept_per_session(self) -> float:
        """Expected retained demonstrations per session; fractional because retention is probabilistic.

        Equation: self.attempts_per_session * self.retention
        """
        return self.attempts_per_session * self.retention

    @property
    def kept_time(self) -> Quantity:
        """Expected retained motion duration in minutes, excluding resets and rejected attempts.

        Equation: (self.kept_per_session * self.attempt).to('minute')
        """
        return (self.kept_per_session * self.attempt).to("minute")

    @property
    def kept_share(self) -> float:
        """Fraction of the scheduled session represented by retained demonstration motion.

        Equation: (self.kept_time / self.session).to('').magnitude
        """
        return (self.kept_time / self.session).to("").magnitude

    @property
    def reset_time(self) -> Quantity:
        """Seconds spent resetting across all complete attempts in a session.

        Equation: (self.attempts_per_session * self.reset).to('second')
        """
        return (self.attempts_per_session * self.reset).to("second")

    @property
    def attempt_time(self) -> Quantity:
        """Seconds spent executing all complete attempts, whether retained or rejected.

        Equation: (self.attempts_per_session * self.attempt).to('second')
        """
        return (self.attempts_per_session * self.attempt).to("second")

    @property
    def failed_time(self) -> Quantity:
        """Expected seconds of attempt execution lost to failures.

        Equation: (self.failure_rate * self.attempt_time).to('second')
        """
        return (self.failure_rate * self.attempt_time).to("second")

    @property
    def qa_time(self) -> Quantity:
        """Expected seconds rejected by QA after excluding failed attempts.

        Equation: (self.qa_rate * (self.attempt_time - self.failed_time)).to('second')
        """
        return (self.qa_rate * (self.attempt_time - self.failed_time)).to("second")

    @property
    def failed_episodes(self) -> float:
        """Expected failed attempt count per session.

        Equation: self.failure_rate * self.attempts_per_session
        """
        return self.failure_rate * self.attempts_per_session

    @property
    def qa_episodes(self) -> float:
        """Expected QA-rejected count among attempts that did not fail.

        Equation: self.qa_rate * (self.attempts_per_session - self.failed_episodes)
        """
        return self.qa_rate * (self.attempts_per_session - self.failed_episodes)

    @property
    def campaign_attempts(self) -> int:
        """Attempts required in expectation to retain ``target_kept``, rounding target divided by retention upward; not a confidence guarantee.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        import math

        return int(math.ceil(self.target_kept / self.retention))

    @property
    def campaign_time(self) -> Quantity:
        """Scheduled campaign hours using the average attempts-per-session rate; allows a fractional final session.

        Equation: (self.campaign_attempts / self.attempts_per_session * self.session).to('hour')
        """
        return (self.campaign_attempts / self.attempts_per_session * self.session).to("hour")

    @property
    def kept_motion_time(self) -> Quantity:
        """Hours of motion in exactly ``target_kept`` retained demonstrations.

        Equation: (self.target_kept * self.attempt).to('hour')
        """
        return (self.target_kept * self.attempt).to("hour")

    @property
    def overhead_factor(self) -> float:
        """Scheduled campaign hours divided by retained motion hours; includes session overhead and expected rejections.

        Equation: (self.campaign_time / self.kept_motion_time).to('').magnitude
        """
        return (self.campaign_time / self.kept_motion_time).to("").magnitude

    @property
    def cable_flexes_per_attempt(self) -> int:
        """Cable-flex count per attempt, adding opening and return motions.

        Equation: self.cable_flexes_opening + self.cable_flexes_return
        """
        return self.cable_flexes_opening + self.cable_flexes_return

    @property
    def gear_cycles(self) -> int:
        """Gear-reversal count across the modeled campaign attempts.

        Equation: self.campaign_attempts * self.gear_reversals_per_attempt
        """
        return self.campaign_attempts * self.gear_reversals_per_attempt

    @property
    def cable_cycles(self) -> int:
        """Cable-flex count across the modeled campaign attempts.

        Equation: self.campaign_attempts * self.cable_flexes_per_attempt
        """
        return self.campaign_attempts * self.cable_flexes_per_attempt

    @property
    def gear_life_used(self) -> float:
        """Campaign reversal count as a fraction of the declared rated gear-cycle life; no wear distribution is modeled.

        Equation: self.gear_cycles / self.gear_rated_cycles
        """
        return self.gear_cycles / self.gear_rated_cycles

    @property
    def cable_life_used(self) -> float:
        """Campaign flex count as a fraction of the declared rated cable-cycle life.

        Equation: self.cable_cycles / self.cable_rated_cycles
        """
        return self.cable_cycles / self.cable_rated_cycles

    @property
    def repeat_cable_life_used(self) -> float:
        """Cumulative fraction of rated cable life for the declared number of identical campaigns.

        Equation: self.repeat_campaigns * self.cable_life_used
        """
        return self.repeat_campaigns * self.cable_life_used


class StateGridScenario(BaseScenario):
    """A state space of ``state_components`` axes cut into ``bins`` per axis, against the records
    ``fleet`` robots log at ``log_rate`` for ``duration``.

    Notes
    -----
    Equation: Cells=``bins**sum(state_components)``; coverage upper bound=``fleet*rate*duration/cells``.

    Assumptions: Uniform axis discretization and at most one newly occupied cell per recorded observation.

    Excludes: Actual occupancy distribution, repeated-state correction, and proof that a grid represents task coverage.

    References
    ----------
    Derivation: The Cartesian product of equally binned state axes contains bins raised to the summed dimension count. At most one new cell can be represented by each record, so records divided by cells is only an upper coverage bound.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseManipulatorStateGrid
    >>> scenario.dims
    28
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    state_components: Dict[str, int]
    bins: int
    fine_bins: int
    fleet: int
    log_rate: Quantity
    duration: Quantity

    @property
    def dims(self) -> int:
        """Total state dimensions, summing the declared component widths.

        Equation: sum(self.state_components.values())
        """
        return sum(self.state_components.values())

    @property
    def cells(self) -> float:
        """Number of coarse grid cells: ``bins ** dims``; a combinatorial count stored as float.

        Equation: float(self.bins) ** self.dims
        """
        return float(self.bins) ** self.dims

    @property
    def fine_cells(self) -> float:
        """Number of fine grid cells: ``fine_bins ** dims``.

        Equation: float(self.fine_bins) ** self.dims
        """
        return float(self.fine_bins) ** self.dims

    @property
    def records(self) -> float:
        """Total records from all fleet members at constant logging rate over the duration.

        Equation: self.fleet * (self.log_rate * self.duration).to('').magnitude
        """
        return self.fleet * (self.log_rate * self.duration).to("").magnitude

    @property
    def max_coverage(self) -> float:
        """Records divided by coarse cells, an upper bound on distinct-cell coverage assuming every record occupies a new cell; not clipped to one.

        Equation: self.records / self.cells
        """
        return self.records / self.cells


class ContactGridScenario(BaseScenario):
    """A contact task binned by contact angle and normal force, and the cells each archive occupies.

    ``archives`` maps a name to (angle_low, angle_high, force_low, force_high, held-out RMS action
    error); each archive's occupied region is a rectangle of whole bins.

    Notes
    -----
    Equation: Cells=``angle_bins*force_bins``; archive support is its rectangular bin span.

    Assumptions: Archive boundaries align with angle/force bins and the supplied RMS errors are external inputs.

    Excludes: Interpolation accuracy or probability mass within occupied bins.

    References
    ----------
    Derivation: Aligned angular and force bins form a Cartesian product. Each archive is represented by a rectangle whose angular count times force count gives occupied cells; the RMS error is carried from the input and is not inferred from occupancy.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ContactAngleCoverage
    >>> scenario.cells
    600
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    angle_min: Quantity
    angle_max: Quantity
    angle_bin: Quantity
    force_max: Quantity
    force_bin: Quantity
    records: int  # transitions each archive records
    record_rate: Quantity
    archives: Dict[str, Any]

    @property
    def angle_bins(self) -> int:
        """Integer angular bins spanning the declared angle range; rounds range divided by bin width.

        Equation: int(round(((self.angle_max - self.angle_min) / self.angle_bin).to('').magnitude))
        """
        return int(round(((self.angle_max - self.angle_min) / self.angle_bin).to("").magnitude))

    @property
    def force_bins(self) -> int:
        """Integer force bins from zero to ``force_max``, rounding the ratio to ``force_bin``.

        Equation: int(round((self.force_max / self.force_bin).to('').magnitude))
        """
        return int(round((self.force_max / self.force_bin).to("").magnitude))

    @property
    def cells(self) -> int:
        """Total contact grid cells: angular bins times force bins.

        Equation: self.angle_bins * self.force_bins
        """
        return self.angle_bins * self.force_bins

    def bin_span(self, name: str):
        """Archive rectangle as ``(first_angle, angle_count, first_force, force_count)`` integer bin coordinates; assumes edges align with the grid.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        a_lo, a_hi, f_lo, f_hi, _ = self.archives[name]
        first_a = int(round(((a_lo - self.angle_min) / self.angle_bin).to("").magnitude))
        n_a = int(round(((a_hi - a_lo) / self.angle_bin).to("").magnitude))
        first_f = int(round((f_lo / self.force_bin).to("").magnitude))
        n_f = int(round(((f_hi - f_lo) / self.force_bin).to("").magnitude))
        return first_a, n_a, first_f, n_f

    def occupied(self, name: str) -> int:
        """Grid-cell count in archive ``name``'s rectangular support, from its angular and force bin counts.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        _, n_a, _, n_f = self.bin_span(name)
        return n_a * n_f

    def share(self, name: str) -> float:
        """Fraction of the grid occupied by archive ``name``'s rectangle; no distribution within it is modeled.

        Equation: self.occupied(name) / self.cells
        """
        return self.occupied(name) / self.cells

    def empty(self, name: str) -> int:
        """Number of grid cells outside archive ``name``'s rectangle.

        Equation: self.cells - self.occupied(name)
        """
        return self.cells - self.occupied(name)

    def empty_share(self, name: str) -> float:
        """Fraction of grid cells outside the archive's declared rectangular support.

        Equation: self.empty(name) / self.cells
        """
        return self.empty(name) / self.cells

    def rms_error(self, name: str) -> Quantity:
        """Supplied RMS-error entry for archive ``name``; returns the input value without estimating or extrapolating error.

        Equation: self.archives[name][4]
        """
        return self.archives[name][4]


class ScenarioTableCoverageScenario(BaseScenario):
    """A scenario table of ``offsets`` by ``speeds`` joined against an archive demonstrated within
    ``logged_offset`` at ``demonstrated_speed``, plus recovery collections at ``added_offsets``.

    Notes
    -----
    Equation: Supported cells require the demonstrated speed and an offset in the logged interval or added set.

    Assumptions: Exact discrete matching, with no extrapolation between speed values.

    Excludes: A claim of safety or generalization outside demonstrated cells.

    References
    ----------
    Derivation: Support is an explicit set-membership rule on discrete offset/speed pairs: speed must match exactly, and offset must lie within the logged interval or the supplied added set. Enumerating the Cartesian table counts support; subtraction counts absence.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchScenarioTable
    >>> scenario.supported
    1
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    offsets: List[Quantity]
    speeds: List[Quantity]
    logged_offset: Quantity
    demonstrated_speed: Quantity
    added_offsets: List[Quantity] = []

    @property
    def cells(self) -> int:
        """Number of offset/speed combinations in the Cartesian scenario table.

        Equation: len(self.offsets) * len(self.speeds)
        """
        return len(self.offsets) * len(self.speeds)

    def supported_cell(self, offset: Quantity, speed: Quantity) -> bool:
        """Whether an offset/speed pair lies on the demonstrated speed and inside the logged offset interval or explicit added offsets.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        if speed != self.demonstrated_speed:
            return False
        return abs(offset) <= self.logged_offset or any(offset == a for a in self.added_offsets)

    @property
    def supported(self) -> int:
        """Count of table cells satisfying the exact support predicate; no interpolation across speed is assumed.

        Equation: sum((self.supported_cell(d, v) for d in self.offsets for v in self.speeds))
        """
        return sum(self.supported_cell(d, v) for d in self.offsets for v in self.speeds)

    @property
    def absences(self) -> int:
        """Number of unsupported combinations: total cells minus supported cells.

        Equation: self.cells - self.supported
        """
        return self.cells - self.supported


class GripperTransferScenario(BaseScenario):
    """A normalized gripper command from a donor gripper replayed on a target with a shorter stroke
    and a lower continuous force rating, against a part of ``part_width``.

    Notes
    -----
    Equation: Opening=``normalized_command*stroke``; clearance compares opening with part_width.

    Assumptions: A linear command-to-stroke map and fixed donor/target force specifications.

    Excludes: Contact-force prediction, compliant deformation, and transfer success beyond geometric clearance.

    References
    ----------
    Derivation: The assumed actuator map is linear: normalized opening command times stroke gives jaw opening. Comparing that opening with part width gives geometric clearance; comparing donor force with target rating is a separate specification ratio.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.GripperStrokeTransfer
    >>> scenario.target_clears
    False
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    command: float  # normalized opening command, 0 to 1
    donor_stroke: Quantity
    target_stroke: Quantity
    part_width: Quantity
    donor_hold_force: Quantity
    target_force_rating: Quantity

    @property
    def donor_opening(self) -> Quantity:
        """Donor opening in mm: normalized command times donor stroke.

        Equation: (self.command * self.donor_stroke).to('mm')
        """
        return (self.command * self.donor_stroke).to("mm")

    @property
    def target_opening(self) -> Quantity:
        """Target opening in mm under the same normalized command and its different stroke.

        Equation: (self.command * self.target_stroke).to('mm')
        """
        return (self.command * self.target_stroke).to("mm")

    @property
    def stroke_excess(self) -> Quantity:
        """Donor commanded opening minus the target's full available stroke, in mm; positive values exceed what the target can reproduce even fully open.

        Equation: (self.donor_opening - self.target_stroke).to('mm')
        """
        return (self.donor_opening - self.target_stroke).to("mm")

    @property
    def force_ratio(self) -> float:
        """Dimensionless ratio of donor hold force to target rating; this comparison does not predict contact force.

        Equation: (self.donor_hold_force / self.target_force_rating).to('').magnitude
        """
        return (self.donor_hold_force / self.target_force_rating).to("").magnitude

    @property
    def donor_clears(self) -> bool:
        """Whether donor opening exceeds part width; equality does not count as clearance.

        Equation: self.donor_opening > self.part_width
        """
        return self.donor_opening > self.part_width

    @property
    def target_clears(self) -> bool:
        """Whether target opening exceeds part width under the transferred command.

        Equation: self.target_opening > self.part_width
        """
        return self.target_opening > self.part_width

    @property
    def result(self):
        """Collect the geometric and force-rating comparisons into ``GripperTransferResult``; opening fields are in mm, the rating ratio is dimensionless, and clearance fields are booleans.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...results import GripperTransferResult

        return GripperTransferResult(
            donor_opening=self.donor_opening,
            target_opening=self.target_opening,
            stroke_excess=self.stroke_excess,
            force_ratio=self.force_ratio,
            donor_clears=self.donor_clears,
            target_clears=self.target_clears,
        )
