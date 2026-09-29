"""Embodied AI, cyber-physical machines, and robotics platform types."""

import math
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.units import ureg
from ..core.types import Quantity, Metadata, require_dimensionality
from ..hardware.types import HardwareNode
from ..sensors.types import CameraSensor, LiDARSensor, IMUSensor, SafetyLaserScanner
from ..models.types import EmbodiedWorkload
from ..actuators.types import WheelHubMotor


class EmbodiedPlatform(BaseModel):
    """Vetted cyber-physical machine platform (quadruped, humanoid, manipulator, drone, AMR, AV)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    archetype: str  # Class 1: Mobility, Class 2: Manipulation, Class 3: Process, Class 4: Humanoid
    mass: Quantity
    payload_capacity: Optional[Quantity] = None
    max_velocity: Optional[Quantity] = None
    max_acceleration: Optional[Quantity] = None
    max_jerk: Optional[Quantity] = None
    max_torque: Optional[Quantity] = None
    dofs: Optional[int] = None
    control_frequency: Quantity
    nominal_power: Optional[Quantity] = None
    compute_soc: Optional[HardwareNode] = None
    transmission: Optional[str] = None
    # Serial-arm kinematics and interface limits (manipulators only; None otherwise).
    # upper_arm_length / forearm_length are the shoulder-to-elbow and elbow-to-wrist
    # offsets of the kinematic chain (Denavit-Hartenberg d3 and d5 on a Panda).
    upper_arm_length: Optional[Quantity] = None
    forearm_length: Optional[Quantity] = None
    joint_velocity_limits: Optional[tuple[Quantity, ...]] = None      # per joint, base outward
    joint_acceleration_limits: Optional[tuple[Quantity, ...]] = None
    joint_jerk_limits: Optional[tuple[Quantity, ...]] = None
    joint_torque_rate_limit: Optional[Quantity] = None                # same on every joint
    cartesian_velocity_limit: Optional[Quantity] = None               # commanded TCP translation
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("upper_arm_length", "forearm_length", mode="after")
    @classmethod
    def _validate_link_length(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter, info.field_name)

    @field_validator("joint_velocity_limits", "joint_acceleration_limits", "joint_jerk_limits", mode="after")
    @classmethod
    def _validate_joint_limits(cls, v, info):
        if v is None:
            return v
        unit = {
            "joint_velocity_limits": ureg.radian / ureg.second,
            "joint_acceleration_limits": ureg.radian / ureg.second**2,
            "joint_jerk_limits": ureg.radian / ureg.second**3,
        }[info.field_name]
        return tuple(require_dimensionality(q, unit, info.field_name) for q in v)

    @field_validator("joint_torque_rate_limit", mode="after")
    @classmethod
    def _validate_torque_rate(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.newton * ureg.meter / ureg.second, "joint_torque_rate_limit")

    @field_validator("cartesian_velocity_limit", mode="after")
    @classmethod
    def _validate_cartesian_velocity(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter / ureg.second, "cartesian_velocity_limit")

    @field_validator("mass", "payload_capacity", mode="after")
    @classmethod
    def _validate_mass(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.kg, info.field_name)

    @field_validator("max_velocity", mode="after")
    @classmethod
    def _validate_velocity(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter / ureg.second, "max_velocity")

    @field_validator("max_acceleration", mode="after")
    @classmethod
    def _validate_acceleration(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter / ureg.second**2, "max_acceleration")

    @field_validator("control_frequency", mode="after")
    @classmethod
    def _validate_frequency(cls, v):
        return require_dimensionality(v, 1 / ureg.second, "control_frequency")

    @field_validator("nominal_power", mode="after")
    @classmethod
    def _validate_power(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.watt, "nominal_power")




# ---------------------------------------------------------------------------
# Composite machine and site scenario (added 2026-09-21, add-only).
# A mobile manipulator is two platforms, so EmbodiedPlatform is not extended.
# ---------------------------------------------------------------------------

_HZ = 1 / ureg.second
_MPS = ureg.meter / ureg.second
_MPS2 = ureg.meter / ureg.second**2


class MobileManipulatorPlatform(BaseModel):
    """Class 1 base carrying a Class 2 arm, with an application processor and a separate permission MCU."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    archetype: str
    base: EmbodiedPlatform
    arm: EmbodiedPlatform
    onboard_payload_capacity: Quantity      # kg
    footprint_width: Quantity               # m
    brain_soc: HardwareNode
    permission_mcu: HardwareNode
    permission_rate: Quantity               # Hz
    current_loop_rate: Quantity             # Hz
    fieldbus_cycle: Quantity                # ms
    servo_axes: int
    nav_camera: CameraSensor
    wrist_camera: CameraSensor
    lidar: LiDARSensor
    imu: IMUSensor
    # Certified protective-field sensor for persons in the path (ISO 3691-4);
    # separate from the perception LiDAR, which carries no safety rating.
    safety_scanner: Optional[SafetyLaserScanner] = None
    intent_model: EmbodiedWorkload
    chunk_model: EmbodiedWorkload
    intent_rate: Quantity                   # Hz
    # Intent-model deployment. Its inference latency is not stored: the
    # intent_inference_latency property computes it from these fields with
    # calc_vla_inference_latency (compute-bound prefill plus bandwidth-bound
    # decode), so the rate can be checked against it.
    intent_weight_bytes_per_param: Quantity  # byte/param, stored weight precision
    intent_input_tokens: int                # image-patch plus prompt tokens per evaluation
    intent_parallel_decode: bool            # True: all action tokens in one pass (OFT)
    intent_compute_utilization: float       # achieved fraction of the dense 16-bit peak
    intent_reference_latency: Quantity      # ms, published measurement of this deployment
    chunk_rate: Quantity                    # Hz
    chunk_inference_latency: Quantity       # ms
    chunk_step: Quantity                    # ms
    chunk_horizon: int
    dram_efficiency: float
    tcp_speed_limit: Quantity               # m/s
    # 24 V control rail shared by the application processor and the permission MCU
    control_rail_nominal: Quantity          # V
    control_rail_battery_low: Quantity      # V, battery at the low end of its discharge
    control_rail_resistance: Quantity       # ohm, shared harness resistance
    control_rail_dropout: Quantity          # V, point-of-load regulator dropout
    # Permission-path rail: isolated from the application processor's supply and
    # held up by a local supercapacitor store behind an ideal-diode controller. It
    # feeds the safety MCU, the drive-logic and gate-driver supplies, the encoder
    # interfaces, and every spring-brake coil. The permission loads are assumed to
    # share the control rail's regulator dropout.
    permission_rail_holdup: Quantity        # s, ride-through time after the feed is lost
    permission_rail_load: Quantity          # W, total load on the permission-path rail
    spring_brake_engage_min: Quantity       # ms, spring-brake engage time after coil decay (fast end)
    spring_brake_engage_max: Quantity       # ms, spring-brake engage time after coil decay (slow end)
    # Braking drivetrain (2026-09-28, M6). The two driven wheels are direct-drive
    # hub motors; casters carry the rest of the weight. Geometry fields that no
    # source supplies stay None ("needs source") rather than being estimated.
    drive_unit: Optional[WheelHubMotor] = None
    drive_wheel_count: int = 2
    drive_current_limit: Optional[Quantity] = None      # A, configured drive current limit
    drive_wheel_load_share: Optional[float] = None      # needs source: weight fraction on driven wheels
    cg_height: Optional[Quantity] = None                # needs source: loaded CoG height
    tip_arm_stowed: Optional[Quantity] = None           # needs source: CoG to leading support, arm stowed
    # 48 V drive bus (2026-09-28, M6): battery pack, source path, the 48-to-24 V
    # converter's undervoltage lockout, and the regenerative clamp and switch rating.
    drive_bus_full_charge: Optional[Quantity] = None    # V
    drive_bus_nominal: Optional[Quantity] = None        # V
    drive_bus_battery_low: Optional[Quantity] = None    # V
    drive_bus_resistance: Optional[Quantity] = None     # ohm, cells + BMS switches + fuse + harness
    drive_bus_uvlo: Optional[Quantity] = None           # V, converter input turn-off
    drive_bus_converter_max: Optional[Quantity] = None  # V, converter continuous input maximum
    drive_bus_converter_efficiency: Optional[float] = None
    drive_bus_clamp: Optional[Quantity] = None          # V, brake-chopper clamp threshold
    drive_bus_switch_rating: Optional[Quantity] = None  # V, inverter switch voltage rating
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator(
        "drive_current_limit", mode="after",
    )
    @classmethod
    def _validate_drive_current(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.ampere, info.field_name)

    @field_validator("cg_height", "tip_arm_stowed", mode="after")
    @classmethod
    def _validate_drive_geometry(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter, info.field_name)

    @field_validator(
        "drive_bus_full_charge", "drive_bus_nominal", "drive_bus_battery_low",
        "drive_bus_uvlo", "drive_bus_converter_max", "drive_bus_clamp",
        "drive_bus_switch_rating", mode="after",
    )
    @classmethod
    def _validate_drive_bus_voltage(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.volt, info.field_name)

    @field_validator("drive_bus_resistance", mode="after")
    @classmethod
    def _validate_drive_bus_resistance(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.ohm, info.field_name)

    @field_validator("drive_wheel_load_share", "drive_bus_converter_efficiency", mode="after")
    @classmethod
    def _validate_fraction(cls, v, info):
        if v is None:
            return v
        if not 0.0 < v <= 1.0:
            raise ValueError(f"{info.field_name} must be in (0, 1]; got {v}")
        return v

    @model_validator(mode="after")
    def _drive_bus_ordering(self):
        # 2026-09-28 (M6): the clamp must sit above full charge (or it fires while
        # charging), below the converter's continuous input maximum, and the
        # switches need at least 1.5x derating over the clamp.
        vals = (self.drive_bus_battery_low, self.drive_bus_nominal, self.drive_bus_full_charge,
                self.drive_bus_clamp, self.drive_bus_switch_rating)
        if any(x is None for x in vals):
            return self
        low, nom, full, clamp, switch = vals
        if not (low < nom < full < clamp):
            raise ValueError("drive bus must satisfy battery_low < nominal < full_charge < clamp")
        if self.drive_bus_converter_max is not None and clamp > self.drive_bus_converter_max:
            raise ValueError("drive_bus_clamp exceeds the converter's continuous input maximum")
        if switch < 1.5 * clamp:
            raise ValueError("drive_bus_switch_rating must be at least 1.5x the clamp")
        if self.drive_bus_uvlo is not None and not self.drive_bus_uvlo < low:
            raise ValueError("drive_bus_uvlo must sit below the battery-low voltage")
        return self

    @field_validator("onboard_payload_capacity", mode="after")
    @classmethod
    def _validate_mass(cls, v, info):
        return require_dimensionality(v, ureg.kg, info.field_name)

    @field_validator("footprint_width", mode="after")
    @classmethod
    def _validate_length(cls, v, info):
        return require_dimensionality(v, ureg.meter, info.field_name)

    @field_validator("permission_rate", "current_loop_rate", "intent_rate", "chunk_rate", mode="after")
    @classmethod
    def _validate_rate(cls, v, info):
        return require_dimensionality(v, _HZ, info.field_name)

    @field_validator(
        "fieldbus_cycle", "intent_reference_latency", "chunk_inference_latency", "chunk_step",
        mode="after",
    )
    @classmethod
    def _validate_time(cls, v, info):
        return require_dimensionality(v, ureg.second, info.field_name)

    @field_validator("tcp_speed_limit", mode="after")
    @classmethod
    def _validate_speed(cls, v, info):
        return require_dimensionality(v, _MPS, info.field_name)

    @field_validator("servo_axes", "chunk_horizon", mode="after")
    @classmethod
    def _validate_count(cls, v, info):
        if v < 1:
            raise ValueError(f"{info.field_name} must be at least 1; got {v}")
        return v

    @field_validator("dram_efficiency", mode="after")
    @classmethod
    def _validate_efficiency(cls, v):
        if not 0.0 < v <= 1.0:
            raise ValueError(f"dram_efficiency must be in (0, 1]; got {v}")
        return v

    @field_validator("intent_compute_utilization", mode="after")
    @classmethod
    def _validate_utilization(cls, v):
        if not 0.0 < v <= 1.0:
            raise ValueError(f"intent_compute_utilization must be in (0, 1]; got {v}")
        return v

    @field_validator("intent_weight_bytes_per_param", mode="after")
    @classmethod
    def _validate_bytes_per_param(cls, v, info):
        return require_dimensionality(v, ureg.byte / ureg.param, info.field_name)

    @field_validator("intent_input_tokens", mode="after")
    @classmethod
    def _validate_tokens(cls, v, info):
        if v < 1:
            raise ValueError(f"{info.field_name} must be at least 1; got {v}")
        return v

    @field_validator(
        "control_rail_nominal", "control_rail_battery_low", "control_rail_dropout",
        mode="after",
    )
    @classmethod
    def _validate_voltage(cls, v, info):
        return require_dimensionality(v, ureg.volt, info.field_name)

    @field_validator("control_rail_resistance", mode="after")
    @classmethod
    def _validate_resistance(cls, v, info):
        return require_dimensionality(v, ureg.ohm, info.field_name)

    @field_validator(
        "permission_rail_holdup", "spring_brake_engage_min", "spring_brake_engage_max",
        mode="after",
    )
    @classmethod
    def _validate_permission_rail_time(cls, v, info):
        return require_dimensionality(v, ureg.second, info.field_name)

    @field_validator("permission_rail_load", mode="after")
    @classmethod
    def _validate_permission_rail_power(cls, v, info):
        return require_dimensionality(v, ureg.watt, info.field_name)

    @model_validator(mode="after")
    def _intent_rate_fits_latency(self):
        # 2026-09-28 (M2): the stated intent rate must be achievable. One
        # evaluation per period, not pipelined, so the period must cover the
        # computed latency; a 5 Hz / 160 ms stipulation had priced only the
        # decode weight sweep.
        period = (1 / self.intent_rate).to(ureg.millisecond)
        if period < self.intent_inference_latency:
            raise ValueError(
                f"intent_rate {self.intent_rate:~P} gives a {period:~P} period, shorter than "
                f"the computed intent latency {self.intent_inference_latency:~P}"
            )
        return self

    @model_validator(mode="after")
    def _chunk_renewal_on_setpoint_grid(self):
        # 2026-09-28 (M5): the chunk policy renews on a whole number of setpoint
        # steps, finishes within its renewal period, and supplies enough
        # setpoints to bridge one renewal (calc_chunk_renewal_schedule raises
        # otherwise). A 50 ms renewal over 20 ms steps (2.5 steps) broke the
        # temporal-ensembling index.
        self.chunk_schedule  # noqa: B018 -- evaluated for its validation
        return self

    @property
    def chunk_schedule(self):
        """Renewal timing of the chunk policy (see ``calc_chunk_renewal_schedule``)."""
        from ..physics.robotics import calc_chunk_renewal_schedule  # avoid import cycle

        return calc_chunk_renewal_schedule(
            self.chunk_rate, self.chunk_step, self.chunk_horizon, self.chunk_inference_latency,
        )

    @property
    def chunk_period(self):
        """Chunk-policy renewal period, 1 / chunk_rate."""
        return self.chunk_schedule["renewal_period"]

    @property
    def intent_weight_memory(self):
        """Stored intent-model weights at the deployed precision."""
        return (self.intent_model.parameters * self.intent_weight_bytes_per_param).to(ureg.GB)

    @property
    def intent_latency_breakdown(self):
        """Roofline terms of one fresh intent-model evaluation on the application processor.

        Prefill runs at the dense FP16 rate (weight-only quantization keeps 16-bit
        activations); decode sweeps the stored weights at the sustained bandwidth.
        See ``calc_vla_inference_latency``.
        """
        from ..physics.transformer import calc_vla_inference_latency  # avoid import cycle

        soc = self.brain_soc
        return calc_vla_inference_latency(
            self.intent_model.parameters,
            self.intent_weight_memory,
            self.intent_input_tokens,
            self.intent_model.action_dim,
            soc.compute.precision_flops["fp16"],
            soc.memory.bandwidth,
            bus_efficiency=self.dram_efficiency,
            compute_utilization=self.intent_compute_utilization,
            parallel_decode=self.intent_parallel_decode,
            n_layers=self.intent_model.layers,
            hidden_dim=self.intent_model.hidden_dim,
        )

    @property
    def intent_inference_latency(self):
        """Computed latency of one fresh intent-model evaluation (prefill plus decode)."""
        return self.intent_latency_breakdown["t_total"]

    @property
    def unloaded_mass(self):
        """Base plus arm, without the tote-rack payload."""
        return (self.base.mass + self.arm.mass).to(ureg.kg)

    @property
    def loaded_mass(self):
        """Unloaded mass plus the full tote-rack capacity (an upper bound)."""
        return (self.unloaded_mass + self.onboard_payload_capacity).to(ureg.kg)

    @property
    def arm_effective_contact_mass(self):
        """Robot-side effective mass at the TCP: half the arm's moving mass plus payload."""
        return (self.arm.mass / 2 + self.arm.payload_capacity).to(ureg.kg)

    @property
    def drive_wheel_load_share_max(self):
        """Largest weight fraction the driven wheels may carry within their load rating.

        The hub motors' datasheet load rating per pair bounds the static load on
        the two driven wheels, so the casters must carry the remainder of the
        loaded mass: phi <= rating / m_loaded (capped at 1).
        """
        if self.drive_unit is None:
            return None
        pairs = self.drive_wheel_count / 2
        share = (pairs * self.drive_unit.load_rating_per_pair / self.loaded_mass).to("dimensionless").magnitude
        return min(1.0, float(share))

    @property
    def permission_rail_holdup_capacitance(self):
        """Hold-up store sized to carry the permission load for the hold-up time.

        The store is charged to the depleted-battery voltage and may discharge to
        the regulator dropout: C = 2 P t / (V_low^2 - V_dropout^2).
        """
        return (2 * self.permission_rail_load * self.permission_rail_holdup
                / (self.control_rail_battery_low**2 - self.control_rail_dropout**2)).to(ureg.farad)


class EmbodiedSiteScenario(BaseModel):
    """Budget terms for one embodied machine at one site. Illustrative unless a field is sourced."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    machine: MobileManipulatorPlatform
    # stopping-budget terms
    delta_loc: Quantity
    delta_margin: Quantity
    eps_track: Quantity
    t_brake_onset: Quantity
    t_lease: Quantity
    t_bus: Quantity
    # navigation-camera age terms after readout
    t_transport: Quantity
    t_dma: Quantity
    t_isp: Quantity
    t_backbone: Quantity
    t_ipc: Quantity
    # permission-path liveness and enforcer timing
    heartbeat_rate: Quantity
    heartbeat_timeout: Quantity
    mcu_self_watchdog: Quantity
    enforcer_unloaded: Quantity
    enforcer_wcet: Quantity
    enforcer_deadline: Quantity
    # dispatch-to-admission terms after the proposer's inference (the total is
    # the t_dispatch_to_admission property)
    chunk_wire_bytes: Quantity            # serialized chunk with header and evidence epoch
    board_link_clock: Quantity            # board-link (SPI) clock, one bit per clock
    t_chunk_deserialize: Quantity
    t_plan_validate: Quantity             # whole-plan validation on the safety MCU
    t_plan_activate: Quantity
    # site geometry, people, and floor
    v_human_approach: Quantity
    d_clear: Quantity
    v_aisle: Quantity
    aisle_width: Quantity
    aisle_length: Quantity
    side_clearance: Quantity
    mu_dry: float
    mu_inspected_floor: float
    mu_oil_film: float
    # arm tasks: latch, conveyor grasp, coworker contact
    k_latch: Quantity
    v_latch_approach: Quantity
    v_latch_high: Quantity
    f_latch_tripwire: Quantity
    t_contact_response: Quantity
    f_latch_limit: Quantity
    v_conveyor: Quantity
    a_conveyor_slip: Quantity
    grasp_tolerance: Quantity
    grasp_initial_error: Quantity
    k_contact_human: Quantity             # ISO/TS 15066 Annex A hand effective spring constant
    m_contact_human: Quantity             # ISO/TS 15066 Annex A hand effective mass
    f_contact_criterion: Quantity
    # coworker handover and remote takeover (shared by more than one chapter)
    v_handover: Quantity
    # Remote-takeover response terms (2026-09-28, M4). Human terms are sourced;
    # the totals t_takeover_in_loop / t_takeover_out_of_loop are properties.
    t_human_perceptual: Quantity          # Model Human Processor, nominal
    t_human_perceptual_slow: Quantity     # Model Human Processor, slow bound
    t_human_cognitive: Quantity
    t_human_cognitive_slow: Quantity
    t_human_motor: Quantity
    t_human_motor_slow: Quantity
    t_brake_response_expected: Quantity   # driver perception-brake time, expected signal
    t_brake_response_unexpected: Quantity # unexpected but common signal
    t_brake_response_surprise: Quantity   # surprise event
    t_takeover_driving_p50: Quantity      # percentiles of condition-mean take-over time
    t_takeover_driving_mean: Quantity
    t_takeover_driving_p90: Quantity
    t_takeover_driving_p99: Quantity
    t_video_glass_to_glass: Quantity      # remote console video latency
    t_handshake_request: Quantity         # four-phase protocol budgets (commit is one tick)
    t_handshake_acknowledge: Quantity
    t_handshake_confirm: Quantity
    # The in-loop slow-down is derived (v_takeover_in_loop property): the
    # Delta_da-charged in-loop ceiling rounded down to this resolution.
    v_takeover_resolution: Quantity
    v_takeover_out_of_loop: Quantity
    # coincident current transient on the shared control rail (inference burst
    # plus drive motors climbing the cage-door threshold)
    i_rail_transient: Quantity
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator(
        "delta_loc", "delta_margin", "eps_track", "d_clear", "aisle_width", "aisle_length",
        "side_clearance", "grasp_tolerance", "grasp_initial_error",
        mode="after",
    )
    @classmethod
    def _validate_length(cls, v, info):
        return require_dimensionality(v, ureg.meter, info.field_name)

    @field_validator(
        "t_brake_onset", "t_lease", "t_bus", "t_transport", "t_dma", "t_isp", "t_backbone",
        "t_ipc", "heartbeat_timeout", "mcu_self_watchdog", "enforcer_unloaded",
        "enforcer_wcet", "enforcer_deadline", "t_contact_response",
        "t_human_perceptual", "t_human_perceptual_slow", "t_human_cognitive",
        "t_human_cognitive_slow", "t_human_motor", "t_human_motor_slow",
        "t_brake_response_expected", "t_brake_response_unexpected", "t_brake_response_surprise",
        "t_takeover_driving_p50", "t_takeover_driving_mean", "t_takeover_driving_p90",
        "t_takeover_driving_p99", "t_video_glass_to_glass", "t_handshake_request",
        "t_handshake_acknowledge", "t_handshake_confirm",
        "t_chunk_deserialize", "t_plan_validate", "t_plan_activate",
        mode="after",
    )
    @classmethod
    def _validate_time(cls, v, info):
        return require_dimensionality(v, ureg.second, info.field_name)

    @field_validator("chunk_wire_bytes", mode="after")
    @classmethod
    def _validate_bytes(cls, v, info):
        return require_dimensionality(v, ureg.byte, info.field_name)

    @field_validator("board_link_clock", mode="after")
    @classmethod
    def _validate_clock(cls, v, info):
        return require_dimensionality(v, _HZ, info.field_name)

    @field_validator("heartbeat_rate", mode="after")
    @classmethod
    def _validate_rate(cls, v, info):
        return require_dimensionality(v, _HZ, info.field_name)

    @field_validator(
        "v_human_approach", "v_aisle", "v_latch_approach", "v_latch_high", "v_conveyor",
        "v_handover", "v_takeover_resolution", "v_takeover_out_of_loop",
        mode="after",
    )
    @classmethod
    def _validate_speed(cls, v, info):
        return require_dimensionality(v, _MPS, info.field_name)

    @field_validator("a_conveyor_slip", mode="after")
    @classmethod
    def _validate_acceleration(cls, v, info):
        return require_dimensionality(v, _MPS2, info.field_name)

    @field_validator("k_latch", "k_contact_human", mode="after")
    @classmethod
    def _validate_stiffness(cls, v, info):
        return require_dimensionality(v, ureg.newton / ureg.meter, info.field_name)

    @field_validator("m_contact_human", mode="after")
    @classmethod
    def _validate_mass(cls, v, info):
        return require_dimensionality(v, ureg.kilogram, info.field_name)

    @field_validator("f_latch_tripwire", "f_latch_limit", "f_contact_criterion", mode="after")
    @classmethod
    def _validate_force(cls, v, info):
        return require_dimensionality(v, ureg.newton, info.field_name)

    @field_validator("mu_dry", "mu_inspected_floor", "mu_oil_film", mode="after")
    @classmethod
    def _validate_friction(cls, v, info):
        if not 0.0 < v < 1.0:
            raise ValueError(f"{info.field_name} must be in (0, 1); got {v}")
        return v

    @field_validator("i_rail_transient", mode="after")
    @classmethod
    def _validate_current(cls, v, info):
        return require_dimensionality(v, ureg.ampere, info.field_name)

    @property
    def a_brake(self):
        """Credible loaded deceleration on a dry floor: the base's registry value."""
        return self.machine.base.max_acceleration.to(ureg.meter / ureg.second**2)

    @property
    def t_tick(self):
        """Permission-loop period."""
        return (1 / self.machine.permission_rate).to(ureg.millisecond)

    @property
    def perception_age(self):
        """Navigation-camera observation age at dispatch (mid-exposure to the proposer)."""
        cam = self.machine.nav_camera
        return (cam.exposure_time / 2 + cam.readout_time + self.t_transport + self.t_dma
                + self.t_isp + self.t_backbone + self.t_ipc).to(ureg.millisecond)

    @property
    def tau_delay(self):
        """Total pre-brake delay: brake onset, lease, one tick, one bus cycle, and observation age."""
        return (self.t_brake_onset + self.t_lease + self.t_tick + self.t_bus
                + self.perception_age).to(ureg.millisecond)

    @property
    def t_handshake(self):
        """Four-phase takeover exchange: request, acknowledge, commit on one tick, confirm."""
        return (self.t_handshake_request + self.t_handshake_acknowledge + self.t_tick
                + self.t_handshake_confirm).to(ureg.millisecond)

    @property
    def t_takeover_transport(self):
        """Transport a remote takeover adds to the human response.

        The scene reaches the operator through the console video
        (glass-to-glass latency), and the operator's request returns through
        the four-phase exchange.
        """
        return (self.t_video_glass_to_glass + self.t_handshake).to(ureg.millisecond)

    @property
    def t_human_simple_reaction(self):
        """Model Human Processor simple reaction time: one perceptual, cognitive, and motor cycle."""
        return (self.t_human_perceptual + self.t_human_cognitive
                + self.t_human_motor).to(ureg.millisecond)

    @property
    def t_human_simple_reaction_slow(self):
        """Simple reaction time with every processor at the slow end of its range."""
        return (self.t_human_perceptual_slow + self.t_human_cognitive_slow
                + self.t_human_motor_slow).to(ureg.millisecond)

    @property
    def t_takeover_in_loop(self):
        """Design in-loop takeover: slow-bound simple reaction plus takeover transport.

        Holds only for an attending operator with hands on the controls who
        responds to the stimulus the task leads them to expect; an unexpected
        hazard takes the longer perception-brake times.
        """
        return (self.t_human_simple_reaction_slow + self.t_takeover_transport).to(ureg.millisecond)

    @property
    def t_takeover_out_of_loop(self):
        """Design out-of-loop takeover: 90th-percentile condition-mean take-over time plus transport."""
        return (self.t_takeover_driving_p90 + self.t_takeover_transport).to(ureg.second)

    @property
    def t_board_link(self):
        """Serialization of one chunk over the board link: 8 B / f_link."""
        bits = self.chunk_wire_bytes.to(ureg.bit).magnitude
        return (bits / self.board_link_clock).to(ureg.millisecond)

    @property
    def t_dispatch_to_admission(self):
        """Allocated dispatch-to-admission interval Delta_da of the discrete placement.

        Proposer inference charged in full after dispatch (conservative), board-link
        serialization, deserialization, whole-plan validation, activation, and the
        wait for the next permission tick. An allocation to qualify, not a
        measurement.

        Source: additive stage budget; stage allowances are the scenario's
        illustrative values, inference is the machine's registry latency.
        """
        return (self.machine.chunk_inference_latency + self.t_board_link
                + self.t_chunk_deserialize + self.t_plan_validate + self.t_plan_activate
                + self.t_tick).to(ureg.millisecond)

    @property
    def v_takeover_in_loop_ceiling(self):
        """Highest slow-down speed whose in-loop takeover, charged with Delta_da, still fits.

        S-curve ceiling (calc_scurve_max_permitted_velocity) with pre-brake delay
        tau_delay + T_takeover,in + Delta_da, the credible deceleration, the
        stipulated jerk bound of the reference stop, and the localization,
        tracking, and protective margins.
        """
        from ..physics.robotics import calc_c2_stop_suffix, calc_scurve_max_permitted_velocity

        j_brake = calc_c2_stop_suffix(self.machine.base.max_velocity, self.a_brake)["peak_jerk"]
        return calc_scurve_max_permitted_velocity(
            self.d_clear,
            self.tau_delay + self.t_takeover_in_loop + self.t_dispatch_to_admission,
            self.a_brake, j_brake, self.delta_loc + self.eps_track, self.delta_margin,
        ).to(_MPS)

    @property
    def v_takeover_in_loop(self):
        """Slow-down before an in-loop takeover request.

        The Delta_da-charged in-loop ceiling rounded down to v_takeover_resolution,
        so the chosen speed always sits below the ceiling by less than one step.
        """
        step = self.v_takeover_resolution.to(_MPS).magnitude
        ceiling = self.v_takeover_in_loop_ceiling.to(_MPS).magnitude
        return (math.floor(ceiling / step) * step) * _MPS

    @property
    def pre_trigger_window(self):
        """Pre-trigger logging window: the out-of-loop design takeover, rounded up to whole seconds.

        The buffer must reach back past the divergence that preceded the
        takeover, so it spans at least the design takeover interval.
        """
        return math.ceil(self.t_takeover_out_of_loop.to(ureg.second).magnitude) * ureg.second

    @property
    def fixed_overhead(self):
        """Localization bound, protective clearance, and tracking bound."""
        return (self.delta_loc + self.delta_margin + self.eps_track).to(ureg.meter)
