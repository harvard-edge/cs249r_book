"""Embodied AI, cyber-physical machines, and robotics platform types."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.units import ureg
from ..core.types import Quantity, Metadata, require_dimensionality
from ..hardware.types import HardwareNode
from ..sensors.types import CameraSensor, LiDARSensor, IMUSensor
from ..models.types import EmbodiedWorkload


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
    metadata: Metadata = Field(default_factory=Metadata)

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
    intent_model: EmbodiedWorkload
    chunk_model: EmbodiedWorkload
    intent_rate: Quantity                   # Hz
    intent_inference_latency: Quantity      # ms
    chunk_rate: Quantity                    # Hz
    chunk_inference_latency: Quantity       # ms
    chunk_step: Quantity                    # ms
    chunk_horizon: int
    dram_efficiency: float
    tcp_speed_limit: Quantity               # m/s
    metadata: Metadata = Field(default_factory=Metadata)

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
        "fieldbus_cycle", "intent_inference_latency", "chunk_inference_latency", "chunk_step",
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
    k_contact_human: Quantity
    f_contact_criterion: Quantity
    # coworker handover and remote takeover (shared by more than one chapter)
    v_handover: Quantity
    t_takeover_in_loop: Quantity
    t_takeover_out_of_loop: Quantity
    v_takeover_in_loop: Quantity
    v_takeover_out_of_loop: Quantity
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
        "t_takeover_in_loop", "t_takeover_out_of_loop",
        mode="after",
    )
    @classmethod
    def _validate_time(cls, v, info):
        return require_dimensionality(v, ureg.second, info.field_name)

    @field_validator("heartbeat_rate", mode="after")
    @classmethod
    def _validate_rate(cls, v, info):
        return require_dimensionality(v, _HZ, info.field_name)

    @field_validator(
        "v_human_approach", "v_aisle", "v_latch_approach", "v_latch_high", "v_conveyor",
        "v_handover", "v_takeover_in_loop", "v_takeover_out_of_loop",
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
    def fixed_overhead(self):
        """Localization bound, protective clearance, and tracking bound."""
        return (self.delta_loc + self.delta_margin + self.eps_track).to(ureg.meter)
