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
    cartesian_acceleration_limit: Optional[Quantity] = None           # commanded TCP translation
    emergency_stop_time: Optional[Quantity] = None                    # worst-case stop, request to rest
    # Standing center-of-mass height above the support surface (legged/bipedal
    # platforms; the inverted-pendulum length z_c). None where not applicable.
    com_height: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("com_height", mode="after")
    @classmethod
    def _validate_com_height(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter, "com_height")

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

    @field_validator("emergency_stop_time", mode="after")
    @classmethod
    def _validate_emergency_stop_time(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.second, "emergency_stop_time")

    @field_validator("cartesian_acceleration_limit", mode="after")
    @classmethod
    def _validate_cartesian_acceleration(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter / ureg.second**2, "cartesian_acceleration_limit")

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
        """Derive the chunk renewal schedule from the platform's timing fields.

        Returns
        -------
        dict
            Renewal period, chunk span, and open-loop span as time quantities;
            setpoints_per_renewal and min_supply_setpoints as counts. The helper
            rejects a renewal that misses its grid or lacks sufficient setpoints.

        See Also
        --------
        mlsysim.physics.robotics.calc_chunk_renewal_schedule : Schedule equations and validation.

        Notes
        -----
        Equation: renewal_period = 1/chunk_rate; chunk_span = ``chunk_horizon*chunk_step``;
            open_loop_span = renewal_period + chunk_inference_latency. The positive
            chunk_rate is in Hz, chunk_step and chunk_inference_latency are times,
            and chunk_horizon is an integer setpoint count.
        Assumptions: periodic renewal, constant setpoint spacing, and inference
            completing within the period; renewal aligns to whole setpoint steps.
        Excludes: runtime jitter, missed transmissions, and a guarantee that the
            commanded trajectory remains dynamically feasible.

        References
        ----------
        Derivation: reciprocal frequency gives renewal time; multiplying setpoint
            spacing by count gives supply duration. The referenced helper checks
            whether that supply covers the scheduled replacement interval.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> machine.chunk_schedule['min_supply_setpoints'] <= machine.chunk_horizon
        True
        """
        from ..physics.robotics import calc_chunk_renewal_schedule  # avoid import cycle

        return calc_chunk_renewal_schedule(
            self.chunk_rate, self.chunk_step, self.chunk_horizon, self.chunk_inference_latency,
        )

    @property
    def chunk_period(self):
        """Return the interval between scheduled chunk-policy renewals.

        Returns
        -------
        Quantity
            Renewal period as a time quantity returned by chunk_schedule.

        See Also
        --------
        MobileManipulatorPlatform.chunk_schedule : Complete renewal contract.

        Notes
        -----
        Equation: chunk_period = 1/chunk_rate, where chunk_rate is positive Hz.
        Assumptions: the periodic setpoint schedule passes chunk_schedule validation.
        Excludes: realized inference jitter, dropped renewals, and transport delay.

        References
        ----------
        Derivation: a periodic process at f cycles per second has period 1/f seconds.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> bool(machine.chunk_period * machine.chunk_rate == 1)
        True
        """
        return self.chunk_schedule["renewal_period"]

    @property
    def intent_weight_memory(self):
        """Count stored intent-model weight bytes at deployment precision.

        Returns
        -------
        Quantity
            Weight storage in decimal GB, excluding activation and KV state.

        Notes
        -----
        Equation: intent_weight_memory = intent_model.parameters *
            intent_weight_bytes_per_param. Parameter count is in param and deployed
            element storage is in byte/param; both must be positive.
        Assumptions: all parameters are resident and stored at one uniform precision.
        Excludes: quantization scales, alignment, allocator reserve, KV cache,
            activations, and runtime workspaces.

        References
        ----------
        Derivation: parameter count multiplied by bytes per parameter gives bytes;
            the stored-precision field is an input, not an inferred compression ratio.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> bool(machine.intent_weight_memory ==
        ...      machine.intent_model.parameters * machine.intent_weight_bytes_per_param)
        True
        """
        return (self.intent_model.parameters * self.intent_weight_bytes_per_param).to(ureg.GB)

    @property
    def intent_latency_breakdown(self):
        """Estimate one fresh intent evaluation's prefill and decode times.

        Returns
        -------
        dict
            Compute and memory timing terms, total latency, token counts, and
            throughput from calc_vla_inference_latency. Time values carry Pint
            units; these are modeled durations, not the reference measured latency.

        See Also
        --------
        mlsysim.physics.transformer.calc_vla_inference_latency : Phase equations and model exclusions.

        Notes
        -----
        Equation: delegate to calc_vla_inference_latency with model parameter count,
            deployed weight bytes, input token count, action dimension, layer count,
            and hidden width. Each phase compares its compute and bandwidth bounds;
            FP16 compute supply is multiplied by intent_compute_utilization and
            byte bandwidth by dram_efficiency. intent_parallel_decode selects
            parallel action-token prefill instead of sequential decode sweeps.
        Assumptions: positive work/rates; efficiencies in (0, 1]; weight-only
            quantization retains FP16 arithmetic. An explicit fp16 compute rate takes
            precedence over the processor's canonical peak_flops. The referenced helper
            defines attention work and the first-token convention.
        Excludes: vision-encoder work outside supplied input tokens, runtime
            workspace admission, scheduling jitter, and action-quality prediction.

        References
        ----------
        Derivation: the platform supplies declared workload and processor fields to
            the linked VLA phase model. It does not calibrate utilization from the
            separate intent_reference_latency field.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> from mlsysim import Q_
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> machine.intent_latency_breakdown['t_total'] > Q_(0, 's')
        True
        """
        from ..physics.transformer import calc_vla_inference_latency  # avoid import cycle

        soc = self.brain_soc
        return calc_vla_inference_latency(
            self.intent_model.parameters,
            self.intent_weight_memory,
            self.intent_input_tokens,
            self.intent_model.action_dim,
            soc.compute.precision_flops.get("fp16", soc.compute.peak_flops),
            soc.memory.bandwidth,
            bus_efficiency=self.dram_efficiency,
            compute_utilization=self.intent_compute_utilization,
            parallel_decode=self.intent_parallel_decode,
            n_layers=self.intent_model.layers,
            hidden_dim=self.intent_model.hidden_dim,
        )

    @property
    def intent_inference_latency(self):
        """Return total modeled time for one fresh intent-model evaluation.

        Returns
        -------
        Quantity
            Prefill-plus-decode time from the VLA timing model, with time units.

        See Also
        --------
        MobileManipulatorPlatform.intent_latency_breakdown : Timing model inputs and exclusions.

        Notes
        -----
        Equation: intent_inference_latency = intent_latency_breakdown['t_total'];
            total time sums prefill and decode under the configured execution mode.
        Assumptions: the workload, precision, and efficiencies of the linked
            breakdown apply to this evaluation, with no pipelining across requests.
        Excludes: replacing the modeled total with intent_reference_latency,
            external sensor-processing stages, or a measured runtime guarantee.

        References
        ----------
        Derivation: select the total from the same phase calculation used for
            validation, avoiding a separately stored latency that could diverge.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> bool(machine.intent_inference_latency * machine.intent_rate <= 1)
        True
        """
        return self.intent_latency_breakdown["t_total"]

    @property
    def unloaded_mass(self):
        """Add the base and arm masses before onboard payload is loaded.

        Returns
        -------
        Quantity
            Combined base and arm mass in kg.

        Notes
        -----
        Equation: unloaded_mass = base.mass + arm.mass; both masses are in kg.
        Assumptions: the base and arm entries contain disjoint installed masses.
        Excludes: onboard payload and equipment not represented in either entry.

        References
        ----------
        Derivation: mass is additive for the two non-overlapping components.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> machine.unloaded_mass == machine.base.mass + machine.arm.mass
        True
        """
        return (self.base.mass + self.arm.mass).to(ureg.kg)

    @property
    def loaded_mass(self):
        """Add the full onboard payload capacity to the unloaded machine mass.

        Returns
        -------
        Quantity
            Fully loaded mass in kg under the stated component inventory.

        See Also
        --------
        MobileManipulatorPlatform.unloaded_mass : Base-plus-arm mass.

        Notes
        -----
        Equation: loaded_mass = unloaded_mass + onboard_payload_capacity; payload
            capacity is a nonnegative mass, not the actual payload of a particular run.
        Assumptions: capacity is fully used and each component is counted once.
        Excludes: overload, separate unlisted attachments, and payload distribution.

        References
        ----------
        Derivation: adding the full allowed payload to the component sum gives the
            upper load case for this inventory.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> machine.loaded_mass - machine.unloaded_mass == machine.onboard_payload_capacity
        True
        """
        return (self.unloaded_mass + self.onboard_payload_capacity).to(ureg.kg)

    @property
    def arm_effective_contact_mass(self):
        """Apply the declared half-arm-plus-payload contact-mass approximation.

        Returns
        -------
        Quantity
            Approximate robot-side contact mass at the tool center point, in kg.

        Notes
        -----
        Equation: arm_effective_contact_mass = arm.mass/2 + arm.payload_capacity;
            both required inputs are masses in kg.
        Assumptions: half of the stated arm mass and the full payload capacity
            participate in contact. This is a fixed modeling convention, not a
            configuration-dependent operational-space inertia calculation.
        Excludes: joint posture, contact direction, controller impedance, reflected
            drivetrain inertia, and proof of a safe impact force.

        References
        ----------
        Derivation: the implementation assigns a one-half participation factor to
            arm mass and full participation to payload. That factor is an assumed
            approximation rather than a consequence of general rigid-body mechanics.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> machine.arm_effective_contact_mass == machine.arm.mass/2 + machine.arm.payload_capacity
        True
        """
        return (self.arm.mass / 2 + self.arm.payload_capacity).to(ureg.kg)

    @property
    def drive_wheel_load_share_max(self):
        """Bound the static mass share supported by the installed drive wheels.

        Returns
        -------
        float or None
            Dimensionless supported-mass fraction capped at one, or None when no
            drive_unit is specified. This is a rating constraint, not actual axle load.

        Notes
        -----
        Equation: share = min(1, (drive_wheel_count/2) *
            drive_unit.load_rating_per_pair / loaded_mass). Load rating and loaded
            mass are positive masses in kg; wheel count is a positive integer.
        Assumptions: each installed pair has the same published static load rating,
            and remaining machine load can be carried by the casters.
        Excludes: uneven wheel loading, transient load transfer, tire friction,
            suspension dynamics, and measurement of the actual drive-wheel share.

        References
        ----------
        Derivation: multiply the per-pair rated mass by pair count, divide by total
            loaded mass, and cap the admissible fraction at the whole vehicle mass.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> 0 < machine.drive_wheel_load_share_max <= 1
        True
        """
        if self.drive_unit is None:
            return None
        pairs = self.drive_wheel_count / 2
        share = (pairs * self.drive_unit.load_rating_per_pair / self.loaded_mass).to("dimensionless").magnitude
        return min(1.0, float(share))

    @property
    def permission_rail_holdup_capacitance(self):
        """Size an ideal capacitor for the declared permission-rail ride-through.

        Returns
        -------
        Quantity
            Required ideal hold-up capacitance in farads.

        Notes
        -----
        Equation: C = ``2*permission_rail_load*permission_rail_holdup`` /
            ``(control_rail_battery_low**2`` - ``control_rail_dropout**2)``. Load is watts,
            hold-up is seconds, voltages are volts, and low voltage must exceed
            regulator dropout for usable positive stored energy.
        Assumptions: constant power, lossless energy delivery, initial voltage equal
            to depleted-battery voltage, and discharge ending exactly at dropout.
        Excludes: capacitor ESR, tolerance, aging, converter loss, leakage, wiring
            drops, and qualification of the permission-path electrical design.

        References
        ----------
        Derivation: capacitor energy released between two voltages is
            ``C*(V_initial**2`` - ``V_final**2)/2``; equate it to required energy ``P*t``.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> released = machine.permission_rail_holdup_capacitance * (
        ...     machine.control_rail_battery_low**2 - machine.control_rail_dropout**2)/2
        >>> abs((released/machine.permission_rail_holdup_energy).m_as('') - 1) < 1e-12
        True
        """
        return (2 * self.permission_rail_load * self.permission_rail_holdup
                / (self.control_rail_battery_low**2 - self.control_rail_dropout**2)).to(ureg.farad)

    @property
    def permission_rail_holdup_energy(self):
        """Compute permission-rail energy demand over the configured ride-through.

        Returns
        -------
        Quantity
            Energy demand in joules, excluding conversion and storage losses.

        Notes
        -----
        Equation: E = permission_rail_load * permission_rail_holdup, using positive
            load power in watts and hold-up duration in seconds.
        Assumptions: constant total permission-path load throughout the interval.
        Excludes: load transients, reserve margin, converter loss, and storage leakage.

        References
        ----------
        Derivation: integrating constant power P for time t gives energy ``P*t``.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> machine = Embodied.MobileManipulator.WarehouseMobileManipulator
        >>> machine.permission_rail_holdup_energy == machine.permission_rail_load * machine.permission_rail_holdup
        True
        """
        return (self.permission_rail_load * self.permission_rail_holdup).to(ureg.joule)


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
    # Added 2026-10-20: the base's yaw rate turning into the aisle, and the bound
    # on converting a capture time into the safety MCU's clock.
    yaw_rate_turn: Quantity
    clock_conversion_bound: Quantity
    # permission-path liveness and enforcer timing
    heartbeat_rate: Quantity
    heartbeat_timeout: Quantity
    mcu_self_watchdog: Quantity
    enforcer_unloaded: Quantity
    enforcer_wcet: Quantity
    enforcer_deadline: Quantity
    enforcer_blocking: Quantity           # longest tick-masking section in lower-priority code (B)
    enforcer_interference: Quantity       # timer and bus-interrupt work per response window (I)
    # dispatch-to-admission terms after the proposer's inference (the total is
    # the t_dispatch_to_admission property)
    chunk_wire_bytes: Quantity            # serialized chunk with header and evidence epoch
    board_link_clock: Quantity            # board-link (SPI) clock, one bit per clock
    t_chunk_deserialize: Quantity
    t_plan_validate: Quantity             # aggregate host geometry and local MCU admission allowance
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
        "enforcer_wcet", "enforcer_deadline", "enforcer_blocking", "enforcer_interference", "t_contact_response",
        "t_human_perceptual", "t_human_perceptual_slow", "t_human_cognitive",
        "t_human_cognitive_slow", "t_human_motor", "t_human_motor_slow",
        "t_brake_response_expected", "t_brake_response_unexpected", "t_brake_response_surprise",
        "t_takeover_driving_p50", "t_takeover_driving_mean", "t_takeover_driving_p90",
        "t_takeover_driving_p99", "t_video_glass_to_glass", "t_handshake_request",
        "t_handshake_acknowledge", "t_handshake_confirm",
        "t_chunk_deserialize", "t_plan_validate", "t_plan_activate", "clock_conversion_bound",
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
        """Read the scenario braking deceleration from its base specification.

        Returns
        -------
        Quantity
            Braking-deceleration magnitude in m/s^2.

        Notes
        -----
        Equation: a_brake = machine.base.max_acceleration, converted to m/s^2.

        Assumptions: The base specification is deliberately reused as a loaded dry-floor
        deceleration assumption. The property does not establish that a particular floor
        or load attains it.

        Excludes: Independent braking tests, tire friction limits, center-of-gravity
        effects, and braking-ramp dynamics.

        References
        ----------
        Derivation: This is a unit-preserving projection of the scenario's base
        specification, not a new empirical estimate.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.a_brake == site.machine.base.max_acceleration)
        True
        """
        return self.machine.base.max_acceleration.to(ureg.meter / ureg.second**2)

    @property
    def t_tick(self):
        """Convert permission-controller frequency to its period.

        Returns
        -------
        Quantity
            Permission-loop period in milliseconds.

        Notes
        -----
        Equation: t_tick = 1 / machine.permission_rate.

        Assumptions: The permission rate is positive and periodic; one whole period is
        used where a worst-phase tick wait is budgeted.

        Excludes: Interrupt masking, execution time, and jitter beyond the nominal
        period.

        References
        ----------
        Derivation: Frequency is cycles per second; its reciprocal is seconds per cycle.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool((site.t_tick * site.machine.permission_rate).to("").magnitude == 1)
        True
        """
        return (1 / self.machine.permission_rate).to(ureg.millisecond)

    @property
    def perception_age(self):
        """Accumulate navigation-camera age from exposure midpoint to dispatch.

        Returns
        -------
        Quantity
            Observation age in milliseconds.

        Notes
        -----
        Equation: perception_age = exposure_time/2 + readout_time + t_transport + t_dma
        + t_isp + t_backbone + t_ipc. Camera times come from machine.nav_camera; the
        other terms are site stage allowances.

        Assumptions: The exposure midpoint represents capture time and all listed stages
        are serial. These stage allowances may be illustrative rather than measured.

        Excludes: Unlisted queueing, frame batching, variable exposure/readout, and
        overlap between stages.

        References
        ----------
        Derivation: Elapsed time along a serial path is the sum of its nonoverlapping
        stage durations, starting halfway through exposure.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.perception_age >= site.machine.nav_camera.exposure_time / 2)
        True
        """
        cam = self.machine.nav_camera
        return (cam.exposure_time / 2 + cam.readout_time + self.t_transport + self.t_dma
                + self.t_isp + self.t_backbone + self.t_ipc).to(ureg.millisecond)

    @property
    def dispatch_age_limit(self):
        """Add clock-conversion uncertainty to the observation-age budget.

        Returns
        -------
        Quantity
            Dispatch age allowance in milliseconds.

        Notes
        -----
        Equation: dispatch_age_limit = perception_age + clock_conversion_bound.

        Assumptions: The clock-conversion bound is nonnegative and bounds the epoch
        conversion error; it is added in the unfavorable direction.

        Excludes: Clock faults outside the supplied bound, event reordering, and
        enforcement of this allowance on actual packets.

        References
        ----------
        Derivation: The largest age consistent with an additive absolute timestamp error
        equals the nominal age plus that error bound.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> from math import isclose
        >>> isclose((site.dispatch_age_limit - site.perception_age).to("ms").magnitude,
        ...         site.clock_conversion_bound.to("ms").magnitude)
        True
        """
        return (self.perception_age + self.clock_conversion_bound).to(ureg.millisecond)

    @field_validator("yaw_rate_turn", mode="after")
    @classmethod
    def _validate_yaw_rate(cls, v, info):
        return require_dimensionality(v, ureg.radian / ureg.second, info.field_name)

    @property
    def tau_delay(self):
        """Sum the legacy observation-to-brake delay budget.

        Returns
        -------
        Quantity
            Pre-brake delay in milliseconds, excluding dispatch-to-admission allocation.

        Notes
        -----
        Equation: tau_delay = t_brake_onset + t_lease + t_tick + t_bus + perception_age.

        Assumptions: A stalled proposer is noticed through lease expiry followed by one
        permission tick and one bus cycle; brake-onset delay follows. Terms are charged
        serially.

        Excludes: The dispatch-to-admission allocation Delta_da, actual brake travel,
        and timing overruns beyond the supplied allowances.

        References
        ----------
        Derivation: The expression adds the observation-age and lease-response
        intervals. The separate capture-to-brake ledger includes Delta_da explicitly.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.tau_delay >= site.perception_age)
        True
        """
        return (self.t_brake_onset + self.t_lease + self.t_tick + self.t_bus
                + self.perception_age).to(ureg.millisecond)

    @property
    def t_handshake(self):
        """Budget the four-phase remote takeover exchange.

        Returns
        -------
        Quantity
            Request-through-confirmation interval in milliseconds.

        Notes
        -----
        Equation: t_handshake = t_handshake_request + t_handshake_acknowledge + t_tick +
        t_handshake_confirm; the commit phase occupies one permission tick.

        Assumptions: Request, acknowledgment, commit, and confirmation execute in order
        with their specified allowances.

        Excludes: Lost-message retransmission, authentication delays not included in the
        fields, and network-tail guarantees.

        References
        ----------
        Derivation: Four serial protocol phases contribute additive elapsed time; the
        local commit is represented by the permission period.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.t_handshake >= site.t_tick)
        True
        """
        return (self.t_handshake_request + self.t_handshake_acknowledge + self.t_tick
                + self.t_handshake_confirm).to(ureg.millisecond)

    @property
    def t_takeover_transport(self):
        """Add video delivery and request-return transport for remote takeover.

        Returns
        -------
        Quantity
            Transport contribution to takeover in milliseconds.

        Notes
        -----
        Equation: t_takeover_transport = t_video_glass_to_glass + t_handshake.

        Assumptions: The operator first sees the scene through the video link and then
        returns a request through the four-phase handshake. The video field may be a
        reference measurement rather than a site-specific worst case.

        Excludes: Human reaction time, unmodeled video buffering, and a guarantee on
        wireless tail latency.

        References
        ----------
        Derivation: The scene-to-display and request-to-machine paths occur on opposite
        sides of the operator response, so their elapsed times add.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.t_takeover_transport >= site.t_video_glass_to_glass)
        True
        """
        return (self.t_video_glass_to_glass + self.t_handshake).to(ureg.millisecond)

    @property
    def t_human_simple_reaction(self):
        """Compose nominal perceptual, cognitive, and motor reaction stages.

        Returns
        -------
        Quantity
            Nominal simple-reaction estimate in milliseconds.

        Notes
        -----
        Equation: t_human_simple_reaction = t_human_perceptual + t_human_cognitive +
        t_human_motor.

        Assumptions: One serial cycle of each Model Human Processor stage represents an
        expected simple response to a known stimulus. Stage values retain the provenance
        of the scenario fields.

        Excludes: Decision complexity, surprise, divided attention, operator
        variability, and a worst-case response guarantee.

        References
        ----------
        Derivation: The property applies serial-stage accounting to the three supplied
        human-processing durations; it does not infer their values or a population
        quantile.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.t_human_simple_reaction >= site.t_human_motor)
        True
        """
        return (self.t_human_perceptual + self.t_human_cognitive
                + self.t_human_motor).to(ureg.millisecond)

    @property
    def t_human_simple_reaction_slow(self):
        """Compose the slow-end simple-reaction stage estimates.

        Returns
        -------
        Quantity
            Sum of slow perceptual, cognitive, and motor stage durations in milliseconds.

        Notes
        -----
        Equation: t_human_simple_reaction_slow = t_human_perceptual_slow +
        t_human_cognitive_slow + t_human_motor_slow.

        Assumptions: Each supplied stage is taken at the slow end of its reference range
        and the stages execute serially for an expected simple stimulus.

        Excludes: A joint statistical confidence bound, a guaranteed maximum human
        response, and responses to unexpected hazards.

        References
        ----------
        Derivation: Adding slow-end stage values constructs a conservative scenario
        relative to those stage choices; a sum of range endpoints is not a measured
        response quantile.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.t_human_simple_reaction_slow >= site.t_human_motor_slow)
        True
        """
        return (self.t_human_perceptual_slow + self.t_human_cognitive_slow
                + self.t_human_motor_slow).to(ureg.millisecond)

    @property
    def t_takeover_in_loop(self):
        """Combine attending-operator reaction and remote transport allowances.

        Returns
        -------
        Quantity
            Design in-loop takeover interval in milliseconds.

        Notes
        -----
        Equation: t_takeover_in_loop = t_human_simple_reaction_slow +
        t_takeover_transport.

        Assumptions: The operator is attending, has hands on the controls, and expects
        the task stimulus. The slow-end simple-reaction estimate and the transport
        allowance apply to the same event path.

        Excludes: Unexpected-hazard perception/braking time, disengaged operators, and
        certification of a response deadline.

        References
        ----------
        Derivation: The sequential path runs from scene delivery through human reaction
        to the completed takeover exchange; these durations add.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.t_takeover_in_loop >= site.t_takeover_transport)
        True
        """
        return (self.t_human_simple_reaction_slow + self.t_takeover_transport).to(ureg.millisecond)

    @property
    def t_takeover_out_of_loop(self):
        """Compose an out-of-loop takeover design interval from reference data.

        Returns
        -------
        Quantity
            Reference takeover duration plus transport, in seconds.

        Notes
        -----
        Equation: t_takeover_out_of_loop = t_takeover_driving_p90 +
        t_takeover_transport.

        Assumptions: The supplied driving reference is the 90th percentile of
        condition-mean takeover times, not the 90th percentile of individual operators.
        Applying it to this machine is a scenario assumption.

        Excludes: An individual-response probability guarantee, direct validation for
        this site, and unexpected delays beyond the transport field.

        References
        ----------
        Derivation: The property adds the selected condition-mean reference duration to
        remote transport; it does not transform the source statistic into a person-level
        quantile.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.t_takeover_out_of_loop >= site.t_takeover_driving_p90)
        True
        """
        return (self.t_takeover_driving_p90 + self.t_takeover_transport).to(ureg.second)

    @property
    def t_board_link(self):
        """Calculate serialized chunk transfer time on the board link.

        Returns
        -------
        Quantity
            Payload serialization time in milliseconds.

        Notes
        -----
        Equation: t_board_link = 8 * chunk_wire_bytes / board_link_clock when chunk size
        is expressed in bytes and clock in cycles per second.

        Assumptions: One bit transfers per clock and chunk_wire_bytes already includes
        the intended packet/header bytes. Board-link clock is positive.

        Excludes: Idle cycles, framing not included in chunk size, retransmission,
        contention, and deserialization.

        References
        ----------
        Derivation: Transfer duration is the bit count divided by bits per second;
        conversion from bytes supplies eight bits per byte.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.t_board_link * site.board_link_clock == site.chunk_wire_bytes.to("bit").magnitude)
        True
        """
        bits = self.chunk_wire_bytes.to(ureg.bit).magnitude
        return (bits / self.board_link_clock).to(ureg.millisecond)

    @property
    def t_dispatch_to_admission(self):
        """Allocate the discrete dispatch-to-admission processing path.

        Returns
        -------
        Quantity
            Delta_da allocation in milliseconds; this is not a measured latency.

        Notes
        -----
        Equation: Delta_da = machine.chunk_inference_latency + t_board_link +
        t_chunk_deserialize + t_plan_validate + t_plan_activate + t_tick.

        Assumptions: Inference is charged in full after dispatch, followed by
        serialization, deserialization, whole-plan validation, activation, and the next
        permission tick. The stage allowances are supplied by the scenario.

        Excludes: Proof of stage WCETs, runtime measurements, pipeline overlap, and
        overruns not represented by the fields.

        References
        ----------
        Derivation: Adding the declared serial stage budgets defines an allocation to
        qualify. The inference term comes from the machine; other allowances may be
        illustrative.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.t_dispatch_to_admission >= site.machine.chunk_inference_latency)
        True
        """
        return (self.machine.chunk_inference_latency + self.t_board_link
                + self.t_chunk_deserialize + self.t_plan_validate + self.t_plan_activate
                + self.t_tick).to(ureg.millisecond)

    # ── The capture-to-brake-onset event ledger (2026-09-29) ─────────────────
    # Four events (capture, dispatch, admission, brake onset) bound three
    # intervals. One interval, one owner: perception_age (capture to dispatch),
    # t_dispatch_to_admission (dispatch to admission), and
    # t_admission_to_brake_onset (admission to brake onset for a stalled
    # proposer). The replacement latency and the full pre-brake interval are
    # sums of those, never separate stage lists.

    @property
    def t_capture_to_admission(self):
        """Join capture-to-dispatch and dispatch-to-admission allocations.

        Returns
        -------
        Quantity
            Replacement latency allocation in milliseconds.

        Notes
        -----
        Equation: L = perception_age + t_dispatch_to_admission.

        Assumptions: Capture is represented by the exposure midpoint. Full inference
        inside Delta_da is charged after dispatch, so the vision backbone already
        present in perception_age is counted twice; the accounting is deliberately
        conservative.

        Excludes: A measured end-to-end latency, overlap recovery, and proof that
        supplied stage allocations hold on hardware.

        References
        ----------
        Derivation: The event ledger concatenates the capture-to-dispatch interval and
        Delta_da; the documented duplicate backbone charge is retained rather than
        silently canceled.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.t_capture_to_admission == site.perception_age + site.t_dispatch_to_admission)
        True
        """
        return (self.perception_age + self.t_dispatch_to_admission).to(ureg.millisecond)

    @property
    def t_admission_to_brake_onset(self):
        """Sum the stalled-proposer response after command admission.

        Returns
        -------
        Quantity
            Admission-to-brake-onset allocation in milliseconds.

        Notes
        -----
        Equation: t_admission_to_brake_onset = t_lease + t_tick + t_bus + t_brake_onset.

        Assumptions: The lease starts at admission. After it expires, response may wait
        one permission tick, one bus cycle, and the supplied brake-onset interval.

        Excludes: Capture age, dispatch-to-admission work, braking travel after onset,
        and unmodeled enforcement delays.

        References
        ----------
        Derivation: The lease-response stages occur serially after the admission event,
        so their duration sum is the corresponding event-ledger interval.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.t_admission_to_brake_onset >= site.t_lease)
        True
        """
        return (self.t_lease + self.t_tick + self.t_bus
                + self.t_brake_onset).to(ureg.millisecond)

    @property
    def t_capture_to_brake_onset(self):
        """Join the event-ledger intervals from capture through brake onset.

        Returns
        -------
        Quantity
            Full stalled-proposer pre-brake allocation in milliseconds.

        Notes
        -----
        Equation: t_capture_to_brake_onset = t_capture_to_admission +
        t_admission_to_brake_onset = tau_delay + t_dispatch_to_admission.

        Assumptions: The capture, dispatch, admission, and brake-onset events occur in
        order. Each ledger interval is charged once, including the conservative
        inference allocation inside capture-to-admission.

        Excludes: Braking distance after onset, measured execution guarantees, and a
        safety certification of the composed path.

        References
        ----------
        Derivation: Adjacent event intervals concatenate by addition; expanding their
        definitions gives the equality with tau_delay plus Delta_da.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.t_capture_to_brake_onset == site.tau_delay + site.t_dispatch_to_admission)
        True
        """
        return (self.t_capture_to_admission
                + self.t_admission_to_brake_onset).to(ureg.millisecond)

    @property
    def v_takeover_in_loop_ceiling(self):
        """Solve the delayed S-curve stopping model for an in-loop speed ceiling.

        Returns
        -------
        Quantity
            Modeled speed ceiling in meters per second; not a certified operating limit.

        See Also
        --------
        mlsysim.physics.robotics.stopping.calc_scurve_max_permitted_velocity : Delayed S-curve speed inverse.

        Notes
        -----
        Equation: Choose the largest v satisfying ``v*T`` + d_scurve(v, a_brake, j_brake) +
        delta_loc + eps_track + delta_margin <= d_clear, where T = tau_delay +
        t_takeover_in_loop + t_dispatch_to_admission.

        Assumptions: The clearance and error margins bound this scenario. Available
        deceleration is a_brake; jerk is stipulated from the reference C2 stop at the
        base drive-limit speed. The attending-operator assumptions of t_takeover_in_loop
        apply.

        Excludes: Validation of traction or jerk capability, moving-obstacle approach,
        operator-response guarantees, and enforcement on a deployed robot.

        References
        ----------
        Derivation: The linked S-curve inverse solves a monotone delayed
        stopping-distance inequality after adding the in-loop takeover and admission
        allocations.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.v_takeover_in_loop_ceiling >= site.v_takeover_in_loop)
        True
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
        """Round the modeled in-loop takeover speed ceiling down to a command step.

        Returns
        -------
        Quantity
            Quantized speed in meters per second, at or below the modeled ceiling.

        Notes
        -----
        Equation: v_takeover_in_loop = floor(v_takeover_in_loop_ceiling /
        v_takeover_resolution) * v_takeover_resolution.

        Assumptions: The command resolution is positive and the ceiling is nonnegative.
        Rounding preserves the upstream stopping-model assumptions rather than
        independently validating them.

        Excludes: Tracking overshoot, actuator quantization error beyond the chosen
        step, and certification of the commanded speed.

        References
        ----------
        Derivation: Floor division selects the largest integer multiple of the command
        resolution no greater than the continuous ceiling.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(0 <= (site.v_takeover_in_loop_ceiling - site.v_takeover_in_loop).to("m/s").magnitude < site.v_takeover_resolution.to("m/s").magnitude)
        True
        """
        step = self.v_takeover_resolution.to(_MPS).magnitude
        ceiling = self.v_takeover_in_loop_ceiling.to(_MPS).magnitude
        return (math.floor(ceiling / step) * step) * _MPS

    @property
    def pre_trigger_window(self):
        """Round the out-of-loop takeover interval up for pre-trigger logging.

        Returns
        -------
        Quantity
            Required logging-window allocation in whole seconds.

        Notes
        -----
        Equation: pre_trigger_window = ceil(t_takeover_out_of_loop expressed in seconds)
        * one second.

        Assumptions: The retained history should span at least the selected design
        takeover interval, and logging windows are allocated in whole seconds.

        Excludes: An assurance that all causal precursors lie inside the window, storage
        sizing, and a bound on individual human takeover time.

        References
        ----------
        Derivation: Ceiling to an integer number of seconds ensures rounding cannot
        shorten the reference takeover interval.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.pre_trigger_window >= site.t_takeover_out_of_loop)
        True
        """
        return math.ceil(self.t_takeover_out_of_loop.to(ureg.second).magnitude) * ureg.second

    @property
    def fixed_overhead(self):
        """Add localization, protective, and tracking distance margins.

        Returns
        -------
        Quantity
            Total fixed clearance deduction in meters.

        Notes
        -----
        Equation: fixed_overhead = delta_loc + delta_margin + eps_track.

        Assumptions: The three nonnegative distance allowances cover distinct
        contributions and are combined additively in the unfavorable direction.

        Excludes: Probabilistic cancellation, overlap correction, speed-dependent
        braking travel, and validation of the supplied bounds.

        References
        ----------
        Derivation: Adding absolute error and clearance budgets gives the
        worst-direction distance reserved before motion-dependent terms.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.fixed_overhead == site.delta_loc + site.delta_margin + site.eps_track)
        True
        """
        return (self.delta_loc + self.delta_margin + self.eps_track).to(ureg.meter)

    @property
    def j_brake(self):
        """Read the stipulated jerk ceiling from the reference C2 stopping profile.

        Returns
        -------
        Quantity
            Reference peak jerk in meters per second cubed.

        See Also
        --------
        mlsysim.physics.robotics.stopping.calc_c2_stop_suffix : Reference polynomial stop and peak jerk.

        Notes
        -----
        Equation: j_brake = calc_c2_stop_suffix(machine.base.max_velocity,
        a_brake)["peak_jerk"]. The reference profile supplies the jerk limit used by the
        resident S-curve model.

        Assumptions: The drive-limit speed and scenario deceleration define the C2
        polynomial reference. Its peak jerk is a stipulated design input, not a measured
        drivetrain capability.

        Excludes: Actuator identification, jerk tracking, traction changes, and
        qualification of the resulting resident stop.

        References
        ----------
        Derivation: The reference stopping-profile helper differentiates its polynomial
        acceleration to obtain peak jerk; this property forwards that value with
        explicit units.

        Examples
        --------
        >>> from mlsysim.embodied.registry import Embodied
        >>> site = Embodied.Scenario.WarehouseAisle
        >>> bool(site.j_brake.to("m/s^3").magnitude > 0)
        True
        """
        from ..physics.robotics import calc_c2_stop_suffix

        return calc_c2_stop_suffix(self.machine.base.max_velocity,
                                   self.a_brake)["peak_jerk"].to(ureg.meter / ureg.second**3)
