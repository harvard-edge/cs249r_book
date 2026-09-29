"""Actuator and joint transmission types for physical AI and robotics."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.units import ureg
from ..core.types import Quantity, Metadata, require_dimensionality


class ActuatorJoint(BaseModel):
    """Vetted robotic joint actuator (motor + transmission + driver)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    actuator_type: str  # HarmonicDrive, QuasiDirectDrive, Planetary, DirectDrive, Servo
    gear_ratio: float
    rated_torque: Quantity
    peak_torque: Optional[Quantity] = None
    max_velocity: Optional[Quantity] = None
    rotor_inertia: Optional[Quantity] = None
    reflected_inertia: Optional[Quantity] = None
    internal_resistance: Optional[Quantity] = None
    max_current: Optional[Quantity] = None
    mass: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("rated_torque", "peak_torque", mode="after")
    @classmethod
    def _validate_torque(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.newton * ureg.meter, info.field_name)

    @field_validator("max_velocity", mode="after")
    @classmethod
    def _validate_velocity(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.radian / ureg.second, "max_velocity")

    @field_validator("rotor_inertia", "reflected_inertia", mode="after")
    @classmethod
    def _validate_inertia(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.kg * ureg.meter**2, info.field_name)

    @field_validator("internal_resistance", mode="after")
    @classmethod
    def _validate_resistance(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.ohm, "internal_resistance")

    @field_validator("max_current", mode="after")
    @classmethod
    def _validate_current(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.ampere, "max_current")

    @field_validator("mass", mode="after")
    @classmethod
    def _validate_mass(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.kg, "mass")

    @model_validator(mode="after")
    def _compute_reflected_inertia(self):
        if self.reflected_inertia is None and self.rotor_inertia is not None:
            calc_val = (self.gear_ratio ** 2) * self.rotor_inertia
            object.__setattr__(self, "reflected_inertia", calc_val.to(ureg.kg * ureg.meter**2))
        return self


# ---------------------------------------------------------------------------
# Wheel hub motors and frameless joint motors (added 2026-09-28, vol4 M6).
# Every field is a datasheet value; fields a datasheet does not publish stay
# None ("needs source") rather than being estimated.
# ---------------------------------------------------------------------------

_NM = ureg.newton * ureg.meter


class WheelHubMotor(BaseModel):
    """Direct-drive brushless wheel hub motor (motor, tire, and encoder in one wheel).

    A hub motor has no gearbox (N = 1), so its datasheet torque is the torque at
    the tire. ``rotor_inertia`` is None when the datasheet does not publish it;
    ``rotating_mass_bound`` then gives a sourced upper bound on its translational
    equivalent J / r^2 (no part of the motor lies outside the tire radius, so
    J <= m r^2).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    rated_voltage: Quantity
    wheel_diameter: Quantity
    rated_torque: Quantity
    peak_torque: Quantity
    rated_current: Quantity
    peak_current: Quantity
    rated_speed: Quantity
    max_speed: Quantity
    back_emf_constant: Quantity          # line-to-line, per unit shaft speed
    line_resistance: Quantity            # terminal-to-terminal winding resistance
    line_inductance: Quantity
    pole_pairs: int
    mass: Quantity
    load_rating_per_pair: Quantity       # maximum static load carried by two motors
    rotor_inertia: Optional[Quantity] = None   # needs source when None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("rated_voltage", mode="after")
    @classmethod
    def _v_voltage(cls, v, info):
        return require_dimensionality(v, ureg.volt, info.field_name)

    @field_validator("wheel_diameter", mode="after")
    @classmethod
    def _v_length(cls, v, info):
        return require_dimensionality(v, ureg.meter, info.field_name)

    @field_validator("rated_torque", "peak_torque", mode="after")
    @classmethod
    def _v_torque(cls, v, info):
        return require_dimensionality(v, _NM, info.field_name)

    @field_validator("rated_current", "peak_current", mode="after")
    @classmethod
    def _v_current(cls, v, info):
        return require_dimensionality(v, ureg.ampere, info.field_name)

    @field_validator("rated_speed", "max_speed", mode="after")
    @classmethod
    def _v_speed(cls, v, info):
        return require_dimensionality(v, ureg.revolution / ureg.minute, info.field_name)

    @field_validator("back_emf_constant", mode="after")
    @classmethod
    def _v_kemf(cls, v, info):
        return require_dimensionality(v, ureg.volt / (ureg.revolution / ureg.minute), info.field_name)

    @field_validator("line_resistance", mode="after")
    @classmethod
    def _v_res(cls, v, info):
        return require_dimensionality(v, ureg.ohm, info.field_name)

    @field_validator("line_inductance", mode="after")
    @classmethod
    def _v_ind(cls, v, info):
        return require_dimensionality(v, ureg.henry, info.field_name)

    @field_validator("mass", "load_rating_per_pair", mode="after")
    @classmethod
    def _v_mass(cls, v, info):
        return require_dimensionality(v, ureg.kg, info.field_name)

    @field_validator("rotor_inertia", mode="after")
    @classmethod
    def _v_inertia(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.kg * ureg.meter**2, info.field_name)

    @property
    def wheel_radius(self):
        return (self.wheel_diameter / 2).to(ureg.meter)

    @property
    def torque_constant(self):
        """Torque per ampere at the datasheet peak point, peak torque / peak current."""
        return (self.peak_torque / self.peak_current).to(_NM / ureg.ampere)

    @property
    def rotating_mass_bound(self):
        """Upper bound on J / r^2 when rotor inertia is unpublished: the whole motor mass."""
        if self.rotor_inertia is not None:
            return (self.rotor_inertia / self.wheel_radius**2).to(ureg.kg)
        return self.mass.to(ureg.kg)


class FramelessMotor(BaseModel):
    """Frameless brushless servo motor kit (stator and rotor) for integration in a joint.

    Current is the amplitude-invariant q-axis current (peak phase current), so
    copper loss at current I is (3/2) I^2 R_phase; for a star winding the
    terminal resistance is 2 R_phase and ``loss_resistance`` = 0.75 R_terminal.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    winding: str                          # e.g. "star-serial"
    rated_voltage: Quantity
    rated_torque: Quantity
    peak_torque: Quantity                 # datasheet peak (at 20 percent deviation from linearity)
    rated_current: Quantity
    torque_constant: Quantity
    motor_constant: Quantity              # K_M = K_t / sqrt(P_loss per unit current^2)
    terminal_resistance: Quantity         # at 20 C
    terminal_inductance: Quantity
    rotor_inertia: Quantity
    pole_pairs: int
    mass: Quantity
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("rated_voltage", mode="after")
    @classmethod
    def _v_voltage(cls, v, info):
        return require_dimensionality(v, ureg.volt, info.field_name)

    @field_validator("rated_torque", "peak_torque", mode="after")
    @classmethod
    def _v_torque(cls, v, info):
        return require_dimensionality(v, _NM, info.field_name)

    @field_validator("rated_current", mode="after")
    @classmethod
    def _v_current(cls, v, info):
        return require_dimensionality(v, ureg.ampere, info.field_name)

    @field_validator("torque_constant", mode="after")
    @classmethod
    def _v_kt(cls, v, info):
        return require_dimensionality(v, _NM / ureg.ampere, info.field_name)

    @field_validator("terminal_resistance", mode="after")
    @classmethod
    def _v_res(cls, v, info):
        return require_dimensionality(v, ureg.ohm, info.field_name)

    @field_validator("terminal_inductance", mode="after")
    @classmethod
    def _v_ind(cls, v, info):
        return require_dimensionality(v, ureg.henry, info.field_name)

    @field_validator("rotor_inertia", mode="after")
    @classmethod
    def _v_inertia(cls, v, info):
        return require_dimensionality(v, ureg.kg * ureg.meter**2, info.field_name)

    @field_validator("mass", mode="after")
    @classmethod
    def _v_mass(cls, v, info):
        return require_dimensionality(v, ureg.kg, info.field_name)

    @property
    def loss_resistance(self):
        """Effective copper-loss resistance for amplitude-invariant q-axis current, star winding."""
        if not self.winding.startswith("star"):
            raise ValueError(f"loss_resistance assumes a star winding; got {self.winding}")
        return (0.75 * self.terminal_resistance).to(ureg.ohm)

    @property
    def rated_copper_loss(self):
        """Copper loss at rated current and 20 C: I_r^2 * loss_resistance."""
        return (self.rated_current**2 * self.loss_resistance).to(ureg.watt)
