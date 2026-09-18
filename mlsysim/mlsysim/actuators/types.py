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
