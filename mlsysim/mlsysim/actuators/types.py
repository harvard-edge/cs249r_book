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
        """Convert the specified wheel diameter to radius.

        Returns
        -------
        Quantity
            Nominal wheel radius in meters, derived from wheel_diameter.

        Notes
        -----
        Equation: wheel_radius = wheel_diameter / 2.
        Assumptions: wheel_diameter is a positive geometric diameter of a circular
            wheel; no tire deformation correction is applied.
        Excludes: loaded rolling radius, tire wear, compression, and slip.

        References
        ----------
        Derivation: a diameter spans two radii of the same circle, so dividing the
            stored diameter by two preserves its length dimension.

        Examples
        --------
        >>> from mlsysim import Actuators
        >>> motor = Actuators.HubMotor.ZLTECH_ZLLG10ASM800
        >>> motor.wheel_radius * 2 == motor.wheel_diameter
        True
        """
        return (self.wheel_diameter / 2).to(ureg.meter)

    @property
    def torque_constant(self):
        """Compute the peak-point ratio of torque to current.

        Returns
        -------
        Quantity
            Peak torque divided by peak current, in newton meters per ampere.
            This ratio is an effective operating-point constant, not a fitted
            linear torque-current curve.

        Notes
        -----
        Equation: torque_constant = peak_torque / peak_current.
        Assumptions: peak_torque and positive peak_current describe the same
            operating point and use the same current convention.
        Excludes: magnetic saturation away from the supplied point, winding
            temperature effects, driver losses, and a continuous torque rating.

        References
        ----------
        Derivation: the proportional model torque = K_t * current gives
            K_t = torque/current at the supplied peak point. No inference about
            linearity at other points follows from this single ratio.

        Examples
        --------
        >>> from mlsysim import Actuators
        >>> motor = Actuators.HubMotor.ZLTECH_ZLLG10ASM800
        >>> expected = motor.peak_torque / motor.peak_current
        >>> motor.torque_constant == expected
        True
        """
        return (self.peak_torque / self.peak_current).to(_NM / ureg.ampere)

    @property
    def rotating_mass_bound(self):
        """Return the translational equivalent of rotor inertia or a mass bound.

        Returns
        -------
        Quantity
            Equivalent rotating mass J/r**2 in kilograms when rotor_inertia is
            supplied. Otherwise, the whole motor mass in kilograms is used as
            an upper bound under the radial-envelope assumption.

        Notes
        -----
        Equation: with known rotor inertia J and wheel radius r, equivalent mass
            is ``J/r**2``. Without J, the bound is motor mass m because every rotating
            mass element at radius rho <= r contributes ``rho**2`` dm <= ``r**2`` dm.
        Assumptions: rotation and translation are linked by rolling at the stated
            positive wheel radius. The fallback assumes no rotating part lies
            beyond that radius and rotating mass does not exceed total motor mass.
        Excludes: translated vehicle mass, additional wheels or transmissions,
            tire slip, and the inference of an unpublished inertia measurement.

        References
        ----------
        Derivation: equate rotational energy ``J*omega**2/2`` to translational energy
            ``m_equivalent*v**2/2`` with v = ``r*omega``. Integrating ``rho**2`` dm within
            radius r gives J <= ``m*r**2`` and therefore ``J/r**2`` <= m.

        Examples
        --------
        An illustrative copy with unknown inertia uses the motor-mass bound.

        >>> from mlsysim import Actuators
        >>> motor = Actuators.HubMotor.ZLTECH_ZLLG10ASM800
        >>> unknown = motor.model_copy(update={"rotor_inertia": None})
        >>> unknown.rotating_mass_bound == unknown.mass
        True
        """
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
    no_load_speed: Optional[Quantity] = None  # at rated voltage, theoretical (added 2026-10-20)
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
        """Convert star-winding terminal resistance to copper-loss resistance.

        Returns
        -------
        Quantity
            Effective resistance in ohms, such that copper power is
            q_axis_current**2 * loss_resistance for amplitude-invariant current.

        Raises
        ------
        ValueError
            If winding does not begin with "star"; the conversion is not defined
            here for delta or unknown winding connections.

        Notes
        -----
        Equation: loss_resistance = 0.75 * terminal_resistance. Balanced
            three-phase copper power is ``(3/2)*I_q**2*R_phase``, while a star winding
            has R_terminal = ``2*R_phase``.
        Assumptions: balanced sinusoidal phase currents, amplitude-invariant
            q-axis current equal to peak phase current, and a star connection.
            The stored terminal resistance applies at its stated temperature.
        Excludes: iron loss, inverter loss, mechanical friction, temperature
            corrections, and alternative current or winding conventions.

        References
        ----------
        Derivation: the sum of squared balanced sinusoidal phase currents is
            ``(3/2)*I_q**2``. Replacing R_phase by R_terminal/2 in their Joule loss
            gives the factor 3/4 multiplying terminal resistance.

        Examples
        --------
        >>> from mlsysim import Actuators
        >>> motor = Actuators.FramelessMotor.TQ_ILM70x18
        >>> motor.loss_resistance == motor.terminal_resistance * 0.75
        True
        """
        if not self.winding.startswith("star"):
            raise ValueError(f"loss_resistance assumes a star winding; got {self.winding}")
        return (0.75 * self.terminal_resistance).to(ureg.ohm)

    @property
    def rated_copper_loss(self):
        """Calculate winding copper power at the rated q-axis current.

        Returns
        -------
        Quantity
            Copper loss in watts at rated_current and the temperature of the
            stored terminal resistance, documented as 20 degrees Celsius.

        Raises
        ------
        ValueError
            If the winding is not star-connected, because loss_resistance cannot
            apply its star-winding conversion.

        See Also
        --------
        FramelessMotor.loss_resistance : Current-convention and winding conversion.

        Notes
        -----
        Equation: rated_copper_loss = ``rated_current**2`` * loss_resistance.
        Assumptions: rated_current uses the amplitude-invariant q-axis convention;
            winding resistance stays at its stored reference-temperature value.
        Excludes: hot-winding resistance growth, transient temperature evolution,
            iron and inverter losses, and total electrical input power.

        References
        ----------
        Derivation: Joule heating is current squared times effective resistance.
            The loss_resistance property accounts for the balanced three-phase
            star-winding geometry and the chosen current convention.

        Examples
        --------
        >>> from mlsysim import Actuators
        >>> motor = Actuators.FramelessMotor.TQ_ILM70x18
        >>> expected = motor.rated_current**2 * motor.loss_resistance
        >>> motor.rated_copper_loss == expected
        True
        """
        return (self.rated_current**2 * self.loss_resistance).to(ureg.watt)


class SafetyRelay(BaseModel):
    """Dual-channel safety relay that removes a drive's torque enables when an E-stop opens.

    No processor sits in its path; its datasheet states the delay from the
    E-stop opening to its outputs opening as a typical and a maximum value.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    estop_release_typical: Quantity       # delay-on de-energisation with E-STOP, typical
    estop_release_max: Quantity           # the same, maximum
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("estop_release_typical", "estop_release_max", mode="after")
    @classmethod
    def _validate_times(cls, v):
        return require_dimensionality(v, ureg.second, "estop_release")
