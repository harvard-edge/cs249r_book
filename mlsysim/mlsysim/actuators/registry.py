"""Registry of actuators and joint transmissions for physical AI platforms."""

from ..core.registry import Registry
from ..core.types import Metadata
from ..core.units import ureg
from ..core import provenance_catalog as pc
from .types import ActuatorJoint


class HarmonicDrives(Registry):
    """Harmonic drive (strain wave) zero-backlash robotic joint actuators."""

    CSG_17_50 = ActuatorJoint(
        name="Harmonic Drive CSG-17-50",
        actuator_type="HarmonicDrive",
        gear_ratio=50.0,
        rated_torque=21.0 * (ureg.newton * ureg.meter),
        peak_torque=54.0 * (ureg.newton * ureg.meter),
        rotor_inertia=0.19e-4 * (ureg.kg * ureg.meter**2),
        reflected_inertia=0.0475 * (ureg.kg * ureg.meter**2),  # 50^2 * 0.19e-4
        internal_resistance=0.45 * ureg.ohm,
        max_current=12.0 * ureg.ampere,
        mass=0.68 * ureg.kg,
        metadata=Metadata(provenance=pc.HARMONIC_DRIVE_CSG),
    )

    CSG_20_100 = ActuatorJoint(
        name="Harmonic Drive CSG-20-100",
        actuator_type="HarmonicDrive",
        gear_ratio=100.0,
        rated_torque=40.0 * (ureg.newton * ureg.meter),
        peak_torque=107.0 * (ureg.newton * ureg.meter),
        rotor_inertia=0.38e-4 * (ureg.kg * ureg.meter**2),
        reflected_inertia=0.38 * (ureg.kg * ureg.meter**2),  # 100^2 * 0.38e-4
        internal_resistance=0.55 * ureg.ohm,
        max_current=15.0 * ureg.ampere,
        mass=0.98 * ureg.kg,
        metadata=Metadata(provenance=pc.HARMONIC_DRIVE_CSG),
    )

    CSG_25_50 = ActuatorJoint(
        name="Harmonic Drive CSG-25-50",
        actuator_type="HarmonicDrive",
        gear_ratio=50.0,
        rated_torque=67.0 * (ureg.newton * ureg.meter),
        peak_torque=150.0 * (ureg.newton * ureg.meter),
        rotor_inertia=1.0e-4 * (ureg.kg * ureg.meter**2),
        reflected_inertia=0.25 * (ureg.kg * ureg.meter**2),  # 50^2 * 1.0e-4
        internal_resistance=0.60 * ureg.ohm,
        max_current=18.0 * ureg.ampere,
        mass=1.5 * ureg.kg,
        metadata=Metadata(provenance=pc.HARMONIC_DRIVE_CSG),
    )


class QuasiDirectDrives(Registry):
    """Quasi-direct-drive (QDD) low-gear-ratio actuators for dynamic legged robots."""

    Unitree_M107 = ActuatorJoint(
        name="Unitree M107",
        actuator_type="QuasiDirectDrive",
        gear_ratio=10.0,
        rated_torque=120.0 * (ureg.newton * ureg.meter),
        peak_torque=360.0 * (ureg.newton * ureg.meter),
        rotor_inertia=1.2e-4 * (ureg.kg * ureg.meter**2),
        reflected_inertia=0.012 * (ureg.kg * ureg.meter**2),  # 10^2 * 1.2e-4
        internal_resistance=0.18 * ureg.ohm,
        max_current=60.0 * ureg.ampere,
        mass=1.9 * ureg.kg,
        metadata=Metadata(provenance=pc.UNITREE_M107_MOTOR),
    )

    TMotor_AK80_9 = ActuatorJoint(
        name="T-Motor AK80-9",
        actuator_type="QuasiDirectDrive",
        gear_ratio=9.0,
        rated_torque=9.0 * (ureg.newton * ureg.meter),
        peak_torque=18.0 * (ureg.newton * ureg.meter),
        rotor_inertia=0.8e-4 * (ureg.kg * ureg.meter**2),
        reflected_inertia=0.00648 * (ureg.kg * ureg.meter**2),  # 9^2 * 0.8e-4
        internal_resistance=0.15 * ureg.ohm,
        max_current=24.0 * ureg.ampere,
        mass=0.485 * ureg.kg,
        metadata=Metadata(provenance=pc.TMOTOR_AK80_9),
    )


class Servos(Registry):
    """Smart networked position/torque bus actuators for manipulators."""

    Dynamixel_XM430 = ActuatorJoint(
        name="Dynamixel XM430-W350",
        actuator_type="Servo",
        gear_ratio=353.5,
        rated_torque=3.0 * (ureg.newton * ureg.meter),
        peak_torque=4.1 * (ureg.newton * ureg.meter),
        internal_resistance=1.2 * ureg.ohm,
        max_current=2.3 * ureg.ampere,
        mass=0.082 * ureg.kg,
        metadata=Metadata(provenance=pc.DYNAMIXEL_XM430),
    )


class Actuators(Registry):
    """Authoritative registry of robotic actuators and joint transmissions."""

    HarmonicDrive = HarmonicDrives
    QuasiDirectDrive = QuasiDirectDrives
    Servo = Servos
