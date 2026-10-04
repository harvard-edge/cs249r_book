"""Registry of actuators and joint transmissions for physical AI platforms."""

from ..core.registry import Registry
from ..core.types import Metadata
from ..core.units import ureg
from ..core import provenance_catalog as pc
from ..core.provenance import sourced_qty
from .types import ActuatorJoint, WheelHubMotor, FramelessMotor


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
        metadata=Metadata(
            provenance=pc.HARMONIC_DRIVE_CSG,
            description="Harmonic Drive CSG-17-50 zero-backlash robotic joint (50:1 gear ratio, 21 N·m rated torque).",
        ),
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
        metadata=Metadata(
            provenance=pc.HARMONIC_DRIVE_CSG,
            description="Harmonic Drive CSG-20-100 zero-backlash robotic joint (100:1 gear ratio, 40 N·m rated torque).",
        ),
    )

    CSG_25_50 = ActuatorJoint(
        name="Harmonic Drive CSG-25-50",
        actuator_type="HarmonicDrive",
        gear_ratio=50.0,
        rated_torque=51.0 * (ureg.newton * ureg.meter),  # L10 rated, CSG-25-50-2UH
        peak_torque=127.0 * (ureg.newton * ureg.meter),  # repeated peak limit
        rotor_inertia=1.0e-4 * (ureg.kg * ureg.meter**2),
        reflected_inertia=0.25 * (ureg.kg * ureg.meter**2),  # 50^2 * 1.0e-4
        internal_resistance=0.60 * ureg.ohm,
        max_current=18.0 * ureg.ampere,
        mass=1.5 * ureg.kg,
        metadata=Metadata(
            provenance=pc.HARMONIC_DRIVE_CSG,
            description="Harmonic Drive CSG-25-50 zero-backlash robotic joint (50:1 gear ratio, 51 N·m rated torque).",
        ),
    )


class FramelessMotors(Registry):
    """Frameless brushless servo kits (stator and rotor) integrated into robot joints."""

    TQ_ILM70x18 = FramelessMotor(
        name="TQ-RoboDrive ILM 70x18 (star-serial)",
        winding="star-serial",
        rated_voltage=48.0 * ureg.volt,
        rated_torque=1.25 * (ureg.newton * ureg.meter),
        peak_torque=4.05 * (ureg.newton * ureg.meter),       # at 20 percent deviation from linearity
        rated_current=6.7 * ureg.ampere,                      # amplitude-invariant (peak phase) current
        torque_constant=0.187 * (ureg.newton * ureg.meter / ureg.ampere),
        motor_constant=0.266 * (ureg.newton * ureg.meter / ureg.watt**0.5),
        terminal_resistance=660.0 * ureg.milliohm,            # at 20 C
        terminal_inductance=1460.0 * ureg.microhenry,
        rotor_inertia=0.321e-4 * (ureg.kg * ureg.meter**2),   # 0.321 kg cm^2
        pole_pairs=10,
        mass=0.330 * ureg.kg,
        metadata=Metadata(
            provenance=pc.TQ_ROBODRIVE_ILM,
            description="TQ-RoboDrive ILM 70x18 frameless brushless servo kit (star-serial winding, 1.25 N·m rated torque).",
        ),
    )


# CSG-25 gear-unit moment of inertia at the input (Harmonic Drive AG engineering data).
_CSG25_INPUT_INERTIA = sourced_qty(
    0.413e-4 * (ureg.kg * ureg.meter**2), pc.HARMONIC_DRIVE_CSG_2UH_ENGINEERING,
    name="CSG-25-2UH moment of inertia at the input",
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
        metadata=Metadata(
            provenance=pc.UNITREE_M107_MOTOR,
            description="Unitree M107 high-torque quasi-direct-drive joint motor for dynamic legged robotics (360 N·m peak).",
        ),
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
        metadata=Metadata(
            provenance=pc.TMOTOR_AK80_9,
            description="T-Motor AK80-9 low-gear-ratio quasi-direct-drive dynamic actuator (9:1 ratio, 18 N·m peak).",
        ),
    )


class PandaClassJoints(Registry):
    """Sourced stand-ins for a Franka Panda-class strain-wave joint.

    Franka publishes the Panda's joint torque limits (87 N m on axes 1-4) and its
    harmonic-drive transmission, not the joint motor or gear parameters. The
    stand-in pairs a CSG-25-50 gear unit with the smallest TQ-RoboDrive ILM motor
    whose peak torque through 50:1 covers the gear's repeatable peak.
    """

    CSG_25_50_ILM70x18 = ActuatorJoint(
        name="CSG-25-50-2UH strain-wave joint with TQ-RoboDrive ILM 70x18",
        actuator_type="HarmonicDrive",
        gear_ratio=50.0,
        rated_torque=51.0 * (ureg.newton * ureg.meter),       # gear L10 rated torque
        peak_torque=127.0 * (ureg.newton * ureg.meter),       # gear repeatable peak torque
        # Motor rotor plus gear input inertia, both sourced: 0.321e-4 + 0.413e-4.
        rotor_inertia=(FramelessMotors.TQ_ILM70x18.rotor_inertia + _CSG25_INPUT_INERTIA).to(ureg.kg * ureg.meter**2),
        internal_resistance=FramelessMotors.TQ_ILM70x18.loss_resistance,  # copper-loss resistance, amplitude-invariant i_q
        mass=1.5 * ureg.kg + FramelessMotors.TQ_ILM70x18.mass,
        metadata=Metadata(
            provenance=pc.FRANKA_CLASS_STRAIN_WAVE_JOINT,
            description="Franka Panda-class strain-wave joint stand-in pairing CSG-25-50 gear unit with TQ-RoboDrive ILM 70x18.",
        ),
    )


class HubMotors(Registry):
    """Direct-drive wheel hub motors for AMR drive wheels."""

    ZLTECH_ZLLG10ASM800 = WheelHubMotor(
        name="ZLTECH ZLLG10ASM800-48 10-inch hub servo motor",
        rated_voltage=48.0 * ureg.volt,
        wheel_diameter=266.0 * ureg.millimeter,
        rated_torque=20.0 * (ureg.newton * ureg.meter),
        peak_torque=60.0 * (ureg.newton * ureg.meter),
        rated_current=8.5 * ureg.ampere,
        peak_current=25.0 * ureg.ampere,
        rated_speed=150.0 * (ureg.revolution / ureg.minute),
        max_speed=180.0 * (ureg.revolution / ureg.minute),
        back_emf_constant=0.262 * (ureg.volt / (ureg.revolution / ureg.minute)),  # line-to-line
        line_resistance=0.7 * ureg.ohm,
        line_inductance=1.885 * ureg.millihenry,
        pole_pairs=20,
        mass=9.5 * ureg.kg,
        load_rating_per_pair=300.0 * ureg.kg,
        rotor_inertia=None,  # needs source: the datasheet does not publish it
        metadata=Metadata(
            provenance=pc.ZLTECH_ZLLG10ASM800,
            description="ZLTECH ZLLG10ASM800-48 10-inch direct-drive wheel hub servo motor for industrial AMRs (60 N·m peak).",
        ),
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
        metadata=Metadata(
            provenance=pc.DYNAMIXEL_XM430,
            description="ROBOTIS Dynamixel XM430-W350 smart networked bus servo actuator (353.5:1 ratio, 4.1 N·m peak).",
        ),
    )


class Actuators(Registry):
    """Authoritative registry of robotic actuators and joint transmissions."""

    HarmonicDrive = HarmonicDrives
    QuasiDirectDrive = QuasiDirectDrives
    Servo = Servos
    FramelessMotor = FramelessMotors
    PandaClassJoint = PandaClassJoints
    HubMotor = HubMotors
