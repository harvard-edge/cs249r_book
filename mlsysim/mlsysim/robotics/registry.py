"""Registry of physical AI robot platforms and machine archetypes."""

from ..core.registry import Registry
from ..core.types import Metadata
from ..core.units import ureg
from ..core import provenance_catalog as pc
from ..hardware.registry import Hardware
from .types import RobotPlatform


class Quadrupeds(Registry):
    """Dynamic legged quadruped platforms (Class 1: Mobility)."""

    Spot = RobotPlatform(
        name="Boston Dynamics Spot",
        archetype="Class 1: Mobility",
        mass=32.7 * ureg.kg,
        payload_capacity=14.0 * ureg.kg,
        max_velocity=1.6 * (ureg.meter / ureg.second),
        max_acceleration=2.0 * (ureg.meter / ureg.second**2),
        dofs=12,
        control_frequency=1000.0 * ureg.Hz,
        nominal_power=400.0 * ureg.watt,
        transmission="Quasi-Direct Drive (QDD) / Brushless Outrunner",
        compute_soc=Hardware.Edge.JetsonAGXOrin,
        metadata=Metadata(provenance=pc.BOSTON_DYNAMICS_SPOT),
    )


class Humanoids(Registry):
    """General-purpose bipedal humanoid embodiments (Class 4: Integrative Capstone)."""

    Atlas = RobotPlatform(
        name="Boston Dynamics Atlas",
        archetype="Class 4: Humanoid Capstone",
        mass=89.0 * ureg.kg,
        payload_capacity=11.0 * ureg.kg,
        max_velocity=2.5 * (ureg.meter / ureg.second),
        max_acceleration=5.0 * (ureg.meter / ureg.second**2),
        dofs=28,
        control_frequency=1000.0 * ureg.Hz,
        nominal_power=1500.0 * ureg.watt,
        transmission="High-bandwidth electro-hydraulic / QDD electric",
        compute_soc=Hardware.Edge.JetsonAGXOrin,
        metadata=Metadata(provenance=pc.BOSTON_DYNAMICS_ATLAS),
    )

    Unitree_H1 = RobotPlatform(
        name="Unitree H1",
        archetype="Class 4: Humanoid Capstone",
        mass=47.0 * ureg.kg,
        payload_capacity=30.0 * ureg.kg,
        max_velocity=3.3 * (ureg.meter / ureg.second),
        dofs=19,
        control_frequency=1000.0 * ureg.Hz,
        nominal_power=800.0 * ureg.watt,
        transmission="High-torque M107 joint motors (360 N·m)",
        compute_soc=Hardware.Edge.JetsonAGXOrin,
        metadata=Metadata(provenance=pc.UNITREE_H1_HUMANOID),
    )


class Manipulators(Registry):
    """Articulated robotic manipulators (Class 2: Contact-Dominated Manipulation)."""

    Panda = RobotPlatform(
        name="Franka Emika Panda",
        archetype="Class 2: Manipulation",
        mass=18.0 * ureg.kg,
        payload_capacity=3.0 * ureg.kg,
        max_velocity=2.0 * (ureg.meter / ureg.second),
        dofs=7,
        control_frequency=1000.0 * ureg.Hz,
        nominal_power=150.0 * ureg.watt,
        transmission="Harmonic Drive with joint torque sensors",
        metadata=Metadata(provenance=pc.FRANKA_EMIKA_PANDA),
    )


class Drones(Registry):
    """Aerial robotic platforms and quadrotors (Class 1: Aerial Mobility)."""

    DJI_Matrice = RobotPlatform(
        name="DJI Matrice 350 RTK",
        archetype="Class 1: Aerial Mobility",
        mass=6.47 * ureg.kg,
        payload_capacity=2.7 * ureg.kg,
        max_velocity=21.0 * (ureg.meter / ureg.second),
        control_frequency=400.0 * ureg.Hz,
        nominal_power=1000.0 * ureg.watt,
        transmission="Direct-drive brushless outrunner motors",
        compute_soc=Hardware.Edge.JetsonOrinNano,
        metadata=Metadata(provenance=pc.DJI_MATRICE_350_RTK),
    )


class AMRs(Registry):
    """Industrial Autonomous Mobile Robots (Class 1: Ground Mobility)."""

    LogisticsAMR = RobotPlatform(
        name="Industrial Logistics AMR",
        archetype="Class 1: Mobility",
        mass=150.0 * ureg.kg,
        payload_capacity=100.0 * ureg.kg,
        max_velocity=1.8 * (ureg.meter / ureg.second),
        max_acceleration=2.5 * (ureg.meter / ureg.second**2),
        control_frequency=100.0 * ureg.Hz,
        nominal_power=300.0 * ureg.watt,
        transmission="Differential drive with integrated spring-applied brakes",
        compute_soc=Hardware.Edge.JetsonAGXOrin,
        metadata=Metadata(provenance=pc.INDUSTRIAL_AMR),
    )


class Vehicles(Registry):
    """Automated road vehicles and robotaxis (Class 1: Mass-Dominated Mobility)."""

    Robotaxi = RobotPlatform(
        name="Autonomous Vehicle / Robotaxi",
        archetype="Class 1: Mobility",
        mass=2400.0 * ureg.kg,
        payload_capacity=400.0 * ureg.kg,
        max_velocity=25.0 * (ureg.meter / ureg.second),
        max_acceleration=6.5 * (ureg.meter / ureg.second**2),
        control_frequency=100.0 * ureg.Hz,
        nominal_power=3500.0 * ureg.watt,
        compute_soc=Hardware.Embodied.DriveThor,
        metadata=Metadata(provenance=pc.AUTONOMOUS_VEHICLE_ROBOTAXI),
    )


class Robotics(Registry):
    """Authoritative registry of cyber-physical robot platforms and machine archetypes."""

    Quadruped = Quadrupeds
    Humanoid = Humanoids
    Manipulator = Manipulators
    Drone = Drones
    AMR = AMRs
    Vehicle = Vehicles
