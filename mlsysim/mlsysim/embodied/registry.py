"""Registry of embodied AI platforms, cyber-physical machines, and robotic embodiments."""

from ..core.registry import Registry
from ..core.types import Metadata
from ..core.units import ureg
from ..core import provenance_catalog as pc
from ..hardware.registry import Hardware
from .types import EmbodiedPlatform


class Quadrupeds(Registry):
    """Dynamic legged quadruped platforms (Class 1: Mobility)."""

    Spot = EmbodiedPlatform(
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

    Atlas = EmbodiedPlatform(
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

    Unitree_H1 = EmbodiedPlatform(
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
    """Articulated robotic arms and dexterous manipulators (Class 2: Manipulation)."""

    Panda = EmbodiedPlatform(
        name="Franka Emika Panda",
        archetype="Class 2: Manipulation",
        mass=18.0 * ureg.kg,
        payload_capacity=3.0 * ureg.kg,
        max_velocity=2.0 * (ureg.meter / ureg.second),
        max_torque=87.0 * (ureg.newton * ureg.meter),
        dofs=7,
        control_frequency=1000.0 * ureg.Hz,
        nominal_power=300.0 * ureg.watt,
        transmission="Harmonic drive with integrated joint torque sensors",
        compute_soc=Hardware.Edge.JetsonAGXOrin,
        metadata=Metadata(provenance=pc.FRANKA_EMIKA_PANDA),
    )

    ALOHA_Bimanual = EmbodiedPlatform(
        name="ALOHA Bimanual Manipulator",
        archetype="Class 2: Manipulation",
        mass=15.0 * ureg.kg,
        payload_capacity=0.5 * ureg.kg,
        max_velocity=1.0 * (ureg.meter / ureg.second),
        dofs=14,
        control_frequency=50.0 * ureg.Hz,
        nominal_power=150.0 * ureg.watt,
        transmission="Dynamixel servo actuators / 4-bar linkage gripper",
        compute_soc=Hardware.Edge.JetsonAGXOrin,
        metadata=Metadata(provenance=pc.ALOHA_BIMANUAL_MANIPULATOR),
    )


class Drones(Registry):
    """Unmanned aerial vehicles and micro-aerial platforms (Class 1: Mobility)."""

    DJI_Matrice = EmbodiedPlatform(
        name="DJI Matrice 350 RTK",
        archetype="Class 1: Aerial Mobility",
        mass=6.47 * ureg.kg,
        payload_capacity=2.7 * ureg.kg,
        max_velocity=23.0 * (ureg.meter / ureg.second),
        max_acceleration=6.0 * (ureg.meter / ureg.second**2),
        control_frequency=400.0 * ureg.Hz,
        nominal_power=1000.0 * ureg.watt,
        transmission="Direct-drive brushless DC motors (FOC)",
        compute_soc=Hardware.Edge.JetsonOrinNano,
        metadata=Metadata(provenance=pc.DJI_MATRICE_350_RTK),
    )


class AMRs(Registry):
    """Autonomous mobile robots and industrial warehouse transport vehicles (Class 1: Mobility)."""

    LogisticsAMR = EmbodiedPlatform(
        name="Industrial Logistics AMR",
        archetype="Class 1: Industrial Mobility",
        mass=150.0 * ureg.kg,
        payload_capacity=500.0 * ureg.kg,
        max_velocity=1.8 * (ureg.meter / ureg.second),
        max_acceleration=2.5 * (ureg.meter / ureg.second**2),
        control_frequency=100.0 * ureg.Hz,
        nominal_power=500.0 * ureg.watt,
        transmission="Differential wheel drive with planetary gearboxes",
        compute_soc=Hardware.Edge.JetsonAGXOrin,
        metadata=Metadata(provenance=pc.INDUSTRIAL_AMR),
    )

    WarehouseAMR = EmbodiedPlatform(
        name="Warehouse Fulfillment AMR",
        archetype="Class 1: Industrial Mobility",
        mass=300.0 * ureg.kg,
        payload_capacity=1000.0 * ureg.kg,
        max_velocity=1.5 * (ureg.meter / ureg.second),
        max_acceleration=2.0 * (ureg.meter / ureg.second**2),
        control_frequency=100.0 * ureg.Hz,
        nominal_power=600.0 * ureg.watt,
        transmission="Differential dual-drive in-wheel brushless motors",
        compute_soc=Hardware.Edge.JetsonAGXOrin,
        metadata=Metadata(provenance=pc.WAREHOUSE_AMR),
    )

    HeavyAMR = EmbodiedPlatform(
        name="Heavy Logistics AMR",
        archetype="Class 1: Industrial Mobility",
        mass=250.0 * ureg.kg,
        payload_capacity=800.0 * ureg.kg,
        max_velocity=1.8 * (ureg.meter / ureg.second),
        max_acceleration=2.0 * (ureg.meter / ureg.second**2),
        control_frequency=100.0 * ureg.Hz,
        nominal_power=550.0 * ureg.watt,
        transmission="Differential drive with planetary reduction",
        compute_soc=Hardware.Edge.JetsonAGXOrin,
        metadata=Metadata(provenance=pc.HEAVY_AMR),
    )


class Vehicles(Registry):
    """Autonomous driving vehicles and robotaxis (Class 1: Heavy Mobility)."""

    Robotaxi = EmbodiedPlatform(
        name="Autonomous Vehicle / Robotaxi",
        archetype="Class 1: Heavy Mobility",
        mass=2200.0 * ureg.kg,
        max_velocity=33.3 * (ureg.meter / ureg.second),  # 120 km/h
        max_acceleration=4.0 * (ureg.meter / ureg.second**2),
        control_frequency=100.0 * ureg.Hz,
        nominal_power=2500.0 * ureg.watt,
        transmission="Electric multi-motor powertrain",
        compute_soc=Hardware.Embodied.DriveThor,
        metadata=Metadata(provenance=pc.AUTONOMOUS_VEHICLE_ROBOTAXI),
    )

    UberATG_VolvoXC90 = EmbodiedPlatform(
        name="Uber ATG Volvo XC90",
        archetype="Class 1: Heavy Mobility",
        mass=2100.0 * ureg.kg,
        max_velocity=25.0 * (ureg.meter / ureg.second),
        max_acceleration=8.0 * (ureg.meter / ureg.second**2),
        control_frequency=100.0 * ureg.Hz,
        nominal_power=2500.0 * ureg.watt,
        transmission="Hydraulic friction braking with steer-by-wire",
        compute_soc=Hardware.Edge.JetsonAGXOrin,
        metadata=Metadata(provenance=pc.UBER_ATG_VOLVO_XC90),
    )


class ContinuousProcesses(Registry):
    """Continuous physical processes and industrial manufacturing systems (Class 3: Process)."""

    DED_MeltPool = EmbodiedPlatform(
        name="Direct Energy Deposition (DED) Melt Pool",
        archetype="Class 3: Process",
        mass=500.0 * ureg.kg,
        control_frequency=1000.0 * ureg.Hz,
        nominal_power=5000.0 * ureg.watt,
        transmission="CNC gantry with fiber laser and powder nozzle",
        compute_soc=Hardware.Edge.JetsonAGXOrin,
        metadata=Metadata(provenance=pc.DED_MELT_POOL_PROCESS),
    )


class Embodied(Registry):
    """Authoritative registry of embodied AI platforms, cyber-physical machines, and robotic embodiments."""

    Quadruped = Quadrupeds
    Humanoid = Humanoids
    Manipulator = Manipulators
    Drone = Drones
    AMR = AMRs
    Vehicle = Vehicles
    ContinuousProcess = ContinuousProcesses
