"""Registry of embodied AI platforms, cyber-physical machines, and robotic embodiments."""

from ..core.registry import Registry
from ..core.types import Metadata
from ..core.units import ureg
from ..core import provenance_catalog as pc
from ..hardware.registry import Hardware
from .types import EmbodiedPlatform
from ..core.provenance import sourced_qty
from ..sensors.registry import Sensors
from ..models.registry import Models
from .types import MobileManipulatorPlatform, EmbodiedSiteScenario


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


class MobileManipulators(Registry):
    """Composite mobile manipulators: a Class 1 base carrying a Class 2 arm."""

    WarehouseMobileManipulator = MobileManipulatorPlatform(
        name="Warehouse mobile manipulator",
        archetype="Class 1: Mobility + Class 2: Manipulation",
        base=AMRs.WarehouseAMR,
        arm=Manipulators.Panda,
        onboard_payload_capacity=50.0 * ureg.kg,               # chosen tote-rack capacity
        footprint_width=1.00 * ureg.meter,
        brain_soc=Hardware.Edge.JetsonAGXOrin,
        permission_mcu=Hardware.Tiny.LockstepSafetyMCU_Reference,
        permission_rate=1000.0 * ureg.Hz,                      # chosen
        current_loop_rate=20_000.0 * ureg.Hz,                  # chosen
        fieldbus_cycle=1.0 * ureg.millisecond,                 # EtherCAT, illustrative
        servo_axes=9,                                          # 7 arm joints + 2 drive wheels
        nav_camera=Sensors.Camera.Sony_IMX477,
        wrist_camera=Sensors.Camera.Sony_IMX296,
        lidar=Sensors.LiDAR.Ouster_OS1_64,
        imu=Sensors.IMU.Bosch_BMI088,
        intent_model=Models.Embodied.OpenVLA_7B,
        chunk_model=Models.Embodied.ACT_ALOHA,
        # Intent model: OpenVLA-7B with INT4 weight-only quantization (OpenVLA
        # reports INT4 matching bfloat16 task success; Kim et al. 2024, Table 2),
        # decoding its 7 action tokens autoregressively. Latency is computed by
        # the intent_inference_latency property (compute-bound prefill plus six
        # bandwidth-bound decode sweeps), about 336 ms; the rate is the largest
        # whole-hertz rate whose period covers it (2026-09-28, M2; was a
        # stipulated 5 Hz / 160 ms that priced only one weight sweep).
        intent_rate=2.0 * ureg.Hz,                             # chosen, <= 1 / computed latency
        intent_weight_bytes_per_param=0.5 * (ureg.byte / ureg.param),  # chosen: INT4 weights
        intent_input_tokens=288,                               # 256 image patches (224 px, P = 14) + 32 prompt tokens, illustrative
        intent_parallel_decode=False,                          # OpenVLA: one token per action dimension
        intent_compute_utilization=0.50,                       # illustrative achieved fraction of dense FP16 peak
        intent_reference_latency=sourced_qty(
            336.0 * ureg.millisecond, pc.JETSON_AI_LAB_OPENVLA_ORIN,
            name="OpenVLA-7B INT4 per-frame latency, Jetson AGX Orin 64GB",
        ),
        chunk_rate=20.0 * ureg.Hz,                             # chosen
        chunk_inference_latency=40.0 * ureg.millisecond,       # illustrative P99 incl. vision
        chunk_step=20.0 * ureg.millisecond,                    # chosen setpoint period
        chunk_horizon=16,                                      # chosen
        dram_efficiency=0.70,                                  # sustained fraction of peak
        tcp_speed_limit=1.0 * (ureg.meter / ureg.second),      # chosen free-space limit
        control_rail_nominal=24.0 * ureg.volt,                 # chosen control-rail voltage
        control_rail_battery_low=21.5 * ureg.volt,             # illustrative depleted battery
        control_rail_resistance=60.0 * ureg.milliohm,          # illustrative harness resistance
        control_rail_dropout=18.0 * ureg.volt,                 # illustrative regulator dropout
        permission_rail_holdup=2.0 * ureg.second,              # chosen hold-up time
        permission_rail_load=70.0 * ureg.watt,                 # illustrative permission-path load
        spring_brake_engage_min=30.0 * ureg.millisecond,       # illustrative spring-brake engage, fast end
        spring_brake_engage_max=80.0 * ureg.millisecond,       # illustrative spring-brake engage, slow end
        metadata=Metadata(provenance=pc.WAREHOUSE_MOBILE_MANIPULATOR),
    )


class SiteScenarios(Registry):
    """Budget terms for an embodied machine at a named site (illustrative unless sourced)."""

    WarehouseAisle = EmbodiedSiteScenario(
        name="Warehouse aisle",
        machine=MobileManipulators.WarehouseMobileManipulator,
        delta_loc=0.050 * ureg.meter,                          # illustrative
        delta_margin=0.100 * ureg.meter,                       # chosen
        eps_track=0.040 * ureg.meter,                          # illustrative
        t_brake_onset=20.0 * ureg.millisecond,                 # illustrative
        t_lease=60.0 * ureg.millisecond,                       # chosen
        t_bus=MobileManipulators.WarehouseMobileManipulator.fieldbus_cycle,
        t_transport=0.8 * ureg.millisecond,                    # illustrative
        t_dma=1.2 * ureg.millisecond,                          # illustrative
        t_isp=2.5 * ureg.millisecond,                          # illustrative
        t_backbone=22.0 * ureg.millisecond,                    # illustrative
        t_ipc=0.5 * ureg.millisecond,                          # illustrative
        heartbeat_rate=100.0 * ureg.Hz,                        # chosen
        heartbeat_timeout=30.0 * ureg.millisecond,             # chosen
        mcu_self_watchdog=5.0 * ureg.millisecond,              # chosen
        enforcer_unloaded=135.0 * ureg.microsecond,            # illustrative
        enforcer_wcet=250.0 * ureg.microsecond,                # chosen (declared)
        enforcer_deadline=400.0 * ureg.microsecond,            # chosen
        v_human_approach=sourced_qty(
            1.6 * (ureg.meter / ureg.second), pc.ISO_13855_APPROACH_SPEED,
            name="ISO 13855 walking approach speed K",
        ),
        d_clear=1.10 * ureg.meter,                             # illustrative rack-end clear distance
        v_aisle=1.3 * (ureg.meter / ureg.second),              # chosen, below the derived ceiling
        aisle_width=1.30 * ureg.meter,
        aisle_length=30.0 * ureg.meter,
        side_clearance=0.15 * ureg.meter,
        mu_dry=0.60,
        mu_inspected_floor=0.12,
        mu_oil_film=0.05,
        k_latch=4.0e5 * (ureg.newton / ureg.meter),
        v_latch_approach=0.03 * (ureg.meter / ureg.second),
        v_latch_high=0.10 * (ureg.meter / ureg.second),
        f_latch_tripwire=15.0 * ureg.newton,
        t_contact_response=2.0 * ureg.millisecond,
        f_latch_limit=100.0 * ureg.newton,
        v_conveyor=0.20 * (ureg.meter / ureg.second),
        a_conveyor_slip=0.50 * (ureg.meter / ureg.second**2),
        grasp_tolerance=15.0 * ureg.millimeter,
        grasp_initial_error=3.0 * ureg.millimeter,
        k_contact_human=15_000.0 * (ureg.newton / ureg.meter),
        f_contact_criterion=50.0 * ureg.newton,                # chosen
        v_handover=0.10 * (ureg.meter / ureg.second),          # chosen coworker handover TCP speed
        # Remote takeover (2026-09-28, M4; replaces an illustrative 150 ms in-loop
        # and 2.0 s out-of-loop takeover). The design totals are properties:
        # t_takeover_in_loop = slow-bound simple reaction + video + handshake;
        # t_takeover_out_of_loop = p90 condition-mean take-over time + video + handshake.
        t_human_perceptual=sourced_qty(
            100.0 * ureg.millisecond, pc.CARD_MORAN_NEWELL_MHP,
            name="Model Human Processor perceptual cycle, nominal",
        ),
        t_human_perceptual_slow=sourced_qty(
            200.0 * ureg.millisecond, pc.CARD_MORAN_NEWELL_MHP,
            name="Model Human Processor perceptual cycle, slow bound",
        ),
        t_human_cognitive=sourced_qty(
            70.0 * ureg.millisecond, pc.CARD_MORAN_NEWELL_MHP,
            name="Model Human Processor cognitive cycle, nominal",
        ),
        t_human_cognitive_slow=sourced_qty(
            170.0 * ureg.millisecond, pc.CARD_MORAN_NEWELL_MHP,
            name="Model Human Processor cognitive cycle, slow bound",
        ),
        t_human_motor=sourced_qty(
            70.0 * ureg.millisecond, pc.CARD_MORAN_NEWELL_MHP,
            name="Model Human Processor motor cycle, nominal",
        ),
        t_human_motor_slow=sourced_qty(
            100.0 * ureg.millisecond, pc.CARD_MORAN_NEWELL_MHP,
            name="Model Human Processor motor cycle, slow bound",
        ),
        t_brake_response_expected=sourced_qty(
            0.75 * ureg.second, pc.GREEN_2000_PERCEPTION_BRAKE,
            name="Perception-brake time, expected signal (upper end of 0.70-0.75 s)",
        ),
        t_brake_response_unexpected=sourced_qty(
            1.25 * ureg.second, pc.GREEN_2000_PERCEPTION_BRAKE,
            name="Perception-brake time, unexpected but common signal",
        ),
        t_brake_response_surprise=sourced_qty(
            1.5 * ureg.second, pc.GREEN_2000_PERCEPTION_BRAKE,
            name="Perception-brake time, surprise event",
        ),
        t_takeover_driving_p50=sourced_qty(
            2.37 * ureg.second, pc.ZHANG_2019_TAKEOVER_META,
            name="Take-over time, median of 520 condition means",
        ),
        t_takeover_driving_mean=sourced_qty(
            2.72 * ureg.second, pc.ZHANG_2019_TAKEOVER_META,
            name="Take-over time, average of 520 condition means",
        ),
        t_takeover_driving_p90=sourced_qty(
            4.33 * ureg.second, pc.ZHANG_2019_TAKEOVER_META,
            name="Take-over time, 90th percentile of 520 condition means",
        ),
        t_takeover_driving_p99=sourced_qty(
            7.0 * ureg.second, pc.ZHANG_2019_TAKEOVER_META,
            name="Take-over time, 99th percentile of 520 condition means",
        ),
        t_video_glass_to_glass=sourced_qty(
            104.0 * ureg.millisecond, pc.SCHIMPE_2022_TELEOP_G2G,
            name="Teleoperation video glass-to-glass latency, wired best case",
        ),                                                     # site wireless link: needs a measured value
        t_handshake_request=10.0 * ureg.millisecond,           # chosen protocol budget
        t_handshake_acknowledge=5.0 * ureg.millisecond,        # chosen protocol budget
        t_handshake_confirm=15.0 * ureg.millisecond,           # chosen protocol budget
        v_takeover_in_loop=0.8 * (ureg.meter / ureg.second),   # chosen slow-down, under the sourced in-loop ceiling (was 1.2)
        v_takeover_out_of_loop=0.15 * (ureg.meter / ureg.second),  # chosen crawl, under the sourced out-of-loop ceiling (was 0.3)
        i_rail_transient=70.0 * ureg.ampere,                   # illustrative coincident inference + drive transient
        metadata=Metadata(provenance=pc.WAREHOUSE_AISLE_SCENARIO),
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
    MobileManipulator = MobileManipulators
    Scenario = SiteScenarios
