"""Unit tests for mlsysim.embodied registry and EmbodiedPlatform models."""

from __future__ import annotations

import pytest
from mlsysim import Embodied, Hardware, Sensors, Actuators, Models, Datasets
from mlsysim.embodied.types import EmbodiedPlatform
from mlsysim.sensors.types import CameraSensor, LiDARSensor, IMUSensor
from mlsysim.actuators.types import ActuatorJoint
from mlsysim.models.types import EmbodiedWorkload
from mlsysim.datasets.types import DatasetProfile


def test_embodied_registry_platforms_exist():
    assert Embodied.Quadruped.Spot.name == "Boston Dynamics Spot"
    assert Embodied.Humanoid.Atlas.name == "Boston Dynamics Atlas"
    assert Embodied.Humanoid.Unitree_H1.name == "Unitree H1"
    assert Embodied.Manipulator.Panda.name == "Franka Emika Panda"
    assert Embodied.Manipulator.ALOHA_Bimanual.name == "ALOHA Bimanual Manipulator"
    assert Embodied.Drone.DJI_Matrice.name == "DJI Matrice 350 RTK"
    assert Embodied.AMR.LogisticsAMR.name == "Industrial Logistics AMR"
    assert Embodied.Vehicle.Robotaxi.name == "Autonomous Vehicle / Robotaxi"
    assert Embodied.Vehicle.UberATG_VolvoXC90.name == "Uber ATG Volvo XC90"
    assert Embodied.ContinuousProcess.DED_MeltPool.name == "Direct Energy Deposition (DED) Melt Pool"


def test_embodied_registry_units():
    spot = Embodied.Quadruped.Spot
    assert spot.mass.to("kg").magnitude == pytest.approx(32.7)
    assert spot.payload_capacity.to("kg").magnitude == pytest.approx(14.0)
    assert spot.max_velocity.to("m/s").magnitude == pytest.approx(1.6)
    assert spot.control_frequency.to("Hz").magnitude == pytest.approx(1000.0)

    amr = Embodied.AMR.LogisticsAMR
    assert amr.mass.to("kg").magnitude == pytest.approx(150.0)
    assert amr.max_velocity.to("m/s").magnitude == pytest.approx(1.8)
    assert amr.max_acceleration.to("m/s^2").magnitude == pytest.approx(2.5)

    volvo = Embodied.Vehicle.UberATG_VolvoXC90
    assert volvo.mass.to("kg").magnitude == pytest.approx(2100.0)
    assert volvo.max_acceleration.to("m/s^2").magnitude == pytest.approx(8.0)


def test_embodied_registry_provenance():
    for platform in [
        Embodied.Quadruped.Spot,
        Embodied.Humanoid.Atlas,
        Embodied.Humanoid.Unitree_H1,
        Embodied.Manipulator.Panda,
        Embodied.Manipulator.ALOHA_Bimanual,
        Embodied.Drone.DJI_Matrice,
        Embodied.AMR.LogisticsAMR,
        Embodied.Vehicle.Robotaxi,
        Embodied.Vehicle.UberATG_VolvoXC90,
        Embodied.ContinuousProcess.DED_MeltPool,
    ]:
        assert isinstance(platform, EmbodiedPlatform)
        assert platform.metadata.provenance is not None
        assert platform.metadata.provenance.ref != ""


def test_hardware_embodied_drive_thor():
    thor = Hardware.Embodied.DriveThor
    assert thor.name == "NVIDIA DRIVE Thor"
    assert thor.tdp.to("W").magnitude == pytest.approx(100.0)


def test_sensors_registry():
    cam477 = Sensors.Camera.Sony_IMX477
    assert isinstance(cam477, CameraSensor)
    assert cam477.shutter_type == "rolling"
    assert cam477.resolution_width == 4056
    assert cam477.nominal_latency.to("ms").magnitude == pytest.approx(33.3, rel=0.01)

    cam296 = Sensors.Camera.Sony_IMX296
    assert cam296.shutter_type == "global"
    assert cam296.nominal_latency.to("ms").magnitude == pytest.approx(10.0)

    lidar = Sensors.LiDAR.Ouster_OS1_64
    assert isinstance(lidar, LiDARSensor)
    assert lidar.channels == 64
    assert lidar.scan_rate.to("Hz").magnitude == pytest.approx(20.0)

    imu = Sensors.IMU.Bosch_BMI088
    assert isinstance(imu, IMUSensor)
    assert imu.sample_rate.to("Hz").magnitude == pytest.approx(1000.0)


def test_actuators_registry():
    csg17 = Actuators.HarmonicDrive.CSG_17_50
    assert isinstance(csg17, ActuatorJoint)
    assert csg17.gear_ratio == 50.0
    assert csg17.rated_torque.to("N*m").magnitude == pytest.approx(21.0)
    # Reflected inertia computed or verified: 50^2 * 0.19e-4 = 0.0475 kg*m^2
    assert csg17.reflected_inertia.to("kg*m^2").magnitude == pytest.approx(0.0475, rel=0.01)

    m107 = Actuators.QuasiDirectDrive.Unitree_M107
    assert isinstance(m107, ActuatorJoint)
    assert m107.peak_torque.to("N*m").magnitude == pytest.approx(360.0)
    assert m107.reflected_inertia.to("kg*m^2").magnitude == pytest.approx(0.012, rel=0.01)


def test_models_embodied_registry():
    openvla = Models.Embodied.OpenVLA_7B
    assert isinstance(openvla, EmbodiedWorkload)
    assert openvla.parameters.to("count").magnitude == pytest.approx(7e9)
    assert openvla.action_chunk_horizon == 1
    # Check lowering to computation graph
    cg = openvla.lower()
    assert cg.parameter_count.to("count").magnitude == pytest.approx(7e9)

    act = Models.Embodied.ACT_ALOHA
    assert act.action_chunk_horizon == 100
    assert act.action_dim == 14


def test_datasets_embodied_registry():
    oxe = Datasets.OpenXEmbodiment
    assert isinstance(oxe, DatasetProfile)
    assert oxe.episodes.to("count").magnitude == pytest.approx(1e6)
    assert oxe.embodiments == 22

    droid = Datasets.DROID
    assert droid.episodes.to("count").magnitude == pytest.approx(76000)
    assert droid.total_duration.to("hour").magnitude == pytest.approx(350.0)

