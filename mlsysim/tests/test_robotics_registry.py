"""Unit tests for mlsysim.robotics registry and RobotPlatform models."""

from __future__ import annotations

import pytest
from mlsysim import Robotics, Hardware
from mlsysim.robotics.types import RobotPlatform


def test_robotics_registry_platforms_exist():
    assert Robotics.Quadruped.Spot.name == "Boston Dynamics Spot"
    assert Robotics.Humanoid.Atlas.name == "Boston Dynamics Atlas"
    assert Robotics.Humanoid.Unitree_H1.name == "Unitree H1"
    assert Robotics.Manipulator.Panda.name == "Franka Emika Panda"
    assert Robotics.Drone.DJI_Matrice.name == "DJI Matrice 350 RTK"
    assert Robotics.AMR.LogisticsAMR.name == "Industrial Logistics AMR"
    assert Robotics.Vehicle.Robotaxi.name == "Autonomous Vehicle / Robotaxi"


def test_robotics_registry_units():
    spot = Robotics.Quadruped.Spot
    assert spot.mass.to("kg").magnitude == pytest.approx(32.7)
    assert spot.payload_capacity.to("kg").magnitude == pytest.approx(14.0)
    assert spot.max_velocity.to("m/s").magnitude == pytest.approx(1.6)
    assert spot.control_frequency.to("Hz").magnitude == pytest.approx(1000.0)

    amr = Robotics.AMR.LogisticsAMR
    assert amr.mass.to("kg").magnitude == pytest.approx(150.0)
    assert amr.max_velocity.to("m/s").magnitude == pytest.approx(1.8)
    assert amr.max_acceleration.to("m/s^2").magnitude == pytest.approx(2.5)


def test_robotics_registry_provenance():
    for platform in [
        Robotics.Quadruped.Spot,
        Robotics.Humanoid.Atlas,
        Robotics.Humanoid.Unitree_H1,
        Robotics.Manipulator.Panda,
        Robotics.Drone.DJI_Matrice,
        Robotics.AMR.LogisticsAMR,
        Robotics.Vehicle.Robotaxi,
    ]:
        assert isinstance(platform, RobotPlatform)
        assert platform.metadata.provenance is not None
        assert platform.metadata.provenance.ref != ""


def test_hardware_embodied_drive_thor():
    thor = Hardware.Embodied.DriveThor
    assert thor.name == "NVIDIA DRIVE Thor"
    assert thor.tdp.to("W").magnitude == pytest.approx(100.0)
