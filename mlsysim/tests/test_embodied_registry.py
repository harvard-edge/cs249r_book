"""Unit tests for mlsysim.embodied registry and EmbodiedPlatform models."""

from __future__ import annotations

import pytest
from mlsysim import Embodied, Hardware
from mlsysim.embodied.types import EmbodiedPlatform


def test_embodied_registry_platforms_exist():
    assert Embodied.Quadruped.Spot.name == "Boston Dynamics Spot"
    assert Embodied.Humanoid.Atlas.name == "Boston Dynamics Atlas"
    assert Embodied.Humanoid.Unitree_H1.name == "Unitree H1"
    assert Embodied.Manipulator.Panda.name == "Franka Emika Panda"
    assert Embodied.Drone.DJI_Matrice.name == "DJI Matrice 350 RTK"
    assert Embodied.AMR.LogisticsAMR.name == "Industrial Logistics AMR"
    assert Embodied.Vehicle.Robotaxi.name == "Autonomous Vehicle / Robotaxi"


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


def test_embodied_registry_provenance():
    for platform in [
        Embodied.Quadruped.Spot,
        Embodied.Humanoid.Atlas,
        Embodied.Humanoid.Unitree_H1,
        Embodied.Manipulator.Panda,
        Embodied.Drone.DJI_Matrice,
        Embodied.AMR.LogisticsAMR,
        Embodied.Vehicle.Robotaxi,
    ]:
        assert isinstance(platform, EmbodiedPlatform)
        assert platform.metadata.provenance is not None
        assert platform.metadata.provenance.ref != ""


def test_hardware_embodied_drive_thor():
    thor = Hardware.Embodied.DriveThor
    assert thor.name == "NVIDIA DRIVE Thor"
    assert thor.tdp.to("W").magnitude == pytest.approx(100.0)
