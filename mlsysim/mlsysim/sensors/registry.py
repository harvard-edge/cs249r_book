"""Registry of perception sensors (cameras, LiDARs, IMUs) for physical AI."""

from ..core.registry import Registry
from ..core.types import Metadata
from ..core.units import ureg
from ..core import provenance_catalog as pc
from .types import CameraSensor, LiDARSensor, IMUSensor


class Cameras(Registry):
    """Authoritative image sensors and depth cameras."""

    Sony_IMX477 = CameraSensor(
        name="Sony IMX477",
        resolution_width=4056,
        resolution_height=3040,
        frame_rate=60.0 * ureg.Hz,
        shutter_type="rolling",
        exposure_time=16.0 * ureg.millisecond,
        readout_time=16.6 * ureg.millisecond,
        nominal_latency=33.3 * ureg.millisecond,
        interface="MIPI CSI-2 (4-lane, 10-bit full-resolution 60 fps mode)",
        nominal_power=0.6 * ureg.watt,
        metadata=Metadata(provenance=pc.SONY_IMX477),
    )

    Sony_IMX296 = CameraSensor(
        name="Sony IMX296",
        resolution_width=1440,
        resolution_height=1080,
        frame_rate=60.0 * ureg.Hz,
        shutter_type="global",
        exposure_time=5.0 * ureg.millisecond,
        readout_time=5.0 * ureg.millisecond,
        nominal_latency=10.0 * ureg.millisecond,
        interface="MIPI CSI-2 (1-lane)",
        nominal_power=0.3 * ureg.watt,
        metadata=Metadata(provenance=pc.SONY_IMX296),
    )

    Intel_RealSense_D435i = CameraSensor(
        name="Intel RealSense D435i",
        resolution_width=1920,
        resolution_height=1080,
        frame_rate=30.0 * ureg.Hz,
        shutter_type="global",
        nominal_latency=25.0 * ureg.millisecond,
        interface="USB 3.1 Gen 1",
        nominal_power=2.0 * ureg.watt,
        metadata=Metadata(provenance=pc.INTEL_REALSENSE_D435I),
    )


class LiDARs(Registry):
    """Authoritative LiDAR ranging sensors."""

    Ouster_OS1_64 = LiDARSensor(
        name="Ouster OS1-64",
        channels=64,
        scan_rate=20.0 * ureg.Hz,
        max_range=120.0 * ureg.meter,
        nominal_latency=25.0 * ureg.millisecond,
        interface="Gigabit Ethernet (UDP)",
        nominal_power=18.0 * ureg.watt,
        metadata=Metadata(provenance=pc.OUSTER_OS1_64),
    )


class IMUs(Registry):
    """Authoritative inertial measurement units."""

    Bosch_BMI088 = IMUSensor(
        name="Bosch BMI088",
        dofs=6,
        sample_rate=1000.0 * ureg.Hz,
        nominal_latency=1.0 * ureg.millisecond,
        interface="SPI / I2C",
        nominal_power=0.01 * ureg.watt,
        metadata=Metadata(provenance=pc.BOSCH_BMI088),
    )


class Sensors(Registry):
    """Authoritative registry of robotics perception sensors."""

    Camera = Cameras
    LiDAR = LiDARs
    IMU = IMUs
