"""Registry of perception sensors (cameras, LiDARs, IMUs) for physical AI."""

from ..core.registry import Registry
from ..core.types import Metadata
from ..core.units import ureg
from ..core import provenance_catalog as pc
from .types import CameraSensor, LiDARSensor, IMUSensor, SafetyLaserScanner


class Cameras(Registry):
    """Authoritative image sensors and depth cameras."""

    Sony_IMX477 = CameraSensor(
        name="Sony IMX477",
        resolution_width=4056,
        resolution_height=3040,
        frame_rate=60.0 * ureg.Hz,
        shutter_type="rolling",
        # exposure_time is a configured setting (category: chosen), not a datasheet
        # property: the sensor supports a range of exposures.
        exposure_time=16.0 * ureg.millisecond,
        readout_time=16.6 * ureg.millisecond,
        nominal_latency=33.3 * ureg.millisecond,
        interface="MIPI CSI-2 (4-lane, 10-bit full-resolution 60 fps mode)",
        nominal_power=0.6 * ureg.watt,
        metadata=Metadata(
            provenance=pc.SONY_IMX477,
            description="Sony IMX477 12.3 MP rolling-shutter CMOS perception camera (4-lane MIPI CSI-2).",
        ),
    )

    Sony_IMX296 = CameraSensor(
        name="Sony IMX296",
        resolution_width=1440,
        resolution_height=1080,
        frame_rate=60.0 * ureg.Hz,
        shutter_type="global",
        exposure_time=5.0 * ureg.millisecond,  # configured setting (category: chosen)
        readout_time=5.0 * ureg.millisecond,
        nominal_latency=10.0 * ureg.millisecond,
        interface="MIPI CSI-2 (1-lane)",
        nominal_power=0.3 * ureg.watt,
        metadata=Metadata(
            provenance=pc.SONY_IMX296,
            description="Sony IMX296 1.58 MP global-shutter CMOS camera for high-speed robotic vision.",
        ),
    )

    Intel_RealSense_D435i = CameraSensor(
        name="Intel RealSense D435i",
        resolution_width=1920,
        resolution_height=1080,
        frame_rate=30.0 * ureg.Hz,
        # 2026-09-29: the color sensor (OV2740) is rolling shutter; only the
        # stereo depth imagers (OV9282, D430 wide module) are global shutter.
        # D400 Series Datasheet rev 023 (Mar 2026), Tables 3-18 and 3-49.
        shutter_type="rolling",
        depth_shutter_type="global",
        nominal_latency=25.0 * ureg.millisecond,
        interface="USB 3.1 Gen 1",
        nominal_power=2.0 * ureg.watt,
        metadata=Metadata(
            provenance=pc.INTEL_REALSENSE_D435I,
            description="Intel RealSense D435i active stereo depth camera with integrated IMU and RGB imager.",
        ),
    )


class LiDARs(Registry):
    """Authoritative LiDAR ranging sensors."""

    Ouster_OS1_64 = LiDARSensor(
        name="Ouster OS1-64",
        channels=64,
        scan_rate=20.0 * ureg.Hz,
        max_range=120.0 * ureg.meter,  # needs source: Rev7 sheet gives 170 m (80% target) and 90 m (10%)
        # 2026-09-29: the Rev7 datasheet gives data latency < 10 ms (was 25 ms,
        # unsupported); nominal_latency stores that upper bound.
        nominal_latency=10.0 * ureg.millisecond,
        range_precision_min=5.0 * ureg.millimeter,
        range_precision_max=30.0 * ureg.millimeter,
        range_accuracy=25.0 * ureg.millimeter,
        range_accuracy_retroreflective=50.0 * ureg.millimeter,  # Rev7 datasheet: +/-5 cm, retroreflective targets
        vertical_fov=42.4 * ureg.degree,  # Rev7 datasheet: 42.4 deg (+21.2 to -21.2)
        time_sync_accuracy=1.0 * ureg.millisecond,
        internal_clock_drift=10.0 * ureg.ppm,
        laser_wavelength=865.0 * ureg.nanometer,  # Rev7 datasheet (Class 1, IEC 60825-1:2014)
        interface="Gigabit Ethernet (UDP)",
        nominal_power=18.0 * ureg.watt,
        metadata=Metadata(
            provenance=pc.OUSTER_OS1_64,
            description="Ouster OS1-64 64-channel digital LiDAR with 120 m range and 20 Hz scan rate.",
        ),
    )


    # Added 2026-10-20: the lidar of the KITTI recording platform, as that
    # platform's paper states it (KITTI ran it at 10 Hz).
    Velodyne_HDL_64E = LiDARSensor(
        name="Velodyne HDL-64E",
        channels=64,
        scan_rate=10.0 * ureg.Hz,
        max_range=120.0 * ureg.meter,
        vertical_fov=26.8 * ureg.degree,
        interface="Ethernet (UDP)",
        metadata=Metadata(
            provenance=pc.KITTI_GEIGER_2013,
            description="Velodyne HDL-64E 64-beam spinning lidar of the KITTI recording platform, run at 10 Hz.",
        ),
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
        # Typical values, data sheet rev 1.9 (Tables 4 and 5); accelerometer
        # noise is the X/Y value (Z is 190 ug/sqrt(Hz)).
        gyro_noise_density=0.014 * ureg.degree / ureg.second / ureg.hertz**0.5,
        gyro_zero_rate_offset=1.0 * ureg.degree / ureg.second,
        accel_noise_density=160.0e-6 * ureg.gravity / ureg.hertz**0.5,
        accel_zero_g_offset=20.0e-3 * ureg.gravity,
        metadata=Metadata(
            provenance=pc.BOSCH_BMI088,
            description="Bosch BMI088 6-DoF low-noise automotive-grade IMU for robotic navigation.",
        ),
    )


class SafetyScanners(Registry):
    """Certified safety laser scanners (IEC 61496-3 AOPDDR)."""

    Keyence_SZ_V32 = SafetyLaserScanner(
        name="Keyence SZ-V32",
        protective_field_range=5.7 * ureg.meter,
        response_time=80.0 * ureg.millisecond,
        scan_angle=190.0 * ureg.degree,
        iec61496_type="Type 3",
        performance_level="PL d, Category 3",
        configuration="High-speed mode, 70/150 mm objects, 2 scans",
        metadata=Metadata(
            provenance=pc.KEYENCE_SZ_V32,
            description="Keyence SZ-V32 Type 3 / PL d optical safety laser scanner (5.7 m protection zone).",
        ),
    )


class Sensors(Registry):
    """Authoritative registry of robotics perception sensors."""

    Camera = Cameras
    LiDAR = LiDARs
    IMU = IMUs
    SafetyScanner = SafetyScanners
