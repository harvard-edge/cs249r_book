"""Named robotics scenarios for perception.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ....physics.constants import ALUMINUM_THERMAL_EXPANSION_COEFFICIENT as _ALUMINUM_ALPHA
from ....physics.constants import MOR_CONTRAST_THRESHOLD as _MOR_CONTRAST


# --- Sensor perception on the warehouse mobile manipulator (added 2026-10-20) ---
# The machine's navigation camera, lidar, stage allowances, clearances, yaw
# rate, and clock-conversion bound are in Embodied.Scenario.WarehouseAisle;
# inputs neither record carries are stated here.
WarehouseNavCameraAge = _st.ObservationAgeScenario(
    name="The warehouse machine's navigation-camera observation age at its drive limit",
    description="Exposure midpoint to dispatch, stage by stage, priced as travel at 1.5 m/s; "
    "a four-frame temporal batch for comparison.",
    site=_Embodied.Scenario.WarehouseAisle,
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    batch_frames=4,  # scenario assumption: a four-frame batch across capture times
)


WarehouseNavCameraRollingShutter = _st.RollingShutterScenario(
    name="Rolling-shutter skew of the warehouse machine's navigation camera at its drive limit",
    description="Row interval, along-track skew, and the sideways skew of the rack-end upright "
    "while the base turns into the aisle.",
    site=_Embodied.Scenario.WarehouseAisle,
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
)


WarehouseNavCameraRollingShutterAisleSpeed = WarehouseNavCameraRollingShutter.variant(
    name="The same rolling-shutter skew at the chosen aisle speed",
    description="The drive-limit case with the speed lowered to the site's 1.3 m/s aisle speed.",
    speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
)


WarehouseRackEndTimingAttitude = _st.TimingAttitudeErrorScenario(
    name="One millisecond and one degree of error at the warehouse rack end",
    description="Travel and misorientation per millisecond of misdating while turning, and the "
    "sideways error per degree of attitude error at the 1.10 m rack end.",
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    yaw_rate=_Embodied.Scenario.WarehouseAisle.yaw_rate_turn,
    distance=_Embodied.Scenario.WarehouseAisle.d_clear,
    timing_error=Q_(1.0, "ms"),  # scenario assumption: one millisecond of misdating
    angle_error=Q_(1.0, "degree"),  # scenario assumption: one degree of mount error
)


WarehouseLongAisleMountError = WarehouseRackEndTimingAttitude.variant(
    name="A one-degree mount error read 15 m down a long aisle",
    description="The same uncalibrated degree at a 15 m detection range, against the side clearance.",
    distance=Q_(15.0, "m"),  # scenario assumption: detection range down a long aisle
    reference=_Embodied.Scenario.WarehouseAisle.side_clearance,
)


WarehouseLidarExtrinsicShift = WarehouseRackEndTimingAttitude.variant(
    name="A 0.005 rad extrinsic shift after a shock, at a 10 m target",
    description="A small lidar-to-base rotation after a bump, against the localization allowance.",
    distance=Q_(10.0, "m"),  # scenario assumption: target range
    angle_error=Q_(0.005, "radian"),  # scenario assumption: extrinsic shift after a shock
    reference=_Embodied.Scenario.WarehouseAisle.delta_loc,
)


WarehouseRackEndLidarAge = _st.LidarSweepAgeScenario(
    name="The warehouse machine's lidar at its drive limit: one sweep, and the rack-end sector's age",
    description="Travel and turn during one 20 Hz sweep; the aisle sector at the rack end streamed "
    "as it arrives or held for the full sweep, against the camera's age.",
    site=_Embodied.Scenario.WarehouseAisle,
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
)


WarehouseStereoRigDepthError = _st.StereoDepthErrorScenario(
    name="An illustrative stereo rig against the warehouse aisle's clearances",
    description="One-sigma depth error at 2 m, 10 m, and the rack end; the ranges where it spends "
    "the localization allowance and the side clearance; a baseline change's bias at 20 m.",
    site=_Embodied.Scenario.WarehouseAisle,
    baseline=Q_(0.20, "m"),  # scenario assumption: stereo baseline
    focal_length_px=800.0,  # scenario assumption: focal length in pixels
    disparity_sigma_px=0.5,  # scenario assumption: disparity standard deviation in pixels
    near_range=Q_(2.0, "m"),  # scenario assumption
    far_range=Q_(10.0, "m"),  # scenario assumption
    ray_clearance=Q_(150.0, "mm"),  # scenario assumption: clearance kept along the viewing ray
    baseline_change=Q_(0.1, "mm"),  # scenario assumption: unmodeled baseline change
    bias_range=Q_(20.0, "m"),  # scenario assumption
)


VisualServoDelayPhaseMargin = _st.DelayPhaseMarginScenario(
    name="A visual-servo loop with 45 degrees of margin at 10 rad/s, closed through 60 ms of perception",
    description="The pure delay that erases the margin, and the share a 60 ms exposure, transport, "
    "and inference age spends.",
    phase_margin=Q_(45.0, "degree"),  # scenario assumption: nominal phase margin
    crossover=Q_(10.0, "rad/s"),  # scenario assumption: gain crossover
    delay=Q_(60.0, "ms"),  # scenario assumption: exposure, transport, and inference
)


ClearanceInsetConformalRank = _st.ConformalInsetScenario(
    name="A 99th-percentile clearance inset from 1,000 measured residuals",
    description="The split-conformal rank for a one percent miss rate, and the fewest residuals that state it.",
    alpha=0.01,  # scenario assumption: tail mass for the clearance inset
    calibration_count=1000,  # scenario assumption: calibration residuals
)


WarehouseLidarClockConversion = _st.ClockConversionScenario(
    name="The warehouse machine's lidar clock between one-second syncs",
    description="Drift of the lidar's free-running clock between syncs, the producer's dispatch limit, "
    "and the conversion bound's travel at the drive limit.",
    site=_Embodied.Scenario.WarehouseAisle,
    drift_rate=_Embodied.MobileManipulator.WarehouseMobileManipulator.lidar.internal_clock_drift,
    sync_interval=Q_(1.0, "s"),  # scenario assumption: sync interval
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
)


DenseFogVisibility = _st.VisibilityRangeScenario(
    name="Dense fog: the range at which contrast falls to the meteorological threshold",
    description="Koschmieder's law at an extinction coefficient of 0.05 per meter.",
    extinction=Q_(0.05, "1/m"),  # scenario assumption: dense-fog extinction coefficient
    contrast_threshold=_MOR_CONTRAST,
)


AluminumLinkWarming = _st.ThermalExpansionScenario(
    name="A 1 m aluminum link warmed by 10 K",
    description="Length change that shifts a calibrated sensor mount.",
    coefficient=_ALUMINUM_ALPHA,
    length=Q_(1.0, "m"),  # scenario assumption: link length
    temperature_rise=Q_(10.0, "K"),  # scenario assumption: uniform warming
)
