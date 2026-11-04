"""Named robotics scenarios for spatial memory.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ..robotics.perception import WarehouseNavCameraAge, WarehouseStereoRigDepthError


# --- Spatial memory on the warehouse mobile manipulator (added 2026-10-20) ---
# The machine, clearances, walking speed, conveyor, grasp record, and
# observation age are in Embodied.Scenario.WarehouseAisle; the cited
# odometry, mapping, and tracking figures in published record and
# Inputs neither record carries are stated here.
WarehouseRackEndRangeFusion = _st.RangeFusionScenario(
    name="The range to the warehouse rack-end upright, fused from lidar and stereo",
    description="One sweep's prediction, a lidar update, a stereo update, and the stereo update "
    "misdated by the navigation camera's arrival stamp at the drive limit.",
    site=_Embodied.Scenario.WarehouseAisle,
    stereo=WarehouseStereoRigDepthError,
    age=WarehouseNavCameraAge,
    prior_sigma=Q_(20.0, "mm"),  # scenario assumption: posterior from the previous sweep
    speed_sigma=Q_(0.1, "m/s"),  # scenario assumption: odometry speed disturbance (wheel slip)
    comparison_scale=2.0,  # scenario assumption: the stereo spread doubled, for comparison
)


RackEndMisdatedUpdateConsistency = _st.EstimatorConsistencyScenario(
    name="Consistency tests for the rack-end range estimator",
    description="The 95 percent band for mean NEES over 50 qualification runs, and the 95 and 99 percent "
    "NIS gates against one misdated stereo update.",
    fusion=WarehouseRackEndRangeFusion,
    runs=50,  # scenario assumption: qualification runs with ground truth
    confidence=0.95,  # scenario assumption: two-sided acceptance level
    gate_confidences=[0.95, 0.99],  # scenario assumption: single-update gate levels
)


WarehouseObservationReplayBuffer = _st.ReplayBufferScenario(
    name="Out-of-sequence replay buffer for the warehouse navigation camera's observation age",
    description="Slots at the base IMU's rate that reach back one observation age.",
    site=_Embodied.Scenario.WarehouseAisle,
)


WarehouseImuOneSecondCoast = _st.InertialCoastScenario(
    name="The warehouse machine's IMU coasting one second without a range fix",
    description="Heading and position error from white noise and from the datasheet offsets, "
    "with the heading offset seen at the rack-end upright.",
    site=_Embodied.Scenario.WarehouseAisle,
    duration=Q_(1.0, "s"),  # scenario assumption: interval without a range update
)


WarehouseUmbmarkOdometryDrift = _st.OdometryDriftScenario(
    name="UMBmark odometry drift against the warehouse localization bound",
    description="Borenstein and Feng's return errors before and after calibration as drift rates, "
    "priced against the 50 mm bound at the aisle speed and over one aisle.",
    site=_Embodied.Scenario.WarehouseAisle,
    square_side=Q_(4, "m"),  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
    legs=4,  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
    return_error_uncalibrated=Q_(317, "mm"),  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
    return_error_calibrated=Q_(21, "mm"),  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
    run_sigma=Q_(25, "mm"),  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
    test_speed=Q_(0.2, "m/s"),  # Borenstein, J. and Feng, L. (1996), Measurement and Correction of S...
)


LatchSlotFingerBelief = _st.BeliefExpiryScenario(
    name="The finger's position in the cage latch-release slot while the finger hides it",
    description="Initial error, unobserved drift, and disturbance acceleration against the slot tolerance.",
    initial_error=Q_(0.1, "mm"),  # scenario assumption: finger-to-slot error at the wrist capture
    growth_rate=Q_(12.0, "mm/s"),  # scenario assumption: unobserved lateral velocity bound
    growth_acceleration=Q_(50.0, "mm/s**2"),  # scenario assumption: disturbance acceleration bound
    tolerance=Q_(1.0, "mm"),  # scenario assumption: latch-release slot tolerance
)


LatchSlotFingerBeliefTightSlot = LatchSlotFingerBelief.variant(
    name="The same finger belief in a slot machined to 0.30 mm",
    description="A tighter slot; the fixed initial error takes a larger share of it.",
    tolerance=Q_(0.30, "mm"),  # scenario assumption: tighter machined slot
)


LatchSlotStaleWristTransform = _st.StaleTransformScenario(
    name="A one-frame-stale camera-to-slot edge composed with the current wrist pose",
    description="Drift and lever-arm swing over one wrist-camera frame while the wrist aligns the finger.",
    site=_Embodied.Scenario.WarehouseAisle,
    belief=LatchSlotFingerBelief,
    lever=Q_(0.15, "m"),  # scenario assumption: wrist-to-fingertip lever arm
    angular_rate=Q_(0.30, "rad/s"),  # scenario assumption: wrist rate while aligning the finger
)


ConveyorMugHandleBelief = _st.BeliefExpiryScenario(
    name="The mug handle's pose on the warehouse takeaway conveyor",
    description="Grasp-record initial error growing at the belt speed against the grasp tolerance, "
    "admitted 35 ms after capture under the clock-conversion bound.",
    initial_error=_Embodied.Scenario.WarehouseAisle.grasp_initial_error,
    growth_rate=_Embodied.Scenario.WarehouseAisle.v_conveyor,
    tolerance=_Embodied.Scenario.WarehouseAisle.grasp_tolerance,
    use_age=Q_(35.0, "ms"),  # scenario assumption: capture-to-use age on the consumer's clock
    age_bound=_Embodied.Scenario.WarehouseAisle.clock_conversion_bound,
)


CoworkerBesideArmBelief = _st.BeliefExpiryScenario(
    name="A coworker tracked beside the warehouse arm",
    description="Position error at the last observation growing at the walking speed against the "
    "tolerance to the arm's swept volume.",
    initial_error=Q_(15.0, "mm"),  # scenario assumption: coworker position error at the last observation
    growth_rate=_Embodied.Scenario.WarehouseAisle.v_human_approach,
    tolerance=Q_(150.0, "mm"),  # scenario assumption: belief tolerance to the arm's swept volume
)


CoworkerBesideArmBeliefTightTolerance = CoworkerBesideArmBelief.variant(
    name="The same coworker belief with a 100 mm tolerance",
    description="A tighter swept-volume tolerance.",
    tolerance=Q_(100.0, "mm"),  # scenario assumption: tighter swept-volume tolerance
)


WarehouseRackEndFreeSpace = _st.FreeSpaceExpiryScenario(
    name="Free space at the warehouse rack end, ahead and abreast, eroding at walking speed",
    description="The camera frame's free-space claim ahead (clear distance) and beside the rack end "
    "(side clearance); what the observation age leaves; a walker during a stop; and a "
    "single 100 ms staleness timeout for the whole robot.",
    site=_Embodied.Scenario.WarehouseAisle,
    uniform_timeout=Q_(100.0, "ms"),  # scenario assumption: one staleness timeout for every belief
)


ArmSensorEpochGap = _st.EpochGapScenario(
    name="Three arm sensors latching at different epochs in one cycle",
    description="Tactile, camera, and encoder epochs; the camera-to-encoder gap at the arm's Cartesian limit.",
    epochs={
        "tactile": Q_(12.0, "ms"),  # scenario assumption: tactile array latches force
        "camera": Q_(18.5, "ms"),  # scenario assumption: camera exposure midpoint
        "encoders": Q_(21.0, "ms"),
    },  # scenario assumption: encoders sample angles
    pair=["camera", "encoders"],
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.arm.cartesian_velocity_limit,
)


WarehouseSkewedRecordPair = _st.SkewedRecordsScenario(
    name="A person detection and current odometry 20 ms apart, compared at the warehouse drive limit",
    description="Closing speed of the base and a walker, the envelope across the skew, and the slack "
    "left in a 200 mm separation.",
    site=_Embodied.Scenario.WarehouseAisle,
    skew=Q_(20.0, "ms"),  # scenario assumption: acquisition-time gap between the two records
    separation=Q_(200.0, "mm"),  # scenario assumption: available separation
)


WristForceBiasDrift = _st.SensorBiasDriftScenario(
    name="A wrist strain gauge whose bias drifts with nearby joint heat",
    description="A steady 5 N reading while the bias drifts 0.2 N per minute for 15 minutes.",
    reading=Q_(5.0, "N"),  # scenario assumption: steady strain-gauge reading
    reported_sigma=Q_(0.05, "N"),  # scenario assumption: the estimator's reported bound
    drift_rate=Q_(0.2, "N/min"),  # scenario assumption: thermoelastic bias drift
    span=Q_(15.0, "min"),  # scenario assumption: drift interval
    excitation=Q_(40.0, "ms"),  # scenario assumption: deliberate contact break
)


ConveyorMugFrozenTransform = _st.FrozenTransformScenario(
    name="The mug's last transform republished after the wrist camera is lost",
    description="Fresh stamps every millisecond pass a 5 ms freshness check while the belt carries the mug for 100 ms.",
    site=_Embodied.Scenario.WarehouseAisle,
    republish_period=Q_(1.0, "ms"),  # scenario assumption: middleware republish period
    freshness_threshold=Q_(5.0, "ms"),  # scenario assumption: publish-time freshness threshold
    loss_time=Q_(10.0, "ms"),  # scenario assumption: the camera stops producing evidence
    window=Q_(100.0, "ms"),  # scenario assumption: elapsed time with the frozen transform
)


WarehousePermissionClearanceQuery = _st.ClearanceQueryScenario(
    name="Sphere-hull clearance queries on the warehouse permission controller",
    description="The 100 us collision allocation over 50 hulls, the SRAM read budget, reads per "
    "representation at 2, 5, and 10 cm, and the stopping-corridor snapshot.",
    site=_Embodied.Scenario.WarehouseAisle,
    collision_allocation=Q_(100.0, "us"),  # scenario assumption: collision checks within the permission tick
    hulls=50,  # scenario assumption: sphere hulls covering the machine
    fewest_hulls=20,  # scenario assumption: fewest evaluation spheres considered
    hull_radius=Q_(0.25, "m"),  # scenario assumption: sphere-hull radius
    esdf_cell_bytes=Q_(8, "B"),  # scenario assumption: 4 B distance, 2 B weight, 2 B evidence epoch
    occupancy_cell_bytes=Q_(4, "B"),  # scenario assumption: 2 B log-odds, 2 B evidence epoch
    map_edge=Q_(10.0, "m"),  # scenario assumption: host-map edge for the octree depth
    slab_height=Q_(2.0, "m"),  # scenario assumption: snapshot height covering the arm's reach
    pitches=[Q_(2, "cm"), Q_(5, "cm"), Q_(10, "cm")],  # scenario assumption: voxel pitches compared
    trilinear_reads=8,  # corner cells of one trilinear interpolation
)


WorkcellHostMapSizing = _st.HostMapSizingScenario(
    name="A 10 m by 10 m by 3 m workcell mapped at 1 cm",
    description="A dense voxel grid against a hashed 8-cubed-block band around 120 square meters of surface.",
    extent=[Q_(10.0, "m"), Q_(10.0, "m"), Q_(3.0, "m")],  # scenario assumption: workcell extent
    pitch=Q_(0.01, "m"),  # scenario assumption: manipulation resolution
    voxel_bytes=Q_(8, "B"),  # scenario assumption: distance, weight, and epoch payload
    surface_area=Q_(120.0, "m**2"),  # scenario assumption: obstacle surface area
    band_half_width=Q_(0.04, "m"),  # scenario assumption: truncation band half-width
    block_edge=8,  # voxels per block edge, the nvblox layout
)


WarehouseDepthRaymarchTraffic = _st.RaymarchTrafficScenario(
    name="One VGA depth stream ray-marched into a 2 cm map on the warehouse host SoC",
    description="Naive per-voxel read-modify-write traffic against the SoC's peak DRAM bandwidth.",
    site=_Embodied.Scenario.WarehouseAisle,
    width_px=640,  # scenario assumption: VGA depth stream
    height_px=480,  # scenario assumption: VGA depth stream
    frame_rate=Q_(30, "Hz"),  # scenario assumption: depth frame rate
    ray_length=Q_(2.0, "m"),  # scenario assumption: mean marched distance
    pitch=Q_(0.02, "m"),  # scenario assumption: voxel pitch
    bytes_per_step=Q_(8, "B"),  # scenario assumption: read-modify-write traffic per voxel, low end
)


WarehouseFoundationEncoderAtCameraRate = _st.EncoderRateScenario(
    name="A 3 TFLOP foundation encoder on every warehouse navigation frame",
    description="Sustained rate the encoder would need against the host SoC's dense FP16 peak, "
    "and the semantic keyframe rates of the multi-rate alternative.",
    site=_Embodied.Scenario.WarehouseAisle,
    flop_per_frame=Q_(3.0, "TFLOP"),  # scenario assumption: encoder cost per frame
    keyframe_rate_low=Q_(1.0, "Hz"),  # scenario assumption: semantic keyframe rate, low end
    keyframe_rate_high=Q_(5.0, "Hz"),  # scenario assumption: semantic keyframe rate, high end
)
