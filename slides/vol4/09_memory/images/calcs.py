#!/usr/bin/env python3
"""Every number on the Lecture 9 slides and notes (Spatial Memory), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* machine and its site record, the
Literature.RobotPerception and vol4/09_memory/data/kitti_odometry_leaderboard.csv figures) and the
calculations (Scenarios.*, each backed by a solver in
engine/solvers/estimation.py or mapping.py); the chapter reads the same
scenarios. This file only selects values, converts units, and formats them;
it types no numbers. Each value is recorded with v() under the key the deck
prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports it for the scenario results.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Embodied, Scenarios, Literature = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature
Q_ = mlsysim.core.units.Q_

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
LIT = Literature.RobotPerception

AGE = Scenarios.WarehouseNavCameraAge.result
FUSION = Scenarios.WarehouseRackEndRangeFusion
FUSED = FUSION.result
CONSISTENCY = Scenarios.RackEndMisdatedUpdateConsistency
BUFFER = Scenarios.WarehouseObservationReplayBuffer.result
IMU = Scenarios.WarehouseImuOneSecondCoast
TRACK = Scenarios.WarehouseTrackConfirmationDelay.result
CLOCK = Scenarios.WarehouseLidarClockConversion.result
EPOCHS = Scenarios.ArmSensorEpochGap
STALE = Scenarios.LatchSlotStaleWristTransform
LATCH = Scenarios.LatchSlotFingerBelief
LATCH_TIGHT = Scenarios.LatchSlotFingerBeliefTightSlot
UMB = Scenarios.WarehouseUmbmarkOdometryDrift
KITTI = Scenarios.WarehouseKittiOdometryDrift
QUERY = Scenarios.WarehousePermissionClearanceQuery
HOST = Scenarios.WorkcellHostMapSizing
FREE = Scenarios.WarehouseRackEndFreeSpace
COWORKER = Scenarios.CoworkerBesideArmBelief
MUG = Scenarios.ConveyorMugHandleBelief
FROZEN = Scenarios.ConveyorMugFrozenTransform
BIAS = Scenarios.WristForceBiasDrift
SKEW = Scenarios.WarehouseSkewedRecordPair


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


print("== The running machine at the rack end")
v("v_drive", m(RM.base.max_velocity, "m/s"), "{:.1f}", "drive limit, m/s")
v("v_walk", m(AISLE.v_human_approach, "m/s"), "{:.1f}", "ISO 13855 walking speed, m/s")
v("d_clear", m(AISLE.d_clear, "m"), "{:.2f}", "rack-end clear distance, m")
v("d_clear_fig", m(AISLE.d_clear, "m"), "{:.1f}", "the same, figure label")
v("side_clearance", m(AISLE.side_clearance, "mm"), "{:.0f}", "side clearance, mm")
v("side_clearance_m", m(AISLE.side_clearance, "m"), "{:.2f}", "side clearance, m")
v("delta_loc", m(AISLE.delta_loc, "mm"), "{:.0f}", "localization bound, mm")
v("delta_loc_m", m(AISLE.delta_loc, "m"), "{:.2f}", "localization bound, m")
v("delta_margin", m(AISLE.delta_margin, "mm"), "{:.0f}", "protective clearance, mm")
v("delta_margin_m", m(AISLE.delta_margin, "m"), "{:.2f}", "protective clearance, m")
v("aisle_m", m(AISLE.aisle_length, "m"), "{:.0f}", "aisle length, m")
v("v_aisle", m(AISLE.v_aisle, "m/s"), "{:.1f}", "aisle speed, m/s")

print("== Recall: the observation age (Scenarios.WarehouseNavCameraAge)")
v("t_age", m(AGE.age, "ms"), "{:.1f}", "frame age at dispatch, ms")
v("d_age", m(AGE.age_travel, "mm"), "{:.1f}", "base travel during the age, mm")
v("t_frame", m(AGE.frame_period, "ms"), "{:.1f}", "navigation and wrist frame period, ms")

print("== Kalman predict and update (Scenarios.WarehouseRackEndRangeFusion)")
v("sig_prior", m(FUSION.prior_sigma, "mm"), "{:.0f}", "prior spread, mm")
v("sig_prior_fig", m(FUSION.prior_sigma, "mm"), "{:.1f}", "prior spread, mm, figure label")
v("speed_sigma", m(FUSION.speed_sigma, "m/s"), "{:.1f}", "odometry speed disturbance, m/s")
v("t_sweep", m(1 / RM.lidar.scan_rate, "ms"), "{:.0f}", "lidar sweep interval, ms")
v("sig_odo", m(FUSED.process_sigma, "mm"), "{:.0f}", "odometry spread over one sweep, mm")
v("sig_pred", m(FUSED.predicted_sigma, "mm"), "{:.1f}", "predicted spread, mm")
v("sig_lidar", m(RM.lidar.range_precision_max, "mm"), "{:.0f}", "OS1-64 range precision, worst case, mm")
v("lidar_bias", m(RM.lidar.range_accuracy, "mm"), "{:.0f}", "OS1-64 systematic accuracy offset, mm")
v("g_lidar", FUSED.first_gain, "{:.2f}", "lidar gain")
v("sig_l", m(FUSED.first_sigma, "mm"), "{:.1f}", "spread after lidar, mm")
v("sig_stereo", m(FUSED.second_sensor_sigma, "mm"), "{:.1f}", "stereo spread at the rack end, mm")
v("g_stereo", FUSED.second_gain, "{:.2f}", "stereo gain")
v("sig_s", m(FUSED.fused_sigma, "mm"), "{:.1f}", "fused spread, mm")
v("sig_stereo_x2", m(FUSION.comparison_scale * FUSED.second_sensor_sigma, "mm"), "{:.1f}",
  "checkpoint: stereo spread doubled, mm")
v("g_stereo_x2", FUSED.comparison_gain, "{:.2f}", "checkpoint: stereo gain")
v("sig_s_x2", m(FUSED.comparison_sigma, "mm"), "{:.1f}", "checkpoint: fused spread, mm")

print("== Misdated update and replay buffer")
v("misdate", m(FUSED.misdating_bias, "mm"), "{:.1f}", "arrival-stamp travel, mm")
v("shift", m(FUSED.misdated_shift, "mm"), "{:.1f}", "estimate pulled away, mm")
v("nsig_words", round(FUSED.misdated_shift_sigmas), "words", "error in reported standard deviations, about")
v("slots", BUFFER.slots, "{:.0f}", "ring-buffer slots")
v("f_imu_khz", m(RM.imu.sample_rate, "kHz"), "{:.0f}", "IMU sample rate, kHz")

print("== NIS gates (Scenarios.RackEndMisdatedUpdateConsistency)")
cons = CONSISTENCY.result
v("nis", cons.nis_expected, "{:.2f}", "expected NIS of the misdated update")
v("gate95", cons.gates[0], "{:.2f}", "95 percent gate")
v("gate99", cons.gates[1], "{:.2f}", "99 percent gate")
v("gate95_pct", 100 * CONSISTENCY.gate_confidences[0], "{:.0f}", "first gate level, percent")
v("gate99_pct", 100 * CONSISTENCY.gate_confidences[1], "{:.0f}", "second gate level, percent")
v("p95", cons.detection[0], "{:.2f}", "probability caught at the 95 percent gate")
v("p99", cons.detection[1], "{:.2f}", "probability caught at the 99 percent gate")
v("n_runs", CONSISTENCY.runs, "{:.0f}", "innovations averaged")
v("band_lo", cons.band_low, "{:.2f}", "honest mean-NIS band, low")
v("band_hi", cons.band_high, "{:.2f}", "honest mean-NIS band, high")
v("nees_band_pct", 100 * CONSISTENCY.confidence, "{:.0f}", "band confidence, percent")

print("== IMU coasting (Scenarios.WarehouseImuOneSecondCoast)")
imu = IMU.result
v("t_coast", m(IMU.duration, "s"), "{:.0f}", "coasting interval, s")
v("t_coast_words", m(IMU.duration, "s"), "words", "the same, in words")
v("gyro_bias", m(RM.imu.gyro_zero_rate_offset, "deg/s"), "{:.0f}", "BMI088 gyro zero-rate offset, deg/s")
v("acc_bias_mg", m(RM.imu.accel_zero_g_offset, "milligravity"), "{:.0f}", "BMI088 zero-g offset, mg")
v("gyro_noise_deg", m(imu.heading_noise, "deg"), "{:.3f}", "gyro-noise heading after the interval, deg")
v("imu_lateral", m(imu.lateral_bias, "mm"), "{:.1f}", "gyro-offset lateral error at the upright, mm")
v("acc_bias_mm", m(imu.position_bias, "mm"), "{:.1f}", "accelerometer-offset position error, mm")
v("acc_noise_mm", m(imu.position_noise, "mm"), "{:.1f}", "accelerometer-noise position error, mm")

print("== SORT confirmation (Scenarios.WarehouseTrackConfirmationDelay)")
v("min_hits_words", LIT.SortMinHits, "words", "SORT min_hits, in words")
v("held_words", TRACK.held_frames, "words", "frame periods withheld, in words")
v("min_hits", LIT.SortMinHits, "{:.0f}", "SORT min_hits")
v("max_age", LIT.SortMaxAge, "{:.0f}", "SORT max_age, frames")
v("sort_wait", m(TRACK.wait, "ms"), "{:.1f}", "wait after the first detection, ms")
v("sort_travel", m(TRACK.travel, "mm"), "{:.1f}", "travel at the drive limit meanwhile, mm")

print("== Epochs and clocks (Scenarios.WarehouseLidarClockConversion, ArmSensorEpochGap)")
v("eps_sync", m(AISLE.clock_conversion_bound, "ms"), "{:.1f}", "clock-conversion bound, ms")
v("d_sync", m(CLOCK.bound_travel, "mm"), "{:.1f}", "its travel at the drive limit, mm")
gap = EPOCHS.result
v("epoch_gap", m(gap.gap, "ms"), "{:.1f}", "camera-to-encoder gap, ms")
v("d_gap", m(gap.travel, "mm"), "{:.2f}", "tool travel across it, mm")
v("v_arm", m(EPOCHS.speed, "m/s"), "{:.1f}", "arm Cartesian speed limit, m/s")

print("== Stale edge at the latch slot (Scenarios.LatchSlotStaleWristTransform)")
stale = STALE.result
v("lever", m(STALE.lever, "m"), "{:.2f}", "wrist-to-fingertip lever arm, m")
v("omega", m(STALE.angular_rate, "rad/s"), "{:.2f}", "wrist rate, rad/s")
v("v_drift", m(LATCH.growth_rate, "mm/s"), "{:.0f}", "unseen drift bound, mm/s")
v("e0_slot", m(LATCH.initial_error, "mm"), "{:.1f}", "initial error, mm")
v("e0_slot_fig", m(LATCH.initial_error, "mm"), "{:.2f}", "the same, figure label")
v("r_tol", m(LATCH.tolerance, "mm"), "{:.0f}", "slot tolerance, mm")
v("e_drift", m(stale.drift_term, "mm"), "{:.2f}", "drift term, mm")
v("e_lever", m(stale.lever_term, "mm"), "{:.2f}", "lever-arm term, mm")
v("e_total", m(stale.total, "mm"), "{:.2f}", "total, mm")
v("over_pct", 100 * stale.over_tolerance, "{:.0f}", "over the tolerance, percent")
v("e_same", m(stale.same_epoch_error, "mm"), "{:.2f}", "same-epoch chain, mm")

print("== UMBmark (Scenarios.WarehouseUmbmarkOdometryDrift)")
umb = UMB.result
v("umb_side", m(UMB.square_side, "m"), "{:.0f}", "square side, m")
v("umb_path", m(umb.path_length, "m"), "{:.0f}", "path per run, m")
v("umb_uncal", m(UMB.return_error_uncalibrated, "mm"), "{:.0f}", "return error, uncalibrated, mm")
v("umb_cal", m(UMB.return_error_calibrated, "mm"), "{:.0f}", "return error, calibrated, mm")
v("umb_sigma", m(UMB.run_sigma, "mm"), "{:.0f}", "run-to-run spread, mm")
v("umb_speed", m(UMB.test_speed, "m/s"), "{:.1f}", "benchmark speed, m/s")
v("r_unc_pct", 100 * umb.rate_uncalibrated, "{:.1f}", "uncalibrated rate, percent")
v("r_cal_pct", 100 * umb.rate_calibrated, "{:.2f}", "calibrated rate, percent")
v("d_unc", m(umb.distance_uncalibrated, "m"), "{:.1f}", "uncalibrated: bound spent after, m")
v("t_unc", m(umb.time_uncalibrated, "s"), "{:.1f}", "uncalibrated: seconds at the aisle speed")
v("e_aisle_unc", m(umb.segment_error_uncalibrated, "mm"), "{:.1f}", "uncalibrated: drift over one aisle, mm")
v("aisle_over_bound", umb.segment_over_bound_uncalibrated, "{:.1f}", "uncalibrated: times the bound")
v("cal_gain", umb.calibration_gain, "{:.1f}", "calibration gain")
v("d_cal", m(umb.distance_calibrated, "m"), "{:.1f}", "calibrated: bound spent after, m")
v("speed_ratio", umb.speed_ratio, "{:.1f}", "production over lab speed")

print("== KITTI (Scenarios.WarehouseKittiOdometryDrift)")
kit = KITTI.result
v("kitti_n", kit.count, "{:.0f}", "ranked methods")
v("kitti_best", kit.best_pct, "{:.2f}", "best, percent")
v("kitti_laser", kit.laser_median_pct, "{:.2f}", "median lidar, percent")
v("kitti_stereo", kit.stereo_median_pct, "{:.2f}", "median stereo, percent")
v("kitti_d_best", m(kit.distance_best, "m"), "{:.1f}", "best: meters to spend the bound")
v("kitti_d_laser", m(kit.distance_laser_median, "m"), "{:.1f}", "median lidar: meters")
v("kitti_d_stereo", m(kit.distance_stereo_median, "m"), "{:.1f}", "median stereo: meters")

print("== Read budget and reads per query (Scenarios.WarehousePermissionClearanceQuery)")
q = QUERY.result
v("hulls", QUERY.hulls, "{:.0f}", "body spheres")
v("f_tick_khz", m(RM.permission_rate, "kHz"), "{:.0f}", "permission tick rate, kHz")
v("t_tick_ms", m(q.tick, "ms"), "{:.0f}", "permission tick, ms")
v("t_reflex", m(QUERY.collision_allocation, "us"), "{:.0f}", "collision allocation, us")
v("t_query", m(q.query_time, "us"), "{:.0f}", "per-sphere allocation, us")
v("sram_bw", m(RM.permission_mcu.memory.sram_bandwidth, "GB/s"), "{:.1f}", "permission MCU SRAM bandwidth, GB/s")
v("cell_b", m(QUERY.esdf_cell_bytes, "B"), "{:.0f}", "bytes per ESDF cell")
v("reads", q.reads_budget, "{:.0f}", "read budget per query")
v("wcet", m(AISLE.enforcer_wcet, "us"), "{:.0f}", "declared WCET of the permission job, us")
v("wcet_pct", 100 * q.wcet_share, "{:.0f}", "collision share of the WCET, percent")
v("spare_pct", 100 * q.tick_spare_share, "{:.0f}", "tick left for other work, percent")
v("r_hull", m(QUERY.hull_radius, "m"), "{:.2f}", "sphere radius, m")
v("reach", m(q.reach, "m"), "{:.2f}", "radius plus clearance, m")
v("map_edge", m(QUERY.map_edge, "m"), "{:.0f}", "host map edge, m")
v("trilinear", QUERY.trilinear_reads, "{:.0f}", "corner reads of one interpolation")
for cm, row in q.rows.items():
    v(f"pitch{cm}", cm, "{:.0f}", f"pitch, cm")
    v(f"occ{cm}", row.occupancy_reads, "{:,.0f}", f"{cm} cm: occupancy reads")
    v(f"oct{cm}", row.octree_reads, "{:,.0f}", f"{cm} cm: octree reads")
    v(f"depth{cm}", row.octree_depth, "{:.0f}", f"{cm} cm: octree depth")
    v(f"quant{cm}", m(row.quantization, "mm"), "{:.0f}", f"{cm} cm: quantization, mm")
    v(f"esdf{cm}_kb", m(row.esdf_bytes, "kB"), "{:,.0f}", f"{cm} cm: ESDF snapshot, kB")
v("esdf2_mb", m(q.rows[2].esdf_bytes, "MB"), "{:.2f}", "2 cm snapshot, MB")
v("sram_mb", m(RM.permission_mcu.memory.sram_capacity, "MB"), "{:.2f}", "permission MCU SRAM, MB")
v("slab", m(q.slab_volume, "m**3"), "{:.2f}", "stopping-corridor snapshot volume, m^3")
v("slab_h", m(QUERY.slab_height, "m"), "{:.0f}", "snapshot height, m")

print("== nvblox (Literature.RobotPerception, Millane et al. 2024, Table II)")
v("esdf_nvblox", LIT.NvbloxEsdfUpdateMs, "{:g}", "nvblox ESDF update, ms")
v("esdf_fiesta", LIT.FiestaEsdfUpdateMs, "{:g}", "Fiesta ESDF update, ms")
v("esdf_voxblox", LIT.VoxbloxEsdfUpdateMs, "{:g}", "voxblox ESDF update, ms")

print("== Host map (Scenarios.WorkcellHostMapSizing)")
host = HOST.result
lx, ly, lz = HOST.extent
v("ws_x", m(lx, "m"), "{:.0f}", "workcell length, m")
v("ws_y", m(ly, "m"), "{:.0f}", "workcell width, m")
v("ws_z", m(lz, "m"), "{:.0f}", "workcell height, m")
v("ws_m3", m(host.workspace_volume, "m**3"), "{:.0f}", "workcell volume, m^3")
v("host_pitch", m(HOST.pitch, "cm"), "{:.0f}", "host pitch, cm")
DENSE_EXP = math.floor(math.log10(host.dense_voxels))
v("dense_mant", host.dense_voxels / 10 ** DENSE_EXP, "{:.0f}", "dense voxels, mantissa")
v("dense_exp", DENSE_EXP, "{:.0f}", "dense voxels, power of ten")
v("dense_gb", m(host.dense_bytes, "GB"), "{:.1f}", "dense grid, GB")
v("surface", m(HOST.surface_area, "m**2"), "{:.0f}", "surface area, m^2")
v("band_cm", m(HOST.band_half_width, "cm"), "{:.0f}", "band half-width, cm")
v("band_pct", 100 * host.band_share, "{:.1f}", "band share of the volume, percent")
v("hashed_mb", m(host.hashed_bytes, "MB"), "{:.1f}", "hashed band, MB")
v("dense_over_hashed", host.ratio, "{:.0f}", "dense over hashed")
v("block_edge", HOST.block_edge, "{:.0f}", "voxels per block edge")

print("== Free-space expiry at the rack end (Scenarios.WarehouseRackEndFreeSpace)")
free = FREE.result
v("t_ahead", m(free.horizon_ahead, "ms"), "{:.2f}", "ahead, ms")
v("t_ahead_whole", m(free.horizon_ahead, "ms"), "{:.0f}", "ahead, whole ms")
v("t_ahead_s", m(free.horizon_ahead, "s"), "{:.1f}", "ahead, s")
v("t_abreast", m(free.horizon_abreast, "ms"), "{:.0f}", "abreast, ms")
v("depth_ahead_m", m(free.certified_depth_ahead, "m"), "{:.2f}", "certified depth ahead, m")
v("depth_ahead_mm", m(free.certified_depth_ahead, "mm"), "{:.0f}", "certified depth ahead, mm")
v("depth_abreast_m", m(free.certified_depth_abreast, "m"), "{:.0f}", "certified depth abreast, m")
v("ms_per_mm", m(free.time_per_depth, "ms/mm"), "{:.3f}", "ms per mm of depth")
v("d_age_eroded", m(free.depth_eroded_by_age, "mm"), "{:.1f}", "depth spent by the frame age, mm")
v("t_left", m(free.time_left_at_dispatch, "ms"), "{:.2f}", "left after the frame age, ms")
v("frames_left", free.frames_bridged, "{:.0f}", "frame periods left, about")
v("walk_intrusion", m(free.walker_intrusion, "m"), "{:.2f}", "walker during delay plus stop, m")
v("timeout_ms", m(FREE.uniform_timeout, "ms"), "{:.0f}", "one staleness timeout, ms")
v("timeout_refused_s", m(free.timeout_refused, "s"), "{:.2f}", "certified time the timeout refuses ahead, s")

print("== Latch slot (Scenarios.LatchSlotFingerBelief, LatchSlotFingerBeliefTightSlot)")
lat = LATCH.result
v("a_dist", m(LATCH.growth_acceleration, "mm/s**2"), "{:.0f}", "disturbance acceleration bound, mm/s^2")
v("a_half", m(lat.quadratic_coefficient, "mm/s**2"), "{:.0f}", "half of it, the dt^2 coefficient, mm/s^2")
v("slot_slack", m(LATCH.tolerance - LATCH.initial_error, "mm"), "{:.1f}", "tolerance less initial error, mm")
v("t_latch", m(lat.horizon, "ms"), "{:.0f}", "horizon, ms")
v("t_lin", m(lat.constant_velocity_horizon, "ms"), "{:.0f}", "constant-velocity horizon, ms")
v("e_at_lin", m(lat.error_at_constant_velocity_horizon, "mm"), "{:.2f}", "bounded error there, mm")
v("lin_over_pct", 100 * lat.overshoot, "{:.1f}", "over the tolerance, percent")
v("overrun", m(lat.overrun, "ms"), "{:.0f}", "blind overrun, ms")
v("r_tight", m(LATCH_TIGHT.tolerance, "mm"), "{:.2f}", "tight slot, mm")
v("t_tight", m(LATCH_TIGHT.result.horizon, "ms"), "{:.1f}", "tight-slot horizon, ms")
v("horizon_ratio", ratio(lat.horizon, LATCH_TIGHT.result.horizon), "{:.1f}", "horizon ratio")
v("tol_ratio", ratio(LATCH.tolerance, LATCH_TIGHT.tolerance), "{:.1f}", "tolerance ratio")

print("== Frozen transform (Scenarios.ConveyorMugFrozenTransform)")
frz = FROZEN.result
v("t_loss", m(FROZEN.loss_time, "ms"), "{:.0f}", "camera lost at, ms")
v("f_repub_khz", m(frz.republish_rate, "kHz"), "{:.0f}", "republish rate, kHz")
v("t_repub", m(FROZEN.republish_period, "ms"), "{:.0f}", "republish period, ms")
v("tau_max", m(FROZEN.freshness_threshold, "ms"), "{:.0f}", "freshness threshold, ms")
v("v_belt", m(AISLE.v_conveyor, "m/s"), "{:.2f}", "belt speed, m/s")
v("frozen_err", m(frz.error, "mm"), "{:.0f}", "mug moved, mm")
v("frozen_window", m(FROZEN.window, "ms"), "{:.0f}", "frozen window, ms")
v("grasp_tol", m(AISLE.grasp_tolerance, "mm"), "{:.0f}", "grasp tolerance, mm")

print("== Four beliefs (Scenarios.CoworkerBesideArmBelief and its tight variant)")
v("cw_e0", m(COWORKER.initial_error, "mm"), "{:.0f}", "coworker initial error, mm")
v("cw_tol", m(COWORKER.tolerance, "mm"), "{:.0f}", "coworker tolerance, mm")
v("t_cw", m(COWORKER.result.horizon, "ms"), "{:.0f}", "coworker horizon, ms")
tight = Scenarios.CoworkerBesideArmBeliefTightTolerance
v("cw_tol_tight", m(tight.tolerance, "mm"), "{:.0f}", "checkpoint: tighter tolerance, mm")
v("t_cw_tight", m(tight.result.horizon, "ms"), "{:.0f}", "checkpoint: coworker horizon, ms")

print("== Mug admission (Scenarios.ConveyorMugHandleBelief)")
mug = MUG.result
v("mug_e0", m(MUG.initial_error, "mm"), "{:.0f}", "initial error, mm")
v("v_belt_mm_ms", m(AISLE.v_conveyor, "mm/ms"), "{:.2f}", "belt, mm per ms")
v("v_belt_mm_ms_short", m(AISLE.v_conveyor, "mm/ms"), "{:.1f}", "belt, mm per ms, one decimal")
v("t_use", m(MUG.use_age, "ms"), "{:.0f}", "use age after capture, ms")
v("eps_sync_short", m(MUG.age_bound, "ms"), "{:.1f}", "conversion bound, ms")
v("age_upper", m(mug.use_age_upper, "ms"), "{:.1f}", "upper age, ms")
v("e_use", m(mug.error_at_use, "mm"), "{:.2f}", "error at use, mm")
v("mug_margin", m(mug.margin_at_use, "mm"), "{:.2f}", "margin, mm")
v("mug_left", m(mug.time_left, "ms"), "{:.1f}", "lifetime left, ms")
v("mug_horizon", m(mug.horizon, "ms"), "{:.0f}", "mug horizon, ms")
v("mug_latest", m(mug.latest_use_age, "ms"), "{:.1f}", "latest use age, ms")

print("== Force bias (Scenarios.WristForceBiasDrift)")
bias = BIAS.result
v("f_read", m(BIAS.reading, "N"), "{:.0f}", "reading, N")
v("f_sigma", m(BIAS.reported_sigma, "N"), "{:.2f}", "reported spread, N")
v("bias_rate", m(BIAS.drift_rate, "N/min"), "{:.1f}", "drift, N per minute")
v("bias_span", m(BIAS.span, "min"), "{:.0f}", "drift interval, minutes")
v("bias_shift", m(bias.shift, "N"), "{:.0f}", "bias shift, N")
v("f_true", m(bias.true_value, "N"), "{:.0f}", "true force, N")
v("grip_lost", 100 * bias.lost_share, "{:.0f}", "grip lost, percent")

print("== Skewed records (Scenarios.WarehouseSkewedRecordPair)")
sk = SKEW.result
v("skew", m(SKEW.skew, "ms"), "{:.0f}", "skew, ms")
v("v_close", m(sk.speed, "m/s"), "{:.1f}", "closing speed, m/s")
v("d_skew", m(sk.travel, "mm"), "{:.0f}", "envelope, mm")
v("sep_slack", m(sk.slack, "mm"), "{:.0f}", "slack, mm")
v("separation", m(SKEW.separation, "mm"), "{:.0f}", "separation, mm")


if __name__ == "__main__":
    write(__file__)
