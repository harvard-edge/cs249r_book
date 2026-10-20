#!/usr/bin/env python3
"""Every number on the Lecture 8 slides and notes (Sensor Perception), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* machine and its site record, the
Sensors.* lidars, Literature.*) and the calculations (Scenarios.*, each backed
by a solver in engine/solvers/perception.py, embodied.py, or realtime.py); the
chapter reads the same scenarios. This file only selects values, converts
units, and formats them; it types no numbers. Each value is recorded with v()
under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports it for the scenario results and the
measured samples.

Measured data in data/, with sources in SOURCES.yml:
  kitti/: KITTI raw drive 2011_09_26_0001 (Geiger et al., IJRR 2013;
    CC BY-NC-SA 3.0): host-clock timestamp logs of the Velodyne HDL-64E and
    four cameras (extract), and sync sweep 0 with its OXTS record and the
    color camera's frame 0.
  mlperf_v3.1_orin_retinanet_singlestream_summary.txt: MLPerf Inference v3.1,
    closed, NVIDIA Jetson AGX Orin 64G, RetinaNet INT8, SingleStream, run 1.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Embodied, Scenarios, Sensors, Literature = (mlsysim.Embodied, mlsysim.Scenarios,
                                            mlsysim.Sensors, mlsysim.Literature)
Q_ = mlsysim.core.units.Q_

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
HDL = Sensors.LiDAR.Velodyne_HDL_64E

AGE = Scenarios.WarehouseNavCameraAge.result
SHUTTER = Scenarios.WarehouseNavCameraRollingShutter.result
SHUTTER_AISLE = Scenarios.WarehouseNavCameraRollingShutterAisleSpeed.result
TIMING = Scenarios.WarehouseRackEndTimingAttitude.result
MOUNT = Scenarios.WarehouseLongAisleMountError
SWEEP = Scenarios.WarehouseRackEndLidarAge.result
STEREO = Scenarios.WarehouseStereoRigDepthError
PHASE = Scenarios.VisualServoDelayPhaseMargin
INSET = Scenarios.ClearanceInsetConformalRank
CLOCK = Scenarios.WarehouseLidarClockConversion
FOG = Scenarios.DenseFogVisibility
LINK = Scenarios.AluminumLinkWarming
LEASE = Scenarios.WarehouseChunkLease
BODY = Scenarios.WarehouseBaseStopBodyTerms.result
FULL = Scenarios.WarehouseBaseStopFullDelay.result

DATA = Path(__file__).resolve().parent / "data"


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


# Module-level values make_figs.py draws with (all from MLSysIM).
V = m(RM.base.max_velocity, "m/s")                        # mm per ms
STAGES = [(name, m(t, "ms"), m(cum, "ms")) for name, t, cum in AGE.stages]
T_AGE = m(AGE.age, "ms")
T_ARRIVAL = m(AGE.arrival_offset, "ms")
T_CAPTURE = m(AGE.capture_before_compute, "ms")
T_EXP = m(RM.nav_camera.exposure_time, "ms")
T_READ = m(RM.nav_camera.readout_time, "ms")
ROWS = RM.nav_camera.resolution_height
D_LOC = m(AISLE.delta_loc, "mm")
SIDE = m(AISLE.side_clearance, "mm")
SYNC_BOUND = m(AISLE.clock_conversion_bound, "ms")

print("== The running machine at the rack end")
v("v_drive", m(RM.base.max_velocity, "m/s"), "{:.1f}", "drive limit, m/s")
v("d_per_ms", m(BODY.travel_over(Q_(1, "ms")), "mm"), "{:.1f}", "travel per millisecond, mm")
v("d_clear", m(AISLE.d_clear, "m"), "{:.2f}", "rack-end clear distance, m")
v("d_clear_mm", m(AISLE.d_clear, "mm"), "{:,.0f}", "the same, mm")
v("a_brake", m(AISLE.a_brake, "m/s**2"), "{:.0f}", "credible deceleration, m/s^2")
v("delta_loc", D_LOC, "{:.0f}", "localization allowance, mm")
v("delta_margin", m(AISLE.delta_margin, "mm"), "{:.0f}", "protective clearance, mm")
v("side_clearance", SIDE, "{:.0f}", "side clearance, mm")
v("t_onset", m(AISLE.t_brake_onset, "ms"), "{:.0f}", "brake onset, ms")
v("yaw_rate", m(AISLE.yaw_rate_turn, "rad/s"), "{:.1f}", "yaw rate turning into the aisle, rad/s")
v("v_aisle", m(AISLE.v_aisle, "m/s"), "{:.1f}", "chosen aisle speed, m/s")

print("== Recall: stopping-budget rows 1 to 4 (Scenarios.WarehouseChunkLease)")
lease = LEASE.result
v("d_braking", m(BODY.braking_travel, "mm"), "{:.1f}", "braking travel, mm")
v("d_overhead", m(BODY.overhead, "mm"), "{:.0f}", "localization plus protective clearance, mm")
v("d_onset", m(BODY.delay_travel, "mm"), "{:.0f}", "brake-onset travel, mm")
v("t_lease_path", m(lease.lease_path, "ms"), "{:.0f}", "lease plus tick plus bus, ms")
v("d_lease_path", m(lease.lease_path_travel, "mm"), "{:.0f}", "lease-path travel, mm")
v("d_rows14", m(lease.running_total, "mm"), "{:,.1f}", "rows 1 to 4, mm")
v("d_spare14", m(lease.spare, "mm"), "{:.1f}", "clear distance left before perception, mm")

print("== Hook: Williston (NTSB HAR-17/02)")
mph = Q_(Literature.IncidentReports.WillistonImpactSpeedMph, "mph")
v("williston_mph", m(mph, "mph"), "{:.0f}", "speed just before impact, mph")
v("williston_mps", m(mph, "m/s"), "{:.1f}", "the same, m/s")

print("== Capture epoch: exposure midpoint, arrival stamp, trigger stamp")
v("t_exp", T_EXP, "{:.0f}", "navigation-camera exposure, ms")
v("d_exposure", m(AGE.exposure_travel, "mm"), "{:.0f}", "travel during the exposure, mm")
v("t_readout", T_READ, "{:.1f}", "rolling readout, ms")
v("t_arrival", T_ARRIVAL, "{:.1f}", "exposure midpoint to DMA completion, ms")
v("d_arrival", m(AGE.arrival_travel, "mm"), "{:.1f}", "arrival-stamp travel, mm")
v("t_trigger", m(AGE.trigger_offset, "ms"), "{:.0f}", "trigger stamp early by, ms")
v("d_trigger", m(AGE.trigger_travel, "mm"), "{:.0f}", "trigger-stamp travel, mm")
v("t_link_dma", m(AISLE.t_transport + AISLE.t_dma, "ms"), "{:.1f}", "link plus DMA, ms")
v("misorient", m(TIMING.misorientation, "mrad"), "{:.1f}", "misorientation per ms of jitter, mrad")

print("== KITTI 2011_09_26_drive_0001: host-clock timestamp logs (measured)")


def kitti_ms(rel):
    """Timestamps in a KITTI log, in ms after the drive's first Velodyne start stamp."""
    lines = [ln for ln in (DATA / "kitti" / rel).read_text().splitlines() if ln.strip()]
    stamps = np.array([np.datetime64(ln.replace(" ", "T")) for ln in lines])
    return (stamps - KITTI_T0) / np.timedelta64(1, "ms")


EX = "extract"
KITTI_T0 = np.datetime64((DATA / "kitti" / EX / "velodyne_points" / "timestamps_start.txt")
                         .read_text().split("\n")[0].replace(" ", "T"))
K_VS = kitti_ms(f"{EX}/velodyne_points/timestamps_start.txt")
K_VF = kitti_ms(f"{EX}/velodyne_points/timestamps.txt")
K_VE = kitti_ms(f"{EX}/velodyne_points/timestamps_end.txt")
K_CAM = {d.name: kitti_ms(f"{EX}/{d.name}/timestamps.txt") for d in sorted((DATA / "kitti" / EX).glob("image_0*"))}


def cam_offsets(c):
    """Camera stamp minus the nearest Velodyne forward-facing (trigger) stamp, ms,
    for camera frames that fall inside the logged sweeps."""
    t = K_CAM[c]
    t = t[(t > K_VS[0]) & (t < K_VE[-1])]
    j = np.array([np.argmin(abs(K_VF - x)) for x in t])
    return t - K_VF[j]


GRAY, COLOR = "image_00", "image_02"  # gray left and color left cameras (KITTI devkit)
K_OFF = {c: cam_offsets(c) for c in K_CAM}
K_SWEEP = K_VE - K_VS
v("kitti_sweeps", len(K_VS), "{:,.0f}", "sweeps logged")
v("kitti_sweep_ms", np.median(K_SWEEP), "{:.0f}", "sweep duration, median, ms")
v("kitti_fwd", np.median(K_VF - K_VS), "{:.1f}", "forward-facing after sweep start, median, ms")
v("kitti_color_off", np.median(K_OFF[COLOR]), "{:.1f}", "color stamp after trigger, median, ms")
v("kitti_gray_off", np.median(K_OFF[GRAY]), "{:.1f}", "gray stamp after trigger, median, ms")
gap = np.median(K_OFF[GRAY]) - np.median(K_OFF[COLOR])
v("kitti_gap", gap, "{:.1f}", "gray minus color, ms")
v("kitti_gap_whole", gap, "{:.0f}", "the same, whole ms")
for key, series, label in (("lidar", K_VF, "Velodyne forward"), ("color", K_CAM[COLOR], "color camera"),
                           ("gray", K_CAM[GRAY], "gray camera")):
    d = np.diff(series)
    v(f"kitti_per_{key}_min", d.min(), "{:.1f}", f"{label} stamp period, min, ms")
    v(f"kitti_per_{key}_max", d.max(), "{:.1f}", f"{label} stamp period, max, ms")
v("hdl_vfov", m(HDL.vertical_fov, "deg"), "{:.1f}", "HDL-64E vertical field of view, deg")

print("== KITTI sync sweep 0 and its OXTS record (measured)")
SYNC_S = kitti_ms("sync/timestamps_start.txt")
SYNC_E = kitti_ms("sync/timestamps_end.txt")
SWEEP0 = SYNC_E[0] - SYNC_S[0]                                   # ms
OX = [float(x) for x in (DATA / "kitti" / "sync" / "oxts_0000000000.txt").read_text().split()]
VF = OX[8]                     # forward velocity, m/s (OXTS dataformat field 9, vf)
_xyzr = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "<f4")])  # KITTI velodyne .bin record
PTS = np.fromfile(DATA / "kitti" / "sync" / "velo_0000000000.bin", dtype=_xyzr)
PTS = np.stack([PTS["x"], PTS["y"], PTS["z"], PTS["r"]], axis=1)
v("kitti_sweep0", SWEEP0, "{:.0f}", "sync sweep 0 duration, ms")
v("kitti_v", VF, "{:.1f}", "car forward speed, m/s")
v("kitti_kmh", m(Q_(VF, "m/s"), "km/h"), "{:.0f}", "car speed, km/h")
v("kitti_sweep0_travel", m(Q_(VF, "m/s") * Q_(SWEEP0, "ms"), "m"), "{:.2f}", "car travel during sweep 0, m")
v("kitti_points", len(PTS), "{:,.0f}", "returns in sweep 0")
v("kitti_gray_travel_cm", m(Q_(VF, "m/s") * Q_(np.median(K_OFF[GRAY]), "ms"), "cm"), "{:.0f}",
  "car travel during the gray stamps' delay, cm")

print("== Rolling-shutter rows (Scenarios.WarehouseNavCameraRollingShutter)")
v("rows", ROWS, "{:,.0f}", "rows, full-resolution mode")
v("t_row", m(SHUTTER.row_interval, "us"), "{:.2f}", "row interval, us")
v("d_skew", m(SHUTTER.skew, "mm"), "{:.1f}", "along-track skew, mm")
v("d_lateral", m(SHUTTER.lateral_skew, "mm"), "{:.1f}", "lateral skew, mm")
v("lateral_pct", 100 * SHUTTER.lateral_share, "{:.1f}", "lateral skew, % of side clearance")
v("d_skew_aisle", m(SHUTTER_AISLE.skew, "mm"), "{:.1f}", "skew at the aisle speed, mm")
v("f_imu", m(RM.imu.sample_rate, "Hz"), "{:,.0f}", "IMU sample rate, Hz")

print("== One lidar sweep (Scenarios.WarehouseRackEndLidarAge)")
v("f_lidar", m(RM.lidar.scan_rate, "Hz"), "{:.0f}", "lidar scan rate, Hz")
v("t_sweep", m(SWEEP.sweep_period, "ms"), "{:.0f}", "sweep period, ms")
v("d_sweep", m(SWEEP.sweep_travel, "mm"), "{:.0f}", "travel per sweep, mm")
v("rot_sweep", m(SWEEP.sweep_rotation, "mrad"), "{:.0f}", "turn per sweep, mrad")

print("== Frames and mount error (Scenarios.WarehouseRackEndTimingAttitude, WarehouseLongAisleMountError)")
v("angle_err", m(Scenarios.WarehouseRackEndTimingAttitude.angle_error, "deg"), "{:.0f}", "mount error, degrees")
v("d_deg_rack", m(TIMING.angle_lateral, "mm"), "{:.1f}", "per degree at the rack end, mm")
v("mount_range", m(MOUNT.distance, "m"), "{:.0f}", "long-aisle detection range, m")
v("d_deg_far", m(MOUNT.result.angle_lateral, "mm"), "{:.0f}", "per degree at that range, mm")
v("mount_over_side", MOUNT.result.angle_lateral_share, "{:.1f}", "over the side clearance")
v("link_len", m(LINK.length, "m"), "{:.0f}", "aluminum link, m")
v("link_dt", m(LINK.temperature_rise, "K"), "{:.0f}", "warming, K")
v("d_link", m(LINK.result.expansion, "mm"), "{:.2f}", "link growth, mm")

print("== The latency waterfall (Scenarios.WarehouseNavCameraAge)")
for (name, t, cum), key in zip(STAGES, ("exphalf", "readout", "link", "dma", "isp", "backbone", "ipc")):
    v(f"t_st_{key}", t, "{:g}", f"{name}, ms")
v("t_capture", T_CAPTURE, "{:.1f}", "half the exposure plus readout, ms")
v("t_through_isp", STAGES[4][2], "{:.1f}", "age after the ISP, ms")
v("t_age", T_AGE, "{:.1f}", "observation age at dispatch, ms")
v("d_age", m(AGE.age_travel, "mm"), "{:.1f}", "travel during the age, mm")
v("pct_backbone", 100 * AGE.backbone_share, "{:.0f}", "backbone share of the age, %")
v("pct_capture", 100 * AGE.capture_share, "{:.0f}", "capture share of the age, %")

print("== A measured detector: MLPerf v3.1 RetinaNet on Jetson AGX Orin (measured)")


def mlperf(key):
    for line in (DATA / "mlperf_v3.1_orin_retinanet_singlestream_summary.txt").read_text().splitlines():
        if line.startswith(key):
            return Q_(int(line.split(":")[1].strip()), "ns")
    raise KeyError(key)


RN_P99 = m(mlperf("99.00 percentile latency (ns)"), "ms")
v("rn_p99", RN_P99, "{:.2f}", "p99, ms")
v("rn_p99_short", RN_P99, "{:.1f}", "p99, ms, one decimal")
v("backbone_over_rn", ratio(AISLE.t_backbone, Q_(RN_P99, "ms")), "{:.1f}", "backbone allocation / p99")
v("capture_over_rn", ratio(AGE.capture_before_compute, Q_(RN_P99, "ms")), "{:.1f}", "capture / p99")

print("== Row 5 of the stopping budget (Scenarios.WarehouseBaseStopFullDelay)")
v("t_tau", m(AISLE.tau_delay, "ms"), "{:.1f}", "total pre-brake delay, ms")
v("t_rest", m(AISLE.t_admission_to_brake_onset, "ms"), "{:.0f}", "lease, tick, bus, onset, ms")
v("d_rows15", m(FULL.required_distance, "mm"), "{:,.1f}", "rows 1 to 5, mm")
v("d_spare15", m(FULL.spare, "mm"), "{:.1f}", "clear distance left, mm")

print("== Evidence keeps aging; a dropped frame")
v("t_sync", SYNC_BOUND, "{:.1f}", "clock-conversion bound, ms")
v("t_dispatch_limit", m(CLOCK.result.dispatch_age_limit, "ms"), "{:.1f}", "dispatch limit, ms")
v("f_frame", m(RM.nav_camera.frame_rate, "Hz"), "{:.0f}", "frame rate, Hz")
v("t_frame", m(AGE.frame_period, "ms"), "{:.1f}", "frame period, ms")
v("d_frame", m(AGE.frame_travel, "mm"), "{:.1f}", "one dropped frame, mm")
v("frame_pct", 100 * ratio(AGE.frame_travel, FULL.spare), "{:.1f}", "of the remainder, %")

print("== Batching four frames (Scenarios.WarehouseNavCameraAge)")
v("batch_words", AGE.batch_frames, "words", "frames batched")
v("held_frames_words", len(AGE.batch_waits), "words", "frame periods the oldest waits")
v("t_batch_wait", m(AGE.batch_wait, "ms"), "{:.0f}", "oldest frame held, ms")
v("d_batch_wait", m(AGE.batch_wait_travel, "mm"), "{:.0f}", "travel while held, mm")
v("t_batched", m(AGE.batched_age, "ms"), "{:.1f}", "batched age, ms")

print("== Phase margin (Scenarios.VisualServoDelayPhaseMargin)")
phase = PHASE.result
v("pm_deg", m(PHASE.phase_margin, "deg"), "{:.0f}", "phase margin, degrees")
v("pm_rad", m(phase.phase_margin, "rad"), "{:.3f}", "phase margin, rad")
v("wc", m(PHASE.crossover, "rad/s"), "{:.0f}", "crossover, rad/s")
v("f_c", m(phase.crossover_frequency, "Hz"), "{:.1f}", "crossover, Hz")
v("t_pm", m(phase.delay_ceiling, "ms"), "{:.1f}", "delay ceiling, ms")
v("t_pm_used", m(PHASE.delay, "ms"), "{:.0f}", "perception delay, ms")
v("pm_used_pct", 100 * phase.share_used, "{:.1f}", "share spent, %")

print("== Stereo range wall (Scenarios.WarehouseStereoRigDepthError)")
stereo = STEREO.result
v("st_b", m(STEREO.baseline, "m"), "{:.2f}", "baseline, m")
v("st_f", STEREO.focal_length_px, "{:.0f}", "focal length, px")
v("st_sd", STEREO.disparity_sigma_px, "{:.1f}", "disparity sigma, px")
v("st_rack", m(stereo.sigma_target, "mm"), "{:.1f}", "sigma at the rack end, mm")
v("st_z_loc", m(stereo.range_at_loc_allowance, "m"), "{:.1f}", "range where sigma = allowance, m")
v("st_z_side", m(stereo.range_at_side_clearance, "m"), "{:.1f}", "range where sigma = side clearance, m")
v("st_z_side_half", m(stereo.range_at_side_clearance_half_baseline, "m"), "{:.1f}", "the same, half baseline, m")
v("st_range_ratio_words", stereo.range_ratio, "words", "far over near range")
v("st_growth", stereo.sigma_growth, "{:.0f}", "sigma growth")
v("st_near", m(STEREO.near_range, "m"), "{:.0f}", "near range, m")
v("st_far", m(STEREO.far_range, "m"), "{:.0f}", "far range, m")
v("st_sigma_near", m(stereo.sigma_near, "mm"), "{:.1f}", "sigma near, mm")
v("st_sigma_far", m(stereo.sigma_far, "mm"), "{:.1f}", "sigma far, mm")
v("inset_near", m(stereo.inset_near, "mm"), "{:.1f}", "shortcut inset near, mm")
v("inset_far", m(stereo.inset_far, "mm"), "{:.1f}", "shortcut inset far, mm")

print("== Ouster OS1-64 datasheet values (registry)")
os1 = RM.lidar
v("os1_prec_min", m(os1.range_precision_min, "mm"), "{:.0f}", "precision min, mm")
v("os1_prec_max", m(os1.range_precision_max, "mm"), "{:.0f}", "precision max, mm")
v("os1_prec_min_cm", m(os1.range_precision_min, "cm"), "{:.1f}", "precision min, cm")
v("os1_prec_max_cm", m(os1.range_precision_max, "cm"), "{:.0f}", "precision max, cm")
v("os1_acc", m(os1.range_accuracy, "mm"), "{:.0f}", "accuracy, Lambertian, mm")
v("os1_acc_retro_cm", m(os1.range_accuracy_retroreflective, "cm"), "{:.0f}", "accuracy, retroreflective, cm")
v("os1_ptp", m(os1.time_sync_accuracy, "ms"), "{:.0f}", "PTP accuracy bound, ms")
v("os1_drift_ppm", m(os1.internal_clock_drift, "ppm"), "{:.0f}", "internal clock drift, ppm")
v("drift_us", m(CLOCK.result.drift_per_interval, "us"), "{:.0f}", "drift per sync interval, us")
v("sync_interval", m(CLOCK.sync_interval, "s"), "{:.0f}", "sync interval, s")

print("== Clearance inset (Scenarios.ClearanceInsetConformalRank)")
rank = INSET.result
v("alpha_pct", 100 * INSET.alpha, "{:.0f}", "tail mass, %")
v("n_min", rank.min_count, "{:.0f}", "fewest residuals")
v("n_resid", INSET.calibration_count, "{:,.0f}", "residuals")
v("rank", rank.rank, "{:.0f}", "rank of the bound")
v("rank_top", rank.rank_from_top, "{:.0f}", "counted from the top")

print("== Metric veto rates")
v("f_vlm", m(RM.intent_rate, "Hz"), "{:.0f}", "vision-language model rate, Hz")
v("sweeps_per_vlm_words", ratio(RM.lidar.scan_rate, RM.intent_rate), "words", "lidar sweeps per model evaluation")
T_VLM = m(1 / RM.intent_rate, "ms")
T_SWEEP = m(SWEEP.sweep_period, "ms")

print("== Degradation (Scenarios.DenseFogVisibility; KITTI frame 0)")
from PIL import Image  # noqa: E402

IMG = np.asarray(Image.open(DATA / "kitti" / "sync" / "image_02_0000000000.png").convert("L"))
SAT = float((IMG >= np.iinfo(IMG.dtype).max).mean())
v("sat_pct", 100 * SAT, "{:.1f}", "pixels clipped at full scale, %")
v("pix_max", np.iinfo(IMG.dtype).max, "{:.0f}", "8-bit full scale")
v("fog_beta", m(FOG.extinction, "1/m"), "{:.2f}", "fog extinction, per m")
v("fog_eps", FOG.contrast_threshold, "{:.2f}", "contrast threshold")
v("d_fog", m(FOG.result.visibility_range, "m"), "{:.0f}", "visibility range, m")


if __name__ == "__main__":
    write(__file__)
