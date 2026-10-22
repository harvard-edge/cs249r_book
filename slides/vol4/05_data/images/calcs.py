#!/usr/bin/env python3
"""Every number on the Lecture 5 slides and notes (Physical Data), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* machine and its site record, the
Literature registries for DROID, Open X-Embodiment, the LeRobot Hub,
ThriftyDAgger, 49 CFR 563, and the Mars Climate Orbiter report) and the
calculations (the Scenarios.* entries for physical demonstration data); the
chapter reads the same scenarios. This file only selects values, converts
units, and formats them; it types no numbers. Each value is recorded with v()
under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports it for the scenario objects and the
plotted magnitudes.

Run from this folder: python3 calcs.py
"""

import datetime as dt
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.embodied.datasheet import machine_card  # noqa: E402

Embodied, Scenarios, Literature = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature
ureg = mlsysim.core.units.ureg

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
DATA = Literature.RobotDataSystems
EDR = Literature.EventDataRecorders
MCO = Literature.SpaceflightAnomalies

ESS = Scenarios.PermissionTickTelemetryHour
BOUNDS = Scenarios.ImitationBoundsFiveSecondEpisode
BOUNDS10 = Scenarios.ImitationBoundsTenSecondEpisode
PAIR = Scenarios.LatchPairingLag
TIGHT = Scenarios.LatchPairingLagTightPlate
LOOP = Scenarios.TeleopCorrectionLoop
BYTES = Scenarios.WarehouseTeleopIngestion
CAMP = Scenarios.LatchDemonstrationCampaign
HUB = Scenarios.LeRobotHubDatasetCounts
OXE = Scenarios.OpenXEmbodimentPoolComposition
STATE = Scenarios.WarehouseManipulatorStateGrid
GRID = Scenarios.ContactAngleCoverage
TABLE = Scenarios.LatchScenarioTable
RECOVERY = Scenarios.LatchScenarioTableAfterRecovery
THIRD = Scenarios.LatchScenarioTableThirdSpeed
GRIP = Scenarios.GripperStrokeTransfer


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


print("== Hook: DROID")
v("droid_recal", float(DATA.DroidRecalibratedEpisodes), "{:,.0f}", "episodes recalibrated")
v("droid_traj", float(DATA.DroidTrajectories), "{:,.0f}", "trajectories")
v("droid_hours", float(DATA.DroidHours), "{:.0f}", "hours")
v("droid_scenes", float(DATA.DroidScenes), "{:.0f}", "scenes")

print("== Running case: the latch")
v("f_trip", m(AISLE.f_latch_tripwire, "N"), "{:.0f}", "contact tripwire, N")
v("f_latch_limit", m(AISLE.f_latch_limit, "N"), "{:.0f}", "latch damage limit, N")
v("v_guard", m(AISLE.v_latch_approach, "m/s"), "{:.2f}", "guarded approach, m/s")
v("v_fast", m(AISLE.v_latch_high, "m/s"), "{:.2f}", "fast approach, m/s")
v("tick_khz", m(RM.permission_rate, "kHz"), "{:.0f}", "permission tick, kHz")
v("chunk_hz", m(RM.chunk_rate, "Hz"), "{:.0f}", "chunk renewal rate, Hz")

print("== Compounding error bounds")
v("s_steps", BOUNDS.horizon_steps, "{:.0f}", "horizon S, steps")
v("episode_s", m(BOUNDS.episode, "s"), "{:.0f}", "episode, s")
v("ctrl_hz", m(BOUNDS.control_rate, "Hz"), "{:.0f}", "control rate, Hz")
v("eps", BOUNDS.per_step_error, "{:.3f}", "per-step error")
v("u_rec", BOUNDS.recoverability, "{:.0f}", "recoverability constant u")
v("bound_open", BOUNDS.open_loop, "{:.1f}", "open loop S eps")
v("bound_bc", BOUNDS.cloning, "{:.0f}", "cloning S^2 eps")
v("bound_dagger", BOUNDS.aggregation, "{:.1f}", "aggregation u S eps")
v("eps_vacuous", BOUNDS.vacuous_error, "{:.3f}", "vacuous above eps = 1/S")
v("s10_steps", BOUNDS10.horizon_steps, "{:,.0f}", "horizon of a 10 s episode, steps")
v("eps10_vacuous", BOUNDS10.vacuous_error, "{:.3f}", "its vacuous threshold")
assert BOUNDS.cloning + BOUNDS.cloning == BOUNDS.horizon_steps, "notes say the cloning bound is half the episode maximum"

print("== Effective sample size")
v("ess_records", ESS.records, "{:,.0f}", "records in an hour")
v("ess_records_m", ESS.records / 10 ** 6, "{:.1f}", "the same, millions")
v("ess_n", round(ESS.effective_samples, -2), "{:,.0f}", "effective samples (nearest hundred)")
v("rho", ESS.lag1_correlation, "{:.3f}", "lag-one correlation")
v("tick_ms", m(ESS.record_interval, "ms"), "{:.0f}", "record interval, ms")

print("== LeRobot Hub")
v("hub_2024", HUB.count_at("2024-12"), "{:,.0f}", "datasets, end of 2024")
v("hub_2025", HUB.count_at("2025-12"), "{:,.0f}", "datasets, end of 2025")
v("hub_now", HUB.latest_count, "{:,.0f}", "datasets, latest month")
v("hub_now_month", dt.datetime.strptime(HUB.latest_month, "%Y-%m"), "{:%B %Y}", "latest month")
v("hub_growth", HUB.growth_since("2024-12"), "{:.0f}", "growth since end of 2024")
v("hub_counted", HUB.counted, "{:,.0f}", "datasets with an episode count")
v("hub_small", HUB.small_episodes, "{:.0f}", "small threshold, episodes")
v("hub_large", HUB.large_episodes, "{:.0f}", "large threshold, episodes")
v("hub_small_pct", 100 * HUB.small_share, "{:.0f}", "share under the small threshold, %")
v("hub_large_pct", 100 * HUB.large_share, "{:.0f}", "share at or above the large threshold, %")
v("hub_median_lo", HUB.median_bin_low, "{:.0f}", "median bin, low edge")
v("hub_median_hi", HUB.median_bin_high - 1, "{:.0f}", "median bin, high edge")

print("== LeRobot example artifact (lerobot/aloha_static_coffee)")
v("aloha_ep", float(DATA.AlohaStaticCoffeeEpisodes), "{:.0f}", "total_episodes")
v("aloha_fps", float(DATA.AlohaStaticCoffeeFps), "{:.0f}", "fps")
v("aloha_dim", float(DATA.AlohaStaticCoffeeStateDim), "{:.0f}", "state, effort, action width")

print("== Pairing lag (Napkin Math 5.1)")
v("lag_ms", m(PAIR.lag, "ms"), "{:.0f}", "uncompensated lag, ms")
v("clear_mm", m(PAIR.clearance, "mm"), "{:.0f}", "latch clearance, mm")
v("v_transit", m(PAIR.transit_speed, "m/s"), "{:.2f}", "transit speed, m/s")
v("dx_transit", m(PAIR.displacement_transit, "mm"), "{:.1f}", "transit displacement, mm")
v("transit_x", PAIR.transit_over_clearance, "{:.2f}", "transit displacement over clearance")
v("dx_guard", m(PAIR.displacement_guarded, "mm"), "{:.2f}", "guarded displacement, mm")
v("dx_fast", m(PAIR.displacement_fast, "mm"), "{:.1f}", "fast displacement, mm")
v("ceil_guard", m(PAIR.lag_ceiling_guarded, "ms"), "{:.1f}", "guarded lag ceiling, ms")
v("ceil_fast", m(PAIR.lag_ceiling_fast, "ms"), "{:.0f}", "fast lag ceiling, ms")
v("skew_lo", m(PAIR.shutter_skew_low, "ms"), "{:.0f}", "rolling-shutter skew, fast readout, ms")
v("skew_hi", m(PAIR.shutter_skew_high, "ms"), "{:.0f}", "rolling-shutter skew, one frame, ms")
v("clear_tight", m(TIGHT.clearance, "mm"), "{:.0f}", "tighter clearance, mm")
v("ceil_tight", m(TIGHT.lag_ceiling_fast, "ms"), "{:.0f}", "tight-plate lag ceiling, ms")
v("tight_x", TIGHT.fast_over_clearance, "{:.1f}", "35 ms error over the tight clearance")
v("v_fast_mm_ms", m(AISLE.v_latch_high, "mm/ms"), "{:.1f}", "fast approach, mm per ms")

print("== Teleoperation loop")
v("t_disp", m(LOOP.display, "ms"), "{:.0f}", "display, ms")
v("t_hum", m(LOOP.human, "ms"), "{:.0f}", "human, ms")
v("t_cmd", m(LOOP.command, "ms"), "{:.0f}", "command, ms")
v("t_mech", m(LOOP.mechanical, "ms"), "{:.0f}", "mechanical, ms")
v("t_loop", m(LOOP.loop, "ms"), "{:.0f}", "loop, ms")
v("t_issued", m(LOOP.command_issued, "ms"), "{:.0f}", "correction commanded at, ms")
v("hum_pct", 100 * LOOP.human_share, "{:.1f}", "human share, %")

print("== Open X-Embodiment")
v("oxe_n", OXE.datasets, "{:.0f}", "datasets with an episode count")
v("oxe_total", OXE.total, "{:,.0f}", "episodes")
v("oxe_total_m", OXE.total / 10 ** 6, "{:.1f}", "episodes, millions")
v("oxe_rate10", 10, "{:.0f}", "the 10 Hz category, Hz")
v("oxe_10hz_pct", 100 * OXE.rate_share(10.0), "{:.0f}", "share at 10 Hz, %")
v("oxe_norate_pct", 100 * OXE.rate_share(None), "{:.0f}", "share with no declared rate, %")
v("oxe_fast_pct", 100 * OXE.at_or_above_threshold_share, "{:.0f}", "share at or above the chunk rate, %")
v("oxe_single_pct", 100 * OXE.morphology_share("Single Arm"), "{:.0f}", "single-arm share, %")
v("oxe_table_pct", 100 * OXE.table_top_share, "{:.0f}", "table-top share, %")
v("oxe_mobile", OXE.by_morphology["Mobile Manipulator"], "{:,.0f}", "mobile-manipulator episodes")
v("oxe_mobile_pct", 100 * OXE.morphology_share("Mobile Manipulator"), "{:.0f}", "mobile-manipulator share, %")
v("oxe_top_n_words", OXE.top_datasets, "words", "largest mobile datasets counted, spelled")
v("oxe_top2_pct", 100 * OXE.top_share_within("Mobile Manipulator"), "{:.0f}",
  "share of mobile episodes from the largest datasets, %")
ROWS = ("QT-Opt", "Language Table", "DROID", "ConqHose", "MobileALOHA")
for i, name in enumerate(ROWS):
    r = OXE.row(name)
    v(f"oxe_row{i}_hz", r["control_frequency_hz"], "{:.0f}", f"{name}: declared rate, Hz")
    v(f"oxe_row{i}_ep", r["episodes"], "{:,.0f}", f"{name}: episodes")
rates = [OXE.row(name)["control_frequency_hz"] for name in ROWS]
v("oxe_rows_hz_lo", min(rates), "{:.0f}", "lowest rate among the five rows, Hz")
v("oxe_rows_hz_hi", max(rates), "{:.0f}", "highest rate among the five rows, Hz")
v("oxe_rows_words", len(ROWS), "words", "rows shown, spelled")

print("== Yield and wear (Napkin Math 5.3)")
v("startup_min", m(CAMP.startup, "minute"), "{:.0f}", "startup, min")
v("calib_min", m(CAMP.calibration, "minute"), "{:.0f}", "calibration, min")
v("rest_min", m(CAMP.rest, "minute"), "{:.0f}", "rest, min")
v("active_min", m(CAMP.active, "minute"), "{:.0f}", "active window, min")
v("attempts", CAMP.attempts_per_session, "{:.0f}", "attempts per session")
v("t_att", m(CAMP.attempt, "s"), "{:.0f}", "attempt, s")
v("t_rst", m(CAMP.reset, "s"), "{:.0f}", "reset, s")
v("resets_min", m(CAMP.reset_time, "minute"), "{:.0f}", "resets per session, min")
v("attempts_min", m(CAMP.attempt_time, "minute"), "{:.0f}", "attempts per session, min (whole)")
v("fail_pct", 100 * CAMP.failure_rate, "{:.0f}", "trial failures, %")
v("qa_pct", 100 * CAMP.qa_rate, "{:.0f}", "quality-filter discards, %")
v("retention", CAMP.retention, "{:.2f}", "retention")
v("kept", CAMP.kept_per_session, "{:.2f}", "demonstrations kept per session")
v("kept_min", m(CAMP.kept_time, "minute"), "{:.2f}", "kept minutes")
v("kept_min_1", m(CAMP.kept_time, "minute"), "{:.1f}", "kept minutes, one decimal")
v("kept_pct", 100 * CAMP.kept_share, "{:.1f}", "kept share of the hour, %")
v("target", CAMP.target_kept, "{:,.0f}", "campaign target, kept demonstrations")
v("camp_attempts", CAMP.campaign_attempts, "{:,.0f}", "campaign attempts")
v("camp_h", m(CAMP.campaign_time, "hour"), "{:.1f}", "campaign cell time, h")
v("motion_h", m(CAMP.kept_motion_time, "hour"), "{:.1f}", "kept motion, h")
v("overhead_x", CAMP.overhead_factor, "{:.2f}", "cell time over kept motion")
v("gear_rev", CAMP.gear_reversals_per_attempt, "{:.0f}", "gear reversals per attempt")
v("cable_flex", CAMP.cable_flexes_per_attempt, "{:.0f}", "cable bends per attempt")
v("gear_cycles", CAMP.gear_cycles, "{:,.0f}", "campaign gear cycles")
v("cable_cycles", CAMP.cable_cycles, "{:,.0f}", "campaign cable bends")
v("gear_pct", 100 * CAMP.gear_life_used, "{:.1f}", "gear rated life used, %")
v("cable_pct", 100 * CAMP.cable_life_used, "{:.1f}", "cable rated life used, %")
v("gear_life_m", CAMP.gear_rated_cycles / 10 ** 6, "{:.0f}", "gear rating, million cycles")
v("cable_life", CAMP.cable_rated_cycles, "{:,.0f}", "cable rating, bends")
v("campaigns_words", CAMP.repeat_campaigns, "words", "campaigns in the wear comparison")
v("cable_pct_3", 100 * CAMP.repeat_cable_life_used, "{:.0f}", "those campaigns, cable life used, %")

print("== Bytes (Napkin Math 5.2)")
v("nav_w", RM.nav_camera.resolution_width, "{}", "navigation camera width, px")
v("nav_h", RM.nav_camera.resolution_height, "{}", "navigation camera height, px")
v("nav_fps", m(RM.nav_camera.frame_rate, "Hz"), "{:.0f}", "navigation camera rate, Hz")
v("bw_nav", m(BYTES.nav_rate, "MB/s"), "{:,.0f}", "navigation camera, MB/s")
v("nav_over_sata", BYTES.nav_over_slow_drive, "{:.1f}", "navigation camera over SATA")
v("bw_total", m(BYTES.total_rate, "MB/s"), "{:,.0f}", "all streams, MB/s")
v("bw_total_gb", m(BYTES.total_rate, "GB/s"), "{:.2f}", "all streams, GB/s")
v("tb_hour", m(BYTES.storage_per_hour, "TB"), "{:.2f}", "storage per hour, TB")
v("compress_words", BYTES.compression_ratio, "words", "compression ratio, spelled")
v("compress_x", BYTES.compression_ratio, "{:.0f}", "compression ratio")
v("bw_comp", m(BYTES.compressed_rate, "MB/s"), "{:.0f}", "compressed, MB/s")
v("axes_words", RM.servo_axes, "words", "logged axes, spelled")
v("bw_kin", m(BYTES.kinematic_rate, "MB/s"), "{:.2f}", "kinematic stream, MB/s")
v("kin_pct", 100 * BYTES.kinematic_share, "{:.3f}", "kinematic share, %")
v("sata", m(BYTES.slow_drive_bandwidth, "MB/s"), "{:,.0f}", "SATA SSD, MB/s")
v("nvme", m(BYTES.fast_drive_bandwidth, "MB/s"), "{:,.0f}", "NVMe Gen4, MB/s")

print("== Contact-angle grid and the curse of dimensionality")
a_lo, a_hi = m(GRID.angle_min, "degree"), m(GRID.angle_max, "degree")
assert -a_lo == a_hi, "the slide prints the angle range as plus or minus one value"
assert GRID.records == 10 ** 6, "the slides say a million records"
v("ang_max", a_hi, "{:.0f}", "angle range, plus or minus, deg")
v("ang_bin", m(GRID.angle_bin, "degree"), "{:.0f}", "angle bin, deg")
v("ang_bin_words", m(GRID.angle_bin, "degree"), "words", "the same, spelled")
v("force_max", m(GRID.force_max, "N"), "{:.0f}", "force range top, N")
v("force_bin", m(GRID.force_bin, "N"), "{:.0f}", "force bin, N")
v("force_bin_words", m(GRID.force_bin, "N"), "words", "the same, spelled")
v("grid_cells", GRID.cells, "{:.0f}", "grid cells")
v("grid_hz", m(GRID.record_rate, "Hz"), "{:.0f}", "record rate, Hz")
c_alo, c_ahi, c_flo, c_fhi, c_rms = GRID.archives["cautious"]
e_alo, e_ahi, e_flo, e_fhi, e_rms = GRID.archives["exploratory"]
v("caut_ang", m(c_ahi, "degree"), "{:.0f}", "cautious angle, plus or minus, deg")
v("caut_f_lo", m(c_flo, "N"), "{:.0f}", "cautious force low, N")
v("caut_f_hi", m(c_fhi, "N"), "{:.0f}", "cautious force high, N")
v("caut_cells", GRID.occupied("cautious"), "{:.0f}", "cautious cells")
v("caut_pct", 100 * GRID.share("cautious"), "{:.1f}", "cautious share, %")
v("expl_f_lo", m(e_flo, "N"), "{:.0f}", "exploratory force low, N")
v("expl_f_hi", m(e_fhi, "N"), "{:.0f}", "exploratory force high, N")
v("expl_cells", GRID.occupied("exploratory"), "{:.0f}", "exploratory cells")
v("expl_pct", 100 * GRID.share("exploratory"), "{:.0f}", "exploratory share, %")
v("rms_caut", m(c_rms, "rad"), "{:.3f}", "cautious held-out error, rad")
v("rms_expl", m(e_rms, "rad"), "{:.3f}", "exploratory held-out error, rad")
v("state_dims", STATE.dims, "{:.0f}", "state dimensions")
v("state_bins", STATE.bins, "{:.0f}", "bins per axis")
v("state_cells_exp", STATE.dims * (STATE.bins // 10), "{:.0f}", "cells, power of ten (bins per axis is 10)")

print("== Scenario table")
v("tab_cells", TABLE.cells, "{:.0f}", "cells the door requires")
v("tab_supported", TABLE.supported, "{:.0f}", "cells the archive fills")
v("tab_absent", TABLE.absences, "{:.0f}", "known absences")
v("tab_offsets_words", len(TABLE.offsets), "words", "plate offsets, spelled")
v("tab_speeds_words", len(TABLE.speeds), "words", "approach speeds, spelled")
v("tab_supported_words", TABLE.supported, "words", "cells filled, spelled")
v("tab_absent_words", TABLE.absences, "words", "known absences, spelled")
v("tab_cells_words", TABLE.cells, "words", "cells, spelled")
v("plate_mm", m(max(TABLE.offsets), "mm"), "{:.0f}", "plate offset, mm")
v("rec_supported", RECOVERY.supported, "{:.0f}", "cells filled after latch-recovery-02")
v("rec_absent", RECOVERY.absences, "{:.0f}", "known absences after latch-recovery-02")
v("rec_absent_words", RECOVERY.absences, "words", "the same, spelled")
v("third_offsets", len(THIRD.offsets), "{:.0f}", "plate offsets")
v("third_speeds", len(THIRD.speeds), "{:.0f}", "approach speeds with the third")
v("third_offsets_words", len(THIRD.offsets), "words", "the same, spelled")
v("third_speeds_words", len(THIRD.speeds), "words", "the same, spelled")
v("third_cells", THIRD.cells, "{:.0f}", "cells with the third speed")
v("third_absent", THIRD.absences, "{:.0f}", "known absences for the original archive")

print("== Interventions")
# The machine card prints this value (half away from zero); the chapter quotes the card.
CARD = {r.key: r for r in machine_card()}
assert CARD["t_takeover_out"].value == AISLE.t_takeover_out_of_loop
v("takeover_s", CARD["t_takeover_out"].value_str.split()[0], "{}", "out-of-loop takeover budget, s (as the card prints it)")
v("lookback_s", m(AISLE.pre_trigger_window, "s"), "{:.0f}", "lookback window, s")
v("lookback_ticks", m(AISLE.pre_trigger_window * RM.permission_rate, ""), "{:,.0f}", "ticks in the window")
v("edr_s", float(EDR.PreCrashWindowS), "{:.0f}", "EDR pre-crash window, s")
v("edr_hz", float(EDR.PreCrashRateHz), "{:.0f}", "EDR sample rate, per s")
v("edr_n", float(EDR.PreCrashWindowS) * float(EDR.PreCrashRateHz), "{:.0f}", "EDR samples per element")
v("edr_old_s", float(EDR.LegacyPreCrashWindowS), "{:.1f}", "pre-phase-in window, s")
v("edr_old_s0", float(EDR.LegacyPreCrashWindowS), "{:.0f}", "the same, whole seconds")
v("edr_old_hz", float(EDR.LegacyPreCrashRateHz), "{:.0f}", "pre-phase-in rate, per s")
v("edr_old_n", float(EDR.LegacyPreCrashWindowS) * float(EDR.LegacyPreCrashRateHz), "{:.0f}",
  "pre-phase-in samples per element")
v("thr_trials", float(DATA.ThriftyTrials), "{:.0f}", "ThriftyDAgger trials per method")
v("thr_bc", float(DATA.ThriftyBehaviorCloningSuccesses), "{:.0f}", "behavior cloning successes")
v("thr_hg", float(DATA.ThriftyHgDaggerSuccesses), "{:.0f}", "HG-DAgger successes")
v("thr_td", float(DATA.ThriftyDaggerSuccesses), "{:.0f}", "ThriftyDAgger successes")
v("thr_hg_int", float(DATA.ThriftyHgDaggerInterventionsMean), "{:.2f}", "HG-DAgger interventions, mean")
v("thr_hg_sd", float(DATA.ThriftyHgDaggerInterventionsStd), "{:.2f}", "HG-DAgger interventions, std")
v("thr_td_int", float(DATA.ThriftyDaggerInterventionsMean), "{:.2f}", "ThriftyDAgger interventions, mean")
v("thr_td_sd", float(DATA.ThriftyDaggerInterventionsStd), "{:.2f}", "ThriftyDAgger interventions, std")
v("thr_int_avg", statistics.fmean([float(DATA.ThriftyHgDaggerInterventionsMean), float(DATA.ThriftyDaggerInterventionsMean)]),
  "{:.1f}", "takeovers per training episode, both methods")

print("== Normalization fallacy")
assert GRIP.donor_hold_force == GRIP.target_force_rating + GRIP.target_force_rating, "the note says twice"
v("grip_cmd", GRIP.command, "{:.2f}", "normalized command")
v("stroke_a", m(GRIP.donor_stroke, "mm"), "{:.0f}", "Machine A stroke, mm")
v("stroke_b", m(GRIP.target_stroke, "mm"), "{:.0f}", "Machine B stroke, mm")
v("part_mm", m(GRIP.part_width, "mm"), "{:.0f}", "part width, mm")
v("open_a", m(GRIP.donor_opening, "mm"), "{:.0f}", "Machine A opening, mm")
v("open_b", m(GRIP.target_opening, "mm"), "{:.0f}", "Machine B opening, mm")
v("hold_a", m(GRIP.donor_hold_force, "N"), "{:.0f}", "Machine A hold, N")
v("rating_b", m(GRIP.target_force_rating, "N"), "{:.0f}", "Machine B continuous rating, N")

print("== Mars Climate Orbiter")
v("lbf_n", m(1 * ureg.lbf * ureg.second, "N*s"), "{:.2f}", "one lbf*s in N*s")
v("mco_planned", float(MCO.MarsClimateOrbiterPlannedAltitudeKm), "{:.0f}", "planned altitude, km")
v("mco_actual", float(MCO.MarsClimateOrbiterActualAltitudeKm), "{:.0f}", "estimated altitude, km")

if __name__ == "__main__":
    write(__file__)
