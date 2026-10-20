#!/usr/bin/env python3
"""Every number on the Lecture 6 slides and notes (Policy Training), read from MLSysIM.

MLSysIM holds the inputs (the Embodied.* machine and its site record, the
Literature.RobotLearning figures for Dactyl, legged_gym, HIL-SERL, QT-Opt, the
grasping farm, Diffusion Policy, and OpenVLA, and the Dactyl trial series) and
the calculations (the Scenarios.* entries for policy training: the discounted
return over one opening, the decoder budget, the trial campaign, the parallel
simulation run, the contact deadlines, the soft-spring exploit, the linear-drive
reality gap, residual fine-tuning, and the latch scenario table); the chapter
reads the same scenarios. This file only selects values, converts units, and
formats them; it types no numbers. Each value is recorded with v() under the
key the deck prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports it for the scenario objects and the plotted magnitudes.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.physics.constants import INSULATION_CLASS_F_LIMIT  # noqa: E402

Embodied, Scenarios, Literature = mlsysim.Embodied, mlsysim.Scenarios, mlsysim.Literature

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
LIT = Literature.RobotLearning

DACTYL = Scenarios.DactylPhysicalAblation
OPENING = Scenarios.LatchOpeningDiscount
OPENING_FAST = Scenarios.LatchOpeningDiscountDoubledRate
DECODER = Scenarios.LatchDecoderBudget
TABLE = Scenarios.LatchScenarioTable
CAMPAIGN = Scenarios.LatchTrialCampaign
RUN = Scenarios.LeggedGymQuadrupedRun
DEADLINES = Scenarios.LatchContactDeadlines
EXPLOIT = Scenarios.LatchSoftSpringExploit
GAP = Scenarios.LinearDriveRealityGap
TUNE = Scenarios.LatchResidualFineTune

ALL = "All randomizations"


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def sci(key, x, mant_fmt, label):
    """Record x as a mantissa (key_m) and a power of ten (key_e)."""
    e = math.floor(math.log10(x))
    v(f"{key}_m", x / 10 ** e, mant_fmt, f"{label}, mantissa")
    v(f"{key}_e", e, "{:.0f}", f"{label}, power of ten")


print("== Hook: Dactyl (Andrychowicz et al. 2020)")
v("dactyl_dof", float(LIT.DactylHandDof), "{:.0f}", "Shadow Hand degrees of freedom")
v("dactyl_sim_trials", float(LIT.DactylSimulationTrials), "{:.0f}", "simulated trials")
v("dactyl_phys_trials", float(LIT.DactylPhysicalTrials), "{:.0f}", "physical trials per condition")
v("dactyl_sim_median", float(LIT.DactylSimulationMedianRotations), "{:.0f}", "simulation median, rotations")
v("dactyl_cap", float(LIT.DactylRotationCap), "{:.0f}", "rotation cap of the test")
all_trials = DACTYL.trials(ALL, "4")
v("dactyl_all_trials", ", ".join(f"{x:.0f}" for x in all_trials), "{}", "the ten physical trials, sorted")
v("dactyl_all_median", DACTYL.median(ALL), "{:g}", "physical median, all randomizations")
v("dactyl_all_min", min(all_trials), "{:.0f}", "fewest rotations")
v("dactyl_all_max", max(all_trials), "{:.0f}", "most rotations")
v("dactyl_gap_words", round(float(LIT.DactylSimulationMedianRotations) / DACTYL.median(ALL)), "words",
  "simulation median over physical median, nearest whole factor")
v("dactyl_none_median", DACTYL.median("No randomizations"), "{:g}", "median, no randomizations")
v("dactyl_nophys_median", DACTYL.median("No physics randomizations"), "{:g}", "median, no physics randomizations")
v("dactyl_noeffects_median", DACTYL.median("No unmodeled effects"), "{:g}", "median, no unmodeled effects")
v("dactyl_ff_lstm_median", DACTYL.median("FF policy, LSTM value"), "{:g}", "median, FF policy, LSTM value")
v("dactyl_ff_ff_median", DACTYL.median("FF policy, FF value"), "{:g}", "median, FF policy, FF value")
v("dactyl_locked_median", float(LIT.DactylLockedWristMedianRotations), "{:g}", "locked-wrist median")
v("dactyl_none_hours", float(LIT.DactylNoRandomizationTrainingHours), "{:g}", "training hours, no randomization")
v("dactyl_all_hours", float(LIT.DactylFullRandomizationTrainingHours), "{:.0f}", "training hours, all randomizations")
v("dactyl_none_years", float(LIT.DactylNoRandomizationSimulatedYears), "{:.0f}", "simulated years, none")
v("dactyl_all_years", float(LIT.DactylFullRandomizationSimulatedYears), "{:.0f}", "simulated years, all")

print("== Running case: the latch on the running machine")
v("f_lim", m(AISLE.f_latch_limit, "N"), "{:.0f}", "latch force limit, N")
v("f_trip", m(AISLE.f_latch_tripwire, "N"), "{:.0f}", "contact tripwire, N")
v("v_guard", m(AISLE.v_latch_approach, "m/s"), "{:.2f}", "guarded approach, m/s")
v("v_fast", m(AISLE.v_latch_high, "m/s"), "{:.2f}", "fast approach, m/s")
v("chunk_h", RM.chunk_horizon, "{:.0f}", "ACT setpoints per chunk")
v("step_ms", m(RM.chunk_step, "ms"), "{:.0f}", "setpoint step, ms")
v("renew_ms", m(RM.chunk_period, "ms"), "{:.0f}", "renewal period, ms")
v("act_ms", m(RM.chunk_inference_latency, "ms"), "{:.0f}", "ACT p99 inference, ms")
v("class_f_c", m(INSULATION_CLASS_F_LIMIT, "degC"), "{:.0f}", "Class F winding limit, degrees C")

print("== ACT in LeRobot (Zhao et al. 2023)")
v("act_chunk", float(Literature.VlaSystems.ActChunkSize), "{:.0f}", "ACT chunk size, setpoints")

print("== Diffusion Policy (Chi et al. 2023)")
v("dp_rollouts", float(LIT.DiffusionPolicyPushTRollouts), "{:.0f}", "Push-T rollouts per policy")

print("== Decoder budget (Napkin Math 6.1)")
v("ddim_n", DECODER.denoising_steps, "{:.0f}", "DDIM denoising steps")
v("flow_n", DECODER.flow_steps, "{:.0f}", "flow-matching evaluations")
v("enc_ms", m(DECODER.encode, "ms"), "{:.0f}", "visual encoding, ms")
v("ddim_ms", m(DECODER.denoising_latency, "ms"), "{:.0f}", "DDIM inference, ms")
v("eval_ms", m(DECODER.per_evaluation, "ms"), "{:.1f}", "per-evaluation cost, ms")
v("flow_ms", m(DECODER.flow_latency, "ms"), "{:.0f}", "flow-head latency, ms")
v("jitter_ms", m(DECODER.renewal_jitter, "ms"), "{:.0f}", "renewal-jitter allowance, ms")
v("act_slack_ms", m(DECODER.one_pass_slack, "ms"), "{:.0f}", "ACT slack to renewal, ms")
v("ddim_slack_ms", m(DECODER.denoising_slack, "ms"), "{:.0f}", "DDIM slack to renewal, ms")
v("ddim_over_ms", m(DECODER.denoising_overrun_under_jitter, "ms"), "{:.0f}", "DDIM past renewal under jitter, ms")
v("act_end_ms", m(DECODER.one_pass_end_under_jitter, "ms"), "{:.0f}", "ACT end under jitter, ms")

print("== Estimate: evaluations per deadline")
v("n_step", DECODER.evaluations_within_step, "{:.0f}", "evaluations within the setpoint step")
v("n_step_ms", m(DECODER.latency_at(DECODER.evaluations_within_step), "ms"), "{:.0f}", "latency at that count, ms")
v("n_step_room", DECODER.evaluation_room(DECODER.setpoint_step), "{:.1f}", "unrounded evaluations, step")
v("budget_ms", m(DECODER.renewal_less_jitter, "ms"), "{:.0f}", "renewal less jitter, ms")
v("n_renew", DECODER.evaluations_within_renewal, "{:.0f}", "evaluations within the renewal, jitter free")
v("n_renew_ms", m(DECODER.latency_at(DECODER.evaluations_within_renewal), "ms"), "{:.0f}", "latency at that count, ms")
v("n_renew_room", DECODER.evaluation_room(DECODER.renewal_less_jitter), "{:.1f}", "unrounded evaluations, renewal")

print("== Reward and discount (section 6.2.1)")
v("decisions", OPENING.decisions, "{:.0f}", "decisions per opening")
v("rollout_s", m(OPENING.rollout, "s"), "{:.0f}", "opening, s")
v("step_hz", m(1 / OPENING.step, "Hz"), "{:.0f}", "setpoint rate, Hz")
v("gamma", OPENING.discount, "{:.2f}", "discount")
v("w_terminal", OPENING.terminal_weight, "{:.2f}", "terminal weight at the first decision")
v("horizon_steps", OPENING.horizon_steps, "{:.0f}", "effective horizon, steps")
v("horizon_s", m(OPENING.horizon_time, "s"), "{:.1f}", "effective horizon, s")
v("gamma_half", OPENING.retaining_discount, "{:.4f}", "discount keeping half the weight")
v("w_retained", OPENING.retained_weight, "{:.2f}", "weight it keeps")
v("step_hz_fast", m(1 / OPENING_FAST.step, "Hz"), "{:.0f}", "doubled setpoint rate, Hz")
v("horizon_s_fast", m(OPENING_FAST.horizon_time, "s"), "{:.2f}", "horizon at the doubled rate, s")

print("== Pretraining and the scenario table")
v("oxe_used", float(LIT.OpenVlaPretrainingEpisodes), "{:,.0f}", "OpenVLA pretraining episodes")
v("oxe_used_k", float(LIT.OpenVlaPretrainingEpisodes) / 10 ** 3, "{:.0f}", "the same, thousands")
v("plates_words", len(TABLE.offsets), "words", "plate positions")
v("speeds_words", len(TABLE.speeds), "words", "approach speeds")
v("cells_words", TABLE.cells, "words", "cells")
v("supported", TABLE.supported, "{:.0f}", "supported cells")
v("supported_words", TABLE.supported, "words", "supported cells, spelled")
v("absences", TABLE.absences, "{:.0f}", "known absences")

print("== Reset wall (Napkin Math 6.2)")
v("trials_e", math.log10(CAMPAIGN.trials), "{:.0f}", "trials, power of ten")
v("reset_auto_s", m(CAMPAIGN.automated_reset, "s"), "{:.0f}", "automated reset, s")
v("reset_manual_s", m(CAMPAIGN.manual_reset, "s"), "{:.0f}", "manual reset, s")
v("exec_days", m(CAMPAIGN.execution_time, "day"), "{:.1f}", "execution only, days")
v("auto_days", m(CAMPAIGN.automated_time, "day"), "{:.1f}", "automated resets, days")
v("manual_days", m(CAMPAIGN.manual_time, "day"), "{:.1f}", "manual resets, days")
v("manual_days_r", m(CAMPAIGN.manual_time, "day"), "{:.0f}", "manual resets, whole days")
v("reset_pct", 100 * CAMPAIGN.manual_reset_share, "{:.1f}", "reset share of a manual trial, %")
v("farm_rigs", CAMPAIGN.farm_rigs, "{:.0f}", "rigs in the farm")
v("farm_rigs_words", CAMPAIGN.farm_rigs, "words", "rigs in the farm, spelled")
v("farm_days", m(CAMPAIGN.farm_manual_time, "day"), "{:.1f}", "farm, manual resets, days")
v("reversals_m", CAMPAIGN.reversals / 10 ** 6, "{:.1f}", "load reversals, millions (mantissa of 10^6)")
v("reversals_m_words", CAMPAIGN.reversals / 10 ** 6, "words", "load reversals, millions, spelled")
v("hil_hours", m(CAMPAIGN.seeded_budget, "hour"), "{:.1f}", "seeded budget, h")
v("trial_s", m(CAMPAIGN.rollout + CAMPAIGN.automated_reset, "s"), "{:.0f}", "one trial with automated reset, s")
v("hil_trials", CAMPAIGN.seeded_trials, "{:.0f}", "trials in the seeded budget")
v("hil_factor", CAMPAIGN.trial_ratio, "{:,.0f}", "from-scratch trials over seeded trials")
v("hil_reversals", CAMPAIGN.seeded_reversals, "{:,.0f}", "reversals in the seeded budget")
v("sim_min", m(CAMPAIGN.simulated_time, "minute"), "{:.1f}", "the million openings in simulation, min")
v("qtopt", float(LIT.QtOptGraspAttempts), "{:,.0f}", "QT-Opt grasp attempts")

print("== Grasping farm (Levine et al. 2018)")
v("farm_attempts", float(LIT.GraspingFarmAttempts), "{:,.0f}", "grasp attempts")
v("farm_months_words", float(LIT.GraspingFarmMonths), "words", "collection months, spelled")

print("== HIL-SERL (Luo et al. 2025)")
v("hil_min_h", float(LIT.HilSerlMinTrainingHours), "{:.0f}", "shortest training, h")
v("hil_demo_lo", float(LIT.HilSerlMinDemonstrations), "{:.0f}", "fewest seed demonstrations")
v("hil_demo_hi", float(LIT.HilSerlMaxDemonstrations), "{:.0f}", "most seed demonstrations")
v("hil_avg", float(LIT.HilSerlRunningAverageEpisodes), "{:.0f}", "episodes per running average")
v("hil_ram_min", float(LIT.HilSerlRamInsertionMinutes), "{:.0f}", "RAM insertion to full success, min (read)")
v("hil_usb_min", float(LIT.HilSerlUsbInsertMinutes), "{:.0f}", "USB grasp-insert to full success, min (read)")

print("== Parallel simulation (Rudin et al. 2022)")
v("robots", RUN.robots, "{:,.0f}", "simulated robots")
v("steps_upd", RUN.steps_per_update, "{:.0f}", "steps per robot per update")
v("batch", RUN.batch, "{:,.0f}", "transitions per batch")
v("updates", RUN.updates, "{:,.0f}", "updates")
v("train_min", m(RUN.training_time, "minute"), "{:.0f}", "training time bound, min")
v("policy_hz", m(RUN.policy_rate, "Hz"), "{:.0f}", "policy rate, Hz")
v("sim_days", m(RUN.simulated_experience, "day"), "{:.1f}", "simulated walking, days")
sci("samples", RUN.samples, "{:.2f}", "total policy steps")
sci("rate", m(RUN.throughput, "1/s"), "{:.2f}", "throughput lower bound, steps/s")

print("== Soft-spring exploit (Napkin Math 6.3)")
v("mass_kg", m(EXPLOIT.moving_mass, "kg"), "{:.0f}", "moving mass, kg")
v("pen_mm", m(EXPLOIT.penalty_penetration, "mm"), "{:.1f}", "penalty penetration, mm")
sci("k_pen", m(EXPLOIT.penalty_stiffness, "N/m"), "{:.1f}", "penalty stiffness, N/m")
sci("k_plate", m(AISLE.k_latch, "N/m"), "{:.1f}", "plate stiffness, N/m")
sci("k_star", m(EXPLOIT.break_even_arm_stiffness, "N/m"), "{:.2f}", "break-even arm stiffness, N/m")
v("f_sim", m(EXPLOIT.simulated_peak, "N"), "{:.1f}", "simulated peak, N")
v("f_rigid", m(EXPLOIT.rigid_peak, "N"), "{:.1f}", "rigid-arm peak, N")
v("f_rigid_r", m(EXPLOIT.rigid_peak, "N"), "{:.0f}", "rigid-arm peak, whole newtons")
v("f_guard", m(EXPLOIT.guarded_rigid_peak, "N"), "{:.1f}", "guarded rigid peak, N")
v("plate_x", EXPLOIT.plate_over_penalty, "{:.1f}", "plate over penalty stiffness")
v("rigid_x", EXPLOIT.rigid_over_limit, "{:.1f}", "rigid peak over the limit")

print("== Reality gap (section 6.5, illustrative linear drive)")
v("gap_v", m(GAP.approach_speed, "m/s"), "{:.2f}", "approach speed, m/s")
v("gap_f", m(GAP.target_force, "N"), "{:.0f}", "regulated force, N")
v("gap_drive_ms", m(GAP.drive_delay, "ms"), "{:.0f}", "unmodeled drive delay, ms")
v("gap_t_sim", m(GAP.simulated_time_to_target, "ms"), "{:.0f}", "time to target, simulated, ms")
v("gap_t_real", m(GAP.physical_time_to_target, "ms"), "{:.0f}", "time to target, hardware, ms")
v("gap_p99_ms", m(GAP.p99_delay, "ms"), "{:.1f}", "p99 actuation delay, ms")
v("gap_travel_mm", m(GAP.unmodeled_travel, "mm"), "{:.1f}", "unobserved travel, mm")
v("gap_bias_mm", m(GAP.range_bias, "mm"), "{:.1f}", "range bias, mm")
v("gap_dropout", GAP.dropout_frames, "{:.0f}", "dropout burst, frames")
v("gap_pixel_pct", 100 * GAP.pixel_difference, "{:.1f}", "pixel difference, %")
v("gap_patch_mm", m(GAP.patch_shift, "mm"), "{:.0f}", "contact-patch shift, mm")
v("gap_tol_mm", m(GAP.chamfer_tolerance, "mm"), "{:.0f}", "chamfer tolerance, mm")

print("== Residual fine-tuning (section 6.6)")
v("ft_offset_mm", m(TUNE.plate_offset, "mm"), "{:.1f}", "plate offset, mm")
v("ft_steps", TUNE.interaction_steps, "{:,.0f}", "interaction steps")
v("ft_active_s", m(TUNE.active_time, "s"), "{:.0f}", "active contact time, s")
v("ft_pct", 100 * TUNE.exceed_fraction, "{:.1f}", "steps past the tripwire, %")
v("ft_samples", TUNE.exceed_samples, "{:.0f}", "above-tripwire samples")

print("== Contact deadline (section 6.4.2)")
v("t_yield_fast", m(DEADLINES.deadline_fast, "ms"), "{:.1f}", "fast deadline, ms")
v("t_trip_fast", m(DEADLINES.tripwire_fast, "ms"), "{:.3f}", "fast tripwire crossing, ms")
v("t_brake_fast", m(DEADLINES.brake_onset_fast, "ms"), "{:.3f}", "fast brake onset, ms")
v("t_left_fast", m(DEADLINES.time_left_fast, "ms"), "{:.3f}", "fast time left, ms")

if __name__ == "__main__":
    write(__file__)
