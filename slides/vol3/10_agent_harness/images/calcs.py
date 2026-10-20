#!/usr/bin/env python3
"""Every number on the Volume III Lecture 10 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations, and the chapter reads the same
entries: the rule kept per turn (Scenarios.FreezeRuleOverStepBudget), the
pause behind one call's output allowance (CanceledStreamDecodeSavings
.allowance_decode_time), approval escrow, weighted deficit round-robin,
bounded-liability admission, and the budget meter are Scenarios.* backed by
the solvers in engine/solvers/harness.py; the public runs are Literature
series read through Scenarios.SweBenchCallCeilingP90,
MiniSweAgentOpus46RunCost, and OSWorldStepCeilingReturn. The one other source
is the deck's own data (data/): a pause measured on a laptop
(pause_latency.json, written by pause_latency.py) and one public trajectory
(trajectory_django16263.json, fetched by fetch_trajectory.py), read as
recorded. This file only selects values, converts units, and formats them; it
types no numbers. Each value is recorded with v() under the key the deck
prints as \\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.physics.agents import calc_step_error_for_target, calc_unchecked_success  # noqa: E402

Scenarios = mlsysim.Scenarios
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

RULE = Scenarios.FreezeRuleOverStepBudget
CALL = Scenarios.CanceledStreamDecodeSavings
ESCROW = Scenarios.RepairRemediationApprovalEscrow
ESCROW_FAST = Scenarios.RepairRemediationEscrowFasterReview
DRR_S = Scenarios.InteractiveMaintenanceDrr
DRR_C = Scenarios.InteractiveMaintenanceDrrCompacted
ADMIT_S = Scenarios.RepairNextCallAdmission
METER_S = Scenarios.RepairTrajectoryBudgetMeter
CEIL_S = Scenarios.SweBenchCallCeilingP90
RUN_S = Scenarios.MiniSweAgentOpus46RunCost
STEPS_S = Scenarios.OSWorldStepCeilingReturn

CEIL = CEIL_S.result
CONFIGS = CEIL.configurations
PICK = next(c for c in CONFIGS if c["submission"] == RUN_S.submission)
RUN = RUN_S.result
STEPS = STEPS_S.result
DRR = DRR_S.result
ADMIT = ADMIT_S.result
METER = METER_S.result


def mag(q, unit):
    return q.to(unit).magnitude


def pct(x):
    return x * 100


print("== A rule kept per turn (Scenarios.FreezeRuleOverStepBudget)")
hold = RULE.result.success[RULE.short_horizon]
v("fr_step", RULE.step_success, "{:.2f}", "per-turn compliance")
v("fr_step_pct", pct(RULE.step_success), "{:.0f}", "per-turn compliance, %")
v("fr_turns", RULE.short_horizon, "{:.0f}", "turns in the run")
v("fr_hold", hold, "{:.2f}", "rule held on every turn")
v("fr_break_pct", pct(1 - hold), "{:.0f}", "runs that break it at least once, %")
v("fr_target_pct", pct(RULE.target_success), "{:.0f}", "share of runs that must keep it, %")

print("== Median calls of five public configurations (Scenarios.SweBenchCallCeilingP90)")
v("swe_runs", CEIL.runs, "{:,.0f}", "public trajectories")
v("swe_configs_words", len(CONFIGS), "words", "configurations")
v("swe_tasks", max(c["runs"] for c in CONFIGS), "{:.0f}", "tasks in a full submission")
v("med_lo", CEIL.median_calls_low, "{:.0f}", "lowest median calls")
v("med_hi", CEIL.median_calls_high, "{:.0f}", "highest median calls")
v("fr_hold_hi_pct", pct(calc_unchecked_success(RULE.step_success, CEIL.median_calls_high)), "{:.0f}",
  "rule held on every one of the highest median calls, %")
v("fr_need_pct", pct(1 - calc_step_error_for_target(RULE.target_success, CEIL.median_calls_high)),
  "{:.2f}", "per-turn compliance that keeps the target there, %")
v("mini_runs", sum(c["runs"] for c in CONFIGS if c["scaffold"].startswith("mini-SWE-agent")),
  "{:,.0f}", "public runs by mini-SWE-agent 2.x")

print("== One public trajectory (data/trajectory_django16263.json)")
TRAJ = json.loads((DATA / "trajectory_django16263.json").read_text())
cum = TRAJ["cumulative_cost"]
limit = TRAJ["limits"]["cost_limit"]
assert limit == mag(RUN_S.cost_limit, "USD"), "the run's limit is the scenario's"
v("tj_cost_full", TRAJ["info"]["model_stats"]["instance_cost"], "{:.7f}", "instance_cost, 7 decimals")
v("tj_calls", len(cum), "{:.0f}", "model calls")
v("tj_msgs", TRAJ["n_messages"], "{:.0f}", "messages saved")
v("tj_limit", limit, "{:.2f}", "cost limit, $")
v("tj_limit_cfg", limit, "{:.1f}", "cost limit as configured")
*_, _before, _final = cum
v("tj_before", _before, "{:.3f}", "settled before the last call, $")
v("tj_last", TRAJ["call_costs"][-1], "{:.2f}", "last call, $")
v("tj_final", _final, "{:.2f}", "final cost, $")
v("tj_over_pct", pct((_final - limit) / limit), "{:.1f}", "over the limit, %")

print("== Runs of that configuration over the limit (Scenarios.MiniSweAgentOpus46RunCost)")
v("ov_runs", RUN.tasks, "{:.0f}", "runs")
v("ov_n", RUN.over_limit_runs, "{:.0f}", "runs that ended above the limit")
v("ov_mean", RUN.mean_overshoot, "{:.2f}", "mean overshoot, $")
v("ov_max", RUN.max_overshoot, "{:.2f}", "largest overshoot, $")
v("ov_max_cents", pct(RUN.max_overshoot), "{:.0f}", "largest overshoot, cents")

print("== Pause behind one call (CanceledStreamDecodeSavings; data/pause_latency.json)")
v("bw_tdec", mag(CALL.floors.result.memory_time, "ms"), "{:.3f}", "decode floor, ms per token")
v("bw_kmax", CALL.output_allowance, "{:,.0f}", "output allowance, tokens")
v("bw_wait", mag(CALL.allowance_decode_time, "s"), "{:.1f}", "boundary wait, s")
_pause = json.loads((DATA / "pause_latency.json").read_text())
_rows = _pause["rows"]
PAUSE = {"k": sorted({r["k"] for r in _rows}), "end": {}, "cancel": {}}
for _k in PAUSE["k"]:
    for _pol, _key in (("end-check", "end"), ("cancel", "cancel")):
        _e = [r["pause_effect_s"] for r in _rows if r["k"] == _k and r["policy"] == _pol]
        PAUSE[_key][_k] = (min(_e), statistics.mean(_e), max(_e))
_kmin, _kmax = PAUSE["k"][0], PAUSE["k"][-1]
_reps = {sum(1 for r in _rows if r["k"] == _k and r["policy"] == _p)
         for _k in PAUSE["k"] for _p in ("end-check", "cancel")}
assert len(_reps) == 1, "every allowance and policy ran the same number of times"
_ms_tok = statistics.mean(r["eval_duration_s"] / r["eval_count"] for r in _rows
                          if r["policy"] == "end-check" and r.get("eval_duration_s"))
_cancel_max = max(c[2] for c in PAUSE["cancel"].values())
v("ps_at", _pause["meta"]["pause_at_s"], "{:.1f}", "pause sent after the call, s")
_n_reps = _reps.pop()
v("ps_reps", _n_reps, "{:.0f}", "runs per allowance and policy")
v("ps_reps_words", _n_reps, "words", "the same, in words")
v("ps_k_lo", _kmin, "{:,.0f}", "smallest allowance, tokens")
v("ps_k_hi", _kmax, "{:,.0f}", "largest allowance, tokens")
v("ps_wait_lo", PAUSE["end"][_kmin][1], "{:.1f}", "wait for the reply, smallest allowance, s")
v("ps_wait_hi", PAUSE["end"][_kmax][1], "{:.1f}", "wait for the reply, largest allowance, mean s")
v("ps_wait_hi_round", PAUSE["end"][_kmax][1], "{:.0f}", "the same, whole seconds")
v("ps_wait_hi_min", PAUSE["end"][_kmax][0], "{:.1f}", "fastest of the runs, s")
v("ps_wait_hi_max", PAUSE["end"][_kmax][2], "{:.1f}", "slowest of the runs, s")
v("ps_ms_tok", _ms_tok * 10 ** 3, "{:.0f}", "measured decode, ms per token")
v("ps_cancel_ms", math.ceil(_cancel_max * 10 ** 3), "{:.0f}", "cancel held within, ms (ceiling)")
v("ps_next_first", max(r["next_call_first_token_s"] for r in _rows
                       if r["policy"] == "cancel" and r.get("next_call_first_token_s") is not None),
  "{:.2f}", "first token of the next call after a cancel, s (max)")

print("== Approval escrow (Scenarios.RepairRemediationApprovalEscrow)")
e, ef = ESCROW.result, ESCROW_FAST.result
v("es_rate", mag(ESCROW.repair_rate, "1/hour"), "{:.0f}", "repair review requests per hour")
v("es_wait", mag(ESCROW.repair_wait, "hour"), "{:.0f}", "mean decision, h")
v("es_lease", mag(ESCROW.lease, "MiB"), "{:.0f}", "workspace lease, MiB")
v("es_held", e.held, "{:.0f}", "held repair trajectories")
v("es_mem", mag(e.retained_memory, "GiB"), "{:.0f}", "retained workspaces, GiB")
v("es_mem_fast", mag(ef.retained_memory, "GiB"), "{:.0f}", "the same at half the decision time, GiB")
v("rm_rate", mag(ESCROW.remediation_rate, "1/hour"), "{:.0f}", "remediation requests per hour")
v("rm_wait", mag(ESCROW.remediation_wait, "minute"), "{:.0f}", "remediation decision, min")
v("rm_wait_h", mag(ESCROW.remediation_wait, "hour"), "{:.2f}", "remediation decision, h")
v("rm_diag", mag(ESCROW.diagnosis, "minute"), "{:.0f}", "diagnosis, min")
v("rm_apply", mag(ESCROW.apply_verify, "minute"), "{:.0f}", "apply and verify, min")
v("rm_held", e.remediation_held, "{:.1f}", "held remediation trajectories")
v("rm_ttm", mag(e.time_to_mitigate, "minute"), "{:.0f}", "time to mitigate, min")
v("rm_share", pct(e.approval_share), "{:.0f}", "approval share of time to mitigate, %")
v("rm_saved", mag(e.time_to_mitigate - ef.time_to_mitigate, "minute"), "{:.1f}",
  "minutes removed by twice-as-fast decisions")

print("== Weighted deficit round-robin (Scenarios.InteractiveMaintenanceDrr)")
_i, _m = DRR_S.classes["interactive"], DRR_S.classes["maintenance"]
v("dr_w_int", _i["weight"], "{:.0f}", "interactive weight")
v("dr_w_mnt", _m["weight"], "{:.0f}", "maintenance weight")
v("dr_quantum", DRR_S.quantum, "{:,.0f}", "quantum per unit weight, tokens")
v("dr_credit_int", DRR.credit_per_round["interactive"], "{:,.0f}", "interactive credit per round")
v("dr_c_int", DRR.charge_per_call["interactive"], "{:,.0f}", "interactive charge per call")
v("dr_c_mnt", DRR.charge_per_call["maintenance"], "{:,.0f}", "maintenance charge per call")
v("dr_calls_int", DRR.calls_per_round["interactive"], "{:.1f}", "interactive calls per round")
v("dr_rounds_mnt", DRR.rounds_per_call["maintenance"], "{:.2f}", "rounds per maintenance call")
v("dr_share_int", pct(DRR.charge_share["interactive"]), "{:.0f}", "interactive charge share, %")
v("dr_ratio", DRR.call_ratio, "{:.0f}", "interactive / maintenance calls")
v("dr_call_share", pct(DRR.call_share["interactive"]), "{:.0f}", "interactive call share, %")
v("dr_rounds", DRR.simulated_rounds, "{:,.0f}", "simulated rounds")
v("dr_compact_in", DRR_C.classes["maintenance"]["input"], "{:,.0f}", "compacted maintenance input")
v("dr_compact_rounds", DRR_C.result.rounds_per_call["maintenance"], "{:.2f}",
  "rounds per compacted maintenance call")

print("== Bounded-liability admission (Scenarios.RepairNextCallAdmission)")
v("ad_allow", mag(ADMIT_S.allowance, "USD"), "{:.2f}", "allowance, $")
v("ad_settled", mag(ADMIT_S.settled, "USD"), "{:.2f}", "settled, $")
v("ad_reserved", mag(ADMIT_S.reserved, "USD"), "{:.2f}", "test job reservation, $")
v("ad_final", mag(ADMIT_S.finalization_reserve, "USD"), "{:.2f}", "finalization reserve, $")
v("ad_in", ADMIT_S.input_tokens, "{:,.0f}", "next call input, tokens")
v("ad_out", ADMIT_S.output_allowance, "{:,.0f}", "next call output allowance, tokens")
v("ad_p_in", mag(ADMIT_S.input_price, "USD/Mtoken"), "{:.0f}", "$ per million input tokens")
v("ad_p_out", mag(ADMIT_S.output_price, "USD/Mtoken"), "{:.0f}", "$ per million output tokens")
v("ad_liab_in", ADMIT.input_liability, "{:.2f}", "input liability, $")
v("ad_liab_out", ADMIT.output_liability, "{:.2f}", "output liability, $")
v("ad_liab", ADMIT.liability, "{:.2f}", "liability, $")
v("ad_spend_head", ADMIT.spend_headroom, "{:.2f}", "spend-check headroom, $")
v("ad_head", ADMIT.headroom, "{:.2f}", "headroom after reservations, $")
v("ad_short", ADMIT.finalization_shortfall, "{:.2f}", "finalization shortfall if admitted, $")
v("ad_in_share", pct(ADMIT.input_share), "{:.0f}", "input share of liability, %")
v("ad_max_in", round(ADMIT.max_input_tokens, -2), "{:,.0f}", "largest input that fits, to 100 tokens")
v("ad_res_share", pct(ADMIT.reservation_share), "{:.0f}", "reservation share of the allowance, %")

print("== The budget meter (Scenarios.RepairTrajectoryBudgetMeter)")
T = METER_S.trajectory
v("mt_turns", T.turns, "{:.0f}", "turns done")
v("mt_first", T.first_prompt, "{:,.0f}", "first prompt, tokens")
v("mt_append", T.appended, "{:,.0f}", "appended per turn, tokens")
v("mt_out", T.output_per_turn, "{:.0f}", "output per turn, tokens")
v("mt_in", METER.input_tokens, "{:,.0f}", "input tokens over the run")
v("mt_cached", METER.cached_tokens, "{:,.0f}", "cached input tokens")
v("mt_h_max", METER_S.turn_allowance, "{:.0f}", "turn allowance")
v("mt_n_max_m", METER_S.token_allowance / 10 ** 6, "{:.0f}", "token allowance, millions")
v("mt_t_max", mag(METER_S.time_allowance, "s"), "{:.0f}", "time allowance, s")
v("mt_c_max", mag(METER_S.dollar_allowance, "USD"), "{:.2f}", "dollar allowance")
v("mt_clock", mag(METER_S.elapsed, "s"), "{:.0f}", "clock at the readout, s")
v("mt_cost", METER.cost_with_reuse, "{:.2f}", "cost with reuse, $")
v("mt_cost_none", METER.cost_without_reuse, "{:.2f}", "cost billed uncached, $")
v("mt_turns_pct", pct(METER.used_with_reuse["turns"]), "{:.0f}", "turns used, %")
v("mt_tokens_pct", pct(METER.used_with_reuse["tokens"]), "{:.1f}", "tokens used, %")
v("mt_time_pct", pct(METER.used_with_reuse["time"]), "{:.1f}", "time used, %")
v("mt_dollars_pct", pct(METER.used_with_reuse["dollars"]), "{:.1f}", "dollars used, %")
v("mt_dollars_none_pct", pct(METER.used_without_reuse["dollars"]), "{:.1f}", "dollars used uncached, %")

print("== Choosing a ceiling (Scenarios.SweBenchCallCeilingP90)")
_same = sorted(c["cap"] for c in CONFIGS if c["model"] == "Claude Sonnet 4")
v("ce_q", pct(CEIL.quantile), "{:.0f}", "ceiling quantile of accepted runs, %")
v("ce_fail_lo", pct(CEIL.failed_halted_low), "{:.0f}", "failed runs halted, lowest, %")
v("ce_fail_hi", pct(CEIL.failed_halted_high), "{:.0f}", "failed runs halted, highest, %")
v("ce_ok_hi", pct(CEIL.accepted_halted_high), "{:.0f}", "accepted runs halted, at most, %")
v("ce_cap_lo", CEIL.cap_low, "{:.0f}", "lowest ceiling, calls")
v("ce_cap_hi", CEIL.cap_high, "{:.0f}", "highest ceiling, calls")
v("ce_same_lo", _same[0], "{:.0f}", "same model, first ceiling")
v("ce_same_hi", _same[-1], "{:.0f}", "same model, second ceiling")
v("ce_pick_ok_med", PICK["median_accepted"], "{:.0f}", "one configuration: accepted median calls")
v("ce_pick_bad_med", PICK["median_failed"], "{:.0f}", "failed median calls")
v("ce_pick_cap", PICK["cap"], "{:.0f}", "its ceiling, calls")
v("ce_pick_ok", pct(PICK["accepted_halted"]), "{:.0f}", "accepted runs halted, %")
v("ce_pick_bad", pct(PICK["failed_halted"]), "{:.0f}", "failed runs halted, %")
v("ce_pick_cpa", RUN.cost_per_resolved, "{:.2f}", "cost per accepted task, $")

print("== Raising the step ceiling (Scenarios.OSWorldStepCeilingReturn)")
_lo, _hi = min(STEPS.steps), max(STEPS.steps)
_mid = next(s for s in STEPS.steps if _lo < s < _hi)  # the middle ceiling
_o3 = next(m for m in STEPS.scores if m.startswith("o3"))
v("ow_lo", _lo, "{:.0f}", "lowest ceiling, steps")
v("ow_mid", _mid, "{:.0f}", "middle ceiling, steps")
v("ow_hi", _hi, "{:.0f}", "highest ceiling, steps")
v("ow_extra", _hi - _mid, "{:.0f}", "steps added from the middle to the highest ceiling")
v("ow_flat_words", STEPS.flat_after_middle, "words", "models flat or lower after the middle ceiling")
v("ow_gain_words", len(STEPS.scores) - STEPS.flat_after_middle, "words", "models still gaining")
v("ow_top", max(s[_hi] for s in STEPS.scores.values()), "{:.0f}", "top model at the highest ceiling, %")
v("ow_o3", STEPS.scores[_o3][_hi], "{:.0f}", "o3 at the highest ceiling, %")
v("ow_best_gain", STEPS.best_gain, "{:.0f}", "largest gain, lowest to highest ceiling, points")

if __name__ == "__main__":
    write(__file__)
