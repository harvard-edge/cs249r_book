#!/usr/bin/env python3
"""Every number on the Volume III Lecture 11 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations, and the chapter reads the same
entries: the group commit (Scenarios.IntentQuorumGroupCommit), full snapshots
against deltas (GrowingTrajectorySnapshots), checkpoint strides on a crash
trace (TwoCrashCheckpointTrace), Young's interval and its flat minimum
(ReexecutedWorkCheckpointInterval), the cold prefill of a resumed context
(ResumedContextColdPrefill), a greedy rerun's reproduction
(GreedyRerunReproduction), and a year of recovery data
(RepairTrajectoryRetention), backed by engine/solvers/durability.py. The test
reservation is the one the harness lecture admits against
(RepairNextCallAdmission), the frontier task length is METR's
(FrontierTaskHorizonTrend), and the temperature-zero completions are He and
Thinking Machines Lab's (Literature.InferenceDeterminism).

The one other source is the deck's own data (data/), runs measured for this
lecture on one laptop (Apple M5 Max, macOS 26.4) and read as recorded; the
script that produced each file sits beside it. The Temporal runs need the
temporalio package (1.34.0) and a local Ollama; the other scripts need only
the standard library.

This file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex. make_figs.py imports
it so figures and slides agree.

Run from this folder: python3 calcs.py
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Scenarios = mlsysim.Scenarios
Literature = mlsysim.Literature
Q_ = mlsysim.core.units.Q_
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

GROUP = Scenarios.IntentQuorumGroupCommit
SNAP = Scenarios.GrowingTrajectorySnapshots
TRACE = Scenarios.TwoCrashCheckpointTrace
YOUNG = Scenarios.ReexecutedWorkCheckpointInterval
PREFILL = Scenarios.ResumedContextColdPrefill
RERUN = Scenarios.GreedyRerunReproduction
KEEP = Scenarios.RepairTrajectoryRetention
TM = Literature.InferenceDeterminism


def load(name):
    return json.loads((DATA / name).read_text())


def mag(q, unit):
    return q.to(unit).magnitude


def pct(x):
    return x * 100


# ------------------------------------------------------------------ hook
# data/resample.json: the repair agent's task prompt sent repeatedly to
# qwen2.5:7b at a sampling temperature and at temperature 0.
def resample():
    d = load("resample.json")
    temps = sorted((k for k in d if k.replace(".", "").isdigit()), key=float)
    cold, hot = d[temps[0]], d[temps[-1]]
    by_tool = {}
    for reply, n in hot["counts"]:
        tool = reply.split("(")[0].strip()
        by_tool[tool] = by_tool.get(tool, 0) + n
    return dict(temp=float(temps[-1]), n=hot["n"], distinct=hot["distinct"], by_tool=by_tool,
                n0=cold["n"], distinct0=cold["distinct"])


# --------------------------------------------- the real crash run (Temporal)
# data/summary_Repair*.json: a worker SIGKILLed during the r18 test activity.
def crash_runs():
    out = {}
    for name in ("RepairNaive", "RepairKeyed"):
        s = load(f"summary_{name}.json")
        led = s["ledger"]
        t0 = led[0]["t"]
        out[name] = dict(
            model_calls=sum(e["event"] == "model_call" for e in led),
            launches=sum(e["event"] == "job_launched" for e in led),
            reused=sum(e["event"] == "job_reused" for e in led),
            ledger=[(e["event"], e.get("job"), e["attempt"], e["t"] - t0) for e in led])
    return out


# ------------------------------------------------ acknowledgment latency
# data/ack_latency.json: appends of one intent record on the laptop's SSD.
def acks():
    d = load("ack_latency.json")
    w, f, ff = (d[k]["median_ms"] for k in ("write", "fsync", "full_fsync"))
    g = {int(k): x for k, x in d["group_commit"].items()}
    return dict(write=w, fsync=f, full=ff, group=g, n=d["write"]["n"], record_bytes=d["record_bytes"])


# ------------------------------------------------------- resume prefill
# data/resume_prefill.json: next call after a resume, cold and warm server.
def prefill():
    d = load("resume_prefill.json")
    cold = statistics.median(r["wall_s"] for r in d["cold"])
    warm = statistics.median(r["wall_s"] for r in d["warm"])
    load_s = Q_(statistics.median(r["load_duration"] for r in d["cold"]), "ns").to("s").magnitude
    return dict(tokens=d["cold"][0]["prompt_eval_count"], cold=cold, warm=warm, load=load_s,
                n=len(d["cold"]))


# ---------------------------------------------- figure helpers (make_figs)
def snapshots():
    r = SNAP.result
    return dict(full_gb=mag(r.full_bytes, "GB"), delta_mb=mag(r.delta_bytes, "MB"))


def snapshot_curves():
    """Cumulative bytes (GB) after k checkpoints, full and delta, k = 0 to the scenario's count."""
    from mlsysim.physics.reliability import calc_snapshot_write_volume
    ks = list(range(SNAP.result.checkpoints + 1))
    full, delta = [0.0], [0.0]
    for k in ks[1:]:
        f, dl = calc_snapshot_write_volume(SNAP.state_growth, SNAP.stride, k)
        full.append(mag(f, "GB"))
        delta.append(mag(dl, "GB"))
    return ks, full, delta


PER_TURN = TRACE.rebuild_per_turn
REEXEC = TRACE.reexecute_per_turn


def cadence():
    r = TRACE.result
    return dict(s_log=r.stride_logged, s_unlog=r.stride_reexecuted)


def expected_overhead_curve(stride, lost_turn_cost):
    """Expected seconds per turn on the trace's crash rate: pause/stride + rate * cost * stride / 2."""
    return mag(TRACE.stride_overhead(stride, lost_turn_cost), "s")


def retention():
    r = KEEP.result
    return dict(log_mb=mag(r.log_bytes, "MB"), snap_mb=mag(KEEP.workspace, "MB"))


# ======================================================================
print("== Hook: restarts from the prompt (data/resample.json)")
RS = resample()
v("rs_n", RS["n"], "{:.0f}", "restarts at the sampling temperature")
v("rs_temp", RS["temp"], "{:.1f}", "sampling temperature")
v("rs_distinct", RS["distinct"], "{:.0f}", "distinct first tool calls")
v("rs_read", RS["by_tool"]["read_file"], "{:.0f}", "restarts that read a file")
v("rs_grep", RS["by_tool"]["grep"], "{:.0f}", "restarts that searched the code")
v("rs_tests", RS["by_tool"]["run_tests"], "{:.0f}", "restarts that ran the tests")
v("rs_n0", RS["n0"], "{:.0f}", "runs at temperature 0")
v("rs_distinct0", RS["distinct0"], "{:.0f}", "distinct replies at temperature 0")
v("rs_distinct0_words", RS["distinct0"], "words", "the same, in words")

print("== The frontier task length (Scenarios.FrontierTaskHorizonTrend)")
_trend = Scenarios.FrontierTaskHorizonTrend.result
v("metr_last_h", mag(Q_(_trend.last_frontier_minutes, "minute"), "hour"), "{:.0f}", "latest frontier, h")

print("== The test's reservation (Scenarios.RepairNextCallAdmission)")
v("res_usd", mag(Scenarios.RepairNextCallAdmission.reserved, "USD"), "{:.2f}", "reservation on the test job, $")

print("== Temporal crash runs (data/summary_*.json)")
CR = crash_runs()
_naive = CR["RepairNaive"]
v("cr_calls", _naive["model_calls"], "{:.0f}", "model calls in the run")
v("cr_launches", _naive["launches"], "{:.0f}", "test jobs launched, unkeyed executor")
v("cr_second_s", _naive["ledger"][-1][3], "{:.1f}", "second launch after the first entry, s")

print("== Acknowledgment latency (data/ack_latency.json)")
A = acks()
_ks = sorted(A["group"])
_flush = [A["group"][k]["flush_ms_median"] for k in _ks]
v("ack_n", A["n"], "{:.0f}", "appends per call")
v("ack_bytes", A["record_bytes"], "{:.0f}", "bytes per intent")
v("ack_write", A["write"], "{:.3f}", "write() median, ms")
v("ack_fsync", A["fsync"], "{:.3f}", "fsync() median, ms")
v("ack_full", A["full"], "{:.1f}", "F_FULLFSYNC median, ms")
v("ack_ratio", A["full"] / A["fsync"], "{:.0f}", "F_FULLFSYNC / fsync")
v("gc_kmin", _ks[0], "{:.0f}", "fewest intents per flush")
v("gc_kmax", _ks[-1], "{:.0f}", "most intents per flush")
v("gc_flush_lo", min(_flush), "{:.0f}", "shortest flush, ms")
v("gc_flush_hi", max(_flush), "{:.1f}", "longest flush, ms")
v("gc_per_hi", A["group"][_ks[0]]["per_record_ms"], "{:.1f}", "storage work per intent, one per flush, ms")
v("gc_per_lo", A["group"][_ks[-1]]["per_record_ms"], "{:.3f}", "storage work per intent, most per flush, ms")

print("== Group commit (Scenarios.IntentQuorumGroupCommit)")
G = GROUP.result
v("gc_timer", mag(GROUP.batch_timer, "ms"), "{:.0f}", "batch timer, ms")
v("gc_flush", mag(GROUP.flush_time, "ms"), "{:.0f}", "quorum flush, ms")
v("gc_batch", GROUP.batch_size, "{:.0f}", "intents per flush")
v("gc_work", mag(G.work_per_record, "ms"), "{:.2f}", "storage work per intent, ms")
v("gc_first", mag(G.first_wait, "ms"), "{:.0f}", "first arrival waits, ms")
v("gc_last", mag(G.last_wait, "ms"), "{:.0f}", "last arrival waits, ms")

print("== Full snapshots against deltas (Scenarios.GrowingTrajectorySnapshots)")
SN = SNAP.result
v("sn_turns", SNAP.turns, "{:.0f}", "turns")
v("sn_growth", mag(SNAP.state_growth, "kB"), "{:.0f}", "state growth per turn, KB")
v("sn_stride", SNAP.stride, "{:.0f}", "checkpoint stride, turns")
v("sn_n", SN.checkpoints, "{:.0f}", "checkpoints")
v("sn_full", mag(SN.full_bytes, "GB"), "{:.1f}", "full snapshots, GB")
v("sn_delta", mag(SN.delta_bytes, "MB"), "{:.0f}", "deltas, MB")
v("sn_ratio", SN.ratio, "{:.1f}", "full / delta")

print("== Checkpoint strides on the trace (Scenarios.TwoCrashCheckpointTrace)")
T = TRACE.result
_c1, _c2 = TRACE.crashes
v("cd_turns", TRACE.turns, "{:.0f}", "turns")
v("cd_crash_a", _c1, "{:.0f}", "first crash, after turn")
v("cd_crash_b", _c2, "{:.0f}", "second crash, after turn")
v("cd_crashes_words", len(TRACE.crashes), "words", "crashes, in words")
v("cd_pause", mag(TRACE.pause, "s"), "{:.1f}", "pause per checkpoint, s")
v("cd_restore", mag(TRACE.restore, "s"), "{:.0f}", "restore per crash, s")
v("cd_rebuild", mag(TRACE.rebuild_per_turn, "s"), "{:.1f}", "rebuild one logged turn, s")
v("cd_reexec", mag(TRACE.reexecute_per_turn, "s"), "{:.0f}", "re-execute one unlogged turn, s")
for _k, _name in zip(TRACE.strides, ("a", "b", "c")):
    _lost_a, _lost_b = T.lost_turns[_k]
    v(f"cd_s_{_name}", _k, "{:.0f}", "stride, turns")
    v(f"cd_ck_{_name}", T.checkpoints[_k], "{:.0f}", "checkpoints")
    v(f"cd_cap_{_name}", mag(T.capture_time[_k], "s"), "{:.0f}", "checkpoint pauses, s")
    v(f"cd_lost_a_{_name}", _lost_a, "{:.0f}", "turns rebuilt after the first crash")
    v(f"cd_lost_b_{_name}", _lost_b, "{:.0f}", "turns rebuilt after the second crash")
    v(f"cd_rb_{_name}", mag((_lost_a + _lost_b) * TRACE.rebuild_per_turn, "s"), "{:.0f}", "rebuild, s")
    v(f"cd_tot_{_name}", mag(T.total_time[_k], "s"), "{:.0f}", "total, s")
v("cd_rest_all", mag(len(TRACE.crashes) * TRACE.restore, "s"), "{:.0f}", "restores, s")
v("cd_every", mag(T.every_turn_time, "s"), "{:.0f}", "checkpoint every turn, s")
v("cd_rate", T.crash_rate, "{:.2f}", "crashes per turn")
v("cd_k_log", T.stride_logged, "{:.0f}", "balance stride, logged turns")
v("cd_k_unlog", T.stride_reexecuted, "{:.1f}", "balance stride, re-executed turns")
v("cd_gain", T.stride_gain, "{:.1f}", "stride gain")
v("cd_gain_round", T.stride_gain, "{:.0f}", "stride gain, whole")
_small, _big = YOUNG.miss_factors
v("yg_miss", _small, "{:.0f}", "miss factor")
v("yg_miss_pct", pct(YOUNG.miss_penalty(_small)), "{:.0f}", "extra overhead at that miss, %")

print("== Next call after a resume (data/resume_prefill.json; Scenarios.ResumedContextColdPrefill)")
P = prefill()
v("pf_tokens", P["tokens"], "{:,.0f}", "context, tokens")
v("pf_cold", P["cold"], "{:.1f}", "model reloaded, median s")
v("pf_warm", P["warm"], "{:.2f}", "prefix held, median s")
v("pf_ratio", P["cold"] / P["warm"], "{:.0f}", "cold / warm")
v("pf_load", P["load"], "{:.2f}", "model load, median s")
v("pf_n", P["n"], "{:.0f}", "runs per case")
v("pf_ctx", PREFILL.prompt_tokens, "{:,.0f}", "chapter's resumed context, tokens")
v("pf_floor", mag(PREFILL.result.prefill_time, "s"), "{:.1f}", "chapter's prefill floor, s")

print("== Temperature zero (Literature.InferenceDeterminism; Scenarios.GreedyRerunReproduction)")
v("tm_n", float(TM.ThinkingMachines2025Temp0Completions), "{:,.0f}", "completions")
v("tm_unique", float(TM.ThinkingMachines2025Temp0UniqueCompletions), "{:.0f}", "distinct completions")
v("tm_prefix", float(TM.ThinkingMachines2025Temp0IdenticalPrefixTokens), "{:.0f}", "identical tokens")
v("tm_major", float(TM.ThinkingMachines2025Temp0MajorityContinuations), "{:.0f}", "Queens, New York")
v("tm_minor", float(TM.ThinkingMachines2025Temp0MinorityContinuations), "{:.0f}", "New York City")
v("rp_h", RERUN.short_horizon, "{:.0f}", "calls in the trajectory")
v("rp_q", RERUN.step_success, "{:.2f}", "per-call reproduction")
v("rp_q_pct", pct(RERUN.step_success), "{:.0f}", "per-call reproduction, %")
v("rp_all_pct", pct(RERUN.result.success[RERUN.short_horizon]), "{:.0f}", "all calls reproduce, %")

print("== A year of recovery data (Scenarios.RepairTrajectoryRetention)")
K = KEEP.result
v("rt_turns", KEEP.turns, "{:.0f}", "turns")
v("rt_env", mag(KEEP.envelope, "kB"), "{:.0f}", "envelopes per turn, KB")
v("rt_payload", mag(KEEP.payload, "kB"), "{:.0f}", "inputs and results per turn, KB")
v("rt_log", mag(K.log_bytes, "MB"), "{:.2f}", "log, MB")
v("rt_ws", mag(KEEP.workspace, "MB"), "{:.0f}", "workspace snapshot, MB")
v("rt_total", mag(K.total_bytes, "MB"), "{:.0f}", "total, MB")
v("rt_rate", mag(KEEP.rate, "USD / TB / month"), "{:.0f}", "$ per TB-month")
v("rt_months", mag(KEEP.period, "month"), "{:.0f}", "months kept")
v("rt_cents", pct(mag(K.cost, "USD")), "{:.1f}", "cost of the year, cents")
v("rt_attempt", mag(KEEP.attempt_cost, "USD"), "{:.2f}", "one attempt, $")
v("rt_share_pct", pct(K.attempt_share), "{:.1f}", "share of an attempt, %")
v("rt_log_pct", pct(K.log_share), "{:.2f}", "log alone, share of an attempt, %")
v("rt_ws_pct", pct(K.workspace_share), "{:.0f}", "workspace share of the bytes, %")

if __name__ == "__main__":
    write(__file__)
