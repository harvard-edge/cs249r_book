"""Numbers for the Durable Execution study sheet (Volume III, Chapter 11).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(CheckpointCadence, GroupCommit, RerunReproduction, RetentionCost), and written as
\\setnum{key}{value} to durable_execution_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Scenarios as SC


def fx(x, nd):
    """Round half up, as a reader would, and format with nd decimals."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(x, 9))).quantize(q, rounding=ROUND_HALF_UP))   # round(.., 9) drops float noise


def r(x, nd):
    return float(fx(x, nd))


def pct(x, nd=0):
    return fx(x * 100, nd)


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def trace(turns, crashes, stride, pause, restore, rebuild):
    """Capture and reconstruction time on a fixed crash trace, checkpointing every stride and at the end."""
    marks = sorted(set(range(stride, turns + 1, stride)) | {turns})
    lost = [c - max([0] + [m for m in marks if m <= c]) for c in crashes]
    return len(marks), lost, len(marks) * pause, len(crashes) * restore + rebuild * sum(lost)


def numbers() -> dict:
    n = {}

    # ---- Worked example: checkpoint cadence on a fixed crash trace (chapter: CheckpointCadence)
    sc = SC.TwoCrashCheckpointTrace
    res = sc.result
    turns, crashes, strides = sc.turns, list(sc.crashes), list(sc.strides)
    pause = sc.pause.to("second").magnitude
    restore = sc.restore.to("second").magnitude
    rebuild = sc.rebuild_per_turn.to("second").magnitude
    reexec = sc.reexecute_per_turn.to("second").magnitude
    names = {10: "ten", 25: "quarter", 50: "half"}
    totals = {}
    for k in strides:
        count, lost, cap, rec = trace(turns, crashes, k, pause, restore, rebuild)
        assert count == res.checkpoints[k] and tuple(lost) == tuple(res.lost_turns[k])
        near(cap, res.capture_time[k].to("second").magnitude, 1e-9)
        near(rec, res.recovery_time[k].to("second").magnitude, 1e-9)
        totals[k] = cap + rec
        w = names[k]
        n.update({f"stride{w}": f"{k}", f"count{w}": f"{count}", f"lostA{w}": f"{lost[0]}",
                  f"lostB{w}": f"{lost[1]}", f"lostsum{w}": f"{sum(lost)}", f"capture{w}": fx(cap, 0),
                  f"recovery{w}": fx(rec, 0), f"total{w}": fx(cap + rec, 0)})
    best = min(totals, key=totals.get)
    assert best == res.best_stride == 25
    rate = len(crashes) / turns
    near(rate, res.crash_rate, 1e-12)
    k_log = math.sqrt(2 * pause / (rate * rebuild))
    k_exec = math.sqrt(2 * pause / (rate * reexec))
    near(k_log, res.stride_logged, 1e-9)
    near(k_exec, res.stride_reexecuted, 1e-9)
    gain = math.sqrt(reexec / rebuild)
    near(gain, res.stride_gain, 1e-9)
    every = trace(turns, crashes, 1, pause, restore, rebuild)
    near(every[2] + every[3], res.every_turn_time.to("second").magnitude, 1e-9)
    n.update(turns=f"{turns}", crashA=f"{crashes[0]}", crashB=f"{crashes[1]}", ncrash=f"{len(crashes)}",
             pause=fx(pause, 1), restore=fx(restore, 0), rebuild=fx(rebuild, 1), reexec=fx(reexec, 0),
             best=f"{best}", rate=fx(rate, 2), klog=fx(k_log, 0), kexec=fx(k_exec, 1), gain=fx(gain, 0),
             restoretwo=fx(len(crashes) * restore, 0), everyturn=fx(every[2] + every[3], 0),
             balanceinside=fx(2 * pause / (rate * rebuild), 0), execinside=fx(2 * pause / (rate * reexec), 1))

    # ---- Your turn. ASSUMED: the capture pause rises to 2 s (four times); all else unchanged
    pause2 = 2.0
    t2 = {}
    for k in strides:
        count, lost, cap, rec = trace(turns, crashes, k, pause2, restore, rebuild)
        t2[k] = (cap, rec)
        w = names[k]
        n.update({f"slowcap{w}": fx(cap, 0), f"slowtotal{w}": fx(cap + rec, 0)})
    assert min(t2, key=lambda k: sum(t2[k])) == 25
    k_slow = math.sqrt(2 * pause2 / (rate * rebuild))
    near(k_slow / k_log, math.sqrt(pause2 / pause), 1e-9)                # quadrupling the pause doubles it
    n.update(slowpause=fx(pause2, 0), kslow=fx(k_slow, 0), slowinside=fx(2 * pause2 / (rate * rebuild), 0),
             pausefactor=fx(pause2 / pause, 0), strideratio=fx(k_slow / k_log, 0))

    # ---- Q1. ASSUMED: a 3 ms batch timer, a 5 ms quorum flush, 25 intents per batch
    # (method as the chapter's GroupCommit: first waits timer + flush, last waits the flush, work = flush / batch)
    gc = SC.IntentQuorumGroupCommit
    gt, gf, gb = (gc.batch_timer.to("ms").magnitude, gc.flush_time.to("ms").magnitude, gc.batch_size)
    near(gt + gf, gc.result.first_wait.to("ms").magnitude, 1e-9)
    near(gf / gb, gc.result.work_per_record.to("ms").magnitude, 1e-9)
    q_timer, q_flush, q_batch = 3.0, 5.0, 25
    n.update(qtimer=fx(q_timer, 0), qflush=fx(q_flush, 0), qbatch=f"{q_batch}",
             qfirst=fx(q_timer + q_flush, 0), qlast=fx(q_flush, 0), qwork=fx(q_flush / q_batch, 1),
             qtimertwo=fx(2 * q_timer, 0), qfirsttwo=fx(2 * q_timer + q_flush, 0))

    # ---- Mistake: a temperature-zero rerun (chapter: RerunReproduction, Scenarios.GreedyRerunReproduction)
    rr = SC.GreedyRerunReproduction
    q, h = rr.step_success, rr.short_horizon
    p_all = rr.result.success[h]
    near(p_all, q ** h, 1e-12)
    n.update(rerunq=pct(q), rerunh=f"{h}", rerunall=pct(p_all), rerunqfrac=fx(q, 2))

    # ---- Mistake: pruning the log (chapter: RetentionCost, Scenarios.RepairTrajectoryRetention)
    rt = SC.RepairTrajectoryRetention
    rres = rt.result
    cost_usd = rres.cost.to("USD").magnitude
    attempt = rt.attempt_cost.to("USD").magnitude
    near(rres.attempt_share, cost_usd / attempt, 1e-12)
    n.update(retcents=fx(cost_usd * 100, 1), retmonths=fx(rt.period.to("month").magnitude, 0),
             retattempt=fx(attempt, 2), retshare=pct(rres.attempt_share, 1),
             retlogshare=pct(rres.log_share, 2), retworkspace=pct(rres.workspace_share))

    # ---- Resume budget: the cold prefill floor (chapter: ResumePrefill, Scenarios.ResumedContextColdPrefill)
    cp = SC.ResumedContextColdPrefill
    t_cold = cp.result.prefill_time.to("second").magnitude
    assert 2.0 < t_cold < 2.7                                          # the chapter's guard: about 2.3 s
    n.update(coldtokens=f"{cp.prompt_tokens:,}", coldprefill=fx(t_cold, 1))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
