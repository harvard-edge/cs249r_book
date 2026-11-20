"""Numbers for the Physical Data study sheet (Volume IV, Chapter 5).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to data_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Scenarios
from mlsysim.core.units import meter, millimeter, second


def f(x, nd):
    """Round half up (as a student would) and print with thousands separators."""
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"{q:,.{nd}f}"


def numbers() -> dict:
    n = {}

    # ---- Front yield card and worked example: the chapter's latch campaign
    # (chapter: DemonstrationYieldFatigue, Scenarios.LatchDemonstrationCampaign)
    camp = Scenarios.LatchDemonstrationCampaign
    sched = camp.session.to("min").magnitude
    motion = camp.active.to("min").magnitude
    att = camp.attempt.to("s").magnitude
    rst = camp.reset.to("s").magnitude
    cyc = att + rst
    attempts = math.floor(motion * 60 / cyc)
    assert attempts == camp.attempts_per_session
    ret = (1 - camp.failure_rate) * (1 - camp.qa_rate)
    assert abs(ret - camp.retention) < 1e-12
    kept = attempts * round(ret, 2)
    target = camp.target_kept
    total = math.ceil(target / round(ret, 2))
    assert total == camp.campaign_attempts
    hours = total / attempts  # one session is one scheduled hour
    assert abs(hours - camp.campaign_time.to("h").magnitude) < 1e-9
    ncable = camp.cable_flexes_per_attempt
    ngear = camp.gear_reversals_per_attempt
    lcable = camp.cable_rated_cycles
    lgear = camp.gear_rated_cycles
    cable = total * ncable
    gear = total * ngear
    assert cable == camp.cable_cycles and gear == camp.gear_cycles
    n.update(sched=f(sched, 0), motion=f(motion, 0), attempt=f(att, 0), reset=f(rst, 0), cycle=f(cyc, 0),
             failpct=f(camp.failure_rate * 100, 0), qapct=f(camp.qa_rate * 100, 0),
             failshare=f(camp.failure_rate, 2), qashare=f(camp.qa_rate, 2), retention=f(ret, 2),
             attempts=f(attempts, 0), motionsec=f(motion * 60, 0), kept=f(kept, 1), target=f(target, 0),
             campaign=f(total, 0), hours=f(hours, 1), cableper=f(ncable, 0), gearper=f(ngear, 0),
             cablecycles=f(cable, 0), gearcycles=f(gear, 0), cablelife=f(lcable, 0), gearlife=f(lgear, 0),
             cablepct=f(cable / lcable * 100, 1), gearpct=f(gear / lgear * 100, 1))

    # picture: to-scale stacked bar of the scheduled hour, 3 pt per minute
    over = (camp.startup + camp.calibration + camp.rest + camp.idle).to("min").magnitude
    resets = camp.reset_time.to("min").magnitude
    rejected = (camp.failed_time + camp.qa_time).to("min").magnitude
    keptmin = camp.kept_time.to("min").magnitude
    assert abs(over + resets + rejected + keptmin - sched) < 1e-9
    sc = 3.0
    n.update(baroverhead=f(over * sc, 2), barresets=f(resets * sc, 2), barrejected=f(rejected * sc, 2),
             barkept=f(keptmin * sc, 2), bartotal=f(sched * sc, 2), keptmin=f(keptmin, 1),
             keptpct=f(camp.kept_share * 100, 1))

    # ---- Your turn: ASSUMED faster fixture reset, 10 s instead of 15 s; nothing else changes
    rst2 = 10.0
    turn = camp.model_copy(update={"reset": rst2 * second})
    cyc2 = att + rst2
    attempts2 = math.floor(motion * 60 / cyc2)
    assert attempts2 == turn.attempts_per_session
    kept2 = attempts2 * round(ret, 2)
    total2 = math.ceil(target / round(ret, 2))
    assert total2 == turn.campaign_attempts
    hours2 = total2 / attempts2
    assert abs(hours2 - turn.campaign_time.to("h").magnitude) < 1e-9
    assert abs(turn.cable_life_used - camp.cable_life_used) < 1e-12
    n.update(treset=f(rst2, 0), tcycle=f(cyc2, 0), tattempts=f(attempts2, 0), tkept=f(kept2, 1),
             thours=f(hours2, 1), tsaved=f(float(f(hours, 1)) - float(f(hours2, 1)), 1))

    # ---- Front pairing-lag card (chapter: SyncDisplacement, Scenarios.LatchPairingLag)
    pair = Scenarios.LatchPairingLag
    lag = pair.lag.to("ms").magnitude
    clear = pair.clearance.to("mm").magnitude
    vg = pair.guarded_speed.to("m/s").magnitude
    vf = pair.fast_speed.to("m/s").magnitude
    assert abs(vg * lag - pair.displacement_guarded.to("mm").magnitude) < 1e-9
    n.update(lag=f(lag, 0), clearance=f(clear, 0), vguarded=f(vg, 2), vfast=f(vf, 2),
             dxguarded=f(vg * lag, 2), dxfast=f(vf * lag, 1),
             ceilfast=f(pair.lag_ceiling_fast.to("ms").magnitude, 0),
             skewlow=f(pair.shutter_skew_low.to("ms").magnitude, 0),
             skewhigh=f(pair.shutter_skew_high.to("ms").magnitude, 0))

    # ---- Q1: ASSUMED approach at 0.06 m/s with the same logger lag, clearance and shutter
    v1 = 0.06  # m/s, ASSUMED
    skew = pair.shutter_skew_high.to("ms").magnitude
    n.update(qspeed=f(v1, 2), qdx=f(v1 * lag, 1), qceil=f(clear / v1, 1), qskewdx=f(v1 * skew, 2))

    # ---- Front bound card (chapter: BoundComparison, Scenarios.ImitationBoundsFiveSecondEpisode)
    bnd = Scenarios.ImitationBoundsFiveSecondEpisode
    S = bnd.horizon_steps
    n.update(episode=f(bnd.episode.to("s").magnitude, 0), ctrl=f(bnd.control_rate.to("Hz").magnitude, 0),
             horizon=f(S, 0), eps=f(bnd.per_step_error, 3), clone=f(bnd.cloning, 0),
             agg=f(bnd.aggregation, 1), vacuous=f(bnd.vacuous_error, 3))

    # ---- Q2: ASSUMED 2 s grasp episode at the same rate, eps = 0.004, recoverable step (u = 1)
    q2 = bnd.model_copy(update={"episode": 2 * second, "per_step_error": 0.004})
    S2 = q2.horizon_steps
    assert abs(q2.cloning - S2 ** 2 * 0.004) < 1e-9
    n.update(qepisode=f(2, 0), qeps=f(0.004, 3), qhorizon=f(S2, 0), qvacuous=f(1 / S2, 3),
             qclone=f(q2.cloning, 0), qagg=f(q2.aggregation, 1), urec=f(bnd.recoverability, 0))

    # ---- Mistake 1: one hour of permission-tick telemetry (chapter: EffectiveSampleSize)
    log = Scenarios.PermissionTickTelemetryHour
    n.update(lograte=f(log.rate.to("kHz").magnitude, 0), records=f(log.records / 1e6, 1),
             rho=f(log.lag1_correlation, 3), effective=f(round(log.effective_samples, -2), 0))

    # ---- Mistake 2: two archives on the contact grid (chapter: ContactGridCoverage)
    grid = Scenarios.ContactAngleCoverage
    c_rms = grid.archives["cautious"][4].to("rad").magnitude
    e_rms = grid.archives["exploratory"][4].to("rad").magnitude
    n.update(gridcells=f(grid.cells, 0), cautiouscells=f(grid.occupied("cautious"), 0),
             exploratorycells=f(grid.occupied("exploratory"), 0), cautiousloss=f(c_rms, 3),
             exploratoryloss=f(e_rms, 3), cautiouspct=f(grid.share("cautious") * 100, 1))

    # ---- Key idea 5: the latch scenario table (chapter: LatchScenarioTable)
    tbl = Scenarios.LatchScenarioTable
    n.update(cells=f(tbl.cells, 0), supported=f(tbl.supported, 0))

    # ---- Q4: chapter's two grippers; ASSUMED normalized command 0.55 and a 40 mm part
    grip = Scenarios.GripperStrokeTransfer.model_copy(update={"command": 0.55, "part_width": 40 * millimeter})
    sa = grip.donor_stroke.to("mm").magnitude
    sb = grip.target_stroke.to("mm").magnitude
    open_a = grip.donor_opening.to("mm").magnitude
    open_b = grip.target_opening.to("mm").magnitude
    assert grip.donor_clears and not grip.target_clears
    n.update(strokedonor=f(sa, 0), stroketarget=f(sb, 0), gcommand=f(grip.command, 2),
             part=f(grip.part_width.to("mm").magnitude, 0), opendonor=f(open_a, 0), opentarget=f(open_b, 0),
             relabel=f(open_a / sb, 2))
    _ = meter
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
