"""Numbers for the Workload Placement study sheet (Volume IV, Chapter 13).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to placement_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Embodied, Scenarios
from mlsysim.core.units import Q_


def f(x, nd):
    return f"{x:,.{nd}f}"


def us(q):
    return q.to("us").magnitude


def numbers() -> dict:
    n = {}
    aisle = Embodied.Scenario.WarehouseAisle
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator

    # ---- Front: the permission tick's budget (chapter: PlacementMachine)
    unloaded = us(aisle.enforcer_unloaded)
    wcet = us(aisle.enforcer_wcet)
    deadline = us(aisle.enforcer_deadline)
    period = aisle.t_tick.to("ms").magnitude
    blocking = us(aisle.enforcer_blocking)
    interf = us(aisle.enforcer_interference)
    resp = Scenarios.WarehousePermissionTickResponse.result
    ceiling = us(resp.execution_ceiling)
    allowance = us(resp.margin)
    assert abs(ceiling - (deadline - blocking - interf)) < 1e-9 and abs(allowance - (ceiling - wcet)) < 1e-9
    n.update(unloaded=f(unloaded, 0), wcet=f(wcet, 0), deadline=f(deadline, 0), period=f(period, 0),
             blocking=f(blocking, 0), interf=f(interf, 0), bplusi=f(blocking + interf, 0),
             ceiling=f(ceiling, 0), allowance=f(allowance, 0),
             watchdog=f(aisle.mcu_self_watchdog.to("ms").magnitude, 0))
    # picture: stage strip, 1 us = 0.5 pt
    s = 0.5
    n.update(sunl=f(unloaded * s, 2), swcet=f(wcet * s, 2), sceil=f(ceiling * s, 2), sdead=f(deadline * s, 2))

    # ---- Front: stall share (chapter: PlacementInterference)
    mshr = Scenarios.WarehouseTickMshrStallShare
    wb = Scenarios.WarehouseTickWritebackStallShare
    n.update(smshr=f(mshr.slowdown, 0), fmshr=f(mshr.breakeven_share * 100, 1),
             swb=f(wb.slowdown, 0), fwb=f(wb.breakeven_share * 100, 2))

    # ---- Front: relocation toll (chapter: PlacementTransport)
    n.update(tlink=f(aisle.t_board_link.to("ms").magnitude, 2),
             dda=f(aisle.t_dispatch_to_admission.to("ms").magnitude, 1))

    # ---- Worked example: candidate (b)'s thermal hangover (chapter: RejectedPlacementThrottle)
    th = Scenarios.WarehouseSharedDieThermalThrottle
    r = th.result
    tamb = th.ambient.to("degC").magnitude
    pq = th.quiet_power.to("W").magnitude
    pl = th.loaded_power.to("W").magnitude
    rth = th.theta.to("K/W").magnitude
    trip = th.trip.to("degC").magnitude
    clear = th.clear.to("degC").magnitude
    tau = th.time_constant.to("s").magnitude
    fnom = th.nominal_clock.to("GHz").magnitude
    fthr = th.throttled_clock.to("MHz").magnitude
    tq = tamb + pq * rth
    tl = tamb + pl * rth
    assert abs(tq - r.junction_quiet.to("degC").magnitude) < 1e-9
    assert abs(tl - r.junction_loaded.to("degC").magnitude) < 1e-9
    cycles = r.cycles
    texec = cycles / (fthr * 1e6) * 1e6
    assert abs(texec - us(r.execution_throttled)) < 1e-9
    rthr = texec + blocking + interf
    assert abs(rthr - us(r.response_throttled)) < 1e-9
    trec = tau * math.log((trip - tq) / (clear - tq))
    assert abs(trec - r.recovery.to("s").magnitude) < 1e-9
    trec_p = round(trec, 3)
    ticks = round(trec_p * 1e3 / period)  # whole ticks in the printed recovery time
    n.update(tamb=f(tamb, 0), pq=f(pq, 0), pl=f(pl, 0), rth=f(rth, 0), trip=f(trip, 0), clear=f(clear, 0),
             tau=f(tau, 0), fnom=f(fnom, 1), fthr=f(fthr, 0), tq=f(tq, 0), tl=f(tl, 0),
             cycles=f(cycles, 0), texec=f(texec, 0), rthr=f(rthr, 0), over=f(rthr - deadline, 0),
             wnum=f(trip - tq, 0), wden=f(clear - tq, 0), trec=f(trec_p, 3), ticks=f(ticks, 0))
    # picture: cooling curve, x = 20 pt per s over 0..8 s, y = 1.3 pt per degC above 50 degC
    n.update(ptq=f(tq, 1), ptrip=f(trip, 1), pclear=f(clear, 1), ptau=f(tau, 2), ptrec=f(trec * 20, 2))

    # ---- Your turn: ASSUMED hotter bay late in a shift (ambient 50 degC); everything else unchanged
    hot = th.variant(ambient=Q_(50.0, "degC"))
    hr = hot.result
    tamb_y = hot.ambient.to("degC").magnitude
    tq_y = tamb_y + pq * rth
    tl_y = tamb_y + pl * rth
    trec_y = tau * math.log((trip - tq_y) / (clear - tq_y))
    assert abs(trec_y - hr.recovery.to("s").magnitude) < 1e-9
    trec_yp = round(trec_y, 3)
    n.update(yamb=f(tamb_y, 0), ytq=f(tq_y, 0), ytl=f(tl_y, 0), ynum=f(trip - tq_y, 0), yden=f(clear - tq_y, 0),
             ytrec=f(trec_yp, 3), yticks=f(round(trec_yp * 1e3 / period), 0),
             ylonger=f((trec_yp - trec_p) * 1e3, 0))

    # ---- Q2: ASSUMED slowdown of 6 on the stalled share (same tick, same declared bound)
    six = mshr.variant(slowdown=6.0)
    f2 = (wcet / unloaded - 1) / (six.slowdown - 1)
    assert abs(f2 - six.breakeven_share) < 1e-12
    n.update(one=f(1, 0),  # the unit subtracted in the stall-share formula, printed in the answer
             qslow=f(six.slowdown, 0), qratio=f(wcet / unloaded, 3), qshare=f(f2 * 100, 1),
             qstall=f(f2 * unloaded, 0))

    # ---- Q3: ASSUMED loaded cycle of 270 us from a stress campaign on a shared-die candidate
    loaded = Scenarios.WarehouseLoadedCycleCheckpoint.variant(wcet=Q_(270.0, "us"))
    t3 = us(loaded.wcet)
    n.update(qloaded=f(t3, 0), qover=f(t3 - wcet, 0), qresp=f(t3 + blocking + interf, 0),
             qspare=f(us(loaded.result.margin), 0))

    # ---- Q4: watchdog window; c_max = declared WCET + B + I; ASSUMED c_min and phase jitter
    cmax = wcet + blocking + interf
    cmin = 60.0  # ASSUMED shortest healthy completion offset, us
    jit = 20.0  # ASSUMED cycle-to-cycle phase jitter bound, us
    tp = period * 1e3
    lo = tp - (cmax - cmin) - jit
    hi = tp + (cmax - cmin) + jit
    n.update(qcmax=f(cmax, 0), qcmin=f(cmin, 0), qjit=f(jit, 0), qtp=f(tp, 0), qspread=f(cmax - cmin, 0),
             qlo=f(lo, 0), qhi=f(hi, 0))

    # ---- Mistakes: average utilization (chapter: PlacementFallacyMargins), partitioned cache (MSHR)
    read_tail = 320.0  # chapter's illustrative shared-DRAM read tail (PlacementFallacyMargins.read_tail)
    after = (Scenarios.WarehouseEnforcerUnloadedTick.stage("constraint check")
             + Scenarios.WarehouseEnforcerUnloadedTick.stage("drive-frame staging"))
    n.update(readtail=f(read_tail, 0), afterread=f(deadline - read_tail, 0), compute=f(us(after), 0),
             sense=f(us(Scenarios.WarehouseEnforcerUnloadedTick.stage("sense")), 0))
    n.update(mcuclock=f(rm.permission_mcu.compute.clock_rate.to("MHz").magnitude, 0))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
