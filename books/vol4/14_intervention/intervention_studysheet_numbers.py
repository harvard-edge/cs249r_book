"""Numbers for the Supervisory Intervention study sheet (Volume IV, Chapter 14).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to intervention_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Embodied, Scenarios


def f(x, nd):
    return f"{x:,.{nd}f}"


def resident_stop_mm(v, a, ramp_ms):
    """Jerk-limited resident stop v^2/(2a) + v * a/(2j), from printed values (mm)."""
    brake = round(v ** 2 / (2 * a) * 1e3, 1)
    rampt = round(v * ramp_ms, 1)
    return brake, rampt, round(brake + rampt, 1)


def numbers() -> dict:
    n = {}
    aisle = Embodied.Scenario.WarehouseAisle

    # ---- Shared givens (chapter: InterventionTakeoverBudget, InterventionCeilingRoot)
    at_aisle = Scenarios.WarehouseInLoopTakeoverAtAisleSpeed
    v = at_aisle.speed.to("m/s").magnitude
    tau = round(aisle.tau_delay.to("ms").magnitude, 1)
    a = aisle.a_brake.to("m/s^2").magnitude
    j = aisle.j_brake.to("m/s^3").magnitude
    ramp = round(a / (2 * j) * 1e3, 1)                     # ms, the half jerk ramp a/(2j)
    dclear = aisle.d_clear.to("mm").magnitude
    over = aisle.fixed_overhead.to("mm").magnitude
    avail = dclear - over
    dda = round(aisle.t_dispatch_to_admission.to("ms").magnitude, 1)
    tin = at_aisle.takeover.to("ms").magnitude
    tool = Scenarios.WarehouseOutOfLoopTakeoverAtAisleSpeed.takeover.to("s").magnitude
    chunk = aisle.t_lease.to("ms").magnitude
    n.update(wv=f(v, 1), wtau=f(tau, 1), wa=f(a, 0), wj=f(j, 2), wramp=f(ramp, 1),
             wclear=f(dclear / 1e3, 2), wclearmm=f(dclear, 0), wover=f(over, 0), wavail=f(avail, 0),
             wdda=f(dda, 1), tin=f(tin, 0), tool=f(tool, 1), toolms=f(tool * 1e3, 0), chunk=f(chunk, 0),
             vknee=f(a ** 2 / j, 2))

    # ---- Takeover delay terms (Mode 1, slow end of the Model Human Processor; console = video + handshake)
    ts = aisle.t_human_perceptual_slow.to("ms").magnitude
    tc = aisle.t_human_cognitive_slow.to("ms").magnitude
    tm = aisle.t_human_motor_slow.to("ms").magnitude
    tvid = aisle.t_video_glass_to_glass.to("ms").magnitude
    thandshake = aisle.t_handshake.to("ms").magnitude
    tcomm = aisle.t_takeover_transport.to("ms").magnitude
    assert abs(ts + tc + tm + tcomm - tin) < 1e-9 and abs(tvid + thandshake - tcomm) < 1e-9
    n.update(tsens=f(ts, 0), tcog=f(tc, 0), tmot=f(tm, 0), tvid=f(tvid, 0), thand=f(thandshake, 0), tcomm=f(tcomm, 0))

    # ---- Worked example: latest fallback time at the aisle speed, complete budget
    wbrake, wrampt, wres = resident_stop_mm(v, a, ramp)
    assert abs(wres - at_aisle.result.machine.scurve_travel.to("mm").magnitude) < 0.1
    wroom = round(avail - wres, 1)
    wper = round(wroom / v, 1)                              # mm / (m/s) = ms
    wdead = round(wper - tau, 1)
    assert abs(wdead - at_aisle.result.deadline.to("ms").magnitude) < 0.06
    wleft = round(wdead - dda, 1)
    assert abs(wleft - at_aisle.result.deadline_after_admission.to("ms").magnitude) < 0.06
    n.update(wbrake=f(wbrake, 1), wrampt=f(wrampt, 1), wres=f(wres, 1), wroom=f(wroom, 1), wper=f(wper, 1),
             wdead=f(wdead, 1), wleft=f(wleft, 1), wratio=f(tin / wleft, 0))

    # ---- Worked-example picture: the complete budget at the aisle speed to scale (1 mm = 0.2 pt)
    deltrav = round(v * tau, 1)
    datrav = round(v * dda, 1)
    wspare = round(dclear - over - wres - deltrav - datrav, 1)
    assert abs(wspare - at_aisle.result.machine_with_dispatch.spare.to("mm").magnitude) < 0.15
    taketrav = round(v * tin, 1)
    ps = 0.12
    wstopall = round(deltrav + datrav + taketrav + wres + over, 1)
    assert abs(wstopall - at_aisle.result.with_takeover_and_dispatch.required_distance.to("mm").magnitude) < 0.15
    n.update(wstopall=f(wstopall, 1), pres2=f(wres * ps, 2))
    n.update(wdeltrav=f(deltrav, 1), wdatrav=f(datrav, 1), wspare=f(wspare, 1), wtaketrav=f(taketrav, 1),
             pover=f(over * ps, 2), pres=f(wres * ps, 2), pdel=f(deltrav * ps, 2), pda=f(datrav * ps, 2),
             pclear=f(dclear * ps, 2), ptake=f(taketrav * ps, 2), pspare=f(wspare * ps, 2))

    # ---- Picture: the in-loop takeover as a to-scale stage strip, 1 ms = 0.36 pt
    sc = 0.36
    n.update(sv=f(tvid * sc, 2), ss=f(ts * sc, 2), sc=f(tc * sc, 2), sm=f(tm * sc, 2), sh=f(thandshake * sc, 2),
             sleft=f(wleft * sc, 2), stot=f(tin * sc, 2))

    # ---- Your turn: the same takeover at the retired 0.8 m/s slow-down (one condition changed: speed)
    ret = Scenarios.WarehouseInLoopTakeoverAtRetiredSlowdown
    yv = ret.speed.to("m/s").magnitude
    ybrake, yrampt, yres = resident_stop_mm(yv, a, ramp)
    yroom = round(avail - yres, 1)
    yper = round(yroom / yv, 1)
    ydead = round(yper - tau, 1)
    yafter = round(ydead - tin, 1)
    yleft = round(yafter - dda, 1)
    assert abs(yafter - ret.result.time_left.to("ms").magnitude) < 0.06
    assert abs(yleft - ret.result.time_left_after_admission.to("ms").magnitude) < 0.06 and yleft < 0
    ceil_partial = at_aisle.result.with_takeover.speed_ceiling.to("m/s").magnitude
    n.update(yv=f(yv, 2), ybrake=f(ybrake, 1), yrampt=f(yrampt, 1), yres=f(yres, 1), yroom=f(yroom, 1),
             yper=f(yper, 1), ydead=f(ydead, 1), yafter=f(yafter, 1), yshort=f(-yleft, 1),
             ceilpartial=f(ceil_partial, 2))

    # ---- Session budget table: in-loop session at the 0.75 m/s slow-down (InterventionAssistantSession)
    req = Scenarios.WarehouseInLoopTakeoverAtRequestSpeed
    r = req.result
    vin = req.speed.to("m/s").magnitude
    mlease = req.manual_lease.to("ms").magnitude
    tspare = round(r.machine.spare.to("mm").magnitude, 1)
    ttake = round(vin * tin, 1)
    tleft1 = round(tspare - ttake, 1)
    tda = round(vin * dda, 1)
    tleft2 = round(tleft1 - tda, 1)
    tlease = round(vin * (mlease - chunk), 1)
    tleft3 = round(tleft2 - tlease, 1)
    assert abs(tleft1 - r.spare_after_takeover.to("mm").magnitude) < 0.15
    assert abs(tleft2 - r.spare_after_admission.to("mm").magnitude) < 0.15
    assert abs(tleft3 - r.spare_after_lease.to("mm").magnitude) < 0.15
    ceil_da = at_aisle.result.with_takeover_and_dispatch.speed_ceiling.to("m/s").magnitude
    n.update(vin=f(vin, 2), mlease=f(mlease, 0), tspare=f(tspare, 1), ttake=f(ttake, 1), tleft1=f(tleft1, 1),
             tda=f(tda, 1), tleft2=f(tleft2, 1), tlease=f(tlease, 1), tleft3=f(tleft3, 1),
             ceilda=f(ceil_da, 3), vool=f(aisle.v_takeover_out_of_loop.to("m/s").magnitude, 2))

    # ---- Q1: the chapter's out-of-loop (monitoring) takeover delay; ceiling on the complete budget
    tin_o = Scenarios.WarehouseOutOfLoopTakeoverAtAisleSpeed.takeover.to("ms").magnitude
    b = round((tau + dda + tin_o + ramp) / 1e3, 4)          # s
    rad = round(b ** 2 + 2 * (avail / 1e3) / a, 4)
    root = round(math.sqrt(rad), 4)
    qv = a * (root - b)
    # below the knee the card's plateau stop overstates the true jerk-limited stop, so this bound is conservative
    true_ceiling = Scenarios.WarehouseOutOfLoopTakeoverAtAisleSpeed.result.with_takeover_and_dispatch.speed_ceiling.to("m/s").magnitude
    assert qv < a ** 2 / j and qv <= true_ceiling and abs(qv - true_ceiling) < 0.005
    n.update(qb=f(b, 4), qrad=f(rad, 4), qroot=f(root, 4), qceil=f(qv, 3))

    # ---- Q2: ASSUMED arm joint-torque change 24 N.m under the chapter's 750 N.m/s torque-rate ceiling
    smax = 15 / 8                                           # peak slope of the quintic smoothstep
    rate = 750.0                                            # N.m/s, chapter InterventionBlendBudget arm_rate
    dq = 24.0
    qblend = smax * dq / rate * 1e3
    n.update(smax=f(smax, 3), qrate=f(rate, 0), qdtorque=f(dq, 0), qblend=f(qblend, 0),
             armdelta=f(15, 0), armblend=f(smax * 15 / rate * 1e3, 1))

    # ---- Q3: ASSUMED a 150 ms manual lease for a lossy link at the 0.75 m/s slow-down
    ql = 150.0
    qtravel = round(vin * (ql - chunk), 1)
    n.update(qlease=f(ql, 0), qextra=f(ql - chunk, 0), qtravel=f(qtravel, 1), qmiss=f(qtravel - tleft2, 1))

    # ---- Mistake 1: a fixed 500 ms window (chapter: InterventionFixedWindowPitfall, partial budget)
    fw = Scenarios.WarehouseFixedWindowAtAisleSpeed.result
    fc = Scenarios.WarehouseFixedWindowAtCrawl
    n.update(window=f(Scenarios.WarehouseFixedWindowAtAisleSpeed.takeover.to("ms").magnitude, 0),
             fwtravel=f(fw.takeover_travel.to("m").magnitude, 2),
             fwover=f(-fw.with_takeover.spare.to("m").magnitude, 2),
             fwcrawl=f(fc.speed.to("m/s").magnitude, 2), fwmargin=f(fc.result.deadline.to("s").magnitude, 1))
    # book finding: the same overrun with the dispatch-to-admission travel charged (complete budget)
    n.update(fwovercomplete=f(-fw.with_takeover.spare.to("m").magnitude
                              + v * aisle.t_dispatch_to_admission.to("s").magnitude, 2))

    # ---- Mistake 2: unannounced split channels (definition box, mode confusion)
    trav = at_aisle.result.takeover_travel.to("mm").magnitude
    sp = at_aisle.result.machine_with_dispatch.spare.to("mm").magnitude
    n.update(m2travel=f(trav, 1), m2spare=f(sp, 1), m2ratio=f(trav / sp, 1))
    # ---- Session table: the base's braking blend beyond the resident stop (chapter: InterventionBlendBudget)
    blend = Scenarios.WarehouseTakeoverBrakingBlend.result
    bover = round(blend.excess_over_scurve.to("mm").magnitude, 1)
    assert abs(blend.speed.to("m/s").magnitude - vin) < 1e-12 and bover > tleft3
    n.update(blendover=f(bover, 1), blendshort=f(bover - tleft3, 1))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
