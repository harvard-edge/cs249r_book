"""Numbers for the Spatial Memory study sheet (Volume IV, Chapter 9).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to memory_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Embodied, Scenarios


def f(x, nd):
    return f"{x:,.{nd}f}"


def numbers() -> dict:
    n = {}
    aisle = Embodied.Scenario.WarehouseAisle
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator

    # ---- Front: predict and update on the rack-end range (chapter: AisleKalman)
    fusion = Scenarios.WarehouseRackEndRangeFusion
    kf = fusion.result
    sig_prior = fusion.prior_sigma.to("mm").magnitude
    sig_w = kf.process_sigma.to("mm").magnitude
    sig_pred = round(kf.predicted_sigma.to("mm").magnitude, 1)
    sig_lidar = rm.lidar.range_precision_max.to("mm").magnitude
    g1 = kf.first_gain
    assert abs(g1 - kf.predicted_sigma.magnitude ** 2 / (kf.predicted_sigma.magnitude ** 2 + sig_lidar ** 2)) < 1e-9
    n.update(sigprior=f(sig_prior, 0), sigw=f(sig_w, 0), sigpred=f(sig_pred, 1), siglidar=f(sig_lidar, 0),
             glidar=f(g1, 2), siglidarpost=f(kf.first_sigma.to("mm").magnitude, 1),
             sigfused=f(kf.fused_sigma.to("mm").magnitude, 1), misshift=f(kf.misdated_shift.to("mm").magnitude, 0),
             missigmas=f(kf.misdated_shift_sigmas, 0))

    # ---- Front tiles and worked example: free space ahead of the hull at the rack end
    # (chapter: RackEndBelief, Scenarios.WarehouseRackEndFreeSpace)
    fs = Scenarios.WarehouseRackEndFreeSpace.result
    dloc = aisle.delta_loc.to("mm").magnitude
    dmar = aisle.delta_margin.to("mm").magnitude
    vh = aisle.v_human_approach.to("m/s").magnitude
    dclear = aisle.d_clear.to("mm").magnitude
    side = aisle.side_clearance.to("mm").magnitude
    age = aisle.perception_age.to("ms").magnitude
    frame = (1 / rm.nav_camera.frame_rate).to("ms").magnitude
    tol_ahead = dclear - dloc - dmar
    t_ahead = tol_ahead / vh  # mm / (m/s) = ms
    assert abs(t_ahead - fs.horizon_ahead.to("ms").magnitude) < 1e-9
    eroded = vh * age  # mm
    t_left = t_ahead - age
    frames_left = t_left / round(frame, 1)
    tol_abreast = side - dloc - dmar
    assert abs(tol_abreast) < 1e-9 and fs.horizon_abreast.magnitude == 0.0
    n.update(dloc=f(dloc, 0), dmar=f(dmar, 0), vh=f(vh, 1), dclear=f(dclear / 1e3, 2), dclearmm=f(dclear, 0),
             side=f(side, 0), age=f(age, 1), frame=f(frame, 1), tolahead=f(tol_ahead, 0),
             tahead=f(t_ahead, 2), eroded=f(eroded, 1), tleft=f(t_left, 2), framesleft=f(math.floor(frames_left), 0),
             tolabreast=f(tol_abreast, 0))
    # picture: to-scale clear distance, 1 mm = 0.19 pt
    sc = 0.19
    n.update(preserve=f((dloc + dmar) * sc, 2), pcert=f(tol_ahead * sc, 2), page=f(eroded * sc, 2),
             pclear=f(dclear * sc, 2))

    # ---- Your turn: same claim, but the hull drives toward the rack end at its drive limit;
    # the certified depth shrinks at v + v_h (chapter, after the erosion notebook).
    v = rm.base.max_velocity.to("m/s").magnitude
    vclose = v + vh
    t_move = round(tol_ahead / vclose, 1)
    t_move_left = t_move - age
    n.update(vdrive=f(v, 1), vclose=f(vclose, 1), tmove=f(t_move, 1), tmoveleft=f(t_move_left, 1),
             framesmove=f(math.floor(t_move_left / round(frame, 1)), 0))

    # ---- Front picture and Q2: latch-slot finger, quadratic growth (chapter: LatchSlotBelief)
    latch_s = Scenarios.LatchSlotFingerBelief
    latch = latch_s.result
    e0 = latch_s.initial_error.to("mm").magnitude
    vd = latch_s.growth_rate.to("mm/s").magnitude
    ad = latch_s.growth_acceleration.to("mm/s**2").magnitude
    rtol = latch_s.tolerance.to("mm").magnitude
    t_latch = latch.horizon.to("ms").magnitude
    t_cv = latch.constant_velocity_horizon.to("ms").magnitude
    n.update(ezero=f(e0, 1), vdrift=f(vd, 0), adist=f(ad, 0), halfa=f(ad / 2, 0), rtol=f(rtol, 1),
             tlatch=f(t_latch, 1), tcv=f(t_cv, 0))
    # curve coefficients for the picture (error in mm, time in ms)
    top = 1.15  # picture: draw the bounded curve up to 1.15 mm (plot top 1.2 mm)
    n.update(ccoef=f(vd / 1e3, 3), qcoef=f(ad / 2 / 1e6, 6), tmaxplot=f(90, 0),
             tmlimit=f((-vd + math.sqrt(vd ** 2 + 2 * ad * (top - e0))) / ad * 1e3, 1))

    # Q2: ASSUMED slot tolerance 0.60 mm; same initial error, drift and disturbance bounds
    rtol2 = 0.60
    slack2 = rtol2 - e0
    t2 = (-vd + math.sqrt(vd ** 2 + 2 * ad * slack2)) / ad * 1e3  # ms
    t2cv = slack2 / vd * 1e3
    n.update(qrtol=f(rtol2, 2), qslack=f(slack2, 1), qdisc=f(vd ** 2 + 2 * ad * slack2, 0),
             qroot=f(math.sqrt(vd ** 2 + 2 * ad * slack2), 2), qtlatch=f(t2, 1), qtcv=f(t2cv, 1),
             qover=f(round(t2cv, 1) - round(t2, 1), 1))

    # ---- Mistake: a stale camera-to-slot edge on the current wrist (chapter: LatchTransformStaleness)
    stale_s = Scenarios.LatchSlotStaleWristTransform
    st = stale_s.result
    n.update(wframe=f(st.interval.to("ms").magnitude, 1), drift=f(st.drift_term.to("mm").magnitude, 2),
             lever=f(st.lever_term.to("mm").magnitude, 2), etool=f(st.total.to("mm").magnitude, 2),
             rarm=f(stale_s.lever.to("mm").magnitude, 0), omega=f(stale_s.angular_rate.to("rad/s").magnitude, 2),
             esame=f(st.same_epoch_error.to("mm").magnitude, 2))
    assert abs(st.total.to("mm").magnitude - (e0 + st.drift_term.to("mm").magnitude + st.lever_term.to("mm").magnitude)) < 1e-9

    # ---- Mistake: frozen transform restamped by the middleware (chapter: MemoryFrozenTransform)
    fz_s = Scenarios.ConveyorMugFrozenTransform
    fz = fz_s.result
    mug = Scenarios.ConveyorMugHandleBelief.result
    n.update(repub=f(fz.republish_rate.to("Hz").magnitude, 0), fresh=f(fz_s.freshness_threshold.to("ms").magnitude, 0),
             window=f(fz_s.window.to("ms").magnitude, 0), vbelt=f(aisle.v_conveyor.to("m/s").magnitude, 1),
             frozenerr=f(fz.error.to("mm").magnitude, 0), grasptol=f(aisle.grasp_tolerance.to("mm").magnitude, 0),
             mughorizon=f(mug.horizon.to("ms").magnitude, 0))

    # ---- Q1: ASSUMED a dusty lidar window doubles the range spread; same predicted prior
    sig_l1 = 2 * sig_lidar
    g_q1 = sig_pred ** 2 / (sig_pred ** 2 + sig_l1 ** 2)
    sig_q1 = math.sqrt(1 - round(g_q1, 3)) * sig_pred
    n.update(qsiglidar=f(sig_l1, 0), qgain=f(g_q1, 3), qsigpost=f(sig_q1, 1))
    # Q1 evidence: ASSUMED mean square of 200 innovations; NIS from the printed rounded spreads
    nis_count, nu2_mean = 200, 12000.0
    s_innov = round(sig_pred, 1) ** 2 + round(sig_l1, 0) ** 2
    nis = nu2_mean / round(s_innov)
    n.update(qnisn=f(nis_count, 0), qnutwo=f(nu2_mean, 0), qinnovs=f(s_innov, 0), qnis=f(nis, 1))
    assert nis > 2

    # ---- Q3: clearance query on the permission controller (chapter: RepresentationQuery)
    cq_s = Scenarios.WarehousePermissionClearanceQuery
    cq = cq_s.result
    mcu = rm.permission_mcu.memory
    tq = cq.query_time.to("us").magnitude
    bw = mcu.sram_bandwidth.to("GB/s").magnitude
    cell = cq_s.esdf_cell_bytes.to("B").magnitude
    reads = tq * 1e-6 * bw * 1e9 / cell
    assert abs(reads - cq.reads_budget) < 1e-6
    rh = cq_s.hull_radius.to("m").magnitude
    p2, p5 = (q.to("m").magnitude for q in cq_s.pitches[:2])
    reach = rh + aisle.delta_margin.to("m").magnitude
    occ5 = 4 / 3 * math.pi * (reach / p5) ** 3
    assert abs(occ5 - cq.rows[5].occupancy_reads) < 1e-6
    vol = cq.slab_volume.to("m**3").magnitude
    esdf5 = vol / p5 ** 3 * cell / 1e3  # kB
    esdf2 = vol / p2 ** 3 * cell / 1e6  # MB
    assert abs(esdf5 - cq.rows[5].esdf_bytes.to("kB").magnitude) < 1e-6
    sram_mib = mcu.sram_capacity.to("MiB").magnitude
    sram_mb = mcu.sram_capacity.to("MB").magnitude
    n.update(tquery=f(tq, 0), sramgbs=f(bw, 1), cellbytes=f(cell, 0), reads=f(reads, 0), rhull=f(rh, 2),
             reach=f(reach, 2), occfive=f(occ5, 0), slab=f(vol, 2), esdffive=f(esdf5, 0), esdftwo=f(esdf2, 2),
             srammib=f(sram_mib, 0), srammb=f(sram_mb, 2), trilinear=f(cq_s.trilinear_reads, 0),
             pitchfive=f(p5, 2), pitchfivecm=f(p5 * 100, 0), pitchtwocm=f(p2 * 100, 0))

    # ---- Q4: wrist strain gauge with drifting bias (chapter: MemoryBiasDrift)
    b_s = Scenarios.WristForceBiasDrift
    b = b_s.result
    n.update(freading=f(b_s.reading.to("N").magnitude, 0), fsigma=f(b_s.reported_sigma.to("N").magnitude, 2),
             frate=f(b_s.drift_rate.to("N/min").magnitude, 1), fspan=f(b_s.span.to("min").magnitude, 0),
             fshift=f(b.shift.to("N").magnitude, 0), ftrue=f(b.true_value.to("N").magnitude, 0),
             flost=f(b.lost_share * 100, 0), fexcite=f(b_s.excitation.to("ms").magnitude, 0))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
