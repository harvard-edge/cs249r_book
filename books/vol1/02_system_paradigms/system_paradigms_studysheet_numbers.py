"""Numbers for the System Paradigms study sheet (Volume I, Chapter 2).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to system_paradigms_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED. Results that the sheet
works through by hand are computed from the rounded inputs it prints, so they reproduce.
"""
import math
from pathlib import Path

from mlsysim import Platforms
from mlsysim.engine.scenarios import Scenarios as SC
from mlsysim.physics import SPEED_OF_LIGHT_FIBER_KM_S


BITS = 8                                                                  # bits per byte


def g(x, nd=3):
    return f"{x:.{nd}g}"


def c(x):
    return f"{x:,.0f}"


def numbers() -> dict:
    n = {}

    # ---- Light barrier (chapter: light-barrier-calc, distance-penalty)
    v_fiber = SPEED_OF_LIGHT_FIBER_KM_S.to("km/s").magnitude
    per100 = 2 * 100 / v_fiber * 1e3                                       # ms of round trip per 100 km
    n.update(cfiber=c(v_fiber), per100=g(per100))
    arm = SC.RoboticArmRegionalCloud
    arm_km, arm_budget = arm.distance.to("km").magnitude, arm.budget.to("ms").magnitude
    arm_rtt = 2 * arm_km / v_fiber * 1e3
    n.update(armkm=c(arm_km), armbudget=g(arm_budget), armrtt=g(arm_rtt), armover=g(arm_rtt - arm_budget))
    assert abs(arm_rtt - arm.round_trip.to("ms").magnitude) < 1e-9

    # ---- Mobile thermal envelope and the thermal wall (chapter: DeploymentOverviewTable, ThermalQuantCalc)
    tdp_lo, tdp_hi = (int(v) for v in Platforms.Mobile.tdp_range_w.split("-"))
    n.update(tdp=f"{tdp_lo}--{tdp_hi}")
    th = SC.UnoptimizedVisionModelPhone
    p0, k, ceil = th.power.to("W").magnitude, round(th.reduction_factor), th.sustained_limit.to("W").magnitude
    n.update(thermw=g(p0), thk=g(k), thnew=g(p0 / k), thceil=g(ceil), thtrip=g(th.trip_time.to("s").magnitude),
             thfps0=g(th.burst_rate), thfps1=g(th.throttled_rate))

    # ---- Worked example: drone offload (chapter scenario Scenarios.Drone4KObstacleOffload)
    dr = SC.Drone4KObstacleOffload
    mb = round(dr.payload.to("MB").magnitude, 1)                         # printed payload, 24.9 MB
    mbit = dr.uplink_bandwidth.to("Mbit/s").magnitude
    resp = dr.remote_response.to("ms").magnitude                          # network latency + remote compute
    dl = round(dr.deadline.to("ms").magnitude, 1)                          # printed deadline, 16.7 ms
    tx_s = mb * BITS / mbit                                                   # from the printed payload
    tot_s = tx_s + resp / 1e3
    n.update(drmb=g(mb), drmbit=g(mbit), drnet=g(dr.network_latency.to("ms").magnitude),
             drcmp=g(dr.remote_compute_time.to("ms").magnitude), drresp=g(resp), drdl=g(dl),
             drfps=g(dr.frame_rate.to("Hz").magnitude), drtx=f"{tx_s:.2f}", drtot=f"{tot_s:.2f}",
             drx=f"{round(tot_s, 2) * 1e3 / dl:.0f}", drrespx=f"{resp / dl:.1f}")

    # ---- Your turn, ASSUMED: same uplink and remote response, a 2 MB photo, a 500 ms deadline
    ymb, ydl = 2.0, 500.0
    ytx = ymb * BITS / mbit * 1e3
    ytot = ytx + resp
    ycross = (ydl - resp) * 1e-3 * mbit / BITS                                # MB that use up the deadline
    n.update(ymb=g(ymb), ydl=g(ydl), ytx=g(ytx), ytot=g(ytot), yslack=g(ydl - ytot),
             ycross=f"{math.floor(ycross * 100) / 100:.2f}",                 # rounded down: the largest that passes
             yleft=g(ydl - resp), ydrfit=f"{ycross / mb * 100:.0f}")

    # ---- Energy of transmission (chapter scenario Scenarios.RadioUploadVsMobileNetV2)
    radio = SC.RadioUploadVsMobileNetV2.transmit_energy_per_byte.to("mJ/MB").magnitude
    n["radio"] = g(radio)

    # ---- Mistake 3: edge TCO pitfall (chapter: TcoPitfallCalc scenario inputs, USD per month)
    cloud, hw, net, maint, rel = 2000, 500, 3000, 500, 2000
    edge = hw + net + maint + rel
    n.update(tcocloud=c(cloud), tcohw=c(hw), tconet=c(net), tcomaint=c(maint), tcorel=c(rel), tcoedge=c(edge),
             tcox=g(edge / cloud), tcolabor=f"{(net + maint + rel) / edge * 100:.0f}")

    # ---- Q1, ASSUMED: one inference moves 50 MB and does 4 GFLOP; 100 GB/s, 20 TFLOP/s achieved, 0.1 ms overhead
    qmb, qbw, qgf, qtf, qL, qk = 50, 100, 4, 20, 0.1, 2                 # qk: ASSUMED faster accelerator
    td, tc = qmb / qbw, qgf / qtf                                          # ms (MB / (GB/s) = ms; GFLOP / (TFLOP/s) = ms)
    n.update(bits=str(BITS), qk=g(qk))                                        # bits per byte; speedup
    n.update(qmb=g(qmb), qbw=g(qbw), qgf=g(qgf), qtf=g(qtf), qL=g(qL), qtd=g(td), qtc=g(tc),
             qser=g(td + tc + qL), qpipe=g(max(td, tc) + qL), qtc2=g(tc / qk), qser2=g(td + tc / qk + qL),
             qpipe2=g(max(td, tc / qk) + qL))

    # ---- Q3, ASSUMED: factory inspection screening
    n.update(fdl=g(20.0), fkm=c(1200), frtt=g(2 * 1200 / v_fiber * 1e3), fgb=g(4.0), fw=g(30.0))

    # ---- Q4, ASSUMED: wearable sends a 0.05 MB raw clip per inference, or runs a 2 mJ local inference
    wmb, wloc = 0.05, 2.0
    wtx = wmb * radio
    n.update(wmb=g(wmb), wloc=g(wloc), wtx=g(wtx), wx=g(wtx / wloc), wbe=g(wloc / radio * 1e3))   # wbe in KB
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
