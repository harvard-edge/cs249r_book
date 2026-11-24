"""Numbers for the Adversarial Verification study sheet (Volume IV, Chapter 15).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(RackEndVerification, RareEventOvershoot), and written as \\setnum{key}{value} to
verification_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
from pathlib import Path

from mlsysim import Embodied, Scenarios


def f(x, nd):
    return f"{x:,.{nd}f}"


def numbers() -> dict:
    n = {}
    aisle = Embodied.Scenario.WarehouseAisle

    # ---- Front + worked example: the F1 oracle from the last valid renewal at the aisle speed
    # (chapter: Scenarios.WarehouseRenewalReferencedStop, notebook "The reference-event gap")
    lease = aisle.t_lease.to("ms").magnitude
    tick = aisle.t_tick.to("ms").magnitude
    bus = aisle.t_bus.to("ms").magnitude
    act = aisle.t_brake_onset.to("ms").magnitude
    onset = lease + tick + bus + act
    renewal = Scenarios.WarehouseRenewalReferencedStop
    r = renewal.result
    assert abs(onset - renewal.delay.to("ms").magnitude) < 1e-9
    v = aisle.v_aisle.to("m/s").magnitude
    a = round(aisle.a_brake.to("m/s^2").magnitude, 2)
    j = round(aisle.j_brake.to("m/s^3").magnitude, 2)  # 64/9 m/s^3, printed as 7.11
    inset = aisle.eps_track.to("mm").magnitude
    dclear = aisle.d_clear.to("mm").magnitude
    pre = round(v * onset, 1)  # mm (m/s x ms = mm)
    brake = round(v ** 2 / (2 * a) * 1e3, 1)
    ramp = round(v * a / (2 * j) * 1e3, 1)
    ontime = pre + brake + ramp
    bound = ontime + inset
    assert abs(bound - r.required_distance.to("mm").magnitude) < 0.1
    assert abs(ontime - r.stop_distance.to("mm").magnitude) < 0.1
    capture = Scenarios.WarehouseFinishedStopAtAisleSpeed.result.required_distance.to("mm").magnitude
    gap = round(capture, 1) - round(bound, 1)
    n.update(lease=f(lease, 0), tick=f(tick, 0), bus=f(bus, 0), act=f(act, 0), onset=f(onset, 0),
             v=f(v, 1), a=f(a, 1), j=f(j, 2), inset=f(inset, 0), clear=f(dclear, 0),
             pre=f(pre, 1), brake=f(brake, 1), ramp=f(ramp, 1), ontime=f(ontime, 1), bound=f(bound, 1),
             capture=f(capture, 1), gap=f(gap, 1), gapms=f(gap / v, 0))
    # picture: to-scale bars, 1 mm = 0.2 pt (1,100 mm = 220 pt)
    s = 0.2
    n.update(ppre=f(pre * s, 2), pstop=f((brake + ramp) * s, 2), pinset=f(inset * s, 2),
             pcap=f(capture * s, 2), pclear=f(dclear * s, 2), pbound=f(bound * s, 2))

    # ---- Your turn: same oracle at the restricted point on the inspected floor
    # (chapter: Scenarios.WarehouseRenewalReferencedStopRestricted; deceleration is the inspected
    # floor's optimistic all-weight cap, printed at three decimals so the sum reproduces)
    restricted = Scenarios.WarehouseRenewalReferencedStopRestricted
    vr = restricted.speed.to("m/s").magnitude
    ar = round(restricted.deceleration.to("m/s^2").magnitude, 3)
    pre_r = round(vr * onset, 1)
    brake_r = round(vr ** 2 / (2 * ar) * 1e3, 1)
    ramp_r = round(vr * ar / (2 * j) * 1e3, 1)
    bound_r = pre_r + brake_r + ramp_r + inset
    assert abs(bound_r - restricted.result.required_distance.to("mm").magnitude) < 0.1
    n.update(vr=f(vr, 1), ar=f(ar, 3), mu=f(aisle.mu_inspected_floor, 2), prer=f(pre_r, 1),
             braker=f(brake_r, 1), rampr=f(ramp_r, 1), boundr=f(bound_r, 1),
             cutr=f((1 - bound_r / bound) * 100, 0), cutv=f((1 - vr / v) * 100, 0))

    # ---- Mistake: the chapter's late-onset trace (Scenarios.WarehouseRenewalStopLateOnsetTrial)
    late = Scenarios.WarehouseRenewalStopLateOnsetTrial.result
    n.update(lateonset=f(late.onset.to("ms").magnitude, 0), latemargin=f(late.onset_margin.to("ms").magnitude, 0),
             latetravel=f(late.travel.to("mm").magnitude, 1), latedist=f(late.distance_margin.to("mm").magnitude, 0),
             lateshort=f(dclear - late.travel.to("mm").magnitude, 0))

    # ---- Mistake: falsification fraction vs operational rate (chapter: RareEventOvershoot,
    # Scenarios.WarehouseQueueDelaySpareExhaustion)
    rare = Scenarios.WarehouseQueueDelaySpareExhaustion.result
    p = rare.failure_probability
    mant, expo = f"{p:.1e}".split("e")
    n.update(failshare=f(rare.proposal_failure_share * 100, 0), pmant=mant, pexp=str(int(expo)))
    # same delay model once the dispatch-to-admission allocation is charged
    # (chapter: RareEventCharged, Scenarios.WarehouseQueueDelayChargedSpareExhaustion)
    pc = Scenarios.WarehouseQueueDelayChargedSpareExhaustion.result.failure_probability
    cmant, cexp = f"{pc:.1e}".split("e")
    n.update(cmant=cmant, cexp=str(int(cexp)))

    # ---- Front card: zero-miss injection count (chapter: InjectedPopulationTrials,
    # Scenarios.WarehouseDiagnosticCoverageHundredFit); n >= ln(1 - C)/ln C_diag, rounded up
    import math
    cov = Scenarios.WarehouseDiagnosticCoverageHundredFit
    conf = cov.confidence
    cdiag = round(cov.result.coverage * 100, 2)  # printed percent
    ncov = math.ceil(math.log(1 - conf) / math.log(cdiag / 100))
    assert ncov == cov.result.injections
    n.update(conf=f(conf * 100, 0), cdiag=f(cdiag, 2), ncov=f(ncov, 0),
             rule3=f(3 / (1 - cdiag / 100), 0))
    # Q4(b): ASSUMED coverage target 99.5% at the same confidence
    qc = 99.5
    qn_exact = math.log(1 - conf) / math.log(qc / 100)
    n.update(qcov=f(qc, 1), qconfm=f(1 - conf, 2), qcovf=f(qc / 100, 3), qnexact=f(qn_exact, 1),
             qn=f(math.ceil(qn_exact), 0))

    # ---- Q1: ASSUMED measured onset of 95 ms at the aisle speed, lease and stop command on time
    q1 = 95.0
    q1m = onset - q1
    q1x = round(v * (q1 - onset), 1)
    q1t = ontime + q1x
    n.update(qonset=f(q1, 0), qlate=f(q1 - onset, 0), qmargin=f(q1m, 0), qnorm=f(q1m / onset, 2), qextra=f(q1x, 1),
             qtravel=f(q1t, 1), qdist=f(bound - q1t, 1), qshort=f(dclear - q1t, 0))

    # ---- Q2: ASSUMED derated speed before injection (verdict question)
    n.update(qslow=f(1.1, 1))

    # ---- Q4: ASSUMED simulated distance margin and discrepancy bounds; position error normalized
    # by a 1 m scale with unit weight, so L_phi = 1 m (= 1,000 mm) per unit of the norm (chapter's choice)
    rho_sim = 60.0  # mm, ASSUMED
    eps = 0.03  # ASSUMED, measured on an unloaded, cool machine
    lphi = 1000.0  # mm per unit discrepancy
    n.update(rhosim=f(rho_sim, 0), eps=f(eps, 2), lphi=f(lphi, 0), rhoreal=f(rho_sim - lphi * eps, 0),
             epsmax=f(rho_sim / lphi, 2), spare=f(dclear - capture, 1), spareeps=f((dclear - capture) / lphi, 3))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
