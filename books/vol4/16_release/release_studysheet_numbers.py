"""Numbers for the Deployment Release study sheet (Volume IV, Chapter 16).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to release_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Embodied, Scenarios
from mlsysim.physics import calc_conjunctive_confidence


def f(x, nd):
    return f"{x:,.{nd}f}"


def numbers() -> dict:
    n = {}
    aisle = Embodied.Scenario.WarehouseAisle

    # ---- Worked example: the aisle spare (chapter: BiomechanicalImpactEnvelope, nbk-release-aisle-spare)
    stop = Scenarios.WarehouseFinishedStopAtAisleSpeed
    s = stop.result
    disp = Scenarios.WarehouseFinishedStopWithDispatch
    v = stop.speed.to("m/s").magnitude
    tau = stop.delay.to("ms").magnitude
    age = aisle.perception_age.to("ms").magnitude
    onset_delay = Scenarios.WarehouseRenewalReferencedStop.delay.to("ms").magnitude
    assert abs(age + onset_delay - tau) < 1e-9
    a = aisle.a_brake.to("m/s^2").magnitude
    j = s.jerk.to("m/s^3").magnitude
    assert v >= a ** 2 / j  # the stop reaches full deceleration, so the two-term S-curve form holds
    loc = aisle.delta_loc.to("mm").magnitude
    trk = aisle.eps_track.to("mm").magnitude
    mar = aisle.delta_margin.to("mm").magnitude
    clear = aisle.d_clear.to("mm").magnitude
    dda = aisle.t_dispatch_to_admission.to("ms").magnitude
    react = round(v * tau, 1)
    quad = round(s.braking_travel.to("mm").magnitude, 1)
    jerkterm = round(s.jerk_ramp_travel.to("mm").magnitude, 1)
    brake = round(quad + jerkterm, 1)
    total = round(react + brake + loc + trk + mar, 1)
    spare = round(clear - total, 1)
    assert abs(total - s.required_distance.to("mm").magnitude) < 0.1
    assert abs(spare - s.spare.to("mm").magnitude) < 0.1
    travel_da = round(v * dda, 1)
    reserve = round(spare - travel_da, 1)
    assert abs(reserve - disp.result.spare.to("mm").magnitude) < 0.1
    n.update(v=f(v, 1), tau=f(tau, 1), age=f(age, 1), onsetdelay=f(onset_delay, 0), abrake=f(a, 1),
             jerk=f(j, 2), loc=f(loc, 0), track=f(trk, 0), margin=f(mar, 0), allow=f(loc + trk + mar, 0),
             clear=f(clear / 1e3, 2), clearmm=f(clear, 0), dda=f(dda, 1),
             react=f(react, 1), quad=f(quad, 1), jerkterm=f(jerkterm, 1), brake=f(brake, 1),
             total=f(total, 1), spare=f(spare, 1), sparems=f(spare / v, 1),
             travelda=f(travel_da, 1), reserve=f(reserve, 1))
    # picture: to-scale bar, 1 mm = 0.19 pt
    sc = 0.19
    n.update(preact=f(react * sc, 2), pbrake=f(brake * sc, 2), pallow=f((loc + trk + mar) * sc, 2),
             pda=f(travel_da * sc, 2), preserve=f(reserve * sc, 2), pclear=f(clear * sc, 2))

    # ---- Your turn: loaded trials put the low tail at 1.6 m/s^2 (the chapter's checkpoint input,
    # ReleaseWorkedVerdict.a_trial; its answer is not printed in the chapter). Student works from
    # the printed jerk bound.
    a_t = 1.6  # chapter value (chk-release-refusal-and-path-to-cond), illustrative
    j_print = round(j, 2)
    assert v >= a_t ** 2 / j
    yquad = round(v ** 2 / (2 * a_t) * 1e3, 1)
    yjerk = round(v * a_t / (2 * j_print) * 1e3, 1)
    ybrake = round(yquad + yjerk, 1)
    ytotal = round(react + ybrake + loc + trk + mar, 1)
    yspare = round(clear - ytotal, 1)
    yreserve = round(yspare - travel_da, 1)
    exact = disp.variant(deceleration=a_t * aisle.a_brake.units).result.spare.to("mm").magnitude
    assert abs(yreserve - exact) < 0.2 and yreserve < 0 < yspare
    n.update(atrial=f(a_t, 1), yquad=f(yquad, 1), yjerk=f(yjerk, 1), ybrake=f(ybrake, 1),
             ytotal=f(ytotal, 1), yspare=f(yspare, 1), yshort=f(-yreserve, 1))

    # ---- Q1: ASSUMED stopping claim with six required premises at 0.97, then a seventh at 0.99
    p, k, p7 = 0.97, 6, 0.99
    c6 = round(calc_conjunctive_confidence([p] * k), 3)
    c7 = round(c6 * p7, 3)
    assert abs(calc_conjunctive_confidence([p] * k + [p7]) - c7) < 1e-3
    n.update(qp=f(p, 2), qk=f(k, 0), qpseven=f(p7, 2), qcsix=f(c6, 3), qcseven=f(c7, 3))

    # ---- Q2: ASSUMED flag latency 75 ms; chapter's silent and flagged routes (ReleaseBlindTravel)
    silent = Scenarios.WarehouseRenewalReferencedStop
    flagged = Scenarios.WarehouseFlagReferencedStop
    lease = aisle.t_lease.to("ms").magnitude
    tick = aisle.t_tick.to("ms").magnitude
    bus = aisle.t_bus.to("ms").magnitude
    onset = aisle.t_brake_onset.to("ms").magnitude
    t_sil = silent.delay.to("ms").magnitude
    t_flg = flagged.delay.to("ms").magnitude
    assert abs(lease + tick + bus + onset - t_sil) < 1e-9 and abs(tick + bus + onset - t_flg) < 1e-9
    dflag = 75.0  # ASSUMED, ms
    t_q2 = dflag + t_flg
    n.update(lease=f(lease, 0), tick=f(tick, 0), bus=f(bus, 0), onset=f(onset, 0),
             tsilent=f(t_sil, 0), tflag=f(t_flg, 0), qdflag=f(dflag, 0), qtflag=f(t_q2, 0),
             qdsilent=f(v * t_sil, 1), qdflagmm=f(v * t_q2, 1), qextra=f(v * (t_q2 - t_sil), 1),
             crossover=f(t_sil - t_flg, 0))

    # ---- Q3: inspected-floor restricted mode (chapter: ReleaseEnvelope)
    restricted = Scenarios.WarehouseFinishedStopRestricted
    n.update(mudry=f(aisle.mu_dry, 2), muinsp=f(aisle.mu_inspected_floor, 2), muoil=f(aisle.mu_oil_film, 2),
             vrestricted=f(restricted.speed.to("m/s").magnitude, 1))

    # ---- Mistake 1: the exposure wall at 95 percent confidence (chapter prose: ln 20, 1e-9 per hour)
    wall = math.log(1 / (1 - 0.95))
    lam_exp = -9
    hours = round(wall, 1) * 10 ** (-lam_exp)
    n.update(wall=f(wall, 1), lamexp=f"{lam_exp}", wallbillion=f(hours / 1e9, 1))

    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
