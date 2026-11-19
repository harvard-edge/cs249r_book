"""Numbers for The Physical Body study sheet (Volume IV, Chapter 2).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to body_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Embodied, Scenarios
from mlsysim.physics import (calc_max_permitted_velocity, calc_safe_stopping_distance,
                             calc_traction_deceleration_limit)


def f(x, nd):
    return f"{x:,.{nd}f}"


def numbers() -> dict:
    n = {}
    aisle = Embodied.Scenario.WarehouseAisle
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    gravity = 9.80665  # m/s^2, standard gravity (as mlsysim.physics.robotics uses)

    # ---- Front: freshness deadline on the pick-station conveyor (chapter: GraspLatencyProfile)
    grasp = Scenarios.WristCameraGraspAgeProfile
    g = grasp.result
    n.update(tol=f(aisle.grasp_tolerance.to("mm").magnitude, 0), err=f(aisle.grasp_initial_error.to("mm").magnitude, 0),
             emax=f(g.allowed_error.to("mm").magnitude, 0), vconv=f(aisle.v_conveyor.to("m/s").magnitude, 2),
             fresh=f(g.deadline.to("ms").magnitude, 0), backbone=f(g.focus_stage.to("ms").magnitude, 1),
             agemed=f(g.stage_sum_p50.to("ms").magnitude, 0), agepnn=f(grasp.system_p99.to("ms").magnitude, 1),
             agetail=f(grasp.system_p999.to("ms").magnitude, 1), traveltail=f(g.travel_tail.to("mm").magnitude, 1))

    # ---- Front: deceleration ceilings of the loaded base (chapter: BrakingLimits)
    lim = Scenarios.WarehouseBaseBrakingCeilings.result
    a_req = aisle.a_brake.to("m/s^2").magnitude
    a_dry = lim.a_traction_dry.to("m/s^2").magnitude
    a_drive = lim.a_torque.to("m/s^2").magnitude
    a_insp = lim.a_traction_inspected.to("m/s^2").magnitude
    n.update(areq=f(a_req, 1), adry=f(a_dry, 2), adrive=f(a_drive, 2), ainsp=f(a_insp, 2),
             mudry=f(aisle.mu_dry, 2), muinsp=f(aisle.mu_inspected_floor, 2))

    # ---- Worked example: the chapter's full-delay stop (Scenarios.WarehouseBaseStopFullDelay)
    full = Scenarios.WarehouseBaseStopFullDelay
    v = full.speed.to("m/s").magnitude
    tau = full.delay.to("s").magnitude
    loc = aisle.delta_loc.to("m").magnitude
    mar = aisle.delta_margin.to("m").magnitude
    dclear = aisle.d_clear.to("m").magnitude
    blind = v * tau
    brake = v ** 2 / (2 * a_req)
    total = calc_safe_stopping_distance(full.speed, full.delay, aisle.a_brake, aisle.delta_loc,
                                        aisle.delta_margin).to("m").magnitude
    assert abs(total - (blind + brake + loc + mar)) < 1e-9
    vmax = calc_max_permitted_velocity(aisle.d_clear, full.delay, aisle.a_brake, aisle.delta_loc,
                                       aisle.delta_margin).to("m/s").magnitude
    n.update(wv=f(v, 1), wtau=f(tau * 1e3, 1), wloc=f(loc * 1e3, 0), wmar=f(mar * 1e3, 0),
             wover=f((loc + mar) * 1e3, 0), wclear=f(dclear, 2), wclearmm=f(dclear * 1e3, 0),
             wblind=f(blind * 1e3, 1), wbrake=f(brake * 1e3, 1), wtotal=f(total * 1e3, 1),
             wspare=f((dclear - total) * 1e3, 1), wvmax=f(vmax, 2), wavail=f(dclear - loc - mar, 2))
    # picture: to-scale bar, 1 m = 200 pt
    n.update(pblind=f(blind * 200, 2), pbrake=f(brake * 200, 2), pover=f((loc + mar) * 200, 2),
             pclear=f(dclear * 200, 2))

    # ---- Your turn: same stop on the inspected floor (chapter's mu; all weight on the driven wheels,
    # an upper bound). The deceleration is the rounded value printed, as a student would use it.
    a_t = round(calc_traction_deceleration_limit(aisle.mu_inspected_floor).to("m/s^2").magnitude, 2)
    assert abs(a_t - round(a_insp, 2)) < 1e-12
    brake_t = v ** 2 / (2 * a_t)
    total_t = round(blind, 4) + round(brake_t, 4) + loc + mar
    vmax_t = -a_t * tau + math.sqrt((a_t * tau) ** 2 + 2 * a_t * (dclear - loc - mar))
    n.update(ybrake=f(brake_t * 1e3, 1), ytotal=f(total_t * 1e3, 1), yover=f((total_t - dclear) * 1e3, 1),
             yvmax=f(vmax_t, 2))

    # ---- Q1: ASSUMED conveyor speeds up to 0.30 m/s; same tolerance, same illustrative age profile
    v1 = 0.30
    d1 = g.allowed_error.to("m").magnitude / v1
    n.update(qvconv=f(v1, 2), qfresh=f(d1 * 1e3, 0), qmedspare=f(d1 * 1e3 - g.stage_sum_p50.to("ms").magnitude, 0),
             qpnnover=f(grasp.system_p99.to("ms").magnitude - d1 * 1e3, 1))

    # ---- Q2: burst horizon from a heat-soaked estimate (chapter: PandaClassJointBurst inputs)
    burst = Scenarios.PandaClassJointBurst
    b = burst.result
    tamb = burst.ambient.to("degC").magnitude
    tlim = burst.limit.to("degC").magnitude
    rth = burst.thermal_resistance.to("K/W").magnitude
    tauth = b.time_constant.to("s").magnitude
    ppk = b.burst_loss.to("W").magnitude
    tinf = tamb + round(ppk) * rth
    t0 = 120.0  # ASSUMED observer estimate after a long hold, degC
    case = 70.0  # ASSUMED case thermistor reading, degC
    h = tauth * math.log((tinf - t0) / (tinf - tlim))
    hcold = tauth * math.log((tinf - tamb) / (tinf - tlim))
    n.update(tamb=f(tamb, 0), tlim=f(tlim, 0), rth=f(rth, 1), tauth=f(tauth, 0), ppk=f(ppk, 0),
             tinf=f(tinf, 0), qtzero=f(t0, 0), qcase=f(case, 0), qhot=f(h, 1), qcold=f(hcold, 1),
             qnum=f(tinf - t0, 0), qden=f(tinf - tlim, 0))

    # ---- Q3: ASSUMED aged, cold pack doubles the source resistance; chapter's coincident step and stop
    bus = Scenarios.WarehouseDriveBusStop
    u = bus.result
    r0 = rm.drive_bus_resistance.to("ohm").magnitude
    r3 = 2 * r0
    di = aisle.i_rail_transient.to("A").magnitude
    vlow = rm.drive_bus_battery_low.to("V").magnitude
    vuvlo = rm.drive_bus_uvlo.to("V").magnitude
    sag = di * r3
    vfull = rm.drive_bus_full_charge.to("V").magnitude
    eregen = u.regenerated.to("J").magnitude
    c0 = bus.link_capacitance.to("F").magnitude
    c3 = 2 * c0
    vfin = math.sqrt(vfull ** 2 + 2 * eregen / c3)
    n.update(rbus=f(r0 * 1e3, 0), qrbus=f(r3 * 1e3, 0), di=f(di, 0), vlow=f(vlow, 0), vuvlo=f(vuvlo, 0),
             qsag=f(sag, 1), qvbus=f(vlow - sag, 1), qmargin=f(vlow - sag - vuvlo, 1),
             vfull=f(vfull, 0), eregen=f(eregen, 1), cbus=f(c0 * 1e6, 0), qcbus=f(c3 * 1e6, 0),
             qvfin=f(vfin, 1), vswitch=f(rm.drive_bus_switch_rating.to("V").magnitude, 0),
             vfinchap=f(u.capacitor_only_endpoint.to("V").magnitude, 1))

    # ---- Q4: current limit lowered to the rated current; K_t x I route (the chapter's 0.78 uses rated torque)
    hub = rm.drive_unit
    kt = hub.torque_constant.to("N*m/A").magnitude
    irated = hub.rated_current.to("A").magnitude
    ipeak = rm.drive_current_limit.to("A").magnitude
    rw = hub.wheel_radius.to("m").magnitude
    meff = round(lim.m_eff.to("kg").magnitude)
    a4 = 2 * kt * irated / (rw * meff)
    n.update(kt=f(kt, 1), irated=f(irated, 1), ipeak=f(ipeak, 0), rwheel=f(rw * 1e3, 0), meff=f(meff, 0),
             qforce=f(2 * kt * irated / rw, 0), qadrive=f(a4, 2))

    # ---- Mistakes
    gears = Scenarios.PandaClassJointGearRatios
    gr = gears.result
    n1, n2, _ = gears.ratios
    n.update(gone=f(n1, 0), gtwo=f(n2, 0), lossone=f(gr.copper_loss[n1].to("W").magnitude, 1),
             losstwo=f(gr.copper_loss[n2].to("W").magnitude, 1),
             jone=f(gr.reflected_inertia[n1].to("kg*m^2").magnitude, 3),
             jtwo=f(gr.reflected_inertia[n2].to("kg*m^2").magnitude, 3))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
