"""Numbers for the Safety Enforcement study sheet (Volume IV, Chapter 12).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to enforcement_studysheet_numbers.tex next to this file.
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

    # ---- Stopping budget at the aisle speed (chapter: EnforcementStoppingBudget, AISLE_STOP)
    stop = Scenarios.WarehouseFinishedStopAtAisleSpeed
    r = stop.result
    v = stop.speed.to("m/s").magnitude
    tau = stop.delay.to("ms").magnitude
    a = r.deceleration.to("m/s^2").magnitude
    j = r.jerk.to("m/s^3").magnitude
    dclear = r.clear_distance.to("mm").magnitude
    loc = aisle.delta_loc.to("mm").magnitude
    mar = aisle.delta_margin.to("mm").magnitude
    eps = r.tracking_inset.to("mm").magnitude
    ramp = r.ramp_delay.to("ms").magnitude
    dda = aisle.t_dispatch_to_admission.to("ms").magnitude
    # Rounded values the sheet prints, then the results from them
    v_p, tau_p, ramp_p, dda_p = round(v, 1), round(tau, 1), round(ramp, 1), round(dda, 1)
    blind = round(v_p * tau_p, 1)                      # mm (m/s x ms = mm)
    brake = round(v_p ** 2 / (2 * a) * 1e3, 1)          # mm
    rampt = round(v_p * ramp_p, 1)                      # mm
    stopt = round(brake + rampt, 1)
    allow = loc + mar + eps
    total = round(blind + stopt + allow, 1)
    spare = round(dclear - total, 1)
    datravel = round(v_p * dda_p, 1)
    reserve = round(spare - datravel, 1)
    # check against mlsysim
    assert abs(total - r.required_distance.to("mm").magnitude) < 0.2
    assert abs(spare - r.spare.to("mm").magnitude) < 0.2
    assert abs(reserve - Scenarios.WarehouseFinishedStopWithDispatch.result.spare.to("mm").magnitude) < 0.2
    # ceiling from the printed values
    teff = round(tau_p + ramp_p, 1)
    davail = dclear - allow
    atau = round(a * teff / 1e3, 3)
    root = round(math.sqrt(atau ** 2 + 2 * a * davail / 1e3), 3)
    vmax = root - atau
    assert abs(vmax - r.speed_ceiling.to("m/s").magnitude) < 0.005
    n.update(wv=f(v_p, 1), wtau=f(tau_p, 1), wa=f(a, 1), wj=f(j, 2), wramp=f(ramp_p, 1),
             wloc=f(loc, 0), wmar=f(mar, 0), weps=f(eps, 0), wallow=f(allow, 0),
             wclear=f(dclear / 1e3, 1), wclearmm=f(dclear, 0), wdda=f(dda_p, 1),
             wblind=f(blind, 1), wbrake=f(brake, 1), wrampt=f(rampt, 1), wstop=f(stopt, 1),
             wtotal=f(total, 1), wspare=f(spare, 1), wdatravel=f(datravel, 1), wreserve=f(reserve, 1),
             wteff=f(teff, 1), wavail=f(davail, 0), wavailm=f(davail / 1e3, 3), wlocmar=f(loc + mar, 0), watau=f(atau, 3), wroot=f(root, 3), wvmax=f(vmax, 2))

    # ---- Picture: the same terms at the drive limit vs the aisle speed, 1 m = 140 pt
    drive = Scenarios.WarehouseFinishedStopAtDriveLimit
    d = drive.result
    vd = drive.speed.to("m/s").magnitude
    dd_blind = d.delay_travel.to("mm").magnitude
    dd_stop = (d.braking_travel + d.jerk_ramp_travel).to("mm").magnitude
    dd_total = d.required_distance.to("mm").magnitude
    sc = 0.14
    n.update(vdrive=f(vd, 1), dtotal=f(dd_total, 0), dover=f(dd_total - dclear, 1), atotal=f(total, 0),
             pdblind=f(dd_blind * sc, 2), pdstop=f(dd_stop * sc, 2), pallow=f(allow * sc, 2),
             pablind=f(r.delay_travel.to("mm").magnitude * sc, 2),
             pastop=f(r.scurve_travel.to("mm").magnitude * sc, 2), pclear=f(dclear * sc, 2))

    # ---- Your turn: ASSUMED wear doubles the tracking bound (the chapter's doubled-error case)
    worn = stop.model_copy(update={"tracking_bound": 2 * stop.tracking_bound}).result
    eps2 = 2 * eps
    total2 = round(blind + stopt + loc + mar + eps2, 1)
    spare2 = round(dclear - total2, 1)
    reserve2 = round(spare2 - datravel, 1)
    davail2 = dclear - loc - mar - eps2
    root2 = round(math.sqrt(atau ** 2 + 2 * a * davail2 / 1e3), 3)
    vmax2 = root2 - atau
    assert abs(total2 - worn.required_distance.to("mm").magnitude) < 0.2
    assert abs(vmax2 - worn.speed_ceiling.to("m/s").magnitude) < 0.005
    n.update(yeps=f(eps2, 0), ytotal=f(total2, 1), yspare=f(spare2, 1), yreserve=f(reserve2, 1),
             yavail=f(davail2, 0), yroot=f(root2, 3), yvmax=f(vmax2, 2))

    # ---- Q1: gravity tax with an ASSUMED lighter load (chapter: LoadedArmJointGravityStop)
    joint = Scenarios.LoadedArmJointGravityStop
    tmax = joint.torque_limit.to("N*m").magnitude
    jin = joint.inertia.to("kg*m^2").magnitude
    lev = joint.lever.to("m").magnitude
    om = joint.joint_speed.magnitude
    frac = 0.40  # ASSUMED: gravity on an emptier gripper takes 40% of peak torque
    bias1 = tmax * frac
    alpha1 = (tmax - bias1) / jin
    tool1 = lev * om ** 2 / (2 * alpha1) * 1e3
    tool0 = joint.tool_stop_free.to("mm").magnitude
    n.update(tmax=f(tmax, 0), jin=f(jin, 0), lever=f(lev, 1), omega=f(om, 1), qfrac=f(frac * 100, 0),
             qbias=f(bias1, 1), qnet=f(tmax - bias1, 1), qalpha=f(alpha1, 1), qtool=f(tool1, 1),
             toolfree=f(tool0, 1), qratio=f(1 / (1 - frac), 2), qremain=f((1 - frac) * 100, 0), qremainfrac=f(1 - frac, 2), one="1",
             biasch=f(joint.bias_fraction * 100, 0), toolch=f(joint.tool_stop_descending.to("mm").magnitude, 0))

    # ---- Q2: projection onto the chapter's two-joint half-space with ASSUMED proposals
    proj = Scenarios.TwoJointTorqueProjection
    na, nb = proj.normal
    bnd = proj.bound
    lim = proj.actuator_limit
    u1 = (3.0, 1.0)   # ASSUMED proposal that breaches
    u2 = (1.0, 2.0)   # ASSUMED proposal that is already safe
    s1 = na * u1[0] + nb * u1[1]
    lam = max(0.0, (s1 - bnd) / (na ** 2 + nb ** 2))
    up = (u1[0] - lam * na, u1[1] - lam * nb)
    s2 = na * u2[0] + nb * u2[1]
    assert abs(na * up[0] + nb * up[1] - bnd) < 1e-12 and max(abs(up[0]), abs(up[1])) <= lim and s2 <= bnd
    n.update(na=f(na, 0), nb=f(nb, 0), bnd=f(bnd, 0), ulim=f(lim, 0), qua=f(u1[0], 0), qub=f(u1[1], 0),
             qsone=f(s1, 0), qnormsq=f(na ** 2 + nb ** 2, 0), qlam=f(lam, 1), qpa=f(up[0], 1), qpb=f(up[1], 1),
             qva=f(u2[0], 0), qvb=f(u2[1], 0), qstwo=f(s2, 0))

    # ---- Q3: rung timers (chapter: EnforcementLeaseStop) and the resident stop (EnforcementStopRung)
    n.update(lease=f(aisle.t_lease.to("ms").magnitude, 0), hbtimeout=f(aisle.heartbeat_timeout.to("ms").magnitude, 0),
             watchdog=f(aisle.mcu_self_watchdog.to("ms").magnitude, 0),
             stoptime=f(r.stop_time.to("ms").magnitude, 0), stopdist=f(r.scurve_travel.to("mm").magnitude, 0))

    # ---- Mistakes: kinetic barrier demand along the descent (chapter: KineticBarrierArm) and handover
    far, near = joint.barrier_clearances
    n.update(bfar=f(far.to("mm").magnitude, 0), bnear=f(near.to("mm").magnitude, 0),
             tfar=f(joint.barrier_torque_demand(far).to("N*m").magnitude, 1),
             tnear=f(joint.barrier_torque_demand(near).to("N*m").magnitude, 1))
    from mlsysim.physics.robotics import calc_transient_impact_force
    meff = rm.arm_effective_contact_mass
    khand = aisle.k_contact_human
    vh = aisle.v_handover
    # free hand: two-body reduced mass; printed rounded inputs reproduce both forces
    mh, me = aisle.m_contact_human.to("kg").magnitude, meff.to("kg").magnitude
    mred = 1 / (1 / mh + 1 / me)
    kh = khand.to("kN/m").magnitude
    ffree_p = vh.to("m/s").magnitude * math.sqrt(kh * 1e3 * round(mred, 3))
    fclamp_p = vh.to("m/s").magnitude * math.sqrt(kh * 1e3 * me)
    assert abs(ffree_p - calc_transient_impact_force(vh, meff, khand, human_mass=aisle.m_contact_human).to("N").magnitude) < 0.05
    assert abs(fclamp_p - calc_transient_impact_force(vh, meff, khand).to("N").magnitude) < 1e-9
    n.update(khand=f(kh, 0), mred=f(mred, 3))
    n.update(vhand=f(vh.to("m/s").magnitude, 2), crit=f(aisle.f_contact_criterion.to("N").magnitude, 0),
             ffree=f(calc_transient_impact_force(vh, meff, khand, human_mass=aisle.m_contact_human).to("N").magnitude, 1),
             fclamp=f(calc_transient_impact_force(vh, meff, khand).to("N").magnitude, 1),
             meff=f(meff.to("kg").magnitude, 0), mhand=f(aisle.m_contact_human.to("kg").magnitude, 1))

    # ---- Mistake: inter-sample (zero-order hold) drift at the drive limit (chapter: EnforcementZOHBuffer)
    hold = Scenarios.WarehouseBaseHoldDrift
    zv, za, zt = hold.speed.to("m/s").magnitude, hold.acceleration.to("m/s^2").magnitude, hold.period.to("ms").magnitude
    zdrift = zv * zt + za * zt ** 2 / 2 * 1e-3          # mm (m/s x ms = mm; m/s^2 x ms^2 = 1e-3 mm)
    assert abs(zdrift - hold.drift.to("mm").magnitude) < 1e-9
    n.update(zv=f(zv, 1), za=f(za, 1), ztick=f(zt, 0), zdrift=f(zdrift, 3))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
