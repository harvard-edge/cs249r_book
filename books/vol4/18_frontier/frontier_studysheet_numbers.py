"""Numbers for the Engineering Physical AI study sheet (Volume IV, Chapter 18).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(PackagingCell, OilFilm, ArchitecturalShieldDilution, RestrictionCost), and written as
\\setnum{key}{value} to frontier_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Embodied, Scenarios
from mlsysim.physics.drivetrain import calc_required_friction


def f(x, nd):
    return f"{x:,.{nd}f}"


def sci(x, nd):
    """A rate as LaTeX math, e.g. 1.39\\times10^{-5}."""
    exp = math.floor(math.log10(abs(x)))
    man = x / 10 ** exp
    if round(man, nd) >= 10:
        man, exp = man / 10, exp + 1
    head = "" if abs(round(man, nd) - 1) < 1e-12 and nd == 0 else f"{man:.{nd}f}\\times"
    return f"{head}10^{{{exp}}}"


def numbers() -> dict:
    n = {}
    aisle = Embodied.Scenario.WarehouseAisle
    gravity = 9.80665  # m/s^2, standard gravity (the chapter's OilFilm.g)

    # ---- Worked example: the packaging-cell carriage (chapter: PackagingCell notebook)
    cell = Scenarios.PackagingCellTransit
    mode = Scenarios.PackagingCellRestrictedMode
    r, rm = cell.result, mode.result
    v = cell.speed.to("m/s").magnitude
    tau = cell.delay.to("s").magnitude
    a = cell.deceleration.to("m/s^2").magnitude
    clear = cell.clearance.to("mm").magnitude
    reserve = cell.reserve.to("mm").magnitude
    budget = cell.stop_budget.to("mm").magnitude
    assert abs(budget - (clear - reserve)) < 1e-9
    blind = r.delay_travel.to("mm").magnitude
    brake = r.braking_travel.to("mm").magnitude
    stop = r.stop_distance.to("mm").magnitude
    assert abs(stop - (blind + brake)) < 1e-9
    a_tau, two_a_d, root = (q.magnitude for q in cell.root_terms)
    ceiling = r.speed_ceiling.to("m/s").magnitude
    vm = mode.speed.to("m/s").magnitude
    blind_m = rm.delay_travel.to("mm").magnitude
    brake_m = rm.braking_travel.to("mm").magnitude
    stop_m = rm.stop_distance.to("mm").magnitude
    # chapter prints: 171 mm stop, 0.55 m/s ceiling, 40.3 mm restricted stop, 19.7 mm left
    assert round(stop) == 171 and round(ceiling, 2) == 0.55 and round(stop_m, 1) == 40.3
    assert abs(mode.remaining.to("mm").magnitude - (clear - stop_m)) < 1e-9
    n.update(cv=f(v, 1), ctau=f(tau * 1e3, 0), ca=f(a, 0), cclear=f(clear, 0), creserve=f(reserve, 0),
             cbudget=f(budget, 0), cblind=f(blind, 0), cbrake=f(brake, 0), cstop=f(stop, 0),
             catau=f(a_tau, 2), ctwoad=f(two_a_d, 2), croot=f(root, 2),
             cceiling=f(ceiling, 2), cvm=f(vm, 1), cblindm=f(blind_m, 0), cbrakem=f(brake_m, 1),
             cstopm=f(stop_m, 1), cinside=f(budget - stop_m, 1))
    # picture: to-scale bars, 1 mm = 1.1 pt
    s = 1.1
    n.update(pblind=f(blind * s, 2), pbrake=f(brake * s, 2), pblindm=f(blind_m * s, 2), pbrakem=f(brake_m * s, 2),
             pbudget=f(budget * s, 2), pclear=f(clear * s, 2))

    # ---- Your turn: ASSUMED a power cut leaves only the spring-applied track brake, qualified at 8 m/s^2
    # (chapter: the power-cut case has a lower deceleration and a speed ceiling of its own)
    a_spring = 8.0  # ASSUMED
    brake_y = round(vm ** 2 / (2 * a_spring) * 1e3, 1)
    total_y = round(blind_m) + brake_y
    atau_y = round(a_spring * tau, 3)
    twoad_y = round(2 * a_spring * budget / 1e3, 2)
    vmax_y = -atau_y + math.sqrt(atau_y ** 2 + twoad_y)
    assert total_y > budget and vmax_y < vm
    n.update(ya=f(a_spring, 0), ybrake=f(brake_y, 1), ytotal=f(total_y, 1), yover=f(total_y - budget, 1),
             yatau=f(atau_y, 3), ytwoad=f(twoad_y, 2), yvmax=f(vmax_y, 2))

    # ---- Front: friction floor (chapter: OilFilm)
    a_brake = aisle.a_brake.to("m/s^2").magnitude
    mu_min = calc_required_friction(aisle.a_brake)
    a_oil = min(a_brake, aisle.mu_oil_film * gravity)
    assert abs(mu_min - a_brake / gravity) < 1e-9
    n.update(abrake=f(a_brake, 0), grav=f(gravity, 2), mumin=f(mu_min, 3), mudry=f(aisle.mu_dry, 2),
             muinsp=f(aisle.mu_inspected_floor, 2), muoil=f(aisle.mu_oil_film, 2), aoil=f(a_oil, 2),
             vaisle=f(aisle.v_aisle.to("m/s").magnitude, 1))

    # ---- Q1: ASSUMED friction survey reads mu = 0.18 on one aisle
    mu_q = 0.18  # ASSUMED
    a_q = min(a_brake, mu_q * round(gravity, 2))
    assert a_q < a_brake
    n.update(qmu=f(mu_q, 2), qa=f(a_q, 2))

    # ---- Mistake: shield coverage bill (chapter: ArchitecturalShieldDilution)
    bill = Scenarios.ShieldCoverageBill
    b = bill.result
    lam = b.proposal_rate_upper.to("1/hour").magnitude
    resid = b.residual_rate.to("1/hour").magnitude
    target = bill.target_rate.to("1/hour").magnitude
    assert resid > target
    n.update(lamprop=sci(lam, 2), shieldc=f(bill.coverage, 4), lamres=sci(resid, 2), lamtarget=sci(target, 0),
             covtrials=f(b.coverage_trials, 0), loghours=sci(b.exposure.to("hour").magnitude, 2),
             conf=f(bill.confidence * 100, 0))

    # ---- Mistake (restricted mode speed) and Q4: price of the restriction (chapter: RestrictionCost)
    tr = Scenarios.WarehouseRestrictedTransit
    leg = aisle.aisle_length.to("m").magnitude
    v_nom = aisle.v_aisle.to("m/s").magnitude
    v_res = tr.restricted_speed.to("m/s").magnitude
    t_nom = tr.leg_time_nominal.to("s").magnitude
    assert abs(t_nom - leg / v_nom) < 1e-9
    n.update(leg=f(leg, 0), vres=f(v_res, 1), tnom=f(t_nom, 1))
    v_q4 = 0.8  # ASSUMED lower restricted speed after a spill program lapses on part of the route
    t_q4 = leg / v_q4
    n.update(qvres=f(v_q4, 1), qtres=f(t_q4, 1), qtadd=f(t_q4 - t_nom, 1),
             qinc=f((t_q4 / t_nom - 1) * 100, 1), qcut=f((1 - v_q4 / v_nom) * 100, 1),
             qratio=f(v_nom / v_q4, 3))

    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
