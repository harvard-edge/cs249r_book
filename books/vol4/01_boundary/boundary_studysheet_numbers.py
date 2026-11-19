"""Numbers for The Causal Boundary study sheet (Volume IV, Chapter 1).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(MachineCard, StoppingBudget, IncidentRecord, FloorTraction, InferenceStall), and written as
\\setnum{key}{value} to boundary_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Embodied, Scenarios as SC
from mlsysim.core.units import Q_
from mlsysim.embodied.datasheet import data_sheet_lookup, machine_card


def fx(x, nd):
    """Round half up, as a reader would, and format with nd decimals."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(x, 9))).quantize(q, rounding=ROUND_HALF_UP))   # round(.., 9) drops float noise


def r(x, nd):
    return float(fx(x, nd))


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def numbers() -> dict:
    n = {}
    aisle = Embodied.Scenario.WarehouseAisle
    robot = Embodied.MobileManipulator.WarehouseMobileManipulator

    # ---- Front: the pre-brake delay and its five terms (chapter: StoppingBudget, nbk-boundary-stopping-budget)
    ms = lambda q: q.to("ms").magnitude
    age, backbone = ms(aisle.perception_age), ms(aisle.t_backbone)
    lease, tick, bus, onset = ms(aisle.t_lease), ms(aisle.t_tick), ms(aisle.t_bus), ms(aisle.t_brake_onset)
    tau = ms(aisle.tau_delay)
    near(age + lease + tick + bus + onset, tau, 1e-9)
    n.update(age=fx(age, 1), backbone=fx(backbone, 0), agerest=fx(age - backbone, 1), lease=fx(lease, 0),
             tick=fx(tick, 0), bus=fx(bus, 0), onset=fx(onset, 0), tau=fx(tau, 1), taus=fx(tau / 1000, 4))

    # ---- Front rates of the five-level picture (chapter: LocatorRates)
    n.update(intenthz=fx(robot.intent_rate.to("Hz").magnitude, 0),
             stephz=fx((1 / robot.chunk_step).to("Hz").magnitude, 0),
             tickkhz=fx(robot.permission_rate.to("kHz").magnitude, 0))

    # ---- Worked example: the elementary stop at the drive limit (Scenarios.WarehouseAisleStopAtDriveLimit)
    stop = SC.WarehouseAisleStopAtDriveLimit
    res = stop.result
    v = stop.speed.to("m/s").magnitude
    a = stop.deceleration.to("m/s**2").magnitude
    clear = stop.clear_distance.to("mm").magnitude
    t_s = r(tau / 1000, 4)
    delay_mm = v * t_s * 1000
    brake_mm = v ** 2 / (2 * a) * 1000
    stop_mm = r(delay_mm, 1) + r(brake_mm, 1)
    spare_mm = clear - stop_mm
    vstar = 2 * a * t_s
    near(delay_mm, res.delay_travel.to("mm").magnitude, 1e-6)
    near(brake_mm, res.braking_travel.to("mm").magnitude, 1e-6)
    near(stop_mm, res.stop_distance.to("mm").magnitude, 1e-6)
    near(spare_mm, res.spare.to("mm").magnitude, 1e-6)
    near(vstar, res.crossover_speed.to("m/s").magnitude, 1e-9)
    assert vstar < v and spare_mm > 0
    per_ms = v                                                       # mm of travel per ms of delay at v m/s
    n.update(speed=fx(v, 1), speedsq=fx(v ** 2, 2), decel=fx(a, 1), twodecel=fx(2 * a, 1), clear=fx(clear / 1000, 2),
             clearmm=fx(clear, 0), delaytravel=fx(delay_mm, 1), braketravel=fx(brake_mm, 1), stopmm=fx(stop_mm, 1),
             spare=fx(spare_mm, 1), crossover=fx(vstar, 2), permm=fx(per_ms, 1),
             brakeratio=fx(brake_mm / delay_mm, 1),
             backbonemm=fx(backbone * per_ms, 0), leasemm=fx(lease * per_ms, 0))

    # ---- Your turn. ASSUMED: a more accurate perception backbone takes 80 ms instead of the card's 22 ms
    big = 80.0
    tau_big = tau - backbone + big
    t_big_s = r(tau_big / 1000, 4)
    delay_big = v * t_big_s * 1000
    stop_big = r(delay_big, 1) + r(brake_mm, 1)
    spare_big = clear - stop_big
    vstar_big = 2 * a * t_big_s
    near((big - backbone) * per_ms, delay_big - delay_mm, 1e-6)
    big_res = stop.variant(name="sheet: bigger backbone", delay=Q_(tau_big, "ms")).result
    near(stop_big, big_res.stop_distance.to("mm").magnitude, 1e-6)
    assert spare_big > 0 and vstar_big < v
    n.update(bigbackbone=fx(big, 0), taubig=fx(tau_big, 1), taubigs=fx(t_big_s, 4), delaybig=fx(delay_big, 1),
             stopbig=fx(stop_big, 1), sparebig=fx(spare_big, 1), crossoverbig=fx(vstar_big, 2),
             addedmm=fx((big - backbone) * per_ms, 0), addedms=fx(big - backbone, 0))

    # ---- Mistake: the human takeover (machine card t_takeover_in, at the drive limit)
    card = {row.key: row for row in machine_card()}
    takeover = card["t_takeover_in"].value.to("ms").magnitude
    take_mm = takeover * per_ms
    n.update(takeover=fx(takeover, 0), takeoverticks=fx(takeover / tick, 0), takeovermm=fx(take_mm, 0),
             takeovershare=fx(100 * r(take_mm, 0) / clear, 0))

    # ---- Q1 Tempe (Scenarios.TempeActionSuppression; NTSB HAR-19/03 values as the chapter prints them)
    supp = SC.TempeActionSuppression
    v1 = supp.speed.to("m/s").magnitude
    v2 = supp.speed_at_delay_end.to("m/s").magnitude
    t_supp = supp.delay.to("s").magnitude
    d_supp = (r(v1, 1) + r(v2, 1)) / 2 * t_supp
    near(d_supp, supp.result.delay_travel.to("m").magnitude, 0.05)
    n.update(tempempha=fx(supp.speed.to("mph").magnitude, 1), tempemphb=fx(supp.speed_at_delay_end.to("mph").magnitude, 1),
             tempea=fx(v1, 1), tempeb=fx(v2, 1), tempemean=fx((r(v1, 1) + r(v2, 1)) / 2, 1),
             tempesec=fx(t_supp, 0), tempedist=fx(d_supp, 1))

    # ---- Q3 stall (Scenarios.WarehouseInferenceHiccup): travel on the last command
    hic = SC.WarehouseInferenceHiccup
    stall = hic.delay.to("ms").magnitude
    stall_mm = stall * per_ms
    near(stall_mm, hic.result.delay_travel.to("mm").magnitude, 1e-6)
    assert stall > lease
    n.update(stall=fx(stall, 0), stallmm=fx(stall_mm, 0))

    # ---- Q4 slick floor (Scenarios.HydraulicFluidSlickTraction) against the card's friction premise
    slick = SC.HydraulicFluidSlickTraction
    tr = slick.result
    mu_dry, mu_oil = slick.mu_before, slick.mu_after
    mu_need = data_sheet_lookup()["mu_req"].value
    g = Q_(1, "standard_gravity").to("m/s**2").magnitude
    ceil = r(mu_oil * r(g, 2), 2)                                    # printed 0.05 x 9.81 = 0.49 m/s^2
    near(ceil, tr.traction_ceiling.to("m/s**2").magnitude, 0.005)
    brake_slick = v ** 2 / (2 * ceil)
    assert not tr.premise_holds and brake_slick * 1000 > clear
    n.update(mudry=fx(mu_dry, 2), muoil=fx(mu_oil, 2), muneed=fx(mu_need, 3), gravity=fx(g, 2),
             tractionceiling=fx(ceil, 2), twoceiling=fx(2 * ceil, 2), brakeslick=fx(brake_slick, 2))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
