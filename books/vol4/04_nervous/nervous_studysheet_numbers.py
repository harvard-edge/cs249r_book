"""Numbers for The Nervous System study sheet (Volume IV, Chapter 4).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to nervous_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer; asserts tie each chain
back to the chapter's scenario results.
"""
import math
from pathlib import Path

from mlsysim import Embodied, Scenarios


def f(x, nd):
    return f"{x:,.{nd}f}"


def numbers() -> dict:
    n = {}
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    aisle = Embodied.Scenario.WarehouseAisle
    lease = Scenarios.WarehouseChunkLease
    lr = lease.result

    # ---- Front: the four tiers (chapter: MachineTiers)
    t_current_us = (1 / rm.current_loop_rate).to("microsecond").magnitude
    t_tick_ms = (1 / rm.permission_rate).to("millisecond").magnitude
    t_chunk_ms = rm.chunk_period.to("millisecond").magnitude
    t_intent_ms = (1 / rm.intent_rate).to("millisecond").magnitude
    n.update(tiercurrent=f(t_current_us, 0), tiertick=f(t_tick_ms, 0), tierchunk=f(t_chunk_ms, 0),
             tierintent=f(t_intent_ms, 0), chunkinfer=f(rm.chunk_inference_latency.to("ms").magnitude, 0),
             intentinfer=f(rm.intent_inference_latency.to("ms").magnitude, 0))
    # ladder positions on a log axis from 10 us to 1 s, in seconds
    n.update(ladcurrent=f(t_current_us * 1e-6, 6), ladtick=f(t_tick_ms * 1e-3, 6), ladchunk=f(t_chunk_ms * 1e-3, 6),
             ladintent=f(t_intent_ms * 1e-3, 6))

    # ---- Stopping-budget inputs (chapter: MachineLeaseBudget, notebook "The price of a lease")
    v = lease.speed.to("m/s").magnitude
    a = aisle.a_brake.to("m/s^2").magnitude
    dclear_mm = aisle.d_clear.to("mm").magnitude
    over_mm = (aisle.delta_loc + aisle.delta_margin).to("mm").magnitude
    tick = aisle.t_tick.to("ms").magnitude
    bus = aisle.t_bus.to("ms").magnitude
    onset = aisle.t_brake_onset.to("ms").magnitude
    t_lease = aisle.t_lease.to("ms").magnitude
    renew = rm.chunk_period.to("ms").magnitude
    jitter = lease.renewal_jitter.to("ms").magnitude
    age_dispatch = aisle.perception_age.to("ms").magnitude
    t_da = aisle.t_dispatch_to_admission.to("ms").magnitude
    age_admit = aisle.t_capture_to_admission.to("ms").magnitude
    assert abs(age_admit - (age_dispatch + t_da)) < 1e-9
    mm_per_ms = v  # m/s == mm/ms
    n.update(speed=f(v, 1), decel=f(a, 1), clear=f(dclear_mm, 0), clearm=f(dclear_mm / 1e3, 2), over=f(over_mm, 0),
             tick=f(tick, 0), bus=f(bus, 0), onset=f(onset, 0), lease=f(t_lease, 0), renew=f(renew, 0),
             jitter=f(jitter, 0), agedispatch=f(age_dispatch, 1), dispatchadmit=f(t_da, 1), ageadmit=f(age_admit, 1),
             mmperms=f(mm_per_ms, 1), fixedpath=f(tick + bus + onset, 0))

    # ---- Worked example: size the lease at the drive limit (chapter's numbers)
    brake_mm = v ** 2 / (2 * a) * 1e3
    assert abs(brake_mm - Scenarios.WarehouseBaseStopBodyTerms.result.braking_travel.to("mm").magnitude) < 1e-9
    ceiling = (dclear_mm - over_mm - round(brake_mm, 1)) / v
    assert abs(ceiling - lr.delay_ceiling.to("ms").magnitude) < 1e-6
    upper = round(ceiling, 1) - tick - bus - onset - round(age_admit, 1)
    assert abs(upper - lr.lease_ceiling_at_admission.to("ms").magnitude) < 0.05
    floor = renew + jitter
    assert abs(floor - lr.lease_floor.to("ms").magnitude) < 1e-9
    cost = t_lease * mm_per_ms
    spare = upper - t_lease
    n.update(wbrake=f(brake_mm, 1), wceiling=f(ceiling, 1), wupper=f(upper, 1), wfloor=f(floor, 0),
             wcost=f(cost, 0), wspare=f(spare, 1))
    # front picture: the pre-brake ceiling split (age at admission | lease | tick+bus+onset | spare), 1 ms = 0.9 pt
    sc = 0.9
    # drawn in the order of the rule: age at admission | tick+bus+onset | lease | spare (lease + spare = upper limit)
    n.update(page=f(round(age_admit, 1) * sc, 2), plea=f(t_lease * sc, 2), pfix=f((tick + bus + onset) * sc, 2),
             pspare=f(spare * sc, 2), pceil=f(round(ceiling, 1) * sc, 2),
             pfloor=f((round(age_admit, 1) + tick + bus + onset + floor) * sc, 2))

    # ---- Your turn: ASSUMED worn, dusty floor; credible deceleration drops to 1.6 m/s^2, all else unchanged
    a2 = 1.6
    brake2 = v ** 2 / (2 * a2) * 1e3
    ceiling2 = (dclear_mm - over_mm - round(brake2, 1)) / v
    upper2 = round(ceiling2, 1) - tick - bus - onset - round(age_admit, 1)
    # highest speed at which the bounds still separate (upper limit == floor), solved for v
    need = (floor + tick + bus + onset + round(age_admit, 1)) / 1e3  # s
    avail = (dclear_mm - over_mm) / 1e3  # m
    # (avail - v^2/(2 a2)) / v = need  ->  v^2/(2 a2) + need v - avail = 0
    qa, qb, qc = 1 / (2 * a2), need, -avail
    vmax2 = (-qb + math.sqrt(qb ** 2 - 4 * qa * qc)) / (2 * qa)
    # highest speed that keeps the chosen 60 ms lease, printed rounded down so the printed speed fits
    need60 = (t_lease + tick + bus + onset + round(age_admit, 1)) / 1e3
    vkeep = (-need60 + math.sqrt(need60 ** 2 + 4 * qa * avail)) / (2 * qa)
    vkeep_print = math.floor(vkeep * 100) / 100
    assert ((dclear_mm - over_mm) / 1e3 - vkeep_print ** 2 / (2 * a2)) / vkeep_print >= need60
    n.update(ydecel=f(a2, 1), ykeep=f(vkeep_print, 2), ybrake=f(brake2, 1), yceiling=f(ceiling2, 1), yupper=f(upper2, 1), yvmax=f(vmax2, 2))
    assert upper2 < floor < upper

    # ---- Q1: response-time bound (chapter: TickResponseBudget); ASSUMED new 45 us interrupt, 210 us bench maximum
    rta = Scenarios.WarehousePermissionTickResponse
    c = aisle.enforcer_wcet.to("us").magnitude
    b = rta.site.enforcer_blocking.to("us").magnitude
    i = rta.site.enforcer_interference.to("us").magnitude
    d = aisle.enforcer_deadline.to("us").magnitude
    assert abs(c + b + i - rta.result.response.to("us").magnitude) < 1e-9
    extra, bench = 45.0, 210.0
    r1 = c + b + i + extra
    n.update(wcet=f(c, 0), blocking=f(b, 0), interference=f(i, 0), deadline=f(d, 0), response=f(c + b + i, 0),
             qextra=f(extra, 0), qinterf=f(i + extra, 0), qresp=f(r1, 0), qmargin=f(d - r1, 0), qbench=f(bench, 0),
             unloaded=f(aisle.enforcer_unloaded.to("us").magnitude, 0))

    # ---- Q2 and front card: phase lag through the tick and bus (chapter: TickDelayBudget)
    loop = Scenarios.WarehouseJointLoopThroughTick
    jr = loop.result
    hold = loop.sample_period.to("ms").magnitude / 2
    comp = loop.compute_delay.to("ms").magnitude
    tbus = loop.transport_delay.to("ms").magnitude
    total = hold + comp + tbus
    allow = loop.phase_allowance.to("degree").magnitude
    fc = loop.crossover.to("Hz").magnitude
    phase20 = 360 * fc * total / 1e3
    assert abs(phase20 - jr.phase_total.to("degree").magnitude) < 1e-9
    ftrim = 6.0  # ASSUMED outer compliance-trim crossover, Hz
    phase6 = 360 * ftrim * total / 1e3
    fmax = allow / (360 * total / 1e3)
    assert abs(fmax - jr.crossover_ceiling.to("Hz").magnitude) < 1e-9
    n.update(fullturn=f(360, 0), hold=f(hold, 1), comp=f(comp * 1e3, 0), loopbus=f(tbus, 0), totaldelay=f(total, 2), allow=f(allow, 0),
             crossover=f(fc, 0), phasejoint=f(phase20, 1), trim=f(ftrim, 0), phasetrim=f(phase6, 1), fmax=f(fmax, 1),
             phasebus=f(jr.phase_transport.to("degree").magnitude, 1))

    # ---- Q3: timers (chapter: MachineLeaseBudget)
    n.update(heartbeat=f(aisle.heartbeat_rate.to("Hz").magnitude, 0), hbtimeout=f(aisle.heartbeat_timeout.to("ms").magnitude, 0),
             hbperiod=f((1 / aisle.heartbeat_rate).to("ms").magnitude, 0),
             watchdog=f(aisle.mcu_self_watchdog.to("ms").magnitude, 0))

    # ---- Mistake: doubling the lease (each ms costs v of clearance)
    long_lease = 2 * t_lease
    n.update(longlease=f(long_lease, 0), longcost=f((long_lease - t_lease) * mm_per_ms, 0),
             longleft=f(upper - long_lease, 1))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
