#!/usr/bin/env python3
"""Every derived number in Lecture 16 (Deployment Release), one block per slide.

Inputs carry their source. "DR" is the rendered chapter, Deployment Release
(Volume IV, Chapter 16). Running-machine values are read from the MLSysIM
registry, the same source the chapter prints from (Embodied.MobileManipulator.
WarehouseMobileManipulator and Embodied.Scenario.WarehouseAisle), with the
chapter's own functions, so the slide and the book agree. Outside sources are
named where they are used. Every printed value is what the slide or its note
shows, with the label used there. make_figs.py imports from this file.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

MLSYSIM = Path(__file__).resolve().parents[4] / "mlsysim"
sys.path.insert(0, str(MLSYSIM))

from mlsysim import Embodied, Literature  # noqa: E402
from mlsysim.core.units import meter, millimeter, millisecond, second, standard_gravity  # noqa: E402
from mlsysim.physics.robotics import (  # noqa: E402
    calc_c2_stop_suffix,
    calc_latch_contact_stopping_profile,
    calc_scurve_max_permitted_velocity,
    calc_scurve_stop_distance,
)


def head(title):
    print(f"\n--- {title}")


def show(label, value):
    print(f"  {label}: {value}")


RM = Embodied.MobileManipulator.WarehouseMobileManipulator  # running machine (illustrative)
AISLE = Embodied.Scenario.WarehouseAisle                      # aisle scene (illustrative; speeds chosen)
G = (1 * standard_gravity).to(meter / second**2).magnitude    # 9.80665 m/s^2

# =============================================================================
# Shared inputs (DR Napkin Math 16.1 variables; registry values)
# =============================================================================
V = AISLE.v_aisle.to(meter / second).magnitude                 # 1.3 m/s, chosen
D_CLEAR = AISLE.d_clear.to(millimeter).magnitude               # 1,100 mm, illustrative
TAU = AISLE.tau_delay.to(millisecond).magnitude                # 133.6 ms = 51.6 + 82
A_BRAKE = AISLE.a_brake.to(meter / second**2).magnitude        # 2 m/s^2, illustrative
LOC = AISLE.delta_loc.to(millimeter).magnitude                 # 50 mm, illustrative
TRACK = AISLE.eps_track.to(millimeter).magnitude               # 40 mm, illustrative
MARGIN = AISLE.delta_margin.to(millimeter).magnitude           # 100 mm, chosen
T_DA = AISLE.t_dispatch_to_admission.to(millisecond).magnitude  # 63.3 ms, illustrative allocation
# Jerk bound: the reference C2 stop's peak jerk at the drive limit (DR NM 16.1)
J = calc_c2_stop_suffix(RM.base.max_velocity, AISLE.a_brake)["peak_jerk"].to(meter / second**3).magnitude


def braking_mm(v, a, j=J):
    """Jerk-limited stop distance (DR NM 16.1 step 2 form), mm; valid when v >= a^2/j."""
    return (v * v / (2 * a) + v * a / (2 * j)) * 1e3


# =============================================================================
# Recall slide: Lecture 15's fault F1 (DR 16.5.3 and NM 16.2 use the same 82 ms)
# =============================================================================
T_LEASE = AISLE.t_lease.to(millisecond).magnitude      # 60 ms, chosen
T_TICK = AISLE.t_tick.to(millisecond).magnitude        # 1 ms, chosen
T_BUS = AISLE.t_bus.to(millisecond).magnitude          # 1 ms, illustrative
T_ONSET = AISLE.t_brake_onset.to(millisecond).magnitude  # 20 ms, illustrative
T_SILENT = T_LEASE + T_TICK + T_BUS + T_ONSET            # 82 ms
T_FLAGGED = T_TICK + T_BUS + T_ONSET                     # 22 ms

# =============================================================================
# Hook and segment 1: the Challenger flight record
# Dalal, Fowlkes, and Hoadley, "Risk Analysis of the Space Shuttle: Pre-
# Challenger Prediction of Failure," JASA 84 (1989), table 1, as posted in the
# UCI Machine Learning Repository, "Challenger USA Space Shuttle O-Ring" data
# set (file o-ring-erosion-or-blowby.data): per flight, O-rings at risk (6),
# O-rings with thermal distress (erosion or blow-by), launch temperature (F),
# leak-check pressure (psi), temporal order. Typed from the file.
# =============================================================================
ORING = [  # (order, launch temperature F, O-rings with distress)
    (1, 66, 0), (2, 70, 1), (3, 69, 0), (4, 68, 0), (5, 67, 0), (6, 72, 0),
    (7, 73, 0), (8, 70, 0), (9, 57, 1), (10, 63, 1), (11, 70, 1), (12, 78, 0),
    (13, 67, 0), (14, 53, 2), (15, 67, 0), (16, 75, 0), (17, 70, 0), (18, 81, 0),
    (19, 76, 0), (20, 79, 0), (21, 75, 2), (22, 76, 0), (23, 58, 1),
]
T_CHALLENGER = 36        # F, air at launch, 28 January 1986 (DR War Story 16.1; Rogers Commission ch. IV finding 6)
T_JOINT_51L = 28         # F, estimated right aft field joint, +/- 5 F (Rogers Commission ch. IV finding 6a)
T_JOINT_51L_ERR = 5
# The data set's temperatures are the calculated joint temperatures of Rogers
# Commission figure 7 (its distressed flights sit at 53, 57, 58, 63, 70, 70, 75 F).
T_THIOKOL = 53           # F, Thiokol's recommended floor, the coldest prior flight (DR War Story 16.1)

# =============================================================================
# Segment 1: the three criteria and the envelope (DR 16.2, table 1)
# =============================================================================
latch_lo = calc_latch_contact_stopping_profile(AISLE.v_latch_approach, scenario=AISLE)
latch_hi = calc_latch_contact_stopping_profile(AISLE.v_latch_high, scenario=AISLE)
V_LATCH = AISLE.v_latch_approach.to(meter / second).magnitude   # 0.03 m/s, chosen
V_LATCH_HI = AISLE.v_latch_high.to(meter / second).magnitude    # 0.10 m/s, illustrative
F_LATCH_LIMIT = AISLE.f_latch_limit.to("newton").magnitude      # 100 N, illustrative
F_CONTACT = AISLE.f_contact_criterion.to("newton").magnitude    # 50 N, chosen (site assessment)
V_HANDOVER = AISLE.v_handover.to(meter / second).magnitude      # 0.10 m/s, chosen
M_LOADED = RM.loaded_mass.to("kilogram").magnitude              # 368 kg (derived upper bound)
WCET_US, DEADLINE_US = 250, 400   # enforcer WCET declared, permission deadline (DR table 2, C2)
F_HANDOVER = 20.7                  # N, free-hand transient force at 0.10 m/s (DR 16.2; Lecture 12's model)
F_HANDOVER_CLAMPED = 94.9          # N, clamped hand at 0.10 m/s (DR 16.2)
V_HAND_CEIL, V_HAND_CEIL_CLAMPED = 0.24, 0.053  # m/s ceilings (DR 16.2)

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    head("Recall slide: Lecture 15 fault F1 (proposer stall), DR NM 16.2 step 1")
    show("lease + tick + bus + onset (ms)", f"{T_LEASE:g} + {T_TICK:g} + {T_BUS:g} + {T_ONSET:g} = {T_SILENT:g}")

    head("Hook and O-ring slide: the flight record (Dalal et al. 1989 via UCI; joint temperatures)")
    temps = [t for _, t, _ in ORING]
    cold = [(t, d) for _, t, d in ORING if t <= 63]
    warm = [(t, d) for _, t, d in ORING if t >= 66]
    show("flights in the record", len(ORING))
    show("prior flights (Rogers Commission ch. V: 24 previous times)", 24)
    show("coldest / warmest launch (F)", f"{min(temps)} / {max(temps)}")
    show("flights with distress", sum(1 for _, _, d in ORING if d > 0))
    show("flights at 63 F or below, with distress", f"{sum(1 for _, d in cold if d > 0)} of {len(cold)}")
    show("flights at 66 F or above, with distress", f"{sum(1 for _, d in warm if d > 0)} of {len(warm)}")
    show("Challenger air below the coldest flight joint (F)", min(temps) - T_CHALLENGER)
    show("Challenger joint estimate below the coldest flight (F)", min(temps) - T_JOINT_51L)
    show("distressed O-rings, cold flights / warm flights",
         f"{sum(d for _, d in cold)} / {sum(d for _, d in warm)}")

    head("Claims slide: the three criteria (DR equation 1, table 1)")
    show("aisle speed (m/s)", V)
    show("clear distance (mm)", D_CLEAR)
    show("latch: guarded approach (m/s), limit (N)", f"{V_LATCH:.2f}, {F_LATCH_LIMIT:.0f}")
    show("handover: speed (m/s), criterion (N)", f"{V_HANDOVER:.2f}, {F_CONTACT:.0f}")
    show("enforcer WCET / deadline (us)", f"{WCET_US} / {DEADLINE_US}")
    show("WCET share of deadline (percent)", f"{100 * WCET_US / DEADLINE_US:.1f}")

    head("Estimate slide: the aisle spare (DR Napkin Math 16.1)")
    reaction = V * TAU
    braking = braking_mm(V, A_BRAKE)
    brake_const = V * V / (2 * A_BRAKE) * 1e3
    brake_jerk = V * A_BRAKE / (2 * J) * 1e3
    show("jerk bound (m/s^3)", f"{J:.2f}")
    show("full deceleration reached (v >= a^2/j, m/s)", f"{A_BRAKE**2 / J:.3f} <= {V}")
    show("reaction: 1.3 m/s x 133.6 ms (mm)", f"{reaction:.1f}")
    show("braking: v^2/2a + v a/2j (mm)", f"{brake_const:.1f} + {brake_jerk:.1f} = {braking:.1f}")
    # The chapter rounds each term to 0.1 mm before summing (173.7 + 605.3 + 190).
    reaction_r, braking_r = round(reaction, 1), round(braking, 1)
    total = reaction_r + braking_r + LOC + TRACK + MARGIN
    spare = D_CLEAR - total
    check_lib = (AISLE.v_aisle * AISLE.tau_delay + calc_scurve_stop_distance(AISLE.v_aisle, AISLE.a_brake, J * meter / second**3)
                 + AISLE.delta_loc + AISLE.eps_track + AISLE.delta_margin).to(millimeter).magnitude
    show("reaction + braking (mm)", f"{reaction_r + braking_r:.1f}")
    show("budget total (mm)", f"{total:.1f}  (MLSysIM S-curve check {check_lib:.1f})")
    show("spare (mm)", f"{spare:.1f}")
    show("spare as delay at 1.3 m/s (ms)", f"{spare / V:.1f}")
    room = D_CLEAR - reaction_r - LOC - TRACK - MARGIN   # braking room
    # Solve v^2/(2a) + v a/(2j) = room for a (smaller root: the weakest braking that still fits)
    R = room / 1e3
    qa, qb, qc = V / J, -2 * R, V * V
    a_be = (-qb - math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
    show("braking room (mm)", f"{room:.1f}")
    show("break-even deceleration (m/s^2)", f"{a_be:.2f}")
    show("shortfall that spends the spare (percent)", f"{100 * (1 - a_be / A_BRAKE):.0f}")
    admission = round(V * T_DA, 1)
    reserve = spare - admission
    show("dispatch-to-admission allocation (ms)", f"{T_DA:.1f}")
    show("admission travel: 1.3 m/s x 63.3 ms (mm)", f"{admission:.1f}")
    show("reserve (mm)", f"{reserve:.1f}")
    show("reserve as time at 1.3 m/s (ms)", f"{reserve / V:.1f}")
    show("used before admission work (percent of clear distance)", f"{100 * total / D_CLEAR:.1f}")
    v_ceiling = calc_scurve_max_permitted_velocity(AISLE.d_clear, AISLE.tau_delay, AISLE.a_brake,
                                                   J * meter / second**3, AISLE.delta_loc + AISLE.eps_track,
                                                   AISLE.delta_margin).to(meter / second).magnitude
    show("partial ceiling on surveyed floors (m/s)", f"{v_ceiling:.2f}")

    head("Warrant slide: the exposure wall at 95 percent (Lecture 7; DR 16.3.1)")
    ln20 = math.log(20)
    show("ln 20", f"{ln20:.2f}")
    for lam in (1e-6, 1e-9):
        hours = ln20 / lam
        show(f"failure-free hours to show {lam:g} per hour", f"{hours:.2e}")
        show(f"  as years of continuous operation", f"{hours / (24 * 365):,.0f}")

    head("Premises slide: conjunctive premises and diverse legs (DR equations 2 and 3; stipulated)")
    show("10 premises at 0.99 each", f"{0.99 ** 10:.3f}")
    show("6 claims at 0.99 each", f"{0.99 ** 6:.3f}")
    show("2 independent legs at 0.90 each", f"{1 - (1 - 0.90) ** 2:.2f}")
    show("10 premises at 0.999 each", f"{0.999 ** 10:.3f}")

    head("Certification slide: Keyence SZ-V32 datasheet (MLSysIM registry provenance)")
    sc = RM.safety_scanner
    show("scanner", sc.name)
    show("registry configuration", sc.configuration)
    show("response time (ms) / protection zone (m)", f"{sc.response_time.to(millisecond).magnitude:g} / "
         f"{sc.protective_field_range.to(meter).magnitude:g}")
    show("travel at 1.3 m/s during the 80 ms response (mm)", f"{V * sc.response_time.to(millisecond).magnitude:.0f}")
    show("travel at 1.3 m/s during the 160 ms standard-mode response (mm)", f"{V * 160:.0f}")

    head("Door predict: the latch branch (DR 16.6; Lecture 12's equations 12.11 and 12.12)")
    show("onset force, guarded / fast (N)", f"{latch_lo['f_at_brake_onset'].to('newton').magnitude:.0f} / "
         f"{latch_hi['f_at_brake_onset'].to('newton').magnitude:.0f}")
    show("remaining latch travel, guarded (mm)", f"{latch_lo['x_remaining'].to(millimeter).magnitude:.2f}")
    show("remaining latch travel, fast (mm)", f"{latch_hi['x_remaining'].to(millimeter).magnitude:.4f}")
    a_lo = latch_lo["a_arm_needed"].to(meter / second**2).magnitude
    a_hi = latch_hi["a_arm_needed"].to(meter / second**2).magnitude
    show("arm deceleration needed, guarded / fast (m/s^2)", f"{a_lo:.2f} / {a_hi:.0f}")
    show("guarded stop time (ms)", f"{latch_lo['t_arm_stop'].to(millisecond).magnitude:.1f}")
    show("ratio fast / guarded", f"{a_hi / a_lo:.0f}")
    show("fast demand in g", f"{a_hi / G:.0f}")

    head("COND slide: restricted mug delivery (DR 16.4.1, 16.6.1)")
    mu_min = A_BRAKE / G
    a_insp = min(A_BRAKE, AISLE.mu_inspected_floor * G)
    v_insp = calc_scurve_max_permitted_velocity(AISLE.d_clear, AISLE.tau_delay, a_insp * meter / second**2,
                                                J * meter / second**3, AISLE.delta_loc + AISLE.eps_track,
                                                AISLE.delta_margin).to(meter / second).magnitude
    show("friction floor for 2 m/s^2 (all weight driven)", f"{mu_min:.3f}")
    show("inspected floor friction", AISLE.mu_inspected_floor)
    show("optimistic traction ceiling 0.12 x 9.81 (m/s^2)", f"{a_insp:.2f}")
    show("partial ceiling on inspected floors (m/s)", f"{v_insp:.2f}")
    show("signed restricted speed (m/s), chosen", 1.0)
    show("margin of 1 m/s below the inspected ceiling (percent)", f"{100 * (1 - 1.0 / v_insp):.0f}")

    head("Silent-proposer estimate (DR Napkin Math 16.2)")
    show("silent proposer: lease+tick+bus+onset (ms)", T_SILENT)
    show("silent travel (mm)", f"{V * T_SILENT:.1f}")
    show("monitored, immediate flag (ms)", T_FLAGGED)
    show("monitored travel (mm)", f"{V * T_FLAGGED:.1f}")
    show("travel monitoring can save (mm)", f"{V * T_SILENT - V * T_FLAGGED:.0f}")
    show("crossover flag latency (ms)", T_SILENT - T_FLAGGED)
    show("travel per ms at 1.3 m/s (mm)", V)
    ke = 0.5 * M_LOADED * V * V
    show("kinetic energy, 368 kg at 1.3 m/s (J)", f"{ke:.0f}")

    head("Waymo slide: rider-only crash rates by outcome (Waymo Safety Impact Data Hub via MLSysIM)")
    rows = Literature.WaymoSafetyImpact.entries
    last = max(r["data_cut"] for r in rows)
    cuts = sorted({r["data_cut"] for r in rows})
    latest = {r["outcome"]: r for r in rows if r["data_cut"] == last}
    show("latest data cut", int(last))
    show("releases in the registry", len(cuts))
    show("rider-only miles (millions)", f"{latest['Any-injury-reported']['ro_miles_millions']:.0f}")
    for name, r in sorted(latest.items(), key=lambda kv: -kv[1]["waymo_count"]):
        lo, hi = -100 * r["pct_reduction_ci_hi"], -100 * r["pct_reduction_ci_lo"]
        show(f"{name}: n, reduction, 95% interval (percent), width",
             f"{r['waymo_count']:.0f}, {-100 * r['pct_reduction']:.0f}, {lo:.0f} to {hi:.0f}, {hi - lo:.0f}")
    first = min(cuts)
    inj0 = [r for r in rows if r["data_cut"] == first and r["outcome"] == "Any-injury-reported"][0]
    show("first data cut and its miles (millions)", f"{int(first)}, {inj0['ro_miles_millions']:.1f}")
    w_inj = 100 * (latest["Any-injury-reported"]["pct_reduction_ci_hi"] - latest["Any-injury-reported"]["pct_reduction_ci_lo"])
    w_ssi = 100 * (latest["Suspected Serious Injury+"]["pct_reduction_ci_hi"] - latest["Suspected Serious Injury+"]["pct_reduction_ci_lo"])
    show("interval width ratio, serious injury / injury", f"{w_ssi / w_inj:.1f}")

    head("Recall-records slide: NHTSA campaigns (api.nhtsa.gov/recalls/campaignNumber, fetched 2026-10-19)")
    recalls = [("24E-013", "2024-02-13", 444), ("24E-049", "2024-06-10", 672), ("25E-034", "2025-05-12", 1212)]
    for c, d, n in recalls:
        show(f"{c} received {d}, vehicles", n)
    show("fleet growth across the three recalls (x)", f"{recalls[-1][2] / recalls[0][2]:.1f}")
    # 25E-034: remedy complete 2024-12-26, report received 2025-05-12
    from datetime import date
    d0, d1 = date(2024, 2, 13), date(2025, 5, 12)
    show("months from first to third recall report", (d1.year - d0.year) * 12 + d1.month - d0.month)
    show("days from fix complete to recall report (25E-034)", (date(2025, 5, 12) - date(2024, 12, 26)).days)
    show("days from software cutoff to fix complete (24E-013)", (date(2024, 1, 12) - date(2023, 12, 20)).days)

    head("Canary note: kinetic energy (DR 16.10 fallacy)")
    show("0.5 x 368 kg x (1.3 m/s)^2 (J)", f"{ke:.0f}")

    head("Synthesis: Challenger as a release case")
    show("Challenger joint estimate below the 53 F floor (F)", T_THIOKOL - T_JOINT_51L)
