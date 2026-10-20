"""Every derived number on the Lecture 18 slides (Volume IV, Engineering Physical AI).

Run:  python3 calcs.py
Each block names the slide it serves, lists its inputs with their sources, and
prints each result with the label the slide uses. Warehouse-aisle inputs come
from the MLSysIM registry (Embodied.Scenario.WarehouseAisle), as the chapter's
own cells do; other inputs are typed from the rendered chapter or a cited
source. Only derived values are computed here.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))

from mlsysim import Embodied  # noqa: E402
from mlsysim.physics import (  # noqa: E402
    calc_c2_stop_suffix,
    calc_scurve_max_permitted_velocity,
    calc_scurve_stop_distance,
    calc_scurve_stop_time,
)
from mlsysim.core.units import meter, millimeter, ms, second  # noqa: E402

AISLE = Embodied.Scenario.WarehouseAisle
RM = Embodied.MobileManipulator.WarehouseMobileManipulator
G = 9.80665  # m/s^2, standard gravity (defined constant; the chapter prints 9.81)


def mag(q, unit):
    return q.to(unit).magnitude


# --- Warehouse aisle inputs (registry; categories as the chapter labels them)
A_BRAKE = mag(AISLE.a_brake, meter / second**2)    # 2 m/s^2, illustrative
MU_DRY = AISLE.mu_dry                              # 0.60, illustrative
MU_INSP = AISLE.mu_inspected_floor                 # 0.12, illustrative
MU_OIL = AISLE.mu_oil_film                         # 0.05, illustrative
V_AISLE = mag(AISLE.v_aisle, meter / second)       # 1.3 m/s, chosen
D_CLEAR = mag(AISLE.d_clear, meter)                # 1.10 m, illustrative
TAU = mag(AISLE.tau_delay, second)                 # 133.6 ms, derived (known components)
DELTA_DA = mag(AISLE.t_dispatch_to_admission, second)  # 63.3 ms allocation, illustrative
OVERHEAD = mag(AISLE.fixed_overhead, meter)        # 190 mm, derived
DELTA_LOC = mag(AISLE.delta_loc, meter)            # 50 mm
DELTA_MARGIN = mag(AISLE.delta_margin, meter)      # 100 mm
EPS_TRACK = mag(AISLE.eps_track, meter)            # 40 mm
V_RESTRICTED = 1.0                                 # m/s, chosen (register's restricted mode, sec. 18.3.3)
AISLE_LEG = mag(AISLE.aisle_length, meter)         # 30 m aisle segment (sec. 18.4.2)
T_DETECT = 0.100                                   # s, chosen film-detector latency (sec. 18.3.4)


def a_avail(mu):
    """Equation 1: available deceleration = min(brake, mu g)."""
    return min(A_BRAKE, mu * G)


J_BRAKE = mag(calc_c2_stop_suffix(RM.base.max_velocity, AISLE.a_brake)["peak_jerk"], meter / second**3)


def budget_terms(v, mu):
    """Known-component budget (Napkin Math 18.2): delay, braking, jerk ramp, overheads, in m."""
    a = a_avail(mu)
    return dict(delay=v * TAU, braking=v**2 / (2 * a), ramp=v * a / (2 * J_BRAKE), overhead=OVERHEAD)


def scurve_total(v, mu):
    a = a_avail(mu) * meter / second**2
    vv = v * meter / second
    return v * TAU + mag(calc_scurve_stop_distance(vv, a, J_BRAKE * meter / second**3), meter) + OVERHEAD


# --- Packaging cell inputs (Napkin Math 18.1; all illustrative)
CELL_M = 12.0          # kg
CELL_V = 1.50          # m/s
CELL_CLEAR = 0.060     # m
CELL_RESERVE = 0.015   # m
CELL_A = 15.0          # m/s^2
CELL_CHAIN = [("image", 16.0), ("transfer, inference, admission", 36.0), ("bus", 2.0), ("brake onset", 10.0)]  # ms
CELL_V_MODE = 0.50     # m/s candidate mode
CURTAIN_MS = 10.0      # ms, SICK deTec4 Set 30/450 response time, uncoded (product data sheet)


def cell_tau(chain=None):
    return sum(v for _, v in (chain or CELL_CHAIN)) / 1000


def cell_stop(v, tau, a=CELL_A):
    return v * tau, v**2 / (2 * a)


def ceiling(tau, a, d):
    """Positive root of v tau + v^2/(2a) = d (equation 2.3)."""
    return -a * tau + math.sqrt((a * tau) ** 2 + 2 * a * d)


# --- Coverage bill inputs (Napkin Math 18.3; illustrative)
P_TARGET = 1e-9
ALPHA = 0.05
FLEET, DAYS, HPD = 100, 90.0, 24.0
C_SHIELD = 0.9999

# --- Latch inputs (registry; sec. 18.3.4)
K_PLATE = mag(AISLE.k_latch, AISLE.k_latch.units)  # N/m
V_GUARD = mag(AISLE.v_latch_approach, meter / second)
V_FAST = mag(AISLE.v_latch_high, meter / second)
F_TRIP = 15.0     # N, chosen tripwire (AISLE.f_latch_tripwire)
F_FLOOR = 100.0   # N, declared strength floor (AISLE.f_latch_limit)
T_RESP = mag(AISLE.t_contact_response, second)
M_EFF = 12.0      # kg, RM.arm_effective_contact_mass


def zero_failure_n(p_fail, conf=0.95):
    return math.ceil(math.log(1 - conf) / math.log(1 - p_fail))


def main():
    print("== Hook and Napkin Math 18.2: the friction floor")
    mu_min = A_BRAKE / G
    print(f"friction floor mu = a_brake/g: {mu_min:.3f}")
    print(f"film deceleration mu g: {a_avail(MU_OIL):.2f} m/s^2; inspected floor: {a_avail(MU_INSP):.2f} m/s^2")
    for name, mu in (("dry", MU_DRY), ("film", MU_OIL)):
        t = budget_terms(V_AISLE, mu)
        tot = sum(t.values())
        print(f"{name}: delay {t['delay']*1000:.1f} mm, braking {t['braking']*1000:.1f} mm, "
              f"ramp {t['ramp']*1000:.1f} mm, overheads {t['overhead']*1000:.1f} mm, total {tot*1000:.1f} mm")
        print(f"   S-curve check total: {scurve_total(V_AISLE, mu)*1000:.1f} mm")
    ratio = budget_terms(V_AISLE, MU_OIL)["braking"] / budget_terms(V_AISLE, MU_DRY)["braking"]
    print(f"braking term grows by: {ratio:.2f}x (= {A_BRAKE:.0f} / {a_avail(MU_OIL):.3f})")
    print(f"jerk bound: {J_BRAKE:.2f} m/s^3")

    print("\n== Synthesis: the finished aisle budget at 1.3 m/s (dry)")
    t = budget_terms(V_AISLE, MU_DRY)
    tot = sum(t.values())
    spare = D_CLEAR - tot
    print(f"overheads: localization {DELTA_LOC*1000:.0f} + margin {DELTA_MARGIN*1000:.0f} + tracking {EPS_TRACK*1000:.0f} mm")
    print(f"known-component budget {tot*1000:.1f} mm; spare {spare*1000:.1f} mm = {spare/V_AISLE*1000:.1f} ms")
    da_mm = V_AISLE * DELTA_DA * 1000
    print(f"dispatch-to-admission {DELTA_DA*1000:.1f} ms = {da_mm:.1f} mm; spare after it {spare*1000 - da_mm:.1f} mm")
    print(f"timing reserve after allocation: {spare/V_AISLE*1000 - DELTA_DA*1000:.1f} ms")
    v_ceil_dry = mag(calc_scurve_max_permitted_velocity(AISLE.d_clear, AISLE.tau_delay, AISLE.a_brake,
                                                        J_BRAKE * meter / second**3, AISLE.delta_loc + AISLE.eps_track,
                                                        AISLE.delta_margin), meter / second)
    v_ceil_insp = mag(calc_scurve_max_permitted_velocity(AISLE.d_clear, AISLE.tau_delay, a_avail(MU_INSP) * meter / second**2,
                                                         J_BRAKE * meter / second**3, AISLE.delta_loc + AISLE.eps_track,
                                                         AISLE.delta_margin), meter / second)
    print(f"known-component ceiling dry {v_ceil_dry:.2f} m/s; inspected floor {v_ceil_insp:.2f} m/s")
    print(f"film-detector range at 100 ms: {(V_AISLE * T_DETECT + tot):.2f} m")

    print("\n== Time form against distance form (pitfall, sec. 18.3.4)")
    vv = V_AISLE * meter / second
    a = A_BRAKE * meter / second**2
    jj = J_BRAKE * meter / second**3
    t_stop = mag(calc_scurve_stop_time(vv, a, jj), ms)
    d_stop = mag(calc_scurve_stop_distance(vv, a, jj), millimeter)
    t_eq = d_stop / V_AISLE
    print(f"S-curve stop time {t_stop:.1f} ms; distance {d_stop:.1f} mm; distance/v {t_eq:.1f} ms")
    print(f"over-reserve {t_stop - t_eq:.1f} ms = {(t_stop - t_eq) * V_AISLE:.1f} mm; spare {spare/V_AISLE*1000:.1f} ms")

    print("\n== Napkin Math 18.1: the delay wall (packaging cell)")
    tau = cell_tau()
    d1, d2 = cell_stop(CELL_V, tau)
    budget = CELL_CLEAR - CELL_RESERVE
    print(f"chain total {tau*1000:.0f} ms; stop at {CELL_V} m/s: {d1*1000:.0f} + {d2*1000:.0f} = {(d1+d2)*1000:.0f} mm; "
          f"deficit {(d1+d2-CELL_CLEAR)*1000:.0f} mm against {CELL_CLEAR*1000:.0f} mm")
    print(f"a tau = {CELL_A*tau:.2f} m/s; 2 a d = {2*CELL_A*budget:.2f} m^2/s^2; root {math.sqrt((CELL_A*tau)**2 + 2*CELL_A*budget):.2f}")
    v_max = ceiling(tau, CELL_A, budget)
    print(f"speed ceiling: {v_max:.2f} m/s")
    e1, e2 = cell_stop(CELL_V_MODE, tau)
    print(f"0.50 m/s: {e1*1000:.0f} + {e2*1000:.1f} = {(e1+e2)*1000:.1f} mm; left {(CELL_CLEAR-e1-e2)*1000:.1f} mm")
    ke1, ke2 = 0.5 * CELL_M * CELL_V**2, 0.5 * CELL_M * CELL_V_MODE**2
    print(f"kinetic energy {ke1:.1f} J -> {ke2:.1f} J, reduction {100*(1-ke2/ke1):.1f} percent")
    print(f"delay travel falls {d1/e1:.0f}x, braking {d2/e2:.0f}x")
    print("-- sensitivity")
    half_proc = [(n, v / 2 if n.startswith("transfer") else v) for n, v in CELL_CHAIN]
    print(f"processing halved to 18 ms: tau {cell_tau(half_proc)*1000:.0f} ms, ceiling {ceiling(cell_tau(half_proc), CELL_A, budget):.2f} m/s")
    print(f"braking halved to 7.5: ceiling {ceiling(tau, CELL_A / 2, budget):.2f} m/s")
    curtain = [("light curtain (datasheet)", CURTAIN_MS), ("bus", 2.0), ("brake onset", 10.0)]
    print(f"certified curtain path: tau {cell_tau(curtain)*1000:.0f} ms, ceiling {ceiling(cell_tau(curtain), CELL_A, budget):.2f} m/s (derived)")
    print("-- Checkpoint 18.1: brake qualified at 12 m/s^2")
    v12 = ceiling(tau, 12.0, budget)
    f1, f2 = cell_stop(CELL_V_MODE, tau, 12.0)
    print(f"ceiling {v12:.2f} m/s; 0.50 m/s stop {f1*1000:.0f} + {f2*1000:.1f} = {(f1+f2)*1000:.1f} mm against 45 mm budget, "
          f"{(budget - f1 - f2)*1000:.1f} mm left")

    print("\n== Napkin Math 18.3: the coverage bill")
    hours = FLEET * DAYS * HPD
    p_up = -math.log(ALPHA) / hours
    resid = p_up * (1 - C_SHIELD)
    c_req = 1 - P_TARGET / p_up
    p_lim = P_TARGET / (1 - C_SHIELD)
    hours_need = -math.log(ALPHA) / p_lim
    days_need = hours_need / (FLEET * HPD)
    n_cov = math.ceil(math.log(ALPHA) / math.log(C_SHIELD))
    n_cov_j = math.ceil(math.log(ALPHA / 2) / math.log(C_SHIELD))
    p_up_j = -math.log(ALPHA / 2) / hours
    print(f"proposal-hours {hours:.3g}; -ln alpha {-math.log(ALPHA):.3f}; bound {p_up:.4g} per h")
    print(f"residual {resid:.4g} per h against target {P_TARGET:g}; ratio {resid/P_TARGET:.2f}")
    print(f"coverage needed at this bound {c_req:.7f}")
    print(f"proposal bound needed {p_lim:.0e}; hours {hours_need:.3g} = {days_need:.1f} days, {days_need-DAYS:.1f} beyond the cycle")
    print(f"coverage campaign {n_cov:,} trials; joint (97.5 percent) {n_cov_j:,}; joint residual {p_up_j*(1-C_SHIELD):.4g}")

    print("\n== Residual-claims register closure tests")
    n_film = zero_failure_n(0.001)
    p_1000 = 1 - ALPHA ** (1 / 1000)
    print(f"zero-failure trials for 0.1 percent at 95 percent: {n_film:,}; 1,000 trials bound {p_1000:.6f}")

    print("\n== What a restriction costs (sec. 18.4.2)")
    cut = 1 - V_RESTRICTED / V_AISLE
    t13, t10 = AISLE_LEG / V_AISLE, AISLE_LEG / V_RESTRICTED
    print(f"speed cut {100*cut:.1f} percent; 30 m leg {t13:.1f} s -> {t10:.0f} s, +{100*(t10/t13-1):.0f} percent")

    print("\n== Latch compliance (sec. 18.3.4)")
    a_guard = V_GUARD**2 / (2 * ((F_FLOOR - F_TRIP) / K_PLATE - V_GUARD * T_RESP))
    print(f"guarded stop deceleration on the rigid plate: {a_guard:.2f} m/s^2")
    stroke = V_FAST * T_RESP + V_FAST**2 / (2 * a_guard)
    k_eq = (F_FLOOR - F_TRIP) / stroke
    k_spring = 1 / (1 / k_eq - 1 / K_PLATE)
    print(f"driven stroke {stroke*1000:.2f} mm; k_eq {k_eq/1000:.1f} kN/m; spring {k_spring/1000:.1f} kN/m; "
          f"stroke needed {F_FLOOR/k_spring*1000:.1f} mm")
    coast = lambda k, v: math.sqrt(F_TRIP**2 + k * M_EFF * v**2)  # noqa: E731
    print(f"coasting peaks: rigid 0.03 {coast(K_PLATE, V_GUARD):.1f} N; rigid 0.10 {coast(K_PLATE, V_FAST):.1f} N; "
          f"spring 0.10 {coast(k_eq, V_FAST):.1f} N")

    print("\n== Measured friction (TRL PPR073, Lambourn and Viner 2006, table 5.1)")
    for lab, mu in TRL_ROWS:
        print(f"   {lab}: {mu}  binds={'floor' if mu < mu_min else 'brake'}")


# TRL PPR073 table 5.1, averaged locked-wheel coefficients (typed from the report)
TRL_ROWS = [
    ("Concrete, dry, skid cars", 0.85),
    ("Concrete, wet, smooth tyre", 0.52),
    ("Concrete, diesel, skid cars", 0.44),
    ("Concrete, diesel, smooth tyre", 0.24),
    ("Asphalt, engine oil, skid cars", 0.41),
    ("Asphalt, engine oil, smooth tyre", 0.01),
]


if __name__ == "__main__":
    main()
