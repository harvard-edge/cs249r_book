#!/usr/bin/env python3
"""Every derived number in Lecture 2 (The Physical Body), one block per slide.

Inputs carry their source. "PB" is the rendered chapter, The Physical Body
(Volume IV, Chapter 2); "NM 2.x" is its Napkin Math 2.x. Values the chapter
labels illustrative stay illustrative on the slides. Every printed value is
what the slide or its note shows, with the label used there. make_figs.py
imports from this file.

Run from this folder: python3 calcs.py
"""

import math

G = 9.80665                    # standard gravity, m/s^2


def head(title):
    print(f"\n--- {title}")


def show(label, value):
    print(f"  {label}: {value}")


# =============================================================================
# Shared inputs: the running machine (PB NM 2.2, NM 2.3, NM 2.7)
# =============================================================================
MASS = 368.0                   # kg, loaded upper bound (PB NM 2.2, derived upper bound)
V_DRIVE = 1.5                  # m/s, drive limit (PB NM 2.2, illustrative)
D_CLEAR = 1.10                 # m, clear distance to the person (PB NM 2.2, illustrative)
A_BRAKE = 2.0                  # m/s^2, credible deceleration (PB NM 2.2, illustrative)
LOC = 0.050                    # m, localization bound (PB NM 2.2, illustrative)
MARGIN = 0.100                 # m, protective clearance (PB NM 2.2, chosen)
T_ACT = 0.020                  # s, brake onset (PB NM 2.2, illustrative)
TAU_ELEM = 0.1336              # s, elementary pre-brake delay (PB 2.3.1; Lecture 1, NM 1.1)
V_FAST = 2.0                   # m/s, hypothetical speed above the drive limit (PB NM 2.2)

# Hub motor, ZLTECH ZLLG10ASM800-48 datasheet (PB NM 2.3; MLSysIM HubMotors)
R_WHEEL = 0.133                # m, rolling radius
KT_HUB = 2.4                   # N m/A, peak torque / peak current
I_PEAK = 25.0                  # A, peak current = configured limit
TAU_RATED_HUB = 20.0           # N m, rated torque (datasheet)
HUB_MASS = 9.5                 # kg, motor mass, bounds the rotor term
LOAD_PAIR = 300.0              # kg, rated load per pair
KE_HUB_RPM = 0.262             # V/rpm, line-to-line back-EMF constant
R_LINE = 0.7                   # ohm, line resistance
L_LINE = 1.885e-3              # H, line inductance
POLE_PAIRS_HUB = 20
MU_DRY = 0.60                  # dry floor friction (PB NM 2.3)
MU_INSPECTED = 0.12            # inspected low-traction floor (PB NM 2.3)

# Bus (PB NM 2.7)
V_FULL, V_DEPLETED = 54.0, 42.0      # V, pack envelope (MiR250 class)
R_BUS = 0.060                  # ohm, source resistance (illustrative)
C_BUS = 22000e-6               # F, bulk link capacitance
V_UVLO = 33.0                  # V, converter lockout (MEAN WELL DDR-240)
V_CONV_MAX = 67.2              # V, converter continuous input max
V_SWITCH = 100.0               # V, inverter switch rating
V_CLAMP = 60.0                 # V, chopper clamp (chosen)
I_STEP = 70.0                  # A, coincident load step (illustrative)
REGEN_SHARE = 0.75             # share of kinetic energy returned (illustrative)


# =============================================================================
# Hook: Schiaparelli (PB War Story 2.1; ESA 2017 press release)
# =============================================================================
head("Hook: Schiaparelli")
impact_kmh = 540.0                                  # ESA press release
show("impact speed m/s (from 540 km/h)", f"{impact_kmh / 3.6:.0f}")
show("thruster burn share of plan (3 s of 30 s)", f"{3 / 30:.0%}")


# =============================================================================
# Recall (Lecture 1, NM 1.1): the elementary stop
# =============================================================================
head("Recall: elementary stop")
blind_elem = V_DRIVE * TAU_ELEM * 1000
brake = V_DRIVE ** 2 / (2 * A_BRAKE) * 1000
show("blind travel mm", f"{blind_elem:.1f}")
show("braking travel mm", f"{brake:.1f}")
show("elementary stop mm", f"{blind_elem + brake:.1f}")
show("crossover speed m/s (2 a tau)", f"{2 * A_BRAKE * TAU_ELEM:.2f}")


# =============================================================================
# Five budgets slide
# =============================================================================
head("Five budgets: kinetic energy")
KE = 0.5 * MASS * V_DRIVE ** 2
show("kinetic energy J", f"{KE:.0f}")


# =============================================================================
# Segment 1: freshness (PB 2.2, table 1, NM 2.1)
# =============================================================================
head("Freshness: stage medians (table 1, illustrative)")
STAGES_P50 = [("Exposure and readout", 11.2), ("Transport and DMA", 1.4),
              ("Backbone and policy", 18.5), ("Scheduling and IPC", 0.6),
              ("Fieldbus", 1.0), ("Current rise", 2.3)]
STAGES_P99 = [16.6, 3.2, 24.1, 2.1, 1.0, 3.0]
P99_E2E = 46.2                 # ms, system-level p99 (table 1)
P999_E2E = 68.4                # ms, p99.9 age (NM 2.1)
p50_sum = sum(v for _, v in STAGES_P50)
show("p50 chain ms", f"{p50_sum:.1f}")
show("p99 stage sum ms", f"{sum(STAGES_P99):.1f}")
show("inference share of median age", f"{18.5 / p50_sum:.0%}")
show("benchmark rate Hz (1 / 18.5 ms)", f"{1000 / 18.5:.1f}")
show("non-inference median ms", f"{p50_sum - 18.5:.1f}")

head("Freshness deadline (NM 2.1)")
TOL, INIT_ERR, V_BELT = 15.0, 3.0, 0.20   # mm, mm, m/s (illustrative)
e_max = TOL - INIT_ERR
E_DEADLINE = e_max / V_BELT                # mm / (m/s) = ms
show("allowed error mm", f"{e_max:.0f}")
show("freshness deadline ms", f"{E_DEADLINE:.0f}")
mug_p50 = V_BELT * p50_sum
mug_tail = V_BELT * P999_E2E
show("mug travel at median mm", f"{mug_p50:.0f}")
show("mug travel at p99.9 mm", f"{mug_tail:.1f}")
show("tail overrun mm", f"{mug_tail - e_max:.1f}")
show("median slack mm", f"{e_max - mug_p50:.0f}")

head("Encoder quantization (PB 2.2.1)")
step12 = 2 * math.pi / 2 ** 12
show("12-bit step mrad", f"{step12 * 1000:.2f}")
show("one-count spike at 1 kHz deg/s", f"{math.degrees(step12 / 1e-3):.0f}")


# =============================================================================
# Segment 2: stopping (PB 2.3, NM 2.2, figure 2)
# =============================================================================
head("Stopping: three cases")
overhead = (LOC + MARGIN) * 1000
onset = V_DRIVE * T_ACT * 1000
case_body = brake + overhead + onset
case_elem = blind_elem + brake + overhead
blind_fast = V_FAST * TAU_ELEM * 1000
brake_fast = V_FAST ** 2 / (2 * A_BRAKE) * 1000
case_fast = blind_fast + brake_fast + overhead
show("overhead mm", f"{overhead:.0f}")
show("brake onset travel mm", f"{onset:.0f}")
show("case 1 body terms mm", f"{case_body:.1f}")
show("left for delays the body does not own mm", f"{D_CLEAR * 1000 - case_body:.1f}")
show("case 2 elementary delay mm", f"{case_elem:.1f}")
show("case 2 margin mm", f"{D_CLEAR * 1000 - case_elem:.1f}")
show("case 3 at 2 m/s mm", f"{case_fast:.1f}")
show("case 3 breach mm", f"{case_fast - D_CLEAR * 1000:.1f}")

head("Estimate: speed against delay (NM 2.2 step 5)")
extra = brake_fast - brake
show("braking at 2 m/s mm", f"{brake_fast:.0f}")
show("growth over drive limit", f"{brake_fast / brake - 1:.1%}")
show("braking added mm", f"{extra:.1f}")
show("equivalent delay at 1.5 m/s ms", f"{extra / V_DRIVE:.1f}")
show("each 100 ms of delay at 1.5 m/s mm", f"{V_DRIVE * 100:.0f}")


def v_ceiling(d_clear, tau, a):
    """Equation 3: the admissible speed for a clear distance."""
    d_avail = d_clear - LOC - MARGIN
    if d_avail <= 0:
        return 0.0
    return -a * tau + math.sqrt((a * tau) ** 2 + 2 * a * d_avail)


head("Velocity ceiling (equation 3)")
show("body-only ceiling at 1.10 m (tau 20 ms) m/s", f"{v_ceiling(D_CLEAR, T_ACT, A_BRAKE):.2f}")
show("elementary-delay ceiling at 1.10 m (tau 133.6 ms) m/s", f"{v_ceiling(D_CLEAR, TAU_ELEM, A_BRAKE):.2f}")
show("inspected floor ceiling at 1.10 m (a 1.18) m/s", f"{v_ceiling(D_CLEAR, TAU_ELEM, MU_INSPECTED * G):.2f}")

# smallest clear distance at which the drive limit is still admissible
def d_for_v(v, tau, a):
    return v * tau + v ** 2 / (2 * a) + LOC + MARGIN


show("clear distance needed for 1.5 m/s, tau 133.6 ms, a 2 (m)", f"{d_for_v(V_DRIVE, TAU_ELEM, A_BRAKE):.3f}")
show("clear distance needed for 1.5 m/s, inspected floor (m)", f"{d_for_v(V_DRIVE, TAU_ELEM, MU_INSPECTED * G):.3f}")

head("Credible deceleration ceilings (NM 2.3)")
mu_min_all = A_BRAKE / G
phi_rated = LOAD_PAIR / MASS
mu_min_rated = mu_min_all / phi_rated
phi_min_dry = mu_min_all / MU_DRY
a_dry = MU_DRY * G * phi_rated
a_inspected = MU_INSPECTED * G * 1.0
tau_wheel = KT_HUB * I_PEAK
force = 2 * tau_wheel / R_WHEEL
m_eff = MASS + 2 * HUB_MASS
a_torque = force / m_eff
a_rated = 2 * TAU_RATED_HUB / R_WHEEL / m_eff
cg_ratio = G / A_BRAKE
A_CEIL = {"dry": a_dry, "peak": a_torque, "inspected": a_inspected, "rated": a_rated}
show("mu needed, all weight on driven wheels", f"{mu_min_all:.3f}")
show("rated load share phi", f"{phi_rated:.3f}")
show("mu needed at rated share", f"{mu_min_rated:.3f}")
show("min driven-wheel share on dry floor", f"{phi_min_dry:.2f}")
show("dry floor ceiling m/s^2", f"{a_dry:.2f}")
show("inspected floor ceiling m/s^2", f"{a_inspected:.2f}")
show("torque per wheel N m", f"{tau_wheel:.0f}")
show("floor force N", f"{force:.1f}")
show("m_eff upper bound kg", f"{m_eff:.0f}")
show("peak-current ceiling m/s^2", f"{a_torque:.2f}")
show("margin over 2 m/s^2", f"{a_torque / A_BRAKE - 1:.1%}")
show("rated-current ceiling m/s^2", f"{a_rated:.2f}")
show("tip-over h_cg/d_lead max", f"{cg_ratio:.2f}")
show("rotating mass share of loaded mass", f"{2 * HUB_MASS / MASS:.1%}")


# =============================================================================
# Segment 3: actuators (PB 2.4, NM 2.3 step 4, NM 2.4)
# =============================================================================
head("Torque is current (PB 2.4.2)")
tau_e = L_LINE / R_LINE
t_rise = I_PEAK * L_LINE / V_DEPLETED
rpm = V_DRIVE / R_WHEEL * 60 / (2 * math.pi)
e_bemf = KE_HUB_RPM * rpm
show("electrical time constant ms", f"{tau_e * 1000:.1f}")
show("resistance-free rise to 25 A on 42 V ms", f"{t_rise * 1000:.1f}")
show("rise share of 20 ms brake onset", f"{t_rise / T_ACT:.1%}")
show("wheel speed at drive limit rpm", f"{rpm:.1f}")
show("back-EMF at drive limit V", f"{e_bemf:.1f}")
show("braking terminal voltage needed V", f"{e_bemf - I_PEAK * R_LINE:.1f}")
show("driving terminal voltage needed V", f"{e_bemf + I_PEAK * R_LINE:.1f}")
show("depleted pack above back-EMF V", f"{V_DEPLETED - e_bemf:.1f}")

head("Reflected inertia (PB 2.4.3)")
J_ROTOR = 5.6e-6               # kg m^2, illustrative rotor (PB 2.4.3)
J_LOAD = 8.5e-3                # kg m^2, illustrative link (PB 2.4.3)
N_GEAR = 100.0
j_ref = N_GEAR ** 2 * J_ROTOR
m_fly = 2 * j_ref / 0.10 ** 2
N_STAR = math.sqrt(J_LOAD / J_ROTOR)
x = N_GEAR / N_STAR
share100 = 2 * x / (1 + x ** 2)
show("reflected inertia at 100:1 kg m^2", f"{j_ref:.3f}")
show("equivalent 10 cm solid flywheel kg", f"{m_fly:.1f}")
show("matched ratio N*", f"{N_STAR:.1f}")
show("acceleration share at 100:1", f"{share100:.1%}")
show("acceleration lost at 100:1", f"{1 - share100:.1%}")

head("Heat against inertia (NM 2.4)")
KT_J, R_EFF = 0.187, 0.495     # TQ ILM 70x18: N m/A; effective loss resistance, ohm
J_IN = 7.34e-5                 # kg m^2, rotor + CSG-25 input (NM 2.4)
TAU_OUT, ETA = 87.0, 0.85      # N m max joint torque; gear efficiency (illustrative)
TAU_PEAK_MOTOR, TAU_CONT_MOTOR = 4.05, 1.25   # N m, TQ datasheet
km = KT_J / math.sqrt(R_EFF)
show("motor constant N m/sqrt(W)", f"{km:.3f}")
GEAR_ROWS = []
for n in (10, 50, 100):
    tm = TAU_OUT / (ETA * n)
    loss = (tm / km) ** 2
    jr = n ** 2 * J_IN
    GEAR_ROWS.append((n, tm, loss, jr))
    show(f"{n}:1 motor torque N m", f"{tm:.2f}")
    show(f"{n}:1 copper loss W", f"{loss:.1f}")
    show(f"{n}:1 reflected inertia kg m^2", f"{jr:.3f}")
    show(f"{n}:1 loss x inertia W kg m^2", f"{loss * jr:.2f}")
show("10:1 demand over peak", f"{GEAR_ROWS[0][1] / TAU_PEAK_MOTOR:.1f}x")
show("peak to continuous ratio", f"{TAU_PEAK_MOTOR / TAU_CONT_MOTOR:.1f}")

head("Contact (Systems Perspective 2.3)")
K_PLATE = 4.0e5                # N/m, steel plate under the latch
for v in (0.03, 0.10):
    rate = K_PLATE * v / 1000  # N per ms
    show(f"force ramp at {v} m/s N/ms", f"{rate:.0f}")
    show(f"time to the 100 N latch limit at {v} m/s ms", f"{100 / rate:.1f}")


# =============================================================================
# Segment 4: heat and power (PB 2.5, 2.6, NM 2.5 to 2.7)
# =============================================================================
head("Thermal burst (NM 2.5)")
I_RATED, C_TH, R_TH, T_AMB, T_LIM = 6.7, 120.0, 2.5, 40.0, 155.0
TAU_TH = R_TH * C_TH
t_eq_rated = T_AMB + I_RATED ** 2 * R_EFF * R_TH
I_BURST = 3 * I_RATED
P_BURST = I_BURST ** 2 * R_EFF
T_INF = T_AMB + P_BURST * R_TH


def t_survive(t0):
    return TAU_TH * math.log((T_INF - t0) / (T_INF - T_LIM))


t_cold, t_hot = t_survive(T_AMB), t_survive(t_eq_rated)
show("thermal time constant s", f"{TAU_TH:.0f}")
show("rated-current equilibrium C", f"{t_eq_rated:.1f}")
show("burst current A", f"{I_BURST:.1f}")
show("burst dissipation W", f"{P_BURST:.1f}")
show("burst asymptote C", f"{T_INF:.1f}")
show("cold-start horizon s", f"{t_cold:.1f}")
show("heat-soaked horizon s", f"{t_hot:.1f}")
show("heat-soaked shortfall", f"{1 - t_hot / t_cold:.1%}")
T_CEIL = 140.0
duty = (T_CEIL - T_AMB) / (P_BURST * R_TH)
show("duty-cycle screen", f"{duty:.1%}")
show("rest after a 3 s burst s", f"{3 * (1 - duty) / duty:.0f}")
t_on, period = 3.0, 3.0 / duty
peak_rise = P_BURST * R_TH * (1 - math.exp(-t_on / TAU_TH)) / (1 - math.exp(-period / TAU_TH))
show("periodic peak C", f"{T_AMB + peak_rise:.1f}")

head("Stall derating (NM 2.6)")
N_ELBOW, R_TH_STALL, ALPHA, T_REF = 50, 1.80, 0.00393, 20.0
tm = TAU_OUT / (ETA * N_ELBOW)
i_stall = tm / KT_J
p_cold = i_stall ** 2 * R_EFF
gain = i_stall ** 2 * R_EFF * R_TH_STALL * ALPHA
num = i_stall ** 2 * R_EFF * (1 + ALPHA * (T_AMB - T_REF)) * R_TH_STALL
rise = num / (1 - gain)
r_hot = R_EFF * (1 + ALPHA * (T_LIM - T_REF))
p_sus = (T_LIM - T_AMB) / R_TH_STALL
i_sus = math.sqrt(p_sus / r_hot)
tau_sus = ETA * N_ELBOW * KT_J * i_sus
show("stall current A", f"{i_stall:.1f}")
show("cold stall loss W", f"{p_cold:.1f}")
show("feedback factor", f"{gain:.4f}")
show("steady rise K", f"{rise:.1f}")
show("stall equilibrium C", f"{T_AMB + rise:.1f}")
show("sustainable loss W", f"{p_sus:.2f}")
show("hot resistance ohm", f"{r_hot:.3f}")
show("sustainable current A", f"{i_sus:.2f}")
show("sustainable hold N m", f"{tau_sus:.1f}")
show("derating", f"{1 - tau_sus / TAU_OUT:.1%}")
show("copper resistance rise 20 to 140 C", f"{ALPHA * (T_CEIL - T_REF):.1%}")

head("Bus sag (NM 2.7)")
sag = I_STEP * R_BUS
show("sustained sag V", f"{sag:.1f}")
show("full pack under step V", f"{V_FULL - sag:.1f}")
show("depleted pack under step V", f"{V_DEPLETED - sag:.1f}")
show("margin above lockout V", f"{V_DEPLETED - sag - V_UVLO:.1f}")
show("depleted pack to lockout V", f"{V_DEPLETED - V_UVLO:.0f}")
show("step that spends it A", f"{(V_DEPLETED - V_UVLO) / R_BUS:.0f}")
show("processor current A (60 W / 0.91 / 48 V)", "1.4 (chapter value)")

head("Regenerative surge (NM 2.7)")
e_regen = REGEN_SHARE * KE
v_final = math.sqrt(V_FULL ** 2 + 2 * e_regen / C_BUS)
e_to_clamp = 0.5 * C_BUS * (V_CLAMP ** 2 - V_FULL ** 2)
t_stop = V_DRIVE / A_BRAKE
p_peak = 2 * e_regen / t_stop
r_max = V_CLAMP ** 2 / p_peak
show("regenerated energy J", f"{e_regen:.1f}")
show("capacitor-only endpoint V", f"{v_final:.1f}")
show("energy capacitor absorbs to 60 V J", f"{e_to_clamp:.1f}")
show("stop duration ms", f"{t_stop * 1000:.0f}")
show("peak regenerated power W", f"{p_peak:.0f}")
show("largest chopper resistor ohm", f"{r_max:.2f}")

head("Unpowered states and permission rail (PB 2.6.2, 2.8.2)")
w_e = R_LINE / L_LINE
rpm_peak = w_e / POLE_PAIRS_HUB * 60 / (2 * math.pi)
show("phase-short torque peak, wheel rpm", f"{rpm_peak:.1f}")
show("drive-limit wheel rpm", f"{rpm:.1f}")
C_HOLD, V_HI, V_LO, P_PERM = 2.025, 21.5, 18.0, 70.0
e_hold = 0.5 * C_HOLD * (V_HI ** 2 - V_LO ** 2)
show("hold-up energy J", f"{e_hold:.0f}")
show("hold-up time at 70 W s", f"{e_hold / P_PERM:.1f}")
show("stop plus slow brake ms", f"{t_stop * 1000 + 80:.0f}")
show("DVFS added travel mm (11 ms at 1.5 m/s)", f"{V_DRIVE * 11:.1f}")


# =============================================================================
# Segment 5: limit record (PB 2.7)
# =============================================================================
head("Zero-failure bound (PB 2.7.2)")
show("rule of three, 3,000 stops", f"1 in {3000 / 3:.0f}")
show("stops for 1 in 10,000", f"{3 * 10000:,}")


if __name__ == "__main__":
    pass
