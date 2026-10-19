#!/usr/bin/env python3
"""Every derived number in Lecture 12 (Safety Enforcement), one block per slide.

Inputs carry their source. "SE" is the rendered chapter, Safety Enforcement
(Volume IV, Chapter 12). Outside sources are named where they are used.
Every printed value is what the slide or its note shows, with the label used
there. make_figs.py imports from this file.

Run from this folder: python3 calcs.py
"""

import math

G = 9.80665  # standard gravity, m/s^2


def head(title):
    print(f"\n--- {title}")


def show(label, value):
    print(f"  {label}: {value}")


# =============================================================================
# Shared inputs: the running machine at the rack end (SE Napkin Math 12.1, table 1)
# =============================================================================
D_CLEAR = 1.100        # m, clear distance at the rack end (illustrative)
LOC = 0.050            # m, localization bound (illustrative)
PROT = 0.100           # m, protective clearance (chosen)
TRACK = 0.040          # m, tracking bound (illustrative qualification target)
A_BRAKE = 2.0          # m/s^2, credible loaded deceleration, dry floor (illustrative)
V_DRIVE = 1.5          # m/s, drive limit (illustrative)
V_AISLE = 1.3          # m/s, candidate aisle speed (SE 12.2.1)
T_ACT = 0.020          # s, brake onset (SE table 1 row 3: 30 mm at 1.5 m/s)
T_LEASE_PATH = 0.062   # s, lease 60 ms + one tick + one bus cycle (SE table 1 row 4)
T_LEASE = 0.060        # s, chunk lease (SE 12.5)
T_AGE = 0.0516         # s, observation age (SE table 1 row 5)
T_DISPATCH = 0.0633    # s, dispatch-to-admission allocation (SE 12.2.1, owned by Lecture 13)
# Jerk limit: SE stipulates 7.11 m/s^3. Lecture 11's S-curve from 1.5 m/s takes
# 1,031.25 ms against 750 ms at constant deceleration (SE 12.10), so
# a/J = 0.28125 s and J = 2 / 0.28125 (derived; prints as 7.11).
T_SCURVE_DRIVE = 1.03125
T_CONST_DRIVE = V_DRIVE / A_BRAKE
J = A_BRAKE / (T_SCURVE_DRIVE - T_CONST_DRIVE)
T_DELAY = T_ACT + T_LEASE_PATH + T_AGE   # pre-brake delay, s


def stop_dist(v, a=A_BRAKE, j=J):
    """Jerk-limited resident stop distance (SE equation 11.17 form), m."""
    if v >= a * a / j:
        return v * v / (2 * a) + v * a / (2 * j)
    t = 2 * math.sqrt(v / j)      # triangular deceleration, peak never reached
    return v * t / 2


def stop_time(v, a=A_BRAKE, j=J):
    if v >= a * a / j:
        return v / a + a / j
    return 2 * math.sqrt(v / j)


def rows(v):
    """Table 1 rows 1 to 7 at speed v, in mm."""
    return [
        ("Braking", v * v / (2 * A_BRAKE) * 1e3),
        ("Fixed allowances", (LOC + PROT) * 1e3),
        ("Brake onset", v * T_ACT * 1e3),
        ("Lease path", v * T_LEASE_PATH * 1e3),
        ("Observation age", v * T_AGE * 1e3),
        ("Jerk ramps", v * A_BRAKE / (2 * J) * 1e3),
        ("Tracking inset", TRACK * 1e3),
    ]


def ceiling(delay, a=A_BRAKE, avail=D_CLEAR - LOC - PROT - TRACK, j=J, walker=0.0):
    """Highest speed whose stop subtotal fits the clear distance (root search)."""
    lo, hi = 0.0, 5.0
    for _ in range(200):
        v = (lo + hi) / 2
        need = v * delay + stop_dist(v, a, j) + walker * (delay + stop_time(v, a, j))
        if need <= avail:
            lo = v
        else:
            hi = v
    return lo


# =============================================================================
# Slide: Recall (Lecture 11): constant-deceleration stop against the S-curve
# =============================================================================
head("Recall: Lecture 11 stop from 1.5 m/s")
show("constant deceleration (ms)", f"{T_CONST_DRIVE * 1e3:.0f}")
show("S-curve (ms)", f"{T_SCURVE_DRIVE * 1e3:,.2f}")
show("jerk limit J (m/s^3)", f"{J:.2f}")
show("a/(2J) (ms)", f"{A_BRAKE / (2 * J) * 1e3:.1f}")

# =============================================================================
# Slide: the finished budget overruns at the drive limit (budget build)
# =============================================================================
head("Budget at the drive limit and at the aisle speed")
run = 0.0
for name, mm in rows(V_DRIVE):
    run += mm
    show(f"{name} at 1.5 (mm) / running total", f"{mm:,.2f} / {run:,.2f}")
TOT_DRIVE = sum(mm for _, mm in rows(V_DRIVE))
SUB5_DRIVE = sum(mm for _, mm in rows(V_DRIVE)[:5])
SUB6_DRIVE = sum(mm for _, mm in rows(V_DRIVE)[:6])
show("rows 1-5 total at 1.5 (mm)", f"{SUB5_DRIVE:,.1f}")
show("rows 1-6 overrun at 1.5 (mm)", f"{SUB6_DRIVE - D_CLEAR * 1e3:,.2f}")
show("rows 1-7 overrun at 1.5 (mm)", f"{TOT_DRIVE - D_CLEAR * 1e3:,.2f}")
for name, mm in rows(V_AISLE):
    show(f"{name} at 1.3 (mm)", f"{mm:,.2f}")
TOT_AISLE = sum(mm for _, mm in rows(V_AISLE))
SPARE = D_CLEAR * 1e3 - TOT_AISLE
show("total at 1.3 (mm)", f"{TOT_AISLE:,.2f}")
show("spare at 1.3 (mm)", f"{SPARE:,.2f}")
show("wear doubles tracking error: unbudgeted (mm)", f"{(2 * TRACK - TRACK) * 1e3:.0f}")

# =============================================================================
# Worked example: the ceiling the budget sets (SE Napkin Math 12.1)
# =============================================================================
head("Napkin Math 12.1")
avail = D_CLEAR - LOC - PROT - TRACK
t_eff = T_DELAY + A_BRAKE / (2 * J)
a_t = A_BRAKE * t_eff
root = math.sqrt(a_t ** 2 + 2 * A_BRAKE * avail)
V_CEIL = -a_t + root
show("pre-brake delay (ms)", f"{T_DELAY * 1e3:.1f}")
show("available braking distance (mm)", f"{avail * 1e3:,.0f}")
show("effective delay (ms)", f"{t_eff * 1e3:.1f}")
show("a * t_eff (m/s)", f"{a_t:.4f}")
show("(a t)^2", f"{a_t ** 2:.4f}")
show("2 a D", f"{2 * A_BRAKE * avail:.2f}")
show("root (m/s)", f"{root:.4f}")
show("ceiling (m/s)", f"{V_CEIL:.2f}")
SPARE_MS = SPARE / V_AISLE
show("spare as delay at 1.3 (ms)", f"{SPARE_MS:.1f}")
DISP_MM = V_AISLE * T_DISPATCH * 1e3
show("dispatch-to-admission travel (mm)", f"{DISP_MM:.1f}")
RESERVE = SPARE - DISP_MM
show("reserve after dispatch (mm)", f"{RESERVE:.1f}")
show("reserve as time (ms)", f"{RESERVE / V_AISLE:.1f}")
show("ceiling with dispatch charged (m/s)", f"{ceiling(T_DELAY + T_DISPATCH):.2f}")
show("one ms of delay at 1.3 (mm)", f"{V_AISLE * 1e-3 * 1e3:.1f}")
show("10 ms more delay lowers ceiling by (mm/s)",
     f"{(V_CEIL - ceiling(T_DELAY + 0.010)) * 1e3:.0f}")

# =============================================================================
# Activity: a 200 ms host stall instead of the 60 ms lease (SE 12.2.1)
# =============================================================================
head("Estimate: host-enforced lease, 200 ms stall")
STALL = 0.200
extra = (STALL - T_LEASE) * V_AISLE * 1e3
show("extra unbraked travel (mm)", f"{extra:.0f}")
show("overrun past the spare (mm)", f"{extra - SPARE:.2f}")
show("integrator unwind 150 ms at 1.3 (mm)", f"{0.150 * V_AISLE * 1e3:.0f}")
show("unwind overrun past spare (mm)", f"{0.150 * V_AISLE * 1e3 - SPARE:.2f}")

# =============================================================================
# Slide: three ceilings (SE 12.2.1, figure 3)
# =============================================================================
head("Three ceilings, before dispatch allocation")
MU_INSP = 0.12
A_INSP = MU_INSP * G
V_WALK = 1.6       # m/s, ISO 13855 approach speed (SE 12.2.1)
show("dry floor (m/s)", f"{ceiling(T_DELAY):.2f}")
show("inspected floor decel (m/s^2)", f"{A_INSP:.3f}")
show("inspected floor mu=0.12 (m/s)", f"{ceiling(T_DELAY, a=A_INSP):.2f}")
V_WALKER = ceiling(T_DELAY, walker=V_WALK)
show("walking person (m/s)", f"{V_WALKER:.2f}")
show("min mu for 2 m/s^2", f"{A_BRAKE / G:.3f}")
show("jerk threshold a^2/J (m/s)", f"{A_BRAKE ** 2 / J:.2f}")
show("stop time from walker ceiling (ms)", f"{stop_time(V_WALKER) * 1e3:.0f}")
show("base stop travel at walker ceiling (mm)", f"{stop_dist(V_WALKER) * 1e3:.0f}")
show("person travel during that stop (mm)", f"{V_WALK * stop_time(V_WALKER) * 1e3:.0f}")
show("dry, dispatch charged (m/s)", f"{ceiling(T_DELAY + T_DISPATCH):.2f}")
show("inspected, dispatch charged (m/s)", f"{ceiling(T_DELAY + T_DISPATCH, a=A_INSP):.2f}")
show("walker, dispatch charged (m/s)", f"{ceiling(T_DELAY + T_DISPATCH, walker=V_WALK):.2f}")

# =============================================================================
# Slide: the scanner path (SICK nanoScan3 NANS3-CAAZ30AA1 datasheet, 2024-12-17)
# =============================================================================
head("Certified scanner path")
SCAN_RESP = 0.085   # s, response time (datasheet)
show("scanner response travel at 1.3 (mm)", f"{SCAN_RESP * V_AISLE * 1e3:.1f}")
show("software pre-brake delay travel at 1.3 (mm)", f"{T_DELAY * V_AISLE * 1e3:.2f}")
show("scanner response travel at 1.5 (mm)", f"{SCAN_RESP * V_DRIVE * 1e3:.1f}")

# =============================================================================
# Slide: the gravity braking tax (SE Napkin Math 12.2)
# Franka Research 3 datasheet R02212 2.3: A1-A4 +/-87 Nm, A5-A7 +/-12 Nm, ~18.3 kg
# =============================================================================
head("Napkin Math 12.2")
W0 = 1.2          # rad/s (illustrative)
LEVER = 0.80      # m (illustrative)
INERTIA = 3.0     # kg m^2 (illustrative)
TAU_MAX = 87.0    # N m (datasheet)
TAU_BIAS = 0.75 * TAU_MAX   # N m (illustrative, 75 percent of peak)


def tool_stop_mm(alpha):
    return W0 ** 2 / (2 * alpha) * LEVER * 1e3


A_FREE = TAU_MAX / INERTIA
A_DOWN = (TAU_MAX - TAU_BIAS) / INERTIA
A_UP = (TAU_MAX + TAU_BIAS) / INERTIA
show("bias torque (N m)", f"{TAU_BIAS:.2f}")
show("no bias: decel / angle deg / tool mm",
     f"{A_FREE:.2f} / {math.degrees(W0**2/(2*A_FREE)):.2f} / {tool_stop_mm(A_FREE):.2f}")
show("descending: torque left / decel / tool mm",
     f"{TAU_MAX - TAU_BIAS:.2f} / {A_DOWN:.2f} / {tool_stop_mm(A_DOWN):.2f}")
show("ascending: torque / decel / tool mm",
     f"{TAU_MAX + TAU_BIAS:.2f} / {A_UP:.2f} / {tool_stop_mm(A_UP):.2f}")
show("descending / free", f"{tool_stop_mm(A_DOWN) / tool_stop_mm(A_FREE):.1f}")
show("unbudgeted overrun (mm)", f"{tool_stop_mm(A_DOWN) - tool_stop_mm(A_FREE):.2f}")
show("joint stop time descending (ms)", f"{W0 / A_DOWN * 1e3:.0f}")
show("FR3 half mass + payload (kg)", f"{18.3 / 2 + 3:.1f}")

# =============================================================================
# Slide: the barrier's torque demand (SE 12.4.1, equation 6)
# =============================================================================
head("Kinetic barrier torque demand")
GAMMA = 10.0


def barrier_brake_torque(clear_m, w=W0):
    theta = clear_m / LEVER
    h = theta - w * w / (2 * A_DOWN)
    # equation 6: tau <= -bias + I a_net (gamma h / w - 1); braking demand is its negative
    return -(-TAU_BIAS + INERTIA * A_DOWN * (GAMMA * h / w - 1)), theta, h


for c in (0.110, tool_stop_mm(A_DOWN) / 1e3, 0.070):
    tq, th, h = barrier_brake_torque(c)
    show(f"clearance {c * 1e3:.2f} mm: theta / h / braking torque",
         f"{th:.4f} / {h:.4f} / {tq:.1f}")

# =============================================================================
# Activity + slide: projection against per-axis clamp (SE 12.6.1)
# =============================================================================
head("Two-axis projection")
u = (3.5, 3.0)
a = (2.0, 1.0)
b = 5.0
au = a[0] * u[0] + a[1] * u[1]
lam = max(0.0, (au - b) / (a[0] ** 2 + a[1] ** 2))
ustar = (u[0] - lam * a[0], u[1] - lam * a[1])
clamp = (2.0, 3.0)
show("a.u_nom", f"{au:g}")
show("breach", f"{au - b:g}")
show("lambda*", f"{lam:g}")
show("u*", f"({ustar[0]:g}, {ustar[1]:g})")
show("a.u*", f"{a[0] * ustar[0] + a[1] * ustar[1]:g}")
show("intervention norm", f"{math.hypot(u[0] - ustar[0], u[1] - ustar[1]):.3f}")
show("a.clamp", f"{a[0] * clamp[0] + a[1] * clamp[1]:g}")
show("nominal angle (deg)", f"{math.degrees(math.atan2(u[1], u[0])):.1f}")
show("clamp angle (deg)", f"{math.degrees(math.atan2(clamp[1], clamp[0])):.1f}")
show("projected angle (deg)", f"{math.degrees(math.atan2(ustar[1], ustar[0])):.1f}")
show("latch speed projection lambda (1-D, a=1)", f"{0.10 - 0.03:.2f}")

# =============================================================================
# Slide: force rises at the latch (SE 12.6.2, equations 11 and 12)
# =============================================================================
head("Latch")
K_PLATE = 4.0e5     # N/m (SE)
TRIP = 15.0         # N
T_LOOP = 0.002      # s
F_LATCH = 100.0     # N


def latch(v):
    rate = K_PLATE * v
    f_on = TRIP + rate * T_LOOP
    room = (F_LATCH - f_on) / K_PLATE
    acc = v * v / (2 * room) if room > 0 else float("inf")
    return rate, f_on, room, acc, v / acc


for v in (0.03, 0.10):
    rate, f_on, room, acc, t = latch(v)
    show(f"v={v}: rate N/ms / force at brake / room mm / decel / stop ms",
         f"{rate / 1e3:.0f} / {f_on:.0f} / {room * 1e3:.4f} / {acc:.2f} / {t * 1e3:.1f}")
show("0.10: time from first contact to 100 N, unbraked (ms)", f"{F_LATCH / (K_PLATE * 0.10) * 1e3:.1f}")
show("0.03: time from first contact to 100 N, unbraked (ms)", f"{F_LATCH / (K_PLATE * 0.03) * 1e3:.1f}")

# =============================================================================
# Activity: the clamped-hand penalty (SE Napkin Math 12.3)
# =============================================================================
head("Napkin Math 12.3")
M_ROBOT = 12.0      # kg (SE, half-arm mass plus payload)
M_HAND = 0.6        # kg (ISO/TS 15066 body model, via SE and Kirschner et al. 2022)
K_HAND = 75e3       # N/m (75 N/mm)
V_HAND = 0.10       # m/s (chosen)
F_CRIT = 50.0       # N (chosen)
mu = 1 / (1 / M_HAND + 1 / M_ROBOT)
s_free = math.sqrt(K_HAND * mu)
s_clamp = math.sqrt(K_HAND * M_ROBOT)
show("reduced mass (kg)", f"{mu:.3f}")
show("free stiffness term (N per m/s)", f"{s_free:.0f}")
show("free force at 0.10 (N)", f"{V_HAND * s_free:.1f}")
show("free speed ceiling (m/s)", f"{F_CRIT / s_free:.2f}")
show("clamped stiffness term (N per m/s)", f"{s_clamp:.0f}")
show("clamped force at 0.10 (N)", f"{V_HAND * s_clamp:.1f}")
show("clamped speed ceiling (m/s)", f"{F_CRIT / s_clamp:.3f}")
show("clamp multiplier", f"{s_clamp / s_free:.1f}")
show("free force at 1 m/s (N) / multiple of 50", f"{s_free:.0f} / {s_free / F_CRIT:.1f}")

# =============================================================================
# Slide: measured clamped-hand forces (Kirschner et al. 2022, arXiv 2203.02706, Table II)
# Interpretation B1: TS hand values, constrained hand (m_h -> infinity),
# speeds set for F_max = 280 N; forces measured with a Pilz PRMS2 device.
# =============================================================================
head("Kirschner et al. Table II, hand, B1")
KIRSCHNER = {  # robot: (v_B1 m/s, F at C-position N, F at N-position N)
    "UR10e": (0.28, 316, 230),
    "Franka (FE)": (0.40, 289, 228),
    "KUKA LBR iiwa 14": (0.36, 368, 279),
    "Techman TM5": (0.28, 427, 292),
}
F_TS = 280
allf = [f for _, c, n in KIRSCHNER.values() for f in (c, n)]
show("measured range (N)", f"{min(allf)} to {max(allf)}")
show("largest over 280 (N)", f"{max(allf) - F_TS}")
show("largest / 280", f"{max(allf) / F_TS:.2f}")
show("FR3 clamped speed from formula, 280 N, m_r=5.54 (m/s)",
     f"{F_TS / math.sqrt(K_HAND * 5.54):.2f}")

# =============================================================================
# Slide: speed and separation monitoring (SE 12.8.3)
# =============================================================================
head("Protective separation distance, descending arm")
T_R = T_DELAY       # 133.6 ms vision-path latency
T_S = W0 / A_DOWN   # s, gravity-biased joint stop
V_TCP = 0.96        # m/s
S_STOP = tool_stop_mm(A_DOWN)
s_h = V_WALK * (T_R + T_S) * 1e3
s_r = V_TCP * T_R * 1e3
show("stop duration (ms)", f"{T_S * 1e3:.0f}")
show("person travel (mm)", f"{s_h:.0f}")
show("robot reaction travel (mm)", f"{s_r:.0f}")
show("robot stop travel (mm)", f"{S_STOP:.0f}")
show("sum (mm)", f"{s_h + s_r + S_STOP:.0f}")
show("person share", f"{s_h / (s_h + s_r + S_STOP):.0%}")

# =============================================================================
# Slide: the cycle budget and the hold (SE 12.6.3, 12.6.4)
# =============================================================================
head("Cycle budget")
STAGES = [("State in", 30), ("Lease check", 15), ("Tracking", 10),
          ("QP solve", 150), ("Output", 15), ("Reserve", 30)]
show("declared WCET (us)", f"{sum(s for _, s in STAGES)}")
show("headroom to 400 us deadline (us)", f"{400 - sum(s for _, s in STAGES)}")
show("QP share of WCET", f"{150 / sum(s for _, s in STAGES):.0%}")
zoh = V_DRIVE * 1e-3 + 0.5 * 2.0 * 1e-6
show("zero-order-hold drift at 1.5 m/s, 2 m/s^2 (mm)", f"{zoh * 1e3:.3f}")
show("missed tick at 1.3 (mm)", f"{V_AISLE * 1e-3 * 1e3:.1f}")

# =============================================================================
# Slide + activity: STOP against STO from the aisle speed (SE 12.8.1)
# =============================================================================
head("Resident stop from 1.3 m/s")
show("STOP duration (ms)", f"{stop_time(V_AISLE) * 1e3:.2f}")
show("STOP distance (mm)", f"{stop_dist(V_AISLE) * 1e3:.2f}")
show("STOP from 1.5 duration / distance", f"{stop_time(V_DRIVE) * 1e3:.2f} / {stop_dist(V_DRIVE) * 1e3:.2f}")
show("lease stall onset bound (ms)", f"{(T_LEASE_PATH + T_ACT) * 1e3:.0f}")
show("coast at 1.3 for 80 ms brake set (mm)", f"{V_AISLE * 0.080 * 1e3:.0f}")
show("direct-control lease row at 1.3: 2 ms (mm)", f"{V_AISLE * 0.002 * 1e3:.1f}")

# =============================================================================
# Slide: Cruise, October 2023 (SE War Story 12.2)
# =============================================================================
head("Cruise")
show("20 ft in m", f"{20 * 0.3048:.1f}")
show("7.7 mph in m/s", f"{7.7 * 0.44704:.1f}")
