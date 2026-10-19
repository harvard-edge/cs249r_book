#!/usr/bin/env python3
"""Every derived number in Lecture 11 (Trajectory Planning), one block per slide.

Inputs carry their source. "TP" is the rendered chapter, Trajectory Planning
(Volume IV, Chapter 11). Every printed value is what the slide or its note
shows, with the label used there. make_figs.py imports from this file.

Run from this folder: python3 calcs.py
(The reflected rotor inertia comes from the MLSysIM registry when it is
importable; the fallback value is the same registry value, 0.1835 kg m^2.)
"""

import math
import sys
from pathlib import Path


def head(title):
    print(f"\n--- {title}")


def show(label, value):
    print(f"  {label}: {value}")


# =============================================================================
# Shared inputs
# =============================================================================
G = 9.80665                   # m/s^2, standard gravity (the chapter uses it)

# Arm (TP Sec. 11.2.1, 11.4.1; Franka Robotics 2023 datasheet values)
L1, L2 = 0.316, 0.384         # m, Panda planar two-link abstraction (TP Sec. 11.2.1)
V_TCP = 1.0                   # m/s, TCP speed limit (TP Sec. 11.2.1)
QDOT_MAX = 2.175              # rad/s, joints 1-4 velocity limit (Franka 2023)
QDDOT_MAX = 7.5               # rad/s^2, joint acceleration limit (Franka 2023; TP NM 11.3)
TAU_MAX = 87.0                # N m, joints 1-4 torque limit (Franka 2023)
TAU_RATE_MAX = 1000.0         # N m/s, torque-rate limit (Franka 2023; TP NM 11.2)
M_PAY = 3.0                   # kg, rated payload (Franka 2023; TP Sec. 11.4.1)
LAMBDA = 0.050                # m, illustrative DLS damping (TP Sec. 11.2.1)

# Reflected rotor inertia of the Panda-class stand-in joint (MLSysIM registry,
# Actuators.PandaClassJoint.CSG_25_50_ILM70x18: 50^2 x 7.34e-5 kg m^2)
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
    from mlsysim.actuators.registry import Actuators  # noqa: E402
    J_REFL = Actuators.PandaClassJoint.CSG_25_50_ILM70x18.reflected_inertia.to("kg*m**2").magnitude
    TAU_RATED = Actuators.PandaClassJoint.CSG_25_50_ILM70x18.rated_torque.to("N*m").magnitude
    TAU_PEAK = Actuators.PandaClassJoint.CSG_25_50_ILM70x18.peak_torque.to("N*m").magnitude
except Exception:  # registry not importable: same values, typed from it
    J_REFL, TAU_RATED, TAU_PEAK = 0.1835, 51.0, 127.0

# Timing (TP Sec. 11.2.2, 11.3.1)
T_AGE = 51.6                  # ms, capture-to-dispatch age (TP Sec. 11.3.1, from Lecture 8's waterfall)
T_DA = 63.3                   # ms, dispatch-to-admission allocation (TP Sec. 11.3.1)
T_INFER = 40.0                # ms, chunk policy p99 pass (illustrative, TP Sec. 11.1)
T_VALIDATE = 18.0             # ms, whole-plan validation allowance (TP Sec. 11.3.1)
T_RENEW = 50.0                # ms, desired admission interval (TP Sec. 11.2.2)
T_LEASE = 60.0                # ms, chunk lease (TP Sec. 11.2.2)
T_SETPOINT = 25.0             # ms, setpoint spacing (TP Sec. 11.3.2)
N_SETPOINTS = 16              # setpoints per chunk (TP Sec. 11.3.2)

# =============================================================================
# Slide: Predict, a p99 replanner across a ten-minute mission (hook, NM 11.1)
# =============================================================================
head("Hook: handoff deadlines across a mission (TP Napkin Math 11.1)")
MISSION_S, REPLAN_HZ = 600, 5          # illustrative
P_LATE = 0.01                          # p99 budget, definitional
MISSION_BUDGET = 1e-3                  # chosen
N_HANDOFF = MISSION_S * REPLAN_HZ
P_ALL = (1 - P_LATE) ** N_HANDOFF
P_TARGET = MISSION_BUDGET / N_HANDOFF
show("handoffs", f"{N_HANDOFF:,}")
show("expected late handoffs", f"{N_HANDOFF * P_LATE:.0f}")
show("P(every handoff on time)", f"{P_ALL:.1e}")
show("per-handoff design target", f"{P_TARGET:.1e}")
show("design quantile (percent)", f"{100 * (1 - P_TARGET):.5f}")
show("deeper than p99 by", f"{P_LATE / P_TARGET:.0e}")
show("orders of magnitude deeper", f"{math.log10(P_LATE / P_TARGET):.1f}")

# =============================================================================
# Slide: A 1 m/s tool speed can ask a joint for 37.8 rad/s (Jacobian)
# =============================================================================
head("Jacobian: weakest-direction joint rate for 1 m/s of tool speed (TP Sec. 11.2.1)")


def sing_vals(q2_deg, q1_deg=0.0):
    q1, q2 = math.radians(q1_deg), math.radians(q2_deg)
    j11 = -L1 * math.sin(q1) - L2 * math.sin(q1 + q2)
    j12 = -L2 * math.sin(q1 + q2)
    j21 = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
    j22 = L2 * math.cos(q1 + q2)
    # singular values of a 2x2 from J^T J
    a = j11 * j11 + j21 * j21
    b = j11 * j12 + j21 * j22
    d = j12 * j12 + j22 * j22
    tr, det = a + d, a * d - b * b
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    return math.sqrt(tr / 2 + disc), math.sqrt(max(tr / 2 - disc, 0.0))


def dls_rate(sigma):
    return V_TCP * sigma / (sigma ** 2 + LAMBDA ** 2)


JAC = {}
for q2 in (90, 10):
    smax, smin = sing_vals(q2)
    exact = V_TCP / smin
    JAC[q2] = dict(smin=smin, smax=smax, exact=exact, damped=dls_rate(smin),
                   realized=smin ** 2 / (smin ** 2 + LAMBDA ** 2))
    show(f"elbow {q2} deg: sigma_min (m/rad)", f"{smin:.4f}")
    show(f"elbow {q2} deg: exact joint-rate norm (rad/s)", f"{exact:.1f}")
    show(f"elbow {q2} deg: damped joint-rate norm (rad/s)", f"{dls_rate(smin):.1f}")
    show(f"elbow {q2} deg: realized tool motion (percent)", f"{100 * JAC[q2]['realized']:.1f}")
show("largest joint at 90 deg, at least (rad/s)", f"{JAC[90]['exact'] / math.sqrt(2):.1f}")
show("against the joint limit (rad/s)", f"{QDOT_MAX}")
show("planar reach (m)", f"{L1 + L2:.2f}")
show("DLS ceiling 1/(2 lambda) (rad/s per m/s)", f"{V_TCP / (2 * LAMBDA):.0f}")

# =============================================================================
# Slide: Replacement latency outlasts the renewal period (TP Sec. 11.3.1)
# =============================================================================
head("Replacement latency (TP equation 3)")
T_REPL = T_AGE + T_DA
show("replacement latency (ms)", f"{T_REPL:.1f}")
show("other dispatch-to-admission stages (ms)", f"{T_DA - T_INFER - T_VALIDATE:.1f}")
show("latency / renewal period", f"{T_REPL / T_RENEW:.2f}")
show("requests in flight at once, at least", f"{math.ceil(T_REPL / T_RENEW)}")
show("latency beyond the lease (ms)", f"{T_REPL - T_LEASE:.1f}")

# =============================================================================
# Slide: Predict far, execute little (published chunk configurations)
# =============================================================================
head("Horizon against executed prefix")
CHUNKS = [
    # (name, predicted steps, executed steps, source)
    ("Diffusion Policy, Push-T", 16, 6, "Chi et al. 2025 via TP Sec. 11.3.2"),
    ("pi0, 50 Hz robots", 50, 25, "Black et al. 2024 via TP table 4"),
    ("Running machine", N_SETPOINTS, int(T_RENEW / T_SETPOINT), "TP Sec. 11.3.2"),
]
for n, h, e, s in CHUNKS:
    show(f"{n}: executed share (percent)", f"{100 * e / h:.1f}")
show("running machine nominal trajectory (ms)", f"{N_SETPOINTS * T_SETPOINT:.0f}")
show("pi0 re-inference interval (s)", f"{25 / 50:.1f}")

# =============================================================================
# Slide: Plan from the seam, not from the request (future-state conditioning)
# =============================================================================
head("Stale anchor (TP Sec. 11.3.3)")
W_JOINT, T_DELAY = 0.50, 0.080          # rad/s, s (illustrative)
GAP = W_JOINT * T_DELAY
show("joint gap (rad)", f"{GAP:.3f}")
show("joint gap (deg)", f"{math.degrees(GAP):.2f}")
show("tool displacement at full reach (mm)", f"{1000 * (L1 + L2) * GAP:.0f}")

# =============================================================================
# Slide: Holding the payload already spends torque (TP Sec. 11.4.1)
# =============================================================================
head("One-joint dynamics (TP equation 7)")
REACH = L1 + L2
TAU_GRAV = M_PAY * G * REACH
J_EFF = M_PAY * REACH ** 2 + J_REFL
show("holding torque (N m)", f"{TAU_GRAV:.1f}")
show("share of 87 N m (percent)", f"{100 * TAU_GRAV / TAU_MAX:.1f}")
show("reflected rotor inertia (kg m^2)", f"{J_REFL:.4f}")
show("effective inertia with payload (kg m^2)", f"{J_EFF:.2f}")

# =============================================================================
# Slide: The seam spike (TP Napkin Math 11.2)
# =============================================================================
head("Seam spike (TP NM 11.2)")
DW, DT_TICK, T_BRIDGE = 0.2, 0.001, 0.050
TAU_STEP = J_REFL * DW / DT_TICK
RATE_STEP = TAU_STEP / DT_TICK
A_BRIDGE = 1.5 * DW / T_BRIDGE
JERK_BRIDGE = 6 * DW / T_BRIDGE ** 2
show("one-tick torque (N m)", f"{TAU_STEP:.1f}")
show("one-tick torque rate (N m/s)", f"{RATE_STEP:,.0f}")
show("multiple of the torque-rate limit", f"{RATE_STEP / TAU_RATE_MAX:.1f}")
show("bridge peak acceleration (rad/s^2)", f"{A_BRIDGE:.0f}")
show("bridge peak torque (N m)", f"{J_REFL * A_BRIDGE:.1f}")
show("bridge peak jerk (rad/s^3)", f"{JERK_BRIDGE:.0f}")
show("bridge peak torque rate (N m/s)", f"{J_REFL * JERK_BRIDGE:.1f}")
show("bridge torque rate share of limit (percent)", f"{100 * J_REFL * JERK_BRIDGE / TAU_RATE_MAX:.1f}")
show("torque peak ratio step/bridge", f"{TAU_STEP / (J_REFL * A_BRIDGE):.1f}")
show("one-tick step with payload inertia (N m)", f"{J_EFF * DW / DT_TICK:.0f}")
show("one-tick velocity ceiling on 127 N m (rad/s)", f"{TAU_PEAK * DT_TICK / J_REFL:.2f}")

head("Estimate: the bridge squeezed to 10 ms (Checkpoint 11.2)")
T_SHORT = 0.010
A_S = 1.5 * DW / T_SHORT
JK_S = 6 * DW / T_SHORT ** 2
show("10 ms bridge peak acceleration (rad/s^2)", f"{A_S:.0f}")
show("10 ms bridge peak torque (N m)", f"{J_REFL * A_S:.1f}")
show("10 ms bridge peak jerk (rad/s^3)", f"{JK_S:,.0f}")
show("10 ms bridge torque rate (N m/s)", f"{J_REFL * JK_S:,.0f}")
show("multiple of the torque-rate limit", f"{J_REFL * JK_S / TAU_RATE_MAX:.1f}")
show("shortest bridge inside the limit (ms)", f"{1000 * math.sqrt(6 * DW * J_REFL / TAU_RATE_MAX):.1f}")

# =============================================================================
# Slide: The fastest swing asks for a step the drive refuses (TP NM 11.3)
# =============================================================================
head("Time-optimal swing (TP NM 11.3)")
SWING = 0.5
show("torque-limited acceleration up (rad/s^2)", f"{(TAU_MAX - TAU_GRAV) / J_EFF:.1f}")
show("torque-limited braking (rad/s^2)", f"{(TAU_MAX + TAU_GRAV) / J_EFF:.1f}")
V_PEAK_SW = math.sqrt(QDDOT_MAX * SWING)
T_BANG = 2 * math.sqrt(SWING / QDDOT_MAX)
JERK_SW = TAU_RATE_MAX / J_EFF
RAMP = QDDOT_MAX / JERK_SW
T1 = (-RAMP + math.sqrt(RAMP ** 2 + 4 * SWING / QDDOT_MAX)) / 2
T_S = 2 * (T1 + RAMP)
show("peak speed (rad/s)", f"{V_PEAK_SW:.2f}")
show("bang-bang swing (ms)", f"{1000 * T_BANG:.1f}")
show("switch-point torque step (N m)", f"{2 * J_EFF * QDDOT_MAX:.1f}")
show("jerk bound from torque rate (rad/s^3)", f"{JERK_SW:.1f}")
show("jerk ramp a/j (ms)", f"{1000 * RAMP:.1f}")
show("S-curve swing (ms)", f"{1000 * T_S:.1f}")
show("penalty (ms)", f"{1000 * (T_S - T_BANG):.1f}")
show("penalty (percent)", f"{100 * (T_S - T_BANG) / T_BANG:.1f}")

# =============================================================================
# Stops: equation 16 polynomial and equation 17 S-curve
# =============================================================================


def poly_jerk(v, a):
    """Peak jerk of the equation 16 polynomial stop with peak deceleration a."""
    t = 1.5 * v / a
    return 6 * v / t ** 2


def scurve(v, a, j):
    """Distance (m) and duration (s) of the jerk-limited S-curve stop, eq. 17."""
    if v >= a * a / j:
        return v * v / (2 * a) + v * a / (2 * j), v / a + a / j
    return v * math.sqrt(v / j), 2 * math.sqrt(v / j)


# =============================================================================
# Slide: The stall runway (TP Napkin Math 11.4)
# =============================================================================
head("Stall runway, empty gripper (TP NM 11.4)")
A_ARM = 5.0                   # m/s^2, illustrative
D_CLEAR_ARM = 300.0           # mm, illustrative
T_TICK, T_BUS, T_ONSET = 1.0, 1.0, 20.0   # ms (derived, illustrative, illustrative)
J_ARM = poly_jerk(V_TCP, A_ARM)
T_PATH = T_LEASE + T_TICK + T_BUS + T_ONSET
D_PATH = V_TCP * T_PATH       # mm, since 1 m/s = 1 mm/ms
D_STOP_ARM, T_STOP_ARM = scurve(V_TCP, A_ARM, J_ARM)
D_STOP_ARM *= 1000
TOTAL_ARM = D_PATH + D_STOP_ARM
show("jerk bound (m/s^3)", f"{J_ARM:.1f}")
show("plateau threshold a^2/j (m/s)", f"{A_ARM ** 2 / J_ARM:.3f}")
show("lease path (ms)", f"{T_PATH:.0f}")
show("lease path travel (mm)", f"{D_PATH:.0f}")
show("S-curve stop (mm)", f"{D_STOP_ARM:.1f}")
show("S-curve stop (ms)", f"{1000 * T_STOP_ARM:.0f}")
show("constant-deceleration part (mm)", f"{1000 * V_TCP ** 2 / (2 * A_ARM):.0f}")
show("jerk-ramp part (mm)", f"{1000 * V_TCP * A_ARM / (2 * J_ARM):.1f}")
show("polynomial stop, same peaks (mm)", f"{1000 * 0.75 * V_TCP ** 2 / A_ARM:.0f}")
show("lease path + stop (mm)", f"{TOTAL_ARM:.1f}")
show("margin (mm)", f"{D_CLEAR_ARM - TOTAL_ARM:.1f}")
show("lease travel / stop (percent)", f"{100 * D_PATH / D_STOP_ARM:.1f}")
show("rest at (ms after last admission)", f"{T_PATH + 1000 * T_STOP_ARM:.0f}")

head("Stall runway, open mug (TP Sec. 11.5.2)")
R_MUG, FREEBOARD = 40.0, 10.0  # mm, illustrative
show("liquid rise at 5 m/s^2 (mm)", f"{R_MUG * A_ARM / G:.1f}")
A_MUG = G * FREEBOARD / R_MUG
J_MUG = poly_jerk(V_TCP, A_MUG)
D_STOP_MUG = 1000 * scurve(V_TCP, A_MUG, J_MUG)[0]
TOTAL_MUG = D_PATH + D_STOP_MUG
show("slosh deceleration bound (m/s^2)", f"{A_MUG:.2f}")
show("jerk at that peak (m/s^3)", f"{J_MUG:.1f}")
show("open-mug stop (mm)", f"{D_STOP_MUG:.1f}")
show("lease path + stop (mm)", f"{TOTAL_MUG:.1f}")
show("overrun (mm)", f"{TOTAL_MUG - D_CLEAR_ARM:.1f}")
lo, hi = 0.1, 1.0
for _ in range(60):
    mid = (lo + hi) / 2
    need = mid * T_PATH + 1000 * scurve(mid, A_MUG, J_MUG)[0]
    lo, hi = (mid, hi) if need <= D_CLEAR_ARM else (lo, mid)
V_MUG_MAX = lo
show("highest speed that fits (m/s)", f"{V_MUG_MAX:.2f}")

# =============================================================================
# Slide: The record's own deadlines (alternate longer-lease schedule)
# =============================================================================
head("Alternate schedule (TP Sec. 11.3.2, 11.5.2)")
T_REQ, T_Q, T_SLACK, T_RESP = 60.0, 160.0, 20.0, 22.0   # ms, chosen/illustrative
T_COMMIT = 300.0
T_PREF = T_REQ + T_Q + T_SLACK
T_CUTOFF = T_COMMIT - T_RESP
T_END = T_COMMIT + 1000 * T_STOP_ARM
CRUISE_ALT = V_TCP * (T_COMMIT - T_PREF)
TOTAL_ALT = CRUISE_ALT + D_STOP_ARM
show("preferred admission (ms)", f"{T_PREF:.0f}")
show("full reserve (ms)", f"{T_SLACK + T_RESP:.0f}")
show("minimum stop onset (ms)", f"{T_PREF + T_RESP:.0f}")
show("admission cutoff (ms)", f"{T_CUTOFF:.0f}")
show("rest at (ms)", f"{T_END:.0f}")
show("nominal end (ms)", f"{N_SETPOINTS * T_SETPOINT:.0f}")
show("setpoint periods in window", f"{T_END / T_SETPOINT:.0f}")
show("cruise charged 240->300 (mm)", f"{CRUISE_ALT:.0f}")
show("used (mm)", f"{TOTAL_ALT:.1f}")
show("margin (mm)", f"{D_CLEAR_ARM - TOTAL_ALT:.1f}")
show("200 ms late: breach (mm)", f"{TOTAL_ALT + 200 * V_TCP - D_CLEAR_ARM:.1f}")

# =============================================================================
# Slide: The jerk-limited stop at the rack end (TP Sec. 11.5.4)
# =============================================================================
head("Base stop at 1.5 m/s (TP Sec. 11.5.4)")
V_BASE, A_BASE, D_CLEAR_BASE = 1.5, 2.0, 1100.0   # illustrative card values
BUDGET_BEFORE = 912.9          # mm, running budget after Lecture 8's waterfall (TP Sec. 11.5.4)
J_BASE = poly_jerk(V_BASE, A_BASE)
D_CONST = 1000 * V_BASE ** 2 / (2 * A_BASE)
T_CONST = 1000 * V_BASE / A_BASE
D_SC, T_SC = scurve(V_BASE, A_BASE, J_BASE)
D_SC, T_SC = 1000 * D_SC, 1000 * T_SC
D_POLY = 1000 * 0.75 * V_BASE ** 2 / A_BASE
T_POLY = 1000 * 1.5 * V_BASE / A_BASE
show("jerk bound (m/s^3)", f"{J_BASE:.2f}")
show("plateau speed (m/s)", f"{A_BASE ** 2 / J_BASE:.4f}")
show("constant deceleration: mm, ms", f"{D_CONST:.1f}, {T_CONST:.0f}")
show("S-curve: mm, ms", f"{D_SC:.2f}, {T_SC:.2f}")
show("polynomial: mm, ms", f"{D_POLY:.2f}, {T_POLY:.0f}")
show("S-curve adds (mm)", f"{D_SC - D_CONST:.2f}")
show("equivalent cruise (ms)", f"{(D_SC - D_CONST) / V_BASE:.1f}")
show("budget before stop term (mm)", f"{BUDGET_BEFORE - D_CONST:.1f}")
show("running total with S-curve (mm)", f"{BUDGET_BEFORE + D_SC - D_CONST:.2f}")
show("overrun (mm)", f"{BUDGET_BEFORE + D_SC - D_CONST - D_CLEAR_BASE:.2f}")
show("running total with polynomial (mm)", f"{BUDGET_BEFORE + D_POLY - D_CONST:.2f}")

# =============================================================================
# Slide: DWA admits speeds the full budget refuses (TP Sec. 11.6.1)
# =============================================================================
head("Dynamic window (TP Sec. 11.6.1)")
show("window half-width (m/s)", f"{A_BASE * T_RENEW / 1000:.1f}")
show("textbook admissible speed (m/s)", f"{math.sqrt(2 * D_CLEAR_BASE / 1000 * A_BASE):.2f}")

# =============================================================================
# Slide: Generators, reported compute (TP table 4)
# =============================================================================
head("Generator timing")
RUCKIG_MEAN, RUCKIG_WORST = 19.8e-3, 123e-3   # ms, Berscheid and Kroger 2021 (TP table 4)
show("Ruckig worst case share of a 1 ms tick (percent)", f"{100 * RUCKIG_WORST / 1.0:.1f}")
show("chunk pass / Ruckig worst case", f"{T_INFER / RUCKIG_WORST:.0f}")
