#!/usr/bin/env python3
"""Every derived number in Lecture 10 (Grounded Intent), Volume IV.

One block per slide, in deck order. Inputs carry their source as a comment
("ch" is the rendered chapter, Grounded Intent; "RM" and "AISLE" are the
running machine and its aisle scenario in the MLSysIM registry, which the
chapter's LEGO cells print from). Each printed line is a value the slide or
its note shows, under the label used there. make_figs.py imports from this
file so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Embodied  # noqa: E402

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


def mag(q, unit):
    return q.to(unit).magnitude


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
# AutoRT (Ahn et al. 2024, arXiv 2401.12963), as quoted in ch War Story 10.1
AUTORT_PROPOSED = 259
AUTORT_ACCEPTABLE = 228
AUTORT_KEPT = 214
AUTORT_KEPT_ACCEPTABLE = 200
AUTORT_REJECTED_UNSUITABLE = 17

# Running machine and aisle scenario (MLSysIM registry; ch prints these)
R_TOL = mag(AISLE.grasp_tolerance, "mm")          # 15 mm grasp tolerance
E0 = mag(AISLE.grasp_initial_error, "mm")         # 3 mm initial handle error
V_BELT = mag(AISLE.v_conveyor, "mm/s")            # 200 mm/s conveyor
A_SLIP = mag(AISLE.a_conveyor_slip, "mm/s**2")    # 500 mm/s^2 residual slip
T_TICK = mag(AISLE.t_tick, "ms")                  # 1 ms permission tick
T_BUS = mag(AISLE.t_bus, "ms")                    # 1 ms fieldbus cycle
T_BRAKE = mag(AISLE.t_brake_onset, "ms")          # 20 ms brake onset
V_HUMAN = mag(AISLE.v_human_approach, "m/s")      # 1.6 m/s, ISO 13855 design value
AISLE_W = mag(AISLE.aisle_width, "m")             # 1.30 m aisle
K_LATCH = mag(AISLE.k_latch, "N/mm")              # 400 N/mm plate stiffness
V_LATCH = mag(AISLE.v_latch_approach, "mm/s")     # 30 mm/s guarded approach
F_TRIP = mag(AISLE.f_latch_tripwire, "N")         # 15 N tripwire
F_LIMIT = mag(AISLE.f_latch_limit, "N")           # 100 N latch limit
T_CONTACT = mag(AISLE.t_contact_response, "ms")   # 2 ms contact loop
T_CHUNK_LEASE = mag(AISLE.t_lease, "ms")          # 60 ms chunk lease (Lecture 4)
T_INFER = mag(RM.intent_inference_latency, "ms")  # 335.6 ms intent inference (derived in the book)
F_INTENT = mag(RM.intent_rate, "Hz")              # 2 Hz intent rate
V_TCP = mag(RM.tcp_speed_limit, "mm/s")           # 1,000 mm/s TCP speed ceiling

# ch scenario inputs (illustrative, Napkin Math 10.1 to 10.3 and Sec. 10.1, 10.3)
D_GOAL_IN, D_GOAL_OUT = 10.0, 20.0     # mm, terminal TCP offsets along x
SCORE_MUG, SCORE_JAR, SCORE_JAR_LOW = 0.62, 0.48, 0.10   # raw box scores
N_CAL, ALPHA, QHAT = 200, 0.10, 0.60   # calibration scenes, miscoverage, assumed score
SIGMA = (4.0, 5.0, 12.0)               # mm, wrist-camera principal std devs
CHI2_3DOF_95 = 7.815                   # ch Sec. 10.3.3 (chi-square, 3 dof, 0.95)
T_STALL = 200.0                        # ms, host or contact stall
V_DRIFT_SLOW = 20.0                    # mm/s, settling drift
V_DRIFT_CHECK = 50.0                   # mm/s, Checkpoint 10.1
D_TARGET = 300.0                       # mm, gripper-to-handle distance at proposal
T_FORECAST = 2.0                       # s, multi-second planning horizon

# ---------------------------------------------------------------------------
head("Hook: AutoRT filter (War Story 10.1)")
unsuitable = AUTORT_PROPOSED - AUTORT_ACCEPTABLE
passed = AUTORT_KEPT - AUTORT_KEPT_ACCEPTABLE
show("unsuitable proposed", unsuitable)
show("acceptable share of proposed (%)", f"{100 * AUTORT_ACCEPTABLE / AUTORT_PROPOSED:.0f}")
show("acceptable share of kept (%)", f"{100 * AUTORT_KEPT_ACCEPTABLE / AUTORT_KEPT:.0f}")
show("unsuitable passed by the filter", passed)
show("rejected + passed (check = unsuitable)", AUTORT_REJECTED_UNSUITABLE + passed)
show("filter recall (%)", f"{100 * AUTORT_REJECTED_UNSUITABLE / unsuitable:.0f}")
show("share of unsuitable passed (%)", f"{100 * passed / unsuitable:.0f}")
show("tasks the filter removed", AUTORT_PROPOSED - AUTORT_KEPT)
show("acceptable tasks removed", AUTORT_ACCEPTABLE - AUTORT_KEPT_ACCEPTABLE)

# ---------------------------------------------------------------------------
head("Two-speed brain recall (Lecture 3)")
t_period = 1000.0 / F_INTENT
show("intent inference (ms)", f"{T_INFER:.1f} -> {T_INFER:.0f}")
show("intent period (ms)", f"{t_period:.0f}")

# ---------------------------------------------------------------------------
head("Goal set is a specification (Sec. 10.1)")
norm_in = D_GOAL_IN / R_TOL
norm_out = D_GOAL_OUT / R_TOL
show("normalized distance, 10 mm offset", f"{norm_in:.2f}")
show("normalized distance, 20 mm offset", f"{norm_out:.2f}")

# ---------------------------------------------------------------------------
head("Conformal refusal: the jar beside the mug (Sec. 10.3.1)")
tot = SCORE_MUG + SCORE_JAR
p_mug, p_jar = SCORE_MUG / tot, SCORE_JAR / tot
entropy = -(p_mug * math.log(p_mug) + p_jar * math.log(p_jar))
rank = math.ceil((N_CAL + 1) * (1 - ALPHA))
thresh = 1 - QHAT
show("normalized mug", f"{p_mug:.2f}")
show("normalized jar", f"{p_jar:.2f}")
show("entropy (nats)", f"{entropy:.3f}")
show("two-candidate maximum ln 2 (nats)", f"{math.log(2):.3f}")
show("calibration rank of qhat", rank)
show("inclusion threshold 1 - qhat", f"{thresh:.2f}")
set_size = sum(p >= thresh for p in (p_mug, p_jar))
show("prediction-set size", set_size)
tot_low = SCORE_MUG + SCORE_JAR_LOW
p_mug_low, p_jar_low = SCORE_MUG / tot_low, SCORE_JAR_LOW / tot_low
show("low jar case: normalized mug", f"{p_mug_low:.2f}")
show("low jar case: normalized jar", f"{p_jar_low:.2f}")
show("low jar case: set size", sum(p >= thresh for p in (p_mug_low, p_jar_low)))
# The jar in place of the mug: one retained candidate normalizes to 1.
show("jar alone: normalized score", f"{SCORE_JAR / SCORE_JAR:.2f}")

# ---------------------------------------------------------------------------
head("Imprecise estimate: the distant view (Sec. 10.3.3)")
k = math.sqrt(CHI2_3DOF_95)
semi = [k * s for s in SIGMA]
show("chi radius sqrt(7.815)", f"{k:.3f}")
show("95% semi-axes (mm)", ", ".join(f"{s:.1f}" for s in semi))
show("depth semi-axis over tolerance", f"{semi[2] / R_TOL:.2f}")
show("axes inside 15 mm", sum(s <= R_TOL for s in semi))

# ---------------------------------------------------------------------------
head("Latch ceilings (Napkin Math 10.1)")
f_brake = F_TRIP + K_LATCH * V_LATCH * T_CONTACT / 1000.0   # N/mm * mm/s * s
x_stall = V_LATCH * T_STALL / 1000.0
f_stall = K_LATCH * x_stall
show("force at brake command (N)", f"{f_brake:.0f}")
show("stall travel (mm)", f"{x_stall:.0f}")
show("stall force (N)", f"{f_stall:,.0f}")
show("stall force over limit (x)", f"{f_stall / F_LIMIT:.0f}")
show("force rise rate (N per ms)", f"{K_LATCH * V_LATCH / 1000.0:.0f}")
x_left = (F_LIMIT - f_brake) / K_LATCH
show("compression left after brake command (mm)", f"{x_left:.2f}")

# ---------------------------------------------------------------------------
head("Cadence gap (Napkin Math 10.2)")
budget = R_TOL - E0
tau_u = budget / V_BELT * 1000.0
tau_t = math.sqrt(2 * budget / A_SLIP) * 1000.0
err_period = E0 + V_BELT * t_period / 1000.0
overrun = T_INFER - tau_u
age_renew = t_period + T_INFER
v_renew = budget / (age_renew / 1000.0)
show("error budget (mm)", f"{budget:.0f}")
show("untracked horizon (ms)", f"{tau_u:.0f}")
show("tracked horizon (ms)", f"{tau_t:.2f} -> {tau_t:.0f}")
show("error at one period (mm)", f"{err_period:.0f}")
show("past the tolerance at one period (mm)", f"{err_period - R_TOL:.0f}")
show("inference overruns horizon by (ms)", f"{overrun:.1f} -> {overrun:.0f}")
show("evidence age at model-only renewal (ms)", f"{age_renew:.1f} -> {age_renew:.0f}")
show("model-only renewal ceiling (mm/s)", f"{v_renew:.1f}")
show("as share of belt speed (%)", f"{100 * v_renew / V_BELT:.1f}")
tau_slow = budget / V_DRIFT_SLOW * 1000.0
show("settling-drift horizon (ms)", f"{tau_slow:.0f}")
show("inference / untracked horizon (x)", f"{T_INFER / tau_u:.1f}")

# ---------------------------------------------------------------------------
head("Host stall against both horizons (Sec. 10.5)")
err_u = E0 + V_BELT * T_STALL / 1000.0
err_t = E0 + 0.5 * A_SLIP * (T_STALL / 1000.0) ** 2
onset = T_TICK + T_BUS + T_BRAKE
show("untracked target error after 200 ms (mm)", f"{err_u:.0f}")
show("tracked target error after 200 ms (mm)", f"{err_t:.0f}")
show("stop onset after expiry (ms)", f"{onset:.0f}")
show("untracked expiry + onset (ms)", f"{tau_u + onset:.0f}")
show("tracked expiry + onset (ms)", f"{tau_t + onset:.0f}")

# ---------------------------------------------------------------------------
head("Checkpoint estimate: 50 mm/s drift")
tau_chk = budget / V_DRIFT_CHECK * 1000.0
show("untracked horizon at 50 mm/s (ms)", f"{tau_chk:.0f}")
show("model-only renewal age (ms)", f"{age_renew:.0f}")
show("renewal fits", age_renew < tau_chk)

# ---------------------------------------------------------------------------
head("Dynamic reachability (Sec. 10.6, Definition 10.2)")
reach_t = V_TCP * tau_t / 1000.0
reach_u = V_TCP * tau_u / 1000.0
t_transit = D_TARGET / V_TCP * 1000.0
n_capt = math.ceil(t_transit / tau_t)
show("reach in tracked window (mm)", f"{reach_t:.0f}")
show("reach in untracked window (mm)", f"{reach_u:.0f}")
show("transit floor for 300 mm (ms)", f"{t_transit:.0f}")
show("shortfall against tracked window (ms)", f"{t_transit - tau_t:.0f}")
show("fresh captures needed", n_capt)

# ---------------------------------------------------------------------------
head("Freezing radius (Napkin Math 10.3)")
d_u = V_HUMAN * tau_u
d_t = V_HUMAN * tau_t
d_long = V_HUMAN * T_FORECAST
t_block = AISLE_W / V_HUMAN
show("growth over untracked horizon (mm)", f"{d_u:.0f}")
show("growth over tracked horizon (mm)", f"{d_t:.1f} -> {d_t:.0f}")
show("growth over 2 s (m)", f"{d_long:.1f}")
show("2 s set over aisle width (x)", f"{d_long / AISLE_W:.1f}")
show("horizon at which one person spans the aisle (s)", f"{t_block:.4f} -> {t_block:.2f}")

# ---------------------------------------------------------------------------
head("Lease expiry, chunk lease contrast (Sec. 10.4, 10.8)")
show("chunk lease (ms)", f"{T_CHUNK_LEASE:.0f}")
show("tracked evidence term over chunk lease (x)", f"{tau_t / T_CHUNK_LEASE:.1f}")


# ---------------------------------------------------------------------------
head("OK-Robot failure analysis (Liu et al. 2024, Fig. 4 counts)")
OKR = dict(trials=375, success=254, nav_wrong_object=35, nav_other=7,
           grasp_model=45, hardware=28, placing=6)   # Fig. 4, three cleanup levels
parts = OKR["success"] + OKR["nav_wrong_object"] + OKR["nav_other"] + OKR["grasp_model"] \
    + OKR["hardware"] + OKR["placing"]
show("categories sum (check = 375)", parts)
for k in ("success", "nav_wrong_object", "grasp_model", "hardware"):
    show(f"{k} share (%)", f"{100 * OKR[k] / OKR['trials']:.1f}")

# ---------------------------------------------------------------------------
head("KnowNo hardware result (Ren et al. 2023, Table 2, target 0.85)")
KN = dict(knowno_succ=0.76, knowno_help=0.67, simple_succ=0.75, simple_help=0.81,
          nohelp_succ=0.51)
show("help-rate reduction vs simple set (points)", f"{100 * (KN['simple_help'] - KN['knowno_help']):.0f}")

# ---------------------------------------------------------------------------
head("Published action-model rates as periods (ms)")
VLA = [  # (label, low period ms, high period ms, source)
    ("RT-2 55B", 1000 / 3, 1000 / 1, "Brohan et al. 2023, Sec. 3.3: 1-3 Hz, cloud TPUs"),
    ("RT-1", 1000 / 3, 1000 / 3, "Brohan et al. 2023 (RT-1), Sec. 5.1: 3 Hz"),
    ("RT-2 5B", 1000 / 5, 1000 / 5, "Sec. 3.3: about 5 Hz"),
    ("OpenVLA 7B", 1000 / 6, 1000 / 6, "Kim et al. 2024, Sec. 3.5: about 6 Hz, RTX 4090, bf16"),
    ("pi0 chunk", 73.0, 73.0, "Black et al. 2024, Table I: 73 ms on-board, RTX 4090"),
]
for lab, lo, hi, _ in VLA:
    show(f"{lab} period (ms)", f"{lo:.0f}" if lo == hi else f"{lo:.0f} to {hi:.0f}")
show("pi0 chunk over untracked horizon (x)", f"{73.0 / tau_u:.2f}")

# ---------------------------------------------------------------------------
head("Walking speeds (Bohannon 1997, reference table)")
WALK = [  # (group, comfortable cm/s, maximum cm/s)
    ("men 20s", 139.3, 253.3), ("men 50s", 139.3, 206.9),
    ("women 20s", 140.7, 246.7), ("women 50s", 139.5, 201.0)]
for g, c, m in WALK:
    show(f"{g} comfortable / max (m/s)", f"{c / 100:.2f} / {m / 100:.2f}")
show("fastest max over ISO 1.6 m/s (x)", f"{max(m for _, _, m in WALK) / 100 / V_HUMAN:.2f}")

# Values make_figs.py reuses
VALUES = dict(
    tau_u=tau_u, tau_t=tau_t, t_infer=T_INFER, t_period=t_period, onset=onset,
    err_u=err_u, err_t=err_t, f_brake=f_brake, f_stall=f_stall, f_trip=F_TRIP,
    f_limit=F_LIMIT, semi=semi, r_tol=R_TOL, e0=E0, v_belt=V_BELT, a_slip=A_SLIP,
    d_u=d_u, d_t=d_t, d_long=d_long, aisle_w=AISLE_W, t_transit=t_transit,
    reach_t=reach_t, v_human=V_HUMAN, tau_slow=tau_slow, norm_in=norm_in,
    norm_out=norm_out, p_mug=p_mug, p_jar=p_jar, thresh=thresh,
    p_mug_low=p_mug_low, p_jar_low=p_jar_low, age_renew=age_renew,
    okr=OKR, vla=VLA, walk=WALK,
)

if __name__ == "__main__":
    pass
