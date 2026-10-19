#!/usr/bin/env python3
"""Every derived number in Lecture 17 (Responsible Physical AI), Volume IV.

One block per slide, in deck order. Inputs carry their source as a comment
("ch" is the rendered chapter, Responsible Physical AI; "RM" and "AISLE" are
the running machine and its aisle scenario in the MLSysIM registry, which the
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
from mlsysim.physics import calc_queue_latency_mmc  # noqa: E402
from scipy.stats import beta, norm  # noqa: E402

RM = Embodied.MobileManipulator.WarehouseMobileManipulator
AISLE = Embodied.Scenario.WarehouseAisle
VALUES = {}


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


def mag(q, unit):
    return q.to(unit).magnitude


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
V_AISLE = mag(AISLE.v_aisle, "m/s")              # 1.3 m/s, chosen aisle speed
D_CLEAR = mag(AISLE.d_clear, "m")                # 1.10 m clear distance (illustrative)
T_OOL = mag(AISLE.t_takeover_out_of_loop, "s")   # 4.465 s out-of-loop takeover (derived in the registry)
T_INL = mag(AISLE.t_takeover_in_loop, "s")       # 0.605 s in-loop takeover (derived in the registry)
A_BRAKE = mag(AISLE.a_brake, "m/s**2")           # 2 m/s^2 credible deceleration (illustrative)
TAU_DELAY = mag(AISLE.tau_delay, "s")            # 133.6 ms pre-brake delay
M_LOADED = mag(RM.loaded_mass, "kg")             # 368 kg loaded mass (derived)
F_CAM = mag(RM.nav_camera.frame_rate, "Hz")      # 60 Hz navigation camera (datasheet)
T_PRE = mag(AISLE.pre_trigger_window, "s")       # 5 s pre-trigger window (derived)

# Cruise, 2 October 2023 (ch Sec. 17.1 and 17.4.2, from the sources the chapter cites)
DRAG_FT = 20.0             # ft, pullover distance (Quinn Emanuel 2024)
DRAG_MPH = 7.7             # mph, peak pullover speed (same source)
DAYS_WITHHELD = 15         # days, 3 to 18 October (CPUC D.24-07-004)
CPUC_DAILY = 7_500         # USD per day (CPUC D.24-07-004)
NHTSA_PENALTY = 1_500_000  # USD, consent order, September 2024
DOJ_FINE = 500_000         # USD, deferred prosecution agreement, November 2024

# ---------------------------------------------------------------------------
head("Hook and timeline: the Cruise pullover and the record (Sec. 17.1, 17.4.2)")
drag_m = DRAG_FT * 0.3048
drag_ms = DRAG_MPH * 0.44704
cpuc_fine = DAYS_WITHHELD * CPUC_DAILY
show("pullover distance", f"{drag_m:.1f} m")
show("peak pullover speed", f"{drag_ms:.1f} m/s")
show("CPUC fine = 15 days x $7,500", f"${cpuc_fine:,}")
VALUES["cruise"] = dict(drag_m=drag_m, cpuc=cpuc_fine, nhtsa=NHTSA_PENALTY, doj=DOJ_FINE)

# ---------------------------------------------------------------------------
head("Running machine: one admitted aisle command (Sec. 17.2)")
d_brake = V_AISLE**2 / (2 * A_BRAKE)
show("aisle speed", f"{V_AISLE:.1f} m/s")
show("clear distance", f"{D_CLEAR:.2f} m")
show("loaded mass", f"{M_LOADED:.0f} kg")
show("braking distance v^2/(2a)", f"{d_brake:.2f} m")
show("kinetic energy at aisle speed 1/2 m v^2", f"{0.5 * M_LOADED * V_AISLE**2:.0f} J")
VALUES["ke_aisle"] = 0.5 * M_LOADED * V_AISLE**2

# ---------------------------------------------------------------------------
head("Estimate: a six-link chain of decisions (Sec. 17.2.1)")
K_LINKS = 6                       # ch: six decisions behind the aisle command
P_LINKS = (0.99, 0.95, 0.90)      # ch: retrievability per link (illustrative)
chain = {p: p**K_LINKS for p in P_LINKS}
for p, c in chain.items():
    show(f"chain resolves at p = {p:.2f} per link", f"{c:.2f}")
VALUES["chain"] = chain

# ---------------------------------------------------------------------------
head("Time to act at the rack end (Sec. 17.3.1)")
t_window = D_CLEAR / V_AISLE
travel = V_AISLE * T_OOL
ratio = T_OOL / t_window
show("time the hazard leaves = d_clear / v", f"{t_window:.2f} s")
show("out-of-loop takeover", f"{T_OOL:.1f} s ({T_OOL:.3f} s)")
show("travel during the takeover", f"{travel:.1f} m")
show("takeover / window", f"{ratio:.1f}, about {round(ratio)}x")
show("in-loop takeover (Lecture 14)", f"{T_INL:.3f} s")
VALUES.update(t_window=t_window, t_ool=T_OOL, travel=travel, ratio=ratio)

# ---------------------------------------------------------------------------
head("One assistant, many machines (Napkin Math 17.1)")
RATE_PER_MACHINE_H = 0.5   # requests per machine per operating hour (illustrative)
T_SERVICE_S = 180.0        # s per request (illustrative)


def queue(n, rate_h=RATE_PER_MACHINE_H, s=T_SERVICE_S):
    lam = n * rate_h / 3600.0
    rho = lam * s
    wq = rho * s / (1 - rho)
    _, _, p99 = calc_queue_latency_mmc(lam, 1 / s, 1)
    return rho, wq, p99


Q = {}
for n in (10, 20, 32):
    rho, wq, p99 = queue(n)
    Q[n] = (rho, wq, p99)
    show(f"{n} machines: utilization, mean wait", f"{rho:.2f}, {wq:.0f} s ({wq / 60:.0f} min)")
p99_20 = Q[20][2]
p99_20_s = p99_20.to("s").magnitude if hasattr(p99_20, "to") else float(p99_20)
show("20 machines: 99th-percentile wait", f"{p99_20_s:.0f} s ({p99_20_s / 60:.1f} min)")
n_max = math.floor(0.5 / (RATE_PER_MACHINE_H / 3600.0 * T_SERVICE_S))
show("machines at a 0.5 utilization cap", n_max)
VALUES.update(queue=Q, p99_20=p99_20_s)

# Checkpoint 17.1 load question (ch): 24 machines, 0.8 per machine-hour, 75 s
rho_c, wq_c, _ = queue(24, 0.8, 75.0)
show("Checkpoint 17.1: 24 machines, 0.8/h, 75 s -> utilization, mean wait", f"{rho_c:.2f}, {wq_c:.0f} s")
VALUES.update(rho_c=rho_c, wq_c=wq_c)

# ---------------------------------------------------------------------------
head("Leaving a stage on evidence (Sec. 17.4.1)")
RATE_TARGET = 1.0e-3       # safety-relevant interventions per operating hour (chosen)
ALPHA = 0.05               # one-sided, 95 percent confidence (chosen)
N_STAGE = 10               # machines in the supervised stage (illustrative)
H_PER_DAY = 16.0           # operating hours per machine-day (illustrative)
t_req = math.log(1 / ALPHA) / RATE_TARGET
days_req = t_req / (N_STAGE * H_PER_DAY)
show("failure-free hours = ln(1/0.05) / 1e-3 per h", f"{t_req:,.0f} h")
show("fleet hours per day", f"{N_STAGE * H_PER_DAY:.0f} h")
show("days to reach them", f"{days_req:.1f}, so {math.ceil(days_req)} days")
VALUES.update(t_req=t_req, days_req=days_req, fleet_h_day=N_STAGE * H_PER_DAY)

# ---------------------------------------------------------------------------
head("Withdrawal and disclosure: penalties for the record (Sec. 17.4.2)")
total_pen = cpuc_fine + NHTSA_PENALTY + DOJ_FINE
show("penalties for the record, total", f"${total_pen:,}")

# ---------------------------------------------------------------------------
head("Published per-group detection (Sec. 17.5)")
# Wilson, Hoffman, Morgenstern 2019, arXiv 1902.11097, Table 4 (COCO-trained
# Detectron models on BDD100K validation people), AP75 in percent, LS then DS.
WILSON_AP75 = {
    "Faster R-CNN R-50-C4": (64.5, 53.6),
    "Faster R-CNN R-50-FPN": (63.1, 53.6),
    "Faster R-CNN R-101-FPN": (70.0, 57.3),
    "Faster R-CNN X-101-FPN": (69.8, 62.1),
    "Mask R-CNN R-50-C4": (66.9, 51.5),
    "Mask R-CNN R-50-FPN": (67.6, 51.2),
    "Mask R-CNN R-101-FPN": (65.6, 55.2),
    "Mask R-CNN X-101-FPN": (69.2, 62.9),
}
WILSON_AVG = (67.1, 55.9)          # Table 4, "Average" row
WILSON_HOLDOUT = (387, 100)        # Sec. 4.2: holdout people, LS and DS
WILSON_NEEDED = (12_000, 4_000)    # Sec. 4.2: to verify a 0.05 AP gap
avg = sum(v[0] for v in WILSON_AP75.values()) / 8, sum(v[1] for v in WILSON_AP75.values()) / 8
show("Wilson Table 4 AP75 average recomputed (LS, DS)", f"{avg[0]:.1f}, {avg[1]:.1f}")
show("Wilson gap in points", f"{WILSON_AVG[0] - WILSON_AVG[1]:.1f}")
# Li et al. 2025 (TOSEM 34(3); arXiv 2308.02935v4), Table 7: miss rate (%) adult, child
LI_MR = {
    "YOLOX": (12.64, 42.47),
    "RetinaNet": (14.36, 44.33),
    "Faster RCNN": (5.24, 26.06),
    "Cascade RCNN": (5.12, 26.57),
    "ALFNet": (36.62, 53.47),
    "CSP": (37.74, 50.42),
    "MGAN": (33.58, 46.53),
    "PRNet": (44.69, 61.25),
}
LI_AVG = (23.75, 43.89)            # Table 7, "Average" row
show("Li Table 7 child minus adult miss rate (points)", f"{LI_AVG[1] - LI_AVG[0]:.2f}")
VALUES.update(wilson=WILSON_AP75, wilson_avg=WILSON_AVG, li=LI_MR, li_avg=LI_AVG)

# ---------------------------------------------------------------------------
head("Two frames late (Napkin Math 17.2, Table 1)")
P_MISS = 1.0e-3                     # chosen all-frames miss probability
T_FRAME = 1.0 / F_CAM               # s, 16.7 ms
P_REF = 0.95                        # illustrative, best-detected group
P_LOWS = (0.85, 0.75, 0.60)         # illustrative, lower-detection groups


def frames(p, p_miss=P_MISS):
    return math.ceil(math.log(p_miss) / math.log(1.0 - p) - 1e-12)


def shortfall(dn, v=V_AISLE):
    dr = v * T_FRAME * dn
    v_imp = math.sqrt(2 * A_BRAKE * dr)
    e_imp = M_LOADED * A_BRAKE * dr
    return dr, v_imp, e_imp


n_ref = frames(P_REF)
show("frame period", f"{T_FRAME * 1e3:.1f} ms")
show("reference group frames", n_ref)
ROWS = []
for p in P_LOWS:
    n = frames(p)
    dr, vi, ei = shortfall(n - n_ref)
    ROWS.append((p, n, n - n_ref, dr, vi, ei))
    show(f"p = {p:.2f}: frames, extra, mm lost, impact m/s, J",
         f"{n}, {n - n_ref}, {dr * 1e3:.0f} mm, {vi:.2f} m/s, {ei:.0f} J")
show("two extra frames in ms", f"{2 * T_FRAME * 1e3:.1f} ms")
show("braking distance at aisle speed", f"{d_brake:.2f} m")
# miss probability in every one of n frames, for the figure
for p in (P_REF, 0.75):
    show(f"(1-p)^n for p = {p}", ", ".join(f"n={n}: {(1 - p) ** n:.1e}" for n in range(1, 7)))
VALUES.update(n_ref=n_ref, rows=ROWS, t_frame=T_FRAME, d_brake=d_brake, p_miss=P_MISS)

# ---------------------------------------------------------------------------
head("Published miss rates in meters and joules (derived, Li et al. Table 7)")
# Treat each detector's per-image miss rate as a per-frame miss probability,
# independent across frames (the chapter's best-case assumption), at the
# running machine's aisle speed, camera, mass, and deceleration.
LI_CONV = {}
for name, (ma, mc) in LI_MR.items():
    na, nc = frames(1 - ma / 100), frames(1 - mc / 100)
    dr, vi, ei = shortfall(nc - na)
    LI_CONV[name] = (na, nc, dr, vi, ei)
    show(f"{name}: adult frames, child frames, mm, m/s, J",
         f"{na}, {nc}, {dr * 1e3:.0f} mm, {vi:.2f} m/s, {ei:.0f} J")
na, nc = frames(1 - LI_AVG[0] / 100), frames(1 - LI_AVG[1] / 100)
dr, vi, ei = shortfall(nc - na)
show("Li average: adult frames, child frames, mm, m/s, J", f"{na}, {nc}, {dr * 1e3:.0f} mm, {vi:.2f} m/s, {ei:.0f} J")
VALUES.update(li_conv=LI_CONV, li_avg_conv=(na, nc, dr, vi, ei))

# ---------------------------------------------------------------------------
head("Sample sizes for each group (Sec. 17.5.2)")
N_TOTAL, SHARE_SMALL, RECALL = 10_000, 0.02, 0.95   # illustrative
P_A, P_B, ALPHA2, POWER = 0.95, 0.92, 0.05, 0.80    # chosen


def cp(k, n, conf=0.95):
    a = 1.0 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


n_small = round(N_TOTAL * SHARE_SMALL)
n_large = N_TOTAL - n_small
ci_s = cp(round(RECALL * n_small), n_small)
ci_l = cp(round(RECALL * n_large), n_large)
w_ratio = (ci_s[1] - ci_s[0]) / (ci_l[1] - ci_l[0])
z_a, z_b = norm.ppf(1 - ALPHA2 / 2), norm.ppf(POWER)
pbar = (P_A + P_B) / 2
n_needed = math.ceil((z_a * math.sqrt(2 * pbar * (1 - pbar))
                      + z_b * math.sqrt(P_A * (1 - P_A) + P_B * (1 - P_B))) ** 2 / (P_A - P_B) ** 2)
show("group sizes", f"{n_small} and {n_large}")
show("95% exact interval, 9,800 people", f"{ci_l[0]:.3f} to {ci_l[1]:.3f}")
show("95% exact interval, 200 people", f"{ci_s[0]:.3f} to {ci_s[1]:.3f}")
show("width ratio", f"{w_ratio:.1f}, about {round(w_ratio)}x")
show("people per group to tell 0.95 from 0.92 at 80% power", f"{n_needed:,}")
show("share of a 10,000-person set per group", f"{n_needed / N_TOTAL:.1%}")
VALUES.update(ci_s=ci_s, ci_l=ci_l, n_needed=n_needed, n_small=n_small, n_large=n_large)

# ---------------------------------------------------------------------------
head("Who bears the risk (Sec. 17.6)")
ENC_SHARE, WORKER_SHARE = 0.80, 0.10   # illustrative
exp_ratio = ENC_SHARE / WORKER_SHARE
rest_ratio = (1 - ENC_SHARE) / (1 - WORKER_SHARE)
show("crossing users' exposure / site average", f"{exp_ratio:.0f}x")
show("everyone else / site average", f"{rest_ratio:.0%}")
VALUES.update(exp_ratio=exp_ratio, rest_ratio=rest_ratio)

# ---------------------------------------------------------------------------
head("Keeping less (Sec. 17.7.2)")
T_POST = 25.0          # s after the trigger (chosen)
EVENTS = 20            # per machine-day (illustrative)
DAY_H = 16.0           # h (illustrative)
window = T_PRE + T_POST
kept = EVENTS * window / (DAY_H * 3600)
show("window per event", f"{window:.0f} s")
show("share of the day kept", f"{kept:.2%}")
show("reduction", f"{1 / kept:.0f}x")
show("footage kept per machine-day", f"{EVENTS * window / 60:.0f} min of {DAY_H * 60:.0f} min")
VALUES.update(window=window, kept=kept, events=EVENTS, day_h=DAY_H, t_pre=T_PRE)

# ---------------------------------------------------------------------------
head("Where the labor went (Napkin Math 17.3)")
SWEEPS, T_SWEEP_H, SHIFT_H = 4, 20 / 60, 8.0   # illustrative
rho20 = Q[20][0]
assist_h = rho20 * DAY_H
idle_h = DAY_H - assist_h
sweep_h = SWEEPS * T_SWEEP_H
total_h = DAY_H + sweep_h
show("request hours", f"{assist_h:.0f} h")
show("watching and waiting hours", f"{idle_h:.0f} h")
show("sweep hours", f"{sweep_h:.1f} h")
show("person-hours per day", f"{total_h:.1f} h")
show("full-time shifts", f"{total_h / SHIFT_H:.1f}")
VALUES.update(assist_h=assist_h, idle_h=idle_h, sweep_h=sweep_h, total_h=total_h)

if __name__ == "__main__":
    pass
