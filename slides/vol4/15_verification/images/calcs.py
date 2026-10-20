#!/usr/bin/env python3
"""Every derived number in Lecture 15 (Adversarial Verification), one block per slide.

Inputs carry their source. "AV" is the rendered chapter, Adversarial
Verification (Volume IV, Chapter 15); "NM 15.x" is its Napkin Math 15.x.
Values the chapter labels illustrative or chosen stay so on the slides.
Outside sources are named where they are used, with the table read from the
paper itself. Every printed value is what the slide or its note shows, with
the label used there. make_figs.py and make_record.py import from this file.

Run from this folder: python3 calcs.py
"""

import math

G = 9.80665                    # standard gravity, m/s^2


def head(title):
    print(f"\n--- {title}")


def show(label, value):
    print(f"  {label}: {value}")


# =============================================================================
# Shared inputs: the running machine at the rack end (AV 15.2, NM 15.1)
# =============================================================================
V_AISLE = 1.3                  # m/s, candidate aisle speed (chosen)
V_RESTRICTED = 1.0             # m/s, candidate restricted point on inspected floors (AV 15.2)
D_CLEAR = 1.100                # m, clear distance (illustrative)
T_LEASE = 0.060                # s, chunk lease (chosen)
T_TICK = 0.001                 # s, permission-loop tick (chosen)
T_BUS = 0.001                  # s, one fieldbus cycle (illustrative)
T_ACT = 0.020                  # s, delivered stop command to drive onset (illustrative)
A_BRAKE = 2.0                  # m/s^2, credible deceleration, dry floor (illustrative)
# Jerk limit: AV NM 15.1 states 7.11 m/s^3 (derived, the reference stop's peak
# jerk). Same derivation as Lecture 12: the S-curve stop from 1.5 m/s takes
# 1,031.25 ms against 750 ms at constant deceleration, so J = 2 / 0.28125.
J_MAX = A_BRAKE / (1.03125 - 1.5 / A_BRAKE)
TRACK = 0.040                  # m, tracking inset (illustrative)
LOC = 0.050                    # m, localization bound (illustrative)
MARGIN = 0.100                 # m, protective clearance (chosen)
T_AGE = 0.0516                 # s, evidence age at dispatch (derived; AV 15.2, Lecture 8)
TAU_PREBRAKE = 0.1336          # s, pre-brake subtotal from capture (AV 15.2)
T_DA = 0.0633                  # s, dispatch-to-admission allocation (AV 15.2; Lecture 13)
MU_DRY = 0.60                  # dry concrete friction (AV 15.6.1)
MU_INSPECTED = 0.12            # inspected-floor guarantee (AV 15.2)
PAYLOAD = 50                   # kg, full rack payload (AV 15.2)


def suffix(v, a=A_BRAKE, j=J_MAX):
    """Jerk-limited resident stop distance, AV NM 15.1 (equation 11.17 form), m."""
    return v ** 2 / (2 * a) + v * a / (2 * j)


def onset_bound():
    return T_LEASE + T_TICK + T_BUS + T_ACT


def f1_bound(v, a=A_BRAKE):
    """Renewal-referenced F1 travel bound, AV equation 1, m."""
    return v * onset_bound() + suffix(v, a) + TRACK


def capture_subtotal(v, a=A_BRAKE):
    """Capture-referenced design subtotal, AV NM 15.1, m."""
    return v * TAU_PREBRAKE + suffix(v, a) + LOC + MARGIN + TRACK


def ceiling(mu, tau=TAU_PREBRAKE):
    """Optimistic all-weight screen: largest v with the capture subtotal within D_CLEAR."""
    a = min(A_BRAKE, mu * G)
    lo, hi = 0.0, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if capture_subtotal(mid, a) <= D_CLEAR:
            lo = mid
        else:
            hi = mid
    return lo


SPARE = D_CLEAR - capture_subtotal(V_AISLE)          # m
T_SPARE = SPARE / V_AISLE                            # s

# =============================================================================
# Recall (Lecture 14): what the spare buys at aisle speed
# =============================================================================
head("Recall: the spare at aisle speed (AV 15.2; Lecture 14)")
show("jerk limit m/s^3 (derived)", f"{J_MAX:.2f}")
show("capture subtotal at 1.3 m/s, mm", f"{capture_subtotal(V_AISLE)*1e3:.1f}")
show("spare mm", f"{SPARE*1e3:.1f}")
show("spare as waiting time ms", f"{T_SPARE*1e3:.1f}")
da_travel = V_AISLE * T_DA
show("dispatch-to-admission 63.3 ms costs mm", f"{da_travel*1e3:.1f}")
show("spare left after allocation mm", f"{(SPARE - da_travel)*1e3:.1f}")
T_LEFT = (SPARE - da_travel) / V_AISLE
show("waiting time left after allocation ms", f"{T_LEFT*1e3:.1f}")
T_TAKEOVER = 0.605             # s, in-loop takeover allowance (AV 15.2; Lecture 14)
show("in-loop takeover 605 ms: travel mm", f"{V_AISLE*T_TAKEOVER*1e3:.1f}")
show("in-loop takeover overrun past the spare mm", f"{(V_AISLE*T_TAKEOVER - SPARE)*1e3:.1f}")

# =============================================================================
# Segment 1: the envelope and its coupled corner (AV 15.2, 15.5.1)
# =============================================================================
head("Envelope: speed ceiling against floor friction (AV 15.5.1, optimistic all-weight screen)")
mu_knee = A_BRAKE / G
show("friction where traction stops binding (knee)", f"{mu_knee:.3f}")
show("ceiling on a dry floor m/s", f"{ceiling(MU_DRY):.2f}")
show("ceiling on the inspected floor m/s", f"{ceiling(MU_INSPECTED):.2f}")
a_insp = MU_INSPECTED * G
show("traction-limited deceleration on 0.12 m/s^2", f"{a_insp:.2f}")
stop_corner = capture_subtotal(V_AISLE, a_insp)
show("1.3 m/s on the inspected floor: capture subtotal m", f"{stop_corner:.2f}")
show("overrun past 1.10 m, mm", f"{(stop_corner - D_CLEAR)*1e3:.0f}")
show("30 ms memory stall at aisle speed, mm", f"{V_AISLE*0.030*1e3:.0f}")

head("DriveFI (Jha et al., DSN 2019, arXiv 1907.01051, sec. I and VI)")
DFI_FAULTS = 98400             # faults in the list (sec. I)
DFI_EXHAUSTIVE_DAYS = 615      # exhaustive evaluation (sec. I)
DFI_BAYES_HOURS = 4            # "less than 4 hours" (abstract, sec. I)
DFI_FOUND, DFI_HAZARD = 561, 460
DFI_SCENES_HAZ, DFI_SCENES = 68, 7200
DFI_RANDOM_FI = 5000           # random FI experiments over "several weeks", 0 hazards
DFI_GPU_FI = 2400              # GPU fault injections, DriveAV (sec. VI): SDC 1.9 %, DUE 7.35 %, 0 hazards
show("exhaustive hours", f"{DFI_EXHAUSTIVE_DAYS*24:,}")
show("acceleration (paper: 3690x)", f"{DFI_EXHAUSTIVE_DAYS*24/DFI_BAYES_HOURS:,.0f}")
show("hazardous share of scenes %", f"{100*DFI_SCENES_HAZ/DFI_SCENES:.2f}")
show("found faults that manifested as hazards %", f"{100*DFI_HAZARD/DFI_FOUND:.0f}")

# =============================================================================
# Segment 2: the F1 oracle (NM 15.1), the trace, the verdicts
# =============================================================================
head("NM 15.1: the reference-event gap")
show("onset bound ms", f"{onset_bound()*1e3:.0f}")
show("resident stop at 1.3 m/s, mm", f"{suffix(V_AISLE)*1e3:.1f}")
show("  v^2/2a mm", f"{V_AISLE**2/(2*A_BRAKE)*1e3:.1f}")
show("  v a / 2j mm", f"{V_AISLE*A_BRAKE/(2*J_MAX)*1e3:.1f}")
show("travel before onset mm", f"{V_AISLE*onset_bound()*1e3:.1f}")
show("F1 oracle at 1.3 m/s, mm", f"{f1_bound(V_AISLE)*1e3:.1f}")
show("capture subtotal at 1.3 m/s, mm", f"{capture_subtotal(V_AISLE)*1e3:.1f}")
gap = capture_subtotal(V_AISLE) - f1_bound(V_AISLE)
show("gap mm", f"{gap*1e3:.1f}")
show("  evidence-age travel mm", f"{V_AISLE*T_AGE*1e3:.1f}")
show("late onset the wrong bound would tolerate ms", f"{gap/V_AISLE*1e3:.0f}")
show("F1 oracle at 1.0 m/s on 0.12, mm", f"{f1_bound(V_RESTRICTED, a_insp)*1e3:.1f}")
show("  resident stop at 1.0 m/s on 0.12, mm", f"{suffix(V_RESTRICTED, a_insp)*1e3:.1f}")
show("capture subtotal at 1.0 m/s on 0.12, mm", f"{capture_subtotal(V_RESTRICTED, a_insp)*1e3:.1f}")
red_v = 1 - V_RESTRICTED / V_AISLE
red_d = 1 - f1_bound(V_RESTRICTED, a_insp) / f1_bound(V_AISLE)
show("speed reduction %", f"{red_v*100:.1f}")
show("oracle shortening %", f"{red_d*100:.1f}")

head("Diagnose: onset at 90 ms (Checkpoint 15.1)")
ON_90 = 0.090
show("onset margin ms", f"{(onset_bound()-ON_90)*1e3:.0f}")
show("extra travel before braking mm", f"{V_AISLE*(ON_90-onset_bound())*1e3:.1f}")

head("Signed margins: the late F1 trace (AV 15.9.1, figure 4)")
ON_LATE = 0.112                # s, onset of the late trace (stipulated)
nominal_travel = f1_bound(V_AISLE) - TRACK
late_travel = nominal_travel + V_AISLE * (ON_LATE - onset_bound())
rho_on = onset_bound() - ON_LATE
rho_d = f1_bound(V_AISLE) - late_travel
show("on-time travel mm", f"{nominal_travel*1e3:.1f}")
show("late travel mm", f"{late_travel*1e3:.1f}")
show("onset margin ms", f"{rho_on*1e3:.0f}")
show("distance margin mm", f"{rho_d*1e3:.0f}")
show("normalized onset (scale 82 ms)", f"{rho_on/onset_bound():.3f}")
show("normalized distance (scale 751.9 mm)", f"{rho_d/f1_bound(V_AISLE):.4f}")
show("normalized distance (scale 40 mm, Checkpoint 15.3)", f"{rho_d/TRACK:.3f}")
t_stop = V_AISLE / A_BRAKE + A_BRAKE / J_MAX
show("stop duration after onset ms", f"{t_stop*1e3:.0f}")
show("rest time, on time ms", f"{(onset_bound()+t_stop)*1e3:.0f}")
show("rest time, late ms", f"{(ON_LATE+t_stop)*1e3:.0f}")
show("lease replay probe ms", "65")

# =============================================================================
# Segment 3: the fault list (AV 15.6; NM 15.2) and published attacks
# =============================================================================
head("NM 15.2: supply headroom (F4a, F4b)")
V_LOW, V_UVLO, R_SRC, I_STEP = 42.0, 33.0, 0.060, 70.0   # datasheet, datasheet, illustrative, illustrative
v_min = V_LOW - I_STEP * R_SRC
show("sag under the 70 A transient V", f"{v_min:.1f}")
show("headroom above lockout V", f"{v_min - V_UVLO:.1f}")
show("step that reaches lockout A", f"{(V_LOW - V_UVLO)/R_SRC:.0f}")
show("resistance where headroom vanishes mOhm", f"{(V_LOW - V_UVLO)/I_STEP*1e3:.0f}")
v_ceil_insp = ceiling(MU_INSPECTED)
t_set = (T_TICK + T_BUS + T_ACT) + v_ceil_insp / a_insp + a_insp / J_MAX + 0.080
show("F4b flag to brakes set ms", f"{t_set*1e3:.1f}")
show("hold-up spare of 2 s ms", f"{(2.0 - t_set)*1e3:.1f}")
show("F4a travel bound from flag mm", f"{(V_AISLE*(T_TICK+T_BUS+T_ACT) + suffix(V_AISLE) + TRACK)*1e3:.1f}")
SOC_W = 60.0
show("SoC current at 37.8 V, A", f"{SOC_W/v_min:.1f}")
show("share of the 70 A transient %", f"{100*SOC_W/v_min/I_STEP:.1f}")
show("F3: travel per lost 1 ms cycle mm", f"{V_AISLE*1e-3*1e3:.1f}")

head("Wang et al. 2025 (arXiv 2411.13587v4, Table 1a): OpenVLA, LIBERO")
WANG_SUITES = ["Spatial", "Object", "Goal", "Long"]
WANG_BENIGN_FR = [15.3, 11.6, 20.8, 46.3]        # failure rate %, benign
WANG_UADA_FR = [100.0, 100.0, 100.0, 100.0]      # UADA, DoF1-3, failure rate %
for s, b, u in zip(WANG_SUITES, WANG_BENIGN_FR, WANG_UADA_FR):
    show(f"{s}: success benign -> patched %", f"{100-b:.1f} -> {100-u:.0f}")
show("average benign failure (paper 23.5 %)", f"{sum(WANG_BENIGN_FR)/4:.1f}")

head("Robey et al. 2025 (arXiv 2410.13691, Tables 2 and 3): attack success")
ROBEY = {"Jackal UGV, GPT-4o planner": (1, 35, 35, 35),
         "Unitree Go2, GPT-3.5": (3, 35, 35, 35)}
for k, (d, n, r, m) in ROBEY.items():
    show(f"{k}: direct prompt %", f"{100*d/n:.1f}")
    show(f"{k}: RoboPAIR %", f"{100*r/m:.0f}")

head("Brown et al. 2017 (arXiv 1712.09665, Figure 1): patch on a table")
show("VGG16 banana before / toaster after %", "97 / 99")

# =============================================================================
# Segment 4: ladder, proof scope, search, transfer
# =============================================================================
head("VNN-COMP 2025 (Kaulen et al., arXiv 2512.19007, Table 3)")
VNN_MAX_PARAMS = {             # upper end of each scored benchmark's '# Params'
    "cGAN": 68e6, "NN4Sys": 37e6, "LinearizeNN": 203e3, "ml4acopf": 680e3,
    "ViT": 76e3, "Collins Aerospace": 1.8e6, "LSNC-ReLU": 275, "CCTSDB": 100e3,
    "Collins RUL CNN": 262e3, "VGGNet16": 138e6, "Traffic Signs": 1.7e6,
    "cifar100": 3.8e6, "tinyimagenet": 3.6e6, "Metaroom": 7.4e6, "Yolo": 37e6,
    "SoundnessBench": 1.7e6, "Relusplitter": 625e3, "MalBeWare": 1.5e6,
    "cersyve": 4e3, "TLL Verify Bench": 67e6, "Acas XU": 13e3,
    "Dist Shift": 855e3, "safeNLP": 4e3, "CORA": 1.1e6, "SAT ReLU": 35e3,
}
INTENT_PARAMS, CHUNK_PARAMS = 7e9, 80e6   # running machine card (AV 15.7.3; Lecture 3)
show("scored benchmarks", len(VNN_MAX_PARAMS))
show("below one million parameters", sum(v < 1e6 for v in VNN_MAX_PARAMS.values()))
show("largest M", f"{max(VNN_MAX_PARAMS.values())/1e6:.0f}")
show("7B intent model / largest", f"{INTENT_PARAMS/max(VNN_MAX_PARAMS.values()):.1f}")
show("80M chunk policy / largest", f"{CHUNK_PARAMS/max(VNN_MAX_PARAMS.values()):.2f}")

head("Transfer discount (AV 15.9.4)")
show("largest discrepancy bound that keeps 131.0 mm positive (L = 1 m)", f"{SPARE:.3f}")

head("Clock drift (AV 15.8.3, illustrative)")
ppm = 100 + 50 + 50
show("relative drift ppm", ppm)
show("drift in 100 ms us", f"{ppm*1e-6*0.1*1e6:.0f}")
show("time to spend 200 us s", f"{200e-6/(ppm*1e-6):.1f}")
show("4-period blackout us", f"{ppm*1e-6*0.4*1e6:.0f}")
show("plus 15 us jitter", f"{ppm*1e-6*0.4*1e6+15:.0f}")
show("5-period blackout us", f"{ppm*1e-6*0.5*1e6:.0f}")

# =============================================================================
# Segment 5: statistics and records
# =============================================================================
head("NM 15.3: the coverage count")
FIT = {"sensors": 1500, "compute": 1000, "transceivers and power": 2500, "gate drivers": 4800}
lam_raw = sum(FIT.values())
show("raw FIT", f"{lam_raw:,}")
for target in (100, 10):
    c_diag = 1 - target / lam_raw
    n_exact = math.log(0.05) / math.log(c_diag)
    show(f"{target} FIT: coverage %", f"{c_diag*100:.2f}")
    show(f"{target} FIT: injections (exact)", f"{n_exact:.1f} -> {math.ceil(n_exact):,}")
    show(f"{target} FIT: rule of three", f"{3*lam_raw/target:,.0f}")
for k, v in FIT.items():
    show(f"share of injections, {k} %", f"{100*v/lam_raw:.1f}")
    show(f"injections of 293, {k}", f"{293*v/lam_raw:.0f}")

head("Exposure wall (Lecture 7; AV 15.10.1)")
show("failure-free hours to bound 1e-9 per hour", f"{3/1e-9:.0e}")

head("Importance sampling (AV 15.10.2): exponential queue delay, mean 5 ms")
MU_Q = 5.0                     # ms, illustrative mean delay
gamma = T_SPARE * 1e3          # ms, uncharged threshold
p = math.exp(-gamma / MU_Q)
show("threshold ms", f"{gamma:.2f}")
show("failure probability", f"{p:.2e}")
show("charged threshold ms", f"{T_LEFT*1e3:.2f}")
show("charged failure probability", f"{math.exp(-T_LEFT*1e3/MU_Q):.1e}")
show("ratio charged / uncharged", f"{math.exp(-T_LEFT*1e3/MU_Q)/p:.0e}")
RE = 0.10
n_mc = (1 - p) / (p * RE ** 2)
show("plain Monte Carlo runs", f"{n_mc:.1e}")
show("zero-failure runs 3/p", f"{3/p:.1e}")
# cross-entropy, elite fraction 10 %
levels, m = [], MU_Q
while True:
    lev = m * math.log(10)
    if lev >= gamma:
        lev = gamma
    levels.append(lev)
    m = lev + MU_Q
    if lev >= gamma:
        break
show("CE levels ms", ", ".join(f"{x:.1f}" for x in levels))
show("final proposal mean ms", f"{m:.1f}")
q_fail = math.exp(-gamma / m)
k = 2 / MU_Q - 1 / m
second = (m / MU_Q ** 2) * math.exp(-gamma * k) / k
relvar = second / p ** 2 - 1
n_is = relvar / RE ** 2
show("share of proposal runs that fail %", f"{q_fail*100:.1f}")
show("per-draw relative variance", f"{relvar:.1f}")
show("IS runs", f"{n_is:,.0f}")
show("plain MC / IS", f"{n_mc/n_is:.1e}")
ew2 = (m / MU_Q ** 2) / k
show("expected ESS of the IS runs", f"{n_is/ew2:.0f}")

head("O'Kelly et al. 2018 (arXiv 1811.00145, Table 1): relative standard error")
OK_GAMMA = [0.14, 0.15, 0.19, 0.20]
OK = {"Naive MC, 1.3M runs": [(12.4e-6, 3.1e-6), (80.6e-6, 7.91e-6), (133e-5, 3.2e-5), (186e-5, 3.79e-5)],
      "Cross-entropy, 100K": [(19.8e-6, 8.88e-6), (66.1e-6, 15e-6), (108e-5, 9.51e-5), (164e-5, 14e-5)],
      "Naive MC, 100K": [(20e-6, 14.1e-6), (100e-6, 31.6e-6), (132e-5, 11.5e-5), (185e-5, 13.6e-5)]}
OK_RSE = {k: [100 * se / est for est, se in v] for k, v in OK.items()}
for k, v in OK_RSE.items():
    show(f"{k} relative SE %", ", ".join(f"{x:.0f}" for x in v))
show("variance ratio naive/CE at 100K, gamma 0.14",
     f"{(OK['Naive MC, 100K'][0][1]/OK['Cross-entropy, 100K'][0][1])**2:.1f}")

head("Fallacy: local watchdogs (AV 15.12)")
show("spare as share of the 133.6 ms pre-brake interval %", f"{100*T_SPARE/TAU_PREBRAKE:.1f}")
