#!/usr/bin/env python3
"""Every derived number in Volume IV Lecture 6 (Policy Training).

One block per slide, in deck order. Inputs carry their source as a comment
("ch" is the rendered chapter, Policy Training, dev text of 2026-10-18; "NM"
is a Napkin Math box). Each printed line is a value the slide or its note
shows, under the label used there. make_figs.py imports from this file so
the figures and the slides share one computation.

Run from this folder: python3 calcs.py
"""
import math


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
# Dactyl, Andrychowicz et al. 2020 (arXiv 1808.00177v5)
DACTYL_SIM_MEDIAN = 50          # Table 3, block (state), simulation, 100 trials
DACTYL_REAL = {                 # Table 4, physical robot, 10 trials each (sorted)
    "All randomizations": [50, 41, 29, 27, 14, 12, 6, 4, 4, 1],
    "No observation noise": [45, 35, 23, 11, 9, 8, 7, 6, 6, 1],
    "No physics randomizations": [7, 7, 7, 3, 2, 2, 2, 2, 2, 1],
    "No unmodeled effects": [16, 7, 3, 3, 2, 2, 1, 1, 0, 0],
    "No randomizations": [6, 2, 2, 1, 0, 0, 0, 0, 0, 0],
}
DACTYL_MEDIAN = {"All randomizations": 13, "No observation noise": 8.5,
                 "No physics randomizations": 2, "No unmodeled effects": 2,
                 "No randomizations": 0}   # Table 4, as printed
DACTYL_MEMORY = {               # Table 5, physical robot, all randomizations
    "LSTM policy": [50, 41, 29, 27, 14, 12, 6, 4, 4, 1],
    "FF policy, LSTM value": [15, 7, 6, 5, 4, 3, 3, 2, 2, 0],
    "FF policy, FF value": [15, 8, 6, 5, 3, 3, 2, 2, 2, 0],
}
DACTYL_MEMORY_MEDIAN = {"LSTM policy": 13, "FF policy, LSTM value": 3.5,
                        "FF policy, FF value": 3}   # Table 5, as printed
DACTYL_LOCKED_MEDIAN = 28.5     # Table 3, block (state, locked wrist), physical
DACTYL_YEARS = (3, 100)         # section 6.3: simulated years, none vs all randomizations
DACTYL_HOURS = (1.5, 50)        # section 6.3: wall-clock hours, same two runs

# Latch task (ch section 6.2.1, NM 6.1, NM 6.2, NM 6.3, section 6.4.2)
STEP_MS = 25.0                  # setpoint step (chosen)
RENEW_MS = 50.0                 # renewal period (derived in ch, two steps)
EPISODE_S = 8.0                 # one opening, autonomous contact phase
GAMMA = 0.99                    # ch section 6.2.1
ACT_MS = 40.0                   # ACT p99 inference target (illustrative)
DDIM_N, ENC_MS, DDIM_MS = 16, 8.0, 48.0   # NM 6.1 (illustrative, stipulated)
FLOW_N = 4                      # NM 6.1 flow candidate
JITTER_MS = 5.0                 # NM 6.1 renewal-jitter allowance (chosen)
TRIALS = 1.0e6                  # NM 6.2 (illustrative)
ROLL_S, AUTO_S, MANUAL_S = 8.0, 4.0, 30.0  # NM 6.2 (illustrative)
RIGS = 14                       # NM 6.2: largest arm count of the farm (measured)
REVERSALS = 4                   # NM 6.2 load reversals per opening (illustrative)
HIL_HOURS = 2.5                 # ch section 6.3.1: HIL-SERL upper training time (Luo et al. 2025)
MASS = 9.0                      # NM 6.3 moving mass, kg (stipulated)
V_FAST, V_GUARD = 0.10, 0.03    # m/s (illustrative)
PEN_M = 2.4e-3                  # penalty penetration at fast approach (illustrative)
K_PLATE = 4.0e5                 # plate stiffness, N/m (illustrative)
F_LIMIT = 100.0                 # latch limit, N
F_TRIP = 15.0                   # contact tripwire, N
LOOP_MS = 2.0                   # sampling, intervention, drive response (ch section 6.4.2)

# Rudin et al. 2022 (ch section 6.4.1)
ROBOTS, STEPS_PER, UPDATES = 4096, 24, 1500
RUDIN_MIN = 20.0                # "less than 20 min"
RUDIN_HZ = 50.0                 # policy control frequency

# Reality gap scenario (ch section 6.5, illustrative, linear drive)
GAP_V = 0.20                    # m/s approach
GAP_P99_MS = 49.5               # p99 actuation delay
GAP_TARGET_N = 10.0
GAP_RISE = (500.0, 250.0)       # N/s, simulation and hardware

# Residual fine-tuning (ch section 6.6)
FT_STEPS, FT_TRIP_FRAC = 20000, 0.015

# Coverage (ch section 6.2.8): 6 cells, 1 supported
CELLS, SUPPORTED = 6, 1


# ---------------------------------------------------------------------------
# Derived values, one block per slide
# ---------------------------------------------------------------------------
def dactyl():
    head("Hook and randomization slide: Dactyl")
    for k, v in DACTYL_REAL.items():
        med = (sorted(v)[4] + sorted(v)[5]) / 2
        show(f"{k}: median of 10 trials (check vs Table 4 {DACTYL_MEDIAN[k]})", med)
    show("sim hours ratio, all vs none", f"{DACTYL_HOURS[1] / DACTYL_HOURS[0]:.0f}x")
    show("sim years ratio, all vs none", f"{DACTYL_YEARS[1] / DACTYL_YEARS[0]:.0f}x")
    for k, v in DACTYL_MEMORY.items():
        med = (sorted(v)[4] + sorted(v)[5]) / 2
        show(f"{k}: median (check vs Table 5 {DACTYL_MEMORY_MEDIAN[k]})", med)


def discount():
    head("Reward slide: discount horizon")
    steps = EPISODE_S / (STEP_MS / 1000)
    show("decisions per opening", f"{steps:.0f}")
    show("weight of final reward at first decision, 0.99^320", f"{GAMMA ** steps:.2f}")
    horizon = 1 / (1 - GAMMA)
    show("effective horizon, steps", f"{horizon:.0f}")
    show("effective horizon, s", f"{horizon * STEP_MS / 1000:.1f}")
    gmin = 0.5 ** (1 / steps)
    show("gamma for half weight over 320 steps", f"{math.floor(gmin * 1e4) / 1e4:.4f}")
    return steps, gmin


def decoder():
    head("Decoder slack (NM 6.1)")
    per = (DDIM_MS - ENC_MS) / DDIM_N
    flow = ENC_MS + FLOW_N * per
    show("per-evaluation cost, ms", per)
    show("flow 4-step latency, ms", flow)
    show("ACT slack to renewal, ms", RENEW_MS - ACT_MS)
    show("DDIM slack to renewal, ms", RENEW_MS - DDIM_MS)
    show("DDIM stretched past renewal, ms", DDIM_MS + JITTER_MS - RENEW_MS)
    show("ACT stretched end, ms", ACT_MS + JITTER_MS)
    head("Estimate activity: evaluation budget")
    n_step = math.floor((STEP_MS - ENC_MS) / per)
    n_renew = math.floor((RENEW_MS - JITTER_MS - ENC_MS) / per)
    show("max evaluations within 25 ms step", n_step)
    show("  latency at that count, ms", ENC_MS + n_step * per)
    show("max evaluations within 50 ms less 5 ms jitter", n_renew)
    show("  latency at that count, ms", ENC_MS + n_renew * per)
    return per, flow


def reset_wall():
    head("Reset wall (NM 6.2)")
    day = 86400
    exe = TRIALS * ROLL_S / day
    auto = TRIALS * (ROLL_S + AUTO_S) / day
    man = TRIALS * (ROLL_S + MANUAL_S) / day
    show("execution only, days", f"{exe:.1f}")
    show("automated resets, days", f"{auto:.1f}")
    show("manual resets, days", f"{man:.1f}")
    show("reset share of a manual trial, %", f"{100 * MANUAL_S / (ROLL_S + MANUAL_S):.1f}")
    show("14-rig farm, manual, days", f"{man / RIGS:.1f}")
    show("drivetrain load reversals", f"{TRIALS * REVERSALS:.1e}")
    rate = ROBOTS * STEPS_PER * UPDATES / (RUDIN_MIN * 60)
    sim_steps = TRIALS * EPISODE_S / (STEP_MS / 1000)
    sim_min = sim_steps / rate / 60
    show("sim policy steps for 1e6 openings", f"{sim_steps:.1e}")
    show("sim time at reported rate, min", f"{sim_min:.1f}")
    hil = HIL_HOURS * 3600 / (ROLL_S + AUTO_S)
    show("HIL-SERL 2.5 h rollouts", f"{hil:.0f}")
    show("trial-count factor, 1e6 / 750", f"{TRIALS / hil:,.0f}")
    show("HIL-SERL load reversals", f"{hil * REVERSALS:,.0f}")
    return dict(exe=exe, auto=auto, man=man, farm=man / RIGS,
                sim_days=sim_min / 1440, hil_days=HIL_HOURS / 24, hil=hil)


def rudin():
    head("Simulation throughput slide: Rudin et al. 2022")
    batch = ROBOTS * STEPS_PER
    total = batch * UPDATES
    show("batch, transitions", f"{batch:,}")
    show("total policy steps", f"{total:.3g}")
    show("throughput, steps/s (at 20 min)", f"{total / (RUDIN_MIN * 60):.3g}")
    show("simulated days at 50 Hz", f"{total / RUDIN_HZ / 86400:.1f}")


def spring():
    head("Soft-spring exploit (NM 6.3)")
    k_pen = MASS * V_FAST ** 2 / PEN_M ** 2
    f_sim = MASS * V_FAST ** 2 / PEN_M
    f_rigid = V_FAST * math.sqrt(K_PLATE * MASS)
    k_eff_star = (F_LIMIT / V_FAST) ** 2 / MASS
    k_rob_star = 1 / (1 / k_eff_star - 1 / K_PLATE)
    f_guard = V_GUARD * math.sqrt(K_PLATE * MASS)
    show("penalty stiffness, N/m", f"{k_pen:.2g}")
    show("simulated peak, N", f"{f_sim:.1f}")
    show("rigid-arm peak, N", f"{f_rigid:.1f}")
    show("rigid peak over limit", f"{f_rigid / F_LIMIT:.1f}x")
    show("break-even series stiffness, N/m", f"{k_eff_star:.3g}")
    show("break-even robot stiffness, N/m", f"{k_rob_star:.3g}")
    show("guarded rigid peak, N", f"{f_guard:.1f}")
    show("plate over penalty stiffness", f"{K_PLATE / k_pen:.1f}x")
    head("Contact deadline (section 6.4.2)")
    for name, v in (("fast", V_FAST), ("guarded", V_GUARD)):
        t_lim = F_LIMIT / (K_PLATE * v) * 1e3
        t_trip = F_TRIP / (K_PLATE * v) * 1e3
        t_brake = t_trip + LOOP_MS
        show(f"{name}: limit reached, ms", f"{t_lim:.4g}")
        show(f"{name}: tripwire, ms", f"{t_trip:.4g}")
        show(f"{name}: brake onset, ms", f"{t_brake:.4g}")
        show(f"{name}: left before limit, ms", f"{t_lim - t_brake:.4g}")
    return dict(k_pen=k_pen, f_sim=f_sim, f_rigid=f_rigid,
                k_rob_star=k_rob_star, k_eff_star=k_eff_star)


def gap():
    head("Reality gap vector (section 6.5, illustrative)")
    show("unobserved travel at p99, mm", f"{GAP_V * GAP_P99_MS:.1f}")
    for name, r in zip(("simulated", "hardware"), GAP_RISE):
        show(f"{name}: time to 10 N, ms", f"{GAP_TARGET_N / r * 1e3:.0f}")


def residual():
    head("Update slide: residual fine-tuning (section 6.6)")
    show("active contact time, s", f"{FT_STEPS * STEP_MS / 1000:.0f}")
    show("above-tripwire samples", f"{FT_STEPS * FT_TRIP_FRAC:.0f}")


def cells():
    head("Pretraining and manifest slides")
    show("known absences", CELLS - SUPPORTED)


if __name__ == "__main__":
    dactyl()
    discount()
    decoder()
    reset_wall()
    rudin()
    spring()
    gap()
    residual()
    cells()
