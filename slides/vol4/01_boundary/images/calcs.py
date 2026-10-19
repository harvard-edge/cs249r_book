#!/usr/bin/env python3
"""Every derived number in Lecture 1 (The Causal Boundary), Volume IV.

One block per slide, in deck order. Inputs carry their source as a comment
("ch" is the rendered chapter, The Causal Boundary; "NTSB" is the Tempe
report HAR-19/03, read from the report itself). Each printed line is a value
the slide or its note shows, under the label used there. make_figs.py imports
from this file so the figures and the slides share one computation.

Run from this folder: python3 calcs.py
"""
import math
import sys
from pathlib import Path

MPH = 0.44704          # m/s per mile per hour (exact)
FT = 0.3048            # m per foot (exact)
G0 = 9.80665           # m/s^2, standard gravity


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
# The running machine (ch §1.4.2, table 2; Napkin Math 1.1).
V = 1.5                # m/s, drive limit (illustrative)
A_BRAKE = 2.0          # m/s^2, credible deceleration (illustrative)
D_CLEAR = 1100.0       # mm, rack-end clear distance (illustrative)
AGE = 51.6             # ms, observation age of the frame that shows the worker
BACKBONE = 22.0        # ms, perception backbone, part of the observation age
LEASE = 60.0           # ms, chunk lease (chosen)
TICK = 1.0             # ms, one permission tick
BUS = 1.0              # ms, one bus cycle
ONSET = 20.0           # ms, brake onset
MU_PREMISE = 0.204     # ch §1.4.2 and 1.9: friction the 2 m/s^2 rests on
MU_DRY, MU_SLICK = 0.8, 0.2   # ch §1.9: dry concrete, hydraulic fluid (illustrative)

# Five-level rates (ch §1.7, figure 7).
RATE_INTENT_HZ = 2
RATE_CURRENT_HZ = 25_000

# Tempe (NTSB HAR-19/03, "Selected parameters recorded by vehicle's ADS",
# pp. 15-16; footnotes 33 and 91; §2.1.2).
T_DETECT = 5.6         # s before impact, first detection (radar)
T_EMERG = 1.2          # s before impact, emergency recognized; suppression begins
T_SUPP_END = 0.2       # s before impact, suppression ends
V_EMERG_MPH = 43.2     # mph at -1.2 s
V_SUPP_END_MPH = 40.5  # mph at -0.2 s
V_DETECT_MPH = 44.0    # mph, "initial vehicle speed of 44 mph" (footnote 91)
G_ROAD = 0.92          # g, road capability measured by Tempe police (footnote 33)
G_ADS = 0.71           # g, the ADS's own braking cap (footnote 33)
DETECT_FT = 350        # ft from the pedestrian at first detection (§2.1.2)

# Epoch AI training compute (mlsysim Literature.ModelsByDomain, Epoch AI 2026),
# the largest language model of each year against the open robot policy the
# chapter names (§1.2).
REPO = Path(__file__).resolve().parents[4]


def epoch_rows():
    sys.path.insert(0, str(REPO / "mlsysim"))
    from mlsysim import Literature  # noqa: E402
    return Literature.ModelsByDomain.entries


def tau_s(lease=LEASE):
    return (AGE + lease + TICK + BUS + ONSET) / 1000.0


def stop_mm(v=V, lease=LEASE, a=A_BRAKE):
    t = tau_s(lease)
    return v * t * 1000.0, v * v / (2 * a) * 1000.0


def stopping_distance(v_ms, g):
    a = g * G0
    return v_ms * v_ms / (2 * a), v_ms / a


def main():
    head("Hook: Tempe at first detection (NTSB)")
    show("350 ft in m", f"{DETECT_FT * FT:.0f} m")
    d_full, t_full = stopping_distance(V_DETECT_MPH * MPH, G_ROAD)
    show("full stop from 44 mph at 0.92 g (distance)", f"{d_full:.1f} m (derived)")
    show("full stop from 44 mph at 0.92 g (time; NTSB says 2.1 s)", f"{t_full:.2f} s")
    show("detection distance / full stop", f"{DETECT_FT * FT / d_full:.1f}x")

    head("Recall and budget slide (Napkin Math 1.1)")
    t = tau_s()
    dd, db = stop_mm()
    show("pre-brake delay tau", f"{t * 1000:.1f} ms")
    show("delay travel v*tau", f"{dd:.1f} mm")
    show("braking travel v^2/2a", f"{db:.1f} mm")
    show("stop", f"{dd + db:.1f} mm of {D_CLEAR:.0f} mm")
    show("spare", f"{D_CLEAR - dd - db:.1f} mm")
    show("rest of observation age (age - backbone)", f"{AGE - BACKBONE:.1f} ms")
    show("tick + bus", f"{TICK + BUS:.0f} ms")
    show("travel per ms at the drive limit", f"{V:.1f} mm/ms")
    show("backbone travel", f"{V * BACKBONE:.0f} mm")
    show("lease travel", f"{V * LEASE:.0f} mm")
    show("braking / delay", f"{db / dd:.1f}x")
    show("delay adds to braking", f"{dd / db * 100:.1f} percent")

    head("Crossover slide")
    v_star = 2 * A_BRAKE * t
    show("crossover v* = 2 a tau", f"{v_star:.2f} m/s")
    show("delay travel at v* (= braking travel)", f"{v_star * t * 1000:.0f} mm")
    # Fastest speed whose elementary stop fits 1.10 m: v^2/(2a) + v tau = D.
    D = D_CLEAR / 1000.0
    v_max = A_BRAKE * (-t + math.sqrt(t * t + 2 * D / A_BRAKE))
    show("elementary speed ceiling for 1.10 m", f"{v_max:.2f} m/s")

    head("Estimate slide")
    hd, hb = stop_mm(v=V / 2)
    show("half speed", f"{V / 2:.2f} m/s")
    show("half-speed delay travel", f"{hd:.1f} mm")
    show("half-speed braking travel", f"{hb:.1f} mm")
    show("half-speed stop", f"{hd + hb:.1f} mm")
    ld, lb = stop_mm(lease=2 * LEASE)
    show("doubled lease, tau", f"{tau_s(2 * LEASE) * 1000:.1f} ms")
    show("doubled lease, delay travel", f"{ld:.1f} mm")
    show("doubled lease, stop", f"{ld + lb:.1f} mm")
    show("doubled lease, spare", f"{D_CLEAR - ld - lb:.1f} mm")

    head("Tempe through the budget (NTSB inputs, derived)")
    v1, v2 = V_EMERG_MPH * MPH, V_SUPP_END_MPH * MPH
    supp = (T_EMERG - T_SUPP_END) * (v1 + v2) / 2
    show("speed at -1.2 s", f"{v1:.1f} m/s")
    show("speed at -0.2 s", f"{v2:.1f} m/s")
    show("travel during 1 s suppression", f"{supp:.1f} m")
    d_road, _ = stopping_distance(v1, G_ROAD)
    d_ads, _ = stopping_distance(v1, G_ADS)
    show("full stop from 43.2 mph at 0.92 g", f"{d_road:.1f} m")
    show("full stop from 43.2 mph at 0.71 g", f"{d_ads:.1f} m")
    show("0.92 g in m/s^2", f"{G_ROAD * G0:.1f} m/s^2")
    show("0.71 g in m/s^2", f"{G_ADS * G0:.1f} m/s^2")
    vstar_t = 2 * G_ROAD * G0 * (T_EMERG - T_SUPP_END)
    show("crossover for a 1 s delay at 0.92 g", f"{vstar_t:.1f} m/s ({vstar_t / MPH:.0f} mph)")

    head("Scanner path slide")
    leave = BACKBONE + LEASE
    show("terms that leave the delay (backbone + lease)", f"{leave:.0f} ms")
    show("travel they cost at the drive limit", f"{V * leave:.0f} mm")

    head("Five levels slide")
    show("intent 2 Hz to current loop 25 kHz", f"{math.log10(RATE_CURRENT_HZ / RATE_INTENT_HZ):.1f} orders of magnitude")

    head("Convergence slide (slick floor)")
    show("braking distance ratio, dry 0.8 to slick 0.2", f"{MU_DRY / MU_SLICK:.0f}x")
    show("traction ceiling at mu = 0.2 (all weight on driven wheels)",
         f"{MU_SLICK * G0:.2f} m/s^2 against the card's {A_BRAKE:.0f}")
    show("traction the card's 2 m/s^2 needs", f"mu >= {A_BRAKE / G0:.3f} (card: {MU_PREMISE})")

    head("Epoch AI slide (mlsysim registry)")
    rows = epoch_rows()
    lang = [r for r in rows if r["domain_group"] == "Language"]
    robo = [r for r in rows if r["domain_group"] == "Robotics" and r["training_flop"]]
    def top(year):
        return max((r for r in lang if str(r["publication_date"]).startswith(year)),
                   key=lambda r: r["training_flop"])
    ov = next(r for r in robo if r["model"] == "OpenVLA")
    gr = next(r for r in robo if r["model"] == "GR00T N1 2B")
    l24, l25 = top("2024"), top("2025")
    show("2024 gap", f"{l24['model']} / OpenVLA = {l24['training_flop'] / ov['training_flop']:.0f}x")
    show("2025 gap", f"{l25['model']} / GR00T N1 = {l25['training_flop'] / gr['training_flop']:.0f}x")
    show("robotics models with a compute estimate", f"{len(robo)}")


if __name__ == "__main__":
    main()
