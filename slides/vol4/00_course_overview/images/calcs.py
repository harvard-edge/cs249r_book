"""Every derived number on the Lecture 0 slides (Volume IV, Physical AI).

Run:  python3 calcs.py
Each block names the slide it serves, lists its inputs with their sources,
and prints each result with the label the slide uses. Inputs are typed from
the rendered chapters; only derived values are computed here.
"""

import math
import sys
from pathlib import Path

MLSYSIM = Path(__file__).resolve().parents[4] / "mlsysim"

# --- Inputs: the running machine's card (The Causal Boundary, sec. 1.4.2, table 2)
V_DRIVE = 1.5        # m/s, drive limit (illustrative)
A_BRAKE = 2.0        # m/s^2, credible deceleration, loaded, dry floor (illustrative)
D_CLEAR = 1.10       # m, rack-end clear distance (illustrative)
MASS = 368           # kg, loaded mass (derived upper bound)

# --- Inputs: observation age stages (Sensor Perception, Napkin Math 8.2), ms
AGE_STAGES = [
    ("Half exposure", 8.0),      # 16 ms exposure, chosen
    ("Readout", 16.6),           # datasheet
    ("Link", 0.8),               # illustrative
    ("DMA queue", 1.2),          # illustrative
    ("ISP", 2.5),                # illustrative
    ("Backbone", 22.0),          # INT8 vision backbone, illustrative
    ("Dispatch", 0.5),           # illustrative
]

# --- Inputs: rest of the pre-brake delay (The Causal Boundary, Napkin Math 1.1), ms
LEASE = 60.0
TICK = 1.0
BUS = 1.0
BRAKE_ONSET = 20.0


def obs_age():
    return sum(v for _, v in AGE_STAGES)


def tau_delay_ms():
    return obs_age() + LEASE + TICK + BUS + BRAKE_ONSET


def stop_mm(v, tau_ms=None, a=A_BRAKE):
    tau = (tau_delay_ms() if tau_ms is None else tau_ms) / 1000
    return v * tau * 1000, v ** 2 / (2 * a) * 1000


def main():
    print("== Hook: one frame (Napkin Math 8.2)")
    age = obs_age()
    print(f"observation age at dispatch: {age:.1f} ms")
    print(f"travel during that age at {V_DRIVE} m/s: {V_DRIVE * age:.1f} mm")
    print(f"exposure + readout before any computation: {AGE_STAGES[0][1] + AGE_STAGES[1][1]:.1f} ms")
    for name, v in AGE_STAGES:
        print(f"   stage {name}: {v} ms -> {V_DRIVE * v:.1f} mm")

    print("\n== Worked example 1: does the base stop inside its clearance? (Napkin Math 1.1)")
    tau = tau_delay_ms()
    d1, d2 = stop_mm(V_DRIVE)
    print(f"pre-brake delay: {tau:.1f} ms")
    print(f"delay travel: {d1:.1f} mm")
    print(f"braking travel: {d2:.1f} mm")
    print(f"stop: {d1 + d2:.1f} mm of {D_CLEAR * 1000:.0f} mm")
    print(f"spare: {D_CLEAR * 1000 - d1 - d2:.1f} mm")
    print(f"braking / delay: {d2 / d1:.2f}x ; delay adds {100 * d1 / d2:.1f} percent to braking")
    vstar = 2 * A_BRAKE * tau / 1000
    print(f"crossover speed: {vstar:.4f} m/s (slide: {vstar:.2f} m/s)")
    print(f"share of stop used by delay: {100 * d1 / (d1 + d2):.1f} percent")

    print("\n== Activity: halve the speed")
    h1, h2 = stop_mm(V_DRIVE / 2)
    print(f"0.75 m/s: delay {h1:.1f} mm, braking {h2:.1f} mm, stop {h1 + h2:.1f} mm")
    print(f"stop shrinks by factor {(d1 + d2) / (h1 + h2):.2f}")
    print(f"wrong answer (halving both terms): {(d1 + d2) / 2:.1f} mm")

    print("\n== Pre-brake delay bar")
    backbone = dict(AGE_STAGES)["Backbone"]
    print(f"backbone {backbone:.0f} ms -> {V_DRIVE * backbone:.0f} mm")
    print(f"rest of observation age {age - backbone:.1f} ms")
    print(f"lease {LEASE:.0f} ms -> {V_DRIVE * LEASE:.0f} mm")
    print(f"tick + bus {TICK + BUS:.0f} ms")
    print(f"each added ms costs {V_DRIVE * 1:.1f} mm")
    print(f"backbone share of delay: {100 * backbone / tau:.1f} percent")
    ke = 0.5 * MASS * V_DRIVE ** 2
    print(f"kinetic energy at drive limit: {ke:.0f} J")

    print("\n== Three machine classes")
    # Napkin Math 10.1: tripwire 15 N, plate stiffness 400 N/mm, 0.03 m/s, 2 ms loop
    f_brake = 15 + 400 * (0.03 * 1000) * 0.002   # N/mm * mm/s * s
    print(f"latch force at brake command: {f_brake:.0f} N against 100 N")
    # Napkin Math 2.6: hold 87 N m, derated to 73.0 N m (chapter-derived)
    print(f"derating 87 -> 73.0 N m: {100 * (1 - 73.0 / 87):.1f} percent")

    print("\n== Clean runs (Closed-Loop Evaluation, Napkin Math 7.1, 7.2)")
    alpha = 0.05
    lo20 = alpha ** (1 / 20)
    print(f"20 clean runs: lower limit {100 * lo20:.1f} percent at 95 percent")
    for p in (0.9, 0.99, 0.999):
        n = math.ceil(math.log(alpha) / math.log(p))
        print(f"target {p}: {n:,} clean runs")
    trial_s = 25 + 15   # s per trial: 25 s run + 15 s reset (illustrative)
    print(f"20 runs: {20 * trial_s / 60:.1f} min ; 2,995 runs: {2995 * trial_s / 3600:.1f} h")
    print(f"299 runs: {299 * trial_s / 3600:.1f} h")

    print("\n== RAND against Waymo rider-only miles (sec. 7.9.1)")
    rand_m, waymo_m = 275e6, 56.7e6
    print(f"275 M / 56.7 M = {rand_m / waymo_m:.1f}x")

    print("\n== Epoch AI compute (sec. 1.2; mlsysim Literature.ModelsByDomain)")
    sys.path.insert(0, str(MLSYSIM))
    from mlsysim import Literature
    rows = Literature.ModelsByDomain.entries

    def year_max(group, year):
        return max(r["training_flop"] for r in rows
                   if r["domain_group"] == group and str(r["publication_date"]).startswith(year))

    def model(name):
        return next(r for r in rows if r["model"] == name and r["domain_group"] == "Robotics")

    for name in ("OpenVLA", "GR00T N1 2B"):
        m = model(name)
        g = year_max("Language", m["publication_date"][:4]) / m["training_flop"]
        print(f"{name} ({m['publication_date'][:4]}): {m['training_flop']:.2e} FLOP; gap {g:,.0f}x")
    print(f"robotics models with compute: {sum(r['domain_group'] == 'Robotics' for r in rows)}")

    print("\n== California DMV filings, Waymo (sec. 14.1; Literature.CaliforniaDisengagements)")
    dm = [r for r in Literature.CaliforniaDisengagements.entries
          if r["company"] == "Waymo" and r["filer_type"] in ("annual", "first_report")]

    def rate(year, permit):
        return next(r for r in dm if r["report_year"] == year and r["permit_type"] == permit)

    f, l = rate(2015, "drivered_AVT"), rate(2025, "drivered_AVT")
    d = rate(2025, "driverless_AVDT")
    print(f"2015 safety driver: {f['miles_per_disengagement']:,.0f} mi/dis")
    print(f"2025 safety driver: {l['miles_per_disengagement']:,.0f} mi/dis, {l['disengagements']:.0f} takeovers")
    print(f"2025 driverless: {d['miles_per_disengagement']:,.0f} mi/dis, {d['disengagements']:.0f} disengagements")
    for y in (2015, 2020, 2021, 2025):
        r = rate(y, "drivered_AVT")
        print(f"  explore row {y}: miles {r['autonomous_miles']:,.1f} diseng {r['disengagements']:.0f} "
              f"mi/diseng {r['miles_per_disengagement']:,.1f}")
    print(f"2020 to 2021 miles growth: {rate(2021, 'drivered_AVT')['autonomous_miles'] / rate(2020, 'drivered_AVT')['autonomous_miles']:.1f}x")
    print(f"improvement 2015 to 2025: {l['miles_per_disengagement'] / f['miles_per_disengagement']:.1f}x")

    print("\n== Course")
    print("lectures 18 + overview; records 16 (sec. 1.11 figure)")


if __name__ == "__main__":
    main()
