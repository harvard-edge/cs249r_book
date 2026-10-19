"""Every derived number on the Lecture 0 slides (Volume IV, Physical AI).

Run:  /Users/.../MLSysBook/.venv/bin/python calcs.py   (needs numpy for MLSysIM)
Each block names the slide it serves, lists its inputs with their sources,
and prints each result with the label the slide uses. Inputs typed from a
source carry that source; only derived values are computed here.

The last block keeps the running machine's elementary stopping budget for the
stopping-budget bar generator in make_figs.py (a kit candidate). Lecture 0 no
longer shows that budget; Lecture 1 owns it.
"""

import sys
from pathlib import Path

MLSYSIM = Path(__file__).resolve().parents[4] / "mlsysim"

# --- Inputs: the running machine's card (The Causal Boundary, sec. 1.4.2, table 2)
V_DRIVE = 1.5        # m/s, drive limit (illustrative)
A_BRAKE = 2.0        # m/s^2, credible deceleration (illustrative)
D_CLEAR = 1.10       # m, rack-end clear distance (illustrative)
# Pre-brake delay terms (Napkin Math 1.1), ms: observation age, lease, tick, bus, brake onset
DELAY_TERMS = [51.6, 60.0, 1.0, 1.0, 20.0]

# --- Inputs: IFR World Robotics 2025 press release, Frankfurt, 25 Sep 2025
IFR_INSTALLED_2024 = 542_000        # "542,000 robots installed in 2024"
IFR_STOCK_2024 = 4_664_000          # "4,664,000 units in 2024" in operational use

# --- Inputs: UR16e datasheet (Universal Robots) and MiR250 specifications (MiR, 2023-06-28)
UR16E = {"payload kg": 16, "reach mm": 900, "joints": 6, "mass kg (with cable)": 33.1}
MIR250 = {"payload kg": 250, "mass kg": 97, "safety laser scanners": 2, "3D cameras": 2}


def tau_delay_ms():
    return sum(DELAY_TERMS)


def stop_mm(v, tau_ms=None, a=A_BRAKE):
    tau = (tau_delay_ms() if tau_ms is None else tau_ms) / 1000
    return v * tau * 1000, v ** 2 / (2 * a) * 1000


def waymo_rows():
    sys.path.insert(0, str(MLSYSIM))
    from mlsysim import Literature
    return [(r["data_through"], r["cumulative_ro_miles_millions"])
            for r in Literature.Series.WaymoRiderOnlyMiles.entries]


def waymo_last_year():
    rows = dict(waymo_rows())
    return rows["2026-06"] - rows["2025-06"]


def main():
    print("== Hook and plot: Waymo rider-only miles (Waymo Safety Impact Data Hub via MLSysIM)")
    rows = waymo_rows()
    for d, m in rows:
        print(f"   {d}: {m:.3f} million cumulative")
    first, last = rows[0], rows[-1]
    yr = waymo_last_year()
    print(f"miles July 2025 to June 2026: {yr:.3f} million (slide: about {yr:.0f} million)")
    print(f"hook choices (factor-ten ladder): {yr / 100:.2f} M, {yr / 10:.1f} M, {yr:.0f} M, {yr * 10 / 1000:.2f} B")
    print(f"per day over that year: {yr * 1e6 / 365:,.0f} miles")
    print(f"growth {first[0]} to {last[0]}: {last[1] / first[1]:.1f} times")
    print(f"first: {first[1]:.1f} million ; last: {last[1]:.1f} million")

    print("\n== Industrial robots (IFR World Robotics 2025)")
    print(f"installed 2024: {IFR_INSTALLED_2024:,} ; in operation: {IFR_STOCK_2024:,}")
    print(f"in operation, millions: {IFR_STOCK_2024 / 1e6:.2f}")
    print(f"stock per 2024 installation: {IFR_STOCK_2024 / IFR_INSTALLED_2024:.1f}")

    print("\n== Anatomy slide: datasheet values (typed, none derived)")
    print(f"UR16e: {UR16E}")
    print(f"MiR250: {MIR250}")

    print("\n== Course map")
    lectures = list(range(1, 19))
    print(f"lectures after this one: {len(lectures)} ; parts: 4 ; records: 16 (sec. 1.11 figure)")

    print("\n== Kit candidate only (stopping-budget bar; not shown in Lecture 0)")
    d1, d2 = stop_mm(V_DRIVE)
    print(f"pre-brake delay {tau_delay_ms():.1f} ms ; delay {d1:.1f} mm ; braking {d2:.1f} mm ; "
          f"stop {d1 + d2:.1f} of {D_CLEAR * 1000:.0f} mm")


if __name__ == "__main__":
    main()
