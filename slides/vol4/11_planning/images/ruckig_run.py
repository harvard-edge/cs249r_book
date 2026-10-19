#!/usr/bin/env python3
"""Run the Ruckig online trajectory generator on the lecture's two stops.

Ruckig (Berscheid and Kroger, RSS 2021; pip package `ruckig`, MIT license) is
the jerk-limited generator the chapter names in Sec. 11.4.5 and table 4. This
script asks it for a stop to zero velocity and acceleration from cruise, with
the chapter's bounds, and writes its sampled output to ruckig_stops.csv, which
make_figs.py plots. The plot is the tool's own output, not a redrawn formula.

  arm:  1.0 m/s, 5 m/s^2, 66.7 m/s^3 (TP Napkin Math 11.4)
  base: 1.5 m/s, 2 m/s^2, 7.11 m/s^3 (TP Sec. 11.5.4)

Needs `pip install ruckig` (tested with ruckig 0.19.4). Run from this folder:
    python3 ruckig_run.py
"""

import csv
from importlib.metadata import version
from pathlib import Path

from ruckig import ControlInterface, InputParameter, Result, Ruckig, Trajectory

DT = 0.001                      # s, the arm's 1 kHz control cycle (TP Sec. 11.1)


def poly_jerk(v, a):
    """Peak jerk of the chapter's equation 16 stop at peak deceleration a."""
    return 6 * v / (1.5 * v / a) ** 2


def stop(v, a):
    inp = InputParameter(1)
    inp.control_interface = ControlInterface.Velocity
    inp.current_position = [0.0]
    inp.current_velocity = [v]
    inp.current_acceleration = [0.0]
    inp.target_velocity = [0.0]
    inp.target_acceleration = [0.0]
    inp.max_acceleration = [a]
    inp.max_jerk = [poly_jerk(v, a)]
    traj = Trajectory(1)
    assert Ruckig(1).calculate(inp, traj) == Result.Working
    n = int(round(traj.duration / DT))
    rows = []
    for k in range(n + 1):
        t = min(k * DT, traj.duration)
        p, vel, acc = traj.at_time(t)
        rows.append((t, p[0], vel[0], acc[0]))
    return traj.duration, rows


out = Path(__file__).parent / "ruckig_stops.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "t_s", "position_m", "velocity_mps", "acceleration_mps2"])
    for name, v, a in (("arm", 1.0, 5.0), ("base", 1.5, 2.0)):
        dur, rows = stop(v, a)
        print(f"{name}: ruckig {version('ruckig')} duration {1000 * dur:.2f} ms, "
              f"distance {1000 * rows[-1][1]:.2f} mm")
        for r in rows:
            w.writerow([name, f"{r[0]:.4f}", f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}"])
print(f"wrote {out.name}")
