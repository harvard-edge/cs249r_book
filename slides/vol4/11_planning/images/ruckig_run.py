#!/usr/bin/env python3
"""Run the Ruckig online trajectory generator on the lecture's two stops.

Ruckig (Berscheid and Kroger, RSS 2021; pip package `ruckig`, MIT license) is
the jerk-limited generator the chapter names in Sec. 11.4.5 and table 4. This
script asks it for a stop to zero velocity and acceleration from cruise, with
the chapter's bounds, and writes its sampled output to ruckig_stops.csv, which
make_figs.py plots. The plot is the tool's own output, not a redrawn formula.

  arm:  the TCP limit, its deceleration bound, and its jerk bound (TP Napkin Math 11.4)
  base: the drive limit, a_brake, and its jerk bound (TP Sec. 11.5.4)
Every input comes from MLSysIM through calcs.py (Scenarios.WarehouseArmStallStop
and Scenarios.WarehouseBaseJerkLimitedStop).

Needs `pip install ruckig` (tested with ruckig 0.19.4). Run from this folder:
    python3 ruckig_run.py
"""

import contextlib
import csv
import io
import sys
from importlib.metadata import version
from pathlib import Path

from ruckig import ControlInterface, InputParameter, Result, Ruckig, Trajectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import calcs as K  # noqa: E402  every input comes from MLSysIM through calcs.py

DT = K.T_TICK / 1000            # s, the arm's servo tick (TP Sec. 11.1)


def stop(v, a, j):
    inp = InputParameter(1)
    inp.control_interface = ControlInterface.Velocity
    inp.current_position = [0.0]
    inp.current_velocity = [v]
    inp.current_acceleration = [0.0]
    inp.target_velocity = [0.0]
    inp.target_acceleration = [0.0]
    inp.max_acceleration = [a]
    inp.max_jerk = [j]
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
    for name, v, a, j in (("arm", K.V_TCP, K.A_ARM, K.J_ARM), ("base", K.V_BASE, K.A_BASE, K.J_BASE)):
        dur, rows = stop(v, a, j)
        print(f"{name}: ruckig {version('ruckig')} duration {1000 * dur:.2f} ms, "
              f"distance {1000 * rows[-1][1]:.2f} mm")
        for r in rows:
            w.writerow([name, f"{r[0]:.4f}", f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}"])
print(f"wrote {out.name}")
