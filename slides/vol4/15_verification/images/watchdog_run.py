#!/usr/bin/env python3
"""Freeze a control task and see which watchdog notices (our run).

A small model of the F1 injection and of the task-blind watchdog hypothesis
in Bookout v. Toyota (Barr 2013): a control task issues a command every
10 ms; at t = 500 ms the harness freezes it while its timer keeps running.
Watchdog A is fed by the periodic timer, as the testimony described.
Watchdog B is fed only when the control task finishes a cycle. Both bite
after 60 ms without a feed. Threads on a laptop stand in for an interrupt
and a task; the timing is this machine's, not a vehicle's.

Writes data/watchdog_run.log. Run from this folder: python3 watchdog_run.py
"""

import threading
import time
from pathlib import Path

PERIOD, TIMEOUT, FREEZE_AT, END = 0.010, 0.060, 0.500, 0.800
t0 = time.perf_counter()
now = lambda: time.perf_counter() - t0  # noqa: E731
lines = []
lock = threading.Lock()
fed = {"A": 0.0, "B": 0.0}
bitten = {}
frozen = threading.Event()
command = {"value": 0.0, "at": 0.0}


def log(msg):
    with lock:
        lines.append(f"[{now()*1e3:7.1f} ms] {msg}")


def control_task():
    k = 0
    while now() < END:
        if now() >= FREEZE_AT and not frozen.is_set():
            frozen.set()
            log("HARNESS  control task frozen (deadlock injected); timer left running")
            threading.Event().wait()          # never returns: the task is dead
        k += 1
        command["value"], command["at"] = round(0.1 * k, 1), now()
        fed["B"] = now()                       # progress check: a finished cycle
        time.sleep(PERIOD)


def timer_tick():
    while now() < END + 0.1:
        fed["A"] = now()                       # serviced from the timer, task-blind
        time.sleep(PERIOD)


def watchdogs():
    last_report = 0.0
    while now() < END + 0.1:
        t = now()
        for name in ("A", "B"):
            if name not in bitten and t - fed[name] > TIMEOUT:
                bitten[name] = t
                log(f"WATCHDOG {name}  BITE: no feed for {(t - fed[name])*1e3:.0f} ms -> stop commanded")
        if t - last_report >= 0.100:
            last_report = t
            age = (t - command["at"]) * 1e3
            log(f"STATUS   cmd={command['value']:4.1f} age={age:5.0f} ms  "
                f"A={'BITTEN' if 'A' in bitten else 'ok'}  B={'BITTEN' if 'B' in bitten else 'ok'}")
        time.sleep(0.001)


threads = [threading.Thread(target=f, daemon=True) for f in (control_task, timer_tick, watchdogs)]
for th in threads:
    th.start()
time.sleep(END + 0.15)
out = Path(__file__).resolve().parent / "data" / "watchdog_run.log"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
