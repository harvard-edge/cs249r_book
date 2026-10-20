#!/usr/bin/env python3
"""Every number on the Volume III Lecture 8 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the chapter's system-call tax
and Napkin Math 8.1 to 8.3 are Scenarios.* backed by the solvers in
engine/solvers/sandboxes.py (the chapter reads the same scenarios); the cited
results are Literature entries (Cybench, the gVisor system-call latencies of
Young et al. 2019, and Firecracker's released boot CDFs, read through
Scenarios.FirecrackerPreconfiguredBoot*). The one other source is the deck's
own runs, written by measure.py into data/ (escape.txt, clone.csv) and read
as recorded. This file only selects values, converts units, and formats them;
it types no numbers. Each value is recorded with v() under the key the deck
prints as \\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import csv
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CYBENCH = Literature.CybenchSolveRate.entries
ISO = Literature.SandboxIsolation
TAX = Scenarios.SyscallTaxPerMillionCalls
SETUP = Scenarios.ParserCallsEnvelopeSetup
WORK = Scenarios.CopyOnWriteRepairWorkspaces
POOL = Scenarios.CodingSandboxPoolTwoPerSecond
BOOT1 = Scenarios.FirecrackerPreconfiguredBootSerial
BOOTN = Scenarios.FirecrackerPreconfiguredBootHundred

SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def mag(q, unit):
    return q.to(unit).magnitude


def recorded(fn):
    """Rows of one of our own runs (data/), skipping the # header lines."""
    with open(DATA / fn) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


print("== Cybench offensive capability (Literature.CybenchSolveRate)")
first_2025 = "2025-01-01"
v("cy_tasks", ISO.CybenchTasks, "{:.0f}", "Cybench tasks")
v("cy_best_2024", max(r["unguided_solved"] for r in CYBENCH if r["release_date"] < first_2025) * 100,
  "{:.1f}", "best unguided solve rate among 2024 releases, %")
v("cy_best_latest", max(r["unguided_solved"] for r in CYBENCH) * 100, "{:.0f}",
  "best unguided solve rate to date, %")

print("== The in-process escape, our run (data/escape.txt)")
escape = (DATA / "escape.txt").read_text()
v("esc_names", int(re.search(r"dict with (\d+) names", escape).group(1)), "{:.0f}",
  "names in the supervisor's module")
v("esc_classes", int(re.search(r"(\d+) classes reachable", escape).group(1)), "{:.0f}",
  "classes reachable from an empty tuple")

print("== System-call tax (Scenarios.SyscallTaxPerMillionCalls)")
t = TAX.result
exponent = round(math.log10(TAX.calls))
assert 10 ** exponent == TAX.calls, "the axis label writes the call count as a power of ten"
v("tax_calls", TAX.calls, "{:,.0f}", "ordinary calls")
v("tax_calls_exp", exponent, "{:.0f}", "the same, as a power of ten")
v("tax_container", mag(t.container_total, "s"), "{:.2f}", "container, added s")
v("tax_microvm", mag(t.microvm_total, "s"), "{:.2f}", "microVM, added s")
v("tax_userspace", mag(t.userspace_total, "s"), "{:.1f}", "user-space kernel, added s")
v("tax_ratio", t.userspace_vs_microvm, "{:.0f}", "user-space kernel / microVM")

print("== gVisor gettimeofday latency (Literature.SandboxIsolation, Young et al. 2019 fig 3)")
v("gv_native", ISO.GvisorNativeGettimeofdayUs, "{:.2f}", "native, us")
v("gv_runc", ISO.GvisorRuncGettimeofdayUs, "{:.2f}", "runc, us")
v("gv_sentry", ISO.GvisorSentryGettimeofdayUs, "{:.2f}", "Sentry on KVM, us")
v("gv_host", ISO.GvisorHostGettimeofdayUs, "{:.2f}", "invoked on the host, us")
v("gv_gofer", ISO.GvisorGoferGettimeofdayUs, "{:.1f}", "through the Gofer, us")
v("gv_gofer_x", ISO.GvisorGoferGettimeofdayUs / ISO.GvisorNativeGettimeofdayUs, "{:.0f}",
  "Gofer / native")

print("== Napkin Math 8.1 (Scenarios.ParserCallsEnvelopeSetup)")
s = SETUP.result
v("su_calls", SETUP.calls, "{:.0f}", "parser calls")
v("su_setup", mag(SETUP.setup, "ms"), "{:.0f}", "envelope setup, ms")
v("su_work", mag(SETUP.work_per_call, "ms"), "{:.0f}", "useful work per call, ms")
v("su_fresh_each", mag(SETUP.setup + SETUP.work_per_call, "ms"), "{:.0f}", "one fresh call, ms")
v("su_useful", mag(s.useful_work, "ms"), "{:.0f}", "useful work in all, ms")
v("su_fresh", mag(s.per_call_total, "ms"), "{:,.0f}", "fresh per call, ms")
v("su_reuse", mag(s.per_trajectory_total, "ms"), "{:.0f}", "retained for the trajectory, ms")
v("su_ratio", s.reuse_speedup, "{:.1f}", "fresh / retained")
v("su_share", s.setup_share * 100, "{:.1f}", "setup share of a fresh call, %")
v("su_cap", SETUP.overhead_cap * 100, "{:.0f}", "setup cap, %")
v("su_cap_work", mag(s.work_for_cap, "ms"), "{:.0f}", "work per call for the cap, ms")

print("== Napkin Math 8.2 (Scenarios.CopyOnWriteRepairWorkspaces)")
w = WORK.result
v("wk_n", WORK.workspaces, "{:.0f}", "workspaces")
v("wk_base", mag(WORK.base_image, "GB"), "{:.1f}", "pinned base, GB")
v("wk_written", mag(WORK.written, "MB"), "{:.0f}", "written per task, MB")
v("wk_full", mag(w.full_copy, "GB"), "{:.1f}", "full copy, GB")
v("wk_cow", mag(w.cow_written, "GB"), "{:.2f}", "copy-on-write, GB")
v("wk_ratio", w.copy_vs_cow, "{:.0f}", "full / copy-on-write")

print("== Clone against copy, our run (data/clone.csv)")
clone = {r["operation"]: float(r["ms"]) for r in recorded("clone.csv")}
base_line = next(l for l in (DATA / "clone.csv").read_text().splitlines() if "base file" in l)
clone_base = float(re.search(r"base file ([\d.]+) GB", base_line).group(1))
assert f"{clone_base:.1f}" == f"{mag(WORK.base_image, 'GB'):.1f}", "the run cloned the scenario's base"
v("cl_base", clone_base, "{:.1f}", "base file cloned, GB")
v("cl_full", clone["full copy"], "{:.0f}", "full copy, ms")
v("cl_clone", clone["copy-on-write clone (cp -c)"], "{:.0f}", "clone, ms")
v("cl_write", clone["1-byte change to the clone"], "{:.1f}", "1-byte change to the clone, ms")

print("== Firecracker boot (Literature.FirecrackerBootPreconfigured)")
b1, bn = BOOT1.result, BOOTN.result
v("fc_serial_p50", mag(b1.median, "ms"), "{:.0f}", "one at a time, median, ms")
v("fc_par_n", bn.booting, "{:.0f}", "booting at once")
v("fc_par_p99", mag(bn.tail, "ms"), "{:.0f}", "many at once, p99, ms")

print("== Napkin Math 8.3 (Scenarios.CodingSandboxPoolTwoPerSecond)")
p = POOL.result
v("pl_rate", mag(POOL.arrival_rate, "1/s"), "{:.0f}", "trajectories per second")
v("pl_hold", mag(POOL.hold, "s"), "{:.0f}", "lease hold, s")
v("pl_active", p.active, "{:,.0f}", "active leases")
v("pl_restore", mag(POOL.restore, "ms"), "{:.0f}", "snapshot restore, ms")
v("pl_target", POOL.target_miss * 100, "{:.1f}", "miss target, %")
v("pl_reserve_words", p.reserve, "words", "smallest reserve")
v("pl_max_restore", math.floor(mag(p.max_restore, "ms")), "{:.0f}",
  "restore the reserve tolerates, ms (floored, as the chapter)")
v("pl_slow_words", POOL.slow_factor, "words", "cold boot / restore")
v("pl_slow_reserve_words", p.slow_reserve, "words", "reserve at the slower boot")
v("pl_private", mag(p.private_memory, "GB"), "{:.1f}", "private images, GB")
v("pl_cow", mag(p.pool_memory, "GB"), "{:.1f}", "copy-on-write pool, GB")
v("pl_ratio", p.memory_ratio, "{:.1f}", "private / copy-on-write")

if __name__ == "__main__":
    write(__file__)
