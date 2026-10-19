#!/usr/bin/env python3
"""Every derived number in Lecture 8 (Agent Sandboxes).

One block per slide, in deck order. Inputs carry their source as a comment
("ch" is the rendered chapter, Agent Sandboxes). Scenario inputs that the
chapter labels hypothetical are marked "illustrative"; only numbers derived
from them by arithmetic are computed here, and the slide that shows one says
"illustrative". make_figs.py imports this file so figures and slides share
one computation.

Run from this folder: python3 calcs.py
"""
import csv
import math
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Cybench: measured offensive capability (real data, not a scenario)
# ch §8.2, figure 1; MLSysIM Literature.CybenchSolveRate, via Epoch AI.
# ---------------------------------------------------------------------------
def cybench():
    head("Cybench offensive capability (ch figure 1; MLSysIM)")
    rows = []
    with open(DATA / "cybench_unguided.csv") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["release_date"], float(r["unguided_solved"]), r["source"]))
    best_2024 = max(v for d, v, s in rows if d < "2025-01-01")
    best_latest = max(v for d, v, s in rows)
    show("best unguided 2024 (percent)", f"{best_2024 * 100:.1f}")
    show("best unguided latest (percent)", f"{best_latest * 100:.0f}")
    show("n models", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Napkin Math 8.1: paying envelope setup once or per call (ch §8.5.5)
# Illustrative sandbox profile: setup 15 ms, useful work 3 ms, 100 calls.
# ---------------------------------------------------------------------------
def setup_amortization():
    head("NM 8.1 envelope setup (ch §8.5.5, illustrative)")
    setup, work, n = 15.0, 3.0, 100           # ms, ms, calls (illustrative)
    cap = 0.10                                 # hold setup to 10 percent of a call
    fresh = n * (setup + work)
    reuse = setup + n * work
    ratio = fresh / reuse
    share = setup / (setup + work)
    work_for_cap = setup * (1 - cap) / cap
    show("fresh per call (ms)", f"{fresh:.0f}")
    show("retained for trajectory (ms)", f"{reuse:.0f}")
    show("ratio", f"{ratio:.1f}")
    show("setup share of a fresh call (percent)", f"{share * 100:.1f}")
    show("useful work per call for 10 percent cap (ms)", f"{work_for_cap:.0f}")
    return dict(setup=setup, work=work, n=n, share=share, cap_work=work_for_cap)


# ---------------------------------------------------------------------------
# The system-call tax by envelope (ch §8.5.4, equation 3; illustrative)
# ---------------------------------------------------------------------------
def syscall_tax():
    head("System-call tax per 1,000,000 ordinary calls (ch §8.5.4, illustrative)")
    trap, filt, emul, switch = 0.1, 0.05, 1.5, 2.0     # microseconds (illustrative)
    calls = 1_000_000
    us = 1e-6
    container = calls * (trap + filt) * us
    microvm = calls * trap * us                        # device exits counted separately
    userspace = calls * (trap + emul + switch) * us
    show("container added time (s)", f"{container:.2f}")
    show("microVM added time (s)", f"{microvm:.2f}")
    show("user-space kernel added time (s)", f"{userspace:.1f}")
    show("user-space vs microVM", f"{userspace / microvm:.0f}x")
    return dict(container=container, microvm=microvm, userspace=userspace,
                ratio=userspace / microvm)


# ---------------------------------------------------------------------------
# Napkin Math 8.2: provisioning by copy, and first-modification granularity
# ch §8.6.2 (illustrative): base 2.4 GB, 40 MB written per task, 32 tasks,
# 7 GB/s drive; 1.2 GB fixture, 64 KB block.
# ---------------------------------------------------------------------------
def provisioning():
    head("NM 8.2 provisioning and first write (ch §8.6.2, illustrative)")
    base_gb, write_mb, n, rate = 2.4, 40.0, 32, 7.0    # GB, MB, tasks, GB/s
    full = n * base_gb
    cow = n * write_mb / 1000                           # MB -> GB
    show("full copy payload (GB)", f"{full:.1f}")
    show("full copy drive time (s)", f"{full / rate:.1f}")
    show("copy-on-write payload (GB)", f"{cow:.2f}")
    show("copy-on-write drive time (s)", f"{cow / rate:.2f}")
    show("provisioning ratio", f"{full / cow:.0f}x")
    fixture_gb, block_kb = 1.2, 64.0
    file_payload_gb = n * fixture_gb
    block_payload_kb = n * block_kb
    ratio = (fixture_gb * 1e9) / (block_kb * 1e3)
    show("file-granularity first-write payload (GB)", f"{file_payload_gb:.1f}")
    show("block-granularity first-write payload (KB)", f"{block_payload_kb:.0f}")
    show("first-write ratio", f"{ratio:,.0f}x")
    return dict(full=full, cow=cow, ratio=full / cow, first_ratio=ratio)


# ---------------------------------------------------------------------------
# Napkin Math 8.3: sizing a pool from its arrival rate (ch §8.8.3)
# Illustrative: 2 trajectories/s, 300 s lease, 15 ms restore, 512 MB image,
# 64 MB private per lease, 8 MB per standby; miss target 0.1 percent.
# ---------------------------------------------------------------------------
def pool_sizing():
    head("NM 8.3 pool sizing (ch §8.8.3, illustrative)")
    lam, hold, restore = 2.0, 300.0, 0.015              # /s, s, s
    image_mb, private_mb, standby_mb = 512.0, 64.0, 8.0
    target = 1e-3
    active = lam * hold
    rho = lam * restore                                 # arrivals during one restore
    # request-level miss = Poisson tail at or above the reserve S (mean rho)
    def poisson_tail(S, mean):
        # P(N >= S) = 1 - sum_{k<S} e^-m m^k/k!
        cdf = sum(math.exp(-mean) * mean**k / math.factorial(k) for k in range(S))
        return 1 - cdf
    miss1 = poisson_tail(1, rho)
    miss2 = poisson_tail(2, rho)
    # smallest reserve meeting the target
    S = 0
    while poisson_tail(S, rho) > target:
        S += 1
    # memory: private images vs copy-on-write (eq 6). The chapter's worked
    # notebooks quote configured guest memory as a binary capacity in
    # "conventional megabytes" (ch §8.12 note), so 512 MB is 512 MiB. Private
    # images count the full binary image (322.1 GB decimal); copy-on-write
    # reports the MiB sum as decimal GB (ch Notebook 8.3 and the fallacy).
    MiB = 2**20
    private_images_gb = active * image_mb * MiB / 1e9
    cow_gb = (image_mb + active * private_mb + S * standby_mb) / 1000
    # restore time at which reserve of 2 just meets the target
    r = restore
    while poisson_tail(2, lam * r) <= target:
        r += 0.0005
    show("active leases (mean)", f"{active:.0f}")
    show("arrivals per restore (rho)", f"{rho:.3f}")
    show("miss with 1 standby (percent)", f"{miss1 * 100:.2f}")
    show("miss with 2 standby (percent)", f"{miss2 * 100:.3f}")
    show("smallest reserve meeting 0.1 percent", S)
    show("private images memory (GB)", f"{private_images_gb:.1f}")
    show("copy-on-write memory (GB)", f"{cow_gb:.1f}")
    show("memory ratio", f"{private_images_gb / cow_gb:.1f}x")
    show("reserve 2 holds while restore below (ms)", f"{(r - 0.0005) * 1000:.0f}")
    # a 10x slower boot needs what reserve
    rho10 = lam * restore * 10
    S10 = 0
    while poisson_tail(S10, rho10) > target:
        S10 += 1
    show("reserve at 10x slower boot", S10)
    return dict(active=active, rho=rho, miss1=miss1, miss2=miss2, S=S,
                private=private_images_gb, cow=cow_gb,
                ratio=private_images_gb / cow_gb, S10=S10)


# ---------------------------------------------------------------------------
# Firecracker boot time (real data: Agache et al. 2020, NSDI; the paper's own
# released CDF, data/boot-*-fc-api-cdf.dat, pre-configured block, m5d.metal).
# ---------------------------------------------------------------------------
def firecracker_boot():
    head("Firecracker boot (Agache et al. 2020, released CDF data)")
    import numpy as np

    def preconf_ms(fn):
        blocks = open(DATA / fn).read().strip().split("\n\n")
        arr = np.array([[float(x) for x in l.split()]
                        for l in blocks[1].strip().splitlines() if l.strip()])
        p50 = arr[np.searchsorted(arr[:, 1], 0.5), 0]
        p99 = arr[min(np.searchsorted(arr[:, 1], 0.99), len(arr) - 1), 0]
        return p50, p99

    s50, s99 = preconf_ms("boot-serial-fc-api-cdf.dat")
    p50, p99 = preconf_ms("boot-parallel-100-fc-api-cdf.dat")
    show("serial pre-configured p50 (ms)", f"{s50:.0f}")
    show("serial pre-configured p99 (ms)", f"{s99:.0f}")
    show("100-parallel pre-configured p50 (ms)", f"{p50:.0f}")
    show("100-parallel pre-configured p99 (ms)", f"{p99:.0f}")
    return dict(s50=s50, s99=s99, p50=p50, p99=p99)


# ---------------------------------------------------------------------------
# Our runs (real measurements produced by measure.py), read back for slides
# ---------------------------------------------------------------------------
def our_runs():
    head("Our runs (images/measure.py)")

    def read_csv(fn):
        with open(DATA / fn) as fh:
            lines = [l for l in fh if not l.startswith("#")]
        return list(csv.DictReader(lines))

    startup = {r["envelope"]: float(r["median_ms"]) for r in read_csv("startup.csv")}
    clone = {r["operation"]: float(r["ms"]) for r in read_csv("clone.csv")}
    for k, v in startup.items():
        show(f"startup {k} (ms, median)", f"{v:.1f}")
    for k, v in clone.items():
        show(f"clone {k} (ms)", f"{v:.1f}")
    show("full vs clone ratio", f"{clone['full copy'] / clone['copy-on-write clone (cp -c)']:.1f}x")
    return dict(startup=startup, clone=clone)


if __name__ == "__main__":
    cybench()
    setup_amortization()
    syscall_tax()
    provisioning()
    pool_sizing()
    firecracker_boot()
    our_runs()
    print()
