#!/usr/bin/env python3
"""Every derived number in Lecture 4 (The Nervous System), one block per slide.

Inputs carry their source. "NS" is the rendered chapter, The Nervous System
(Volume IV, Chapter 4); "NM 4.x" is its Napkin Math 4.x. Values the book
labels illustrative or chosen keep that label on the slide. Every printed
value is what the slide or its note shows, with the label used there.
make_figs.py imports from this file.

Measured data in data/ (run by the lecture authors, 19 October 2026, on an
Apple M5 Max laptop, macOS 26.4, C compiled with cc -O2):
  jitter_light.npy / jitter_heavy.npy: lateness of 60,000 releases of a 1 ms
    periodic task (tick.c), first with the machine's ordinary background work,
    then with 18 memory-streaming load processes (load.c); run_jitter.sh.
  mailbox_<variant>.npy: 20,000 reads of a 1 kHz real-time reader against a
    20 Hz writer that stalls 3 ms mid-write on every 4th write (mailbox.c);
    columns: read time in ns, outcome (0 fresh, 1 kept prior, 2 rejected torn).

Run from this folder: python3 calcs.py
"""

import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def head(title):
    print(f"\n--- {title}")


def show(label, value):
    print(f"  {label}: {value}")


# =============================================================================
# Shared inputs (running machine; NS and NM 4.1)
# =============================================================================
V = 1.5                 # m/s, drive limit (illustrative, NM 4.1)
A_BRAKE = 2.0           # m/s^2, loaded, dry floor (illustrative, NM 4.1)
D_CLEAR = 1100.0        # mm, clear distance (illustrative, NM 4.1)
LOC = 50.0              # mm, localization error (illustrative, NM 4.1)
PROT = 100.0            # mm, protective clearance (chosen, NM 4.1)
T_ONSET = 20.0          # ms, brake onset (illustrative, NM 4.1)
T_TICK = 1.0            # ms, permission tick (chosen, NM 4.1)
T_BUS = 1.0             # ms, bus cycle (illustrative, NM 4.1)
T_LEASE = 60.0          # ms, chunk lease (chosen, NM 4.1)
T_RENEW = 50.0          # ms, chunk renewal period (chosen, NM 4.1)
T_JIT = 5.0             # ms, renewal variation allowance (chosen, NM 4.1)
T_INTENT = 336.0        # ms, intent model per evaluation (NS 4.1, from Lecture 3)
T_CHUNK_P99 = 40.0      # ms, chunk policy p99 (illustrative, NS 4.3.1)
SETPOINT = 25.0         # ms, setpoint spacing (NS 4.3.1)
H = 16                  # setpoints per chunk (NS 4.3.1)
OBS_AGE = 51.6          # ms, capture to dispatch (NS table 1; owned by Lecture 8)
DISP_ADM = 63.3         # ms, dispatch to admission (NS table 1; owned by Lecture 13)
L1_STOP = 762.9         # mm, Lecture 1 elementary stop at 1.5 m/s (NM 1.1, ownership table)

MM_PER_MS = V           # 1.5 m/s is 1.5 mm per ms

# =============================================================================
# Slide: Recall (how far the base moves while the brain thinks)
# =============================================================================
head("Recall: travel during inference")
show("travel in one 336 ms intent evaluation (mm, derived; the chapter prints 503 mm "
     "from the unrounded latency)", f"{V * T_INTENT:.0f}")
show("travel in one 40 ms chunk inference (mm)", f"{V * T_CHUNK_P99:.0f}")
show("chunk span, 16 setpoints x 25 ms (ms)", f"{H * SETPOINT:.0f}")
show("setpoints per 50 ms renewal", f"{T_RENEW / SETPOINT:.0f}")
CAL_WRITE_A = 35.0     # A, test value a diagnostic daemon writes through a calibration port (NS 4.6, illustrative)

# =============================================================================
# Slide: rate ladder (four tiers)
# =============================================================================
head("Rate ladder")
TIERS = [("Intent model", 500.0), ("Chunk policy", 50.0), ("Permission tick", 1.0),
         ("Current loop", 0.05)]   # ms; NS 4.1 figure and 4.3.1
for (a, ta), (b, tb) in zip(TIERS, TIERS[1:]):
    show(f"{a} / {b} period ratio", f"{ta / tb:.0f}")
show("intent / current period ratio", f"{TIERS[0][1] / TIERS[-1][1]:,.0f}")
TAU_E = 2.7e-3          # s, hub motor L/R (NS 4.3.1)
show("winding pole 1/(2 pi tau_e) (Hz; chapter: near 59 Hz)", f"{1 / (2 * math.pi * TAU_E):.1f}")
show("current loop rate (kHz)", f"{1 / 0.05e-3 / 1e3:.0f}")
show("base travel per 1 ms tick (mm)", f"{V * T_TICK:.1f}")

# =============================================================================
# Slide: Predict, phase lag at a 20 Hz crossover (NS 4.3.1, equation 1)
# =============================================================================
head("Phase lag at 20 Hz")
F_C = 20.0              # Hz, illustrative crossover (NS 4.3.1)
ALLOW = 5.0             # degrees, chosen erosion allowance (NS 4.3.1)


def lag_deg(t_ms, f=F_C):
    return 360.0 * f * t_ms / 1000.0


HOLD, COMPUTE, BUS = 0.5, 0.25, 1.0     # ms: half of a 1 ms tick, declared WCET, bus cycle
PHASE = [("Hold (half a tick)", HOLD), ("Compute (WCET)", COMPUTE), ("Bus cycle", BUS)]
for lab, t in PHASE:
    show(f"{lab} {t} ms (deg)", f"{lag_deg(t):.1f}")
TOT = HOLD + COMPUTE + BUS
show("total delay (ms)", f"{TOT:.2f}")
show("total lag (deg)", f"{lag_deg(TOT):.1f}")
show("ratio to allowance", f"{lag_deg(TOT) / ALLOW:.1f}")
show("delay the allowance tolerates (ms)", f"{ALLOW / (360 * F_C) * 1000:.2f}")
show("highest crossover for 1.75 ms (Hz)", f"{ALLOW / (360 * TOT / 1000):.1f}")
show("10 kHz tick: hold + compute + bus (deg)", f"{lag_deg(0.05 + COMPUTE + BUS):.1f}")
show("bus alone (deg)", f"{lag_deg(BUS):.1f}")
show("release at next tick, 2.5 ms (deg)", f"{lag_deg(2.5):.0f}")

# =============================================================================
# Slide: measured 1 ms task on a general-purpose OS (data/jitter_*.npy)
# =============================================================================
head("Measured: 1 ms periodic task on macOS")
JIT = {k: np.load(DATA / f"jitter_{k}.npy").astype(float) for k in ("light", "heavy")}
JSTAT = {}
for k, a in JIT.items():
    s = dict(n=len(a), p50=np.percentile(a, 50), p99=np.percentile(a, 99),
             p999=np.percentile(a, 99.9), mx=a.max(), over1ms=int((a > 1000).sum()))
    JSTAT[k] = s
    show(f"{k}: releases", f"{s['n']:,}")
    show(f"{k}: p50 / p99 / p99.9 / max lateness (us)",
         f"{s['p50']:.0f} / {s['p99']:.0f} / {s['p999']:,.0f} / {s['mx']:,.0f}")
    show(f"{k}: releases more than 1 ms late", s["over1ms"])
    show(f"{k}: travel at 1.5 m/s during the worst lateness (mm)", f"{V * s['mx'] / 1000:.1f}")
show("heavy max (ms)", f"{JSTAT['heavy']['mx'] / 1000:.1f}")
show("light max (ms)", f"{JSTAT['light']['mx'] / 1000:.1f}")

# =============================================================================
# Slide: Pilz PNOZ s4 datasheet against the laptop thread
# =============================================================================
head("Safety relay datasheet (Pilz PNOZ s4, 1001742-EN-05, Technical details, Times)")
PNOZ = dict(estop_typ=10.0, estop_max=20.0, power_typ=40.0, power_max=80.0)   # ms
show("E-STOP delay-on de-energisation typ / max (ms)", f"{PNOZ['estop_typ']:.0f} / {PNOZ['estop_max']:.0f}")
show("travel at 1.5 m/s in the 20 ms max (mm)", f"{V * PNOZ['estop_max']:.0f}")
show("travel at 1.5 m/s in the laptop's 35.8 ms (mm)", f"{V * JSTAT['heavy']['mx'] / 1000:.1f}")

# =============================================================================
# Slide: Predict, measured reader wait for three exchanges (data/mailbox_*.npy)
# =============================================================================
head("Measured: 1 kHz reader against a stalling 20 Hz writer")
MBX = {}
for k in ("mutex", "trylock", "seqlock"):
    a = np.load(DATA / f"mailbox_{k}.npy").astype(float)
    t = a[:, 0] / 1000.0           # us
    o = a[:, 1]
    MBX[k] = dict(n=len(t), p99=np.percentile(t, 99), mx=t.max(), over1ms=int((t > 1000).sum()),
                  fresh=int((o == 0).sum()), torn=int((o == 2).sum()))
    s = MBX[k]
    show(f"{k}: reads / p99 / max read time (us)", f"{s['n']:,} / {s['p99']:.2f} / {s['mx']:,.1f}")
    show(f"{k}: reads longer than 1 ms", s["over1ms"])
    show(f"{k}: fresh chunks taken / torn reads rejected", f"{s['fresh']} / {s['torn']}")
show("mutex max (ms)", f"{MBX['mutex']['mx'] / 1000:.1f}")
show("mutex max / seqlock max", f"{MBX['mutex']['mx'] / MBX['seqlock']['mx']:,.0f}")
show("writer stalls in 20 s (every 4th of 400 writes)", 400 // 4)

# =============================================================================
# Slides: lease timeline and the price of a lease (NM 4.1, equation 2)
# =============================================================================
head("Price of a lease (NM 4.1)")
BRAKE = V ** 2 / (2 * A_BRAKE) * 1000          # mm
CEIL = (D_CLEAR - LOC - PROT - BRAKE) / V        # ms (mm / (mm/ms))
ONSET_MM = V * T_ONSET
CHARGED = BRAKE + LOC + PROT + ONSET_MM
LEASE_PATH_MS = T_LEASE + T_TICK + T_BUS
LEASE_PATH = V * LEASE_PATH_MS
RUN = CHARGED + LEASE_PATH
show("braking distance v^2/2a (mm)", f"{BRAKE:.1f}")
show("pre-brake ceiling (ms)", f"{CEIL:.1f}")
show("onset travel (mm)", f"{ONSET_MM:.0f}")
show("already charged: braking + overheads + onset (mm)", f"{CHARGED:.1f}")
show("lease path 60 + 1 + 1 ms (mm)", f"{LEASE_PATH:.0f}")
show("running total (mm)", f"{RUN:.1f}")
show("left of 1,100 mm (mm)", f"{D_CLEAR - RUN:.1f}")
show("last admission to brake onset (ms)", f"{LEASE_PATH_MS + T_ONSET:.0f}")
show("left under the ceiling (ms)", f"{CEIL - (LEASE_PATH_MS + T_ONSET):.1f}")
show("renewal floor 50 + 5 (ms)", f"{T_RENEW + T_JIT:.0f}")
# lease timeline figure (NS figure 4 panel a): renewals at 0-40 and 50-90, stall from 100
T_LAST_ADM = 90.0
show("lease expiry (ms on the timeline)", f"{T_LAST_ADM + T_LEASE:.0f}")
show("brake onset (ms on the timeline)", f"{T_LAST_ADM + T_LEASE + T_TICK + T_BUS + T_ONSET:.0f}")
show("tick + bus + onset (ms)", f"{T_TICK + T_BUS + T_ONSET:.0f}")
show("travel from last admission to brake onset (mm)", f"{V * (LEASE_PATH_MS + T_ONSET):.0f}")

# =============================================================================
# Slide: measured OpenVLA latency against the lease (Franklin 2024)
# =============================================================================
head("OpenVLA-7B on Jetson AGX Orin 64GB (Franklin 2024, Bridge table)")
OPENVLA = [("FP16", 840.0), ("FP8", 471.0), ("INT4", 336.0)]
for lab, t in OPENVLA:
    show(f"{lab} {t:.0f} ms: times the 258.3 ms ceiling", f"{t / CEIL:.2f}")
    show(f"{lab}: travel at 1.5 m/s (mm)", f"{V * t:.0f}")
show("INT4 over the 60 ms lease", f"{336 / T_LEASE:.1f}")

# =============================================================================
# Slide: Estimate, a 100 ms lease
# =============================================================================
head("Estimate: 100 ms lease")
NEW = 100.0
show("added lease (ms)", f"{NEW - T_LEASE:.0f}")
show("added distance (mm)", f"{V * (NEW - T_LEASE):.0f}")
show("lease path (mm)", f"{V * (NEW + T_TICK + T_BUS):.0f}")
show("running total (mm)", f"{CHARGED + V * (NEW + T_TICK + T_BUS):.1f}")
show("last admission to brake onset (ms)", f"{NEW + T_TICK + T_BUS + T_ONSET:.0f}")

# =============================================================================
# Slide: the lease rule (equation 4)
# =============================================================================
head("Lease rule (equation 4)")
UP0 = CEIL - T_TICK - T_BUS - T_ONSET
show("ceiling - tick - bus - onset (ms)", f"{UP0:.1f}")
show("after observation age 51.6 ms (ms)", f"{UP0 - OBS_AGE:.1f}")
AGE_ADM = OBS_AGE + DISP_ADM
show("age at admission 51.6 + 63.3 (ms)", f"{AGE_ADM:.1f}")
UPPER = UP0 - AGE_ADM
show("upper limit on the lease (ms)", f"{UPPER:.1f}")
show("floor (ms)", f"{T_RENEW + T_JIT:.0f}")
show("100 ms lease under the limit?", NEW <= UPPER)

# =============================================================================
# Slide: authority-expiry timers
# =============================================================================
head("Timers (NS 4.4.1)")
HB_HZ, HB_TIMEOUT, WDOG = 100.0, 30.0, 5.0
show("heartbeat period (ms)", f"{1000 / HB_HZ:.0f}")
show("missed beats before timeout", f"{HB_TIMEOUT / (1000 / HB_HZ):.0f}")
show("ticks per watchdog window", f"{WDOG / T_TICK:.0f}")

# =============================================================================
# Slide: handoff record, event table (NS table 1)
# =============================================================================
head("Event table (NS table 1)")
STALL = T_LEASE + T_TICK + T_BUS + T_ONSET
show("rows 1 + 2, replacement latency (ms)", f"{OBS_AGE + DISP_ADM:.1f}")
show("row 3, stalled-proposer response (ms)", f"{STALL:.0f}")
show("capture to brake onset (ms)", f"{OBS_AGE + DISP_ADM + STALL:.1f}")

# =============================================================================
# Slide: response time R = C + B + I (NM 4.2) and the Estimate
# =============================================================================
head("Response-time margin (NM 4.2)")
C_UNLOADED, C_DECL, B, I, DEADLINE = 135.0, 250.0, 50.0, 40.0, 400.0   # us
show("WCET margin to justify (us)", f"{C_DECL - C_UNLOADED:.0f}")
show("slack after declared WCET (us)", f"{DEADLINE - C_DECL:.0f}")
R = C_DECL + B + I
show("R = C + B + I (us)", f"{R:.0f}")
show("D - R (us)", f"{DEADLINE - R:.0f}")
R2 = C_DECL + 120.0 + I
show("blocking 120 us: R (us)", f"{R2:.0f}")
show("blocking 120 us: miss by (us)", f"{R2 - DEADLINE:.0f}")
show("largest blocking that still fits (us)", f"{DEADLINE - C_DECL - I:.0f}")

# =============================================================================
# Slide: measured maximum grows with the test (data/jitter_heavy.npy)
# =============================================================================
head("Running maximum against test length")
a = JIT["heavy"]
for n in (100, 1000, 10000, 60000):
    show(f"first {n:,} releases: p99 / max (us); max (ms)",
         f"{np.percentile(a[:n], 99):.0f} / {a[:n].max():,.0f}; {a[:n].max() / 1000:.1f}")

# =============================================================================
# Slide: fieldbus occupancy (NM 4.3)
# =============================================================================
head("Fieldbus occupancy (NM 4.3)")
T_FRAME = 34 / 1.0 + 113 / 5.0           # us: 34 bits at 1 Mbps, 113 bits at 5 Mbps
show("one CAN FD frame (us)", f"{T_FRAME:.1f}")
CYCLE = 1000.0
BUSES = [("One CAN FD bus, 9 axes", 18 * T_FRAME), ("CAN FD, arm (7 axes)", 14 * T_FRAME),
         ("CAN FD, base (2 axes)", 4 * T_FRAME)]
for lab, t in BUSES:
    show(f"{lab} (us, percent of cycle)", f"{t:.1f}, {100 * t / CYCLE:.1f}")
ECAT = (64 + 9 * 16) * 8 / 100.0 + 9 * 0.6
show("EtherCAT (us, percent)", f"{ECAT:.2f}, {100 * ECAT / CYCLE:.1f}")
show("EtherCAT left (us)", f"{CYCLE - ECAT:.1f}")
show("arm bus left (us)", f"{CYCLE - 14 * T_FRAME:.1f}")
show("axes one CAN FD bus can serve (2 frames each)", f"{int(CYCLE // (2 * T_FRAME))}")
SCOPE_CAN = 47.0                          # us, classic CAN, 1 data byte, 1 Mbit/s (photo)
show("measured classic CAN 1-byte frame SOF to ACK (us)", f"{SCOPE_CAN:.0f}")

# =============================================================================
# Slide: synthesis
# =============================================================================
head("Synthesis")
show("unchecked travel at the laptop's worst lateness (mm)", f"{V * JSTAT['heavy']['mx'] / 1000:.1f}")
show("stall to brake onset (ms) / travel (mm)", f"{STALL:.0f} / {V * STALL:.0f}")
show("running total of the stopping budget (mm)", f"{RUN:.1f}")
