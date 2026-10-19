#!/usr/bin/env python3
"""Every derived number in Lecture 8 (Sensor Perception), one block per slide.

Inputs carry their source. "SP" is the rendered chapter, Sensor Perception
(Volume IV, Chapter 8); "PB" is The Physical Body (Chapter 2) and "NS" The
Nervous System (Chapter 4), which priced the first four rows of the stopping
budget. Outside data sit in data/ with their sources in SOURCES.yml: KITTI raw
logs (Geiger et al., IJRR 2013), the MLPerf Inference v3.1 Orin log, and the
Ouster OS1 datasheet (rev7, firmware 3.1).

Each printed value is what the slide or its note shows, with the label used
there. make_figs.py imports this module.

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
# Shared inputs: the running machine on the aisle run
# =============================================================================
V = 1.5                 # m/s, drive limit (SP Napkin Math 8.2; illustrative)
V_AISLE = 1.3           # m/s, chosen aisle speed (SP Napkin Math 8.1, sensitivity)
D_CLEAR = 1100.0        # mm, rack-end clear distance (NS Napkin Math 4.1; illustrative)
A_BRAKE = 2.0           # m/s^2, credible deceleration (NS Napkin Math 4.1)
D_LOC = 50.0            # mm, localization allowance (PB eq. 2; illustrative)
D_MARGIN = 100.0        # mm, protective clearance (PB eq. 2; chosen)
T_ACT = 20.0            # ms, brake onset (NS: onset travel 30 mm at 1.5 m/s)
T_LEASE, T_TICK, T_BUS = 60.0, 1.0, 1.0   # ms (NS Napkin Math 4.1)
SIDE = 150.0            # mm, side clearance to the racks (SP Napkin Math 8.1)
RANGE_RACK = 1.10       # m, rack-end range (SP Napkin Math 8.1)
YAW = 0.5               # rad/s, turn into the aisle (SP Napkin Math 8.1; illustrative)

# Navigation camera stages (SP Napkin Math 8.2, table 3; budgeted allocations)
T_EXP = 16.0            # ms exposure (chosen low-light setting)
STAGES = [              # (name on the slide, ms)
    ("Exposure midpoint", T_EXP / 2),
    ("Rolling readout", 16.6),   # datasheet, full-resolution mode
    ("Link transport", 0.8),
    ("DMA and queue", 1.2),
    ("ISP", 2.5),
    ("Vision backbone", 22.0),   # INT8 (illustrative)
    ("Dispatch", 0.5),
]
T_READ = 16.6           # ms rolling readout (datasheet)
ROWS = 3040             # rows, full-resolution mode (SP Napkin Math 8.1; Sony IMX477)
FRAME_HZ = 60.0         # navigation camera rate (SP §8.4.3)


def mm(v_ms, t_ms):
    """Travel in mm at v_ms (m/s) over t_ms (ms)."""
    return v_ms * t_ms


# =============================================================================
# Slide: Recall (rows 1 to 4 of the stopping budget)
# =============================================================================
head("Recall: stopping budget rows 1 to 4 (PB, NS)")
BRAKE = (V ** 2) / (2 * A_BRAKE) * 1000          # mm
ONSET = mm(V, T_ACT)
LEASE_ROW = mm(V, T_LEASE + T_TICK + T_BUS)
ROWS_1_4 = BRAKE + (D_LOC + D_MARGIN) + ONSET + LEASE_ROW
show("braking v^2/2a (mm)", f"{BRAKE:.1f}")
show("overhead loc + margin (mm)", f"{D_LOC + D_MARGIN:.0f}")
show("onset travel (mm)", f"{ONSET:.0f}")
show("lease path (mm)", f"{LEASE_ROW:.0f}")
show("rows 1-4 (mm)", f"{ROWS_1_4:.1f}")
show("left before perception (mm)", f"{D_CLEAR - ROWS_1_4:.1f}")
show("ms per mm at 1.5 m/s: mm per ms", f"{V:.1f}")

# =============================================================================
# Slide: Hook, Williston (NTSB HAR-17/02: 74 mph before impact)
# =============================================================================
head("Hook: Williston speed")
MPH = 0.44704
show("74 mph in m/s", f"{74 * MPH:.1f}")
show("mm per ms at 74 mph", f"{74 * MPH:.1f}")

# =============================================================================
# Slide: A frame is dated at its exposure midpoint (SP §8.3)
# =============================================================================
head("Capture epoch: arrival and trigger stamps")
T_ARRIVAL = sum(t for n, t in STAGES[:4])          # midpoint .. DMA complete
show("arrival offset (ms)", f"{T_ARRIVAL:.1f}")
show("arrival-stamp travel (mm)", f"{mm(V, T_ARRIVAL):.1f}")
show("trigger edge without T/2: early by (ms)", f"{T_EXP / 2:.0f}")
show("trigger-stamp travel (mm)", f"{mm(V, T_EXP / 2):.0f}")
show("travel during the exposure itself (mm)", f"{mm(V, T_EXP):.0f}")
JIT = 1.0  # ms of scheduling jitter (SP §8.3)
show("1 ms jitter at 0.5 rad/s: misorientation (mrad)", f"{YAW * JIT:.1f}")
show("lateral at rack end (mm)", f"{YAW * JIT / 1000 * RANGE_RACK * 1000:.2f}")

# =============================================================================
# Slide: KITTI timestamp log (real data; images/data/kitti)
# =============================================================================
head("KITTI 2011_09_26_drive_0001 timestamps (host system clock)")


def kitti_times(rel):
    out = []
    for line in (DATA / "kitti" / rel).read_text().splitlines():
        if not line.strip():
            continue
        _, t = line.split()
        h, m, s = t.split(":")
        out.append(int(h) * 3600 + int(m) * 60 + float(s))
    return np.array(out)


EX = "extract"
K_VS = kitti_times(f"{EX}/velodyne_points/timestamps_start.txt")
K_VF = kitti_times(f"{EX}/velodyne_points/timestamps.txt")
K_VE = kitti_times(f"{EX}/velodyne_points/timestamps_end.txt")
K_CAM = {c: kitti_times(f"{EX}/image_0{c}/timestamps.txt") for c in range(4)}


def cam_offsets(c):
    """Camera stamp minus the Velodyne forward-facing (trigger) stamp, ms, for
    camera frames that fall inside the logged Velodyne sweeps."""
    t = K_CAM[c]
    t = t[(t > K_VS[0]) & (t < K_VE[-1])]
    j = np.array([np.argmin(abs(K_VF - x)) for x in t])
    return (t - K_VF[j]) * 1000


K_SWEEP = (K_VE - K_VS) * 1000
K_FWD = (K_VF - K_VS) * 1000
K_OFF = {c: cam_offsets(c) for c in range(4)}
show("sweeps logged", len(K_VS))
show("sweep duration median (ms)", f"{np.median(K_SWEEP):.1f}")
show("sweep duration min..max (ms)", f"{K_SWEEP.min():.1f} .. {K_SWEEP.max():.1f}")
show("forward-facing after start, median (ms)", f"{np.median(K_FWD):.1f}")
for c, name in [(0, "gray left"), (1, "gray right"), (2, "color left"), (3, "color right")]:
    o = K_OFF[c]
    show(f"cam {c} ({name}) stamp - trigger, median (ms)", f"{np.median(o):.1f}")
    show(f"cam {c} ({name}) stamp - trigger, min..max (ms)", f"{o.min():.1f} .. {o.max():.1f}")
K_GAP = np.median(K_OFF[0]) - np.median(K_OFF[2])
show("gray minus color median (ms)", f"{K_GAP:.1f}")
K_PER0 = np.diff(K_CAM[0]) * 1000
show("gray camera stamp period min..max (ms)", f"{K_PER0.min():.1f} .. {K_PER0.max():.1f}")
show("Velodyne forward period min..max (ms)",
     f"{np.diff(K_VF).min() * 1000:.1f} .. {np.diff(K_VF).max() * 1000:.1f}")

# =============================================================================
# Slide: rolling-shutter rows (SP Napkin Math 8.1)
# =============================================================================
head("Readout shear (Napkin Math 8.1)")
T_ROW_US = T_READ / ROWS * 1000
SKEW = mm(V, T_READ)
LAT = RANGE_RACK * YAW * T_READ                      # mm (m * rad/s * ms)
show("row interval (us)", f"{T_ROW_US:.2f}")
show("along-track skew (mm)", f"{SKEW:.1f}")
show("skew / localization allowance", f"{SKEW / D_LOC:.2f}")
show("lateral skew (mm)", f"{LAT:.1f}")
show("lateral / side clearance (%)", f"{LAT / SIDE * 100:.1f}")
show("skew at 1.3 m/s (mm)", f"{mm(V_AISLE, T_READ):.1f}")
show("share of allowance at 1.3 m/s (%)", f"{mm(V_AISLE, T_READ) / D_LOC * 100:.1f}")
show("blur band: travel during 16 ms exposure (mm)", f"{mm(V, T_EXP):.0f}")

# =============================================================================
# Slide: one lidar sweep takes time (KITTI sweep 0; SP §8.2.5)
# =============================================================================
head("Lidar sweep (KITTI sync sweep 0; the machine's OS1-64)")
SYNC_S = kitti_times("sync/timestamps_start.txt")
SYNC_E = kitti_times("sync/timestamps_end.txt")
SWEEP0 = (SYNC_E[0] - SYNC_S[0]) * 1000
OX = [float(x) for x in (DATA / "kitti" / "sync" / "oxts_0000000000.txt").read_text().split()]
VF = OX[8]                     # forward velocity, m/s (oxts dataformat: field 9, vf)
WZ = OX[19]                    # angular rate about z, rad/s (field 20, wz)
PTS = np.fromfile(DATA / "kitti" / "sync" / "velo_0000000000.bin", dtype=np.float32).reshape(-1, 4)
show("sweep 0 duration (ms)", f"{SWEEP0:.1f}")
show("car forward speed vf (m/s)", f"{VF:.2f}")
show("car speed (km/h)", f"{VF * 3.6:.0f}")
show("car travel during sweep 0 (m)", f"{VF * SWEEP0 / 1000:.2f}")
show("yaw rate wz (rad/s)", f"{WZ:.3f}")
show("points in sweep 0", f"{len(PTS):,}")
LIDAR_HZ = 20.0                # OS1-64 configured at 20 Hz (SP §8.2.5)
T_SWEEP = 1000 / LIDAR_HZ
show("OS1-64 sweep at 20 Hz (ms)", f"{T_SWEEP:.0f}")
show("machine travel per sweep (mm)", f"{mm(V, T_SWEEP):.0f}")
show("machine rotation per sweep (mrad)", f"{YAW * T_SWEEP:.0f}")
show("machine rotation per sweep (deg)", f"{math.degrees(YAW * T_SWEEP / 1000):.1f}")

# =============================================================================
# Slide: calibration chain; 1 degree mount error (SP §8.2.2, 8.3)
# =============================================================================
head("Mount error")
DEG1 = math.radians(1.0)
show("1 deg at 1.10 m (mm)", f"{RANGE_RACK * math.sin(DEG1) * 1000:.1f}")
show("1 deg at 15 m (mm)", f"{15 * math.sin(DEG1) * 1000:.0f}")
show("1 deg at 15 m / side clearance", f"{15 * math.sin(DEG1) * 1000 / SIDE:.1f}")
show("0.005 rad at 10 m (mm)", f"{10 * 0.005 * 1000:.0f}")
F_KITTI = 721.5377             # px, P_rect_02 focal length (KITTI calib_cam_to_cam.txt)
show("1 deg in KITTI color-camera pixels (rotation, any depth)", f"{F_KITTI * math.tan(DEG1):.1f}")

# =============================================================================
# Slide: error sources against the allowances (SP figure, §8.3)
# =============================================================================
head("Error bars (mm)")
ERR_BARS = [
    ("1° mount error at 15 m", 15 * math.sin(DEG1) * 1000),
    ("Dispatch age", mm(V, sum(t for _, t in STAGES))),
    ("Arrival stamp", mm(V, T_ARRIVAL)),
    ("Row skew", SKEW),
    ("Trigger stamp", mm(V, T_EXP / 2)),
]
for n, v in ERR_BARS:
    show(n, f"{v:.1f}")

# =============================================================================
# Slide: clock domains (SP §8.6.2; Ouster datasheet)
# =============================================================================
head("Clock conversion")
PPM = 10e-6
show("10 ppm drift over a 1 s sync interval (us)", f"{PPM * 1.0 * 1e6:.0f}")
show("PTP asymmetry: offset error is half the asymmetry", "Delta/2")
SYNC_BOUND = 0.2               # ms, nav camera record (SP table 5; illustrative)
show("travel during 0.2 ms clock bound (mm)", f"{mm(V, SYNC_BOUND):.1f}")

# =============================================================================
# Slide: the latency waterfall (SP Napkin Math 8.2)
# =============================================================================
head("Waterfall (Napkin Math 8.2)")
cum = 0.0
WATERFALL = []
for n, t in STAGES:
    cum += t
    WATERFALL.append((n, t, cum))
    show(f"{n} +{t:g} ms -> serial age (ms)", f"{cum:.1f}")
T_AGE = cum
D_AGE = mm(V, T_AGE)
show("observation age at dispatch (ms)", f"{T_AGE:.1f}")
show("travel at dispatch (mm)", f"{D_AGE:.1f}")
show("exposure + readout (ms)", f"{STAGES[0][1] + STAGES[1][1]:.1f}")
show("exposure + readout share of age (%)", f"{(STAGES[0][1] + STAGES[1][1]) / T_AGE * 100:.0f}")
show("backbone share of age (%)", f"{22.0 / T_AGE * 100:.0f}")

# =============================================================================
# Slide: a measured detector (MLPerf v3.1, Orin, RetinaNet SingleStream)
# =============================================================================
head("MLPerf Inference v3.1, Jetson AGX Orin, RetinaNet INT8, SingleStream")


def mlperf_ns(key):
    for line in (DATA / "mlperf_v3.1_orin_retinanet_singlestream_summary.txt").read_text().splitlines():
        if line.startswith(key):
            return int(line.split(":")[1].strip())
    raise KeyError(key)


RN_P50 = mlperf_ns("50.00 percentile latency (ns)") / 1e6
RN_P90 = mlperf_ns("90th percentile latency (ns)") / 1e6
RN_P99 = mlperf_ns("99.00 percentile latency (ns)") / 1e6
RN_MAX = mlperf_ns("Max latency (ns)") / 1e6
show("p50 (ms)", f"{RN_P50:.2f}")
show("p90 (ms)", f"{RN_P90:.2f}")
show("p99 (ms)", f"{RN_P99:.2f}")
show("max (ms)", f"{RN_MAX:.2f}")
EXPREAD = STAGES[0][1] + STAGES[1][1]
show("exposure + readout / RetinaNet p99", f"{EXPREAD / RN_P99:.1f}")
show("backbone allocation / RetinaNet p99", f"{22.0 / RN_P99:.1f}")

# =============================================================================
# Slide: age is the fifth row of the stopping budget (SP §8.4.1)
# =============================================================================
head("Stopping budget with row 5")
ROWS_1_5 = ROWS_1_4 + D_AGE
show("rows 1-4 (mm)", f"{ROWS_1_4:.1f}")
show("row 5, v * t_age (mm)", f"{D_AGE:.1f}")
show("rows 1-5 (mm)", f"{ROWS_1_5:.1f}")
show("remainder of 1,100 mm (mm)", f"{D_CLEAR - ROWS_1_5:.1f}")
show("pre-brake delay 51.6 + 82 (ms)", f"{T_AGE + T_LEASE + T_TICK + T_BUS + T_ACT:.1f}")
show("lease + tick + bus + onset (ms)", f"{T_LEASE + T_TICK + T_BUS + T_ACT:.0f}")

# =============================================================================
# Slide: age keeps growing after dispatch; dropped frame (SP §8.8, Checkpoint 8.1)
# =============================================================================
head("Current evidence age")
T_FRAME = 1000 / FRAME_HZ
show("dispatch-local threshold (ms)", f"{T_AGE + SYNC_BOUND:.1f}")
show("frame period at 60 Hz (ms)", f"{T_FRAME:.1f}")
show("dropped frame: extra travel (mm)", f"{mm(V, T_FRAME):.1f}")
show("share of 187.1 mm remainder (%)", f"{mm(V, T_FRAME) / (D_CLEAR - ROWS_1_5) * 100:.1f}")

# =============================================================================
# Slide: diagnose, refused under camera load (SP §8.4.3, 8.5)
# =============================================================================
head("Batching and contention")
HOLD = 3 * T_FRAME
show("batch of 4: first frame held (ms)", f"{HOLD:.0f}")
show("age with batching (ms)", f"{T_AGE + HOLD:.1f}")
show("travel with batching (mm)", f"{mm(V, T_AGE + HOLD):.1f}")
W, H, BPP, N_CAM, COPIES = 1920, 1080, 3, 4, 3
per_copy = W * H * BPP * FRAME_HZ * N_CAM / 1e9
show("four 1080p 24-bit 60 Hz cameras, per copy (GB/s)", f"{per_copy:.2f}")
show("three copies (GB/s)", f"{per_copy * COPIES:.1f}")
show("renewal jitter allowance (ms)", "5")
show("lease lapses when a renewal is late by (ms)", f"{T_LEASE - 50:.0f}")

# =============================================================================
# Slide: streaming the lidar sector (SP §8.4.2, table 4)
# =============================================================================
head("Lidar ingestion")
AISLE_W = 1.30
SECTOR_DEG = 2 * math.degrees(math.atan((AISLE_W / 2) / RANGE_RACK))
T_SECTOR = T_SWEEP * SECTOR_DEG / 360
LIDAR_LAT, PTP = 10.0, 1.0     # ms, OS1 datasheet: data latency < 10 ms, PTP < 1 ms
STREAM = T_SECTOR + LIDAR_LAT + PTP
FULL = T_SWEEP + LIDAR_LAT + PTP
show("forward sector (deg)", f"{SECTOR_DEG:.1f}")
show("sector sweep time (ms)", f"{T_SECTOR:.1f}")
show("stream: upper age before processing (ms)", f"{STREAM:.1f}")
show("full sweep: upper age before processing (ms)", f"{FULL:.0f}")
show("stream travel (mm)", f"{mm(V, STREAM):.1f}")
show("full-sweep travel (mm)", f"{mm(V, FULL):.1f}")
show("headroom inside 51.6 ms when streaming (ms)", f"{T_AGE - STREAM:.1f}")

# =============================================================================
# Slide: delay spends phase margin (SP §8.4.4)
# =============================================================================
head("Phase margin")
PM = math.radians(45)
WC = 10.0
T_PM0 = PM / WC * 1000
show("PM 45 deg in rad", f"{PM:.3f}")
show("zero-margin age (ms)", f"{T_PM0:.1f}")
show("crossover (Hz)", f"{WC / (2 * math.pi):.1f}")
show("60 ms spends (%)", f"{60 / T_PM0 * 100:.1f}")

# =============================================================================
# Slide: stereo range wall (SP Napkin Math 8.3)
# =============================================================================
head("Stereo range wall (Napkin Math 8.3)")
B_ST, F_ST, SD = 0.20, 800.0, 0.5


def sigma_z_mm(z):
    return z ** 2 * SD / (F_ST * B_ST) * 1000


def crossover_m(c_mm):
    return math.sqrt(c_mm / 1000 * F_ST * B_ST / SD)


show("sigma_z at 1.10 m (mm)", f"{sigma_z_mm(1.10):.1f}")
show("sigma_z at 2 m (mm)", f"{sigma_z_mm(2):.1f}")
show("sigma_z at 10 m (mm)", f"{sigma_z_mm(10):.1f}")
show("ratio 10 m / 2 m", f"{sigma_z_mm(10) / sigma_z_mm(2):.0f}")
show("range where sigma = 50 mm (m)", f"{crossover_m(50):.1f}")
show("range where sigma = 150 mm (m)", f"{crossover_m(150):.1f}")
show("halved baseline: range where sigma = 150 mm (m)",
     f"{math.sqrt(0.150 * F_ST * B_ST / 2 / SD):.1f}")
OS1_PREC = (5.0, 30.0)          # mm, 1 sigma beyond 1 m, 1024x10 (Ouster datasheet)
OS1_ACC = (25.0, 50.0)          # mm, lambertian, retroreflective (Ouster datasheet)
show("OS1 precision band (mm)", f"{OS1_PREC[0]:g}-{OS1_PREC[1]:g}")
show("OS1 accuracy lambertian / retroreflective (mm)", f"{OS1_ACC[0]:g} / {OS1_ACC[1]:g}")

# =============================================================================
# Slide: clearance inset (SP eq. 9, §8.6.2)
# =============================================================================
head("Clearance inset")
C_SAFE = 150.0
show("shortcut inset at 2 m (mm)", f"{C_SAFE + sigma_z_mm(2):.1f}")
show("shortcut inset at 10 m (mm)", f"{C_SAFE + sigma_z_mm(10):.1f}")


def conformal_rank(n, alpha):
    return math.ceil((n + 1) * (1 - alpha))


ALPHA = 0.01
n_min = next(n for n in range(1, 1000) if conformal_rank(n, ALPHA) <= n)
show("fewest residuals for alpha = 1% (n)", n_min)
show("rank at n = 99", conformal_rank(99, ALPHA))
show("rank at n = 1000", conformal_rank(1000, ALPHA))
show("that is the k-th largest", 1000 - conformal_rank(1000, ALPHA) + 1)

# =============================================================================
# Slide: metric veto (SP Systems Perspective 8.3)
# =============================================================================
head("Metric veto rates")
show("VLM period at 2 Hz (ms)", f"{1000 / 2:.0f}")
show("lidar sweep at 20 Hz (ms)", f"{1000 / 20:.0f}")
show("lidar sweeps per VLM evaluation", f"{20 / 2:.0f}")

# =============================================================================
# Slide: lost frame vs misdated frame (SP §8.7, eq. 11)
# =============================================================================
head("Silent vs detectable")
show("dropped frame: evidence older by (ms)", f"{T_FRAME:.1f}")
show("misdated by the arrival stamp: offset (mm)", f"{mm(V, T_ARRIVAL):.1f}")
show("stale by three frames (ms)", f"{3 * T_FRAME:.0f}")
show("travel for three frames (mm)", f"{mm(V, 3 * T_FRAME):.0f}")

# =============================================================================
# Slide: environmental degradation (SP §8.7.1; KITTI frame)
# =============================================================================
head("Degradation")
BETA = 0.05
show("contrast left after 60 m of fog at 0.05 per m", f"{math.exp(-BETA * 60):.3f}")
show("range where contrast reaches 0.05 (m)", f"{-math.log(0.05) / BETA:.0f}")
try:
    from PIL import Image
    img = np.asarray(Image.open(DATA / "kitti" / "sync" / "image_02_0000000000.png").convert("L"))
    SAT = float((img >= 255).mean())
    show("KITTI frame 0 pixels clipped at 255 (%)", f"{SAT * 100:.1f}")
    show("KITTI frame size", f"{img.shape[1]} x {img.shape[0]}")
except ImportError:  # Pillow missing: the slide value stays from the last run
    SAT = None
show("FMCW 4 GHz sweep: range resolution (cm)", f"{3e8 / (2 * 4e9) * 100:.2f}")
show("wrist camera blur scale 0.5 m x 3 rad/s x 5 ms (mm)", f"{0.5 * 3 * 5:.1f}")

# =============================================================================
# Slide: observation contract (SP table 5, §8.8)
# =============================================================================
head("Contract")
show("dispatch-local threshold (ms)", f"{T_AGE + SYNC_BOUND:.1f}")
show("middle-row sensor latency T/2 + T_read/2 (ms)", f"{T_EXP / 2 + T_READ / 2:.1f}")
show("top-row (oldest) T/2 + T_read (ms)", f"{T_EXP / 2 + T_READ:.1f}")
show("top row exceeds the 16.7 ms handoff allotment by (ms)",
     f"{T_EXP / 2 + T_READ - T_FRAME:.1f}")
show("wrist: 10 ms skew at 1 m/s + 2 rad/s x 0.15 m (mm)", f"{10 * (1 + 2 * 0.15):.0f}")
show("wrist: skew bound for 5 mm (ms)", f"{5 / (1 + 2 * 0.15):.1f}")

# =============================================================================
# Slide: synthesis
# =============================================================================
head("Synthesis")
show("rows 1-5 (mm)", f"{ROWS_1_5:.1f}")
show("remainder (mm)", f"{D_CLEAR - ROWS_1_5:.1f}")
