#!/usr/bin/env python3
"""Every derived number in Lecture 9 (Spatial Memory), Volume IV.

One block per slide, in deck order. Inputs carry their source as a comment:
"ch" is the rendered chapter (Spatial Memory, chapter_text_dev 2026-10-18),
"NM" a Napkin Math box in it. Each printed line is a value the slide or its
note shows, under the label used there. make_figs.py imports from this file so
figures and slides share one computation.

Run from this folder: python3 calcs.py
"""
import math
import sys
from pathlib import Path

from scipy import stats

MLSYSIM = Path(__file__).resolve().parents[4] / "mlsysim"
sys.path.insert(0, str(MLSYSIM))


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
V_DRIVE = 1.5            # m/s, drive limit (ch 9.4, machine card)
FRAME_MS = 16.7          # ms, navigation and wrist camera frame period (ch NM 9.2, 9.5)
T_AGE_MS = 51.6          # ms, frame age at dispatch (ch NM 9.2, from the latency waterfall)
DELTA_LOC = 0.050        # m, localization bound (ch 9.7.1, illustrative)
DELTA_MARGIN = 0.100     # m, protective clearance (ch 9.7.1, chosen)
V_WALK = 1.6             # m/s, ISO 13855 walking approach speed (ch 9.7.1)
D_CLEAR = 1.10           # m, clear distance to the rack end (ch 9.7.1)
SIDE_CLEAR = 0.150       # m, side clearance abreast of the rack end (ch NM 9.2)
TAU_DELAY = 0.1336       # s, pre-brake delay (machine card, ch 9.7.1 uses it)
A_BRAKE = 2.0            # m/s^2, credible deceleration (machine card)

# Segment 1: estimation (ch 9.2)
SIG0 = 20.0              # mm, prior range std (ch 9.2)
V_DIST = 0.1             # m/s, illustrative odometry speed disturbance (ch 9.2)
DT_SWEEP = 0.050         # s, lidar sweep interval (ch 9.2)
SIG_LIDAR = 30.0         # mm, Ouster OS1-64 worst-case datasheet range precision (ch 9.2)
LIDAR_BIAS = 25.0        # mm, Ouster systematic accuracy offset (ch 9.2)
SIG_STEREO = 3.8         # mm, stereo at 1.10 m (ch 9.2)
MISDATE = 39.9           # mm, arrival-stamp travel (ch 9.2, from the observation contract)
N_TRIALS = 50            # trials for the NEES/NIS band (ch 9.2.1)
GATE95, GATE99 = 3.84, 6.63   # chi-square(1) 95 and 99 percent gates (ch 9.2.1)
# BMI088 datasheet values (Bosch Sensortec 2024, via ch 9.2)
GYRO_ND = 0.014          # deg/s/sqrt(Hz)
GYRO_BIAS = 1.0          # deg/s
ACC_ND = 160e-6          # g/sqrt(Hz)
ACC_BIAS = 20e-3         # g
G = 9.81                 # m/s^2
T_IMU = 1.0              # s, coasting interval (ch 9.2)
RANGE_UPRIGHT = 1.10     # m (ch 9.2)
IMU_RATE_MS = 1.0        # ms, IMU sample period, 1 kHz (ch 9.2.3)
SORT_MIN_HITS = 3        # abewley/sort sort.py, Sort.__init__ default min_hits

# Segment 2: records and frames (ch 9.4, 9.5)
SYNC_MS = 0.2            # ms, clock-conversion bound (ch 9.4)
T_TACT, T_CAM, T_ENC = 12.0, 18.5, 21.0   # ms, illustrative epochs (ch 9.4)
V_ARM = 1.7              # m/s, arm Cartesian speed limit (ch 9.4)
E0_SLOT = 0.1            # mm, initial calibration error (ch 9.5)
V_DRIFT = 12.0           # mm/s, unobserved lateral velocity bound (ch 9.5)
R_ARM = 0.15             # m, wrist-to-fingertip lever arm (ch 9.5)
OMEGA = 0.30             # rad/s, wrist alignment rate (ch 9.5)
R_TOL = 1.0              # mm, latch-slot tolerance (ch 9.5)
UMB_PATH = 16.0          # m, UMBmark 4 m square path (Borenstein and Feng 1996, via ch 9.5.1)
UMB_UNCAL, UMB_CAL = 317.0, 21.0   # mm, average return error (Borenstein and Feng 1996)
V_AISLE = 1.3            # m/s, aisle cruising speed (ch 9.5.1)
AISLE_M = 30.0           # m, one aisle (ch 9.5.1)
UMB_LAB_SPEED = 0.2      # m/s, benchmark speed (ch 9.5.1)

# Segment 3: maps (ch 9.6)
F_TICK_US = 1000.0       # us, 1 kHz permission loop (ch 9.6.1)
T_REFLEX_US = 100.0      # us, collision-check allocation (ch 9.6.1)
N_HULLS = 50             # evaluation spheres (ch 9.6.1)
WCET_US = 250.0          # us, declared WCET of the permission job (ch 9.6.1)
SRAM_BW = 3.2e9          # B/s, permission MCU SRAM bandwidth (ch 9.6.1)
SRAM_CAP_MB = 1.05       # MB, permission MCU SRAM (ch 9.6.1)
CELL_B = 8               # B per ESDF cell (ch 9.6.1)
OCC_B = 4                # B per occupancy cell (book figure, ch 9.6.1)
R_HULL = 0.25            # m, sphere-hull radius (ch 9.6.1)
L_MAP = 10.0             # m, host map edge for octree depth (book calc cell)
SLAB_M3 = 2.86           # m^3, stopping-corridor snapshot (ch 9.6.1)
PITCHES_CM = (2, 5, 10)
N_TRILINEAR = 8
WORKCELL_M3 = 300.0      # m^3 (ch NM 9.1)
PITCH_HOST = 0.01        # m (ch NM 9.1)
SURFACE_M2 = 120.0       # m^2 (ch NM 9.1)
BAND_HALF = 0.04         # m (ch NM 9.1)
BLOCK_EDGE = 8           # voxels, nvblox layout (ch NM 9.1)
RAYS = 640 * 480         # VGA depth (ch 9.6.1)
DEPTH_HZ = 30
RAY_LEN, RAY_PITCH = 2.0, 0.02
SOC_BW = 204e9           # B/s host SoC peak (ch 9.6.1)
FM_TFLOP, ORIN_TFLOPS, CAM_HZ = 3.0, 43.0, 60   # ch 9.6.2
# nvblox paper (Millane et al., ICRA 2024, arXiv 2311.00626), Table II,
# "Average" row: incremental ESDF runtime in ms
ESDF_MS = {"nvblox (GPU)": 1.9, "Fiesta (CPU)": 13.0, "voxblox (CPU)": 58.0}
NVBLOX_QUERY_NS = 3      # Table III caption: Jetson, uncorrelated queries

# Segment 4: expiry (ch 9.7 to 9.10)
A_DIST = 50.0            # mm/s^2, disturbance acceleration bound (ch 9.7.3)
R_TOL_TIGHT = 0.30       # mm (ch 9.7.3)
E0_COWORKER, TOL_COWORKER = 15.0, 150.0   # mm (ch table 4)
BIAS_RATE = 0.2          # N/min (ch 9.8)
T_DRIFT_MIN = 15.0       # min (ch 9.8)
F_READ = 5.0             # N (ch 9.8)
E0_MUG = 3.0             # mm (ch NM 9.3)
V_BELT = 0.20            # m/s (ch NM 9.3)
TOL_MUG = 15.0           # mm (ch NM 9.3)
T_USE_MS = 35.0          # ms (ch NM 9.3)
FROZEN_MS = 100.0        # ms (ch 9.9)
SKEW_MS = 20.0           # ms (ch 9.11)
SEP_PITFALL = 0.200      # m (ch 9.11)


def kalman():
    head("Predict and update (slide: The filter predicts, then each update narrows)")
    sig_odo = V_DIST * DT_SWEEP * 1000
    sig_pred = math.hypot(SIG0, sig_odo)
    g_l = sig_pred**2 / (sig_pred**2 + SIG_LIDAR**2)
    sig_l = math.sqrt((1 - g_l) * sig_pred**2)
    g_s = sig_l**2 / (sig_l**2 + SIG_STEREO**2)
    sig_s = math.sqrt((1 - g_s) * sig_l**2)
    show("odometry std over one sweep (mm)", f"{sig_odo:.1f}")
    show("predicted std (mm)", f"{sig_pred:.1f}")
    show("lidar gain", f"{g_l:.2f}")
    show("after lidar std (mm)", f"{sig_l:.1f}")
    show("stereo gain", f"{g_s:.2f}")
    show("after stereo std (mm)", f"{sig_s:.1f}")
    return dict(sig_odo=sig_odo, sig_pred=sig_pred, g_l=g_l, sig_l=sig_l,
                g_s=g_s, sig_s=sig_s)


def misdated(k):
    head("Misdated update (slide: A misdated update moves the mean, not the spread)")
    shift = k["g_s"] * MISDATE
    nsig = shift / k["sig_s"]
    show("estimate pulled away (mm)", f"{shift:.1f}")
    show("error in reported std", f"{nsig:.1f}")
    # Checkpoint: stereo std doubled
    s2 = 2 * SIG_STEREO
    g2 = k["sig_l"]**2 / (k["sig_l"]**2 + s2**2)
    f2 = math.sqrt((1 - g2) * k["sig_l"]**2)
    show("checkpoint: stereo 7.6 mm -> gain", f"{g2:.2f}")
    show("checkpoint: fused std (mm)", f"{f2:.1f}")
    return dict(shift=shift, nsig=nsig)


def gates(k):
    head("NIS gates (slide: Does one gate catch a misdated update?)")
    s_innov = k["sig_l"]**2 + SIG_STEREO**2
    lam = MISDATE**2 / s_innov
    p95 = stats.ncx2.sf(GATE95, 1, lam)
    p99 = stats.ncx2.sf(GATE99, 1, lam)
    lo = stats.chi2.ppf(0.025, N_TRIALS) / N_TRIALS
    hi = stats.chi2.ppf(0.975, N_TRIALS) / N_TRIALS
    show("noncentrality (bias^2 / S)", f"{lam:.2f}")
    show("expected NIS", f"{1 + lam:.2f}")
    show("P(caught at 3.84)", f"{p95:.2f}")
    show("P(caught at 6.63)", f"{p99:.2f}")
    show("50-trial mean band", f"{lo:.2f} to {hi:.2f}")
    return dict(p95=p95, p99=p99, lo=lo, hi=hi, nis=1 + lam)


def imu():
    head("Dead reckoning (slide: Over one second, the IMU's bias outgrows its noise)")
    gyro_noise = GYRO_ND * math.sqrt(T_IMU)
    gyro_bias = GYRO_BIAS * T_IMU
    lateral = RANGE_UPRIGHT * math.tan(math.radians(gyro_bias)) * 1000
    acc_bias = 0.5 * ACC_BIAS * G * T_IMU**2 * 1000
    acc_noise = ACC_ND * G * math.sqrt(T_IMU**3 / 3) * 1000
    show("gyro noise heading (deg)", f"{gyro_noise:.3f}")
    show("gyro bias heading (deg)", f"{gyro_bias:.0f}")
    show("lateral at 1.10 m (mm)", f"{lateral:.1f}")
    show("accel bias position (mm)", f"{acc_bias:.1f}")
    show("accel noise position (mm)", f"{acc_noise:.1f}")
    show("bias / localization bound", f"{acc_bias / (DELTA_LOC * 1000):.1f}x")
    return dict(gyro_noise=gyro_noise, lateral=lateral, acc_bias=acc_bias,
                acc_noise=acc_noise)


def tracker():
    head("Occupancy before identity (slide: SORT code)")
    wait_ms = (SORT_MIN_HITS - 1) * FRAME_MS
    travel = V_DRIVE * wait_ms
    show("frames before confirmation", SORT_MIN_HITS)
    show("wait after first detection (ms)", f"{wait_ms:.1f}")
    show("travel at drive limit (mm)", f"{travel:.1f}")
    slots = math.ceil(T_AGE_MS / IMU_RATE_MS)
    show("ring buffer slots for 51.6 ms at 1 kHz", slots)
    return dict(wait=wait_ms, travel=travel, slots=slots)


def epochs():
    head("Three epochs (slide: A stamped pose carries one time)")
    gap = T_ENC - T_CAM
    show("camera-to-encoder gap (ms)", f"{gap:.1f}")
    show("tool travel at 1.7 m/s (mm)", f"{V_ARM * gap:.2f}")
    show("sync margin at drive limit (mm)", f"{V_DRIVE * SYNC_MS:.1f}")


def lever():
    head("Frames age (slide: A stale camera edge misses the slot)")
    e_tr = V_DRIFT * FRAME_MS / 1000
    e_rot = R_ARM * 1000 * OMEGA * FRAME_MS / 1000
    tot = E0_SLOT + e_tr + e_rot
    show("drift term (mm)", f"{e_tr:.2f}")
    show("lever-arm term (mm)", f"{e_rot:.2f}")
    show("total (mm)", f"{tot:.2f}")
    show("over tolerance (percent)", f"{(tot / R_TOL - 1) * 100:.0f}")
    show("lever / drift", f"{e_rot / e_tr:.1f}")
    return dict(e_tr=e_tr, e_rot=e_rot, tot=tot)


def umbmark():
    head("Odometry drift (slide: UMBmark)")
    r_unc = UMB_UNCAL / (UMB_PATH * 1000)
    r_cal = UMB_CAL / (UMB_PATH * 1000)
    d_unc = DELTA_LOC / r_unc
    d_cal = DELTA_LOC / r_cal
    show("uncalibrated rate (percent)", f"{r_unc * 100:.1f}")
    show("calibrated rate (percent)", f"{r_cal * 100:.2f}")
    show("calibration gain", f"{UMB_UNCAL / UMB_CAL:.1f}x")
    show("uncalibrated: bound spent after (m)", f"{d_unc:.1f}")
    show("uncalibrated: seconds at 1.3 m/s", f"{d_unc / V_AISLE:.1f}")
    show("uncalibrated: drift over 30 m (mm)", f"{r_unc * AISLE_M * 1000:.1f}")
    show("uncalibrated: times the bound", f"{r_unc * AISLE_M / DELTA_LOC:.1f}x")
    show("calibrated: bound spent after (m)", f"{d_cal:.1f}")
    show("calibrated: seconds at 1.3 m/s", f"{d_cal / V_AISLE:.1f}")
    show("calibrated: drift over 30 m (mm)", f"{r_cal * AISLE_M * 1000:.1f}")
    show("production / lab speed", f"{V_AISLE / UMB_LAB_SPEED:.1f}x")
    return dict(r_unc=r_unc, r_cal=r_cal, d_unc=d_unc, d_cal=d_cal)


def kitti():
    head("KITTI (slide: Every ranked odometry method spends 50 mm within one aisle)")
    from mlsysim import Literature
    import statistics
    rows = Literature.KittiOdometry.entries
    st = [r["translation_err_pct"] for r in rows if r["sensor_group"] == "stereo"]
    la = [r["translation_err_pct"] for r in rows if r["sensor_group"] == "laser"]
    best = min(min(st), min(la))
    allv = [r["translation_err_pct"] for r in rows]
    show("ranked methods", len(rows))
    show("stereo / lidar / other counts", f"{len(st)} / {len(la)} / {len(rows) - len(st) - len(la)}")
    show("best (percent)", best)
    show("median stereo (percent)", round(statistics.median(st), 2))
    show("median lidar (percent)", round(statistics.median(la), 2))
    show("best: meters to spend 50 mm", f"{DELTA_LOC / (best / 100):.1f}")
    show("median lidar: meters", f"{DELTA_LOC / (statistics.median(la) / 100):.1f}")
    show("median stereo: meters", f"{DELTA_LOC / (statistics.median(st) / 100):.1f}")
    show("aisle error threshold (percent)", f"{100 * DELTA_LOC / AISLE_M:.3f}")
    show("methods below that threshold", sum(1 for v in allv if v < 100 * DELTA_LOC / AISLE_M))
    return rows


def budget():
    head("Read budget (slide: The permission controller gets 2 us per hull)")
    t_q = T_REFLEX_US / N_HULLS
    reads = SRAM_BW * t_q * 1e-6 / CELL_B
    show("per-hull allocation (us)", f"{t_q:.0f}")
    show("read budget per query", f"{reads:.0f}")
    show("collision share of WCET (percent)", f"{100 * T_REFLEX_US / WCET_US:.0f}")
    show("tick left for other work (percent)", f"{100 * (1 - T_REFLEX_US / F_TICK_US):.0f}")
    return dict(t_q=t_q, reads=reads)


def reads():
    head("Reads per query (slide: Only a distance field answers in eight reads)")
    reach = R_HULL + DELTA_MARGIN
    out = {}
    for cm in PITCHES_CM:
        r = cm / 100
        scan = 4 / 3 * math.pi * (reach / r) ** 3
        depth = math.ceil(math.log2(L_MAP / r))
        cells = SLAB_M3 / r**3
        out[cm] = dict(occ=scan, depth=depth, oct=scan * depth, esdf=N_TRILINEAR,
                       esdf_kb=cells * CELL_B / 1e3, occ_kb=cells * OCC_B / 1e3,
                       quant=math.sqrt(3) / 2 * r * 1000)
        show(f"{cm} cm: occupancy reads", f"{scan:,.0f}")
        show(f"{cm} cm: octree depth / reads", f"{depth} / {scan * depth:,.0f}")
        show(f"{cm} cm: ESDF snapshot (kB)", f"{cells * CELL_B / 1e3:,.0f}")
        show(f"{cm} cm: quantization (mm)", f"{math.sqrt(3) / 2 * r * 1000:.0f}")
    return out


def empty_air():
    head("Empty-air tax (NM 9.1)")
    vox = WORKCELL_M3 / PITCH_HOST**3
    dense = vox * CELL_B
    band = 2 * BAND_HALF * SURFACE_M2
    blocks = band / (BLOCK_EDGE * PITCH_HOST) ** 3
    hashed = blocks * BLOCK_EDGE**3 * CELL_B
    show("voxels", f"{vox:.1e}")
    show("dense (GB)", f"{dense / 1e9:.1f}")
    show("band (m^3)", f"{band:.1f}")
    show("band share (percent)", f"{100 * band / WORKCELL_M3:.1f}")
    show("blocks", f"{blocks:,.0f}")
    show("hashed (MB)", f"{hashed / 1e6:.1f}")
    show("ratio", f"{dense / hashed:.0f}x")
    bw = RAYS * DEPTH_HZ * (RAY_LEN / RAY_PITCH) * CELL_B
    show("ray-marching traffic (GB/s)", f"{bw / 1e9:.2f}")
    show("share of 204 GB/s (percent)", f"{100 * bw / SOC_BW:.1f}")
    show("foundation model at 60 Hz (TFLOP/s)", f"{FM_TFLOP * CAM_HZ:.0f}")
    show("vs Orin peak", f"{FM_TFLOP * CAM_HZ / ORIN_TFLOPS:.1f}x")
    show("nvblox / voxblox ESDF speedup", f"{ESDF_MS['voxblox (CPU)'] / ESDF_MS['nvblox (GPU)']:.1f}x")
    return dict(dense=dense, hashed=hashed)


def erosion():
    head("Rack end (NM 9.2)")
    t_ab = (SIDE_CLEAR - DELTA_LOC - DELTA_MARGIN) / V_WALK
    t_ah = (D_CLEAR - DELTA_LOC - DELTA_MARGIN) / V_WALK
    left = t_ah * 1000 - T_AGE_MS
    show("abreast (ms)", f"{t_ab * 1000:.2f}")
    show("ahead (ms)", f"{t_ah * 1000:.2f}")
    show("left after frame age (ms)", f"{left:.2f}")
    show("frame periods left", f"{left / FRAME_MS:.1f}")
    show("depth spent by frame age (mm)", f"{V_WALK * T_AGE_MS:.1f}")
    show("recall: base travel during frame age at 1.5 m/s (mm)", f"{V_DRIVE * T_AGE_MS:.1f}")
    show("ms per mm of depth", f"{1 / V_WALK:.3f}")
    show("initial certified depth (mm)", f"{(D_CLEAR - DELTA_LOC - DELTA_MARGIN) * 1000:.0f}")
    t_stop = V_DRIVE / A_BRAKE
    intr = V_WALK * (TAU_DELAY + t_stop)
    show("walking intrusion over delay + stop (m)", f"{intr:.2f}")
    return dict(t_ah=t_ah * 1000, t_ab=t_ab * 1000)


def quad_root(e0, v, a_half, tol):
    # a_half * t^2 + v * t + (e0 - tol) = 0, t in s, e in mm
    c = e0 - tol
    return (-v + math.sqrt(v * v - 4 * a_half * c)) / (2 * a_half)


def latch():
    head("Latch slot (ch 9.7.3)")
    t = quad_root(E0_SLOT, V_DRIFT, A_DIST / 2, R_TOL)
    t_lin = (R_TOL - E0_SLOT) / V_DRIFT
    e_at = E0_SLOT + V_DRIFT * t_lin + A_DIST / 2 * t_lin**2
    t_tight = quad_root(E0_SLOT, V_DRIFT, A_DIST / 2, R_TOL_TIGHT)
    one = E0_SLOT + V_DRIFT * FRAME_MS / 1000 + A_DIST / 2 * (FRAME_MS / 1000) ** 2
    show("horizon (ms)", f"{t * 1000:.1f}")
    show("constant-velocity horizon (ms)", f"{t_lin * 1000:.1f}")
    show("bounded error at 75 ms (mm)", f"{e_at:.3f}")
    show("over tolerance (percent)", f"{(e_at / R_TOL - 1) * 100:.1f}")
    show("blind overrun (ms)", f"{(t_lin - t) * 1000:.1f}")
    show("tight slot horizon (ms)", f"{t_tight * 1000:.1f}")
    show("tolerance ratio", f"{R_TOL / R_TOL_TIGHT:.1f}x")
    show("horizon ratio", f"{t / t_tight:.1f}x")
    show("error over one frame, common epoch (mm)", f"{one:.2f}")
    return dict(t=t * 1000, t_lin=t_lin * 1000, e_at=e_at, one=one)


def beliefs():
    head("Four beliefs (ch table 4)")
    cw = (TOL_COWORKER - E0_COWORKER) / (V_WALK * 1000) * 1000
    cw100 = (100 - E0_COWORKER) / (V_WALK * 1000) * 1000
    show("coworker (ms)", f"{cw:.1f}")
    show("checkpoint: coworker at 100 mm (ms)", f"{cw100:.1f}")
    ahead = (D_CLEAR - DELTA_LOC - DELTA_MARGIN) / V_WALK
    show("100 ms timeout: certified time refused ahead (s)", f"{ahead - 0.100:.2f}")


def observability():
    head("Force bias (ch 9.8)")
    shift = BIAS_RATE * T_DRIFT_MIN
    true = F_READ - shift
    show("bias shift (N)", f"{shift:.0f}")
    show("true force (N)", f"{true:.0f}")
    show("grip lost (percent)", f"{100 * shift / F_READ:.0f}")


def mug():
    head("Mug admission (NM 9.3)")
    t_inv = (TOL_MUG - E0_MUG) / (V_BELT)            # mm / (m/s) -> ms
    up = T_USE_MS + SYNC_MS
    e = E0_MUG + V_BELT * up
    show("horizon (ms)", f"{t_inv:.0f}")
    show("upper age (ms)", f"{up:.1f}")
    show("error at use (mm)", f"{e:.2f}")
    show("margin (mm)", f"{TOL_MUG - e:.2f}")
    show("lifetime left (ms)", f"{t_inv - up:.1f}")
    show("limiting use age (ms)", f"{t_inv - SYNC_MS:.1f}")


def frozen():
    head("Frozen transform (ch 9.9) and skew pitfall (ch 9.11)")
    show("mug error after 100 ms (mm)", f"{V_BELT * FROZEN_MS:.0f}")
    closing = V_DRIVE + V_WALK
    env = closing * SKEW_MS
    slack = (SEP_PITFALL - DELTA_LOC - DELTA_MARGIN) * 1000
    show("closing speed (m/s)", f"{closing:.1f}")
    show("skew envelope (mm)", f"{env:.0f}")
    show("slack (mm)", f"{slack:.0f}")


if __name__ == "__main__":
    k = kalman()
    misdated(k)
    gates(k)
    imu()
    tracker()
    epochs()
    lever()
    umbmark()
    kitti()
    budget()
    reads()
    empty_air()
    erosion()
    latch()
    beliefs()
    observability()
    mug()
    frozen()
