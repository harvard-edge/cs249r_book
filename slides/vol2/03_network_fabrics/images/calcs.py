#!/usr/bin/env python3
"""Every derived number in Lecture 3 (Network Fabrics), computed from sourced
inputs. Run: python3 calcs.py. Each line prints the value as the slide or
note shows it, with its slide label. Inputs cite the rendered chapter
(Network Fabrics, Vol. II Ch. 3), Compute Infrastructure (Ch. 2), or MLSysIM.
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Literature  # noqa: E402

GB = 1e9
MB = 1e6
KB = 1e3


def show(slide, label, value):
    print(f"[{slide}] {label}: {value}")


# --- Recall: staircase (Compute Infrastructure, Napkin Math 2.3) -------------
payload = 10 * GB                       # 10 GB buffer
for name, bw in (("HBM", 3350), ("NVLink", 450), ("InfiniBand", 50)):  # GB/s per direction
    show("Recall", f"10 GB over {name}", f"{payload / (bw * GB) * 1e3:.1f} ms")

# --- Node cliff (MLSysIM interconnect series; Network Fabrics Sec. 3.1) -----
nvlink_h100_gbps = 3600                 # MLSysIM: NVLink 4 (H100), per direction, Gb/s
ndr_gbps = 400                          # MLSysIM: NDR (4x), per direction, Gb/s
show("Node cliff", "H100 NVLink per direction", f"{nvlink_h100_gbps / 8:.0f} GB/s")
show("Node cliff", "NDR port per direction", f"{ndr_gbps / 8:.0f} GB/s")
show("Node cliff", "ratio", f"{nvlink_h100_gbps / ndr_gbps:.0f}x")

# --- Exposed AllReduce (figure 2, illustrative inputs) -----------------------
step_ms = 100 + 100 + 30
show("Timeline", "exposed share", f"30 of {step_ms} ms = {30 / step_ms:.0%}")
show("Timeline", "compute share, inter-node", f"{200 / step_ms:.0%}")

# --- ECMP (Sec. 3.6.4: 4 paths, 8 flows; hashed 4/2/2/0) ---------------------
show("ECMP", "worst over even drain time", f"{4 / 2:.0f}x")

# --- Running case: per-rank shard (Sec. 3.1) ---------------------------------
grad_bytes = 175e9 * 2                  # 175B parameters, 2 bytes per 16-bit gradient
show("Goals/Napkin", "gradient state", f"{grad_bytes / GB:.0f} GB")
show("Napkin", "1/1,024 share", f"{grad_bytes / 1024 / MB:.0f} MB (chapter rounds the shard to 350 MB)")

# --- Optics cost (Systems Perspective 3.2: assumptions) ----------------------
links = 10_000 * 2                      # two spine links per GPU, assumed
show("Optics", "spine links", f"{links:,}")
show("Optics", "cabling", f"${links * 500 / 1e6:.0f}M")       # $500 per link, assumed
show("Optics", "module power", f"{links * 20 / 1e3:.0f} kW")  # 20 W per module, assumed

# --- Go-Back-N window (Sec. 3.4.3) -------------------------------------------
show("IB/RoCE", "resend window", f"{50 * GB * 5e-6 / KB:.0f} KB")   # 50 GB/s x 5 us round trip


# --- Alpha-beta (Napkin Math 3.1) --------------------------------------------
def t_us(alpha_us, beta_gbs, n_bytes):
    return alpha_us + n_bytes / (beta_gbs * GB) * 1e6


ndr = (1.5, 50.0)                        # alpha us, beta GB/s (modeled NDR path)
roce100 = (5.0, 12.5)                    # modeled RoCE path, 100 Gb/s
roce400 = (5.0, 50.0)                    # RoCE alpha at 400 Gb/s
small, large = 4 * KB, 350 * MB
a, b = t_us(*ndr, small), t_us(*roce100, small)
show("Napkin", "4 KB NDR / RoCE", f"{a:.2f} us / {b:.2f} us, {b / a:.1f}x")
show("Napkin", "4 KB bandwidth share on NDR", f"{(a - 1.5) / a:.1%}")
a, b = t_us(*ndr, large), t_us(*roce100, large)
show("Napkin", "350 MB NDR / RoCE", f"{a / 1e3:.1f} ms / {b / 1e3:.2f} ms, {b / a:.0f}x")
show("Napkin", "RoCE 400G", f"{t_us(*roce400, small):.2f} us; {t_us(*roce400, large) / 1e3:.1f} ms")
show("Napkin", "crossover NDR", f"{1.5e-6 * 50 * GB / KB:.0f} KB")
show("Crossover", "crossover RoCE 100G", f"{5e-6 * 12.5 * GB / KB:.1f} KB")

# --- 10 KB estimate (Sec. 3.11) ----------------------------------------------
hdr, ndr10 = t_us(1.5, 25.0, 10 * KB), t_us(1.5, 50.0, 10 * KB)
show("Estimate", "10 KB HDR / NDR", f"{hdr:.2f} / {ndr10:.2f} us, {(hdr - ndr10) / hdr:.1%} faster")

# --- NCCL measured (MLSysIM NCCLAllReduceH100Node) ----------------------------
rows = Literature.Series.NCCLAllReduceH100Node.entries
floor = statistics.median(r["oop_time_us"] for r in rows if r["size_bytes"] <= 64 * 1024)
bw = rows[-1]["oop_algbw_GBps"]
knee = floor * 1e-6 * bw * GB
ratio = max(r["oop_time_us"] / (floor + r["size_bytes"] / (bw * GB) * 1e6)
            for r in rows if 8 * 2**20 <= r["size_bytes"] <= 128 * 2**20)
mid = [r["oop_busbw_GBps"] for r in rows if 8 * 2**20 <= r["size_bytes"] <= 128 * 2**20]
show("NCCL", "floor", f"{floor:.0f} us")
show("NCCL", "beta (algbw at 16 GB)", f"{bw:.0f} GB/s")
show("NCCL", "model knee", f"{knee / MB:.1f} MB")
show("NCCL", "worst ratio 8-128 MB", f"{ratio:.1f}x")
show("NCCL", "busbw factor 2(N-1)/N, N=8", f"{2 * 7 / 8:.2f}")
show("Busbw", "8-128 MB busbw", f"{min(mid):.0f} to {max(mid):.0f} GB/s")
show("Busbw", "peak busbw", f"{max(r['oop_busbw_GBps'] for r in rows):.0f} GB/s")

# --- Bisection (Def. 3.4) ----------------------------------------------------
cut = 512 * 50                           # 512 links x 50 GB/s
show("Bisection", "1:1 cut", f"{cut / 1e3:.1f} TB/s")
show("Bisection", "4:1 cut", f"{cut / 4 / 1e3:.1f} TB/s")

# --- Fat-tree (Def. 3.6, Sec. 3.5.2, 3.10.1) ----------------------------------
for k in (64, 256):
    show("Fat-tree", f"k={k} two tiers / three tiers", f"{k**2 // 2:,} / {k**3 // 4:,}")
show("Fat-tree", "k=64 three-tier switches", f"{5 * 64**2 // 4:,}")
show("Fat-tree", "102.4 Tb/s as 400G ports", f"{102.4e3 / 400:.0f}")

# --- Oversubscription (Napkin Math 3.2; 30% is a scenario input) -------------
n = 1024
factor = 2 * (n - 1) / n
crossing = 5 * factor * (n / 2)          # GB, 5 GB payload per GPU
t11, t41 = crossing / 25_600, crossing / 6_400
rel = 1 / (0.70 + 0.30 * 4)
rel2 = 1 / (0.70 + 0.30 * 2)
show("Oversub", "traffic factor", f"{factor:.3f}")
show("Oversub", "crossing bytes", f"{crossing:,.0f} GB")
show("Oversub", "1:1 / 4:1 time", f"{t11:.3f} s / {t41:.3f} s")
show("Oversub", "relative throughput", f"{rel:.2f}; loss {1 - rel:.1%}")
show("Oversub", "idle capacity", f"${(1 - rel) * 300:.1f}M of $300M")
show("Read", "2:1 loss", f"{1 - rel2:.1%}")

# --- Rails (Sec. 3.5.3, Napkin Math 3.3) --------------------------------------
show("Rails", "GPUs per rail", f"{1024 // 8}")
show("Rails", "3 vs 1 traversals at 0.6 us", f"{3 * 0.6:.1f} / {1 * 0.6:.1f} us")

# --- Torus against fat-tree at 4,096 (Sec. 3.5.4) -----------------------------
show("Torus", "endpoints", f"{16**3:,}")
show("Torus", "cut links / capacity", f"{2 * 16 * 16} / {2 * 16 * 16 * 50 / 1e3:.1f} TB/s")
show("Torus", "fat-tree cut", f"{4096 // 2:,} links / {4096 // 2 * 50 / 1e3:.1f} TB/s")
show("Torus", "worst path hops", f"{3 * (16 // 2)}")

# --- Transceivers (Napkin Math 3.4; 0.001%/day assumed) -----------------------
xcvr = 4096 * 3 * 2
show("PFC", "transceivers", f"{xcvr:,}")
show("PFC", "daily P(at least one)", f"{1 - (1 - 1e-5) ** xcvr:.1%}")

# --- Incast (Def. 3.10, deliberate worst case) --------------------------------
offered = 256 * 50                       # GB/s
show("Incast", "offered load", f"{offered / 1e3:.1f} TB/s, {offered / 50:.0f}x the egress")
show("Incast", "fill time", f"{32 * MB / ((offered - 50) * GB) * 1e6:.1f} us")
show("Incast", "drain time", f"{32 * MB / (50 * GB) * 1e6:.0f} us")

# --- Silent waste (Sec. 3.9; scenario) ----------------------------------------
gpu_h = 1024 * 24 * 30
show("Silent", "GPU-hours in 30 days", f"{gpu_h:,}")
for f in (0.02, 0.03):
    show("Silent", f"{f:.0%} slower", f"{gpu_h * f:,.0f} GPU-h, ${gpu_h * f * 2:,.0f} at $2")
show("Silent", "no-overlap slowdown, 10% loss, 30% comm",
     f"{(0.70 + 0.30 / 0.9) - 1:.1%} (upper bound; chapter says 2-3% after overlap)")
show("Silent", "expected degraded links of 3,000", f"{3000 * 0.001:.0f}")
show("Silent", "P(at least one)", f"{1 - 0.999 ** 3000:.0%}")

# --- Sharing (Sec. 3.8) -------------------------------------------------------
show("SR-IOV", "VFs at 50 Gb/s on 400 Gb/s", f"{400 // 50}")
show("SR-IOV", "100 GB at 400 Gb/s", f"{100 / 50:.0f} s")

# --- SuperPOD (Sec. 3.7.2) ----------------------------------------------------
show("SuperPOD", "GPUs per unit", f"{32 * 8}")
show("SuperPOD", "injection per unit", f"{256 * 50 / 1e3:.1f} TB/s")
show("SuperPOD", "four units", f"{4 * 256:,} GPUs")

# --- Diagnose (Checkpoint 3.5) -------------------------------------------------
show("Diagnose", "step slowdown", f"{4.8 / 4.2 - 1:.0%}")
show("Diagnose", "pair bandwidth", f"{24 / 48:.0%} of expected")
