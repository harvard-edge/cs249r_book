#!/usr/bin/env python3
"""Every derived number in Lecture 5 (Collective Communication), computed.

Inputs carry their source as a comment. Each print line names the slide (or
note) that shows the value. Run: python3 calcs.py
"""

import math

KB, MB, GB = 1e3, 1e6, 1e9
US, MS = 1e-6, 1e-3


def show(label, value):
    print(f"{label:<62} {value}")


# --- Hook: Horovod at 128 Pascal GPUs (read from Sergeev and Del Balso 2018, Fig. 6)
ideal = {"Inception V3": 17000, "ResNet-101": 15000}
tf = {"Inception V3": 9000, "ResNet-101": 6200}
hv = {"Inception V3": 15000, "ResNet-101": 13200}
for m in ideal:
    show(f"Hook: TensorFlow efficiency, {m}", f"{100 * tf[m] / ideal[m]:.0f}%")
    show(f"Hook: Horovod efficiency, {m}", f"{100 * hv[m] / ideal[m]:.0f}%")

# --- FSDP call counts (Sec. 5.4.2: 2L, or 3L with resharding); L = 32 (Napkin Math 5.9 model)
L = 32
show("FSDP slide note: calls per step, 32 layers, reshard (3L)", 3 * L)

# --- Running case link: InfiniBand NDR alpha, beta (Napkin Math 5.1)
alpha, beta = 1.5 * US, 50 * GB
show("Critical size NDR, alpha*beta (KB)", f"{alpha * beta / KB:.0f}")

# --- 25 MB bucket on 64 and 1,024 ranks (Napkin Math 5.1)
M = 25 * MB
res = {}
for N in (64, 1024):
    chunk = M / N
    rounds = 2 * (N - 1)
    byte = rounds * chunk / beta
    total = rounds * (alpha + chunk / beta)
    res[N] = total
    show(f"Bucket slide: N={N} chunk (KB)", f"{chunk / KB:.1f}")
    show(f"Bucket slide: N={N} rounds", rounds)
    show(f"Bucket slide: N={N} byte term (ms)", f"{byte / MS:.1f}")
    show(f"Bucket slide: N={N} total (ms)", f"{total / MS:.1f}")
show("Bucket slide title: 1,024 vs 64 ratio", f"{res[1024] / res[64]:.1f}x")
show("Bucket note: 25 MB / 75 KB knee", f"{M / (alpha * beta):.0f}")
fused_sep = 100 * alpha
fused_one = alpha + 400 * KB / beta
show("Fusion: 100 x 4 KB separately (us)", f"{fused_sep / US:.0f}")
show("Fusion: one 400 KB message (us)", f"{fused_one / US:.1f}")
show("Fusion: ratio", f"{fused_sep / fused_one:.0f}x")

# --- Scoreboard: 1 GB buffer, 64 GPUs, 50 GB/s (Sec. 5.6.1, 5.6.2, 5.6.4)
Mref, N = 1 * GB, 64
ps = N * Mref / beta
ring_b = 2 * (N - 1) / N * Mref / beta
tree_b = 2 * math.log2(N) * Mref / beta
show("Scoreboard: parameter server N*M/beta (ms)", f"{ps / MS:,.0f}")
show("Scoreboard: ring byte term (ms)", f"{ring_b / MS:.1f}")
show("Scoreboard: simple tree, 2 log2 N transfers (ms)", f"{tree_b / MS:.0f}")
show("Scoreboard: server/ring ratio (title, ~N/2)", f"{ps / ring_b:.1f}x (N/2 = {N // 2})")

# --- Estimate: 70B BF16 gradient on 64 GPUs (Napkin Math 5.2)
P = 70e9
Mg = P * 2
bw = 2 * (N - 1) / N * Mg / beta
lat = 2 * (N - 1) * alpha
show("Estimate: payload (GB)", f"{Mg / GB:.0f}")
show("Estimate: factor 2(N-1)/N", 2 * (N - 1) / N)
show("Estimate: byte term (ms)", f"{bw / MS:,.1f}")
show("Estimate: startup term (ms)", f"{lat / MS:.2f}")
show("Estimate: startup share of total (%)", f"{100 * lat / (bw + lat):.4f}")
saved = bw / 2
show("Estimate: halving saves per step (s)", f"{saved:.2f}")
show("Estimate: GPU-hours over 100,000 steps on 64 GPUs", f"{saved * 64 * 100000 / 3600:,.0f}")

# --- Rounds at N = 1,024 (Sec. 5.6.4)
N = 1024
show("Rounds slide: ring 2(N-1)", 2 * (N - 1))
show("Rounds slide: tree 2 log2 N", int(2 * math.log2(N)))
show("Rounds note: ratio", f"{2 * (N - 1) / (2 * math.log2(N)):.0f}x")

# --- Crossover: N = 64, alpha = 10 us, beta = 10 GB/s, M = 1 MB (Napkin Math 5.3)
N, a, b, Mx = 64, 10 * US, 10 * GB, 1 * MB
r_lat, t_lat = 2 * (N - 1) * a, 2 * math.log2(N) * a
r_bw, t_bw = 2 * (N - 1) / N * Mx / b, 2 * math.log2(N) * Mx / b
show("Crossover table: ring startup (us)", f"{r_lat / US:,.0f}")
show("Crossover table: tree startup (us)", f"{t_lat / US:,.0f}")
show("Crossover table: ring bytes (us)", f"{r_bw / US:.1f}")
show("Crossover table: tree bytes (us)", f"{t_bw / US:,.0f}")
show("Crossover table: ring total (us)", f"{(r_lat + r_bw) / US:,.1f}")
show("Crossover table: tree total (us)", f"{(t_lat + t_bw) / US:,.0f}")
show("Crossover note: tree faster by (%)", f"{100 * (1 - (t_lat + t_bw) / (r_lat + r_bw)):.0f}")
show("Crossover estimate N a b / log2 N (MB)", f"{N * a * b / math.log2(N) / MB:.1f}")
exact = (2 * (N - 1) - 2 * math.log2(N)) * a * b / (2 * math.log2(N) - 2 * (N - 1) / N)
show("Crossover exact (MB), drawn on plot", f"{exact / MB:.2f}")
show("Crossover on 1,024 NDR GPUs (MB)", f"{1024 * alpha * beta / math.log2(1024) / MB:.1f}")

# --- NVLink against InfiniBand (Sec. 5.7.1)
show("NVLink / InfiniBand per GPU", f"{450 / 50:.0f}x")

# --- Hierarchical AllReduce, 1 GB, 8 nodes of 8 (Napkin Math 5.4)
G, nodes, nvl = 8, 8, 450 * GB
p1 = (G - 1) / G * Mref / nvl
shard = Mref / G
p2 = 2 * (nodes - 1) / nodes * shard / beta + 2 * (nodes - 1) * alpha
flat = 2 * Mref / beta
show("Hier: phase 1 bytes per GPU (MB)", f"{(G - 1) / G * Mref / MB:.0f}")
show("Hier: phase 1 and 3 (ms)", f"{p1 / MS:.2f}")
show("Hier: shard (MB)", f"{shard / MB:.0f}")
show("Hier: phase 2 (ms)", f"{p2 / MS:.2f}")
show("Hier: total (ms)", f"{(2 * p1 + p2) / MS:.1f}")
show("Hier: flat ring 2M/beta (ms)", f"{flat / MS:.0f}")
show("Hier: speedup", f"{flat / (2 * p1 + p2):.1f}x")
show("Hier note: 175B BF16 gradient (GB)", f"{175e9 * 2 / GB:.0f}")
show("Hier note: 14.0 s / 3.1 s (chapter values)", f"{14.0 / 3.1:.1f}x")

# --- MoE dispatch (Napkin Math 5.5, 5.6)
tok_b = 2048 * 2  # bytes per token, BF16
per_pair = 64 * tok_b
sent = 7 * per_pair
show("MoE: bytes per token (KB)", f"{tok_b / KB:.3f}")
show("MoE: per pair (KB)", f"{per_pair / KB:.3f}")
show("MoE: sent per GPU (MB)", f"{sent / MB:.2f}")
show("MoE: uniform receive bound (us)", f"{sent / beta / US:.1f}")
show("MoE: injection bound alpha + bytes/beta (us)", f"{(alpha + sent / beta) / US:.1f}")
show("MoE: hot receiver tokens 448 x 1.5", f"{448 * 1.5:.0f}")
show("MoE: hot receiver bytes (MB)", f"{672 * tok_b / MB:.2f}")
show("MoE: hot receive bound (us)", f"{672 * tok_b / beta / US:.1f}")
show("MoE note: seven ports' worth (GB/s)", 7 * 50)

# --- Compression payback (Napkin Math 5.7) and run level (Sec. 5.9.5)
comm, r, codec = 40, 8, 2
show("Payback: compressed transfer (ms)", comm / r)
show("Payback: total (ms)", comm / r + codec)
show("Payback: speedup", f"{comm / (comm / r + codec):.1f}x")
for name, c in (("slow", 200), ("fast", 30)):
    base = 100000 * (300 + c) / 1000
    comp = 110000 * (300 + c / 4) / 1000
    show(f"Run level {name}: baseline (s)", f"{base:,.0f}")
    show(f"Run level {name}: compressed (s)", f"{comp:,.0f}")
    show(f"Run level {name}: change (%)", f"{100 * (comp - base) / base:+.1f}")

# --- Error feedback trace (Example 5.2): round to nearest whole unit
g = [0.4, 0.3, 0.2, 0.4, 0.3]
e, sent_total = 0.0, 0
for t, gt in enumerate(g, 1):
    v = round(gt + e + 1e-12)
    show(f"EF step {t}: g, e, g+e, v, new e",
         f"{gt}, {e:+.1f}, {gt + e:.1f}, {v}, {gt + e - v:+.1f}")
    e = gt + e - v
    sent_total += v
show("EF: sent total, final residual, true total", f"{sent_total}, {e:+.1f}, {sum(g):.1f}")
e, sends = 0.0, 0
for _ in range(10):
    v = round(0.3 + e + 1e-12)
    e = 0.3 + e - v
    sends += v
show("EF note: constant 0.3 for 10 steps, units sent", sends)
show("1-bit vs FP32 raw ratio", "32x")

# --- Overlap budget (Napkin Math 5.9; 7.6 ms per layer from Sec. 5.10.4)
Lyr, tb, tc = 32, 15, 27
seq = Lyr * tb + Lyr * tc
ov = Lyr * tb + tc + (Lyr - 1) * (tc - tb)
show("Overlap: sequential (ms)", f"{seq:,}")
show("Overlap: overlapped (ms)", ov)
show("Overlap: backlog per layer (ms)", tc - tb)
show("Overlap: reduction (%)", f"{100 * (1 - ov / seq):.1f}")
show("Overlap: 9 buckets x 0.85 ms (ms)", f"{9 * 0.85:.2f} (chapter: about 7.6)")
show("Overlap: faster buckets, 480 + 7.6 (ms)", f"{Lyr * tb + 7.6:.1f}")

# --- Horovod Timeline (Sergeev and Del Balso 2018, Figure 5), redrawn as
# horovod_trace.svg. Phase edges read from the screenshot against its
# 18,500 to 18,900 ms ticks (about +/-2 ms), in ms after 18,500 ms. One value is
# printed in the screenshot's detail pane: tensors 2 to 4 start ALLREDUCE at
# 18,663.501 ms and run 184.063 ms. Each row of the timeline is one tensor.
HVD_T0 = 18500.0
hvd_fast = {"negotiate": (23, 35), "wait for data": (35, 124), "NCCL": (126, 138)}
hvd_ar_start = 18663.501 - HVD_T0      # detail pane
hvd_ar_wall = 184.063                  # detail pane
hvd_slow = {"negotiate": (23, 154), "wait for other tensors": (165, 273),
            "schedule": (274, 312), "NCCL": (318, 337)}
slow_total = hvd_ar_start + hvd_ar_wall - hvd_slow["negotiate"][0]
show("Horovod slide: tensors 2-4, negotiate (ms)", f"{hvd_slow['negotiate'][1] - hvd_slow['negotiate'][0]:.0f}")
show("Horovod slide: tensors 2-4, ALLREDUCE ends (ms after 18,500)", f"{hvd_ar_start + hvd_ar_wall:.1f}")
show("Horovod slide: tensors 2-4, negotiate start to end (ms)", f"{slow_total:.0f}")
show("Horovod slide: tensors 2-4, NCCL (ms)", f"{hvd_slow['NCCL'][1] - hvd_slow['NCCL'][0]:.0f}")
show("Horovod note: NCCL share of tensors 2-4 span", f"{100 * (hvd_slow['NCCL'][1] - hvd_slow['NCCL'][0]) / slow_total:.0f}%")
show("Horovod note: tensors 0-1, negotiate start to end (ms)", f"{hvd_fast['NCCL'][1] - hvd_fast['negotiate'][0]:.0f}")
