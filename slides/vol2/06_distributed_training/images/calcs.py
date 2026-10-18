#!/usr/bin/env python3
"""Every derived number in Lecture 6 (Distributed Training), computed from
sourced inputs. Run: python3 calcs.py. Each line prints the value and the
slide (or note) that shows it. Inputs cite the rendered chapter
(Distributed Training, Vol. II) or an outside source.
"""

import math
import sys
from pathlib import Path

ROWS = []


def show(label, value, fmt="{:,.2f}"):
    ROWS.append((label, fmt.format(value)))


# --- Opening ----------------------------------------------------------------
P_REF = 175e9            # parameters, reference model (Definition 6.1)
B_STATE = 16             # bytes per parameter, mixed-precision Adam (Napkin Math 6.4)
HBM = 80                 # GB per H100 / A100 (Definition 6.1)
state = P_REF * B_STATE / 1e9
show("Running job: training state (GB)", state, "{:,.0f}")
show("Running job: GPUs of memory, ceil(state/80)", math.ceil(state / HBM), "{:.0f}")
show("Hook: batch growth 8192/256 (x)", 8192 / 256, "{:.0f}")
show("Hook: servers 256 GPUs / 8 per Big Basin server (Goyal section 4.3)", 256 / 8, "{:.0f}")
show("Hook: images per GPU 8192/256 GPUs", 8192 / 256, "{:.0f}")
GRAD = P_REF * 2 / 1e9   # FP16 gradient payload, 350 GB (Napkin Math 6.5)
N = 1024
ring = 2 * (N - 1) / N * GRAD
show("Retrieval: ring bytes per GPU, 2(N-1)/N x 350 GB", ring, "{:,.1f}")
show("Retrieval: seconds at 50 GB/s", ring / 50, "{:.1f}")

# --- Replicas ----------------------------------------------------------------
flops = 6 * P_REF * 2e6  # 6 FLOPs/param/token, 2M tokens (Napkin Math 6.5)
show("175B step FLOPs (x1e18)", flops / 1e18, "{:.2f}")
tc = flops / (N * 0.5 * 1979e12)  # FP8 peak 1,979 TFLOP/s at 50% (Napkin Math 6.5)
show("175B per-GPU compute (s)", tc, "{:.2f}")
show("175B efficiency 2.1/(2.1+11.9) (%)", 100 * 2.1 / (2.1 + 11.9), "{:.1f}")
# GPT-2 (Napkin Math 6.1, 6.2): chapter's ms values
show("GPT-2 one node efficiency 1800/1811.7 (%)", 100 * 1800 / 1811.7, "{:.1f}")
show("GPT-2 four nodes efficiency 1800/6461.7 (%)", 100 * 1800 / 6461.7, "{:.1f}")
show("GPT-2 four nodes speedup 32x1800/6461.7 (x)", 32 * 1800 / 6461.7, "{:.1f}")
show("GPT-2 check: 5.8 GB / 1.25 GB/s (s)", 5.8 / 1.25, "{:.2f}")
# MLPerf (MLSysIM series; chapter section 6.4.2)
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Literature  # noqa: E402
rows = Literature.Series.MLPerfTrainingLLMScaling.entries
specs = [("v3.1", "NVIDIA", "gpt3", "H100-SXM5"), ("v4.1", "Google", "gpt3", "TPU-v5p"),
         ("v4.1", "Google", "gpt3", "TPU-trillium"), ("v5.0", "NVIDIA", "llama31_405b", "H100-SXM5"),
         ("v5.0", "IBM+CoreWeave", "llama31_405b", "Blackwell")]
effs = []
for rnd, sub, b, a in specs:
    pts = sorted([r for r in rows if r["round"] == rnd and r["submitter"].startswith(sub)
                  and r["benchmark"] == b and a in r["accelerator"]], key=lambda r: r["accelerators"])
    effs.append((pts[0]["score_minutes"] * pts[0]["accelerators"]) / (pts[-1]["score_minutes"] * pts[-1]["accelerators"]))
    if a == "H100-SXM5" and b == "gpt3":
        eos = pts
show("MLPerf Eos GPUs ratio 10752/512 (x)", eos[-1]["accelerators"] / eos[0]["accelerators"], "{:.0f}")
show("MLPerf Eos time ratio (x)", eos[0]["score_minutes"] / eos[-1]["score_minutes"], "{:.1f}")
show("MLPerf Eos minutes at 512", eos[0]["score_minutes"], "{:.1f}")
show("MLPerf Eos minutes at 10,752", eos[-1]["score_minutes"], "{:.1f}")
show("MLPerf min efficiency (%)", 100 * min(effs), "{:.0f}")
show("MLPerf max efficiency (%)", 100 * max(effs), "{:.0f}")

# --- Batches -----------------------------------------------------------------
for n in (8, 100, 1000):
    show(f"Straggler: 0.99^{n} (%)", 100 * 0.99 ** n, "{:.4f}")
show("Straggler: expected late GPUs at 1,000", 1000 * 0.01, "{:.0f}")
bcrit = 1.6e6 / 100      # Checkpoint 6.2 inputs
show("Estimate: B_crit", bcrit, "{:,.0f}")
show("Estimate: global batch 64x4x32", 64 * 4 * 32, "{:,.0f}")
show("Estimate: 8192 / B_crit", 8192 / bcrit, "{:.2f}")
show("Estimate: crossover GPUs 16000/(4x32)", bcrit / (4 * 32), "{:.0f}")
goyal = {128: 23.49, 256: 23.60, 512: 23.48, 1024: 23.53, 2048: 23.49, 4096: 23.56, 8192: 23.74}
show("Goyal: spread up to 8k (points)", max(goyal.values()) - min(goyal.values()), "{:.2f}")

# --- Sharding ----------------------------------------------------------------
P7, NG = 7e9, 64         # Napkin Math 6.4
show("ZeRO DDP (GB)", P7 * 16 / 1e9, "{:.1f}")
show("ZeRO-1 (GB)", P7 * (2 + 2 + 12 / NG) / 1e9, "{:.1f}")
show("ZeRO-2 (GB)", P7 * (2 + 2 / NG + 12 / NG) / 1e9, "{:.1f}")
show("ZeRO-3 (GB)", P7 * 16 / NG / 1e9, "{:.2f}")
show("ZeRO-3 traffic 3/2 (x)", 3 / 2, "{:.1f}")
show("175B ZeRO-3 on 64: 2800/64 (GB)", state / 64, "{:.2f}")
show("175B ZeRO-3 + 4.8 GB activations (GB)", state / 64 + 4.8, "{:.1f}")

# --- Splitting ---------------------------------------------------------------
def bubble(p, m):
    return 100 * (p - 1) / (m + p - 1)


for p, m in ((4, 4), (8, 32), (16, 32), (4, 32), (8, 8)):
    show(f"Bubble p={p}, m={m} (%)", bubble(p, m), "{:.1f}")
show("Bubble <10% needs m > 9(p-1), p=16", 9 * 15, "{:.0f}")
show("GPipe p=m=4 idle slots 6/14 (%)", 100 * 6 / 14, "{:.1f}")
show("NVLink / NDR 450/50 (x)", 450 / 50, "{:.0f}")

# --- Placement ---------------------------------------------------------------
show("Config C GPUs 8x4x32", 8 * 4 * 32, "{:.0f}")
show("Config B DP shard 350/8 (GB)", GRAD / 8, "{:.2f}")
show("Config C DP shard 350/32 (GB)", GRAD / 32, "{:.2f}")
tB = 2 * GRAD / 8 / 50
show("Config B AllReduce 2 x 43.75/50 (s)", tB, "{:.2f}")
show("Config B efficiency, no overlap (%)", 100 * 2.1 / (2.1 + tB), "{:.1f}")
show("Config B efficiency, 75% hidden (%)", 100 * 2.1 / (2.1 + 0.25 * tB), "{:.1f}")
tC = 2 * GRAD / 32 / 50
show("Config C AllReduce (s)", tC, "{:.2f}")
show("Config C efficiency (bubble 3/35, 75% hidden) (%)",
     100 * (1 - 3 / 35) * 2.1 / (2.1 + 0.25 * tC), "{:.1f}")
show("State per GPU A (GB)", state, "{:,.0f}")
show("State per GPU B, 2800/8 (GB)", state / 8, "{:.0f}")
show("State per GPU C, 2800/32 (GB)", state / 32, "{:.1f}")
z1 = (P_REF * 4 / 32 + P_REF * 12 / 32 / 32) / 1e9
show("State per GPU C + ZeRO-1 (GB)", z1, "{:.1f}")
show("16,384 layout 8x16x128", 8 * 16 * 128, "{:,.0f}")
show("16,384 layout state 2800/128 (GB)", state / 128, "{:.1f}")
show("Recomputation 4/3 - 1 (%)", 100 * (4 / 3 - 1), "{:.0f}")
pol, ref, rew, val = 70e9 * 16 / 1e9, 70e9 * 2 / 1e9, 13e9 * 2 / 1e9, 13e9 * 16 / 1e9
show("PPO policy (GB)", pol, "{:,.0f}")
show("PPO reference (GB)", ref, "{:,.0f}")
show("PPO reward (GB)", rew, "{:,.0f}")
show("PPO value (GB)", val, "{:,.0f}")
show("PPO total (GB)", pol + ref + rew + val, "{:,.0f}")
show("DPO total (GB)", pol + ref, "{:,.0f}")
show("DPO reduction (%)", 100 * (1 - (pol + ref) / (pol + ref + rew + val)), "{:.1f}")
show("Policy per GPU 1120/32 (GB)", pol / 32, "{:.1f}")
show("Reference per GPU 140/32 (GB)", ref / 32, "{:.1f}")
show("Rollout KV per GPU 85.9/32 (GB)", 85.9 / 32, "{:.1f}")

# --- Wrap-up -----------------------------------------------------------------
show("Fleet MFU 0.50 x 0.83 (%)", 100 * 0.50 * 0.83, "{:.1f}")
show("Double-counted 0.415 x 0.83 (%)", 100 * 0.415 * 0.83, "{:.1f}")
show("7B FP16 gradient buffer (GB)", 7e9 * 2 / 1e9, "{:.0f}")
show("Next: GPU failure interval 50,000/1,024 (h)", 50000 / 1024, "{:.1f}")

if __name__ == "__main__":
    for label, v in ROWS:
        print(f"{v:>12}  {label}")
