#!/usr/bin/env python3
"""Every derived number in Lecture 10 (Serving at Scale), computed from sourced inputs.

Run from this folder: python3 calcs.py
Each line prints a slide label and the value the deck shows. Inputs carry their source.
"Ch." is the rendered chapter, Serving at Scale (Volume II, Chapter 10).
"""

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
IX = (Path(__file__).resolve().parents[4] / "mlsysim" / "mlsysim" / "literature" / "data"
      / "series" / "inferencex_dsr1_fp8_1k1k.csv")


def show(label, value):
    print(f"{label:<62} {value}")


# ------------------------------------------------------------- hook (Napkin Math 10.1)
train = 2_000_000                 # Ch. Napkin Math 10.1
users, per_day, days = 1_000_000, 50, 365   # Ch. Napkin Math 10.1
cost_req = 0.001                  # Ch. Napkin Math 10.1 (assumed)
reqs = users * per_day * days
serve = reqs * cost_req
show("hook: requests per year (billion)", f"{reqs / 1e9:.2f}")
show("hook: first-year serving ($M)", f"{serve / 1e6:.2f}")
show("hook: serving / training", f"{serve / train:.1f}x")
show("hook: 10% cut saves ($M)", f"{0.1 * serve / 1e6:.1f}")

# ------------------------------------------------------------- replica (Ch. Sec. 10.7)
w_fp16 = 70e9 * 2                 # 70B parameters x 2 bytes (Ch. eq. 11)
show("replica: FP16 weights (GB)", f"{w_fp16 / 1e9:.0f}")
show("replica: weights per GPU, 8 GPUs (GB)", f"{w_fp16 / 8 / 1e9:.1f}")

# ------------------------------------------------------------- Azure 2023 trace (Patel et al.)
def trace(name):
    r = list(csv.DictReader(open(DATA / name)))
    t = [datetime.strptime(x["TIMESTAMP"][:26], "%Y-%m-%d %H:%M:%S.%f") for x in r]
    c = np.array([int(x["ContextTokens"]) for x in r])
    g = np.array([int(x["GeneratedTokens"]) for x in r])
    return t, c, g


tv, cv, gv = trace("azure_llm_conv_2023.csv")
tc, cc, gc = trace("azure_llm_code_2023.csv")
dur = (tv[-1] - tv[0]).total_seconds()
show("trace: chat requests", f"{len(gv):,}")
show("trace: chat rate (req/s)", f"{len(gv) / dur:.1f}")
show("trace: chat median prompt / answer", f"{np.median(cv):.0f} / {np.median(gv):.0f}")
show("trace: chat mean answer / max answer", f"{gv.mean():.0f} / {gv.max()}")
show("trace: code median prompt / answer", f"{np.median(cc):.0f} / {np.median(gc):.0f}")


def idle(g, B=8):
    n = len(g) // B
    gb = g[: n * B].reshape(n, B)
    return float((1 - gb.mean(1) / gb.max(1)).mean())


show("lengths: chat idle share, batch of 8", f"{100 * idle(gv):.0f}%")
show("lengths: code idle share, batch of 8", f"{100 * idle(gc):.0f}%")
secs = np.array([(x - tc[0]).total_seconds() for x in tc]).astype(int)
per = np.bincount(secs)
show("traffic: code trace mean per second / busiest second", f"{per.mean():.1f} / {per.max()}")

# ------------------------------------------------------------- vote (Ch. Sec. 10.3 self-check)
show("vote: X SLO goodput", f"{1200 * 0.70:.0f}")
show("vote: Y SLO goodput", f"{1000 * 0.95:.0f}")

# ------------------------------------------------------------- tax (Ch. Sec. 10.3.2)
show("tax: budget outside the model (ms)", f"{100 - 70}")
show("tax: 4 exchanges at 50-500 us (ms)", f"{4 * 0.05:.1f} to {4 * 0.5:.0f}")

# ------------------------------------------------------------- InferenceX
ix = list(csv.DictReader(open(IX)))
for x in ix:
    for f in ("conc", "disagg", "interactivity_tok_s_user", "output_tok_s_gpu"):
        x[f] = float(x[f])


def best_at(rows, t):
    return max(r["output_tok_s_gpu"] for r in rows if r["interactivity_tok_s_user"] >= t)


def hw(h):
    return [r for r in ix if r["hardware"] == h]


b200 = hw("b200")
peak = max(b200, key=lambda r: r["output_tok_s_gpu"])
show("frontier: configurations", f"{len(ix)}")
show("frontier: B200 peak (tok/s/GPU at tok/s/user)",
     f"{peak['output_tok_s_gpu']:,.0f} at {peak['interactivity_tok_s_user']:.0f}")
show("frontier: B200 best at >=100 tok/s/user", f"{best_at(b200, 100):,.0f}")
show("frontier: B200 span", f"{peak['output_tok_s_gpu'] / best_at(b200, 100):.0f}x")
h100_50, gb200_50 = best_at(hw("h100"), 50), best_at(hw("gb200"), 50)
show("frontier: H100 / GB200 at 50 tok/s/user", f"{h100_50:.0f} / {gb200_50:,.0f}")
show("frontier: GB200 / H100 at 50", f"{gb200_50 / h100_50:.0f}x")

sg = sorted([r for r in b200 if r["framework"] == "sglang"], key=lambda r: r["conc"])
lo, hi = sg[0], sg[-1]
show("conc: per-GPU tok/s, conc 1 -> 512",
     f"{lo['output_tok_s_gpu']:.0f} -> {hi['output_tok_s_gpu']:,.0f} "
     f"({hi['output_tok_s_gpu'] / lo['output_tok_s_gpu']:.0f}x)")
show("conc: per-user tok/s, conc 1 -> 512",
     f"{lo['interactivity_tok_s_user']:.0f} -> {hi['interactivity_tok_s_user']:.0f} "
     f"({lo['interactivity_tok_s_user'] / hi['interactivity_tok_s_user']:.1f}x)")
ok50 = [r for r in sg if r["interactivity_tok_s_user"] >= 50]
show("conc: highest concurrency at >=50 tok/s/user", f"{max(r['conc'] for r in ok50):.0f}")

for h in ("b200", "h200"):
    rows = hw(h)
    for t in (25, 50, 100):
        s = best_at([r for r in rows if r["disagg"] == 1], t)
        c = best_at([r for r in rows if r["disagg"] == 0], t)
        show(f"disagg: {h} at {t} split / colocated", f"{s:,.0f} / {c:,.0f} ({s / c:.2f}x)")

# ------------------------------------------------------------- knee (Napkin Math 10.2)
show("knee: vision t0/treq", f"{2 / 1:.0f}")
show("knee: LLM decode t0/treq", f"{40 / 0.5:.0f}")
show("knee: 140 GB / 3.35 TB/s (ms)", f"{140e9 / 3.35e12 * 1e3:.1f}")

# ------------------------------------------------------------- Little's law (Theorem 10.1)
lam = 1000
show("little: in flight, 100 ms", f"{lam * 0.1:.0f}")
show("little: replica rate (req/s)", f"{8 / 0.08:.0f}")
show("little: replicas at 80%", f"{math.ceil(lam / (0.8 * 100))}")
tpot = 0.050                      # Ch. Sec. 10.15.1 illustrative TPOT target
t_chat = round(gv.mean()) * tpot  # mean answer 211 tokens
show("little: chat time in system (s)", f"{t_chat:.2f}")
show("little: chat in flight", f"{lam * t_chat:,.0f}")

# ------------------------------------------------------------- KV (eq. 5, Napkin Math 10.5)
layers, kvh, hidden, heads, b = 80, 8, 8192, 64, 2   # Llama 3.1 70B config.json; FP16
head_dim = hidden // heads
m_tok = 2 * layers * kvh * head_dim * b
show("kv: head width from config (8192 / 64)", f"{head_dim}")
show("kv: bytes per token", f"{m_tok:,}")
free = 687.2 - 141.2 - 20         # Ch. Napkin Math 10.5 (registered byte capacity, GB)
long_req = 131072 * m_tok / 1e9
show("kv: free for KV (GB)", f"{free:.1f}")
show("kv: 128K request (GB)", f"{long_req:.1f}")
show("kv: 128K requests that fit", f"{math.floor(free / long_req)}")
chat_tok = int(np.median(cv) + np.median(gv))
chat_mb = chat_tok * m_tok / 1e6
show("kv: median chat tokens / MB", f"{chat_tok:,} / {chat_mb:.0f}")
n_chat = math.floor(free * 1e3 / chat_mb)
show("kv: median chats that fit", f"{n_chat:,}")
show("kv: chats / long requests", f"{n_chat / math.floor(free / long_req):.0f}x")
avg_resident = (cv.mean() + gv.mean() / 2) * m_tok   # prompt plus half the answer, mid-generation
fleet_kv = lam * t_chat * avg_resident
show("kv: chat KV in flight (TB)", f"{fleet_kv / 1e12:.1f}")
show("kv: replicas by KV memory alone", f"{fleet_kv / (free * 1e9):.1f}")

# ------------------------------------------------------------- paging (Napkin Math 10.6)
mha_tok = 2 * 80 * 8192 * 2       # 64 heads x 128 = 8192 width, FP16
before = 2500 * 1000 * mha_tok
after = (2000 + 500 * 1000) * mha_tok
show("paging: without / with prefix sharing (TB)", f"{before / 1e12:.1f} / {after / 1e12:.1f}")
show("paging: reduction", f"{100 * (1 - after / before):.1f}%")
show("paging diagram: slots holding tokens", f"{3 + 5} of 16")

# ------------------------------------------------------------- precision (Sec. 10.9, Napkin Math 10.7)
show("precision: 70B INT4 / 175B INT4 (GB)", f"{70e9 * 0.5 / 1e9:.0f} / {175e9 * 0.5 / 1e9:.1f}")
kv_fp16 = 2 * 80 * 8 * 128 * 4096 * 2 / 1e9 / 8     # GQA, 4,096 tokens, per GPU
b_fp16 = math.floor((80 - 17.5) / kv_fp16)
b_q = math.floor((80 - 17.5 / 4) / (kv_fp16 / 2))
show("precision: max batch FP16 / INT4 W + INT8 KV", f"{b_fp16} / {b_q} ({b_q / b_fp16:.1f}x)")
show("precision: KV budget gain from INT4 weights", f"{(80 - 17.5 / 4) / (80 - 17.5):.2f}x")

# ------------------------------------------------------------- roofline, chunking
show("roofline: H100 FP16 ridge (FLOP/byte)", f"{989e12 / 3.35e12:.0f}")
show("chunk: iterations for 10,000 tokens at 512", f"{math.ceil(10000 / 512)}")

# ------------------------------------------------------------- handoff (eq. 10)
show("handoff: median chat over NVLink 450 GB/s (ms)", f"{chat_mb / 1e3 / 450 * 1e3:.1f}")
show("handoff: median chat over InfiniBand 50 GB/s (ms)", f"{chat_mb / 1e3 / 50 * 1e3:.1f}")
show("handoff: 128K request over NVLink (ms)", f"{long_req / 450 * 1e3:.0f}")

# ------------------------------------------------------------- tensor parallel (Example 10.6)
per_layer = (12 + 18) / 8 + 2 * 0.3
show("tp: compute per layer on 8 (ms)", f"{(12 + 18) / 8:.2f}")
show("tp: layer with collectives (ms)", f"{per_layer:.2f}")
show("tp: 80 layers, 1 GPU -> 8 GPUs (ms)", f"{80 * 30:,} -> {80 * per_layer:.0f}")
show("tp: speedup", f"{30 / per_layer:.1f}x")

# ------------------------------------------------------------- decode budget (table 26)
base = [34, 20.4, 10.2, 10.2, 10.2]
q = [34 * 0.55, 20.4, 10.2, 10.2, 10.2]
f = [34 * 0.55, 20.4 * 0.75, 3.4, 10.2, 2.0]
show("budget: baseline / INT4 / fused (ms)", f"{sum(base):.1f} / {sum(q):.1f} / {sum(f):.1f}")
show("budget: non-GEMM intervals (ms)", f"{sum(base[1:]):.0f}")
show("budget: baseline / 5.2 ms floor", f"{sum(base) / 5.2:.0f}x")
show("budget: per AllReduce, 160 per step (us)", f"{10.2 / 160 * 1e3:.0f}")
a, K = 0.78, 5
tokens = (1 - a ** (K + 1)) / (1 - a)
rnd = 4 + sum(f)
show("budget: tokens per round", f"{tokens:.2f}")
show("budget: round (ms) / per token (ms)", f"{rnd:.1f} / {rnd / tokens:.1f}")
show("budget: TPOT gain", f"{sum(f) / (rnd / tokens):.2f}x")

# ------------------------------------------------------------- routing and tail
show("hetero: utilization at 15,000 QPS", f"{100 * 15000 / (10 * 1000 + 20 * 600):.1f}%")
show("fanout: 100 leaves at p99", f"{100 * (1 - 0.99 ** 100):.0f}%")
show("fanout: 100 leaves at p99.99", f"{100 * (1 - 0.9999 ** 100):.1f}%")
show("fanout: slow events per second at 1e7 QPS, 1e-4", f"{1e7 * 1e-4:,.0f}")
show("cold start: 140 GB at 0.5 GB/s (min)", f"{140 / 0.5 / 60:.1f}")
show("cold start: total (min)", f"{1 + 140 / 0.5 / 60 + 1:.1f}")
r24 = list(csv.DictReader(open(DATA / "azure_llm_conv_2024_10min.csv")))
v = np.array([int(x["requests"]) for x in r24]) / 600
show("traffic: week peak / minimum (req/s)", f"{v.max():.0f} / {v.min():.0f}")

# ------------------------------------------------------------- synthesis
price = 4.00                      # Ch. table 25, hypothetical H100 on-demand $/hour
costs = []
for t in (13, 50, 100):
    tok = best_at(hw("h100"), t)
    c = price / (tok * 3600) * 1e6
    costs.append(c)
    show(f"synth: H100 at {t} tok/s/user: tok/s/GPU, $ per M output", f"{tok:.0f}, ${c:.2f}")
show("synth: spread", f"{costs[2] / costs[0]:.0f}x")
show("synth: 211-token answer at 50 tok/s/user ($)", f"{costs[1] * round(gv.mean()) / 1e6:.4f}")
wk = [x for x in r24 if datetime.strptime(x["bin10min"][:10], "%Y-%m-%d").weekday() < 5]
by_day = {}
for x in wk:
    by_day.setdefault(x["bin10min"][:10], []).append(int(x["requests"]) / 600)
troughs = [min(d) for d in by_day.values()]
peaks = [max(d) for d in by_day.values()]
show("traffic: weekday troughs / peaks (req/s)",
     f"{min(troughs):.0f}-{max(troughs):.0f} / {min(peaks):.0f}-{max(peaks):.0f}")
show("traffic: weekday peak / trough", f"{np.mean([p / t for p, t in zip(peaks, troughs)]):.1f}x")
