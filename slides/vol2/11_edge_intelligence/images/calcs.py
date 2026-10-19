#!/usr/bin/env python3
"""Every derived number in Lecture 11 (Edge Intelligence), computed, not typed.

Inputs carry their source as a comment. Each printed line names the slide
(frame title) and the label the value carries there. Run from this folder:
    python3 calcs.py
"EI" is the rendered chapter, Edge Intelligence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))


def show(slide, label, value):
    print(f"[{slide}] {label}: {value}")


# --- Hook: Gboard (Hard et al. 2018) --------------------------------------
ngram_cache, server_cache, fed_cache = 12.5, 15.0, 15.8   # Table 4, client caches (%)
show("Predict: did phones train a better keyboard?", "federated minus data-center (points)",
     f"{fed_cache - server_cache:.1f}")
show("Predict: did phones train a better keyboard?", "relative gain over data-center model",
     f"{(fed_cache / server_cache - 1) * 100:.0f}%")   # paper states 'a relative 5%'

# --- Training working set (EI figure 3, illustrative) ---------------------
params = 10e6                       # EI figure 3
fp32_bytes = 4                      # FP32
weights_mb = params * fp32_bytes / 1e6
grads_mb = weights_mb               # one gradient per weight
adam_mb = 2 * weights_mb            # two Adam moments
acts_mb = 200                       # EI figure 3, assumed (5x weights)
total_mb = weights_mb + grads_mb + adam_mb + acts_mb
show("Predict: will this model train in 300 MB?", "weights (MB)", f"{weights_mb:.0f}")
show("Predict: will this model train in 300 MB?", "weights + gradients + Adam (MB)",
     f"{weights_mb + grads_mb + adam_mb:.0f}")
show("Predict: will this model train in 300 MB?", "total stack (MB)", f"{total_mb:.0f}")
show("Predict: will this model train in 300 MB?", "amplification over weights",
     f"{total_mb / weights_mb:.0f}x")
kb_update, kb_allow = 75, 300       # EI Sec. 11.3.1 keyboard profile, selected point
show("Predict: will this model train in 300 MB? (note)", "keyboard update share of allowance",
     f"{kb_update / kb_allow * 100:.0f}%")

# --- Device range (EI Sec. 11.3.2, 11.3.4) --------------------------------
phone_b, mcu_b = 16 * 2**30, 256 * 2**10   # 16 GB, 256 KB (binary, as the chapter's 65,536x)
show("One fleet spans phones, edge boards, and microcontrollers", "phone / microcontroller memory",
     f"{phone_b / mcu_b:,.0f}x")
img_kb = 224 * 224 * 3 / 1000       # one 224x224x3 8-bit image
show("One fleet spans ... (note)", "224x224x3 image (KB)", f"{img_kb:.0f}")
show("One fleet spans ... (note)", "image share of 256 KB", f"{img_kb / 256 * 100:.0f}%")

# --- NPU and bytes (EI Sec. 11.3.4) ---------------------------------------
step_mb = 800                       # EI: one update step moves about 800 MB
a17_gbs = 51                        # A17 Pro peak DRAM bandwidth (EI)
show("Peak NPU ratings grew, yet an update step is bound by bytes", "traffic time per step (ms)",
     f"{step_mb / 1000 / a17_gbs * 1000:.1f}")
show("Peak NPU ratings grew ...", "update bytes / weight bytes", f"{step_mb / weights_mb:.0f}x")
hbm, lo, hi = 3350, 51, 77          # GB/s, EI
show("Peak NPU ratings grew ...", "HBM3 / mobile memory bandwidth",
     f"{hbm / hi:.1f}x to {hbm / lo:.1f}x")
weight_reads_mb = 3 * weights_mb    # EI: three weight reads per step
show("Peak NPU ratings grew ... (note)", "weight reads share of step traffic",
     f"{weight_reads_mb / step_mb * 100:.0f}%")
from mlsysim import Literature  # noqa: E402
ane = {r["chip"]: r["tops"] for r in Literature.Series.AppleNeuralEngine.entries}
show("Peak NPU ratings grew ... (map)", "A17 Pro / A11 stated rating",
     f"{ane['A17 Pro'] / ane['A11 Bionic']:.0f}x")

# --- Budget terms (EI Sec. 11.4.3) -----------------------------------------
d, r = 768, 4
full = d * d
lora = r * (d + d)
show("Each adaptation method shrinks a different budget term", "full layer parameters", f"{full:,}")
show("Each adaptation method shrinks a different budget term", "LoRA rank-4 parameters", f"{lora:,}")
show("Each adaptation method shrinks a different budget term", "reduction",
     f"{(1 - lora / full) * 100:.0f}%")
d2, r2 = 768, 16
adapter = 2 * d2 * r2 + r2 + d2
show("Ten contexts cost one backbone ... (note)", "adapter parameters (d=768, r=16)", f"{adapter:,}")
show("Ten contexts cost one backbone ... (note)", "adapter FP16 (KB)", f"{adapter * 2 / 1000:.1f}")

# --- Personalization (EI Napkin Math 11.2) ---------------------------------
ctx, model_mb, adapter_mb = 10, 40, 0.2
copies = ctx * model_mb
shared = model_mb + ctx * adapter_mb
show("Ten contexts cost one backbone plus ten small adapters", "ten copies (MB)", f"{copies:.0f}")
show("Ten contexts cost one backbone plus ten small adapters", "backbone + adapters (MB)", f"{shared:.0f}")
show("Ten contexts cost one backbone plus ten small adapters", "total saving", f"{copies / shared:.1f}x")
show("Ten contexts cost one backbone plus ten small adapters", "per-context saving",
     f"{model_mb / adapter_mb:.0f}x")

# --- Battery (EI Napkin Math 11.1, Lighthouse 11.1) -------------------------
p_w, hours, batt_wh = 4.5, 0.5, 15
e_wh = p_w * hours
show("Estimate: what share of battery is one update?", "energy (Wh)", f"{e_wh:.2f}")
show("Estimate: what share of battery is one update?", "share", f"{e_wh / batt_wh * 100:.0f}%")
show("Archetype C turns the fleet into millions of phones (note)", "15 Wh at 10 W (h)", f"{batt_wh / 10:.1f}")

# --- Energy per inference against a training second (EI fig. 6, MLSysIM) ---
kws_lo, kws_hi = 31.5e-6, 156.5e-6       # J, Syntiant Core 2 KWS, STM32N6 KWS
train_lo, train_hi = 2.0, 3.0            # J per second at 2 to 3 W
show("A training second costs thousands of wake-word checks", "training second / KWS check",
     f"{train_lo / kws_hi:,.0f} to {train_hi / kws_lo:,.0f}")
checks_per_s = 50                        # illustrative detector rate (assumed)
detector_w = checks_per_s * kws_lo
show("A training second ... (note)", "detector power at 50 checks/s (mW)", f"{detector_w * 1000:.2f}")
show("A training second ... (note)", "minutes to spend one training second",
     f"{train_lo / detector_w / 60:.0f} to {train_hi / detector_w / 60:.0f}")

# --- Uplink (EI Napkin Math 11.3) ------------------------------------------
raw_mb, upd_mb = 200, 2.5
show("Estimate: is a 2.5 MB update a saving?", "per-round ratio", f"{raw_mb / upd_mb:.0f}x")
show("Estimate: is a 2.5 MB update a saving?", "break-even rounds", f"{raw_mb / upd_mb:.0f}")
show("Estimate: is a 2.5 MB update a saving?", "daily: 7 rounds (MB)", f"{7 * upd_mb:.1f}")
show("Estimate: is a 2.5 MB update a saving?", "hourly: 168 rounds (MB)", f"{24 * 7 * upd_mb:.0f}")
adapter_payload_mb, link_mbps = 50, 1     # EI reference adapter; 1 Mb/s uplink
show("Estimate: is a 2.5 MB update a saving? (note)", "50 MB at 1 Mb/s (s, decimal)",
     f"{adapter_payload_mb * 8 / link_mbps:.0f}")

# --- FedAvg (McMahan et al. 2017, Table 2) ---------------------------------
show("Predict: does more local work cut the rounds?", "IID 626 / 18", f"{626 / 18:.1f}x")
show("Predict: does more local work cut the rounds?", "non-IID 483 / 173", f"{483 / 173:.1f}x")
show("Predict: does more local work cut the rounds?", "non-IID E=5 1,000 vs 483", f"{483 / 1000:.1f}x")

# --- Secure aggregation cells (make_figs.py values, illustrative) ----------
da = [0.2, -0.1, 0.4, 0.3]
db = [0.1, 0.3, -0.2, 0.1]
s = [7.3, -4.6, 2.9, 9.1]
show("Masks hide each update and cancel in the sum", "A + s", [round(a + m, 1) for a, m in zip(da, s)])
show("Masks hide each update and cancel in the sum", "sum", [round(a + b, 1) for a, b in zip(da, db)])

# --- Sessions (Bonawitz et al. 2019, Table 1) ------------------------------
ok, late, intr = 1_116_401, 327_478, 29_771
tot = ok + late + intr
show("About a quarter of training sessions never count", "not counted share",
     f"{(late + intr) / tot * 100:.0f}%")

# --- AI Benchmark (MLSysIM series) -----------------------------------------
rows = [x for x in Literature.Series.AIBenchmarkSoCs.entries if x["ai_score"]]
win = [x for x in rows if 2020 <= x["year"] <= 2026]
gen = [x for x in rows if 2024 <= x["year"] <= 2025]
show("Phones in one fleet differ about 400 times in AI score", "2020-2026 spread",
     f"{max(x['ai_score'] for x in win) / min(x['ai_score'] for x in win):.0f}x")
show("Phones in one fleet differ ... (note)", "2024-2025 spread",
     f"{max(x['ai_score'] for x in gen) / min(x['ai_score'] for x in gen):.0f}x")

# --- Coverage (illustrative counts per 100 deployed, make_figs.py) ---------
deployed = {"flagship": 20, "mid-tier": 60, "budget": 20}
completed = {"flagship": 8, "mid-tier": 11, "budget": 1}
n_done = sum(completed.values())
for k in deployed:
    show("Diagnose: whose typing does the model learn?", f"coverage ratio, {k}",
         f"{(completed[k] / n_done) / (deployed[k] / 100):.2f}")
show("Diagnose: whose typing does the model learn?", "budget share of completed",
     f"{completed['budget'] / n_done * 100:.0f}%")
show("Diagnose: whose typing does the model learn?", "selected = 1.3 x completed", f"{1.3 * n_done:.0f}")

# --- Shadow gate (illustrative letters, make_figs.py) ----------------------
base, cand = "AABCABCA", "AABBABAA"
dis = sum(a != b for a, b in zip(base, cand))
show("Once models learn after release, the release is a policy", "disagreement",
     f"{dis} of {len(base)} = {dis / len(base) * 100:.0f}%")
