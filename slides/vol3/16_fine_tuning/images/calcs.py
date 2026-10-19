#!/usr/bin/env python3
"""Every derived number in Lecture 16 (Trajectory Fine-Tuning), one block per slide.

Inputs are typed with their source. Sources:
  TF   = the rendered chapter, Trajectory Fine-Tuning (Volume III), dev text
         2026-10-18. NM = its Napkin Math notebooks; Ex = its exercises (the
         chapter labels all of their inputs hypothetical: slides say
         "illustrative").
  SWE  = MLSysIM Literature.SweBenchTrajectories: 2,100 public SWE-bench
         Verified trajectories (github.com/SWE-bench/experiments), the data
         behind the chapter's figure 2.
  Ross = Ross, Gordon, and Bagnell, AISTATS 2011, Figure 2 (Super Tux Kart,
         average falls per lap; values read from the figure) and Sec. 5.1 text.
  LoRA = Hu et al., arXiv 2106.09685v2, Sec. 4.2 ("Practical benefits").
  runs = data/*.txt, written by tool_runs.py (real library runs).

Run from this folder: python3 calcs.py   (prints "label: value")
make_figs.py imports these values so figures and slides agree.
"""

import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))


def show(label, value):
    print(f"  {label}: {value}")


# ------------------------------------------------ hook: recorded role shares
def role_shares():
    """SWE: share of transcript characters by role, per configuration."""
    from mlsysim import Literature
    tot = {}
    for r in Literature.SweBenchTrajectories.entries:
        t = tot.setdefault((r["scaffold"], r["model"]), [0.0, 0.0, 0.0, 0.0, 0])
        t[0] += r["prompt_chars"]
        t[1] += r["obs_chars"]
        t[2] += r["model_chars"]
        t[3] += r["total_chars"]
        t[4] += 1
    rows = []
    for (scaffold, model), v in tot.items():
        rows.append(dict(scaffold=scaffold, model=model, runs=v[4],
                         prompt=100 * v[0] / v[3], obs=100 * v[1] / v[3],
                         model_out=100 * v[2] / v[3]))
    return rows


def swe_tokens():
    """SWE, SWE-agent 0.x with Claude 3.5 Sonnet (500 runs): the per-call input
    tokens the agent recorded, summed over a run, against its output tokens."""
    from mlsysim import Literature
    rs = [r for r in Literature.SweBenchTrajectories.entries
          if r["scaffold"].startswith("SWE-agent 0.x")]
    ins = [r["rep_input_tokens_sum"] for r in rs]
    outs = [r["rep_output_tokens_sum"] for r in rs]
    calls = [r["n_model_calls"] for r in rs]
    return dict(runs=len(rs), calls=st.median(calls), input_med=st.median(ins),
                output_med=st.median(outs),
                ratio_med=st.median([i / o for i, o in zip(ins, outs) if o]),
                out_share_pct=100 * sum(outs) / (sum(ins) + sum(outs)))


# --------------------------------------------- serialization (TF Sec. 16.3.3)
H_SER = 30                      # TF: stipulated 30-turn horizon


def serialization(h=H_SER, a=1):
    per_inv = a * h * (h + 1) / 2
    single = a * h
    return dict(expansion=per_inv / single)


# ------------------------------------------------------ NM 16.1 inventory
INV = dict(prompt=1200, task=184, rationale=1400, responses=960, obs=12640)
REFUSED = 160                    # NM 16.1: refused-proposal positions


def nm161(obs=None):
    inv = dict(INV)
    if obs is not None:
        inv["obs"] = obs
    total = sum(inv.values())
    pred = total - 1                         # first boundary has no prediction
    tgt_resp = inv["responses"] - REFUSED    # rationale kept as context
    tgt_rat = tgt_resp + inv["rationale"]    # rationale selected
    return dict(total=total, pred=pred, obs_pct=100 * inv["obs"] / total,
                tgt=tgt_resp, tgt_rat=tgt_rat,
                tgt_pct=100 * tgt_resp / total, tgt_rat_pct=100 * tgt_rat / total,
                dil=pred / tgt_resp, dil_rat=pred / tgt_rat,
                ctx_pct=100 * (total - inv["obs"] - tgt_resp) / total,
                refused_share_resp=100 * REFUSED / inv["responses"],
                refused_vs_tgt=100 * REFUSED / tgt_resp)


# ------------------------------------------- normalization (NM 16.2, Ex 3)
def nm162_weights(a=256, b=8192):
    return dict(w_short=100 * a / (a + b), w_long=100 * b / (a + b), ratio=b / a)


def ex3(long_t=6000, n_short=15, short_t=200, split=20):
    tot = long_t + n_short * short_t
    tok = 100 * long_t / tot
    ex = 100 / (1 + n_short)
    ex_split = 100 * split / (split + n_short)
    return dict(tok=tok, ex=ex, tok_split=tok, ex_split=ex_split)


# ------------------------------------------------ packing (NM 16.2)
LENGTHS = [32768, 8192, 4096, 2048, 1024, 1024, 512, 512]
BUF = 32768


def nm162_pack():
    stored = sum(LENGTHS)
    padded = len(LENGTHS) * max(LENGTHS)
    buffers = 2                              # longest alone; the rest share one
    packed = buffers * BUF
    return dict(stored=stored, rest=stored - max(LENGTHS),
                pad_occ=100 * stored / padded, pack_occ=100 * stored / packed,
                fewer=padded / packed, mean=stored / len(LENGTHS))


def run_value(fname, pattern):
    txt = (HERE / "data" / fname).read_text()
    return re.search(pattern, txt).group(1)


def runs():
    return dict(
        causal=float(run_value("packing_test.txt", r"ordinary causal mask:\s+(\S+)")),
        block=float(run_value("packing_test.txt", r"pos reset:\s+(\S+)")),
        block_off=float(run_value("packing_test.txt", r"pos offset:\s+(\S+)")),
        mom=float(run_value("accum_test.txt", r"mean of micro-batch means:\s+(\S+)")),
        glob=float(run_value("accum_test.txt", r"num_items_in_batch=\d+:\s+(\S+)")),
        lora_tr=run_value("lora_count.txt", r"trainable params: ([\d,]+)"),
        lora_all=run_value("lora_count.txt", r"all params: ([\d,]+)"),
        lora_pct=run_value("lora_count.txt", r"trainable%: (\S+)"),
        t_ctx=int(run_value("template_render.txt", r"system\+tools\+user (\d+)")),
        t_asst=int(run_value("template_render.txt", r"assistant turn (\d+)")),
        t_tool=int(run_value("template_render.txt", r"two tool replies (\d+)")),
        t_all=int(run_value("template_render.txt", r"total (\d+)")))


# --------------------------------------------- exposure bias (NM 16.3)
H_EB, EPS, U = 30, 0.02, 1


def nm163(h=H_EB, eps=EPS, u=U):
    ref = min(h, h * h * eps)
    lrn = min(h, u * h * eps)
    return dict(ref=ref, lrn=lrn, gap=ref / lrn, sat=1 / eps,
                need_pct=100 * lrn / (h * h))


# Ross Figure 2, average falls per lap after 1, 5, 10, 15, 20 iterations
# (about 1,000 data points per iteration); read from the figure.
STK_X = [1, 5, 10, 15, 20]
STK = {"Supervised": [3.0, 2.9, 3.2, 2.9, 4.0],
       "SMILe": [3.0, 2.35, 2.1, 1.85, 1.95],
       "DAgger": [3.0, 0.3, 0.35, 0.0, 0.0]}


# --------------------------------------------------------- DPO (NM 16.4)
def sig(x):
    return 1 / (1 + math.exp(-x))


def nm164(beta=0.2, pol_w=-8, ref_w=-10, pol_l=-11, ref_l=-10):
    zw, zl = pol_w - ref_w, pol_l - ref_l
    m = beta * (zw - zl)
    return dict(zw=zw, zl=zl, margin=m, score=sig(m), loss=-math.log(sig(m)),
                loss0=math.log(2), sat=sig(-m), coef=beta * sig(-m), coef0=beta / 2)


# ---------------------------------------- context distillation (Ex 8)
EX8 = dict(checklist=3000, turns=4, sessions=2e6, price=2.0, cached_frac=0.1,
           redistill=15000, revisions=1)


def ex8():
    e = EX8
    toks = e["checklist"] * e["turns"] * e["sessions"]
    unc = toks / 1e6 * e["price"]
    cac = unc * e["cached_frac"]
    return dict(tokens_b=toks / 1e9, uncached=unc, cached=cac,
                net_unc=unc - e["redistill"] * e["revisions"],
                net_cac=cac - e["redistill"] * e["revisions"],
                breakeven_revs=unc / e["redistill"])


# ------------------------------------------------------ LoRA (NM 16.5)
P_BASE, LAYERS, MATS, RANK, D_IN, D_OUT = 8e9, 32, 2, 16, 4096, 4096
B_TRAIN, B_FROZEN = 16, 2       # bytes: 2 w + 2 g + 4 master + 4 + 4 moments


def nm165(rank=RANK):
    p_ad = LAYERS * MATS * rank * (D_IN + D_OUT)
    ad_state = p_ad * B_TRAIN
    frozen = P_BASE * B_FROZEN
    lora_total = frozen + ad_state
    full = P_BASE * B_TRAIN
    return dict(p_ad=p_ad, pct=100 * p_ad / P_BASE, ad_mb=ad_state / 1e6,
                frozen_gb=frozen / 1e9, lora_gb=lora_total / 1e9, full_gb=full / 1e9,
                saved_gb=(full - lora_total) / 1e9, red=full / lora_total,
                keep_pct=100 * lora_total / full, frozen_share=100 * frozen / lora_total,
                bound=B_TRAIN / B_FROZEN)


LORA_PAPER = dict(vram_full_tb=1.2, vram_lora_gb=350, ckpt_gb=350, ckpt_mb=35, n=100)


def lora_paper():
    p = LORA_PAPER
    return dict(vram_red=p["vram_full_tb"] * 1000 / p["vram_lora_gb"],
                hundred_lora_gb=p["ckpt_gb"] + p["ckpt_mb"] * p["n"] / 1000,
                hundred_full_tb=p["ckpt_gb"] * p["n"] / 1000)


# ------------------------------------------------ release (Ex 9)
EX9 = dict(n=500, p_b=0.60, c_b=1.20, p_c=0.66, c_c=1.50)


def ex9():
    e = EX9
    ce_b, ce_c = e["c_b"] / e["p_b"], e["c_c"] / e["p_c"]
    se = math.sqrt(e["p_b"] * (1 - e["p_b"]) / e["n"] + e["p_c"] * (1 - e["p_c"]) / e["n"])
    return dict(ce_b=ce_b, ce_c=ce_c, max_cost=ce_b * e["p_c"], se=se,
                z=(e["p_c"] - e["p_b"]) / se, ce_rise=100 * (ce_c / ce_b - 1),
                pre_ratio=EX9_PRE[1] / EX9_PRE[0])


EX9_PRE = (0.02, 0.12)          # Ex 9: PRE on renamed schemas, baseline and candidate


if __name__ == "__main__":
    print("Hook: role shares (SWE)")
    for r in role_shares():
        show(f"{r['scaffold']} / {r['model']} ({r['runs']} runs)",
             f"prompt {r['prompt']:.0f}%, observations {r['obs']:.0f}%, model output {r['model_out']:.0f}%")
    print("Serialization")
    show("expansion at H=30", f"{serialization()['expansion']:.1f}x")
    t = swe_tokens()
    show("SWE-agent 0.x runs", t["runs"])
    show("median model calls", t["calls"])
    show("median summed input tokens", f"{t['input_med']:,.0f}")
    show("median output tokens", f"{t['output_med']:,.0f}")
    show("median input/output ratio", f"{t['ratio_med']:.0f}")
    show("output share of all tokens (pooled)", f"{t['out_share_pct']:.1f}%")
    print("NM 16.1")
    n = nm161()
    for k in ("total", "pred", "obs_pct", "tgt", "tgt_rat", "tgt_pct", "tgt_rat_pct",
              "dil", "dil_rat", "ctx_pct", "refused_share_resp", "refused_vs_tgt"):
        show(k, f"{n[k]:,.1f}" if isinstance(n[k], float) else f"{n[k]:,}")
    show("dilution, observations doubled", f"{nm161(obs=2 * INV['obs'])['dil']:.1f}x")
    show("dilution, observations 30,000 (Checkpoint 16.1)", f"{nm161(obs=30000)['dil']:.1f}x")
    print("Normalization")
    w = nm162_weights()
    show("token-average weights", f"{w['w_short']:.2f}% and {w['w_long']:.2f}%; ratio {w['ratio']:.0f}x")
    e3 = ex3()
    show("Ex 3 long share: token, example", f"{e3['tok']:.1f}%, {e3['ex']:.2f}%")
    show("Ex 3 after split into 20: token, example", f"{e3['tok_split']:.1f}%, {e3['ex_split']:.1f}%")
    show("Ex 3 example-average rise", f"{e3['ex_split'] / e3['ex']:.1f}x")
    print("Packing")
    p = nm162_pack()
    show("stored", f"{p['stored']:,}; rest {p['rest']:,}; mean {p['mean']:,.0f}")
    show("occupancy padded / packed", f"{p['pad_occ']:.1f}% / {p['pack_occ']:.1f}%; {p['fewer']:.0f}x fewer positions")
    rr = runs()
    show("packing run: causal / block / block no reset",
         f"{rr['causal']:.3g} / {rr['block']:.3g} / {rr['block_off']:.3g}")
    show("accumulation run: mean of means / global", f"{rr['mom']:.3f} / {rr['glob']:.3f}")
    show("accumulation ratio", f"{rr['mom'] / rr['glob']:.1f}x")
    show("template: assistant share", f"{100 * rr['t_asst'] / rr['t_all']:.1f}% of {rr['t_all']} tokens")
    show("PEFT count", f"{rr['lora_tr']} of {rr['lora_all']} ({rr['lora_pct']}%)")
    print("Exposure bias (NM 16.3)")
    x = nm163()
    show("bounds", f"ref {x['ref']:.1f}, learner {x['lrn']:.1f}, gap {x['gap']:.0f}x")
    show("saturation horizon", f"{x['sat']:.0f} steps")
    show("reference error needed", f"{x['need_pct']:.3f}%")
    print("DPO (NM 16.4)")
    for b in (0.2, 0.4):
        d = nm164(beta=b)
        show(f"beta {b}", f"scores {d['zw']}, {d['zl']}; margin {d['margin']:.1f}; score {d['score']:.3f}; "
             f"loss {d['loss']:.3f} (from {d['loss0']:.3f}); saturation {d['sat']:.3f}; "
             f"coef {d['coef']:.4f} (from {d['coef0']:.2f})")
    print("Distillation (Ex 8)")
    d8 = ex8()
    show("checklist tokens per month", f"{d8['tokens_b']:.0f} billion")
    show("spend removed: uncached / cached", f"${d8['uncached']:,.0f} / ${d8['cached']:,.0f}")
    show("net after redistillation: uncached / cached", f"${d8['net_unc']:,.0f} / ${d8['net_cac']:,.0f}")
    show("revisions per month to zero (uncached)", f"{d8['breakeven_revs']:.1f}")
    print("LoRA (NM 16.5)")
    l = nm165()
    show("adapter params", f"{l['p_ad']:,} ({l['pct']:.2f}%)")
    show("adapter state", f"{l['ad_mb']:.2f} MB")
    show("LoRA run / full", f"{l['lora_gb']:.2f} GB / {l['full_gb']:.0f} GB; {l['red']:.1f}x; saved {l['saved_gb']:.2f} GB")
    show("kept share, frozen share", f"{l['keep_pct']:.1f}%, {l['frozen_share']:.1f}%")
    show("bound", f"{l['bound']:.0f}x")
    show("doubling rank adds", f"{nm165(32)['ad_mb'] - l['ad_mb']:.2f} MB")
    lp = lora_paper()
    show("LoRA paper VRAM reduction", f"{lp['vram_red']:.1f}x")
    show("100 adapted models", f"{lp['hundred_lora_gb']:.1f} GB vs {lp['hundred_full_tb']:.0f} TB")
    print("Release (Ex 9)")
    r9 = ex9()
    show("C_eff baseline / candidate", f"${r9['ce_b']:.2f} / ${r9['ce_c']:.2f}")
    show("max candidate attempt cost", f"${r9['max_cost']:.2f}")
    show("SE of difference, z", f"{r9['se']:.4f}, {r9['z']:.2f}")
    show("C_eff rise", f"{r9['ce_rise']:.0f}%")
    show("PRE ratio", f"{r9['pre_ratio']:.0f}x")
