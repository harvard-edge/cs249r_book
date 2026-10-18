#!/usr/bin/env python3
"""Every derived number Lecture 0 (Volume III) shows, computed from sourced inputs.

Run from this folder: python3 calcs.py
Each line prints the value with the slide (or note) that shows it. Copy the
printed values onto the slides; never compute a slide number by hand.
make_figs.py imports the functions below so plots and slides agree.
"""

import csv
import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent

# --- Horizon compounding (Introduction, Napkin Math 1.1; equation 1) --------
STEP_OK = 0.99          # per-step success, illustrative (NM 1.1)
H_SHORT, H_LONG = 10, 100
TARGET = 0.90           # success to hold at 100 turns (NM 1.1)
H_FREEZE = 30           # turns (Agent Harness, Sec. 10.10 figure)


def horizon():
    eps_needed = 1 - TARGET ** (1 / H_LONG)
    return dict(
        p10=STEP_OK ** H_SHORT,
        p100=STEP_OK ** H_LONG,
        eps_needed=eps_needed,
        eps_ratio=(1 - STEP_OK) / eps_needed,
        p30=STEP_OK ** H_FREEZE,
    )


# --- The bill of one trajectory (Foundation Model NM 2.2; Agent Economics
#     NM 19.2; scenario values the chapters label hypothetical) -------------
TURNS = 30
FIRST_PROMPT = 8_000     # tokens
APPEND = 1_500           # tokens appended per turn
OUT_PER_CALL = 200       # output tokens per call
WINDOW = 128_000         # repair agent's context window (Lighthouse 1.1)
PRICE_IN = 3.00 / 1e6    # $ per fresh input token (NM 19.2)
PRICE_CACHED = 0.30 / 1e6  # 10 percent of fresh (NM 19.2)
PRICE_OUT = 15.00 / 1e6  # $ per output token (NM 19.2)


def resent_after(turns):
    return turns * FIRST_PROMPT + APPEND * turns * (turns - 1) // 2


def fresh_after(turns):
    return FIRST_PROMPT + (turns - 1) * APPEND


def bill():
    sent, fresh = resent_after(TURNS), fresh_after(TURNS)
    out = TURNS * OUT_PER_CALL
    no_reuse = sent * PRICE_IN + out * PRICE_OUT
    reuse = fresh * PRICE_IN + (sent - fresh) * PRICE_CACHED + out * PRICE_OUT
    last_call = FIRST_PROMPT + (TURNS - 1) * APPEND
    return dict(sent=sent, fresh=fresh, out=out, ratio=sent / fresh,
                no_reuse=no_reuse, reuse=reuse, bill_ratio=no_reuse / reuse,
                in_no_reuse=sent * PRICE_IN, out_cost=out * PRICE_OUT,
                last_call=last_call, window_share=last_call / WINDOW)


# --- Cost per accepted task (Introduction, Napkin Math 1.2) -----------------
TOKENS_USD, OTHER_USD, P_ACC = 1.30, 0.80, 0.40
P_ACC_HI, PRICE_CUT = 0.60, 0.5


def cost_per_accepted():
    attempt = TOKENS_USD + OTHER_USD
    cheap = TOKENS_USD * (1 - PRICE_CUT) + OTHER_USD
    share = TOKENS_USD / attempt
    return dict(attempt=attempt, base=attempt / P_ACC, cheaper=cheap / P_ACC,
                better=attempt / P_ACC_HI, attempts=1 / P_ACC,
                breakeven=P_ACC / (1 - PRICE_CUT * share))


# --- Real runs: 500 SWE-bench Verified repairs (swebench_data.py) ------------
BINS = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 70), (70, 1000)]


def runs():
    rows = list(csv.DictReader(open(HERE / "swebench_runs.csv")))
    n = len(rows)
    cost = [float(r["cost_usd"]) for r in rows]
    ok = [r["resolved"] == "1" for r in rows]
    calls = [int(r["api_calls"]) for r in rows]
    accepted = sum(ok)
    submitted = sum(r["exit_status"] == "Submitted" for r in rows)
    rejected_spend = sum(c for c, a in zip(cost, ok) if not a)
    sent = sum(int(r["chars_sent"]) for r in rows)
    new = sum(int(r["chars_new"]) for r in rows)
    bins = []
    for lo, hi in BINS:
        sel = [a for c, a in zip(calls, ok) if lo <= c < hi]
        bins.append((lo, hi, len(sel), sum(sel) / len(sel)))
    srt = sorted(calls)
    return dict(n=n, accepted=accepted, rejected=n - accepted, submitted=submitted,
                acc_rate=accepted / n, total=sum(cost), per_attempt=sum(cost) / n,
                per_accepted=sum(cost) / accepted, rejected_spend=rejected_spend,
                rejected_share=rejected_spend / sum(cost),
                mean_calls=sum(calls) / n, median_calls=(srt[n // 2 - 1] + srt[n // 2]) / 2,
                min_calls=min(calls), max_calls=max(calls),
                resend_ratio=sent / new, bins=bins,
                at_80=(sum(cost) / n) / 0.80,   # same attempt cost, 80% accepted (note)
                mean_cost_ok=sum(c for c, a in zip(cost, ok) if a) / accepted,
                mean_cost_rejected=rejected_spend / (n - accepted))


def one_run():
    rows = list(csv.DictReader(open(HERE / "swebench_run_django12050.csv")))
    last = rows[-1]
    det = json.loads((HERE / "swebench_per_instance.json").read_text())["django__django-12050"]
    return dict(calls=len(rows), sent=int(last["cum_sent_chars"]),
                new=int(last["cum_new_chars"]),
                ratio=int(last["cum_sent_chars"]) / int(last["cum_new_chars"]),
                cost=det["cost"], rows=rows)


def sealed():
    rep = json.loads((HERE / "swebench_report_astropy13398.json").read_text())
    ts = rep["astropy__astropy-13398"]["tests_status"]
    det = json.loads((HERE / "swebench_per_instance.json").read_text())["astropy__astropy-13398"]
    f2p, p2p = ts["FAIL_TO_PASS"], ts["PASS_TO_PASS"]
    return dict(f2p_pass=len(f2p["success"]), f2p_total=len(f2p["success"]) + len(f2p["failure"]),
                p2p_broke=len(p2p["failure"]), p2p_total=len(p2p["success"]) + len(p2p["failure"]),
                calls=det["api_calls"], cost=det["cost"])


# --- Task length (METR Time Horizon 1.1; Kwa et al. 2025) -------------------
DAYS_PER_MONTH = 365.25 / 12


def metr():
    d = yaml.safe_load(open(HERE / "metr_benchmark_results_1_1.yaml"))
    pts = [(str(v["release_date"]), v["metrics"]["p50_horizon_length"]["estimate"],
            bool(v["metrics"].get("is_sota"))) for v in d["results"].values()]
    pts.sort()
    lo, hi = min(p[1] for p in pts), max(p[1] for p in pts)
    dt = d["doubling_time_in_days"]
    return dict(points=pts, lo_s=lo * 60, hi_h=hi / 60, ratio=hi / lo,
                dbl_all=dt["all_time_stitched"]["point_estimate"] / DAYS_PER_MONTH,
                dbl_2023=dt["from_2023_on"]["point_estimate"] / DAYS_PER_MONTH)


if __name__ == "__main__":
    h = horizon()
    print("[Predict / curve] 0.99^10 =", f"{h['p10']:.1%}", " 0.99^100 =", f"{h['p100']:.1%}")
    print("[Curve] eps for 90% at 100 turns =", f"{h['eps_needed']:.3%}",
          f" ({h['eps_ratio']:.1f}x lower than 1%)")
    print("[Freeze predict] 0.99^30 =", f"{h['p30']:.1%}", " violate at least once:", f"{1 - h['p30']:.1%}")
    b = bill()
    print("[Bill] resent input tokens =", f"{b['sent']:,}", " fresh =", f"{b['fresh']:,}",
          " ratio =", f"{b['ratio']:.1f}x", " output =", f"{b['out']:,}")
    print("[Bill] last call reads", f"{b['last_call']:,}", "tokens =", f"{b['window_share']:.0%}", "of 128,000")
    print("[Bill] $ no reuse =", f"{b['no_reuse']:.3f}", "(input", f"{b['in_no_reuse']:.3f}",
          "+ output", f"{b['out_cost']:.3f})", " $ reuse =", f"{b['reuse']:.3f}",
          " ratio =", f"{b['bill_ratio']:.1f}x")
    print("[Bill] resent after 60 turns / 30 turns =", f"{resent_after(60) / resent_after(30):.1f}x",
          " fresh =", f"{fresh_after(60) / fresh_after(30):.1f}x")
    c = cost_per_accepted()
    print("[Discuss bars] attempt $", f"{c['attempt']:.2f}", " base $", f"{c['base']:.2f}",
          " half token price $", f"{c['cheaper']:.3f}", " 60% acceptance $", f"{c['better']:.2f}",
          " attempts", f"{c['attempts']:.1f}", " break-even acceptance", f"{c['breakeven']:.2f}")
    r = runs()
    print("[Real] runs", r["n"], " submitted", r["submitted"], " accepted", r["accepted"],
          f"({r['acc_rate']:.1%})", " rejected", r["rejected"])
    print("[Real cost] total $", f"{r['total']:.2f}", " per attempt $", f"{r['per_attempt']:.3f}",
          " per accepted $", f"{r['per_accepted']:.3f}", " ratio", f"{r['per_accepted'] / r['per_attempt']:.2f}x",
          " rejected spend $", f"{r['rejected_spend']:.2f}", f"({r['rejected_share']:.0%})")
    print("[Real cost note] at 80% acceptance $", f"{r['at_80']:.2f}", " mean cost accepted $",
          f"{r['mean_cost_ok']:.2f}", " rejected $", f"{r['mean_cost_rejected']:.2f}")
    print("[Real calls] mean", f"{r['mean_calls']:.1f}", " median", r["median_calls"],
          " min", r["min_calls"], " max", r["max_calls"])
    print("[Real resend] all 500 runs: sent / new characters =", f"{r['resend_ratio']:.1f}x")
    for lo, hi, n, a in r["bins"]:
        print(f"[Real bins] calls {lo}-{hi - 1 if hi < 1000 else '+'}: n={n} resolved={a:.0%}")
    o = one_run()
    print("[One run] django-12050 calls", o["calls"], " sent chars", f"{o['sent']:,}", " new chars",
          f"{o['new']:,}", " ratio", f"{o['ratio']:.1f}x", " cost $", f"{o['cost']:.2f}")
    s = sealed()
    print("[Sealed] astropy-13398: FAIL_TO_PASS", f"{s['f2p_pass']} of {s['f2p_total']}",
          " PASS_TO_PASS broke", f"{s['p2p_broke']} of {s['p2p_total']}", " calls", s["calls"],
          " cost $", f"{s['cost']:.2f}")
    m = metr()
    print("[METR] shortest p50", f"{m['lo_s']:.1f} s", " longest", f"{m['hi_h']:.1f} h",
          " ratio", f"{m['ratio']:,.0f}x", " doubling", f"{m['dbl_all']:.1f}", "months (all),",
          f"{m['dbl_2023']:.1f}", "months (2023 on)", " models", len(m["points"]))
