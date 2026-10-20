#!/usr/bin/env python3
"""Every derived number in Lecture 19 (Agent Economics), Volume III.

Run from this folder: python3 calcs.py
Each block names the slide it feeds and the source of its inputs. "AE" is the
rendered chapter, Agent Economics (chapter_text_dev/vol3/19_agent_economics),
read 2026-10-19. "NM 19.k" is its Napkin Math notebook k. Inputs typed here are
the chapter's stated scenario values (hypothetical unless marked measured);
everything printed is computed. Real data come from MLSysIM's literature
series (CSV files read directly, so this script needs only the standard
library): Epoch AI price series, SWE-bench trajectories, Brown et al. (2024)
coverage, FrontierSWE trials, and SemiAnalysis InferenceX.
"""

import csv
import datetime as dt
import math
import statistics as st
from pathlib import Path

SERIES = (Path(__file__).resolve().parents[4] / "mlsysim" / "mlsysim" / "literature" / "data" / "series")


def rows(name):
    with open(SERIES / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


M = 1e-6  # dollars per token from dollars per million tokens


# ---------------------------------------------------------------- NM 19.1 hook
def nm191():
    """Hook and synthesis. AE NM 19.1, table 1: two configurations, 100 tickets,
    one attempt each. Prices $/M (in, out); turns; tokens per turn (in, out);
    accepted tickets; environment $ per turn; assessment $ per attempt."""
    cfg = {
        "frontier": dict(pin=3.0, pout=15.0, turns=4, tin=16000, tout=800, acc=75),
        "lowprice": dict(pin=0.30, pout=1.20, turns=7, tin=28000, tout=1200, acc=25),
    }
    env, assess, n, triage = 0.03, 0.05, 100, 10.0
    out = {}
    for k, c in cfg.items():
        model = c["turns"] * (c["tin"] * c["pin"] + c["tout"] * c["pout"]) * M
        envc = c["turns"] * env
        attempt = model + envc + assess
        total = n * attempt
        rej = n - c["acc"]
        out[k] = dict(model=model, env=envc, assess=assess, attempt=attempt, total=total,
                      per_acc=total / c["acc"], rej_per_acc=rej / c["acc"],
                      total_tri=total + triage * rej, per_acc_tri=(total + triage * rej) / c["acc"],
                      acc=c["acc"])
    f, l = out["frontier"], out["lowprice"]
    out["ratio"] = l["per_acc"] / f["per_acc"]
    out["ratio_tri"] = l["per_acc_tri"] / f["per_acc_tri"]
    out["token_cut"] = 1 - l["model"] / f["model"]
    out["attempt_cut"] = 1 - l["attempt"] / f["attempt"]
    out["acc_drop"] = 1 - l["acc"] / f["acc"]
    return out


# --------------------------------------------------- Epoch AI price trend (real)
def price_trend():
    """Epoch AI (2025), lowest price at a fixed capability (MLSysIM
    LowestPriceAtCapability). Fitted decline per year per threshold series,
    as the chapter's LEGO cell computes it."""
    r = rows("epoch_lowest_price_at_capability")
    series = {}
    for x in r:
        series.setdefault((x["benchmark"], x["threshold_model"]), []).append(x)

    def rate(pts):
        t = [dt.date.fromisoformat(p["release_date"]).toordinal() / 365.25 for p in pts]
        y = [math.log10(float(p["usd_per_mtok"])) for p in pts]
        tm, ym = st.mean(t), st.mean(y)
        slope = sum((a - tm) * (b - ym) for a, b in zip(t, y)) / sum((a - tm) ** 2 for a in t)
        return 10 ** -slope

    rates = sorted(rate(v) for v in series.values())
    chosen = {}
    for key in [("MMLU", "GPT-3"), ("MMLU", "GPT-4-0314"), ("GPQA Diamond", "GPT-4-0314")]:
        pts = sorted(series[key], key=lambda p: p["release_date"])
        chosen[key] = dict(pts=pts, rate=rate(pts))
    return dict(n=len(series), lo=rates[0], med=st.median(rates), hi=rates[-1], chosen=chosen)


# --------------------------------------------- SWE-bench trajectories (real)
SUBS = [
    ("20240620_sweagent_claude3.5sonnet", "2024 run A"),
    ("20250522_sweagent_claude-4-sonnet-20250514", "2025 run B"),
    ("20260217_mini-v2.0.0_claude-4-6-opus", "2026 run C"),
    ("20260219_mini-v2.0.0_gpt-5-2-codex", "2026 run D"),
]


def swe_cohorts():
    """MLSysIM SweBenchTrajectories: per-trajectory cost as each agent recorded
    it (model tokens at list prices; no sandbox or human cost). Run B is a
    seed-0 sample of 100 of 500. Cohort cost per resolved = total / resolved."""
    r = rows("swe_bench_verified_trajectories")
    out = []
    for sub, label in SUBS:
        v = [x for x in r if x["submission"] == sub]
        cost = [float(x["rep_cost_usd"]) for x in v]
        res = [x for x in v if x["resolved"] == "True"]
        cr = [float(x["rep_cost_usd"]) for x in res]
        cu = [float(x["rep_cost_usd"]) for x in v if x["resolved"] != "True"]
        tin = sum(float(x["rep_input_tokens_sum"]) for x in v)
        tout = sum(float(x["rep_output_tokens_sum"]) for x in v)
        out.append(dict(sub=sub, label=label, scaffold=v[0]["scaffold"], model=v[0]["model"],
                        n=len(v), resolved=len(res), total=sum(cost),
                        per_resolved=sum(cost) / len(res), mean_resolved=st.mean(cr),
                        mean_unresolved=st.mean(cu), input_share=tin / (tin + tout),
                        calls=st.mean(int(x["n_model_calls"]) for x in v),
                        understate=(sum(cost) / len(res)) / st.mean(cr)))
    return out


def swe_cap():
    """Run A (SWE-agent 0.x, 2024): Kapoor et al. (2024, p. 1) report a $4
    cap per run. Count recorded trajectory costs above it."""
    r = [x for x in rows("swe_bench_verified_trajectories") if x["submission"] == SUBS[0][0]]
    c = [float(x["rep_cost_usd"]) for x in r]
    over = [v for v in c if v > 4.0]
    return dict(n=len(c), over=len(over), max=max(c), max_over=max(c) - 4.0,
                near=sum(1 for v in c if v > 3.5))


def kapoor():
    """Kapoor et al. (2024), arXiv:2407.01502, Table A1 (measured): mean
    accuracy (%) and mean total cost (USD, April 2024 prices) over five runs
    on the 164 HumanEval problems. Typed from the table."""
    t = [("LATS (GPT-4)", 88.0, 134.50), ("LATS (GPT-3.5)", 80.4, 9.49),
         ("LDB (GPT-4, GPT-3.5)", 91.0, 2.19), ("LDB (Reflexion, GPT-4)", 92.9, 7.26),
         ("LDB (Reflexion, GPT-3.5)", 88.9, 4.19), ("LDB (GPT-4)", 93.3, 6.36),
         ("LDB (GPT-3.5)", 80.2, 0.63), ("GPT-4", 89.6, 1.93), ("GPT-3.5", 73.9, 0.05),
         ("Reflexion (GPT-4)", 87.8, 3.90), ("Warming (GPT-4)", 93.2, 2.45),
         ("Retry (GPT-4)", 92.0, 2.51), ("Escalation", 85.0, 0.27)]
    d = dict((n, (a, c)) for n, a, c in t)
    return dict(rows=t, esc_vs_ldb35=d["LDB (GPT-3.5)"][1] / d["Escalation"][1],
                lats_vs_warm=d["LATS (GPT-4)"][1] / d["Warming (GPT-4)"][1])


# ------------------------------------------------------------- NM 19.2 resend
def nm192():
    """AE NM 19.2: 30 turns, 8,000-token first prompt, 1,500 appended per turn,
    200 output tokens per call; $3 fresh, $0.30 cached, $15 output per M."""
    H, P0, a, out = 30, 8000, 1500, 200
    pin, pc, pout = 3.0, 0.30, 15.0
    lengths = [P0 + (t - 1) * a for t in range(1, H + 1)]
    logical = sum(lengths)
    fresh = P0 + (H - 1) * a
    in_no = logical * pin * M
    in_re = (fresh * pin + (logical - fresh) * pc) * M
    o = H * out * pout * M
    per_turn_no = [L * pin * M + out * pout * M for L in lengths]
    per_turn_re = [((L if t == 1 else a) * pin + (0 if t == 1 else L - a) * pc) * M + out * pout * M
                   for t, L in enumerate(lengths, 1)]
    cached_read = (logical - fresh) * pc * M
    miss = 0.1 * (logical - fresh) * (pin - pc) * M
    return dict(logical=logical, fresh=fresh, in_no=in_no, in_re=in_re, out=o,
                total_no=in_no + o, total_re=in_re + o, ratio=(in_no + o) / (in_re + o),
                reread_share=cached_read / (in_re + o), miss=miss, miss_pct=miss / (in_re + o),
                per_turn_no=per_turn_no, per_turn_re=per_turn_re, resend_mult=logical / fresh)


# ---------------------------------------------------------- NM 19.3 compaction
def nm193(cached_frac=0.10):
    """AE NM 19.3: compact at turn 16 from 30,500 to 12,000 tokens; summary
    emits 1,500; 15 ordinary calls remain; tariff of NM 19.2."""
    S, Sc, K, R = 30500, 12000, 1500, 15
    pin, pout = 3.0, 15.0
    pc = cached_frac * pin
    U = (pout * K + pin * Sc) * M
    s = pc * (S - Sc) * M
    nstar = 1 + math.ceil(U / s)
    diff = {n: U - (n - 1) * s for n in range(1, 31)}
    return dict(U=U, s=s, nstar=nstar, diff=diff, R=R, diff_R=diff[R], diff_11=diff[11],
                diff_12=diff[12])


# ------------------------------------------------------------ NM 19.4 retries
def nm194():
    """AE NM 19.4: cheap p = 0.30 at $0.10; strong p = 0.70 at $0.40; cap 5."""
    p, c, k, ps, cs = 0.30, 0.10, 5, 0.70, 0.40
    cov = 1 - (1 - p) ** k
    att = cov / p
    seq = c / p
    par = k * c / cov
    return dict(cov=cov, att=att, spend=c * att, seq=seq, par_task=k * c, par=par,
                strong=cs / ps, par_ratio=par / seq)


def coverage_real():
    """Brown et al. (2024) via MLSysIM RepeatedSamplingCoverage: SWE-bench Lite
    (all 300), DeepSeek-Coder-V2-Instruct with Moatless Tools, coverage at k.
    Independent-trials prediction 1 - (1 - p1)^k from the measured k = 1."""
    r = [x for x in rows("repeated_sampling_coverage")
         if x["dataset"] == "SWE-bench Lite (all 300)"]
    pts = sorted((int(x["k"]), float(x["coverage"])) for x in r)
    p1 = pts[0][1]
    pred = [(k, 1 - (1 - p1) ** k) for k, _ in pts]
    return dict(pts=pts, pred=pred, p1=p1)


# ------------------------------------------------------- NM 19.5 critical path
def nm195():
    """AE NM 19.5, table 2: disjoint serial intervals in seconds."""
    model, tests, sandbox, runtime = 29.25, 245.60, 37.35, 0.20
    T = model + tests + sandbox + runtime
    f = model / T
    t4 = T * ((1 - f) + f / 4)
    t_inf = T - model
    half_tests = tests / 2
    # Speculative decoding, AE §19.4.4: gamma = 5 drafted, r = 0.08 per draft
    gamma, r = 5, 0.08
    S = lambda m: (m + 1) / (gamma * r + 1)  # noqa: E731
    s_hi, s_lo = S(3.6), S(0.3)
    return dict(model=model, tests=tests, sandbox=sandbox, runtime=runtime, T=T, f=f,
                t4=t4, save4=T - t4, sp4=T / t4, t_inf=t_inf, sp_inf=T / t_inf,
                ceiling=1 / (1 - f), half_tests=half_tests,
                breakeven=gamma * r, s_hi=s_hi, s_lo=s_lo,
                t_hi=T - model + model / s_hi, t_lo=T - model + model / s_lo,
                sp_hi=T / (T - model + model / s_hi))


# -------------------------------------------------------- NM 19.6 cascade
def nm196():
    """AE NM 19.6: 100,000 tasks; tier costs and times; gate $0.0010, 0.40 s."""
    n = 100000
    cost = [0.0004, 0.0020, 0.0600]
    time = [0.25, 0.80, 4.50]
    gate_c, gate_t = 0.0010, 0.40
    reached = [100000, 40000, 12000]
    accepted = [60000, 28000, 11400]
    total = sum(r * (c + gate_c) for r, c in zip(reached, cost))
    acc = sum(accepted)
    direct_acc = 97000
    direct = n * (cost[2] + gate_c)
    mean_t = sum((r / n) * (t + gate_t) for r, t in zip(reached, time))
    worst_t = sum(t + gate_t for t in time)
    # AE §19.10 fallacy: relaxed tier-1 gate falsely accepts 0.9% of its exits,
    # each costing $10 to remediate.
    harm = accepted[0] * 0.009 * 10.0
    return dict(total=total, per_init=total / n, acc=acc, per_acc=total / acc,
                direct=direct, direct_per_acc=direct / direct_acc, ratio=(direct / direct_acc) / (total / acc),
                mean_t=mean_t, worst_t=worst_t, direct_t=time[2] + gate_t,
                tolls=cost[0] + gate_c + cost[1] + gate_c, saving=direct / direct_acc - total / acc,
                cond=[a / r for a, r in zip(accepted, reached)], reached=reached, accepted=accepted,
                harm=harm, per_acc_harm=(total + harm) / acc)


def frontierswe():
    """FrontierSWE V2 via MLSysIM FrontierSweTrials (measured): 19
    configurations, 5 trials each, cost and hours per trial."""
    r = rows("frontier_swe_v2_trials")
    pts = [dict(model=x["model"], mean=100 * float(x["mean5"]), best=100 * float(x["best5"]),
                worst=100 * float(x["worst5"]), usd=float(x["usd_per_trial"]),
                hours=float(x["hours_per_trial"])) for x in r]
    pts.sort(key=lambda p: -p["mean"])
    top, ru = pts[0], pts[1]
    return dict(pts=pts, n=len(pts), lo=min(p["usd"] for p in pts), hi=max(p["usd"] for p in pts),
                cost_ratio=top["usd"] / ru["usd"], gap=top["mean"] - ru["mean"],
                spread=st.median(p["best"] - p["worst"] for p in pts),
                h_lo=min(p["hours"] for p in pts), h_hi=max(p["hours"] for p in pts))


# ------------------------------------------------------ NM 19.7 capacity
def nm197():
    """AE NM 19.7: 1 admission per minute, 20 minutes present, 60 calls,
    30,000 quota tokens per call, 2,000,000 tokens per minute allowance."""
    lam, T, calls, tok, quota = 1.0, 20.0, 60, 30000, 2_000_000
    conc = lam * T
    call_rate = lam * calls
    tok_rate = call_rate * tok
    use = tok_rate / quota
    lam_max = quota / (calls * tok)
    after_drop = use / 0.9      # acceptance falls 10 percent, goodput held
    return dict(conc=conc, call_rate=call_rate, tok_traj=calls * tok, tok_rate=tok_rate,
                use=use, lam_max=lam_max, after_drop=after_drop)


def inferencex():
    """SemiAnalysis InferenceX via MLSysIM InferenceXDeepSeekR1 (measured):
    DeepSeek-R1 FP8, 1,024 in and 1,024 out, 8x H200, SGLang, run of
    2026-05-20. Per-user tokens/s and output tokens/s per GPU by concurrency."""
    r = [x for x in rows("inferencex_dsr1_fp8_1k1k")
         if x["hardware"] == "h200" and x["framework"] == "sglang" and x["disagg"] == "0"
         and x["date"] == "2026-05-20"]
    pts = sorted((int(x["conc"]), float(x["interactivity_tok_s_user"]), float(x["output_tok_s_gpu"]))
                 for x in r)
    lo, hi = pts[0], pts[-1]
    return dict(pts=pts, thr_mult=hi[2] / lo[2], speed_drop=lo[1] / hi[1],
                tpot_lo=1000 / lo[1], tpot_hi=1000 / hi[1], conc_mult=hi[0] / lo[0])


# ------------------------------------------------------- NM 19.8 queueing
def nm198():
    """AE NM 19.8: E[S] = 1 s, ca^2 = 2, cs^2 = 4, holding $0.002 per s."""
    ES, ca2, cs2, hold = 1.0, 2.0, 4.0, 0.002
    v = (ca2 + cs2) / 2
    W = lambda rho: v * rho / (1 - rho) * ES  # noqa: E731
    curve = [(x / 100, W(x / 100)) for x in range(5, 96)]
    rho_one = 1 / (1 + v)    # wait equals one service time
    tail = 1 - (1 - 0.01) ** 30
    return dict(v=v, w70=W(0.7), w90=W(0.9), h70=W(0.7) * hold, h90=W(0.9) * hold,
                thr=0.9 / 0.7 - 1, wait_mult=W(0.9) / W(0.7), curve=curve, rho_one=rho_one,
                tail30=tail, tail_curve=[(n, 1 - 0.99 ** n) for n in range(1, 101)])


# ------------------------------------------------- NM 19.9 dedicated capacity
def nm199(cached=0.80):
    """AE NM 19.9: 5 input per output; $3 / $0.30 / $15 per M; dedicated
    $28 per hour plus $2 per M output; reference 1,200 output tokens/s."""
    ratio, pin, pc, pout = 5, 3.0, 0.30, 15.0
    F, v, R = 28.0, 2.0, 1200
    w_in = (1 - cached) * pin + cached * pc
    P = pout + ratio * w_in
    ustar = F / (3600 * R * (P - v) * M)
    u = 0.52
    tok = 3600 * R * u
    api = tok * P * M
    ded = F + tok * v * M
    return dict(w_in=w_in, P=P, ustar=ustar, tok=tok, api=api, ded=ded, save=api - ded,
                line=lambda x: (3600 * R * x * P * M, F + 3600 * R * x * v * M))


# --------------------------------------------------- NM 19.10 conservation
def nm1910():
    """AE NM 19.10: $5 budget; planning $0.042 settled; trace child $1.20,
    closes at $0.38; fuzzing child $2.50, $0.85 settled and $1 reserved for a
    remote operation at local stop; late bill $0.70."""
    B, plan, trace_a, trace_c, fz_a, fz_s, fz_r, late = 5.0, 0.042, 1.20, 0.38, 2.50, 0.85, 1.00, 0.70
    rows_ = [
        ("Planning settled", plan, 0.0),
        ("Both allocations granted", plan, trace_a + fz_a),
        ("Trace child closed", plan + trace_c, fz_a),
        ("Fuzzing stopped; bill pending", plan + trace_c, fz_a),
        ("Late bill settled; child closed", plan + trace_c + fz_s + late, 0.0),
    ]
    table = [(n, s, r, B - s - r) for n, s, r in rows_]
    return dict(table=table, naive=fz_a - fz_s, safe=fz_a - fz_s - fz_r,
                child_free=fz_a - fz_s - fz_r, returned=fz_a - fz_s - late, late=late)


# ------------------------------------------------- NM 19.11 architectures
def nm1911():
    """AE NM 19.11: 10,000 upgrades a month; check $0.002 per candidate;
    human fallback $35, resolves 90 percent."""
    check, H, h = 0.002, 35.0, 0.90
    q = [0.55, 0.42, 0.30]
    reach = [1.0]
    for x in q[:-1]:
        reach.append(reach[-1] * (1 - x))
    loop_p = 1 - math.prod(1 - x for x in q)
    turns = sum(reach)
    designs = {
        "Rule-based rewriter": (0.001 + check, 0.32),
        "Single model call": (0.045 + check, 0.68),
        "Bounded 3-turn loop": ((0.006 + check) * turns, loop_p),
    }
    out = {}
    for k, (c, p) in designs.items():
        res = p + (1 - p) * h
        comb = c + (1 - p) * H
        out[k] = dict(c=c, p=p, per_auto=c / p, res=res, comb=comb, per_res=comb / res)
    loop = out["Bounded 3-turn loop"]
    return dict(d=out, turns=turns, loop_p=loop_p, month_spend=10000 * loop["comb"],
                month_res=10000 * loop["res"], marginal=[x * H for x in q])


def nm1912():
    """AE NM 19.12: adaptation payback."""
    base = 5 * (12000 * 3 + 800 * 15) * M
    adapt = 4 * (1500 * 1 + 800 * 4) * M
    C = 4000 + 500 + 8000
    N = C / (base - adapt)
    return dict(base=base, adapt=adapt, save=base - adapt, C=C, N=N, Nint=math.ceil(N),
                months=N / 20000, months_q=N / 5000, eval_share=8000 / C, train_share=500 / C)


# --------------------------------------------------- Lighthouse 19.1 table
def lighthouse():
    """AE Lighthouse 19.1, table 8. Inputs typed from the table; the repair
    row's tokens are NM 19.2's cached bill. Support: $4 handoff on each
    rejected session (expected handoff = 4 (1 - p))."""
    nm = nm192()
    L = {
        "Repair": dict(tok=nm["total_re"], sbx=0.90, asmt=0.10, hum=0.0, p=0.40, big="sbx"),
        "Research": dict(tok=0.90, sbx=0.10, asmt=0.05, hum=0.0, p=0.60, big="tok"),
        "Remediation": dict(tok=0.15, sbx=0.10, asmt=0.20, hum=20.0, p=0.70, big="hum"),
        "Support": dict(tok=0.03, sbx=0.01, asmt=0.01, hum=4 * (1 - 0.85), p=0.85, big="hum"),
    }
    out = {}
    for k, x in L.items():
        a = x["tok"] + x["sbx"] + x["asmt"] + x["hum"]
        ce = a / x["p"]
        half_tok = (x["tok"] / 2) / a
        half_big = (x[x["big"]] / 2) / a
        if k == "Support":
            p2 = x["p"] + 0.10
            a2 = x["tok"] + x["sbx"] + x["asmt"] + 4 * (1 - p2)
            plus10 = 1 - (a2 / p2) / ce
        else:
            plus10 = 0.10 / (x["p"] + 0.10)
        out[k] = dict(attempt=a, ce=ce, half_tok=half_tok, half_big=half_big, plus10=plus10,
                      model_share=x["tok"] / a)
    return out


if __name__ == "__main__":
    h = nm191()
    print("== NM 19.1 hook")
    for k in ("frontier", "lowprice"):
        x = h[k]
        print(f"  {k}: model ${x['model']:.3f}, env ${x['env']:.2f}, attempt ${x['attempt']:.3f}, "
              f"cohort ${x['total']:.3f}, per accepted ${x['per_acc']:.2f}, rejected per accepted {x['rej_per_acc']:.2f}, "
              f"with triage ${x['total_tri']:.3f} -> ${x['per_acc_tri']:.2f}")
    print(f"  ratio {h['ratio']:.1f}x; with triage {h['ratio_tri']:.1f}x; token cut {h['token_cut']:.1%}; "
          f"attempt cut {h['attempt_cut']:.1%}; acceptance drop {h['acc_drop']:.1%}")
    p = price_trend()
    print(f"== Epoch price trend: {p['n']} series, decline {p['lo']:.0f}x to {p['hi']:.0f}x per year, median {p['med']:.0f}x")
    for key, v in p["chosen"].items():
        print(f"  {key}: {v['rate']:.0f}x per year, first ${v['pts'][0]['usd_per_mtok']} last ${v['pts'][-1]['usd_per_mtok']}")
    print("== SWE-bench cohorts (recorded model-token cost)")
    for c in swe_cohorts():
        print(f"  {c['label']} {c['scaffold']} / {c['model']}: n={c['n']} resolved={c['resolved']} "
              f"total=${c['total']:.2f} per resolved=${c['per_resolved']:.2f} mean resolved=${c['mean_resolved']:.2f} "
              f"mean unresolved=${c['mean_unresolved']:.2f} understatement {c['understate']:.1f}x "
              f"input share {c['input_share']:.1%} calls {c['calls']:.1f}")
    kp = kapoor()
    print(f"== Kapoor Table A1: LDB (GPT-3.5) costs {kp['esc_vs_ldb35']:.1f}x Escalation; LATS (GPT-4) costs {kp['lats_vs_warm']:.0f}x Warming (GPT-4)")
    cap = swe_cap()
    print(f"== run A cap: {cap['over']} of {cap['n']} above $4, max ${cap['max']:.2f} (over by ${cap['max_over']:.2f}), above $3.50: {cap['near']}")
    r = nm192()
    print(f"== NM 19.2: logical {r['logical']:,}, fresh {r['fresh']:,} ({r['resend_mult']:.1f}x), "
          f"input ${r['in_no']:.3f} vs ${r['in_re']:.3f}, totals ${r['total_no']:.3f} vs ${r['total_re']:.3f} "
          f"({r['ratio']:.1f}x); cached re-read share {r['reread_share']:.1%}; 10% miss +${r['miss']:.3f} ({r['miss_pct']:.1%})")
    print(f"  per turn: first ${r['per_turn_no'][0]:.4f}/${r['per_turn_re'][0]:.4f}, last ${r['per_turn_no'][-1]:.4f}/${r['per_turn_re'][-1]:.4f}")
    for cf in (0.10, 0.05):
        c = nm193(cf)
        print(f"== NM 19.3 cached {cf:.0%}: U ${c['U']:.5f}, saving/call ${c['s']:.5f}, payback {c['nstar']} calls, "
              f"diff at 11 ${c['diff_11']:.5f}, at 12 ${c['diff_12']:.5f}, at 15 ${c['diff_R']:.5f}")
    t = nm194()
    print(f"== NM 19.4: coverage {t['cov']:.1%}, attempts {t['att']:.3f}, spend ${t['spend']:.3f}, seq ${t['seq']:.3f}, "
          f"parallel ${t['par_task']:.3f}/task ${t['par']:.3f}/acc ({t['par_ratio']:.1f}x), strong ${t['strong']:.3f}")
    cv = coverage_real()
    print(f"== Brown et al. SWE-bench Lite: p1 {cv['p1']:.3f}; " +
          ", ".join(f"k={k}: {c:.3f} vs {pr:.3f}" for (k, c), (_, pr) in zip(cv["pts"], cv["pred"])))
    c5 = nm195()
    print(f"== NM 19.5: T {c5['T']:.2f}, f {c5['f']:.1%}, 4x -> {c5['t4']:.2f} s (saves {c5['save4']:.2f}, {c5['sp4']:.2f}x), "
          f"instant {c5['t_inf']:.2f} ({c5['sp_inf']:.2f}x), ceiling {c5['ceiling']:.2f}x, halving tests saves {c5['half_tests']:.1f}")
    print(f"  spec: breakeven {c5['breakeven']:.1f}; S(3.6) {c5['s_hi']:.2f}x -> {c5['t_hi']:.1f} s ({c5['sp_hi']:.2f}x); "
          f"S(0.3) {c5['s_lo']:.2f}x -> {c5['t_lo']:.1f} s")
    c6 = nm196()
    print(f"== NM 19.6: total ${c6['total']:,.0f}, per init ${c6['per_init']:.5f}, accepted {c6['acc']:,}, per acc ${c6['per_acc']:.5f}; "
          f"direct ${c6['direct']:,.0f}, per acc ${c6['direct_per_acc']:.5f}, ratio {c6['ratio']:.1f}x; "
          f"mean {c6['mean_t']:.3f} s worst {c6['worst_t']:.2f} s direct {c6['direct_t']:.2f} s; tolls ${c6['tolls']:.4f}; "
          f"saving/acc ${c6['saving']:.3f}; cond {[round(x, 2) for x in c6['cond']]}")
    print(f"  fallacy: harm ${c6['harm']:,.0f}; cascade with harm ${c6['total'] + c6['harm']:,.0f} -> ${c6['per_acc_harm']:.5f} per accepted vs direct ${c6['direct_per_acc']:.5f}")
    fs = frontierswe()
    print(f"== FrontierSWE: {fs['n']} configs, ${fs['lo']:.0f} to ${fs['hi']:,.0f}, hours {fs['h_lo']:.0f} to {fs['h_hi']:.0f}, "
          f"top/runner-up cost {fs['cost_ratio']:.1f}x for {fs['gap']:.1f} points, median spread {fs['spread']:.1f} points")
    c7 = nm197()
    print(f"== NM 19.7: concurrency {c7['conc']:.0f}, calls/min {c7['call_rate']:.0f}, tokens/traj {c7['tok_traj']:,}, "
          f"tokens/min {c7['tok_rate']:,.0f} ({c7['use']:.0%}), max admission {c7['lam_max']:.2f}/min, after 10% drop {c7['after_drop']:.0%}")
    ix = inferencex()
    print(f"== InferenceX H200 SGLang: " + ", ".join(f"c{c}: {u:.1f} tok/s/user, {g:.1f} tok/s/GPU" for c, u, g in ix["pts"]))
    print(f"  conc x{ix['conc_mult']:.0f}: throughput x{ix['thr_mult']:.1f}, per-user speed /{ix['speed_drop']:.1f}, "
          f"TPOT {ix['tpot_lo']:.1f} ms -> {ix['tpot_hi']:.1f} ms")
    q = nm198()
    print(f"== NM 19.8: v {q['v']:.0f}; wait {q['w70']:.0f} s -> {q['w90']:.0f} s ({q['wait_mult']:.1f}x) for +{q['thr']:.1%} throughput; "
          f"holding ${q['h70']:.3f} -> ${q['h90']:.3f}; wait < service below rho {q['rho_one']:.2f}; tail at 30 calls {q['tail30']:.2f}")
    for cf in (0.80, 0.0):
        d = nm199(cf)
        print(f"== NM 19.9 cached {cf:.0%}: input ${d['w_in']:.2f}/M, API ${d['P']:.1f}/M out, u* {d['ustar']:.1%}; "
              f"at 52%: {d['tok']:,.0f} tok/h, API ${d['api']:.2f}, dedicated ${d['ded']:.2f}, saving ${d['save']:.2f}")
    b = nm1910()
    print("== NM 19.10:")
    for n, s, rr, fr in b["table"]:
        print(f"  {n}: settled ${s:.3f} reserved ${rr:.2f} free ${fr:.3f}")
    print(f"  naive release ${b['naive']:.2f}, safe ${b['safe']:.2f}, returned at close ${b['returned']:.2f}")
    a = nm1911()
    print(f"== NM 19.11: loop acceptance {a['loop_p']:.2%}, turns {a['turns']:.3f}")
    for k, x in a["d"].items():
        print(f"  {k}: auto spend ${x['c']:.6f}, per auto acc ${x['per_auto']:.5f}, resolution {x['res']:.3%}, "
              f"combined ${x['comb']:.6f}, per resolved ${x['per_res']:.2f}")
    print(f"  month spend ${a['month_spend']:,.2f} for {a['month_res']:,.1f} resolved; marginal q*H {a['marginal']}")
    ad = nm1912()
    print(f"== NM 19.12: base ${ad['base']:.4f}, adapted ${ad['adapt']:.4f}, saving ${ad['save']:.4f}, N* {ad['N']:,.2f} -> {ad['Nint']:,}, "
          f"{ad['months']:.2f} months, at 5,000/month {ad['months_q']:.1f}; eval {ad['eval_share']:.0%}, training {ad['train_share']:.0%}")
    lh = lighthouse()
    print("== Lighthouse 19.1")
    for k, x in lh.items():
        print(f"  {k}: attempt ${x['attempt']:.4f}, C_eff ${x['ce']:.2f}, half token {x['half_tok']:.1%}, "
              f"half largest {x['half_big']:.1%}, +10 points {x['plus10']:.1%}, model share {x['model_share']:.1%}")
