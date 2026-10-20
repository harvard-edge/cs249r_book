#!/usr/bin/env python3
"""Every derived number in Lecture 18 (Multi-Agent Coordination), one block per slide.

Inputs are typed with their source. Sources:
  MA    = the rendered chapter, Multi-Agent Coordination (Volume III), dev text
          2026-10-18. NM = a Napkin Math notebook there (scenario values the
          chapter labels hypothetical; slides say "illustrative").
  RL    = the rendered chapter Reinforcement Learning from Verifiable Rewards,
          Napkin Math 17.1 (recall slide).
  Hadfield 2025 = Hadfield et al., "How we built our multi-agent research
          system," Anthropic Engineering, 13 June 2025 (read 2026-10-19).
  Dean 2013 = Dean and Barroso, "The Tail at Scale," CACM 56(2), 2013,
          Table 1 and the hedged-request paragraph.
  Kim 2025 = Kim, Garg, Peng, Garg, "Correlated Errors in Large Language
          Models," ICML 2025, arXiv 2506.07962v1, Introduction (Helm leaderboard).
  MAST  = Cemri et al., "Why Do Multi-Agent LLM Systems Fail?", NeurIPS 2025
          Datasets and Benchmarks, arXiv 2503.13657v3, Figure 1 (1,642 traces).

Run from this folder: python3 calcs.py   (prints "label: value")
make_figs.py imports these functions so figures and slides agree.
"""

import math

from scipy.special import betainc, betaln, comb


def show(label, value):
    print(f"  {label}: {value}")


# ------------------------------------------------------------- hook / NM 18.6
# Hadfield 2025: agents use about 4x the tokens of chat; multi-agent systems
# about 15x; the multi-agent system beat a single agent by 90.2% on an
# internal research evaluation. MA NM 18.6 builds the break-even on these.
CHAT_MULT_AGENT = 4.0
CHAT_MULT_MULTI = 15.0
EVAL_GAIN = 0.902


def hook():
    m = CHAT_MULT_MULTI / CHAT_MULT_AGENT          # delegation token multiple
    breakeven_ps = 1 / m                           # pM = m pS <= 1
    acc_mult = 1 + EVAL_GAIN                        # if the gain were an acceptance multiple
    cost_ratio = m / acc_mult                       # delegated / single cost per accepted task
    return dict(token_multiple=m, breakeven_single_success=breakeven_ps,
                acceptance_multiple=acc_mult, cost_ratio=cost_ratio)


# ------------------------------------------------------- recall (RL NM 17.1)
def recall():
    n, p, fa = 8192, 0.10, 0.025                    # RL NM 17.1 inputs
    true_acc = n * p * 1.0
    false_acc = n * (1 - p) * fa
    return dict(true_accepts=true_acc, false_accepts=false_acc,
                false_share=false_acc / (true_acc + false_acc))


# --------------------------------------------------- NM 18.1 (Amdahl screen)
def elapsed(n, p, c):
    """Equation 1 denominator with linear overhead c(N-1)."""
    return (1 - p) + p / n + c * (n - 1)


def screen(p=0.80, c=0.02, nmax=16):
    t = {n: elapsed(n, p, c) for n in range(1, nmax + 1)}
    best = min(t, key=t.get)
    return dict(t=t, best=best, t_best=t[best], speedup=1 / t[best],
                ceiling=1 / (1 - p), approx_best=math.sqrt(p / c))


# ------------------------------------------------ USL (MA Theorem 18.1)
def usl(n, sigma, kappa):
    return n / (1 + sigma * (n - 1) + kappa * n * (n - 1))


def usl_summary(sigma=0.20, kappa=0.02, nmax=60):
    nstar = math.sqrt((1 - sigma) / kappa)
    ints = {n: usl(n, sigma, kappa) for n in range(1, nmax + 1)}
    best = max(ints, key=ints.get)
    slower = min((n for n in range(2, nmax + 1) if ints[n] < 1), default=None)
    return dict(nstar=nstar, best=best, peak=ints[best], first_slower=slower,
                retro_threshold=(1 - sigma) / kappa)


# ------------------------------------------------- mesh vs supervisor (§18.3)
def deliveries(n):
    return dict(mesh=n * (n - 1), supervisor=2 * n, ratio=(n - 1) / 2)


# ------------------------------------------------------- NM 18.2 (DAG bound)
DAG = {  # MA Table 3: duration (s), predecessors
    "v1": (20.5, []), "v2": (42, ["v1"]), "v3": (42, ["v1"]), "v4": (42, ["v1"]),
    "v5": (43, ["v2", "v3"]), "v6": (16.5, ["v4"]), "v7": (69, ["v5", "v6"]),
}


def dag(durations=None, slots=4):
    d = {k: (durations or {}).get(k, v[0]) for k, v in DAG.items()}
    start, finish = {}, {}
    for k in DAG:  # dict order is topological
        start[k] = max([finish[p] for p in DAG[k][1]], default=0.0)
        finish[k] = start[k] + d[k]
    work = sum(d.values())
    span = finish["v7"]
    return dict(start=start, finish=finish, work=work, work_bound=work / slots,
                makespan=span, speedup=work / span,
                slack_v6=start["v7"] - finish["v6"])


# --------------------------------------------------- NM 18.3 (reference passing)
def reference():
    kids, slice_tok, env_tok, read_tok, price = 6, 48000, 350, 3600, 3e-6
    paste = kids * slice_tok
    ref = kids * (env_tok + read_tok)
    return dict(paste_tokens=paste, paste_usd=paste * price, ref_tokens=ref,
                ref_usd=ref * price, reduction=paste / ref,
                read_fraction=read_tok / slice_tok,
                breakeven_reads=slice_tok - env_tok)


# ------------------------------------------------- equation 4 (shared writes)
def no_shared_write(n, files=1000, w=10):
    pr = 1.0
    for k in range(1, n):
        pr *= comb(files - k * w, w, exact=True) / comb(files, w, exact=True)
    return pr


def collisions(files=1000, w=10, nmax=16):
    pc = {n: 1 - no_shared_write(n, files, w) for n in range(1, nmax + 1)}
    even = min(n for n in pc if pc[n] > 0.5)
    return dict(p=pc, even=even)


# --------------------------------------------- NM 18.4 (correlated reviewers)
def majority_indep(n, q):
    return sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range((n + 1) // 2, n + 1))


def betabin_pmf(k, n, a, b):
    return comb(n, k) * math.exp(betaln(k + a, n - k + b) - betaln(a, b))


def reviewers(q=0.20, rho=0.40, nmax=15):
    s = 1 / rho - 1
    a, b = q * s, (1 - q) * s
    ind = {n: majority_indep(n, q) for n in range(1, nmax + 1, 2)}
    mix = {n: sum(betabin_pmf(k, n, a, b) for k in range((n + 1) // 2, n + 1))
           for n in range(1, nmax + 1, 2)}
    floor = 1 - betainc(a, b, 0.5)
    unan_ind = q ** 5
    unan_mix = betabin_pmf(5, 5, a, b)
    return dict(alpha=a, beta=b, ind=ind, mix=mix, floor=floor,
                unan_ind=unan_ind, unan_mix=unan_mix, unan_ratio=unan_mix / unan_ind)


# Kim 2025: on Helm (71 LLMs, 14,042 questions), model pairs agree about 60%
# of the time when both are wrong; uniform choice among the three wrong
# answers of a four-option question would agree 1/3 of the time.
KIM_MEASURED = 0.60
KIM_RANDOM = 1 / 3

# ---------------------------------------------- NM 18.5 (barrier vs quorum)
def barrier(f=0.90, n=8, k=6, nmax=16):
    allp = {m: f ** m for m in range(1, nmax + 1)}
    below = min(m for m in allp if allp[m] < 0.5)
    quorum = sum(comb(n, j) * f ** j * (1 - f) ** (n - j) for j in range(k, n + 1))
    return dict(all=allp, below_half=below, all_n=f ** n, quorum=quorum)


# Dean 2013, Table 1 (a real Google fan-out service, measured at the root):
DEAN = {"one random leaf": (1, 5, 10), "95% of all leaves": (12, 32, 70),
        "all leaves": (40, 87, 140)}               # 50th, 95th, 99th pct, ms
DEAN_HEDGE = dict(before_ms=1800, after_ms=74, extra=0.02)   # 1,000 keys, 100 servers


def dean():
    p99 = DEAN["all leaves"][2] / DEAN["one random leaf"][2]
    return dict(p99_ratio=p99,
                hedge_ratio=DEAN_HEDGE["before_ms"] / DEAN_HEDGE["after_ms"])


# ----------------------------------------------------- §18.7.4 (credit window)
def credit(lam=5.0, crit_tokens=500, decode=40.0, minutes=10, window=8, target_s=60):
    mu = decode / crit_tokens
    backlog = (lam - mu) * minutes * 60
    wait_h = backlog / mu / 3600
    residence = window / mu
    w_target = math.floor(target_s * mu)
    return dict(mu=mu, backlog=backlog, wait_h=wait_h, residence=residence,
                window_for_target=w_target)


# ------------------------------------- reservation estimate (MA Exercise 7a)
def reservation(remaining=50, reserve=5, own=6, asks=(20, 20, 20)):
    avail = remaining - reserve - own
    copied = sum(asks)
    admitted, left = [], avail
    for a in asks:
        if a <= left:
            admitted.append(True)
            left -= a
        else:
            admitted.append(False)
    return dict(available=avail, copied_liability=copied, admitted=admitted,
                left_after=left, equal_max=avail // len(asks))


# --------------------------------------------- NM 18.7 (deadline coverage)
PRICE_IN, PRICE_OUT, PRICE_SVC = 3e-6, 10e-6, 0.01   # MA NM 18.7, illustrative
COHORT = {  # MA Table 9
    "single": dict(acc=80, ontime=70, tin=2_000_000, tout=400_000, svc=200, failed=3),
    "delegated": dict(acc=80, ontime=78, tin=3_000_000, tout=600_000, svc=300, failed=6),
}


def cohort():
    out = {}
    for k, v in COHORT.items():
        spend = v["tin"] * PRICE_IN + v["tout"] * PRICE_OUT + v["svc"] * PRICE_SVC
        out[k] = dict(spend=spend, per_acc=spend / v["acc"], failed_share=v["failed"] / spend)
    d, s = COHORT["delegated"], COHORT["single"]
    out["ratio"] = out["delegated"]["per_acc"] / out["single"]["per_acc"]
    out["extra_spend"] = out["delegated"]["spend"] - out["single"]["spend"]
    out["extra_ontime"] = d["ontime"] - s["ontime"]
    out["marginal"] = out["extra_spend"] / out["extra_ontime"]
    return out


# MAST Figure 1 (prevalence over 1,642 traces), percent.
MAST_CAT = {"System design": 44.2, "Inter-agent misalignment": 32.3, "Task verification": 23.5}
MAST_MODES = [  # (label, percent, category)
    ("Step repetition", 15.7, "design"),
    ("Reasoning-action mismatch", 13.2, "align"),
    ("Unaware of termination", 12.4, "design"),
    ("Disobey task specification", 11.8, "design"),
    ("Incorrect verification", 9.1, "verify"),
    ("No or incomplete verification", 8.2, "verify"),
    ("Task derailment", 7.4, "align"),
]


if __name__ == "__main__":
    print("hook / NM 18.6")
    h = hook()
    show("token multiple 15/4", f"{h['token_multiple']:.2f}")
    show("break-even single-agent success", f"{100 * h['breakeven_single_success']:.1f}%")
    show("acceptance multiple if 90.2% gain", f"{h['acceptance_multiple']:.3f}")
    show("cost per accepted task ratio", f"{h['cost_ratio']:.2f}")
    print("recall / RL NM 17.1")
    r = recall()
    show("true accepts", f"{r['true_accepts']:.2f}")
    show("false accepts", f"{r['false_accepts']:.2f}")
    show("false share", f"{100 * r['false_share']:.1f}%")
    print("NM 18.1 screen p=0.80 c=0.02")
    s = screen()
    show("t(4)", f"{s['t'][4]:.3f}")
    show("best N", s["best"])
    show("t(best)", f"{s['t_best']:.3f}")
    show("speedup", f"{s['speedup']:.2f}")
    show("t(8)", f"{s['t'][8]:.3f}")
    show("ceiling 1/(1-p)", f"{s['ceiling']:.1f}")
    show("sqrt(p/c)", f"{s['approx_best']:.2f}")
    s2 = screen(c=0.01)
    show("c=0.01 best N", s2["best"])
    show("c=0.01 speedup", f"{s2['speedup']:.2f}")
    print("Checkpoint 18.1 estimate p=0.9 c=0.02")
    e = screen(p=0.9, c=0.02, nmax=40)
    show("sqrt(p/c)", f"{e['approx_best']:.2f}")
    show("best N", e["best"])
    show("speedup", f"{e['speedup']:.2f}")
    show("ceiling", f"{e['ceiling']:.0f}")
    e1 = screen(p=0.9, c=0.01, nmax=40)
    show("halve c: best N", e1["best"])
    show("halve c: speedup", f"{e1['speedup']:.2f}")
    e2 = screen(p=0.95, c=0.02, nmax=40)
    show("halve serial: best N", e2["best"])
    show("halve serial: speedup", f"{e2['speedup']:.2f}")
    show("halve serial: ceiling", f"{e2['ceiling']:.0f}")
    print("USL repair (0.20, 0.02)")
    u = usl_summary()
    show("N* continuous", f"{u['nstar']:.2f}")
    show("best integer", u["best"])
    show("peak capacity", f"{u['peak']:.2f}")
    show("first count slower than one worker", u["first_slower"])
    show("capacity at 41", f"{usl(41, 0.2, 0.02):.3f}")
    print("USL research (0.10, 0.002)")
    ur = usl_summary(0.10, 0.002, nmax=600)
    show("N* continuous", f"{ur['nstar']:.2f}")
    show("best integer", ur["best"])
    show("peak capacity", f"{ur['peak']:.2f}")
    show("fan-out ratio 21/6", f"{ur['best'] / u['best']:.1f}")
    print("mesh vs supervisor")
    for n in (2, 4, 8):
        d = deliveries(n)
        show(f"N={n} mesh / supervisor / ratio", f"{d['mesh']} / {d['supervisor']} / {d['ratio']:.1f}")
    print("NM 18.2 DAG")
    g = dag()
    show("aggregate slot-work", f"{g['work']:.1f} s")
    show("work bound (4 slots)", f"{g['work_bound']:.2f} s")
    show("makespan", f"{g['makespan']:.1f} s")
    show("speedup vs serial", f"{g['speedup']:.2f}")
    show("slack before v7 on v4-v6 branch", f"{g['slack_v6']:.1f} s")
    show("v6 20 s shorter: makespan", f"{dag({'v6': max(0.0, 16.5 - 20)})['makespan']:.1f} s")
    show("v5 20 s shorter: makespan", f"{dag({'v5': 23})['makespan']:.1f} s")
    print("NM 18.3 reference passing")
    rf = reference()
    show("paste tokens / usd", f"{rf['paste_tokens']} / {rf['paste_usd']:.3f}")
    show("reference tokens / usd", f"{rf['ref_tokens']} / {rf['ref_usd']:.3f}")
    show("reduction", f"{rf['reduction']:.1f}")
    show("read fraction", f"{100 * rf['read_fraction']:.1f}%")
    print("equation 4 shared writes (1,000 files, 10 each)")
    c = collisions()
    for n in (2, 5, 8):
        show(f"P(shared write) N={n}", f"{100 * c['p'][n]:.1f}%")
    show("first N above even odds", c["even"])
    show("P(disjoint) N=2", f"{100 * (1 - c['p'][2]):.1f}%")
    c2 = collisions(files=2000)
    show("2,000 files: first N above even odds", c2["even"])
    print("NM 18.4 reviewers q=0.20 rho=0.40")
    rv = reviewers()
    show("alpha, beta", f"{rv['alpha']:.2f}, {rv['beta']:.2f}")
    show("majority 5 independent", f"{100 * rv['ind'][5]:.2f}%")
    show("majority 5 mixture", f"{100 * rv['mix'][5]:.2f}%")
    show("unanimous 5 independent", f"{100 * rv['unan_ind']:.3f}%")
    show("unanimous 5 mixture", f"{100 * rv['unan_mix']:.2f}%")
    show("unanimous ratio", f"{rv['unan_ratio']:.0f}")
    show("floor P(theta>1/2)", f"{100 * rv['floor']:.1f}%")
    show("majority 15 independent", f"{100 * rv['ind'][15]:.2f}%")
    show("majority 15 mixture", f"{100 * rv['mix'][15]:.1f}%")
    show("Kim measured / random", f"{100 * KIM_MEASURED:.0f}% / {100 * KIM_RANDOM:.0f}%")
    print("NM 18.5 barrier")
    b = barrier()
    show("all 8 on time", f"{100 * b['all_n']:.2f}%")
    show("6 of 8 on time", f"{100 * b['quorum']:.2f}%")
    show("first N below even odds", b["below_half"])
    dn = dean()
    show("Dean p99 all leaves / one leaf", f"{dn['p99_ratio']:.0f}")
    show("Dean hedge p99.9 reduction", f"{dn['hedge_ratio']:.1f}")
    print("credit window")
    cr = credit()
    show("service rate", f"{cr['mu']:.2f} items/s")
    show("backlog after 10 min", f"{cr['backlog']:.0f}")
    show("newest wait", f"{cr['wait_h']:.1f} h")
    show("residence W=8", f"{cr['residence']:.0f} s")
    show("largest window for 60 s residence", cr["window_for_target"])
    print("reservation estimate")
    rs = reservation()
    show("available", rs["available"])
    show("copied liability", rs["copied_liability"])
    show("admitted in order", rs["admitted"])
    show("left after", rs["left_after"])
    show("largest equal allowance", rs["equal_max"])
    print("NM 18.7 cohort")
    co = cohort()
    for k in ("single", "delegated"):
        show(f"{k} spend / per accepted", f"{co[k]['spend']:.2f} / {co[k]['per_acc']:.3f}")
    show("delegated failed share", f"{100 * co['delegated']['failed_share']:.1f}%")
    show("cost ratio", f"{co['ratio']:.2f}")
    show("marginal cost per extra on-time", f"{co['marginal']:.2f}")
    show("MAST category sum", f"{sum(MAST_CAT.values()):.1f}")
