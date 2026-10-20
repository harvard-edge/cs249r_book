#!/usr/bin/env python3
"""Every derived number in Lecture 14 (Operating Agents), one block per slide.

Inputs are typed with their source. Sources:
  OA    = the rendered chapter, Operating Agents (Volume III), dev text 2026-10-18.
  NM    = a Napkin Math notebook of that chapter (scenario values the chapter
          labels hypothetical; slides say "illustrative").
  Ex    = Example 14.1 of that chapter (hypothetical branch outcomes).
  AE    = the rendered chapter Agent Evaluation, Napkin Math 13.1 (§13.5).
  OpenAI 2025a = "Expanding on what we missed with sycophancy," 2 May 2025
          (read from the Wayback Machine copy of 10 Oct 2026).
  Dapper = Sigelman et al., "Dapper," Google Technical Report dapper-2010-1,
          Table 2 (web search cluster; errors 2.5% latency, 0.15% throughput).
  SWE-agent = Yang et al., "SWE-agent," NeurIPS 2024, arXiv 2405.15793v3,
          Table 3 (GPT-4 Turbo, 300 SWE-bench Lite instances).
  Argo  = Argo Rollouts docs, docs/features/analysis.md, commit 176a66f
          (background-analysis example).
  run   = trace_run.json, spans the OpenTelemetry Python SDK 1.27.0 recorded
          when trace_run.py replayed NM 14.1's intervals with sleep().

Run from this folder: python3 calcs.py   (prints "label: value")
make_figs.py imports these values so figures and slides agree.
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).parent
Z95 = 1.959964  # two-sided 95 percent normal quantile


def show(label, value):
    print(f"  {label}: {value}")


def wilson(x, n, z=Z95):
    """Wilson score interval for x successes in n trials (Lecture 13's method)."""
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ------------------------------------------------------------ hook (OpenAI)
# OpenAI 2025a: rollout started Thu 24 Apr and completed Fri 25 Apr 2025;
# two days of monitoring; system-prompt mitigation late Sunday (27 Apr);
# full rollback initiated Monday (28 Apr) and took around 24 hours.
HOOK = dict(rollout_start=24, rollout_done=25, prompt_fix=27, rollback_start=28,
            rollback_hours=24)


def hook():
    h = HOOK
    return dict(days_live_before_rollback=h["rollback_start"] - h["rollout_done"],
                days_first_exposure_to_rollback_start=h["rollback_start"] - h["rollout_start"],
                rollback_done_day=h["rollback_start"] + h["rollback_hours"] / 24)


# --------------------------------------------- purpose number (OA Purpose)
# A failure that recurs ten times under a one-percent sample.
def purpose_miss():
    return (1 - 0.01) ** 10


# --------------------------------------------------- NM 14.1 critical path
# OA Table 2 intervals (s): retrieve [0,0.2]; propose [0.2,3.2];
# tests [3.2,9.2]; lint [3.2,5.2]; synthesize [9.7,11.7].
NM141 = dict(retrieve=0.2, propose=3.0, tests=6.0, lint=2.0, gap=0.5, synth=2.0)


def critical_path(tests=None, lint=None, d=NM141):
    t = d["tests"] if tests is None else tests
    li = d["lint"] if lint is None else lint
    return d["retrieve"] + d["propose"] + max(t, li) + d["gap"] + d["synth"]


def nm141():
    d = NM141
    leaf_sum = d["retrieve"] + d["propose"] + d["tests"] + d["lint"] + d["synth"]
    explained = d["retrieve"] + d["propose"] + max(d["tests"], d["lint"]) + d["synth"]
    total = critical_path()
    tests1 = critical_path(tests=1.0)
    lint1 = critical_path(lint=1.0)
    return dict(leaf_sum=leaf_sum, explained=explained, total=total,
                unattributed=total - explained,
                tests_cut_to_1=tests1, tests_saving=total - tests1,
                tests_cut=d["tests"] - 1.0,
                lint_cut_to_1=lint1, lint_saving=total - lint1)


def run_spans():
    """Times from the real OpenTelemetry run, seconds after the root span starts."""
    recs = json.loads((HERE / "trace_run.json").read_text())
    t0 = min(r["start_ns"] for r in recs)
    spans = {r["name"]: ((r["start_ns"] - t0) / 1e9, (r["end_ns"] - t0) / 1e9)
             for r in recs}
    leaves = [v for k, v in spans.items() if not k.startswith("invoke_agent")]
    leaf_sum = sum(e - s for s, e in leaves)
    root = spans["invoke_agent repair_agent"]
    tests = spans["execute_tool run_tests"]
    synth = spans["chat synthesize_report"]
    gap = synth[0] - tests[1]
    return dict(spans=spans, leaf_sum=leaf_sum, root=root[1] - root[0], gap=gap,
                explained=(root[1] - root[0]) - gap)


# --------------------------------------------------- head sampling (OA 14.3)
def miss(p, n):
    return (1 - p) ** n


# --------------------------------------------------------- NM 14.2 retention
NM142 = dict(per_hour=10_000, mb=1.4, failed=400, pending=100, ordinary=9_500,
             hours=24, head_p=0.05, control_p=0.02)


def nm142():
    d = NM142
    gb = d["mb"] * d["hours"] / 1000           # GB per (trace per hour) per day
    full = d["per_hour"] * gb
    head_n = d["per_hour"] * d["head_p"]
    head = head_n * gb
    tail_n = d["failed"] + d["pending"] + d["ordinary"] * d["control_p"]
    tail = tail_n * gb
    floor = (d["failed"] + d["pending"]) * gb
    per_point = d["ordinary"] * 0.01 * gb
    return dict(full_gb=full, head_per_hour=head_n, head_gb=head,
                head_failed=d["failed"] * d["head_p"],
                head_pending=d["pending"] * d["head_p"],
                tail_per_hour=tail_n, tail_gb=tail, full_over_tail=full / tail,
                floor_gb=floor, per_point_gb=per_point)


# ------------------------------------------------------- Dapper (Table 2)
DAPPER = [("1/1", 16.3), ("1/2", 9.40), ("1/4", 6.38), ("1/8", 4.12),
          ("1/16", 2.12), ("1/1024", -0.20)]   # avg latency change, percent


# ------------------------------------------------------ Example 14.1 branches
EX141 = [("Original interface", 1, 19, 0), ("Warning only", 8, 0, 12),
         ("Paginated interface", 20, 0, 0)]   # accepted, forbidden, budget stop


def ex141():
    out = {}
    for name, acc, forb, bud in EX141:
        n = acc + forb + bud
        lo, hi = wilson(acc, n)
        out[name] = dict(n=n, acc=acc, rate=acc / n, lo=lo, hi=hi)
    # 20 of 20 accepted: upper limit on the failure rate
    lo, hi = wilson(0, 20)
    out["fail_upper_20of20"] = hi
    # consecutive clean runs before the Wilson upper failure limit drops below 5%
    n = 1
    while wilson(0, n)[1] >= 0.05:
        n += 1
    out["runs_for_5pct"] = n
    return out


def wilson_curve(ns):
    return [wilson(0, n)[1] for n in ns]


# ------------------------------------------------------- SWE-agent Table 3
SWE_N = 300
SWE_VIEWER = [("30 lines", 14.3), ("100 lines", 18.0), ("Full file", 12.7)]
SWE_SEARCH = [("Summarized", 18.0), ("Iterative", 12.0), ("No search", 15.7)]


def swe():
    out = {}
    for name, pct in SWE_VIEWER + SWE_SEARCH:
        x = round(pct / 100 * SWE_N)
        lo, hi = wilson(x, SWE_N)
        out[name] = dict(pct=pct, x=x, lo=100 * lo, hi=100 * hi)
    return out


# ----------------------------------------------- Jensen-Shannon (OA 14.5.1)
def kl(p, q):
    return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0)


def js(p, q):
    m = [(a + b) / 2 for a, b in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


JS_BASE, JS_CAND = (0.70, 0.30), (0.50, 0.50)   # tests, lint


# ---------------------------------------------------------------- NM 14.3
NM143 = dict(p0=0.90, p1=0.75, alpha=0.01, beta=0.05, rate=60, frac=0.05,
             delay=0.5)


def sprt(p0, p1, alpha, beta):
    a = math.log((1 - beta) / alpha)
    b = math.log(beta / (1 - alpha))
    mu1 = p1 * math.log(p1 / p0) + (1 - p1) * math.log((1 - p1) / (1 - p0))
    num = (1 - beta) * a + beta * b
    return dict(upper=a, lower=b, inc=mu1, terminal=num, n=num / mu1,
                up_fail=math.log((1 - p1) / (1 - p0)), dn_success=math.log(p1 / p0))


def nm143(frac=None):
    d = NM143
    f = d["frac"] if frac is None else frac
    s = sprt(d["p0"], d["p1"], d["alpha"], d["beta"])
    per_hour = d["rate"] * f
    admit_h = s["n"] / per_hour
    excess = (s["n"] + f * d["rate"] * d["delay"]) * (d["p0"] - d["p1"])
    return dict(**s, per_hour=per_hour, admit_h=admit_h,
                decide_h=admit_h + d["delay"], excess=excess)


# ------------------------------------------------------------- Argo example
# Argo: setWeight 20, 40, 60, 80, each followed by a 10 min pause; analysis
# every 5 min, failureLimit 3, successCondition result[0] >= 0.95.
# Applied to the chapter's canary scenario (NM 14.3): 60 eligible tasks/hour,
# each task's label arrives 0.5 h after admission.
ARGO = dict(weights=[0.20, 0.40, 0.60, 0.80], pause_min=10)


def argo():
    d, n = ARGO, NM143
    full_min = len(d["weights"]) * d["pause_min"]
    # candidate tasks admitted in each 10-minute step
    admitted = [w * n["rate"] * d["pause_min"] / 60 for w in d["weights"]]
    # labels available when the rollout reaches 100%: tasks admitted at least
    # delay before that moment
    cutoff = full_min - n["delay"] * 60
    labeled = 0.0
    t = 0
    for w in d["weights"]:
        seg_end = t + d["pause_min"]
        overlap = max(0.0, min(seg_end, cutoff) - t)
        labeled += w * n["rate"] * overlap / 60
        t = seg_end
    return dict(full_min=full_min, admitted=sum(admitted), labeled=labeled,
                needed=nm143()["n"])


# ------------------------------------------------- release decision (AE 13.1)
PAIRED = dict(lo=-3.21, hi=11.12, gain=4.0, margin=2.0, pairs=150, need=217)


def family(k=4, alpha=0.05):
    return dict(bound=k * alpha, independent=1 - (1 - alpha) ** k,
                per_test=alpha / k)


if __name__ == "__main__":
    print("Hook (OpenAI 2025a timeline)")
    for k, v in hook().items():
        show(k, v)
    print("Purpose: P(no detailed record), 10 recurrences at 1% =",
          f"{purpose_miss():.3f}")
    print("NM 14.1 critical path (s)")
    for k, v in nm141().items():
        show(k, f"{v:.1f}")
    r = run_spans()
    print("Real OpenTelemetry run (s)")
    for k, (s, e) in sorted(r["spans"].items(), key=lambda kv: kv[1][0]):
        show(k, f"[{s:.3f}, {e:.3f}]")
    show("leaf_sum", f"{r['leaf_sum']:.2f}")
    show("root duration", f"{r['root']:.2f}")
    show("explained (root minus gap)", f"{r['explained']:.2f}")
    show("gap", f"{r['gap']:.3f}")
    print("Head sampling misses")
    for p, n in [(0.10, 5), (0.10, 50), (0.05, 20), (0.01, 10)]:
        show(f"p={p}, n={n}", f"{miss(p, n):.4f}")
    print("NM 14.2 retention")
    for k, v in nm142().items():
        show(k, f"{v:.3f}")
    print("Example 14.1 branches")
    e = ex141()
    for k, v in e.items():
        if isinstance(v, dict):
            show(k, f"{v['acc']}/{v['n']} = {v['rate']:.2f}, Wilson [{v['lo']:.3f}, {v['hi']:.3f}]")
        else:
            show(k, f"{v:.4f}" if isinstance(v, float) else v)
    print("SWE-agent Table 3 with Wilson intervals (n=300)")
    for k, v in swe().items():
        show(k, f"{v['pct']}% = {v['x']}/300, [{v['lo']:.1f}, {v['hi']:.1f}]")
    s_ = swe()
    show("100 lines minus full file (points)", f"{s_['100 lines']['pct'] - s_['Full file']['pct']:.1f}")
    print("SPRT failures in a row from the start")
    st = nm143()
    for k in (4, 5):
        show(f"{k} failures", f"{k * st['up_fail']:.2f} (hold at {st['upper']:.2f})")
    hw = (PAIRED["gain"] - PAIRED["lo"]) * math.sqrt(PAIRED["pairs"] / PAIRED["need"])
    show("217-pair interval (points)", f"[{PAIRED['gain'] - hw:.2f}, {PAIRED['gain'] + hw:.2f}]")
    print("Jensen-Shannon")
    j = js(JS_BASE, JS_CAND)
    show("JS (nats)", f"{j:.4f}")
    show("share of ln 2", f"{j / math.log(2):.3f}")
    show("ln 2", f"{math.log(2):.3f}")
    print("NM 14.3 SPRT")
    for f in (0.05, 0.10, 1.0):
        s = nm143(f)
        show(f"frac={f}", f"upper {s['upper']:.4f}, lower {s['lower']:.4f}, inc {s['inc']:.5f}, "
             f"terminal {s['terminal']:.4f}, n {s['n']:.2f}, per_hour {s['per_hour']:.1f}, "
             f"admit {s['admit_h']:.2f} h, decide {s['decide_h']:.2f} h, excess {s['excess']:.2f}")
    s = nm143()
    show("step per failure", f"{s['up_fail']:.4f}")
    show("step per success", f"{s['dn_success']:.4f}")
    print("Argo rule applied to the canary scenario")
    for k, v in argo().items():
        show(k, f"{v:.2f}" if isinstance(v, float) else v)
    print("Family-wise error, 4 predicates at 0.05")
    for k, v in family().items():
        show(k, f"{v:.4f}")
    print("Paired comparison (AE NM 13.1): interval",
          PAIRED["lo"], "to", PAIRED["hi"], "against", -PAIRED["margin"])
