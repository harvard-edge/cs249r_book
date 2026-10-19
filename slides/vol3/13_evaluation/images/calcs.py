"""Lecture 13 (Agent Evaluation): every derived number on the slides.

Run from anywhere: python3 calcs.py
Each block names its slide, its inputs and their sources, and prints the
values the slide and its note show. Inputs typed here are copied from the
source named beside them; everything printed is computed.
"""

import json
import math
import sys
from pathlib import Path

from scipy.stats import binom, norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Literature  # noqa: E402

Z95 = norm.ppf(0.975)  # 1.96, two-sided 95 percent


def wilson(x, n, z=Z95):
    """Wilson score interval, Agent Evaluation equation 2."""
    c = (x + z * z / 2) / (n + z * z)
    h = z / (n + z * z) * math.sqrt(x * (n - x) / n + z * z / 4)
    return max(0.0, c - h), min(1.0, c + h)


def wald(x, n, z=Z95):
    p = x / n
    h = z * math.sqrt(p * (1 - p) / n)
    return p - h, p + h


def newcombe10(both, cand_only, base_only, neither, z=Z95):
    """Newcombe (1998) method 10 for a paired difference, candidate minus
    baseline, with the continuity-corrected phi."""
    a, b, c, d = both, cand_only, base_only, neither
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    num = a * d - b * c
    num = max(num - n / 2, 0) if num > 0 else num
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = num / den if den else 0.0
    dl = math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    du = math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    theta = p1 - p2
    return theta, theta - dl, theta + du


def mcnemar_exact(b, c):
    m = b + c
    return min(1.0, 2 * binom.cdf(min(b, c), m, 0.5))


def head(t):
    print(f"\n== {t}")


# ---------------------------------------------------------------------------
head("Hook: pass^k at 90 percent (NM 13.2, Agent Evaluation §13.5.4)")
p, k, target = 0.90, 8, 0.90
print(f"pass^8 at 0.90: {p**k:.3f}  ({100*p**k:.0f} percent)")
print(f"pass^8 at 0.80: {0.8**k:.3f}  ({100*0.8**k:.0f} percent)")
print(f"pass^8 at 0.95: {0.95**k:.3f}  ({100*0.95**k:.0f} percent)")
print(f"pass@8 at 0.90: {1-(1-p)**k:.8f}")
need = target ** (1 / k)
print(f"single-run rate for 90% over 8: {need:.4f} ({100*need:.1f} percent); failure {100*(1-need):.1f}%; cut {(1-p)/(1-need):.1f}x")
for kk in range(1, 9):
    print(f"  k={kk}: pass^k {p**kk:.3f}  pass@k {1-(1-p)**kk:.6f}")

# ---------------------------------------------------------------------------
head("Real data: Terminal-Bench 2.0 scaffold spread (MLSysIM, Literature.TerminalBenchHarness)")
by = {}
for r in Literature.TerminalBenchHarness.entries:
    by.setdefault(r["model"], []).append(r["accuracy"])
by = {m: v for m, v in by.items() if len(v) >= 5}
spreads = {m: max(v) - min(v) for m, v in by.items()}
widest = max(spreads, key=spreads.get)
ses = sorted(r["accuracy_se"] for r in Literature.TerminalBenchHarness.entries if r["accuracy_se"])
med = lambda xs: sorted(xs)[len(xs) // 2] if len(xs) % 2 else sum(sorted(xs)[len(xs)//2-1:len(xs)//2+1]) / 2
print(f"models with >=5 scaffolds: {len(by)}")
print(f"widest: {widest} {100*min(by[widest]):.0f} to {100*max(by[widest]):.0f} percent over {len(by[widest])} scaffolds")
print(f"median spread: {100*med(list(spreads.values())):.0f} points; median reported SE: {100*med(ses):.0f} points")

# ---------------------------------------------------------------------------
head("Four verdicts: dropping unresolved trials (illustrative, §13.2.3)")
n_pass, n_fail, n_unres = 60, 25, 15
n = n_pass + n_fail + n_unres
print(f"default (unresolved = not accepted): {100*n_pass/n:.0f} percent")
print(f"bounds: {100*n_pass/n:.0f} to {100*(n_pass+n_unres)/n:.0f} percent")
print(f"dropping unresolved: {100*n_pass/(n_pass+n_fail):.1f} percent")

# ---------------------------------------------------------------------------
head("METR reward hacking by o3 (METR, 5 June 2025, task table)")
re_hacks, re_runs, hc_hacks, hc_runs = 39, 128, 8, 1087
re_pct, hc_pct = 30.4, 0.7  # METR's stated percentages
print(f"from counts: RE-Bench {100*re_hacks/re_runs:.2f}%  HCAST {100*hc_hacks/hc_runs:.2f}%")
print(f"stated: RE-Bench {re_pct}%  HCAST {hc_pct}%  ratio of stated {re_pct/hc_pct:.1f}x (METR: more than 43x)")
for name, h, rr in [("Optimize a Kernel", 6, 24), ("Rust Codecontest scaffold", 12, 28), ("Optimize LLM Foundry", 21, 21)]:
    print(f"  {name}: {100*h/rr:.1f}%")

# ---------------------------------------------------------------------------
head("Grader error rates (Agent Evaluation, exercise 3, §13.9; hypothetical)")
ref_fail, b_acc_fail, ref_pass, b_acc_pass = 80, 16, 120, 114
far = b_acc_fail / ref_fail
lo, hi = wilson(b_acc_fail, ref_fail)
fam = b_acc_fail / (b_acc_fail + b_acc_pass)
print(f"false-acceptance rate {100*far:.0f}% (Wilson {100*lo:.1f} to {100*hi:.1f}%)")
print(f"failure among accepted {b_acc_fail}/{b_acc_fail+b_acc_pass} = {100*fam:.1f}%")
print(f"false rejection {100*(ref_pass-b_acc_pass)/ref_pass:.0f}%")

# ---------------------------------------------------------------------------
head("Kappa from chance agreement (§13.4.3; checkpoint uses 80 percent)")
for rate in (0.9, 0.8):
    po = rate * rate + (1 - rate) * (1 - rate)
    pe = po  # independent raters: observed agreement equals chance
    print(f"each passes {rate:.0%}: raw agreement {100*po:.0f}%, kappa {(po-pe)/(1-pe):.0f}")

# ---------------------------------------------------------------------------
head("Zheng et al. 2023, Table 2 (default prompt): position consistency")
for j, cons in [("Claude-v1", 23.8), ("GPT-3.5", 46.2), ("GPT-4", 65.0)]:
    print(f"  {j}: consistent {cons}%, verdict changed by order {100-cons:.1f}%")

# ---------------------------------------------------------------------------
head("Tasks or repeats (Agent Evaluation, exercise 1, §13.9; hypothetical)")
mean_p, var_between = 0.6, 0.08
var_within = mean_p * (1 - mean_p) - var_between  # E[p(1-p)] = mean(1-mean) - Var(p)
print(f"within-task variance {var_within:.2f}; ratio {var_within/var_between:.1f} runs per task")
budget = 600
for n_tasks, r in [(600, 1), (200, 3), (100, 6)]:
    se = math.sqrt(var_between / n_tasks + var_within / (n_tasks * r))
    print(f"  {n_tasks} tasks x {r}: SE {100*se:.2f} points")
fixture = 10
best = None
for r in range(1, 31):
    n_tasks = budget // (fixture + r)
    v = var_between / n_tasks + var_within / (n_tasks * r)
    if best is None or v < best[0]:
        best = (v, r, n_tasks)
    if r <= 6:
        print(f"  fixture 10 runs, r={r}: {n_tasks} tasks, SE {100*math.sqrt(v):.2f}")
print(f"best with fixture cost: r={best[1]}, {best[2]} tasks, SE {100*math.sqrt(best[0]):.2f} points")

# ---------------------------------------------------------------------------
head("Wald and Wilson at 50 tasks (§13.5.2, figure 4 panel B)")
for x in (0, 2, 25):
    wl, wu = wald(x, 50)
    sl, su = wilson(x, 50)
    print(f"  {x}/50: Wald {100*wl:.2f} to {100*wu:.2f}; Wilson {100*sl:.2f} to {100*su:.2f}")
print(f"half-width at p=0.5, 100 tasks: {100*Z95*math.sqrt(0.25/100):.1f} points; Wilson {100*(wilson(50,100)[1]-wilson(50,100)[0])/2:.1f}")
print(f"tasks for 2-point half-width: {math.ceil((Z95/0.02)**2*0.25):,}")

# ---------------------------------------------------------------------------
head("Real data: SWE-bench Verified, Epoch AI runs (MLSysIM, Literature.SweBenchVerifiedRuns)")
rows = sorted(Literature.SweBenchVerifiedRuns.entries, key=lambda r: -r["mean_score"])
top = rows[:25]
pairs = list(zip(top[:-1], top[1:]))
tied = sum(abs(a["mean_score"] - b["mean_score"]) < Z95 * math.hypot(a["stderr"], b["stderr"]) for a, b in pairs)
se_typ = med([r["stderr"] for r in top])
print(f"models {len(rows)}; top 25 neighbor pairs tied {tied} of {len(pairs)}; typical SE {100*se_typ:.1f}; gap needed {100*Z95*math.sqrt(2)*se_typ:.1f} points")
print(f"median SE of top 25: {100*se_typ:.3f} points; 1.96 x sqrt(2) x median SE = {100*Z95*math.sqrt(2)*se_typ:.2f}")
print(f"top score {100*top[0]['mean_score']:.1f} ({top[0]['model']}); 25th {100*top[-1]['mean_score']:.1f}")

# ---------------------------------------------------------------------------
head("Paired comparison, NM 13.1 (§13.5.3, table 4)")
both, cand, base, neither = 70, 18, 12, 50
npairs = both + cand + base + neither
th, lo10, hi10 = newcombe10(both, cand, base, neither)
tol = 0.02
print(f"gain {100*th:.0f} points; method 10 interval {100*lo10:.2f} to {100*hi10:.2f}")
lw = th - lo10
print(f"lower half-width {100*lw:.2f}; needed below {100*(th+tol):.0f}")
print(f"pairs if gain holds: {math.ceil(npairs*(lw/(th+tol))**2)}")
print(f"pairs if true gain zero: {math.ceil(npairs*(lw/tol)**2):,} (chapter: about 1,900)")
print(f"McNemar exact p: {mcnemar_exact(base, cand):.3f}")
disc = (cand + base) / npairs
pc, pb = (both + cand) / npairs, (both + base) / npairs
var_pair = (disc - th ** 2)
var_ind = pc * (1 - pc) + pb * (1 - pb)
print(f"discordant {100*disc:.0f}%; paired variance share {100*var_pair/var_ind:.0f}%; independent needs {var_ind/var_pair:.1f}x tasks")
print(f"normal-approx half-width {100*Z95*math.sqrt(var_pair/npairs):.1f}")
pbar = 0.5
for d in (0.04, 0.02):
    narm = 2 * (Z95 + norm.ppf(0.8)) ** 2 * pbar * (1 - pbar) / d ** 2
    print(f"  independent arms, detect {100*d:.0f} points: {narm:,.0f} per arm")

# ---------------------------------------------------------------------------
head("Batch vs sequential, NM 13.3 (§13.5.5)")
p, kc, gen, chk = 0.4, 3, 0.03, 0.01
q = 1 - (1 - p) ** kc
batch = kc * (gen + chk)
en = q / p
seq = en * (gen + chk)
print(f"availability {100*q:.1f}%; batch per trial ${batch:.2f}; sequential mean candidates {en:.2f}, ${seq:.4f}")
print(f"per accepted: batch ${batch/q:.3f}, sequential ${seq/q:.3f}; premium {100*(batch/q/(seq/q)-1):.0f}%")
print(f"doubling p: ${(gen+chk)/(2*p):.2f}")

# ---------------------------------------------------------------------------
head("Kapoor et al. 2024, Table A1 (HumanEval, 164 problems, 5 runs)")
kap = [("LATS (GPT-4)", 88.0, 134.50), ("Warming (GPT-4)", 93.2, 2.45), ("GPT-4 baseline", 89.6, 1.93),
       ("Reflexion (GPT-4)", 87.8, 3.90), ("LDB (GPT-4)", 93.3, 6.36)]
print(f"LATS/Warming cost ratio {134.50/2.45:.0f}x; per problem LATS ${134.50/164:.2f}, Warming ${2.45/164:.3f}")

# ---------------------------------------------------------------------------
head("Our run (images/data/eval_run.json)")
run = HERE / "data" / "eval_run.json"
if run.exists():
    d = json.loads(run.read_text())
    R = d["rows"]
    tasks = [t["id"] for t in d["tasks"]]
    runs = d["runs_done"]
    print(f"model {d['model']}, T={d['temperature']}, runs {runs}, measured {d['measured']}, elapsed {d['elapsed_s']} s")
    for cfg in ("baseline_v1", "candidate_v2"):
        rr = [r for r in R if r["config"] == cfg]
        acc = sum(r["verdict"] == "pass" for r in rr)
        unk = sum(r["verdict"] == "unknown" for r in rr)
        vis = sum(r["visible"] == "pass" for r in rr)
        vis_but_fail = sum(r["visible"] == "pass" and r["verdict"] != "pass" for r in rr)
        lo, hi = wilson(acc, len(rr))
        tok = sum(r["input_tokens"] + r["output_tokens"] for r in rr)
        outt = sum(r["output_tokens"] for r in rr)
        sec = sum(r["seconds"] for r in rr)
        calls = sum(r["calls"] for r in rr)
        print(f"{cfg}: trials {len(rr)}, accepted {acc} ({100*acc/len(rr):.1f}%, Wilson {100*lo:.1f} to {100*hi:.1f}), unknown {unk}, "
              f"visible pass {vis} ({100*vis/len(rr):.1f}%), visible pass but sealed not pass {vis_but_fail} ({100*vis_but_fail/max(vis,1):.1f}% of visible passes)")
        print(f"   calls {calls}, tokens {tok:,} (output {outt:,}), seconds {sec:.0f}; tokens per accepted {tok/max(acc,1):,.0f}; "
              f"seconds per accepted {sec/max(acc,1):.1f}")
        per = []
        for t in tasks:
            vv = [r["verdict"] == "pass" for r in rr if r["task"] == t]
            per.append(sum(vv) / len(vv))
        print("   per-task pass rate: " + ", ".join(f"{t} {x:.2f}" for t, x in zip(tasks, per)))
        for kk in (1, 2, 4, 8):
            if kk <= runs:
                pk = sum(math.comb(round(x * runs), kk) / math.comb(runs, kk) for x in per) / len(per)
                ak = sum(1 - math.comb(runs - round(x * runs), kk) / math.comb(runs, kk) for x in per) / len(per)
                print(f"   k={kk}: pass^k {100*pk:.1f}%  pass@k {100*ak:.1f}%")
    tpa = {}
    for cfg in ("baseline_v1", "candidate_v2"):
        rr = [r for r in R if r["config"] == cfg]
        tpa[cfg] = sum(r["input_tokens"] + r["output_tokens"] for r in rr) / sum(r["verdict"] == "pass" for r in rr)
    print(f"tokens per accepted, candidate over baseline: {tpa['candidate_v2']/tpa['baseline_v1']:.3f} ({100*(tpa['candidate_v2']/tpa['baseline_v1']-1):.0f}% more)")
    # paired, run by run
    key = {(r["task"], r["run"], r["config"]): r["verdict"] == "pass" for r in R}
    a = b = c = dd = 0
    for t in tasks:
        for rn in range(runs):
            x, y = key[(t, rn, "candidate_v2")], key[(t, rn, "baseline_v1")]
            a += x and y; b += x and not y; c += y and not x; dd += (not x) and (not y)
    th, l, u = newcombe10(a, b, c, dd)
    print(f"paired 2x2: both {a}, candidate only {b}, baseline only {c}, neither {dd}")
    print(f"gain {100*th:.1f} points, method 10 {100*l:.1f} to {100*u:.1f}; McNemar exact p {mcnemar_exact(c, b):.3f}")
    print("   (pairs share a task and run index; tasks repeat, so this interval treats runs as independent pairs)")
    # buggy originals pass their own visible test
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_run", HERE / "data" / "eval_run.py")
    er = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(er)
    orig_vis = sum(er.run_checks(t["code"], [t["visible"]])[0] == "pass" for t in er.TASKS)
    print(f"buggy originals passing their visible test: {orig_vis} of {len(er.TASKS)}")
else:
    print("eval_run.json not yet written")
