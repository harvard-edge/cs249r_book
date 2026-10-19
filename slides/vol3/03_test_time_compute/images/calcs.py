#!/usr/bin/env python3
"""Every derived number in Lecture 3 (Test-Time Compute), Volume III.

Run from this folder: python3 calcs.py
Each block prints the values with the label the slide uses. Inputs come
from the rendered chapter (TTC = Test-Time Compute, Volume III chapter 3;
NM = Napkin Math), from MLSysIM, or from a cited source, named in comments.
Most chapter inputs are hypothetical scenario values; the slides label them
"illustrative". make_figs.py imports this module, so plots and slides share
one computation.
"""

import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "mlsysim"))

# -- Shared scenario inputs (TTC NM 3.1, 3.2, 3.6, 3.7) ----------------------
T_TOK = 21.075e-3          # s per generated token, illustrative (TTC NM 3.6, 3.7; "21.07 ms" in NM 3.1)
P_IN = 3e-6                # $ per input token, hypothetical (NM 3.2)
P_OUT = 15e-6              # $ per output token, hypothetical (NM 3.1, 3.2)
P_SCORER = 0.5e-6          # $ per scorer input token, hypothetical (NM 3.6)


def hook():
    """NM 3.1: a 1,024-token explanation vs a 40 ms test."""
    tokens, test_s = 1024, 0.040
    gen_s = tokens * T_TOK
    return dict(gen_s=gen_s, ratio=gen_s / test_s, bill=tokens * P_OUT)


def axes():
    """NM 3.2: one 4,096-token output ceiling spent three ways."""
    prompt, ceiling = 4096, 4096
    # Depth: one path of 4,096 tokens
    depth = dict(inp=prompt, out=ceiling, time=ceiling * T_TOK)
    # Breadth: 4 concurrent streams of 1,024, each billed its own prompt copy
    k, each = 4, 1024
    breadth = dict(inp=k * prompt, out=k * each, time=each * T_TOK)
    # Feedback: 3 turns of 1,024 tokens, each followed by a 1.5 s check
    # returning 340 tokens; later requests append candidates and reports
    turns, check_s, report = 3, 1.5, 340
    inp = sum(prompt + j * (each + report) for j in range(turns))
    feedback = dict(inp=inp, out=turns * each, time=turns * (each * T_TOK + check_s))
    for d in (depth, breadth, feedback):
        d["bill"] = d["inp"] * P_IN + d["out"] * P_OUT
    premium = (k - 1) * prompt
    premium_8x = (k - 1) * prompt * 8          # repository context grows eightfold
    return dict(depth=depth, breadth=breadth, feedback=feedback,
                premium=premium, premium_8x=premium_8x,
                premium_8x_usd=premium_8x * P_IN,
                breadth_vs_depth_bill=breadth["bill"] / depth["bill"],
                serial_breadth_s=k * each * T_TOK)   # Checkpoint 3.1: one stream at a time


def pass_at(p, k):
    return 1 - (1 - p) ** k


def pool():
    """NM 3.3: pools of 1, 2, 4, 8 at a 30 percent pass rate."""
    p, ks = 0.30, [1, 2, 4, 8]
    avail = {k: pass_at(p, k) for k in ks}
    last_gain = p * (1 - p) ** 7
    par_per_acc = 8 / avail[8]
    seq_attempts = avail[8] / p                 # [1-(1-p)^8]/p
    seq_per_acc = seq_attempts / avail[8]       # = 1/p
    # Pitfall (TTC 3.8): half the tasks at 60 percent, half never pass
    plug = pass_at(0.5 * 0.6 + 0.5 * 0.0, 8)
    true = 0.5 * pass_at(0.6, 8) + 0.5 * pass_at(0.0, 8)
    return dict(avail=avail, last_gain=last_gain, par_per_acc=par_per_acc,
                seq_attempts=seq_attempts, seq_per_acc=seq_per_acc,
                ratio=par_per_acc / seq_per_acc, plug=plug, true=true)


def estimator():
    """TTC section 3.3.3 subset estimator; TTC Exercise 2(d): 4 of 20 pass, k = 5."""
    n, c, k = 20, 4, 5
    subset = 1 - math.comb(n - c, k) / math.comb(n, k)
    plug = 1 - (1 - c / n) ** k
    return dict(n=n, c=c, k=k, subset=subset, plug=plug)


def coverage():
    """Brown et al. 2024 via MLSysIM (Literature.RepeatedSamplingMajority/Coverage)."""
    from mlsysim import Literature
    maj = Literature.RepeatedSamplingMajority.entries
    cov = Literature.RepeatedSamplingCoverage.entries

    def at(rows, model, dataset, k, field):
        return next(r[field] for r in rows if r["model"] == model and r["dataset"] == dataset and r["k"] == k)
    s, l = "Llama-3-8B-Instruct", "Llama-3-70B-Instruct"
    rep, rset = "DeepSeek-Coder-V2-Instruct + Moatless Tools", "SWE-bench Lite (all 300)"
    return dict(
        cov_small=at(maj, s, "MATH", 10_000, "coverage"),
        cov_large=at(maj, l, "MATH", 10_000, "coverage"),
        vote_small=at(maj, s, "MATH", 10_000, "majority_vote_accuracy"),
        vote_large=at(maj, l, "MATH", 10_000, "majority_vote_accuracy"),
        cov_small_1=at(maj, s, "MATH", 1, "coverage"),
        cov_large_1=at(maj, l, "MATH", 1, "coverage"),
        vote_gain_small=at(maj, s, "MATH", 10_000, "majority_vote_accuracy") - at(maj, s, "MATH", 256, "majority_vote_accuracy"),
        cov_gain_small=at(maj, s, "MATH", 10_000, "coverage") - at(maj, s, "MATH", 256, "coverage"),
        repair_1=at(cov, rep, rset, 1, "coverage"),
        repair_250=at(cov, rep, rset, 250, "coverage"),
    )


def precision(v, fr, fa):
    return v * (1 - fr) / (v * (1 - fr) + (1 - v) * fa)


def verifier():
    """NM 3.4: 64 candidates, 5 percent valid, FR 10 percent, FA 2 percent."""
    n, v, fr, fa, loss, out_tok = 64, 0.05, 0.10, 0.02, 500.0, 1024
    prec = precision(v, fr, fa)
    p_pass = v * (1 - fr) + (1 - v) * fa
    p_any = 1 - (1 - p_pass) ** n
    defect_sel = 1 - prec
    return dict(p_pass=p_pass, valid_pass=n * v * (1 - fr), invalid_pass=n * (1 - v) * fa,
                prec=prec, defect_sel=defect_sel, loss_sel=loss * defect_sel,
                p_any=p_any, abstain=1 - p_any,
                defect_attempt=p_any * defect_sel, loss_attempt=p_any * defect_sel * loss,
                out_bill=n * out_tok * P_OUT,
                loss_ratio=p_any * defect_sel * loss / (n * out_tok * P_OUT),
                prec_half_fa=precision(v, fr, fa / 2), prec_double_v=precision(2 * v, fr, fa),
                false_pass_exposure=1 - (1 - (1 - v) * fa) ** n)


def expected_max_normal(n, steps=20001, lo=-8.0, hi=8.0):
    """Exact E[max of n standard normals] by numerical integration."""
    h = (hi - lo) / (steps - 1)
    total = 0.0
    for i in range(steps):
        x = lo + i * h
        cdf = 0.5 * (1 + math.erf(x / math.sqrt(2)))
        pdf = math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
        w = 0.5 if i in (0, steps - 1) else 1.0
        total += w * x * n * cdf ** (n - 1) * pdf
    return total * h


def selection():
    """TTC equation 4 and section 3.4.3: winner's scoring error; loophole in a pool."""
    exact = {n: expected_max_normal(n) for n in (1, 2, 4, 8, 16, 32, 64)}
    asym = {n: math.sqrt(2 * math.log(n)) for n in (2, 4, 8, 16, 32, 64)}
    loophole = 1 - (1 - 0.005) ** 64
    return dict(exact=exact, asym=asym, loophole=loophole)


def prm800k():
    """PRM800K README annotated line (github.com/openai/prm800k @ 7ecc794)."""
    total_ms = 278270            # "total_time" of the annotated line
    steps = 16                   # pre_generated_steps in the line
    first_error = 3              # the first step rated -1
    return dict(minutes=total_ms / 60000, steps=steps, first_error=first_error,
                after_error=steps - first_error)


def pruning():
    """TTC section 3.6.2: b = 5 children, r = 0.68 rejected, depth 3."""
    b, r, d = 5, 0.68, 3
    beff = b * (1 - r)
    unpruned = sum(b ** i for i in range(1, d + 1))
    pruned = sum(b * beff ** (i - 1) for i in range(1, d + 1))
    return dict(beff=beff, unpruned=unpruned, unpruned_leaves=b ** d,
                pruned=pruned, pruned_leaves=beff ** d, cut=1 - pruned / unpruned,
                flat=unpruned * (1 - r), critical=1 - 1 / b)


def revision():
    """NM 3.5: 12 nodes of 150 tokens, 3 need replacement."""
    nodes, per, bad = 12, 150, 3
    full, scoped = nodes * per, bad * per
    return dict(full=full, scoped=scoped, share=scoped / full, avoided=1 - scoped / full)


def beam():
    """NM 3.6: 16 flat candidates vs beam width 4, 3 extensions, 4 segments of 128."""
    prompt, seg, segs, streams, score_s, test_s = 4096, 128, 4, 8, 0.015, 0.5
    # Flat: 16 complete candidates
    n = 16
    flat = dict(out=n * seg * segs, inp=n * prompt, scorer=0)
    flat["time"] = math.ceil(n / streams) * seg * segs * T_TOK + math.ceil(n / 8) * test_s
    # Beam: stage sizes 3, 9, 12, 12 (3 extensions per retained prefix, keep 4)
    width, ext = 4, 3
    retained, stages = 1, []
    for _ in range(segs):
        stages.append(retained * ext)
        retained = min(width, retained * ext)
    out = sum(stages) * seg
    inp = sum(m * (prompt + i * seg) for i, m in enumerate(stages))
    scorer = sum(m * (i + 1) * seg for i, m in enumerate(stages))
    waves = sum(math.ceil(m / streams) for m in stages)
    bm = dict(out=out, inp=inp, scorer=scorer,
              time=waves * seg * T_TOK + sum(stages) * score_s + math.ceil(retained / 8) * test_s)
    for d in (flat, bm):
        d["bill"] = d["inp"] * P_IN + d["out"] * P_OUT + d["scorer"] * P_SCORER
    # Input price fraction below which the beam's bill is lower
    fixed_flat = flat["out"] * P_OUT
    fixed_beam = bm["out"] * P_OUT + bm["scorer"] * P_SCORER
    breakeven = (fixed_flat - fixed_beam) / ((bm["inp"] - flat["inp"]) * P_IN)
    return dict(stages=stages, flat=flat, beam=bm, waves=waves,
                out_saved=1 - bm["out"] / flat["out"],
                out_saved_usd=(flat["out"] - bm["out"]) * P_OUT,
                inp_extra_usd=(bm["inp"] - flat["inp"]) * P_IN,
                breakeven=breakeven, beam_prompt_inputs=sum(stages) * prompt)


def swe_runs():
    """Public SWE-bench Verified runs via MLSysIM (Literature.SweBenchTrajectories)."""
    from mlsysim import Literature
    rows = Literature.SweBenchTrajectories.entries
    out = []
    for sub in sorted({r["submission"] for r in rows}):
        rs = [r for r in rows if r["submission"] == sub]
        costs = [r["rep_cost_usd"] for r in rs]
        if any(c is None or c != c for c in costs):
            continue                      # this run records no cost
        total = sum(costs)
        resolved = sum(1 for r in rs if r["resolved"])
        inp = sum(r["rep_input_tokens_sum"] for r in rs)
        outp = sum(r["rep_output_tokens_sum"] for r in rs)
        out.append(dict(submission=sub, year=sub[:4], n=len(rs), resolved=resolved,
                        rate=resolved / len(rs), per_attempt=total / len(rs),
                        per_resolved=total / resolved,
                        unresolved_share=sum(r["rep_cost_usd"] for r in rs if not r["resolved"]) / total,
                        in_out=inp / outp,
                        median_calls=statistics.median(r["n_model_calls"] for r in rs)))
    return out


def policies():
    """NM 3.7: three policies under a 75 s deadline, p = 0.40, revision 0.35."""
    p, q, prompt, out, report, test_s, deadline = 0.40, 0.35, 2048, 1000, 340, 10.0, 75.0
    rnd = out * T_TOK + test_s
    a_bill = prompt * P_IN + out * P_OUT
    b_bill = 8 * a_bill
    b_pass = pass_at(p, 8)
    rev_prompt = prompt + out + report
    c_pass = p + (1 - p) * q
    c_rounds = 1 + (1 - p)
    c_prompt = prompt + (1 - p) * rev_prompt
    c_bill = c_prompt * P_IN + c_rounds * out * P_OUT
    res = dict(round=rnd, three_rounds=3 * rnd, rounds_fit=int(deadline // rnd),
               A=dict(pass_=p, bill=a_bill, per_acc=a_bill / p, time=rnd, out_per_acc=out / p),
               B=dict(pass_=b_pass, bill=b_bill, per_acc=b_bill / b_pass, time=rnd, out_per_acc=8 * out / b_pass),
               C=dict(pass_=c_pass, bill=c_bill, per_acc=c_bill / c_pass, time=c_rounds * rnd,
                      time_max=2 * rnd, out_per_acc=c_rounds * out / c_pass, prompt=c_prompt))
    res["B_vs_A"] = res["B"]["per_acc"] / res["A"]["per_acc"]
    return res


def reproducer_trace():
    """The real run of pool_reproducer.py saved in pool_trace.txt."""
    text = (Path(__file__).parent / "pool_trace.txt").read_text()
    blocked = [float(line.split()[2]) for line in text.splitlines() if line.startswith("B blocked")]
    wall = [float(line.split(",")[1].split()[0]) for line in text.splitlines() if line.startswith("probe")]
    return dict(blocked_bug=blocked[0], blocked_t60=blocked[1], blocked_fix=blocked[2],
                wall_bug=wall[0], wall_t60=wall[1], wall_fix=wall[2],
                explanation_vs_probe=hook()["gen_s"] / (wall[0] / 1000))


if __name__ == "__main__":
    h = hook()
    print("== Hook (NM 3.1)")
    print(f"explanation time: {h['gen_s']:.2f} s; ratio to 40 ms test: {h['ratio']:.1f}x; bill ${h['bill']:.5f}")

    a = axes()
    print("== Three axes (NM 3.2)")
    for k in ("depth", "breadth", "feedback"):
        d = a[k]
        print(f"{k}: input {d['inp']:,}, output {d['out']:,}, time {d['time']:.2f} s, bill ${d['bill']:.5f}")
    print(f"breadth premium: {a['premium']:,} tokens; at 8x context {a['premium_8x']:,} tokens = ${a['premium_8x_usd']:.4f}")
    print(f"breadth bill / depth bill: {a['breadth_vs_depth_bill']:.2f}x; breadth one stream at a time: {a['serial_breadth_s']:.2f} s")

    pl = pool()
    print("== Pool (NM 3.3)")
    print("pass@k: " + ", ".join(f"{k}: {v * 100:.1f}%" for k, v in pl["avail"].items()))
    print(f"8th candidate adds {pl['last_gain'] * 100:.1f} points")
    print(f"parallel per accepted {pl['par_per_acc']:.2f}; sequential attempts {pl['seq_attempts']:.2f}, per accepted {pl['seq_per_acc']:.2f}; ratio {pl['ratio']:.2f}x")
    print(f"mixed workload: plug-in {pl['plug'] * 100:.1f}%, true {pl['true'] * 100:.1f}%; easy half {pass_at(0.6, 8) * 100:.1f}%")

    e = estimator()
    print("== Estimator (Exercise 2d)")
    print(f"n={e['n']} c={e['c']} k={e['k']}: subset {e['subset']:.3f}, plug-in {e['plug']:.3f}")

    c = coverage()
    print("== Coverage vs vote (Brown et al. 2024, MLSysIM)")
    print(f"8B: coverage@1 {c['cov_small_1'] * 100:.0f}%, @10k {c['cov_small'] * 100:.0f}%, vote@10k {c['vote_small'] * 100:.0f}%")
    print(f"70B: coverage@1 {c['cov_large_1'] * 100:.0f}%, @10k {c['cov_large'] * 100:.0f}%, vote@10k {c['vote_large'] * 100:.0f}%")
    print(f"8B after 256: vote +{c['vote_gain_small'] * 100:.1f} pts, coverage +{c['cov_gain_small'] * 100:.1f} pts")
    print(f"repair coverage: 1 sample {c['repair_1'] * 100:.0f}%, 250 samples {c['repair_250'] * 100:.0f}%")

    v = verifier()
    print("== Verifier precision (NM 3.4)")
    print(f"pass probability per candidate {v['p_pass']:.3f}")
    print(f"expected passes: valid {v['valid_pass']:.2f}, invalid {v['invalid_pass']:.3f}")
    print(f"precision {v['prec'] * 100:.2f}%; defective selection {v['defect_sel'] * 100:.2f}%; loss per selection ${v['loss_sel']:.2f}")
    print(f"any selection {v['p_any'] * 100:.2f}%; abstain {v['abstain'] * 100:.2f}%")
    print(f"defective per attempt {v['defect_attempt'] * 100:.2f}%; loss per attempt ${v['loss_attempt']:.2f}; output bill ${v['out_bill']:.2f}; ratio {v['loss_ratio']:.0f}x")
    print(f"precision with FA halved {v['prec_half_fa'] * 100:.1f}%; with valid doubled {v['prec_double_v'] * 100:.1f}%")
    print(f"false-pass exposure in the batch {v['false_pass_exposure'] * 100:.2f}%")

    s = selection()
    print("== Selection pressure (equation 4)")
    print("exact E[max]: " + ", ".join(f"{n}: {x:.2f}" for n, x in s["exact"].items()))
    print("sqrt(2 ln N): " + ", ".join(f"{n}: {x:.2f}" for n, x in s["asym"].items()))
    print(f"0.5% loophole in a pool of 64: {s['loophole'] * 100:.1f}%")

    r = prm800k()
    print("== PRM800K annotated line")
    print(f"labeling time {r['minutes']:.1f} min; first error at step {r['first_error']} of {r['steps']}; {r['after_error']} steps follow it")

    pr = pruning()
    print("== Pruning (section 3.6.2)")
    print(f"b_eff {pr['beff']:.1f}; critical rejection {pr['critical'] * 100:.0f}%")
    print(f"unpruned {pr['unpruned']} nodes, {pr['unpruned_leaves']} leaves; pruned {pr['pruned']:.1f} nodes, {pr['pruned_leaves']:.1f} leaves; cut {pr['cut'] * 100:.1f}%; flat 68% cut {pr['flat']:.1f}")

    rv = revision()
    print("== Scoped revision (NM 3.5)")
    print(f"full {rv['full']:,} tokens, scoped {rv['scoped']} tokens, {rv['share'] * 100:.0f}% of output, avoids {rv['avoided'] * 100:.0f}%")

    b = beam()
    print("== Flat vs beam (NM 3.6)")
    print(f"stages {b['stages']}, generation waves {b['waves']}")
    for k in ("flat", "beam"):
        d = b[k]
        print(f"{k}: output {d['out']:,}, input {d['inp']:,}, scorer {d['scorer']:,}, time {d['time']:.2f} s, bill ${d['bill']:.4f}")
    print(f"beam output saved {b['out_saved'] * 100:.2f}% = ${b['out_saved_usd']:.3f}; extra input ${b['inp_extra_usd']:.3f}; breakeven input price {b['breakeven'] * 100:.1f}% of list")
    print(f"beam common-prompt input alone: {b['beam_prompt_inputs']:,} tokens")

    print("== Public repair-agent runs (SWE-bench Verified, MLSysIM)")
    for w in swe_runs():
        print(f"{w['submission']}: n {w['n']}, resolved {w['rate'] * 100:.0f}%, ${w['per_attempt']:.2f} per attempt, ${w['per_resolved']:.2f} per resolved, {w['unresolved_share'] * 100:.0f}% of spend on unresolved, input/output {w['in_out']:.0f}, median calls {w['median_calls']:.0f}")

    po = policies()
    print("== Three policies (NM 3.7)")
    print(f"round {po['round']:.2f} s; three rounds {po['three_rounds']:.2f} s; rounds that fit 75 s: {po['rounds_fit']}")
    for k in "ABC":
        d = po[k]
        print(f"{k}: pass {d['pass_'] * 100:.2f}%, bill ${d['bill']:.5f}, per accepted ${d['per_acc']:.4f}, time {d['time']:.2f} s, output per accepted {d['out_per_acc']:,.0f}")
    print(f"C expected prompt tokens {po['C']['prompt']:.1f}; C max time {po['C']['time_max']:.2f} s; B per accepted / A {po['B_vs_A']:.1f}x")

    t = reproducer_trace()
    print("== Real reproducer run (pool_trace.txt)")
    print(f"without notify: B blocked {t['blocked_bug']:.1f} ms after the return, probe {t['wall_bug']:.1f} ms")
    print(f"timeout raised to 60 ms: B blocked {t['blocked_t60']:.1f} ms, probe {t['wall_t60']:.1f} ms")
    print(f"with notify: B blocked {t['blocked_fix']:.1f} ms, probe {t['wall_fix']:.1f} ms")
    print(f"explanation 21.58 s / measured probe: {t['explanation_vs_probe']:.0f}x")
