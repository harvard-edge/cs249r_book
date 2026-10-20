"""Lecture 20 (Conclusion): every derived number on the slides and in the notes.

Run: python3 calcs.py   (prints each value with the label the slide uses).
Inputs are typed with their source; only derived values are computed here.
"Ch N" is the rendered Volume III chapter (dev text, read 2026-10-19).
make_figs.py imports these functions, so the figures and the slides agree.
"""

import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- recall (L19)
def recall_l19():
    """NM 19.1, Ch 19 §19.2, table 1: 100 tickets per configuration."""
    tickets = 100
    acc_frontier, acc_cheap = 75, 25           # accepted tickets (table 1)
    cpa_frontier, cpa_cheap = 0.55, 1.32       # $ per accepted repair (NM 19.1)
    return dict(
        rej_per_acc_frontier=(tickets - acc_frontier) / acc_frontier,
        rej_per_acc_cheap=(tickets - acc_cheap) / acc_cheap,
        cpa_ratio=cpa_cheap / cpa_frontier,
        cpa_frontier=cpa_frontier, cpa_cheap=cpa_cheap)


# ---------------------------------------------------------------- the lost reply
P_REQ, P_REP = 0.001, 0.004      # Exercise 3, Ch 20 §20.13
CALLS = 25_000                   # mutating calls per day (Exercise 3)
DUP_COST, LOOKUP_COST = 2_000, 0.01   # $ per duplicated change, $ per status lookup


def lost_reply():
    """Exercise 3: timeouts, duplicates under a fresh-identity blind retry."""
    p_timeout = P_REQ + (1 - P_REQ) * P_REP
    p_dup = (1 - P_REQ) * P_REP * (1 - P_REQ)   # settlement relation, table 8
    timeouts = CALLS * p_timeout
    dups = CALLS * p_dup
    lookups = timeouts * LOOKUP_COST
    lost_req_share = P_REQ / p_timeout
    false_closures = CALLS * P_REQ             # assume success after a lost request
    return dict(timeouts=timeouts, dups=dups, dup_cost=dups * DUP_COST,
                lookup_cost=lookups, lost_req_share=lost_req_share, false_closures=false_closures,
                p_dup=p_dup)


def measured_run():
    """images/data/settle_run.json: our localhost run of the four runtimes."""
    return json.loads((HERE / "data" / "settle_run.json").read_text())


# ---------------------------------------------------------------- the incident's bill
def incident_bill():
    """Ch 20 §20.3.2: one incident_42 attempt (remediation profile, Lighthouse 19.1)."""
    model, tools, health, approval = 0.15, 0.10, 0.20, 20.00
    acc = 0.70
    attempt = model + tools + health + approval
    p_wrong, consequence = 0.04, 12_000          # §20.3.2, eq. 12.5 with no check
    loss = p_wrong * consequence
    # Doubling the model calls (§20.3.2): break-even acceptance at the new cost
    attempt2 = attempt + model
    acc_needed = acc * attempt2 / attempt
    return dict(attempt=attempt, cpa=attempt / acc, model_share=model / attempt,
                approval_share=approval / attempt, loss=loss,
                loss_over_attempt=loss / attempt, attempt2=attempt2,
                attempt_rise=attempt2 / attempt - 1, acc_needed=acc_needed,
                parts=dict(model=model, tools=tools, health=health, approval=approval))


# ---------------------------------------------------------------- verification
def passing_set():
    """§20.10 pitfall, eq. 3.3: 30% valid, FR 10%, FA 5%."""
    valid, fr, fa = 0.30, 0.10, 0.05
    pass_valid = valid * (1 - fr)
    pass_invalid = (1 - valid) * fa
    precision = pass_valid / (pass_valid + pass_invalid)
    return dict(invalid_before=1 - valid, invalid_after=1 - precision,
                precision=precision, pass_rate=pass_valid + pass_invalid)


def swebench_plus():
    """Aleithan et al. 2024 (SWE-bench+), arXiv 2410.06992v2, Table 1:
    the 251 passing patches of SWE-Agent + GPT-4 on SWE-bench."""
    leak, incorrect, other_files, incomplete = 82, 32, 9, 37
    different_ok, more_complete = 76, 15
    total = leak + incorrect + other_files + incomplete + different_ok + more_complete
    weak = incorrect + other_files + incomplete
    correct = different_ok + more_complete
    return dict(total=total, leak=leak, weak=weak, correct=correct,
                leak_share=leak / total, weak_share=weak / total,
                correct_share=correct / total, suspicious_share=(leak + weak) / total)


def swe_agent():
    """Yang et al. 2024 (SWE-agent), arXiv 2405.15793v3, SWE-bench Lite,
    GPT-4 Turbo unless named: Table 1 and Table 3 (percent resolved)."""
    rows = [("SWE-agent interface", 18.0), ("Viewer shows whole file", 12.7),
            ("Iterative search", 12.0), ("Shell only", 11.0), ("No edit command", 10.3)]
    swap = ("Same interface, Claude 3 Opus", 13.0)
    return dict(rows=rows, swap=swap, n=300,
                resolved_full=round(18.0 / 100 * 300), shell_rel=18.0 / 11.0 - 1)


# ---------------------------------------------------------------- fallacies, limits
def fallacy_capable():
    """§20.10: 0.1% per step, 50 turns, 10,000 trajectories a day."""
    eps, H, daily = 0.001, 50, 10_000
    p_any = 1 - (1 - eps) ** H
    return dict(p_any=p_any, per_day=p_any * daily)


def window_halving():
    """§20.10: halving the window halves the samples; CI widens by sqrt(2)."""
    return dict(widen=math.sqrt(2))


def tasks_per_arm():
    """eq. 13.3: n ∝ 1/Δ²; halving Δ quadruples n; doubling n shrinks Δ by sqrt(2)."""
    return dict(halve_delta=(1 / 0.5) ** 2, double_n_shrink=1 / math.sqrt(2))


def attribution():
    """§20.8: 4 candidate actions per step, 50 steps; one checked 5-step window,
    1.2 s per replayed check."""
    b, H, w, sec = 4, 50, 5, 1.2
    full = b ** H
    window = b ** w
    return dict(full=full, full_exp=math.log10(full), window=window,
                minutes=window * sec / 60)


def amdahl_spec():
    """§20.9: half the effort stays human specification and review."""
    serial = 0.5
    return dict(cap=1 / serial)


# ---------------------------------------------------------------- the two relations
def horizon():
    """eq. 1.1 at 99% per step (NM 1.1, §20.11 figure)."""
    p = 0.99
    return dict(p10=p ** 10, p100=p ** 100, curve=[(h, p ** h) for h in range(0, 101)])


def freeze():
    """Ch 10 §10.10: an instruction honored on 99% of turns, over 30 turns."""
    p30 = 0.99 ** 30
    return dict(p30=p30, violate=1 - p30)


def cost_bars():
    """NM 1.2 (Ch 1 §1.6): $2.10 attempt, 40% acceptance; tokens 62% of the attempt."""
    attempt, acc, token_share = 2.10, 0.40, 0.62
    base = attempt / acc
    # The chapter rounds the cheaper attempt to $1.45 before dividing
    # (Ch 1 NM 1.2: "lowers the attempt to $1.45, giving 1.45 / 0.40 = 3.63").
    cheap = round(attempt * (1 - 0.5 * token_share), 2)
    half_price = float(Decimal(str(cheap)) / Decimal("0.40"))
    better_acc = attempt / 0.60
    return dict(base=base, half_price=half_price, better_acc=better_acc,
                cheap_attempt=attempt * (1 - 0.5 * token_share))


def metr():
    """METR Time Horizon 1.1 (benchmark_results_1_1.yaml): p50 and p80 horizons
    (minutes) of models METR marks state of the art, released 2024 or later."""
    d = yaml.safe_load((HERE / "data" / "metr_benchmark_results_1_1.yaml").read_text())
    rows = []
    for name, v in d["results"].items():
        m = v["metrics"]
        if m.get("is_sota") and str(v["release_date"]) >= "2024-01-01":
            rows.append((str(v["release_date"]), name,
                         m["p50_horizon_length"]["estimate"],
                         m["p80_horizon_length"]["estimate"]))
    rows.sort()
    ratios = sorted(r[2] / r[3] for r in rows)
    mid = len(ratios) // 2
    median = ratios[mid] if len(ratios) % 2 else (ratios[mid - 1] + ratios[mid]) / 2
    return dict(rows=rows, ratio_min=ratios[0], ratio_max=ratios[-1], ratio_median=median)


if __name__ == "__main__":
    r = recall_l19()
    print("[Recall L19] rejected per accepted: frontier", f"{r['rej_per_acc_frontier']:.2f}",
          " cheap", f"{r['rej_per_acc_cheap']:.0f}", " cost ratio", f"{r['cpa_ratio']:.1f}x")
    lr = lost_reply()
    print("[Estimate ex3] timeouts/day", f"{lr['timeouts']:.1f}", " duplicates/day", f"{lr['dups']:.1f}",
          " $/day", f"{lr['dup_cost']:,.0f}", " lookups $/day", f"{lr['lookup_cost']:.2f}",
          " share of timeouts with request lost", f"{lr['lost_req_share']:.1%}",
          " expected false closures", f"{lr['false_closures']:.0f}")
    try:
        m = measured_run()
        for k, v in m["runtimes"].items():
            print(f"[Measured] {k:7s}", {kk: v[kk] for kk in ('lost_replies_seen', 'duplicates',
                  'false_closures', 'unowned_unknowns', 'applied_but_unknown')})
    except FileNotFoundError:
        print("[Measured] run data/settle_run.py first")
    b = incident_bill()
    print("[Bill] attempt", f"${b['attempt']:.2f}", " per accepted", f"${b['cpa']:.2f}",
          " model share", f"{b['model_share']:.1%}", " approval share", f"{b['approval_share']:.1%}")
    print("[Bill] expected irreversible loss", f"${b['loss']:.0f}", "=", f"{b['loss_over_attempt']:.1f}x attempt")
    print("[Estimate] doubled model calls: attempt", f"${b['attempt2']:.2f}", f"(+{b['attempt_rise']:.2%})",
          " acceptance needed", f"{b['acc_needed']:.4f}")
    ps = passing_set()
    print("[Predict] invalid before", f"{ps['invalid_before']:.0%}", " invalid in passing set",
          f"{ps['invalid_after']:.1%}", " precision", f"{ps['precision']:.1%}", " pass rate", f"{ps['pass_rate']:.1%}")
    s = swebench_plus()
    print("[SWE-bench+] total", s["total"], " leak", s["leak"], f"({s['leak_share']:.2%})",
          " weak tests", s["weak"], f"({s['weak_share']:.2%})", " correct", s["correct"],
          f"({s['correct_share']:.2%})", " suspicious", f"{s['suspicious_share']:.2%}")
    sa = swe_agent()
    print("[SWE-agent] 18.0% of 300 =", sa["resolved_full"], " relative gain over shell only",
          f"{sa['shell_rel']:.0%}")
    f = fallacy_capable()
    print("[Fallacy] 1-(0.999)^50 =", f"{f['p_any']:.2%}", " per day of 10,000:", f"{f['per_day']:.0f}")
    print("[Pitfall] halving the window widens the interval by", f"{window_halving()['widen']:.2f}x")
    t = tasks_per_arm()
    print("[Release] halving the detectable difference: x", f"{t['halve_delta']:.0f}", "tasks;",
          " doubling tasks shrinks it to", f"{t['double_n_shrink']:.2f}", "of before")
    a = attribution()
    print("[Attribution] 4^50 =", f"{a['full']:.3e}", f"(10^{a['full_exp']:.1f})", " 4^5 =", a["window"],
          " replay minutes", f"{a['minutes']:.1f}")
    print("[Spec boundary] speedup cap", f"{amdahl_spec()['cap']:.0f}x")
    h = horizon()
    print("[Horizon] 0.99^10 =", f"{h['p10']:.1%}", " 0.99^100 =", f"{h['p100']:.1%}")
    fz = freeze()
    print("[Part IV] 0.99^30 =", f"{fz['p30']:.0%}", " violate at least once", f"{fz['violate']:.0%}")
    c = cost_bars()
    print("[Cost] base", f"${c['base']:.2f}", " half token price", f"${Decimal(str(c['half_price'])).quantize(Decimal('0.01'), ROUND_HALF_UP)}",
          f"(attempt ${c['cheap_attempt']:.2f})", " 60% acceptance", f"${c['better_acc']:.2f}")
    me = metr()
    print("[METR] SOTA models since 2024:", len(me["rows"]), " p50/p80 ratio min", f"{me['ratio_min']:.1f}",
          " median", f"{me['ratio_median']:.1f}", " max", f"{me['ratio_max']:.1f}")
    for row in me["rows"]:
        print("   ", row[0], row[1], f"p50 {row[2]:.1f} min", f"p80 {row[3]:.1f} min", f"ratio {row[2]/row[3]:.1f}")
