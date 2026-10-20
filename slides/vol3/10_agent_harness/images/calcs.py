#!/usr/bin/env python3
"""Every derived number in Lecture 10 (The Agent Harness), one block per slide.

Inputs are typed with their source. Sources:
  AH    = the rendered chapter, The Agent Harness (Volume III), dev text 2026-10-18.
  NM    = a Napkin Math notebook of that chapter (scenario values the chapter
          labels hypothetical; slides say "illustrative").
  MLSysIM Literature.SweBenchTrajectories = 2,100 public SWE-bench Verified
          trajectories of five agent configurations (calls, outcome, cost as
          each agent recorded it); Literature.OSWorldStepBudget = OSWorld
          success at 15, 50, 100 steps (Xie et al. 2024, via Epoch AI).
  data/ = files measured or fetched for this deck (scripts beside them).

Run from this folder:
  /Users/.../MLSysBook/.venv/bin/python calcs.py   (prints "label: value")
make_figs.py imports these functions so figures and slides agree.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Literature  # noqa: E402


def show(label, value):
    print(f"  {label}: {value}")


# ------------------------------------------------- horizon compounding
# AH Sec. 10.10 (fallacy, equation 1.1): a rule honored with probability
# 0.99 on each turn, over a 30-turn trajectory.
P_TURN = 0.99
H_RUN = 30


def compounding(p=P_TURN, h=H_RUN):
    return p ** h


def real_horizons():
    """Median and 90th-percentile model calls per trajectory, per config."""
    runs = Literature.SweBenchTrajectories.entries
    out = {}
    for s in sorted({r["submission"] for r in runs}):
        g = [r for r in runs if r["submission"] == s]
        c = np.array([r["n_model_calls"] for r in g])
        out[s] = dict(label=f"{g[0]['scaffold']}, {g[0]['model']}", n=len(g),
                      median=float(np.median(c)), p90=float(np.quantile(c, 0.9)),
                      hold_at_median=P_TURN ** float(np.median(c)))
    return out


# ------------------------------------------- turn-boundary control
# AH Sec. 10.5.1: output allowance 4,096 tokens at 21.075 ms per token (the
# chapter's single-accelerator, single-request reference floor).
K_MAX = 4096
T_DEC_MS = 21.075


def boundary_delay_s(k=K_MAX, t_ms=T_DEC_MS):
    return k * t_ms / 1000


def measured_pause():
    """data/pause_latency.json, written by data/pause_latency.py."""
    d = json.loads((DATA / "pause_latency.json").read_text())
    rows = d["rows"]
    ks = sorted({r["k"] for r in rows})
    out = {"meta": d["meta"], "k": ks, "end": {}, "cancel": {}, "next_first": [],
           "ms_per_token": []}
    for k in ks:
        e = [r["pause_effect_s"] for r in rows if r["k"] == k and r["policy"] == "end-check"]
        c = [r["pause_effect_s"] for r in rows if r["k"] == k and r["policy"] == "cancel"]
        out["end"][k] = (min(e), float(np.mean(e)), max(e))
        out["cancel"][k] = (min(c), float(np.mean(c)), max(c))
    for r in rows:
        if r["policy"] == "end-check" and r.get("eval_duration_s"):
            out["ms_per_token"].append(1000 * r["eval_duration_s"] / r["eval_count"])
        if r["policy"] == "cancel" and r.get("next_call_first_token_s") is not None:
            out["next_first"].append(r["next_call_first_token_s"])
    out["ms_per_token_mean"] = float(np.mean(out["ms_per_token"]))
    out["cancel_max_s"] = max(v[2] for v in out["cancel"].values())
    out["idle_first"] = d["meta"].get("idle_first_token_s", [])
    return out


# ------------------------------------------------- approval escrow
# AH Lighthouse 10.1 (scenario inputs): repair review requests 6 per hour,
# mean decision 2 h, workspace lease 512 MiB; remediation 2 per hour,
# 15-minute decision, 6 min diagnosis, 9 min apply and verify.
REPAIR_RATE_PER_H = 6
REPAIR_WAIT_H = 2
LEASE_MIB = 512
REMED_RATE_PER_H = 2
REMED_WAIT_MIN = 15
REMED_DIAG_MIN = 6
REMED_APPLY_MIN = 9


def escrow():
    held = REPAIR_RATE_PER_H * REPAIR_WAIT_H                      # Little's law
    mem_gib = held * LEASE_MIB / 1024
    remed_held = REMED_RATE_PER_H * REMED_WAIT_MIN / 60
    ttm = REMED_DIAG_MIN + REMED_WAIT_MIN + REMED_APPLY_MIN
    return dict(held=held, mem_gib=mem_gib, held_half_wait=held / 2,
                mem_half_wait_gib=mem_gib / 2, remed_held=remed_held,
                ttm_min=ttm, approval_share=REMED_WAIT_MIN / ttm)


# --------------------------------------- deficit round-robin (NM 10.1)
# AH NM 10.1: weights 3 (interactive) and 1 (maintenance), quantum 8,000
# charge tokens per unit weight per round; interactive call 2,000 input +
# 500 output allowance; maintenance 40,000 + 2,000.
QUANTUM = 8000
CLASSES = {"interactive": (3, 2000 + 500), "maintenance": (1, 40000 + 2000)}


def drr_formula():
    per_round = {c: w * QUANTUM / charge for c, (w, charge) in CLASSES.items()}
    tot_w = sum(w for w, _ in CLASSES.values())
    charge_share = {c: w / tot_w for c, (w, _) in CLASSES.items()}
    tot_calls = sum(per_round.values())
    call_share = {c: v / tot_calls for c, v in per_round.items()}
    return dict(per_round=per_round, charge_share=charge_share, call_share=call_share,
                ratio=per_round["interactive"] / per_round["maintenance"],
                rounds_per_maint=CLASSES["maintenance"][1] / QUANTUM,
                rounds_per_maint_20k=(20000 + 2000) / QUANTUM)


def drr_simulate(rounds=10000):
    """Run weighted deficit round-robin with both classes backlogged."""
    credit = {c: 0.0 for c in CLASSES}
    calls = {c: 0 for c in CLASSES}
    charge = {c: 0.0 for c in CLASSES}
    for _ in range(rounds):
        for c, (w, size) in CLASSES.items():
            credit[c] += w * QUANTUM
            while credit[c] >= size:
                credit[c] -= size
                calls[c] += 1
                charge[c] += size
    tc, tg = sum(calls.values()), sum(charge.values())
    return dict(calls=calls, call_share={c: calls[c] / tc for c in calls},
                charge_share={c: charge[c] / tg for c in charge})


# ------------------------------------ bounded-liability admission (NM 10.2)
# AH NM 10.2 (illustrative): allowance $3.50, settled $2.58, test-job
# reservation $0.50, finalization reserve $0.30; next call 50,000 input
# tokens, 4,096 output allowance, $3 and $15 per million tokens.
ALLOW = 3.50
SETTLED = 2.58
RESERVED = 0.50
FINAL_RESERVE = 0.30
IN_TOK = 50000
OUT_ALLOW = 4096
P_IN = 3.0 / 1e6
P_OUT = 15.0 / 1e6
P_CACHED = 0.30 / 1e6


def admission():
    liab_in = P_IN * IN_TOK
    liab_out = P_OUT * OUT_ALLOW
    liab = liab_in + liab_out
    spend_headroom = ALLOW - SETTLED
    true_headroom = ALLOW - SETTLED - RESERVED - FINAL_RESERVE
    max_input = (true_headroom - liab_out) / P_IN
    return dict(liab_in=liab_in, liab_out=liab_out, liab=liab,
                spend_headroom=spend_headroom, true_headroom=true_headroom,
                shortfall=true_headroom - liab, input_share=liab_in / liab,
                max_input_tokens=max_input, tokens_per_dime=0.10 / P_IN,
                reservation_share=RESERVED / ALLOW)


# ------------------------------------------ the meter (NM 10.3)
# AH NM 10.3 (illustrative): 30 turns done; allowances 40 turns, 2 million
# tokens, 900 s, $3.50; clock 460 s; base prompt 8,000, 1,500 appended per
# turn, 200 output per turn; $3 uncached, $0.30 cached, $15 output per M.
TURNS = 30
B_TURNS, B_TOKENS, B_TIME, B_DOLLARS = 40, 2_000_000, 900, 3.50
CLOCK = 460
BASE, APPEND, OUT_PER_TURN = 8000, 1500, 200


def meter():
    inputs = [BASE + APPEND * t for t in range(TURNS)]          # equation 2.9
    total_in = sum(inputs)
    total_out = OUT_PER_TURN * TURNS
    # with reuse, each turn's earlier prefix is cached: turn t's prompt minus
    # the newly appended tokens (the first turn has nothing cached)
    cached = sum(inputs[t - 1] for t in range(1, TURNS))
    uncached = total_in - cached
    cost_reuse = P_IN * uncached + P_CACHED * cached + P_OUT * total_out
    cost_none = P_IN * total_in + P_OUT * total_out
    used = dict(turns=TURNS / B_TURNS, tokens=(total_in + total_out) / B_TOKENS,
                time=CLOCK / B_TIME)
    return dict(total_in=total_in, total_tokens=total_in + total_out, cached=cached,
                uncached=uncached, cost_reuse=cost_reuse, cost_none=cost_none,
                ratio=cost_none / cost_reuse,
                used_reuse=dict(used, dollars=cost_reuse / B_DOLLARS),
                used_none=dict(used, dollars=cost_none / B_DOLLARS))


# ------------------------------ a real harness overshoots its cost limit
# data/trajectory_django16263.json (fetch_trajectory.py): mini-SWE-agent
# 2.1.0, Claude Opus 4.6, SWE-bench Verified django__django-16263,
# cost_limit 3.0, step_limit 250. The limit check runs before each call
# against settled cost only (minisweagent/agents/default.py, v2.1.0, l. 105).
def overshoot():
    d = json.loads((DATA / "trajectory_django16263.json").read_text())
    cum = d["cumulative_cost"]
    limit = d["limits"]["cost_limit"]
    runs = [r for r in Literature.SweBenchTrajectories.entries
            if r["submission"] == "20260217_mini-v2.0.0_claude-4-6-opus"]
    over = sorted(r["rep_cost_usd"] for r in runs if r["rep_cost_usd"] > limit)
    return dict(limit=limit, calls=len(cum), before_last=cum[-2], final=cum[-1],
                last_call=d["call_costs"][-1], over_by=cum[-1] - limit,
                over_pct=(cum[-1] - limit) / limit, n_runs=len(runs), n_over=len(over),
                max_over=max(over) - limit, mean_over=float(np.mean(over)) - limit,
                cumulative=cum, n_messages=d["n_messages"],
                step_limit=d["limits"]["step_limit"])


# ------------------------------------------- choosing a ceiling
# AH Sec. 10.9.5 and figures 5 and 6: a call ceiling at the accepted runs'
# 90th percentile, applied post hoc to five public configurations.
Q = 0.90


def ceiling_evidence():
    runs = Literature.SweBenchTrajectories.entries
    rows = []
    for s in sorted({r["submission"] for r in runs}):
        g = [r for r in runs if r["submission"] == s]
        ok = np.array([r["n_model_calls"] for r in g if r["resolved"]])
        bad = np.array([r["n_model_calls"] for r in g if not r["resolved"]])
        cap = float(np.quantile(ok, Q))
        costs = [r["rep_cost_usd"] for r in g if r["rep_cost_usd"]]
        rows.append(dict(sub=s, scaffold=g[0]["scaffold"], model=g[0]["model"],
                         n=len(g), accepted=len(ok), cap=cap,
                         fail_halt=float((bad > cap).mean()),
                         ok_halt=float((ok > cap).mean()),
                         med_ok=float(np.median(ok)), med_bad=float(np.median(bad)),
                         ok_calls=np.sort(ok), bad_calls=np.sort(bad),
                         cost_per_accepted=(sum(costs) / len(ok)) if costs else None,
                         mean_cost=(sum(costs) / len(g)) if costs else None))
    return rows


def osworld():
    by = {}
    for r in Literature.OSWorldStepBudget.entries:
        by.setdefault(r["model"], {})[int(r["steps"])] = r["score_pct"]
    flat = [m for m, v in by.items() if v[100] <= v[50]]
    gains = {m: v[100] - v[15] for m, v in by.items()}
    return dict(by=by, n_flat=len(flat), best_gain=max(gains.values()),
                best=max(gains, key=gains.get))


# ---------------------------------------------------------------- print
def main():
    print("Hook and continuation (AH Sec. 10.10):")
    show("0.99^30 (rule holds on all 30 turns)", f"{compounding():.4f}")
    show("runs that break it at least once", f"{1 - compounding():.4f}")
    for s, v in real_horizons().items():
        show(f"median calls {v['label']}", f"{v['median']:.0f} (p90 {v['p90']:.0f}); "
             f"0.99^median = {v['hold_at_median']:.3f}")

    print("Turn-boundary control (AH Sec. 10.5.1):")
    show("4,096 x 21.075 ms", f"{boundary_delay_s():.1f} s")
    try:
        m = measured_pause()
        show("measured ms per output token (end-check runs)", f"{m['ms_per_token_mean']:.1f}")
        for k in m["k"]:
            e, c = m["end"][k], m["cancel"][k]
            show(f"K={k}: wait for reply (min/mean/max s)", f"{e[0]:.1f} / {e[1]:.1f} / {e[2]:.1f}")
            show(f"K={k}: cancel (max s)", f"{c[2]:.3f}")
        show("cancel max over all runs (s)", f"{m['cancel_max_s']:.3f}")
        show("first token after cancel (s, max)", f"{max(m['next_first']):.3f}")
        show("first token on idle server (s, max)", f"{max(m['idle_first']):.3f}" if m["idle_first"] else "n/a")
    except FileNotFoundError:
        show("measured pause", "run data/pause_latency.py first")

    print("Approval escrow (AH Lighthouse 10.1):")
    e = escrow()
    show("repair held trajectories (6/h x 2 h)", f"{e['held']:.0f}")
    show("repair retained memory", f"{e['mem_gib']:.0f} GiB")
    show("half the decision time", f"{e['held_half_wait']:.0f} held, {e['mem_half_wait_gib']:.0f} GiB")
    show("remediation held (2/h x 0.25 h)", f"{e['remed_held']:.1f}")
    show("remediation time to mitigate", f"{e['ttm_min']} min; approval share {e['approval_share']:.0%}")

    print("Fair scheduling (AH NM 10.1):")
    f = drr_formula()
    show("interactive calls per round", f"{f['per_round']['interactive']:.2f}")
    show("maintenance calls per round", f"{f['per_round']['maintenance']:.4f}")
    show("rounds per maintenance call", f"{f['rounds_per_maint']:.2f}")
    show("rounds per maintenance call at 20,000 input", f"{f['rounds_per_maint_20k']:.2f}")
    show("charge share interactive", f"{f['charge_share']['interactive']:.0%}")
    show("call share interactive", f"{f['call_share']['interactive']:.1%}")
    show("interactive : maintenance calls", f"{f['ratio']:.1f}")
    sim = drr_simulate()
    show("simulated 10,000 rounds: call share interactive", f"{sim['call_share']['interactive']:.1%}")
    show("simulated charge share interactive", f"{sim['charge_share']['interactive']:.1%}")

    print("Bounded-liability admission (AH NM 10.2):")
    a = admission()
    show("liability input", f"${a['liab_in']:.2f}")
    show("liability output", f"${a['liab_out']:.4f}")
    show("liability", f"${a['liab']:.4f}")
    show("spend-check headroom", f"${a['spend_headroom']:.2f}")
    show("headroom after reservations", f"${a['true_headroom']:.2f}")
    show("shortfall if admitted", f"${a['shortfall']:.2f}")
    show("input share of liability", f"{a['input_share']:.1%}")
    show("largest input that fits", f"{a['max_input_tokens']:,.0f} tokens")
    show("input tokens per $0.10 of unsettled liability", f"{a['tokens_per_dime']:,.0f}")
    show("test reservation as share of allowance", f"{a['reservation_share']:.1%}")

    print("Metering (AH NM 10.3):")
    mt = meter()
    show("total input tokens", f"{mt['total_in']:,}")
    show("input + output tokens", f"{mt['total_tokens']:,}")
    show("cached input tokens", f"{mt['cached']:,}")
    show("uncached input tokens", f"{mt['uncached']:,}")
    show("cost with reuse", f"${mt['cost_reuse']:.4f}")
    show("cost without reuse", f"${mt['cost_none']:.4f}")
    show("no-reuse / reuse", f"{mt['ratio']:.2f}")
    for k, v in mt["used_reuse"].items():
        show(f"used fraction with reuse, {k}", f"{v:.1%}")
    show("used fraction without reuse, dollars", f"{mt['used_none']['dollars']:.1%}")

    print("A real harness overshoots (data/trajectory_django16263.json):")
    o = overshoot()
    show("model calls", o["calls"])
    show("messages in the saved trajectory", o["n_messages"])
    show("cost before the last call", f"${o['before_last']:.4f}")
    show("last call", f"${o['last_call']:.4f}")
    show("final cost", f"${o['final']:.4f}")
    show("over the limit", f"${o['over_by']:.4f} ({o['over_pct']:.1%})")
    show("runs of 500 ending above the limit", f"{o['n_over']} of {o['n_runs']}")
    show("largest overshoot", f"${o['max_over']:.4f}")
    show("mean overshoot of those runs", f"${o['mean_over']:.4f}")

    print("Choosing a ceiling (AH Sec. 10.9.5):")
    rows = ceiling_evidence()
    for r in rows:
        cpa = f"${r['cost_per_accepted']:.2f}" if r["cost_per_accepted"] else "n/a"
        show(f"{r['scaffold']}, {r['model']}",
             f"p90 cap {r['cap']:.0f}; halts {r['fail_halt']:.0%} failed, {r['ok_halt']:.0%} accepted; "
             f"median calls {r['med_ok']:.0f} accepted vs {r['med_bad']:.0f} failed; "
             f"accepted {r['accepted']}/{r['n']}; cost per accepted task {cpa}")
    show("failed halted range", f"{min(r['fail_halt'] for r in rows):.0%} to {max(r['fail_halt'] for r in rows):.0%}")
    show("accepted halted max", f"{max(r['ok_halt'] for r in rows):.0%}")
    show("p90 cap range", f"{min(r['cap'] for r in rows):.0f} to {max(r['cap'] for r in rows):.0f}")
    ow = osworld()
    show("OSWorld models flat or lower from 50 to 100 steps", ow["n_flat"])
    show("largest gain 15 -> 100 steps", f"{ow['best_gain']:.1f} points ({ow['best']})")


if __name__ == "__main__":
    main()
