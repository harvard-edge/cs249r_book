#!/usr/bin/env python3
"""Every derived number in Lecture 7 (Tool Calling), one block per slide.

Inputs are typed with their source. Sources:
  TC   = the rendered chapter, Tool Calling (Volume III), dev text 2026-10-18.
  NM   = a Napkin Math notebook of that chapter (scenario values the chapter
         itself labels hypothetical; slides say "illustrative").
  data/ = files measured or copied for this deck (see data/README in the
         docstrings of collect_tools.py, obs_stats.py, retry_harness.py).

Run from this folder: python3 calcs.py   (prints "label: value" per slide)
make_figs.py imports the functions below so figures and slides agree.
"""

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def show(label, value):
    print(f"  {label}: {value}")


# ---------------------------------------------------------------- NM 7.1
# TC Sec. 7.5, Napkin Math 7.1: request and reply loss 2 percent each,
# one blind retry per timeout, non-idempotent call, no deduplication;
# fleet of 50 workers x 4,000 mutating calls per day.
P_REQ = 0.02
P_REP = 0.02
WORKERS = 50
CALLS_PER_WORKER = 4000


def p_timeout(p_req=P_REQ, p_rep=P_REP):
    return 1 - (1 - p_req) * (1 - p_rep)


def p_duplicate(p_req=P_REQ, p_rep=P_REP):
    # original delivered, its reply lost, the retry delivered
    return (1 - p_req) * p_rep * (1 - p_req)


def nm71():
    fleet = WORKERS * CALLS_PER_WORKER
    return dict(p_timeout=p_timeout(), p_dup=p_duplicate(), fleet_calls=fleet,
                dups_per_day=p_duplicate() * fleet)


# ---------------------------------------------- measured: retry harness
# data/retry_results.json, written by data/retry_harness.py: 20,000
# operations per setting over a real localhost HTTP socket, request loss
# 2 percent, reply loss 1, 2, 4 percent, one blind retry per timeout; the
# last setting adds endpoint deduplication by Idempotency-Key.
def harness():
    rows = json.loads((DATA / "retry_results.json").read_text())
    out = []
    for r in rows:
        pred = p_duplicate(r["p_req"], r["p_rep"]) * r["n"]
        if r["dedup"]:
            pred = 0.0
        out.append(dict(r, predicted=pred,
                        pred_timeouts=p_timeout(r["p_req"], r["p_rep"]) * r["n"]))
    return out


# ---------------------------------------------------------------- NM 7.2
# TC Sec. 7.6, Napkin Math 7.2: a 24 MB log, 3 bytes per token (assumed),
# 128,000-token window, shaped observation 4,000 tokens carried by 30 later
# calls, $3 per million uncached input tokens, cached at 10 percent.
LOG_BYTES = 24e6
BYTES_PER_TOKEN = 3
WINDOW = 128_000
SHAPED = 4_000
CARRY_CALLS = 30
PRICE_IN = 3.0 / 1e6
CACHED_SHARE_OF_PRICE = 0.10


def nm72():
    tokens = LOG_BYTES / BYTES_PER_TOKEN
    carried = SHAPED * CARRY_CALLS
    return dict(tokens=tokens, windows=tokens / WINDOW, shaped_share=SHAPED / WINDOW,
                carried=carried, cost_uncached=carried * PRICE_IN,
                cost_cached=carried * PRICE_IN * CACHED_SHARE_OF_PRICE)


# ------------------------------------------------- measured: real pytest log
# data/pytest_test_pool.py run with pytest 9.1.1 (20,001 tests, one fails),
# measured 2026-10-18: 1,601,040 bytes of -v output; 381,274 tokens under
# the cl100k_base tokenizer (tiktoken). Failure block is the last lines.
REAL_LOG_BYTES = 1_601_040
REAL_LOG_TOKENS = 381_274


def real_log():
    return dict(bytes_per_token=REAL_LOG_BYTES / REAL_LOG_TOKENS,
                windows=REAL_LOG_TOKENS / WINDOW)


# --------------------------------------- measured: 500 real repair runs
# data/swebench_observations.csv from data/obs_stats.py over the 500 public
# trajectories of mini-SWE-agent 1.0.0 (bash only) on SWE-bench Verified.
# The agent stages any output over 10,000 characters as the first 5,000,
# an "N characters elided" notice, and the last 5,000.
STAGE_CAP = 10_000


def swebench_obs():
    rows = list(csv.DictReader(open(DATA / "swebench_observations.csv")))
    raw = [int(r["raw_chars"]) for r in rows]
    tr = [r for r in rows if r["truncated"] == "1"]
    big = max(rows, key=lambda r: int(r["raw_chars"]))
    return dict(n=len(rows), truncated=len(tr), share=len(tr) / len(rows),
                max_raw=int(big["raw_chars"]), max_staged=int(big["staged_chars"]),
                over_cap=sum(1 for x in raw if x > STAGE_CAP),
                rc0_share=sum(1 for r in rows if r["returncode"] == "0") / len(rows),
                raw=raw)


RUNS_WITH_TRUNCATION = 253   # obs_stats.py output: runs with >= 1 elided observation
RUNS = 500


# ---------------------------------------------------------------- NM 7.3
# TC Sec. 7.8.3, Napkin Math 7.3: 300 s job, model polls every 5 s,
# 32,768-token prompt and 40 output tokens per poll, 95 percent cached at
# $0.30 per million, uncached $3, output $15 per million.
JOB_S = 300
POLL_S = 5
POLL_IN = 32_768
POLL_OUT = 40
CACHE_HIT = 0.95
PRICE_CACHED = 0.30 / 1e6
PRICE_OUT = 15.0 / 1e6


def poll_cost(poll_s=POLL_S, job_s=JOB_S):
    n = math.ceil(job_s / poll_s) - 1
    tin = n * POLL_IN
    cin = tin * (CACHE_HIT * PRICE_CACHED + (1 - CACHE_HIT) * PRICE_IN)
    cout = n * POLL_OUT * PRICE_OUT
    return dict(polls=n, tokens_in=tin, tokens_out=n * POLL_OUT, cost_in=cin,
                cost_out=cout, cost=cin + cout, worst_delay_s=poll_s)


# ---------------------------------------------------------------- NM 7.4
# TC Sec. 7.9.1, Napkin Math 7.4: 50 descriptors x 300 tokens; selected 6
# plus 400 tokens of selection overhead.
REG, SEL, DESC, OVER = 50, 6, 300, 400


def nm74():
    full = REG * DESC
    sel = SEL * DESC + OVER
    return dict(full=full, selected=sel, released=full - sel,
                breakeven_descriptors=OVER / DESC, removed_tokens=(REG - SEL) * DESC)


# ----------------------------------- measured: real GitHub MCP descriptors
# data/github_mcp_tools.json from data/collect_tools.py over the tool
# snapshots of github/github-mcp-server at commit e3809fe (2026-10-15);
# tokens: cl100k_base over {name, description, input_schema} compact JSON.
def github_tools():
    rows = json.loads((DATA / "github_mcp_tools.json").read_text())
    tok = sorted(r["tokens"] for r in rows)
    biggest = max(rows, key=lambda r: r["tokens"])
    return dict(n=len(rows), total=sum(tok), median=tok[len(tok) // 2],
                max=biggest["tokens"], max_name=biggest["name"],
                read_only=sum(r["read_only"] for r in rows),
                mutating=sum(not r["read_only"] for r in rows),
                idempotent=sum(r["idempotent"] for r in rows),
                destructive=sum(r["destructive"] for r in rows),
                repo_free=sum(r["repo_free_string"] for r in rows),
                window_share=sum(tok) / WINDOW, rows=rows)


# --------------------------------------------- adapters: N x M vs N + M
# TC Sec. 7.4.1 figure: 3 hosts, 4 tool services.
HOSTS, SERVICES = 3, 4


def adapters():
    return dict(pairwise=HOSTS * SERVICES, shared=HOSTS + SERVICES)


# ------------------------------------------------- Exercise 9 (discussion)
# TC Sec. 7.14, Exercise 9 (hypothetical givens): 12 structured tools of
# 350 tokens vs one run_shell of 500; 30 calls; $3 per million input;
# structured: attempt $1.20, acceptance 0.80, unconfined change 0.001;
# shell: acceptance 0.78, unconfined change 0.004; each change costs $2,000.
N_TOOLS, TOOL_TOK, SHELL_TOK, CALLS = 12, 350, 500, 30
ATTEMPT_STRUCT, ACC_STRUCT, P_CHANGE_STRUCT = 1.20, 0.80, 0.001
ACC_SHELL, P_CHANGE_SHELL, CHANGE_COST = 0.78, 0.004, 2000.0


def ex9():
    saved_tokens = (N_TOOLS * TOOL_TOK - SHELL_TOK) * CALLS
    saved = saved_tokens * PRICE_IN
    shell_attempt = ATTEMPT_STRUCT - saved
    c_struct = ATTEMPT_STRUCT / ACC_STRUCT
    c_shell = shell_attempt / ACC_SHELL
    c_struct_loss = (ATTEMPT_STRUCT + P_CHANGE_STRUCT * CHANGE_COST) / ACC_STRUCT
    c_shell_loss = (shell_attempt + P_CHANGE_SHELL * CHANGE_COST) / ACC_SHELL
    p_break = (c_struct_loss * ACC_SHELL - shell_attempt) / CHANGE_COST
    return dict(saved_tokens=saved_tokens, saved=saved, shell_attempt=shell_attempt,
                c_struct=c_struct, c_shell=c_shell, c_struct_loss=c_struct_loss,
                c_shell_loss=c_shell_loss, p_break=p_break)


# ------------------------------------------------- SWE-agent (published)
# Yang et al., NeurIPS 2024 (arXiv 2405.15793v3), SWE-bench Lite, GPT-4
# Turbo: Table 1 shell-only agent 11.00, SWE-agent 18.00 percent resolved;
# Table 3 file viewer 30 lines 14.3, 100 lines 18.0, full file 12.7; no
# edit command 10.3; full history 15.0 vs last 5 observations 18.0.
SWE = dict(shell_only=11.0, swe_agent=18.0, no_edit=10.3,
           viewer_30=14.3, viewer_100=18.0, viewer_full=12.7)


def swe():
    return dict(aci_over_shell=SWE["swe_agent"] / SWE["shell_only"],
                full_file_drop=SWE["viewer_100"] - SWE["viewer_full"])


if __name__ == "__main__":
    print("Hook / synthesis: no derived numbers (Invariant Labs, 26 May 2025)")

    print("Mediation, adapters (TC 7.4.1):")
    a = adapters()
    show("pairwise adapters", a["pairwise"]); show("with a shared protocol", a["shared"])

    print("Real descriptors (GitHub MCP server @ e3809fe):")
    g = github_tools()
    for k in ("n", "read_only", "mutating", "destructive", "idempotent", "repo_free",
              "total", "median", "max", "max_name"):
        show(k, g[k])
    show("share of a 128,000-token window", f"{g['window_share']:.1%}")

    print("NM 7.1 (retry):")
    m = nm71()
    show("P(timeout)", f"{m['p_timeout']:.2%}"); show("P(duplicate)", f"{m['p_dup']:.4%}")
    show("fleet calls/day", f"{m['fleet_calls']:,}"); show("duplicates/day", f"{m['dups_per_day']:,.0f}")

    print("Measured retry harness (20,000 operations each):")
    for r in harness():
        show(f"reply loss {r['p_rep']:.0%} dedup={r['dedup']}",
             f"measured {r['duplicates']} vs predicted {r['predicted']:.0f}; "
             f"timeouts {r['timeouts']} vs {r['pred_timeouts']:.0f}")

    print("Real pytest log:")
    rl = real_log()
    show("bytes per token", f"{rl['bytes_per_token']:.1f}"); show("windows", f"{rl['windows']:.1f}")

    print("NM 7.2 (observation):")
    o = nm72()
    show("tokens", f"{o['tokens']:,.0f}"); show("windows", f"{o['windows']:.1f}")
    show("shaped share", f"{o['shaped_share']:.1%}"); show("carried tokens", f"{o['carried']:,}")
    show("carried cost uncached", f"${o['cost_uncached']:.2f}")
    show("carried cost cached", f"${o['cost_cached']:.3f}")

    print("500 real repair runs (mini-SWE-agent, SWE-bench Verified):")
    s = swebench_obs()
    show("observations", f"{s['n']:,}"); show("truncated", s["truncated"])
    show("over 10,000 chars", s["over_cap"])
    show("share truncated", f"{s['share']:.1%}")
    show("runs with a truncation", f"{RUNS_WITH_TRUNCATION} of {RUNS} ({RUNS_WITH_TRUNCATION / RUNS:.1%})")
    show("largest raw output chars", f"{s['max_raw']:,}"); show("staged as", f"{s['max_staged']:,}")
    show("reduction", f"{s['max_raw'] / s['max_staged']:.0f}x")
    show("exit status 0 share", f"{s['rc0_share']:.1%}")

    print("SWE-agent (published):")
    w = swe()
    show("ACI over shell-only", f"{w['aci_over_shell']:.2f}x"); show("full-file drop (points)", f"{w['full_file_drop']:.1f}")

    print("NM 7.3 (polling):")
    p = poll_cost()
    show("polls", p["polls"]); show("input tokens", f"{p['tokens_in']:,}")
    show("output tokens", f"{p['tokens_out']:,}"); show("input cost", f"${p['cost_in']:.4f}")
    show("output cost", f"${p['cost_out']:.4f}"); show("total", f"${p['cost']:.4f}")
    show("ratio to NM 7.2 uncached carry", f"{p['cost'] / nm72()['cost_uncached']:.2f}")
    for s_ in (10, 30, 60):
        show(f"poll every {s_} s", f"${poll_cost(s_)['cost']:.4f}, {poll_cost(s_)['polls']} polls")

    print("NM 7.4 (catalog):")
    c = nm74()
    show("full", f"{c['full']:,}"); show("selected", f"{c['selected']:,}")
    show("released", f"{c['released']:,}"); show("removed descriptors' tokens", f"{c['removed_tokens']:,}")
    show("break-even unused descriptors", f"{c['breakeven_descriptors']:.2f}")
    show("real catalog vs NM full", f"{g['total'] / c['full']:.2f}x")

    print("Exercise 9 (discussion):")
    e = ex9()
    show("tokens saved per trajectory", f"{e['saved_tokens']:,}"); show("dollars saved", f"${e['saved']:.3f}")
    show("shell attempt cost", f"${e['shell_attempt']:.3f}")
    show("C_eff tokens only: structured", f"${e['c_struct']:.2f}"); show("shell", f"${e['c_shell']:.2f}")
    show("C_eff with losses: structured", f"${e['c_struct_loss']:.2f}"); show("shell", f"${e['c_shell_loss']:.2f}")
    show("shell break-even change probability", f"{e['p_break']:.5f}")
