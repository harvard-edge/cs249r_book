#!/usr/bin/env python3
"""Every derived number in Lecture 11 (Durable Execution), one block per slide.

Inputs are typed with their source. Sources:
  DE    = the rendered chapter, Durable Execution (Volume III), dev text 2026-10-18.
  NM    = a Napkin Math notebook of that chapter (scenario values the chapter
          labels hypothetical; slides say "illustrative").
  MLSysIM Literature.AgentTimeHorizon = METR Time Horizon v1.1 results.
  TM    = Horace He and Thinking Machines Lab, "Defeating Nondeterminism in
          LLM Inference," 10 September 2025 (blog post, read 2026-10-19).
  data/ = runs measured for this lecture on one laptop (Apple M5 Max, macOS
          26.4); the script that produced each file sits beside it. The
          Temporal runs need the temporalio package (1.34.0) and a local
          Ollama; the other scripts need only the standard library.

Run from this folder with the MLSysBook venv:  python3 calcs.py
make_figs.py imports these functions so figures and slides agree.
"""

import json
import statistics
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Literature  # noqa: E402


def show(label, value):
    print(f"  {label}: {value}")


def load(name):
    return json.loads((DATA / name).read_text())


# ------------------------------------------------------------------ hook
# data/resample.json: the repair agent's original task prompt sent 30 times
# to qwen2.5:7b at temperature 0.8 and 30 times at temperature 0.
def resample():
    d = load("resample.json")
    hot, cold = d["0.8"], d["0.0"]
    # group the first steps by tool name for the plot
    by_tool = {}
    for reply, n in hot["counts"]:
        tool = reply.split("(")[0].strip()
        by_tool[tool] = by_tool.get(tool, 0) + n
    return dict(n=hot["n"], distinct=hot["distinct"], top=hot["counts"][0][1],
                top_reply=hot["counts"][0][0], by_tool=by_tool,
                n0=cold["n"], distinct0=cold["distinct"])


# --------------------------------------------- horizon (segment 1 opener)
# MLSysIM Literature.AgentTimeHorizon: METR 50% time horizon, minutes of
# expert human time, state-of-the-art models only.
def horizons():
    rows = [r for r in Literature.AgentTimeHorizon.entries if r["is_sota"]]
    pts = [(date.fromisoformat(r["release_date"]), r["p50_min"]) for r in rows]
    gpt4 = next(r for r in rows if r["model_key"] == "gpt_4")
    last = rows[-1]
    return dict(points=pts, gpt4_min=gpt4["p50_min"], gpt4_date=gpt4["release_date"],
                last_min=last["p50_min"], last_h=last["p50_min"] / 60,
                last_date=last["release_date"], ratio=last["p50_min"] / gpt4["p50_min"])


# --------------------------------------------- the real crash run (Temporal)
# data/summary_Repair*.json: a worker SIGKILLed during the r18 test activity.
def crash_runs():
    out = {}
    for name in ("RepairNaive", "RepairKeyed"):
        s = load(f"summary_{name}.json")
        led = s["ledger"]
        t0 = led[0]["t"]
        out[name] = dict(
            kill_s=s["kill_after_s"], restart_s=s["restart_after_s"], done_s=s["done_after_s"],
            model_calls=sum(e["event"] == "model_call" for e in led),
            launches=sum(e["event"] == "job_launched" for e in led),
            reused=sum(e["event"] == "job_reused" for e in led),
            events=len(s["events"]),
            ledger=[(e["event"], e.get("job"), e["attempt"], round(e["t"] - t0, 2))
                    for e in led])
    return out


def history_excerpt(name="RepairNaive"):
    """Event id, type, and the few attributes the slide shows."""
    h = load(f"history_{name}.json")
    rows = []
    for e in h["events"]:
        typ = e["eventType"].replace("EVENT_TYPE_", "")
        attrs = next((v for k, v in e.items() if k.endswith("Attributes")), {})
        extra = {}
        if "activityType" in attrs:
            extra["activity"] = attrs["activityType"]["name"]
        if "attempt" in attrs:
            extra["attempt"] = attrs["attempt"]
        if "lastFailure" in attrs:
            extra["lastFailure"] = attrs["lastFailure"]["message"]
        rows.append((int(e["eventId"]), typ, extra))
    return rows


# ------------------------------------------------ acknowledgment latency
# data/ack_latency.json: 512-byte appends on the laptop's internal SSD.
def acks():
    d = load("ack_latency.json")
    w, f, ff = (d[k]["median_ms"] for k in ("write", "fsync", "full_fsync"))
    g = {int(k): v for k, v in d["group_commit"].items()}
    return dict(write=w, fsync=f, full=ff, full_over_fsync=ff / f, group=g,
                g1=g[1]["per_record_ms"], g16=g[16]["per_record_ms"],
                g64=g[64]["per_record_ms"], flush64=g[64]["flush_ms_median"],
                n=d["write"]["n"])


# DE Sec. 11.3.4 (group commit, worked numbers): 2 ms timer, 4 ms quorum
# acknowledgment, a batch of 16 intents.
def group_commit_chapter(timer=2.0, flush=4.0, k=16):
    return dict(per_intent=flush / k, first_wait=timer + flush, last_wait=flush)


# ------------------------------------------ full vs incremental snapshots
# DE Sec. 11.4.2 (equation 3): 500 turns, 128 KB of state per turn,
# a checkpoint every 5 turns.
def snapshots(turns=500, growth_kb=128, stride=5):
    n = turns // stride
    full = growth_kb * stride * n * (n + 1) / 2          # KB
    delta = growth_kb * stride * n                         # KB
    return dict(n=n, full_gb=full / 1e6, delta_mb=delta / 1e3, ratio=full / delta)


def snapshot_curves(growth_kb=128, stride=5, nmax=100):
    ks = list(range(0, nmax + 1))
    return ks, [growth_kb * stride * k * (k + 1) / 2 / 1e6 for k in ks], \
        [growth_kb * stride * k / 1e6 for k in ks]


# ------------------------------------------------- checkpoint cadence
# DE NM 11.1: 100 turns, crashes after turns 37 and 83, 0.5 s per checkpoint,
# 1 s restore per crash, 0.1 s per logged turn reconstructed; an initial
# checkpoint exists and each policy also checkpoints at the final turn.
TURNS, CRASHES, PAUSE, RESTORE, PER_TURN, REEXEC = 100, (37, 83), 0.5, 1.0, 0.1, 20.0


def trace_overhead(stride, per_turn=PER_TURN):
    marks = set(range(stride, TURNS + 1, stride)) | {TURNS}
    ckpts = len(marks)
    lost = [c - (c // stride) * stride for c in CRASHES]
    return dict(ckpts=ckpts, lost=lost,
                total=ckpts * PAUSE + len(CRASHES) * RESTORE + per_turn * sum(lost))


def cadence():
    rate = len(CRASHES) / TURNS
    s_log = (2 * PAUSE / (rate * PER_TURN)) ** 0.5
    s_unlog = (2 * PAUSE / (rate * REEXEC)) ** 0.5
    return dict(rate=rate, s_log=s_log, s_unlog=s_unlog, gain=s_log / s_unlog,
                **{f"t{s}": trace_overhead(s) for s in (1, 10, 25, 50)})


# DE theorem 11.1 (Young 1974): 2 s pause, 10-minute mean time between failures.
def young(pause=2.0, mtbf=600.0):
    t = (2 * pause * mtbf) ** 0.5
    ovh = (2 * pause / mtbf) ** 0.5

    def miss(m):     # overhead at m times the optimum, relative to the optimum
        return (m + 1 / m) / 2
    return dict(t_opt=t, overhead=ovh, miss2=miss(2) - 1, miss10=miss(10) - 1)


def expected_overhead_curve(stride, per_turn):
    """Expected seconds per turn: pause/stride + rate * per_turn * stride / 2."""
    rate = len(CRASHES) / TURNS
    return PAUSE / stride + rate * per_turn * stride / 2


# ------------------------------------------------------- resume prefill
# data/resume_prefill.json: next call after a resume, 10,202-token context.
def prefill():
    d = load("resume_prefill.json")
    cold = statistics.median(r["wall_s"] for r in d["cold"])
    warm = statistics.median(r["wall_s"] for r in d["warm"])
    load_s = statistics.median(r["load_duration"] for r in d["cold"]) / 1e9
    return dict(tokens=d["cold"][0]["prompt_eval_count"], cold=cold, warm=warm,
                ratio=cold / warm, load=load_s, n=len(d["cold"]))


# --------------------------------------------------- temperature zero
# TM: 1,000 completions of Qwen3-235B-A22B-Instruct-2507 at temperature 0,
# "Tell me about Richard Feynman", 1,000 tokens each: 80 unique, the most
# common 78 times, identical through token 102.
TM_N, TM_UNIQUE, TM_TOP, TM_SAME_PREFIX = 1000, 80, 78, 102


# DE Sec. 11.9 (fallacy, equation 1.1): each call reproduces with 0.99, 30 turns.
def reproduce(p=0.99, h=30):
    return p ** h


# ------------------------------------------------------- retention
# DE NM 11.2: 30 turns of 4 KB envelopes + 128 KB inputs and results, one
# 50 MB workspace snapshot, $23 per TB-month, 12 months, $2.10 attempt.
def retention(turns=30, env_kb=4, body_kb=128, snap_mb=50, usd_tb_month=23.0,
              months=12, attempt=2.10):
    log_mb = turns * (env_kb + body_kb) / 1e3
    total_gb = (log_mb + snap_mb) / 1e3
    usd_gb_month = usd_tb_month / 1e3
    cost = total_gb * usd_gb_month * months
    log_cost = log_mb / 1e3 * usd_gb_month * months
    return dict(log_mb=log_mb, total_mb=log_mb + snap_mb, cost_cents=cost * 100,
                share=cost / attempt, log_share=log_cost / attempt,
                snap_share=snap_mb / (log_mb + snap_mb))


if __name__ == "__main__":
    r = resample()
    print("Hook: restarts from the original prompt (data/resample.json)")
    show("restarts at temperature 0.8", r["n"])
    show("distinct first steps", r["distinct"])
    show("most common first step and count", f"{r['top_reply']} x {r['top']}")
    show("first steps by tool", r["by_tool"])
    show("distinct at temperature 0, one machine", f"{r['distinct0']} of {r['n0']}")

    h = horizons()
    print("Segment 1: METR 50% time horizon")
    show("GPT-4 (2023-03-14), minutes", f"{h['gpt4_min']:.1f}")
    show(f"latest frontier point ({h['last_date']}), hours", f"{h['last_h']:.1f}")
    show("growth factor", f"{h['ratio']:.0f}")

    c = crash_runs()
    print("Segment 1-2: Temporal crash runs (data/summary_*.json)")
    for k, v in c.items():
        show(k, {x: v[x] for x in ("kill_s", "restart_s", "done_s", "model_calls",
                                   "launches", "reused", "events")})
        show(k + " ledger (event, job, attempt, s after first entry)", v["ledger"])
    for row in history_excerpt():
        if row[2]:
            show("history", row)

    a = acks()
    print("Segment 2: acknowledgment latency on one laptop (data/ack_latency.json)")
    show("write() median, ms", f"{a['write']:.3f}")
    show("fsync() median, ms", f"{a['fsync']:.3f}")
    show("F_FULLFSYNC median, ms", f"{a['full']:.2f}")
    show("F_FULLFSYNC / fsync", f"{a['full_over_fsync']:.0f}x")
    show("group commit per record, k=1 / 16 / 64, ms",
         f"{a['g1']:.2f} / {a['g16']:.2f} / {a['g64']:.3f}")
    show("group commit flush at k=64, ms", f"{a['flush64']:.2f}")
    g = group_commit_chapter()
    show("chapter: amortized per intent, ms", g["per_intent"])
    show("chapter: first / last arrival waits, ms", f"{g['first_wait']} / {g['last_wait']}")

    s = snapshots()
    print("Segment 3: full vs delta (equation 3)")
    show("checkpoints", s["n"])
    show("full snapshots, GB", f"{s['full_gb']:.2f}")
    show("deltas, MB", f"{s['delta_mb']:.0f}")
    show("ratio", f"{s['ratio']:.1f}")

    k = cadence()
    print("Segment 3: cadence on the NM 11.1 trace")
    for st in (1, 10, 25, 50):
        show(f"stride {st}", k[f"t{st}"])
    show("crash rate per turn", k["rate"])
    show("logged balance stride", f"{k['s_log']:.1f}")
    show("unlogged balance stride", f"{k['s_unlog']:.1f}")
    show("stride gain", f"{k['gain']:.1f}x")
    y = young()
    show("Young: T* s / overhead", f"{y['t_opt']:.0f} / {100 * y['overhead']:.1f}%")
    show("miss by 2x / 10x, extra overhead", f"{100 * y['miss2']:.0f}% / {100 * y['miss10']:.0f}%")

    p = prefill()
    print("Segment 3: next call after a resume (data/resume_prefill.json)")
    show("prompt tokens", p["tokens"])
    show("cold (model reloaded) median wall, s", f"{p['cold']:.1f}")
    show("warm (prefix held) median wall, s", f"{p['warm']:.2f}")
    show("ratio", f"{p['ratio']:.0f}x")
    show("model load median, s", f"{p['load']:.2f}")

    print("Segment 4: reproduction")
    show("0.99^30", f"{reproduce():.3f}")
    show("TM unique completions", f"{TM_UNIQUE} of {TM_N}; most common {TM_TOP}")

    t = retention()
    print("Segment 4: retention (NM 11.2)")
    show("log, MB", f"{t['log_mb']:.2f}")
    show("total, MB", f"{t['total_mb']:.1f}")
    show("year of retention, cents", f"{t['cost_cents']:.2f}")
    show("share of a $2.10 attempt", f"{100 * t['share']:.2f}%")
    show("log alone, share", f"{100 * t['log_share']:.3f}%")
    show("snapshot share of bytes", f"{100 * t['snap_share']:.1f}%")
