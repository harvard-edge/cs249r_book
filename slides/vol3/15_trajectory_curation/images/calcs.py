"""Lecture 15 (Trajectory Curation): every derived number on the slides.

Run from anywhere: python3 calcs.py
Each block names its slide, its inputs and their sources, and prints the
values the slide and its note show. Inputs typed here are copied from the
source named beside them; everything printed is computed. The measured
corpus numbers come from data/corpus_stats.json, written by
data/corpus_stats.py from the public nebius/SWE-agent-trajectories corpus.
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "data" / "corpus_stats.json"


def out(label, value):
    print(f"  {label}: {value}")


# ---------------------------------------------------------------------------
# Hook and synthesis: Napkin Math 15.1 (§15.2.2), hypothetical failure mix
FAILS_RUNS = 50_000          # trajectories per month, NM 15.1
FAIL_RATE = 0.06             # failing share, NM 15.1
SCHEMA_SHARE = 0.70          # probes attribute to a tool schema missing a field
SANDBOX_SHARE = 0.20         # probes attribute to sandbox startup timeouts


def hook():
    failures = FAILS_RUNS * FAIL_RATE
    schema = failures * SCHEMA_SHARE
    sandbox = failures * SANDBOX_SHARE
    residual = failures - schema - sandbox
    print("Hook / synthesis (NM 15.1)")
    out("failures per month", f"{failures:,.0f}")
    out("schema-attributed", f"{schema:,.0f}")
    out("sandbox-attributed", f"{sandbox:,.0f}")
    out("residual after every probe", f"{residual:,.0f}")
    out("removed by system repairs", f"{schema + sandbox:,.0f} ({(schema + sandbox) / failures:.0%})")
    return dict(failures=failures, schema=schema, sandbox=sandbox, residual=residual)


# ---------------------------------------------------------------------------
# Estimate frame: pricing the fine-tuning run, NM 15.1 (§15.2.2)
PARAMS = 70.6e9              # model parameters, NM 15.1
SEQ = 4096                   # tokens per training sequence, NM 15.1
EPOCHS = 3                   # NM 15.1
FLOP_PER_PARAM_TOKEN = 6     # 6 N D training rule, NM 15.1
ACCELS = 8                   # one node of 8 accelerators, NM 15.1
PEAK = 989e12                # dense 16-bit FLOP/s per device, NM 15.1
MFU = 0.30                   # model FLOPs utilization, NM 15.1
PRICE_HR = 4.0               # dollars per accelerator-hour, NM 15.1


def finetune_cost(n_failures):
    tokens = n_failures * SEQ * EPOCHS
    flops = FLOP_PER_PARAM_TOKEN * PARAMS * tokens
    seconds = flops / (ACCELS * PEAK * MFU)
    acc_hours = seconds / 3600 * ACCELS
    return tokens, flops, seconds / 3600, acc_hours, acc_hours * PRICE_HR


def finetune():
    print("Estimate: fine-tuning on every failure (NM 15.1)")
    tokens, flops, hours, acc_h, dollars = finetune_cost(3000)
    out("training tokens", f"{tokens:,.0f}")
    out("training FLOPs", f"{flops:.3g}")
    out("node throughput (FLOP/s)", f"{ACCELS * PEAK * MFU:.4g}")
    out("wall-clock hours", f"{hours:.2f}")
    out("accelerator-hours", f"{acc_h:.1f}")
    out("invoice", f"${dollars:.2f}")
    t2, f2, h2, a2, d2 = finetune_cost(300)
    out("residual 300 only: invoice", f"${d2:.2f}")


# ---------------------------------------------------------------------------
# Splits: random trace split, §15.4 and its figure; Exercise 1 (§15.14)
def straddle(f, k):
    """Share of tasks with attempts on both sides of a random trace split."""
    return 1 - f ** k - (1 - f) ** k


def splits():
    print("Splits (§15.4)")
    for k in range(1, 9):
        out(f"straddle at 80/20, k={k}", f"{straddle(0.8, k):.4f}")
    out("Estimate: k=5, f=0.9 straddle", f"{straddle(0.9, 5):.4f}")
    out("Estimate: tasks of 2,000", f"{2000 * straddle(0.9, 5):.0f}")
    k = 1
    while straddle(0.9, k) <= 0.5:
        k += 1
    out("smallest k with a majority straddling at 90/10", k)
    out("P(test attempt has a sibling in training), k=5, f=0.9", f"{1 - 0.1 ** 4:.4f}")


# ---------------------------------------------------------------------------
# MinHash and LSH, §15.4.3 (128 components, 32 bands of 4 rows)
M_PERM = 128                 # signature components, §15.4.3
BANDS, ROWS = 32, 4          # §15.4.3
BYTES_PER_COMPONENT = 4      # §15.4.3
RECORDS = 100_000            # §15.4.3


def lsh_p(j, b=BANDS, r=ROWS):
    return 1 - (1 - j ** r) ** b


def minhash():
    print("MinHash and LSH (§15.4.3)")
    out("max standard error at J=0.5", f"{0.5 / math.sqrt(M_PERM):.4f}")
    out("bytes per signature", M_PERM * BYTES_PER_COMPONENT)
    out("MB for 100,000 records", f"{M_PERM * BYTES_PER_COMPONENT * RECORDS / 1e6:.1f}")
    for j in (0.2, 0.3, 0.5, 0.8):
        out(f"candidate probability at J={j}", f"{lsh_p(j):.4f}")
    out("steep region (1/b)^(1/r)", f"{(1 / BANDS) ** (1 / ROWS):.3f}")


# ---------------------------------------------------------------------------
# Admission ordering, NM 15.2 (§15.5.3), table 6 inputs (hypothetical)
CHECKS = {  # name: (pass probability, slot seconds)
    "record": (0.95, 0.01),
    "use": (0.90, 0.04),
    "proc": (0.80, 0.20),
    "out": (0.40, 8.0),
    "target": (0.75, 2.0),
}


def expected_time(order, checks=CHECKS):
    t, reach = 0.0, 1.0
    for name in order:
        p, s = checks[name]
        t += reach * s
        reach *= p
    return t, reach


def cascade():
    print("Admission ordering (NM 15.2)")
    for name, (p, s) in CHECKS.items():
        out(f"cost per rejection, {name}", f"{s / (1 - p):.2f} s")
    a = ["record", "use", "proc", "out", "target"]
    b = ["record", "use", "out", "proc", "target"]
    ta, ya = expected_time(a)
    tb, yb = expected_time(b)
    out("proc first: s per attempted", f"{ta:.4f}")
    out("proc first: s per admitted", f"{ta / ya:.2f}")
    out("out first: s per attempted", f"{tb:.4f}")
    out("out first: s per admitted", f"{tb / yb:.2f}")
    out("yield (both)", f"{ya:.4f}")
    out("admitted of 100,000", f"{100_000 * ya:,.0f}")
    out("saving", f"{(tb - ta) / tb:.3f}")
    out("saving per attempted record", f"{tb - ta:.2f} s")
    out("joint reach of proc and out", f"{0.95 * 0.90:.3f}")
    # Checkpoint 15.1 variant: proc costs 2 s and passes 95 percent
    alt = dict(CHECKS, proc=(0.95, 2.0))
    out("variant: proc cost per rejection", f"{2.0 / (1 - 0.95):.1f} s")
    tc, _ = expected_time(a, alt)
    td, _ = expected_time(b, alt)
    out("variant: proc first s per attempted", f"{tc:.4f}")
    out("variant: out first s per attempted", f"{td:.4f}")
    out("variant: saving with out first", f"{(tc - td) / tc:.3f}")


# ---------------------------------------------------------------------------
# Three denominators, NM 15.3 (§15.6.4), table 10 inputs (hypothetical)
POOL = {  # class: (records, stored tokens per record, target tokens per record)
    "direct": (32_500, 7_200, 1_200),
    "recovery": (12_500, 18_500, 2_000),
    "negative": (5_000, 9_800, 0),
}


def denominators():
    print("Three denominators (NM 15.3)")
    recs = sum(v[0] for v in POOL.values())
    stored = {k: v[0] * v[1] for k, v in POOL.items()}
    target = {k: v[0] * v[2] for k, v in POOL.items()}
    rec = POOL["recovery"]
    out("recovery share of records", f"{rec[0] / recs:.4f}")
    out("recovery share of stored tokens", f"{stored['recovery'] / sum(stored.values()):.4f}")
    admitted = stored["direct"] + stored["recovery"]
    out("recovery share of admitted (processed) tokens", f"{stored['recovery'] / admitted:.4f}")
    out("recovery share of selected targets", f"{target['recovery'] / sum(target.values()):.4f}")
    out("stored tokens, total", f"{sum(stored.values()):,}")
    out("targets, total", f"{sum(target.values()):,}")
    out("recovery context per target", f"{rec[1] / rec[2]:.2f}")
    d = POOL["direct"]
    out("direct context per target", f"{d[1] / d[2]:.2f}")


# ---------------------------------------------------------------------------
# Self-selection, equation 3 (§15.7.1), 8 attempts per fixture
K_ATTEMPTS = 8


def yield_any(p, k=K_ATTEMPTS):
    return 1 - (1 - p) ** k


def mixed(p, k=K_ATTEMPTS):
    return 1 - (1 - p) ** k - p ** k


def selfselect():
    print("Self-selection (equation 3)")
    for p in (0.05, 0.5, 0.95):
        out(f"p={p}: success yield", f"{yield_any(p):.4f}")
        out(f"p={p}: mixed fraction", f"{mixed(p):.4f}")
    out("attempts per harvested success at p=0.05", f"{K_ATTEMPTS / yield_any(0.05):.1f}")


# ---------------------------------------------------------------------------
# Staging burst, NM 15.4 (§15.8.2), hypothetical collector
ROLLOUTS = 1024              # concurrent rollouts, NM 15.4
ROLLOUT_S = 45.0             # seconds per rollout, NM 15.4
SLOTS = 512                  # verifier slots, NM 15.4
SERVICE_S = 16.875           # mean verifier seconds per record, NM 15.4
TIMEOUT_S = 120.0            # timeout during the burst, NM 15.4
CEIL_MB = 8.0                # per-record ceiling, MB (decimal), NM 15.4
STORE_GB = 32.0              # staging store, GB (decimal), NM 15.4
REACT_S = 30.0               # controller reaction delay, NM 15.4


def staging():
    print("Staging burst (NM 15.4)")
    lam = ROLLOUTS / ROLLOUT_S
    cap = SLOTS / SERVICE_S
    burst = SLOTS / TIMEOUT_S
    net = lam - burst
    mbps = net * CEIL_MB
    t = STORE_GB * 1000 / mbps
    inflight = ROLLOUTS * CEIL_MB / 1000
    react = REACT_S * mbps / 1000
    out("arrival rate (records/s)", f"{lam:.2f}")
    out("nominal capacity (records/s)", f"{cap:.2f}")
    out("utilization", f"{lam / cap:.3f}")
    out("burst capacity (records/s)", f"{burst:.2f}")
    out("net growth (records/s)", f"{net:.2f}")
    out("growth (MB/s)", f"{mbps:.1f}")
    out("time to full (s)", f"{t:.1f}")
    out("time to full (min)", f"{t / 60:.1f}")
    out("in-flight reservation (GB)", f"{inflight:.1f}")
    out("growth during reaction (GB)", f"{react:.2f}")
    out("high watermark at most", f"{1 - (inflight + react) / STORE_GB:.3f}")
    return dict(lam=lam, burst=burst, mbps=mbps, t_full=t, inflight=inflight, react=react)


# ---------------------------------------------------------------------------
# Cost per emitted artifact, NM 15.7 (§15.10.3), hypothetical cohort
ATTEMPTS = 500               # NM 15.7
COHORT_COST = 300.0          # dollars, NM 15.7
EMITTED = 15                 # NM 15.7, table 15


def emitted():
    print("Cost per emitted artifact (NM 15.7)")
    rate = EMITTED / ATTEMPTS
    per_attempt = COHORT_COST / ATTEMPTS
    out("emission yield", f"{rate:.3f}")
    out("cost per attempt", f"${per_attempt:.2f}")
    out("cost per emitted artifact", f"${per_attempt / rate:.2f}")
    out("at 6 percent yield", f"${per_attempt / 0.06:.2f}")
    out("attempts not emitted", f"{1 - rate:.0%}")


# ---------------------------------------------------------------------------
# Measured: the public corpus (data/corpus_stats.json)
def corpus():
    if not CORPUS.exists():
        print("Corpus: data/corpus_stats.json missing; run data/corpus_stats.py")
        return
    s = json.loads(CORPUS.read_text())
    print("Measured corpus (nebius/SWE-agent-trajectories)")
    for k, v in s["print"].items():
        out(k, v)
    rc = s["recovery"]
    d_mean = rc["direct_chars"] / rc["direct"]
    r_mean = rc["recovery_chars"] / rc["recovery"]
    out("derived: recovery runs are longer by", f"{r_mean / d_mean:.2f}x")
    res = rc["direct"] + rc["recovery"]
    out("derived: resolved runs a linter-error filter removes", f"{rc['recovery']:,} ({rc['recovery'] / res:.1%})")
    out("derived: resolved runs left after that filter", f"{rc['direct']:,}")
    out("derived: mixed issues among issues with a resolved run",
        f"{s['outcome_mix']['mixed'] / rc['tasks_resolved']:.1%}")


if __name__ == "__main__":
    hook()
    finetune()
    splits()
    minhash()
    cascade()
    denominators()
    selfselect()
    staging()
    emitted()
    corpus()
