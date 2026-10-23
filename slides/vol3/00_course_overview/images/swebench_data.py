#!/usr/bin/env python3
"""Real agent runs for Lecture 0: 500 SWE-bench Verified repairs.

Source: the public SWE-bench submission "20250726_mini-v1.0.0_claude-sonnet-4-
20250514" (mini-SWE-agent 1.0.0 driving Claude 4 Sonnet, one attempt per task,
bash-only track).
  - Per-task cost, model calls, and resolved flag: swebench_per_instance.json,
    copied from github.com/SWE-bench/experiments, evaluation/verified/
    20250726_mini-v1.0.0_claude-sonnet-4-20250514/per_instance_details.json.
  - Trajectories: https://swe-bench-submissions.s3.amazonaws.com/bash-only/
    20250726_mini-v1.0.0_claude-sonnet-4-20250514/trajs/<id>/<id>.traj.json
  - Sealed-test reports for the two runs the slides show:
    .../logs/<id>/report.json (swebench_report_*.json here).

This script downloads the 500 trajectories (about 83 MB) into a cache folder
outside the repository and writes two small CSV files that make_figs.py reads:
  swebench_runs.csv           one row per task: calls, cost, resolved, exit
                              status, characters sent to the model over the
                              whole run, and characters newly added
  swebench_run_django12050.csv one row per model call of the run on the
                              "One real repair" slide: characters in the
                              context it read, cumulative sent, cumulative new

Characters, not tokens: the trajectories store message text only, so the
resend measure is counted in characters (exact) rather than tokens.

Run from this folder: python3 swebench_data.py [--cache DIR]
"""

import argparse
import csv
import json
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
RUN = "bash-only/20250726_mini-v1.0.0_claude-sonnet-4-20250514"
S3 = "https://swe-bench-submissions.s3.amazonaws.com/" + RUN


def text(message):
    c = message["content"]
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c)


def resend(messages):
    """Characters each model call reads (all earlier messages), summed, and
    the characters added since the previous call (the new suffix)."""
    rows, cum, prev = [], 0, 0
    sent = fresh = 0
    for m in messages:
        if m["role"] == "assistant":
            new = cum - prev
            sent += cum
            fresh += new
            rows.append((len(rows) + 1, cum, sent, fresh))
            prev = cum
        cum += len(text(m))
    return rows


def fetch(iid, cache):
    path = cache / f"{iid}.json"
    if not path.exists():
        urllib.request.urlretrieve(f"{S3}/trajs/{iid}/{iid}.traj.json", path)
    return iid, json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "swebench_mini_sonnet4"))
    cache = Path(ap.parse_args().cache)
    cache.mkdir(parents=True, exist_ok=True)
    det = json.loads((HERE / "swebench_per_instance.json").read_text())
    with ThreadPoolExecutor(16) as pool:
        trajs = dict(pool.map(lambda i: fetch(i, cache), sorted(det)))

    with open(HERE / "swebench_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "api_calls", "cost_usd", "resolved", "exit_status",
                    "chars_sent", "chars_new"])
        for iid in sorted(det):
            rows = resend(trajs[iid]["messages"])
            w.writerow([iid, det[iid]["api_calls"], det[iid]["cost"], int(det[iid]["resolved"]),
                        trajs[iid]["info"]["exit_status"], rows[-1][2], rows[-1][3]])

    with open(HERE / "swebench_run_django12050.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["call", "context_chars", "cum_sent_chars", "cum_new_chars"])
        w.writerows(resend(trajs["django__django-12050"]["messages"]))
    print("wrote swebench_runs.csv and swebench_run_django12050.csv")


if __name__ == "__main__":
    main()
