#!/usr/bin/env python3
"""Reduce one public SWE-bench Verified agent run to a small per-task CSV.

The run: mini-swe-agent 1.0.0 with Claude 4 Sonnet (20250514) on the 500
tasks of SWE-bench Verified, "bash only" track, one attempt per task.

  Submission record (cost, model calls, resolved, per task):
    https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/
      20250726_mini-v1.0.0_claude-sonnet-4-20250514/per_instance_details.json
  Trajectories (one JSON file per task):
    https://swe-bench-submissions.s3.amazonaws.com/bash-only/
      20250726_mini-v1.0.0_claude-sonnet-4-20250514/trajs/<id>/<id>.traj.json

Download both into one folder (per_instance_details.json beside a trajs/
folder holding <id>.json for each task), then run

    python3 extract_run.py <that folder>

It writes data/swebench_verified_mini_sonnet4.csv next to this script. The
deck's calcs.py and make_figs.py read only that CSV. Each trajectory is read
as data; nothing in it is executed.

Command classes (side-effect class in the container, an A1 sandbox):
  read   only reads files or the repository (cat, sed -n, grep, ls, git log)
  write  changes files in the working copy (sed -i, cp, rm, a redirect, git checkout)
  run    executes code in the working copy (python, pytest, sphinx-build)
  net    tries to install or fetch over the network (pip, curl, wget)
"""

import csv
import json
import re
import sys
from pathlib import Path

RUN = re.compile(r"^(python3?|pytest|sphinx-build|timeout|PYTHON\w*=\S+|tox|make|bash|sh)\b")
NET = re.compile(r"\b(pip3?|curl|wget|conda|apt-get)\b")
WRITE = re.compile(
    r"(sed\s+-i|\bcp\b|\brm\b|\bmv\b|\bmkdir\b|\btouch\b|\btee\b|\bpatch\b|"
    r"git\s+(checkout|restore|stash|apply|reset)|"
    r"(?<![0-9&])>{1,2}(?!&)\s*(?!/dev/null)\S)"
)


def text(content):
    if isinstance(content, list):
        return " ".join(x.get("text", "") for x in content if isinstance(x, dict))
    return str(content)


def classify(cmd):
    c = cmd.strip()
    if c.startswith("cd ") and "&&" in c:
        c = c.split("&&", 1)[1].strip()
    if NET.search(c.split("\n")[0]):
        return "net"
    if WRITE.search(c):
        return "write"
    if RUN.match(c):
        return "run"
    return "read"


def main(folder):
    folder = Path(folder)
    details = json.loads((folder / "per_instance_details.json").read_text())
    rows = []
    for iid in sorted(details):
        t = json.loads((folder / "trajs" / f"{iid}.json").read_text())
        counts = {"read": 0, "write": 0, "run": 0, "net": 0}
        n_obs = n_nonzero = n_trunc = 0
        syntax = False
        for m in t["messages"][2:]:
            body = text(m.get("content"))
            if m["role"] == "assistant":
                cmd = re.findall(r"```bash\n(.*?)```", body, re.S)
                if cmd:
                    counts[classify(cmd[0])] += 1
            elif m["role"] == "user" and "<returncode>" in body:
                n_obs += 1
                rc = re.findall(r"<returncode>(-?\d+)</returncode>", body)
                if rc and rc[0] != "0":
                    n_nonzero += 1
                if "output of your last command was too long" in body:
                    n_trunc += 1
                if "SyntaxError" in body or "IndentationError" in body:
                    syntax = True
        files = re.findall(r"^diff --git a/(\S+)", t["info"].get("submission", ""), re.M)
        d = details[iid]
        rows.append({
            "instance_id": iid,
            "model_calls": d["api_calls"],
            "cost_usd": round(d["cost"], 6),
            "resolved": int(bool(d["resolved"])),
            "exit_status": t["info"].get("exit_status", ""),
            "observations": n_obs,
            "nonzero_exit": n_nonzero,
            "truncated": n_trunc,
            "syntax_error_seen": int(syntax),
            "cmd_read": counts["read"],
            "cmd_write": counts["write"],
            "cmd_run": counts["run"],
            "cmd_net": counts["net"],
            "files_in_submission": len(files),
        })
    out = Path(__file__).resolve().parent / "data" / "swebench_verified_mini_sonnet4.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} tasks -> {out.name}")


if __name__ == "__main__":
    main(sys.argv[1])
