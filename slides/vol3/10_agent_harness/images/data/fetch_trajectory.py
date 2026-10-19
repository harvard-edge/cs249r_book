#!/usr/bin/env python3
"""Fetch one public repair-agent trajectory and keep what the slides use.

Source: the SWE-bench submissions bucket, bash-only track, submission
20260217_mini-v2.0.0_claude-4-6-opus (mini-SWE-agent, Claude Opus 4.6),
instance django__django-16263. The file records the agent's limits, every
model call's reported cost, and how the run ended.

Writes trajectory_django16263.json: the limits, the info block, the number
of messages, and the cumulative cost after each model call.
Run: python3 fetch_trajectory.py
"""

import json
import urllib.request
from itertools import accumulate
from pathlib import Path

URL = ("https://swe-bench-submissions.s3.amazonaws.com/bash-only/"
       "20260217_mini-v2.0.0_claude-4-6-opus/trajs/django__django-16263/"
       "django__django-16263.traj.json")


def main():
    with urllib.request.urlopen(URL, timeout=120) as r:
        d = json.load(r)
    agent = d["info"]["config"]["agent"]
    calls = [m.get("extra", {}).get("cost", 0.0) or 0.0
             for m in d["messages"] if m.get("role") == "assistant"]
    out = {
        "source": URL,
        "limits": {"step_limit": agent["step_limit"], "cost_limit": agent["cost_limit"]},
        "info": {"model_stats": d["info"]["model_stats"],
                 "mini_version": d["info"]["mini_version"],
                 "exit_status": d["info"]["exit_status"],
                 "submission": d["info"]["submission"]},
        "n_messages": len(d["messages"]),
        "last_message": d["messages"][-1],
        "call_costs": calls,
        "cumulative_cost": list(accumulate(calls)),
    }
    Path(__file__).with_name("trajectory_django16263.json").write_text(json.dumps(out, indent=1))
    print(len(calls), out["cumulative_cost"][-2:], out["info"])


if __name__ == "__main__":
    main()
