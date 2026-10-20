#!/usr/bin/env python3
"""Fetch every SWE-bench Verified leaderboard submission and its resolved count.

Source: github.com/SWE-bench/experiments, evaluation/verified/<date>_<name>/
  results/results.json ("resolved": list of task ids) and metadata.yaml
  (tags.system.attempts, info.name). The folder's date prefix is the
  submission date used on the plot. Writes swebench_verified_trend.csv.

Run from this folder (network needed once): python3 swebench_trend.py
"""

import csv
import json
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).parent
API = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/verified"
RAW = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"
TASKS = 500  # SWE-bench Verified size (Chowdhury et al. 2024)


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def main():
    names = sorted(x["name"] for x in json.loads(get(API)) if x["type"] == "dir")
    rows = []
    for n in names:
        try:
            res = json.loads(get(f"{RAW}/{n}/results/results.json"))
            meta = yaml.safe_load(get(f"{RAW}/{n}/metadata.yaml")) or {}
        except Exception as e:  # missing file: skip and report
            print("skip", n, e)
            continue
        attempts = str(((meta.get("tags") or {}).get("system") or {}).get("attempts", ""))
        rows.append(dict(folder=n, date=f"{n[:4]}-{n[4:6]}-{n[6:8]}",
                         name=(meta.get("info") or {}).get("name", ""),
                         attempts=attempts, resolved=len(res.get("resolved", []))))
    with open(HERE / "swebench_verified_trend.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "submissions written")


if __name__ == "__main__":
    main()
