import csv, sys, urllib.request, concurrent.futures as cf
from pathlib import Path
runs = sys.argv[1]; out = Path(sys.argv[2])
base = "https://swe-bench-submissions.s3.amazonaws.com/bash-only/20250726_mini-v1.0.0_claude-sonnet-4-20250514/trajs/{i}/{i}.traj.json"
ids = [r["instance_id"] for r in csv.DictReader(open(runs))]
def get(i):
    p = out / f"{i}.json"
    if p.exists() and p.stat().st_size > 0: return i, True
    try:
        with urllib.request.urlopen(base.format(i=i), timeout=60) as r: p.write_bytes(r.read())
        return i, True
    except Exception as e:
        return i, repr(e)
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(get, ids))
print(sum(1 for _, ok in res if ok is True), "of", len(ids))
