#!/usr/bin/env python3
"""Fields that differ between two MLPerf system description files.

The files in mlperf_v40_systems/ are copied unchanged from
mlcommons/training_results_v4.0 (Apache 2.0): NVIDIA/systems/eos_ngc24.04_mxnet.json,
NVIDIA/systems/h200_ngc24.04_mxnet.json, Dell/systems/XE9680x8H100-SXM-80GB.json.
The nested software block is flattened. Run from this folder:
  python3 sysdiff.py eos_ngc24.04_mxnet h200_ngc24.04_mxnet
"""
import json
import sys
from pathlib import Path

DIR = Path(__file__).parent / "mlperf_v40_systems"


def flat(name):
    out = {}
    for k, v in json.load(open(DIR / f"{name}.json")).items():
        v = v[0] if isinstance(v, list) else v
        if isinstance(v, dict):
            out.update(v)
        else:
            out[k] = v
    return out


a, b = flat(sys.argv[1]), flat(sys.argv[2])
keys = sorted(set(a) | set(b))
diff = [k for k in keys if a.get(k) != b.get(k)]
print(f"{len(diff)} of {len(keys)} fields differ")
for k in diff:
    print(f"  {k}: {a.get(k)!r} | {b.get(k)!r}")
