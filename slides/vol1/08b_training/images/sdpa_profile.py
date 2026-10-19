#!/usr/bin/env python3
"""Real profiler output for the "you already call it" slide of Lecture 8b.

One call of torch.nn.functional.scaled_dot_product_attention at GPT-2 XL's
attention shape for one sequence (25 heads, 1,024 tokens, head width 64,
FP32, causal), on CPU, profiled with torch.profiler (profile_memory=True):
  1. the default dispatch, which picks PyTorch's fused flash kernel for CPU;
  2. the same call forced to the reference math backend with
     torch.nn.attention.sdpa_kernel(SDPBackend.MATH), which materializes
     the score matrix.
For each run it prints the profiler's top operators by self memory, with
the operator name, self CPU memory (MB) and total CPU time (ms), as
reported by prof.key_averages(). Output: data/sdpa_profile.txt.

Needs torch (not in the MLSysBook venv). Run with the mlperf-edu venv:
  ../../../../mlperf-edu/.venv/bin/python sdpa_profile.py
"""
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

torch.manual_seed(0)
q = torch.randn(1, 25, 1024, 64)
k = torch.randn_like(q)
v = torch.randn_like(q)
lines = [f"# torch {torch.__version__}, CPU, {platform.platform()}",
         "# q, k, v: (1, 25, 1024, 64) FP32, is_causal=True"]


def run(title, ctx):
    with ctx:
        F.scaled_dot_product_attention(q, k, v, is_causal=True)  # warm-up
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    lines.append(f"\n== {title}")
    lines.append(f"{'operator':<52}{'self MB':>9}{'total ms':>10}")
    evs = sorted(prof.key_averages(), key=lambda e: -e.cpu_time_total)
    for e in evs[:6]:
        lines.append(f"{e.key:<52}{e.self_cpu_memory_usage / 1e6:>9.1f}"
                     f"{e.cpu_time_total / 1e3:>10.2f}")
    return out


class Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


o1 = run("default dispatch", Null())
o2 = run("forced math backend", sdpa_kernel(SDPBackend.MATH))
lines.append(f"\nmax |difference| between the two outputs: {(o1 - o2).abs().max().item():.2e}")
text = "\n".join(lines) + "\n"
(Path(__file__).parent / "data" / "sdpa_profile.txt").write_text(text)
print(text)
