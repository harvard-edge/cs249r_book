#!/usr/bin/env python3
"""Profile one GPT-2 decode step on the CPU with torch.profiler (Lecture 1).

A real measurement, not a mock: Hugging Face GPT-2 (124M, the published
weights from the 'gpt2' model card) generates one token with a 64-token
KV cache, on this machine's CPU in FP32. Prints the profiler's
key_averages table (the PyTorch profiler recipe,
pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), and writes it
to data/gpt2_profile.txt together with the versions and machine.

Needs torch and transformers (not in the MLSysBook venv):
  uv venv tvenv && uv pip install --python tvenv/bin/python torch transformers
  tvenv/bin/python profile_gpt2.py
"""
import platform
import subprocess
import time
from pathlib import Path

import torch
import transformers
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import GPT2LMHeadModel

torch.manual_seed(0)
torch.set_num_threads(1)          # one core, so the timing is easy to reason about
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
n_params = sum(p.numel() for p in model.parameters())
ctx = torch.randint(0, 50257, (1, 64))
with torch.no_grad():
    out = model(ctx, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    for _ in range(5):            # warm up
        model(nxt, past_key_values=past, use_cache=True)
        past.crop(-1)   # drop the new token so every step sees 64
    t0 = time.perf_counter()
    for _ in range(20):
        model(nxt, past_key_values=past, use_cache=True)
        past.crop(-1)   # drop the new token so every step sees 64
    step_ms = (time.perf_counter() - t0) / 20 * 1e3
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with record_function("decode_step"):
            model(nxt, past_key_values=past, use_cache=True)

table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=6)
cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                     capture_output=True, text=True).stdout.strip() or platform.processor()
head = (f"# GPT-2 (124M) decode step, batch 1, 64-token KV cache, FP32, 1 CPU thread\n"
        f"# torch {torch.__version__}, transformers {transformers.__version__}, {cpu}\n"
        f"# parameters: {n_params}\n"
        f"# mean of 20 unprofiled steps: {step_ms:.2f} ms\n"
        f"# command: tvenv/bin/python profile_gpt2.py\n")
print(head + table)
(Path(__file__).parent / "data" / "gpt2_profile.txt").write_text(head + table)
