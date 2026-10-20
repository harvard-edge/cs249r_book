#!/usr/bin/env python3
"""Host timer against a synchronize barrier, run for Lecture 12's harness slide.

Run: python3 timer_demo.py > timer_demo_output.txt
(Apple M5 Max GPU through PyTorch's MPS backend; torch 2.13.0.)
The slide quotes the lines below the imports and the printed output as is.
"""
import time

import torch

x = torch.randn(8192, 8192, device="mps")
for _ in range(5): y = x @ x          # warm-up
torch.mps.synchronize()

t0 = time.perf_counter(); y = x @ x
t1 = time.perf_counter()              # no barrier
print(f"host timer:       {1e3*(t1-t0):.3f} ms")
torch.mps.synchronize()

t0 = time.perf_counter(); y = x @ x
torch.mps.synchronize()               # barrier
t1 = time.perf_counter()
print(f"with synchronize: {1e3*(t1-t0):.3f} ms")
