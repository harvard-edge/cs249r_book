#!/usr/bin/env python3
"""Time a ResNet-50 replica's cold start on this machine's CPU (Lecture 13).

Real tool output, produced by running PyTorch here (not a mock). A fresh
Python process plays a new replica: it imports PyTorch, builds ResNet-50,
loads its weights from a file, compiles the model with torch.compile (the
CPU analog of building an inference engine), runs warm-up inferences, and
reports ready. Each line is stamped with seconds since the process started.

Setup (run once): `coldstart_log.py --setup DIR` writes the same randomly
initialized weights to DIR as a pickle checkpoint (torch.save) and as
safetensors. Weights are random: load and compile time do not depend on the
values. The files sit in the OS page cache after setup, so the load lines
measure deserialization, not the disk.

Run (the MLSysBook venv has no PyTorch; the mlperf-edu venv does):
  PY=../../../../mlperf-edu/.venv/bin/python
  $PY coldstart_log.py --setup /tmp/cs && $PY coldstart_log.py /tmp/cs > coldstart_log.txt
The slide quotes coldstart_log.txt as is; calcs.py reads it.
"""

import time

T0 = time.perf_counter()

import platform  # noqa: E402
import sys  # noqa: E402


def stamp(msg):
    print(f"[{time.perf_counter() - T0:7.3f} s] {msg}", flush=True)


def setup(d):
    import os
    import torch
    import torchvision
    from safetensors.torch import save_file
    os.makedirs(d, exist_ok=True)
    torch.manual_seed(0)
    m = torchvision.models.resnet50(weights=None).eval()
    sd = {k: v.contiguous() for k, v in m.state_dict().items()}
    torch.save(sd, f"{d}/resnet50.pt")
    save_file(sd, f"{d}/resnet50.safetensors")


def main(d):
    stamp("process start")
    import torch
    import torchvision
    from safetensors.torch import load_file
    stamp(f"imported torch {torch.__version__}, torchvision {torchvision.__version__}")
    torch.set_num_threads(4)
    m = torchvision.models.resnet50(weights=None).eval()
    stamp(f"built ResNet-50 ({sum(p.numel() for p in m.parameters()) / 1e6:.1f} M parameters)")
    t = time.perf_counter()
    sd = torch.load(f"{d}/resnet50.pt", weights_only=True)
    stamp(f"loaded weights with torch.load (pickle): {time.perf_counter() - t:.3f} s")
    t = time.perf_counter()
    sd = load_file(f"{d}/resnet50.safetensors")
    stamp(f"loaded weights with safetensors: {time.perf_counter() - t:.3f} s")
    m.load_state_dict(sd)
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        t = time.perf_counter()
        m(x)
        stamp(f"first eager inference, batch 1: {1e3 * (time.perf_counter() - t):.1f} ms")
        t = time.perf_counter()
        m(x)
        stamp(f"second eager inference: {1e3 * (time.perf_counter() - t):.1f} ms")
        mc = torch.compile(m)
        t = time.perf_counter()
        mc(x)
        stamp(f"torch.compile, first call (compiles): {time.perf_counter() - t:.1f} s")
        t = time.perf_counter()
        mc(x)
        stamp(f"compiled inference: {1e3 * (time.perf_counter() - t):.1f} ms")
    stamp("ready")
    print(f"# {platform.machine()} {platform.system()} {platform.mac_ver()[0]}, "
          f"Python {platform.python_version()}, 4 CPU threads")


if __name__ == "__main__":
    if sys.argv[1] == "--setup":
        setup(sys.argv[2])
    else:
        main(sys.argv[1])
