#!/usr/bin/env python3
"""Every derived number in Lecture 7 (ML Frameworks), computed from sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports from this file, so figures and slides share
one computation. Measured inputs are read from data/, which
profile_frameworks.py writes by running PyTorch on this machine.

Run from this folder with the MLSysBook venv:
  /Users/VJ/GitHub/MLSysBook/.venv/bin/python calcs.py
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n{title}")


# ---------------------------------------------------------------------------
# Measured data (data/, written by profile_frameworks.py)
# ---------------------------------------------------------------------------
def load_resnet_memory():
    """Rows (event, step, phase, module, bytes) of the ResNet-50 batch-64 run,
    and the parameter count from the file header."""
    rows, n_params = [], None
    for line in open(DATA / "resnet50_memory.csv"):
        if line.startswith("#"):
            m = re.search(r"(\d+) parameters", line)
            if m:
                n_params = int(m.group(1))
            continue
        if line.startswith("event"):
            continue
        e, s, phase, mod, b = line.rstrip("\n").split(",")
        rows.append((int(e), int(s), phase, mod, int(b)))
    return rows, n_params


def load_resnet_b128():
    t = (DATA / "resnet50_peak_b128.txt").read_text()
    start = int(re.search(r"start bytes: (\d+)", t).group(1))
    peak = int(re.search(r"peak bytes: (\d+)", t).group(1))
    return start, peak


def load_mnist_times():
    """Median eager step time (us) per batch size, Apple GPU (MPS)."""
    t = (DATA / "mnist_step.txt").read_text()
    return {int(b): float(us) for b, us in re.findall(r"batch (\d+): median step ([\d.]+) us", t)}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
# ML Frameworks, Sec. 7.1 (GPT-2 XL decode on an H100)
GPT2XL_WEIGHT_GB = 3.0            # FP16 weights
HBM_TBPS = 3.35                   # H100 HBM3
BLOCKS, KERNELS_PER_BLOCK = 48, 12
T_DISPATCH_US, T_LAUNCH_US = 10.0, 5.0   # chapter's illustrative overhead model

# Sec. 7.3.4 (dispatch share): kernel times
SMALL_KERNEL_US, LARGE_KERNEL_US = 1.0, 100.0

# Napkin Math 7.2 (dispatch overhead law)
MLP_DISPATCH_US, MLP_DEVICE_US = 90.0, 2.6
GPT3_LAYER_DEVICE_MS = 100.0

# Table 4 and Sec. 7.10 (breakeven, hypothetical)
EAGER_IPS, COMPILED_IPS, T_COMPILE_S = 1450.0, 2150.0, 30.0
PROTO_IMAGES, PROTO_CHANGES = 10_000, 10

# Sec. 7.3.6, figure 6 (operator registries, 2025)
OPS = {"PyTorch ATen": 2658, "TensorFlow": 1563, "ONNX": 203,
       "Core ATen IR": 192, "JAX lax": 135, "XLA HLO": 123}

# Ansel et al., "PyTorch 2," ASPLOS 2024, Table 3: TorchInductor geometric-mean
# speedup over eager, NVIDIA A100, float32 (inference, training)
PT2_TABLE3 = {"TorchBench": (2.73, 1.38), "Hugging Face": (1.47, 1.24), "TIMM": (2.48, 1.38)}
# Table 4: Hugging Face, float16, A100, inference geomean speedup
PT2_ALL, PT2_NO_FUSION_INLINING, PT2_NO_CUDAGRAPHS = 1.91, 0.80, 1.81

# Sec. 7.4 (Jacobian argument): one 100,000-wide layer, FP32
JAC_WIDTH, FP32_B = 100_000, 4

# Sec. 7.4.3 (illustrative ResNet-50 scenario)
RESNET_PARAMS = 25.6e6
ACT_GB_LO, ACT_GB_HI = 8.0, 12.0          # saved activations at batch 64
H100_GB = 80

# Sec. 7.5.2, table 7 (bandwidths, GB/s)
HBM_GBPS, NVLINK_GBPS, PCIE_GBPS = 3350.0, 450.0, 64.0
TENSOR_MB = 4.0

# Sec. 7.5.3 (copy estimate) and listing 15 (DataLoader)
BATCH, H, W, CH = 64, 224, 224, 3
STEP_MS_BIG, STEP_MS_SMALL = 5.0, 0.5
WORKERS, PREFETCH = 4, 2
FEED_IPS = 1000                          # images per second, uint8 pixels

# Sec. 7.7 (MNIST step, 784-128-10, batch 32, H100)
D_IN, D_H, D_OUT, MNIST_BATCH = 784, 128, 10, 32
ADAM_BYTES_PER_PARAM = 28
STEP_MFLOP, STEP_MB = 13.1, 1.2
FP32_TFLOPS = 67.0
PY_ENTRIES, LAUNCHES = 5, 9


def figure_values():
    """The derived values make_figs.py draws, from the same inputs as main()."""
    t_op = T_DISPATCH_US + T_LAUNCH_US
    n_k = BLOCKS * KERNELS_PER_BLOCK
    w_mb = RESNET_PARAMS * FP32_B / 1e6
    fixed_gb = (w_mb + w_mb + 2 * w_mb) / 1e3
    return {
        "decode_read_ms": GPT2XL_WEIGHT_GB / (HBM_TBPS * 1e3) * 1e3,
        "decode_launch_ms": n_k * T_LAUNCH_US / 1e3,
        "decode_host_ms": n_k * t_op / 1e3,
        "decode_kernels": n_k,
        "t_op_us": t_op,
        "resnet_weights_mb": w_mb,
        "resnet_total_gb": (fixed_gb + ACT_GB_LO, fixed_gb + ACT_GB_HI),
        "resnet_act512_gb": (ACT_GB_LO * 512 / 64, ACT_GB_HI * 512 / 64),
        "hbm_over_pcie": HBM_GBPS / PCIE_GBPS,
        "mnist_compute_us": STEP_MFLOP * 1e6 / (FP32_TFLOPS * 1e12) * 1e6,
        "mnist_memory_us": STEP_MB * 1e6 / (HBM_TBPS * 1e12) * 1e6,
        "mnist_host_us": PY_ENTRIES * T_DISPATCH_US + LAUNCHES * T_LAUNCH_US,
    }


def main():
    head("Slide: units (Mars Climate Orbiter)")
    # Mishap Investigation Board Phase I report, p. 14: 1 lbf = 4.45 N, input only
    show("factor (input, MIB report p. 14)", 4.45)

    head("Slide: fusion")
    k = 3
    show("unfused traffic / fused traffic, k = 3", f"{(2 * k) / 2:.0f}x")
    show("HBM trips unfused, launches", f"{2 * k}, {k}")

    head("Slide: operator surfaces")
    show("PyTorch ATen / Core ATen IR", f"{OPS['PyTorch ATen'] / OPS['Core ATen IR']:.1f}")
    show("PyTorch ATen / ONNX", f"{OPS['PyTorch ATen'] / OPS['ONNX']:.1f}")

    head("Slide: predict host time (GPT-2 XL decode)")
    read_ms = GPT2XL_WEIGHT_GB / (HBM_TBPS * 1e3) * 1e3
    n_k = BLOCKS * KERNELS_PER_BLOCK
    launch_ms = n_k * T_LAUNCH_US / 1e3
    host_ms = n_k * (T_DISPATCH_US + T_LAUNCH_US) / 1e3
    show("weight read (ms)", f"{read_ms:.2f}")
    show("kernels", n_k)
    show("launches only (ms)", f"{launch_ms:.2f}")
    show("dispatch + launch (ms)", f"{host_ms:.2f}")
    show("launches / read", f"{launch_ms / read_ms:.1f}x")
    show("host / read", f"{host_ms / read_ms:.1f}x")

    head("Slide: dispatch share")
    t_op = T_DISPATCH_US + T_LAUNCH_US
    show("per-op host cost (us)", f"{t_op:.0f}")
    show("small op dispatch share", f"{100 * t_op / (t_op + SMALL_KERNEL_US):.0f}%")
    show("large op dispatch share", f"{100 * t_op / (t_op + LARGE_KERNEL_US):.0f}%")

    head("Focus: dispatch overhead law (Napkin Math 7.2)")
    show("small MLP ratio", f"{MLP_DISPATCH_US / MLP_DEVICE_US:.1f}")
    show("small MLP host share", f"{100 * MLP_DISPATCH_US / (MLP_DISPATCH_US + MLP_DEVICE_US):.0f}%")
    gpt3_dispatch_ms = KERNELS_PER_BLOCK * t_op / 1e3
    show("GPT-3 block dispatch (ms)", f"{gpt3_dispatch_ms:.2f}")
    show("GPT-3 layer ratio", f"{gpt3_dispatch_ms / GPT3_LAYER_DEVICE_MS:.4f}")

    head("Slide: breakeven (hypothetical)")
    t_e = 1e3 / EAGER_IPS
    t_c = 1e3 / COMPILED_IPS
    n_be = T_COMPILE_S * 1e3 / (t_e - t_c)
    show("eager per image (ms)", f"{t_e:.3f}")
    show("compiled per image (ms)", f"{t_c:.3f}")
    show("saving per image (ms)", f"{t_e - t_c:.3f}")
    show("breakeven images", f"{n_be:,.0f}")
    proto_e = PROTO_IMAGES / EAGER_IPS
    proto_run = PROTO_IMAGES / COMPILED_IPS
    proto_c = PROTO_CHANGES * T_COMPILE_S + proto_run
    show("prototype eager (s)", f"{proto_e:.1f}")
    show("prototype compiled running (s)", f"{proto_run:.1f}")
    show("prototype compiled total (s)", f"{proto_c:.1f}")
    show("prototype slowdown", f"{proto_c / proto_e:.1f}x")

    head("Slide: PyTorch 2 measured speedups (Ansel et al. 2024, Tables 3 and 4)")
    for suite, (inf, trn) in PT2_TABLE3.items():
        show(f"{suite} inference / training (input)", f"{inf} / {trn}")
    show("HF fp16: all opts / no fusion+inlining / no cudagraphs (input)",
         f"{PT2_ALL} / {PT2_NO_FUSION_INLINING} / {PT2_NO_CUDAGRAPHS}")

    head("Slide: reverse-mode tape")
    x = 1.0
    y = x * 2
    z = y + 1
    show("y, z, dz/dx", f"{y:g}, {z:g}, {1 * 2:g}")
    jac_gb = JAC_WIDTH * JAC_WIDTH * FP32_B / 1e9
    show("Jacobian of one 100,000-wide layer, FP32 (GB)", f"{jac_gb:.0f}")

    head("Slide: saved forward values")
    x, w1, w2 = 1.0, 2.0, 3.0
    hidden = x * w1
    activated = max(0.0, hidden)
    d_w2 = activated
    d_hidden = w2 * (1.0 if hidden > 0 else 0.0)
    d_w1 = x * d_hidden
    show("hidden, activated", f"{hidden:g}, {activated:g}")
    show("d_w2, d_hidden, d_w1", f"{d_w2:g}, {d_hidden:g}, {d_w1:g}")

    head("Slide: ResNet-50 training memory (chapter's illustrative estimate)")
    w_mb = RESNET_PARAMS * FP32_B / 1e6
    adam_mb = 2 * w_mb
    lo_gb = (w_mb + w_mb + adam_mb) / 1e3 + ACT_GB_LO
    hi_gb = (w_mb + w_mb + adam_mb) / 1e3 + ACT_GB_HI
    show("FP32 weights (MB)", f"{w_mb:.1f}")
    show("gradients (MB)", f"{w_mb:.1f}")
    show("Adam moments (MB)", f"{adam_mb:.1f}")
    show("total, batch 64 (GB)", f"{lo_gb:.1f} to {hi_gb:.1f}")
    show("total / weights", f"{lo_gb * 1e3 / w_mb:.1f} to {hi_gb * 1e3 / w_mb:.1f}x")
    show("activations at batch 512 (GB)", f"{ACT_GB_LO * 512 / 64:.0f} to {ACT_GB_HI * 512 / 64:.0f}")

    head("Slide: ResNet-50 training memory, measured (Apple GPU, MPS)")
    rows, n_params = load_resnet_memory()
    mw_mb = n_params * FP32_B / 1e6
    s1 = [r for r in rows if r[1] == 1]
    s2 = [r for r in rows if r[1] == 2]
    start1 = s1[0][4]
    peak1, peak2 = max(r[4] for r in s1), max(r[4] for r in s2)
    after_bwd1 = next(r[4] for r in s1 if r[2] == "after_backward")
    after_step1 = next(r[4] for r in s1 if r[2] == "after_step")
    start2 = s2[0][4]
    show("parameters (measured model)", f"{n_params:,}")
    show("FP32 weights (MB)", f"{mw_mb:.1f}")
    show("step 1 start (GB)", f"{start1 / 1e9:.2f}")
    show("step 1 peak (GB)", f"{peak1 / 1e9:.2f}")
    show("step 2 peak (GB)", f"{peak2 / 1e9:.2f}")
    show("peak / weights", f"{peak2 / 1e6 / mw_mb:.0f}x")
    show("Adam state added at first step (MB)", f"{(after_step1 - after_bwd1) / 1e6:.0f}")
    show("step 2 start, weights + Adam state + input (GB)", f"{start2 / 1e9:.2f}")
    rise2 = peak2 - start2
    show("step 2 rise above start (GB)", f"{rise2 / 1e9:.2f}")
    st128, pk128 = load_resnet_b128()
    rise128 = pk128 - st128
    show("batch 128 peak (GB)", f"{pk128 / 1e9:.2f}")
    show("batch 128 rise above start (GB)", f"{rise128 / 1e9:.2f}")
    show("rise ratio, batch 128 / batch 64", f"{rise128 / rise2:.2f}")
    show("batch-64 rise scaled to batch 512 (GB)", f"{rise2 * 512 / 64 / 1e9:.0f}")
    show("measured peak / estimate low end", f"{peak2 / 1e9 / lo_gb:.2f}")

    head("Slide: host link")
    show("HBM / PCIe", f"{HBM_GBPS / PCIE_GBPS:.1f}x")
    show("4 MB over HBM (us)", f"{TENSOR_MB * 1e6 / (HBM_GBPS * 1e9) * 1e6:.2f}")
    show("4 MB over PCIe (us)", f"{TENSOR_MB * 1e6 / (PCIE_GBPS * 1e9) * 1e6:.1f}")

    head("Slide: copy estimate")
    nbytes = BATCH * H * W * CH * FP32_B
    copy_ms = nbytes / (PCIE_GBPS * 1e9) * 1e3
    show("batch bytes (MB)", f"{nbytes / 1e6:.1f}")
    show("over PCIe (ms)", f"{copy_ms:.2f}")
    show("inside HBM (us)", f"{nbytes / (HBM_GBPS * 1e9) * 1e6:.1f}")
    show("serial copy / 5 ms step", f"{100 * copy_ms / STEP_MS_BIG:.0f}%")

    head("Slide: DataLoader")
    show("batches in flight", WORKERS * PREFETCH)
    show("raw feed at 1,000 images/s (MB/s)", f"{FEED_IPS * H * W * CH / 1e6:.1f}")

    head("Slide: one training step (MNIST 784-128-10)")
    n_p = D_IN * D_H + D_H + D_H * D_OUT + D_OUT
    show("parameters", f"{n_p:,}")
    show("Adam bytes moved (MB)", f"{n_p * ADAM_BYTES_PER_PARAM / 1e6:.2f}")

    head("Slide: host 95 us, device under 1 us")
    t_comp = STEP_MFLOP * 1e6 / (FP32_TFLOPS * 1e12) * 1e6
    t_mem = STEP_MB * 1e6 / (HBM_TBPS * 1e12) * 1e6
    t_host = PY_ENTRIES * T_DISPATCH_US + LAUNCHES * T_LAUNCH_US
    show("arithmetic (us)", f"{t_comp:.2f}")
    show("memory (us)", f"{t_mem:.2f}")
    show("host (us)", f"{t_host:.0f}")
    show("host / (arithmetic + memory)", f"{t_host / (t_comp + t_mem):.0f}x")
    mt = load_mnist_times()
    show("measured eager step, batch 32 (ms)", f"{mt[32] / 1e3:.2f}")
    show("measured eager step, batch 1024 (ms)", f"{mt[1024] / 1e3:.2f}")
    show("images per step, 1024 / 32", f"{1024 / 32:.0f}x")
    show("step time, 1024 / 32", f"{mt[1024] / mt[32]:.2f}")

    head("Slide: the orbiter and the framework (close)")
    show("host vs read (ms)", f"{host_ms:.1f} vs {read_ms:.1f}")
    show("training memory vs weights", f"{lo_gb:.1f} to {hi_gb:.1f} GB vs {w_mb:.1f} MB")
    show("PCIe below HBM", f"{HBM_GBPS / PCIE_GBPS:.0f}x")


if __name__ == "__main__":
    main()
