#!/usr/bin/env python3
"""Every derived number in Lecture 5 (Neural Computation), computed from sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports these functions, so the figures and the
slides share one computation.

The chapter's own calculation cells are executed as well (chapter()), for
two jobs only: they supply the MLSysIM hardware specs the roofline and the
epoch estimate need, and they give the chapter's published values, which
check() compares against this file's independent arithmetic. Any
disagreement prints a line starting with DISAGREES.

Measured values come from the real runs in this folder:
  data/torch_runs.txt   (torch_runs.py: autograd saved tensors; profiler)
  data/overfit.json     (overfit_run.py: MNIST training against held-out)

Run from this folder with the MLSysBook venv (it imports mlsysim):
  /Users/VJ/GitHub/MLSysBook/.venv/bin/python calcs.py
"""

import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parents[3]                     # repository root (holds books/ and mlsysim/)

_CH = None
PROBLEMS = []


def chapter():
    """Execute the chapter's calculation cells once; return their namespace."""
    global _CH
    if _CH is not None:
        return _CH
    import matplotlib
    import matplotlib.pyplot as plt
    for d in (ROOT, ROOT / "mlsysim"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    qmd = next((ROOT / "books" / "vol1" / "05_nn_computation").glob("*.qmd"))
    cells = re.findall(r"```\{python\}\n(.*?)```", qmd.read_text(), re.S)
    saved = dict(matplotlib.rcParams)
    here = os.getcwd()
    os.chdir(ROOT)
    ns = {}
    try:
        for cell in cells:
            if "plt.show(" in cell or "setup_plot(" in cell:
                continue  # figure cells; only the calculations are needed
            exec(cell, ns)
    finally:
        os.chdir(here)
        plt.close("all")
        matplotlib.rcParams.update(saved)
    _CH = ns
    return ns


def show(label, value):
    print(f"  {label}: {value}")


def check(label, ours, book, tol=1e-6):
    """Compare this file's value with the chapter's published one."""
    ok = math.isclose(float(ours), float(book), rel_tol=tol, abs_tol=tol)
    if not ok:
        PROBLEMS.append(f"{label}: calcs {ours} vs chapter {book}")
        print(f"  DISAGREES {label}: calcs {ours} vs chapter {book}")


# ---------------------------------------------------------------------------
# Inputs: the running network (chapter Sec. 5.1 and Table 5)
WIDTHS = [784, 128, 64, 10]     # 28x28 input, two hidden layers, 10 classes
BATCH = 32                      # chapter Napkin Math 5.1
BYTES_FP32 = 4                  # IEEE 754 single precision
TRAIN_IMAGES = 60_000           # MNIST training set (chapter Sec. 5.4.3)


def network():
    w = [a * b for a, b in zip(WIDTHS[:-1], WIDTHS[1:])]
    b = WIDTHS[1:]
    return {"weights": w, "biases": b, "W": sum(w), "B": sum(b), "P": sum(w) + sum(b)}


TEST_IMAGES = 10_000            # MNIST test set (LeCun, Cortes, Burges)


def slide_hook():
    """Slide 'Reading one handwritten digit takes 109,184 multiply-adds'."""
    n = network()
    macs = n["W"]                       # one MAC per weight per example
    return {"macs": macs, "flops": 2 * macs, "mnist_total": TRAIN_IMAGES + TEST_IMAGES}


def slide_ladder():
    """Slide 'Learned features cost 1,100 times the work of rules'.
    Inputs: about 100 comparisons (rules) and about 8,000 operations
    (HOG + linear SVM), chapter Sec. 5.2, illustrative implementations."""
    rules, hog = 100, 8000
    macs = slide_hook()["macs"]
    return {"rules": rules, "hog": hog, "net": macs, "net_over_rules": macs / rules,
            "hog_over_rules": hog / rules}


def slide_neuron():
    """Slide 'One neuron is a dot product...': a first-layer unit sees 784 pixels."""
    return {"unit_macs": WIDTHS[0], "layer1_macs": WIDTHS[0] * WIDTHS[1]}


def slide_xor():
    """Slide 'Two ReLU units compute XOR' (chapter Example 5.2)."""
    relu = lambda v: max(0.0, v)
    rows = []
    for x1, x2 in ((0, 0), (0, 1), (1, 0), (1, 1)):
        h1, h2 = relu(x1 + x2), relu(x1 + x2 - 1)
        rows.append((x1, x2, h1, h2, h1 - 2 * h2))
    return {"rows": rows}


def slide_slopes():
    """Slide 'Predict: how much gradient survives 20 sigmoid layers?'.
    Input: the sigmoid's slope peaks at 0.25 (chapter Sec. 5.3.2)."""
    return {"survive_20": 0.25 ** 20}


def slide_alexnet():
    """Slide 'Predict: how much gradient survives...', step 3. Inputs read off
    Krizhevsky et al. 2012, Figure 1: the ReLU curve reaches 25 percent
    training error at epoch 6, the tanh curve at epoch 36."""
    relu_ep, tanh_ep = 6, 36
    return {"relu_ep": relu_ep, "tanh_ep": tanh_ep, "ratio": tanh_ep / relu_ep}


def slide_transistor():
    """Slide on activation cost. Inputs: ReLU about 50 and an exponential
    datapath about 2,500 transistor equivalents, chapter Sec. 5.3.2, a
    stylized teaching ratio (illustrative)."""
    relu, sig = 50, 2500
    return {"relu": relu, "sigmoid": sig, "ratio": sig / relu}


def slide_layer():
    """Slide 'A layer is one matrix multiply; a batch reuses each weight'."""
    return {"layer1_b32_macs": BATCH * WIDTHS[0] * WIDTHS[1]}


def slide_count():
    """Slide 'Your turn: what does the MNIST network weigh?'."""
    n = network()
    bytes_ = n["P"] * BYTES_FP32
    side2 = 56 * 56                     # chapter self-check: a 56x56 input
    return {**n, "bytes": bytes_, "kb": bytes_ / 1000,
            "layer1_share": n["weights"][0] / n["W"],
            "w1_56": side2 * WIDTHS[1]}


def macs_of(widths):
    return sum(a * b for a, b in zip(widths[:-1], widths[1:]))


def slide_depth_width():
    """Slide 'Width costs quadratically, depth costs linearly' (chapter
    Sec. 5.3.5, Figure 12): two hidden layers swept in width; width 512 swept
    in depth; saved activations = input + every layer output."""
    inp, out = WIDTHS[0], WIDTHS[-1]
    net = lambda L, w: [inp] + [w] * L + [out]
    wid = macs_of(net(2, 4096)) / macs_of(net(2, 2048))
    dep = macs_of(net(32, 512)) / macs_of(net(16, 512))
    ret = sum(net(32, 512)) / sum(net(16, 512))
    return {"width_double": wid, "depth_double": dep, "retained_double": ret}


def fwd_flops(widths, batch=BATCH):
    """Chapter Napkin Math 5.1: 2*B*n_in*n_out per matrix multiply, and two
    operations per output value for the bias add and the activation."""
    mm = [2 * batch * a * b for a, b in zip(widths[:-1], widths[1:])]
    other = [2 * batch * b for b in widths[1:]]
    return mm, other


def slide_fwd_ops():
    """Slide 'Matrix multiply is over 99 percent of the forward pass'."""
    mm, other = fwd_flops(WIDTHS)
    total = sum(mm) + sum(other)
    dmm, doth = fwd_flops([784, 256, 128, 10])
    dtotal = sum(dmm) + sum(doth)
    return {"mm": mm, "other": other, "other_total": sum(other), "total": total,
            "per_digit": total / BATCH, "mm_share": sum(mm) / total,
            "other_share": sum(other) / total, "layer1_share_of_mm": mm[0] / sum(mm),
            "double_total": dtotal, "double_ratio": dtotal / total}


def load_profile():
    """Parse data/torch_runs.txt (the profiler table from torch_runs.py)."""
    txt = (DATA / "torch_runs.txt").read_text()
    rows = {}
    for m in re.finditer(r"^\s*(aten::\w+)\s+([\d.]+)%\s+([\d.]+)(ms|us)\s+([\d.]+)%"
                         r"\s+([\d.]+)(ms|us)\s+([\d.]+)(ms|us)\s+(\d+)", txt, re.M):
        name = m.group(1)
        to_us = lambda v, u: float(v) * (1000 if u == "ms" else 1)
        rows[name] = {"self_pct": float(m.group(2)), "self_us": to_us(m.group(3), m.group(4)),
                      "avg_us": to_us(m.group(8), m.group(9)), "calls": int(m.group(10))}
    tot = re.search(r"Self CPU time total: ([\d.]+)(ms|us)", txt)
    total_us = float(tot.group(1)) * (1000 if tot.group(2) == "ms" else 1)
    passes = 200                                  # torch_runs.py: 200 profiled forward passes
    head = txt.splitlines()[0]
    return rows, total_us, passes, head


def slide_profile():
    """New slide: the profiler on the running network at batch 32."""
    rows, total_us, passes, head = load_profile()
    mm, _ = fwd_flops(WIDTHS)
    per_pass_us = total_us / passes
    addmm = rows["aten::addmm"]
    # aten::addmm, aten::clamp_min (the ReLU's arithmetic): the kernels that compute
    compute_pct = addmm["self_pct"] + rows["aten::clamp_min"]["self_pct"]
    return {"versions": head, "per_pass_us": per_pass_us, "addmm_pct": addmm["self_pct"],
            "addmm_calls_per_pass": addmm["calls"] / passes, "addmm_avg_us": addmm["avg_us"],
            "relu_kernel_pct": rows["aten::clamp_min"]["self_pct"],
            "bookkeeping_pct": 100 - compute_pct,
            "addmm_rate_gflops": sum(mm) / (addmm["self_us"] / passes) / 1e3}


def traffic_bytes(widths, batch):
    """Chapter Napkin Math 5.1 traffic model: every weight and bias once;
    input and output once; each hidden activation written and read back."""
    n = network()
    hidden = sum(widths[1:-1])
    return n["P"] * BYTES_FP32 + batch * (widths[0] + 2 * hidden + widths[-1]) * BYTES_FP32


def slide_intensity():
    """Slide on arithmetic intensity, and the predict slide after it."""
    f = slide_fwd_ops()
    b1 = traffic_bytes(WIDTHS, 1)
    b32 = traffic_bytes(WIDTHS, BATCH)
    i1 = f["per_digit"] / b1
    i32 = f["total"] / b32
    return {"bytes_b1": b1, "bytes_b32": b32, "i1": i1, "i32": i32, "ratio": i32 / i1}


def machines():
    """MLSysIM registry via the chapter: the reference desktop CPU and the H100."""
    CH = chapter()
    H = CH["Hardware"]
    TF = CH["TFLOPs"] / CH["second"]
    TBs = CH["TB"] / CH["second"]
    out = {}
    for key, hw in (("cpu", H.Cloud.ReferenceCPU), ("acc", H.Cloud.H100)):
        out[key] = {"peak_tflops": hw.compute.precision_flops["fp32"].to(TF).magnitude,
                    "bw_tbs": hw.memory.bandwidth.to(TBs).magnitude}
    return out


def slide_roofline():
    """Slides 'Predict: how much faster per image at batch 32?' and the roofline."""
    I = slide_intensity()
    out = {}
    for k, m in machines().items():
        r1 = min(m["peak_tflops"], m["bw_tbs"] * I["i1"])
        r32 = min(m["peak_tflops"], m["bw_tbs"] * I["i32"])
        out[k] = {"ridge": m["peak_tflops"] / m["bw_tbs"], "rate_b1": r1, "rate_b32": r32,
                  "speedup": r32 / r1, **m}
    out["mean_ridge"] = (out["cpu"]["ridge"] + out["acc"]["ridge"]) / 2
    return out


def slide_epoch_steps():
    """Slide 'Training repeats one loop'."""
    return {"steps": TRAIN_IMAGES // BATCH}


def slide_cross_entropy():
    """Slide 'Cross-entropy punishes confident mistakes hardest'."""
    return {p: -math.log(p) for p in (0.9, 0.8, 0.1)} | {"uniform": math.log(10)}


# Napkin Math 5.2 inputs: x = (1.0, 0.5); W1 columns feed h1 and h2;
# W2 = (0.6, -0.4); biases zero; squared-error loss; target 1.0; lr 0.1.
X = (1.0, 0.5)
W1 = ((0.5, -0.3), (0.8, 0.2))     # W1[i][j]: input i to hidden unit j
W2 = (0.6, -0.4)
TARGET, LR = 1.0, 0.1


def slide_backprop():
    """Slides 'Forward pass: compute each value' and 'Backward pass'."""
    z = [X[0] * W1[0][j] + X[1] * W1[1][j] for j in range(2)]
    a = [max(0.0, v) for v in z]
    yhat = a[0] * W2[0] + a[1] * W2[1]
    loss = 0.5 * (yhat - TARGET) ** 2
    err = yhat - TARGET
    gW2 = [a[j] * err for j in range(2)]
    back = [err * W2[j] for j in range(2)]
    delta = [back[j] if z[j] > 0 else 0.0 for j in range(2)]
    gW1 = [[X[i] * delta[j] for j in range(2)] for i in range(2)]
    w11_new = W1[0][0] - LR * gW1[0][0]
    return {"z": z, "a": a, "yhat": yhat, "loss": loss, "err": err, "gW2": gW2,
            "back": back, "delta": delta, "gW1": gW1, "w11_new": w11_new}


def slide_train_flops():
    """Slide 'A training step here costs 2.08 forward passes, not 3'
    (chapter Sec. 5.4.5, Napkin Math 5.3): forward matrix multiplies, the same
    again for weight gradients, and input gradients for layers 2 and 3 only."""
    mm, _ = fwd_flops(WIDTHS)
    fwd = sum(mm)
    wgrad = fwd
    igrad = sum(mm[1:])
    total = fwd + wgrad + igrad
    return {"fwd": fwd, "wgrad": wgrad, "igrad": igrad, "total": total, "ratio": total / fwd}


def slide_epoch():
    """Slide 'Predict: CPU or accelerator, which finishes first?' (Napkin Math 5.3).
    Inputs: 12 launched operations per step (chapter: 3 forward layers, loss,
    5 backward products, 2 ReLU masks, 1 fused update); Python dispatch 10 us
    and kernel launch 5 us (MLSysIM Ops.RuntimeOverheads); rates from the
    roofline at batch-32 intensity."""
    CH = chapter()
    us = CH["microsecond"]
    dispatch = CH["Ops"].RuntimeOverheads.PythonDispatch.to(us).magnitude
    launch = CH["Ops"].RuntimeOverheads.KernelLaunch.to(us).magnitude
    ops = 3 + 1 + 5 + 2 + 1
    steps = slide_epoch_steps()["steps"]
    # The chapter scales the forward FLOPs per image (2 x 109,184) by the 2.08 ratio.
    T = slide_train_flops()
    step_flops = T["ratio"] * slide_hook()["flops"] * BATCH
    R = slide_roofline()
    work = {k: steps * step_flops / (R[k]["rate_b32"] * 1e12) * 1e3 for k in ("cpu", "acc")}
    ovh = {"cpu": steps * ops * dispatch / 1e3, "acc": steps * ops * (dispatch + launch) / 1e3}
    tot = {k: work[k] + ovh[k] for k in work}
    return {"dispatch_us": dispatch, "launch_us": launch, "ops": ops, "step_mflop": step_flops / 1e6,
            "work_ms": work, "ovh_ms": ovh, "total_ms": tot,
            "work_speedup": work["cpu"] / work["acc"], "ovh_share_acc": ovh["acc"] / tot["acc"]}


def slide_ranges():
    """Slide 'Vote: a gradient of 9.1e-13 is stored in FP16': IEEE 754 limits."""
    import numpy as np
    f16 = np.finfo(np.float16)
    f32 = np.finfo(np.float32)
    g = slide_slopes()["survive_20"]
    return {"fp16_min_normal": float(f16.tiny), "fp16_min_sub": float(f16.smallest_subnormal),
            "fp16_max": float(f16.max), "fp32_min_normal": float(f32.tiny),
            "fp32_max": float(f32.max), "int16_max": int(np.iinfo(np.int16).max),
            "grad_in_fp16": float(np.float16(g)), "grad_in_fp32": float(np.float32(g))}


def slide_memory():
    """Slide 'Training holds 3.4 times the memory of inference' (chapter Table 9).
    Inputs: training keeps input and every layer output for the batch;
    inference keeps the largest adjacent pair (rolling buffers); Adam keeps
    2 values per parameter (chapter Sec. 5.4.5)."""
    n = network()
    params = n["P"] * BYTES_FP32
    act_train = BATCH * sum(WIDTHS) * BYTES_FP32
    pairs = [a + b for a, b in zip(WIDTHS[:-1], WIDTHS[1:])]
    act_inf = BATCH * max(pairs) * BYTES_FP32
    grads = params
    adam = 2 * params
    train = params + act_train + grads + adam
    infer = params + act_inf
    return {"params_kb": params / 1000, "act_train_kb": act_train / 1000,
            "act_inf_kb": act_inf / 1000, "grads_kb": grads / 1000, "adam_kb": adam / 1000,
            "train_kb": train / 1000, "infer_kb": infer / 1000, "ratio": train / infer}


def slide_memory_trace():
    """New slide: the PyTorch memory-profiler timeline (ResNet-50, batch 1,
    SGD with momentum; DeVito and Shi, PyTorch blog, 2023). Input: ResNet-50
    has 25.6 million parameters (MLSysIM, chapter Lighthouse 5.1)."""
    p = 25.6e6
    weights = p * BYTES_FP32
    return {"weights_mb": weights / 1e6, "weights_gib": weights / 2 ** 30,
            "momentum_gib": weights / 2 ** 30}   # SGD momentum: one value per parameter


def slide_gpt2():
    """Slide 'Estimate: does GPT-2 training fit in 80 GB?' (Lighthouse 5.1).
    Inputs: GPT-2 1.5 billion parameters; ResNet-50 25.6 million (MLSysIM)."""
    p = 1.5e9
    w = p * BYTES_FP32
    return {"weights_gb": w / 1e9, "train_gb": 4 * w / 1e9,
            "resnet_mb": 25.6e6 * BYTES_FP32 / 1e6, "jump": p / network()["P"]}


def slide_reject():
    """Slide 'A threshold sends uncertain digits to people' (LeCun et al. 1989,
    chapter Table 11): 12.1 percent rejected."""
    return {"accepted_pct": 100 - 12.1}


def slide_overfit():
    """Open-questions slide: the real training run (data/overfit.json)."""
    d = json.loads((DATA / "overfit.json").read_text())
    out = {"setup": d["setup"], "versions": d["versions"]}
    for n, h in d["runs"].items():
        out[n] = {"train": h["train_acc"][-1] * 100, "test": h["test_acc"][-1] * 100,
                  "gap": (h["train_acc"][-1] - h["test_acc"][-1]) * 100,
                  "epochs": len(h["train_acc"]),
                  "first_100": next((i + 1 for i, v in enumerate(h["train_acc"]) if v >= 1.0), None)}
    return out


# ---------------------------------------------------------------------------
def main():
    CH = chapter()
    print("Lecture 5 derived numbers\n")

    h = slide_hook()
    print("Reading one handwritten digit takes 109,184 multiply-adds")
    show("MACs per digit", f"{h['macs']:,}")
    show("FLOPs per digit (1 MAC = 2 FLOPs)", f"{h['flops']:,}")
    show("MNIST images, training + test", f"{h['mnist_total']:,}")
    check("MACs", h["macs"], CH["MNISTInference"].total_macs)

    L = slide_ladder()
    print("Learned features cost 1,100 times the work of rules")
    show("network / rules", f"{L['net_over_rules']:,.0f}x (slide rounds to 1,100x)")
    show("HOG / rules", f"{L['hog_over_rules']:.0f}x")
    check("ladder ratio", L["net_over_rules"], CH["ParadigmSystemsCost"].dl_ops_ratio)

    N = slide_neuron()
    print("One neuron is a dot product followed by an activation")
    show("MACs, one first-layer unit", N["unit_macs"])
    show("MACs, all 128 first-layer units", f"{N['layer1_macs']:,}")

    print("Two ReLU units compute XOR (h1, h2, y)")
    for r in slide_xor()["rows"]:
        show(f"x = ({r[0]}, {r[1]})", f"h1 {r[2]:g}, h2 {r[3]:g}, y {r[4]:g}")

    S = slide_slopes()
    print("Predict: how much gradient survives 20 sigmoid layers?")
    show("0.25^20", f"{S['survive_20']:.2e}")
    check("0.25^20", S["survive_20"], CH["EarlySystemsClaims"].gradient_scale)

    A = slide_alexnet()
    show("AlexNet Figure 1: tanh epochs / ReLU epochs to 25% error", f"{A['ratio']:.0f}x")

    T = slide_transistor()
    print("A sigmoid needs an exponential; a ReLU needs a comparison")
    show("sigmoid / ReLU transistor equivalents (illustrative)", f"{T['ratio']:.0f}x")

    print("A layer is one matrix multiply; a batch reuses each weight")
    show("layer 1 MACs at batch 32", f"{slide_layer()['layer1_b32_macs']:,} (about 3.2 million)")

    C = slide_count()
    print("Your turn: what does the MNIST network weigh?")
    show("weights per layer", ", ".join(f"{w:,}" for w in C["weights"]))
    show("weights", f"{C['W']:,}")
    show("biases", C["B"])
    show("parameters", f"{C['P']:,}")
    show("bytes (FP32)", f"{C['bytes']:,} = {C['kb']:.1f} KB")
    show("layer 1 share of weights", f"{100 * C['layer1_share']:.1f}%")
    show("first-layer weights for a 56x56 input", f"{C['w1_56']:,}")
    check("parameters", C["P"], CH["MNISTInference"].total_params)
    check("layer 1 share", C["layer1_share"], CH["DamMnistDataAxis"].w1_share)

    D = slide_depth_width()
    print("Width costs quadratically, depth costs linearly")
    show("width 2,048 -> 4,096, MACs", f"{D['width_double']:.1f}x")
    show("depth 16 -> 32, MACs", f"{D['depth_double']:.1f}x")
    show("depth 16 -> 32, saved activations", f"{D['retained_double']:.1f}x")
    check("width double", D["width_double"], CH["DepthWidthCost"].width_double_factor)
    check("depth double", D["depth_double"], CH["DepthWidthCost"].depth_double_factor)
    check("retained double", D["retained_double"], CH["DepthWidthCost"].retained_double_factor)

    F = slide_fwd_ops()
    print("Matrix multiply is over 99 percent of the forward pass (batch 32)")
    show("layer MatMul FLOPs", ", ".join(f"{v:,}" for v in F["mm"]))
    show("bias + activation, all layers", f"{F['other_total']:,}")
    show("total operations", f"{F['total']:,} (about 7 million)")
    show("per digit", f"{F['per_digit']:,.0f} (about 219 thousand)")
    show("MatMul share", f"{100 * F['mm_share']:.1f}%")
    show("bias + activation share", f"{100 * F['other_share']:.1f}%")
    show("layer 1 share of the MatMul work", f"{100 * F['layer1_share_of_mm']:.1f}%")
    show("doubled hidden widths (784-256-128-10): total", f"{F['double_total']:,} (about 15 million)")
    show("doubled hidden widths: ratio", f"{F['double_ratio']:.2f}x (about 2.1)")
    check("forward total", F["total"], CH["MnistFlopsCalc"].total_flops_value)
    check("double ratio", F["double_ratio"], CH["MnistFlopsCalc"].double_ratio_value)

    P = slide_profile()
    print("New: the profiler on the running network (data/torch_runs.txt)")
    show("run", P["versions"])
    show("time per forward pass", f"{P['per_pass_us']:.1f} us")
    show("aten::addmm share of self CPU time", f"{P['addmm_pct']:.1f}%")
    show("addmm calls per forward pass", f"{P['addmm_calls_per_pass']:.0f}")
    show("addmm average per call", f"{P['addmm_avg_us']:.1f} us")
    show("ReLU kernel (clamp_min) share", f"{P['relu_kernel_pct']:.1f}%")
    show("everything else (dispatch and bookkeeping)", f"{P['bookkeeping_pct']:.0f}%")
    show("addmm achieved rate", f"{P['addmm_rate_gflops']:.0f} GFLOP/s")

    I = slide_intensity()
    print("Batching raises a dense layer's arithmetic intensity")
    show("bytes moved, batch 1", f"{I['bytes_b1']:,} B = {I['bytes_b1'] / 1000:.0f} KB")
    show("bytes moved, batch 32", f"{I['bytes_b32']:,} B = {I['bytes_b32'] / 1000:.0f} KB")
    show("intensity, batch 1", f"{I['i1']:.2f} FLOP/byte (about 0.5)")
    show("intensity, batch 32", f"{I['i32']:.1f} FLOP/byte")
    show("intensity ratio", f"{I['ratio']:.0f}x")
    check("intensity b1", I["i1"], CH["MnistFlopsCalc"].single_image_arith_intensity)
    check("intensity b32", I["i32"], CH["MnistFlopsCalc"].batch_arith_intensity)

    R = slide_roofline()
    print("Predict / roofline: attainable FP32 rate on the bandwidth slope")
    for k, name in (("cpu", "desktop CPU"), ("acc", "H100")):
        r = R[k]
        show(f"{name}: peak, bandwidth", f"{r['peak_tflops']:.1f} TFLOP/s, {r['bw_tbs']:.3f} TB/s")
        show(f"{name}: ridge", f"{r['ridge']:.1f} FLOP/byte")
        show(f"{name}: batch 1 rate", f"{r['rate_b1'] * 1000:,.0f} GFLOP/s = {r['rate_b1']:.1f} TFLOP/s")
        show(f"{name}: batch 32 rate", f"{r['rate_b32'] * 1000:,.0f} GFLOP/s = {r['rate_b32']:.1f} TFLOP/s")
        show(f"{name}: speedup per image", f"{r['speedup']:.0f}x")
    show("mean ridge", f"{R['mean_ridge']:.0f} FLOP/byte")

    print("Training repeats one loop")
    show("steps per epoch", f"{slide_epoch_steps()['steps']:,}")

    X_ = slide_cross_entropy()
    print("Cross-entropy punishes confident mistakes hardest")
    for p in (0.9, 0.8, 0.1):
        show(f"-ln {p}", f"{X_[p]:.2f}")
    show("ln 10 (uniform guess)", f"{X_['uniform']:.2f}")

    B = slide_backprop()
    print("Forward and backward pass, Napkin Math 5.2")
    show("z", [round(v, 4) for v in B["z"]])
    show("a", [round(v, 4) for v in B["a"]])
    show("y-hat", f"{B['yhat']:.2f}")
    show("loss", f"{B['loss']:.4f}")
    show("output error", f"{B['err']:.2f}")
    show("dL/dW2", [round(v, 4) for v in B["gW2"]])
    show("error sent back", [round(v, 4) for v in B["back"]])
    show("deltas after ReLU mask", [round(v, 4) for v in B["delta"]])
    show("dL/dW1", [[round(v, 4) for v in r] for r in B["gW1"]])
    show("w11 after one step", f"{B['w11_new']:.4f}")

    Tf = slide_train_flops()
    print("A training step here costs 2.08 forward passes, not 3")
    show("forward", f"{Tf['fwd']:,} = {Tf['fwd'] / 1e6:.2f} M")
    show("weight gradients", f"{Tf['wgrad']:,} = {Tf['wgrad'] / 1e6:.2f} M")
    show("input gradients", f"{Tf['igrad']:,} = {Tf['igrad'] / 1e6:.2f} M")
    show("total", f"{Tf['total']:,} = {Tf['total'] / 1e6:.1f} M")
    show("ratio", f"{Tf['ratio']:.2f}x")
    check("training ratio", Tf["ratio"], CH["MnistTrainingArithmetic"].training_ratio)

    E = slide_epoch()
    print("Predict: CPU or accelerator, which finishes first? (one epoch)")
    show("FLOPs per step", f"{E['step_mflop']:.1f} MFLOP")
    show("work, CPU", f"{E['work_ms']['cpu']:.0f} ms")
    show("work, accelerator", f"{E['work_ms']['acc']:.1f} ms")
    show("work speedup", f"{E['work_speedup']:.0f}x")
    show("overhead, CPU", f"{E['ovh_ms']['cpu']:.1f} ms")
    show("overhead, accelerator", f"{E['ovh_ms']['acc']:.1f} ms")
    show("total, CPU", f"{E['total_ms']['cpu']:.0f} ms")
    show("total, accelerator", f"{E['total_ms']['acc']:.0f} ms")
    show("overhead share, accelerator", f"{100 * E['ovh_share_acc']:.1f}%")
    check("overhead share", E["ovh_share_acc"], CH["EpochTimeEnergy"].ovh_share_acc)
    check("work speedup", round(E["work_speedup"]), CH["EpochTimeEnergy"].acc_work_speedup)

    G = slide_ranges()
    print("Vote: a gradient of 9.1e-13 is stored in FP16")
    show("FP16 smallest normal", f"{G['fp16_min_normal']:.2e}")
    show("FP16 smallest subnormal", f"{G['fp16_min_sub']:.2e}")
    show("FP16 max", f"{G['fp16_max']:,.0f}")
    show("FP32 smallest normal", f"{G['fp32_min_normal']:.2e}")
    show("FP32 max", f"{G['fp32_max']:.2e}")
    show("INT16 max", f"{G['int16_max']:,}")
    show("9.1e-13 stored in FP16", G["grad_in_fp16"])
    show("9.1e-13 stored in FP32", f"{G['grad_in_fp32']:.2e}")

    M = slide_memory()
    print("Training holds 3.4 times the memory of inference")
    for k in ("params_kb", "act_train_kb", "act_inf_kb", "grads_kb", "adam_kb", "train_kb", "infer_kb"):
        show(k, f"{M[k]:.1f} KB")
    show("training total", f"{M['train_kb'] / 1000:.1f} MB")
    show("ratio", f"{M['ratio']:.1f}x")
    check("memory ratio", M["ratio"], CH["MnistTrainingMemoryCalc"].training_ratio)
    check("training KB", M["train_kb"], CH["MnistTrainingMemoryCalc"].training_total_kb)
    check("inference KB", M["infer_kb"], CH["MnistTrainingMemoryCalc"].inference_total_kb)

    Mt = slide_memory_trace()
    print("New: the memory-profiler timeline (ResNet-50, batch 1, SGD momentum)")
    show("ResNet-50 weights", f"{Mt['weights_mb']:.1f} MB = {Mt['weights_gib']:.3f} GiB")
    show("momentum buffer", f"{Mt['momentum_gib']:.3f} GiB")

    Gp = slide_gpt2()
    print("Estimate: does GPT-2 training fit in 80 GB?")
    show("weights", f"{Gp['weights_gb']:.0f} GB")
    show("weights, gradients, 2 Adam values", f"{Gp['train_gb']:.0f} GB")
    show("ResNet-50 weights", f"{Gp['resnet_mb']:.1f} MB")
    show("GPT-2 / MNIST parameters", f"{Gp['jump']:,.0f}x (about 14,000)")

    print("A threshold sends uncertain digits to people")
    show("accepted", f"{slide_reject()['accepted_pct']:.1f}%")

    O = slide_overfit()
    print("Open questions: MNIST training against held-out (data/overfit.json)")
    show("setup", O["setup"])
    show("run", O["versions"])
    for n in ("1000", "60000"):
        o = O[n]
        show(f"{int(n):,} training images, after {o['epochs']} epochs",
             f"training {o['train']:.1f}%, held-out {o['test']:.1f}%, gap {o['gap']:.1f} points")
    show("1,000 images: first epoch at 100% training accuracy", O["1000"]["first_100"])

    print()
    if PROBLEMS:
        print(f"{len(PROBLEMS)} value(s) disagree with the chapter:")
        for p in PROBLEMS:
            print("  " + p)
    else:
        print("All checked values agree with the chapter's own calculation cells.")


if __name__ == "__main__":
    main()
