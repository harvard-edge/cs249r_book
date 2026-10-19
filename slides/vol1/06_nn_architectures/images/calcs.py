#!/usr/bin/env python3
"""Every derived number in Lecture 6 (Network Architectures, Part 1).

One block per slide, in deck order. Inputs carry their source as a comment
("ch" is the rendered chapter, Network Architectures); each printed line is
a value the slide or its note shows, under the label used there.
make_figs.py imports from this file so figures and slides share one
computation.

Run from this folder: python3 calcs.py
"""

import math


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
MLP = (784, 128, 64, 10)          # ch, Lecture 5 reference MNIST network
MNIST_HW = 28 * 28                # 28x28 image, pixel count (input)
N_FEAT_MNIST = 32                 # ch Example 6.1: 32 features
K3 = 3                            # 3x3 filter
WIDE = 2048                       # ch §6.2.5: production-width layer
FP32_B = 4                        # bytes per FP32 value
INT8_B = 1
DRAM_PJ_PER_BYTE = 160            # ch §6.2.5 (Horowitz 2014, 45 nm)
MAC_FP32_PJ = 4.6                 # ch §6.2.5 (Horowitz 2014, 45 nm)
IMG = 224                         # ch Definition 6.3: 224x224x64 feature map
C_IN = C_OUT = 64
BIG_IMG = 1024                    # note: "if the image grows to 1,024 by 1,024"
RESNET50_INT = 80.1               # ch Table 8, FLOP per FP32 weight byte
DENSE_INT = None                  # derived below (equation 2)
RESNET50_GFLOP = 8.2              # ch Napkin Math 6.3 / Lighthouse 6.1
MOBILENETV2_GFLOP = 0.6           # ch Table 8 (600 MFLOP)
FPS = 30                          # ch Napkin Math 6.3
GPU_PEAK_TFLOPS = 10              # ch Napkin Math 6.3
GPU_ACHIEVED_TFLOPS = 5           # ch Napkin Math 6.3: 50 to 60% of peak
DETECTOR_GFLOP = 100              # ch Napkin Math 6.3
GAIN = 0.9                        # ch §6.4.1: per-layer gradient factor
DEPTH = 50
KWS_PARAMS = 200_000              # ch Lighthouse 6.3, Table 10
RNN_IN, RNN_H = 100, 128          # ch §6.5 self-check
RNN_STEPS = 1000
LSTM_STATE = 512                  # ch §6.5.5: 512-wide FP32 state

# Keras Applications table (keras.io/api/applications, retrieved 2026-10-18):
# time (ms) per inference step, batch 32, mean of 30 batches x 10 repeats;
# CPU: AMD EPYC (92 core); GPU: Tesla A100.
KERAS = {"ResNet50": dict(params="25.6M", top1=74.9, cpu_ms=58.2, gpu_ms=4.6),
         "MobileNetV2": dict(params="3.5M", top1=71.3, cpu_ms=25.9, gpu_ms=3.8)}

# Arduino Nano 33 BLE Sense Rev2 (store.arduino.cc, tech specs, retrieved
# 2026-10-18): nRF52840, 1 MB flash, 256 KB SRAM (binary units).
FLASH_B = 1 * 1024 * 1024
SRAM_B = 256 * 1024

# Vaswani et al. 2017, Table 2: training cost (FLOPs), EN-DE, and BLEU.
GNMT_RL_FLOPS, GNMT_RL_BLEU = 2.3e19, 24.6
TRANSFORMER_BASE_FLOPS, TRANSFORMER_BASE_BLEU = 3.3e18, 27.3

# PyTorch 2.14.1, torch_shapes.py (run 2026-10-18): nn.Linear(784, 128)
# weights (128, 784) plus bias (128,); nn.LSTM(100, 128) weight_ih_l0 (512, 100).


def main():
    head("Recall: MNIST MLP multiply-adds")
    macs = sum(a * b for a, b in zip(MLP, MLP[1:]))
    show("MACs per digit", f"{macs:,}")

    head("Shared filter: 87x fewer weights per detector")
    dense_det = N_FEAT_MNIST * MNIST_HW
    conv_det = N_FEAT_MNIST * K3 * K3
    show("32 dense features (weights)", f"{dense_det:,}")
    show("32 shared 3x3 filters (weights)", f"{conv_det:,}")
    show("ratio", f"{dense_det / conv_det:.0f}x")

    head("Dense layer weights")
    w = MLP[0] * MLP[1]
    show("784 -> 128 weights", f"{w:,}")
    show("nn.Linear(784,128) parameters (weights + bias)", f"{w + MLP[1]:,}")
    w2 = WIDE * WIDE
    show("2048 -> 2048 weights", f"{w2:,} = {w2 / 1e6:.2f} million")
    show("2048 -> 2048 FP32 MB", f"{w2 * FP32_B / 1e6:.1f} MB")
    show("input doubled in width and height: weight factor", f"{(2 * 2):d}x")

    head("Four pixels into three neurons")
    x = [0.8, 0.2, 0.9, 0.1]
    W = [[0.5, 0.1, -0.2], [-0.3, 0.8, 0.4], [0.2, -0.4, 0.6], [0.7, 0.3, -0.1]]
    s = [sum(x[i] * W[i][j] for i in range(4)) for j in range(3)]
    show("terms of neuron 1", ", ".join(f"{x[i] * W[i][0]:+.2f}" for i in range(4)))
    show("sums", ", ".join(f"{v:.2f}" for v in s))
    show("ReLU", ", ".join(f"{max(0.0, v):.2f}" for v in s))

    head("Dense intensity at batch one")
    i_dense = 2 / FP32_B
    show("I (FLOP/byte)", f"{i_dense}")
    show("batch for 100 FLOP/byte", f"{100 / i_dense:.0f}")

    head("Energy: moving a weight vs. using it")
    e_w = DRAM_PJ_PER_BYTE * FP32_B
    show("read FP32 weight (pJ)", f"{e_w}")
    show("ratio to one FP32 MAC", f"{e_w / MAC_FP32_PJ:.0f}x")

    head("Shift and receptive field")
    show("three 3x3 layers see", f"{1 + 3 * (K3 - 1)}x{1 + 3 * (K3 - 1)}")
    show("3x3 layers to see 21x21", f"{(21 - 1) // (K3 - 1)}")

    head("AlexNet first layer (Krizhevsky et al. 2012, figure 3)")
    # 96 kernels of 11x11x3 (paper, figure 3 caption)
    show("conv1 weights", f"{96 * 11 * 11 * 3:,}")

    head("Weight sharing: 205 million vs 36,864")
    dense_w = IMG * IMG * C_IN * C_OUT
    conv_w = K3 * K3 * C_IN * C_OUT
    show("dense weights", f"{dense_w:,} (about {dense_w / 1e6:.1f} million)")
    show("conv weights", f"{conv_w:,}")
    show("ratio", f"{dense_w / conv_w:,.0f}x")
    show("dense growth at 1024x1024", f"{(BIG_IMG / IMG) ** 2:.0f}x")

    head("Conv reuse and activations")
    pos = IMG * IMG
    show("positions per filter weight", f"{pos:,}")
    show("ResNet-50 / dense intensity", f"{RESNET50_INT / i_dense:.0f}x")
    layer_macs = pos * conv_w
    show("example layer MACs", f"{layer_macs / 1e9:.2f} billion")
    wbytes = conv_w * FP32_B
    show("example layer weight bytes", f"{wbytes / 1e3:.1f} KB")
    outs = pos * C_OUT
    show("output values", f"{outs / 1e6:.1f} million")
    show("activations FP32", f"{outs * FP32_B / 1e6:.1f} MB")
    show("activations / weights", f"{outs * FP32_B / wbytes:.0f}x")

    head("im2col miniature")
    n_in, k, out_hw = 3, 2, 3 - 2 + 1
    show("input values", n_in * n_in)
    show("unfolded entries", (out_hw ** 2) * k * k)
    show("max copies of a pixel, 3x3 filter", K3 * K3)

    head("unfold on a 224x224 image, 3x3 window, no padding")
    show("columns (windows)", f"{(IMG - K3 + 1) ** 2:,}")

    head("Estimate: 30 frames a second")
    need = FPS * RESNET50_GFLOP
    show("GFLOP/s needed", f"{need:.0f}")
    show("headroom", f"{GPU_ACHIEVED_TFLOPS * 1e3 / need:.1f}x")
    det = FPS * DETECTOR_GFLOP
    show("detector need (TFLOP/s)", f"{det / 1e3:.0f}")
    show("detector headroom", f"{GPU_ACHIEVED_TFLOPS * 1e3 / det:.1f}x")
    show("peak-based (wrong) frames/s", f"{GPU_PEAK_TFLOPS * 1e3 / RESNET50_GFLOP:,.0f}")

    head("Skip connections")
    show("0.9^50", f"{GAIN ** DEPTH:.3f}")
    add_bytes = 3 * FP32_B           # two operands read, one result written
    show("residual add FLOP/byte", f"{1 / add_bytes:.2f} (1 FLOP per {add_bytes} bytes)")

    head("Depthwise factoring")
    dw = K3 * K3 * C_IN
    pw = C_IN * C_OUT
    show("depthwise", f"{dw:,}")
    show("pointwise", f"{pw:,}")
    show("total", f"{dw + pw:,}")
    show("ratio", f"{conv_w / (dw + pw):.1f}x")
    frac = 1 / C_OUT + 1 / (K3 * K3)
    show("fraction 1/64 + 1/9", f"{frac:.4f} -> {1 / frac:.1f}x")
    show("limit with many channels", f"{K3 * K3}x")

    head("Check: 13.7x fewer FLOPs (Keras measured)")
    show("FLOP ratio ResNet-50 / MobileNetV2", f"{RESNET50_GFLOP / MOBILENETV2_GFLOP:.1f}x")
    r, m = KERAS["ResNet50"], KERAS["MobileNetV2"]
    show("CPU speedup", f"{r['cpu_ms'] / m['cpu_ms']:.1f}x")
    show("GPU (A100) speedup", f"{r['gpu_ms'] / m['gpu_ms']:.1f}x")
    show("top-1 gap (points)", f"{r['top1'] - m['top1']:.1f}")

    head("KWS on a Nano 33 BLE Sense")
    fp32 = KWS_PARAMS * FP32_B
    int8 = KWS_PARAMS * INT8_B
    show("FP32 weights", f"{fp32 / 1e3:.0f} KB")
    show("INT8 weights", f"{int8 / 1e3:.0f} KB")
    show("FP32 share of 1 MB flash", f"{100 * fp32 / FLASH_B:.0f}%")
    show("INT8 share of 1 MB flash", f"{100 * int8 / FLASH_B:.0f}%")
    show("FP32 weights / 256 KB SRAM", f"{fp32 / SRAM_B:.1f}x")
    show("INT8 weights / 256 KB SRAM", f"{100 * int8 / SRAM_B:.0f}%")

    head("Recurrent step")
    step = RNN_H * RNN_H + RNN_IN * RNN_H
    show("MACs per step", f"{step:,}")

    head("Predict: Transformer vs recurrent training cost (Vaswani Table 2)")
    show("GNMT+RL / Transformer base training FLOPs",
         f"{GNMT_RL_FLOPS / TRANSFORMER_BASE_FLOPS:.0f}x")
    show("BLEU gain", f"{TRANSFORMER_BASE_BLEU - GNMT_RL_BLEU:.1f}")

    head("LSTM")
    show("weight_ih rows (4 x hidden)", 4 * RNN_H)
    show("weight_hh rows (4 x hidden)", 4 * RNN_H)
    show("LSTM weights per layer (ih + hh)", f"{4 * RNN_H * (RNN_IN + RNN_H):,}")
    show("plain RNN weights per layer", f"{step:,}")
    show("LSTM / plain weights", f"{4 * RNN_H * (RNN_IN + RNN_H) / step:.0f}x")
    show("state, 512 wide FP32", f"{LSTM_STATE * FP32_B / 1024:.0f} KB")


def note_minutes():
    """Title-slide note: total note minutes and the total without [OPTIONAL]
    notes, summed from the deck's own [N min] tags."""
    import re
    from pathlib import Path
    tex = (Path(__file__).resolve().parents[1] / "06_nn_architectures.tex").read_text()
    notes = re.findall(r"\\(?:note|mlsysnote)\{\[(\d+(?:\.\d+)?) min\](.*?)\n", tex)
    total = sum(float(m) for m, _ in notes)
    core = sum(float(m) for m, t in notes if "[OPTIONAL" not in t)
    head("Title note: lecture length")
    show("notes total (min)", f"{total:g}")
    show("without [OPTIONAL] notes (min)", f"{core:g}")


if __name__ == "__main__":
    main()
    note_minutes()
