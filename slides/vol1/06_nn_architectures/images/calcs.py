#!/usr/bin/env python3
"""Every number on the Lecture 6 (Network Architectures, Part 1) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the layer scenarios
(Scenarios.MnistMlpLayerStack, ImageNet64ChannelConv, ElmanRnnStep, and the rest), the
model registry (ResNet-50, MobileNetV2, the DS-CNN keyword spotter), the
nRF52840 microcontroller, the 45 nm energy figures in Hardware.Tech, and the
cited results in Literature (Netflix Prize, Keras Applications, He et al. 2016,
Vaswani et al. 2017). The chapter reads the same scenarios. This file only
selects values, converts units, and formats them; it types no numbers. Each
value is recorded with v() under the key the deck prints as \\val{key};
running this file writes images/numbers.tex. make_figs.py imports this
module, so figures and slides share one computation.

Kept typed in the deck, by rule: the PyTorch console listings (recorded output
of images/torch_shapes.py), the toy drawings of Example 6.3 and the im2col
grid, wrong answer choices, years, depths listed in a figure legend, and
lecture and section numbers.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Models, Literature, Scenarios = mlsysim.Hardware, mlsysim.Models, mlsysim.Literature, mlsysim.Scenarios
units = mlsysim.core.units
FP32, FP16, INT8 = units.BYTES_FP32, units.BYTES_FP16, units.BYTES_INT8

NETFLIX = Literature.NetflixPrize
KERAS = Literature.KerasApplications
QUALITY = Literature.PredictionQuality
RUNS = Literature.TrainingRuns

MLP = Scenarios.MnistMlpLayerStack
L1 = Scenarios.MnistMlpFirstLayer
L1_BIG = Scenarios.MnistMlpFirstLayerDoubledInput
WIDE = Scenarios.Dense2048Layer
TOY = Scenarios.FourPixelThreeNeuronLayer
FIRST = Scenarios.MnistFirstConvLayer
SMALL_CNN = Scenarios.MnistSmallCnn
CONV = Scenarios.ImageNet64ChannelConv
CONV_BIG = Scenarios.ImageNet64ChannelConvAt1024
CONV_VALID = Scenarios.ImageNet64ChannelConvUnpadded
MINI = Scenarios.Im2colMiniature
ALEX = Scenarios.AlexNetFirstConv
RF = Scenarios.ThreeByThreeReceptiveField
GRAD = Scenarios.PlainStackGradientDecay
ADD = Scenarios.ResidualAddFp32
SHARE3 = Scenarios.ResNet50ConvKernelShare
STREAM = Scenarios.ResNet50ThirtyFpsStream
KWS32, KWS8 = Scenarios.KwsDsCnnOnNrf52840Fp32, Scenarios.KwsDsCnnOnNrf52840Int8
RNN, LSTM, STATE = Scenarios.ElmanRnnStep, Scenarios.LstmStep, Scenarios.RecurrentState512Fp32
RESNET, MOBILENET, KWS = Models.Vision.ResNet50, Models.Vision.MobileNetV2, Models.Tiny.DS_CNN
MCU = Hardware.Tiny.nRF52840


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def intensity(workload):
    """FLOPs per FP32 weight byte (the chapter's weight-only proxy, Table 8)."""
    return m(workload.inference_flops / workload.size_in_bytes(FP32), "flop/byte")


def mantissa_exponent(x):
    e = math.floor(math.log10(x))
    return x / 10 ** e, e


print("== Hook: the Netflix Prize (Literature.NetflixPrize)")
v("prize_musd", NETFLIX.GrandPrizeUsd / 10 ** 6, "{:.0f}", "Grand Prize, millions of dollars")
v("target_pct", NETFLIX.TargetImprovementPct, "{:.0f}", "required improvement, %")
v("win_pct", NETFLIX.WinningImprovementPct, "{:.2f}", "leaderboard improvement of the winners, %")
v("blend_algos", NETFLIX.ProgressPrizeBlendAlgorithms, "{:.0f}", "algorithms in the 2007 Progress Prize blend")
v("progress_pct", NETFLIX.ProgressPrizeImprovementPct, "{:.2f}", "2007 Progress Prize improvement, %")
v("kept_algos_word", NETFLIX.ProgressPrizeAlgorithmsKept, "words", "algorithms Netflix kept, in prose")
v("contest_ratings_m", NETFLIX.ContestRatings / 10 ** 6, "{:.0f}", "contest ratings, millions")
v("prod_ratings_b", NETFLIX.ProductionRatings / 10 ** 9, "{:.0f}", "production ratings, billions")

print("== Recall: the MNIST network (Scenarios.MnistMlpLayerStack)")
v("mnist_side", mlsysim.Datasets.MNIST.image_width, "{:.0f}", "MNIST image side, pixels")
v("mlp_in", MLP.widths[0], "{:.0f}", "input width")
v("mlp_h1", MLP.widths[1], "{:.0f}", "first hidden width")
v("mlp_h2", MLP.widths[2], "{:.0f}", "second hidden width")
v("mlp_out", MLP.widths[3], "{:.0f}", "classes")
v("mnist_macs", MLP.macs_per_example_dense, "{:,.0f}", "multiply-adds per digit (= weights, biases aside)")

print("== Shared filter against dense units (Scenarios.MnistFirstConvLayer, MnistSmallCnn)")
v("first_feat", FIRST.out_channels, "{:.0f}", "first-layer features")
v("k", FIRST.kernel, "{:.0f}", "filter side")
v("dense_det", FIRST.dense_weights, "{:,.0f}", "weights of 32 dense features")
v("conv_det", FIRST.weights, "{:,.0f}", "weights of 32 shared 3x3 filters")
v("det_ratio", FIRST.sharing_ratio, "{:.0f}", "dense over shared, times")
v("cnn_weights", SMALL_CNN.weights, "{:,.0f}", "small CNN weights")
v("head_pct", 100 * SMALL_CNN.dense_share, "{:.0f}", "dense head share of the CNN's weights, %")

print("== Dense layer (Scenarios.FourPixelThreeNeuronLayer, MnistMlpFirstLayer, Dense2048Layer)")
v("toy_in", len(TOY.inputs), "{:.0f}", "toy layer inputs")
v("toy_out", TOY.neurons, "{:.0f}", "toy layer outputs")
v("toy_weights", TOY.multiply_adds, "{:.0f}", "toy layer weights (and multiply-adds)")
v("toy_in_word", len(TOY.inputs), "words", "the same, in prose")
v("toy_out_word", TOY.neurons, "words", "the same, in prose")
v("toy_weights_word", TOY.multiply_adds, "words", "the same, in prose")
v("l1_weights", L1.weights, "{:,.0f}", "784 -> 128 weights")
v("l1_params", L1.parameters, "{:,.0f}", "784 -> 128 parameters with biases (PyTorch count)")
v("wide", WIDE.in_features, "{:,.0f}", "production layer width")
v("wide_m", WIDE.weights / 10 ** 6, "{:.2f}", "2,048 -> 2,048 weights, millions")
v("wide_mb", m(WIDE.weight_bytes, "MB"), "{:.1f}", "2,048 -> 2,048 weights in FP32, MB")
v("fp32_bytes", m(FP32, "byte"), "{:.0f}", "bytes per FP32 value")
v("fp32_bytes_word", m(FP32, "byte"), "words", "the same, in prose")
v("growth_word", L1_BIG.weights / L1.weights, "words", "weight growth when the input doubles in width and height")

print("== Four pixels into three neurons (Scenarios.FourPixelThreeNeuronLayer)")
for i, x in enumerate(TOY.inputs, start=1):
    v(f"x{i}", x, "{:.1f}", f"pixel {i}")
for i, row in enumerate(TOY.weights, start=1):
    for j, w in enumerate(row, start=1):
        v(f"w{i}{j}", w, "{:.1f}", f"weight, input {i} to neuron {j}")
for i, row in enumerate(TOY.weights, start=1):
    v(f"w{i}1_abs", abs(row[0]), "{:.1f}", f"weight, input {i} to neuron 1 (magnitude; the sign is in the text)")
for i, t in enumerate(TOY.terms(0), start=1):
    v(f"t{i}", abs(t), "{:.2f}", f"term {i} of neuron 1 (magnitude; the sign is in the text)")
for j, z in enumerate(TOY.preactivations, start=1):
    v(f"z{j}", abs(z), "{:.2f}", f"sum of neuron {j} (magnitude; the sign is in the text)")
for j, a in enumerate(TOY.activations, start=1):
    v(f"r{j}", a, "{:.2f}" if a else "{:.0f}", f"ReLU of neuron {j}")

print("== Dense intensity at batch one; energy (MnistMlpFirstLayer; Hardware.Tech, Horowitz 45 nm)")
v("dense_i", L1.intensity, "{:.1f}", "FLOP per byte, FP32 weights, batch one")
v("batch_100", L1.batch_for_intensity(100), "{:.0f}", "batch at which the layer reaches 100 FLOP/byte")
v("dense_i_fp16", L1.variant(bytes_per_weight=FP16).intensity, "{:.0f}", "FLOP per byte with 16-bit weights")
e_mac = m((Hardware.Tech.Op.FlopFp32.energy + Hardware.Tech.Op.AddFp32.energy) * units.flop, "pJ")
e_word = m(FP32 * Hardware.Tech.Memory.DRAM.energy_per_byte, "pJ")
v("mac_ratio", e_word / e_mac, "{:.0f}", "one FP32 DRAM weight read over one FP32 multiply-add")

print("== Convolution: shift, receptive field, AlexNet (Scenarios.ThreeByThreeReceptiveField, AlexNetFirstConv)")
v("k_area", CONV.kernel ** 2, "{:.0f}", "weights in one 3x3 filter (per input channel)")
v("k_area_word", CONV.kernel ** 2, "words", "the same, in prose")
v("rf_growth", RF.growth_per_layer, "{:.0f}", "pixels each 3x3 layer adds")
v("rf_layers_word", RF.layers, "words", "layers in the stack, in prose")
v("rf_field", RF.field, "{:.0f}", "receptive field of three 3x3 layers, pixels")
v("rf_target", RF.target_field, "{:.0f}", "self-check target field, pixels")
v("rf_needed", RF.layers_for_target, "{:.0f}", "3x3 layers to reach it")
v("alex_filters", ALEX.out_channels, "{:.0f}", "AlexNet conv1 filters")
v("alex_k", ALEX.kernel, "{:.0f}", "AlexNet conv1 filter side")
v("alex_c", ALEX.in_channels, "{:.0f}", "AlexNet conv1 input channels")
v("alex_weights", ALEX.weights, "{:,.0f}", "AlexNet conv1 weights, biases aside")
v("img", CONV.height, "{:.0f}", "ImageNet input side, pixels")

print("== Weight sharing and reuse (Scenarios.ImageNet64ChannelConv)")
v("ch", CONV.in_channels, "{:.0f}", "channels in and out")
v("dense_w_m", CONV.dense_weights / 10 ** 6, "{:.1f}", "dense weights, millions")
v("conv_w", CONV.weights, "{:,.0f}", "3x3 convolution weights")
v("share_ratio", CONV.sharing_ratio, "{:,.0f}", "dense over convolution, times")
v("big_img", CONV_BIG.height, "{:,.0f}", "larger image side, pixels")
v("big_growth", CONV_BIG.dense_weights / CONV.dense_weights, "{:.0f}", "dense weight growth at that size, times")
v("positions", CONV.positions, "{:,.0f}", "positions per filter weight")
v("layer_gmac", CONV.macs / 10 ** 9, "{:.2f}", "layer multiply-adds, billions")
v("layer_kb", m(CONV.weight_bytes, "KB"), "{:.1f}", "layer weights in FP32, KB")
v("outputs_m", CONV.outputs / 10 ** 6, "{:.1f}", "output values, millions")
v("act_mb", m(CONV.activation_bytes, "MB"), "{:.1f}", "activations in FP32, MB")
v("act_ratio", CONV.activation_to_weight_ratio, "{:.0f}", "activations over weights, times")
v("resnet_i", intensity(RESNET), "{:.1f}", "ResNet-50 FLOP per FP32 weight byte")
v("resnet_i_round", intensity(RESNET), "{:.0f}", "the same, to the unit")
v("resnet_vs_dense", intensity(RESNET) / L1.intensity, "{:.0f}", "ResNet-50 over the batch-one dense layer, times")

print("== Unfolding (Scenarios.Im2colMiniature, ImageNet64ChannelConvUnpadded)")
v("im_k", MINI.kernel, "{:.0f}", "miniature filter side")
v("im_in", MINI.input_values, "{:.0f}", "miniature input values")
v("im_unf", MINI.unfolded_entries, "{:.0f}", "unfolded entries")
v("im_windows", MINI.positions, "{:.0f}", "windows (patch-matrix columns)")
v("im_row", MINI.kernel ** 2, "{:.0f}", "values per window")
v("max_copies", CONV.max_copies, "{:.0f}", "most copies of one pixel, 3x3 filter")
v("valid_side", CONV_VALID.output_height, "{:.0f}", "window positions per side, no padding")
v("unfold_cols", CONV_VALID.positions, "{:,.0f}", "patch-matrix columns, 224x224 image, 3x3 window")

print("== Thirty frames a second (Scenarios.ResNet50ThirtyFpsStream)")
v("resnet_gflop", m(STREAM.flops_per_frame, "GFLOP"), "{:.1f}", "ResNet-50 GFLOP per image")
v("fps", m(STREAM.frame_rate, "Hz"), "{:.0f}", "frames per second")
v("gpu_tflops", m(STREAM.peak_rate, "TFLOP/s"), "{:.0f}", "GPU peak, TFLOP/s")
v("util_lo", 100 * STREAM.utilization_low, "{:.0f}", "planning utilization, low, %")
v("util_hi", 100 * STREAM.utilization_high, "{:.0f}", "planning utilization, high, %")
v("eff_tflops", m(STREAM.sustained_low, "TFLOP/s"), "{:.0f}", "sustained rate at the low utilization, TFLOP/s")
v("need_gflops", m(STREAM.required_rate, "GFLOP/s"), "{:.0f}", "GFLOP/s the stream needs")
v("need_tflops", m(STREAM.required_rate, "TFLOP/s"), "{:.3f}", "the same, TFLOP/s")
v("headroom", STREAM.headroom, "{:.1f}", "headroom, times")
v("headroom_round", STREAM.headroom, "{:.0f}", "the same, to the unit")
v("det_gflop", m(STREAM.alternative_flops, "GFLOP"), "{:.0f}", "detector GFLOP per frame")
v("det_tflops", m(STREAM.alternative_required_rate, "TFLOP/s"), "{:.0f}", "detector need, TFLOP/s")
v("det_headroom", STREAM.alternative_headroom, "{:.1f}", "detector headroom, times")
v("conv3_pct", 100 * SHARE3.share, "{:.0f}", "3x3 convolutions' share of ResNet-50 multiply-adds, %")

print("== Depth: skips and residual add (Scenarios.PlainStackGradientDecay, ResidualAddFp32)")
v("depth", GRAD.depth, "{:.0f}", "plain-stack depth")
v("depth_word", GRAD.depth, "words", "the same, in prose")
v("gain", GRAD.gain, "{:.1f}", "per-layer gradient factor")
v("grad", GRAD.product, "{:.3f}", "gradient product")
v("add_i", ADD.intensity, "{:.2f}", "residual add, FLOP per byte")
v("add_bytes", m(ADD.bytes_per_element, "byte"), "{:.0f}", "bytes moved per add")

print("== Residual networks on CIFAR-10 (He et al. 2016, Table 6)")
v("r56_err", QUALITY.ResNet56Cifar10TestErrorPct, "{:.1f}", "ResNet-56 test error, %, rounded")
v("r20_err", QUALITY.ResNet20Cifar10TestErrorPct, "{:.1f}", "ResNet-20 test error, %, rounded")
v("r56_err_exact", QUALITY.ResNet56Cifar10TestErrorPct, "{:.2f}", "ResNet-56 test error, %")
v("r20_err_exact", QUALITY.ResNet20Cifar10TestErrorPct, "{:.2f}", "ResNet-20 test error, %")

print("== Depthwise separable factoring (Scenarios.ImageNet64ChannelConv)")
v("dw", CONV.depthwise_weights, "{:,.0f}", "depthwise multiply-adds per position")
v("pw", CONV.pointwise_weights, "{:,.0f}", "pointwise multiply-adds per position")
v("sep", CONV.separable_weights, "{:,.0f}", "separable total")
v("sep_ratio", CONV.separable_ratio, "{:.1f}", "standard over separable, times")
v("sep_round", CONV.separable_ratio, "{:.0f}", "the same, to the unit")
v("sep_round_word", round(CONV.separable_ratio), "words", "the same, in prose")
v("sep_limit", CONV.separable_limit, "{:.0f}", "limit with many channels, times")

print("== Check: ResNet-50 against MobileNetV2 (Models; Literature.KerasApplications)")
v("flop_ratio", m(RESNET.inference_flops / MOBILENET.inference_flops, ""), "{:.1f}", "FLOP ratio")
v("resnet_params_m", m(RESNET.parameters, "Mparam"), "{:.1f}", "ResNet-50 parameters, millions")
v("mobilenet_params_m", m(MOBILENET.parameters, "Mparam"), "{:.1f}", "MobileNetV2 parameters, millions")
v("mobilenet_gflop", m(MOBILENET.inference_flops, "GFLOP"), "{:.1f}", "MobileNetV2 GFLOP per image")
v("resnet_top1", KERAS.ResNet50Top1Pct, "{:.1f}", "Keras ResNet50 top-1, %")
v("mobilenet_top1", KERAS.MobileNetV2Top1Pct, "{:.1f}", "Keras MobileNetV2 top-1, %")
v("top1_gap", KERAS.ResNet50Top1Pct - KERAS.MobileNetV2Top1Pct, "{:.1f}", "top-1 gap, points")
v("resnet_cpu_ms", KERAS.ResNet50CpuMsPerStep, "{:.1f}", "ResNet50 CPU ms/step")
v("mobilenet_cpu_ms", KERAS.MobileNetV2CpuMsPerStep, "{:.1f}", "MobileNetV2 CPU ms/step")
v("resnet_gpu_ms", KERAS.ResNet50GpuMsPerStep, "{:.1f}", "ResNet50 A100 ms/step")
v("mobilenet_gpu_ms", KERAS.MobileNetV2GpuMsPerStep, "{:.1f}", "MobileNetV2 A100 ms/step")
v("cpu_speedup", KERAS.ResNet50CpuMsPerStep / KERAS.MobileNetV2CpuMsPerStep, "{:.1f}", "measured CPU speedup")
v("gpu_speedup", KERAS.ResNet50GpuMsPerStep / KERAS.MobileNetV2GpuMsPerStep, "{:.1f}", "measured A100 speedup")
v("keras_batch", KERAS.BatchSize, "{:.0f}", "Keras timing batch")
v("keras_cores", KERAS.CpuCores, "{:.0f}", "Keras timing CPU cores")

print("== Keyword spotter on a microcontroller (Scenarios.KwsDsCnnOnNrf52840Fp32/Int8)")
v("flash_mb", m(MCU.memory.flash_capacity, "MiB"), "{:.0f}", "nRF52840 flash, MB (binary)")
v("sram_kb", m(MCU.memory.sram_capacity, "KiB"), "{:.0f}", "nRF52840 SRAM, KB (binary)")
v("mcu_mhz", m(MCU.compute.clock_rate, "MHz"), "{:.0f}", "nRF52840 clock, MHz")
v("kws_params", m(KWS.parameters, "param"), "{:,.0f}", "keyword spotter weights")
v("kws_params_k", m(KWS.parameters, "Kparam"), "{:.0f}", "the same, thousands")
v("kws_fp32_kb", m(KWS32.weight_bytes, "KB"), "{:.0f}", "FP32 weights, KB")
v("kws_int8_kb", m(KWS8.weight_bytes, "KB"), "{:.0f}", "INT8 weights, KB")
v("kws_fp32_pct", 100 * KWS32.flash_share, "{:.0f}", "FP32 share of flash, %")
v("kws_int8_pct", 100 * KWS8.flash_share, "{:.0f}", "INT8 share of flash, %")
v("quant_word", m(FP32 / INT8, ""), "words", "FP32 over INT8 bytes, in prose")
v("kws_i", intensity(KWS), "{:.0f}", "keyword spotter FLOP per FP32 weight byte")
v("mobilenet_i", intensity(MOBILENET), "{:.1f}", "MobileNetV2 FLOP per FP32 weight byte")

print("== Recurrence (Scenarios.ElmanRnnStep, LstmStep, RecurrentState512Fp32)")
v("rnn_in", RNN.input_dim, "{:.0f}", "input width")
v("rnn_h", RNN.hidden_dim, "{:.0f}", "state width")
v("rnn_macs", RNN.macs_per_step, "{:,.0f}", "multiply-adds per step")
v("seq", RNN.steps, "{:,.0f}", "sequence length, steps")
v("lstm_rows", LSTM.gate_rows, "{:.0f}", "fused gate rows")
v("lstm_ratio_word", LSTM.weights / RNN.weights, "words", "the same, in prose")
v("gates", LSTM.gates, "{:.0f}", "LSTM gate maps")
v("gates_word", LSTM.gates, "words", "the same, in prose")
v("lstm_ratio", LSTM.weights / RNN.weights, "{:.0f}", "LSTM over plain weights, times")
v("state_w", STATE.hidden_dim, "{:.0f}", "state width, illustrative")
v("state_kb", m(STATE.state_bytes, "KiB"), "{:.0f}", "512-wide FP32 state, KB (binary)")

print("== Transformer against recurrence (Vaswani et al. 2017, Table 2)")
gnmt, base = RUNS.GnmtRlEnDeTrainingFlops, RUNS.TransformerBaseEnDeTrainingFlops
v("gnmt_bleu", QUALITY.GnmtRlEnDeBleu, "{:.1f}", "GNMT + RL BLEU")
v("tf_bleu", QUALITY.TransformerBaseEnDeBleu, "{:.1f}", "Transformer base BLEU")
for key, flops, label in (("gnmt", gnmt, "GNMT + RL"), ("tf", base, "Transformer base")):
    mant, exp = mantissa_exponent(flops)
    v(f"{key}_flops_mant", mant, "{:.1f}", f"{label} training FLOPs, mantissa")
    v(f"{key}_flops_exp", exp, "{:.0f}", f"{label} training FLOPs, power of ten")
v("tf_compute_ratio", gnmt / base, "{:.0f}", "training compute ratio, times")


def note_minutes():
    """Title-slide note: total note minutes and the total without [OPTIONAL]
    notes, summed from the deck's own [N min] tags (a check; not recorded)."""
    import re
    tex = (Path(__file__).resolve().parents[1] / "06_nn_architectures.tex").read_text()
    notes = re.findall(r"\\(?:note|mlsysnote)\{\[(\d+(?:\.\d+)?) min\](.*?)\n", tex)
    total = sum(float(n) for n, _ in notes)
    core = sum(float(n) for n, t in notes if "[OPTIONAL" not in t)
    print(f"\n[Title note: lecture length] notes total {total:g} min; without [OPTIONAL] notes {core:g} min")


if __name__ == "__main__":
    note_minutes()
    write(__file__)
