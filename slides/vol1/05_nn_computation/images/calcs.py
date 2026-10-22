#!/usr/bin/env python3
"""Every number on the Volume I Lecture 5 (Neural Computation) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the running 784-128-64-10
network (Scenarios.MnistDenseNetwork and its variants, DenseNetworkModel), the
width and depth sweep, the CPU-against-accelerator epoch (EpochOverheadModel),
the paradigm, activation, loss, and backpropagation examples, the number-format
ranges (physics.numerics), the parameter-growth fit, and the cited figures in
Literature. The chapter reads the same scenarios. This file only selects values,
converts units, and formats them; it types no numbers. Each value is recorded
with v() under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports this module, so figures and slides
share one computation.

Two inputs are measured here, not in MLSysIM, and read as the tools printed
them: data/torch_runs.txt (torch_runs.py: autograd saved tensors and the
torch.profiler table) and data/overfit.json (overfit_run.py: MNIST training
against held-out accuracy).

Run from this folder: python3 calcs.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Scenarios, Literature, Datasets = mlsysim.Scenarios, mlsysim.Literature, mlsysim.Datasets
units = mlsysim.core.units

NET = Scenarios.MnistDenseNetwork
R = NET.result                                         # batch 32
R1 = NET.variant(batch_size=1).result                  # batch 1
WIDE = Scenarios.MnistDenseNetworkWiderHidden.result
BIG_INPUT = Scenarios.MnistDenseNetworkDoubledInputSide.result
SWEEP = Scenarios.DenseWidthDepthSweep
EPOCH = Scenarios.MnistEpochCpuVersusH100
E = EPOCH.result
PARADIGMS = Scenarios.DigitRecognitionParadigms
LOGIC = Scenarios.ReluVersusSigmoidLogic
CHAIN = Scenarios.TwentySigmoidLayers
FORMATS = Scenarios.VanishedGradientFormats
CE = Scenarios.DigitCrossEntropyExamples
TRACE = Scenarios.TwoTwoOneBackpropTrace
GPT2 = Scenarios.Gpt2Fp32AdamOnH100
RESNET = Scenarios.ResNet50Fp32SgdMomentum
GROWTH = Scenarios.NotableModelParameterGrowth.result
FEED = Scenarios.ResNet50A100InputFeed                 # Lecture 4b's recall question
H100 = EPOCH.accelerator


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def kb(q):
    return m(q, "KB")


def mantissa_exponent(x):
    """(mantissa, power of ten) of x, for spoken and typeset scientific notation."""
    mant, exp = f"{x:.1e}".split("e")
    return float(mant), int(exp)


print("== The running network (Scenarios.MnistDenseNetwork)")
w_in, w_h1, w_h2, w_out = NET.layer_widths
v("w_in", w_in, "{:.0f}", "input width (28 x 28 pixels)")
v("w_h1", w_h1, "{:.0f}", "first hidden width")
v("w_h2", w_h2, "{:.0f}", "second hidden width")
v("w_out", w_out, "{:.0f}", "output width (classes)")
v("w_h1_wide", WIDE.layer_widths[1], "{:.0f}", "first hidden width, doubled")
v("w_h2_wide", WIDE.layer_widths[2], "{:.0f}", "second hidden width, doubled")
v("img_side", Datasets.MNIST.image_width, "{:.0f}", "MNIST image side, pixels")
v("img_side_big", BIG_INPUT.layer_widths[0] ** 0.5, "{:.0f}", "doubled image side, pixels")
v("mnist_total", m(Datasets.MNIST.training_examples + Datasets.MNIST.test_examples, "count"), "{:,.0f}",
  "MNIST images, training plus test")
v("mnist_train", m(Datasets.MNIST.training_examples, "count"), "{:,.0f}", "MNIST training images")
v("mnist_test", m(Datasets.MNIST.test_examples, "count"), "{:,.0f}", "MNIST test images")
v("batch", NET.batch_size, "{:.0f}", "batch size")
v("macs", R.macs_per_example, "{:,.0f}", "MACs per digit")
v("flops", R.flops_per_example, "{:,.0f}", "FLOPs per digit (2 per MAC)")
for i, (wt, b) in enumerate(zip(R.weights_per_layer, R.biases_per_layer), start=1):
    v(f"wt{i}", wt, "{:,.0f}", f"layer {i} weights")
v("biases", R.biases, "{:.0f}", "biases")
v("params", R.parameters, "{:,.0f}", "parameters")
fp32_bytes = m(NET.bytes_per_value, "byte")
v("fp32_bytes", fp32_bytes, "{:.0f}", "bytes per FP32 value")
v("weights_kb", kb(R.parameter_bytes), "{:.1f}", "FP32 weights, KB")
v("weights_kb_int", kb(R.parameter_bytes), "{:.0f}", "FP32 weights, KB, rounded")
v("l1_share", 100 * R.first_layer_share, "{:.1f}", "layer 1 share of weights and work, %")
v("l1_share_int", 100 * R.first_layer_share, "{:.0f}", "the same, rounded, %")
v("w1_big", BIG_INPUT.weights_per_layer[0], "{:,.0f}", "first-layer weights for the doubled input side")

print("== Ladder (Scenarios.DigitRecognitionParadigms)")
v("rules", PARADIGMS.rule_comparisons, "{:.0f}", "rule comparisons per digit")
v("hog_ops", PARADIGMS.feature_operations, "{:,.0f}", "HOG + classifier operations")
v("hog_kb", kb(PARADIGMS.feature_memory), "{:.0f}", "HOG working memory, KB")
v("net_over_rules", round(PARADIGMS.network_to_rule_ratio, -2), "{:,.0f}", "network over rules, to the hundred")

print("== Parameter growth (Scenarios.NotableModelParameterGrowth)")
v("growth_split", GROWTH.split_year, "{:.0f}", "split year")
v("growth_early_mo", round(GROWTH.early_doubling_months), "{:.0f}", "doubling before the split, months")
v("growth_late_mo", round(GROWTH.late_doubling_months), "{:.0f}", "doubling after the split, months")
v("growth_models", GROWTH.model_count, "{:,.0f}", "models with parameter counts")

print("== Neuron, layer, and XOR")
v("layer1_b_macs_m", NET.batch_size * R.weights_per_layer[0] / 10 ** 6, "{:.1f}", "layer 1 MACs at batch 32, millions")

print("== Vanishing gradient (Scenarios.TwentySigmoidLayers; Literature.TrainingRuns)")
v("sig_peak", CHAIN.peak_slope, "{:.2f}", "sigmoid's peak slope")
v("sig_depth", CHAIN.depth, "{:.0f}", "sigmoid layers")
g_mant, g_exp = mantissa_exponent(CHAIN.surviving_fraction)
v("grad_mant", g_mant, "{:.1f}", "surviving fraction, mantissa")
v("grad_exp", g_exp, "{:.0f}", "surviving fraction, power of ten")
v("grad_exp_mag", -g_exp, "{:.0f}", "the same power, magnitude (spoken)")
relu_ep = Literature.TrainingRuns.AlexNetCifar10ReluEpochsTo25PctError
tanh_ep = Literature.TrainingRuns.AlexNetCifar10TanhEpochsTo25PctError
v("relu_ep", relu_ep, "{:.0f}", "AlexNet Fig. 1: ReLU epochs to 25% error")
v("tanh_ep", tanh_ep, "{:.0f}", "AlexNet Fig. 1: tanh epochs to 25% error")
v("relu_speedup", tanh_ep / relu_ep, "{:.0f}", "tanh epochs over ReLU epochs")
v("relu_speedup_word", tanh_ep / relu_ep, "words", "the same, in prose")

print("== Transistor tax (Scenarios.ReluVersusSigmoidLogic)")
v("relu_tr", LOGIC.cheap_transistors, "{:.0f}", "ReLU transistor equivalents")
v("sig_tr", LOGIC.costly_transistors, "{:,.0f}", "sigmoid transistor equivalents")
v("tr_ratio_word", LOGIC.ratio, "words", "sigmoid over ReLU, in prose")
v("tr_ratio", LOGIC.ratio, "{:.0f}", "sigmoid over ReLU")

print("== Width and depth (Scenarios.DenseWidthDepthSweep)")
v("width_lo", SWEEP.width_doubled_from, "{:,.0f}", "width before doubling")
v("width_hi", SWEEP.widths[-1], "{:,.0f}", "width after doubling")
v("width_double", SWEEP.width_double_factor, "{:.1f}", "MACs factor, width doubled")
v("fixed_width", SWEEP.fixed_width, "{:.0f}", "fixed width of the depth sweep")
v("depth_lo", SWEEP.depth_doubled_from, "{:.0f}", "depth before doubling")
v("depth_hi", SWEEP.max_depth, "{:.0f}", "depth after doubling")
v("depth_double", SWEEP.depth_double_factor, "{:.1f}", "MACs factor, depth doubled")
v("retained_double", SWEEP.saved_activations_double_factor, "{:.1f}", "saved activations factor, depth doubled")

print("== Forward pass at batch 32 (DenseNetworkModel)")
for i, f in enumerate(R.matmul_flops_per_layer, start=1):
    v(f"mm{i}", f, "{:,.0f}", f"layer {i} matrix-multiply FLOPs")
v("other_ops", R.elementwise_flops, "{:,.0f}", "bias and activation operations, all layers")
v("other_share", 100 * R.elementwise_share, "{:.1f}", "their share, %")
v("mm_share", 100 * R.matmul_share, "{:.1f}", "matrix-multiply share, %")
v("fwd_m", R.forward_flops / 10 ** 6, "{:.0f}", "forward operations per batch, millions")
v("per_digit_k", R.forward_flops_per_example / 10 ** 3, "{:.0f}", "operations per digit, thousands")
v("wide_m", WIDE.forward_flops / 10 ** 6, "{:.0f}", "forward operations, hidden widths doubled, millions")
v("wide_ratio", WIDE.forward_flops / R.forward_flops, "{:.1f}", "doubled-hidden total over the running one")

print("== Measured profile (data/torch_runs.txt)")
prof = (HERE / "data" / "torch_runs.txt").read_text()


def us(value, unit):
    return float(value) * (10 ** 3 if unit == "ms" else 1)


def row(op):
    """(self %, CPU time avg in us, calls) of one profiler row, as torch.profiler printed it."""
    r = re.search(rf"^\s*{re.escape(op)}\s+(?P<pct>[\d.]+)%\s+\S+\s+\S+\s+\S+\s+"
                  rf"(?P<t>[\d.]+)(?P<u>ms|us)\s+(?P<n>\d+)\s*$", prof, re.M)
    return float(r["pct"]), us(r["t"], r["u"]), int(r["n"])


addmm_pct, addmm_avg, addmm_calls = row("aten::addmm")
relu_pct = row("aten::clamp_min")[0]
total = re.search(r"Self CPU time total: (?P<t>[\d.]+)(?P<u>ms|us)", prof)
passes = addmm_calls / R.layers                          # one addmm per layer per pass
v("prof_passes", passes, "{:.0f}", "profiled forward passes")
v("addmm_pct", addmm_pct, "{:.0f}", "addmm share of self CPU time, %")
v("addmm_calls", addmm_calls, "{:.0f}", "addmm calls")
v("addmm_avg_us", addmm_avg, "{:.1f}", "addmm average per call (CPU time avg), us")
v("pass_us", us(total["t"], total["u"]) / passes, "{:.1f}", "time per forward pass, us")
v("bookkeeping_pct", 100 - addmm_pct - relu_pct, "{:.0f}", "everything else, %")

print("== Arithmetic intensity (DenseNetworkModel at batch 1 and 32)")
v("traffic1_kb", kb(R1.traffic_bytes), "{:.0f}", "bytes moved at batch 1, KB")
v("traffic32_kb", kb(R.traffic_bytes), "{:.0f}", "bytes moved at batch 32, KB")
i1, i32 = m(R1.intensity, "flop/byte"), m(R.intensity, "flop/byte")
v("i1", i1, "{:.1f}", "intensity at batch 1, FLOP/byte")
v("i1_2", i1, "{:.2f}", "the same, two decimals")
v("i32", i32, "{:.1f}", "intensity at batch 32, FLOP/byte")
v("i_ratio", i32 / i1, "{:.0f}", "intensity ratio")

print("== Roofline (Scenarios.MnistEpochCpuVersusH100, FP32)")
v("ridge", m(E.ridge["accelerator"], "flop/byte"), "{:.0f}", "ridge point, both machines, FLOP/byte")
v("acc_b1_tf", m(E.rate_batch1["accelerator"], "Tflop/s"), "{:.1f}", "accelerator at batch 1, TFLOP/s")
v("acc_b32_tf", m(E.rate["accelerator"], "Tflop/s"), "{:.0f}", "accelerator at batch 32, TFLOP/s")
v("cpu_b1_gf", m(E.rate_batch1["host"], "Gflop/s"), "{:.0f}", "CPU at batch 1, GFLOP/s")
v("cpu_b32_gf", m(E.rate["host"], "Gflop/s"), "{:.0f}", "CPU at batch 32, GFLOP/s")
v("batch_speedup", E.batching_speedup["accelerator"], "{:.0f}", "per-image speedup from batching")

print("== Training loop and loss (Scenarios.DigitCrossEntropyExamples)")
v("steps", R.steps_per_epoch, "{:,.0f}", "steps per epoch")
v("ce_p_conf", CE.probabilities["confident"], "{:.1f}", "confident probability")
v("ce_p", CE.probabilities["correct"], "{:.1f}", "correct-class probability")
v("ce_p_miss", CE.misread["true"], "{:.1f}", "probability on the true digit, misread")
v("ce_conf", CE.loss("confident"), "{:.2f}", "loss at the confident probability")
v("ce_loss", CE.loss("correct"), "{:.2f}", "loss at the correct-class probability")
v("ce_uniform", CE.uniform_loss, "{:.2f}", "loss of a uniform guess")
v("ce_miss", CE.misread_loss, "{:.2f}", "loss of the misread digit")

print("== Backpropagation by hand (Scenarios.TwoTwoOneBackpropTrace)")
v("bp_x1", TRACE.x[0], "{:.1f}", "input 1")
v("bp_x2", TRACE.x[1], "{:.1f}", "input 2")
v("bp_w11", TRACE.w1[0][0], "{:.1f}", "W1, input 1 to hidden 1")
v("bp_w12", TRACE.w1[0][1], "{:.1f}", "W1, input 1 to hidden 2")
v("bp_w21", TRACE.w1[1][0], "{:.1f}", "W1, input 2 to hidden 1")
v("bp_w22", TRACE.w1[1][1], "{:.1f}", "W1, input 2 to hidden 2")
v("bp_v1", TRACE.w2[0], "{:.1f}", "W2, hidden 1")
v("bp_v2", TRACE.w2[1], "{:.1f}", "W2, hidden 2")
v("bp_z1", TRACE.z[0], "{:.1f}", "pre-activation, hidden 1")
v("bp_z2", TRACE.z[1], "{:.1f}", "pre-activation, hidden 2")
v("bp_a1", TRACE.a[0], "{:.1f}", "activation, hidden 1")
v("bp_target", TRACE.target, "{:.1f}", "target")
v("bp_yhat", TRACE.y_hat, "{:.2f}", "output")
v("bp_loss", TRACE.loss, "{:.4f}", "squared-error loss")
v("bp_err", TRACE.error, "{:.2f}", "output error")
v("bp_gv1", TRACE.w2_gradient[0], "{:.3f}", "gradient of W2, hidden 1")
v("bp_back2", TRACE.sent_back[1], "{:.3f}", "error sent back, hidden 2")
v("bp_lr", TRACE.learning_rate, "{:.1f}", "learning rate")
v("bp_w11_new", TRACE.w1_updated(0, 0), "{:.4f}", "W1, input 1 to hidden 1, after one step")
v("bp_d1", TRACE.delta[0], "{:.3f}", "hidden 1 error after the ReLU mask")
# Magnitudes of the negative values, for the spoken notes ("minus 0.3")
for key, value in (("bp_w12_mag", TRACE.w1[0][1]), ("bp_z2_mag", TRACE.z[1]),
                   ("bp_err_mag", TRACE.error), ("bp_gv1_mag", TRACE.w2_gradient[0]),
                   ("bp_back1_mag", TRACE.sent_back[0]), ("bp_gw21_mag", TRACE.w1_gradient[1][0])):
    v(key, abs(value), "{:.4g}", "magnitude, spoken")

print("== Training FLOPs at batch 32 (DenseNetworkModel)")
v("train_fwd_m", R.matmul_flops / 10 ** 6, "{:.1f}", "forward matrix work, MFLOP")
v("train_igrad_m", R.input_gradient_flops / 10 ** 6, "{:.2f}", "input gradients, MFLOP")
v("train_step_m", R.training_step_flops / 10 ** 6, "{:.1f}", "training step, MFLOP")
v("train_ratio", R.training_flops_ratio, "{:.2f}", "training step over forward")

print("== One epoch on two machines (Scenarios.MnistEpochCpuVersusH100)")
v("dispatch_us", m(EPOCH.dispatch, "us"), "{:.0f}", "Python dispatch per operation, us")
v("dispatch_launch_us", m(EPOCH.dispatch + EPOCH.launch, "us"), "{:.0f}", "dispatch plus kernel launch, us")
v("launch_us", m(EPOCH.launch, "us"), "{:.0f}", "kernel launch per operation, us")
v("ops_step", E.ops_per_step, "{:.0f}", "operations launched per step")
v("work_cpu_ms", round(m(E.work_time["host"], "ms")), "{:.0f}", "CPU work, ms")
v("work_acc_ms", m(E.work_time["accelerator"], "ms"), "{:.1f}", "accelerator work, ms")
v("work_speedup", round(E.work_speedup), "{:.0f}", "accelerator work speedup")
v("ovh_cpu_ms", m(E.overhead_time["host"], "ms"), "{:.0f}", "CPU overhead, ms")
v("ovh_acc_ms", m(E.overhead_time["accelerator"], "ms"), "{:.1f}", "accelerator overhead, ms")
v("tot_cpu_ms", round(m(E.total_time["host"], "ms")), "{:.0f}", "CPU total, ms")
v("tot_acc_ms", round(m(E.total_time["accelerator"], "ms")), "{:.0f}", "accelerator total, ms")
v("ovh_share", 100 * E.overhead_share_accelerator, "{:.1f}", "overhead share of the accelerator epoch, %")

print("== Number formats (Scenarios.VanishedGradientFormats)")
fp16_max, fp16_norm, fp16_sub = FORMATS.range("fp16")
fp32_max, fp32_norm, _ = FORMATS.range("fp32")
n_mant, n_exp = mantissa_exponent(fp16_norm)
s_mant, s_exp = mantissa_exponent(fp16_sub)
v("fp16_norm_mant", n_mant, "{:.1f}", "FP16 smallest normal, mantissa")
v("fp16_norm_exp", n_exp, "{:.0f}", "FP16 smallest normal, power of ten")
v("fp16_sub_mant", s_mant, "{:.1f}", "FP16 smallest subnormal, mantissa")
v("fp16_sub_exp", s_exp, "{:.0f}", "FP16 smallest subnormal, power of ten")
v("fp16_max", fp16_max, "{:,.0f}", "FP16 largest finite")
v("fp32_range_exp", mantissa_exponent(fp32_max)[1], "{:.0f}", "FP32 range, power of ten (spoken)")
v("int16_max", FORMATS.integer_max, "{:,.0f}", "INT16 largest")
print("== Ariane 501 (Literature.IncidentReports, Lions inquiry report 1996)")
v("ariane_irs_s", Literature.IncidentReports.Ariane501InertialReferenceFailureS, "{:.0f}",
  "inertial reference systems fail, s after H0 (36.7 in the report)")
v("ariane_breakup_s", Literature.IncidentReports.Ariane501BreakupS, "{:.0f}",
  "break-up, s after the flight sequence began")
v("grad_fp16", FORMATS.stored("fp16"), "{:.1f}", "the gradient stored in FP16")

print("== Memory, training against inference (DenseNetworkModel, Adam)")
v("mem_infer_act_kb", kb(R.inference_activation_bytes), "{:.0f}", "inference activation pair, KB")
v("mem_infer_kb", kb(R.inference_memory), "{:.0f}", "inference total, KB")
v("mem_act_kb", kb(R.saved_activation_bytes), "{:.0f}", "saved activations, KB")
v("mem_opt_kb", kb(R.optimizer_bytes), "{:.1f}", "Adam state, KB")
v("mem_train_mb", m(R.training_memory, "MB"), "{:.1f}", "training total, MB")
v("mem_ratio", R.training_memory_ratio, "{:.1f}", "training over inference")
LARGE = Scenarios.MnistDenseNetworkLargeBatch
v("big_batch", LARGE.batch_size, "{:,.0f}", "the larger batch")
v("act_growth", LARGE.result.saved_activation_values / R.saved_activation_values, "{:.0f}",
  "saved-activation growth at the larger batch")

print("== Memory trace (Scenarios.ResNet50Fp32SgdMomentum)")
v("resnet_params_m", m(RESNET.workload.parameters, "Mparam"), "{:.1f}", "ResNet-50 parameters, millions")
v("resnet_weights_mb", m(RESNET.weights, "MB"), "{:.1f}", "ResNet-50 FP32 weights, MB")
v("resnet_weights_gb", m(RESNET.weights, "GB"), "{:.1f}", "the same, GB (rounded)")
v("resnet_weights_mb_int", m(RESNET.weights, "MB"), "{:.0f}", "the same, MB, whole")

print("== GPT-2 fit (Scenarios.Gpt2Fp32AdamOnH100)")
v("gpt2_b", m(GPT2.workload.parameters, "Bparam"), "{:.1f}", "GPT-2 parameters, billions")
v("gpt2_weights_gb", m(GPT2.weights, "GB"), "{:.0f}", "GPT-2 FP32 weights, GB")
v("gpt2_train_gb", m(GPT2.total, "GB"), "{:.0f}", "weights, gradients, Adam, GB")
v("adam_values", GPT2.optimizer_values_per_parameter, "{:.0f}", "Adam values per parameter")
v("train_copies", GPT2.copies, "{:.0f}", "copies of the weights training holds")
v("h100_gb", m(H100.memory.capacity, "GiB"), "{:.0f}", "H100 memory, GB (branded)")
v("gpt2_jump", round(m(GPT2.workload.parameters, "param") / R.parameters, -3), "{:,.0f}",
  "GPT-2 over MNIST parameters, to the thousand")

print("== Rejection (Literature.PredictionQuality, LeCun et al. 1989)")
rej = Literature.PredictionQuality.LeCun1989ZipCodeRejectPct
v("lecun_rej", rej, "{:.1f}", "digits rejected, %")
v("lecun_acc", 100 - rej, "{:.1f}", "digits accepted, %")
v("lecun_err", Literature.PredictionQuality.LeCun1989ZipCodeAcceptedErrorPct, "{:.0f}", "error on accepted digits, %")
v("lecun_params", m(mlsysim.Models.Vision.LeNet1.parameters, "param"), "{:,.0f}", "1989 network parameters (about)")

print("== Lecture 4b recall (Scenarios.ResNet50A100InputFeed)")
v("resnet_train_gflop", m(FEED.training_flops_per_sample, "GFLOP"), "{:.1f}", "ResNet-50 training FLOPs per image")

print("== Overfitting run (data/overfit.json)")
OVERFIT = json.loads((HERE / "data" / "overfit.json").read_text())


def final(n):
    h = OVERFIT["runs"][n]
    return 100 * h["train_acc"][-1], 100 * h["test_acc"][-1]


small, full = sorted(OVERFIT["runs"], key=int)
tr_s, te_s = final(small)
tr_f, te_f = final(full)
first_100 = next(i for i, a in enumerate(OVERFIT["runs"][small]["train_acc"], start=1) if a >= 1)
v("ov_small", int(small), "{:,.0f}", "training images, small run")
v("ov_full", int(full), "{:,.0f}", "training images, full run")
v("ov_train_small", tr_s, "{:.0f}", "final training accuracy, small run, %")
v("ov_first100", first_100, "{:.0f}", "first epoch at 100% training accuracy, small run")
v("ov_test_small", te_s, "{:.1f}", "held-out accuracy, small run, %")
v("ov_test_small_int", te_s, "{:.0f}", "the same, rounded, %")
v("ov_gap_small", tr_s - te_s, "{:.1f}", "gap, small run, points")
v("ov_test_full", te_f, "{:.1f}", "held-out accuracy, full run, %")
v("ov_gap_full", tr_f - te_f, "{:.1f}", "gap, full run, points")

if __name__ == "__main__":
    write(__file__)
