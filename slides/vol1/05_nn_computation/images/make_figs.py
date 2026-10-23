#!/usr/bin/env python3
"""Data plots for Lecture 5 (Neural Computation).

Every derived value is computed in calcs.py (imported here as cx) from the
inputs listed below; nothing derived is typed by hand. Inputs come from the
rendered chapter (Neural Computation) unless noted:
  - One digit by three methods (Sec. 5.2, illustrative implementations):
    about 100 comparisons (rules), about 8,000 operations (HOG + linear SVM),
    109,184 MACs (the 784-128-64-10 network); 1,100x the rule baseline.
  - Activation slopes: sigmoid's derivative peaks at 0.25; ReLU's is 1 on
    its positive branch (Sec. 5.3.2). 0.25**20 = 9.1e-13 (footnote 1).
  - Transistor tax (Sec. 5.3.2): ReLU about 50 transistor equivalents,
    exponential datapath for sigmoid or tanh about 2,500; a stylized
    teaching ratio, labeled illustrative on the slide.
  - Forward pass at batch 32 (Table 7, Napkin Math 5.1): layer MatMuls
    6,422,528 + 524,288 + 40,960 FLOPs; bias and activation 8,192 + 4,096 +
    about 640 operations.
  - Training step FLOPs at batch 32: forward 6,987,776; weight gradients
    the same; input gradients for layers 2 and 3 only, 2 x 32 x (128 x 64 +
    64 x 10) = 565,248; total about 14.5 MFLOP, 2.08x the forward work
    (Sec. 5.4.5 and Napkin Math 5.3).
  - One epoch on two machines (Napkin Math 5.3): work 46 ms (CPU) against
    0.7 ms (accelerator); fixed per-operation overhead 225 ms against
    337.5 ms; totals about 271 ms and 338 ms.
  - Training against inference memory at batch 32, FP32 (Table 9):
    parameters 437.5 KB, saved activations 126 KB (training) or 117 KB
    (inference, rolling pair), gradients 437.5 KB, Adam state 875.1 KB;
    about 1.9 MB against about 554 KB, 3.4x.
  - Number formats (footnote 20 and Sec. 5.4.5): smallest normal 1.18e-38
    (FP32) and 6.10e-5 (FP16); FP16 maximum 65,504; BF16 keeps FP32's
    exponent range. FP32 maximum 3.4e38 and the signed 16-bit integer
    maximum 32,767 are the format definitions (IEEE 754; two's complement).

  - Real runs in data/ (torch_runs.py, overfit_run.py) and two published
    figures, redrawn with slide-size axes over the original pixels:
    Krizhevsky et al. 2012 Figure 1 (alexnet-fig1.png) and the PyTorch
    memory-profiler timeline (pytorch-memory-timeline.png); see SOURCES.yml.

Run from this folder with the MLSysBook venv:
  ../../../../.venv/bin/python make_figs.py
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
import sys
from pathlib import Path
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
out = Path(__file__).parent
plt = sp.plt
sp.hbars_steps(out / 'ops_ladder', [('Hand-written rules', 100, 'neutral', '100 comparisons'), ('HOG features + SVM', 8000, 'neutral', '8,000 operations'), ('Neural network', 109184, 'compute', '109,184 MACs  = 1,100× rules')], 'ops', xlabel='Work to classify one 28×28 digit (log scale)')
z = np.linspace(-6, 6, 600)
sig = 1 / (1 + np.exp(-z))
dsig = sig * (1 - sig)
drelu = (z > 0).astype(float)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    ax.plot(z, dsig, color=sp.C['memory'], lw=3.5)
    ax.annotate('Sigmoid slope:\nat most 0.25', (0, 0.25), xytext=(-5.8, 0.5), fontsize=12, fontweight='bold', color=sp.C['memory'], arrowprops=dict(arrowstyle='-', color=sp.C['memory'], lw=1))
    if k == 2:
        ax.plot(z[z <= 0], drelu[z <= 0], color=sp.C['compute'], lw=3.5)
        ax.plot(z[z > 0], drelu[z > 0], color=sp.C['compute'], lw=3.5)
        ax.text(0.3, 0.8, 'ReLU slope: 1\nwhen z > 0', fontsize=12, va='top', fontweight='bold', color=sp.C['compute'])
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 1.0])
    ax.set_xlabel('Input to the activation, z')
    ax.set_ylabel('Slope f′(z)')
    ax.grid(axis='y', color=sp.C['grid'], linewidth=0.6)
    sp.save(fig, out / f'slopes_step{k}.svg')
sp.hbars_steps(out / 'transistor_tax', [('ReLU: compare and select', 50, 'compute', '50'), ('Sigmoid: exponential', 2500, 'danger', '2,500  = 50× ReLU')], '', xlabel='Transistor equivalents per activation unit (illustrative ratio; log scale)', size=sp.SHORT)
fwd = [('Layer 1 MatMul', 6.422528, 'compute', '6,422,528'), ('Layer 2 MatMul', 0.524288, 'compute', '524,288'), ('Layer 3 MatMul', 0.04096, 'compute', '40,960')]
fwd.append(('Bias + activation, all layers', 12928 / 1000000.0, 'neutral', '12,928  (0.2% of the total)'))
sp.hbars_steps(out / 'fwd_ops', fwd, '', log=False, xlabel='Million operations for one batch of 32 digits (total about 7 million)')

def training_flops(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    parts = [('Forward', 6987776 / 1000000.0, sp.C['compute']), ('Weight gradients', 6987776 / 1000000.0, sp.C['danger']), ('Input gradients', 565248 / 1000000.0, sp.C['network'])]
    left = 0.0
    shown = parts[:step]
    for name, v, col in shown:
        ax.barh(0, v, left=left, color=col, height=0.55, edgecolor='white', lw=1.5)
        if v > 1:
            ax.text(left + v / 2, 0, f'{name}\n{v:.2f} M', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        left += v
    if step == 3:
        ax.annotate('Input gradients\n0.57 M (layer 1 skips it)', (left - 0.28, 0.28), xytext=(left + 0.3, 0.62), fontsize=11, color=sp.C['network'], fontweight='bold', ha='left', arrowprops=dict(arrowstyle='-', color=sp.C['network'], lw=1))
        ax.text(left + 0.3, -0.05, '14.5 M = 2.08× forward', fontsize=13, fontweight='bold', color=sp.C['text'], va='center')
    ax.set_xlim(0, 21.5)
    ax.set_ylim(-0.5, 0.95)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Million FLOPs per training step (batch 32)')
    ax.grid(axis='x', color=sp.C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)
for k in (1, 2, 3):
    training_flops(out / f'train_flops_step{k}.svg', k)

def epoch_plot(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    ms = lambda q: q.to('ms').magnitude
    rows = [('CPU', ms(Q_(45.82446163, 'ms')), ms(Q_(225.0, 'ms'))), ('Accelerator', ms(Q_(0.6839471885, 'ms')), ms(Q_(337.5, 'ms')))]
    for i, (name, work, over) in enumerate(rows):
        y = 1.3 * (1 - i)
        ax.barh(y, work, color=sp.C['compute'], height=0.5)
        ax.text(-4, y, name, ha='right', va='center', fontsize=13, fontweight='bold', color=sp.C['text'])
        if step == 1:
            ax.text(work + 4, y, f'{work:.0f} ms' if work >= 10 else f'{work:.1f} ms', va='center', fontsize=13, fontweight='bold', color=sp.C['compute'])
        else:
            ax.barh(y, over, left=work, color=sp.C['neutral_l'], height=0.5, edgecolor=sp.C['neutral'], lw=0.8)
            ax.text(work + over + 4, y, f'{work + over:.0f} ms', va='center', fontsize=13, fontweight='bold', color=sp.C['text'])
    if step == 1:
        ax.text(80, 0.65, 'Work alone: the accelerator is 67× faster', fontsize=13, color=sp.C['compute'], fontweight='bold')
    else:
        ax.text(338, 0.65, "fixed overhead: 99.8% of the accelerator's epoch", fontsize=13, fontweight='bold', color=sp.C['neutral'], ha='right', va='center')
    alpha = 1.0 if step == 2 else 0.0
    for j, (txt, fc, ec) in enumerate((('arithmetic and data movement', sp.C['compute'], sp.C['compute']), ('fixed per-operation overhead', sp.C['neutral_l'], sp.C['neutral']))):
        x0, y0 = (5, 2.35 - j * 0.32)
        ax.add_patch(plt.Rectangle((x0, y0 - 0.1), 14, 0.2, facecolor=fc, edgecolor=ec, lw=0.8, alpha=alpha, clip_on=False))
        ax.text(x0 + 20, y0, txt, fontsize=13, va='center', color=sp.C['text'], alpha=alpha)
    ax.set_xlim(0, 420)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Time for one epoch of MNIST at batch 32 (ms); estimate, Napkin Math 5.3')
    ax.grid(axis='x', color=sp.C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)
for k in (1, 2):
    epoch_plot(out / f'epoch_step{k}.svg', k)

def memory_plot(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    M = {'params_kb': 437.544, 'act_train_kb': 126.208, 'act_inf_kb': 116.736, 'grads_kb': 437.544, 'adam_kb': 875.088}
    train = [('Weights', M['params_kb'], sp.C['memory']), ('Activations', M['act_train_kb'], sp.C['data']), ('Gradients', M['grads_kb'], sp.C['danger']), ('Adam state', M['adam_kb'], sp.C['network'])]
    infer = [('Weights', M['params_kb'], sp.C['memory']), ('Activations', M['act_inf_kb'], sp.C['data'])]
    left = 0
    for name, v, col in infer:
        ax.barh(0, v, left=left, color=col, height=0.55, edgecolor='white', lw=1.5)
        left += v
    ax.text(left + 25, 0, '554 KB', va='center', fontsize=13, fontweight='bold', color=sp.C['text'])
    left = 0
    for name, v, col in train[:step + 1]:
        if True:
            ax.barh(1, v, left=left, color=col, height=0.55, edgecolor='white', lw=1.5)
            if v > 300:
                ax.text(left + v / 2, 1, name, ha='center', va='center', fontsize=11, color='white', fontweight='bold')
            left += v
    total = left
    label = '1.9 MB  = 3.4×' if step >= 3 else f'{total:,.0f} KB'
    ax.text(total + 25, 1, label, va='center', fontsize=13, fontweight='bold', color=sp.C['text'])
    ax.text(M['params_kb'] / 2, 0, 'Weights', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    ax.annotate('Saved activations', (M['params_kb'] + M['act_train_kb'] / 2, 1.28), xytext=(420, 1.75), fontsize=11, color=sp.C['data'], fontweight='bold', arrowprops=dict(arrowstyle='-', color=sp.C['data'], lw=1))
    for y, name in ((1, 'Training step'), (0, 'Inference')):
        ax.text(-20, y, name, ha='right', va='center', fontsize=13, fontweight='bold', color=sp.C['text'])
    ax.set_xlim(0, 2300)
    ax.set_ylim(-0.5, 2.0)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Memory for the MNIST network, batch 32, FP32 (KB)')
    ax.grid(axis='x', color=sp.C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)
for k in (1, 2, 3):
    memory_plot(out / f'memory_step{k}.svg', k)

def usci(v):
    """2.5e-08 -> '2.5×10⁻⁸' for matplotlib labels."""
    m, e = f'{v:.1e}'.split('e')
    return f'{m}×10' + str(int(e)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))

def ranges_plot(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    (f32_max, f32_norm, _), (b16_max, b16_norm, _), (f16_max, f16_norm, f16_sub) = ((3.402823466e+38, 1.175494351e-38, 1.401298464e-45), (3.389531389e+38, 1.175494351e-38, 9.183549616e-41), (65504.0, 6.103515625e-05, 5.960464478e-08))
    G = {'fp16_min_normal': f16_norm, 'fp16_min_sub': f16_sub, 'fp16_max': f16_max, 'int16_max': 32767}
    rows = [('FP32', f32_norm, f32_max, sp.C['memory']), ('BF16', b16_norm, b16_max, sp.C['memory']), ('FP16', f16_norm, f16_max, sp.C['compute']), ('INT16', 1, 32767, sp.C['neutral'])]
    for i, (name, lo, hi, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, np.log10(hi) - np.log10(lo), left=np.log10(lo), color=col, height=0.55, alpha=0.9)
        ax.text(-41.5, y, name, ha='right', va='center', fontsize=13, fontweight='bold', color=sp.C['text'])
    lo_s, lo_n = (np.log10(G['fp16_min_sub']), np.log10(G['fp16_min_normal']))
    ax.barh(1, lo_n - lo_s, left=lo_s, color=sp.C['compute_l'], height=0.55, edgecolor=sp.C['compute'], lw=0.8, hatch='///')
    ax.text(np.log10(G['fp16_max']) + 1, 1, f"{usci(G['fp16_min_normal'])} to 65,504\nsubnormals to {usci(G['fp16_min_sub'])}", va='center', fontsize=11, color=sp.C['compute'], fontweight='bold', linespacing=1.1)
    ax.text(np.log10(G['int16_max']) + 1, 0, 'max 32,767 (Ariane 501)', va='center', fontsize=11, color=sp.C['neutral'], fontweight='bold')
    if step >= 2:
        g = 9.094947018e-13
        x = np.log10(g)
        ax.axvline(x, color=sp.C['danger'], lw=2, ls='--')
        ax.text(x - 1.2, 3.55, f'20 sigmoid layers: {usci(g)}', ha='right', fontsize=13, color=sp.C['danger'], fontweight='bold')
    ticks = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'10$^{{{t}}}$' for t in ticks])
    ax.set_xlim(-41, 41)
    ax.set_ylim(-0.5, 4.1)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Magnitudes a format holds at full precision (log scale)')
    ax.grid(axis='x', color=sp.C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)
for k in (1, 2):
    ranges_plot(out / f'ranges_step{k}.svg', k)
K = sp.C
ARCH = '-'.join((str(w) for w in [784, 128, 64, 10]))

def param_growth():
    rows = [{'model': 'Grok 4.7', 'parameters': 2100000000.0}, {'model': 'Step 5 Preview', 'parameters': 600000000000.0}, {'model': 'DeepSeek V4.1 Flash', 'parameters': 748000000000.0}, {'model': 'Tencent Hy4 preview', 'parameters': 770000000000.0}, {'model': 'GLM-5.3-Flash', 'parameters': 320000000000.0}, {'model': 'GLM-5.3', 'parameters': 744000000000.0}, {'model': 'Qwen3.8-2.4T-A95B', 'parameters': 2400000000000.0}, {'model': 'Motif-3', 'parameters': 314000000000.0}, {'model': 'K-EXAONE 2.0', 'parameters': 750000000000.0}, {'model': 'A.X K2', 'parameters': 688000000000.0}, {'model': 'Qwen 3.8 Max', 'parameters': 2400000000000.0}, {'model': 'Kimi K3', 'parameters': 2800000000000.0}, {'model': 'Inkling', 'parameters': 975000000000.0}, {'model': 'Tencent Hy3', 'parameters': 295000000000.0}, {'model': 'LongCat-2.0', 'parameters': 1600000000000.0}, {'model': 'Solar Open2 250B', 'parameters': 250300000000.0}, {'model': 'Nemotron 3 Ultra', 'parameters': 550000000000.0}, {'model': 'MiniMax-M3', 'parameters': 428000000000.0}, {'model': 'Composer 2.5', 'parameters': 1040000000000.0}, {'model': 'TML-Interaction-Small', 'parameters': 276000000000.0}, {'model': 'DeepSeek-V4-Pro', 'parameters': 1600000000000.0}, {'model': 'DeepSeek-V4-Flash', 'parameters': 284000000000.0}, {'model': 'Tencent Hy3 preview', 'parameters': 295000000000.0}, {'model': 'MiMo-V2.5-Pro', 'parameters': 1020000000000.0}, {'model': 'Kimi K2.6', 'parameters': 1040000000000.0}, {'model': 'EXAONE 4.5', 'parameters': 33200000000.0}, {'model': 'GLM-5.1', 'parameters': 754000000000.0}, {'model': 'MolmoAct 2', 'parameters': 5485309424.0}, {'model': 'Composer 2', 'parameters': 1040000000000.0}, {'model': 'MiMo-V2-Pro', 'parameters': 1020000000000.0}, {'model': 'Nemotron 3 Super', 'parameters': 120000000000.0}, {'model': 'Qwen3.5-122B-A10B', 'parameters': 122000000000.0}, {'model': 'Grok 4.20', 'parameters': 500000000000.0}, {'model': 'Qwen3.5 397B-A17B', 'parameters': 397000000000.0}, {'model': 'GLM-5', 'parameters': 744000000000.0}, {'model': 'Qwen3-Coder-Next', 'parameters': 80000000000.0}, {'model': 'Kimi K2.5', 'parameters': 1040000000000.0}, {'model': 'Solar Open 100B', 'parameters': 102000000000.0}, {'model': 'K-EXAONE', 'parameters': 236000000000.0}, {'model': 'VAETKI', 'parameters': 100000000000.0}, {'model': 'A.X K1', 'parameters': 519000000000.0}, {'model': 'HyperCLOVA X SEED 32B Think', 'parameters': 32000000000.0}, {'model': 'MiniMax-M2.1', 'parameters': 229000000000.0}, {'model': 'GLM-4.7', 'parameters': 358000000000.0}, {'model': 'Nemotron 3-Nano-30B-A3B', 'parameters': 31600000000.0}, {'model': 'DeepSeekMath-V2', 'parameters': 685000000000.0}, {'model': 'Olmo 3', 'parameters': 32000000000.0}, {'model': 'π0.6 (pi-0.6)', 'parameters': 5260000000.0}, {'model': 'P1-235B-A22B', 'parameters': 235000000000.0}, {'model': 'Kaiju Small', 'parameters': 13000000000.0}, {'model': 'Kaiju Medium', 'parameters': 34000000000.0}, {'model': 'Kaiju Large', 'parameters': 110000000000.0}, {'model': 'Kimi K2 Thinking', 'parameters': 1000000000000.0}, {'model': 'Tongyi DeepResearch', 'parameters': 30500000000.0}, {'model': 'MiniMax-M2', 'parameters': 229000000000.0}, {'model': 'Ling-1T', 'parameters': 1000000000000.0}, {'model': 'GLM-4.6', 'parameters': 357000000000.0}, {'model': 'Qwen3-Omni-30B-A3B', 'parameters': 35300000000.0}, {'model': 'AgentFounder-30B', 'parameters': 30000000000.0}, {'model': 'Qwen3-Max', 'parameters': 1000000000000.0}, {'model': 'LongCat-Flash', 'parameters': 560000000000.0}, {'model': 'gpt-oss-120b', 'parameters': 116830000000.0}, {'model': 'gpt-oss-20b', 'parameters': 20910000000.0}, {'model': 'Qwen Image', 'parameters': 27000000000.0}, {'model': 'Hierarchical Reasoning Model (HPM)', 'parameters': 27000000.0}, {'model': 'MindLink-72B', 'parameters': 72000000000.0}, {'model': 'GLM-4.5', 'parameters': 355000000000.0}, {'model': 'Qwen3-235B-A22B-Thinking (Jul 2025)', 'parameters': 235000000000.0}, {'model': 'Qwen3-235B-A22B (Jul 2025)', 'parameters': 235000000000.0}, {'model': 'Qwen3-Coder-480B-A35B', 'parameters': 480000000000.0}, {'model': 'EXAONE 4.0 (32B)', 'parameters': 32000000000.0}, {'model': 'Kimi K2', 'parameters': 1000000000000.0}, {'model': 'Grok 4', 'parameters': 3000000000000.0}, {'model': 'EXAONE Path 2.0', 'parameters': 175000000.0}, {'model': 'FGN', 'parameters': 720000000.0}, {'model': 'Seed-1.6-Thinking', 'parameters': 230000000000.0}, {'model': 'Qwen3 Embedding', 'parameters': 8000000000.0}, {'model': 'DeepSeek-R1 (May 2025)', 'parameters': 671000000000.0}, {'model': 'Qwen3-235B-A22B', 'parameters': 235000000000.0}, {'model': 'Pangu Ultra', 'parameters': 135000000000.0}, {'model': 'Llama 4 Scout', 'parameters': 109000000000.0}, {'model': 'Llama 4 Maverick', 'parameters': 400000000000.0}, {'model': 'Llama 4 Behemoth (preview)', 'parameters': 2000000000000.0}, {'model': 'DeepSeek-V3 (Mar 2025)', 'parameters': 671000000000.0}, {'model': 'Diffusion Renderer', 'parameters': 1100000000.0}, {'model': 'EXAONE Deep 32B', 'parameters': 32000000000.0}, {'model': 'ERNIE-4.5-VL-424B-A47B (文心大模型4.5)', 'parameters': 424000000000.0}, {'model': 'Hunyuan-TurboS', 'parameters': 560000000000.0}, {'model': 'QwQ-32B', 'parameters': 32500000000.0}, {'model': 'Grok 3', 'parameters': 3000000000000.0}, {'model': 'Eagle 2', 'parameters': 8930000000.0}, {'model': 'DeepSeek-R1', 'parameters': 671000000000.0}, {'model': 'INTELLECT-MATH', 'parameters': 7000000000.0}, {'model': 'Eurus-2-7B-PRIME', 'parameters': 7000000000.0}, {'model': 'STORM-B/8', 'parameters': 100598707.0}, {'model': 'DeepSeek-V3', 'parameters': 671000000000.0}, {'model': 'Apollo 7B', 'parameters': 7000000000.0}, {'model': 'EXAONE 3.5 32B', 'parameters': 32000000000.0}, {'model': 'Llama 3.3 70B', 'parameters': 70000000000.0}, {'model': 'NVILA 15B', 'parameters': 15000000000.0}, {'model': 'Infinity', 'parameters': 2000000000.0}, {'model': 'Fugatto 1', 'parameters': 2500000000.0}, {'model': 'Pixtral Large', 'parameters': 124000000000.0}, {'model': 'Hunyuan-Large', 'parameters': 389000000000.0}, {'model': 'Doubao-pro', 'parameters': 500000000000.0}, {'model': 'NVLM-X 72B', 'parameters': 72000000000.0}, {'model': 'NVLM-H 72B', 'parameters': 72000000000.0}, {'model': 'NVLM-D 72B', 'parameters': 72000000000.0}, {'model': 'RDT-1B', 'parameters': 1200000000.0}, {'model': 'Palmyra X 004', 'parameters': 150000000000.0}, {'model': 'GR-2', 'parameters': 230000000.0}, {'model': 'Movie Gen Video', 'parameters': 30000000000.0}, {'model': 'Llama 3.2 11B', 'parameters': 10600000000.0}, {'model': 'Telechat2-115B', 'parameters': 115000000000.0}, {'model': 'Qwen2.5-72B', 'parameters': 72700000000.0}, {'model': 'Qwen2.5 Instruct (72B)', 'parameters': 72700000000.0}, {'model': 'Oryx 34B', 'parameters': 34000000000.0}, {'model': 'Qwen2.5-32B', 'parameters': 32500000000.0}, {'model': 'DeepSeek-V2.5', 'parameters': 236000000000.0}, {'model': 'Jamba 1.5-Large', 'parameters': 398000000000.0}, {'model': 'Table Tennis Agent', 'parameters': 185000.0}, {'model': 'LLaVA-OV-72B', 'parameters': 72000000000.0}, {'model': 'AFM-on-device', 'parameters': 2730000000.0}, {'model': 'Mistral Large 2', 'parameters': 123000000000.0}, {'model': 'Llama 3.1-405B', 'parameters': 405000000000.0}, {'model': 'Mathstral', 'parameters': 7000000000.0}, {'model': 'SenseChat 5.5', 'parameters': 600000000000.0}, {'model': 'ESM3 (98B)', 'parameters': 98500000000.0}, {'model': 'Cambrian-1-34B', 'parameters': 34000000000.0}, {'model': 'DeepSeek-Coder-V2 236B', 'parameters': 236000000000.0}, {'model': 'Nemotron-4 340B', 'parameters': 340000000000.0}, {'model': 'OpenVLA', 'parameters': 7188100000.0}, {'model': 'Qwen2-72B', 'parameters': 72710000000.0}, {'model': 'ALLaM\xa0adapted 70B', 'parameters': 70000000000.0}, {'model': 'Octo-Base', 'parameters': 93000000.0}, {'model': 'Yi-Large', 'parameters': 100000000000.0}, {'model': 'VILA1.5-13B', 'parameters': 13493916740.0}, {'model': 'Llama 3-70B', 'parameters': 70000000000.0}, {'model': 'Reka Core', 'parameters': 67000000000.0}, {'model': 'ReALM', 'parameters': 3000000000.0}, {'model': 'DBRX', 'parameters': 132000000000.0}, {'model': 'MM1-30B', 'parameters': 30000000000.0}, {'model': 'Aramco Metabrain AI', 'parameters': 250000000000.0}, {'model': 'MegaScale (Production)', 'parameters': 530000000000.0}, {'model': 'Stable Diffusion 3', 'parameters': 8000000000.0}, {'model': 'Aya', 'parameters': 13000000000.0}, {'model': 'Qwen1.5-72B', 'parameters': 72000000000.0}, {'model': 'Qwen-VL-Max', 'parameters': 7000000000.0}, {'model': 'AlphaGeometry', 'parameters': 151000000.0}, {'model': 'Palmyra X 003', 'parameters': 72000000000.0}, {'model': 'CoRe', 'parameters': 12400000000.0}, {'model': 'GQA-8-XXL', 'parameters': 11000000000.0}, {'model': 'nekomata-14b', 'parameters': 14200000000.0}, {'model': 'Gemini Nano-2', 'parameters': 3250000000.0}, {'model': 'Gemini Nano-1', 'parameters': 1800000000.0}, {'model': 'FunSearch', 'parameters': 15000000000.0}, {'model': 'CogAgent', 'parameters': 18000000000.0}, {'model': 'VILA-13B', 'parameters': 13350839300.0}, {'model': 'W.A.L.T', 'parameters': 4719000000.0}, {'model': 'Mixtral 8x7B', 'parameters': 46700000000.0}, {'model': 'SeamlessM4T', 'parameters': 2300000000.0}, {'model': 'Llama Guard', 'parameters': 7000000000.0}, {'model': 'Mamba-24M (SC09)', 'parameters': 23400000.0}, {'model': 'Qwen-72B', 'parameters': 72000000000.0}, {'model': 'PPLX-70B-Online', 'parameters': 70000000000.0}, {'model': 'GNoME for crystal discovery', 'parameters': 16240000.0}, {'model': 'Nemotron-3-8B', 'parameters': 8000000000.0}, {'model': 'Qwen-Audio-Chat', 'parameters': 8460000000.0}, {'model': 'Volcano 13B', 'parameters': 13000000000.0}, {'model': 'SPHINX (Llama 2 13B)', 'parameters': 19900000000.0}, {'model': 'RoFormer', 'parameters': 110000000.0}, {'model': 'mPLUG-Owl2', 'parameters': 7120000000.0}, {'model': 'CogVLM-17B', 'parameters': 17000000000.0}, {'model': 'LLaVA 1.5', 'parameters': 13000000000.0}, {'model': 'Grok-1', 'parameters': 314000000000.0}, {'model': 'BLUUMI', 'parameters': 176000000000.0}, {'model': 'Yi-34B', 'parameters': 34000000000.0}, {'model': 'Skywork-13B', 'parameters': 13000000000.0}, {'model': 'ChatGLM3-6B', 'parameters': 6000000000.0}, {'model': 'DiT-XL/2 + CADS', 'parameters': 675000000.0}, {'model': 'CODEFUSION (Python)', 'parameters': 75000000.0}, {'model': 'PaLI-3', 'parameters': 5000000000.0}, {'model': 'RT-2-X', 'parameters': 55000000000.0}, {'model': 'Ferret (13B)', 'parameters': 13000000000.0}, {'model': 'FinGPT-13B', 'parameters': 13000000000.0}, {'model': 'GPT-3.5 Turbo Instruct', 'parameters': 20000000000.0}, {'model': 'Amazon Titan', 'parameters': 200000000000.0}, {'model': 'AlphaMissense', 'parameters': 93000000.0}, {'model': 'Robot Parkour', 'parameters': 500000.0}, {'model': 'Falcon-180B', 'parameters': 180000000000.0}, {'model': 'Swift', 'parameters': 56804.0}, {'model': 'Jais', 'parameters': 13000000000.0}, {'model': 'Qwen-VL', 'parameters': 9600000000.0}, {'model': 'RT-2', 'parameters': 55000000000.0}, {'model': 'AudioLM', 'parameters': 1500000000.0}, {'model': 'Llama 2-70B', 'parameters': 70000000000.0}, {'model': 'Llama 2-7B', 'parameters': 7000000000.0}, {'model': 'GPT3-2.7B (FlashAttention-2)', 'parameters': 2700000000.0}, {'model': 'xTrimoPGLM -100B', 'parameters': 100000000000.0}, {'model': 'InternLM', 'parameters': 104000000000.0}, {'model': 'Pangu-Weather', 'parameters': 256000000.0}, {'model': 'Stable Diffusion XL (SDXL)', 'parameters': 3400000000.0}, {'model': 'HyenaDNA', 'parameters': 6600000.0}, {'model': 'GPT-4 (Jun 2023)', 'parameters': 1800000000000.0}, {'model': 'GPT-3.5 Turbo', 'parameters': 20000000000.0}, {'model': 'MusicGen', 'parameters': 3359000000.0}, {'model': 'PaLI-X', 'parameters': 55000000000.0}, {'model': 'DPO on Pythia-2.8B', 'parameters': 2800000000.0}, {'model': 'Goat-7B', 'parameters': 7000000000.0}, {'model': 'CodeT5+', 'parameters': 16000000000.0}, {'model': 'ONE-PEACE', 'parameters': 4000000000.0}, {'model': 'CoEdiT-xxl', 'parameters': 11000000000.0}, {'model': 'Med-PaLM 2', 'parameters': 340000000000.0}, {'model': 'InstructBLIP', 'parameters': 13000000000.0}, {'model': 'PaLM 2', 'parameters': 340000000000.0}, {'model': 'StarCoder', 'parameters': 15500000000.0}, {'model': 'ImageBind', 'parameters': 932000000.0}, {'model': 'LLaVA', 'parameters': 13000000000.0}, {'model': 'DINOv2', 'parameters': 1140000000.0}, {'model': 'Incoder-6.7B', 'parameters': 6700000000.0}, {'model': 'Segment Anything Model', 'parameters': 636000000.0}, {'model': 'BloombergGPT', 'parameters': 50558868480.0}, {'model': 'VideoMAE V2', 'parameters': 1000000000.0}, {'model': 'SigLIP 400M', 'parameters': 400000000.0}, {'model': 'PanGu-Σ', 'parameters': 1085000000000.0}, {'model': 'LEP-AD', 'parameters': 3007381000.0}, {'model': 'GPT-4 (Mar 2023)', 'parameters': 1800000000000.0}, {'model': 'Falcon-40B', 'parameters': 40000000000.0}, {'model': 'PaLM-E', 'parameters': 562000000000.0}, {'model': 'AudioGen', 'parameters': 1000000000.0}, {'model': 'DiT-XL/2', 'parameters': 675000000.0}, {'model': 'LLaMA-65B', 'parameters': 65200000000.0}, {'model': 'BASIC-L + Lion', 'parameters': 3070000000.0}, {'model': 'ViT-22B', 'parameters': 21743000000.0}, {'model': 'BLIP-2 (Q-Former)', 'parameters': 1480000000.0}, {'model': 'DDPM-IP (CelebA)', 'parameters': 295000000.0}, {'model': 'MusicLM', 'parameters': 860000000.0}, {'model': 'Ankh_large', 'parameters': 1900000000.0}, {'model': 'Nucleotide Transformer', 'parameters': 2500000000.0}, {'model': 'DreamerV3', 'parameters': 200000000.0}, {'model': 'VALL-E', 'parameters': 353000000.0}, {'model': 'Hybrid H3-2.7B', 'parameters': 2700000000.0}, {'model': 'CaLM', 'parameters': 86000000.0}, {'model': 'RT-1', 'parameters': 35000000.0}, {'model': 'Vega v2', 'parameters': 6000000000.0}, {'model': 'ALM 1.0', 'parameters': 335000000.0}, {'model': 'AR-LDM', 'parameters': 1500000000.0}, {'model': 'Fusion in Encoder', 'parameters': 330000000.0}, {'model': 'Galactica', 'parameters': 120000000000.0}, {'model': 'EVA-01', 'parameters': 1011000000.0}, {'model': 'InternImage', 'parameters': 1080000000.0}, {'model': 'mT0-13B', 'parameters': 13000000000.0}, {'model': 'Mogrifier RLSTM (WT2)', 'parameters': 35000000.0}, {'model': 'eDiff-I', 'parameters': 9100000000.0}, {'model': 'U-PaLM (540B)', 'parameters': 540000000000.0}, {'model': 'LMSI-Palm', 'parameters': 540000000000.0}, {'model': 'Flan-PaLM 540B', 'parameters': 540000000000.0}, {'model': 'GenSLM', 'parameters': 25000000000.0}, {'model': 'Phenaki', 'parameters': 1800000000.0}, {'model': 'DiffDock', 'parameters': 20240000.0}, {'model': 'Whisper', 'parameters': 1550000000.0}, {'model': 'PaLI', 'parameters': 16900000000.0}, {'model': 'BEIT-3', 'parameters': 1900000000.0}, {'model': 'BlenderBot 3', 'parameters': 175000000000.0}, {'model': 'GLM-130B', 'parameters': 130000000000.0}, {'model': 'AlexaTM 20B', 'parameters': 19750000000.0}, {'model': 'OmegaPLM', 'parameters': 670000000.0}, {'model': 'ESM2-15B', 'parameters': 15000000000.0}, {'model': 'BLOOM-176B', 'parameters': 176247271400.0}, {'model': 'NLLB', 'parameters': 54500000000.0}, {'model': 'CodeT5-large', 'parameters': 770000000.0}, {'model': 'Minerva (540B)', 'parameters': 540350000000.0}, {'model': 'ProGen2-xlarge', 'parameters': 6400000000.0}, {'model': 'Parti', 'parameters': 20000000000.0}, {'model': 'CoCa', 'parameters': 2100000000.0}, {'model': 'DITTO', 'parameters': 750000000.0}, {'model': 'CogVideo', 'parameters': 9400000000.0}, {'model': 'Tranception', 'parameters': 700000000.0}, {'model': 'GPT-2 Medium (FlashAttention)', 'parameters': 355000000.0}, {'model': 'Imagen', 'parameters': 7762000000.0}, {'model': 'Gato', 'parameters': 1180000000.0}, {'model': 'UL2', 'parameters': 20000000000.0}, {'model': 'DeBERTaV3large + KEAR', 'parameters': 418000000.0}, {'model': 'OPT-175B', 'parameters': 175000000000.0}, {'model': 'Flamingo', 'parameters': 80000000000.0}, {'model': 'Sparse all-MLP', 'parameters': 9410000000.0}, {'model': 'Stable Diffusion (LDM-KL-8-G)', 'parameters': 1450000000.0}, {'model': 'BERT-RBP', 'parameters': 110000000.0}, {'model': 'DALL·E 2', 'parameters': 3500000000.0}, {'model': 'PaLM (540B)', 'parameters': 540350000000.0}, {'model': 'Chinchilla', 'parameters': 70000000000.0}, {'model': 'Make-A-Scene', 'parameters': 4000000000.0}, {'model': 'Segatron-XL large, M=384 + HCP', 'parameters': 257000000.0}, {'model': 'ViT-G (model soup)', 'parameters': 1843000000.0}, {'model': 'Statement Curriculum Learning', 'parameters': 774000000.0}, {'model': 'DeepNet', 'parameters': 3200000000.0}, {'model': 'PolyCoder', 'parameters': 2700000000.0}, {'model': 'ST-MoE', 'parameters': 269000000000.0}, {'model': 'ProteinBERT', 'parameters': 16000000.0}, {'model': 'LaMDA', 'parameters': 137000000000.0}, {'model': 'GPT-NeoX-20B', 'parameters': 20000000000.0}, {'model': 'MaskGIT (ImageNet)', 'parameters': 227000000.0}, {'model': 'RETRO-7B', 'parameters': 7500000000.0}, {'model': 'AlphaCode', 'parameters': 41100000000.0}, {'model': 'InstructGPT 175B', 'parameters': 175000000000.0}, {'model': 'InstructGPT 6B', 'parameters': 6000000000.0}, {'model': 'InstructGPT 1.3B', 'parameters': 1300000000.0}, {'model': 'OntoProtein', 'parameters': 420000000.0}, {'model': 'AbLang (heavy sequences)', 'parameters': 355000000.0}, {'model': 'data2vec (vision)', 'parameters': 705134592.0}, {'model': 'data2vec (speech)', 'parameters': 705134592.0}, {'model': 'data2vec (language)', 'parameters': 705134592.0}, {'model': 'Detic', 'parameters': 88000000.0}, {'model': 'ERNIE-ViLG', 'parameters': 10000000000.0}, {'model': 'ERNIE 3.0 Titan', 'parameters': 260000000000.0}, {'model': 'XGLM-7.5B', 'parameters': 7500000000.0}, {'model': 'LDM-1.45B', 'parameters': 1450000000.0}, {'model': 'Contriever', 'parameters': 110000000.0}, {'model': 'LongT5', 'parameters': 3000000000.0}, {'model': 'GLaM', 'parameters': 1200000000000.0}, {'model': 'Gopher (280B)', 'parameters': 280000000000.0}, {'model': 'T-NLRv5 XXL', 'parameters': 5400000000.0}, {'model': 'NÜWA', 'parameters': 870000000.0}, {'model': 'Florence', 'parameters': 893000000.0}, {'model': 'BASIC-L', 'parameters': 3070000000.0}, {'model': 'Swin Transformer V2 (SwinV2-G)', 'parameters': 3000000000.0}, {'model': 'ViT-G/14 (LiT)', 'parameters': 3005000000.0}, {'model': 'Masked Autoencoders ViT-H', 'parameters': 632000000.0}, {'model': 'CodeT5-base', 'parameters': 220000000.0}, {'model': 'S4', 'parameters': 249000000.0}, {'model': 'Eve', 'parameters': 15010300.0}, {'model': 'base LM+GNN+kNN', 'parameters': 274000000.0}, {'model': 'Yuan 1.0', 'parameters': 245730000000.0}, {'model': 'Megatron-Turing NLG 530B', 'parameters': 530000000000.0}, {'model': 'Turing ULRv5', 'parameters': 2200000000.0}, {'model': 'TrOCR', 'parameters': 558000000.0}, {'model': 'PLATO-XL', 'parameters': 11000000000.0}, {'model': 'HyperCLOVA 204B', 'parameters': 204000000000.0}, {'model': 'PermuteFormer', 'parameters': 149697024.0}, {'model': 'MEB', 'parameters': 135000000000.0}, {'model': 'FLAN 137B', 'parameters': 137000000000.0}, {'model': 'XLMR-XXL', 'parameters': 10700000000.0}, {'model': 'DNABERT', 'parameters': 110000000.0}, {'model': 'Zidong Taichu', 'parameters': 3200000000.0}, {'model': 'Jurassic-1-Jumbo', 'parameters': 178000000000.0}, {'model': 'W2v-BERT', 'parameters': 1000000000.0}, {'model': 'YOLOX-X', 'parameters': 99100000.0}, {'model': '6-Act Tether', 'parameters': 5000000.0}, {'model': 'SEER', 'parameters': 1300000000.0}, {'model': 'HuBERT', 'parameters': 1000000000.0}, {'model': 'GOAT', 'parameters': 3472816.0}, {'model': 'Codex', 'parameters': 12000000000.0}, {'model': 'ERNIE 3.0', 'parameters': 10000000000.0}, {'model': 'Adaptive Input Transformer + RD', 'parameters': 247000000.0}, {'model': 'Fold2Seq', 'parameters': 12427904.0}, {'model': 'EfficientNetV2-XL', 'parameters': 208000000.0}, {'model': 'StyleGAN3-T', 'parameters': 2230000.0}, {'model': 'StyleGAN3-R', 'parameters': 1580000.0}, {'model': 'Denoising Diffusion Probabilistic Models (LSUN Bedroom)', 'parameters': 256000000.0}, {'model': 'ALIGN', 'parameters': 820000000.0}, {'model': 'DeBERTa', 'parameters': 1500000000.0}, {'model': 'EMDR', 'parameters': 440000000.0}, {'model': 'CoAtNet', 'parameters': 2440000000.0}, {'model': 'ViT-G/14', 'parameters': 1843000000.0}, {'model': 'ByT5-XXL', 'parameters': 12900000000.0}, {'model': 'Transformer local-attention (NesT-B)', 'parameters': 90100000.0}, {'model': 'CogView', 'parameters': 4000000000.0}, {'model': 'MedBERT', 'parameters': 17000000.0}, {'model': 'ADM', 'parameters': 559000000.0}, {'model': 'ProtT5-XL-U50', 'parameters': 3000000000.0}, {'model': 'ProtBERT-BFD', 'parameters': 420000000.0}, {'model': 'PLUG', 'parameters': 27000000000.0}, {'model': 'Unicorn', 'parameters': 11000000000.0}, {'model': 'M6-T', 'parameters': 1002700000000.0}, {'model': 'Generative BST', 'parameters': 9431810048.0}, {'model': 'Meta Pseudo Labels', 'parameters': 480000000.0}, {'model': 'SRU++ Large', 'parameters': 234000000.0}, {'model': 'Rational DQN Average', 'parameters': 1683456.0}, {'model': 'MSA Transformer', 'parameters': 100000000.0}, {'model': 'DLWP', 'parameters': 2676376.0}, {'model': 'DeiT-B', 'parameters': 86000000.0}, {'model': 'Switch', 'parameters': 1571000000000.0}, {'model': 'BigSSL', 'parameters': 8000000000.0}, {'model': 'DALL-E', 'parameters': 12000000000.0}, {'model': 'CLIP (ViT L/14@336px)', 'parameters': 370000000.0}, {'model': 'CLIP (ResNet-50)', 'parameters': 88600000.0}, {'model': 'ERNIE-Doc (247M)', 'parameters': 247000000.0}, {'model': 'CT-MoS (WT2)', 'parameters': 45000000.0}, {'model': 'ESM1b', 'parameters': 652400000.0}, {'model': 'CPM-Large', 'parameters': 2600000000.0}, {'model': 'AlphaFold 2', 'parameters': 93000000.0}, {'model': 'KEPLER', 'parameters': 125000000.0}, {'model': 'wave2vec 2.0 LARGE', 'parameters': 317000000.0}, {'model': 'ViT-Huge/14', 'parameters': 632000000.0}, {'model': 'ViT-Base/32', 'parameters': 86000000.0}, {'model': 'German ELECTRA Large', 'parameters': 335000000.0}, {'model': 'mT5-XXL', 'parameters': 13000000000.0}, {'model': 'Conformer + Wav2vec 2.0 + Noisy Student', 'parameters': 1000000000.0}, {'model': 'LUKE', 'parameters': 483000000.0}, {'model': 'ProBERTa', 'parameters': 44000000.0}, {'model': 'ERNIE-GEN (large)', 'parameters': 340000000.0}, {'model': 'DeLighT', 'parameters': 99000000.0}, {'model': 'EfficientDet', 'parameters': 77000000.0}, {'model': 'GShard (dense)', 'parameters': 2300000000.0}, {'model': 'GPT-3 175B (davinci)', 'parameters': 174600000000.0}, {'model': 'DETR', 'parameters': 60000000.0}, {'model': 'Retrieval-Augmented Generator', 'parameters': 626000000.0}, {'model': 'Conformer', 'parameters': 118800000.0}, {'model': 'ContextNet', 'parameters': 112700000.0}, {'model': 'NAS+ESS (23M)', 'parameters': 23000000.0}, {'model': 'UnifiedQA', 'parameters': 11000000000.0}, {'model': 'Once for All', 'parameters': 7700000.0}, {'model': 'CURL', 'parameters': 907264.0}, {'model': 'MetNet', 'parameters': 225000000.0}, {'model': 'ELECTRA', 'parameters': 335000000.0}, {'model': 'Tensor-Transformer(1core)+PN (WT103)', 'parameters': 85300000.0}, {'model': 'Routing Transformer (WT-103)', 'parameters': 79500000.0}, {'model': 'TransformerXL + spectrum control', 'parameters': 151000000.0}, {'model': 'TCAN (WT2)', 'parameters': 33000000.0}, {'model': 'Feedback Transformer', 'parameters': 126000000.0}, {'model': 'FFN SwiGLU', 'parameters': 220000000.0}, {'model': 'Turing-NLG', 'parameters': 17000000000.0}, {'model': 'SimCLR', 'parameters': 375000000.0}, {'model': 'ALBERT-xxlarge', 'parameters': 235000000.0}, {'model': 'TaLK Convolution', 'parameters': 240000000.0}, {'model': 'Perceiver IO (optical flow)', 'parameters': 27900000.0}, {'model': 'Theseus 6/768', 'parameters': 66000000.0}, {'model': 'Meena', 'parameters': 2600000000.0}, {'model': 'AlphaFold', 'parameters': 16340840.0}, {'model': 'Big Transfer (BiT-L)', 'parameters': 928000000.0}, {'model': 'OpenAI Five Rerun', 'parameters': 159000000.0}, {'model': 'OpenAI Five', 'parameters': 159000000.0}, {'model': 'MMLSTM (WT-2)', 'parameters': 32300000.0}, {'model': 'MMLSTM (PTB)', 'parameters': 21300000.0}, {'model': 'StyleGAN2', 'parameters': 30000000.0}, {'model': 'Transformer-XL DeFINE (141M)', 'parameters': 141000000.0}, {'model': 'Transformer - LibriVox + Decoding/Rescoring', 'parameters': 296000000.0}, {'model': 'MuZero', 'parameters': 36864000.0}, {'model': 'MoCo', 'parameters': 375000000.0}, {'model': 'Noisy Student (L2)', 'parameters': 480000000.0}, {'model': 'Sandwich Transformer', 'parameters': 209000000.0}, {'model': 'CamemBERT', 'parameters': 335000000.0}, {'model': 'XLM-RoBERTa', 'parameters': 550000000.0}, {'model': 'Base LM + kNN LM + Continuous Cache', 'parameters': 247000000.0}, {'model': 'AlphaStar', 'parameters': 139000000.0}, {'model': 'BART-large', 'parameters': 406291456.0}, {'model': 'T5-11B', 'parameters': 11000000000.0}, {'model': 'T5-3B', 'parameters': 2800000000.0}, {'model': 'M4-50B', 'parameters': 50000000000.0}, {'model': 'DistilBERT', 'parameters': 66000000.0}, {'model': 'AlphaX-1', 'parameters': 5400000.0}, {'model': 'ALBERT', 'parameters': 18000000.0}, {'model': 'Adaptive Inputs + LayerDrop', 'parameters': 423000000.0}, {'model': 'Megatron-LM (8.3B)', 'parameters': 8300000000.0}, {'model': 'Megatron-LM (1.2B)', 'parameters': 1200000000.0}, {'model': 'Megatron-BERT', 'parameters': 3900000000.0}, {'model': 'UDSMProt', 'parameters': 28303800.0}, {'model': 'Mogrifier (d2, MoS2, MC) + dynamic eval', 'parameters': 35000000.0}, {'model': 'EN^2AS with performance reward', 'parameters': 23000000.0}, {'model': 'BigBiGAN', 'parameters': 86000000.0}, {'model': 'RoBERTa Large', 'parameters': 355000000.0}, {'model': 'LaNet-L (CIFAR-10)', 'parameters': 44100000.0}, {'model': 'FixRes ResNeXt-101 WSL', 'parameters': 829000000.0}, {'model': 'AWD-LSTM + MoS + Partial Shuffled', 'parameters': 35000000.0}, {'model': 'Transformer-XL Large + Phrase Induction', 'parameters': 257000000.0}, {'model': 'XLNet', 'parameters': 340000000.0}, {'model': 'XLM', 'parameters': 665000000.0}, {'model': 'DLRM-2020', 'parameters': 100000000000.0}, {'model': 'EfficientNet-L2', 'parameters': 480000000.0}, {'model': 'CPC v2', 'parameters': 303000000.0}, {'model': 'AWD-LSTM-DRILL + dynamic evaluation† (WT2)', 'parameters': 34000000.0}, {'model': 'ResNeXt-101 Billion-scale', 'parameters': 193000000.0}, {'model': 'MuseNet', 'parameters': 2038431744.0}, {'model': 'BERT-Large-CAS (PTB+WT2+WT103)', 'parameters': 395000000.0}, {'model': 'Transformer-XL + RMS dynamic eval', 'parameters': 257000000.0}, {'model': 'WeNet (Penn Treebank)', 'parameters': 23000000.0}, {'model': 'True-Regularization+Finetune+Dynamic-Eval', 'parameters': 7000000.0}, {'model': 'SciBERT', 'parameters': 110000000.0}, {'model': 'NMT Transformer 437M', 'parameters': 437700000.0}, {'model': 'KataGo', 'parameters': 2500000.0}, {'model': 'GPT-2 (1.5B)', 'parameters': 1500000000.0}, {'model': 'Hanabi 4 player', 'parameters': 764000.0}, {'model': 'MT-DNN', 'parameters': 330000000.0}, {'model': 'Transformer-XL (257M)', 'parameters': 257000000.0}, {'model': 'Transformer ELMo', 'parameters': 56000000.0}, {'model': 'StyleGAN', 'parameters': 26200000.0}, {'model': 'SPN (ImageNet 128)', 'parameters': 250000000.0}, {'model': 'GPipe (Transformer)', 'parameters': 6000000000.0}, {'model': 'Multi-cell LSTM', 'parameters': 7200000.0}, {'model': 'Fine-tuned-AWD-LSTM-DOC (fin)', 'parameters': 46000000.0}, {'model': 'Mesh-TensorFlow Transformer 4.9B (language)', 'parameters': 4900000000.0}, {'model': 'Mesh-TensorFlow Transformer 2.9B (translation)', 'parameters': 2900000000.0}, {'model': 'TrellisNet', 'parameters': 180000000.0}, {'model': 'MetaMimic', 'parameters': 22000000.0}, {'model': 'BERT-Large', 'parameters': 340000000.0}, {'model': 'Transformer (Adaptive Input Embeddings) WT103', 'parameters': 247000000.0}, {'model': 'LSTM+NeuralCache', 'parameters': 2100000.0}, {'model': 'AWD-LSTM-MoS + dynamic evaluation (WT2, 2018)', 'parameters': 35000000.0}, {'model': 'Transformer + Simple Recurrent Unit', 'parameters': 90000000.0}, {'model': '(ensemble): AWD-LSTM-DOC (fin) × 5 (WT2)', 'parameters': 185000000.0}, {'model': 'AWD-LSTM-MoS+PDR + dynamic evaluation (WT2)', 'parameters': 35000000.0}, {'model': 'Big-Little Net (speech)', 'parameters': 3320000.0}, {'model': 'Big-Little Net', 'parameters': 77360000.0}, {'model': 'FTW (For The Win)', 'parameters': 126001330.0}, {'model': 'MobileNetV2', 'parameters': 3400000.0}, {'model': 'GPT-1', 'parameters': 117000000.0}, {'model': 'aLSTM(depth-2)+RecurrentPolicy (WT2)', 'parameters': 32000000.0}, {'model': 'Dropout-LSTM+Noise(Bernoulli) (WT2)', 'parameters': 51000000.0}, {'model': 'ResNeXt-101 32x48d', 'parameters': 829000000.0}, {'model': 'YOLOv3', 'parameters': 56933216.0}, {'model': '4 layer QRNN (h=2500)', 'parameters': 151000000.0}, {'model': 'TCN (P-MNIST)', 'parameters': 42000.0}, {'model': 'IMPALA', 'parameters': 1600000.0}, {'model': 'QRNN', 'parameters': 135000000.0}, {'model': 'ELMo', 'parameters': 94000000.0}, {'model': 'DL scaling speech', 'parameters': 193000000.0}, {'model': 'DL scaling LM', 'parameters': 177000000.0}, {'model': 'DL scaling Image', 'parameters': 121000000.0}, {'model': 'AWD-LSTM-MoS + dynamic evaluation (WT2, 2017)', 'parameters': 35000000.0}, {'model': 'Fraternal dropout + AWD-LSTM 3-layer (WT2)', 'parameters': 34000000.0}, {'model': 'AlphaGo Zero', 'parameters': 46400244.0}, {'model': 'AWD-LSTM+WT+Cache+IOG (WT2)', 'parameters': 53000000.0}, {'model': 'LSTM + dynamic eval', 'parameters': 50000000.0}, {'model': 'ISS', 'parameters': 11100000.0}, {'model': 'PyramidNet', 'parameters': 26000000.0}, {'model': 'SENet (ImageNet)', 'parameters': 28100000.0}, {'model': 'GL-LWGC-AWD-MoS-LSTM + dynamic evaluation (WT2)', 'parameters': 38000000.0}, {'model': 'EI-REHN-1000D', 'parameters': 19000000.0}, {'model': 'RetinaNet-R101', 'parameters': 53000000.0}, {'model': 'RetinaNet-R50', 'parameters': 34000000.0}, {'model': 'AWD-LSTM - 3-layer LSTM (tied) + continuous cache pointer (WT2)', 'parameters': 33000000.0}, {'model': 'NASNet-A', 'parameters': 89000000.0}, {'model': 'AWD-LSTM', 'parameters': 24000000.0}, {'model': 'JFT', 'parameters': 44654504.0}, {'model': 'ShuffleNet v1', 'parameters': 2430000.0}, {'model': 'Transformer', 'parameters': 213000000.0}, {'model': 'MobileNet', 'parameters': 4200000.0}, {'model': 'MoE-Multi', 'parameters': 8700000000.0}, {'model': 'OR-WideResNet', 'parameters': 18200000.0}, {'model': 'DeepStack', 'parameters': 2500000.0}, {'model': 'YOLOv2', 'parameters': 51000000.0}, {'model': 'EnhanceNet', 'parameters': 814464.0}, {'model': '3DMM-CNN', 'parameters': 44500000.0}, {'model': 'HR-ResNet101', 'parameters': 44500000.0}, {'model': 'PolyNet', 'parameters': 92000000.0}, {'model': 'ResNeXt-101 (64×4d)', 'parameters': 83000000.0}, {'model': 'ResNeXt-50', 'parameters': 25000000.0}, {'model': 'DLDL (PASCAL)', 'parameters': 564000000.0}, {'model': 'NASv3 (CIFAR-10)', 'parameters': 37400000.0}, {'model': 'NAS with base 8 and shared embeddings', 'parameters': 54000000.0}, {'model': 'BIDAF', 'parameters': 2600000.0}, {'model': 'VD-LSTM+REAL Large', 'parameters': 51000000.0}, {'model': 'SPIDER2', 'parameters': 409536.0}, {'model': 'Xception', 'parameters': 22855952.0}, {'model': 'Pointer Sentinel-LSTM (medium)', 'parameters': 21000000.0}, {'model': 'GNMT', 'parameters': 278000000.0}, {'model': 'ResNet-1001', 'parameters': 10200000.0}, {'model': 'MS-ensemble-speech-recognition', 'parameters': 3172117056.0}, {'model': 'LF-MMI', 'parameters': 16600000.0}, {'model': 'DenseNet-264', 'parameters': 34000000.0}, {'model': 'SimpleNet', 'parameters': 5480000.0}, {'model': 'CMS-RCNN', 'parameters': 138000000.0}, {'model': 'LRR-4X', 'parameters': 138000000.0}, {'model': 'Gated HORNN (3rd order)', 'parameters': 8970000.0}, {'model': 'Dueling DQN', 'parameters': 1700000.0}, {'model': 'Template Adaptation', 'parameters': 138000000.0}, {'model': 'Double DQN', 'parameters': 1500000.0}, {'model': 'SqueezeNet', 'parameters': 1200000.0}, {'model': 'Inceptionv4', 'parameters': 43000000.0}, {'model': 'Inception-ResNet-V2', 'parameters': 56000000.0}, {'model': 'Variational (untied weights, MC) LSTM (Large)', 'parameters': 66000000.0}, {'model': 'ResNet-152 (ImageNet)', 'parameters': 60200000.0}, {'model': 'ResNet-110 (CIFAR-10)', 'parameters': 1700000.0}, {'model': 'ResNet-101 (ImageNet)', 'parameters': 44500000.0}, {'model': 'Inception v3', 'parameters': 23626728.0}, {'model': '3DDFA', 'parameters': 5355584.0}, {'model': 'SAF R-CNN', 'parameters': 138000000.0}, {'model': 'AlphaGo Fan', 'parameters': 8209984.0}, {'model': 'DCNN', 'parameters': 5006000.0}, {'model': 'Deep CNN + COTS', 'parameters': 5006000.0}, {'model': 'BatchNorm', 'parameters': 13600000.0}, {'model': 'YOLO', 'parameters': 271684800.0}, {'model': 'CFSS', 'parameters': 17408.0}, {'model': 'U-Net', 'parameters': 37676160.0}, {'model': 'TC-DNN-BLSTM-DNN', 'parameters': 18413568.0}, {'model': 'genCNN + dyn eval', 'parameters': 8000000.0}, {'model': 'DQN-2015', 'parameters': 1693362.0}, {'model': 'TRPO', 'parameters': 33500.0}, {'model': 'MSRA (C, PReLU)', 'parameters': 87048800.0}, {'model': 'VGG-Face', 'parameters': 138000000.0}, {'model': 'ADAM (CIFAR-10)', 'parameters': 2370000.0}, {'model': 'Fractional Max-Pooling', 'parameters': 27000000.0}, {'model': 'SNM-skip', 'parameters': 62000000000.0}, {'model': 'TA-CNN', 'parameters': 706048.0}, {'model': 'LRCN', 'parameters': 142552000.0}, {'model': 'GoogLeNet / InceptionV1', 'parameters': 6797700.0}, {'model': 'SPN-4+KN5', 'parameters': 5000000.0}, {'model': 'Seq2Seq LSTM', 'parameters': 1920000000.0}, {'model': 'VGG19', 'parameters': 144000000.0}, {'model': 'VGG16', 'parameters': 138000000.0}, {'model': 'NPD', 'parameters': 313856.0}, {'model': 'ACF-WIDER', 'parameters': 6144.0}, {'model': 'Multiresolution CNN', 'parameters': 126125568.0}, {'model': 'RNN-WER', 'parameters': 26500000.0}, {'model': 'Fragment embedding', 'parameters': 144496000.0}, {'model': 'Dropout: SVHN', 'parameters': 47795232.0}, {'model': 'AdaRNN', 'parameters': 13040.0}, {'model': 'Paragraph Vector', 'parameters': 32000000.0}, {'model': 'HyperNEAT', 'parameters': 239712.0}, {'model': 'GloVe (32B)', 'parameters': 120000000.0}, {'model': 'GloVe (6B)', 'parameters': 120000000.0}, {'model': 'OverFeat', 'parameters': 144000000.0}, {'model': 'Image generation', 'parameters': 784000.0}, {'model': 'DQN', 'parameters': 836096.0}, {'model': 'RNN for 1B words', 'parameters': 20000000000.0}, {'model': 'TransE', 'parameters': 942000000.0}, {'model': 'R-CNN (T-net)', 'parameters': 69003872.0}, {'model': 'Word2Vec (small)', 'parameters': 207600000.0}, {'model': 'Word2Vec (large)', 'parameters': 692000000.0}, {'model': 'Hierarchical Scene Labeling (Stanford Background)', 'parameters': 51609600.0}, {'model': 'ReLU-Speech', 'parameters': 101706240.0}, {'model': 'Multilingual DNN', 'parameters': 206899200.0}, {'model': 'PreTrans-3L-250H', 'parameters': 43000000.0}, {'model': 'DNN EM segmentation', 'parameters': 218896.0}, {'model': 'DistBelief Vision', 'parameters': 1700000000.0}, {'model': 'DistBelief Speech', 'parameters': 47185920.0}, {'model': 'RNN+LDA+KN5+cache', 'parameters': 9000000.0}, {'model': 'AlexNet', 'parameters': 60000000.0}, {'model': 'LSTM LM', 'parameters': 102720000.0}, {'model': 'Unsupervised High-level Feature Learner', 'parameters': 1000000000.0}, {'model': 'Dropout (TIMIT)', 'parameters': 48840185.0}, {'model': 'Dropout (MNIST)', 'parameters': 5594010.0}, {'model': 'NLP from scratch', 'parameters': 5000000.0}, {'model': 'CNN committee (traffic sign)', 'parameters': 1388800.0}, {'model': 'CNN Committee (NIST)', 'parameters': 128420.0}, {'model': 'CNN Committee (MNIST)', 'parameters': 120620.0}, {'model': 'High Performance CNN (NORB)', 'parameters': 4878300.0}, {'model': 'Vector Space Model', 'parameters': 255000.0}, {'model': 'Deep Autoencoders', 'parameters': 139808256.0}, {'model': 'RNN LM', 'parameters': 70265000.0}, {'model': 'Pooling CNN (NORB)', 'parameters': 268664.0}, {'model': 'Pooling CNN (Caltech 101)', 'parameters': 294912.0}, {'model': 'ReLU (NORB)', 'parameters': 16210006.0}, {'model': 'Feedforward NN', 'parameters': 7082000.0}, {'model': 'Super-vector coding', 'parameters': 1025.0}, {'model': 'LCNP NORB', 'parameters': 16818176.0}, {'model': 'LCNP MNIST', 'parameters': 11616256.0}, {'model': 'LCNP LabelMe', 'parameters': 13729792.0}, {'model': 'Two Stage Feature Extraction (MNIST)', 'parameters': 258800.0}, {'model': 'GPU DBNs', 'parameters': 100000000.0}, {'model': 'RBM Image Classifier', 'parameters': 80000000.0}, {'model': 'BP-DBN', 'parameters': 18030592.0}, {'model': 'GNN', 'parameters': 30.0}, {'model': 'HLBL', 'parameters': 1846400.0}, {'model': 'Deep Multitask NLP Network', 'parameters': 1500000.0}, {'model': 'BLSTM for handwriting (2)', 'parameters': 100881.0}, {'model': 'SB-LM', 'parameters': 300000000000.0}, {'model': 'KN-LM', 'parameters': 21000000000.0}, {'model': 'Dimensionality Reduction', 'parameters': 3800000.0}, {'model': 'Deep Belief Nets', 'parameters': 1600000.0}, {'model': 'SVM-CNN', 'parameters': 90857.0}, {'model': 'RankNet', 'parameters': 5711.0}, {'model': 'LMICA', 'parameters': 4096000.0}, {'model': 'Invariant CNN', 'parameters': 90575.0}, {'model': 'NPLM (Brown)', 'parameters': 4124233.0}, {'model': 'NPLM (AP News)', 'parameters': 11904264.0}, {'model': 'Decision tree (classification)', 'parameters': 12000.0}, {'model': 'PoE MNIST', 'parameters': 3925310.0}, {'model': 'Neural LM', 'parameters': 6906980.0}, {'model': 'RECONTRA-uncategorized', 'parameters': 112000.0}, {'model': 'RECONTRA-categorized', 'parameters': 66780.0}, {'model': 'LSTM with forget gates', 'parameters': 276.0}, {'model': 'LeNet-5', 'parameters': 60000.0}, {'model': 'LSTM', 'parameters': 10504.0}, {'model': 'Bidirectional RNN', 'parameters': 13000.0}, {'model': 'Deep Blue', 'parameters': 8000.0}, {'model': 'SOM-CNN', 'parameters': 32015.0}, {'model': 'System 11', 'parameters': 6452.0}, {'model': 'MUSIC perceptron', 'parameters': 13607.0}, {'model': 'LISSOM', 'parameters': 432800.0}, {'model': 'Support Vector Machines', 'parameters': 100000000.0}, {'model': 'Predictive Coding NN', 'parameters': 206910.0}, {'model': 'NeuroChess', 'parameters': 72251.0}, {'model': 'Mixture of linear models', 'parameters': 384000.0}, {'model': 'JPMAX', 'parameters': 4446.0}, {'model': 'Ceramic-MLP', 'parameters': 1888.0}, {'model': 'ANN Eye Tracker', 'parameters': 5620.0}, {'model': 'Siamese-TDNN', 'parameters': 744.0}, {'model': 'IBM-5', 'parameters': 1658364.0}, {'model': 'Boosting', 'parameters': 2578.0}, {'model': 'Cancer drug mechanism prediction', 'parameters': 594.0}, {'model': 'TD-Gammon', 'parameters': 25000.0}, {'model': 'Weight Decay', 'parameters': 8386.0}, {'model': 'RAAM', 'parameters': 1536.0}, {'model': 'SexNet compression', 'parameters': 72940.0}, {'model': 'SexNet classification', 'parameters': 1640.0}, {'model': 'Bankruptcy-NN', 'parameters': 36.0}, {'model': 'NETtalk reimplementation', 'parameters': 27480.0}, {'model': 'Zip CNN', 'parameters': 9760.0}, {'model': 'Speaker-independent vowel classification', 'parameters': 3040.0}, {'model': 'Handwritten digit recognition network', 'parameters': 2578.0}, {'model': 'Truck backer-upper', 'parameters': 805.0}, {'model': 'MLN-ASR', 'parameters': 10000.0}, {'model': 'Translation-invariant MLP', 'parameters': 816.0}, {'model': 'NetTalk (transcription)', 'parameters': 18629.0}, {'model': 'NetTalk (dictionary)', 'parameters': 18629.0}, {'model': 'MLP with back-propagation', 'parameters': 720.0}, {'model': 'Distributed representation NN', 'parameters': 432.0}, {'model': 'Hierarchical Cognitron', 'parameters': 9315.0}, {'model': 'ASE+ACE', 'parameters': 324.0}, {'model': 'Hopfield network', 'parameters': 9900.0}, {'model': 'Kohonen network', 'parameters': 4096.0}, {'model': 'Neocognitron', 'parameters': 1140576.0}, {'model': 'Cognitron', 'parameters': 21600.0}, {'model': 'Piecewise linear model', 'parameters': 357.0}, {'model': 'Decision tree adaline', 'parameters': 2450.0}, {'model': 'LTE speaker verification system', 'parameters': 2061.0}, {'model': 'ADALINE', 'parameters': 17.0}, {'model': 'Perceptron (1960)', 'parameters': 1000.0}, {'model': 'Pattern recognition and reading by machine', 'parameters': 2625.0}, {'model': 'Samuel Neural Checkers', 'parameters': 16.0}, {'model': 'Perceptron Mark I', 'parameters': 1000.0}, {'model': 'Self Organizing System', 'parameters': 225.0}, {'model': 'SNARC', 'parameters': 40.0}, {'model': 'Theseus', 'parameters': 40.0}]
    years, log_params = (np.array([2026.720055, 2026.711841, 2026.687201, 2026.654346, 2026.632444, 2026.616016, 2026.599589, 2026.596851, 2026.577687, 2026.572211, 2026.544832, 2026.536619, 2026.533881, 2026.50924, 2026.490075, 2026.487337, 2026.421629, 2026.413415, 2026.375086, 2026.355921, 2026.309377, 2026.309377, 2026.306639, 2026.306639, 2026.298426, 2026.268309, 2026.262834, 2026.257358, 2026.210815, 2026.208077, 2026.188912, 2026.147844, 2026.128679, 2026.123203, 2026.11499, 2026.087611, 2026.087611, 2025.996578, 2025.996578, 2025.99384, 2025.99384, 2025.991102, 2025.974675, 2025.971937, 2025.952772, 2025.903491, 2025.884326, 2025.876112, 2025.876112, 2025.848734, 2025.848734, 2025.848734, 2025.845996, 2025.821355, 2025.818617, 2025.772074, 2025.744695, 2025.722793, 2025.706366, 2025.676249, 2025.665298, 2025.591376, 2025.591376, 2025.588638, 2025.588638, 2025.580424, 2025.569473, 2025.561259, 2025.561259, 2025.553046, 2025.533881, 2025.52293, 2025.517454, 2025.517454, 2025.443532, 2025.440794, 2025.424367, 2025.402464, 2025.323066, 2025.271047, 2025.257358, 2025.257358, 2025.257358, 2025.224504, 2025.219028, 2025.202601, 2025.202601, 2025.188912, 2025.175222, 2025.128679, 2025.052019, 2025.052019, 2025.043806, 2025.002738, 2024.999316, 2024.980151, 2024.950034, 2024.939083, 2024.930869, 2024.928131, 2024.928131, 2024.900753, 2024.881588, 2024.848734, 2024.824093, 2024.807666, 2024.807666, 2024.807666, 2024.774812, 2024.772074, 2024.769336, 2024.758385, 2024.731006, 2024.720055, 2024.717317, 2024.717317, 2024.717317, 2024.711841, 2024.681725, 2024.640657, 2024.599589, 2024.596851, 2024.574949, 2024.561259, 2024.558522, 2024.539357, 2024.511978, 2024.481862, 2024.479124, 2024.459959, 2024.451745, 2024.449008, 2024.43258, 2024.386037, 2024.383299, 2024.364134, 2024.336756, 2024.295688, 2024.287474, 2024.240931, 2024.235455, 2024.199863, 2024.172485, 2024.145106, 2024.142368, 2024.11499, 2024.093087, 2024.065708, 2024.043806, 2024.0, 2023.991102, 2023.974675, 2023.969199, 2023.963723, 2023.963723, 2023.950034, 2023.950034, 2023.944559, 2023.941821, 2023.941821, 2023.933607, 2023.930869, 2023.914442, 2023.911704, 2023.908966, 2023.908966, 2023.870637, 2023.867899, 2023.865161, 2023.865161, 2023.851472, 2023.848734, 2023.845996, 2023.843258, 2023.84052, 2023.837782, 2023.835044, 2023.826831, 2023.818617, 2023.81588, 2023.81588, 2023.791239, 2023.780287, 2023.774812, 2023.76386, 2023.73922, 2023.73922, 2023.722793, 2023.695414, 2023.678987, 2023.659822, 2023.657084, 2023.643395, 2023.569473, 2023.563997, 2023.542094, 2023.542094, 2023.542094, 2023.50924, 2023.50924, 2023.506502, 2023.503765, 2023.4846, 2023.44627, 2023.44627, 2023.43258, 2023.405202, 2023.405202, 2023.388775, 2023.380561, 2023.375086, 2023.372348, 2023.36961, 2023.355921, 2023.353183, 2023.350445, 2023.350445, 2023.290212, 2023.281999, 2023.268309, 2023.257358, 2023.240931, 2023.238193, 2023.232717, 2023.213552, 2023.199863, 2023.199863, 2023.199863, 2023.175222, 2023.172485, 2023.164271, 2023.147844, 2023.117728, 2023.109514, 2023.079398, 2023.071184, 2023.068446, 2023.041068, 2023.03833, 2023.024641, 2023.010951, 2022.988364, 2022.963723, 2022.947296, 2022.922656, 2022.906229, 2022.884326, 2022.87885, 2022.873374, 2022.867899, 2022.856947, 2022.837782, 2022.837782, 2022.835044, 2022.799452, 2022.799452, 2022.799452, 2022.774812, 2022.758385, 2022.755647, 2022.720055, 2022.70089, 2022.637919, 2022.605065, 2022.588638, 2022.583162, 2022.553046, 2022.550308, 2022.52293, 2022.50924, 2022.506502, 2022.490075, 2022.4846, 2022.47091, 2022.449008, 2022.427105, 2022.405202, 2022.399726, 2022.399726, 2022.388775, 2022.358658, 2022.353183, 2022.336756, 2022.33128, 2022.323066, 2022.281999, 2022.279261, 2022.262834, 2022.260096, 2022.25462, 2022.238193, 2022.224504, 2022.21629, 2022.186174, 2022.164271, 2022.161533, 2022.15332, 2022.128679, 2022.109514, 2022.109514, 2022.106776, 2022.104038, 2022.1013, 2022.087611, 2022.071184, 2022.071184, 2022.071184, 2022.060233, 2022.057495, 2022.052019, 2022.052019, 2022.052019, 2022.016427, 2021.996578, 2021.974675, 2021.966461, 2021.966461, 2021.95551, 2021.952772, 2021.947296, 2021.933607, 2021.919918, 2021.895277, 2021.889802, 2021.881588, 2021.87885, 2021.870637, 2021.859685, 2021.832307, 2021.829569, 2021.818617, 2021.791239, 2021.77755, 2021.774812, 2021.73922, 2021.720055, 2021.717317, 2021.689938, 2021.678987, 2021.673511, 2021.670773, 2021.62423, 2021.618754, 2021.607803, 2021.607803, 2021.596851, 2021.594114, 2021.5859, 2021.572211, 2021.566735, 2021.566735, 2021.511978, 2021.506502, 2021.487337, 2021.476386, 2021.473648, 2021.468172, 2021.468172, 2021.440794, 2021.440794, 2021.438056, 2021.435318, 2021.435318, 2021.43258, 2021.402464, 2021.396988, 2021.396988, 2021.380561, 2021.355921, 2021.336756, 2021.336756, 2021.295688, 2021.224504, 2021.172485, 2021.172485, 2021.161533, 2021.147844, 2021.131417, 2021.117728, 2021.106776, 2021.03833, 2021.027379, 2021.024641, 2021.010951, 2021.010951, 2021.010951, 2020.999316, 2020.982888, 2020.95551, 2020.91718, 2020.914442, 2020.895277, 2020.807666, 2020.807666, 2020.807666, 2020.804928, 2020.80219, 2020.80219, 2020.752909, 2020.668036, 2020.596851, 2020.588638, 2020.569473, 2020.495551, 2020.405202, 2020.399726, 2020.388775, 2020.372348, 2020.347707, 2020.344969, 2020.334018, 2020.325804, 2020.268309, 2020.227242, 2020.224504, 2020.208077, 2020.194387, 2020.19165, 2020.158795, 2020.13963, 2020.120465, 2020.117728, 2020.117728, 2020.106776, 2020.104038, 2020.104038, 2020.1013, 2020.073922, 2020.03833, 2019.977413, 2019.947296, 2019.947296, 2019.925394, 2019.925394, 2019.919918, 2019.903491, 2019.881588, 2019.881588, 2019.865161, 2019.859685, 2019.856947, 2019.856947, 2019.843258, 2019.832307, 2019.826831, 2019.824093, 2019.807666, 2019.807666, 2019.774812, 2019.750171, 2019.750171, 2019.733744, 2019.731006, 2019.709103, 2019.709103, 2019.709103, 2019.673511, 2019.673511, 2019.553046, 2019.503765, 2019.495551, 2019.457221, 2019.449008, 2019.438056, 2019.421629, 2019.413415, 2019.413415, 2019.410678, 2019.402464, 2019.386037, 2019.364134, 2019.33128, 2019.312115, 2019.298426, 2019.290212, 2019.265572, 2019.265572, 2019.229979, 2019.158795, 2019.156057, 2019.120465, 2019.084873, 2019.082136, 2019.021903, 2019.0, 2018.944559, 2018.922656, 2018.873374, 2018.870637, 2018.862423, 2018.843258, 2018.843258, 2018.785763, 2018.774812, 2018.774812, 2018.73922, 2018.728268, 2018.711841, 2018.709103, 2018.659822, 2018.616016, 2018.520192, 2018.520192, 2018.501027, 2018.459959, 2018.413415, 2018.386037, 2018.334018, 2018.33128, 2018.265572, 2018.219028, 2018.123203, 2018.095825, 2018.084873, 2018.084873, 2017.914442, 2017.914442, 2017.914442, 2017.856947, 2017.829569, 2017.793977, 2017.733744, 2017.720055, 2017.703628, 2017.678987, 2017.676249, 2017.657084, 2017.616016, 2017.596851, 2017.596851, 2017.596851, 2017.550308, 2017.542094, 2017.520192, 2017.501027, 2017.443532, 2017.290212, 2017.060233, 2017.016427, 2017.013689, 2016.982888, 2016.977413, 2016.95551, 2016.950034, 2016.87885, 2016.876112, 2016.876112, 2016.848734, 2016.845996, 2016.845996, 2016.845996, 2016.843258, 2016.824093, 2016.766598, 2016.736482, 2016.736482, 2016.711841, 2016.698152, 2016.687201, 2016.648871, 2016.640657, 2016.459959, 2016.350445, 2016.328542, 2016.260096, 2016.194387, 2016.167009, 2016.147844, 2016.145106, 2016.145106, 2015.95551, 2015.939083, 2015.939083, 2015.939083, 2015.91718, 2015.892539, 2015.821355, 2015.747433, 2015.596851, 2015.563997, 2015.451745, 2015.43258, 2015.429843, 2015.375086, 2015.260096, 2015.205339, 2015.150582, 2015.134155, 2015.098563, 2015.0, 2014.971937, 2014.960986, 2014.919918, 2014.908966, 2014.848734, 2014.709103, 2014.70089, 2014.689938, 2014.673511, 2014.673511, 2014.594114, 2014.533881, 2014.473648, 2014.47091, 2014.468172, 2014.413415, 2014.413415, 2014.364134, 2014.172485, 2014.0, 2014.0, 2013.969199, 2013.966461, 2013.963723, 2013.941821, 2013.925394, 2013.859685, 2013.788501, 2013.788501, 2013.580424, 2013.396988, 2013.396988, 2013.219028, 2012.922656, 2012.922656, 2012.922656, 2012.91718, 2012.747433, 2012.689938, 2012.528405, 2012.421629, 2012.421629, 2011.851472, 2011.752909, 2011.711841, 2011.711841, 2011.536619, 2011.462697, 2011.323066, 2010.733744, 2010.703628, 2010.703628, 2010.451745, 2010.361396, 2010.0, 2009.889802, 2009.889802, 2009.889802, 2009.665298, 2009.451745, 2009.265572, 2009.0, 2008.939083, 2008.936345, 2008.50924, 2007.919918, 2007.47091, 2007.47091, 2006.542094, 2006.542094, 2006.457221, 2005.596851, 2004.91718, 2004.487337, 2003.199863, 2003.199863, 2001.933607, 2000.908966, 2000.908966, 1999.416153, 1999.416153, 1999.002738, 1998.832307, 1997.870637, 1997.832307, 1997.328542, 1997.082136, 1996.462697, 1996.421629, 1995.903491, 1995.665298, 1994.91718, 1994.91718, 1994.91718, 1994.91718, 1994.016427, 1993.908966, 1993.580424, 1993.451745, 1992.914442, 1992.791239, 1992.33128, 1991.91718, 1990.832307, 1990.747433, 1990.747433, 1990.457221, 1990.413415, 1989.914442, 1989.903491, 1989.903491, 1989.459959, 1988.583162, 1987.451745, 1987.427105, 1987.427105, 1986.747433, 1986.618754, 1984.249144, 1983.665298, 1982.246407, 1981.561259, 1980.249144, 1975.665298, 1973.832307, 1969.328542, 1966.832307, 1960.495551, 1960.243669, 1959.914442, 1959.495551, 1957.0, 1955.161533, 1952.019165, 1950.498289]), np.array([9.322219295, 11.77815125, 11.8739016, 11.88649073, 11.50514998, 11.87157294, 12.38021124, 11.49692965, 11.87506126, 11.83758844, 12.38021124, 12.44715803, 11.98900462, 11.46982202, 12.20411998, 11.39846085, 11.74036269, 11.63144377, 12.01703334, 11.44090908, 12.20411998, 11.45331834, 11.46982202, 12.00860017, 12.01703334, 10.52113808, 11.87737135, 9.739201131, 12.01703334, 12.00860017, 11.07918125, 11.08635983, 11.69897, 11.59879051, 11.87157294, 10.90308999, 12.01703334, 11.00860017, 11.372912, 11.0, 11.71516736, 10.50514998, 11.35983548, 11.55388303, 10.49968708, 11.83569057, 10.50514998, 9.720985744, 11.37106786, 10.11394335, 10.53147892, 11.04139269, 12.0, 10.48429984, 11.35983548, 12.0, 11.55266822, 10.54777471, 10.47712125, 12.0, 11.74818803, 11.06755438, 10.32035403, 10.43136376, 7.431363764, 10.8573325, 11.55022835, 11.37106786, 11.37106786, 11.68124124, 10.50514998, 12.0, 12.47712125, 8.243038049, 8.857332496, 11.36172784, 9.903089987, 11.82672252, 11.37106786, 11.13033377, 11.0374265, 11.60205999, 12.30103, 11.82672252, 9.041392685, 10.50514998, 11.62736586, 11.74818803, 10.51188336, 12.47712125, 9.950851459, 11.82672252, 9.84509804, 9.84509804, 8.002592399, 11.82672252, 9.84509804, 10.50514998, 10.84509804, 10.17609126, 9.301029996, 9.397940009, 11.09342169, 11.5899496, 11.69897, 10.8573325, 10.8573325, 10.8573325, 9.079181246, 11.17609126, 8.361727836, 10.47712125, 10.02530587, 11.06069784, 10.86153441, 10.86153441, 10.53147892, 10.51188336, 11.372912, 11.59988307, 5.267171728, 10.8573325, 9.436162647, 11.08990511, 11.60745502, 9.84509804, 11.77815125, 10.99343623, 10.53147892, 11.372912, 11.53147892, 9.85661411, 10.86159414, 10.84509804, 7.968482949, 11.0, 10.13013803, 10.84509804, 10.8260748, 9.477121255, 11.12057393, 10.47712125, 11.39794001, 11.72427587, 9.903089987, 10.11394335, 10.8573325, 9.84509804, 8.178976947, 10.8573325, 10.09342169, 10.04139269, 10.15228834, 9.511883361, 9.255272505, 10.17609126, 10.25527251, 10.12550857, 9.673849977, 10.66931688, 9.361727836, 9.84509804, 7.369215857, 10.8573325, 10.84509804, 7.210586025, 9.903089987, 9.927370363, 10.11394335, 10.29885308, 8.041392685, 9.852479994, 10.23044892, 10.11394335, 11.49692965, 11.24551267, 10.53147892, 10.11394335, 9.77815125, 8.829303773, 7.875061263, 9.698970004, 10.74036269, 10.11394335, 10.11394335, 10.30103, 11.30103, 7.968482949, 5.698970004, 11.25527251, 4.754378919, 10.11394335, 9.982271233, 10.74036269, 9.176091259, 10.84509804, 9.84509804, 9.431363764, 11.0, 11.01703334, 8.408239965, 9.531478917, 6.819543936, 12.25527251, 10.30103, 9.526210004, 10.74036269, 9.447158031, 9.84509804, 10.20411998, 9.602059991, 10.04139269, 11.53147892, 10.11394335, 11.53147892, 10.1903317, 8.969415912, 10.11394335, 9.056904851, 9.826074803, 8.803457116, 10.70379735, 9.0, 8.602059991, 12.03542974, 9.478188452, 12.25527251, 10.60205999, 11.74973632, 9.0, 8.829303773, 10.8142476, 9.487138375, 10.33731947, 9.170261715, 8.469822016, 8.934498451, 9.278753601, 9.397940009, 8.301029996, 8.547774705, 9.431363764, 7.934498451, 7.544068044, 9.77815125, 8.525044807, 9.176091259, 8.51851394, 11.07918125, 9.004751156, 9.033423755, 10.11394335, 7.544068044, 9.959041392, 11.73239376, 11.73239376, 11.73239376, 10.39794001, 9.255272505, 7.306210508, 9.190331698, 10.2278867, 9.278753601, 11.24303805, 11.11394335, 10.2955671, 8.826074803, 10.17609126, 11.2461224, 10.7363965, 8.886490725, 11.73267516, 9.806179974, 10.30103, 9.322219295, 8.875061263, 9.973127854, 8.84509804, 8.550228353, 9.889973638, 9.071882007, 10.30103, 8.621176282, 11.24303805, 10.90308999, 9.973589623, 9.161368002, 8.041392685, 9.544068044, 11.73267516, 10.84509804, 9.602059991, 8.409933123, 9.265525335, 8.888740961, 9.505149978, 9.431363764, 11.42975228, 7.204119983, 11.13672057, 10.30103, 8.356025857, 9.875061263, 10.61384182, 11.24303805, 9.77815125, 9.113943352, 8.62324929, 8.550228353, 8.848272021, 8.848272021, 8.848272021, 7.944482672, 10.0, 11.41497335, 9.875061263, 9.161368002, 8.041392685, 9.477121255, 12.07918125, 11.44715803, 9.73239376, 8.939519253, 8.950851459, 9.487138375, 9.477121255, 9.477844476, 8.800717078, 8.342422681, 8.396199347, 7.176389372, 8.437750563, 11.39045818, 11.72427587, 9.342422681, 8.746634199, 10.04139269, 11.30963017, 8.175213167, 11.13033377, 11.13672057, 10.02938378, 8.041392685, 9.505149978, 11.25042, 9.0, 7.996073654, 6.698970004, 9.113943352, 9.0, 6.540681774, 10.07918125, 10.0, 8.392696953, 7.09439789, 8.318063335, 6.348304863, 6.198657087, 8.408239965, 8.913813852, 9.176091259, 8.643452676, 9.387389826, 9.265525335, 10.11058971, 7.954724791, 9.602059991, 7.230448921, 8.747411808, 9.477121255, 8.62324929, 10.43136376, 10.04139269, 12.00117101, 9.974595046, 8.681241237, 8.369215857, 6.22620177, 8.0, 6.427547127, 7.934498451, 12.19617619, 9.903089987, 10.07918125, 8.568201724, 7.947433722, 8.392696953, 7.653212514, 8.814513952, 9.414973348, 7.968482949, 8.096910013, 8.501059262, 8.800717078, 7.934498451, 8.525044807, 10.11394335, 9.0, 8.683947131, 7.643452676, 8.531478917, 7.995635195, 7.886490725, 9.361727836, 11.24204424, 7.77815125, 8.796574333, 8.074816441, 8.051923916, 7.361727836, 10.04139269, 6.886490725, 5.957733679, 8.352182518, 8.525044807, 7.930949031, 7.900367129, 8.178976947, 7.51851394, 8.100370545, 8.342422681, 10.23044892, 8.574031268, 8.371067862, 8.380211242, 7.445604203, 7.819543936, 9.414973348, 7.213274378, 8.967547976, 8.201397124, 8.201397124, 7.509202522, 7.328379603, 7.477121255, 8.149219113, 8.471291711, 7.566602457, 8.574031268, 8.681241237, 8.320146286, 8.525044807, 8.740362689, 8.392696953, 8.1430148, 8.60883769, 10.04139269, 9.447158031, 10.69897, 7.819543936, 6.73239376, 7.255272505, 8.626340367, 9.919078092, 9.079181246, 9.591064607, 7.451844747, 7.544068044, 7.361727836, 7.934498451, 8.550228353, 7.644438589, 8.918554531, 7.544068044, 8.409933123, 8.531478917, 8.822821645, 11.0, 8.681241237, 8.481442629, 7.531478917, 8.285557309, 9.309296174, 8.596597096, 8.409933123, 7.361727836, 6.84509804, 8.041392685, 8.641176547, 6.397940009, 9.176091259, 5.883093359, 8.51851394, 8.409933123, 7.748188027, 7.418301291, 8.397940009, 9.77815125, 6.857332496, 7.662757832, 9.69019608, 9.462397998, 8.255272505, 7.342422681, 8.531478917, 8.392696953, 6.322219295, 7.544068044, 7.954242509, 8.267171728, 7.544068044, 6.521138084, 7.888516461, 8.100375129, 6.531478917, 8.068185862, 7.505149978, 7.707570176, 8.918554531, 7.755365717, 8.178976947, 4.62324929, 6.204119983, 8.130333768, 7.973127854, 8.285557309, 8.247973266, 8.08278537, 7.544068044, 7.531478917, 7.666520264, 7.72427587, 7.698970004, 7.045322979, 7.414973348, 7.44870632, 7.579783597, 7.278753601, 7.72427587, 7.531478917, 7.51851394, 7.949390007, 7.380211242, 7.64986527, 6.385606274, 8.328379603, 6.62324929, 9.939519253, 7.260071388, 6.397940009, 7.707570176, 5.910871893, 7.648360011, 7.648360011, 7.963787827, 7.919078092, 7.397940009, 8.751279104, 7.572871602, 7.73239376, 6.414973348, 7.707570176, 5.612292084, 7.358999315, 7.322219295, 8.444044796, 7.008600172, 9.501349205, 7.220108088, 7.531478917, 6.738780558, 8.139879086, 8.139879086, 6.952792443, 6.230448921, 8.139879086, 6.176091259, 6.079181246, 7.633468456, 7.748188027, 7.819543936, 7.779596491, 6.230448921, 7.648360011, 7.373403582, 6.728806835, 8.139879086, 6.914342311, 6.699490845, 6.699490845, 7.133538908, 8.434065342, 4.240748878, 7.576066633, 7.26513795, 6.903089987, 6.22874981, 4.525044807, 7.939762789, 8.139879086, 6.374748346, 7.431363764, 10.79239169, 5.848834227, 8.153973315, 6.832361994, 6.698970004, 9.283301229, 8.158362492, 8.139879086, 5.496730435, 3.788451207, 8.100803135, 7.423245874, 8.159855825, 7.679384574, 4.115277591, 7.505149978, 5.379689775, 8.079181246, 8.079181246, 8.158362492, 5.894316063, 5.922256146, 10.30103, 8.974050903, 7.838873461, 8.317227349, 8.840106094, 7.712730493, 8.007347599, 8.315758811, 7.633468456, 5.340237826, 9.230448921, 7.673812427, 6.954242509, 7.77815125, 8.011655011, 9.0, 7.6887773, 6.747723238, 6.698970004, 6.142639708, 5.108632666, 5.081419324, 6.688268505, 5.40654018, 8.145532818, 7.846739051, 5.429209476, 5.469692444, 7.209783176, 6.850155922, 3.010723865, 7.225778893, 7.065066174, 7.137663958, 5.412964272, 8.0, 7.903089987, 7.256009986, 1.477121255, 6.266325791, 6.176091259, 5.003809379, 11.47712125, 10.32221929, 6.579783597, 6.204119983, 4.958358393, 3.75671216, 6.612359948, 4.957008343, 6.615343193, 7.07570255, 4.079181246, 6.593873961, 6.839288198, 5.049218023, 4.824646415, 2.440909082, 4.77815125, 4.021354713, 4.113943352, 3.903089987, 4.505353506, 3.809694359, 4.133762385, 5.636287252, 8.0, 5.315781481, 4.858843862, 5.584331224, 3.647969458, 3.27600199, 3.749736316, 2.871572936, 6.219679861, 3.411282913, 2.773786445, 4.397940009, 3.923554858, 3.186391216, 4.862965759, 3.214843848, 1.556302501, 4.439016728, 3.989449818, 3.482873584, 3.411282913, 2.90579588, 4.0, 2.911690159, 4.270189543, 4.270189543, 2.857332496, 2.635483747, 3.969182859, 2.51054501, 3.995635195, 3.612359948, 6.057124229, 4.334453751, 2.552668216, 3.389166084, 3.314077992, 1.230448921, 3.0, 3.419129308, 1.204119983, 3.0, 2.352182518, 1.602059991, 1.602059991]))
    dom = {'Language': K['danger'], 'Vision': K['memory'], 'Games': K['data']}
    d = np.array(['Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Vision', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Other', 'Language', 'Language', 'Language', 'Language', 'Language', 'Other', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Other', 'Language', 'Language', 'Language', 'Language', 'Language', 'Other', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Vision', 'Other', 'Other', 'Language', 'Language', 'Language', 'Language', 'Other', 'Other', 'Other', 'Language', 'Other', 'Language', 'Other', 'Language', 'Language', 'Language', 'Vision', 'Language', 'Other', 'Other', 'Other', 'Language', 'Other', 'Language', 'Language', 'Vision', 'Other', 'Other', 'Other', 'Language', 'Language', 'Vision', 'Vision', 'Vision', 'Other', 'Language', 'Other', 'Other', 'Other', 'Language', 'Language', 'Language', 'Other', 'Language', 'Language', 'Language', 'Other', 'Other', 'Language', 'Language', 'Language', 'Other', 'Other', 'Other', 'Other', 'Language', 'Language', 'Other', 'Language', 'Language', 'Other', 'Language', 'Other', 'Language', 'Other', 'Language', 'Language', 'Other', 'Language', 'Language', 'Other', 'Language', 'Language', 'Other', 'Other', 'Language', 'Other', 'Language', 'Language', 'Other', 'Other', 'Language', 'Vision', 'Other', 'Other', 'Language', 'Other', 'Language', 'Other', 'Language', 'Language', 'Other', 'Language', 'Language', 'Language', 'Vision', 'Language', 'Vision', 'Other', 'Other', 'Language', 'Language', 'Language', 'Language', 'Other', 'Other', 'Language', 'Other', 'Other', 'Other', 'Language', 'Language', 'Language', 'Other', 'Other', 'Language', 'Other', 'Language', 'Other', 'Other', 'Other', 'Language', 'Language', 'Language', 'Other', 'Language', 'Other', 'Other', 'Other', 'Other', 'Language', 'Other', 'Other', 'Language', 'Language', 'Language', 'Other', 'Language', 'Other', 'Other', 'Language', 'Language', 'Other', 'Other', 'Vision', 'Language', 'Vision', 'Language', 'Other', 'Vision', 'Language', 'Other', 'Other', 'Language', 'Other', 'Other', 'Other', 'Language', 'Vision', 'Vision', 'Vision', 'Other', 'Other', 'Other', 'Other', 'Games', 'Other', 'Language', 'Other', 'Other', 'Language', 'Language', 'Other', 'Language', 'Language', 'Vision', 'Vision', 'Language', 'Language', 'Other', 'Language', 'Language', 'Language', 'Other', 'Other', 'Other', 'Other', 'Language', 'Other', 'Language', 'Language', 'Language', 'Other', 'Other', 'Language', 'Language', 'Language', 'Language', 'Other', 'Other', 'Vision', 'Language', 'Other', 'Other', 'Language', 'Other', 'Other', 'Language', 'Language', 'Language', 'Other', 'Language', 'Other', 'Other', 'Other', 'Language', 'Language', 'Other', 'Language', 'Vision', 'Language', 'Language', 'Language', 'Language', 'Other', 'Language', 'Language', 'Other', 'Language', 'Language', 'Language', 'Language', 'Language', 'Other', 'Other', 'Vision', 'Other', 'Language', 'Vision', 'Other', 'Language', 'Language', 'Other', 'Language', 'Language', 'Language', 'Language', 'Language', 'Other', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Language', 'Language', 'Other', 'Language', 'Language', 'Language', 'Language', 'Vision', 'Language', 'Language', 'Language', 'Other', 'Language', 'Language', 'Other', 'Other', 'Language', 'Other', 'Vision', 'Other', 'Vision', 'Other', 'Games', 'Language', 'Language', 'Language', 'Other', 'Vision', 'Other', 'Other', 'Vision', 'Other', 'Language', 'Language', 'Vision', 'Vision', 'Language', 'Vision', 'Other', 'Other', 'Other', 'Other', 'Other', 'Language', 'Language', 'Other', 'Language', 'Vision', 'Language', 'Games', 'Other', 'Other', 'Vision', 'Language', 'Other', 'Other', 'Other', 'Other', 'Language', 'Language', 'Other', 'Language', 'Other', 'Language', 'Other', 'Vision', 'Vision', 'Language', 'Language', 'Other', 'Language', 'Other', 'Language', 'Language', 'Vision', 'Language', 'Language', 'Vision', 'Language', 'Other', 'Other', 'Language', 'Language', 'Vision', 'Games', 'Other', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Vision', 'Language', 'Language', 'Other', 'Language', 'Language', 'Other', 'Vision', 'Games', 'Games', 'Language', 'Language', 'Other', 'Language', 'Other', 'Games', 'Vision', 'Vision', 'Language', 'Language', 'Language', 'Language', 'Games', 'Language', 'Language', 'Language', 'Language', 'Language', 'Vision', 'Language', 'Language', 'Language', 'Language', 'Language', 'Other', 'Language', 'Language', 'Vision', 'Language', 'Vision', 'Vision', 'Language', 'Language', 'Language', 'Language', 'Other', 'Vision', 'Vision', 'Language', 'Vision', 'Other', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Games', 'Language', 'Games', 'Language', 'Language', 'Language', 'Other', 'Other', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Other', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Language', 'Other', 'Vision', 'Games', 'Vision', 'Language', 'Language', 'Language', 'Vision', 'Vision', 'Language', 'Vision', 'Games', 'Language', 'Language', 'Other', 'Language', 'Vision', 'Language', 'Language', 'Games', 'Language', 'Language', 'Language', 'Vision', 'Vision', 'Language', 'Language', 'Vision', 'Vision', 'Language', 'Vision', 'Language', 'Vision', 'Vision', 'Language', 'Vision', 'Language', 'Vision', 'Games', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Language', 'Language', 'Language', 'Other', 'Vision', 'Language', 'Language', 'Vision', 'Other', 'Other', 'Vision', 'Vision', 'Vision', 'Vision', 'Language', 'Games', 'Vision', 'Games', 'Vision', 'Vision', 'Vision', 'Language', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Games', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Other', 'Language', 'Games', 'Games', 'Vision', 'Vision', 'Vision', 'Vision', 'Language', 'Vision', 'Other', 'Vision', 'Language', 'Language', 'Vision', 'Vision', 'Vision', 'Vision', 'Other', 'Other', 'Vision', 'Vision', 'Language', 'Language', 'Games', 'Language', 'Language', 'Vision', 'Vision', 'Games', 'Language', 'Language', 'Vision', 'Language', 'Language', 'Vision', 'Other', 'Other', 'Other', 'Vision', 'Vision', 'Other', 'Language', 'Vision', 'Language', 'Vision', 'Other', 'Vision', 'Language', 'Vision', 'Vision', 'Vision', 'Vision', 'Language', 'Vision', 'Language', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Other', 'Vision', 'Other', 'Other', 'Language', 'Language', 'Vision', 'Language', 'Language', 'Vision', 'Vision', 'Vision', 'Other', 'Vision', 'Vision', 'Language', 'Language', 'Vision', 'Vision', 'Language', 'Language', 'Language', 'Language', 'Vision', 'Language', 'Other', 'Games', 'Vision', 'Vision', 'Vision', 'Vision', 'Vision', 'Language', 'Games', 'Vision', 'Vision', 'Other', 'Vision', 'Vision', 'Language', 'Vision', 'Other', 'Games', 'Other', 'Other', 'Vision', 'Vision', 'Other', 'Other', 'Vision', 'Other', 'Vision', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Vision', 'Other', 'Vision', 'Other', 'Other', 'Vision', 'Vision', 'Vision', 'Games', 'Other', 'Other', 'Other', 'Other'])
    fig, ax = plt.subplots(figsize=(5.3, 2.0))
    for k, col in list(dom.items()) + [('Other', '#BBBBBB')]:
        m = d == k
        ax.scatter(years[m], 10 ** log_params[m], s=12, color=col, alpha=0.75, edgecolors='none', zorder=3 if k != 'Other' else 2)
    for lo, hi, sl, ic, txt, xm, lift in ((1950, 2010, 0.08366932442, -162.1122855, '43', 1975, 60), (2010, 2026.8, 0.344835656, -687.8321121, '10', 1998, 1)):
        xx = np.linspace(lo, hi, 50)
        ax.plot(xx, 10 ** (sl * xx + ic), '--', color=K['text'], lw=1.6, zorder=4)
    yr = lambda x, sl, ic: 10 ** (sl * x + ic)
    ax.text(1972, yr(1972, 0.08366932442, -162.1122855) * 150, '43-month doubling', fontsize=11, fontweight='bold', ha='center', color=K['text'], zorder=6)
    ax.annotate('10-month doubling', (2014, yr(2014, 0.344835656, -687.8321121)), xytext=(1978, 300000000000.0), fontsize=11, fontweight='bold', ha='center', va='center', color=K['text'], zorder=6, arrowprops=dict(arrowstyle='-', color=K['text'], lw=1))
    offs = {'Perceptron Mark I': (4, -16, 'left'), 'LeNet-5': (-6, 6, 'right'), 'AlexNet': (-6, 6, 'right'), 'GPT-3 175B (davinci)': (-30, 2, 'right'), 'PaLM (540B)': (-8, 4, 'right')}
    for n, (dx, dy, ha) in offs.items():
        for x, r in zip(years, rows):
            if r['model'] == n:
                ax.annotate(n.replace(' (davinci)', '').replace(' (540B)', ''), (x, r['parameters']), xytext=(dx, dy), textcoords='offset points', ha=ha, fontsize=10.5, color=K['text'], zorder=7, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=0.5))
                break
    mn = 109386
    ax.axhline(mn, color=K['compute'], lw=1.4, ls=':', zorder=1)
    ax.text(2027, mn, ' running\n network', fontsize=10.5, color=K['compute'], fontweight='bold', va='center', ha='left')
    for i, (k, col) in enumerate(list(dom.items()) + [('Other', '#999999')]):
        ax.text(0.02, 0.95 - i * 0.1, k, transform=ax.transAxes, fontsize=11, fontweight='bold', color=col, va='top')
    ax.set_yscale('log')
    ax.set_xlim(1948, 2027)
    ax.set_yticks([10.0, 10000.0, 10000000.0, 10000000000.0, 10000000000000.0])
    ax.minorticks_off()
    ax.set_xlabel('Publication date')
    ax.set_ylabel('Parameters (log scale)')
    ax.grid(axis='y', color=K['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'param_growth.svg')

def depth_width():
    h1, mnist_macs = (128, 109184)
    fig, (aw, ad) = plt.subplots(1, 2, figsize=(5.6, 1.9), gridspec_kw={'wspace': 0.75})
    w = [64, 128, 256, 512, 1024, 2048, 4096]
    mw = [54912, 118016, 268800, 668672, 1861632, 5820416, 20029440]
    guide = [mw[-1] * (x / w[-1]) ** 2 for x in w]
    aw.plot(w, guide, color=K['neutral'], lw=1.4, ls='--')
    aw.plot(w, mw, color=K['memory'], lw=2.5, marker='o', ms=5)
    aw.plot([h1], [mnist_macs], marker='D', ms=9, color=K['danger'], ls='none', zorder=5)
    aw.text(w[0] * 0.95, mw[3] * 4, 'two hidden\nlayers', color=K['memory'], fontsize=10.5, fontweight='bold', ha='left', va='bottom')
    aw.text(w[4] * 1.25, guide[4] / 5, 'slope two', color=K['neutral'], fontsize=10.5, ha='left', va='top')
    aw.annotate(f'MNIST\n{ARCH}', (h1, mnist_macs), xytext=(300, 13000.0), textcoords='data', fontsize=10.5, color=K['danger'], fontweight='bold', va='center', ha='left', arrowprops=dict(arrowstyle='-', color=K['danger'], lw=1))
    aw.set_xscale('log', base=2)
    aw.set_yscale('log')
    aw.set_xticks([64, 256, 1024, 4096])
    aw.set_xticklabels(['64', '256', '1,024', '4,096'])
    aw.minorticks_off()
    aw.set_xlabel('Hidden width')
    aw.set_ylabel('MACs per example')
    aw.set_title('Width: quadratic', fontsize=12, fontweight='bold')
    dp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    md = [0.406528, 0.668672, 0.930816, 1.19296, 1.455104, 1.717248, 1.979392, 2.241536, 2.50368, 2.765824, 3.027968, 3.290112, 3.552256, 3.8144, 4.076544, 4.338688, 4.600832, 4.862976, 5.12512, 5.387264, 5.649408, 5.911552, 6.173696, 6.43584, 6.697984, 6.960128, 7.222272, 7.484416, 7.74656, 8.008704, 8.270848, 8.532992]
    act = [1.306, 1.818, 2.33, 2.842, 3.354, 3.866, 4.378, 4.89, 5.402, 5.914, 6.426, 6.938, 7.45, 7.962, 8.474, 8.986, 9.498, 10.01, 10.522, 11.034, 11.546, 12.058, 12.57, 13.082, 13.594, 14.106, 14.618, 15.13, 15.642, 16.154, 16.666, 17.178]
    ad.plot(dp, md, color=K['memory'], lw=2.5)
    ad.set_xlim(0, dp[-1] + 1)
    ad.set_ylim(0, None)
    ad.set_xlabel('Hidden layers (width 512)')
    ad.set_ylabel('MACs (millions)', color=K['memory'])
    ad.set_title('Depth: linear', fontsize=12, fontweight='bold')
    aa = ad.twinx()
    aa.plot(dp, act, color=K['compute'], lw=2.5, ls='--')
    aa.set_ylim(0, None)
    aa.spines['right'].set_visible(True)
    aa.set_ylabel('Saved activations\n(thousand values)', color=K['compute'])
    ad.text(2, md[-1] * 0.95, 'MACs', color=K['memory'], fontsize=10.5, fontweight='bold', va='top')
    ad.text(30, md[-1] * 0.2, 'saved\nactivations', color=K['compute'], fontsize=10.5, fontweight='bold', ha='right', va='bottom')
    sp.save(fig, out / 'depth_width.svg')

def mnist_roofline():
    tf = lambda q: q.to('Tflop/s').magnitude
    ai = np.logspace(-1.8, 3, 400)
    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    machines = [('Desktop CPU', 'host', '#8B5A2B', '--', (12, -8)), ('Data-center accelerator', 'accelerator', K['memory'], '-', (10, -10))]
    i1, i32 = (0.4946727687, 11.89932147)
    for name, key, col, ls, off in machines:
        peak = tf({'host': Q_(1000000000000.0, 'flop / s'), 'accelerator': Q_(67000000000000.0, 'flop / s')}[key])
        bw = {'host': 0.05, 'accelerator': 3.35}[key]
        ax.plot(ai, np.minimum(peak, bw * ai), color=col, lw=2.5, ls=ls)
        ax.text(900, peak * 1.4, name, color=col, fontsize=11, fontweight='bold', ha='right', va='bottom')
        for lab, x, y in (('batch 1', i1, tf({'host': Q_(24733638440.0, 'flop / s'), 'accelerator': Q_(1657153775000.0, 'flop / s')}[key])), ('batch 32', i32, tf({'host': Q_(594966073300.0, 'flop / s'), 'accelerator': Q_(39862726910000.0, 'flop / s')}[key]))):
            ax.plot(x, y, 'o', color=K['danger'], ms=8, zorder=4)
            rate_lbl = f'{y * 1000:,.0f} GFLOP/s' if y < 1 else f'{y:.1f} TFLOP/s' if y < 10 else f'{y:,.0f} TFLOP/s'
            up = key == 'accelerator'
            left = up and lab == 'batch 1'
            ax.annotate(f'{lab}:\n{rate_lbl}' if lab == 'batch 1' else f'{lab}: {rate_lbl}', (x, y), xytext=(-10, 6) if left else (10, -4) if up else off, textcoords='offset points', ha='right' if left else 'left', va='bottom' if left else 'top', fontsize=12, color=K['text'])
    ax.text(max((r.to('flop/byte').magnitude for r in {'host': Q_(20.0, 'flop / B'), 'accelerator': Q_(20.0, 'flop / B')}.values())) * 2.5, 0.0004, 'ridge ≈ 20 FLOP/byte\non both machines', fontsize=10.5, color=K['text'], ha='left', va='bottom')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(ai[0], ai[-1])
    ax.set_ylim(0.0002, 400)
    ax.minorticks_off()
    ax.set_xlabel('Arithmetic intensity (FLOP/byte)')
    ax.set_ylabel('FP32 rate (TFLOP/s)')
    ax.grid(color=K['grid'], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'mnist_roofline.svg')

def model_bytes():
    by = lambda q: q.to('byte').magnitude
    models = [('MNIST', by(Q_(437544, 'B'))), ('ResNet-50', by(Q_(102228128.0, 'B'))), ('GPT-2', by(Q_(6000000000.0, 'B')))]
    tiers = [('MCU SRAM', by(Q_(520, 'KiB'))), ('H100 L2', by(Q_(50, 'MiB'))), ('H100 HBM', by(Q_(80, 'GiB')))]
    y = np.arange(len(models))
    wts = np.array([b for _, b in models])
    fig, ax = plt.subplots(figsize=(3.8, 1.45))
    ax.barh(y + 0.19, wts * 4.0, height=0.36, color=K['compute_l'], edgecolor=K['compute'])
    ax.barh(y - 0.19, wts, height=0.36, color=K['memory_l'], edgecolor=K['memory'])
    ax.set_xscale('log')
    ax.set_xlim(10000.0, 10000000000000.0)
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _ in models], fontsize=13)
    ax.invert_yaxis()
    for (nm, cap), ha in zip(tiers, ('right', 'left', 'left')):
        ax.axvline(cap, color=K['danger'], lw=1.5, ls='--')
        ax.text(cap, 1.03, nm, transform=ax.get_xaxis_transform(), fontsize=13, color=K['danger'], ha=ha, va='bottom', fontweight='bold')
    for j, (txt, fc, ec) in enumerate((('weights', K['memory_l'], K['memory']), ('training state (×4)', K['compute_l'], K['compute']))):
        x0 = 0.0 + j * 0.38
        ax.add_patch(plt.Rectangle((x0, 1.27), 0.05, 0.1, transform=ax.transAxes, facecolor=fc, edgecolor=ec, clip_on=False))
        ax.text(x0 + 0.07, 1.32, txt, transform=ax.transAxes, fontsize=13, va='center', color=K['text'])
    ax.set_xticks([10000.0, 1000000.0, 100000000.0, 10000000000.0, 1000000000000.0])
    ax.set_xticklabels(['10 KB', '1 MB', '100 MB', '10 GB', '1 TB'], fontsize=12)
    ax.minorticks_off()
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    sp.save(fig, out / 'model_bytes.svg')
param_growth()
depth_width()
mnist_roofline()
model_bytes()

def mnist_digit():
    d = json.loads((out / 'data' / 'mnist_test0.json').read_text())
    side = 28
    return np.array(d['pixels'], dtype=float).reshape(side, side) / (256 - 1)

def mnist_net(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    cols = list(zip((1.9, 3.25, 4.45, 5.5), [784, 128, 64, 10], (7, 6, 5, 4)))
    ys = {}
    for x, n, k in cols:
        yy = np.linspace(0.15, 1.45, k) if k > 4 else np.linspace(0.35, 1.25, k)
        ys[x] = yy
    for (x0, _, _), (x1, _, _) in zip(cols[:-1], cols[1:]):
        for a in ys[x0]:
            for b in ys[x1]:
                ax.plot([x0, x1], [a, b], color='#BBBBBB', lw=0.6, zorder=1)
    for x, n, k in cols:
        ax.scatter([x] * k, ys[x], s=70, color='white', edgecolors=K['memory'], lw=1.5, zorder=3)
        ax.text(x, 1.62, f'{n:,}', ha='center', va='bottom', fontsize=14, fontweight='bold', color=K['text'])
    ax.imshow(mnist_digit(), cmap='Greys', extent=(0.05, 1.0, 0.3, 1.25), zorder=2, interpolation='nearest')
    ax.annotate('', (1.75, 0.8), xytext=(1.1, 0.8), arrowprops=dict(arrowstyle='-|>', color=K['text'], lw=1.4))
    ax.text(0.52, 1.62, '28×28 MNIST digit', ha='center', va='bottom', fontsize=12, color=K['neutral'])
    if step >= 2:
        for (x0, n0, _), (x1, n1, _) in zip(cols[:-1], cols[1:]):
            ax.text((x0 + x1) / 2, -0.08, f'{n0 * n1:,}', ha='center', va='top', fontsize=14, fontweight='bold', color=K['compute'])
    ax.set_xlim(0, 5.8)
    ax.set_ylim(-0.42, 1.9)
    ax.set_aspect('auto')
    ax.axis('off')
    sp.save(fig, path)
for k in (1, 2):
    mnist_net(out / f'mnist_net_step{k}.svg', k)

def xor_plot(path, step):
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    if step == 2:
        xx = np.array([-0.3, 1.3])
        ax.fill_between(xx, 0.5 - xx, 1.5 - xx, color=K['compute_l'], zorder=0)
        ax.plot(xx, 0.5 - xx, color=K['compute'], lw=1.5, ls='--')
        ax.plot(xx, 1.5 - xx, color=K['compute'], lw=1.5, ls='--')
        ax.text(0.5, 0.5, 'y > 0.5', rotation=-45, ha='center', va='center', fontsize=15, fontweight='bold', color=K['compute'])
    pts = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
    for (a, b), c in pts:
        ax.scatter(a, b, s=260, color=K['compute'] if c else 'white', edgecolors=K['compute'] if c else K['memory'], lw=2.5, zorder=3)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16, rotation=0, labelpad=10)
    ax.tick_params(labelsize=15)
    ax.set_aspect('equal')
    sp.save(fig, path)
for k in (1, 2):
    xor_plot(out / f'xor_step{k}.svg', k)

def cross_entropy():
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    p = np.linspace(0.02, 1.0, 400)
    ax.plot(p, -np.log(p), color=K['memory'], lw=3)
    for pv, txt, dx, dy, ha in ((0.1, 'p = 0.1: 2.30\n(uniform guess, or a\nconfident mistake)', 0.05, 0.25, 'left'), (0.8, 'p = 0.8: 0.22', 0.0, 0.75, 'center'), (0.9, 'p = 0.9: 0.11', 0.06, 0.45, 'left')):
        lv = -np.log(pv)
        ax.plot(pv, lv, 'o', color=K['danger'], ms=9, zorder=4)
        ax.annotate(txt, (pv, lv), xytext=(pv + dx, lv + dy), fontsize=13, color=K['text'], ha=ha, va='bottom', arrowprops=dict(arrowstyle='-', color=K['neutral'], lw=1))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel('Probability the network gives the correct digit')
    ax.set_ylabel('Loss, −log p')
    ax.grid(color=K['grid'], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'cross_entropy.svg')
cross_entropy()

def risk_coverage():
    fig, ax = plt.subplots(figsize=(3.7, 2.3))
    c = np.linspace(0.55, 1.0, 200)
    acc, err = (87.9, 1.0)
    risk = err * np.exp(9.0 * (c - acc / 100))
    ax.plot(c * 100, risk, color=K['memory'], lw=3)
    ax.plot(acc, err, 'o', color=K['danger'], ms=10, zorder=4)
    ax.annotate('1989 ZIP-code\nreader: 1% error,\n12.1% rejected', (acc, err), xytext=(57, 1.6), fontsize=12, fontweight='bold', color=K['danger'], arrowprops=dict(arrowstyle='-', color=K['danger'], lw=1))
    ax.axvline(acc, color=K['neutral'], ls=':', lw=1.5)
    ax.text(acc + 0.7, 3.5, 'threshold', fontsize=12, color=K['neutral'], ha='left', va='top')
    ax.set_xlim(55, 101)
    ax.set_ylim(0, 3.6)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['0%', '1%', '2%', '3%'])
    ax.set_xlabel('Digits accepted by the machine (%)')
    ax.set_ylabel('Error among\naccepted digits')
    ax.text(1.0, 1.02, 'curve schematic; point measured', transform=ax.transAxes, ha='right', va='bottom', fontsize=11, color=K['neutral'])
    ax.grid(axis='y', color=K['grid'], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'risk_coverage.svg')
risk_coverage()

def load_plot_box(png, box):
    """Crop a PNG to its axes box (left, top, right, bottom pixels)."""
    from PIL import Image
    return np.asarray(Image.open(out / png).convert('RGB').crop(box))

def alexnet_relu():
    img = load_plot_box('alexnet-fig1.png', (215, 36, 1413, 987))
    fig, ax = plt.subplots(figsize=(2.25, 1.45))
    ax.imshow(img, extent=(0, 40, 0, 0.75), aspect='auto', interpolation='lanczos')
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 0.75)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_yticklabels(['0', '25%', '50%', '75%'])
    ax.tick_params(labelsize=11)
    ax.set_xlabel('Epochs (CIFAR-10)', fontsize=11)
    ax.set_ylabel('Training error', fontsize=11)
    ax.text(7.0, 0.2, 'ReLU', fontsize=12, fontweight='bold', color=K['compute'], va='top')
    ax.text(24, 0.36, 'tanh', fontsize=12, fontweight='bold', color=K['memory'], va='bottom')
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    sp.save(fig, out / 'alexnet_relu.svg')

def memory_timeline():
    x0, x1, y_top, y_bot = (201, 1150, 116, 854)
    px_ms = (1294 - 256) / 250
    px_gb = (821 - 254) / 0.4
    ext = ((x0 - 256) / px_ms, (x1 - 256) / px_ms, (821 - y_bot) / px_gb, (821 - y_top) / px_gb)
    img = load_plot_box('pytorch-memory-timeline.png', (x0, y_top, x1, y_bot))
    fig, ax = plt.subplots(figsize=(4.3, 2.45))
    ax.imshow(img, extent=ext, aspect='auto', interpolation='lanczos')
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.tick_params(labelsize=12)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Memory (GB)')
    for y_lab, (x_pt, y_pt), txt, col in ((0.047, (205, 0.047), 'weights', '#3B7D3B'), (0.145, (205, 0.145), 'optimizer state (momentum)', '#A87C12'), (0.235, (196, 0.225), 'saved activations', '#D23C3C'), (0.325, (210, 0.3), 'gradients', '#2B2BA8'), (0.425, (207, 0.4), 'temporary buffers', K['neutral'])):
        ax.annotate(txt, (x_pt, y_pt), xytext=(226, y_lab), annotation_clip=False, fontsize=12, fontweight='bold', color=col, ha='left', va='center', arrowprops=dict(arrowstyle='-', color=col, lw=1.1))
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    sp.save(fig, out / 'memory_timeline.svg')

def overfit_curves():
    d = json.loads((out / 'data' / 'overfit.json').read_text())
    small, full = (d['runs']['1000'], d['runs']['60000'])
    ep = np.arange(1, len(small['train_acc']) + 1)
    fig, ax = plt.subplots(figsize=(3.2, 2.35))
    ax.plot(ep, np.array(small['train_acc']) * 100, color=K['compute'], lw=2.5)
    ax.plot(ep, np.array(small['test_acc']) * 100, color=K['compute'], lw=2.5, ls='--')
    ax.plot(ep, np.array(full['test_acc']) * 100, color=K['memory'], lw=2.5, ls='--')
    ax.text(40, 100.6, 'training, 1,000 images: 100%', fontsize=11, color=K['compute'], fontweight='bold', ha='right', va='bottom')
    ax.text(40, 87.7 - 0.8, 'held-out: 87.7%', fontsize=11, color=K['compute'], fontweight='bold', ha='right', va='top')
    ax.text(40, 97.8 - 0.8, 'held-out, trained on\n60,000 images: 97.8%', fontsize=11, color=K['memory'], fontweight='bold', ha='right', va='top', linespacing=1.0)
    ax.set_xlim(0, 41)
    ax.set_ylim(78, 103)
    ax.set_yticks([80, 85, 90, 95, 100])
    ax.set_yticklabels(['80%', '85%', '90%', '95%', '100%'])
    ax.tick_params(labelsize=11)
    ax.set_xlabel('Epochs', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.grid(axis='y', color=K['grid'], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'overfit_curves.svg')
alexnet_relu()
memory_timeline()
overfit_curves()
print('figures written')
