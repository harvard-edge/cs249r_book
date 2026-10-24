#!/usr/bin/env python3
"""Figures for Lecture 11 (Hardware Acceleration).

Every number comes from MLSysIM through calcs.py: val(key) is the string a
slide prints and num(key) its value; bar lengths and curves read the same
registry entries and Scenarios.* the chapter reads. Layout values (axis
limits, ticks, label offsets, the working-set sizes where the measured
latency plateaus begin, read from the Chips and Cheese chart's axis) are
typed here. Lecture 11b's figures live in that deck's own folder.

Run from this folder: python3 make_figs.py
"""
_VALS = {'voice_minutes_words': 'three', 'tpu_mhz': '700', 'tpu_dim': '256', 'tpu_macs': '65,536', 'tpu_buffer_mb': '24', 'tpu_thr_lo': '15', 'tpu_thr_hi': '30', 'tpu_ppw_lo': '30', 'tpu_ppw_hi': '80', 'tpu_workloads_words': 'six', 'lat_fp32_ms': '0.38', 'lat_int8_ms': '0.30', 'lat_int8_ms_short': '0.3', 'int8_b1_gain': '1.3', 'int8_big_gain': '5.9', 'h100_fp16': '989', 'cpu_lo': '1', 'cpu_hi': '2', 'cpu_gap_lo': '490', 'cpu_gap_hi': '990', 'h100_transistors_b': '80', 'die_buffers': '37', 'die_compute': '30', 'die_io': '10', 'die_control': '2', 'die_ub': '29', 'die_mxu': '24', 'h100_cuda_cores': '16,896', 'h100_tensor_cores': '528', 'e_int8_add': '0.03', 'e_fp32_add': '0.9', 'e_fp32_mul': '3.7', 'e_sram8k': '5', 'e_dram': '640', 'dram_per_mul': '173', 'dram_per_sram': '128', 'dram_per_fma': '139', 'e_fp32_mul_7nm': '1.31', 'mul_gain_7nm': '2.8', 'e_ddr64_45': '1,300', 'e_ddr64_7': '1,300', 'p100_fp16': '21.2', 'v100_fp16': '125', 'tc_gain': '6', 'mma_macs': '2,048', 'mma_per_thread': '64', 'resnet_classes': '1,000', 'tpu4_dim': '128', 'tpu4_macs': '16,384', 'sys_dim': '128', 'sys_edge_loads': '2', 'sys_edge_loads_words': 'two', 'naive_accesses': '4', 'naive_accesses_words': 'four', 'mac_pj': '1', 'naive_pj': '2,561', 'sys_access': '0.016', 'sys_pj': '11', 'sys_ratio': '232.8', 'sys_ratio_int': '233', 'wall_params': '3,900', 'wall_bw': '33', 'gpt2_gb': '3', 'e_reg': '0.1', 'smem_kb': '228', 'smem_ns': '17', 'l2_mb': '50', 'l2_ns': '150', 'hbm_gb': '80', 'h100_bw': '3.35', 'hbm_ns': '273', 'e_dram_byte': '160', 'clock_ghz': '2', 'cycle_ps': '500', 'hbm_cycles': '546', 'cnc_l1': '22.4', 'cnc_l2': '159.7', 'cnc_l2far': '294', 'cnc_hbm': '377.9', 'cnc_hbm_l1': '17', 'pcie_sms': '114', 'pcie_bw': '2', 'need_tbs': '989', 'need_short': '295', 'hbm_gbs': '3,350', 'nvlink_gbs': '900', 'pcie_gbs': '64', 'net_gbs': '50', 'hbm_pcie': '52', 'a100_fp16': '312', 'a100_bw': '2.04', 'a100_ridge': '153', 'h100_fp32': '67', 'h100_int8': '1,979', 'ridge32': '20', 'ridge16': '295.2', 'ridge16_int': '295', 'ridge8': '590.7', 'ridge8_int': '591', 'int8_over_fp16': '2', 'fp16_over_fp32': '15', 'k_ai': '200', 'k_compute_ratio': '3.2', 'k_bw_ratio': '1.6', 'k_a100': '312', 'k_h100': '670', 'k_gain': '2.1', 'k_gain_2': '2.15', 'ridge_v100': '139', 'ridge_a100': '153', 'ridge_h100': '295', 'ridge_b200': '281', 'gen_first_year': '2017', 'gen_last_year': '2024', 'gen_compute': '18', 'gen_bw': '8.9', 'ridge_years_words': 'five', 'env_devices': '52', 'env_compute': '55', 'env_bw': '13', 'env_best_bw': '8', 'env_since_year': '2025', 'qkv_batch': '8', 'conv_batch': '32', 'qkv_gflop': '125.8', 'qkv_mb': '120.2', 'qkv_ai': '1,046.7', 'conv_gflop': '7.4', 'conv_mb': '13.1', 'conv_ai': '563.1', 'head_mflop': '131.1', 'head_mb': '4.3', 'head_ai': '30.5', 'ln_mflop': '104.9', 'ln_mb': '52.4', 'ln_ai': '2.0', 'ln_ops_words': 'eight', 'dec_gflop': '3', 'dec_ai': '1', 'head_tflops': '102', 'head_pct': '10.3', 'ln_tflops': '6.7', 'ln_pct': '0.7', 'dec_tflops': '3.4', 'dec_pct': '0.3', 'top_pct': '100', 'softmax_ai': '0.75', 'nsight_dram_pct': '0.34', 'dense_width': '2048', 'batch_mid': '32', 'batch_big': '512', 'ai_b1': '1', 'ai_b32': '31', 'ai_b512': '341', 'ai_b512_1': '341.3', 'crossing': '415', 'resnet_gflop': '8.2', 'floor16_us': '8', 'floor8_us': '4', 's3_fp16': '563', 's3_int8': '1,126', 's3_mixed': '569', 's2_fp16': '287', 's2_int8': '574', 'p2_fp16': '51', 'p2_int8': '102', 'p_resnet': '0.95', 'p_resnet_pct': '95', 'serial_resnet': '0.05', 'serial_resnet_pct': '5', 'p_gpt': '0.80', 'p_gpt_pct': '80', 'cpu_tops': '8', 'gain': '247', 'amdahl_rn': '18.6', 'amdahl_gpt': '4.9', 'bw_gain': '67', 'amdahl_bw': '4.7', 'amdahl_ceiling': '5', 'lh_resnet16': '51.1', 'lh_resnet8': '25.6', 'lh_mobilenet16': '7.0', 'lh_mobilenet8': '3.5', 'lh_mobilenet_gflop': '0.6', 'lh_gpt8': '1.5', 'lh_dlrm_gb': '100', 'lh_kws16': '400', 'lh_kws8': '200', 'lh_kws_mflop': '20', 'a100_t4_peak': '4.8', 'a100_t4_bw': '6.4', 'ds_tf32': '989', 'ds_fp16': '1,979', 'ds_int8': '3,958', 'mapping_gain': '75'}
_NUMS = {'tpu_mhz': 700.0, 'tpu_dim': 256.0, 'tpu_macs': 65536.0, 'tpu_buffer_mb': 24.0, 'tpu_thr_lo': 15.0, 'tpu_thr_hi': 30.0, 'tpu_ppw_lo': 30.0, 'tpu_ppw_hi': 80.0, 'lat_fp32_ms': 0.38, 'lat_int8_ms': 0.3, 'lat_int8_ms_short': 0.3, 'int8_b1_gain': 1.3, 'int8_big_gain': 5.9, 'h100_fp16': 989.0, 'cpu_lo': 1.0, 'cpu_hi': 2.0, 'cpu_gap_lo': 490.0, 'cpu_gap_hi': 990.0, 'h100_transistors_b': 80.0, 'die_buffers': 37.0, 'die_compute': 30.0, 'die_io': 10.0, 'die_control': 2.0, 'die_ub': 29.0, 'die_mxu': 24.0, 'h100_cuda_cores': 16896.0, 'h100_tensor_cores': 528.0, 'e_int8_add': 0.03, 'e_fp32_add': 0.9, 'e_fp32_mul': 3.7, 'e_sram8k': 5.0, 'e_dram': 640.0, 'dram_per_mul': 173.0, 'dram_per_sram': 128.0, 'dram_per_fma': 139.0, 'e_fp32_mul_7nm': 1.31, 'mul_gain_7nm': 2.8, 'e_ddr64_45': 1300.0, 'e_ddr64_7': 1300.0, 'p100_fp16': 21.2, 'v100_fp16': 125.0, 'tc_gain': 6.0, 'mma_macs': 2048.0, 'mma_per_thread': 64.0, 'resnet_classes': 1000.0, 'tpu4_dim': 128.0, 'tpu4_macs': 16384.0, 'sys_dim': 128.0, 'sys_edge_loads': 2.0, 'naive_accesses': 4.0, 'mac_pj': 1.0, 'naive_pj': 2561.0, 'sys_access': 0.016, 'sys_pj': 11.0, 'sys_ratio': 232.8, 'sys_ratio_int': 233.0, 'wall_params': 3900.0, 'wall_bw': 33.0, 'gpt2_gb': 3.0, 'e_reg': 0.1, 'smem_kb': 228.0, 'smem_ns': 17.0, 'l2_mb': 50.0, 'l2_ns': 150.0, 'hbm_gb': 80.0, 'h100_bw': 3.35, 'hbm_ns': 273.0, 'e_dram_byte': 160.0, 'clock_ghz': 2.0, 'cycle_ps': 500.0, 'hbm_cycles': 546.0, 'cnc_l1': 22.4, 'cnc_l2': 159.7, 'cnc_l2far': 294.0, 'cnc_hbm': 377.9, 'cnc_hbm_l1': 17.0, 'pcie_sms': 114.0, 'pcie_bw': 2.0, 'need_tbs': 989.0, 'need_short': 295.0, 'hbm_gbs': 3350.0, 'nvlink_gbs': 900.0, 'pcie_gbs': 64.0, 'net_gbs': 50.0, 'hbm_pcie': 52.0, 'a100_fp16': 312.0, 'a100_bw': 2.04, 'a100_ridge': 153.0, 'h100_fp32': 67.0, 'h100_int8': 1979.0, 'ridge32': 20.0, 'ridge16': 295.2, 'ridge16_int': 295.0, 'ridge8': 590.7, 'ridge8_int': 591.0, 'int8_over_fp16': 2.0, 'fp16_over_fp32': 15.0, 'k_ai': 200.0, 'k_compute_ratio': 3.2, 'k_bw_ratio': 1.6, 'k_a100': 312.0, 'k_h100': 670.0, 'k_gain': 2.1, 'k_gain_2': 2.15, 'ridge_v100': 139.0, 'ridge_a100': 153.0, 'ridge_h100': 295.0, 'ridge_b200': 281.0, 'gen_first_year': 2017.0, 'gen_last_year': 2024.0, 'gen_compute': 18.0, 'gen_bw': 8.9, 'env_devices': 52.0, 'env_compute': 55.0, 'env_bw': 13.0, 'env_best_bw': 8.0, 'env_since_year': 2025.0, 'qkv_batch': 8.0, 'conv_batch': 32.0, 'qkv_gflop': 125.8, 'qkv_mb': 120.2, 'qkv_ai': 1046.7, 'conv_gflop': 7.4, 'conv_mb': 13.1, 'conv_ai': 563.1, 'head_mflop': 131.1, 'head_mb': 4.3, 'head_ai': 30.5, 'ln_mflop': 104.9, 'ln_mb': 52.4, 'ln_ai': 2.0, 'dec_gflop': 3.0, 'dec_ai': 1.0, 'head_tflops': 102.0, 'head_pct': 10.3, 'ln_tflops': 6.7, 'ln_pct': 0.7, 'dec_tflops': 3.4, 'dec_pct': 0.3, 'top_pct': 100.0, 'softmax_ai': 0.75, 'nsight_dram_pct': 0.34, 'dense_width': 2048.0, 'batch_mid': 32.0, 'batch_big': 512.0, 'ai_b1': 1.0, 'ai_b32': 31.0, 'ai_b512': 341.0, 'ai_b512_1': 341.3, 'crossing': 415.0, 'resnet_gflop': 8.2, 'floor16_us': 8.0, 'floor8_us': 4.0, 's3_fp16': 563.0, 's3_int8': 1126.0, 's3_mixed': 569.0, 's2_fp16': 287.0, 's2_int8': 574.0, 'p2_fp16': 51.0, 'p2_int8': 102.0, 'p_resnet': 0.95, 'p_resnet_pct': 95.0, 'serial_resnet': 0.05, 'serial_resnet_pct': 5.0, 'p_gpt': 0.8, 'p_gpt_pct': 80.0, 'cpu_tops': 8.0, 'gain': 247.0, 'amdahl_rn': 18.6, 'amdahl_gpt': 4.9, 'bw_gain': 67.0, 'amdahl_bw': 4.7, 'amdahl_ceiling': 5.0, 'lh_resnet16': 51.1, 'lh_resnet8': 25.6, 'lh_mobilenet16': 7.0, 'lh_mobilenet8': 3.5, 'lh_mobilenet_gflop': 0.6, 'lh_gpt8': 1.5, 'lh_dlrm_gb': 100.0, 'lh_kws16': 400.0, 'lh_kws8': 200.0, 'lh_kws_mflop': 20.0, 'a100_t4_peak': 4.8, 'a100_t4_bw': 6.4, 'ds_tf32': 989.0, 'ds_fp16': 1979.0, 'ds_int8': 3958.0, 'mapping_gain': 75.0}
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
import contextlib, io
C = sp.C
plt = sp.plt
out = Path(__file__).parent
fig, ax = plt.subplots(figsize=(3.0, 1.45))
rows = [('Throughput', 15.0, 30.0, 'compute', '15–30×'), ('Performance per watt', 30.0, 80.0, 'data', '30–80×')]
for i, (label, low, high, color_key, text) in enumerate(rows):
    y = 1 - i
    ax.barh(y, high - low, left=low, color=C[color_key], height=0.55)
    ax.text(0.9, y, label, ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
    ax.text(high * 1.12, y, text, va='center', fontsize=14, fontweight='bold', color=C[color_key])
ax.axvline(1, color=C['neutral'], lw=2)
ax.text(1.12, -0.7, '2015 CPUs, GPUs = 1×', color=C['neutral'], fontsize=10, va='center')
ax.set_xscale('log')
ax.set_xlim(0.9, 400)
ax.set_ylim(-1.0, 1.6)
ax.set_yticks([])
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(['1×', '10×', '100×'])
ax.spines['left'].set_visible(False)
ax.minorticks_off()
ax.grid(axis='x', color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / 'hook_gains.svg')
sp.hbars(out / 'recall_int8.svg', [('Batch one', 1.3, 'neutral', '1.3×   (0.38 → 0.30 ms)'), ('Large batch', 5.9, 'compute', '5.9×')], '×', log=False, xlabel='INT8 against FP32, ResNet-50, same H100', size=sp.SHORT)
energy_rows = [('INT8 add', 0.03, 'compute', '0.03 pJ'), ('FP32 add', 0.9, 'compute', '0.9 pJ'), ('FP32 multiply', 3.7, 'compute', '3.7 pJ'), ('SRAM read (8 KB)', 5.0, 'memory', '5 pJ'), ('DRAM read (32-bit)', 640.0, 'danger', '640 pJ = 173 multiplies')]
for k, shown in ((1, 3), (2, 4), (3, 5)):
    sp.hbars(out / f'energy_step{k}.svg', energy_rows, 'pJ', step=shown, xlabel='Energy per operation or access (pJ, log scale; 45 nm)', size=(5.8, 2.3))
sp.hbars(out / 'tensorcore.svg', [(f'P100 ({2016})', 21.2, 'neutral', '21.2'), (f'V100 ({2017})', 125.0, 'compute', '125: about 6×')], 'TFLOP/s', log=False, xlabel='Peak FP16 (TFLOP/s)', size=(2.6, 1.3))
FONT = 'font-family="Helvetica, Arial, sans-serif"'

def svg_text(x, y, s, size=20, weight='400', fill=C['text'], anchor='middle'):
    return f'<text x="{x}" y="{y}" {FONT} font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" dominant-baseline="central">{s}</text>'

def units_svg(step):
    lanes = [('Vector unit', 'ReLU, residual add', C['memory'], C['memory_l'], 'ResNet-50: after every conv'), ('Matrix unit', 'convolution, projections', C['compute'], C['compute_l'], 'Both: most of the arithmetic'), ('Special-function unit', 'softmax exp, GELU, rsqrt', C['network'], C['network_l'], 'GPT-2 XL: every step')]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="440" viewBox="0 0 720 440">', '<rect width="720" height="440" fill="white"/>']
    for i, (name, ops, stroke, fill, tag) in enumerate(lanes):
        y = 8 + i * 145
        parts.append(f'<rect x="8" y="{y}" width="704" height="130" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        parts.append(svg_text(30, y + 32, name, 30, '700', stroke, 'start'))
        parts.append(svg_text(30, y + 70, ops, 27, '400', C['text'], 'start'))
        opacity = '1' if step >= 2 else '0'
        parts.append(f'<g opacity="{opacity}">')
        parts.append(svg_text(30, y + 106, '→ ' + tag, 26, '700', stroke, 'start'))
        parts.append('</g>')
    parts.append('</svg>')
    return '\n'.join(parts)
for k in (1, 2):
    (out / f'units_step{k}.svg').write_text(units_svg(k))

def arrow(x1, y1, x2, y2, color, width=3):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" marker-end="url(#head_{color[1:]})"/>'

def systolic_svg(step):
    cell, gap, x0, y0, n = (74, 30, 150, 50, 4)
    colors = [C['memory'], C['data'], C['compute']]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="530" viewBox="0 0 640 530">', '<rect width="640" height="530" fill="white"/>', '<defs>']
    for col in colors:
        parts.append(f'<marker id="head_{col[1:]}" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{col}"/></marker>')
    parts.append('</defs>')
    for r in range(n):
        for c in range(n):
            x = x0 + c * (cell + gap)
            y = y0 + r * (cell + gap)
            parts.append(f'''<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="6" fill="{C['compute_l']}" stroke="{C['compute']}" stroke-width="2.5"/>''')
            parts.append(svg_text(x + cell / 2, y + cell / 2, f'w{r + 1}{c + 1}', 30, '700', C['compute']))
    show = lambda k: '1' if step >= k else '0'
    parts.append(f'<g opacity="{show(2)}">')
    for r in range(n):
        y = y0 + r * (cell + gap) + cell / 2
        parts.append(svg_text(x0 - 84, y, f'x{r + 1}', 30, '700', C['memory']))
        parts.append(arrow(x0 - 52, y, x0 - 6, y, C['memory']))
        for c in range(n - 1):
            xa = x0 + c * (cell + gap) + cell + 3
            parts.append(arrow(xa, y, xa + gap - 8, y, C['memory']))
    parts.append('</g>')
    parts.append(f'<g opacity="{show(3)}">')
    bottom = y0 + n * (cell + gap) - gap
    for c in range(n):
        x = x0 + c * (cell + gap) + cell / 2
        for r in range(n - 1):
            ya = y0 + r * (cell + gap) + cell + 3
            parts.append(arrow(x, ya, x, ya + gap - 8, C['data']))
        parts.append(arrow(x, bottom + 3, x, bottom + 34, C['data']))
        parts.append(svg_text(x, bottom + 58, f'y{c + 1}', 30, '700', C['data']))
    parts.append('</g>')
    parts.append('</svg>')
    return '\n'.join(parts)
for k in (1, 2, 3):
    (out / f'systolic_step{k}.svg').write_text(systolic_svg(k))
BW = 3.35
XLIM = (0.5, 5000)
YLIM = (1, 6000)

def roof_axes(size=(5.8, 2.9)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.minorticks_off()
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1', '10', '100', '1,000'])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(['1', '10', '100', '1,000'])
    ax.grid(color=C['grid'], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel('Arithmetic intensity (FLOP/byte, log scale)')
    ax.set_ylabel('TFLOP/s (log)')
    return (fig, ax)

def slope(ax, top):
    xs = np.logspace(np.log10(XLIM[0]), np.log10(top / BW), 50)
    ax.plot(xs, xs * BW, color=C['memory'], lw=3, zorder=3)
ceilings = [('FP32', 'h100_fp32', '20', 'neutral', '-'), ('FP16', 'h100_fp16', '295', 'compute', '-'), ('INT8', 'h100_int8', '591', 'danger', '--')]
for k in (1, 2, 3):
    fig, ax = roof_axes()
    slope(ax, {1: 67.0, 2: 989.0, 3: 1979.0}[k])
    ax.text(6, 6 * BW / 3.2, '3.35 TB/s', color=C['memory'], fontsize=12, fontweight='bold', ha='left', va='top')
    for i, (name, peak_key, ridge_text, color_key, style) in enumerate(ceilings):
        alpha = 1.0 if i < k else 0.0
        peak = {'neutral': 67.0, 'compute': 989.0, 'danger': 1979.0}[color_key]
        ridge = peak / BW
        ax.plot([ridge, XLIM[1]], [peak, peak], color=C[color_key], lw=3, ls=style, alpha=alpha, zorder=3)
        ax.plot([ridge, ridge], [YLIM[0], 3.2], color=C[color_key], lw=1.2, ls=':', alpha=alpha)
        below = name == 'FP16'
        ax.text(XLIM[1] / 1.15, peak / 1.15 if below else peak * 1.2, f"{name} { {'neutral': '67', 'compute': '989', 'danger': '1,979'}[color_key]} " + ('TOPS' if name == 'INT8' else 'TFLOP/s'), color=C[color_key], fontsize=11, fontweight='bold', ha='right', va='top' if below else 'bottom', alpha=alpha)
        left = name == 'FP16'
        ax.text(ridge / 1.08 if left else ridge * 1.08, 1.25, f'ridge\n{ridge_text}', color=C[color_key], fontsize=10, fontweight='bold', ha='right' if left else 'left', va='bottom', alpha=alpha)
    sp.save(fig, out / f'roof_h100_step{k}.svg')
PEAK16 = 989.0
layers = [('GPT-2 XL decode, batch 1: 0.3%', 1.0, 'danger', (8, -4), 'left'), ('LayerNorm: 0.7%', 2.0, 'danger', (8, -4), 'left'), ('ResNet-50 FC head: 10.3%', 30.5, 'memory', (8, -4), 'left'), ('ResNet-50 stage-3 conv', 563.1, 'compute', (4, -16), 'left'), ('GPT-2 XL QKV prefill', 1046.7, 'compute', (0, 14), 'center')]
reveal = [2, 3, 4, 5]
for k, shown in enumerate(reveal, start=1):
    fig, ax = roof_axes()
    slope(ax, PEAK16)
    ax.plot([PEAK16 / BW, XLIM[1]], [PEAK16, PEAK16], color=C['compute'], lw=3)
    ax.plot([PEAK16 / BW] * 2, [YLIM[0], 2.2], color=C['neutral'], lw=1.2, ls=':')
    ax.text(PEAK16 / BW / 1.1, 1.3, 'ridge 295', color=C['neutral'], fontsize=10, ha='right')
    ax.text(0.02, 0.95, 'FP16 ceiling: 989 TFLOP/s', transform=ax.transAxes, color=C['compute'], fontsize=11, fontweight='bold', ha='left', va='top')
    for i, (label, x, color_key, offset, align) in enumerate(layers):
        alpha = 1.0 if i < shown else 0.0
        y = min(PEAK16, BW * x)
        ax.plot([x], [y], 'o', ms=10, color=C[color_key], mec='white', mew=1.5, zorder=5, alpha=alpha)
        ax.annotate(label, (x, y), xytext=offset, textcoords='offset points', ha=align, va='center', fontsize=11, fontweight='bold', color=C[color_key], alpha=alpha)
    sp.save(fig, out / f'layers_step{k}.svg')
crossing = 415.0
batches = np.logspace(0, np.log10(4096), 200)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.7, 5000)
    ax.set_ylim(0.5, 1500)
    ax.minorticks_off()
    ax.set_xticks([1, 8, 32, 128, 512, 2048])
    ax.set_xticklabels(['1', '8', '32', '128', '512', '2,048'])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(['1', '10', '100', '1,000'])
    ax.grid(color=C['grid'], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axhline(295.2, color=C['compute'], lw=2, ls='--')
    ax.text(0.8, 295.2 * 1.25, 'H100 FP16 ridge: 295', color=C['compute'], fontsize=11, fontweight='bold')
    if k >= 2:
        ax.plot(batches, [0.9990243902, 1.041623019, 1.086036135, 1.132340859, 1.180617572, 1.230950059, 1.283425645, 1.338135346, 1.395174024, 1.454640545, 1.516637949, 1.581273624, 1.648659482, 1.718912156, 1.79215319, 1.868509242, 1.948112302, 2.031099908, 2.117615378, 2.207808045, 2.301833511, 2.399853901, 2.502038131, 2.608562189, 2.719609426, 2.835370853, 2.956045456, 3.081840521, 3.212971974, 3.349664726, 3.492153044, 3.64068092, 3.795502474, 3.95688235, 4.125096146, 4.300430848, 4.483185285, 4.6736706, 4.872210735, 5.07914294, 5.294818294, 5.519602247, 5.753875181, 5.998032992, 6.252487687, 6.517668008, 6.794020069, 7.082008025, 7.382114751, 7.694842552, 8.020713887, 8.360272127, 8.714082324, 9.082732011, 9.466832022, 9.867017337, 10.28394795, 10.71830976, 11.17081549, 11.64220561, 12.13324933, 12.64474554, 13.17752387, 13.73244567, 14.31040508, 14.9123301, 15.53918368, 16.1919648, 16.87170963, 17.57949265, 18.31642779, 19.08366958, 19.88241435, 20.71390139, 21.5794141, 22.48028119, 23.41787784, 24.39362686, 25.40899982, 26.4655182, 27.56475447, 28.70833319, 29.89793201, 31.13528268, 32.42217197, 33.7604426, 35.15199397, 36.59878295, 38.1028245, 39.66619222, 41.29101877, 42.97949617, 44.73387597, 46.5564693, 48.44964663, 50.41583754, 52.4575301, 54.57727015, 56.77766032, 59.06135878, 61.43107772, 63.88958153, 66.4396847, 69.08424935, 71.82618241, 74.66843241, 77.61398592, 80.66586349, 83.8271152, 87.10081574, 90.49005898, 93.99795204, 97.62760885, 101.3821432, 105.264661, 109.2782525, 113.4259833, 117.7108852, 122.1359463, 126.7041005, 131.4182165, 136.2810859, 141.2954116, 146.4637944, 151.78872, 157.2725455, 162.9174845, 168.7255933, 174.6987553, 180.8386661, 187.1468179, 193.6244839, 200.2727026, 207.0922623, 214.0836855, 221.2472135, 228.5827919, 236.0900558, 243.7683164, 251.6165475, 259.633374, 267.8170601, 276.1654996, 284.6762073, 293.3463116, 302.1725488, 311.1512592, 320.2783847, 329.5494686, 338.9596573, 348.503704, 358.1759744, 367.9704552, 377.880764, 387.9001623, 398.0215695, 408.2375806, 418.5404847, 428.9222864, 439.3747284, 449.8893167, 460.4573468, 471.0699318, 481.7180317, 492.3924836, 503.0840332, 513.7833664, 524.4811424, 535.1680253, 545.8347176, 556.4719919, 567.0707228, 577.6219179, 588.1167481, 598.546576, 608.9029835, 619.1777981, 629.3631166, 639.451328, 649.4351333, 659.3075642, 669.0619993, 678.6921781, 688.1922125, 697.5565966, 706.7802141, 715.8583435, 724.7866607, 733.5612406, 742.1785558, 750.635474, 758.9292529, 767.0575347, 775.0183376, 782.8100472, 790.431406, 797.8815016, 805.1597548, 812.2659057, 819.2], color=C['memory'], lw=3)
    pts = [(1, '≈ 1', (12, -12)), (32, '≈ 31', (8, -10)), (512, '341', (-10, 10))]
    for j, (b, text, offset) in enumerate(pts):
        alpha = 1.0 if k >= 2 and j < 2 or k == 3 else 0.0
        ax.plot([b], [{0: 0.9990243902, 1: 31.03030303, 2: 341.3333333}[j]], 'o', ms=10, color=C['memory'], mec='white', mew=1.5, zorder=5, alpha=alpha)
        ax.annotate(text, (b, {0: 0.9990243902, 1: 31.03030303, 2: 341.3333333}[j]), xytext=offset, textcoords='offset points', fontsize=12, fontweight='bold', color=C['memory'], alpha=alpha, ha='right' if offset[0] < 0 else 'left')
    ax.text(512.0, 3, '???', fontsize=16, fontweight='bold', color=C['danger'], ha='center', alpha=1.0 if k == 1 else 0.0)
    alpha = 1.0 if k == 3 else 0.0
    ax.axvline(crossing, color=C['danger'], lw=1.2, ls=':', alpha=alpha)
    ax.text(crossing * 1.1, 1.0, 'crosses near 415\n(derived)', color=C['danger'], fontsize=10, alpha=alpha, va='bottom')
    ax.set_xlabel('Batch size (log scale)')
    ax.set_ylabel('FLOP/byte (log)')
    sp.save(fig, out / f'batch_step{k}.svg')
int8_bars = [('Stage-3 3×3 conv', [(563.0, '563', 'fp16'), (1126.0, '1,126', 'int8'), (569.0, '569: INT8, activations FP16', 'wide')]), ('Stage-2 3×3 conv', [(287.0, '287', 'fp16'), (574.0, '574', 'int8')]), ('Stage-2 1×1 expansion', [(51.0, '51', 'fp16'), (102.0, '102', 'int8')])]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    y = 0.0
    centers = []
    for name, bars in reversed(int8_bars):
        ys = []
        for value, label, style in reversed(bars):
            alpha = 1.0 if style == 'fp16' or k == 3 else 0.0
            if style == 'fp16':
                face, edge, ink = (C['compute'], C['compute'], C['compute'])
            elif style == 'int8':
                face, edge, ink = (C['danger_l'], C['danger'], C['danger'])
            else:
                face, edge, ink = (C['danger'], C['danger'], C['danger'])
            ax.barh(y, value, height=0.8, color=face, edgecolor=edge, alpha=alpha)
            ax.text(value * 1.12, y, label, va='center', fontsize=10.5, fontweight='bold', color=ink, alpha=alpha)
            ys.append(y)
            y += 1
        centers.append((name, sum(ys) / len(ys)))
        y += 0.6
    for name, center in centers:
        ax.text(-0.01, center, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
    top = y - 0.4
    ax.axvline(295.2, color=C['compute'], lw=2, ls='--')
    ax.text(295.2 / 1.04, top, 'FP16 ridge 295', color=C['compute'], fontsize=10, fontweight='bold', ha='right', va='bottom')
    alpha = 1.0 if k >= 2 else 0.0
    ax.axvline(590.7, color=C['danger'], lw=2, ls='--', alpha=alpha)
    ax.text(590.7 * 1.04, top, 'INT8 ridge 591', color=C['danger'], fontsize=10, fontweight='bold', ha='left', va='bottom', alpha=alpha)
    ax.set_xscale('log')
    ax.set_xlim(20, 8000)
    ax.set_ylim(-0.7, top + 0.9)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xticks([30, 100, 300, 1000])
    ax.set_xticklabels(['30', '100', '300', '1,000'])
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Arithmetic intensity at batch 32 (FLOP/byte, log scale)')
    sp.save(fig, out / f'int8ridge_step{k}.svg')
gains = np.logspace(0, 3, 300)
fractions = np.linspace(0.78, 1.0, 300)
grid_g, grid_p = np.meshgrid(gains, fractions)
speedup = 1 / (1 - grid_p + grid_p / grid_g)
fig, ax = plt.subplots(figsize=(3.4, 2.7))
levels = [1, 2, 5, 10, 20, 50, 1000]
shades = ['#EEF4FA', '#D1E6F3', '#A9CFE8', '#7FB6DB', '#4C93C4', '#006395']
ax.contourf(grid_g, grid_p, speedup, levels=levels, colors=shades)
lines = ax.contour(grid_g, grid_p, speedup, levels=[2, 5, 10, 20, 50], colors='white', linewidths=1.2)
for value in (5, 10, 20, 50):
    edge = (1 - 1 / value) / (1 - 1 / 1000)
    ax.text(1150, edge, f'{value}×', va='center', ha='left', fontsize=10, color=C['memory'], fontweight='bold')
G_PATH = 247.0
points = [(0.95, 'ResNet-50: 18.6×', (-8, 10)), (0.8, 'GPT-2 decode: 4.9×', (-8, 10))]
marks = []
for p_frac, label, offset in points:
    dot, = ax.plot([G_PATH], [p_frac], 'o', ms=9, color=C['danger'], mec='white', mew=1.5)
    text = ax.annotate(label, (G_PATH, p_frac), xytext=offset, textcoords='offset points', ha='right', fontsize=11, fontweight='bold', color=C['danger'], bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
    marks.append((dot, text))
ax.set_xscale('log')
ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(['1', '10', '100', '1,000'])
ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
ax.set_xlabel('Accelerator gain (log scale)')
ax.set_ylabel('Accelerated fraction')
fig.savefig(out / 'amdahl.svg', format='svg', bbox_inches='tight', pad_inches=0.15, facecolor='white')
marks[1][0].set_alpha(0.0)
marks[1][1].set_alpha(0.0)
marks[1][1].get_bbox_patch().set_alpha(0.0)
sp.save(fig, out / 'amdahl_step1.svg')
taper_rows = [('HBM', 3350.0, 'memory', '3,350 GB/s'), ('NVLink', 900.0, 'network', '900'), ('PCIe', 64.0, 'danger', '64'), ('Network', 50.0, 'neutral', '50')]
fig, ax = plt.subplots(figsize=(3.6, 2.3))
for i, (label, value, color_key, text) in enumerate(taper_rows):
    y = len(taper_rows) - 1 - i
    ax.barh(y, value, color=C[color_key], height=0.62)
    ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
    ax.text(value * 1.18, y, text, va='center', fontsize=13, fontweight='bold', color=C[color_key])
ax.set_xscale('log')
ax.set_xlim(20, 200000)
ax.set_ylim(-0.6, len(taper_rows) - 0.4)
ax.set_yticks([])
ax.minorticks_off()
ax.set_xticks([100, 1000, 10000])
ax.set_xticklabels(['100', '1,000', '10,000'])
ax.spines['left'].set_visible(False)
ax.grid(axis='x', color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('GB/s from one H100 (log scale)')
sp.save(fig, out / 'taper.svg')
cnc = {'l1': 22.4, 'l2': 159.7, 'l2far': 294.0, 'hbm': 377.9}
starts = [(2, cnc['l1']), (256, cnc['l2']), (40960, cnc['l2far']), (65536, cnc['hbm'])]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
xs = [x for x, _ in starts] + [1048576]
ys = [y for _, y in starts] + [cnc['hbm']]
ax.step(xs, ys, where='post', color=C['memory'], lw=2.5)
labels = [((16, cnc['l1']), 'L1: 22.4 ns', (0, 7), 'center'), ((2048, cnc['l2']), 'L2: 159.7 ns', (0, 7), 'center'), ((40960, cnc['l2far']), 'far L2: 294 ns', (-6, 0), 'right'), ((65536, cnc['hbm']), 'HBM: 377.9 ns', (-6, 0), 'right')]
for (x, y), text, off, ha in labels:
    ax.annotate(text, (x, y), xytext=off, textcoords='offset points', ha=ha, va='bottom' if off[1] else 'center', fontsize=11, fontweight='bold', color=C['text'])
ax.set_xscale('log', base=2)
ax.set_xlim(2, 1048576)
ax.set_ylim(0, 430)
ax.minorticks_off()
ax.set_xticks([16, 1024, 65536, 1048576])
ax.set_xticklabels(['16 KB', '1 MB', '64 MB', '1 GB'])
ax.set_yticks([0, 100, 200, 300, 400])
ax.grid(color=C['grid'], linewidth=0.5)
ax.set_axisbelow(True)
ax.set_xlabel('Working set (log scale)')
ax.set_ylabel('Latency (ns)')
sp.save(fig, out / 'latency_measured.svg')
ridges = [('V100', 2017, 138.8888889), ('A100', 2020, 153.0161844), ('H100', 2022, 295.2238806), ('B200', 2024, 281.25)]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.7, 2.6))
    years = [r[1] for r in ridges]
    vals = [r[2] for r in ridges]
    ax.plot(years, vals, 'o-', color=C['danger'], lw=2.5, ms=9, mec='white', mew=1.5)
    for name, year, ridge in ridges:
        dy = 10 if name in ('V100', 'H100') else -20
        ax.annotate(f"{name}: { {'V100': '139', 'A100': '153', 'H100': '295', 'B200': '281'}[name]}", (year, ridge), xytext=(0, dy), textcoords='offset points', ha='center', fontsize=11, fontweight='bold', color=C['danger'])
    alpha = 1.0 if k == 2 else 0.0
    ax.axhline(200.0, color=C['memory'], lw=2, ls='--', alpha=alpha)
    ax.text(2016.5, 200.0 * 1.04, '200 FLOP/byte kernel', color=C['memory'], fontsize=11, fontweight='bold', ha='left', va='bottom', alpha=alpha)
    ax.set_xlim(2016.3, 2024.8)
    ax.set_ylim(0, 360)
    ax.set_xticks([2017, 2020, 2022, 2024])
    ax.set_yticks([0, 100, 200, 300])
    ax.grid(color=C['grid'], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel('Ridge (FLOP/byte)')
    sp.save(fig, out / f'ridge_rise_step{k}.svg')
sp.hbars(out / 'int8peak.svg', [('FP16', 989.0, 'compute', '989 TFLOP/s'), ('INT8', 1979.0, 'danger', '1,979 TOPS: 2×')], '', log=False, xlabel='H100 dense peak', size=(2.9, 1.5))
sp.hbars(out / 'int8floor.svg', [('FP16', 8.291203236, 'compute', '8 µs'), ('INT8', 4.143506822, 'danger', '4 µs')], '', log=False, xlabel='ResNet-50 arithmetic, one image', size=(2.9, 1.5))
print('figures written')
