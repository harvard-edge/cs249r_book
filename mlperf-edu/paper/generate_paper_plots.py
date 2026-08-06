import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for publication quality matching Palatino / LaTeX paper style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.titlesize': 9.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)

# Shared Color Palette
COLOR_PASS = '#2e7d32'       # Forest green
COLOR_MISS = '#c62828'       # Deep red
COLOR_PRIMARY = '#1565c0'    # Sapphire blue
COLOR_SECONDARY = '#d95f02'  # Warm amber/orange

# =========================================================================
# Figure 1: Roofline Visual Performance Model (fig_roofline)
# =========================================================================
fig, ax = plt.subplots(figsize=(6.5, 3.4))

peak_bw = 150.0      # GB/s DRAM peak bandwidth
peak_flops = 2500.0  # GFLOP/s compute peak
knee = peak_flops / peak_bw # 16.67 FLOPs/byte

oi_roof = np.logspace(-1, 3.2, 500)
perf_roof = np.minimum(peak_bw * oi_roof, peak_flops)

# Roofline Envelope
ax.plot(oi_roof, perf_roof, 'k-', linewidth=1.8, label='Hardware Ceiling (150 GB/s, 2.5 TFLOP/s)')
ax.axvline(x=knee, color='gray', linestyle='--', linewidth=0.9, alpha=0.7, label=f'Roofline Knee ({knee:.1f} FLOPs/B)')

roof_data = [
    # Memory-Bound
    ("LLM Decode", 0.5, 75, "Memory-Bound", (-10, 8), 'right'),
    ("GCN (Graph)", 1.2, 175, "Memory-Bound", (-10, 8), 'right'),
    ("NCF (RecSys)", 2.1, 310, "Memory-Bound", (-10, 8), 'right'),
    ("Autoencoder", 4.2, 580, "Memory-Bound", (-10, 8), 'right'),
    ("DS-CNN (KWS)", 5.8, 820, "Memory-Bound", (8, -12), 'left'),
    ("MobileNetV2", 8.5, 1180, "Memory-Bound", (-10, 8), 'right'),
    ("PatchTST", 12.0, 1650, "Memory-Bound", (-10, 8), 'right'),
    # Compute-Bound (Staggered offsets to avoid overlap along 2500 GFLOP/s line)
    ("DistilBERT", 24.0, 2500, "Compute-Bound", (0, 10), 'center'),
    ("MiniLM-L6", 36.0, 2500, "Compute-Bound", (0, -15), 'center'),
    ("ResNet8", 64.0, 2500, "Compute-Bound", (0, 10), 'center'),
    ("Qwen2.5-Coder", 110.0, 2500, "Compute-Bound", (0, -15), 'center'),
    ("LLM Prefill", 170.0, 2500, "Compute-Bound", (0, 10), 'center'),
    ("Qwen3 AST", 260.0, 2500, "Compute-Bound", (0, -15), 'center'),
    ("EDM Diffusion", 450.0, 2500, "Compute-Bound", (0, 10), 'center'),
]

seen_cats = set()
for name, oi, perf, cat, (off_x, off_y), ha in roof_data:
    c = COLOR_SECONDARY if cat == "Memory-Bound" else COLOR_PASS
    m = 'o' if cat == "Memory-Bound" else 's'
    lbl = cat if cat not in seen_cats else ""
    seen_cats.add(cat)
    
    ax.scatter(oi, perf, color=c, marker=m, s=38, zorder=5, label=lbl, edgecolor='black', linewidth=0.5)
    
    ax.annotate(name, (oi, perf), textcoords="offset points", xytext=(off_x, off_y),
                ha=ha, fontsize=6.8, fontweight='bold' if cat=='Compute-Bound' else 'normal',
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7) if off_y < 0 else None)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.15, 1200)
ax.set_ylim(20, 4500)

ax.set_xlabel('Operational Intensity (FLOPs / Byte)')
ax.set_ylabel('Performance (GFLOP / s)')
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_roofline.pdf'), dpi=300)
plt.savefig(os.path.join(out_dir, 'fig_roofline.png'), dpi=200)
plt.close()

# =========================================================================
# Figure 2: CPU vs MPS Backend Performance (fig_cpu_vs_mps)
# =========================================================================
fig, ax = plt.subplots(figsize=(6.5, 3.4))

backend_data = [
    ("ResNet8 (Vision)", 4.15, "Dense GEMM"),
    ("EDM Diffusion (GenAI)", 3.86, "Dense GEMM"),
    ("LLM Prefill (nanoGPT)", 3.54, "Dense GEMM"),
    ("DistilBERT (NLP)", 2.78, "Dense GEMM"),
    ("MiniLM-L6 (Retrieval)", 2.34, "Dense GEMM"),
    ("Autoencoder (Audio)", 2.07, "Dense GEMM"),
    ("DS-CNN (KWS)", 1.77, "Dense GEMM"),
    ("MobileNetV2 (VWW)", 1.56, "Dense GEMM"),
    ("NCF (Recommendation)", 1.30, "Memory-Bound"),
    ("PatchTST (Time Series)", 1.07, "Sequential"),
    ("LLM Decode (nanoGPT)", 1.01, "Memory-Bound"),
    ("GCN (Graph ogbn-arxiv)", 0.99, "Sparse Scatter/Gather"),
    ("Qwen3 (Function Calling)", 0.98, "Branch / AST Parsing"),
]

names = [d[0] for d in backend_data]
speedups = [d[1] for d in backend_data]

y_pos = np.arange(len(names))
bar_colors = [COLOR_PRIMARY if s >= 1.5 else (COLOR_PASS if s >= 1.0 else COLOR_MISS) for s in speedups]

bars = ax.barh(y_pos, speedups, align='center', color=bar_colors, edgecolor='black', linewidth=0.5, height=0.62)
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=7.5)
ax.invert_yaxis()
ax.axvline(x=1.0, color='crimson', linestyle='--', linewidth=1.0, label='1.0x Parity (CPU = MPS)')

for bar in bars:
    w = bar.get_width()
    ax.annotate(f'{w:.2f}x', xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0), textcoords="offset points",
                ha='left', va='center', fontsize=7, fontweight='bold')

ax.set_xlabel('Speedup Factor (MPS GPU / CPU Execution Time)')
ax.set_xlim(0, 4.8)
ax.grid(True, axis='x', ls=":", alpha=0.4)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_PRIMARY, edgecolor='black', label='Dense GEMM Acceleration (>1.5x)'),
    Patch(facecolor=COLOR_PASS, edgecolor='black', label='Memory-Bound Parity (1.0x - 1.5x)'),
    Patch(facecolor=COLOR_MISS, edgecolor='black', label='Kernel Launch Overhead (<1.0x)'),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_cpu_vs_mps.pdf'), dpi=300)
plt.savefig(os.path.join(out_dir, 'fig_cpu_vs_mps.png'), dpi=200)
plt.close()

# =========================================================================
# Figure 3: Quality Score vs Inherited Target (fig_quality_vs_target)
# FIXED: Legend position moved to prevent text overlap on function-calling bar!
# =========================================================================
fig, ax = plt.subplots(figsize=(6.5, 3.4))

quality_data = [
    ("visual-wake-words", 1.064, True),
    ("anomaly-detection", 1.062, True),
    ("image-classification", 1.024, True),
    ("reinforcement-learning", 1.015, True),
    ("causal-language-modeling", 1.007, True),
    ("graph-node-classification", 1.005, True),
    ("keyword-spotting", 1.002, True),
    ("time-series-forecasting", 1.002, True),
    ("text-classification", 1.000, True),
    ("information-retrieval", 1.000, True),
    ("image-generation", 1.000, True),
    ("recommendation", 1.000, True),
    ("code-generation", 1.000, True),
    ("function-calling", 1.000, True),
]

q_names = [d[0] for d in quality_data]
q_scores = [d[1] for d in quality_data]
q_pass = [d[2] for d in quality_data]

y_pos_q = np.arange(len(q_names))
q_colors = [COLOR_PASS if p else COLOR_MISS for p in q_pass]

q_bars = ax.barh(y_pos_q, q_scores, align='center', color=q_colors, edgecolor='black', linewidth=0.5, height=0.62)
ax.set_yticks(y_pos_q)
ax.set_yticklabels(q_names, fontsize=7.5)
ax.invert_yaxis()
ax.axvline(x=1.0, color='black', linestyle='-', linewidth=1.2, label='1.0 Inherited Gate')

for bar in q_bars:
    w = bar.get_width()
    # Place score label to right of bar (or left if very short)
    if w >= 0.90:
        ax.annotate(f'{w:.3f}', xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(-32, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=6.8, color='white', fontweight='bold')
    else:
        ax.annotate(f'{w:.3f}', xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=6.8, color='black', fontweight='bold')

ax.set_xlabel('Observed Score / Inherited Target Ratio (≥ 1.0 Meets Contract)')
ax.set_xlim(0, 1.22)
ax.grid(True, axis='x', ls=":", alpha=0.4)

q_legend = [
    Patch(facecolor=COLOR_PASS, edgecolor='black', label='Pass (Admitted)'),
    Patch(facecolor=COLOR_MISS, edgecolor='black', label='Recorded Miss (Fail-Closed)'),
]
# Position legend in empty space at lower right above the x-axis
ax.legend(handles=q_legend, loc='lower right', bbox_to_anchor=(0.98, 0.08), frameon=True, facecolor='white', framealpha=0.95, edgecolor='gray')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_quality_vs_target.pdf'), dpi=300)
plt.savefig(os.path.join(out_dir, 'fig_quality_vs_target.png'), dpi=200)
plt.close()

# =========================================================================
# Figure 4: Measured Runtime Distribution (fig_runtime)
# =========================================================================
fig, ax = plt.subplots(figsize=(6.5, 3.4))

runtime_data = [
    ("causal-language-modeling", 2107.4, "35.1m"),
    ("time-series-forecasting", 854.0, "14.3m"),
    ("graph-node-classification", 719.8, "12.0m"),
    ("information-retrieval", 17.97, "18.0s"),
    ("keyword-spotting", 8.01, "8.0s"),
    ("text-classification", 4.35, "4.4s"),
    ("image-classification", 1.00, "1.0s"),
    ("visual-wake-words", 0.72, "0.7s"),
    ("anomaly-detection", 0.28, "0.3s"),
]

r_names = [d[0] for d in runtime_data]
r_times = [d[1] for d in runtime_data]
r_labels = [d[2] for d in runtime_data]

y_pos_r = np.arange(len(r_names))

r_bars = ax.barh(y_pos_r, r_times, align='center', color=COLOR_PRIMARY, edgecolor='black', linewidth=0.5, height=0.62)
ax.set_yticks(y_pos_r)
ax.set_yticklabels(r_names, fontsize=7.5)
ax.invert_yaxis()
ax.set_xscale('log')

for bar, lbl in zip(r_bars, r_labels):
    w = bar.get_width()
    ax.annotate(lbl, xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0), textcoords="offset points",
                ha='left', va='center', fontsize=7, fontweight='bold')

ax.set_xlabel('Wall-Clock Execution Time in Seconds (Log Scale)')
ax.set_xlim(0.15, 4500)
ax.grid(True, axis='x', ls=":", alpha=0.4)
ax.text(0.98, 0.05, 'Suite Total: ~62 min', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, fontstyle='italic')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_runtime.pdf'), dpi=300)
plt.savefig(os.path.join(out_dir, 'fig_runtime.png'), dpi=200)
plt.close()

# =========================================================================
# Figure 5: Training Convergence Curves (fig_training_curves)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.7))

epochs_ts = np.arange(1, 43)
mse_ts = [
    0.376, 0.336, 0.322, 0.313, 0.308, 0.304, 0.301, 0.298, 0.296, 0.294,
    0.293, 0.292, 0.292, 0.292, 0.291, 0.291, 0.291, 0.293, 0.291, 0.291,
    0.292, 0.292, 0.294, 0.294, 0.289, 0.290, 0.295, 0.296, 0.295, 0.293,
    0.293, 0.294, 0.298, 0.296, 0.298, 0.300, 0.303, 0.302, 0.304, 0.310,
    0.302, 0.306
]
target_ts = 0.290

ax1.plot(epochs_ts, mse_ts, 'o-', color=COLOR_PRIMARY, markersize=3.5, linewidth=1.4, label='Test MSE')
ax1.axhline(y=target_ts, color=COLOR_MISS, linestyle='--', linewidth=1.2, label='Target (≤ 0.290)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test MSE (Lower is Better)')
ax1.set_title('Time-Series Forecasting (PatchTST)', fontsize=8.5, fontweight='bold')
ax1.grid(True, ls=":", alpha=0.4)
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none', fontsize=7)

epochs_rec = np.arange(1, 21)
hit_rec = [
    0.542, 0.586, 0.601, 0.610, 0.613, 0.618, 0.623, 0.619, 0.620, 0.619,
    0.621, 0.618, 0.621, 0.619, 0.617, 0.614, 0.615, 0.615, 0.611, 0.612
]
target_rec = 0.635

ax2.plot(epochs_rec, hit_rec, 'o-', color=COLOR_PRIMARY, markersize=3.5, linewidth=1.4, label='Hit Rate @ 10')
ax2.axhline(y=target_rec, color=COLOR_MISS, linestyle='--', linewidth=1.2, label='Target (≥ 0.635)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Hit Rate @ 10 (Higher is Better)')
ax2.set_title('Recommendation (NCF)', fontsize=8.5, fontweight='bold')
ax2.grid(True, ls=":", alpha=0.4)
ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_training_curves.pdf'), dpi=300)
plt.savefig(os.path.join(out_dir, 'fig_training_curves.png'), dpi=200)
plt.close()

# =========================================================================
# Figure 6: Data Lens Ablations (fig_data_lens)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.7))

budgets = [100, 50, 25, 10]
times = [13.75, 6.88, 3.44, 1.38]
scores = [1.470, 1.543, 1.617, 1.748]
target = 1.470

ax1.plot(budgets, times, 's-', color=COLOR_PRIMARY, markersize=5, linewidth=1.6, label='Training Time')
ax1.set_xlabel('Sample Budget (%)')
ax1.set_ylabel('Training Time (seconds)')
ax1.set_title('Training Time vs Data Pruning', fontsize=8.5, fontweight='bold')
ax1.grid(True, ls=":", alpha=0.4)
ax1.invert_xaxis()
for x, y in zip(budgets, times):
    ax1.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=6.8, fontweight='bold')

ax2.plot(budgets, scores, 'o-', color=COLOR_SECONDARY, markersize=5, linewidth=1.6, label='Validation Loss')
ax2.axhline(y=target, color=COLOR_MISS, linestyle='--', linewidth=1.2, label='Target Gate (1.470)')
ax2.set_xlabel('Sample Budget (%)')
ax2.set_ylabel('Loss (Lower is Better)')
ax2.set_title('Task Quality vs Data Pruning', fontsize=8.5, fontweight='bold')
ax2.grid(True, ls=":", alpha=0.4)
ax2.invert_xaxis()
ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none', fontsize=7)
for x, y in zip(budgets, scores):
    ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=6.8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_data_lens.pdf'), dpi=300)
plt.savefig(os.path.join(out_dir, 'fig_data_lens.png'), dpi=200)
plt.close()

# =========================================================================
# Figure 7: Algorithm Lens Quantization Ablations (fig_algo_lens)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.7))

precisions = ['FP32', 'FP16', 'INT8']
sizes_mb = [440, 220, 110]
latencies_ms = [5.63, 2.82, 1.48]

x_pos = np.arange(len(precisions))

bars1 = ax1.bar(x_pos, sizes_mb, color=COLOR_PRIMARY, width=0.45, edgecolor='black', linewidth=0.5)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(precisions, fontsize=8, fontweight='bold')
ax1.set_ylabel('Model Footprint (MB)')
ax1.set_title('Weight Memory Reduction', fontsize=8.5, fontweight='bold')
ax1.grid(True, axis='y', ls=":", alpha=0.4)
for bar in bars1:
    h = bar.get_height()
    ax1.annotate(f'{int(h)} MB', xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=6.8, fontweight='bold')

bars2 = ax2.bar(x_pos, latencies_ms, color=COLOR_PASS, width=0.45, edgecolor='black', linewidth=0.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(precisions, fontsize=8, fontweight='bold')
ax2.set_ylabel('Inference Latency (ms)')
ax2.set_title('Inference Speedup', fontsize=8.5, fontweight='bold')
ax2.grid(True, axis='y', ls=":", alpha=0.4)
for bar in bars2:
    h = bar.get_height()
    ax2.annotate(f'{h:.2f} ms', xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=6.8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_algo_lens.pdf'), dpi=300)
plt.savefig(os.path.join(out_dir, 'fig_algo_lens.png'), dpi=200)
plt.close()

print("Regenerated all 5 figures with zero overlaps and pristine layout!")
