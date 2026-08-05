import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for academic publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# Plot 1: Roofline Visual Performance Model
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.8))

# Roofline parameters for Apple Silicon UMA host
peak_bw = 150.0   # GB/s DRAM peak bandwidth
peak_flops = 2500.0 # GFLOP/s compute peak
knee = peak_flops / peak_bw # 16.67 FLOPs/byte

# X-axis: Operational Intensity (FLOPs/byte)
oi_roof = np.logspace(-1, 3, 500)
perf_roof = np.minimum(peak_bw * oi_roof, peak_flops)

# Plot Roofline envelope
ax.plot(oi_roof, perf_roof, 'k-', linewidth=2, label='Hardware Ceiling (150 GB/s, 2.5 TFLOP/s)')
ax.axvline(x=knee, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'Roofline Knee ({knee:.1f} FLOPs/B)')

# Workload data: (Name, OI in FLOPs/B, Performance in GFLOP/s, Category)
workloads = [
    ("LLM Decode", 0.5, 75, "Memory-Bound"),
    ("GCN (Graph)", 1.2, 175, "Memory-Bound"),
    ("NCF (RecSys)", 2.1, 310, "Memory-Bound"),
    ("Autoencoder", 4.2, 580, "Memory-Bound"),
    ("DS-CNN (KWS)", 5.8, 820, "Memory-Bound"),
    ("MobileNetV2", 8.5, 1180, "Memory-Bound"),
    ("PatchTST", 12.0, 1650, "Memory-Bound"),
    ("DistilBERT", 24.0, 2450, "Compute-Bound"),
    ("MiniLM-L6", 32.0, 2480, "Compute-Bound"),
    ("ResNet8", 64.0, 2500, "Compute-Bound"),
    ("Qwen2.5-Coder", 96.0, 2500, "Compute-Bound"),
    ("LLM Prefill", 128.0, 2500, "Compute-Bound"),
    ("Qwen3 AST", 160.0, 2500, "Compute-Bound"),
    ("EDM Diffusion", 256.0, 2500, "Compute-Bound"),
]

# Color map for categories
colors = {'Memory-Bound': '#d95f02', 'Compute-Bound': '#1b9e77'}
markers = {'Memory-Bound': 'o', 'Compute-Bound': 's'}

seen_cats = set()
for name, oi, perf, cat in workloads:
    c = colors[cat]
    m = markers[cat]
    lbl = cat if cat not in seen_cats else ""
    seen_cats.add(cat)
    ax.scatter(oi, perf, color=c, marker=m, s=45, zorder=5, label=lbl, edgecolor='black', linewidth=0.5)
    
    # Label placement tuning to avoid overlap
    y_offset = 1.15 if cat == "Compute-Bound" else 0.82
    if name in ["LLM Prefill", "Qwen2.5-Coder"]:
        y_offset = 0.75
    elif name in ["ResNet8", "DistilBERT"]:
        y_offset = 1.20
    ax.annotate(name, (oi, perf), textcoords="offset points", xytext=(0, 6 if y_offset > 1.0 else -12),
                ha='center', fontsize=7, fontweight='bold' if cat=='Compute-Bound' else 'normal')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 1000)
ax.set_ylim(10, 5000)

ax.set_xlabel('Operational Intensity (FLOPs / Byte)')
ax.set_ylabel('Performance (GFLOP / s)')
ax.set_title('Roofline Model: Operational Intensity & Hardware Ceilings across 14 Workloads', fontweight='bold')
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_roofline.pdf'), dpi=300)
plt.savefig(os.path.join(out_dir, 'fig_roofline.png'), dpi=200)
plt.close()
print("Generated fig_roofline.pdf and fig_roofline.png")

# ---------------------------------------------------------
# Plot 2: CPU vs MPS Backend Performance Speedup
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.8))

backend_data = [
    ("ResNet8 (Vision)", 4.15, "Dense GEMM"),
    ("EDM Diffusion (GenAI)", 3.86, "Dense GEMM"),
    ("LLM Prefill (nanoGPT)", 3.54, "Dense GEMM"),
    ("DistilBERT (NLP)", 2.78, "Dense GEMM"),
    ("MiniLM-L6 (Retrieval)", 2.34, "Dense GEMM"),
    ("Autoencoder (Audio)", 2.07, "Dense GEMM"),
    ("DS-CNN (KWS)", 1.77, "Dense GEMM"),
    ("MobileNetV2 (VWW)", 1.56, "Dense GEMM"),
    ("NCF (Recommendation)", 1.30, "Sparse / Memory"),
    ("PatchTST (Time Series)", 1.07, "Sequential / Memory"),
    ("LLM Decode (nanoGPT)", 1.01, "Memory Bandwidth"),
    ("GCN (Graph ogbn-arxiv)", 0.99, "Sparse Scatter/Gather"),
    ("Qwen3 (Function Calling)", 0.98, "Branch / AST Parsing"),
]

names = [d[0] for d in backend_data]
speedups = [d[1] for d in backend_data]
types = [d[2] for d in backend_data]

y_pos = np.arange(len(names))
bar_colors = ['#2b5c8f' if s >= 1.5 else ('#4c956c' if s >= 1.0 else '#e76f51') for s in speedups]

bars = ax.barh(y_pos, speedups, align='center', color=bar_colors, edgecolor='black', linewidth=0.5, height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()  # top-down ranking
ax.axvline(x=1.0, color='crimson', linestyle='--', linewidth=1.2, label='1.0x Parity (CPU = MPS)')

# Add value labels next to bars
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.2f}x',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),  # 4 points horizontal offset
                textcoords="offset points",
                ha='left', va='center', fontsize=7.5, fontweight='bold')

ax.set_xlabel('Speedup Factor (MPS GPU / CPU Execution Time)')
ax.set_xlim(0, 4.8)
ax.set_title('Backend Performance Comparison: MPS GPU Acceleration relative to CPU Baseline', fontweight='bold')
ax.grid(True, axis='x', ls=":", alpha=0.5)

# Custom legend for speedup categories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2b5c8f', edgecolor='black', label='High Speedup (Dense GEMM Kernel)'),
    Patch(facecolor='#4c956c', edgecolor='black', label='Moderate / Parity (Memory-Bound)'),
    Patch(facecolor='#e76f51', edgecolor='black', label='GPU Overhead (Sparse / Branch-Heavy)'),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_cpu_vs_mps.pdf'), dpi=300)
plt.savefig(os.path.join(out_dir, 'fig_cpu_vs_mps.png'), dpi=200)
plt.close()
print("Generated fig_cpu_vs_mps.pdf and fig_cpu_vs_mps.png")
