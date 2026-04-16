#!/usr/bin/env python3
"""
Generate the Global Convergence Summary (Figure 3) for MLPerf EDU.

This script creates a publication-quality, full-width figure showing
training convergence for ALL workloads, grouped by division.

Output: paper/figures/convergence_summary.pdf
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(PAPER_DIR, "figures")

# Publication styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.5,
})

# Full color palette — one color per workload
PALETTE = {
    'nanogpt':        '#2563EB',  # Blue
    'nano-moe':       '#D97706',  # Amber
    'micro-dlrm':     '#DC2626',  # Red
    'micro-diffusion':'#7C3AED',  # Purple
    'micro-gcn':      '#059669',  # Emerald
    'micro-bert':     '#0891B2',  # Cyan
    'micro-lstm':     '#EA580C',  # Orange
    'micro-rl':       '#DB2777',  # Pink
    'resnet18':       '#16A34A',  # Green
    'mobilenetv2':    '#65A30D',  # Lime
    'dscnn':          '#0284C7',  # Sky
    'anomaly-ae':     '#4F46E5',  # Indigo
    'vww':            '#9333EA',  # Violet
}

# Verified metrics from all training runs (empirical data)
WORKLOAD_DATA = {
    # Cloud Division
    'nanogpt': {
        'label': 'NanoGPT', 'div': 'Cloud', 'params': '85.9M',
        'time_s': 89, 'target': ('loss', '<', 2.3),
        'final_train': 2.25, 'final_val': 2.31,
    },
    'nano-moe': {
        'label': 'Nano-MoE', 'div': 'Cloud', 'params': '17.4M',
        'time_s': 158, 'target': ('loss', '<', 0.05),
        'final_train': 0.042, 'final_val': 0.048,
    },
    'micro-dlrm': {
        'label': 'DLRM', 'div': 'Cloud', 'params': '23K',
        'time_s': 5, 'target': ('acc', '>', 0.70),
        'final_train': 0.58, 'final_val': 0.61,
    },
    'micro-diffusion': {
        'label': 'Diffusion', 'div': 'Cloud', 'params': '2.0M',
        'time_s': 41, 'target': ('mse', '<', 0.002),
        'final_train': None, 'final_val': None,  # will load from JSON
    },
    'micro-gcn': {
        'label': 'GCN', 'div': 'Cloud', 'params': '94K',
        'time_s': 2, 'target': ('acc', '>', 0.78),
        'final_train': 0.35, 'final_val': 0.52,
    },
    'micro-bert': {
        'label': 'BERT', 'div': 'Cloud', 'params': '1.1M',
        'time_s': 45, 'target': ('acc', '>', 0.75),
        'final_train': 0.15, 'final_val': 0.58,
    },
    'micro-lstm': {
        'label': 'LSTM', 'div': 'Cloud', 'params': '54K',
        'time_s': 20, 'target': ('mse', '<', 0.13),
        'final_train': None, 'final_val': None,
    },
    'micro-rl': {
        'label': 'RL', 'div': 'Cloud', 'params': '18K',
        'time_s': 1, 'target': ('rew', '>', 195),
        'final_train': None, 'final_val': None,
    },
    # Edge Division
    'resnet18': {
        'label': 'ResNet-18', 'div': 'Edge', 'params': '11.2M',
        'time_s': 64, 'target': ('top1', '>', 0.36),
        'final_train': 1.82, 'final_val': 2.48,
    },
    'mobilenetv2': {
        'label': 'MobileV2', 'div': 'Edge', 'params': '2.4M',
        'time_s': 60, 'target': ('top1', '>', 0.40),
        'final_train': 2.10, 'final_val': 2.65,
    },
    # Tiny Division
    'dscnn': {
        'label': 'DS-CNN', 'div': 'Tiny', 'params': '20K',
        'time_s': 51, 'target': ('top1', '>', 0.90),
        'final_train': 0.20, 'final_val': 0.35,
    },
    'anomaly-ae': {
        'label': 'AE', 'div': 'Tiny', 'params': '0.3M',
        'time_s': 6, 'target': ('mse', '<', 0.04),
        'final_train': 0.009, 'final_val': 0.011,
    },
    'vww': {
        'label': 'VWW', 'div': 'Tiny', 'params': '8.5K',
        'time_s': 10, 'target': ('acc', '>', 0.85),
        'final_train': None, 'final_val': None,
    },
}


def load_curves():
    """Load actual training curves from training_data.json."""
    path = os.path.join(PAPER_DIR, "figures", "training_data.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def generate_convergence_summary():
    """Generate full-width 2-panel convergence figure.
    
    Panel (a): 4×4 grid of mini training curves grouped by division
    Panel (b): Final train/val bar chart + training time bars
    """
    curves = load_curves()
    
    # Merge curve data into workload data
    key_map = {
        'nanogpt': 'nanogpt', 'micro-lstm': 'micro-lstm',
        'micro-diffusion': 'micro-diffusion', 'vww': 'vww',
    }
    for json_key, wd_key in key_map.items():
        if json_key in curves and wd_key in WORKLOAD_DATA:
            c = curves[json_key]
            WORKLOAD_DATA[wd_key]['curve_train'] = c['train']
            WORKLOAD_DATA[wd_key]['curve_val'] = c['val']
            if WORKLOAD_DATA[wd_key]['final_train'] is None:
                WORKLOAD_DATA[wd_key]['final_train'] = c['train'][-1]
                WORKLOAD_DATA[wd_key]['final_val'] = c['val'][-1]
    
    # ---- Figure: 2-row layout ----
    # Top row: mini training curves (4 selected, showing variety)
    # Bottom row: bar charts (all workloads)
    
    fig = plt.figure(figsize=(7.0, 5.5))
    
    # Use GridSpec for clean layout
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35,
                          height_ratios=[1.0, 1.0])
    
    # === Top Row: 4 representative training curves ===
    curve_workloads = []
    for key in ['nanogpt', 'micro-lstm', 'micro-diffusion', 'vww']:
        if key in WORKLOAD_DATA and 'curve_train' in WORKLOAD_DATA.get(key, {}):
            curve_workloads.append(key)
    
    # Pad to 4 if needed
    while len(curve_workloads) < 4:
        curve_workloads.append(None)
    
    subtitles = {
        'nanogpt': '(a) NanoGPT — Cloud/LLM',
        'micro-lstm': '(b) LSTM — Cloud/TimeSer.',
        'micro-diffusion': '(c) Diffusion — Cloud/Gen.',
        'vww': '(d) VWW — Tiny/Vision',
    }
    
    for col, key in enumerate(curve_workloads[:4]):
        ax = fig.add_subplot(gs[0, col])
        if key and 'curve_train' in WORKLOAD_DATA[key]:
            wd = WORKLOAD_DATA[key]
            train = wd['curve_train']
            val = wd['curve_val']
            epochs = range(1, len(train) + 1)
            
            color = PALETTE.get(key, '#666')
            ax.plot(epochs, train, '-', color=color, linewidth=1.3, label='Train')
            ax.plot(epochs, val, '--', color=color, linewidth=1.3, alpha=0.65, label='Val')
            
            # Shade overfitting region
            train_arr = np.array(train)
            val_arr = np.array(val)
            if len(train_arr) == len(val_arr):
                gap = val_arr - train_arr
                overfit = gap > 0.15 * np.maximum(train_arr, 1e-8)
                if np.any(overfit):
                    ax.fill_between(range(1, len(train) + 1), train, val,
                                    where=overfit, alpha=0.08, color='red')
            
            ax.set_title(subtitles.get(key, key), fontsize=7.5, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='#999')
            ax.set_title(f'({chr(97+col)}) —', fontsize=7.5)
        
        ax.set_xlabel('Epoch', fontsize=7)
        if col == 0:
            ax.set_ylabel('Loss', fontsize=8)
        ax.legend(fontsize=6, framealpha=0.7, loc='upper right')
    
    # === Bottom Left: Final train vs val loss (bar chart) ===
    ax_bar = fig.add_subplot(gs[1, :2])
    
    # Sort by division then by name
    div_order = {'Cloud': 0, 'Edge': 1, 'Tiny': 2}
    sorted_keys = sorted(
        [k for k in WORKLOAD_DATA if WORKLOAD_DATA[k]['final_train'] is not None],
        key=lambda k: (div_order.get(WORKLOAD_DATA[k]['div'], 9), k)
    )
    
    labels = [WORKLOAD_DATA[k]['label'] for k in sorted_keys]
    train_vals = [WORKLOAD_DATA[k]['final_train'] for k in sorted_keys]
    val_vals = [WORKLOAD_DATA[k]['final_val'] for k in sorted_keys]
    colors = [PALETTE.get(k, '#666') for k in sorted_keys]
    divs = [WORKLOAD_DATA[k]['div'] for k in sorted_keys]
    
    x = np.arange(len(sorted_keys))
    w = 0.35
    
    ax_bar.bar(x - w/2, train_vals, w, color=colors, alpha=0.85, label='Train')
    ax_bar.bar(x + w/2, val_vals, w, color=colors, alpha=0.40,
               edgecolor=colors, linewidth=1.2, label='Val')
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=40, ha='right', fontsize=6.5)
    ax_bar.set_ylabel('Final Loss / MSE')
    ax_bar.set_title('(e) Final Train vs. Val Loss', fontweight='bold', fontsize=9)
    ax_bar.legend(fontsize=6.5, loc='upper right')
    
    # Add division separators
    prev_div = None
    for i, d in enumerate(divs):
        if prev_div and d != prev_div:
            ax_bar.axvline(x=i - 0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        prev_div = d
    
    # === Bottom Right: Training time (horizontal bars) ===
    ax_time = fig.add_subplot(gs[1, 2:])
    
    time_keys = sorted(WORKLOAD_DATA.keys(),
                       key=lambda k: WORKLOAD_DATA[k]['time_s'])
    time_labels = [WORKLOAD_DATA[k]['label'] for k in time_keys]
    times = [WORKLOAD_DATA[k]['time_s'] for k in time_keys]
    time_colors = [PALETTE.get(k, '#666') for k in time_keys]
    time_divs = [WORKLOAD_DATA[k]['div'] for k in time_keys]
    
    y = np.arange(len(time_keys))
    bars = ax_time.barh(y, times, color=time_colors, alpha=0.8, height=0.7)
    
    ax_time.set_yticks(y)
    ax_time.set_yticklabels(time_labels, fontsize=6.5)
    ax_time.set_xlabel('Time (seconds)')
    ax_time.set_title('(f) Training Time (Apple M1 MPS)', fontweight='bold', fontsize=9)
    
    # Add time labels
    for bar, t in zip(bars, times):
        ax_time.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                     f'{t}s', va='center', fontsize=6)
    
    # Add total time annotation
    total = sum(times)
    ax_time.text(0.95, 0.02, f'Total: {total}s ({total/60:.1f} min)',
                 transform=ax_time.transAxes, ha='right', va='bottom',
                 fontsize=7, fontstyle='italic', color='#666')
    
    # Add division color indicators as a legend
    div_patches = [
        mpatches.Patch(color='#2563EB', alpha=0.5, label='Cloud'),
        mpatches.Patch(color='#16A34A', alpha=0.5, label='Edge'),
        mpatches.Patch(color='#4F46E5', alpha=0.5, label='Tiny'),
    ]
    ax_time.legend(handles=div_patches, fontsize=6, loc='lower right',
                   ncol=3, framealpha=0.7)
    
    # Save
    out_pdf = os.path.join(FIG_DIR, "convergence_summary.pdf")
    out_png = os.path.join(FIG_DIR, "convergence_summary.png")
    plt.savefig(out_pdf, pad_inches=0.1)
    plt.savefig(out_png, pad_inches=0.1)
    plt.close()
    print(f"  ✅ Saved {out_pdf}")
    print(f"  ✅ Saved {out_png}")


if __name__ == '__main__':
    print("📊 Generating Global Convergence Summary (Figure 3)...")
    generate_convergence_summary()
    print("✅ Done")
