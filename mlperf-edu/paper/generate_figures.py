#!/usr/bin/env python3
"""
Generate publication-quality figures for the MLPerf EDU paper.
Uses matplotlib to create training curves and architecture diagrams.
Outputs to paper/figures/ as PDF for LaTeX inclusion.
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
os.makedirs(FIG_DIR, exist_ok=True)

# Publication styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Color palette (professional, colorblind-friendly)
COLORS = {
    'nanogpt-12m': '#2196F3',
    'nano-moe-12m': '#FF9800',
    'resnet18': '#4CAF50',
    'micro-dlrm-1m': '#E91E63',
    'micro-diffusion-32px': '#9C27B0',
    'anomaly-ae': '#00BCD4',
}

LABELS = {
    'nanogpt-12m': 'NanoGPT (30.3M)',
    'nano-moe-12m': 'Nano-MoE (17.4M)',
    'resnet18': 'ResNet-18 (11.2M)',
    'micro-dlrm-1m': 'Micro-DLRM (0.6M)',
    'micro-diffusion-32px': 'Micro-Diffusion (2.0M)',
    'anomaly-ae': 'Anomaly AE (0.3M)',
}

def load_data():
    data_path = os.path.join(PAPER_DIR, "training_data.json")
    if os.path.exists(data_path):
        with open(data_path) as f:
            return json.load(f)
    return None


def fig1_training_curves(data):
    """2x3 grid of training/validation loss curves for all workloads."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5))
    axes = axes.flatten()

    models = ['nanogpt-12m', 'nano-moe-12m', 'resnet18',
              'micro-dlrm-1m', 'micro-diffusion-32px', 'anomaly-ae']
    titles = ['(a) NanoGPT — Cloud/LLM', '(b) Nano-MoE — Cloud/MoE',
              '(c) ResNet-18 — Edge/Vision',
              '(d) Micro-DLRM — Cloud/Rec', '(e) Micro-Diffusion — Cloud/Gen',
              '(f) Anomaly AE — Tiny/AD']

    for i, (name, title) in enumerate(zip(models, titles)):
        ax = axes[i]
        if name in data:
            d = data[name]
            epochs = range(len(d['train_losses']))
            ax.plot(epochs, d['train_losses'], '-', color=COLORS[name],
                    linewidth=1.5, label='Train')
            ax.plot(epochs, d['val_losses'], '--', color=COLORS[name],
                    linewidth=1.5, alpha=0.7, label='Val')

            # Shade overfitting region
            train = np.array(d['train_losses'])
            val = np.array(d['val_losses'])
            gap = val - train
            overfit_mask = gap > 0.5 * train
            if np.any(overfit_mask):
                ax.fill_between(range(len(train)), train, val,
                                where=overfit_mask, alpha=0.1, color='red')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')

        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_xlabel('Epoch')
        if i % 3 == 0:
            ax.set_ylabel('Loss')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

    plt.tight_layout(pad=0.5)
    out = os.path.join(FIG_DIR, "training_curves.pdf")
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


def fig2_convergence_summary(data):
    """Bar chart: final train vs val loss for all workloads."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    models = [m for m in ['nanogpt-12m', 'nano-moe-12m', 'resnet18',
               'micro-dlrm-1m', 'micro-diffusion-32px', 'anomaly-ae']
              if m in data]
    labels_short = ['NanoGPT', 'Nano-MoE', 'ResNet-18', 'DLRM', 'Diffusion', 'AE']

    # Panel 1: Final losses
    train_finals = [data[m]['train_losses'][-1] for m in models]
    val_finals = [data[m]['val_losses'][-1] for m in models]
    x = np.arange(len(models))
    w = 0.35
    bars1 = ax1.bar(x - w/2, train_finals, w, label='Train Loss',
                    color=[COLORS[m] for m in models], alpha=0.8)
    bars2 = ax1.bar(x + w/2, val_finals, w, label='Val Loss',
                    color=[COLORS[m] for m in models], alpha=0.4,
                    edgecolor=[COLORS[m] for m in models], linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_short[:len(models)], rotation=30, ha='right', fontsize=7)
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Final Loss (Train vs Val)', fontweight='bold', fontsize=9)
    ax1.legend(fontsize=7)

    # Panel 2: Training time
    times = [data[m]['total_time'] for m in models]
    bars = ax2.barh(x, times, color=[COLORS[m] for m in models], alpha=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels_short[:len(models)], fontsize=7)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('(b) Training Time (Apple M1 MPS)', fontweight='bold', fontsize=9)
    for bar, t in zip(bars, times):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{t:.0f}s', va='center', fontsize=7)

    plt.tight_layout(pad=0.5)
    out = os.path.join(FIG_DIR, "convergence_summary.pdf")
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


def fig3_model_size_comparison():
    """Model size comparison: MLPerf Official vs MLPerf EDU."""
    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    official = {
        'GPT-3': 175_000, 'LLaMA-2': 70_000, 'ResNet-50': 25.6,
        'DLRM': 540, 'Stable Diff.': 860, 'DS-CNN': 0.025
    }
    edu = {
        'NanoGPT': 30.3, 'Nano-MoE': 17.4, 'ResNet-18': 11.2,
        'Micro-DLRM': 0.6, 'Micro-Diff.': 2.0, 'DS-CNN': 0.02
    }

    labels = list(official.keys())
    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w/2, list(official.values()), w, label='MLPerf Official',
           color='#78909C', alpha=0.8)
    ax.bar(x + w/2, list(edu.values()), w, label='MLPerf EDU',
           color='#2196F3', alpha=0.8)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('MLPerf Official vs. EDU Scale', fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "model_scale.pdf")
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


def fig4_dataset_provenance():
    """Dataset provenance and flow diagram (simplified as table-style figure)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.axis('off')

    rows = [
        ['TinyShakespeare', 'Karpathy 2015', '1.1 MB', 'In repo', 'NanoGPT/MoE/Agents'],
        ['CIFAR-100', 'Krizhevsky 2009', '170 MB', 'torchvision', 'ResNet-18'],
        ['CIFAR-10', 'Krizhevsky 2009', '170 MB', 'torchvision', 'Diffusion'],
        ['Speech Cmds', 'Warden 2018', '2 GB', 'torchaudio', 'DS-CNN'],
        ['MNIST', 'LeCun 1998', '12 MB', 'torchvision', 'Anomaly AE'],
    ]
    cols = ['Dataset', 'Source', 'Size', 'Loader', 'Workloads']

    table = ax.table(cellText=rows, colLabels=cols, loc='center',
                     cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor('#2196F3')
        cell.set_text_props(color='white', fontweight='bold', fontsize=7)

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            if i % 2 == 0:
                table[i, j].set_facecolor('#E3F2FD')

    ax.set_title('Dataset Provenance Registry', fontweight='bold', fontsize=9, pad=10)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "dataset_provenance.pdf")
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("📊 Generating MLPerf EDU paper figures...")

    data = load_data()
    if data:
        print(f"  Loaded training data for {len(data)} models")
        fig1_training_curves(data)
        fig2_convergence_summary(data)
    else:
        print("  ⚠️  No training_data.json found — skipping curve plots")

    fig3_model_size_comparison()
    fig4_dataset_provenance()
    print("✅ All figures generated in paper/figures/")
