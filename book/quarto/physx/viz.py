# viz.py
# Centralized Visualization Style for MLSys Book
# Ensures all generated figures across Vol 1 & 2 share a consistent,
# MIT Press-ready aesthetic.

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# --- Brand & Book Palette ---
COLORS = {
    "crimson":    "#A51C30", # Harvard Crimson
    "primary":    "#333333", # Dark Gray (Text)
    "grid":       "#CCCCCC", # Light Gray
    
    # TikZ / Technical Palette (The "Golden" set)
    "BlueLine":   "#006395", "BlueL":   "#D1E6F3", "BlueFill": "#D6EAF8",
    "RedLine":    "#CB202D", "RedL":    "#F5D2D5", "RedFill": "#F2D7D5",
    "GreenLine":  "#008F45", "GreenL":  "#D4EFDF", "GreenFill": "#D5F5E3",
    "OrangeLine": "#E67817", "OrangeL": "#FCE4CC",
    "VioletLine": "#7E317B", "VioletL": "#E6D4E5",
    "BrownLine":  "#78492A", "BrownL":  "#E3D3C8",
    "YellowFill": "#FEF9E0"
}

def set_book_style():
    """Applies the global matplotlib style configuration."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'Inter', 'DejaVu Sans'],
        'font.size': 10,
        'text.color': COLORS['primary'],
        'axes.labelsize': 11,
        'axes.labelcolor': COLORS['primary'],
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.edgecolor': COLORS['primary'],
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.color': COLORS['primary'],
        'ytick.color': COLORS['primary'],
        'axes.grid': True,
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'legend.title_fontsize': 10,
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'figure.figsize': (8, 5),
        'figure.autolayout': True
    })

# --- Lightweight helpers ---

def setup_plot(figsize=None):
    """
    One-line plot setup for QMD blocks.
    Returns (fig, ax, COLORS, plt) after applying book style.
    The plt is returned so code blocks don't need separate matplotlib import.
    """
    set_book_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, COLORS, plt

# --- Data Dictionaries (Single Source of Truth) ---

MODELS_DATA = [
    {'Model': 'AlexNet', 'Year': 2012, 'GFLOPs': 0.7, 'Accuracy': 63.3, 'Family': 'Legacy', 'Source': 'Krizhevsky et al. 2012'},
    {'Model': 'VGG-16', 'Year': 2014, 'GFLOPs': 15.5, 'Accuracy': 71.5, 'Family': 'Legacy', 'Source': 'Simonyan & Zisserman 2015'},
    {'Model': 'ResNet-50', 'Year': 2015, 'GFLOPs': 4.1, 'Accuracy': 76.0, 'Family': 'Dense CNN', 'Source': 'He et al. 2016'},
    {'Model': 'Inception-v3', 'Year': 2015, 'GFLOPs': 5.7, 'Accuracy': 78.8, 'Family': 'Dense CNN', 'Source': 'Szegedy et al. 2016'},
    {'Model': 'MobileNet-v2', 'Year': 2018, 'GFLOPs': 0.3, 'Accuracy': 72.0, 'Family': 'Mobile', 'Source': 'Sandler et al. 2018'},
    {'Model': 'EfficientNet-B0', 'Year': 2019, 'GFLOPs': 0.39, 'Accuracy': 77.1, 'Family': 'Efficient', 'Source': 'Tan & Le 2019'},
    {'Model': 'EfficientNet-B7', 'Year': 2019, 'GFLOPs': 37.0, 'Accuracy': 84.3, 'Family': 'Efficient', 'Source': 'Tan & Le 2019'},
    {'Model': 'ViT-B/16', 'Year': 2020, 'GFLOPs': 17.6, 'Accuracy': 77.9, 'Family': 'Transformer', 'Source': 'Dosovitskiy et al. 2021'},
    {'Model': 'Swin-T', 'Year': 2021, 'GFLOPs': 4.5, 'Accuracy': 81.3, 'Family': 'Transformer', 'Source': 'Liu et al. 2021'},
    {'Model': 'ConvNeXt-T', 'Year': 2022, 'GFLOPs': 4.5, 'Accuracy': 82.1, 'Family': 'Modern CNN', 'Source': 'Liu et al. 2022'}
]

TRAINING_COMPUTE_DATA = [
    {'Model': 'AlexNet', 'Year': 2012, 'FLOPs': 1.2e18, 'Era': 'Deep Learning'},
    {'Model': 'VGG-16', 'Year': 2014, 'FLOPs': 3.0e19, 'Era': 'Deep Learning'},
    {'Model': 'ResNet-50', 'Year': 2015, 'FLOPs': 5.0e19, 'Era': 'Deep Learning'},
    {'Model': 'GoogleNet', 'Year': 2014, 'FLOPs': 2.0e19, 'Era': 'Deep Learning'},
    {'Model': 'AlphaGoZero', 'Year': 2017, 'FLOPs': 3.0e22, 'Era': 'Deep Learning'},
    {'Model': 'Transformer', 'Year': 2017, 'FLOPs': 5.0e20, 'Era': 'Large Scale'},
    {'Model': 'BERT-Large', 'Year': 2018, 'FLOPs': 3.0e21, 'Era': 'Large Scale'},
    {'Model': 'GPT-2-XL', 'Year': 2019, 'FLOPs': 2.0e22, 'Era': 'Large Scale'},
    {'Model': 'T5-11B', 'Year': 2019, 'FLOPs': 5.0e22, 'Era': 'Large Scale'},
    {'Model': 'GPT-3', 'Year': 2020, 'FLOPs': 3.1e23, 'Era': 'Large Scale'},
    {'Model': 'Gopher', 'Year': 2021, 'FLOPs': 6.0e23, 'Era': 'Large Scale'},
    {'Model': 'PaLM', 'Year': 2022, 'FLOPs': 2.5e24, 'Era': 'Large Scale'},
    {'Model': 'GPT-4', 'Year': 2023, 'FLOPs': 2.0e25, 'Era': 'Large Scale'},
    {'Model': 'Llama-3-70B', 'Year': 2024, 'FLOPs': 8.0e24, 'Era': 'Large Scale'},
    {'Model': 'Gemini-Ultra', 'Year': 2024, 'FLOPs': 5.0e25, 'Era': 'Large Scale'}
]

ALGO_EFFICIENCY_DATA = [
    {'Model': 'AlexNet', 'Year': 2012.5, 'Efficiency_Factor': 1.0, 'Family': 'Baseline'},
    {'Model': 'VGG-11', 'Year': 2014.7, 'Efficiency_Factor': 0.8, 'Family': 'Legacy'},
    {'Model': 'GoogLeNet', 'Year': 2014.75, 'Efficiency_Factor': 4.5, 'Family': 'Inception'},
    {'Model': 'ResNet-18', 'Year': 2015.9, 'Efficiency_Factor': 2.9, 'Family': 'ResNet'},
    {'Model': 'DenseNet-121', 'Year': 2016.7, 'Efficiency_Factor': 3.3, 'Family': 'DenseNet'},
    {'Model': 'MobileNet-v1', 'Year': 2017.3, 'Efficiency_Factor': 11.2, 'Family': 'Mobile'},
    {'Model': 'ShuffleNet-v1', 'Year': 2017.5, 'Efficiency_Factor': 20.8, 'Family': 'Mobile'},
    {'Model': 'MobileNet-v2', 'Year': 2018.1, 'Efficiency_Factor': 13.3, 'Family': 'Mobile'},
    {'Model': 'ShuffleNet-v2', 'Year': 2018.5, 'Efficiency_Factor': 24.9, 'Family': 'Mobile'},
    {'Model': 'EfficientNet-B0', 'Year': 2019.4, 'Efficiency_Factor': 44.5, 'Family': 'Efficient'}
]

ENERGY_DATA = [
    {'Operation': 'INT8 Add', 'Energy_pJ': 0.03, 'Source': 'Horowitz 2014'},
    {'Operation': 'FP32 Add', 'Energy_pJ': 0.9, 'Source': 'Horowitz 2014'},
    {'Operation': 'FP32 Mult', 'Energy_pJ': 3.7, 'Source': 'Horowitz 2014'},
    {'Operation': 'SRAM Read (8KB)', 'Energy_pJ': 5.0, 'Source': 'Horowitz 2014'},
    {'Operation': 'DRAM Read', 'Energy_pJ': 640.0, 'Source': 'Horowitz 2014'}
]

QUANTIZATION_DATA = {
    'Bits': [32, 16, 8, 4, 3, 2],
    'ResNet50_Acc': [76.1, 76.1, 76.0, 74.5, 55.0, 10.0],
    'BERT_Acc': [84.0, 84.0, 83.5, 78.0, 40.0, 10.0]
}

BATCHING_DATA = [
    {'BatchSize': 1, 'Throughput': 64, 'Latency': 15.6},
    {'BatchSize': 2, 'Throughput': 120, 'Latency': 16.5},
    {'BatchSize': 4, 'Throughput': 230, 'Latency': 17.4},
    {'BatchSize': 8, 'Throughput': 404, 'Latency': 19.8},
    {'BatchSize': 16, 'Throughput': 650, 'Latency': 24.6},
    {'BatchSize': 32, 'Throughput': 935, 'Latency': 34.2},
    {'BatchSize': 64, 'Throughput': 1100, 'Latency': 60.0},
    {'BatchSize': 128, 'Throughput': 1143, 'Latency': 136.8},
    {'BatchSize': 256, 'Throughput': 1150, 'Latency': 300.0}
]

# --- Golden Plot Generators ---

def plot_efficiency_frontier(ax=None):
    if ax is None: fig, ax = plt.subplots()
    df = pd.DataFrame(MODELS_DATA)
    
    color_map = {
        'Legacy': COLORS['grid'], 'Dense CNN': COLORS['BlueLine'],
        'Mobile': COLORS['GreenLine'], 'Efficient': COLORS['VioletLine'],
        'Transformer': COLORS['RedLine'], 'Modern CNN': COLORS['OrangeLine']
    }
    
    for family, group in df.groupby('Family'):
        ax.scatter(group['GFLOPs'], group['Accuracy'], label=family, 
                   c=color_map.get(family, COLORS['primary']), s=120, alpha=0.9, 
                   edgecolors='white', linewidth=1.2, zorder=3)
        
    offsets = {
        'MobileNet-v2': (5, -15),
        'ResNet-50': (-45, 0),
        'Swin-T': (-25, -10),
        'ConvNeXt-T': (10, 10),
        'EfficientNet-B0': (10, -10)
    }
    
    for _, row in df.iterrows():
        xytext = offsets.get(row['Model'], (5, 5))
        ax.annotate(row['Model'], (row['GFLOPs'], row['Accuracy']), 
                    xytext=xytext, textcoords='offset points', fontsize=8)

    pareto_models = ['MobileNet-v2', 'EfficientNet-B0', 'Swin-T', 'ConvNeXt-T', 'EfficientNet-B7']
    pareto_points = df[df['Model'].isin(pareto_models)].sort_values('GFLOPs')
    ax.plot(pareto_points['GFLOPs'], pareto_points['Accuracy'], '--', color=COLORS['grid'], linewidth=1.5, zorder=1)
    
    ax.set_xscale('log')
    ax.set_xlabel('Compute Cost (GFLOPs per Image)')
    ax.set_ylabel('ImageNet Top-1 Accuracy (%)')
    ax.legend(title='Family', loc='lower right', fontsize=8)
    return ax

def plot_iron_law_heatmap(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    S_vals = np.logspace(0, 3, 100)
    P_vals = np.linspace(0.8, 0.999, 100)
    S_grid, P_grid = np.meshgrid(S_vals, P_vals)
    Speedup = 1 / ((1 - P_grid) + (P_grid / S_grid))
    
    levels = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    norm = mcolors.LogNorm(vmin=1, vmax=500)
    
    cf = ax.contourf(S_grid, P_grid, Speedup, levels=levels, cmap='RdYlBu_r', norm=norm, alpha=0.8)
    cs = ax.contour(S_grid, P_grid, Speedup, levels=levels, colors='white', linewidths=0.8, alpha=0.6)
    ax.clabel(cs, inline=1, fontsize=8, fmt='%gx', colors='black')
    
    ax.set_xscale('log')
    ax.set_xlabel('Accelerator Raw Speedup (S)')
    ax.set_ylabel('Parallelizable Fraction (P)')
    
    ax.text(100, 0.98, "Compute Bound", color='black', ha='center', va='top', fontweight='bold', fontsize=9)
    ax.text(100, 0.82, "Serial Bound", color='white', ha='center', va='bottom', fontweight='bold', fontsize=9)
    return ax, cf

def plot_systems_gap(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    years = np.linspace(2012, 2024.5, 100)
    
    # 1. Moore's Law (CPU Baseline)
    # 2012 (Xeon E5-2690): ~0.37 TF -> 2022 (Xeon 8480+): ~7 TF. Growth ~19x in 10y.
    cpu_slope = np.log10(19) / 10 
    moore = 1.0 * 10**(cpu_slope * (years - 2012))
    
    # 2. Huang's Law (GPU Peak)
    # 2012 (K20X): 3.95 TF -> 2022 (H100): 989 TF. Growth ~250x in 10y.
    gpu_slope = np.log10(250) / 10
    huang = 1.0 * 10**(gpu_slope * (years - 2012))
    
    # 3. Model Demand
    # 2012 (AlexNet): 4.3e16 -> 2023 (GPT-4): 2e25. Growth ~4.6e8x in 11y.
    demand_slope = np.log10(4.6e8) / 11
    demand = 1.0 * 10**(demand_slope * (years - 2012))
    
    ax.plot(years, moore, ':', color=COLORS['grid'], label="CPU Performance Trend", linewidth=2)
    ax.plot(years, huang, '--', color=COLORS['BlueLine'], label="GPU Peak (Huang's Law)", linewidth=2.5)
    ax.plot(years, demand, '-', color=COLORS['RedLine'], label="Model Demand (Scaling Laws)", linewidth=3)
    
    ax.fill_between(years, huang, demand, where=(demand > huang), 
                    color=COLORS['VioletL'], alpha=0.3)
    
    ax.set_yscale('log')
    ax.set_xlabel('Year')
    ax.set_ylabel('Relative Growth (2012 = 1.0)')
    ax.set_xlim(2012, 2024.5)
    ax.set_ylim(0.5, 1e10) 
    
    gap_x = 2020.0
    h_val = 10**(gpu_slope * (gap_x - 2012))
    d_val = 10**(demand_slope * (gap_x - 2012))
    gap_y = np.sqrt(h_val * d_val)
    
    ax.text(gap_x, gap_y, "THE SYSTEMS GAP\n(Closed by Parallelism,\nArchitecture & Co-design)", 
            ha='center', va='center', fontweight='bold', color=COLORS['VioletLine'], fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
    points = [
        (2012, 1.0, "AlexNet"),
        (2015, 10**(demand_slope*3), "ResNet"),
        (2017, 10**(demand_slope*5), "Transformer"),
        (2020, 10**(demand_slope*8), "GPT-3"),
        (2023, 10**(demand_slope*11), "GPT-4")
    ]
    
    model_offsets = {
        "AlexNet": (0, 10),
        "Transformer": (-15, 10),
        "GPT-3": (-15, 8),
        "GPT-4": (0, 8)
    }

    for y, v, l in points:
        ax.scatter(y, v, color=COLORS['RedLine'], s=25, zorder=5, edgecolors='white')
        xytext = model_offsets.get(l, (0, 8))
        ax.annotate(l, (y, v), xytext=xytext, textcoords='offset points', 
                    fontsize=8, ha='center', color=COLORS['RedLine'], fontweight='bold')

    hw_points = [
        (2012, 1.0, "K20X"),
        (2016, 10**(gpu_slope*4), "P100"),
        (2022, 10**(gpu_slope*10), "H100")
    ]
    
    hw_offsets = {
        "K20X": (0, -15),
        "P100": (0, -15),
        "H100": (0, -15)
    }
    
    for y, v, l in hw_points:
        ax.scatter(y, v, color=COLORS['BlueLine'], s=25, zorder=5, edgecolors='white')
        xytext = hw_offsets.get(l, (0, -12))
        ax.annotate(l, (y, v), xytext=xytext, textcoords='offset points',
                    fontsize=8, ha='center', color=COLORS['BlueLine'], fontweight='bold')

    ax.legend(loc='lower right', fontsize=8)
    return ax

def plot_scaling_tax(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    N = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    scenarios = [
        {'r': 0.0, 'name': 'Ideal Linear', 'color': 'black', 'style': '--', 'marker': ''},
        {'r': 0.05, 'name': 'Compute Bound (ResNet)', 'color': COLORS['GreenLine'], 'style': '-', 'marker': 'o'},
        {'r': 0.20, 'name': 'Balanced (LLM + NVLink)', 'color': COLORS['OrangeLine'], 'style': '-', 'marker': 's'}, 
        {'r': 0.50, 'name': 'Bandwidth Bound', 'color': COLORS['RedLine'], 'style': '-', 'marker': '^'}
    ]
    
    for sc in scenarios:
        speedup = N / (1 + (N - 1) * sc['r'])
        ax.plot(N, speedup, linestyle=sc['style'], color=sc['color'], 
                marker=sc['marker'], label=sc['name'], linewidth=2, markersize=5)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(N); ax.set_xticklabels(N)
    ax.set_yticks(N); ax.set_yticklabels(N)
    ax.set_xlabel('Number of GPUs (N)')
    ax.set_ylabel('Effective Speedup')
    ax.legend(loc='upper left', fontsize=8)
    
    ax.annotate("Communication Wall", xy=(32, 32/(1+31*0.5)), xytext=(64, 4),
                arrowprops=dict(facecolor=COLORS['primary'], arrowstyle='->', lw=1.5), fontsize=9)
    return ax

def plot_quantization_free_lunch(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    bits = np.array(QUANTIZATION_DATA['Bits'])
    acc_cnn = np.array(QUANTIZATION_DATA['ResNet50_Acc'])
    acc_trans = np.array(QUANTIZATION_DATA['BERT_Acc'])

    ax.plot(bits, acc_cnn, 'o-', color=COLORS['BlueLine'], label='CNN (ResNet-50)', markersize=5)
    ax.plot(bits, acc_trans, 's-', color=COLORS['RedLine'], label='Transformer (BERT)', markersize=5)

    ax.invert_xaxis()
    ax.set_xlabel('Precision (Bits)')
    ax.set_ylabel('Model Accuracy (%)')
    ax.set_xticks(bits)
    ax.set_xticklabels(['FP32', 'FP16', 'INT8', 'INT4', 'INT3', 'INT2'])

    ax.axvspan(33, 7, color=COLORS['GreenL'], alpha=0.3)
    ax.text(20, 50, "Free Lunch Zone\n(<1% Loss)", color=COLORS['GreenLine'], fontweight='bold', ha='center', fontsize=9)
    ax.axvspan(5, 1, color=COLORS['RedL'], alpha=0.3)
    ax.text(3.5, 30, "The Cliff", color=COLORS['RedLine'], fontweight='bold', ha='center', fontsize=9)
    ax.legend(fontsize=8)
    return ax

def plot_throughput_latency_knee(ax1=None):
    if ax1 is None: fig, ax1 = plt.subplots()
    
    df = pd.DataFrame(BATCHING_DATA)
    batch_sizes = df['BatchSize'].values
    throughput = df['Throughput'].values
    latency = df['Latency'].values

    color_tp = COLORS['BlueLine']
    color_lat = COLORS['OrangeLine']

    ax1.plot(batch_sizes, throughput, 'o-', color=color_tp, label='Throughput')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (Req/Sec)', color=color_tp, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_tp)
    ax1.set_xscale('log', base=2)

    ax2 = ax1.twinx()
    ax2.plot(batch_sizes, latency, 's-', color=color_lat, label='Latency')
    ax2.set_ylabel('Latency (ms)', color=color_lat, fontweight='bold', rotation=270, labelpad=15)
    ax2.tick_params(axis='y', labelcolor=color_lat)
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)

    optimal_idx = 5 
    ax1.axvline(batch_sizes[optimal_idx], color='gray', linestyle='--', alpha=0.5)
    ax1.text(batch_sizes[optimal_idx], 200, " Optimal\n Point", ha='right', color='gray', fontsize=9)
    return ax1

def plot_active_learning_multiplier(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    samples = np.logspace(2, 4, 100)
    acc_random = 50 + 40 * np.log10(samples/100 + 1) / np.log10(101)
    acc_active = 50 + 45 * (1 - np.exp(-samples/1000))
    acc_active = np.minimum(acc_active, 95)
    acc_random = np.minimum(acc_random, 95)

    ax.plot(samples, acc_random, '--', color=COLORS['grid'], label='Random Sampling', linewidth=2)
    ax.plot(samples, acc_active, '-', color=COLORS['GreenLine'], label='Active Learning', linewidth=2.5)
    ax.fill_between(samples, acc_random, acc_active, color=COLORS['GreenL'], alpha=0.3)
    
    ax.set_xscale('log')
    ax.set_xlabel('Labeled Samples')
    ax.set_ylabel('Accuracy (%)')
    ax.annotate("", xy=(2500, 90), xytext=(9000, 90), arrowprops=dict(arrowstyle="->", color='black'))
    ax.text(5000, 91, "4x Data Efficiency", ha='center', fontsize=9)
    ax.legend(loc='lower right', fontsize=8)
    return ax

def plot_dataloader_choke_point(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    workers = np.arange(1, 17)
    cpu_throughput = np.minimum(250 * workers, 3500)
    gpu_limit = 3000
    
    ax.plot(workers, cpu_throughput, 'o-', color=COLORS['BlueLine'], label='DataLoader (CPU)')
    ax.axhline(gpu_limit, linestyle='--', color=COLORS['RedLine'], label='GPU Capacity')
    ax.fill_between(workers, 0, np.minimum(cpu_throughput, gpu_limit), color=COLORS['BlueL'], alpha=0.2)
    
    ax.text(4, 1500, "Starvation", color=COLORS['BlueLine'], fontweight='bold', ha='center')
    ax.text(12, 3200, "Saturated", color=COLORS['RedLine'], fontweight='bold', ha='center')
    ax.set_xlabel('DataLoader Workers')
    ax.set_ylabel('Throughput (img/s)')
    ax.legend(loc='lower right', fontsize=8)
    return ax

def plot_rotting_asset_curve(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    days = np.arange(0, 30, 0.1)
    decay = 95 * np.exp(-0.01 * days)
    sawtooth = decay.copy()
    for i in range(len(days)):
        if int(days[i]) % 7 == 0 and int(days[i]) > 0:
            sawtooth[i:] = sawtooth[i:] + (95 - sawtooth[i])
    triggered = np.maximum(decay, 92)

    ax.plot(days, decay, 'k--', alpha=0.3, label='Natural Decay')
    ax.plot(days, sawtooth, '-', color=COLORS['BlueLine'], label='Scheduled')
    ax.plot(days, triggered, '-', color=COLORS['GreenLine'], label='Triggered')
    
    ax.set_ylim(70, 100)
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='lower left', fontsize=8)
    return ax

def plot_linear_scaling_failure(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    steps = np.arange(0, 1000)
    loss_base = 2.0 * np.exp(-0.01 * steps) + 0.2
    loss_large_naive = 2.0 * np.exp(-0.00125 * steps) + 0.2
    loss_large_scaled = 2.0 * np.exp(-0.009 * steps) + 0.25

    ax.plot(steps, loss_base, '-', color=COLORS['BlueLine'], label='Batch 32')
    ax.plot(steps, loss_large_naive, '--', color=COLORS['grid'], label='Batch 256 (Fixed LR)')
    ax.plot(steps, loss_large_scaled, '-', color=COLORS['GreenLine'], label='Batch 256 (Scaled LR)')

    ax.annotate("Generalization Gap", xy=(800, 0.5), xytext=(800, 1.5), 
                arrowprops=dict(arrowstyle="<->", color=COLORS['RedLine']))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    return ax

def plot_energy_hierarchy(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    df = pd.DataFrame(ENERGY_DATA)
    key_ops = ['INT8 Add', 'FP32 Add', 'FP32 Mult', 'SRAM Read (8KB)', 'DRAM Read']
    df = df[df['Operation'].isin(key_ops)]
    df = df.set_index('Operation').reindex(key_ops).reset_index()
    
    ops = df['Operation'].values
    energy = df['Energy_pJ'].values
    
    colors = [COLORS['GreenLine'], COLORS['BlueLine'], COLORS['BlueLine'], COLORS['OrangeLine'], COLORS['RedLine']]
    
    y_pos = np.arange(len(ops))
    
    ax.barh(y_pos, energy, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ops)
    ax.set_xscale('log')
    ax.set_xlabel('Energy per Operation (picojoules) [Log Scale]')
    
    for i, v in enumerate(energy):
        ax.text(v * 1.1, i, f"{v} pJ", va='center', fontsize=9, fontweight='bold', color=COLORS['primary'])
    
    ax.annotate("", xy=(640, 3), xytext=(5, 3), 
                arrowprops=dict(arrowstyle="->", color=COLORS['RedLine'], lw=1.5))
    ax.text(50, 3.2, "~128x Cost\n(The Memory Wall)", color=COLORS['RedLine'], ha='center', fontsize=9)
    return ax

def plot_python_tax(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    t_dispatch = 10
    t_compute = 1
    n_ops = 5
    
    y_eager = 1
    for i in range(n_ops):
        start = i * (t_dispatch + t_compute)
        ax.barh(y_eager, t_dispatch, left=start, height=0.6, color=COLORS['RedLine'], alpha=0.6, label='Python Overhead' if i==0 else "")
        ax.barh(y_eager, t_compute, left=start+t_dispatch, height=0.6, color=COLORS['BlueLine'], alpha=0.8, label='GPU Kernel' if i==0 else "")
    
    y_compiled = 0
    t_fused_compute = t_compute * n_ops * 0.8
    t_compiled_dispatch = 5
    
    ax.barh(y_compiled, t_compiled_dispatch, left=0, height=0.6, color=COLORS['RedLine'], alpha=0.6)
    ax.barh(y_compiled, t_fused_compute, left=t_compiled_dispatch, height=0.6, color=COLORS['BlueLine'], alpha=0.8)
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Compiled / Fused', 'Eager Execution'])
    ax.set_xlabel('Execution Time (microseconds)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8)
    
    # Move text to middle to act as annotation, not title
    ax.text(30, 0.5, "Gap = The Python Tax", color=COLORS['RedLine'], ha='center', va='center', fontsize=9, fontweight='bold')
    return ax

def plot_compute_growth(ax=None):
    """Visualizes 'The Era of Scale' (Training Compute Growth)."""
    if ax is None: fig, ax = plt.subplots()
    
    df = pd.DataFrame(TRAINING_COMPUTE_DATA)

    for era, group in df.groupby('Era'):
        color = COLORS['BlueLine'] if era == 'Deep Learning' else COLORS['RedLine']
        ax.scatter(group['Year'], group['FLOPs'], color=color, s=80, alpha=0.8, edgecolors='white', label=era)
        
    for _, row in df.iterrows():
        if row['Model'] in ['AlexNet', 'AlphaGoZero', 'GPT-3', 'PaLM', 'GPT-4']:
            xytext = (0, 5)
            if row['Model'] == 'PaLM': xytext = (-5, 5)
            if row['Model'] == 'GPT-4': xytext = (5, 5)
            ax.annotate(row['Model'], (row['Year'], row['FLOPs']), 
                        xytext=xytext, textcoords='offset points', fontsize=8)

    ax.set_yscale('log')
    ax.set_xlabel('Year')
    ax.set_ylabel('Training Compute (FLOPs)')
    
    years = np.linspace(2012, 2024, 100)
    trend = 1e18 * 10**(7/12 * (years - 2012))
    ax.plot(years, trend, '--', color=COLORS['grid'], label='Trend (~6mo Doubling)')
    
    ax.legend(loc='lower right', fontsize=8)
    return ax

def plot_fairness_frontier(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    disparity = np.linspace(0.0, 0.20, 100)
    accuracy = 0.85 + 0.10 * (1 - np.exp(-20 * disparity))
    
    ax.plot(disparity, accuracy, color=COLORS['primary'], linewidth=2, linestyle='--')
    
    ax.plot(0.18, 0.947, 'o', color=COLORS['RedLine'], markersize=8)
    ax.text(0.18, 0.93, "Unconstrained\n(Max Accuracy)", ha='center', va='top', fontsize=8)
    
    ax.plot(0.05, 0.913, 'o', color=COLORS['GreenLine'], markersize=8)
    ax.text(0.05, 0.92, "Sweet Spot\n(96% Acc, 4x Fairer)", ha='center', va='bottom', fontsize=8)
    
    ax.plot(0.0, 0.85, 'o', color=COLORS['BlueLine'], markersize=8)
    ax.text(0.005, 0.85, "Strict Equality\n(Large Drop)", ha='left', va='center', fontsize=8)
    
    ax.set_xlabel('Demographic Disparity (Lower is Fairer)')
    ax.set_ylabel('Model Accuracy')
    ax.invert_xaxis() 
    
    return ax

def plot_efficiency_history(ax=None):
    """Visualizes 'Historical Efficiency Trends' (Gantt Chart)."""
    if ax is None: fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load data
    try:
        df = pd.read_csv('../../../data/efficiency_history.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('book/data/efficiency_history.csv')
        except:
            return ax # Fallback omitted for brevity
            
    # Map Y-positions
    tracks = ['Data Selection', 'Compute Efficiency', 'Algorithmic Efficiency']
    y_map = {t: i for i, t in enumerate(tracks)}
    
    for _, row in df.iterrows():
        y = y_map[row['Track']]
        start = row['Start']
        duration = row['End'] - start
        color = COLORS[row['Color']]
        
        # Draw bar
        ax.barh(y, duration, left=start, height=0.5, color=color, alpha=0.8, edgecolor='white')
        
        # Label era
        mid = start + duration/2
        # Special case for small eras or long text
        label = row['Era']
        fontsize = 9
        if duration < 5: fontsize=8
        
        ax.text(mid, y, label, ha='center', va='center', color='white', fontsize=fontsize, fontweight='bold')

    ax.set_yticks(range(len(tracks)))
    ax.set_yticklabels(tracks)
    ax.set_xlabel('Year')
    ax.set_xlim(1980, 2025)
    
    # Grid and styling
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    
    return ax

def plot_inference_pipeline(ax=None):
    """Visualizes 'The Inference Pipeline' (Serving Flow)."""
    if ax is None: fig, ax = plt.subplots(figsize=(10, 4))
    
    # Disable axes
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    
    # Define Boxes
    boxes = [
        {'label': 'Raw\nInput', 'x': 1, 'color': COLORS['GreenL'], 'edge': COLORS['GreenLine']},
        {'label': 'Pre-\nprocessing', 'x': 3, 'color': COLORS['grid'], 'edge': COLORS['primary']},
        {'label': 'Neural\nNetwork', 'x': 5, 'color': COLORS['BlueL'], 'edge': COLORS['BlueLine']},
        {'label': 'Raw\nOutput', 'x': 7, 'color': COLORS['VioletL'], 'edge': COLORS['VioletLine']},
        {'label': 'Post-\nprocessing', 'x': 9, 'color': COLORS['grid'], 'edge': COLORS['primary']},
        {'label': 'Final\nOutput', 'x': 11, 'color': COLORS['VioletL'], 'edge': COLORS['VioletLine']}
    ]
    
    # Draw Boxes
    from matplotlib.patches import FancyBboxPatch, Rectangle
    
    for i, box in enumerate(boxes):
        # Main Box
        p = FancyBboxPatch((box['x']-0.75, 2), 1.5, 1.2, boxstyle="round,pad=0.1", 
                           fc=box['color'], ec=box['edge'], linewidth=2)
        ax.add_patch(p)
        ax.text(box['x'], 2.6, box['label'], ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Arrow to next
        if i < len(boxes) - 1:
            ax.annotate("", xy=(boxes[i+1]['x']-0.8, 2.6), xytext=(box['x']+0.8, 2.6), 
                        arrowprops=dict(arrowstyle="->", color=COLORS['primary'], lw=1.5))

    # Grouping: Traditional Computing (Pre)
    ax.add_patch(Rectangle((2.1, 1.5), 1.8, 2.5, fill=False, edgecolor=COLORS['grid'], linestyle='--', linewidth=1))
    ax.text(3, 1.2, "Traditional\nComputing", ha='center', fontsize=8, color=COLORS['primary'])

    # Grouping: Deep Learning
    ax.add_patch(Rectangle((4.1, 1.5), 1.8, 2.5, fill=False, edgecolor=COLORS['BlueLine'], linestyle='--', linewidth=1))
    ax.text(5, 1.2, "Deep Learning\n(Accelerator)", ha='center', fontsize=8, color=COLORS['BlueLine'], fontweight='bold')

    # Grouping: Traditional Computing (Post)
    ax.add_patch(Rectangle((8.1, 1.5), 1.8, 2.5, fill=False, edgecolor=COLORS['grid'], linestyle='--', linewidth=1))
    ax.text(9, 1.2, "Traditional\nComputing", ha='center', fontsize=8, color=COLORS['primary'])

    return ax

def plot_ml_lifecycle(ax=None):
    """Visualizes 'ML System Lifecycle' (Circular Flow)."""
    if ax is None: fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    
    from matplotlib.patches import FancyBboxPatch, ConnectionPatch
    
    # Define Nodes (Circular layout roughly)
    nodes = {
        'Collection': {'x': 2, 'y': 6, 'label': 'Data\nCollection', 'color': COLORS['BlueL'], 'edge': COLORS['BlueLine']},
        'Prep':       {'x': 6, 'y': 6, 'label': 'Data\nPreparation', 'color': COLORS['GreenL'], 'edge': COLORS['GreenLine']},
        'Train':      {'x': 10, 'y': 6, 'label': 'Model\nTraining', 'color': COLORS['OrangeL'], 'edge': COLORS['OrangeLine']},
        'Eval':       {'x': 10, 'y': 2, 'label': 'Model\nEvaluation', 'color': COLORS['RedL'], 'edge': COLORS['RedLine']},
        'Deploy':     {'x': 6, 'y': 2, 'label': 'Model\nDeployment', 'color': COLORS['VioletL'], 'edge': COLORS['VioletLine']},
        'Monitor':    {'x': 2, 'y': 2, 'label': 'Model\nMonitoring', 'color': COLORS['OrangeL'], 'edge': COLORS['OrangeLine']}
    }
    
    # Draw Nodes
    for key, node in nodes.items():
        p = FancyBboxPatch((node['x']-0.9, node['y']-0.6), 1.8, 1.2, boxstyle="round,pad=0.1", 
                           fc=node['color'], ec=node['edge'], linewidth=2)
        ax.add_patch(p)
        ax.text(node['x'], node['y'], node['label'], ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw Arrows (Main Cycle)
    arrows = [
        ('Collection', 'Prep'), ('Prep', 'Train'), ('Train', 'Eval'),
        ('Eval', 'Deploy'), ('Deploy', 'Monitor'), ('Monitor', 'Collection')
    ]
    
    for start, end in arrows:
        con = ConnectionPatch(xyA=(nodes[start]['x'], nodes[start]['y']), xyB=(nodes[end]['x'], nodes[end]['y']), 
                              coordsA="data", coordsB="data", 
                              axesA=ax, axesB=ax, 
                              arrowstyle="-|>", connectionstyle="arc3,rad=0.0", color=COLORS['primary'], lw=1.5,
                              shrinkA=20, shrinkB=20)
        ax.add_artist(con)
        
    # Feedback Loops
    # Eval -> Prep (Needs Improvement)
    con = ConnectionPatch(xyA=(nodes['Eval']['x'], nodes['Eval']['y']), xyB=(nodes['Prep']['x'], nodes['Prep']['y']),
                          coordsA="data", coordsB="data", axesA=ax, axesB=ax,
                          arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", color=COLORS['RedLine'], lw=1.5, linestyle='--',
                          shrinkA=20, shrinkB=20)
    ax.add_artist(con)
    ax.text(8, 4, "Needs Improvement", ha='center', va='center', fontsize=8, color=COLORS['RedLine'], rotation=-25, backgroundcolor='white')

    return ax

def plot_distributed_training(ax=None):
    """Visualizes 'Data Parallel Training Flow'."""
    if ax is None: fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    
    from matplotlib.patches import FancyBboxPatch, Rectangle
    
    # Input Data
    p = FancyBboxPatch((4.5, 7), 3, 0.8, boxstyle="round,pad=0.1", fc=COLORS['GreenL'], ec=COLORS['GreenLine'], lw=2)
    ax.add_patch(p)
    ax.text(6, 7.4, "Input Data", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Split Arrows
    ax.annotate("", xy=(3, 6), xytext=(6, 7), arrowprops=dict(arrowstyle="->", color=COLORS['primary'], lw=1.5))
    ax.annotate("", xy=(9, 6), xytext=(6, 7), arrowprops=dict(arrowstyle="->", color=COLORS['primary'], lw=1.5))
    
    # GPU 1 Track
    ax.add_patch(Rectangle((1.5, 2.5), 3, 3.5, fill=False, edgecolor=COLORS['BlueLine'], linestyle='--', lw=1))
    ax.text(3, 6.2, "GPU 1", ha='center', fontweight='bold', color=COLORS['BlueLine'])
    
    ax.text(3, 5.5, "Batch 1", ha='center', fontsize=9, bbox=dict(facecolor='white', edgecolor=COLORS['primary']))
    ax.annotate("", xy=(3, 4.5), xytext=(3, 5.2), arrowprops=dict(arrowstyle="->", color=COLORS['primary']))
    ax.text(3, 4.0, "Forward/\nBackward", ha='center', va='center', fontsize=9, bbox=dict(facecolor=COLORS['BlueL'], edgecolor='none'))
    ax.annotate("", xy=(3, 3.0), xytext=(3, 3.5), arrowprops=dict(arrowstyle="->", color=COLORS['primary']))
    ax.text(3, 2.8, "Gradients", ha='center', fontsize=9, style='italic')

    # GPU 2 Track
    ax.add_patch(Rectangle((7.5, 2.5), 3, 3.5, fill=False, edgecolor=COLORS['BlueLine'], linestyle='--', lw=1))
    ax.text(9, 6.2, "GPU 2", ha='center', fontweight='bold', color=COLORS['BlueLine'])
    
    ax.text(9, 5.5, "Batch 2", ha='center', fontsize=9, bbox=dict(facecolor='white', edgecolor=COLORS['primary']))
    ax.annotate("", xy=(9, 4.5), xytext=(9, 5.2), arrowprops=dict(arrowstyle="->", color=COLORS['primary']))
    ax.text(9, 4.0, "Forward/\nBackward", ha='center', va='center', fontsize=9, bbox=dict(facecolor=COLORS['BlueL'], edgecolor='none'))
    ax.annotate("", xy=(9, 3.0), xytext=(9, 3.5), arrowprops=dict(arrowstyle="->", color=COLORS['primary']))
    ax.text(9, 2.8, "Gradients", ha='center', fontsize=9, style='italic')
    
    # Synchronization
    p = FancyBboxPatch((4, 1), 4, 1, boxstyle="round,pad=0.1", fc=COLORS['VioletL'], ec=COLORS['VioletLine'], lw=2)
    ax.add_patch(p)
    ax.text(6, 1.5, "Gradient Aggregation\n(AllReduce)", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows to Sync
    ax.annotate("", xy=(5, 2), xytext=(3, 2.5), arrowprops=dict(arrowstyle="->", color=COLORS['primary'], lw=1.5))
    ax.annotate("", xy=(7, 2), xytext=(9, 2.5), arrowprops=dict(arrowstyle="->", color=COLORS['primary'], lw=1.5))
    
    # Update Arrow
    ax.annotate("Model Update", xy=(6, 4.0), xytext=(6, 2.0), 
                arrowprops=dict(arrowstyle="->", color=COLORS['RedLine'], lw=2, linestyle='dashed'),
                ha='center', va='center', fontsize=9, color=COLORS['RedLine'], backgroundcolor='white')

    return ax

def plot_algo_efficiency(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    df = pd.DataFrame(ALGO_EFFICIENCY_DATA)

    ax.scatter(df['Year'], df['Efficiency_Factor'], color=COLORS['RedLine'], s=80, alpha=0.9, edgecolors='white', zorder=3)
    
    offsets = {
        'AlexNet': (0, -15),
        'VGG-11': (0, -15),
        'GoogLeNet': (0, 10),
        'ResNet-18': (0, -15),
        'DenseNet-121': (0, -15),
        'MobileNet-v1': (-25, 5),
        'ShuffleNet-v1': (-20, 15),
        'MobileNet-v2': (25, -15),
        'ShuffleNet-v2': (10, 15),
        'EfficientNet-B0': (0, 10)
    }
    
    for _, row in df.iterrows():
        name = row['Model']
        offset = offsets.get(name, (0, 5))
        ax.annotate(name, (row['Year'], row['Efficiency_Factor']), 
                    xytext=offset, textcoords='offset points', 
                    fontsize=8, ha='center', color=COLORS['primary'],
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    years = np.linspace(2012, 2020, 100)
    trend = 1.0 * 1.72**(years - 2012.5)
    
    ax.plot(years, trend, '--', color=COLORS['BlueLine'], label='44x Improvement', linewidth=2, zorder=1)
    
    ax.set_ylim(0, 50)
    ax.set_xlim(2012, 2020)
    ax.set_xlabel('Year')
    ax.set_ylabel('Efficiency Factor (Relative to AlexNet)')
    ax.legend(loc='upper left', fontsize=9)
    return ax

def plot_business_cost(ax=None):
    """Visualizes 'The Cost of Decisions' (Threshold Selection)."""
    if ax is None: fig, ax = plt.subplots()
    
    thresholds = np.linspace(0, 1, 100)
    
    # Scenario: Fraud Detection
    # False Positive (Blocking a user): Cost $10 (Support ticket)
    # False Negative (Missed fraud): Cost $1000 (Chargeback)
    
    # Distributions (simulated)
    from scipy.stats import norm
    neg_dist = norm(loc=0.3, scale=0.15)
    pos_dist = norm(loc=0.7, scale=0.15)
    
    # Error rates vs Threshold
    fpr = 1 - neg_dist.cdf(thresholds)
    fnr = pos_dist.cdf(thresholds)
    
    # Costs
    cost_fp = 10 * fpr
    cost_fn = 1000 * fnr * 0.01 
    total_cost = cost_fp + cost_fn
    
    ax.plot(thresholds, cost_fp, ':', color=COLORS['BlueLine'], label='Cost of False Positives', linewidth=2)
    ax.plot(thresholds, cost_fn, '--', color=COLORS['RedLine'], label='Cost of False Negatives', linewidth=2)
    ax.plot(thresholds, total_cost, '-', color=COLORS['primary'], label='Total Business Cost', linewidth=2.5)
    
    # Optimal Point
    min_idx = np.argmin(total_cost)
    opt_thresh = thresholds[min_idx]
    opt_cost = total_cost[min_idx]
    
    ax.axvline(opt_thresh, color=COLORS['GreenLine'], linestyle='--', alpha=0.5)
    ax.plot(opt_thresh, opt_cost, 'o', color=COLORS['GreenLine'], markersize=8)
    ax.annotate(f"Optimal Threshold\n(T={opt_thresh:.2f})", 
                xy=(opt_thresh, opt_cost), xytext=(0, 30), textcoords='offset points',
                ha='center',
                arrowprops=dict(facecolor=COLORS['primary'], arrowstyle='->', lw=1.5),
                fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Expected Cost ($)')
    ax.legend(loc='upper center', fontsize=8)
    return ax

def plot_tail_latency(ax=None):
    """Visualizes 'The Tail Latency Explosion' (Queuing Theory)."""
    if ax is None: fig, ax = plt.subplots()
    
    utilization = np.linspace(0, 0.95, 100)
    
    mean_latency = 1 / (1 - utilization)
    p99_latency = mean_latency * 4.6 
    
    ax.plot(utilization, mean_latency, '--', color=COLORS['BlueLine'], label='Median Latency (p50)', linewidth=2)
    ax.plot(utilization, p99_latency, '-', color=COLORS['RedLine'], label='Tail Latency (p99)', linewidth=2.5)
    
    ax.set_xlabel('System Utilization (%)')
    ax.set_ylabel('Request Latency (Normalized)')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 50)
    
    ax.axvspan(0, 0.5, color=COLORS['GreenL'], alpha=0.2)
    ax.text(0.25, 5, "Safe Zone", color=COLORS['GreenLine'], fontweight='bold', ha='center')
    
    ax.axvspan(0.7, 1.0, color=COLORS['RedL'], alpha=0.2)
    ax.text(0.85, 40, "Danger Zone\n(Queue Explosion)", color=COLORS['RedLine'], fontweight='bold', ha='center', fontsize=9)
    
    ax.annotate("The Knee", xy=(0.7, 15), xytext=(0.5, 25),
                arrowprops=dict(facecolor=COLORS['primary'], arrowstyle='->', lw=1.5), fontsize=9)
    
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.legend(loc='upper left', fontsize=8)
    return ax

def plot_rising_ridge(ax=None):
    """Visualizes 'The Rising Ridge' (Hardware Arithmetic Intensity)."""
    if ax is None: fig, ax = plt.subplots()
    
    # Data: Peak FLOPS / Memory Bandwidth (FLOPs/Byte)
    years = [2017, 2020, 2022, 2024]
    chips = ['V100', 'A100', 'H100', 'B200']
    ridges = [139, 153, 295, 562]
    
    ax.plot(years, ridges, 'o-', color=COLORS['RedLine'], linewidth=2.5, markersize=8, label='Hardware Ridge Point')
    
    for y, r, c in zip(years, ridges, chips):
        # Position chips above point, and numeric value below
        ax.annotate(c, (y, r), xytext=(0, 12), textcoords='offset points', 
                    ha='center', fontweight='bold', color=COLORS['RedLine'], fontsize=9)
        ax.annotate(f"{r:.0f}", (y, r), xytext=(0, -18), textcoords='offset points', 
                    ha='center', fontsize=8, color=COLORS['primary'])

    # Shaded Zones
    ax.axhspan(0, 100, color=COLORS['BlueL'], alpha=0.2)
    ax.text(2019, 50, "Memory-Rich Zone\n(Legacy Ops Safe)", color=COLORS['BlueLine'], ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.axhspan(100, 600, color=COLORS['OrangeL'], alpha=0.1)
    ax.text(2019, 350, "Compute-Dense Zone\n(Transformers Required)", color=COLORS['OrangeLine'], ha='center', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Release Year')
    ax.set_ylabel('Arithmetic Intensity (FLOPs/Byte)')
    ax.set_ylim(0, 650)
    ax.set_xticks(years)
    
    return ax

def plot_context_explosion(ax=None):
    """Visualizes 'The Context Explosion' (Context Window Growth)."""
    if ax is None: fig, ax = plt.subplots()
    
    data = [
        (2018, 512, "BERT"),
        (2019, 1024, "GPT-2"),
        (2020, 2048, "GPT-3"),
        (2023.2, 32768, "GPT-4"),
        (2023.5, 100000, "Claude 2"),
        (2024.1, 1000000, "Gemini 1.5")
    ]
    
    years = [d[0] for d in data]
    context = [d[1] for d in data]
    labels = [d[2] for d in data]
    
    ax.step(years, context, where='post', color=COLORS['VioletLine'], linewidth=2.5, zorder=2)
    ax.scatter(years, context, color=COLORS['VioletLine'], s=60, zorder=3, edgecolors='white')
    
    for y, c, l in zip(years, context, labels):
        # Adjusting label offsets to avoid step lines
        off_x, off_y = 5, 5
        ha = 'left'
        if l == "Gemini 1.5": 
            off_x, off_y = -10, 10
            ha = 'right'
        elif l == "GPT-4":
            off_x, off_y = -10, 10
            ha = 'right'
            
        c_label = f"{int(c/1000)}k" if c >= 1000 else f"{c/1000:.1f}k"
        ax.annotate(f"{l}\n({c_label})", (y, c), xytext=(off_x, off_y), textcoords='offset points', 
                    fontsize=8, fontweight='bold', color=COLORS['VioletLine'], rotation=45, ha=ha, va='bottom')

    ax.set_yscale('log')
    ax.set_xlabel('Year')
    ax.set_ylabel('Context Window (Tokens)')
    ax.set_ylim(100, 5000000)
    
    # Era Labels (Higher contrast)
    ax.axhline(2000, linestyle='--', color=COLORS['grid'], alpha=0.5)
    ax.text(2018.1, 2600, "Standard Era (2k)", color=COLORS['grid'], fontsize=8, fontweight='bold')
    
    ax.axhline(100000, linestyle='--', color=COLORS['grid'], alpha=0.5)
    ax.text(2018.1, 130000, "RAG Era (100k)", color=COLORS['grid'], fontsize=8, fontweight='bold')
    
    return ax

def plot_intelligence_deflation(ax=None):
    """Visualizes 'Intelligence Deflation' (Token Pricing Trends)."""
    if ax is None: fig, ax = plt.subplots()
    
    # Pricing: Approx USD per 1M Input Tokens
    data = [
        (2020.5, 20.0, "GPT-3 (Davinci)"),
        (2023.1, 2.0, "GPT-3.5 Turbo"),
        (2023.2, 30.0, "GPT-4 (Original)"),
        (2024.2, 15.0, "Claude 3 Opus"),
        (2024.2, 0.25, "Claude 3 Haiku"),
        (2024.3, 5.0, "GPT-4o"),
        (2024.4, 0.075, "Gemini 1.5 Flash"),
        (2024.6, 0.15, "GPT-4o-mini"),
        (2024.9, 0.27, "DeepSeek-V3")
    ]
    
    # Sort for fitting
    data.sort(key=lambda x: x[0])
    years = np.array([d[0] for d in data])
    prices = np.array([d[1] for d in data])
    labels = [d[2] for d in data]
    
    # 1. Trend Line (Log-Linear Fit)
    # Filter for the "Frontier/Efficiency" frontier (exclude high-priced outliers like Opus/GPT-4 for the trend)
    # We want to show the deflation of the "best price for reasonable intelligence"
    trend_years = np.array([2020.5, 2023.1, 2024.2, 2024.4, 2024.9])
    trend_prices = np.array([20.0, 2.0, 0.25, 0.075, 0.27]) # Approximate "Cheapest Competent Model"
    
    # Fit: ln(y) = m*x + c
    slope, intercept = np.polyfit(trend_years, np.log10(trend_prices), 1)
    
    line_years = np.linspace(2020, 2025.5, 100)
    line_prices = 10**(slope * line_years + intercept)
    
    ax.plot(line_years, line_prices, '--', color=COLORS['grid'], linewidth=1.5, label='Deflation Trend (~10x / 18mo)', zorder=1)
    
    # 2. Plot Points
    ax.scatter(years, prices, color=COLORS['GreenLine'], s=50, zorder=3, edgecolors='white', linewidth=1.5)
    
    # 3. Smart Label Placement
    for y, p, l in zip(years, prices, labels):
        off_x, off_y = 5, 5
        va, ha = 'bottom', 'left'
        
        if l == "GPT-3 (Davinci)":
            off_y = 8
        elif l == "GPT-3.5 Turbo":
            off_x, off_y = -8, -12
            ha, va = 'right', 'top'
        elif l == "GPT-4 (Original)":
            off_y = 8
        elif l == "Claude 3 Opus":
            off_x, off_y = -8, 8
            ha = 'right'
        elif l == "Claude 3 Haiku":
            off_x, off_y = -8, 8
            ha, va = 'right', 'bottom'
        elif l == "Gemini 1.5 Flash":
            off_x, off_y = -8, -15 
            ha, va = 'right', 'top'
        elif l == "GPT-4o":
            off_x, off_y = 8, 8
        elif l == "GPT-4o-mini":
            off_x, off_y = 8, -15
            va = 'top'
        elif l == "DeepSeek-V3":
            off_x, off_y = 10, 8
            ha = 'left'

        ax.annotate(l, (y, p), xytext=(off_x, off_y), textcoords='offset points', 
                    fontsize=8, fontweight='bold', ha=ha, va=va, color=COLORS['primary'],
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 4. Styling & Limits
    ax.set_yscale('log')
    ax.set_yticks([100, 10, 1, 0.1, 0.01])
    ax.set_yticklabels(['$100', '$10', '$1', '$0.10', '$0.01'])
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Price per 1M Tokens (USD)')
    ax.grid(True, which="major", ls="-", alpha=0.3)
    
    # Ensure Gemini label (at bottom) fits
    ax.set_ylim(0.01, 500) 
    ax.set_xlim(2020, 2025.5)
    
    # Add Trend Legend
    ax.text(2021, 0.05, "Trend: ~50% Cheaper\nEvery 8 Months", color=COLORS['grid'], fontsize=9, style='italic')
    
    return ax

def plot_data_quality_multiplier(ax=None):
    """Visualizes 'The Data Quality Multiplier' (Clean vs. Noisy Scaling)."""
    if ax is None: fig, ax = plt.subplots()
    
    # Power Law Scaling: Err = a * (Data)^-k
    # Clean Data: k = 0.5 (Fast learning)
    # Noisy Data: k = 0.25 (Slow learning) + Irreducible Error floor
    
    data_size = np.logspace(2, 5, 100) # 100 to 100k samples
    
    # Accuracy = 100 - Error
    acc_clean = 95 - 40 * (data_size / 100)**(-0.3)
    acc_noisy = 85 - 30 * (data_size / 100)**(-0.15)
    
    ax.plot(data_size, acc_clean, '-', color=COLORS['GreenLine'], linewidth=2.5, label='Clean Data (High Quality)')
    ax.plot(data_size, acc_noisy, '--', color=COLORS['RedLine'], linewidth=2.5, label='Noisy Data (Low Quality)')
    
    # Shaded Gap
    ax.fill_between(data_size, acc_noisy, acc_clean, color=COLORS['GreenL'], alpha=0.2)
    
    # Annotations
    ax.text(3000, 90, "The Quality Gap", color=COLORS['GreenLine'], fontweight='bold', fontsize=9)
    ax.annotate("Diminishing Returns", xy=(10000, 82), xytext=(10000, 75),
                arrowprops=dict(facecolor=COLORS['primary'], arrowstyle='->'), fontsize=8)
    
    ax.set_xscale('log')
    ax.set_xlabel('Training Set Size (Samples)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend(loc='lower right', fontsize=9)
    
    return ax

def plot_communication_tax(ax=None):
    """Visualizes 'The Communication Tax' (Distributed Training Scaling)."""
    if ax is None: fig, ax = plt.subplots()
    
    # Extended range for modern clusters
    N = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    
    # Scenarios with Amdahl's Law style scaling: Speedup = N / (1 + (N-1)*r)
    scenarios = [
        {'r': 0.0, 'name': 'Ideal Linear', 'color': COLORS['grid'], 'style': '--', 'marker': None},
        {'r': 0.05, 'name': 'Compute Bound (ResNet)', 'color': COLORS['BlueLine'], 'style': '-', 'marker': 'o'},
        {'r': 0.15, 'name': 'Balanced (LLM + NVLink)', 'color': COLORS['GreenLine'], 'style': '-', 'marker': 's'}, 
        {'r': 0.40, 'name': 'Bandwidth Bound (Slow Net)', 'color': COLORS['RedLine'], 'style': '-', 'marker': '^'}
    ]
    
    # Plot Lines
    for sc in scenarios:
        speedup = N / (1 + (N - 1) * sc['r'])
        if sc['marker']:
            ax.plot(N, speedup, sc['style'], color=sc['color'], label=sc['name'], linewidth=2.5, marker=sc['marker'], markersize=7)
        else:
            ax.plot(N, speedup, sc['style'], color=sc['color'], label=sc['name'], linewidth=2)

    # Shaded Tax Region (Between Ideal and Bandwidth Bound)
    ideal_speedup = N
    worst_speedup = N / (1 + (N - 1) * 0.40)
    ax.fill_between(N, ideal_speedup, worst_speedup, color=COLORS['RedL'], alpha=0.15)
    
    # Annotations
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(N)
    ax.set_xticklabels(N)
    ax.set_yticks(N)
    ax.set_yticklabels(N)
    
    # Tax Annotation
    ax.annotate("The Communication Tax", xy=(128, 128/(1+127*0.4)), xytext=(16, 100),
                arrowprops=dict(facecolor=COLORS['RedLine'], arrowstyle='->', lw=1.5), 
                color=COLORS['RedLine'], fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor=COLORS['RedL'], pad=4))
    
    ax.set_xlabel('Number of GPUs (N)')
    ax.set_ylabel('Effective Speedup')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    return ax

def plot_kv_cache_growth(ax=None):
    """Visualizes 'The KV-Cache Explosion' (Memory vs Sequence Length)."""
    if ax is None: fig, ax = plt.subplots()
    
    # Model: ~70B Params (e.g., Llama-2-70B scale)
    # L=80 layers, d_model=8192
    # Standard Attention (MHA): KV size = 2 * layers * d_model * seq_len * batch * 2_bytes(fp16)
    
    seq_len = np.linspace(0, 32000, 100)
    layers = 80
    d_model = 8192
    bytes_per_param = 2 # FP16
    
    def get_kv_gb(batch, seq):
        bytes_total = 2 * layers * d_model * seq * batch * bytes_per_param
        return bytes_total / 1e9
    
    batches = [1, 4, 16, 32]
    colors = [COLORS['BlueLine'], COLORS['GreenLine'], COLORS['OrangeLine'], COLORS['VioletLine']]
    
    for b, c in zip(batches, colors):
        gb = get_kv_gb(b, seq_len)
        ax.plot(seq_len, gb, label=f'Batch Size {b}', color=c, linewidth=2)

    # Hardware Limit (A100/H100 80GB)
    limit_gb = 80
    ax.axhline(limit_gb, color=COLORS['RedLine'], linestyle='--', linewidth=2)
    ax.text(1000, limit_gb + 2, "A100/H100 Capacity (80GB)", color=COLORS['RedLine'], fontweight='bold', fontsize=9)
    
    # OOM Zone
    ax.axhspan(limit_gb, 140, color=COLORS['RedL'], alpha=0.2)
    ax.text(16000, 100, "Out of Memory (OOM) Zone", color=COLORS['RedLine'], ha='center', fontsize=10, fontweight='bold')
    
    # Annotations
    ax.annotate("Linear Growth", xy=(15000, get_kv_gb(4, 15000)), xytext=(20000, 30),
                arrowprops=dict(facecolor=COLORS['primary'], arrowstyle='->'), fontsize=9)

    ax.set_xlabel('Context Length (Tokens)')
    ax.set_ylabel('KV Cache Size (GB) [FP16]')
    ax.set_xlim(0, 32000)
    ax.set_ylim(0, 120)
    ax.set_xticks([0, 8000, 16000, 24000, 32000])
    ax.set_xticklabels(['0', '8k', '16k', '24k', '32k'])
    
    ax.legend(loc='lower right', fontsize=8)
    
    return ax

def plot_double_descent(ax=None):
    """Visualizes 'The Double Descent' (Generalization vs Complexity)."""
    if ax is None: fig, ax = plt.subplots()
    
    # X-axis: Complexity
    x = np.linspace(0.1, 3.0, 200)
    
    # 1. Classical Regime (U-Curve)
    # Error decreases then increases as we overfit
    y_classical = 0.5 * (x - 0.5)**2 + 0.3
    
    # 2. Modern Regime (Descent)
    # After interpolation (x=1.0), error drops again
    y_modern = 0.3 * np.exp(-1.5 * (x - 1.0)) + 0.15
    
    # Combine with a peak at x=1
    # We blend them: precise peak handling
    y = np.zeros_like(x)
    mask_under = x <= 1.0
    mask_over = x > 1.0
    
    # Classical part with a spike at 1.0
    y[mask_under] = 0.8 * (x[mask_under] - 0.4)**2 + 0.25 + 0.4 * np.exp(-100 * (x[mask_under]-1.0)**2)
    
    # Modern part descending from the peak
    y[mask_over] = 0.3 * np.exp(-2.0 * (x[mask_over] - 1.0)) + 0.2
    
    # Smoothing the join
    y = np.convolve(y, np.ones(5)/5, mode='same')

    ax.plot(x, y, color=COLORS['BlueLine'], linewidth=2.5)
    
    # Zones
    ax.axvspan(0, 1.0, color=COLORS['grid'], alpha=0.1)
    ax.text(0.5, 0.8, "Classical Regime\n(Under-parameterized)", ha='center', fontsize=9, fontweight='bold', color=COLORS['primary'])
    
    ax.axvspan(1.0, 3.0, color=COLORS['GreenL'], alpha=0.1)
    ax.text(2.0, 0.8, "Modern Regime\n(Over-parameterized)", ha='center', fontsize=9, fontweight='bold', color=COLORS['GreenLine'])
    
    # Threshold Line
    ax.axvline(1.0, color=COLORS['RedLine'], linestyle='--', alpha=0.6)
    ax.text(1.05, 0.6, "Interpolation Threshold\n(Zero Training Error)", color=COLORS['RedLine'], fontsize=8)

    ax.set_xlabel('Model Complexity (Parameters / Data Size)')
    ax.set_ylabel('Test Error')
    ax.set_ylim(0.1, 0.9)
    ax.set_xlim(0, 3.0)
    ax.set_yticks([]) # Qualitative
    
    # Annotation
    ax.annotate("Bigger is Better", xy=(2.5, 0.22), xytext=(2.5, 0.35),
                arrowprops=dict(facecolor=COLORS['GreenLine'], arrowstyle='->'), 
                color=COLORS['GreenLine'], ha='center', fontsize=9)
    
    return ax

def plot_mlops_leverage(ax=None):
    """Visualizes 'The MLOps Leverage' (Infrastructure ROI)."""
    if ax is None: fig, ax = plt.subplots()
    
    team_size = np.linspace(1, 20, 100)
    
    # 1. Ad-hoc / Manual (Saturates due to coordination overhead)
    # Velocity = N * (1 - alpha * N)
    velocity_manual = 5 * team_size * np.exp(-0.05 * team_size)
    
    # 2. Platform / Automated (Scales linearly or super-linearly)
    # Velocity = N * Leverage
    velocity_platform = 5 * team_size ** 1.1  # Slight exponential network effect
    
    ax.plot(team_size, velocity_manual, '--', color=COLORS['RedLine'], label='Ad-hoc Scripts', linewidth=2)
    ax.plot(team_size, velocity_platform, '-', color=COLORS['BlueLine'], label='MLOps Platform', linewidth=2.5)
    
    # Fill gap
    ax.fill_between(team_size, velocity_manual, velocity_platform, color=COLORS['BlueL'], alpha=0.2)
    
    ax.text(18, 120, "The Flywheel\nEffect", ha='center', color=COLORS['BlueLine'], fontweight='bold', fontsize=9)
    ax.annotate("", xy=(18, 110), xytext=(18, 40), arrowprops=dict(arrowstyle="->", color=COLORS['BlueLine']))
    
    ax.text(15, 25, "Coordination Tax", ha='center', color=COLORS['RedLine'], fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Team Size (Engineers)')
    ax.legend(loc='upper left', fontsize=9)
    
    return ax

def plot_invariants_cycle(ax=None):
    """Visualizes 'The Cycle of ML Systems' (The 12 Invariants)."""
    if ax is None: fig, ax = plt.subplots(figsize=(10, 8))
    
    # Disable axes
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
    
    # Nodes (Diamond Layout)
    nodes = {
        'Data':      {'x': -0.8, 'y': 0.0, 'label': 'Foundations\n(Data)', 'color': COLORS['GreenL'], 'ec': COLORS['GreenLine']},
        'Model':     {'x': 0.0, 'y': 0.8, 'label': 'Build\n(Model)', 'color': COLORS['BlueL'], 'ec': COLORS['BlueLine']},
        'Hardware':  {'x': 0.8, 'y': 0.0, 'label': 'Optimize\n(Hardware)', 'color': COLORS['OrangeL'], 'ec': COLORS['OrangeLine']},
        'Ops':       {'x': 0.0, 'y': -0.8, 'label': 'Deploy\n(Operations)', 'color': COLORS['VioletL'], 'ec': COLORS['VioletLine']}
    }
    
    # Draw Nodes
    for k, n in nodes.items():
        # Box
        p = FancyBboxPatch((n['x']-0.2, n['y']-0.1), 0.4, 0.2, boxstyle="round,pad=0.05", 
                           fc=n['color'], ec=n['ec'], linewidth=2, zorder=10)
        ax.add_patch(p)
        ax.text(n['x'], n['y'], n['label'], ha='center', va='center', fontsize=11, fontweight='bold', zorder=11)

    # Center: Conservation of Complexity
    ax.add_patch(Circle((0,0), 0.25, fc='white', ec=COLORS['grid'], linestyle='--', linewidth=1.5, zorder=5))
    ax.text(0, 0, "Conservation\nof\nComplexity", ha='center', va='center', fontsize=10, fontweight='bold', color=COLORS['primary'], zorder=6)

    # Transitions (Arrows & Invariants)
    transitions = [
        ('Data', 'Model', ['1. Data as Code', '2. Data Gravity'], (-0.5, 0.5)),
        ('Model', 'Hardware', ['3. Iron Law', '4. Silicon Contract'], (0.5, 0.5)),
        ('Hardware', 'Ops', ['5. Pareto Frontier', '6. Arith. Intensity', '8. Amdahl\'s Law'], (0.5, -0.5)),
        ('Ops', 'Data', ['10. Stat. Drift', '11. Training-Serving Skew', '12. Latency Budget'], (-0.5, -0.5))
    ]
    
    for start, end, invs, pos in transitions:
        s_node = nodes[start]
        e_node = nodes[end]
        
        # Arrow
        con = ConnectionPatch(xyA=(s_node['x'], s_node['y']), xyB=(e_node['x'], e_node['y']), 
                              coordsA="data", coordsB="data", 
                              axesA=ax, axesB=ax, 
                              arrowstyle="-|>,head_width=0.4,head_length=0.4", connectionstyle="arc3,rad=-0.2", 
                              color=COLORS['primary'], lw=2.5, zorder=1)
        ax.add_artist(con)
        
        # Labels (Invariants)
        for i, inv in enumerate(invs):
            offset = 0.12 * (i - (len(invs)-1)/2)
            ax.text(pos[0], pos[1] - offset, inv, ha='center', va='center', 
                    fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1.5))

    return ax

def plot_selection_inequality(ax=None):
    """Visualizes 'The Selection Inequality' (Overhead vs Savings)."""
    if ax is None: fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Baseline', 'Efficient Selection', 'Expensive Selection']
    
    # Data components
    full_train_cost = np.array([100, 0, 0])
    selection_overhead = np.array([0, 5, 60])
    subset_train_cost = np.array([0, 40, 40])
    
    # Plotting
    x = np.arange(len(categories))
    width = 0.6
    
    # Stacked bars
    p1 = ax.bar(x, full_train_cost, width, label='Full Training', color=COLORS['BlueFill'], edgecolor=COLORS['BlueLine'])
    p2 = ax.bar(x, selection_overhead, width, bottom=full_train_cost, label='Selection Overhead', color=COLORS['OrangeL'], edgecolor=COLORS['OrangeLine'])
    p3 = ax.bar(x, subset_train_cost, width, bottom=full_train_cost + selection_overhead, label='Subset Training', color=COLORS['GreenFill'], edgecolor=COLORS['GreenLine'])
    
    # Labels
    ax.set_ylabel('Total Time (Normalized)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 130)
    
    # Legend
    ax.legend(loc='upper right', ncol=3, fontsize=9)
    
    # Annotations
    
    # Arrow for savings (Baseline -> Efficient)
    ax.annotate("", xy=(1, 45), xytext=(1, 100), arrowprops=dict(arrowstyle="<->", color=COLORS['GreenLine'], lw=2))
    ax.text(1.1, 72, "55% Savings", color=COLORS['GreenLine'], fontweight='bold', va='center')
    
    # Arrow for no savings (Baseline -> Expensive)
    ax.annotate("", xy=(2, 100), xytext=(0, 100), arrowprops=dict(arrowstyle="-", linestyle="--", color=COLORS['grid']))
    ax.text(2, 105, "No Savings!", color=COLORS['RedLine'], fontweight='bold', ha='center')
    
    return ax

def plot_ppd_curve(ax=None):
    """Visualizes 'Diminishing Returns of Data'."""
    if ax is None: fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.linspace(0, 100, 200)
    # Random: Slow exponential approach
    y_random = 95 * (1 - np.exp(-0.04 * x))
    # Efficient: Fast exponential approach
    y_efficient = 95 * (1 - np.exp(-0.15 * x))
    
    ax.plot(x, y_random, '--', color=COLORS['grid'], label='Random Sampling', linewidth=2)
    ax.plot(x, y_efficient, '-', color=COLORS['BlueLine'], label='Efficient Selection', linewidth=2.5)
    
    # Fill gap
    ax.fill_between(x, y_random, y_efficient, color=COLORS['BlueL'], alpha=0.1)
    
    ax.set_xlabel('Dataset Size (% of Total)')
    ax.set_ylabel('Model Accuracy (%)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Annotations
    idx = 40 # x=20
    x_val = x[idx]
    y_eff = y_efficient[idx]
    y_rnd = y_random[idx]
    
    # Vertical line showing efficiency gap
    ax.annotate("", xy=(x_val, y_eff), xytext=(x_val, y_rnd),
                arrowprops=dict(arrowstyle="<->", color=COLORS['RedLine'], lw=1.5))
    ax.text(x_val+2, (y_eff+y_rnd)/2, "Efficiency Gap\n(Saved Compute)", color=COLORS['RedLine'], fontsize=9, va='center', fontweight='bold')
    
    ax.legend(loc='lower right', fontsize=9)
    return ax

def plot_power_differentials(ax=None):
    """Visualizes 'Power Consumption Differentials' across ML system types."""
    if ax is None: fig, ax = plt.subplots()
    
    # Data
    system_types = ["Tiny", "Edge", "Datacenter", "Training"]
    min_power = [5.6, 3.9, 266.9, 5.5]
    max_power = [166.6, 1100, 6300, 498000]

    # Plot vertical lines connecting min and max
    for i, (minp, maxp) in enumerate(zip(min_power, max_power)):
        ax.plot([i, i], [minp, maxp], color=COLORS['grid'], linewidth=2, zorder=1)

    # Plot points
    ax.scatter(range(len(system_types)), min_power, color=COLORS['BlueLine'], s=80, 
               label='Minimum Power', zorder=2, edgecolors='white')
    ax.scatter(range(len(system_types)), max_power, color=COLORS['RedLine'], s=80, 
               label='Maximum Power', zorder=2, edgecolors='white')

    # Logarithmic scale
    ax.set_yscale('log')
    ax.set_ylabel('Power Consumption (Log Scale)')
    ax.set_xlabel('System Type')
    ax.set_xticks(range(len(system_types)))
    ax.set_xticklabels(system_types)
    ax.legend(loc='upper left', fontsize=8)
    
    return ax

def plot_imagenet_challenge(ax=None):
    """Visualizes 'ImageNet Challenge Progression' from 2010-2015."""
    if ax is None: fig, ax = plt.subplots()
    
    # Data
    years = [2010, 2011, 2012, 2013, 2014, 2014, 2015]
    models = ["Baseline", "Baseline", "AlexNet", "ZFNet", "VGGNet", "GoogleNet", "ResNet"]
    errors = [28.2, 25.8, 16.4, 11.7, 7.3, 6.7, 3.57]

    # Plot line and points
    ax.plot(years, errors, color=COLORS['BlueLine'], linewidth=1.5, zorder=1)
    ax.scatter(years, errors, color=COLORS['RedLine'], s=50, zorder=2, edgecolors='white')

    # Add labels with smart positioning
    offsets = [
        (5, 8), (5, 8), (5, 8), (5, 8), (-40, 8), (5, -15), (0, 8)
    ]
    for year, model, error, (ox, oy) in zip(years, models, errors, offsets):
        ax.annotate(model, (year, error), textcoords='offset points',
                    xytext=(ox, oy), fontsize=9, ha='left' if ox >= 0 else 'right')

    ax.set_ylim(0, 30)
    ax.set_xlabel('Year')
    ax.set_ylabel('Top-5 Error (%)')
    
    return ax

def plot_compute_memory_imbalance(ax=None):
    """Visualizes 'The Compute-Bandwidth Divergence' (AI Memory Wall)."""
    if ax is None: fig, ax = plt.subplots()
    
    # Data
    years = [2000, 2005, 2010, 2015, 2020, 2025]
    compute_performance = [1e3, 1e5, 1e7, 1e9, 1e12, 1e15]  # FLOPs
    memory_bandwidth = [1, 10, 50, 100, 500, 1000]  # GB/s

    # Shaded area between curves
    ax.fill_between(years, memory_bandwidth, compute_performance, color=COLORS['grid'], alpha=0.3)

    # Plot lines and points
    ax.plot(years, compute_performance, 'o-', color=COLORS['BlueLine'], linewidth=1.5, 
            markersize=6, label='Compute Performance')
    ax.plot(years, memory_bandwidth, 's-', color=COLORS['OrangeLine'], linewidth=1.5, 
            markersize=6, label='Memory Bandwidth')

    # Double-headed arrow at 2023
    ax.annotate('', xy=(2023, 1e13), xytext=(2023, 1e4),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=1.5))

    # Memory Wall label
    ax.text(2022, 3e6, 'Memory Wall', rotation=90, va='bottom', fontsize=10, 
            color=COLORS['primary'], fontweight='bold')

    ax.set_yscale('log')
    ax.set_xlabel('Year')
    ax.set_ylabel('Performance (Log Scale)')
    ax.legend(loc='upper left', frameon=True, edgecolor=COLORS['grid'])
    
    return ax

def plot_model_vs_bandwidth(ax=None):
    """Visualizes 'Model Size vs. Hardware Bandwidth' (Memory Wall Growth)."""
    if ax is None: fig, ax = plt.subplots(figsize=(10, 6))
    
    # AI Processor Data
    proc_names = ["NVIDIA Tesla K80", "Google TPU v2", "NVIDIA Tesla V100",
                  "NVIDIA A100", "Google TPU v4", "NVIDIA H100", "Google TPU v6e"]
    proc_years = [2014, 2017, 2017, 2020, 2021, 2022, 2024]
    proc_bw = [480, 600, 900, 2000, 1200, 3000, 1640]
    proc_log_bw = np.log10(proc_bw)

    # ML Model Data
    model_names = ["AlexNet", "VGG-16", "ResNet-50", "BERT Large",
                   "GPT-3", "PaLM", "GPT-4", "Gemini 1"]
    model_years = [2012, 2014, 2015, 2018, 2020, 2022, 2023, 2024]
    model_params = [60, 138, 25.6, 340, 175000, 540000, 1000000, 1500000]
    model_log_params = np.log10(model_params)

    # Fit trend lines
    proc_fit = np.polyfit(proc_years, proc_log_bw, 1)
    model_fit = np.polyfit(model_years, model_log_params, 1)

    years_range = np.arange(2012, 2025)
    proc_trend = np.polyval(proc_fit, years_range)
    model_trend = np.polyval(model_fit, years_range)

    # Shaded area (from 2016 onward)
    mask = years_range >= 2016
    ax.fill_between(years_range[mask], proc_trend[mask], model_trend[mask], 
                    color=COLORS['grid'], alpha=0.2)

    # Trend lines
    ax.plot(years_range, proc_trend, '--', color=COLORS['BlueLine'], linewidth=1)
    ax.plot(years_range, model_trend, '--', color=COLORS['RedLine'], linewidth=1)

    # Scatter points
    ax.scatter(proc_years, proc_log_bw, color=COLORS['BlueLine'], s=50, zorder=3, edgecolors='white')
    ax.scatter(model_years, model_log_params, color=COLORS['RedLine'], s=50, zorder=3, edgecolors='white')

    # Processor labels
    proc_offsets = [(0, 10), (0, -15), (0, 10), (-40, -15), (0, 15), (0, 10), (0, -15)]
    for name, year, val, offset in zip(proc_names, proc_years, proc_log_bw, proc_offsets):
        ax.annotate(name, (year, val), textcoords='offset points', xytext=offset,
                    fontsize=8, color=COLORS['BlueLine'], ha='center')

    # Model labels
    model_offsets = [(0, -12), (0, 8), (0, 8), (-30, -12), (0, 8), (0, 8), (0, 8), (0, 8)]
    for name, year, val, offset in zip(model_names, model_years, model_log_params, model_offsets):
        ax.annotate(name, (year, val), textcoords='offset points', xytext=offset,
                    fontsize=8, color=COLORS['RedLine'], ha='center')

    # Memory Wall label
    mid_y = (np.polyval(proc_fit, 2020) + np.polyval(model_fit, 2020)) / 2
    ax.text(2020, mid_y, 'AI Memory Wall', fontsize=10, fontweight='bold', 
            ha='center', color=COLORS['primary'])

    ax.set_xlabel('Year')
    ax.set_ylabel('Log Scale (Base 10)')
    ax.set_xlim(2011, 2025)
    
    return ax

def plot_ai_datacenter_demand(ax=None):
    """Visualizes 'Projected AI Data Center Power Demand' to 2030."""
    if ax is None: fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    years = list(range(2015, 2031))
    dc_demand = [200, 200, 200, 200, 200, 210, 220, 230, 250, 290, 340, 400, 480, 570, 670, 780]
    ai_demand = [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 40, 70, 110, 170, 240]
    efficiency_gains = [20, 19, 18, 17, 15, 12, 9, 6, 3, 1.5, 2, 2.5, 3, 3.2, 3.5, 3.7]

    # Stacked bar chart
    bar_width = 0.8
    ax.bar(years, dc_demand, bar_width, label='Data Center ex-AI', 
           color=COLORS['BlueLine'], edgecolor='white')
    ax.bar(years, ai_demand, bar_width, bottom=dc_demand, label='AI', 
           color=COLORS['BlueL'], edgecolor='white')

    ax.set_xlabel('Year')
    ax.set_ylabel('Data Center Power Demand (TWh)')
    ax.legend(loc='upper left', fontsize=9)

    # Secondary axis for efficiency gains
    ax2 = ax.twinx()
    ax2.plot(years, efficiency_gains, color=COLORS['grid'], linewidth=2)
    ax2.set_ylabel('Power Efficiency Gains (%)')
    ax2.set_ylim(0, 25)
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)

    # Vertical dashed line at 2024
    ax.axvline(x=2024, linestyle='--', color=COLORS['OrangeLine'], linewidth=1.5)

    # Annotations
    ax.text(2022, 900, 'Power Demand\nIncreasing', fontsize=9, fontweight='bold', 
            ha='center', color=COLORS['primary'])
    ax.text(2018, 850, 'Efficiency Gains\nDecelerating', fontsize=9, fontweight='bold', 
            ha='center', color=COLORS['primary'])

    ax.set_ylim(0, 1100)
    
    return ax, ax2

def plot_training_roofline(ax=None):
    """Visualizes 'Training Roofline Model' (GPT-2 Operations)."""
    if ax is None: fig, ax = plt.subplots()

    # A100 Specs
    peak_flops = 312
    peak_bw = 2.0 # TB/s approx (actually 1.6-2.0 depending on version, using 2.0 from tikz)
    ridge = peak_flops / peak_bw # 156

    # Roofline
    x = np.logspace(0, np.log10(500), 100)
    y_mem = peak_bw * x
    y_compute = np.full_like(x, peak_flops)
    y = np.minimum(y_mem, y_compute)

    ax.plot(x, y, color=COLORS['BlueLine'], linewidth=2.5, label='A100 Roofline')
    
    # Ridge Line
    ax.vlines(ridge, 1, peak_flops, colors=COLORS['BlueLine'], linestyles='--', alpha=0.6)
    ax.text(ridge * 0.9, 2, f"Ridge: {ridge:.0f}", rotation=90, color=COLORS['BlueLine'], fontsize=9, ha='right')
    
    # Moved A100 Peak label to the right to avoid overlap with FlashAttn
    ax.text(450, 340, "A100 Peak (312 TF)", color=COLORS['BlueLine'], fontsize=9, fontweight='bold', ha='right')

    # Points
    ops = [
        {'name': 'Softmax', 'x': 5, 'y': 10, 'color': COLORS['RedLine'], 'pos': 'top', 'offset': (0, 10)},
        {'name': 'LayerNorm', 'x': 10, 'y': 20, 'color': COLORS['RedLine'], 'pos': 'top', 'offset': (0, 10)},
        {'name': 'Std Attention', 'x': 50, 'y': 100, 'color': COLORS['OrangeLine'], 'pos': 'top', 'offset': (-20, 10)}, # Moved left
        {'name': 'MatMul', 'x': 200, 'y': 312, 'color': COLORS['GreenLine'], 'pos': 'bottom', 'offset': (-15, -20)}, # Moved left/down
        {'name': 'FlashAttn', 'x': 300, 'y': 312, 'color': COLORS['GreenLine'], 'pos': 'bottom', 'offset': (15, -20)} # Moved right/down
    ]

    for op in ops:
        ax.scatter(op['x'], op['y'], color=op['color'], s=100, zorder=3, edgecolors='white')
        
        ax.annotate(op['name'], (op['x'], op['y']), xytext=op['offset'], textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold', color=op['color'],
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Arrow for FlashAttention
    # Adjusted text position to be above the arrow to avoid overlap with line
    ax.annotate("", xy=(300, 312), xytext=(50, 100),
                arrowprops=dict(arrowstyle="->", color=COLORS['VioletLine'], lw=2.5))
    ax.text(90, 180, "Flash Attention", color=COLORS['VioletLine'], rotation=32, fontsize=9, fontweight='bold', ha='right')

    # Regions
    ax.text(15, 200, "Memory-bound", color='gray', style='italic', fontsize=10)
    ax.text(300, 180, "Compute-bound", color='gray', style='italic', fontsize=10)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 500)
    ax.set_ylim(1, 400)
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)')
    ax.set_ylabel('Attainable TFLOP/s')
    
    return ax

def plot_compilation_continuum(ax=None):
    """Visualizes 'The Compilation Continuum' (Eager vs JIT vs AOT)."""
    if ax is None: fig, ax = plt.subplots()

    # X-axis: Production Executions (Log Scale)
    x = np.logspace(2, 7, 200) # 100 to 10,000,000

    # Cost Models (Arbitrary Units)
    # Eager: 0 compile + 10e-6 * x
    y_eager = 0 + 10e-6 * x
    
    # JIT: 10 compile + 5e-6 * x
    y_jit = 10 + 5e-6 * x
    
    # AOT: 30 compile + 2e-6 * x
    y_aot = 30 + 2e-6 * x

    # Plot Lines
    ax.plot(x, y_eager, color=COLORS['BlueLine'], linewidth=2.5, label=r'Eager ($T_{compile}=0, T_{exec}=10\mu s$)')
    ax.plot(x, y_jit, color=COLORS['OrangeLine'], linewidth=2.5, label=r'JIT ($T_{compile}=30s, T_{exec}=5\mu s$)')
    ax.plot(x, y_aot, color=COLORS['GreenLine'], linewidth=2.5, label=r'AOT ($T_{compile}=2min, T_{exec}=2\mu s$)')

    # Region Labels
    # Eager wins: x < 2e6. At x=1000, Eager~0, JIT=10. Place in between.
    ax.text(1000, 5, "Eager wins", color=COLORS['BlueLine'], ha='center', va='center', fontweight='bold', fontsize=9)
    
    # JIT wins: 2e6 < x < 6.6e6. At x=4e6, JIT=30, AOT=38. Place in between.
    ax.text(4e6, 34, "JIT wins", color=COLORS['OrangeLine'], ha='center', va='center', fontweight='bold', fontsize=9)
    
    # AOT wins: x > 6.6e6. At x=8e6, AOT=46, JIT=50. Tight gap. Place below AOT line.
    ax.text(8e6, 42, "AOT wins", color=COLORS['GreenLine'], ha='center', va='top', fontweight='bold', fontsize=9)

    # Styling
    ax.set_xscale('log')
    ax.set_xlabel(r'Production Executions ($N_{prod}$)')
    ax.set_ylabel('Total Time (arbitrary units)')
    ax.set_xlim(100, 1e7)
    ax.set_ylim(0, 100)
    
    ax.legend(loc='upper left', fontsize=8)
    
    return ax
