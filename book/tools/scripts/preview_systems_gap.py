import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add mlsys directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'quarto'))
from mlsys import viz

# Output directory
OUTPUT_DIR = "book/quarto/assets/preview_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_systems_gap(ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    years = np.linspace(2012, 2024.5, 100)

    # 1. Moore's Law (CPU Baseline)
    # 2012 (Xeon E5-2690): ~0.37 TF -> 2022 (Xeon 8480+): ~7 TF. Growth ~19x in 10y.
    cpu_slope = np.log10(19) / 10
    moore = 1.0 * 10 ** (cpu_slope * (years - 2012))

    # 2. Huang's Law (GPU Peak)
    # 2012 (K20X): 3.95 TF -> 2022 (H100): 989 TF. Growth ~250x in 10y.
    gpu_slope = np.log10(250) / 10
    huang = 1.0 * 10 ** (gpu_slope * (years - 2012))

    # 3. Model Demand
    # 2012 (AlexNet): 4.3e16 -> 2023 (GPT-4): 2e25. Growth ~4.6e8x in 11y.
    demand_slope = np.log10(4.6e8) / 11
    demand = 1.0 * 10 ** (demand_slope * (years - 2012))

    ax.plot(years, moore, ':', color=viz.COLORS['grid'], label="CPU Performance Trend", linewidth=2)
    ax.plot(years, huang, '--', color=viz.COLORS['BlueLine'], label="GPU Peak (Huang's Law)", linewidth=2.5)
    ax.plot(years, demand, '-', color=viz.COLORS['RedLine'], label="Model Demand (Scaling Laws)", linewidth=3)

    ax.fill_between(years, huang, demand, where=(demand > huang), color=viz.COLORS['VioletL'], alpha=0.3)

    ax.set_yscale('log')
    ax.set_xlabel('Year')
    ax.set_ylabel('Relative Growth (2012 = 1.0)')
    ax.set_xlim(2012, 2024.5)
    ax.set_ylim(0.5, 1e10)

    gap_x = 2020.0
    h_val = 10 ** (gpu_slope * (gap_x - 2012))
    d_val = 10 ** (demand_slope * (gap_x - 2012))
    gap_y = np.sqrt(h_val * d_val)

    ax.text(
        gap_x,
        gap_y,
        "THE SYSTEMS GAP\n(Closed by Parallelism,\nArchitecture & Co-design)",
        ha='center',
        va='center',
        fontweight='bold',
        color=viz.COLORS['VioletLine'],
        fontsize=8,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2),
    )

    points = [
        (2012, 1.0, "AlexNet"),
        (2015, 10 ** (demand_slope * 3), "ResNet"),
        (2017, 10 ** (demand_slope * 5), "Transformer"),
        (2020, 10 ** (demand_slope * 8), "GPT-3"),
        (2023, 10 ** (demand_slope * 11), "GPT-4"),
    ]

    model_offsets = {
        "AlexNet": (0, 10),
        "Transformer": (-15, 10),
        "GPT-3": (-15, 8),
        "GPT-4": (0, 8),
    }

    for y, v, l in points:
        ax.scatter(y, v, color=viz.COLORS['RedLine'], s=25, zorder=5, edgecolors='white')
        xytext = model_offsets.get(l, (0, 8))
        ax.annotate(
            l,
            (y, v),
            xytext=xytext,
            textcoords='offset points',
            fontsize=8,
            ha='center',
            color=viz.COLORS['RedLine'],
            fontweight='bold',
        )

    hw_points = [
        (2012, 1.0, "K20X"),
        (2016, 10 ** (gpu_slope * 4), "P100"),
        (2022, 10 ** (gpu_slope * 10), "H100"),
    ]

    hw_offsets = {
        "K20X": (0, -15),
        "P100": (0, -15),
        "H100": (0, -15),
    }

    for y, v, l in hw_points:
        ax.scatter(y, v, color=viz.COLORS['BlueLine'], s=25, zorder=5, edgecolors='white')
        xytext = hw_offsets.get(l, (0, -12))
        ax.annotate(
            l,
            (y, v),
            xytext=xytext,
            textcoords='offset points',
            fontsize=8,
            ha='center',
            color=viz.COLORS['BlueLine'],
            fontweight='bold',
        )

    ax.legend(loc='lower right', fontsize=8)
    return ax


# Set style
viz.set_book_style()

print("Generating Systems Gap...")
plot_systems_gap()
plt.savefig(f"{OUTPUT_DIR}/systems_gap.png")
plt.close('all')

print(f"Plot saved to {OUTPUT_DIR}/systems_gap.png")
