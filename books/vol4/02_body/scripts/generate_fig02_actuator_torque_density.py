#| label: fig-02-actuator-torque-density-vs-gear-ratio
#| echo: false
#| warning: false
#| message: false
# ┌─────────────────────────────────────────────────────────────────────────────
# │ ACTUATOR DYNAMICS: TORQUE DENSITY VS. GEAR RATIO (FIGURE)
# ├─────────────────────────────────────────────────────────────────────────────
# │ Context: Section 2.5, @fig-02-actuator-torque-density-vs-gear-ratio.
# │ Goal:    Show the physical trade-off between delivered specific torque density
# │          and reflected rotor inertia (N^2), contrasting stiff industrial
# │          reducers with backdrivable Quasi-Direct Drive (QDD) actuators.
# └─────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0

fig, ax = plt.subplots(figsize=(14, 8), dpi=200, facecolor="#FFFFFF")
ax.set_facecolor("#FFFFFF")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#E2E8F0")
ax.set_axisbelow(True)

# 1. Regimes
ax.axvspan(0.85, 2.0, color="#F8FAFC", alpha=0.9, zorder=0)
ax.axvspan(3.5, 14.0, color="#ECFDF5", alpha=0.7, zorder=0)
ax.axvspan(35.0, 260.0, color="#FFFBEB", alpha=0.6, zorder=0)

regime_y = 126
ax.text(
    1.30,
    regime_y,
    "REGIME I: DIRECT DRIVE\nBackdrivable · Zero Backlash\nLow Torque (<5 Nm/kg)",
    ha="center",
    va="center",
    fontsize=8.0,
    fontweight="bold",
    color="#64748B",
)

ax.text(
    7.0,
    regime_y,
    "REGIME II: QUASI-DIRECT DRIVE (QDD)\nProprioceptive Sweet Spot ($N^2 \\leq 100$)\nHigh Torque Density + Contact Transparency",
    ha="center",
    va="center",
    fontsize=8.2,
    fontweight="bold",
    color="#065F46",
)

ax.text(
    105.0,
    regime_y,
    "REGIME III & IV: HIGH-REDUCTION INDUSTRIAL & SEA\nStiff Harmonic / Cycloidal & Series Elastic ($N^2 \\geq 2,500$)\nNon-Backdrivable · Impact Shock Fragile · Positional Rigidity",
    ha="center",
    va="center",
    fontsize=8.2,
    fontweight="bold",
    color="#92400E",
)

# 2. Thresholds
ax.axvline(10.0, color="#10B981", linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)
ax.text(
    10.0,
    112,
    "Proprioceptive Sensing Limit\n($N \\leq 10, N^2 \\leq 100$)",
    ha="center",
    va="center",
    fontsize=7.8,
    fontweight="bold",
    color="#047857",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#ECFDF5", edgecolor="#10B981", alpha=0.9),
)

ax.axvline(50.0, color="#EF4444", linestyle=":", linewidth=1.5, alpha=0.85, zorder=2)
ax.text(
    50.0,
    112,
    "Non-Backdrivable Boundary\n($N \\geq 50, N^2 \\geq 2,500$)",
    ha="center",
    va="center",
    fontsize=7.8,
    fontweight="bold",
    color="#B91C1C",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#EF4444", alpha=0.9),
)

def add_badge(ax, text, xy, xytext, color="#1E293B", fontsize=7.6, arrow=True, arrow_color="#94A3B8"):
    bbox = dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.96, linewidth=0.8)
    ap = dict(arrowstyle="->", color=arrow_color, linewidth=1.1, shrinkA=3, shrinkB=4) if arrow else None
    ax.annotate(
        text, xy=xy, xytext=xytext, fontsize=fontsize, fontweight="normal", color=color, bbox=bbox, arrowprops=ap, ha="center", va="center", zorder=10
    )

actuators = [
    {
        "name": "CMU Direct Drive Arm (1983)",
        "N": 1.0,
        "plot_N": 1.0,
        "cont": 3.2,
        "peak": 9.0,
        "color": "#475569",
        "marker": "s",
        "badge_xytext": (1.35, 14.5),
        "text": "1983 · CMU Direct Drive\nN = 1:1 · Cont: 3.2 Nm/kg\nAsada & Kanade; zero backlash",
    },
    {
        "name": "Maxon EC90 Flat (Direct Drive)",
        "N": 1.0,
        "plot_N": 1.0,
        "cont": 1.8,
        "peak": 5.0,
        "color": "#1E293B",
        "marker": "o",
        "badge_xytext": (1.35, 4.2),
        "text": "Maxon EC90 Flat (DD)\nN = 1:1 · Cont: 1.8 Nm/kg\nFrameless BLDC outrunner",
    },
    {
        "name": "MIT Cheetah 3 (2017)",
        "N": 7.67,
        "plot_N": 7.67,
        "cont": 15.5,
        "peak": 36.4,
        "color": "#1E793C",
        "marker": "o",
        "badge_xytext": (3.2, 50.0),
        "text": "2017 · MIT Cheetah 3\nN = 7.67:1 · Cont: 15.5 Nm/kg (Peak: 36.4)\nKatz & Wensing; planetary QDD",
    },
    {
        "name": "MIT Mini Cheetah (2019)",
        "N": 6.0,
        "plot_N": 6.0,
        "cont": 12.5,
        "peak": 30.9,
        "color": "#008080",
        "marker": "o",
        "badge_xytext": (4.2, 10.0),
        "text": "2019 · MIT Mini Cheetah\nN = 6.0:1 · Cont: 12.5 Nm/kg (Peak: 31)\nBackdrivable dynamic quad",
    },
    {
        "name": "Unitree Go1/A1 M8010 (2021)",
        "N": 6.33,
        "plot_N": 6.33,
        "cont": 25.0,
        "peak": 69.8,
        "color": "#0284C7",
        "marker": "^",
        "badge_xytext": (3.2, 84.0),
        "text": "2021 · Unitree M8010-6\nN = 6.33:1 · Cont: 25 Nm/kg (Peak: 70)\nCommercial agile quadruped leg",
    },
    {
        "name": "Unitree H1 Hip M107 (2023)",
        "N": 9.0,
        "plot_N": 9.0,
        "cont": 31.5,
        "peak": 94.7,
        "color": "#A51C30",
        "marker": "*",
        "badge_xytext": (12.2, 96.0),
        "text": "2023 · Unitree H1 (M107)\nN = 9.0:1 · Cont: 31.5 Nm/kg (Peak: 95)\nHumanoid high-torque hip/knee",
    },
    {
        "name": "Boston Dynamics Electric Atlas (2024)",
        "N": 8.5,
        "plot_N": 8.5,
        "cont": 35.0,
        "peak": 78.0,
        "color": "#004B87",
        "marker": "D",
        "badge_xytext": (13.5, 68.0),
        "text": "2024 · BD Electric Atlas\nN ≈ 8.5:1 · Cont: 35 Nm/kg (Peak: 78)\nCustom high-power QDD joint",
    },
    {
        "name": "Universal Robots UR5 (2015)",
        "N": 101.0,
        "plot_N": 108.0,
        "cont": 22.7,
        "peak": 68.2,
        "color": "#B45309",
        "marker": "s",
        "badge_xytext": (56.0, 72.0),
        "text": "2015 · Universal Robots UR5\nN = 101:1 · Cont: 22.7 Nm/kg (Peak: 68)\nDual-encoder harmonic drive cobot",
    },
    {
        "name": "Franka Emika Panda (2018)",
        "N": 100.0,
        "plot_N": 100.0,
        "cont": 22.2,
        "peak": 48.3,
        "color": "#D97706",
        "marker": "D",
        "badge_xytext": (56.0, 44.0),
        "text": "2018 · Franka Emika Panda\nN = 100:1 · Cont: 22.2 Nm/kg (Peak: 48)\nStrain wave gear + joint torque sensor",
    },
    {
        "name": "ANYbotics ANYdrive (2018)",
        "N": 100.0,
        "plot_N": 92.0,
        "cont": 21.0,
        "peak": 42.1,
        "color": "#6B21A8",
        "marker": "s",
        "badge_xytext": (56.0, 18.0),
        "text": "2018 · ANYbotics ANYdrive\nN = 100:1 · Cont: 21 Nm/kg (Peak: 42)\nHarmonic drive + titanium spring",
    },
    {
        "name": "KUKA LBR iiwa 7 (2014)",
        "N": 160.0,
        "plot_N": 150.0,
        "cont": 25.0,
        "peak": 55.0,
        "color": "#E11D48",
        "marker": "o",
        "badge_xytext": (180.0, 24.0),
        "text": "2014 · KUKA LBR iiwa 7\nN = 160:1 · Cont: 25 Nm/kg (Peak: 55)\nInternal strain-gauge torque feedback",
    },
    {
        "name": "Nabtesco RV-25N Cycloidal (Industrial)",
        "N": 161.0,
        "plot_N": 165.0,
        "cont": 45.0,
        "peak": 118.0,
        "color": "#A51C30",
        "marker": "^",
        "badge_xytext": (125.0, 88.0),
        "text": "Nabtesco RV Cycloidal\nN = 161:1 · Cont: 45 Nm/kg (Peak: 118)\nPrecision robotics reducer; stiff & heavy",
    },
]

for act in actuators:
    ax.plot([act["plot_N"], act["plot_N"]], [act["cont"], act["peak"]], color=act["color"], linewidth=2.0, alpha=0.6, zorder=5)
    ax.scatter(act["plot_N"], act["cont"], color=act["color"], s=80, marker=act["marker"], edgecolor="#FFFFFF", linewidth=1.2, zorder=6)
    ax.scatter(act["plot_N"], act["peak"], color="#FFFFFF", s=90, marker=act["marker"], edgecolor=act["color"], linewidth=2.0, zorder=6)
    ax.scatter(act["plot_N"], act["peak"], color=act["color"], s=18, marker="o", zorder=7)
    add_badge(
        ax,
        act["text"],
        xy=(act["plot_N"], (act["cont"] + act["peak"]) / 2.0),
        xytext=act["badge_xytext"],
        color="#1E293B",
        fontsize=7.6,
        arrow=True,
        arrow_color=act["color"],
    )

ax2 = ax.twinx()
ax2.set_yscale("log")
ax2.set_ylim(0.8, 90000)
ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_color("#64748B")

n_curve = np.logspace(np.log10(0.85), np.log10(260), 200)
j_refl = n_curve**2
ax2.plot(n_curve, j_refl, color="#94A3B8", linestyle=":", linewidth=1.2, alpha=0.5, zorder=1)

ax2.set_yticks([1, 10, 100, 1000, 10000, 40000])
ax2.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, p: f"{int(y):,}×"))
ax2.set_ylabel(
    "Reflected Rotor Inertia Multiplier $J_{\\mathrm{refl}} / J_{\\mathrm{rotor}} = N^2$ (log scale)",
    fontsize=10,
    fontweight="bold",
    color="#64748B",
    labelpad=10,
)
ax2.tick_params(colors="#64748B", labelsize=8.5)

ax.set_xscale("log")
ax.set_xlim(0.85, 260)
ax.set_ylim(0, 138)

ax.set_xticks([1, 2, 4, 6, 8, 10, 20, 50, 100, 160, 250])
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x)}:1"))

ax.set_xlabel("Transmission Gear Ratio $N:1$ (log scale)", fontsize=11, fontweight="bold", labelpad=8)
ax.set_ylabel("Actuator Specific Torque Density (N·m / kg)", fontsize=11, fontweight="bold", labelpad=8)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#475569', markersize=8, label='Continuous Torque Density ($S1$ Thermal Limit)'),
    Line2D(
        [0],
        [0],
        marker='o',
        color='#475569',
        markerfacecolor='w',
        markeredgewidth=1.8,
        markersize=9,
        label='Peak Intermittent Torque Density (Transient / Jump)',
    ),
    Line2D([0], [0], color='#475569', lw=2, label='Continuous-to-Peak Operating Envelope'),
    Line2D([0], [0], color='#10B981', lw=1.5, ls='--', label='Proprioceptive Current Sensing Boundary ($N \\leq 10$)'),
    Line2D([0], [0], color='#EF4444', lw=1.5, ls=':', label='Non-Backdrivable / Impact Shock Boundary ($N \\geq 50$)'),
]
legend = ax.legend(
    handles=legend_elements,
    bbox_to_anchor=(0.40, 0.04),
    loc="lower left",
    frameon=True,
    framealpha=0.96,
    edgecolor="#CBD5E1",
    fontsize=8.0,
    title="Actuator Mechanical Metrics",
    title_fontsize=8.5,
)
legend.get_title().set_fontweight("bold")
legend.set_zorder(15)

plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.10, left=0.08, right=0.92)
plt.show()
