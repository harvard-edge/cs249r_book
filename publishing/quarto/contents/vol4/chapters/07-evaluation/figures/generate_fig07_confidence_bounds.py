#!/usr/bin/env python3
"""
Generate publication-grade vector figure:
fig07_confidence_bounds.svg and fig07_confidence_bounds.pdf

Topic: Clopper-Pearson Non-Asymptotic Confidence Lower Bound and Physical Testing Budget Trade-off
Chapter 7: Evaluation
Palette: Harvard Crimson / ETH Zurich Academic Semantic Palette
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure matplotlib for crisp publication-grade output
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#E2E8F0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.7
plt.rcParams['text.color'] = '#1E293B'
plt.rcParams['axes.labelcolor'] = '#1E293B'
plt.rcParams['xtick.color'] = '#475569'
plt.rcParams['ytick.color'] = '#475569'

# Semantic Palette (Harvard Crimson & ETH Zurich)
NAVY = '#1F407A'
CRIMSON = '#A51C30'
PETROL = '#007A87'
BRONZE = '#B87333'
SLATE = '#475569'
DARK = '#0F172A'
BG_LIGHT = '#F8FAFC'
CARD_BG = '#FFFFFF'
BORDER = '#CBD5E1'

# Tints
CRIMSON_TINT = '#FDF2F4'
NAVY_TINT = '#EEF4FB'
PETROL_TINT = '#E6F4F6'
AMBER_TINT = '#FEF9C3'

def cp_lower_k0(n, alpha=0.05):
    """Clopper-Pearson lower bound for 0 failures: p = alpha^(1/n)"""
    return alpha ** (1.0 / n)

def cp_lower_k1(n, alpha=0.05):
    """Clopper-Pearson lower bound for 1 failure via bisection"""
    if n <= 1:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(50):
        mid = (low + high) / 2.0
        tail = (mid ** n) + n * (mid ** (n - 1)) * (1.0 - mid)
        if tail > alpha:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0

fig = plt.figure(figsize=(15.2, 8.4), facecolor=BG_LIGHT)

# 2 Columns: Left (2 subplots for curves), Right (Testing Budget & Systems Card)
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.05], height_ratios=[1.0, 1.0],
                      left=0.06, right=0.96, bottom=0.08, top=0.88,
                      wspace=0.22, hspace=0.36)

ax_zoom = fig.add_subplot(gs[0, 0], facecolor='white')
ax_wide = fig.add_subplot(gs[1, 0], facecolor='white')
ax_table = fig.add_subplot(gs[:, 1], facecolor='white')

# -------------------------------------------------------------
# Panel (a): Small-Sample Regime (n = 1 to 100)
# -------------------------------------------------------------
n_zoom = np.arange(1, 101)
p_95_zoom = cp_lower_k0(n_zoom, 0.05)
p_99_zoom = cp_lower_k0(n_zoom, 0.01)
p_90_zoom = cp_lower_k0(n_zoom, 0.10)
p_95_1fail = np.array([cp_lower_k1(n, 0.05) for n in n_zoom])

# Shaded Tail Risk Zone
ax_zoom.fill_between(n_zoom, 0.50, p_95_zoom, color=CRIMSON_TINT, alpha=0.95, label='Unexplored Tail Risk / Failure Probability Zone')
ax_zoom.fill_between(n_zoom, p_95_zoom, 1.00, color=NAVY_TINT, alpha=0.85, label='95% Statistically Supported Envelope')

ax_zoom.plot(n_zoom, p_95_zoom, color=NAVY, lw=2.4, label=r'Zero Failures ($k=0$), 95% Bound: $p_L = 0.05^{1/n}$')
ax_zoom.plot(n_zoom, p_99_zoom, color=PETROL, lw=1.6, linestyle='--', label=r'Zero Failures ($k=0$), 99% Bound: $p_L = 0.01^{1/n}$')
ax_zoom.plot(n_zoom, p_95_1fail, color=BRONZE, lw=1.8, linestyle=':', label=r'One Failure ($k=1$), 95% Bound')

# Naive Wald line
ax_zoom.axhline(1.0, color=CRIMSON, lw=1.2, linestyle='-.', alpha=0.7)
ax_zoom.text(3, 0.982, 'Naive Asymptotic Wald Bound (100% ± 0% — Invalid at Boundary)', color=CRIMSON, fontsize=8.0, fontweight='bold')

# Highlight n = 20 milestone
ax_zoom.scatter([20], [0.05**(1/20)], color=CRIMSON, s=80, zorder=6, edgecolor='white', linewidth=1.5)
ax_zoom.axvline(20, color=CRIMSON, linestyle=':', lw=1.2, alpha=0.8)
ax_zoom.annotate(r'$\mathbf{n = 20\text{ clean trials}}$ (100% observed pass)' + '\n'
                 r'• Exact 95% Lower Bound: $\mathbf{86.1\%}$' + '\n'
                 r'• Residual Tail Failure Risk: $\mathbf{13.9\%}$ (~1 in 7)',
                 xy=(20, 0.05**(1/20)), xytext=(28, 0.68),
                 arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=1.4),
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=CRIMSON, lw=1.2),
                 fontsize=8.5, color=DARK)

# Highlight n = 100 milestone
ax_zoom.scatter([100], [0.05**(1/100)], color=NAVY, s=65, zorder=6, edgecolor='white')
ax_zoom.annotate(r'$\mathbf{n = 100}$: $97.05\%$ bound' + '\n' + r'($2.95\%$ tail risk)',
                 xy=(100, 0.05**(1/100)), xytext=(65, 0.88),
                 arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.1),
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=NAVY, lw=0.9),
                 fontsize=8, color=NAVY)

ax_zoom.set_xlim(0, 100)
ax_zoom.set_ylim(0.50, 1.01)
ax_zoom.set_ylabel(r'Success Lower Bound ($p_{\mathrm{lower}}$)', fontsize=9.2, fontweight='bold')
ax_zoom.set_xlabel(r'Physical Trial Count $n$', fontsize=9.2, fontweight='bold')
ax_zoom.set_title(r'(a) Small-Sample Testing Regime & Tail Risk ($n \leq 100$)', fontsize=10.0, fontweight='bold', color=NAVY, loc='left', pad=6)
ax_zoom.grid(True)
ax_zoom.legend(loc='lower right', fontsize=7.2, framealpha=0.95, edgecolor=BORDER)

# -------------------------------------------------------------
# Panel (b): Wide-Scale High-Reliability Regime (n = 10 to 3500, log-x)
# -------------------------------------------------------------
n_wide = np.logspace(1, np.log10(3500), 400)
p_95_wide = cp_lower_k0(n_wide, 0.05)
p_99_wide = cp_lower_k0(n_wide, 0.01)

ax_wide.fill_between(n_wide, 0.85, p_95_wide, color=CRIMSON_TINT, alpha=0.95)
ax_wide.fill_between(n_wide, p_95_wide, 1.0005, color=NAVY_TINT, alpha=0.85)

ax_wide.plot(n_wide, p_95_wide, color=NAVY, lw=2.4, label=r'95% Lower Bound: $p_L = 0.05^{1/n}$')
ax_wide.plot(n_wide, p_99_wide, color=PETROL, lw=1.6, linestyle='--', label=r'99% Lower Bound: $p_L = 0.01^{1/n}$')

# Reference lines for P99 and P99.9 targets
ax_wide.axhline(0.99, color=PETROL, linestyle='--', lw=1.1, alpha=0.8)
ax_wide.text(12, 0.9915, r'Target $P_{99}$ Reliability ($p \geq 0.990$, Failure $\leq 10^{-2}$)', color=PETROL, fontsize=7.8, fontweight='bold')

ax_wide.axhline(0.999, color=CRIMSON, linestyle='--', lw=1.1, alpha=0.8)
ax_wide.text(12, 0.9972, r'Target $P_{99.9}$ Industrial Reliability ($p \geq 0.999$, Failure $\leq 10^{-3}$)', color=CRIMSON, fontsize=7.8, fontweight='bold')

# Milestones: n = 299, n = 1000, n = 2995
ax_wide.scatter([299], [0.990], color=PETROL, s=70, zorder=6, edgecolor='white', linewidth=1.2)
ax_wide.axvline(299, color=PETROL, linestyle=':', lw=1.1, alpha=0.8)
ax_wide.annotate(r'$\mathbf{n = 299\text{ trials}}$' + '\n' + r'$P_{99}$ Achieved (21.6 h)',
                 xy=(299, 0.990), xytext=(70, 0.940),
                 arrowprops=dict(arrowstyle='->', color=PETROL, lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=PETROL, lw=1.0),
                 fontsize=7.8, fontweight='bold', color=PETROL)

ax_wide.scatter([1000], [0.05**(1/1000)], color=BRONZE, s=70, zorder=6, edgecolor='white', linewidth=1.2)
ax_wide.annotate(r'$\mathbf{n = 1{,}000\text{ trials}}$' + '\n' + r'$99.70\%$ bound (Fails $P_{99.9}$)',
                 xy=(1000, 0.05**(1/1000)), xytext=(300, 0.960),
                 arrowprops=dict(arrowstyle='->', color=BRONZE, lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=BRONZE, lw=1.0),
                 fontsize=7.8, fontweight='bold', color=BRONZE)

ax_wide.scatter([2995], [0.999], color=CRIMSON, s=75, zorder=6, edgecolor='white', linewidth=1.5)
ax_wide.axvline(2995, color=CRIMSON, linestyle=':', lw=1.1, alpha=0.8)
ax_wide.annotate(r'$\mathbf{n = 2{,}995\text{ trials}}$' + '\n' + r'$P_{99.9}$ Achieved' + '\n' + r'(216.3 h / 9 days)',
                 xy=(2995, 0.999), xytext=(1050, 0.910),
                 arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=CRIMSON, lw=1.1),
                 fontsize=7.8, fontweight='bold', color=CRIMSON)

ax_wide.set_xscale('log')
ax_wide.set_xlim(10, 3500)
ax_wide.set_ylim(0.85, 1.001)
ax_wide.set_ylabel(r'Success Lower Bound ($p_{\mathrm{lower}}$)', fontsize=9.2, fontweight='bold')
ax_wide.set_xlabel(r'Physical Trial Count $n$ (Log Scale)', fontsize=9.2, fontweight='bold')
ax_wide.set_title(r'(b) High-Reliability Asymptote & The Nonstationarity Barrier ($n \to 3{,}000$)', fontsize=10.0, fontweight='bold', color=NAVY, loc='left', pad=6)
ax_wide.grid(True, which='both')
ax_wide.legend(loc='lower left', fontsize=7.2, framealpha=0.95, edgecolor=BORDER)

# -------------------------------------------------------------
# Panel (c): Systems Trade-off & Physical Cost Analysis
# -------------------------------------------------------------
ax_table.axis('off')

card_rect = patches.FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                                    boxstyle='round,pad=0.015',
                                    facecolor=CARD_BG, edgecolor=BORDER, lw=1.2)
ax_table.add_patch(card_rect)

ax_table.text(0.04, 0.95, 'PHYSICAL TESTING BUDGET & STATISTICAL LIMITS',
              fontsize=10.2, fontweight='bold', color=NAVY)
ax_table.text(0.04, 0.920, r'Trial unit: $50\,\mathrm{s}$ run + $120\,\mathrm{s}$ reset + $30\,\mathrm{s}$ zero + $60\,\mathrm{s}$ cooling = $260\,\mathrm{s}$ ($4.33\,\mathrm{min}$)',
              fontsize=7.4, color=SLATE, style='italic')

# Table Header
header_y = 0.86
col_x = [0.04, 0.20, 0.38, 0.60, 0.79]
headers = ['Trials (n)', '95% Bound', 'Tail Risk', 'Wall-Clock Time', 'Hardware State']

rect_hdr = patches.Rectangle((0.025, header_y - 0.026), 0.95, 0.045,
                             facecolor=NAVY_TINT, edgecolor='none')
ax_table.add_patch(rect_hdr)

for x, h in zip(col_x, headers):
    ax_table.text(x, header_y - 0.013, h, fontsize=7.5, fontweight='bold', color=NAVY)

rows = [
    ('n = 20', '86.10%', '13.90% (1 in 7)', '1.4 hours', 'Negligible wear', CRIMSON_TINT, CRIMSON),
    ('n = 100', '97.05%', '2.95% (1 in 34)', '7.2 hours', 'Minor thermal rise', 'white', DARK),
    ('n = 299', '99.00%', '1.00% (1 in 100)', '21.6 hours', 'Gear backlash shifts', PETROL_TINT, PETROL),
    ('n = 1,000', '99.70%', '0.30% (3 in 1000)', '72.2 h (3.0 d)', 'Tire friction drops 15%', 'white', BRONZE),
    ('n = 2,995', '99.90%', '0.10% (1 in 1000)', '216.3 h (9.0 d)', 'Exceeds bearing life!', CRIMSON_TINT, CRIMSON)
]

row_y = 0.78
for r in rows:
    bg_rect = patches.Rectangle((0.025, row_y - 0.028), 0.95, 0.048,
                                facecolor=r[5], edgecolor=BORDER, lw=0.5)
    ax_table.add_patch(bg_rect)
    ax_table.text(col_x[0], row_y - 0.013, r[0], fontsize=7.4, fontweight='bold', color=r[6])
    ax_table.text(col_x[1], row_y - 0.013, r[1], fontsize=7.4, fontweight='bold', color=DARK)
    ax_table.text(col_x[2], row_y - 0.013, r[2], fontsize=7.1, color=SLATE)
    ax_table.text(col_x[3], row_y - 0.013, r[3], fontsize=7.1, color=DARK)
    ax_table.text(col_x[4], row_y - 0.013, r[4], fontsize=7.1, fontweight='bold', color=r[6])
    row_y -= 0.055

# Bottom Section: Three Architectural Remedies
rem_top = 0.44
ax_table.text(0.04, rem_top, 'THE THREE ARCHITECTURAL REMEDIES FOR PHYSICAL AI',
              fontsize=9.0, fontweight='bold', color=NAVY)

remedies = [
    ('1. Narrow Operational Domain (ODD)',
     'Restrict machine to benign velocities and tight environmental bounds where modest sample counts (n = 50-200) provide defensible statistical coverage.',
     BRONZE),
    ('2. Parallel HIL / Fleet Testing',
     'Distribute runs across multiple test bays; beware that shared batch tolerances, floor finishes, and weather induce common-mode spatial correlations.',
     PETROL),
    ('3. Enforce Deterministic Gate in Nervous System',
     'Delegate safety invariants, CBFs, and emergency stops to trusted real-time MCU reflex loops (1 kHz), removing unachievable reliability claims from the learned Brain.',
     CRIMSON)
]

rem_y = rem_top - 0.042
for title, desc, color in remedies:
    box = patches.FancyBboxPatch((0.03, rem_y - 0.082), 0.94, 0.082,
                                 boxstyle='round,pad=0.015',
                                 facecolor=BG_LIGHT, edgecolor=color, lw=1.1)
    ax_table.add_patch(box)
    ax_table.text(0.05, rem_y - 0.022, title, fontsize=7.8, fontweight='bold', color=color)
    ax_table.text(0.05, rem_y - 0.066, desc, fontsize=6.9, color=SLATE)
    rem_y -= 0.102

fig.suptitle('Non-Asymptotic Clopper-Pearson Bounds & Physical Testing Budget Limits',
             fontsize=12.5, fontweight='bold', color=NAVY, y=0.96)

out_dir = '/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/07-evaluation/figures'
os.makedirs(out_dir, exist_ok=True)
svg_path = os.path.join(out_dir, 'fig07_confidence_bounds.svg')
pdf_path = os.path.join(out_dir, 'fig07_confidence_bounds.pdf')

plt.savefig(svg_path, format='svg', bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print('Successfully generated:')
print('  ', svg_path)
print('  ', pdf_path)
