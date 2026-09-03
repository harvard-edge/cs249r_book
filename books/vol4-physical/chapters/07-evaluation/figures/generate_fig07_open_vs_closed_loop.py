#!/usr/bin/env python3
"""
Generate publication-grade vector figure:
fig07_open_vs_closed_loop.svg and fig07_open_vs_closed_loop.pdf

Topic: Open-Loop Error Masking vs. Closed-Loop Compounding State Divergence
Chapter 7: Evaluation
Palette: Harvard Crimson / ETH Zurich Academic Semantic Palette
"""

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
GRAY_TINT = '#F1F5F9'

fig = plt.figure(figsize=(15.2, 8.4), facecolor=BG_LIGHT)

# Layout: Two main horizontal rollout swimlanes (Top: Open-Loop, Middle: Closed-Loop), Bottom: Summary Card
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.35, 0.95],
                      left=0.06, right=0.96, bottom=0.06, top=0.92,
                      hspace=0.38)

ax_open = fig.add_subplot(gs[0, 0], facecolor='white')
ax_closed = fig.add_subplot(gs[1, 0], facecolor='white')
ax_summary = fig.add_subplot(gs[2, 0], facecolor='white')

# -------------------------------------------------------------
# Panel (a): Open-Loop Evaluation (Offline Trace Replay)
# -------------------------------------------------------------
x_expert = np.linspace(0, 10, 100)
y_expert = np.zeros_like(x_expert)

# Plot Corridor Boundaries
ax_open.axhline(0.15, color='#94A3B8', linestyle='--', lw=1.2, label='Corridor Clearance Limit (+15 cm)')
ax_open.axhline(-0.15, color='#94A3B8', linestyle='--', lw=1.2)
ax_open.fill_between(x_expert, 0.15, 0.22, color='#E2E8F0', alpha=0.6)
ax_open.fill_between(x_expert, -0.22, -0.15, color='#E2E8F0', alpha=0.6)

# Expert centerline
ax_open.plot(x_expert, y_expert, color=NAVY, lw=2.5, label='Expert Trajectory ($s_t^* \in \mathcal{D}_{\mathrm{demo}}$, $y=0$)')

# Sample points where open-loop policy is tested
test_steps = np.arange(0.5, 9.6, 1.0)
for i, xs in enumerate(test_steps):
    # Expert state point
    ax_open.scatter([xs], [0], color=NAVY, s=40, zorder=5)
    # Predicted action with 1.0 deg bias
    dx = 0.5
    dy = dx * np.tan(np.radians(1.0)) * 6.0 # scaled for visual clarity
    ax_open.arrow(xs, 0, dx, dy, head_width=0.015, head_length=0.08,
                  fc=CRIMSON, ec=CRIMSON, lw=1.3, zorder=6, length_includes_head=True)
    # External reset arrow back to expert centerline at next step
    if i < len(test_steps) - 1:
        next_x = test_steps[i+1]
        ax_open.annotate('', xy=(next_x, 0), xytext=(xs + dx, dy),
                         arrowprops=dict(arrowstyle='->', color=PETROL, lw=1.0, linestyle=':'))

# Open-Loop Score Card Badge on the right
card_ol = patches.FancyBboxPatch((7.6, -0.12), 2.25, 0.22,
                                 boxstyle='round,pad=0.02',
                                 facecolor=NAVY_TINT, edgecolor=NAVY, lw=1.2, zorder=7)
ax_open.add_patch(card_ol)
ax_open.text(7.75, 0.03, 'OFFLINE METRIC RESULT', fontsize=8.2, fontweight='bold', color=NAVY)
ax_open.text(7.75, -0.025, r'• Heading Bias $\epsilon_\theta = +1.0^\circ$ ($0.0175\,\mathrm{rad}$)', fontsize=7.6, color=DARK)
ax_open.text(7.75, -0.075, r'• $\mathrm{MSE} = 3.05 \times 10^{-4}\,\mathrm{rad}^2$ ($\mathbf{99.7\%}$ Accuracy)', fontsize=7.6, fontweight='bold', color=PETROL)

ax_open.text(0.3, -0.10, r'At each step $t$: input is reset to expert state $s_t^*$; predicted action $\hat{a}_t$ is discarded without physical execution.',
             fontsize=8.0, color=SLATE, style='italic')

ax_open.set_xlim(-0.2, 10.2)
ax_open.set_ylim(-0.20, 0.20)
ax_open.set_ylabel('Lateral Offset $y$ (m)', fontsize=9.0, fontweight='bold')
ax_open.set_title(r'(a) Open-Loop Evaluation (Dataset Replay): Errors are Memoryless and Discarded at Every Clock Step',
                  fontsize=10.0, fontweight='bold', color=NAVY, loc='left', pad=6)
ax_open.grid(True, alpha=0.5)

# -------------------------------------------------------------
# Panel (b): Closed-Loop Physical Execution (Endogenous Trajectory)
# -------------------------------------------------------------
# Kinematics: y(t) = v * sin(1 deg) * t, x(t) = v * cos(1 deg) * t, v = 1.0 m/s
# Collision occurs when y(x) = 0.15 m => x = 0.15 / tan(1 deg) = 8.59 m ~ 8.6 m
x_cl_valid = np.linspace(0, 8.59, 200)
y_cl_valid = x_cl_valid * np.tan(np.radians(1.0))

# Corridor Boundaries
ax_closed.axhline(0.15, color=CRIMSON, linestyle='-', lw=1.6, label='Physical Wall Clearance Boundary ($+15\,\mathrm{cm}$)')
ax_closed.axhline(-0.15, color='#94A3B8', linestyle='--', lw=1.2)
ax_closed.fill_between(np.linspace(0, 10, 100), 0.15, 0.22, color='#FEE2E2', alpha=0.7)
ax_closed.fill_between(np.linspace(0, 10, 100), -0.22, -0.15, color='#E2E8F0', alpha=0.6)

# Nominal expert path for reference
ax_closed.plot(x_expert, y_expert, color='#94A3B8', linestyle=':', lw=1.5, label='Nominal Expert Centerline ($y=0$)')

# Shaded drift area
ax_closed.fill_between(x_cl_valid, 0, y_cl_valid, color=CRIMSON_TINT, alpha=0.7)

# Actual Closed-Loop Trajectory
ax_closed.plot(x_cl_valid, y_cl_valid, color=CRIMSON, lw=2.6, label=r'Closed-Loop Rollout: $\dot{y} = v \sin(1.0^\circ) \implies y(t) = 0.0175 \cdot t$')

# Robot poses along trajectory
robot_steps = [0.0, 2.5, 5.0, 7.0]
for xs in robot_steps:
    ys = xs * np.tan(np.radians(1.0))
    ax_closed.scatter([xs], [ys], color=NAVY, s=55, zorder=6, edgecolor='white')
    # Small heading arrow
    dx = 0.45
    dy = dx * np.tan(np.radians(1.0))
    ax_closed.arrow(xs, ys, dx, dy, head_width=0.015, head_length=0.08,
                    fc=CRIMSON, ec=CRIMSON, lw=1.4, zorder=7, length_includes_head=True)

# Out-of-Distribution Annotation at x = 5.0 m
ax_closed.annotate(r'$\mathbf{t = 5.0\,\mathrm{s}}$ ($x = 5.0\,\mathrm{m}$)' + '\n' +
                   r'• Lateral drift: $y = 8.7\,\mathrm{cm}$' + '\n' +
                   r'• State $s_t \notin \mathcal{D}_{\mathrm{demo}}$ (OOD!)',
                   xy=(5.0, 5.0 * np.tan(np.radians(1.0))), xytext=(3.2, 0.08),
                   arrowprops=dict(arrowstyle='->', color=BRONZE, lw=1.3),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=BRONZE, lw=1.1),
                   fontsize=7.8, color=DARK)

# Collision Point at x = 8.6 m
x_crash = 8.594
y_crash = 0.15
ax_closed.scatter([x_crash], [y_crash], color=CRIMSON, s=160, marker='*', zorder=10, edgecolor='black', linewidth=0.8)
ax_closed.annotate(r'$\mathbf{CRASH\ AT\ t = 8.6\,\mathrm{s}}$ ($x = 8.6\,\mathrm{m}$)' + '\n' +
                   r'• Wall collision at $1.0\,\mathrm{m/s}$' + '\n' +
                   r'• $\mathbf{0\%}$ Closed-Loop Task Success!',
                   xy=(x_crash, y_crash), xytext=(6.5, 0.03),
                   arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=1.5),
                   bbox=dict(boxstyle='round,pad=0.35', facecolor=CRIMSON_TINT, edgecolor=CRIMSON, lw=1.3),
                   fontsize=8.2, fontweight='bold', color=CRIMSON)

# Post-crash truncated trajectory
ax_closed.plot([x_crash, 10.0], [y_crash, y_crash + (10.0 - x_crash)*np.tan(np.radians(1.0))],
               color=CRIMSON, linestyle=':', lw=1.5, alpha=0.4)

ax_closed.set_xlim(-0.2, 10.2)
ax_closed.set_ylim(-0.20, 0.20)
ax_closed.set_ylabel('Lateral Offset $y$ (m)', fontsize=9.0, fontweight='bold')
ax_closed.set_xlabel('Forward Corridor Position $x$ (m) [at nominal speed $v = 1.0\,\mathrm{m/s}$]', fontsize=9.0, fontweight='bold')
ax_closed.set_title(r'(b) Closed-Loop Execution (Physical Rollout): Small Systematic Bias Compounds into Catastrophic Collision',
                    fontsize=10.0, fontweight='bold', color=CRIMSON, loc='left', pad=6)
ax_closed.grid(True, alpha=0.5)

# -------------------------------------------------------------
# Panel (c): Systems Comparison Card
# -------------------------------------------------------------
ax_summary.axis('off')

box_sum = patches.FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle='round,pad=0.015',
                                 facecolor=CARD_BG, edgecolor=BORDER, lw=1.2)
ax_summary.add_patch(box_sum)

ax_summary.text(0.03, 0.90, 'STRUCTURAL COMPARISON: OPEN-LOOP REPLAY VS. CLOSED-LOOP EMBODIED EXECUTION',
                fontsize=9.8, fontweight='bold', color=NAVY)

# 4 Column Comparison Boxes
cols = [
    ('1. Input State Source',
     'Open Loop: Stored disk logs ($s_t^* \sim \mathcal{D}_{\mathrm{demo}}$)\n'
     'Closed Loop: Endogenous state ($s_{t+1} = f(s_t, a_t)$)',
     NAVY),
    ('2. Error Propagation',
     'Open Loop: Memoryless, zeroed at every step\n'
     'Closed Loop: Compounding: $y(t) = \int v \sin(\theta) dt$',
     BRONZE),
    ('3. State Support',
     'Open Loop: Evaluates only on expert trajectories\n'
     'Closed Loop: Perturbation pushes policy to unvisited states',
     PETROL),
    ('4. Evaluated Outcome',
     'Open Loop: 99.7% Directional Agreement\n'
     'Closed Loop: 0% Task Success (Wall Collision)',
     CRIMSON)
]

col_x_pos = [0.03, 0.27, 0.52, 0.76]
for (title, desc, color), xp in zip(cols, col_x_pos):
    cbox = patches.FancyBboxPatch((xp, 0.08), 0.22, 0.72,
                                  boxstyle='round,pad=0.012',
                                  facecolor=BG_LIGHT, edgecolor=color, lw=1.0)
    ax_summary.add_patch(cbox)
    ax_summary.text(xp + 0.012, 0.68, title, fontsize=7.8, fontweight='bold', color=color)
    ax_summary.text(xp + 0.012, 0.20, desc, fontsize=7.0, color=DARK, linespacing=1.35)

fig.suptitle('Open-Loop Error Masking vs. Closed-Loop Compounding State Divergence',
             fontsize=12.5, fontweight='bold', color=NAVY, y=0.96)

out_dir = '/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/07-evaluation/figures'
os.makedirs(out_dir, exist_ok=True)
svg_path = os.path.join(out_dir, 'fig07_open_vs_closed_loop.svg')
pdf_path = os.path.join(out_dir, 'fig07_open_vs_closed_loop.pdf')

plt.savefig(svg_path, format='svg', bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print('Successfully generated:')
print('  ', svg_path)
print('  ', pdf_path)
