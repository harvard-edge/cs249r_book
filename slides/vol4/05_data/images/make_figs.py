#!/usr/bin/env python3
"""Figures for Lecture 5 (Physical Data), Volume IV.

Data plots use slideplot (book semantic colors); schematics are hand-written
SVG in the Volume IV causal-boundary palette (proposal blue, nervous-system
green, plant slate, hazard red, sensing petrol). Every derived value comes
from calcs.py. Sources:
  - LeRobot Hub counts and episode histogram, Open X-Embodiment composition:
    Hugging Face Hub API (2026-10-12) and Open X-Embodiment overview sheet,
    as in the chapter's figures.
  - Pairing lag, teleoperation loop, yield, wear, bytes, contact grid,
    scenario table, takeover timeline, normalization: the MLSysIM scenarios
    the chapter reads (Napkin Math 5.1 to 5.3, Sec. 5.6 to 5.9), illustrative
    values labeled so.
  - Autonomous success with intervention data: cited literature
    (Hoque et al. 2022, ThriftyDAgger, Table 3, physical cable routing).
Run from this folder: python3 make_figs.py
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
from mlsysim.core.series_loader import load_series
_NUMS = {'droid_recal': 36000.0, 'droid_traj': 76000.0, 'droid_hours': 350.0, 'droid_scenes': 564.0, 'f_trip': 15.0, 'f_latch_limit': 100.0, 'v_guard': 0.03, 'v_fast': 0.1, 'tick_khz': 1.0, 'chunk_hz': 20.0, 's_steps': 500.0, 'episode_s': 5.0, 'ctrl_hz': 100.0, 'eps': 0.001, 'u_rec': 1.0, 'bound_open': 0.5, 'bound_bc': 250.0, 'bound_dagger': 0.5, 'eps_vacuous': 0.002, 's10_steps': 1000.0, 'eps10_vacuous': 0.001, 'ess_records': 3600000.0, 'ess_records_m': 3.6, 'ess_n': 1800.0, 'rho': 0.999, 'tick_ms': 1.0, 'hub_2024': 1085.0, 'hub_2025': 26548.0, 'hub_now': 80132.0, 'hub_growth': 74.0, 'hub_counted': 59031.0, 'hub_small': 16.0, 'hub_large': 512.0, 'hub_small_pct': 58.0, 'hub_large_pct': 2.0, 'hub_median_lo': 8.0, 'hub_median_hi': 15.0, 'aloha_ep': 50.0, 'aloha_fps': 50.0, 'aloha_dim': 14.0, 'lag_ms': 35.0, 'clear_mm': 2.0, 'v_transit': 0.5, 'dx_transit': 17.5, 'transit_x': 8.75, 'dx_guard': 1.05, 'dx_fast': 3.5, 'ceil_guard': 66.7, 'ceil_fast': 20.0, 'skew_lo': 15.0, 'skew_hi': 33.0, 'clear_tight': 1.0, 'ceil_tight': 10.0, 'tight_x': 3.5, 'v_fast_mm_ms': 0.1, 't_disp': 50.0, 't_hum': 200.0, 't_cmd': 10.0, 't_mech': 80.0, 't_loop': 340.0, 't_issued': 250.0, 'hum_pct': 58.8, 'oxe_n': 72.0, 'oxe_total': 2419193.0, 'oxe_total_m': 2.4, 'oxe_rate10': 10.0, 'oxe_10hz_pct': 57.0, 'oxe_norate_pct': 28.0, 'oxe_fast_pct': 2.0, 'oxe_single_pct': 91.0, 'oxe_table_pct': 89.0, 'oxe_mobile': 119386.0, 'oxe_mobile_pct': 5.0, 'oxe_top2_pct': 95.0, 'oxe_row0_hz': 10.0, 'oxe_row0_ep': 580392.0, 'oxe_row1_hz': 10.0, 'oxe_row1_ep': 442226.0, 'oxe_row2_hz': 15.0, 'oxe_row2_ep': 92233.0, 'oxe_row3_hz': 30.0, 'oxe_row3_ep': 139.0, 'oxe_row4_hz': 50.0, 'oxe_row4_ep': 276.0, 'oxe_rows_hz_lo': 10.0, 'oxe_rows_hz_hi': 50.0, 'startup_min': 8.0, 'calib_min': 7.0, 'rest_min': 10.0, 'active_min': 35.0, 'attempts': 52.0, 't_att': 25.0, 't_rst': 15.0, 'resets_min': 13.0, 'attempts_min': 22.0, 'fail_pct': 20.0, 'qa_pct': 15.0, 'retention': 0.68, 'kept': 35.36, 'kept_min': 14.73, 'kept_min_1': 14.7, 'kept_pct': 24.6, 'target': 5000.0, 'camp_attempts': 7353.0, 'camp_h': 141.4, 'motion_h': 34.7, 'overhead_x': 4.07, 'gear_rev': 14.0, 'cable_flex': 8.0, 'gear_cycles': 102942.0, 'cable_cycles': 58824.0, 'gear_pct': 2.1, 'cable_pct': 29.4, 'gear_life_m': 5.0, 'cable_life': 200000.0, 'cable_pct_3': 88.0, 'nav_w': 4056.0, 'nav_h': 3040.0, 'nav_fps': 60.0, 'bw_nav': 2219.0, 'nav_over_sata': 4.0, 'bw_total': 3122.0, 'bw_total_gb': 3.12, 'tb_hour': 11.24, 'compress_x': 8.0, 'bw_comp': 390.0, 'bw_kin': 0.22, 'kin_pct': 0.007, 'sata': 550.0, 'nvme': 7000.0, 'ang_max': 15.0, 'ang_bin': 1.0, 'force_max': 20.0, 'force_bin': 1.0, 'grid_cells': 600.0, 'grid_hz': 100.0, 'caut_ang': 2.0, 'caut_f_lo': 4.0, 'caut_f_hi': 6.0, 'caut_cells': 8.0, 'caut_pct': 1.3, 'expl_f_lo': 1.0, 'expl_f_hi': 18.0, 'expl_cells': 510.0, 'expl_pct': 85.0, 'rms_caut': 0.001, 'rms_expl': 0.012, 'state_dims': 28.0, 'state_bins': 10.0, 'state_cells_exp': 28.0, 'tab_cells': 6.0, 'tab_supported': 1.0, 'tab_absent': 5.0, 'plate_mm': 3.0, 'rec_supported': 2.0, 'rec_absent': 4.0, 'third_offsets': 3.0, 'third_speeds': 3.0, 'third_cells': 9.0, 'third_absent': 8.0, 'takeover_s': 4.47, 'lookback_s': 5.0, 'lookback_ticks': 5000.0, 'edr_s': 20.0, 'edr_hz': 10.0, 'edr_n': 200.0, 'edr_old_s': 5.0, 'edr_old_s0': 5.0, 'edr_old_hz': 2.0, 'edr_old_n': 10.0, 'thr_trials': 15.0, 'thr_bc': 0.0, 'thr_hg': 10.0, 'thr_td': 12.0, 'thr_hg_int': 1.55, 'thr_hg_sd': 1.16, 'thr_td_int': 1.42, 'thr_td_sd': 1.14, 'thr_int_avg': 1.5, 'grip_cmd': 0.8, 'stroke_a': 100.0, 'stroke_b': 60.0, 'part_mm': 75.0, 'open_a': 80.0, 'open_b': 48.0, 'hold_a': 30.0, 'rating_b': 15.0, 'lbf_n': 4.45, 'mco_planned': 226.0, 'mco_actual': 57.0}
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
plt = sp.plt
C = sp.C
OUT = Path(__file__).resolve().parent
FONT = 'Helvetica, Arial, sans-serif'
NAVY, BLUE, GREEN, RED, PETROL, SLATE, AMBER = ('#1F407A', '#2563EB', '#059669', '#DC2626', '#0F766E', '#334155', '#D97706')
GREY = '#94A3B8'

class Svg:

    def __init__(self, w=900, h=500):
        self.w, self.h, self.items = (w, h, [])

    def rect(self, x, y, w, h, fill='white', stroke=SLATE, sw=2, rx=6, dash=None, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{op}"/>')

    def text(self, x, y, s, size=22, fill=SLATE, anchor='middle', weight='normal', op=1.0, italic=False):
        st = ' font-style="italic"' if italic else ''
        self.items.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" opacity="{op}"{st}>{s}</text>')

    def line(self, x1, y1, x2, y2, stroke=SLATE, sw=2.5, dash=None, arrow=False, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#a{stroke[1:]})"' if arrow else ''
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m} opacity="{op}"/>')

    def path(self, d, stroke=SLATE, sw=2.5, fill='none', dash=None, arrow=False, op=1.0):
        da = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#a{stroke[1:]})"' if arrow else ''
        self.items.append(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{da}{m} opacity="{op}"/>')

    def circle(self, x, y, r, fill=SLATE, stroke='none', op=1.0):
        self.items.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" opacity="{op}"/>')

    def save(self, name):
        cols = {NAVY, BLUE, GREEN, RED, PETROL, SLATE, AMBER, GREY}
        marks = ''.join((f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in cols))
        body = '\n'.join(self.items)
        (OUT / name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">\n<defs>{marks}</defs>\n<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')

def alpha(shown):
    return 1.0 if shown else 0.0

def loops():
    for k in (1, 2):
        s = Svg(900, 360)
        s.text(200, 32, 'Image classifier', 28, NAVY, weight='bold')
        for i, y in enumerate((60, 125, 190)):
            s.rect(15, y, 120, 50, '#F8FAFC', SLATE, rx=4)
            s.text(75, y + 33, f'image {i + 1}', 23)
            s.line(135, y + 25, 190, y + 25, NAVY, arrow=True)
            s.rect(192, y, 105, 50, '#EFF6FF', NAVY)
            s.text(244, y + 33, 'model', 23, NAVY)
            s.line(297, y + 25, 340, y + 25, NAVY, arrow=True)
            s.text(345, y + 33, 'label', 23, NAVY, anchor='start')
        s.text(200, 290, 'A wrong label does not', 25)
        s.text(200, 322, 'change the next image.', 25)
        o = alpha(k >= 2)
        s.line(440, 10, 440, 345, GREY, 2, dash='6,5')
        s.text(680, 32, 'Robot policy', 28, GREEN, weight='bold', op=o)
        s.rect(475, 70, 140, 64, '#EFF6FF', BLUE, dash='6,4', op=o)
        s.text(545, 110, 'policy', 25, BLUE, op=o)
        s.rect(725, 70, 160, 64, '#F8FAFC', SLATE, rx=0, op=o)
        s.text(805, 110, 'body + world', 23, SLATE, op=o)
        s.line(615, 102, 721, 102, BLUE, arrow=True, op=o)
        s.text(668, 90, 'action', 22, BLUE, op=o)
        s.path('M805,134 C805,225 545,225 545,140', PETROL, arrow=True, op=o)
        s.text(675, 240, 'next state and image', 23, PETROL, op=o)
        s.text(680, 290, 'An action chooses the', 25, op=o)
        s.text(680, 322, 'states the robot sees next.', 25, op=o)
        s.save(f'loops_step{k}.svg')

def corridor():
    for k in (1, 2):
        s = Svg(900, 370)
        s.rect(40, 20, 680, 70, '#E2E8F0', 'none', 0, rx=0)
        s.text(380, 64, 'fixture', 26, SLATE)
        s.rect(40, 175, 680, 90, '#DCFCE7', GREEN, 2, rx=0)
        for x in (80, 210, 340, 470, 600):
            s.line(x, 220, x + 90, 220, GREEN, 3, arrow=True)
        s.rect(740, 135, 40, 170, '#F8FAFC', SLATE, 3, rx=0)
        s.text(830, 205, 'strike', 25, anchor='middle')
        s.text(830, 235, 'plate', 25, anchor='middle')
        s.text(380, 305, 'every retained demonstration goes forward', 26, GREEN)
        s.text(380, 338, 'along the corridor', 26, GREEN)
        o = alpha(k >= 2)
        s.rect(330, 108, 40, 40, '#EFF6FF', BLUE, 3, rx=2, op=o)
        s.text(318, 138, 'gripper, 3 mm off', 25, BLUE, anchor='end', op=o)
        s.line(350, 150, 350, 185, NAVY, 3, dash='6,4', arrow=True, op=o)
        s.text(385, 160, 'return action: no data', 25, NAVY, anchor='start', weight='bold', op=o)
        s.line(372, 112, 430, 96, RED, 3, dash='6,4', arrow=True, op=o)
        s.text(445, 112, 'drift into contact', 25, RED, anchor='start', weight='bold', op=o)
        s.save(f'corridor_step{k}.svg')

def bounds():
    S = np.logspace(1, np.log10(3000), 200)
    e = 0.001
    s0 = 500
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.75))
        ax.plot(S, S, color=C['neutral'], lw=1.5, ls='--')
        ax.text(1500, 900, 'episode\nmaximum', color=C['neutral'], fontsize=11, ha='center', va='top')
        ax.plot(S, S * e, color=C['memory'], lw=3)
        ax.text(2200, 2200 * e * 0.45, 'open loop\n$S\\varepsilon$', color=C['memory'], fontsize=12, fontweight='bold', ha='center', va='top')
        a2 = alpha(k >= 2)
        ax.plot(S, np.minimum(S, S ** 2 * e), color=C['danger'], lw=3, alpha=a2)
        ax.text(60, 0.9, 'cloning $S^2\\varepsilon$', color=C['danger'], fontsize=12.5, fontweight='bold', ha='left', va='top', alpha=a2)
        ax.plot([s0], [s0 ** 2 * e], 'o', ms=9, color=C['danger'], alpha=a2)
        ax.text(s0 * 1.12, s0 ** 2 * e * 0.62, '250', color=C['danger'], fontsize=13, fontweight='bold', alpha=a2)
        ax.plot([s0], [s0 * e], 'o', ms=9, color=C['memory'])
        ax.text(s0 * 1.12, s0 * e * 0.4, '0.5', color=C['memory'], fontsize=13, fontweight='bold')
        a3 = alpha(k >= 3)
        ax.plot(S, 1.0 * S * e, color=C['data'], lw=3, ls=(0, (6, 3)), alpha=a3)
        ax.text(12, 0.0042, 'aggregation, $u=1$: $uS\\varepsilon$', color=C['data'], fontsize=12, fontweight='bold', alpha=a3)
        ax.axvline(s0, color=C['grid'], lw=1, zorder=0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(10, 3000)
        ax.set_ylim(0.002, 5000.0)
        ax.minorticks_off()
        ax.set_xticks([10, 100, 1000])
        ax.set_xticklabels(['10', '100', '1,000'])
        ax.set_yticks([0.01, 1, 100])
        ax.set_yticklabels(['0.01', '1', '100'])
        ax.set_xlabel('Horizon $S$ (control steps)')
        ax.set_ylabel('Worst-case excess cost')
        ax.text(0.02, 0.97, '$\\varepsilon = 0.001$ per step', transform=ax.transAxes, ha='left', va='top', fontsize=11, color=C['text'])
        sp.save(fig, OUT / f'bound_step{k}.svg')

def lerobot():
    months, pooled = (({'month': '2024-03', 'new_datasets': 6.0, 'cumulative_datasets': 6.0}, {'month': '2024-04', 'new_datasets': 4.0, 'cumulative_datasets': 10.0}, {'month': '2024-05', 'new_datasets': 30.0, 'cumulative_datasets': 40.0}, {'month': '2024-06', 'new_datasets': 4.0, 'cumulative_datasets': 44.0}, {'month': '2024-07', 'new_datasets': 42.0, 'cumulative_datasets': 86.0}, {'month': '2024-08', 'new_datasets': 36.0, 'cumulative_datasets': 122.0}, {'month': '2024-09', 'new_datasets': 133.0, 'cumulative_datasets': 255.0}, {'month': '2024-10', 'new_datasets': 321.0, 'cumulative_datasets': 576.0}, {'month': '2024-11', 'new_datasets': 172.0, 'cumulative_datasets': 748.0}, {'month': '2024-12', 'new_datasets': 337.0, 'cumulative_datasets': 1085.0}, {'month': '2025-01', 'new_datasets': 422.0, 'cumulative_datasets': 1507.0}, {'month': '2025-02', 'new_datasets': 849.0, 'cumulative_datasets': 2356.0}, {'month': '2025-03', 'new_datasets': 912.0, 'cumulative_datasets': 3268.0}, {'month': '2025-04', 'new_datasets': 1537.0, 'cumulative_datasets': 4805.0}, {'month': '2025-05', 'new_datasets': 1690.0, 'cumulative_datasets': 6495.0}, {'month': '2025-06', 'new_datasets': 2769.0, 'cumulative_datasets': 9264.0}, {'month': '2025-07', 'new_datasets': 2515.0, 'cumulative_datasets': 11779.0}, {'month': '2025-08', 'new_datasets': 2088.0, 'cumulative_datasets': 13867.0}, {'month': '2025-09', 'new_datasets': 2068.0, 'cumulative_datasets': 15935.0}, {'month': '2025-10', 'new_datasets': 2722.0, 'cumulative_datasets': 18657.0}, {'month': '2025-11', 'new_datasets': 3849.0, 'cumulative_datasets': 22506.0}, {'month': '2025-12', 'new_datasets': 4042.0, 'cumulative_datasets': 26548.0}, {'month': '2026-01', 'new_datasets': 4722.0, 'cumulative_datasets': 31270.0}, {'month': '2026-02', 'new_datasets': 5529.0, 'cumulative_datasets': 36799.0}, {'month': '2026-03', 'new_datasets': 5840.0, 'cumulative_datasets': 42639.0}, {'month': '2026-04', 'new_datasets': 4868.0, 'cumulative_datasets': 47507.0}, {'month': '2026-05', 'new_datasets': 7298.0, 'cumulative_datasets': 54805.0}, {'month': '2026-06', 'new_datasets': 5815.0, 'cumulative_datasets': 60620.0}, {'month': '2026-07', 'new_datasets': 6556.0, 'cumulative_datasets': 67176.0}, {'month': '2026-08', 'new_datasets': 5693.0, 'cumulative_datasets': 72869.0}, {'month': '2026-09', 'new_datasets': 5393.0, 'cumulative_datasets': 78262.0}, {'month': '2026-10', 'new_datasets': 1870.0, 'cumulative_datasets': 80132.0}), {1.0: 8567.0, 2.0: 6155.0, 4.0: 5402.0, 8.0: 7490.0, 16.0: 7142.0, 32.0: 8147.0, 64.0: 4370.0, 128.0: 2303.0, 256.0: 1482.0, 512.0: 517.0, 1024.0: 277.0, 2048.0: 144.0, 4096.0: 84.0, 8192.0: 36.0, 0.0: 6841.0, 16384.0: 11.0, 32768.0: 29.0, 65536.0: 34.0})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw=dict(width_ratios=[1, 1.2]))
    x = np.arange(len(months))
    y = [m['cumulative_datasets'] for m in months]
    a1.plot(x, y, color=C['data'], lw=3)
    a1.set_yscale('log')
    a1.set_ylim(3, 300000.0)
    a2.set_ylim(0, 12.5)
    ticks = [i for i, m in enumerate(months) if m['month'].endswith('-01')]
    a1.set_xticks(ticks)
    a1.set_xticklabels([months[i]['month'][:4] for i in ticks])
    a1.set_yticks([10, 1000.0, 100000.0])
    a1.set_yticklabels(['10', '1,000', '100,000'])
    a1.set_ylabel('Datasets on the Hub')
    for key, lab in (('2024-12', '1,085'), ('2026-10', '80,132')):
        i = [m['month'] for m in months].index(key)
        a1.plot([i], [y[i]], 'o', color=C['data'], ms=7)
        a1.text(i - 0.5, y[i] * 2.2, lab, color=C['data'], fontsize=12, fontweight='bold', ha='right' if key == '2026-10' else 'center')
    los = sorted(pooled)
    vals = [pooled[lo] / 1000 for lo in los]
    cols = [C['danger'] if lo < 16 else C['memory'] for lo in los]
    a2.bar(range(len(los)), vals, color=cols, width=0.8)
    lab = ['0' if lo == 0 else f'{int(lo):,}' for lo in los]
    show = {0, 4, 9, 13, 17}
    a2.set_xticks([i for i in range(len(los)) if i in show])
    a2.set_xticklabels([lab[i] for i in range(len(los)) if i in show])
    a2.set_xlabel('Episodes per dataset (bin start)')
    a2.set_ylabel('Datasets (thousands)')
    a2.text(0, 12.2, '58% hold fewer than 16', color=C['danger'], fontsize=12, fontweight='bold', ha='left', va='top')
    a2.text(9.5, 4.0, '2% hold\n512 or more', color=C['memory'], fontsize=12, fontweight='bold', ha='left')
    fig.tight_layout(w_pad=1.5)
    sp.save(fig, OUT / 'lerobot_hub.svg')

def taps():
    for k in range(1, 5):
        s = Svg(900, 390)
        s.rect(20, 10, 410, 200, '#F8FAFC', GREY, 1.5, dash='6,4')
        s.text(225, 42, 'application processor (proposes)', 21, NAVY, weight='bold')
        s.rect(470, 10, 410, 200, '#F0FDF4', GREY, 1.5, dash='6,4')
        s.text(675, 42, 'safety MCU, 1 kHz (may command)', 21, GREEN, weight='bold')
        s.line(450, 0, 450, 360, SLATE, 2.5, dash='8,5')
        s.text(450, 384, 'proposal boundary', 20, SLATE)
        s.rect(35, 80, 155, 74, '#EFF6FF', BLUE, dash='6,4')
        s.text(112, 112, 'operator', 21, BLUE)
        s.text(112, 138, 'or policy', 21, BLUE)
        s.rect(250, 80, 155, 74, '#EFF6FF', NAVY)
        s.text(327, 112, 'kinematic', 21, NAVY)
        s.text(327, 138, 'mapping', 21, NAVY)
        s.line(190, 117, 246, 117, BLUE, arrow=True)
        s.line(405, 117, 496, 117, NAVY, arrow=True)
        s.rect(500, 80, 155, 74, '#DCFCE7', GREEN)
        s.text(577, 112, 'permission', 21, GREEN)
        s.text(577, 138, 'check, clamp', 21, GREEN)
        s.rect(712, 80, 155, 74, '#FFFFFF', SLATE, rx=0)
        s.text(789, 112, 'drives and', 21, SLATE)
        s.text(789, 138, 'plant', 21, SLATE)
        s.line(655, 117, 708, 117, GREEN, arrow=True)
        taps = [(218, '1 Asked', 'request', BLUE), (430, '2 Mapped', 'setpoint', NAVY), (680, '3 Allowed', 'enforced', GREEN), (830, '4 Done', 'measured', PETROL)]
        for i, (x, a, b, col) in enumerate(taps):
            o = alpha(k >= i + 1)
            yy = 275
            if i == 3:
                s.line(789, 154, 789, 185, col, 2.5, op=o)
                s.line(789, 185, x, 185, col, 2.5, op=o)
                s.line(x, 185, x, yy - 34, col, 2.5, dash='4,4', arrow=True, op=o)
            else:
                s.circle(x, 117, 7, col, op=o)
                s.line(x, 124, x, yy - 34, col, 2.5, dash='4,4', arrow=True, op=o)
            bx = min(max(x - 65, 20), 762)
            s.rect(bx, yy - 30, 130, 68, 'white', col, 2.5, op=o)
            s.text(bx + 65, yy - 3, a, 22, col, weight='bold', op=o)
            s.text(bx + 65, yy + 25, b, 20, col, op=o)
        s.text(880, 345, 'one log, one hardware clock (PTP)', 20, PETROL, anchor='end', op=alpha(k >= 4))
        s.save(f'taps_step{k}.svg')

def pairing():
    lag = np.logspace(0, 2, 100)
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.75))
        a2 = alpha(k >= 2)
        ax.axvspan(15.0, 33.0, color=C['neutral_l'], alpha=0.7 * a2, lw=0)
        ax.text(22, 0.045, 'rolling-shutter\nskew', color=C['neutral'], fontsize=10.5, ha='center', alpha=a2)
        for v, name, col in ((0.5, '0.50 m/s\ntransit', C['neutral']), (0.1, '0.10 m/s\nfast', C['danger']), (0.03, '0.03 m/s\nguarded', C['data'])):
            ax.plot(lag, v * lag, color=col, lw=3)
            ax.plot([35.0], [v * 35.0], 'o', color=col, ms=8)
            ax.text(107, v * 100, name, color=col, fontsize=12, fontweight='bold', ha='left', va='center', clip_on=False)
        ax.axhline(2.0, color=C['danger'], lw=1.6, ls='--')
        ax.text(1.1, 2.0 * 1.25, '2 mm clearance', color=C['danger'], fontsize=11, ha='left', bbox=dict(facecolor='white', edgecolor='none', pad=1))
        ax.axvline(35.0, color=C['text'], lw=1.2, ls=':')
        ax.text(35.0 * 1.06, 0.03, '35 ms', fontsize=11.5, color=C['text'])
        for key, ceiling in (('ceil_fast', Q_(20.0, 'ms')), ('ceil_guard', Q_(66.66666667, 'ms'))):
            c = ceiling.to('ms').magnitude
            ax.plot([c], [2.0], 's', color=C['danger'], ms=8, alpha=a2)
            ax.text(c, 4.6 if c < 30 else 0.75, f"≤ { {20.0: '20', 66.66666667: '66.7'}[ceiling.magnitude]} ms", color=C['danger'], fontsize=11.5, fontweight='bold', ha='center', alpha=a2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 100)
        ax.set_ylim(0.02, 60)
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['1', '10', '100'])
        ax.set_yticks([0.1, 1, 10])
        ax.set_yticklabels(['0.1', '1', '10'])
        ax.set_xlabel('Observation-to-action lag (ms)')
        ax.set_ylabel('Pairing error (mm)')
        sp.save(fig, OUT / f'pairing_step{k}.svg')

def teleop():
    tot = 340.0
    fig, ax = plt.subplots(figsize=(5.8, 1.45))
    left = 0
    for lab, key, col in (('display', 't_disp', C['memory']), ('human', 't_hum', C['compute']), ('cmd', 't_cmd', C['network']), ('mechanics', 't_mech', C['neutral'])):
        v = {'display': 50.0, 'human': 200.0, 'cmd': 10.0, 'mechanics': 80.0}[lab]
        ax.barh(0, v, left=left, color=col, height=0.6, edgecolor='white')
        if v >= 40:
            ax.text(left + v / 2, 0, f"{lab}\n{ {'display': '50', 'human': '200', 'cmd': '10', 'mechanics': '80'}[lab]} ms", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        else:
            ax.text(left + v / 2, -0.55, f"{lab} { {'display': '50', 'human': '200', 'cmd': '10', 'mechanics': '80'}[lab]} ms", ha='center', va='top', color=col, fontsize=11)
        left += v
    ax.text(tot, 0.48, '340 ms loop', ha='right', va='bottom', fontsize=13, fontweight='bold', color=C['text'])
    ax.set_xlim(0, tot)
    ax.set_ylim(-1.05, 0.95)
    ax.axis('off')
    sp.save(fig, OUT / 'teleop_strip.svg')

def oxe():
    morph, rate = ({'Mobile Manipulator': 119386.0, 'Single Arm': 2212108.0, 'Quadrupedal Robot': 20.0, 'Bi-Manual': 521.0, 'Wheeled Robot': 22158.0, '3 embodiments: single-armed robot, single-armed human, single-armed human using grasping tools': 61153.0, 'Human': 3847.0}, {3.0: 87589.0, 10.0: 1378291.0, 5.0: 55550.0, 15.0: 95853.0, 20.0: 36505.0, 30.0: 6859.0, 2.0: 150.0, 50.0: 747.0, 1.0: 82628.0, None: 669327.0, 12.5: 110.0, 66.6: 120.0, 500.0: 256.0, 3.75: 5208.0})
    stated = sorted((k for k in rate if k is not None))
    labels = [f'{k:g}' for k in stated] + ['none']
    h = [rate[k] / 1000000.0 for k in stated] + [rate[None] / 1000000.0]
    cols = [C['data'] if k == 20.0 else C['memory'] for k in stated] + [C['neutral']]
    fig, ax = plt.subplots(figsize=(6.2, 2.55))
    ax.bar(range(len(h)), h, color=cols, width=0.75)
    ax.set_xticks(range(len(h)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel('Declared command rate (Hz)')
    ax.set_ylabel('Episodes (millions)')
    i10 = stated.index(10.0)
    ax.text(i10, h[i10] + 0.03, '57%', ha='center', va='bottom', color=C['memory'], fontsize=12.5, fontweight='bold')
    ax.text(len(h) - 1, h[-1] + 0.03, '28%', ha='center', va='bottom', color=C['neutral'], fontsize=12.5, fontweight='bold')
    i20 = stated.index(20.0)
    ax.annotate("20 Hz: the running\nmachine's chunk rate", (i20, h[i20] + 0.02), (i20 - 2.6, 0.75), color=C['data'], fontsize=11.5, fontweight='bold', ha='left', arrowprops=dict(arrowstyle='-', color=C['data'], lw=1.5))
    ax.set_ylim(0, 1.65)
    sp.save(fig, OUT / 'oxe_rate.svg')
    names = {'Single Arm': 'Single arm', 'Mobile Manipulator': 'Mobile manipulator', 'Wheeled Robot': 'Wheeled robot', 'Human': 'Human', 'Bi-Manual': 'Bimanual', 'Quadrupedal Robot': 'Quadruped'}
    order = [k for k in sorted(morph, key=morph.get, reverse=True)]
    rws = []
    for k in order:
        name = names.get(k, 'Mixed human, robot')
        ck = 'data' if k == 'Mobile Manipulator' else 'memory'
        rws.append((name, morph[k], ck, f'{morph[k]:,.0f}'))
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    n = len(rws)
    for i, (name, v, ck, txt) in enumerate(rws):
        y = n - 1 - i
        ax.barh(y, v, color=C[ck], height=0.62)
        ax.text(v * 1.2, y, txt, va='center', fontsize=12.5, fontweight='bold', color=C[ck])
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rws][::-1], fontsize=12.5, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(8, 40000000.0)
    ax.minorticks_off()
    ax.set_xticks([10, 1000, 100000.0, 10000000.0])
    ax.set_xticklabels(['10', '1,000', '100,000', '10 million'])
    ax.set_xlabel('Episodes (log scale)')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    sp.save(fig, OUT / 'oxe_morph.svg')

def yield_fig():
    st = {'startup': 8, 'calibration': 7, 'rest': 10, 'resets': 13.0, 'attempts': 21.66666667}
    fail = 4.333333333
    qa = 2.6
    kept = 14.73333333
    hour = 60
    rows1 = [('startup', st['startup'], C['neutral_l'], C['text']), ('calib.', st['calibration'], C['neutral_l'], C['text']), ('rest', st['rest'], C['neutral_l'], C['text']), ('resets', st['resets'], C['neutral'], 'white'), ('attempts', st['attempts'], C['compute'], 'white')]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.1))
        left = 0
        for lab, v, col, tc in rows1:
            ax.barh(1, v, left=left, color=col, height=0.55, edgecolor='white', lw=1.5)
            ax.text(left + v / 2, 1, f'{lab}\n{v:.0f}' if v >= 9 else f'{lab}\n{v:.0f}', ha='center', va='center', fontsize=10.5, color=tc, fontweight='bold')
            left += v
        ax.text(0, 1.42, 'One scheduled hour (minutes)', fontsize=12, fontweight='bold', color=C['text'])
        a = alpha(k >= 2)
        x0 = hour - st['attempts']
        for lab, v, col in (('failed', fail, C['danger']), ('QA', qa, C['neutral']), ('kept 14.73', kept, C['data'])):
            ax.barh(0, v, left=x0, color=col, height=0.55, edgecolor='white', lw=1.5, alpha=a)
            if v > 8:
                ax.text(x0 + v / 2, 0, lab, ha='center', va='center', fontsize=10.5, color='white', fontweight='bold', alpha=a)
            else:
                up = lab == 'QA'
                ax.text(x0 + v / 2, 0.33 if up else -0.33, lab, ha='center', va='bottom' if up else 'top', fontsize=10.5, color=col, fontweight='bold', alpha=a)
            x0 += v
        ax.text(hour - st['attempts'] - 0.8, 0, 'of the attempts:', ha='right', va='center', fontsize=12, color=C['text'], alpha=a)
        ax.set_xlim(0, hour)
        ax.set_ylim(-0.75, 1.6)
        ax.axis('off')
        sp.save(fig, OUT / f'yield_step{k}.svg')

def wear():
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    keys = ['cable_pct', 'gear_pct']
    vals = [100 * 0.29412, 100 * 0.0205884]
    ax.bar([0, 1], vals, color=[C['danger'], C['memory']], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{ {29.412: '29.4', 2.05884: '2.1'}[v]}%", ha='center', fontsize=15, fontweight='bold', color=[C['danger'], C['memory']][i])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['wrist\nflex cable', 'strain-wave\ngear'], fontsize=13)
    ax.set_ylabel('Rated life used (%)', fontsize=13)
    ax.set_ylim(0, 36)
    ax.tick_params(axis='y', labelsize=12)
    sp.save(fig, OUT / 'wear.svg')

def bytes_fig():
    rows = [('All streams, raw', 3121.6832, C['danger'], '3,122'), ('Navigation camera', 2219.4432, C['danger'], '2,219'), ('Compressed 8×', 390.2104, C['data'], '390'), ('Four action taps', 0.224, C['memory'], '0.22')]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    for i, (lab, v, col, txt) in enumerate(rows):
        y = 3 - i
        ax.barh(y, v, color=col, height=0.6)
        ax.text(0.06, y, lab + '  ', ha='right', va='center', fontsize=12.5, fontweight='bold', color=C['text'])
        ax.text(v * 1.15 if v < 100 else v * 0.85, y, txt, ha='left' if v < 100 else 'right', va='center', fontsize=12, fontweight='bold', color=col if v < 100 else 'white')
    for x, lab, col in ((550.0, 'SATA SSD\n550', C['danger']), (7000.0, 'NVMe\n7,000', C['neutral'])):
        ax.axvline(x, color=col, lw=1.6, ls='--')
        ax.text(x * 1.12, 3.6, lab, color=col, fontsize=10.5, fontweight='bold', va='top')
    ax.set_xscale('log')
    ax.set_xlim(0.06, 60000.0)
    ax.minorticks_off()
    ax.set_xticks([0.1, 10, 1000])
    ax.set_xticklabels(['0.1', '10', '1,000'])
    ax.set_ylim(-0.5, 3.75)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Sustained write rate (MB/s)')
    sp.save(fig, OUT / 'bytes.svg')

def grids():
    a_lo, a_hi = (-15, 15)
    f_hi = 20
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.65))
    for ax, name, arch, mask in ((axes[0], 'Cautious operator', 'cautious', 'c'), (axes[1], 'Exploratory policy', 'exploratory', 'e')):
        g = np.zeros((20, 30))
        first_a, n_a, first_f, n_f = {'c': (13, 4, 4, 2), 'e': (0, 30, 1, 17)}[mask]
        g[first_f:first_f + n_f, first_a:first_a + n_a] = 1
        assert int(g.sum()) == {'c': 8, 'e': 510}[mask]
        tag = f"{ {'c': '8', 'e': '510'}[mask]} of 600 cells"
        rms = {'c': '0.001', 'e': '0.012'}[mask]
        ax.imshow(g, origin='lower', cmap=plt.matplotlib.colors.ListedColormap([C['neutral_l'], C['data']]), extent=(a_lo, a_hi, 0, f_hi), aspect='auto')
        ax.set_title(name, fontsize=12.5, fontweight='bold', color=C['text'], pad=20)
        ax.set_xlabel('Contact angle (deg)', fontsize=11)
        ax.set_xticks([a_lo, 0, a_hi])
        ax.set_yticks([0, f_hi / 2, f_hi])
        ax.text(0, f_hi + 1, f'{tag}; held-out error {rms} rad', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=C['data'] if mask == 'e' else C['danger'])
    axes[0].set_ylabel('Normal force (N)', fontsize=11)
    fig.tight_layout(w_pad=2)
    sp.save(fig, OUT / 'grids.svg')

def scenario():
    assert 1 == 1
    for k in (1, 2, 3):
        s = Svg(900, 370)
        x0, y0, cw, ch = (300, 50, 175, 100)
        s.text(x0 - 130, y0 + ch + 6, 'approach', 23, anchor='middle')
        s.text(x0 - 130, y0 + ch + 34, 'speed', 23, anchor='middle')
        for j, sp_ in enumerate(('0.10 m/s', '0.03 m/s')):
            s.text(x0 - 14, y0 + j * ch + ch / 2 + 8, sp_, 23, anchor='end')
        for i, off in enumerate(('−3 mm', '0 mm', '+3 mm')):
            s.text(x0 + i * cw + cw / 2, y0 + 2 * ch + 32, off, 23)
        s.text(x0 + 1.5 * cw, y0 + 2 * ch + 64, 'strike-plate offset', 23)
        for i in range(3):
            for j in range(2):
                x, y = (x0 + i * cw, y0 + j * ch)
                slow = j == 1
                nominal = i == 1
                if k == 1 and (not (slow and nominal)):
                    s.rect(x + 4, y + 4, cw - 8, ch - 8, 'white', GREY, 1.5, dash='3,5')
                    s.text(x + cw / 2, y + ch / 2 + 8, 'not in table', 21, GREY)
                    continue
                pop = slow and nominal or (k == 3 and slow and (i == 2))
                if pop:
                    s.rect(x + 4, y + 4, cw - 8, ch - 8, '#DCFCE7', GREEN, 3)
                    s.text(x + cw / 2, y + ch / 2 + 8, 'demonstrated', 22, GREEN, weight='bold')
                else:
                    s.rect(x + 4, y + 4, cw - 8, ch - 8, '#FEF2F2', RED, 2.5)
                    s.text(x + cw / 2, y + ch / 2 + 10, '0', 30, RED, weight='bold')
        msg = {1: f'Uncritical table: {1} of {1} cell. The archive looks complete.', 2: 'The table the door needs: 1 of 6 cells; 5 known absences.', 3: 'After latch-recovery-02: 2 of 6 cells; 4 known absences.'}[k]
        s.text(450, 28, msg, 25, SLATE, weight='bold')
        s.save(f'scenario_step{k}.svg')

def takeover():
    t = np.linspace(0, 10, 500)
    td, tt, tb, ts, tr = (2.0, 5.0, 5.6, 7.0, 9.4)
    err = np.where(t < td, 0.05, 0)
    err = np.where((t >= td) & (t < tt), 0.05 + 0.95 * np.clip((t - td) / (tt - td), 0, 1) ** 1.6, err)
    err = np.where((t >= tt) & (t < tb), 1.0 + 0.03 * (t - tt), err)
    err = np.where((t >= tb) & (t < ts), 1.02 + 0.12 * np.sin((t - tb) * 9), err)
    err = np.where(t >= ts, 1.02 * np.clip(1 - (t - ts) / (tr - ts), 0, 1) ** 1.3, err)
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.9))
        ax.plot(t[t < tt], err[t < tt], color=C['danger'], lw=3)
        ax.plot(t[(t >= tt) & (t < tb)], err[(t >= tt) & (t < tb)], color=C['neutral'], lw=3, ls='--')
        ax.plot(t[(t >= tb) & (t < ts)], err[(t >= tb) & (t < ts)], color=C['neutral'], lw=2)
        ax.plot(t[t >= ts], err[t >= ts], color=C['data'], lw=3)
        for x, lab in ((td, '$t_{\\mathrm{div}}$'), (tt, '$t_{\\mathrm{take}}$'), (ts, '$t_{\\mathrm{stab}}$'), (tr, '$t_{\\mathrm{rec}}$')):
            ax.axvline(x, color=C['grid'], lw=1, zorder=0)
            ax.text(x, 1.5, lab, ha='center', fontsize=12.5, color=C['text'])
        look = 5.0
        ax.annotate('', (tt - look + 0.05, 1.22), (tt - 0.05, 1.22), arrowprops=dict(arrowstyle='<->', color=C['memory'], lw=1.8))
        ax.text(tt - look / 2, 1.28, '5 s lookback, frozen at takeover', ha='center', fontsize=10.5, color=C['memory'], fontweight='bold')
        a2, a3 = (alpha(k >= 2), alpha(k >= 3))
        for x, lab, col in (((td + tt) / 2, 'drift', C['danger']), ((tt + tb) / 2 - 0.15, 'blend', C['neutral']), ((tb + ts) / 2 + 0.35, 're-orient', C['neutral']), ((ts + tr) / 2 + 0.35, 'stabilized', C['data'])):
            ax.text(x, -0.28, lab, ha='center', fontsize=11, fontweight='bold', color=col, alpha=a2)
        for x, lab, col in (((td + tt) / 2, 'states: relabel\ncommands: mask', C['danger']), ((tb + ts) / 2 - 0.25, 'excluded', C['neutral']), ((ts + tr) / 2, 'expert labels', C['data'])):
            ax.text(x, -0.55, lab, ha='center', va='top', fontsize=10.5, color=col, alpha=a3)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.05, 1.72)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['bottom'].set_position(('data', -0.12))
        ax.set_ylabel('Tracking error')
        ax.text(10, -0.3, 'time', ha='right', va='top', fontsize=11, color=C['text'])
        sp.save(fig, OUT / f'takeover_step{k}.svg')

def thrifty():
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    keys = ['thr_bc', 'thr_hg', 'thr_td']
    v = [_NUMS[k] for k in keys]
    cols = [C['danger'], C['memory'], C['data']]
    ax.bar(range(3), v, color=cols, width=0.62)
    for i, k in enumerate(keys):
        ax.text(i, v[i] + 0.4, f"{ {'thr_bc': '0', 'thr_hg': '10', 'thr_td': '12'}[k]} of 15", ha='center', fontsize=14, fontweight='bold', color=cols[i])
    ax.set_xticks(range(3))
    ax.set_xticklabels(['cloning\nonly', 'human-gated\nDAgger', 'Thrifty-\nDAgger'], fontsize=12.5)
    ax.set_ylabel('Autonomous successes', fontsize=12.5)
    ax.set_ylim(0, 15.0)
    ax.set_yticks(np.linspace(0, 15.0, 4))
    ax.tick_params(axis='y', labelsize=12)
    sp.save(fig, OUT / 'thrifty.svg')

def normalize():
    for k in (1, 2):
        s = Svg(900, 400)
        sc = 3.4
        for row, (name, key, y) in enumerate((('Machine A', 'stroke_a', 50), ('Machine B', 'stroke_b', 235))):
            stroke = {50: 100.0, 235: 60.0}[y]
            o = 1.0 if row == 0 else alpha(k >= 2)
            cx = 610
            open_key = 'open_a' if row == 0 else 'open_b'
            open_mm = {50: 80.0, 235: 48.0}[y]
            s.text(20, y + 12, f"{name}: { {50: '100', 235: '60'}[y]} mm stroke", 26, SLATE, anchor='start', weight='bold', op=o)
            s.text(20, y + 50, 'command 0.80', 26, BLUE, anchor='start', op=o)
            s.text(20, y + 86, f"opening { {50: '80', 235: '48'}[y]} mm", 26, BLUE, anchor='start', weight='bold', op=o)
            part = 75.0 * sc
            s.rect(cx - part / 2, y - 5, part, 90, '#E2E8F0', SLATE, 2, rx=2, op=o)
            s.text(cx, y + 48, '75 mm part', 25, SLATE, op=o)
            half = open_mm * sc / 2
            col = GREEN if open_mm > 75.0 else RED
            for sx in (-1, 1):
                fx = cx + sx * half
                s.rect(fx - (16 if sx < 0 else 0), y - 25, 16, 130, col, col, 1, rx=2, op=o)
            msg = 'clears the part' if open_mm > 75.0 else 'fingers land inside the part'
            s.text(cx, y + 140, msg, 26, col, weight='bold', op=o)
        s.save(f'normalize_step{k}.svg')
if __name__ == '__main__':
    loops()
    corridor()
    bounds()
    lerobot()
    taps()
    pairing()
    teleop()
    oxe()
    yield_fig()
    wear()
    bytes_fig()
    grids()
    scenario()
    takeover()
    thrifty()
    normalize()
    print('figures written')
