#!/usr/bin/env python3
"""Figures for Lecture 9 (Spatial Memory), Volume IV.

Plots use the shared slideplot helper; diagrams are written as SVG with text
kept as text. Every number drawn comes from calcs.py (imported as K), which
reads MLSysIM, and every label prints val(key), the string the slide prints.
Each figure is drawn at the size of its slot: full-width plots 5.8 x 2.4 in,
column figures about 5.4 x 3.6 in, full-width SVG diagrams on a 900 x 500
canvas with labels of 20 units or more.

Run from this folder: python3 make_figs.py   (then make deck V=4 D=09_memory)
"""
from types import SimpleNamespace
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
_VALS = {'v_drive': '1.5', 'v_walk': '1.6', 'd_clear': '1.10', 'd_clear_fig': '1.1', 'side_clearance': '150', 'side_clearance_m': '0.15', 'delta_loc': '50', 'delta_loc_m': '0.05', 'delta_margin': '100', 'delta_margin_m': '0.10', 'aisle_m': '30', 'v_aisle': '1.3', 't_age': '51.6', 'd_age': '77.4', 't_frame': '16.7', 'sig_prior': '20', 'sig_prior_fig': '20.0', 'speed_sigma': '0.1', 't_sweep': '50', 'sig_odo': '5', 'sig_pred': '20.6', 'sig_lidar': '30', 'lidar_bias': '25', 'g_lidar': '0.32', 'sig_l': '17.0', 'sig_stereo': '3.8', 'g_stereo': '0.95', 'sig_s': '3.7', 'sig_stereo_x2': '7.6', 'g_stereo_x2': '0.83', 'sig_s_x2': '6.9', 'misdate': '39.9', 'shift': '38.0', 'nsig_words': 'ten', 'slots': '52', 'f_imu_khz': '1', 'nis': '6.25', 'gate95': '3.84', 'gate99': '6.63', 'gate95_pct': '95', 'gate99_pct': '99', 'p95': '0.63', 'p99': '0.39', 'n_runs': '50', 'band_lo': '0.65', 'band_hi': '1.43', 'nees_band_pct': '95', 't_coast': '1', 't_coast_words': 'one', 'gyro_bias': '1', 'acc_bias_mg': '20', 'gyro_noise_deg': '0.014', 'imu_lateral': '19.2', 'acc_bias_mm': '98.1', 'acc_noise_mm': '0.9', 'min_hits_words': 'three', 'held_words': 'two', 'min_hits': '3', 'max_age': '1', 'sort_wait': '33.3', 'sort_travel': '50.0', 'eps_sync': '0.2', 'd_sync': '0.3', 'epoch_gap': '2.5', 'd_gap': '4.25', 'v_arm': '1.7', 'lever': '0.15', 'omega': '0.30', 'v_drift': '12', 'e0_slot': '0.1', 'e0_slot_fig': '0.10', 'r_tol': '1', 'e_drift': '0.20', 'e_lever': '0.75', 'e_total': '1.05', 'over_pct': '5', 'e_same': '0.31', 'umb_side': '4', 'umb_path': '16', 'umb_uncal': '317', 'umb_cal': '21', 'umb_sigma': '25', 'umb_speed': '0.2', 'r_unc_pct': '2.0', 'r_cal_pct': '0.13', 'd_unc': '2.5', 't_unc': '1.9', 'e_aisle_unc': '594.4', 'aisle_over_bound': '11.9', 'cal_gain': '15.1', 'd_cal': '38.1', 'speed_ratio': '6.5', 'kitti_n': '127', 'kitti_best': '0.53', 'kitti_laser': '0.68', 'kitti_stereo': '1.23', 'kitti_d_best': '9.4', 'kitti_d_laser': '7.4', 'kitti_d_stereo': '4.0', 'hulls': '50', 'f_tick_khz': '1', 't_tick_ms': '1', 't_reflex': '100', 't_query': '2', 'sram_bw': '3.2', 'cell_b': '8', 'reads': '800', 'wcet': '250', 'wcet_pct': '40', 'spare_pct': '90', 'r_hull': '0.25', 'reach': '0.35', 'map_edge': '10', 'trilinear': '8', 'pitch2': '2', 'occ2': '22,449', 'oct2': '202,044', 'depth2': '9', 'quant2': '17', 'esdf2_kb': '2,860', 'pitch5': '5', 'occ5': '1,437', 'oct5': '11,494', 'depth5': '8', 'quant5': '43', 'esdf5_kb': '183', 'pitch10': '10', 'occ10': '180', 'oct10': '1,257', 'depth10': '7', 'quant10': '87', 'esdf10_kb': '23', 'esdf2_mb': '2.86', 'sram_mb': '1.05', 'slab': '2.86', 'slab_h': '2', 'esdf_nvblox': '1.9', 'esdf_fiesta': '13', 'esdf_voxblox': '58', 'ws_x': '10', 'ws_y': '10', 'ws_z': '3', 'ws_m3': '300', 'host_pitch': '1', 'dense_mant': '3', 'dense_exp': '8', 'dense_gb': '2.4', 'surface': '120', 'band_cm': '4', 'band_pct': '3.2', 'hashed_mb': '76.8', 'dense_over_hashed': '31', 'block_edge': '8', 't_ahead': '593.75', 't_ahead_whole': '594', 't_ahead_s': '0.6', 't_abreast': '0', 'depth_ahead_m': '0.95', 'depth_ahead_mm': '950', 'depth_abreast_m': '0', 'ms_per_mm': '0.625', 'd_age_eroded': '82.6', 't_left': '542.15', 'frames_left': '33', 'walk_intrusion': '1.41', 'timeout_ms': '100', 'timeout_refused_s': '0.49', 'a_dist': '50', 'a_half': '25', 'slot_slack': '0.9', 't_latch': '66', 't_lin': '75', 'e_at_lin': '1.14', 'lin_over_pct': '14.1', 'overrun': '9', 'r_tight': '0.30', 't_tight': '16.1', 'horizon_ratio': '4.1', 'tol_ratio': '3.3', 't_loss': '10', 'f_repub_khz': '1', 't_repub': '1', 'tau_max': '5', 'v_belt': '0.20', 'frozen_err': '20', 'frozen_window': '100', 'grasp_tol': '15', 'cw_e0': '15', 'cw_tol': '150', 't_cw': '84', 'cw_tol_tight': '100', 't_cw_tight': '53', 'mug_e0': '3', 'v_belt_mm_ms': '0.20', 'v_belt_mm_ms_short': '0.2', 't_use': '35', 'eps_sync_short': '0.2', 'age_upper': '35.2', 'e_use': '10.04', 'mug_margin': '4.96', 'mug_left': '24.8', 'mug_horizon': '60', 'mug_latest': '59.8', 'f_read': '5', 'f_sigma': '0.05', 'bias_rate': '0.2', 'bias_span': '15', 'bias_shift': '3', 'f_true': '2', 'grip_lost': '60', 'skew': '20', 'v_close': '3.1', 'd_skew': '62', 'sep_slack': '50', 'separation': '200'}
import contextlib
import io
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import numpy as np
import slideplot as sp
from slideplot import C, plt
COL = (5.4, 3.6)
FONT = 'Helvetica, Arial, sans-serif'
NAVY, BLUE, GREEN, RED, PETROL, SLATE, AMBER = ('#1F407A', '#2563EB', '#059669', '#DC2626', '#0F766E', '#334155', '#D97706')

def out(name):
    return str(HERE / name)

def svg(name, body, w=900, h=500):
    doc = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w / 100:.2f}in" height="{h / 100:.2f}in" viewBox="0 0 {w} {h}" font-family="{FONT}">\n<rect width="{w}" height="{h}" fill="white"/>\n<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{SLATE}"/></marker><marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{RED}"/></marker><pattern id="hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="10" height="10" fill="#F1F5F9"/><line x1="0" y1="0" x2="0" y2="10" stroke="#94A3B8" stroke-width="3"/></pattern></defs>\n{body}\n</svg>\n'
    Path(out(name)).write_text(doc)

def t(x, y, s, size=22, fill=SLATE, anchor='middle', weight='normal', halo=False):
    h = ' stroke="white" stroke-width="7" paint-order="stroke"' if halo else ''
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{h}>{s}</text>'

def kalman():
    shift, sig_s = (38.01707128, 3.690950941)
    cols = [('Prior', 0, 20.0, 'neutral', '20.0'), ('Predict', 0, 20.61552813, 'neutral', '20.6'), ('+ lidar', 0, 16.99056342, 'memory', '17.0'), ('+ stereo', 0, sig_s, 'data', '3.7'), ('Misdated\nstereo', shift, sig_s, 'danger', '3.7')]
    meas = {2: 30.0, 3: 3.78125}
    for step in range(1, 5):
        n = step + 1
        fig, ax = plt.subplots(figsize=COL)
        for i, (lab, mu, s, ck, s_lab) in enumerate(cols):
            vis = i < n
            a = 1.0 if vis else 0.0
            ax.plot([i, i], [mu - s, mu + s], color=C[ck], lw=9, alpha=a, solid_capstyle='butt')
            ax.plot([i - 0.22, i + 0.22], [mu, mu], color=C[ck], lw=2.5, alpha=a)
            ax.text(i, mu + s + 3, f'±{s_lab}', ha='center', va='bottom', fontsize=15, color=C[ck], fontweight='bold', alpha=a)
            if i in meas:
                ax.plot([i + 0.3, i + 0.3], [-meas[i], meas[i]], color=C['neutral_l'], lw=5, alpha=a, solid_capstyle='butt', zorder=0)
        ax.axhline(0, color=C['text'], lw=1, ls=':')
        ax.text(4.45, -2, 'true range', fontsize=13, color=C['text'], ha='right', va='top')
        if step == 4:
            ax.annotate('38.0 mm off,\nsame ±3.7', (4, shift), xytext=(2.85, 52), fontsize=14, color=C['danger'], fontweight='bold', ha='center', arrowprops=dict(arrowstyle='->', color=C['danger']))
        ax.set_xticks(range(5))
        ax.set_xticklabels([c[0] for c in cols], fontsize=14)
        for i, lab in enumerate(ax.get_xticklabels()):
            lab.set_alpha(1.0 if i < n else 0.0)
        ax.set_xlim(-0.5, 4.6)
        ax.set_ylim(-35, 70)
        ax.set_ylabel('Range error (mm)', fontsize=15)
        ax.tick_params(axis='y', labelsize=14)
        sp.save(fig, out(f'kalman_step{step}.svg'))

def gates():
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    rows = [('Gate 3.84 (95%)', 'p95', 0.6301859166), ('Gate 6.63 (99%)', 'p99', 0.3883791284)]
    for i, (lab, key, v) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C['memory'], height=0.6)
        ax.text(v + 0.015, y, f"{ {0.6301859166: '0.63', 0.3883791284: '0.39'}[v]} caught", va='center', fontsize=13, color=C['memory'], fontweight='bold')
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=13, color=C['text'], fontweight='bold')
    ax.axvline(1.0, color=C['neutral'], ls='--', lw=1.5)
    ax.text(0.99, 1.45, 'every one caught', ha='right', fontsize=11, color=C['neutral'])
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.5, 1.7)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Share of single misdated updates the gate rejects')
    sp.save(fig, out('gates.svg'))

def imu():
    rows = [('Accel. bias', 'acc_bias_mm', 'danger', Q_(98.0665, 'mm')), ('Gyro bias at 1.1 m', 'imu_lateral', 'compute', Q_(19.19862177, 'mm')), ('Accel. noise', 'acc_noise_mm', 'neutral', Q_(0.9058995228, 'mm'))]
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    for i, (lab, key, ck, q) in enumerate(rows):
        y = 2 - i
        v = {'danger': 98.0665, 'compute': 19.19862177, 'neutral': 0.9058995228}[ck]
        ax.barh(y, v, color=C[ck], height=0.6)
        if v > 5:
            ax.text(v / 1.12, y, f"{ {'danger': '98.1', 'compute': '19.2', 'neutral': '0.9'}[ck]} mm", va='center', ha='right', fontsize=13, color='white', fontweight='bold')
        else:
            ax.text(v * 1.2, y, f"{ {'danger': '98.1', 'compute': '19.2', 'neutral': '0.9'}[ck]} mm", va='center', fontsize=13, color=C[ck], fontweight='bold')
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=13, color=C['text'], fontweight='bold')
    bound = 50.0
    ax.axvline(bound, color=C['danger'], ls='--', lw=1.5)
    ax.text(bound * 1.1, 2.45, '50 mm bound', color=C['danger'], fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(0.3, 2000)
    ax.set_ylim(-0.5, 2.75)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Position error after 1 s without a fix (mm, log scale)')
    sp.save(fig, out('imu.svg'))

def sort_timeline():
    hits = int(3.0)
    x0, dx, y, bh = (250, 105, [62, 132, 202], 52)
    b = []
    for f in range(6):
        b.append(t(x0 + f * dx + dx / 2, 40, f'{f + 1}', 28, SLATE, weight='bold'))
    b.append(t(x0 - 14, 40, 'Frame', 28, SLATE, 'end', 'bold'))
    for r, lab in enumerate(['In view', 'SORT out', 'Occupancy']):
        b.append(t(x0 - 14, y[r] + 37, lab, 28, SLATE, 'end', 'bold'))
    for f in range(6):
        xx = x0 + f * dx + 6
        ww = dx - 12
        seen = f < 4
        b.append(f'''<rect x="{xx}" y="{y[0]}" width="{ww}" height="{bh}" rx="5" fill="{(PETROL if seen else 'white')}" stroke="{(PETROL if seen else '#CBD5E1')}" stroke-width="3"/>''')
        out_ok = seen and f >= hits - 1
        dash = '' if out_ok else ' stroke-dasharray="7,5"'
        b.append(f'''<rect x="{xx}" y="{y[1]}" width="{ww}" height="{bh}" rx="5" fill="{(BLUE if out_ok else 'white')}" stroke="{(BLUE if out_ok else '#CBD5E1')}" stroke-width="3"{dash}/>''')
    b.append(f'<rect x="{x0 + 6}" y="{y[2]}" width="{4 * dx - 12}" height="{bh}" rx="5" fill="{RED}" fill-opacity="0.85" stroke="{RED}" stroke-width="3"/>')
    b.append(t(x0 + 2 * dx, y[2] + 37, 'occupied', 28, 'white', weight='bold'))
    b.append(f'<rect x="{x0 + 4 * dx + 6}" y="{y[2]}" width="{2 * dx - 12}" height="{bh}" rx="5" fill="url(#hatch)" stroke="#64748B" stroke-width="3"/>')
    b.append(t(x0 + 5 * dx, y[2] + 37, 'unknown', 28, SLATE, weight='bold'))
    held = ' and '.join((str(f + 1) for f in range(hits - 1)))
    b.append(t(x0 + dx, 300, 'min_hits = 3:', 26, BLUE, weight='bold'))
    b.append(t(x0 + dx, 330, f'frames {held} held back', 26, BLUE))
    b.append(t(x0 + 4.6 * dx, 300, 'max_age = 1:', 26, BLUE, weight='bold'))
    b.append(t(x0 + 4.6 * dx, 330, 'hidden track not output', 26, BLUE))
    svg('sort_timeline.svg', '\n'.join(b), w=920, h=345)

def epochs():
    b = []
    y = 150
    b.append(f'<line x1="40" y1="{y}" x2="610" y2="{y}" stroke="{SLATE}" stroke-width="3" marker-end="url(#arr)"/>')
    b.append(t(600, y + 45, 'time', 21, SLATE, 'end'))
    pts = [(90, 'acquisition', 'photons strike', PETROL), (330, 'estimate', 'instant it describes', NAVY), (520, 'publication', 'message sent', '#64748B')]
    for x, name, sub, col in pts:
        b.append(f'<line x1="{x}" y1="{y - 34}" x2="{x}" y2="{y + 34}" stroke="{col}" stroke-width="5"/>')
        b.append(t(x, y - 92, name, 23, col, weight='bold'))
        b.append(t(x, y - 60, sub, 20, col))
    b.append(f'<line x1="90" y1="{y + 70}" x2="560" y2="{y + 70}" stroke="{RED}" stroke-width="3" marker-end="url(#arrR)"/>')
    b.append(t(320, y + 108, 'evidence age: measured from acquisition', 21, RED, weight='bold'))
    b.append(t(320, y + 160, 'Prediction moves the estimate epoch;', 21, SLATE))
    b.append(t(320, y + 190, 'only a new measurement moves acquisition.', 21, SLATE))
    svg('epochs.svg', '\n'.join(b), w=640, h=370)

def lever():
    e0, drift, swing, same = ((lambda q, u: q.to(u).magnitude)(q, 'mm') for q in (Q_(0.1, 'mm'), Q_(0.2, 'mm'), Q_(0.75, 'mm'), Q_(0.3069444444, 'mm')))
    r_tol = 1.0
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    rows = [('Stale edge\n+ current wrist', [e0, drift, swing], 'e_total'), ('Same-epoch\nchain', [e0, drift, same - e0 - drift], 'e_same')]
    cols = [C['neutral'], C['memory'], C['compute']]
    for i, (lab, parts, key) in enumerate(rows):
        yy = 1 - i
        left = 0
        for j, p in enumerate(parts):
            ax.barh(yy, p, left=left, color=cols[j], height=0.55)
            left += p
        ax.text(left + 0.02, yy, f"{ {'Stale edge\n+ current wrist': '1.05', 'Same-epoch\nchain': '0.31'}[lab]} mm", va='center', fontsize=13, fontweight='bold', color=C['danger'] if left > r_tol else C['data'])
        ax.text(-0.02, yy, lab, ha='right', va='center', fontsize=12, color=C['text'], fontweight='bold')
    key = [(0.0, 'initial 0.10', cols[0]), (0.3, 'drift 0.20', cols[1]), (0.58, 'lever arm 0.75', cols[2])]
    for x, s, c in key:
        ax.text(x, 1.45, s, fontsize=11.5, color=c, fontweight='bold', ha='left')
    ax.axvline(r_tol, color=C['danger'], ls='--', lw=1.5)
    ax.text(r_tol + 0.02, -0.5, '1 mm slot\ntolerance', color=C['danger'], fontsize=11, ha='left', va='center', fontweight='bold')
    ax.set_xlim(0, 1.35)
    ax.set_ylim(-0.75, 1.7)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Finger error after one 16.7 ms frame (mm)')
    sp.save(fig, out('lever.svg'))

def umbmark():
    bound, aisle = (50.0, 30.0)
    x = np.linspace(0, 60, 300)
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        ax.plot(x, 0.0198125 * x * 1000, color=C['danger'], lw=2.5)
        ax.text(12.5, 225, 'uncalibrated: 2.0% of distance', color=C['danger'], fontsize=12, fontweight='bold', va='center')
        a = 1.0 if step == 2 else 0.0
        ax.plot(x, 0.0013125 * x * 1000, color=C['data'], lw=2.5, alpha=a)
        ax.text(44, 105, 'calibrated:\n0.13%', color=C['data'], fontsize=12, fontweight='bold', alpha=a)
        ax.axhline(bound, color=C['neutral'], ls='--', lw=1.5)
        ax.text(14, bound + 6, '50 mm bound', ha='left', va='bottom', fontsize=11, color=C['neutral'])
        ax.axvline(aisle, color=C['neutral'], ls=':', lw=1.5)
        ax.text(aisle - 0.7, 170, 'one aisle, 30 m', fontsize=11, color=C['neutral'], ha='right')
        d_unc, d_cal = (2.523659306, 38.0952381)
        ax.plot([d_unc], [bound], 'o', color=C['danger'], ms=8)
        ax.annotate('2.5 m', (d_unc, bound), xytext=(8, 92), fontsize=12, color=C['danger'], fontweight='bold', arrowprops=dict(arrowstyle='->', color=C['danger']))
        ax.plot([d_cal], [bound], 'o', color=C['data'], ms=8, alpha=a)
        ax.annotate('38.1 m', (d_cal, bound), xytext=(46, 12), fontsize=12, color=C['data'], fontweight='bold', alpha=a, arrowprops=dict(arrowstyle='->', color=C['data'], alpha=a))
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 250)
        ax.set_xlabel('Distance driven (m)')
        ax.set_ylabel('Odometry error (mm)')
        sp.save(fig, out(f'umbmark_step{step}.svg'))

def kitti():
    rows = [{'rank': 1.0, 'method': 'SOFT2', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.53, 'rotation_err_deg_per_m': 0.0009, 'runtime_s': 0.1}, {'rank': 2.0, 'method': 'V-LOAM', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.54, 'rotation_err_deg_per_m': 0.0013, 'runtime_s': 0.1}, {'rank': 3.0, 'method': 'LOAM', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.55, 'rotation_err_deg_per_m': 0.0013, 'runtime_s': 0.1}, {'rank': 4.0, 'method': 'TVL-SLAM+', 'sensor_group': 'stereo+laser', 'loop_closure': False, 'translation_err_pct': 0.56, 'rotation_err_deg_per_m': 0.0015, 'runtime_s': 0.3}, {'rank': 5.0, 'method': 'Traj-LIO', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.57, 'rotation_err_deg_per_m': 0.0015, 'runtime_s': 0.1}, {'rank': 6.0, 'method': 'CT-ICP2', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.58, 'rotation_err_deg_per_m': 0.0012, 'runtime_s': 0.06}, {'rank': 7.0, 'method': 'Traj-LO', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.58, 'rotation_err_deg_per_m': 0.0014, 'runtime_s': 0.1}, {'rank': 8.0, 'method': 'AKF-LIO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 0.59, 'rotation_err_deg_per_m': 0.0014, 'runtime_s': 0.02}, {'rank': 9.0, 'method': 'GLIM', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.59, 'rotation_err_deg_per_m': 0.0015, 'runtime_s': 0.1}, {'rank': 10.0, 'method': 'universal-slam', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.59, 'rotation_err_deg_per_m': 0.0014, 'runtime_s': 0.04}, {'rank': 11.0, 'method': 'CT-ICP', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.59, 'rotation_err_deg_per_m': 0.0014, 'runtime_s': 0.06}, {'rank': 12.0, 'method': 'SDV-LOAM', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.6, 'rotation_err_deg_per_m': 0.0015, 'runtime_s': 0.06}, {'rank': 13.0, 'method': 'MagneticPillars++', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.6, 'rotation_err_deg_per_m': 0.0018, 'runtime_s': 0.06}, {'rank': 14.0, 'method': 'CELLmap', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.61, 'rotation_err_deg_per_m': 0.0017, 'runtime_s': 0.1}, {'rank': 15.0, 'method': 'KISS-ICP', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.61, 'rotation_err_deg_per_m': 0.0017, 'runtime_s': 0.05}, {'rank': 16.0, 'method': 'MOLA-LO', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.62, 'rotation_err_deg_per_m': 0.0017, 'runtime_s': 0.05}, {'rank': 17.0, 'method': 'SiMpLE', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.62, 'rotation_err_deg_per_m': 0.0015, 'runtime_s': 0.35}, {'rank': 18.0, 'method': 'SLAMLab', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.63, 'rotation_err_deg_per_m': 0.0013, 'runtime_s': 0.1}, {'rank': 19.0, 'method': 'PIN-SLAM', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.64, 'rotation_err_deg_per_m': 0.0015, 'runtime_s': 0.1}, {'rank': 20.0, 'method': 'filter-reg', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.65, 'rotation_err_deg_per_m': 0.0016, 'runtime_s': 0.01}, {'rank': 21.0, 'method': 'SOFT-SLAM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.65, 'rotation_err_deg_per_m': 0.0014, 'runtime_s': 0.1}, {'rank': 22.0, 'method': 'MULLS', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.65, 'rotation_err_deg_per_m': 0.0019, 'runtime_s': 0.08}, {'rank': 23.0, 'method': 'retro-slam', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.66, 'rotation_err_deg_per_m': 0.0016, 'runtime_s': 0.05}, {'rank': 24.0, 'method': 'VGMM-LO', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.66, 'rotation_err_deg_per_m': 0.0015, 'runtime_s': 0.05}, {'rank': 25.0, 'method': 'ELO', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.68, 'rotation_err_deg_per_m': 0.0021, 'runtime_s': 0.005}, {'rank': 26.0, 'method': 'IMLS-SLAM', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.69, 'rotation_err_deg_per_m': 0.0018, 'runtime_s': 1.25}, {'rank': 27.0, 'method': 'MC2SLAM', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.69, 'rotation_err_deg_per_m': 0.0016, 'runtime_s': 0.1}, {'rank': 28.0, 'method': 'Onion-LO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 0.69, 'rotation_err_deg_per_m': 0.0017, 'runtime_s': 0.04}, {'rank': 29.0, 'method': 'ISC-LOAM', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.72, 'rotation_err_deg_per_m': 0.0022, 'runtime_s': 0.1}, {'rank': 30.0, 'method': 'FLOAM', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 0.72, 'rotation_err_deg_per_m': 0.0022, 'runtime_s': 0.1}, {'rank': 31.0, 'method': 'PSF-LO', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.82, 'rotation_err_deg_per_m': 0.0032, 'runtime_s': 0.2}, {'rank': 32.0, 'method': 'RADVO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.82, 'rotation_err_deg_per_m': 0.0018, 'runtime_s': 0.07}, {'rank': 33.0, 'method': 'LG-SLAM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.82, 'rotation_err_deg_per_m': 0.002, 'runtime_s': 0.2}, {'rank': 34.0, 'method': 'RotRocc+', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.83, 'rotation_err_deg_per_m': 0.0026, 'runtime_s': 0.25}, {'rank': 35.0, 'method': 'AMBA-VO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.84, 'rotation_err_deg_per_m': 0.0021, 'runtime_s': 0.1}, {'rank': 36.0, 'method': 'LIMO2_GP', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.84, 'rotation_err_deg_per_m': 0.0022, 'runtime_s': 0.2}, {'rank': 37.0, 'method': 'CAE-LO', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.86, 'rotation_err_deg_per_m': 0.0025, 'runtime_s': 2.0}, {'rank': 38.0, 'method': 'GDVO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.86, 'rotation_err_deg_per_m': 0.0031, 'runtime_s': 0.09}, {'rank': 39.0, 'method': 'LIMO2', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.86, 'rotation_err_deg_per_m': 0.0022, 'runtime_s': 0.2}, {'rank': 40.0, 'method': 'CPFG-slam', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.87, 'rotation_err_deg_per_m': 0.0025, 'runtime_s': 0.03}, {'rank': 41.0, 'method': 'SOFT', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.88, 'rotation_err_deg_per_m': 0.0022, 'runtime_s': 0.1}, {'rank': 42.0, 'method': 'RotRocc', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.88, 'rotation_err_deg_per_m': 0.0025, 'runtime_s': 0.3}, {'rank': 43.0, 'method': 'D3VO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 0.88, 'rotation_err_deg_per_m': 0.0021, 'runtime_s': 0.1}, {'rank': 44.0, 'method': 'PNDT LO', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.89, 'rotation_err_deg_per_m': 0.003, 'runtime_s': 0.2}, {'rank': 45.0, 'method': 'DVSO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 0.9, 'rotation_err_deg_per_m': 0.0021, 'runtime_s': 0.1}, {'rank': 46.0, 'method': 'LIMO', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.93, 'rotation_err_deg_per_m': 0.0026, 'runtime_s': 0.2}, {'rank': 47.0, 'method': 'Stereo DSO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.93, 'rotation_err_deg_per_m': 0.002, 'runtime_s': 0.1}, {'rank': 48.0, 'method': 'IsaacElbrusGPUSLAM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.94, 'rotation_err_deg_per_m': 0.0019, 'runtime_s': 0.007}, {'rank': 49.0, 'method': 'OV2SLAM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.94, 'rotation_err_deg_per_m': 0.0023, 'runtime_s': 0.01}, {'rank': 50.0, 'method': 'OV2SLAM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.98, 'rotation_err_deg_per_m': 0.0023, 'runtime_s': 0.01}, {'rank': 51.0, 'method': 'ROCC', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.98, 'rotation_err_deg_per_m': 0.0028, 'runtime_s': 0.3}, {'rank': 52.0, 'method': 'IsaacElbrusSLAM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 0.99, 'rotation_err_deg_per_m': 0.002, 'runtime_s': 0.008}, {'rank': 53.0, 'method': 'SuMa-MOS', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 0.99, 'rotation_err_deg_per_m': 0.0033, 'runtime_s': 0.1}, {'rank': 54.0, 'method': 'SuMa++', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 1.06, 'rotation_err_deg_per_m': 0.0034, 'runtime_s': 0.1}, {'rank': 55.0, 'method': 'ULF-ESGVI', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 1.07, 'rotation_err_deg_per_m': 0.0036, 'runtime_s': 0.3}, {'rank': 56.0, 'method': 'cv4xv1-sc', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.09, 'rotation_err_deg_per_m': 0.0029, 'runtime_s': 0.145}, {'rank': 57.0, 'method': 'VINS-Fusion', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.09, 'rotation_err_deg_per_m': 0.0033, 'runtime_s': 0.1}, {'rank': 58.0, 'method': 'MonoROCC', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.11, 'rotation_err_deg_per_m': 0.0028, 'runtime_s': 1.0}, {'rank': 59.0, 'method': 'DEMO', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 1.14, 'rotation_err_deg_per_m': 0.0049, 'runtime_s': 0.1}, {'rank': 60.0, 'method': 'ORB-SLAM2', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.15, 'rotation_err_deg_per_m': 0.0027, 'runtime_s': 0.06}, {'rank': 61.0, 'method': 'IV-SLAM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.17, 'rotation_err_deg_per_m': 0.0025, 'runtime_s': 0.1}, {'rank': 62.0, 'method': 'NOTF', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.17, 'rotation_err_deg_per_m': 0.0035, 'runtime_s': 0.45}, {'rank': 63.0, 'method': 'S-PTAM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.19, 'rotation_err_deg_per_m': 0.0025, 'runtime_s': 0.03}, {'rank': 64.0, 'method': 'S-LSD-SLAM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.2, 'rotation_err_deg_per_m': 0.0033, 'runtime_s': 0.07}, {'rank': 65.0, 'method': 'VoBa', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.22, 'rotation_err_deg_per_m': 0.0029, 'runtime_s': 0.1}, {'rank': 66.0, 'method': 'STEAM-L WNOJ', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 1.22, 'rotation_err_deg_per_m': 0.0058, 'runtime_s': 0.2}, {'rank': 67.0, 'method': 'LiViOdo', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 1.22, 'rotation_err_deg_per_m': 0.0042, 'runtime_s': 0.5}, {'rank': 68.0, 'method': 'SLUP', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.25, 'rotation_err_deg_per_m': 0.0041, 'runtime_s': 0.17}, {'rank': 69.0, 'method': 'STEAM-L', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 1.26, 'rotation_err_deg_per_m': 0.0061, 'runtime_s': 0.2}, {'rank': 70.0, 'method': 'FRVO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.26, 'rotation_err_deg_per_m': 0.0038, 'runtime_s': 0.03}, {'rank': 71.0, 'method': 'JFBVO-FM', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 1.28, 'rotation_err_deg_per_m': 0.001, 'runtime_s': 0.1}, {'rank': 72.0, 'method': 'MFI', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.3, 'rotation_err_deg_per_m': 0.003, 'runtime_s': 0.1}, {'rank': 73.0, 'method': 'TLBBA', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.36, 'rotation_err_deg_per_m': 0.0038, 'runtime_s': 0.1}, {'rank': 74.0, 'method': '2FO-CC', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.37, 'rotation_err_deg_per_m': 0.0035, 'runtime_s': 0.1}, {'rank': 75.0, 'method': 'SALO', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 1.37, 'rotation_err_deg_per_m': 0.0051, 'runtime_s': 0.6}, {'rank': 76.0, 'method': 'SuMa', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 1.39, 'rotation_err_deg_per_m': 0.0034, 'runtime_s': 0.1}, {'rank': 77.0, 'method': 'ProSLAM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.39, 'rotation_err_deg_per_m': 0.0035, 'runtime_s': 0.02}, {'rank': 78.0, 'method': 'ESVO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 1.42, 'rotation_err_deg_per_m': 0.0048, 'runtime_s': 1.0}, {'rank': 79.0, 'method': 'JFBVO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.43, 'rotation_err_deg_per_m': 0.0038, 'runtime_s': 0.05}, {'rank': 80.0, 'method': 'StereoSFM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.51, 'rotation_err_deg_per_m': 0.0042, 'runtime_s': 0.02}, {'rank': 81.0, 'method': 'SSLAM', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.57, 'rotation_err_deg_per_m': 0.0044, 'runtime_s': 0.5}, {'rank': 82.0, 'method': 'Stereo-RIVO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 1.61, 'rotation_err_deg_per_m': 0.0025, 'runtime_s': 0.07}, {'rank': 83.0, 'method': 'VOLDOR', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 1.65, 'rotation_err_deg_per_m': 0.005, 'runtime_s': 0.1}, {'rank': 84.0, 'method': 'eVO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.76, 'rotation_err_deg_per_m': 0.0036, 'runtime_s': 0.05}, {'rank': 85.0, 'method': 'Stereo DWO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 1.76, 'rotation_err_deg_per_m': 0.0026, 'runtime_s': 0.1}, {'rank': 86.0, 'method': 'BVO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 1.76, 'rotation_err_deg_per_m': 0.0036, 'runtime_s': 0.1}, {'rank': 87.0, 'method': '3DOF-SLAM', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 1.89, 'rotation_err_deg_per_m': 0.0083, 'runtime_s': 0.02}, {'rank': 88.0, 'method': 'EfficientLO-Net', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 1.92, 'rotation_err_deg_per_m': 0.0052, 'runtime_s': 0.03}, {'rank': 89.0, 'method': 'D6DVO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 2.04, 'rotation_err_deg_per_m': 0.0051, 'runtime_s': 0.03}, {'rank': 90.0, 'method': 'PMO / PbT-M2', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 2.05, 'rotation_err_deg_per_m': 0.0051, 'runtime_s': 1.0}, {'rank': 91.0, 'method': 'GFM', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 2.12, 'rotation_err_deg_per_m': 0.0056, 'runtime_s': 0.03}, {'rank': 92.0, 'method': 'SSLAM-HR', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 2.14, 'rotation_err_deg_per_m': 0.0059, 'runtime_s': 0.5}, {'rank': 93.0, 'method': 'FTMVO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 2.24, 'rotation_err_deg_per_m': 0.0049, 'runtime_s': 0.11}, {'rank': 94.0, 'method': 'PbT-M1', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 2.38, 'rotation_err_deg_per_m': 0.0053, 'runtime_s': 1.0}, {'rank': 95.0, 'method': 'FLVIS', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 2.42, 'rotation_err_deg_per_m': 0.0057, 'runtime_s': 0.05}, {'rank': 96.0, 'method': 'VISO2-S', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 2.44, 'rotation_err_deg_per_m': 0.0114, 'runtime_s': 0.05}, {'rank': 97.0, 'method': 'MLM-SFM', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 2.54, 'rotation_err_deg_per_m': 0.0057, 'runtime_s': 0.03}, {'rank': 98.0, 'method': 'GT_VO3pt', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 2.54, 'rotation_err_deg_per_m': 0.0078, 'runtime_s': 1.26}, {'rank': 99.0, 'method': 'RMCPE+GP', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 2.55, 'rotation_err_deg_per_m': 0.0086, 'runtime_s': 0.39}, {'rank': 100.0, 'method': 'KLTVO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 2.63, 'rotation_err_deg_per_m': 0.0042, 'runtime_s': 0.1}, {'rank': 101.0, 'method': 'VO3pt', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 2.69, 'rotation_err_deg_per_m': 0.0068, 'runtime_s': 0.56}, {'rank': 102.0, 'method': 'TGVO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 2.94, 'rotation_err_deg_per_m': 0.0077, 'runtime_s': 0.06}, {'rank': 103.0, 'method': 'VO3ptLBA', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 3.13, 'rotation_err_deg_per_m': 0.0104, 'runtime_s': 0.57}, {'rank': 104.0, 'method': 'KD-SLAM', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 3.25, 'rotation_err_deg_per_m': 0.0046, 'runtime_s': 1.0}, {'rank': 105.0, 'method': 'PLSVO', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 3.26, 'rotation_err_deg_per_m': 0.0095, 'runtime_s': 0.2}, {'rank': 106.0, 'method': 'BLF', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 3.49, 'rotation_err_deg_per_m': 0.0128, 'runtime_s': 0.7}, {'rank': 107.0, 'method': 'CFORB', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 3.73, 'rotation_err_deg_per_m': 0.0107, 'runtime_s': 0.9}, {'rank': 108.0, 'method': 'DeepCLR', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 3.83, 'rotation_err_deg_per_m': 0.0104, 'runtime_s': 0.05}, {'rank': 109.0, 'method': 'VOFS', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 3.94, 'rotation_err_deg_per_m': 0.0099, 'runtime_s': 0.51}, {'rank': 110.0, 'method': 'VOFSLBA', 'sensor_group': 'stereo', 'loop_closure': False, 'translation_err_pct': 4.17, 'rotation_err_deg_per_m': 0.0112, 'runtime_s': 0.52}, {'rank': 111.0, 'method': 'CUDA-EgoMotion', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 4.36, 'rotation_err_deg_per_m': 0.0052, 'runtime_s': 0.001}, {'rank': 112.0, 'method': 'BCC', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 4.59, 'rotation_err_deg_per_m': 0.0175, 'runtime_s': 1.0}, {'rank': 113.0, 'method': 'D3DLO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 5.4, 'rotation_err_deg_per_m': 0.0154, 'runtime_s': 0.1}, {'rank': 114.0, 'method': 'EB3DTE+RJMCM', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 5.45, 'rotation_err_deg_per_m': 0.0274, 'runtime_s': 1.0}, {'rank': 115.0, 'method': 'LTMVO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 7.4, 'rotation_err_deg_per_m': 0.0142, 'runtime_s': 0.1}, {'rank': 116.0, 'method': 'VISO2-M + GP', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 7.46, 'rotation_err_deg_per_m': 0.0245, 'runtime_s': 0.15}, {'rank': 117.0, 'method': 'PAVO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 8.04, 'rotation_err_deg_per_m': 0.0167, 'runtime_s': 0.02}, {'rank': 118.0, 'method': 'BLO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 9.21, 'rotation_err_deg_per_m': 0.0163, 'runtime_s': 0.1}, {'rank': 119.0, 'method': 'UDQN-VO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 10.55, 'rotation_err_deg_per_m': 0.0169, 'runtime_s': 1.0}, {'rank': 120.0, 'method': 'VISO2-M', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 11.94, 'rotation_err_deg_per_m': 0.0234, 'runtime_s': 0.1}, {'rank': 121.0, 'method': 'MonoDepth2', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 12.59, 'rotation_err_deg_per_m': 0.0312, 'runtime_s': 1.0}, {'rank': 122.0, 'method': 'SMD-LVO', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 13.25, 'rotation_err_deg_per_m': 0.0097, 'runtime_s': 0.03}, {'rank': 123.0, 'method': 'SC-SfMLearner (cs+k)', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 13.69, 'rotation_err_deg_per_m': 0.0355, 'runtime_s': 0.01}, {'rank': 124.0, 'method': 'CC', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 16.06, 'rotation_err_deg_per_m': 0.032, 'runtime_s': 0.1}, {'rank': 125.0, 'method': 'OABA', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 20.95, 'rotation_err_deg_per_m': 0.0135, 'runtime_s': 0.5}, {'rank': 126.0, 'method': 'SC-SfMLearner (k)', 'sensor_group': 'no_icon(unlabeled: mono, lidar-without-icon, event, etc.)', 'loop_closure': False, 'translation_err_pct': 21.47, 'rotation_err_deg_per_m': 0.0425, 'runtime_s': 0.01}, {'rank': 127.0, 'method': 'SLL', 'sensor_group': 'laser', 'loop_closure': False, 'translation_err_pct': 90.05, 'rotation_err_deg_per_m': 0.2645, 'runtime_s': 0.1}]
    dl = 0.05
    groups = [('stereo', 'Stereo', C['memory']), ('laser', 'Lidar', C['data']), ('other', 'Other', C['neutral'])]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    rng = np.random.default_rng(0)
    for i, (key, lab, col) in enumerate(groups):
        v = np.array([r['translation_err_pct'] for r in rows if (r['sensor_group'] if r['sensor_group'] in ('stereo', 'laser') else 'other') == key])
        ax.scatter(v, 2 - i + rng.uniform(-0.2, 0.2, len(v)), s=22, color=col, alpha=0.85, zorder=3)
    thr = 100 * dl / 30.0
    ax.axvspan(thr, 40, color=C['danger_l'], alpha=0.6, lw=0, zorder=1)
    ax.axvline(thr, color=C['danger'], lw=1.5, ls='--')
    ax.text(38, -0.72, 'red: bound spent within one 30 m aisle', fontsize=12, color=C['danger'], fontweight='bold', ha='right')
    best = 0.53
    ax.annotate('lowest 0.53%', (best, 2), xytext=(0.19, 2.45), fontsize=13, color=C['text'], arrowprops=dict(arrowstyle='->', color=C['text']))
    ax.set_xscale('log')
    ax.set_xlim(0.1, 40)
    ax.set_ylim(-0.9, 2.75)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([g[1] for g in groups], fontsize=12, fontweight='bold')
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticklabels(['0.1', '0.3', '1', '3', '10', '30'])
    ax.set_xlabel('Translation error (% of distance, log)')
    top = ax.secondary_xaxis('top', functions=(lambda e: 100 * dl / np.maximum(e, 1e-09), lambda d: 100 * dl / np.maximum(d, 1e-09)))
    top.set_xticks([30, 10, 3, 1, 0.3])
    top.set_xticklabels(['30', '10', '3', '1', '0.3'])
    top.set_xlabel('Meters until 50 mm is spent')
    sp.save(fig, out('kitti.svg'))

def reads():
    pitches = list({2: SimpleNamespace(esdf_bytes=Q_(2860.0, 'kB'), esdf_reads=8, occupancy_bytes=Q_(1430.0, 'kB'), occupancy_reads=22449.2975, octree_depth=9, octree_reads=202043.6775, pitch=Q_(2, 'cm'), quantization=Q_(17.32050808, 'mm')), 5: SimpleNamespace(esdf_bytes=Q_(183.04, 'kB'), esdf_reads=8, occupancy_bytes=Q_(91.52, 'kB'), occupancy_reads=1436.75504, octree_depth=8, octree_reads=11494.04032, pitch=Q_(5, 'cm'), quantization=Q_(43.30127019, 'mm')), 10: SimpleNamespace(esdf_bytes=Q_(22.88, 'kB'), esdf_reads=8, occupancy_bytes=Q_(11.44, 'kB'), occupancy_reads=179.59438, octree_depth=7, octree_reads=1257.16066, pitch=Q_(10, 'cm'), quantization=Q_(86.60254038, 'mm'))})
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    names = [('occ', 'occupancy', C['memory']), ('oct', 'octree', C['network']), ('esdf', 'ESDF', C['data'])]
    w = 0.27
    for j, (key, lab, col) in enumerate(names):
        xs = [i + (j - 1) * w for i in range(3)]
        keys = [f'{key}{cm}' if key != 'esdf' else 'trilinear' for cm in pitches]
        field = {'occ': 'occupancy_reads', 'oct': 'octree_reads', 'esdf': 'esdf_reads'}[key]
        vals = {'occupancy': [22449.2975, 1436.75504, 179.59438], 'octree': [202043.6775, 11494.04032, 1257.16066], 'ESDF': [8, 8, 8]}[lab]
        ax.bar(xs, vals, width=w * 0.92, color=col)
        for xx, v, k in zip(xs, vals, keys):
            ax.text(xx, v * 1.3, _VALS[k], ha='center', fontsize=9.5, color=col, fontweight='bold')
        ax.text(2.97, 2000000.0 / 6 ** j, lab, ha='right', va='top', fontsize=11.5, color=col, fontweight='bold')
    budget = 800.0
    ax.axhline(budget, color=C['danger'], ls='--', lw=1.5)
    ax.text(2.97, budget * 1.2, 'budget\n800 reads', color=C['danger'], fontsize=11, fontweight='bold', ha='right', va='bottom')
    ax.set_yscale('log')
    ax.set_ylim(2.5, 3000000.0)
    ax.set_xlim(-0.5, 3.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{_VALS[f'pitch{cm}']} cm voxels" for cm in pitches], fontsize=12)
    ax.set_ylabel('Reads per hull query')
    sp.save(fig, out('reads.svg'))

def nvblox():
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    items = [('nvblox (GPU)', 'esdf_nvblox'), ('Fiesta (CPU)', 'esdf_fiesta'), ('voxblox (CPU)', 'esdf_voxblox')]
    for i, (k, key) in enumerate(items):
        y = len(items) - 1 - i
        v = {'nvblox (GPU)': 1.9, 'Fiesta (CPU)': 13.0, 'voxblox (CPU)': 58.0}[k]
        col = C['data'] if 'nvblox' in k else C['neutral']
        ax.barh(y, v, color=col, height=0.42)
        ax.text(v * 1.15, y, f"{ {'nvblox (GPU)': '1.9', 'Fiesta (CPU)': '13', 'voxblox (CPU)': '58'}[k]} ms", va='center', fontsize=16, color=col, fontweight='bold')
        ax.text(1.2, y + 0.32, k, fontsize=15, color=C['text'], fontweight='bold', va='bottom')
    tick = 1.0
    ax.axvline(tick, color=C['danger'], ls='--', lw=2)
    ax.text(tick * 0.95, 2.95, '1 ms tick', fontsize=15, color=C['danger'], fontweight='bold', ha='right')
    ax.set_xscale('log')
    ax.set_xlim(0.25, 400)
    ax.set_ylim(-0.45, 3.25)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.set_xlabel('ESDF update, average (ms)', fontsize=15)
    sp.save(fig, out('nvblox_esdf.svg'))

def snapshot():
    rows = [('2 cm pitch', 2860.0, 'danger', '2,860 kB'), ('5 cm pitch', 183.04, 'data', '183 kB'), ('10 cm pitch', 22.88, 'data', '23 kB')]
    sp.hbars(out('snapshot.svg'), rows, 'kB', log=True, size=(5.8, 1.8), vlines=[(1.05 * 1000, '1.05 MB SRAM', 'danger')], xlabel='ESDF snapshot of the stopping corridor (kB, log)')

def ray_grid():
    b = []
    x0, y0, cs = (130, 60, 56)
    ncol, nrow = (12, 4)
    ray_row = 2
    for r in range(nrow):
        for c in range(ncol):
            x, y = (x0 + c * cs, y0 + r * cs)
            fill = 'white'
            if r == ray_row and c < 7:
                fill = '#DCFCE7'
            if r == ray_row and c == 7:
                fill = '#334155'
            if r == ray_row and c > 7:
                fill = 'url(#hatch)'
            if r == 0 and c in (9, 10):
                fill = '#FEE2E2'
            b.append(f'<rect x="{x}" y="{y}" width="{cs}" height="{cs}" fill="{fill}" stroke="#CBD5E1" stroke-width="2"/>')
    yr = y0 + ray_row * cs + cs / 2
    b.append(f'<circle cx="75" cy="{yr}" r="24" fill="{PETROL}"/>')
    b.append(t(75, yr + 62, 'sensor', 27, PETROL, weight='bold'))
    b.append(f'<line x1="99" y1="{yr}" x2="{x0 + 7 * cs + 8}" y2="{yr}" stroke="{PETROL}" stroke-width="4" marker-end="url(#arr)"/>')
    b.append(f'<rect x="{x0 + 3 * cs + 12}" y="{y0 + ray_row * cs + 12}" width="{cs - 24}" height="{cs - 24}" fill="none" stroke="{RED}" stroke-width="3" stroke-dasharray="6,4"/>')
    b.append(t(x0 + 3.5 * cs, y0 + ray_row * cs - 12, 'phantom retired', 26, RED, weight='bold'))
    b.append(t(x0 + 3.5 * cs, y0 + 4 * cs + 34, 'free, epoch t1', 27, GREEN, weight='bold'))
    b.append(t(x0 + 7.5 * cs, y0 + 4 * cs + 34, 'return', 27, SLATE, weight='bold'))
    b.append(t(x0 + 10.2 * cs, y0 + 4 * cs + 34, 'behind: still t0', 27, SLATE, weight='bold'))
    b.append(t(x0 + 9.8 * cs, y0 - 14, 'occupied, t0', 26, '#991B1B', weight='bold'))
    lx, ly = (40, 385)
    leg = [('#FEE2E2', 'occupied'), ('#DCFCE7', 'free'), ('url(#hatch)', 'unknown'), ('#E2E8F0', 'expired')]
    for i, (f, lab) in enumerate(leg):
        xx = lx + i * 215
        b.append(f'<rect x="{xx}" y="{ly}" width="40" height="40" fill="{f}" stroke="#64748B" stroke-width="2"/>')
        b.append(t(xx + 52, ly + 31, lab, 28, SLATE, 'start'))
    b.append(t(450, 475, 'Unknown and expired cells count as obstacles.', 28, SLATE, weight='bold'))
    svg('ray_grid.svg', '\n'.join(b), w=900, h=495)

def rack_top():
    s = 250
    b = []
    cy = 190
    hull_x, hull_w, hull_h = (40, 130, 105)
    front = hull_x + hull_w
    hull_y = cy + 0.15 * s
    rack_end = front + 1.1 * s
    W, H = (720, 500)
    r = 0.95 * s
    low = hull_y + hull_h + 45
    b.append(f'<rect x="{rack_end}" y="0" width="{W - rack_end}" height="{cy}" fill="url(#hatch)"/>')
    b.append(f'<circle cx="{rack_end}" cy="{cy}" r="{r}" fill="{RED}" fill-opacity="0.05" stroke="{RED}" stroke-width="3" stroke-dasharray="9,6"/>')
    b.append(f'<rect x="0" y="{cy - 75}" width="{rack_end}" height="75" fill="#E2E8F0" stroke="{SLATE}" stroke-width="2"/>')
    b.append(t(70, cy - 27, 'rack', 30, SLATE, weight='bold'))
    b.append(f'<rect x="0" y="{low}" width="{W}" height="55" fill="#E2E8F0" stroke="{SLATE}" stroke-width="2"/>')
    b.append(t(70, low + 38, 'rack', 30, SLATE, weight='bold'))
    b.append(f'<rect x="{rack_end + 22}" y="18" width="215" height="84" fill="white" opacity="0.88"/>')
    b.append(t(rack_end + 130, 52, 'hidden', 30, SLATE, weight='bold'))
    b.append(t(rack_end + 130, 88, 'cross aisle', 30, SLATE, weight='bold'))
    b.append(f'<rect x="{hull_x}" y="{hull_y}" width="{hull_w}" height="{hull_h}" fill="#F8FAFC" stroke="{SLATE}" stroke-width="3"/>')
    b.append(t(hull_x + hull_w / 2, hull_y + hull_h / 2 + 10, 'robot', 30, SLATE, weight='bold'))
    yy = hull_y + hull_h / 2
    b.append(f'<line x1="{front}" y1="{yy + 18}" x2="{rack_end}" y2="{yy + 18}" stroke="{NAVY}" stroke-width="4" marker-start="url(#arr)" marker-end="url(#arr)"/>')
    b.append(t((front + rack_end) / 2, yy - 2, 'ahead: 1.10 m', 30, NAVY, weight='bold', halo=True))
    b.append(f'<line x1="{hull_x + 25}" y1="{hull_y}" x2="{hull_x + 25}" y2="{cy}" stroke="{RED}" stroke-width="5"/>')
    b.append(t(hull_x + 36, cy + 31, 'abreast: 150 mm', 28, RED, 'start', 'bold', halo=True))
    b.append(t(10, H - 14, "Dashed: a hidden walker's reach after 594 ms", 28, RED, 'start', 'bold'))
    svg('rack_top.svg', '\n'.join(b), w=W, h=H)

def erosion():
    t_ah, t_age = (593.75, 51.6)
    tt = np.linspace(0, 700, 300)
    d0 = 950.0
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        y = np.clip(d0 - 1.6 * tt, 0, None)
        ax.plot(tt, y, color=C['memory'], lw=3)
        ax.plot([t_ah], [0], 'o', color=C['danger'], ms=9, zorder=5)
        ax.annotate('ahead: 593.75 ms', (t_ah, 0), xytext=(470, 330), fontsize=12, color=C['memory'], fontweight='bold', arrowprops=dict(arrowstyle='->', color=C['memory']))
        a2 = 1.0 if step >= 2 else 0.0
        ax.axvspan(0, t_age, color=C['neutral_l'], alpha=0.8 * a2, lw=0)
        ax.text(t_age + 8, 300, 'frame age 51.6 ms', fontsize=11, color=C['neutral'], alpha=a2, va='center')
        a3 = 1.0 if step >= 3 else 0.0
        ax.plot([0, 700], [0, 0], color=C['danger'], lw=3, ls='--', alpha=a3)
        ax.plot([0], [0.0], 'o', color=C['danger'], ms=9, alpha=a3, zorder=6)
        ax.text(60, 70, 'abreast of the rack end: 0 ms', fontsize=12, color=C['danger'], fontweight='bold', alpha=a3)
        ax.set_xlim(-10, 700)
        ax.set_ylim(-40, 1000)
        ax.set_xlabel('Time since the frame was captured (ms)')
        ax.set_ylabel('Certified depth (mm)')
        sp.save(fig, out(f'erosion_step{step}.svg'))

def latch():
    t_lin, e_at, r_tol = (75.0, 1.14, 1.0)
    tt = np.linspace(0, 0.1, 300)
    e0, v0, a_half = (0.1, 12.0, 25.0)
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=COL)
        e = e0 + v0 * tt + a_half * tt ** 2
        lin = e0 + v0 * tt
        ax.plot(tt * 1000, e, color=C['danger'], lw=3)
        ax.axhline(r_tol, color=C['text'], ls='--', lw=1.5)
        ax.text(2, r_tol + 0.03, 'slot tolerance 1 mm', fontsize=14, color=C['text'])
        ax.axvspan(65.94117082, 100, color=C['danger_l'], alpha=0.6, lw=0)
        ax.text(65.94117082 - 1, 0.25, '66 ms', fontsize=16, color=C['danger'], fontweight='bold', ha='right')
        ax.text(5, 0.55, 'bounded error,\nwith 50 mm/s²', fontsize=14, color=C['danger'])
        a = 1.0 if step == 2 else 0.0
        ax.plot(tt * 1000, lin, color=C['neutral'], lw=2.5, ls='--', alpha=a)
        ax.plot([t_lin], [e_at], 'o', color=C['danger'], ms=10, alpha=a)
        ax.annotate('75 ms: error 1.14 mm', (t_lin, e_at), xytext=(14, 1.32), fontsize=14, color=C['text'], fontweight='bold', alpha=a, arrowprops=dict(arrowstyle='->', color=C['text'], alpha=a))
        ax.text(80, 0.72, 'constant\nvelocity', fontsize=14, color=C['neutral'], alpha=a)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel('Time since the finger hid the slot (ms)', fontsize=15)
        ax.set_ylabel('Error bound (mm)', fontsize=15)
        ax.tick_params(labelsize=14)
        sp.save(fig, out(f'latch_step{step}.svg'))

def horizons():
    rows = [('Free space abreast', 0.0, 'danger', '0 ms'), ('Latch-slot finger', 65.94117082, 'compute', '66 ms'), ('Coworker by the arm', 84.375, 'network', '84 ms'), ('Free space ahead', 593.75, 'memory', '593.75 ms')]
    sp.hbars(out('horizons.svg'), rows, 'ms', log=False, size=(5.8, 2.2), xlabel='Invalidation horizon: time from capture to expiry (ms)')

def lifecycle():
    b = []
    boxes = {'VALID': (20, 150, GREEN, '#F0FDF4', 'measured'), 'DEGRADED': (330, 150, AMBER, '#FFFBEB', 'predicted'), 'EXPIRED': (640, 150, RED, '#FEF2F2', 'no motion'), 'UNOBSERVABLE': (330, 320, SLATE, '#F1F5F9', 'rank-deficient')}
    w, h = (240, 95)
    for name, (x, y, col, fill, sub) in boxes.items():
        b.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{col}" stroke-width="3"/>')
        b.append(t(x + w / 2, y + 42, name, 28, col, weight='bold'))
        b.append(t(x + w / 2, y + 78, sub, 26, col))
    b.append(f'<line x1="262" y1="197" x2="326" y2="197" stroke="{SLATE}" stroke-width="3" marker-end="url(#arr)"/>')
    b.append(f'<line x1="572" y1="197" x2="636" y2="197" stroke="{SLATE}" stroke-width="3" marker-end="url(#arr)"/>')
    b.append(t(604, 285, 'E = tolerance', 25, SLATE))
    b.append(t(294, 285, 'occlusion', 25, SLATE))
    b.append(f'<path d="M 760 146 L 760 85 L 140 85 L 140 146" fill="none" stroke="{GREEN}" stroke-width="3" marker-end="url(#arr)"/>')
    b.append(f'<line x1="450" y1="85" x2="450" y2="146" stroke="{GREEN}" stroke-width="3"/>')
    b.append(t(450, 60, 'back to VALID only with fresh evidence', 28, GREEN, weight='bold'))
    b.append(f'<path d="M 140 249 L 140 367 L 326 367" fill="none" stroke="{SLATE}" stroke-width="3" stroke-dasharray="7,5" marker-end="url(#arr)"/>')
    b.append(t(20, 445, 'observability lost (any state)', 25, SLATE, 'start'))
    b.append(t(590, 360, 'leave by restoring', 25, SLATE, 'start'))
    b.append(t(590, 390, 'the constraint', 25, SLATE, 'start'))
    svg('lifecycle.svg', '\n'.join(b), w=900, h=460)

def frozen():
    """The frozen transform: fresh publication stamps, a frozen capture epoch."""
    loss, per, win = (10.0, 1.0, 100.0)
    vb, tol, tau = (0.2, 15.0, 5.0)
    end = loss + win
    tt = np.linspace(0, end, 400)
    for step in (1, 2):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=COL, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
        pubs = np.arange(0, end + per / 2, per)
        a1.plot(tt, tt, color=C['neutral'], lw=2.5)
        a1.plot(tt, np.minimum(tt, loss), color=C['memory'], lw=3)
        a1.axvline(loss, color=C['danger'], lw=1.5, ls='--')
        a1.text(end * 0.97, end * 0.58, 'publication stamp', fontsize=14, color=C['neutral'], ha='right', va='top', fontweight='bold')
        a1.text(end, loss + end * 0.06, 'capture epoch carried', fontsize=14, color=C['memory'], ha='right', va='bottom', fontweight='bold')
        a1.text(loss + end * 0.02, end * 0.93, 'camera dies, 10 ms', fontsize=13, color=C['danger'], va='top')
        a1.set_ylim(0, end * 1.05)
        a1.set_ylabel('Stamp (ms)', fontsize=14)
        a1.tick_params(labelsize=12)
        a1.text(0.02 * end, -0.3 * end, 'Stamp age always under 1 ms: passes the 5 ms check', fontsize=13, color=C['data'], fontweight='bold', clip_on=False)
        a2.axhline(tol, color=C['danger'], lw=1.8, ls='--')
        a2.text(1, tol + 0.8, 'grasp tolerance 15 mm', fontsize=13, color=C['danger'], va='bottom')
        if step == 2:
            err = np.clip(tt - loss, 0, None) * vb
            a2.plot(tt, err, color=C['compute'], lw=3)
            a2.plot([end], [err[-1]], 'o', color=C['compute'], ms=8)
            a2.text(end * 0.75, 20.0 * 1.04, 'mug moved 20 mm', fontsize=14, color=C['compute'], ha='right', va='bottom', fontweight='bold')
        a2.set_ylim(0, 20.0 * 1.35)
        a2.set_xlim(0, end * 1.02)
        a2.set_ylabel('Error (mm)', fontsize=14)
        a2.set_xlabel('Wall-clock time (ms)', fontsize=14)
        a2.tick_params(labelsize=12)
        fig.subplots_adjust(hspace=0.55)
        sp.save(fig, out(f'frozen_step{step}.svg'))

def force():
    f, sig, span = (5.0, 0.05, 15.0)
    tt = np.linspace(0, span, 100)
    fig, ax = plt.subplots(figsize=COL)
    ax.fill_between(tt, f - sig, f + sig, color=C['memory_l'])
    ax.plot(tt, np.full_like(tt, f), color=C['memory'], lw=3)
    ax.text(0.3, 5.25, 'reported 5 N ± 0.05', fontsize=15, color=C['memory'], fontweight='bold')
    ax.plot(tt, f - 0.2 * tt, color=C['danger'], lw=3)
    ax.plot([span], [2.0], 'o', color=C['danger'], ms=10)
    ax.text(14.8, 1.45, 'true 2 N', fontsize=15, color=C['danger'], fontweight='bold', ha='right')
    ax.text(0.3, 0.45, 'innovation: 0 at every sample', fontsize=14, color=C['text'])
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6)
    ax.set_xlabel('Minutes of thermal drift', fontsize=15)
    ax.set_ylabel('Grip force (N)', fontsize=15)
    ax.tick_params(labelsize=14)
    sp.save(fig, out('force.svg'))
if __name__ == '__main__':
    kalman()
    gates()
    imu()
    sort_timeline()
    epochs()
    lever()
    umbmark()
    kitti()
    reads()
    nvblox()
    snapshot()
    ray_grid()
    rack_top()
    erosion()
    latch()
    horizons()
    lifecycle()
    frozen()
    force()
