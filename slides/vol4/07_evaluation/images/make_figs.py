#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 7 (Closed-Loop Evaluation), Volume IV.

Every value comes from MLSysIM through calcs.py: the scenario objects for the
plotted curves and val()/num() for every printed number (the chapter, Closed-
Loop Evaluation, reads the same scenarios; Mandlekar et al. 2021, RAND RR-1478
Table 1, and Kusano et al. 2025 are Literature entries). Diagrams are SVG
drawn at the size of their slot (full width 900 x 500, labels 20 px or more);
plots use the shared slideplot helper. Red marks only limits and hazards.

Run from this folder: python3 make_figs.py
"""
_VALS = {'nhtsa_before': '1.3', 'nhtsa_after': '0.8', 'nhtsa_drop_pct': '38.5', 'nhtsa_claim_pct': '40', 'qcs_before': '0.76', 'qcs_after': '1.21', 'qcs_rise_pct': '59', 'qcs_share_pct': '13', 'standoff_m': '2', 'v_guard': '0.03', 'f_trip': '15', 'f_lim': '100', 'run_s': '25', 'reset_s': '15', 'cycle_s': '40', 'n_clean': '20', 'n_clean_words': 'twenty', 'cells_words': 'six', 'supported_words': 'one', 'absences_words': 'five', 'plates_words': 'three', 'speeds_words': 'two', 'plate_offset_mm': '3', 'chunk_hz': '20', 'lease_ms': '60', 'jam_s': '60', 'jam_cycle_s': '85', 'jam_pct': '70.6', 'filter_runs': '10', 'filter_kept': '9', 'filter_pct': '90', 'filter_dropped_pct': '100', 'sq_best': '80.7', 'sq_low': '2.7', 'tr_best': '64.0', 'tr_low': '0.7', 'heading_deg': '1', 'aisle_m': '30', 'v_base': '1.5', 'clear_cm': '15', 'clear_m': '0.15', 'mse_m': '3.05', 'mse_e': '-4', 't_hit': '5.7', 'd_hit': '8.6', 'd_hit_r': '9', 'k_contact_m': '5', 'k_contact_e': '5', 'residual_mm': '2', 'residual_n': '1,000', 'trip_pen_mm': '0.03', 'lim_pen_mm': '0.2', 'conf_pct': '95', 'alpha': '0.05', 'alpha_pct': '5', 'alpha_inv': '20', 'alpha_inv_words': 'twenty', 'lb': '0.861', 'lb_pct': '86.1', 'fail_up_pct': '13.9', 'one_in_words': 'seven', 'weak_one_in_words': 'eight', 'weak_clean_pct': '6.9', 'ln_inv_alpha': '2.996', 'rule3': '3', 'rule3_bound': '0.15', 't90_pct': '90', 't99_pct': '99', 't999_pct': '99.9', 'n90': '29', 'n99': '299', 'n999': '2,995', 'opening_min': '13.3', 'h99': '3.3', 'h999': '33.3', 'op_h999': '66.6', 'operators_words': 'two', 'shifts_words': 'eight', 'est_target': '0.995', 'est_target_pct': '99.5', 'est_rate': '0.005', 'est_r3': '600', 'est_n': '598', 'est_h': '6.6', 'lb_one_fail_pct': '78.4', 'obs_one_fail_pct': '95', 'sprt_p0': '0.10', 'sprt_p1': '0.01', 'sprt_p0_pct': '10', 'sprt_p1_pct': '1', 'sprt_alpha': '0.05', 'sprt_beta': '0.10', 'sprt_up': '0.0953', 'sprt_down': '2.303', 'sprt_accept': '2.89', 'sprt_reject': '2.25', 'sprt_n_accept': '31', 'sprt_trace_n': '25', 'sprt_trace': '0.08', 'sprt_fixed_n': '29', 'sprt_fixed_reject_pct': '25.3', 'looks_words': 'five', 'any_fp_pct': '22.6', 'arm_n': '20', 'arm_a': '20', 'arm_b': '16', 'arm_fail': '4', 'arm_total': '40', 'one_arm_p': '0.053', 'fisher_p': '0.106', 'p_high_pct': '90', 'p_low_pct': '80', 'power_pct': '80', 'runs_per_arm': '195', 'tri_trials': '1,800', 'simpler_episodes': '1,500', 'site_e': '-4', 'e6_e': '-6', 'e9_e': '-9', 'site_h': '29,957', 'site_years': '3.42', 'e6_mh': '3.0', 'e6_years': '342', 'e9_bh': '3.0', 'rigs': '100', 'e9_fleet_years': '3,417', 'rig_year_h': '8,766', 'rig_rate_m': '3.4', 'rig_rate_e': '-4', 'rig_short_e6': '342', 'waymo_m': '56.7', 'rand_fatal_m': '275', 'rand_injury_m': '3.9', 'rand_crash_m': '1.6', 'rand_fatal_rate': '1.09', 'rand_fatal_short': '4.9', 'rand_precision_bm': '8.8', 'rand_precision_x': '155', 'waymo_injury_pct': '79', 'waymo_airbag_pct': '81', 'f_mean': '12', 'f_p99': '115', 'tail_trials_words': 'one hundred', 'fallacy_n': '1,000', 'fallacy_n_exp': '1000', 'fallacy_lb_pct': '99.70', 'fallacy_lb': '0.9970'}
import math
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
SUPERSCRIPT = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
plt = sp.plt
C = sp.C
OUT = Path(__file__).parent
FONT = 'Helvetica, Arial, sans-serif'
PROPOSAL = '#2563EB'
NERVOUS = '#059669'
NERVOUS_L = '#DCFCE7'
PLANT = '#334155'
HAZARD = '#DC2626'
HAZARD_L = '#FEE2E2'
GRAY = '#94A3B8'
GRAY_L = '#F1F5F9'
TEXT = '#1A1A2E'

def svg(path, w, h, body):
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n<rect width="{w}" height="{h}" fill="white"/>\n' + '\n'.join(body) + '\n</svg>\n')

def text(x, y, s, size=20, fill=TEXT, anchor='middle', weight='normal', opacity=1.0):
    s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" opacity="{opacity}">{s}</text>'

def rect(x, y, w, h, fill, stroke, sw=2, rx=6, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>'

def line(x1, y1, x2, y2, stroke, sw=2.5, dash=None, marker=False):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = ' marker-end="url(#arr)"' if marker else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>'

def arrow_defs(color):
    return f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker></defs>'

def coverage(step):
    w, h = (600, 470)
    b = []
    cols = ['Plate −', 'Nominal', 'Plate +']
    rows = ['0.03 m/s', '0.10 m/s']
    x0, y0, cw, ch = (140, 52, 152, 128)
    for j, c in enumerate(cols):
        b.append(text(x0 + j * cw + cw / 2, y0 - 14, c, 26, weight='bold'))
    for i, r in enumerate(rows):
        b.append(text(x0 - 14, y0 + i * ch + ch / 2 + 9, r, 26, anchor='end', weight='bold'))
        for j in range(3):
            supported = i == 0 and j == 1
            fill = NERVOUS_L if supported else GRAY_L
            stroke = NERVOUS if supported else GRAY
            b.append(rect(x0 + j * cw + 5, y0 + i * ch + 5, cw - 10, ch - 10, fill, stroke, 2.5))
            if supported:
                b.append(text(x0 + j * cw + cw / 2, y0 + i * ch + 50, 'demos', 26, NERVOUS, weight='bold'))
                b.append(text(x0 + j * cw + cw / 2, y0 + i * ch + 90, '20 runs', 26, TEXT, weight='bold', opacity=1.0 if step >= 2 else 0.0))
            else:
                b.append(text(x0 + j * cw + cw / 2, y0 + i * ch + 73, 'no data', 24, GRAY))
    a = 1.0 if step >= 2 else 0.0
    b.append(text(w / 2, 370, 'Base stopped, empty rack, Unit 01', 26, TEXT, opacity=a))
    b.append(text(w / 2, 420, 'Five cells and the loaded', 26, HAZARD, weight='bold', opacity=a))
    b.append(text(w / 2, 455, 'machine: no evidence', 26, HAZARD, weight='bold', opacity=a))
    svg(OUT / f'coverage_step{step}.svg', w, h, b)

def mandlekar():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    tasks = ['Square', 'Transport']
    xs = np.arange(len(tasks))
    best = [80.7, 64.0]
    low = [2.7, 0.7]
    ax.bar(xs - 0.2, best, 0.38, color=C['data'])
    ax.bar(xs + 0.2, low, 0.38, color=C['neutral'])
    for x, v in zip(xs, best):
        ax.text(x - 0.2, v + 2, f'{v:g}%', ha='center', fontsize=13, fontweight='bold', color=C['data'])
    for x, v in zip(xs, low):
        ax.text(x + 0.2, v + 2, f'{v:g}%', ha='center', fontsize=13, fontweight='bold', color=C['neutral'])
    ax.set_xticks(xs)
    ax.set_xticklabels(tasks, fontsize=13)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel('Success rate (%)', fontsize=12)
    ax.text(-0.45, 114, 'Top checkpoint', color=C['data'], fontsize=12, fontweight='bold')
    ax.text(-0.45, 103, 'Lowest validation loss', color=C['neutral'], fontsize=12, fontweight='bold')
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, OUT / 'mandlekar_bars.svg')

def aisle(step):
    rad = 0.01745329252
    d_hit = 8.594803275
    aisle_m = 30.0
    clear_cm = 15.0
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    d = np.linspace(0, aisle_m, 300)
    ax.axhline(clear_cm, color=C['danger'], lw=2, ls='--')
    ax.text(29.6, clear_cm + 0.6, 'rack: 15 cm side clearance', ha='right', color=C['danger'], fontsize=12, fontweight='bold')
    ax.plot(d, np.zeros_like(d), color=C['memory'], lw=3)
    ax.text(29.6, 1.6, 'replay: reset each step, MSE 3.05×10⁻⁴ rad²', ha='right', color=C['memory'], fontsize=12, fontweight='bold')
    a = 1.0 if step >= 2 else 0.0
    dd = d[d <= d_hit]
    ax.plot(dd, dd * math.tan(rad) * 100, color=C['compute'], lw=3, alpha=a)
    ax.plot([d_hit], [clear_cm], 'o', ms=10, color=C['danger'], alpha=a)
    ax.text(d_hit + 0.6, 11.2, 'closed loop: 5.7 s, 8.6 m', color=C['compute'], fontsize=12, fontweight='bold', alpha=a)
    ax.set_xlim(0, aisle_m)
    ax.set_ylim(-1, 19)
    ax.set_xlabel('Distance along the 30 m aisle (m)')
    ax.set_ylabel('Lateral offset (cm)')
    ax.grid(color=C['grid'], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, OUT / f'aisle_step{step}.svg')

def contact():
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    k = 500000.0
    dm = np.logspace(-3, math.log10(3), 100)
    ax.plot(dm, k * dm / 1000, color=C['compute'], lw=3)
    pts = [(15.0, '15 N tripwire at 0.03 mm', C['danger']), (100.0, '100 N latch limit at 0.2 mm', C['danger']), (1000.0, '2 mm residual: 1,000 N', C['compute'])]
    for f, lab, col in pts:
        x = f / k * 1000
        ax.plot([x], [f], 'o', ms=9, color=col)
        ax.annotate(lab, (x, f), xytext=(10, -16) if f < 1000 else (-10, 6), textcoords='offset points', ha='left' if f < 1000 else 'right', fontsize=12, fontweight='bold', color=col)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.001, 4)
    ax.set_ylim(0.4, 4000)
    ax.set_xlabel('Penetration into the latch (mm)')
    ax.set_ylabel('Reaction force (N)')
    ax.text(0.0012, 1500, 'stiffness 5×10⁵ N/m', fontsize=12, color=C['neutral'])
    ax.grid(color=C['grid'], lw=0.5)
    sp.save(fig, OUT / 'contact_force.svg')
REGIMES = [(0, 1, 'Recorded replay', 'Latency, memory, numerics', 'Drift: errors never feed back'), (1, 1, 'Closed-loop simulation', 'Recovery, reachability, scale', 'Real contact, slip, sensors'), (0, 0, 'Shadow operation', 'Real buses, heat, sensor noise', 'Recovery: baseline drives'), (1, 0, 'Constrained physical test', 'Contact, friction, compliance', 'Beyond the safety boundary')]

def regimes(step):
    w, h = (900, 420)
    b = [arrow_defs(PLANT)]
    x0, y0, cw, ch = (130, 8, 375, 172)
    b.append(line(x0 - 30, y0 + 2 * ch + 10, x0 - 30, y0 + 5, PLANT, 2.5, marker=True))
    b.append(line(x0 - 30, y0 + 2 * ch + 14, x0 + 2 * cw + 10, y0 + 2 * ch + 14, PLANT, 2.5, marker=True))
    b.append(text(x0 + cw, y0 + 2 * ch + 46, 'Causal feedback: open loop  →  closed loop', 21, PLANT, weight='bold'))
    b.append(f'<g transform="translate({x0 - 52},{y0 + ch}) rotate(-90)">' + text(0, 0, 'Physical realism: logged → live', 21, PLANT, weight='bold') + '</g>')
    for k, (c, r, name, shows, blind) in enumerate(REGIMES):
        shown = k < step
        x, y = (x0 + c * cw + 8, y0 + r * ch + 8)
        live = r == 0
        b.append(rect(x, y, cw - 16, ch - 16, '#F8FAFC' if live else 'white', PLANT if shown else '#E2E8F0', 2.5, rx=8, dash='8,5' if shown and k == 3 else None))
        if shown:
            if k == 3:
                b.append(rect(x + 3, y + 3, cw - 22, ch - 22, 'none', HAZARD, 2.5, rx=8, dash='8,6'))
            b.append(text(x + (cw - 16) / 2, y + 32, name, 24, TEXT, weight='bold'))
            b.append(text(x + 18, y + 64, 'Shows:', 21, NERVOUS, anchor='start', weight='bold'))
            b.append(text(x + 18, y + 88, shows, 21, TEXT, anchor='start'))
            b.append(text(x + 18, y + 116, 'Blind to:', 21, HAZARD, anchor='start', weight='bold'))
            b.append(text(x + 18, y + 140, blind, 21, TEXT, anchor='start'))
    svg(OUT / f'regimes_step{step}.svg', w, h, b)

def inversion(step):
    lb, n, alpha = (0.8608916593, 20, 0.05)
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    p = np.linspace(0.7, 1.0, 400)
    ax.plot(p, p ** n, color=C['memory'], lw=3)
    ax.axhline(alpha, color=C['neutral'], lw=1.5, ls='--')
    ax.text(0.702, alpha * 1.25, '$\\alpha_{\\rm sig}$ = ' + '0.05', color=C['neutral'], fontsize=12, fontweight='bold')
    a = 1.0 if step >= 2 else 0.0
    ax.axvspan(0.7, lb, color=C['neutral_l'], alpha=0.6 * a, lw=0)
    ax.plot([lb], [alpha], 'o', ms=10, color=C['danger'], alpha=a)
    ax.text(0.705, 0.25, 'ruled out: p < 0.861', ha='left', fontsize=12, fontweight='bold', color=C['danger'], alpha=a)
    q = 0.875
    ax.plot([q], [q ** n], 'o', ms=10, color=C['compute'], alpha=a)
    ax.annotate(f'fails 1 run in {8}: clean record in 6.9% of campaigns', (q, q ** n), xytext=(0.882, 0.0016), textcoords='data', fontsize=12, arrowprops=dict(arrowstyle='-', color=C['compute'], lw=1.2, alpha=a), fontweight='bold', color=C['compute'], alpha=a)
    ax.set_yscale('log')
    ax.set_xlim(0.7, 1.0)
    ax.set_ylim(0.0005, 1.5)
    ax.set_xlabel('True per-run success probability p')
    ax.set_ylabel('P(20 clean runs)')
    ax.grid(color=C['grid'], lw=0.5)
    sp.save(fig, OUT / f'inversion_step{step}.svg')

def nines():
    rows = [('86.1%', 20, 'neutral', '20 runs · 13.3 min')]
    for target, key in zip([0.9, 0.99, 0.999], ('90', '99', '999')):
        n = {0.9: 29, 0.99: 299, 0.999: 2995}[target]
        hrs = {0.9: 0.3222222222, 0.99: 3.322222222, 0.999: 33.27777778}[target]
        tl = f'{hrs * 60:.1f} min' if hrs < 1 else f'{hrs:.1f} h'
        rows.append((f"{ {0.9: '90', 0.99: '99', 0.999: '99.9'}[target]}%", n, 'memory', f"{ {0.9: '29', 0.99: '299', 0.999: '2,995'}[target]} runs · {tl}"))
    sp.hbars_steps(str(OUT / 'nines'), rows, 'runs', xlabel='Clean runs needed (log scale)', size=(5.8, 2.3))

def sprt():
    up, down, a, bnd = (0.0953101798, -2.302585093, 2.890371758, -2.251291799)
    n_acc, traced = (31, 0.08016940211)
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    n = np.arange(0, n_acc + 1)
    ax.step(n, n * up, where='post', color=C['data'], lw=3)
    tr = [0.0, 0.0953101798, 0.1906203596, 0.2859305394, 0.3812407192, 0.476550899, 0.5718610788, 0.6671712586, 0.7624814384, 0.8577916182, 0.953101798, 1.048411978, 1.143722158, 1.239032337, 1.334342517, 1.429652697, 1.524962877, 1.620273057, 1.715583236, 1.810893416, 1.906203596, 2.001513776, 2.096823956, 2.192134135, 2.287444315, 2.382754495] + [traced]
    ax.plot([len(tr) - 2, len(tr) - 1], tr[-2:], color=C['compute'], lw=2.5)
    ax.plot([len(tr) - 1], [traced], 'o', ms=8, color=C['compute'])
    ax.axhline(a, color=C['data'], lw=1.8, ls='--')
    ax.axhline(bnd, color=C['danger'], lw=1.8, ls='--')
    ax.text(0.5, a + 0.25, 'accept at 2.89', color=C['data'], fontsize=12, fontweight='bold')
    ax.text(0.5, bnd + 0.25, 'reject at -2.25', color=C['danger'], fontsize=12, fontweight='bold')
    ax.annotate('31 clean runs', (n_acc, n_acc * up), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=12, fontweight='bold', color=C['data'])
    ax.annotate('25 clean, 1 failure: 0.08', (len(tr) - 1, traced), xytext=(8, -18), textcoords='offset points', fontsize=12, fontweight='bold', color=C['compute'])
    ax.set_xlim(0, 34)
    ax.set_ylim(-3, 3.8)
    ax.set_xlabel('Run')
    ax.set_ylabel('Log-likelihood ratio')
    ax.grid(color=C['grid'], lw=0.5)
    sp.save(fig, OUT / 'sprt.svg')

def exposure():
    door = 33.27777778
    hours = [29957.32274, 2995732.274, 2995732274.0]
    rows = [('Door, 99.9%', door, 'neutral', '33.3 h'), ('10⁻⁴ per h', hours[0], 'memory', '29,957 h'), ('10⁻⁶ per h', hours[1], 'memory', '3.0 million h'), ('10⁻⁹ per h', hours[2], 'danger', '3.0 billion h')]
    sp.hbars_steps(str(OUT / 'exposure'), rows, 'h', xlabel='Failure-free hours needed (log scale)', size=(5.8, 2.3), vlines=[(8766.0, 'one rig-year', 'data')])

def rand():
    rows = [('Fatalities', float(275.0), 'danger', '275M'), ('Injuries', float(3.9), 'memory', '3.9M'), ('Crashes', float(1.6), 'memory', '1.6M')]
    sp.hbars(str(OUT / 'rand_waymo.svg'), rows, 'M', xlabel='Zero-event miles to bound the human rate (millions)', size=(4.8, 2.1), vlines=[(float(56.7), 'Waymo 56.7M', 'data')])

def mean_peak():
    fig, ax = plt.subplots(figsize=(5.8, 1.8))
    rows = [('Mean', 12.0, 'compute'), ('99th percentile', 115.0, 'danger')]
    for i, (lab, v, ck) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(-1.5, y, lab, ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
        ax.text(v + 2 if v > 50 else v + 5.5, y, f'{v:g} N', va='center', fontsize=13, fontweight='bold', color=C[ck])
    for x, lab, ck in [(15.0, '15 N tripwire', 'neutral'), (100.0, '100 N limit', 'danger')]:
        ax.axvline(x, color=C[ck], lw=1.5, ls='--')
        ax.text(x + 1, 1.45, lab, color=C[ck], fontsize=11, fontweight='bold', va='bottom')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 140)
    ax.set_ylim(-0.5, 1.8)
    ax.set_xlabel(f'Latch contact force (N), {100} trials')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, OUT / 'mean_peak.svg')
FIELDS = [('target_odd_boundary', 'nominal plate, 0.03 m/s, base stopped'), ('component_hashes', 'manifest digest, weights, Unit 01'), ('sampling_protocol', 'draws, seed, reset, independence'), ('complete_outcome_log', 'every run: task, autonomy, protection'), ('statistical_evidence', '20 clean runs: failure ≤ 13.9% (95%)'), ('unsupported_gap_catalog', f'{5} cells, payload, other units'), ('runtime_monitor_specs', 'signal, threshold, tested response')]
MONITORS = ['Input support', 'Latency and freshness', 'Constraint margin', 'Intervention rate', 'Environmental assumption']

def record(step):
    w, h = (900, 470)
    b = [arrow_defs(NERVOUS)]
    x, y, cw = (6, 6, 548)
    b.append(rect(x, y, cw, 458, 'white', PLANT, 2.5, rx=8))
    b.append(text(x + cw / 2, y + 34, 'Evaluation record: latch-bc-01', 25, TEXT, weight='bold'))
    for i, (f, v) in enumerate(FIELDS):
        yy = y + 50 + i * 57
        hot = step >= 2 and i >= 5
        b.append(rect(x + 10, yy, cw - 20, 52, HAZARD_L if hot else GRAY_L, HAZARD if hot else '#CBD5E1', 1.5, rx=5))
        b.append(text(x + 20, yy + 23, f.replace('_', ' '), 22, TEXT, anchor='start', weight='bold'))
        b.append(text(x + 20, yy + 46, v, 21, '#334155', anchor='start'))
    a = 1.0 if step >= 2 else 0.0
    mx, mw = (612, 282)
    b.append(f'<g opacity="{a}">')
    b.append(text(mx + mw / 2, 30, 'Runtime monitors', 24, NERVOUS, weight='bold'))
    for i, m in enumerate(MONITORS):
        yy = 46 + i * 64
        b.append(rect(mx, yy, mw, 52, NERVOUS_L, NERVOUS, 2, rx=6))
        b.append(text(mx + mw / 2, yy + 34, m, 22, TEXT, weight='bold'))
    b.append(line(x + cw + 4, 410, mx - 6, 220, NERVOUS, 2.5, marker=True))
    b.append(rect(mx, 382, mw, 80, HAZARD_L, HAZARD, 2, rx=6))
    b.append(text(mx + mw / 2, 414, 'Cannot be sensed:', 22, HAZARD, weight='bold'))
    b.append(text(mx + mw / 2, 444, 'hard limit or barrier', 22, TEXT))
    b.append('</g>')
    svg(OUT / f'record_step{step}.svg', w, h, b)

def main():
    for s in (1, 2):
        coverage(s)
        aisle(s)
        inversion(s)
        record(s)
    for s in (1, 2, 3, 4):
        regimes(s)
    mandlekar()
    contact()
    nines()
    sprt()
    exposure()
    rand()
    mean_peak()
    print('  figures written to', OUT)
if __name__ == '__main__':
    main()
