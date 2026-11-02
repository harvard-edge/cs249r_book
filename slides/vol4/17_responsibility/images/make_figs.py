#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 17 (Responsible Physical AI), Volume IV.

Every value comes from MLSysIM through calcs.py (imported as K); printed
numbers use the formatted strings calcs.py records (deckvals.val), and plots
read the scenario objects (Scenarios.*, Literature.* for the two published
per-group detection tables, each named beside its data). Each
figure is drawn at the size of its slot (full width about 5.8 in wide; a
column about 5.4 by 3.6 in with 15 pt labels) so its type stays legible on
the slide. A figure revealed in steps is written as name_step1.svg ...
name_stepN.svg with every element drawn in every step (hidden ones
transparent), so the canvas and the axes never move between steps.

Colors follow the Volume IV palette (volumes/vol4/palette.md): green for the
deterministic permission side and its records, slate for people, parties,
and the plant, red only for hazards, missing records, and harm, amber for
the approval record (the cover's approval seal), navy for learned proposers.

Run from this folder: python3 make_figs.py
"""
from types import SimpleNamespace
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
_VALS = {'drag_m': '6.1', 'cpuc_days': '15', 'cpuc_daily': '7,500', 'cpuc_fine': '112,500', 'nhtsa_m': '1.5', 'doj_fine': '500,000', 'pen_total': '2,112,500', 'pen_total_m': '2.1', 'v_aisle': '1.3', 'd_clear': '1.10', 'mass': '368', 'ke_aisle': '311', 'a_brake': '2', 'links': '6', 'p_good': '0.99', 'c_good': '0.94', 'p_mid': '0.95', 'c_mid': '0.74', 'p_poor': '0.90', 'c_poor': '0.53', 'c_poor_pct': '53', 't_window': '0.85', 't_ool': '4.465', 't_ool1': '4.5', 'travel_ool': '5.8', 'ool_ratio': '5.3', 'ool_ratio_words': 'five', 'ool_ratio_int': '5', 't_in': '0.605', 'req_rate': '0.5', 'service_min': '3', 'service_s': '180', 'n10': '10', 'rho10': '0.25', 'wq10': '60', 'wq10_min': '1', 'n20': '20', 'rho20': '0.50', 'wq20': '180', 'wq20_min': '3', 'n32': '32', 'rho32': '0.80', 'wq32': '720', 'wq32_min': '12', 'n20_words': 'twenty', 'n10_req_h': '5', 'rho20_pct': '50', 'p99_20_s': '1,408', 'p99_20_min': '23.5', 'p99_20_min0': '23', 'rho_cap': '0.5', 'n_max': '20', 'chk_n': '24', 'chk_rate': '0.8', 'chk_s': '75', 'chk_rho': '0.40', 'chk_wq': '50', 'stage_per_h': '1,000', 'stage_conf_pct': '95', 'stage_alpha': '0.05', 'stage_inv_alpha': '20', 'stage_rate_exp_mag': '3', 't_req': '2,996', 'stage_n': '10', 'stage_n_words': 'ten', 'stage_h_day': '16', 'fleet_h_day': '160', 'days_req': '18.7', 'days_req_ceil': '19', 'li_child': '43.89', 'li_adult': '23.75', 'li_gap': '20.14', 'w_ds': '55.9', 'w_ls': '67.1', 'w_hold_ls': '387', 'w_hold_ds': '100', 'w_need_ls': '12,000', 'w_need_ds': '4,000', 'w_gap': '0.05', 'n_detectors_words': 'eight', 'w_models_words': 'eight', 'cam_hz': '60', 't_frame': '16.7', 'p_miss_exp': '-3', 'p_miss_inv': '1,000', 'p_ref': '0.95', 'p_ref_pct': '95', 'n_ref': '3', 'd_brake': '0.42', 'tau': '133.6', 'g0_p': '0.85', 'g0_pct': '85', 'g0_n': '4', 'g0_dn': '1', 'g0_dt': '16.7', 'g0_dr': '22', 'g0_dr_m': '0.022', 'g0_vi': '0.29', 'g0_e': '16', 'g1_p': '0.75', 'g1_pct': '75', 'g1_n': '5', 'g1_dn': '2', 'g1_dt': '33.3', 'g1_dr': '43', 'g1_dr_m': '0.043', 'g1_vi': '0.42', 'g1_e': '32', 'g2_p': '0.60', 'g2_pct': '60', 'g2_n': '8', 'g2_dn': '5', 'g2_dt': '83.3', 'g2_dr': '108', 'g2_dr_m': '0.108', 'g2_vi': '0.66', 'g2_e': '80', 'gap_pts': '20', 'g1_dn_words': 'two', 'li_dr_lo': '65', 'li_dr_hi': '130', 'li_vi_lo': '0.51', 'li_vi_hi': '0.72', 'li_e_lo': '48', 'li_e_hi': '96', 'li_avg_dn': '4', 'li_avg_dr': '87', 'li_avg_vi': '0.59', 'li_avg_e': '64', 's_total': '10,000', 's_share_pct': '2', 's_small': '200', 's_large': '9,800', 's_recall': '0.95', 's_pa': '0.95', 's_pb': '0.92', 's_power_pct': '80', 's_conf_pct': '95', 'ci_l_lo': '0.945', 'ci_l_hi': '0.954', 'ci_s_lo': '0.910', 'ci_s_hi': '0.976', 'w_ratio': '8', 'n_needed': '1,059', 'share_needed_pct': '11', 'enc_pct': '80', 'users_pct': '10', 'enc_share': '0.80', 'users_share': '0.10', 'rest_enc': '0.20', 'rest_users': '0.90', 'exp_ratio': '8', 'rest_ratio_pct': '22', 'ret_pre': '5', 'ret_post': '25', 'ret_window': '30', 'ret_events': '20', 'ret_day_h': '16', 'ret_kept_min': '10', 'ret_day_min': '960', 'ret_kept_pct': '1', 'ret_reduction': '96', 'lab_req_h': '8', 'lab_watch_h': '8', 'lab_day_h': '16', 'lab_sweeps_words': 'four', 'lab_sweep_min': '20', 'lab_sweep_h': '1.3', 'lab_total_h': '17.3', 'lab_shifts': '2.2'}
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import slideplot as sp
plt = sp.plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
OUT = HERE
P = {'navy': '#1F407A', 'navy_l': '#EFF6FF', 'green': '#059669', 'green_l': '#DCFCE7', 'green_d': '#047857', 'red': '#DC2626', 'red_l': '#FEE2E2', 'red_d': '#991B1B', 'slate': '#334155', 'slate_l': '#F1F5F9', 'amber': '#D97706', 'amber_l': '#FEF3C7', 'amber_d': '#92400E', 'gray': '#64748B', 'gray_l': '#E2E8F0', 'petrol': '#0D9488', 'text': '#1A1A2E'}

def save(fig, name):
    fig.savefig(OUT / f'{name}.svg', format='svg', facecolor='white')
    plt.close(fig)

def a(shown):
    return 1.0 if shown else 0.0

def box(ax, x, y, w, h, text, fc, ec, shown=True, fs=13, tc=None, bold=True, rad=0.08, lw=1.8, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={rad}', fc=fc, ec=ec, lw=lw, ls=ls, alpha=a(shown)))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc or P['text'], fontweight='bold' if bold else 'normal', alpha=a(shown))

def arrow(ax, x0, y0, x1, y1, color, shown=True, lw=2.0, ls='-', ms=14):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, ls=ls, alpha=a(shown)))

def canvas(w, h, xlim, ylim):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis('off')
    return (fig, ax)

def cruise_timeline():
    for step in (1, 2):
        fig, ax = canvas(6.0, 2.5, (0, 30), (0, 10))
        s2 = step >= 2
        ax.text(0.2, 8.6, 'The motion', fontsize=12.5, fontweight='bold', color=P['red_d'], va='center')
        box(ax, 7.0, 7.3, 22.6, 2.6, '2 Oct 2023: ran over a pedestrian thrown into its path,\nthen pulled over 6.1 m with her beneath it', P['red_l'], P['red'], fs=11.5, bold=False)
        ax.text(0.2, 5.0, 'The record', fontsize=12.5, fontweight='bold', color=P['slate'], va='center')
        ax.plot([7.0, 29.6], [5.0, 5.0], color=P['gray'], lw=2)
        events = [(8.0, '3 Oct', 'regulator told\nit stopped'), (14.3, '3 to 18 Oct', '15 days without\na full account'), (20.4, '19 Oct', 'full video\nto the CPUC'), (26.6, '24 Oct', 'DMV suspends\ndriverless permits')]
        for x, d, t in events:
            ax.plot([x], [5.0], 'o', color=P['slate'], ms=8)
            ax.text(x, 5.75, d, ha='center', va='bottom', fontsize=11.5, fontweight='bold', color=P['slate'])
            ax.text(x, 4.3, t, ha='center', va='top', fontsize=11, color=P['text'], linespacing=1.05)
        ax.text(0.2, 0.9, '2024', fontsize=12.5, fontweight='bold', color=P['amber_d'], va='center', alpha=a(s2))
        pens = [('CPUC  $112,500', 7.0), (f'NHTSA  ${1500000.0:,.0f}', 14.6), ('DOJ fine  $500,000', 22.2)]
        for t, x in pens:
            box(ax, x, 0.05, 7.3, 1.7, t, P['amber_l'], P['amber'], fs=11.5, shown=s2)
        save(fig, f'cruise_timeline_step{step}')
DECISIONS = [('Policy weights', 'learning team', 'policy manifest'), ('Enforcer thresholds', 'safety engineering', 'enforcement record'), ('Authority state', 'assistant on shift', 'authority record'), ('Operating envelope', 'systems adjudicator', 'release manifest'), ('Crossing rule', 'facility management', 'no record'), ('Staffing ratio', 'operations', 'no record')]

def decision_chain():
    for step in (1, 2, 3):
        fig, ax = canvas(6.4, 3.0, (0, 70), (-4.5, 29.5))
        box(ax, 0.2, 9.0, 9.6, 10.0, 'Admitted\ncommand\n1.3 m/s,\none tick', P['green_l'], P['green'], fs=11.5)
        ax.text(21.8, 27.4, 'Decision', ha='center', fontsize=11.5, color=P['gray'], fontweight='bold')
        ax.text(41.0, 27.4, 'Made by', ha='center', fontsize=11.5, color=P['gray'], fontweight='bold')
        ax.text(58.5, 27.4, 'Held in', ha='center', fontsize=11.5, color=P['gray'], fontweight='bold')
        for i, (dec, who, rec) in enumerate(DECISIONS):
            y = 22.6 - i * 4.4
            missing = rec == 'no record'
            arrow(ax, 9.8, 14.0, 12.4, y + 1.7, P['slate'], lw=1.3, ms=9)
            box(ax, 12.4, y, 18.8, 3.4, dec, P['slate_l'], P['slate'], fs=10.5)
            ax.text(41.0, y + 1.7, who, ha='center', va='center', fontsize=11, color=P['text'])
            red = missing and step >= 2
            col = P['red'] if red else P['green_d'] if not missing else P['text']
            ax.text(58.5, y + 1.7, rec, ha='center', va='center', fontsize=11, color=col, fontweight='bold' if red else 'normal')
        s3 = step >= 3
        ax.add_patch(Rectangle((31.8, -0.5), 37.9, 27.0, fc='none', ec=P['amber'], lw=2.2, ls='--', alpha=a(s3)))
        ax.text(50.7, -3.0, 'Approval record: each decision has an owner', ha='center', va='center', fontsize=11.5, color=P['amber_d'], fontweight='bold', alpha=a(s3))
        save(fig, f'decision_chain_step{step}')

def chain_prob():
    p = np.linspace(0.85, 1.0, 200)
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        fig.subplots_adjust(left=0.19, right=0.96, bottom=0.2, top=0.95)
        ax.plot(p, p, color=P['gray'], lw=2.2)
        ax.text(0.862, 0.905, 'one link', fontsize=15, color=P['gray'])
        s2 = step >= 2
        ax.plot(p, [0.3771495156, 0.3791606771, 0.3811807679, 0.3832098195, 0.3852478638, 0.3872949327, 0.389351058, 0.3914162718, 0.3934906063, 0.3955740935, 0.3976667658, 0.3997686556, 0.4018797953, 0.4040002174, 0.4061299546, 0.4082690395, 0.410417505, 0.412575384, 0.4147427093, 0.4169195141, 0.4191058316, 0.4213016948, 0.4235071372, 0.4257221922, 0.4279468932, 0.4301812738, 0.4324253676, 0.4346792085, 0.4369428303, 0.4392162668, 0.4414995522, 0.4437927205, 0.4460958058, 0.4484088426, 0.4507318651, 0.4530649078, 0.4554080052, 0.4577611921, 0.4601245031, 0.462497973, 0.4648816367, 0.4672755293, 0.4696796858, 0.4720941414, 0.4745189314, 0.4769540911, 0.4793996559, 0.4818556615, 0.4843221433, 0.4867991373, 0.4892866791, 0.4917848046, 0.49429355, 0.4968129511, 0.4993430443, 0.5018838658, 0.5044354519, 0.5069978392, 0.5095710641, 0.5121551633, 0.5147501735, 0.5173561316, 0.5199730744, 0.522601039, 0.5252400626, 0.5278901822, 0.5305514351, 0.5332238589, 0.5359074909, 0.5386023687, 0.54130853, 0.5440260125, 0.5467548542, 0.5494950929, 0.5522467668, 0.5550099139, 0.5577845725, 0.5605707809, 0.5633685776, 0.566178001, 0.5689990899, 0.5718318829, 0.5746764189, 0.5775327367, 0.5804008753, 0.5832808739, 0.5861727717, 0.5890766079, 0.591992422, 0.5949202535, 0.5978601418, 0.6008121269, 0.6037762483, 0.606752546, 0.60974106, 0.6127418304, 0.6157548973, 0.618780301, 0.6218180819, 0.6248682805, 0.6279309373, 0.6310060931, 0.6340937885, 0.6371940645, 0.6403069621, 0.6434325222, 0.6465707862, 0.6497217952, 0.6528855907, 0.656062214, 0.6592517069, 0.6624541109, 0.6656694679, 0.6688978197, 0.6721392082, 0.6753936757, 0.6786612643, 0.6819420162, 0.6852359738, 0.6885431797, 0.6918636764, 0.6951975066, 0.6985447131, 0.7019053389, 0.7052794269, 0.7086670202, 0.7120681621, 0.7154828959, 0.7189112649, 0.7223533127, 0.7258090829, 0.7292786194, 0.7327619658, 0.7362591661, 0.7397702644, 0.7432953049, 0.7468343317, 0.7503873893, 0.7539545221, 0.7575357747, 0.7611311917, 0.764740818, 0.7683646984, 0.772002878, 0.7756554017, 0.7793223149, 0.7830036629, 0.786699491, 0.7904098448, 0.79413477, 0.7978743122, 0.8016285174, 0.8053974315, 0.8091811005, 0.8129795707, 0.8167928883, 0.8206210997, 0.8244642514, 0.82832239, 0.8321955623, 0.8360838151, 0.8399871952, 0.8439057498, 0.847839526, 0.8517885711, 0.8557529324, 0.8597326575, 0.8637277939, 0.8677383893, 0.8717644916, 0.8758061487, 0.8798634086, 0.8839363194, 0.8880249295, 0.8921292873, 0.8962494411, 0.9003854396, 0.9045373315, 0.9087051656, 0.9128889909, 0.9170888564, 0.9213048112, 0.9255369047, 0.9297851862, 0.9340497052, 0.9383305114, 0.9426276544, 0.9469411841, 0.9512711504, 0.9556176035, 0.9599805934, 0.9643601706, 0.9687563853, 0.9731692882, 0.9775989298, 0.982045361, 0.9865086325, 0.9909887955, 0.9954859009, 1.0], color=P['red'], lw=3, alpha=a(s2))
        ax.text(0.952, 0.48, 'six links', fontsize=15, color=P['red'], fontweight='bold', alpha=a(s2))
        for pk, ck in {0.99: 0.9414801494, 0.95: 0.7350918906, 0.9: 0.531441}.items():
            ax.plot([pk], [ck], 'o', color=P['red'], ms=9, alpha=a(s2))
            if pk > 0.98:
                ax.annotate('0.94', xy=(pk, ck), xytext=(0.995, 0.79), ha='right', fontsize=15, color=P['red_d'], fontweight='bold', alpha=a(s2), arrowprops=dict(arrowstyle='-', color=P['red'], lw=1.0, alpha=a(s2)))
            else:
                ax.text(pk - 0.004, ck + 0.035, f'{ck:.2f}', ha='right', fontsize=15, color=P['red_d'], fontweight='bold', alpha=a(s2))
        ax.set_xlim(0.85, 1.0)
        ax.set_ylim(0.3, 1.02)
        ax.set_xticks([0.85, 0.9, 0.95, 1.0])
        ax.tick_params(labelsize=14)
        ax.set_xlabel('Each link retrievable', fontsize=15)
        ax.set_ylabel('Whole chain found', fontsize=15)
        save(fig, f'chain_prob_step{step}')

def time_test():
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    fig.subplots_adjust(left=0.33, right=0.97, bottom=0.3, top=0.95)
    rows = [('Time the hazard leaves', 0.8461538462, P['green'], '0.85 s (1.10 m at 1.3 m/s)'), ('Out-of-loop takeover', 4.465, P['red'], '4.5 s: travels 5.8 m')]
    for i, (lab, v, col, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=col, height=0.55)
        ax.text(-0.08, y, lab, ha='right', va='center', fontsize=12.5, fontweight='bold', color=P['text'])
        ax.text(v + 0.08, y, txt, ha='left', va='center', fontsize=12.5, color=col, fontweight='bold')
    ax.set_xlim(0, 8.2)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xlabel('Seconds', fontsize=12)
    ax.tick_params(labelsize=11.5)
    save(fig, 'time_test')

def queue():
    n = np.arange(1, 35)
    wq = [0.07692307692, 0.1578947368, 0.2432432432, 0.3333333333, 0.4285714286, 0.5294117647, 0.6363636364, 0.75, 0.8709677419, 1.0, 1.137931034, 1.285714286, 1.444444444, 1.615384615, 1.8, 2.0, 2.217391304, 2.454545455, 2.714285714, 3.0, 3.315789474, 3.666666667, 4.058823529, 4.5, 5.0, 5.571428571, 6.230769231, 7.0, 7.909090909, 9.0, 10.33333333, 12.0, 14.14285714, 17.0]
    Q = {10: (0.25, 60.0, 772.530198), 20: (0.5, 180.0, 1408.328282), 32: (0.8, 720.0, 3943.823971)}
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        fig.subplots_adjust(left=0.17, right=0.96, bottom=0.2, top=0.95)
        ax.plot(n, wq, color=P['slate'], lw=3)
        pts = [10, 20] + ([32] if step >= 2 else [])
        for k in (10, 20, 32):
            r, w, _ = Q[k]
            sh = k in pts
            ax.plot([k], [w / 60], 'o', color=P['red'], ms=9, alpha=a(sh))
            tx, ty = {10: (3.0, 4.0), 20: (12.0, 7.4), 32: (17.5, 12.2)}[k]
            ax.annotate(f"{ {10: '10', 20: '20', 32: '32'}[k]} machines\n{ {10: '0.25', 20: '0.50', 32: '0.80'}[k]} busy, { {10: '1', 20: '3', 32: '12'}[k]} min", xy=(k, w / 60), xytext=(tx, ty), ha='left', va='bottom', fontsize=14, color=P['red_d'], alpha=a(sh), linespacing=1.0, arrowprops=dict(arrowstyle='-', color=P['red'], lw=1.0, alpha=a(sh)))
        ax.set_xlim(0, 34)
        ax.set_ylim(0, 15)
        ax.tick_params(labelsize=14)
        ax.set_xlabel('Machines per assistant', fontsize=15)
        ax.set_ylabel('Mean wait (min)', fontsize=15)
        save(fig, f'queue_step{step}')

def ladder():
    road = ['Testing with a\nsafety driver', 'Driverless\ntesting', 'Deployment to\nthe public']
    floor = ['Commissioning\nwith a spotter', 'Unattended in a\nzone people avoid', 'Open floor\namong workers']
    for step in (1, 2):
        fig, ax = canvas(6.0, 2.7, (0, 60), (0, 28))
        s2 = step >= 2
        ax.text(0.3, 21.5, 'California\nroads', fontsize=12, fontweight='bold', color=P['slate'], va='center')
        ax.text(0.3, 7.5, 'A warehouse', fontsize=12, fontweight='bold', color=P['slate'], va='center', alpha=a(s2))
        for i in range(3):
            x = 11.0 + i * 16.6
            box(ax, x, 17.5 + i * 1.6, 14.4, 6.2, road[i], P['green_l'], P['green'], fs=11)
            box(ax, x, 3.5 + i * 1.6, 14.4, 6.2, floor[i], P['green_l'], P['green'], fs=11, shown=s2)
            if i < 2:
                arrow(ax, x + 14.4, 20.6 + i * 1.6, x + 16.6, 22.2 + i * 1.6, P['slate'], lw=1.6)
                arrow(ax, x + 14.4, 6.6 + i * 1.6, x + 16.6, 8.2 + i * 1.6, P['slate'], lw=1.6, shown=s2)
        ax.text(11.5, 14.8, 'Each rung is its own DMV permit; the CPUC adds passenger rungs', fontsize=11, color=P['gray'], va='center')
        ax.text(11.5, 0.9, "Granted by the owner, the integrator, and workers' representatives", fontsize=11, color=P['gray'], va='center', alpha=a(s2))
        save(fig, f'ladder_step{step}')

def withdraw():
    """The DMV suspension of 24 October 2023 steps Cruise down the road ladder:
    the two driverless rungs are withdrawn and the safety-driver permit stays
    (Responsible Physical AI, Sec. 17.4.2; DMV statement, 24 Oct 2023)."""
    road = ['Testing with a\nsafety driver', 'Driverless\ntesting', 'Deployment to\nthe public']
    fig, ax = canvas(3.0, 2.8, (0, 30), (0, 28))
    for i in range(3):
        y = 1.0 + i * 7.4
        kept = i == 0
        box(ax, 1.0, y, 16.0, 5.4, road[i], P['green_l'] if kept else P['red_l'], P['green'] if kept else P['red'], fs=12, ls='-' if kept else '--')
        ax.text(21.5, y + 2.7, 'kept' if kept else 'suspended', ha='left', va='center', fontsize=11.5, fontweight='bold', color=P['green_d'] if kept else P['red_d'])
        if i < 2:
            arrow(ax, 9.0, y + 5.4, 9.0, y + 7.4, P['gray'], lw=1.6, ms=12)
    ax.plot([17.0, 19.6, 19.6, 18.6], [18.1, 18.1, 3.7, 3.7], color=P['red'], lw=2.4, solid_capstyle='butt')
    arrow(ax, 18.8, 3.7, 17.1, 3.7, P['red'], lw=2.4, ms=14)
    ax.text(1.0, 25.4, '24 Oct 2023: the DMV withdraws\nboth driverless rungs', ha='left', va='center', fontsize=11.5, color=P['red_d'], fontweight='bold', linespacing=1.15)
    save(fig, 'withdraw')

def stage_hours():
    days = np.linspace(0, 24, 300)
    hrs = days * 160.0
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.95)
    ax.plot(days, hrs, color=P['green'], lw=3)
    ax.axhline(2995.732274, color=P['red'], lw=1.8, ls='--')
    ax.text(0.5, 2995.732274 + 90, '2,996 failure-free hours', fontsize=15, color=P['red_d'])
    ax.plot([18.72332671], [2995.732274], 'o', color=P['red'], ms=9)
    ax.text(18.72332671 + 0.6, 2995.732274 - 380, 'day 19', fontsize=15, color=P['red_d'], fontweight='bold')
    ax.text(11, 600, '10 machines,\n16 h a day', fontsize=15, color=P['green_d'], ha='left')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 3800)
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Days at the supervised stage', fontsize=15)
    ax.set_ylabel('Fleet hours', fontsize=15)
    save(fig, 'stage_hours')

def fines():
    c = {'drag_m': 6.096, 'cpuc': 112500.0, 'nhtsa': 1500000.0, 'doj': 500000.0}
    rows = [('CPUC', c['cpuc'], '15 days x $7,500'), ('NHTSA', c['nhtsa'], 'reports omitted the drag'), ('DOJ', c['doj'], 'false report, criminal fine')]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.22, top=0.95)
    for i, (lab, v, why) in enumerate(rows):
        y = 2 - i
        ax.barh(y, v / 1000000.0, color=P['amber'], height=0.6)
        ax.text(-0.03, y, lab, ha='right', va='center', fontsize=15, fontweight='bold', transform=ax.get_yaxis_transform())
        ax.text(v / 1000000.0 + 0.04, y + 0.05, f'${v:,.0f}', ha='left', va='bottom', fontsize=14, fontweight='bold', color=P['amber_d'])
        ax.text(v / 1000000.0 + 0.04, y - 0.05, why, ha='left', va='top', fontsize=13, color=P['text'])
    ax.set_xlim(0, 3.0)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Penalty (million USD)', fontsize=15)
    save(fig, 'fines')

def detect_data():
    fig = plt.figure(figsize=(6.0, 2.8))
    ax1 = fig.add_axes([0.2, 0.2, 0.3, 0.68])
    ax2 = fig.add_axes([0.69, 0.2, 0.3, 0.68])
    li = {'YOLOX': (12.64, 42.47), 'RetinaNet': (14.36, 44.33), 'Faster RCNN': (5.24, 26.06), 'Cascade RCNN': (5.12, 26.57), 'ALFNet': (36.62, 53.47), 'CSP': (37.74, 50.42), 'MGAN': (33.58, 46.53), 'PRNet': (44.69, 61.25)}
    names = list(li)
    for i, k in enumerate(names):
        y = len(names) - 1 - i
        ad, ch = li[k]
        ax1.plot([ad, ch], [y, y], color=P['gray_l'], lw=3, zorder=1)
        ax1.plot([ad], [y], 'o', color=P['slate'], ms=6.5, zorder=2)
        ax1.plot([ch], [y], 'o', color=P['red'], ms=6.5, zorder=2)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names[::-1], fontsize=10.5)
    ax1.set_xlim(0, 70)
    ax1.set_xlabel('Miss rate (%)', fontsize=11)
    ax1.tick_params(axis='x', labelsize=10.5)
    ax1.text(3, 8.15, 'adults', color=P['slate'], fontsize=11, fontweight='bold')
    ax1.text(38, 8.15, 'children', color=P['red'], fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.6, 8.9)
    ax1.set_title('Li et al. 2025, Table 7', fontsize=11.5, loc='left', x=-0.55, pad=14)
    w = {'Faster R-CNN R-50-C4': (64.5, 53.6), 'Faster R-CNN R-50-FPN': (63.1, 53.6), 'Faster R-CNN R-101-FPN': (70.0, 57.3), 'Faster R-CNN X-101-FPN': (69.8, 62.1), 'Mask R-CNN R-50-C4': (66.9, 51.5), 'Mask R-CNN R-50-FPN': (67.6, 51.2), 'Mask R-CNN R-101-FPN': (65.6, 55.2), 'Mask R-CNN X-101-FPN': (69.2, 62.9)}
    wn = list(w)
    short = [s.replace('Faster R-CNN ', 'Faster ').replace('Mask R-CNN ', 'Mask ') for s in wn]
    for i, k in enumerate(wn):
        y = len(wn) - 1 - i
        ls_, ds_ = w[k]
        ax2.plot([ds_, ls_], [y, y], color=P['gray_l'], lw=3, zorder=1)
        ax2.plot([ls_], [y], 'o', color=P['slate'], ms=6.5, zorder=2)
        ax2.plot([ds_], [y], 'o', color=P['red'], ms=6.5, zorder=2)
    ax2.set_yticks(range(len(wn)))
    ax2.set_yticklabels(short[::-1], fontsize=10.5)
    ax2.set_xlim(45, 75)
    ax2.set_xlabel('AP75 (%)', fontsize=11)
    ax2.tick_params(axis='x', labelsize=10.5)
    ax2.text(46.2, 8.15, 'darker', color=P['red'], fontsize=11, fontweight='bold')
    ax2.text(62, 8.15, 'lighter skin', color=P['slate'], fontsize=11, fontweight='bold')
    ax2.set_ylim(-0.6, 8.9)
    ax2.set_title('Wilson et al. 2019, Table 4', fontsize=11.5, loc='left', x=-0.62, pad=14)
    save(fig, 'detect_data')

def miss_frames():
    n = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    fig.subplots_adjust(left=0.2, right=0.96, bottom=0.2, top=0.95)
    p_ref, p_low = (0.95, 0.75)
    n_ref, n_low = (3, 5)
    p_miss = 0.001
    for p, col in ((p_ref, P['slate']), (p_low, P['red'])):
        ax.semilogy(n, (1 - p) ** n, 'o-', color=col, lw=2.6, ms=7)
    ax.axhline(p_miss, color=P['green'], lw=1.8, ls='--')
    ax.text(3.75, p_miss * 0.45, '1 in 1,000', fontsize=14, ha='left', color=P['green_d'], va='top', fontweight='bold')
    ax.annotate('3 frames', xy=(n_ref, (1 - p_ref) ** n_ref), xytext=(1.5, 1.2e-05), fontsize=15, color=P['slate'], fontweight='bold', arrowprops=dict(arrowstyle='-', color=P['slate']))
    ax.annotate('5 frames', xy=(n_low, (1 - p_low) ** n_low), xytext=(5.15, 0.02), fontsize=15, color=P['red'], fontweight='bold', arrowprops=dict(arrowstyle='-', color=P['red']))
    ax.text(1.1, 3e-07, '95% per frame', fontsize=14, color=P['slate'])
    ax.text(1.4, 0.3, '75% per frame', fontsize=14, color=P['red'])
    ax.set_xlim(0.8, 6.4)
    ax.set_ylim(1e-08, 1.5)
    ax.set_xticks(n)
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Consecutive frames in view', fontsize=15)
    ax.set_ylabel('Missed in every frame', fontsize=15)
    save(fig, 'miss_frames')

def shortfall():
    v0, acc = (1.3, 2.0)
    x_ref = 0.23868
    x_person = 0.66118
    rows = [(0.85, 4, 1, 0.02166666667, 0.2943920289, 15.94666667), (0.75, 5, 2, 0.04333333333, 0.4163331999, 31.89333333), (0.6, 8, 5, 0.1083333333, 0.6582805886, 79.73333333)]

    def prof(xb):
        x = np.linspace(0, x_person, 600)
        v = np.where(x < xb, v0, np.sqrt(np.clip(v0 ** 2 - 2 * acc * (x - xb), 0, None)))
        return (x, v)
    groups = [(None, P['slate']), (rows[1], P['amber']), (rows[2], P['red'])]
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        fig.subplots_adjust(left=0.17, right=0.97, bottom=0.2, top=0.95)
        for k, (row, col) in enumerate(groups):
            sh = k < step
            dr = 0 if row is None else row[3]
            x, v = prof(x_ref + dr)
            ax.plot(x, v, color=col, lw=3, alpha=a(sh))
            if row is not None:
                ax.plot([x_person], [row[4]], 'o', color=col, ms=10, alpha=a(sh), zorder=5)
                g = rows.index(row)
                ax.annotate(f"seen in {_VALS[f'g{g}_pct']}%: {_VALS[f'g{g}_vi']} m/s, {_VALS[f'g{g}_e']} J", xy=(x_person, row[4]), xytext=(0.52, 0.44 if k == 1 else 0.68), ha='right', va='center', fontsize=14, color=col, fontweight='bold', alpha=a(sh), arrowprops=dict(arrowstyle='-', color=col, lw=1.0, alpha=a(sh)))
        ax.text(0.03, 0.06, 'seen in 95%:\nstops short', fontsize=14, color=P['slate'], fontweight='bold')
        ax.axvline(x_person, color=P['text'], lw=1.4, ls='--')
        ax.text(x_person - 0.02, 1.45, 'person', ha='right', fontsize=14)
        ax.set_xlim(0, x_person * 1.04)
        ax.set_ylim(0, 1.6)
        ax.tick_params(labelsize=14)
        ax.set_xlabel('Distance from first view (m)', fontsize=15)
        ax.set_ylabel('Speed (m/s)', fontsize=15)
        save(fig, f'shortfall_step{step}')

def li_conv():
    names = list({'YOLOX': (4, SimpleNamespace(distance_lost=Q_(0.1083333333, 'm'), extra_frames=5, extra_time=Q_(83.33333333, 'ms'), frames=9, impact_energy=Q_(79.73333333, 'J'), impact_speed=Q_(0.6582805886, 'm / s'), per_frame_detection=0.5753)), 'RetinaNet': (4, SimpleNamespace(distance_lost=Q_(0.1083333333, 'm'), extra_frames=5, extra_time=Q_(83.33333333, 'ms'), frames=9, impact_energy=Q_(79.73333333, 'J'), impact_speed=Q_(0.6582805886, 'm / s'), per_frame_detection=0.5567)), 'Faster RCNN': (3, SimpleNamespace(distance_lost=Q_(0.065, 'm'), extra_frames=3, extra_time=Q_(50.0, 'ms'), frames=6, impact_energy=Q_(47.84, 'J'), impact_speed=Q_(0.5099019514, 'm / s'), per_frame_detection=0.7394)), 'Cascade RCNN': (3, SimpleNamespace(distance_lost=Q_(0.065, 'm'), extra_frames=3, extra_time=Q_(50.0, 'ms'), frames=6, impact_energy=Q_(47.84, 'J'), impact_speed=Q_(0.5099019514, 'm / s'), per_frame_detection=0.7343)), 'ALFNet': (7, SimpleNamespace(distance_lost=Q_(0.1083333333, 'm'), extra_frames=5, extra_time=Q_(83.33333333, 'ms'), frames=12, impact_energy=Q_(79.73333333, 'J'), impact_speed=Q_(0.6582805886, 'm / s'), per_frame_detection=0.4653)), 'CSP': (8, SimpleNamespace(distance_lost=Q_(0.065, 'm'), extra_frames=3, extra_time=Q_(50.0, 'ms'), frames=11, impact_energy=Q_(47.84, 'J'), impact_speed=Q_(0.5099019514, 'm / s'), per_frame_detection=0.4958)), 'MGAN': (7, SimpleNamespace(distance_lost=Q_(0.065, 'm'), extra_frames=3, extra_time=Q_(50.0, 'ms'), frames=10, impact_energy=Q_(47.84, 'J'), impact_speed=Q_(0.5099019514, 'm / s'), per_frame_detection=0.5347)), 'PRNet': (9, SimpleNamespace(distance_lost=Q_(0.13, 'm'), extra_frames=6, extra_time=Q_(100.0, 'ms'), frames=15, impact_energy=Q_(95.68, 'J'), impact_speed=Q_(0.7211102551, 'm / s'), per_frame_detection=0.3875))})
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.22, top=0.97)
    for i, k in enumerate(names):
        y = len(names) - 1 - i
        _, row = {'YOLOX': (4, SimpleNamespace(distance_lost=Q_(0.1083333333, 'm'), extra_frames=5, extra_time=Q_(83.33333333, 'ms'), frames=9, impact_energy=Q_(79.73333333, 'J'), impact_speed=Q_(0.6582805886, 'm / s'), per_frame_detection=0.5753)), 'RetinaNet': (4, SimpleNamespace(distance_lost=Q_(0.1083333333, 'm'), extra_frames=5, extra_time=Q_(83.33333333, 'ms'), frames=9, impact_energy=Q_(79.73333333, 'J'), impact_speed=Q_(0.6582805886, 'm / s'), per_frame_detection=0.5567)), 'Faster RCNN': (3, SimpleNamespace(distance_lost=Q_(0.065, 'm'), extra_frames=3, extra_time=Q_(50.0, 'ms'), frames=6, impact_energy=Q_(47.84, 'J'), impact_speed=Q_(0.5099019514, 'm / s'), per_frame_detection=0.7394)), 'Cascade RCNN': (3, SimpleNamespace(distance_lost=Q_(0.065, 'm'), extra_frames=3, extra_time=Q_(50.0, 'ms'), frames=6, impact_energy=Q_(47.84, 'J'), impact_speed=Q_(0.5099019514, 'm / s'), per_frame_detection=0.7343)), 'ALFNet': (7, SimpleNamespace(distance_lost=Q_(0.1083333333, 'm'), extra_frames=5, extra_time=Q_(83.33333333, 'ms'), frames=12, impact_energy=Q_(79.73333333, 'J'), impact_speed=Q_(0.6582805886, 'm / s'), per_frame_detection=0.4653)), 'CSP': (8, SimpleNamespace(distance_lost=Q_(0.065, 'm'), extra_frames=3, extra_time=Q_(50.0, 'ms'), frames=11, impact_energy=Q_(47.84, 'J'), impact_speed=Q_(0.5099019514, 'm / s'), per_frame_detection=0.4958)), 'MGAN': (7, SimpleNamespace(distance_lost=Q_(0.065, 'm'), extra_frames=3, extra_time=Q_(50.0, 'ms'), frames=10, impact_energy=Q_(47.84, 'J'), impact_speed=Q_(0.5099019514, 'm / s'), per_frame_detection=0.5347)), 'PRNet': (9, SimpleNamespace(distance_lost=Q_(0.13, 'm'), extra_frames=6, extra_time=Q_(100.0, 'ms'), frames=15, impact_energy=Q_(95.68, 'J'), impact_speed=Q_(0.7211102551, 'm / s'), per_frame_detection=0.3875))}[k]
        dr, vi, ei = ({'YOLOX': 108.3333333, 'RetinaNet': 108.3333333, 'Faster RCNN': 65.0, 'Cascade RCNN': 65.0, 'ALFNet': 108.3333333, 'CSP': 65.0, 'MGAN': 65.0, 'PRNet': 130.0}[k], {'YOLOX': 0.6582805886, 'RetinaNet': 0.6582805886, 'Faster RCNN': 0.5099019514, 'Cascade RCNN': 0.5099019514, 'ALFNet': 0.6582805886, 'CSP': 0.5099019514, 'MGAN': 0.5099019514, 'PRNet': 0.7211102551}[k], {'YOLOX': 79.73333333, 'RetinaNet': 79.73333333, 'Faster RCNN': 47.84, 'Cascade RCNN': 47.84, 'ALFNet': 79.73333333, 'CSP': 47.84, 'MGAN': 47.84, 'PRNet': 95.68}[k])
        ax.barh(y, dr, color=P['red'], height=0.62)
        ax.text(dr + 2, y, f'{row.extra_frames} frames later: {vi:.2f} m/s, {ei:.0f} J', va='center', fontsize=11, color=P['red_d'])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=11)
    ax.set_xlim(0, 260)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_xlabel('Clearance lost for children (mm), aisle speed 1.3 m/s', fontsize=11.5)
    save(fig, 'li_conv')

def intervals():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    fig.subplots_adjust(left=0.3, right=0.96, bottom=0.22, top=0.95)
    rows = [('9,800\npeople', (0.9454983457, 0.9542317471), P['slate']), ('200\npeople', (0.9099724623, 0.9757658345), P['red'])]
    for i, (lab, (lo, hi), col) in enumerate(rows):
        y = 1 - i
        ax.plot([lo, hi], [y, y], color=col, lw=6, solid_capstyle='butt')
        ax.plot([0.95], [y], 'o', color='white', mec=col, mew=2.5, ms=11)
        ax.text(0.9035, y, lab, ha='right', va='center', fontsize=15, fontweight='bold', color=col)
        ax.text(0.95, y - 0.22, f'{lo:.3f} to {hi:.3f}', fontsize=14, color=col, ha='center', va='top')
    ax.axvline(0.92, color=P['amber'], lw=1.8, ls='--')
    ax.text(0.92 + 0.0015, -0.62, '0.92', fontsize=14, color=P['amber_d'])
    ax.set_xlim(0.905, 0.99)
    ax.set_ylim(-0.75, 1.6)
    ax.set_yticks([])
    ax.tick_params(labelsize=13)
    ax.set_xlabel('Detection rate, exact 95% interval', fontsize=15)
    save(fig, 'intervals')

def exposure():
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    fig.subplots_adjust(left=0.36, right=0.96, bottom=0.24, top=0.95)
    rows = [('10% who use\nthree crossings', 8.0, P['red']), ('Site average', 1.0, P['gray']), ('Everyone else', 0.2222222222, P['slate'])]
    for i, (lab, v, col) in enumerate(rows):
        y = 2 - i
        ax.barh(y, v, color=col, height=0.6)
        ax.text(-0.15, y, lab, ha='right', va='center', fontsize=15, fontweight='bold')
        ax.text(v + 0.15, y, f'{v:.0f}x' if v >= 1 else f'{v:.2f}x', va='center', fontsize=15, fontweight='bold', color=col)
    ax.set_xlim(0, 9.5)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Exposure per person', fontsize=15)
    save(fig, 'exposure')

def retention():
    day_h = 16.0
    window_s = 30.0
    day_s = 57600.0
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.3, top=0.97)
    ax.barh(1, day_h, color=P['red_l'], ec=P['red'], height=0.6)
    ax.text(day_h / 2, 1, 'every frame, all day', ha='center', va='center', fontsize=15, color=P['red_d'], fontweight='bold')
    rng = np.random.default_rng(17)
    starts = np.sort(rng.uniform(0, day_s - window_s, 20))
    for s0 in starts:
        ax.barh(0, window_s / 3600 * 6, left=s0 / 3600, color=P['green'], height=0.6)
    ax.text(-0.4, 1, 'Continuous', ha='right', va='center', fontsize=15, fontweight='bold')
    ax.text(-0.4, 0, 'Triggered', ha='right', va='center', fontsize=15, fontweight='bold')
    ax.set_xlim(0, day_h)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Hour of a 16 h day (marks 6x wide)', fontsize=14)
    save(fig, 'retention')

def labor():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.2, top=0.95)
    parts = [('requests', 8.0, P['red']), ('watching', 8.0, P['amber']), ('sweeps', 1.333333333, P['slate'])]
    left = 0
    for lab, v, col in parts:
        ax.barh(0, v, left=left, color=col, height=0.7)
        if v >= 2:
            ax.text(left + v / 2, 0, f'{lab}\n{v:.0f} h', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
        else:
            ax.text(left + 0.4, -0.6, f'{lab} {v:.1f} h', ha='left', va='center', fontsize=14, color=col, fontweight='bold')
        left += v
    shift = 8.0
    for k in (1, 2):
        ax.plot([shift * k, shift * k], [-0.8, 0.36], color=P['text'], lw=1.5, ls='--')
        ax.text(shift * k - 0.2, -0.6, f'{k} shift' + ('s' if k > 1 else ''), ha='right', va='center', fontsize=14)
    ax.text(left, 0.5, '17.3 person-hours a day', ha='right', fontsize=15, fontweight='bold', color=P['text'])
    ax.set_xlim(0, 21.5)
    ax.set_ylim(-0.85, 0.85)
    ax.set_yticks([])
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Hours per operating day, 20 machines', fontsize=15)
    save(fig, 'labor')

def main():
    cruise_timeline()
    decision_chain()
    chain_prob()
    time_test()
    queue()
    ladder()
    withdraw()
    stage_hours()
    fines()
    detect_data()
    miss_frames()
    shortfall()
    li_conv()
    intervals()
    exposure()
    retention()
    labor()
if __name__ == '__main__':
    main()
