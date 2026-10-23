#!/usr/bin/env python3
"""Figures for Lecture 3 (ML Lifecycle).

Every value comes from calcs.py, which reads MLSysIM: the retinopathy-screener
scenarios (Scenarios.RetinopathyScreenerRequirements, ...LateMemoryLimit,
AlwaysNormalScreener, FundusPhotoAt224, RetinopathyScreeningCloudVersusEdge,
RetinopathyMonitoringCadences), Literature.Surveys (CrowdFlower 2016), and
ReferenceStats.TorchvisionParameterCounts. Figure code holds only layout
(coordinates, sizes, colors) and labels; the chapter sections each figure
redraws are noted at the figure.
  - Six stages and their artifacts: Sec. 3.1. Stage contracts: table 2;
    audit case: Checkpoint 3.3 (Sec. 3.11).
  - ROC: the floors and the illustrative clinic point drawn over the measured
    curve of Gulshan et al., JAMA 2016, Fig. 2A (inset).
  - 224-pixel demonstration: fundus photo by Hao et al. (CC BY 4.0, Wikimedia
    Commons); the scenario records the photo's size and the measured lesion
    widths, and shrink_demo checks the file against it.

Run from this folder: python3 make_figs.py
"""
_VALS = {'gul_dev_images': '128,000', 'gul_sens': '97.5', 'gul_spec': '93.4', 'gul_hispec_sens': '90.3', 'gul_hispec_spec': '98.1', 'gul_eyepacs': '8,788', 'gul_graders': '8', 'gul_graders_word': 'eight', 'gul_auc': '0.991', 'beede_clinics': '11', 'beede_months_word': 'eight', 'beede_submitted': '1,838', 'beede_ungradable': '393', 'ungradable_pct': '21', 'model_gb': '4', 'budget_mb': '500', 'device_mem_lo': '4', 'device_mem_hi': '8', 'day_request': '1', 'day_val_a': '90', 'day_val_b': '120', 'val_a_pct': '92', 'val_b_pct': '94', 'day_handoff': '150', 'day_found': '151', 'day_check': '152', 'day_discard': '153', 'months_lost_word': 'five', 'overrun_x': '8', 'overrun_word': 'eight', 'fp32_bytes': '4', 'fp32_bytes_word': 'four', 'tv_mobilenet_v2': '3,504,872', 'mb_mobilenet_v2': '14', 'tv_resnet50': '25,557,032', 'mb_resnet50': '102', 'tv_vgg16': '138,357,544', 'mb_vgg16': '553', 'mnv2_mops': '600', 'rep_prep': '1', 'rep_train': '2', 'rep_val': '2', 'rep_pack': '1', 'rep_total': '6', 'rep_total_word': 'six', 'rep_tasks_word': 'four', 'sv_cleaning': '60', 'sv_collecting': '19', 'sv_data_work': '79', 'sv_algorithms': '4', 'it_weeks': '26', 'it_months_word': 'six', 'it_slow_cycle': '1', 'it_slow_cycle_word': 'one', 'it_fast_cycle': '1', 'it_fast_cycle_word': 'one', 'it_slow_start': '95', 'it_slow_gain': '0.15', 'it_fast_start': '90', 'it_fast_gain': '0.1', 'it_ceiling': '99', 'it_slow_runs': '26', 'it_slow_final': '98.9', 'it_hours_week': '168', 'it_fast_possible': '4,368', 'it_fast_runs': '100', 'it_fast_uncapped': '100', 'it_fast_final': '99', 'it_ideas_x': '3.8', 'it_gap': '0.1', 'floor_sens': '90', 'floor_spec': '80', 'power_lo': '7', 'power_hi': '15', 'gate_all_pct': '94', 'gate_cataract_pct': '76', 'trap_pop': '100', 'trap_sick': '10', 'trap_healthy': '90', 'trap_right': '90', 'trap_acc': '90', 'trap_sens': '0', 'trap_missed': '10', 'amazon_years': '10', 'amazon_years_word': 'ten', 'amazon_colleges_word': 'two', 'px_src_h': '857', 'px_src_w': '1,000', 'px_target': '224', 'px_scale': '0.26', 'lesion_lo': '8', 'lesion_hi': '12', 'lesion_mid': '10', 'lesion_lo_224': '2', 'lesion_hi_224': '3', 'lesion_mid_224': '3', 'fda_enrolled': '900', 'fda_sites': '10', 'fda_floor_sens': '85.0', 'fda_floor_spec': '82.5', 'fda_tp': '173', 'fda_sick': '198', 'fda_tn': '556', 'fda_healthy': '621', 'fda_sens': '87.4', 'fda_spec': '89.5', 'fda_sens_r': '87', 'fda_spec_r': '90', 'fda_image_r': '96', 'wilds_id': '93.2', 'wilds_ood': '70.3', 'wilds_drop': '22.9', 'up_patients': '100', 'up_photos': '4', 'up_photo_mb': '5', 'up_uplink_mbit': '2', 'up_uplink_mbs': '0.25', 'up_shift_h': '8', 'up_daily_mb': '2,000', 'up_daily_gb': '2', 'up_secs': '8,000', 'up_hours': '2.2', 'up_share': '28', 'up_share_1': '27.8', 'up_photo_s': '20', 'up_summary_kb': '10', 'up_reduction': '2,000', 'cost_clinics': '500', 'cost_patients': '50', 'cost_images_word': 'one', 'cost_cloud_img': '0.01', 'cost_cloud_inf': '91,250', 'cost_cloud_net': '45,000', 'cost_cloud_year': '136,250', 'cost_device': '500', 'cost_capex': '250,000', 'cost_maint': '25,000', 'cost_edge_inf': '9,125', 'cost_savings': '102,125', 'cost_payback': '2.4', 'cost_cloud_k': '136', 'cost_capex_k': '250', 'cost_edge_k': '34', 'cad_orders_word': 'five'}
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
out = Path(__file__).parent
C = sp.C
FONT = 'Helvetica, Arial, sans-serif'

def svg_open(w=900, h=500):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<defs>' + ''.join((f'<marker id="ar{k}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>' for k, col in (('n', '#555555'), ('d', C['data']), ('r', C['danger']), ('b', C['memory']), ('l', '#BBBBBB')))) + '</defs>', f'<rect width="{w}" height="{h}" fill="white"/>']

def text(x, y, s, size=20, color=None, weight='normal', anchor='middle', opacity=1.0):
    color = color or C['text']
    s = s.replace('&', '&amp;')
    return f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}" opacity="{opacity}">{s}</text>'

def line(x1, y1, x2, y2, color='#555555', w=3, marker='n', dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = f' marker-end="url(#ar{marker})"' if marker else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}"{d}{m} opacity="{opacity}"/>'

def save_svg(parts, name):
    (out / name).write_text('\n'.join(parts + ['</svg>']) + '\n')
BW, BH = (250, 100)
TOP, BOT = (140, 380)
POS = {1: (150, TOP), 2: (450, TOP), 3: (750, TOP), 4: (750, BOT), 5: (450, BOT), 6: (150, BOT)}
NAMES = {1: ('Define the', 'problem'), 2: ('Collect and', 'prepare data'), 3: ('Develop and', 'train model'), 4: ('Evaluate and', 'validate'), 5: ('Deploy and', 'integrate'), 6: ('Monitor and', 'maintain')}
DAM = {1: ('sets all three', C['text'], '#EEEEEE'), 2: ('Data', C['data'], C['data_l']), 3: ('Algorithm', C['compute'], C['compute_l']), 4: ('verifies', C['neutral'], '#F2F2F2'), 5: ('Machine', C['memory'], C['memory_l']), 6: ('verifies', C['neutral'], '#F2F2F2')}
LOOP_H = 500

def stage_box(k, fill='#F4F4F4', stroke='#777777', tcol=None, opacity=1.0, sw=2.5):
    x, y = POS[k]
    a, b = NAMES[k]
    tcol = tcol or C['text']
    return [f'<rect x="{x - BW / 2}" y="{y - BH / 2}" width="{BW}" height="{BH}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>', text(x - BW / 2 + 18, y - BH / 2 + 28, str(k), 24, stroke, 'bold', opacity=opacity), text(x + 10, y - 4, a, 26, tcol, 'bold', opacity=opacity), text(x + 10, y + 27, b, 26, tcol, 'bold', opacity=opacity)]

def flow_arrows(color='#555555', opacity=1.0, marker='n'):
    p = []
    g = BW / 2 + 4
    for a, b in ((1, 2), (2, 3)):
        p.append(line(POS[a][0] + g, TOP, POS[b][0] - g - 3, TOP, color, 3.5, marker, opacity=opacity))
    for a, b in ((4, 5), (5, 6)):
        p.append(line(POS[a][0] - g, BOT, POS[b][0] + g + 3, BOT, color, 3.5, marker, opacity=opacity))
    p.append(line(750, TOP + BH / 2 + 6, 750, BOT - BH / 2 - 10, color, 3.5, marker, opacity=opacity))
    return p
ARTIFACTS = [(300, 40, 'targets', 'and limits', 'middle'), (600, 40, 'versioned', 'dataset', 'middle'), (768, 252, 'trained', 'weights', 'start'), (600, 463, 'release', 'manifest', 'middle'), (300, 463, 'running', 'service', 'middle')]

def artifacts(opacity=1.0, color=None):
    color = color or C['neutral']
    p = []
    for x, y, a, b, align in ARTIFACTS:
        p.append(text(x, y, a, 23, color, 'normal', align, opacity))
        p.append(text(x, y + 26, b, 23, color, 'normal', align, opacity))
    return p

def return_arrow(opacity=1.0, label=True):
    p = [line(150, BOT - BH / 2 - 6, 150, TOP + BH / 2 + 10, C['data'], 4.5, 'd', dash='10,6', opacity=opacity)]
    if label:
        p.append(text(168, 254, 'monitoring', 23, C['data'], 'bold', 'start', opacity))
        p.append(text(168, 281, 'evidence', 23, C['data'], 'bold', 'start', opacity))
    return p

def loop(name, show_art=True, show_return=True, hl=None, dam=False):
    p = svg_open()
    for k in range(1, 7):
        if dam:
            tag, col, fill = DAM[k]
            p += stage_box(k, fill=fill, stroke=col)
            x, y = POS[k]
            p.append(text(x + (105 if k == 1 else 0), y + BH / 2 + 28, tag, 23, col, 'bold'))
        elif hl is None:
            p += stage_box(k)
        elif k == hl:
            p += stage_box(k, fill='#FBE3E6', stroke='#A51C30', sw=4)
        else:
            p += stage_box(k, opacity=0.35)
    op = 1.0 if hl is None else 0.35
    p += flow_arrows(opacity=op)
    if show_art:
        p += artifacts(opacity=op)
    if show_return:
        p += return_arrow(opacity=op)
    save_svg(p, name)
loop('loop_step1.svg', show_art=False, show_return=False)
loop('loop_step2.svg', show_art=True, show_return=False)
loop('loop_step3.svg')
CPOS = {1: (130, 115), 2: (130, 275), 3: (130, 435), 4: (430, 435), 5: (430, 275), 6: (430, 115)}
CNAMES = {1: 'Define', 2: 'Data', 3: 'Train', 4: 'Validate', 5: 'Deploy', 6: 'Monitor'}
CW, CH = (210, 84)

def loop_compact(name, tint=None):
    tint = tint or {}
    p = svg_open(560, 500)
    for k in range(1, 7):
        x, y = CPOS[k]
        col, fill = tint.get(k, ('#777777', '#F4F4F4'))
        tcol = col if k in tint else C['text']
        p.append(f'<rect x="{x - CW / 2}" y="{y - CH / 2}" width="{CW}" height="{CH}" rx="12" fill="{fill}" stroke="{col}" stroke-width="{(4 if k in tint else 2.5)}"/>')
        p.append(text(x, y + 12, f'{k}  {CNAMES[k]}', 34, tcol, 'bold'))
    g = CH / 2 + 4
    for a, b in ((1, 2), (2, 3)):
        p.append(line(130, CPOS[a][1] + g, 130, CPOS[b][1] - g - 4, '#555555', 3.5))
    for a, b in ((4, 5), (5, 6)):
        p.append(line(430, CPOS[a][1] - g, 430, CPOS[b][1] + g + 4, '#555555', 3.5))
    p.append(line(130 + CW / 2 + 4, 435, 430 - CW / 2 - 8, 435, '#555555', 3.5))
    p.append(line(430 - CW / 2 - 4, 115, 130 + CW / 2 + 10, 115, C['data'], 4.5, 'd', dash='10,6'))
    p.append(text(280, 40, 'monitoring evidence', 28, C['data'], 'bold'))
    save_svg(p, name)
loop_compact('loop_compact.svg')
loop_compact('loop_compact_fix.svg', tint={1: (C['compute'], C['compute_l']), 2: (C['data'], C['data_l']), 5: (C['memory'], C['memory_l'])})

def badge(x, y, s, col, fill):
    w = 14 * len(s) + 30
    return [f'<rect x="{x - w / 2}" y="{y - 21}" width="{w}" height="40" rx="20" fill="{fill}" stroke="{col}" stroke-width="2.5"/>', text(x, y + 8, s, 23, col, 'bold')]

def prop(step):
    p = svg_open()
    redo = {2, 3, 4, 5} if step >= 2 else set()
    for k in range(1, 7):
        if k in redo:
            p += stage_box(k, fill=C['danger_l'], stroke=C['danger'])
        elif step == 3 and k == 1:
            p += stage_box(k, fill=C['data_l'], stroke=C['data'], sw=4)
        else:
            p += stage_box(k)
    p += flow_arrows()
    p += return_arrow(label=False)
    p += badge(450, 474, '500 MB limit found here', C['danger'], 'white')
    if step >= 2:
        p.append(line(450, BOT - BH / 2 - 8, 450, TOP + BH / 2 + 12, C['danger'], 4.5, 'r', dash='9,6'))
        p.append(text(468, 268, 'redo', 24, C['danger'], 'bold', 'start'))
    if step == 3:
        p += badge(170, 26, 'state it here, day 1', C['data'], 'white')
    save_svg(p, f'propagation_step{step}.svg')
for s in (1, 2, 3):
    prop(s)

def chain_box(x, y, a, b, col, fill, tag=None, w=266):
    p = [f'<rect x="{x - w / 2}" y="{y - 46}" width="{w}" height="92" rx="10" fill="{fill}" stroke="{col}" stroke-width="3"/>', text(x, y - 6, a, 25, C['text'], 'bold'), text(x, y + 24, b, 25, C['text'], 'bold')]
    if tag:
        p.append(text(x, y - 56, tag, 22, col, 'bold'))
    return p

def directions(step):
    p = svg_open(900, 420)
    xs = (138, 450, 762)
    p.append(text(5, 32, 'Backward', 26, C['danger'], 'bold', 'start'))
    p += chain_box(xs[2], 120, '2 Mb/s uplink', 'at the clinic', C['danger'], C['danger_l'], 'stage 5')
    p += chain_box(xs[1], 120, 'compress photos', 'before upload', C['danger'], C['danger_l'])
    p += chain_box(xs[0], 120, 'train on', 'compressed photos', C['danger'], C['danger_l'], 'stage 2')
    p.append(line(626, 120, 589, 120, C['danger'], 5, 'r'))
    p.append(line(314, 120, 277, 120, C['danger'], 5, 'r'))
    if step >= 2:
        p.append(text(5, 242, 'Forward', 26, C['memory'], 'bold', 'start'))
        p += chain_box(xs[0], 340, 'lesions need', 'high resolution', C['memory'], C['memory_l'], 'stage 2')
        p += chain_box(xs[1], 340, 'larger input', 'to the model', C['memory'], C['memory_l'], 'stage 3')
        p += chain_box(xs[2], 340, 'more operations', 'per inference', C['memory'], C['memory_l'], 'stage 5')
        p.append(line(274, 340, 311, 340, C['memory'], 5, 'b'))
        p.append(line(586, 340, 623, 340, C['memory'], 5, 'b'))
    save_svg(p, f'directions_step{step}.svg')
directions(1)
directions(2)
EVENTS = [('day_request', 'Request:', 'one sentence', 'good'), ('day_val_a', '92%', 'sensitivity', 'good'), ('day_val_b', '94%', 'after tuning', 'good'), ('day_handoff', 'Handed to', 'deployment', 'warn'), ('day_found', 'Model', 'needs 4 GB', 'bad'), ('day_check', 'Reserved:', '500 MB', 'bad'), ('day_discard', 'Five months', 'discarded', 'bad')]

def timeline(step):
    show = {1: 3, 2: 6, 3: 7}[step]
    p = svg_open(900, 215)
    xs = [70, 196, 322, 448, 574, 700, 826]
    p.append(line(10, 85, 890, 85, '#999999', 5, None))
    for i, ((day, a, b, kind), x) in enumerate(zip(EVENTS, xs)):
        if i >= show:
            continue
        col = {'good': C['data'], 'warn': C['neutral'], 'bad': C['danger']}[kind]
        big = day == 'day_discard'
        p.append(f'<circle cx="{x}" cy="85" r="{(19 if big else 13)}" fill="{col}" stroke="white" stroke-width="3"/>')
        p.append(text(x, 45, f"Day { {'day_request': '1', 'day_val_a': '90', 'day_val_b': '120', 'day_handoff': '150', 'day_found': '151', 'day_check': '152', 'day_discard': '153'}[day]}", 24, col, 'bold'))
        p.append(text(x, 140, a, 21, C['text'], 'bold' if big else 'normal'))
        p.append(text(x, 166, b, 21, C['text'], 'bold' if big else 'normal'))
    p.append(text(450, 207, 'Events evenly spaced, not to scale', 16, C['neutral']))
    save_svg(p, f'day153_step{step}.svg')
for s in (1, 2, 3):
    timeline(s)
_mb = {'mobilenet_v2': 14.019488, 'resnet50': 102.228128, 'vgg16': 553.430176}
sp.hbars(out / 'memory_fit.svg', [('Model on day 151', 4000.0, 'danger', '4 GB: 8× the reservation'), ('Reserved for the model', 500, 'neutral', 'about 500 MB'), ('VGG-16 weights', _mb['vgg16'], 'neutral', '553 MB'), ('ResNet-50 weights', _mb['resnet50'], 'compute', '102 MB'), ('MobileNetV2 weights', _mb['mobilenet_v2'], 'compute', '14 MB')], 'MB', size=(5.8, 2.15), xlabel='Memory (MB, log scale; weights in FP32)')

def time_sinks(step):
    rows = []
    for name, v in [('Cleaning and organizing data', 60.0), ('Collecting datasets', 19.0), ('Mining data for patterns', 9.0), ('Other', 5.0), ('Refining algorithms', 4.0), ('Building training sets', 3.0)]:
        ck = 'data' if v in (60.0, 19.0) else 'compute' if 'algorithms' in name else 'neutral'
        if step == 1 and 'algorithms' in name:
            rows.append((name, 0, ck, '?'))
        else:
            rows.append((name, v, ck, f'{v:.0f}%'))
    sp.hbars(out / f'time_sinks_step{step}.svg', rows, '%', log=False, size=(5.8, 2.9), xlabel='Share of respondents naming it their biggest time sink (%)')
time_sinks(1)
time_sinks(2)

def gantt():
    tasks, start = ([], 0)
    for (name, d), key in zip([('Revise preparation', 1), ('Train new weights', 2), ('Repeat validation', 2), ('Package the release', 1)], ('rep_prep', 'rep_train', 'rep_val', 'rep_pack')):
        tasks.append((name, start, d, key))
        start += d
    total = 6.0
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.3))
    for i, (name, s, d, key) in enumerate(tasks):
        y = len(tasks) - 1 - i
        ax.barh(y, d, left=s, color=C['danger'], height=0.6)
        ax.text(-0.15, y, name, ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
        ax.text(s + d / 2, y, f'{_VALS[key]} d', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.annotate('', xy=(total, -0.75), xytext=(0, -0.75), arrowprops=dict(arrowstyle='<->', color=C['text'], lw=1.5))
    ax.text(total / 2, -1.1, '6 days in sequence', ha='center', va='top', fontsize=13, fontweight='bold', color=C['text'])
    ax.set_xlim(0, total + 0.2)
    ax.set_ylim(-1.6, len(tasks) - 0.4)
    ax.set_yticks([])
    ax.set_xticks(range(0, int(total) + 1))
    ax.set_xlabel('Days')
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'repair_gantt.svg')
gantt()

def patients(step):
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.7))
    for i in range(100):
        r, c = divmod(i, 10)
        sick = i >= 90
        if step == 1:
            col = C['danger'] if sick else '#BBBBBB'
        else:
            col = C['danger'] if sick else C['data']
        ax.add_patch(sp.plt.Rectangle((c * 1.1, (9 - r) * 1.1), 0.9, 0.9, color=col))
        if step == 2 and sick:
            ax.text(c * 1.1 + 0.45, (9 - r) * 1.1 + 0.42, '×', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    x0 = 12.3
    if step == 1:
        ax.text(x0, 8.3, '90 healthy', fontsize=15, fontweight='bold', color='#888888')
        ax.text(x0, 6.3, '10 with disease', fontsize=15, fontweight='bold', color=C['danger'])
    else:
        ax.text(x0, 8.6, 'Model says “normal” to all', fontsize=13, color=C['text'])
        ax.text(x0, 6.4, 'Accuracy  90%', fontsize=17, fontweight='bold', color=C['data'])
        ax.text(x0, 4.2, 'Sensitivity  0%', fontsize=17, fontweight='bold', color=C['danger'])
        ax.text(x0, 2.3, '10 of 10 cases missed', fontsize=13, color=C['danger'])
    ax.text(0, -0.9, 'Illustrative population', fontsize=9, color=C['neutral'])
    ax.set_xlim(-0.2, 23)
    ax.set_ylim(-1.3, 11.2)
    ax.set_aspect('equal')
    ax.axis('off')
    sp.save(fig, out / f'patients_step{step}.svg')
patients(1)
patients(2)
CROP0 = (505, 525)
X0, X30, Y70, Y100 = (571, 1125, 1116, 551.5)
CLINIC_POINT = tuple((100 * x for x in (0.88, 0.8)))
SPEC_FLOOR, SENS_FLOOR = (80.0, 90.0)

def roc_px(spec, sens, k):
    fpr = 100 - spec
    x = X0 + (X30 - X0) * fpr / 30 - CROP0[0]
    y = Y70 - (Y70 - Y100) * (sens - 70) / 30 - CROP0[1]
    return (x * k, y * k)

def roc_real(step):
    from PIL import Image, ImageDraw
    k = 3
    base = Image.open(out / 'gulshan2016_fig2a_inset.png').convert('RGB')
    im = base.resize((base.width * k, base.height * k), Image.LANCZOS)
    d = ImageDraw.Draw(im, 'RGBA')
    red = (165, 28, 48)
    if step >= 2:
        x20, y90 = roc_px(SPEC_FLOOR, SENS_FLOOR, k)
        x0, ytop = roc_px(100, 100, k)
        d.rectangle([x0, ytop, x20, y90], fill=(46, 125, 50, 40))
        _, ybot = roc_px(100, 70, k)
        xr, _ = roc_px(70, SENS_FLOOR, k)
        for a, b in (((x0, y90), (xr, y90)), ((x20, ytop), (x20, ybot))):
            n = 24
            for i in range(0, n, 2):
                p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                q = (a[0] + (b[0] - a[0]) * (i + 1) / n, a[1] + (b[1] - a[1]) * (i + 1) / n)
                d.line([p, q], fill=red, width=12)
    if step >= 3:
        cx, cy = roc_px(*CLINIC_POINT, k)
        r = 34
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=red, outline='white', width=8)
    im.thumbnail((1400, 1400))
    im.save(out / f'roc_real_step{step}.png')
for s in (1, 2, 3):
    roc_real(s)
CLOUD_K = 136.25
EDGE_RUN_K = 34.125
CAPEX_K = 250.0

def cost(step):
    import numpy as np
    t = np.linspace(0, 5, 100)
    cloud = CLOUD_K * t
    edge = CAPEX_K + EDGE_RUN_K * t
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.6))
    ax.plot(t, cloud, color=C['neutral'], lw=3.5)
    ax.text(5.05, cloud[-1], 'Cloud\n\\$136k per year', va='center', fontsize=12, fontweight='bold', color=C['neutral'])
    if step >= 2:
        ax.plot(t, edge, color=C['memory'], lw=3.5)
        ax.text(5.05, edge[-1] - 20, 'Edge\n\\$250k + \\$34k per year', va='center', fontsize=12, fontweight='bold', color=C['memory'])
    if step >= 3:
        tc = 2.447980416
        ax.plot([tc], [CLOUD_K * tc], 'o', ms=11, color=C['text'], mec='white', mew=1.5, zorder=5)
        ax.axvline(tc, color=C['text'], lw=1, ls=':')
        ax.text(tc + 0.08, 30, 'pays back at 2.4 years', fontsize=12, fontweight='bold', color=C['text'])
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 720)
    ax.set_xlabel('Years after deployment')
    ax.set_ylabel('Cumulative cost (\\$k)')
    ax.grid(axis='y', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'cost_step{step}.svg')
for s in (1, 2, 3):
    cost(s)
_cm = {'Minute': 1, 'Day': 1440, 'Week': 10080, 'Month': 43800.0, 'Quarter': 131400.0}
with sp.plt.rc_context({'xtick.labelsize': 14}):
    sp.hbars(out / 'cadences.svg', [('Quarter', _cm['Quarter'], 'neutral', 'architecture review'), ('Month', _cm['Month'], 'neutral', 'population coverage'), ('Week', _cm['Week'], 'memory', 'labeled accuracy, drift tests'), ('Day', _cm['Day'], 'memory', 'referral rate, confidence'), ('Minute', _cm['Minute'], 'danger', 'camera misconfigured')], 'min', size=(5.4, 2.6), xlabel='How long each signal takes to trust (minutes, log scale)')

def shrink_demo():
    from PIL import Image, ImageDraw
    im = Image.open(out / 'fundus-dr-hao.png').convert('RGB')
    w, h = im.size
    assert (w, h) == (1009, 857), 'photo differs from FundusPhotoAt224'
    n = 224
    left = 30
    sq = im.crop((left, 0, left + h, h))
    box = (546, 150, 656, 250)
    full = sq.crop(box).resize((440, 400), Image.LANCZOS)
    small = sq.resize((n, n), Image.LANCZOS)
    f = 0.2613768961
    sbox = tuple((round(v * f) for v in box))
    low = small.crop(sbox).resize((440, 400), Image.NEAREST)
    full.save(out / 'retina_full_crop.png')
    low.save(out / 'retina_224_crop.png')
    ov = sq.copy()
    d = ImageDraw.Draw(ov)
    d.rectangle(box, outline=(255, 255, 255), width=5)
    ov.resize((500, 500), Image.LANCZOS).save(out / 'retina_whole.png')
shrink_demo()

def chip_box(x, y, a, b, col, fill, w=290):
    return [f'<rect x="{x - w / 2}" y="{y - 34}" width="{w}" height="68" rx="10" fill="{fill}" stroke="{col}" stroke-width="3"/>', text(x, y - 7, a, 22, col, 'bold'), text(x, y + 22, b, 23, C['text'])]

def contracts(step):
    p = svg_open(900, 560)
    p.append('<g transform="translate(0,30)">')
    for k in range(1, 7):
        p += stage_box(k)
    p += flow_arrows()
    p += return_arrow(label=False)
    p.append('</g>')
    ok = (C['data'], C['data_l'])
    p.append(line(300, 106, 300, 160, C['data'], 3, None))
    p += chip_box(300, 70, '1 → 2', 'criteria are numbers', *ok)
    if step >= 2:
        p.append(line(287, 155, 313, 185, C['danger'], 6, None))
        p.append(line(287, 185, 313, 155, C['danger'], 6, None))
        p.append(text(300, 248, 'blocked:', 24, C['danger'], 'bold'))
        p.append(text(300, 276, 'target “TBD”', 24, C['danger'], 'bold'))
    if step >= 3:
        p.append(line(600, 106, 600, 160, C['data'], 3, None))
        p += chip_box(600, 70, '2 → 3', 'provenance per image', *ok)
        bad = step >= 4
        col, fill = (C['danger'], C['danger_l']) if bad else ok
        p.append(line(600, 470, 600, 418, col, 3, None))
        p += chip_box(600, 506, '4 → 5', 'no subgroup below floor', col, fill, w=320)
    if step >= 4:
        p.append(text(560, 320, 'no result for', 24, C['danger'], 'bold'))
        p.append(text(560, 348, 'patients over 70', 24, C['danger'], 'bold'))
    save_svg(p, f'contracts_step{step}.svg')
for s in (1, 2, 3, 4):
    contracts(s)

def gate(step):
    p = svg_open(980, 430)
    p.append('<rect x="8" y="8" width="694" height="414" rx="12" fill="#FAFAFA" stroke="#999999" stroke-width="2"/>')
    p.append(text(28, 50, 'Release manifest, one model version', 25, C['text'], 'bold', 'start'))
    p.append(text(450, 92, 'Floor', 21, C['neutral'], 'bold', 'start'))
    p.append(text(572, 92, 'Result', 21, C['neutral'], 'bold', 'start'))
    p.append('<line x1="22" y1="104" x2="688" y2="104" stroke="#BBBBBB" stroke-width="2"/>')
    rows = [(1, 'Sensitivity, all clinic photos', '90%', '94% pass', C['data']), (2, 'Sensitivity, cataract patients', '90%', '76% fail', C['danger']), (3, 'Peak memory on clinic device', '500 MB', 'measure', C['neutral']), (3, 'Latency on clinic device', 'in visit', 'measure', C['neutral']), (3, 'Confidence matches accuracy', 'per use', 'measure', C['neutral'])]
    for i, (k, name, floor, res, col) in enumerate(rows):
        if step < k:
            continue
        y = 146 + i * 54
        p.append(text(28, y, name, 24, C['text'], 'normal', 'start'))
        p.append(text(450, y, floor, 24, C['text'], 'normal', 'start'))
        p.append(text(572, y, res, 24, col, 'bold', 'start'))
    if step >= 3:
        p.append(text(28, 405, 'Also recorded: checkpoint hash, preprocessing, slices', 20, C['neutral'], 'normal', 'start'))
    if step >= 2:
        y = 200
        p.append(line(706, y - 8, 736, y - 8, C['danger'], 4, 'r'))
        p.append(f'''<rect x="742" y="{y - 92}" width="232" height="214" rx="12" fill="{C['danger_l']}" stroke="{C['danger']}" stroke-width="3"/>''')
        for j, (s_, wgt) in enumerate((('Gate halts.', 'bold'), ('Back to stage 2:', 'bold'), ('collect cataract', 'normal'), ('images, retrain', 'normal'), ('with lens-opacity', 'normal'), ('augmentation', 'normal'))):
            p.append(text(858, y - 58 + j * 32, s_, 21, C['danger'] if wgt == 'bold' else C['text'], wgt))
    save_svg(p, f'gate_step{step}.svg')
for s in (1, 2, 3):
    gate(s)
print('figures written')
