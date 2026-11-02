#!/usr/bin/env python3
"""Figures for Lecture 14 (Volume II): Robust AI.

Every number is read from MLSysIM through calcs.py (the same Scenarios.* and
Literature entries the chapter reads); the list below records each source.
"RA" is the rendered chapter, Robust AI. Derived values are labeled "derived"
on the figure or its slide; scenario values are labeled "illustrative".
Typed here: layout coordinates, the illustrative shift points (RA figure 3),
and the illustrative conformal score draw.

  - Hook (Eykholt et al., CVPR 2018, arXiv 1707.08945): LISA-CNN 91 percent
    accuracy on the LISA test set (Sec. 1); targeted success in 100 percent
    of lab images (abstract); drive-by field test, camouflage abstract art,
    84.8 percent of frames (Table 3). RA Example 14.1 quotes 100 and 84.8.
  - Tempe, 2018 (RA Sec. 14.2.2, NTSB HAR-19/03): detected 5.6 s before
    impact; reclassified among vehicle, other, bicycle; collision recognized
    1.2 s before impact; action suppressed 1 s; then alert and gradual
    slowdown. The report's table of ADS parameters (HAR-19/03, p. 15 to 16)
    logs the alert at -0.2 s, which equals 1.2 - 1.0 (checked in calcs.py).
    The preliminary report (HWY18MH010, May 2018) gave 1.3 s for the
    recognition; the final report's 1.2 s is used here and in Lecture 16.
  - Fleet silent corruption (RA Sec. 14.3.1 and its self-check 2):
    Pr(>= 1) = 1 - (1 - p)^N with p = 1e-4 per device-hour (assumed).
    Derived: N = 1,000 gives 0.095; N = 10,000 gives 0.632.
  - Shift panels: redrawn from RA figure 3 (illustrative points).
  - ImageNet-V2 (RA figure 4): MLSysIM vol2/14_robust_ai/data/imagenet_testbed_shift.csv
    (Taori et al. 2020 testbed; Recht et al. 2019), models with val >= 55.
  - Mean-shift test (RA Napkin Math 14.1): sigma 0.3, n 1,000, 0.50 -> 0.55;
    standard error 0.0095 and 5.3 standard errors (derived).
  - Fraud PSI (RA table 3): per-country contributions 0.0118, 0.0021,
    0.0007, 0.0036, 0.0470; sum 0.065. Other's share 72 percent (derived).
  - Calibration (Guo et al., ICML 2017, Table 1, ECE with 15 bins):
    CIFAR-100 LeNet 5 4.85 -> 2.02, CIFAR-100 ResNet 110 16.53 -> 1.26,
    ImageNet ResNet 152 5.48 -> 1.86 (uncalibrated -> temperature scaling).
  - Risk-coverage and conformal ranks: illustrative curves; conformal rank
    ceil((n + 1)(1 - alpha)) with n = 1,000 and alpha = 0.1 is 901 (derived).
  - Jailbreak transfer (Zou et al. 2023, arXiv 2307.15043, Table 2, 388
    behaviors): GPT-3.5 1.8, 5.7, 47.4, 86.6; GPT-4 8.0, 13.1, 29.1, 46.9
    (behavior only; + "Sure, here's"; + GCG on Vicuna and Guanacos;
    + ensemble).
  - PGD-7 work model (RA Napkin Math 14.2): 1 + 7 = 8 standard passes;
    FGSM listing about 3 (RA listing 1 and Napkin Math 14.2).
  - RobustBench (RA figure 10): MLSysIM
    vol2/14_robust_ai/data/robustbench_cifar10_linf.csv, dated and not flagged
    unreliable; l-infinity 8/255.
  - Certified radius (RA Sec. 14.6.1.1): sigma 0.5, pA 0.999, R = sigma *
    Phi^-1(pA) = 1.5. Derived: an l-infinity ball of 8/255 on a 32x32x3
    image reaches l2 length 8/255 * sqrt(3,072) = 1.74.
  - BadNets (Gu et al. 2017, arXiv 1708.06733, Table 4, yellow-square
    trigger, single target): baseline F-RCNN average clean 90.0 percent;
    BadNet average clean 89.3 percent; backdoored stop signs classified as
    speed limit 90.3 percent.
  - Bit flip (RA figure 14): 0.042 in FP32, bit 30 flips, x 2^128,
    about 1.4e37.

Run from this folder: python3 make_figs.py (derived slide numbers: calcs.py)
"""
_NUMS = {'lisa_clean_pct': 91.0, 'sticker_lab_pct': 100.0, 'sticker_drive_pct': 84.8, 'tempe_detect_s': 5.6, 'tempe_recog_s': 1.2, 'tempe_gap_s': 4.4, 'tempe_hold_s': 1.0, 'tempe_alert_s': 0.2, 'tempe_prelim_s': 1.3, 'sdc_p_exp': 4.0, 'sdc_n_lo': 1000.0, 'sdc_n_hi': 10000.0, 'sdc_q': 0.9999, 'sdc_p_lo': 0.095, 'sdc_p_hi': 0.63, 'sdc_np': 1.0, 'sdc_limit': 0.63, 'iv2_models': 212.0, 'iv2_floor': 55.0, 'iv2_gap_min': 7.0, 'iv2_gap_max': 14.0, 'iv2_gap_median': 12.0, 'iv2_alex_val': 57.0, 'iv2_alex_v2': 43.0, 'iv2_r50_val': 76.0, 'iv2_r50_v2': 63.0, 'ms_base': 0.5, 'ms_obs': 0.55, 'ms_shift': 0.05, 'ms_rel_pct': 10.0, 'ms_sd': 0.3, 'ms_n': 1000.0, 'ms_n_code': 1000.0, 'ms_se': 0.0095, 'ms_se_ch': 0.009, 'ms_z': 5.3, 'ms_p_bound': 0.001, 'ms_p_bound_pct': 0.1, 'psi_total': 0.065, 'psi_usa': 0.012, 'psi_uk': 0.002, 'psi_germany': 0.001, 'psi_france': 0.004, 'psi_other': 0.047, 'psi_other_pct': 72.0, 'psi_other_base_pct': 10.0, 'psi_other_now_pct': 18.0, 'kl_amount': 0.052, 'kl_trigger': 0.05, 'kl_mu0': 4.2, 'kl_sd0': 1.1, 'kl_mu1': 4.5, 'kl_sd1': 1.3, 'ks_stat': 0.089, 'ks_crit': 0.023, 'ks_n': 10000.0, 'ks_alpha': 0.01, 'ece_bins': 15.0, 'ece_lenet': 4.85, 'ece_lenet_ts': 2.02, 'ece_r110': 16.53, 'ece_r110_title': 16.5, 'ece_r110_ts': 1.26, 'ece_r152': 5.48, 'ece_r152_ts': 1.86, 'sel_rate': 10000.0, 'sel_cov_pct': 90.0, 'sel_rest_pct': 10.0, 'sel_fallback': 1000.0, 'sel_risk_pct': 6.5, 'sel_full_pct': 11.2, 'conf_n': 1000.0, 'conf_n1': 1001.0, 'conf_alpha': 0.1, 'conf_target': 0.9, 'conf_target_pct': 90.0, 'conf_bound': 900.9, 'conf_rank': 901.0, 'gcg_behaviors': 388.0, 'gpt35_only': 1.8, 'gpt4_only': 8.0, 'gpt35_sure': 5.7, 'gpt4_sure': 13.1, 'gpt35_gcg': 47.4, 'gpt4_gcg': 29.1, 'gpt35_ens': 86.6, 'gpt4_ens': 46.9, 'gpt35_ens_title': 87.0, 'fgsm_eps': 0.007, 'fgsm_panda_pct': 57.7, 'fgsm_gibbon_pct': 99.3, 'pgd_attack': 7.0, 'pgd_update': 1.0, 'pgd_passes': 8.0, 'cells_clean': 76.0, 'cells_robust': 50.0, 'cells_gap': 26.0, 'rb_first_year': 2018.0, 'rb_last_year': 2024.0, 'rb_first_pct': 44.0, 'rb_clean_lo': 79.0, 'rb_clean_hi': 95.0, 'rb_clean_min': 45.0, 'rb_last_pct': 75.0, 'rb_last_clean_pct': 94.0, 'rb_fail_pct': 25.0, 'rb_floor_pct': 5.0, 'rb_broken': 4.0, 'rb_chan_clean': 93.79, 'rb_chan_aa': 0.26, 'cert_p': 0.999, 'cert_sigma': 0.5, 'cert_q': 3.09, 'cert_r': 1.5, 'eps_levels': 8.0, 'eps_scale': 255.0, 'cert_d': 3072.0, 'cert_corner': 1.74, 'cifar_w': 32.0, 'cifar_h': 32.0, 'cifar_c': 3.0, 'bn_clean': 90.0, 'bn_back_clean': 89.3, 'bn_drop': 0.7, 'bn_attack': 90.3, 'bn_attack_title': 90.0, 'bn_postit': 95.0, 'flip_w': 0.042, 'flip_bit': 30.0, 'flip_scale': 128.0, 'flip_mant': 1.4, 'flip_exp': 37.0, 'flip_word': 32.0, 'flip_exp_bits': 8.0, 'flip_frac_bits': 23.0, 'llama_sdc': 6.0, 'llama_interrupts': 419.0, 'gpt3_params_b': 175.0}
import struct
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
C = sp.C
plt = sp.plt
out = Path(__file__).parent
V = {'blue': '#1F407A', 'blue_l': '#EAF1F8', 'amber': '#d97706', 'amber_l': '#fef3c7', 'violet': '#7c3aed', 'violet_l': '#f3e8ff', 'green': '#16a34a', 'green_l': '#dcfce7', 'red': '#dc2626', 'red_l': '#fee2e2', 'frame': '#94a3b8', 'frame_l': '#f8fafc', 'ink': '#0f172a', 'gray': '#64748b'}

def svg(path, body, w=900, h=500):
    head = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="Helvetica, Arial, sans-serif">\n<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>\n<defs>' + ''.join((f'<marker id="a-{k}" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="{V[k]}"/></marker>' for k in ('blue', 'amber', 'violet', 'green', 'red', 'gray'))) + '</defs>\n'
    Path(path).write_text(head + '\n'.join(body) + '\n</svg>\n')

def rect(x, y, w, h, fill, stroke, sw=2, r=6, op=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>'

def text(x, y, s, color=V['ink'], size=20, weight=400, anchor='middle', op=1.0, italic=False):
    st = ' font-style="italic"' if italic else ''
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}" opacity="{op}"{st}>{s}</text>'

def arrow(x1, y1, x2, y2, color='gray', sw=2.5, op=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{V[color]}" stroke-width="{sw}" marker-end="url(#a-{color})" opacity="{op}"{d}/>'

def circle(cx, cy, r, fill, stroke, sw=2.5, op=1.0):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>'

def tag(ax, s, x=0.0, y=-0.28):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, color=C['neutral'], ha='left', va='top')
HOOK_ROWS = [('Clean test set', 91.0, 'neutral', '91% correct'), ('Lab photos', 100.0, 'compute', '100% Speed Limit 45'), ('Drive-by frames', 84.8, 'danger', '84.8% Speed Limit 45')]

def hook(path, k, size=(4.0, 1.85), fs=12.5):
    rows = HOOK_ROWS
    fig, ax = plt.subplots(figsize=size)
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 2 - i
        shown = i < 2 and k >= i + 1 or (i == 2 and k >= 3)
        ax.barh(y, v, color=C[ck], height=0.66, alpha=1.0 if shown else 0.0)
        ax.text(-2, y, lab, ha='right', va='center', fontsize=fs, fontweight='bold', alpha=1.0 if i <= k or i == 2 else 0.0)
        ax.text(3, y, txt, va='center', fontsize=fs - 0.5, fontweight='bold', color='white', alpha=1.0 if shown else 0.0)
        if i == 2 and k == 2:
            ax.text(2, y, '???', va='center', fontsize=fs + 2, fontweight='bold', color=C['danger'])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Percent, LISA-CNN road-sign classifier')
    sp.save(fig, path)
for k in (1, 2, 3):
    hook(out / f'hook_step{k}.svg', k)

def hook_synth(path, size=(3.4, 2.7), fs=15):
    """The hook's final step at the size of the closing slide's narrow column:
    each row's name sits above its bar, and the values print inside the bars."""
    rows = HOOK_ROWS
    fig, ax = plt.subplots(figsize=size)
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 2 - i
        ax.barh(y, v, color=C[ck], height=0.44)
        ax.text(0, y + 0.27, lab, ha='left', va='bottom', fontsize=fs, fontweight='bold')
        ax.text(2, y, txt, va='center', fontsize=fs - 1, fontweight='bold', color='white')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.35, 2.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sd in ax.spines.values():
        sd.set_visible(False)
    ax.text(0, -0.4, 'Percent, LISA-CNN road-sign classifier', ha='left', va='top', fontsize=fs - 2, color=C['neutral'])
    sp.save(fig, path)
hook_synth(out / 'hook_synth.svg')

def tempe(path, k):
    fig, ax = plt.subplots(figsize=(4.1, 2.3))
    ax.set_xlim(-6.5, 0.9)
    ax.set_ylim(-1.55, 1.35)
    ax.axhline(0, color=C['neutral'], lw=2)
    for t in (-6, -4, -2, 0):
        ax.plot([t, t], [-0.08, 0.08], color=C['neutral'], lw=1.5)
        ax.text(t, -0.3, f'{t} s' if t else 'impact', ha='center', va='top', fontsize=11.5)
    detect, recog, hold, alert = (5.6, 1.2, 1.0, 0.2)
    ax.plot([-detect], [0], 'o', ms=12, color=C['memory'], zorder=5)
    ax.text(-detect, -0.72, 'first\ndetected\n5.6 s out', ha='center', va='top', fontsize=11.5, fontweight='bold', color=C['memory'])
    ax.add_patch(sp.plt.Rectangle((-5.4, -0.18), 4.2, 0.36, color=C['memory_l'], zorder=1))
    ax.text(-3.4, 0.32, 'label flips: vehicle,\nother, bicycle', ha='center', va='bottom', fontsize=11.5, color=C['memory'])
    a = 1.0 if k >= 2 else 0.0
    ax.plot([-recog], [0], 'o', ms=12, color=C['danger'], zorder=5, alpha=a)
    ax.add_patch(sp.plt.Rectangle((-recog, -0.18), hold, 0.36, color=C['danger_l'], zorder=2, alpha=a))
    ax.text(-1.5, -0.72, 'collision\nrecognized\n1.2 s out', ha='center', va='top', fontsize=11.5, fontweight='bold', color=C['danger'], alpha=a)
    ax.text(-0.5, 0.32, 'held 1 s,\nno alert', ha='center', va='bottom', fontsize=11.5, fontweight='bold', color=C['danger'], alpha=a)
    ax.plot([-alert], [0], 'v', ms=10, color=C['danger'], zorder=5, alpha=a)
    ax.text(0.35, -0.72, 'alert\n0.2 s out', ha='center', va='top', fontsize=11.5, color=C['danger'], alpha=a)
    ax.axis('off')
    sp.save(fig, path)
for k in (1, 2):
    tempe(out / f'tempe_step{k}.svg', k)

def pillars(path, k):
    cols = [('Environmental shift', 'the world moves', 'slices, drift tests', 'recalibrate, retrain', V['blue'], V['blue_l']), ('Input-level attack', 'an adversary picks', 'attack suites', 'harden, filter', V['amber'], V['amber_l']), ('System-level fault', 'hardware or code errs', 'replay, canaries', 'isolate, roll back', V['violet'], V['violet_l'])]
    b = [text(450, 30, 'Latency, uptime, error rate: all green', V['green'], 24, 700)]
    for i, (name, what, ev, resp, s, f) in enumerate(cols):
        x = 20 + i * 293
        op = 1.0 if k >= i + 1 else 0.0
        b.append(rect(x, 48, 274, 228, f, s, 2.5, 8, op))
        b.append(text(x + 137, 84, name, s, 25, 700, op=op))
        b.append(text(x + 137, 118, what, V['ink'], 21, 400, op=op))
        b.append(text(x + 137, 164, 'evidence: ' + ev, V['ink'], 20, 400, op=op))
        b.append(text(x + 137, 204, 'response:', V['gray'], 20, 700, op=op))
        b.append(text(x + 137, 234, resp, V['ink'], 20, 400, op=op))
    op = 1.0 if k >= 4 else 0.0
    b.append(rect(20, 292, 860, 84, V['red_l'], V['red'], 2.5, 8, op, dash='8,5'))
    b.append(text(450, 325, 'Software and pipeline defects cut across all three', V['red'], 23, 700, op=op))
    b.append(text(450, 358, 'a preprocessing bug looks like shift; an overflow looks like an attack', V['ink'], 20, 400, op=op))
    svg(path, b, 900, 385)
for k in (1, 2, 3, 4):
    pillars(out / f'pillars_step{k}.svg', k)

def sdc(path, k):
    p = 0.0001
    n = np.logspace(1, 5, 200)
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.plot(n, 1 - (1 - p) ** n, color=C['danger'], lw=3)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.08)
    ax.set_xlim(10, 100000.0)
    ax.set_xlabel('Accelerators in the job, N')
    ax.set_ylabel('P(silent error\nin an hour)')
    ax.grid(color=C['grid'], lw=0.6)
    pts = [(1000, '1,000: 0.095'), (10000, '10,000: 0.63')]
    for i, (x, lab) in enumerate(pts):
        if k >= 2 or i == 0:
            y = 1 - (1 - p) ** x
            ax.plot([x], [y], 'o', ms=10, color=C['danger'], mec='white', mew=1.5, zorder=5)
            ax.annotate(lab, (x, y), xytext=(-8, 8), textcoords='offset points', ha='right', fontsize=12, fontweight='bold', color=C['danger'])
    sp.save(fig, path)
for k in (1, 2):
    sdc(out / f'sdc_step{k}.svg', k)

def shift(path, k):
    b = []
    titles = [('Covariate shift', 'P(x) changes', 'P(y | x) fixed', V['blue']), ('Label shift', 'P(y) changes', 'P(x | y) fixed', V['amber']), ('Concept drift', 'P(y | x) changes', 'P(x) can stay fixed', V['red'])]
    for i, (t, a1, a2, col) in enumerate(titles):
        x0 = 15 + i * 295
        op = 1.0 if k >= i + 1 else 0.0
        b.append(rect(x0, 8, 280, 372, V['frame_l'], V['frame'], 2, 8, op))
        b.append(text(x0 + 140, 42, t, col, 26, 700, op=op))
        b.append(text(x0 + 140, 74, a1, V['ink'], 22, 700, op=op))
        b.append(text(x0 + 140, 102, a2, V['gray'], 21, 400, op=op))
    x0 = 15
    op = 1.0 if k >= 1 else 0
    b.append(f'''<line x1="{x0 + 40}" y1="360" x2="{x0 + 250}" y2="130" stroke="{V['ink']}" stroke-width="2.5" stroke-dasharray="8,6" opacity="{op}"/>''')
    for cx, cy, filled in [(65, 200, 0), (95, 245, 0), (125, 215, 0), (90, 330, 1), (130, 345, 1)]:
        b.append(circle(x0 + cx, cy, 10, V['blue'] if filled else '#fff', V['blue'], 3, op))
    for cx, cy, filled in [(200, 145, 0), (235, 170, 0), (215, 280, 1), (255, 262, 1), (240, 315, 1)]:
        b.append(circle(x0 + cx, cy, 10, V['amber'] if filled else '#fff', V['amber'], 3, op))
    b.append(text(x0 + 75, 168, 'train', V['blue'], 21, 700, op=op))
    b.append(text(x0 + 225, 360, 'deploy', V['amber'], 21, 700, op=op))
    x0 = 310
    op = 1.0 if k >= 2 else 0
    base = 340
    for j, (h_tr, h_dep) in enumerate([(100, 45), (100, 175)]):
        bx = x0 + 45 + j * 115
        b.append(f'''<rect x="{bx}" y="{base - h_tr}" width="40" height="{h_tr}" fill="{V['blue']}" opacity="{op}"/>''')
        b.append(f'''<rect x="{bx + 45}" y="{base - h_dep}" width="40" height="{h_dep}" fill="{V['amber']}" opacity="{op}"/>''')
        b.append(text(bx + 42, 368, f'y = {j}', V['ink'], 21, 400, op=op))
    b.append(f'''<line x1="{x0 + 30}" y1="{base}" x2="{x0 + 250}" y2="{base}" stroke="{V['ink']}" stroke-width="2" opacity="{op}"/>''')
    b.append(text(x0 + 95, 145, 'train', V['blue'], 21, 700, op=op))
    b.append(text(x0 + 210, 145, 'deploy', V['amber'], 21, 700, op=op))
    x0 = 605
    op = 1.0 if k >= 3 else 0
    b.append(f'''<line x1="{x0 + 40}" y1="350" x2="{x0 + 250}" y2="130" stroke="{V['ink']}" stroke-width="2.5" stroke-dasharray="8,6" opacity="{op}"/>''')
    b.append(f'''<line x1="{x0 + 30}" y1="160" x2="{x0 + 255}" y2="335" stroke="{V['green']}" stroke-width="3.5" opacity="{op}"/>''')
    b.append(text(x0 + 215, 135, 'old', V['ink'], 21, 700, op=op))
    b.append(text(x0 + 60, 145, 'new', V['green'], 21, 700, op=op))
    for cx, cy in [(55, 205), (90, 255), (150, 340), (215, 345)]:
        b.append(circle(x0 + cx, cy, 10, V['blue'], V['blue'], 3, op))
    for cx, cy in [(170, 185), (215, 215), (245, 245)]:
        b.append(circle(x0 + cx, cy, 10, '#fff', V['blue'], 3, op))
    for cx, cy in [(120, 215), (180, 270), (222, 292)]:
        b.append(circle(x0 + cx, cy, 10, V['red'], V['red'], 3, op))
    svg(path, b, 900, 388)
for k in (1, 2, 3):
    shift(out / f'shift_step{k}.svg', k)
IV2 = [{'model': 'efficientnet-l2-noisystudent', 'val': 88.32200623, 'imagenet_v2': 80.84999847, 'objectnet': 68.45090202, 'imagenet_r': 74.67333221, 'imagenet_sketch': 47.64094543}, {'model': 'FixResNeXt101_32x48d_v2', 'val': 86.36200714, 'imagenet_v2': 77.93999481, 'objectnet': 57.84811494, 'imagenet_r': 79.96666718, 'imagenet_sketch': 59.14441299}, {'model': 'FixResNeXt101_32x48d', 'val': 86.25600433, 'imagenet_v2': 77.76999664, 'objectnet': 58.01015448, 'imagenet_r': 79.90333557, 'imagenet_sketch': 59.04026794}, {'model': 'instagram-resnext101_32x48d', 'val': 85.44200134, 'imagenet_v2': 76.95999908, 'objectnet': 59.12282597, 'imagenet_r': 79.65666962, 'imagenet_sketch': 58.81428528}, {'model': 'efficientnet-b8-advprop-autoaug', 'val': 85.36800385, 'imagenet_v2': 76.11000061, 'objectnet': 52.78167873, 'imagenet_r': 57.83333588, 'imagenet_sketch': 45.77806854}, {'model': 'BiT-M-R152x4-ILSVRC2012', 'val': 85.1780014, 'imagenet_v2': 75.02999878, 'objectnet': 50.28626985, 'imagenet_r': 55.34666824, 'imagenet_sketch': 43.01715851}, {'model': 'vit_large_patch16_384', 'val': 85.15400696, 'imagenet_v2': 75.36999512, 'objectnet': 54.48849519, 'imagenet_r': 54.75, 'imagenet_sketch': 43.30405426}, {'model': 'instagram-resnext101_32x32d', 'val': 85.09200287, 'imagenet_v2': 76.76999664, 'objectnet': 58.08577293, 'imagenet_r': 79.45999908, 'imagenet_sketch': 58.38000488}, {'model': 'efficientnet-b7-advprop-autoaug', 'val': 85.09200287, 'imagenet_v2': 75.88999939, 'objectnet': 51.52317165, 'imagenet_r': 53.91666794, 'imagenet_sketch': 41.7909584}, {'model': 'BiT-M-R101x3-ILSVRC2012', 'val': 84.78200531, 'imagenet_v2': 73.45999908, 'objectnet': 45.77616939, 'imagenet_r': 50.86333466, 'imagenet_sketch': 36.73092651}, {'model': 'efficientnet-b6-advprop-autoaug', 'val': 84.76000214, 'imagenet_v2': 75.52999878, 'objectnet': 50.1512369, 'imagenet_r': 53.63666916, 'imagenet_sketch': 41.31148529}, {'model': 'efficientnet-b7-randaug', 'val': 84.72600555, 'imagenet_v2': 74.83999634, 'objectnet': 52.5008102, 'imagenet_r': 52.33333588, 'imagenet_sketch': 40.87524033}, {'model': 'efficientnet-b7-autoaug', 'val': 84.33400726, 'imagenet_v2': 74.52999878, 'objectnet': 49.6381117, 'imagenet_r': 46.06000137, 'imagenet_sketch': 31.86543465}, {'model': 'efficientnet-b5-advprop-autoaug', 'val': 84.30200195, 'imagenet_v2': 74.68000031, 'objectnet': 49.96759209, 'imagenet_r': 54.08000183, 'imagenet_sketch': 41.83615494}, {'model': 'resnext101_32x8d_swsl', 'val': 84.29400635, 'imagenet_v2': 75.33999634, 'objectnet': 56.32494329, 'imagenet_r': 75.59000397, 'imagenet_sketch': 56.43459702}, {'model': 'vit_base_patch16_384', 'val': 84.20000458, 'imagenet_v2': 74.31999969, 'objectnet': 51.11807281, 'imagenet_r': 50.88666916, 'imagenet_sketch': 38.10057449}, {'model': 'instagram-resnext101_32x16d', 'val': 84.1760025, 'imagenet_v2': 75.50999451, 'objectnet': 56.23312088, 'imagenet_r': 78.84999847, 'imagenet_sketch': 57.70009613}, {'model': 'BiT-M-R50x3-ILSVRC2012', 'val': 84.15000153, 'imagenet_v2': 73.19999695, 'objectnet': 41.24446365, 'imagenet_r': 47.90000153, 'imagenet_sketch': 33.74992752}, {'model': 'efficientnet-b6-autoaug', 'val': 84.12800598, 'imagenet_v2': 74.06999969, 'objectnet': 50.19984876, 'imagenet_r': 47.92000198, 'imagenet_sketch': 34.70101929}, {'model': 'FixPNASNet', 'val': 83.69800568, 'imagenet_v2': 73.36000061, 'objectnet': 43.50221454, 'imagenet_r': 42.42333603, 'imagenet_sketch': 31.49207306}, {'model': 'efficientnet-b5-autoaug', 'val': 83.63000488, 'imagenet_v2': 73.51999664, 'objectnet': 49.11958518, 'imagenet_r': 48.65999985, 'imagenet_sketch': 35.95079803}, {'model': 'efficientnet-b5-randaug', 'val': 83.53200531, 'imagenet_v2': 73.22000122, 'objectnet': 50.03240791, 'imagenet_r': 49.06000137, 'imagenet_sketch': 37.82743073}, {'model': 'resnext101_32x4d_swsl', 'val': 83.23400116, 'imagenet_v2': 74.15999603, 'objectnet': 52.5926326, 'imagenet_r': 72.67333221, 'imagenet_sketch': 53.59115219}, {'model': 'efficientnet-b5', 'val': 83.11400604, 'imagenet_v2': 72.76999664, 'objectnet': 46.59716971, 'imagenet_r': 45.0, 'imagenet_sketch': 31.53530312}, {'model': 'vit_large_patch16_224', 'val': 83.05600739, 'imagenet_v2': 72.31999969, 'objectnet': 47.65582802, 'imagenet_r': 50.88999939, 'imagenet_sketch': 40.73375702}, {'model': 'pnasnet5large', 'val': 82.73600769, 'imagenet_v2': 72.30999756, 'objectnet': 46.18126823, 'imagenet_r': 44.02666855, 'imagenet_sketch': 32.86761475}, {'model': 'instagram-resnext101_32x8d', 'val': 82.68800354, 'imagenet_v2': 73.62999725, 'objectnet': 53.39742897, 'imagenet_r': 75.80667114, 'imagenet_sketch': 54.91756821}, {'model': 'efficientnet-b4-advprop-autoaug', 'val': 82.68600464, 'imagenet_v2': 72.45999908, 'objectnet': 47.70443988, 'imagenet_r': 53.22999954, 'imagenet_sketch': 40.42917252}, {'model': 'efficientnet-b4-autoaug', 'val': 82.55000305, 'imagenet_v2': 72.02999878, 'objectnet': 47.89348601, 'imagenet_r': 47.53000259, 'imagenet_sketch': 34.32962036}, {'model': 'BiT-M-R101x1-ILSVRC2012', 'val': 82.52200317, 'imagenet_v2': 71.40999603, 'objectnet': 39.20276547, 'imagenet_r': 44.00666809, 'imagenet_sketch': 29.96718407}, {'model': 'nasnetalarge', 'val': 82.50600433, 'imagenet_v2': 72.29000092, 'objectnet': 47.08868964, 'imagenet_r': 45.20333481, 'imagenet_sketch': 32.86761475}, {'model': 'efficientnet-b4', 'val': 82.22800446, 'imagenet_v2': 70.80999756, 'objectnet': 46.19747218, 'imagenet_r': 43.65666962, 'imagenet_sketch': 30.07722855}, {'model': 'resnext50_32x4d_swsl', 'val': 82.18000031, 'imagenet_v2': 72.65000153, 'objectnet': 49.38424976, 'imagenet_r': 68.97666931, 'imagenet_sketch': 50.44902039}, {'model': 'resnext101_32x16d_ssl', 'val': 81.83600616, 'imagenet_v2': 71.25, 'objectnet': 47.2885384, 'imagenet_r': 50.22999954, 'imagenet_sketch': 34.60865784}, {'model': 'vit_base_patch16_224', 'val': 81.79000092, 'imagenet_v2': 71.33000183, 'objectnet': 44.69050448, 'imagenet_r': 47.33666611, 'imagenet_sketch': 35.76411438}, {'model': 'vit_base_patch32_384', 'val': 81.65800476, 'imagenet_v2': 71.23999786, 'objectnet': 43.76687912, 'imagenet_r': 44.70000076, 'imagenet_sketch': 32.46084595}, {'model': 'resnext101_32x8d_ssl', 'val': 81.62600708, 'imagenet_v2': 71.40999603, 'objectnet': 46.18666955, 'imagenet_r': 49.08666611, 'imagenet_sketch': 34.02110672}, {'model': 'vit_large_patch32_384', 'val': 81.51200104, 'imagenet_v2': 70.58999634, 'objectnet': 44.94976774, 'imagenet_r': 48.23666763, 'imagenet_sketch': 34.67154312}, {'model': 'senet154', 'val': 81.30400085, 'imagenet_v2': 70.19999695, 'objectnet': 43.3077671, 'imagenet_r': 42.22666931, 'imagenet_sketch': 30.00059128}, {'model': 'resnet50_swsl', 'val': 81.18000031, 'imagenet_v2': 71.45999908, 'objectnet': 48.44982176, 'imagenet_r': 68.30667114, 'imagenet_sketch': 49.55098724}, {'model': 'efficientnet-b3-advprop-autoaug', 'val': 81.09000397, 'imagenet_v2': 70.29000092, 'objectnet': 43.95592525, 'imagenet_r': 50.1166687, 'imagenet_sketch': 36.89009476}, {'model': 'efficientnet-b3-autoaug', 'val': 81.05400085, 'imagenet_v2': 70.11999512, 'objectnet': 44.1827806, 'imagenet_r': 45.01000214, 'imagenet_sketch': 32.62198257}, {'model': 'resnet152_3x_simclrv2_finetuned_100pct_tf_port', 'val': 81.04200745, 'imagenet_v2': 70.34999847, 'objectnet': 46.94825537, 'imagenet_r': 47.29666901, 'imagenet_sketch': 35.06258774}, {'model': 'resnext101_32x4d_ssl', 'val': 80.9280014, 'imagenet_v2': 70.26999664, 'objectnet': 45.12801123, 'imagenet_r': 47.16666794, 'imagenet_sketch': 31.43312073}, {'model': 'polynet', 'val': 80.85800171, 'imagenet_v2': 70.06999969, 'objectnet': 44.93896511, 'imagenet_r': 42.22999954, 'imagenet_sketch': 29.36391068}, {'model': 'BiT-M-R152x4-nonfinetuned', 'val': 80.48800659, 'imagenet_v2': 68.79000092, 'objectnet': 32.46192071, 'imagenet_r': 37.84666824, 'imagenet_sketch': 25.57330704}, {'model': 'BiT-M-R50x1-ILSVRC2012', 'val': 80.40200043, 'imagenet_v2': 68.5, 'objectnet': 34.85470455, 'imagenet_r': 39.19000244, 'imagenet_sketch': 25.16654015}, {'model': 'resnext50_32x4d_ssl', 'val': 80.32800293, 'imagenet_v2': 69.47999573, 'objectnet': 43.3077671, 'imagenet_r': 45.41666794, 'imagenet_sketch': 30.59403992}, {'model': 'inceptionresnetv2', 'val': 80.27000427, 'imagenet_v2': 69.29000092, 'objectnet': 43.99913579, 'imagenet_r': 44.11333466, 'imagenet_sketch': 32.73398972}, {'model': 'se_resnext101_32x4d', 'val': 80.23600006, 'imagenet_v2': 69.40000153, 'objectnet': 40.52608837, 'imagenet_r': 40.35666656, 'imagenet_sketch': 26.81915665}, {'model': 'efficientnet-b3', 'val': 80.21200562, 'imagenet_v2': 68.41999817, 'objectnet': 42.81624716, 'imagenet_r': 42.15333557, 'imagenet_sketch': 28.64273262}, {'model': 'inceptionv4', 'val': 80.08200073, 'imagenet_v2': 69.23999786, 'objectnet': 42.79464189, 'imagenet_r': 41.96333313, 'imagenet_sketch': 31.56674385}, {'model': 'BiT-M-R101x3-nonfinetuned', 'val': 79.97400665, 'imagenet_v2': 68.86999512, 'objectnet': 31.83536783, 'imagenet_r': 37.05333328, 'imagenet_sketch': 25.2647934}, {'model': 'resnext101_32x8d_deepaugment_augmix', 'val': 79.89600372, 'imagenet_v2': 68.76000214, 'objectnet': 41.1094307, 'imagenet_r': 52.11000061, 'imagenet_sketch': 38.43267059}, {'model': 'resnet101_cutmix', 'val': 79.82600403, 'imagenet_v2': 68.02999878, 'objectnet': 42.06546397, 'imagenet_r': 38.81666946, 'imagenet_sketch': 28.21042061}, {'model': 'efficientnet-b2-autoaug', 'val': 79.77600098, 'imagenet_v2': 68.52999878, 'objectnet': 40.6881279, 'imagenet_r': 42.51666641, 'imagenet_sketch': 30.40342903}, {'model': 'FixResNet50CutMix_v2', 'val': 79.76200104, 'imagenet_v2': 67.12999725, 'objectnet': 29.58301826, 'imagenet_r': 33.8433342, 'imagenet_sketch': 22.10497475}, {'model': 'dpn107', 'val': 79.7460022, 'imagenet_v2': 67.84999847, 'objectnet': 40.77995031, 'imagenet_r': 43.08666611, 'imagenet_sketch': 30.82788086}, {'model': 'FixResNet50CutMix', 'val': 79.74200439, 'imagenet_v2': 67.08999634, 'objectnet': 29.76126175, 'imagenet_r': 33.81000137, 'imagenet_sketch': 22.0401268}, {'model': 'efficientnet-b2-advprop-autoaug', 'val': 79.602005, 'imagenet_v2': 67.88999939, 'objectnet': 40.81775953, 'imagenet_r': 44.41333389, 'imagenet_sketch': 32.18770218}, {'model': 'dpn131', 'val': 79.43200684, 'imagenet_v2': 67.72999573, 'objectnet': 40.21821324, 'imagenet_r': 43.14666748, 'imagenet_sketch': 30.18334198}, {'model': 'dpn92', 'val': 79.40000153, 'imagenet_v2': 67.23999786, 'objectnet': 40.41266069, 'imagenet_r': 41.38666916, 'imagenet_sketch': 30.05757713}, {'model': 'resnext101_32x8d', 'val': 79.31200409, 'imagenet_v2': 67.44999695, 'objectnet': 41.0500162, 'imagenet_r': 42.56999969, 'imagenet_sketch': 29.43465233}, {'model': 'resnet50_ssl', 'val': 79.22800446, 'imagenet_v2': 68.19999695, 'objectnet': 41.45511505, 'imagenet_r': 44.0, 'imagenet_sketch': 29.42286301}, {'model': 'dpn98', 'val': 79.22400665, 'imagenet_v2': 67.76000214, 'objectnet': 40.27762774, 'imagenet_r': 42.56666946, 'imagenet_sketch': 30.01631165}, {'model': 'google_resnet101_jft-300M', 'val': 79.20400238, 'imagenet_v2': 67.41999817, 'objectnet': 38.1549098, 'imagenet_r': 44.96333313, 'imagenet_sketch': 31.76128578}, {'model': 'FixResNet50_v2', 'val': 79.0980072, 'imagenet_v2': 66.90000153, 'objectnet': 34.16873717, 'imagenet_r': 35.74666595, 'imagenet_sketch': 23.05802917}, {'model': 'se_resnext50_32x4d', 'val': 79.07600403, 'imagenet_v2': 67.76999664, 'objectnet': 39.73749595, 'imagenet_r': 38.88333511, 'imagenet_sketch': 25.21763229}, {'model': 'FixResNet50', 'val': 79.0, 'imagenet_v2': 66.72999573, 'objectnet': 34.11472399, 'imagenet_r': 35.69666672, 'imagenet_sketch': 22.96960068}, {'model': 'resnext101_64x4d', 'val': 78.95600128, 'imagenet_v2': 67.15000153, 'objectnet': 37.71200173, 'imagenet_r': 43.28000259, 'imagenet_sketch': 30.31893158}, {'model': 'BiT-M-R50x3-nonfinetuned', 'val': 78.95000458, 'imagenet_v2': 67.47000122, 'objectnet': 28.8430377, 'imagenet_r': 33.70000076, 'imagenet_sketch': 20.96130943}, {'model': 'resnet152_3x_simclrv2_linear_probe_tf_port', 'val': 78.91200256, 'imagenet_v2': 67.29000092, 'objectnet': 39.3918116, 'imagenet_r': 39.01666641, 'imagenet_sketch': 27.69360924}, {'model': 'efficientnet-b2', 'val': 78.89000702, 'imagenet_v2': 66.87999725, 'objectnet': 39.78610781, 'imagenet_r': 39.46333313, 'imagenet_sketch': 26.17068672}, {'model': 'wide_resnet101_2', 'val': 78.84600067, 'imagenet_v2': 66.34999847, 'objectnet': 39.67808145, 'imagenet_r': 40.35666656, 'imagenet_sketch': 28.10627174}, {'model': 'xception', 'val': 78.82000732, 'imagenet_v2': 67.23999786, 'objectnet': 40.73133845, 'imagenet_r': 40.99666595, 'imagenet_sketch': 29.95735931}, {'model': 'efficientnet-b1-autoaug', 'val': 78.71600342, 'imagenet_v2': 66.98999786, 'objectnet': 38.40877174, 'imagenet_r': 41.85666656, 'imagenet_sketch': 30.43093872}, {'model': 'se_resnet152', 'val': 78.65800476, 'imagenet_v2': 67.43999481, 'objectnet': 38.20892298, 'imagenet_r': 40.04666901, 'imagenet_sketch': 26.67177773}, {'model': 'resnet50_cutmix', 'val': 78.6000061, 'imagenet_v2': 65.81999969, 'objectnet': 38.06848871, 'imagenet_r': 34.83333588, 'imagenet_sketch': 23.74580002}, {'model': 'efficientnet-b1-advprop-autoaug', 'val': 78.53600311, 'imagenet_v2': 66.61000061, 'objectnet': 38.20892298, 'imagenet_r': 43.19333267, 'imagenet_sketch': 30.8455677}, {'model': 'wide_resnet50_2', 'val': 78.46800232, 'imagenet_v2': 66.15000153, 'objectnet': 38.14410716, 'imagenet_r': 38.65333557, 'imagenet_sketch': 25.30998993}, {'model': 'se_resnet101', 'val': 78.39600372, 'imagenet_v2': 67.09999847, 'objectnet': 38.45198228, 'imagenet_r': 39.04666901, 'imagenet_sketch': 25.32767487}, {'model': 'resnet152', 'val': 78.31200409, 'imagenet_v2': 66.83999634, 'objectnet': 40.06697634, 'imagenet_r': 41.33666611, 'imagenet_sketch': 28.53465462}, {'model': 'resnet101-tencent-ml-images', 'val': 78.2480011, 'imagenet_v2': 66.52999878, 'objectnet': 39.84012099, 'imagenet_r': None, 'imagenet_sketch': 27.69753838}, {'model': 'resnet50_feature_cutmix', 'val': 78.21200562, 'imagenet_v2': 66.02999878, 'objectnet': 38.07929135, 'imagenet_r': 35.5933342, 'imagenet_sketch': 24.51217461}, {'model': 'resnext101_32x4d', 'val': 78.18800354, 'imagenet_v2': 66.29000092, 'objectnet': 36.32386302, 'imagenet_r': 42.01333618, 'imagenet_sketch': 29.26762199}, {'model': 'resnet101_lpf3', 'val': 78.11600494, 'imagenet_v2': 66.05999756, 'objectnet': 39.31619315, 'imagenet_r': 40.45333481, 'imagenet_sketch': 27.83312798}, {'model': 'efficientnet-b1', 'val': 77.90800476, 'imagenet_v2': 65.83999634, 'objectnet': 38.00907421, 'imagenet_r': 37.14666748, 'imagenet_sketch': 23.91086578}, {'model': 'resnet101_lpf5', 'val': 77.90800476, 'imagenet_v2': 66.18000031, 'objectnet': 37.71200173, 'imagenet_r': 39.03666687, 'imagenet_sketch': 26.94688606}, {'model': 'vit_small_patch16_224', 'val': 77.85600281, 'imagenet_v2': 65.36999512, 'objectnet': 32.942638, 'imagenet_r': 40.11000061, 'imagenet_sketch': 27.08836937}, {'model': 'resnet101_lpf2', 'val': 77.80000305, 'imagenet_v2': 66.25, 'objectnet': 38.7274495, 'imagenet_r': 39.58666611, 'imagenet_sketch': 25.36501122}, {'model': 'se_resnet50', 'val': 77.63600159, 'imagenet_v2': 65.83999634, 'objectnet': 35.88635627, 'imagenet_r': 36.8666687, 'imagenet_sketch': 23.64361763}, {'model': 'resnext50_32x4d', 'val': 77.61800385, 'imagenet_v2': 65.83000183, 'objectnet': 37.35551475, 'imagenet_r': 37.73666763, 'imagenet_sketch': 25.46916008}, {'model': 'resnet50_augmix', 'val': 77.53800201, 'imagenet_v2': 65.11000061, 'objectnet': 36.36707357, 'imagenet_r': 41.04000092, 'imagenet_sketch': 28.47963333}, {'model': 'resnet50_mixup', 'val': 77.47200012, 'imagenet_v2': 65.01999664, 'objectnet': 37.13406071, 'imagenet_r': 39.55666733, 'imagenet_sketch': 26.90954971}, {'model': 'fbresnet152', 'val': 77.38600159, 'imagenet_v2': 65.47999573, 'objectnet': 36.34546829, 'imagenet_r': 41.12333298, 'imagenet_sketch': 28.93159676}, {'model': 'resnet101', 'val': 77.37400055, 'imagenet_v2': 65.44999695, 'objectnet': 38.37636383, 'imagenet_r': 39.30666733, 'imagenet_sketch': 26.96850204}, {'model': 'inceptionv3', 'val': 77.32000732, 'imagenet_v2': 65.66999817, 'objectnet': 38.44658097, 'imagenet_r': 39.41333389, 'imagenet_sketch': 28.03945923}, {'model': 'densenet161', 'val': 77.13800049, 'imagenet_v2': 65.29000092, 'objectnet': 36.96121854, 'imagenet_r': 40.17666626, 'imagenet_sketch': 28.56806183}, {'model': 'efficientnet-b0-advprop-autoaug', 'val': 77.08400726, 'imagenet_v2': 65.30999756, 'objectnet': 33.7852436, 'imagenet_r': 40.35000229, 'imagenet_sketch': 28.34993935}, {'model': 'resnet50_cutout', 'val': 77.06800079, 'imagenet_v2': 64.43000031, 'objectnet': 35.91876418, 'imagenet_r': 34.92666626, 'imagenet_sketch': 23.90693665}, {'model': 'FixResNet50_no_adaptation', 'val': 77.04000092, 'imagenet_v2': 65.02999878, 'objectnet': 27.83299125, 'imagenet_r': 39.24333572, 'imagenet_sketch': 27.00780296}, {'model': 'dpn68b', 'val': 77.03400421, 'imagenet_v2': 64.76000214, 'objectnet': 37.45273847, 'imagenet_r': 38.77666855, 'imagenet_sketch': 27.98836899}, {'model': 'resnet50_lpf5', 'val': 77.02799988, 'imagenet_v2': 64.5, 'objectnet': 36.04839581, 'imagenet_r': 37.34000015, 'imagenet_sketch': 25.16850471}, {'model': 'densenet201', 'val': 76.89600372, 'imagenet_v2': 64.76000214, 'objectnet': 36.48590256, 'imagenet_r': 40.77666855, 'imagenet_sketch': 27.67002869}, {'model': 'efficientnet-b0-autoaug', 'val': 76.84000397, 'imagenet_v2': 64.41999817, 'objectnet': 34.31457276, 'imagenet_r': 38.60666656, 'imagenet_sketch': 26.49099159}, {'model': 'resnet50_lpf3', 'val': 76.81800079, 'imagenet_v2': 64.65999603, 'objectnet': 36.03219185, 'imagenet_r': 37.15999985, 'imagenet_sketch': 24.99754524}, {'model': 'resnet50_lpf2', 'val': 76.79400635, 'imagenet_v2': 64.51000214, 'objectnet': 37.14486335, 'imagenet_r': 36.72333527, 'imagenet_sketch': 24.67330933}, {'model': 'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN', 'val': 76.72000122, 'imagenet_v2': 64.62999725, 'objectnet': 36.5291131, 'imagenet_r': 38.90333557, 'imagenet_sketch': 26.9174099}, {'model': 'resnet50_deepaugment', 'val': 76.66200256, 'imagenet_v2': 65.33000183, 'objectnet': 35.61088906, 'imagenet_r': 42.16666794, 'imagenet_sketch': 29.50146484}, {'model': 'efficientnet-b0', 'val': 76.52799988, 'imagenet_v2': 63.64999771, 'objectnet': 33.29912499, 'imagenet_r': 34.6166687, 'imagenet_sketch': 22.56283379}, {'model': 'resnet50-vtab-rotation', 'val': 76.50200653, 'imagenet_v2': 63.91999817, 'objectnet': 39.32699579, 'imagenet_r': None, 'imagenet_sketch': 23.05410004}, {'model': 'BiT-M-R101x1-nonfinetuned', 'val': 76.2460022, 'imagenet_v2': 64.90000153, 'objectnet': 25.22955601, 'imagenet_r': 30.3900013, 'imagenet_sketch': 18.48336601}, {'model': 'cafferesnet101', 'val': 76.20000458, 'imagenet_v2': 64.13999939, 'objectnet': 32.57534839, 'imagenet_r': 37.72333527, 'imagenet_sketch': 25.21173668}, {'model': 'resnet152-imagenet11k', 'val': 76.18400574, 'imagenet_v2': 66.12999725, 'objectnet': 45.48989954, 'imagenet_r': 38.99666595, 'imagenet_sketch': 27.93727684}, {'model': 'resnet50_aws_baseline', 'val': 76.13800049, 'imagenet_v2': 63.62999725, 'objectnet': 35.39483634, 'imagenet_r': 35.16666794, 'imagenet_sketch': 23.26829147}, {'model': 'resnet50', 'val': 76.13000488, 'imagenet_v2': 63.20999908, 'objectnet': 35.75132332, 'imagenet_r': 36.17333603, 'imagenet_sketch': 24.09165192}, {'model': 'resnet50_imagenet_100percent_batch64_original_images', 'val': 75.98000336, 'imagenet_v2': 63.22999954, 'objectnet': 34.89251377, 'imagenet_r': 35.1166687, 'imagenet_sketch': 22.64929581}, {'model': 'dpn68', 'val': 75.86800385, 'imagenet_v2': 63.1099968, 'objectnet': 34.51982284, 'imagenet_r': 38.97000122, 'imagenet_sketch': 26.09208298}, {'model': 'resnet50_deepaugment_augmix', 'val': 75.82200623, 'imagenet_v2': 63.5399971, 'objectnet': 35.15717835, 'imagenet_r': 46.77000046, 'imagenet_sketch': 32.61805344}, {'model': 'resnet50-randomized_smoothing_noise_0.00', 'val': 75.69000244, 'imagenet_v2': 63.77000046, 'objectnet': 34.47121098, 'imagenet_r': 35.16333389, 'imagenet_sketch': 23.28597641}, {'model': 'densenet169', 'val': 75.6000061, 'imagenet_v2': 63.54999924, 'objectnet': 35.36782975, 'imagenet_r': 39.63333511, 'imagenet_sketch': 27.30256081}, {'model': 'resnet50-vtab', 'val': 75.54199982, 'imagenet_v2': 63.50999832, 'objectnet': 32.34309171, 'imagenet_r': None, 'imagenet_sketch': 25.33553505}, {'model': 'resnet50_with_brightness_aws', 'val': 75.28200531, 'imagenet_v2': 62.70999908, 'objectnet': 33.57999352, 'imagenet_r': 36.64333344, 'imagenet_sketch': 24.79317856}, {'model': 'resnet50_swav', 'val': 75.26800537, 'imagenet_v2': 62.21999741, 'objectnet': 30.03132764, 'imagenet_r': 30.20666695, 'imagenet_sketch': 18.92747116}, {'model': 'resnet50_with_spatter_aws', 'val': 75.21000671, 'imagenet_v2': 62.61999893, 'objectnet': 33.3639408, 'imagenet_r': 34.47666931, 'imagenet_sketch': 21.94187355}, {'model': 'densenet121_lpf3', 'val': 75.14400482, 'imagenet_v2': 62.47999954, 'objectnet': 35.06535595, 'imagenet_r': 36.1166687, 'imagenet_sketch': 23.68684959}, {'model': 'densenet121_lpf5', 'val': 75.02799988, 'imagenet_v2': 62.82999802, 'objectnet': 35.84854705, 'imagenet_r': 37.00333405, 'imagenet_sketch': 24.54557991}, {'model': 'densenet121_lpf2', 'val': 75.02600098, 'imagenet_v2': 63.1099968, 'objectnet': 36.07000108, 'imagenet_r': 36.41666794, 'imagenet_sketch': 24.34907532}, {'model': 'resnet50_with_saturate_aws', 'val': 74.89600372, 'imagenet_v2': 62.25999832, 'objectnet': 33.52057902, 'imagenet_r': 35.89333344, 'imagenet_sketch': 23.95213127}, {'model': 'resnet50_simclrv2_finetuned_100pct_tf_port', 'val': 74.87200165, 'imagenet_v2': 61.29999924, 'objectnet': 32.98584855, 'imagenet_r': 35.25666809, 'imagenet_sketch': 23.49820328}, {'model': 'resnet50_trained_on_SIN_and_IN', 'val': 74.59000397, 'imagenet_v2': 62.75, 'objectnet': 33.30992762, 'imagenet_r': 41.45333481, 'imagenet_sketch': 29.69797134}, {'model': 'resnet34_lpf2', 'val': 74.47800446, 'imagenet_v2': 62.21999741, 'objectnet': 33.07226963, 'imagenet_r': 36.28000259, 'imagenet_sketch': 24.05628014}, {'model': 'densenet121', 'val': 74.43400574, 'imagenet_v2': 62.14999771, 'objectnet': 33.56919088, 'imagenet_r': 37.25, 'imagenet_sketch': 24.24885559}, {'model': 'resnet34_lpf3', 'val': 74.34200287, 'imagenet_v2': 62.15999985, 'objectnet': 32.62936156, 'imagenet_r': 36.1166687, 'imagenet_sketch': 24.53575516}, {'model': 'vgg19_bn', 'val': 74.21800232, 'imagenet_v2': 61.94999695, 'objectnet': 32.36469699, 'imagenet_r': 33.25, 'imagenet_sketch': 21.63139534}, {'model': 'resnet34_lpf5', 'val': 74.19400024, 'imagenet_v2': 62.19999695, 'objectnet': 32.57534839, 'imagenet_r': 35.85000229, 'imagenet_sketch': 23.36654472}, {'model': 'BiT-M-R50x1-nonfinetuned', 'val': 74.12800598, 'imagenet_v2': 62.27999878, 'objectnet': 22.05358107, 'imagenet_r': 27.66333389, 'imagenet_sketch': 15.82463837}, {'model': 'resnet50-vtab-exemplar', 'val': 74.10400391, 'imagenet_v2': 61.93999863, 'objectnet': 31.52209139, 'imagenet_r': None, 'imagenet_sketch': 25.66959572}, {'model': 'nasnetamobile', 'val': 74.08000183, 'imagenet_v2': 61.31999969, 'objectnet': 31.33844658, 'imagenet_r': 33.59000015, 'imagenet_sketch': 23.75169563}, {'model': 'vgg16_bn_lpf5', 'val': 74.03800201, 'imagenet_v2': 61.16999817, 'objectnet': 30.87393324, 'imagenet_r': 31.65999985, 'imagenet_sketch': 20.6724453}, {'model': 'vgg16_bn_lpf2', 'val': 74.00800323, 'imagenet_v2': 61.65999985, 'objectnet': 31.18720968, 'imagenet_r': 31.10000038, 'imagenet_sketch': 20.39930153}, {'model': 'vgg16_bn_lpf3', 'val': 73.9200058, 'imagenet_v2': 61.93000031, 'objectnet': 31.9163876, 'imagenet_r': 31.87000084, 'imagenet_sketch': 20.75497818}, {'model': 'resnet50_with_frost_aws', 'val': 73.78400421, 'imagenet_v2': 61.56999969, 'objectnet': 30.73349897, 'imagenet_r': 35.84666824, 'imagenet_sketch': 23.39995003}, {'model': 'resnet50_with_jpeg_compression_aws', 'val': 73.62600708, 'imagenet_v2': 60.88999939, 'objectnet': 32.18105218, 'imagenet_r': 36.52000046, 'imagenet_sketch': 23.10519028}, {'model': 'bninception', 'val': 73.52400208, 'imagenet_v2': 61.98999786, 'objectnet': 33.29912499, 'imagenet_r': 36.72666931, 'imagenet_sketch': 23.8558445}, {'model': 'mnasnet1_0', 'val': 73.45600128, 'imagenet_v2': 60.43999863, 'objectnet': 32.07302582, 'imagenet_r': 32.26000214, 'imagenet_sketch': 20.12615776}, {'model': 'vgg16_bn', 'val': 73.36000061, 'imagenet_v2': 60.70999908, 'objectnet': 31.4950848, 'imagenet_r': 30.35666656, 'imagenet_sketch': 19.94930267}, {'model': 'resnet34', 'val': 73.31400299, 'imagenet_v2': 60.86999893, 'objectnet': 30.99276223, 'imagenet_r': 36.07333374, 'imagenet_sketch': 23.47265625}, {'model': 'resnet18_swsl', 'val': 73.28600311, 'imagenet_v2': 62.45999908, 'objectnet': 33.4827698, 'imagenet_r': 52.31666946, 'imagenet_sketch': 35.86040497}, {'model': 'resnet50_with_gaussian_noise_aws', 'val': 72.97000122, 'imagenet_v2': 60.56999969, 'objectnet': 30.38781463, 'imagenet_r': 37.44000244, 'imagenet_sketch': 23.30169678}, {'model': 'resnet50_with_gaussian_noise_contrast_motion_blur_jpeg_compression_aws', 'val': 72.72000122, 'imagenet_v2': 60.13999939, 'objectnet': 31.90018364, 'imagenet_r': 36.38999939, 'imagenet_sketch': 23.46676254}, {'model': 'mobilenet_v2_lpf2', 'val': 72.62200165, 'imagenet_v2': 59.41999817, 'objectnet': 30.76590688, 'imagenet_r': 31.92666817, 'imagenet_sketch': 19.89624596}, {'model': 'resnet18_ssl', 'val': 72.6000061, 'imagenet_v2': 61.18999863, 'objectnet': 30.60386734, 'imagenet_r': 35.62666702, 'imagenet_sketch': 21.27768326}, {'model': 'mobilenet_v2_lpf3', 'val': 72.57200623, 'imagenet_v2': 59.68999863, 'objectnet': 30.32840013, 'imagenet_r': 31.50666809, 'imagenet_sketch': 19.1416626}, {'model': 'mobilenet_v2_lpf5', 'val': 72.50800323, 'imagenet_v2': 59.75, 'objectnet': 30.40401858, 'imagenet_r': 32.36999893, 'imagenet_sketch': 20.49951935}, {'model': 'vgg19', 'val': 72.37600708, 'imagenet_v2': 59.68999863, 'objectnet': 28.389327, 'imagenet_r': 28.57999992, 'imagenet_sketch': 17.93708038}, {'model': 'vgg16_lpf5', 'val': 72.33000183, 'imagenet_v2': 59.79999924, 'objectnet': 28.13546505, 'imagenet_r': 29.47000122, 'imagenet_sketch': 18.08642387}, {'model': 'vgg16_lpf3', 'val': 72.18800354, 'imagenet_v2': 59.29999924, 'objectnet': 28.13006374, 'imagenet_r': 29.38666725, 'imagenet_sketch': 18.66415215}, {'model': 'vgg16_lpf2', 'val': 72.16000366, 'imagenet_v2': 59.25999832, 'objectnet': 27.61153722, 'imagenet_r': 28.96666718, 'imagenet_sketch': 18.33795166}, {'model': 'resnet50_with_contrast_aws', 'val': 72.00200653, 'imagenet_v2': 58.86999893, 'objectnet': 30.04213028, 'imagenet_r': 33.86333466, 'imagenet_sketch': 21.05563164}, {'model': 'mobilenet_v2', 'val': 71.87800598, 'imagenet_v2': 59.02000046, 'objectnet': 30.16636059, 'imagenet_r': 30.99333382, 'imagenet_sketch': 19.09843063}, {'model': 'resnet50_with_fog_aws', 'val': 71.75800323, 'imagenet_v2': 58.18999863, 'objectnet': 29.86388679, 'imagenet_r': 33.55666733, 'imagenet_sketch': 20.39340591}, {'model': 'resnet18_lpf3', 'val': 71.68200684, 'imagenet_v2': 58.50999832, 'objectnet': 28.34611645, 'imagenet_r': 33.86000061, 'imagenet_sketch': 21.60781479}, {'model': 'vgg16', 'val': 71.59200287, 'imagenet_v2': 58.50999832, 'objectnet': 27.8167873, 'imagenet_r': 27.87333488, 'imagenet_sketch': 17.53227615}, {'model': 'vgg13_bn', 'val': 71.58600616, 'imagenet_v2': 58.84000015, 'objectnet': 28.23809009, 'imagenet_r': 28.87333488, 'imagenet_sketch': 17.8073864}, {'model': 'resnet18_lpf2', 'val': 71.39000702, 'imagenet_v2': 58.52999878, 'objectnet': 29.323755, 'imagenet_r': 32.99333572, 'imagenet_sketch': 20.60563278}, {'model': 'resnet18_lpf5', 'val': 71.39000702, 'imagenet_v2': 58.1099968, 'objectnet': 28.30830723, 'imagenet_r': 32.45999908, 'imagenet_sketch': 20.48576355}, {'model': 'resnet18-rotation-standard_40', 'val': 71.28400421, 'imagenet_v2': 58.44999695, 'objectnet': 29.21032732, 'imagenet_r': 34.28666687, 'imagenet_sketch': 22.07746315}, {'model': 'resnet50_simclrv2_linear_probe_tf_port', 'val': 70.84000397, 'imagenet_v2': 57.93000031, 'objectnet': 25.53202982, 'imagenet_r': 29.86666679, 'imagenet_sketch': 18.62878036}, {'model': 'resnet50_imagenet_subsample_1_of_2_batch64_original_images', 'val': 70.52200317, 'imagenet_v2': 57.95999908, 'objectnet': 28.17327428, 'imagenet_r': 30.14333344, 'imagenet_sketch': 18.16502571}, {'model': 'vgg11_bn', 'val': 70.37000275, 'imagenet_v2': 57.40999985, 'objectnet': 26.42324727, 'imagenet_r': 28.42000008, 'imagenet_sketch': 17.40454865}, {'model': 'resnet50-randomized_smoothing_noise_0.25', 'val': 70.28600311, 'imagenet_v2': 57.68999863, 'objectnet': 27.30366209, 'imagenet_r': 37.02666855, 'imagenet_sketch': 22.39383888}, {'model': 'vgg13', 'val': 69.9280014, 'imagenet_v2': 56.75999832, 'objectnet': 24.57059523, 'imagenet_r': 26.2733345, 'imagenet_sketch': 16.09778214}, {'model': 'googlenet/inceptionv1', 'val': 69.77799988, 'imagenet_v2': 57.87999725, 'objectnet': 29.49659717, 'imagenet_r': 36.15999985, 'imagenet_sketch': 23.05606461}, {'model': 'resnet18', 'val': 69.75800323, 'imagenet_v2': 57.30999756, 'objectnet': 26.73652371, 'imagenet_r': 33.05666733, 'imagenet_sketch': 20.22834015}, {'model': 'shufflenet_v2_x1_0', 'val': 69.36199951, 'imagenet_v2': 55.94999695, 'objectnet': 26.9741817, 'imagenet_r': 30.15333366, 'imagenet_sketch': 17.88009262}, {'model': 'resnet18-rotation-worst10_30', 'val': 69.13400269, 'imagenet_v2': 56.89999771, 'objectnet': 28.72420871, 'imagenet_r': 29.13666725, 'imagenet_sketch': 16.30804443}, {'model': 'vgg11', 'val': 69.02000427, 'imagenet_v2': 55.72999954, 'objectnet': 24.10068057, 'imagenet_r': 26.53000069, 'imagenet_sketch': 15.72049046}, {'model': 'resnet18-rotation-random_30', 'val': 68.87800598, 'imagenet_v2': 56.09999847, 'objectnet': 28.47574808, 'imagenet_r': 28.65666771, 'imagenet_sketch': 16.76590347}, {'model': 'resnet18-rotation-worst10_40', 'val': 68.60400391, 'imagenet_v2': 55.56999969, 'objectnet': 28.29750459, 'imagenet_r': 28.09000015, 'imagenet_sketch': 15.42180061}, {'model': 'resnet50_with_pixelate_aws', 'val': 68.5, 'imagenet_v2': 56.68000031, 'objectnet': 29.70184725, 'imagenet_r': 27.84000015, 'imagenet_sketch': 13.98534107}, {'model': 'resnet18-rotation-random_40', 'val': 68.3500061, 'imagenet_v2': 56.07999802, 'objectnet': 28.15707033, 'imagenet_r': 27.88000107, 'imagenet_sketch': 16.1744194}, {'model': 'facebook_adv_trained_resnext101_denoiseAll', 'val': 68.33200073, 'imagenet_v2': 55.21999741, 'objectnet': 26.53667495, 'imagenet_r': 48.99333572, 'imagenet_sketch': 31.28181076}, {'model': 'resnet50-smoothing_adversarial_DNN_2steps_eps_512_noise_0.25', 'val': 67.88600159, 'imagenet_v2': 54.50999832, 'objectnet': 25.12152965, 'imagenet_r': 37.84000015, 'imagenet_sketch': 21.81807518}, {'model': 'resnet50_simsiam', 'val': 67.78200531, 'imagenet_v2': 54.57999802, 'objectnet': 22.33444961, 'imagenet_r': 28.44666672, 'imagenet_sketch': 18.34777641}, {'model': 'mnasnet0_5', 'val': 67.602005, 'imagenet_v2': 54.19999695, 'objectnet': 24.10068057, 'imagenet_r': 25.42000008, 'imagenet_sketch': 14.32726192}, {'model': 'resnet50_with_motion_blur_aws', 'val': 67.45600128, 'imagenet_v2': 55.90999985, 'objectnet': 27.72496489, 'imagenet_r': 28.53000069, 'imagenet_sketch': 13.05783272}, {'model': 'resnet18-rotation-nocrop_40', 'val': 65.37000275, 'imagenet_v2': 53.5, 'objectnet': 25.69947067, 'imagenet_r': 34.77000046, 'imagenet_sketch': 21.63728905}, {'model': 'facebook_adv_trained_resnet152_denoise', 'val': 65.31999969, 'imagenet_v2': 52.59999847, 'objectnet': 24.96489143, 'imagenet_r': 48.57666779, 'imagenet_sketch': 30.09884453}, {'model': 'bninception-imagenet21k', 'val': 65.24000549, 'imagenet_v2': 53.91999817, 'objectnet': 29.31295236, 'imagenet_r': 34.22333527, 'imagenet_sketch': 21.9320488}, {'model': 'resnet50-randomized_smoothing_noise_0.50', 'val': 64.23800659, 'imagenet_v2': 51.11999893, 'objectnet': 22.76115372, 'imagenet_r': 35.57666779, 'imagenet_sketch': 19.75672722}, {'model': 'resnet50_with_greyscale_aws', 'val': 63.33400345, 'imagenet_v2': 50.71999741, 'objectnet': 23.37690396, 'imagenet_r': 29.09000015, 'imagenet_sketch': 21.90650368}, {'model': 'vit_b_32_clip_zeroshot', 'val': 63.30200195, 'imagenet_v2': 55.77000046, 'objectnet': 43.96132656, 'imagenet_r': 69.25666809, 'imagenet_sketch': 42.37261581}, {'model': 'resnet50_imagenet_subsample_1_of_4_batch64_original_images', 'val': 63.23200226, 'imagenet_v2': 50.52999878, 'objectnet': 21.21097548, 'imagenet_r': 23.8433342, 'imagenet_sketch': 12.10281277}, {'model': 'resnet50_linf_eps4_robust', 'val': 62.4240036, 'imagenet_v2': 49.86000061, 'objectnet': 21.32980447, 'imagenet_r': 40.05666733, 'imagenet_sketch': 21.6137085}, {'model': 'facebook_adv_trained_resnet152_baseline', 'val': 62.34000397, 'imagenet_v2': 49.82999802, 'objectnet': 22.43707465, 'imagenet_r': 46.91333389, 'imagenet_sketch': 28.92963219}, {'model': 'resnet50-smoothing_adversarial_DNN_2steps_eps_512_noise_0.50', 'val': 62.2120018, 'imagenet_v2': 49.05999756, 'objectnet': 21.08134385, 'imagenet_r': 35.47333527, 'imagenet_sketch': 18.91961098}, {'model': 'resnet50-vtab-semi-exemplar', 'val': 61.62200165, 'imagenet_v2': 49.52999878, 'objectnet': 18.59133629, 'imagenet_r': None, 'imagenet_sketch': 20.54078484}, {'model': 'resnet50_with_zoom_blur_aws', 'val': 61.25200272, 'imagenet_v2': 49.40999985, 'objectnet': 24.7434374, 'imagenet_r': 23.0233345, 'imagenet_sketch': 9.935349464}, {'model': 'resnet50-vtab-semi-rotation', 'val': 60.9240036, 'imagenet_v2': 48.80999756, 'objectnet': 24.06827266, 'imagenet_r': None, 'imagenet_sketch': 12.7611084}, {'model': 'shufflenet_v2_x0_5', 'val': 60.55200195, 'imagenet_v2': 47.23999786, 'objectnet': 18.42389543, 'imagenet_r': 25.03000069, 'imagenet_sketch': 12.2462616}, {'model': 'resnet50_adv-train-free', 'val': 60.49200439, 'imagenet_v2': 47.41999817, 'objectnet': 20.48719888, 'imagenet_r': 36.74000168, 'imagenet_sketch': 20.04952049}, {'model': 'resnet50-smoothing_adversarial_PGD_1step_eps_512_noise_0.25', 'val': 60.48600388, 'imagenet_v2': 46.96999741, 'objectnet': 18.99103381, 'imagenet_r': 35.15999985, 'imagenet_sketch': 18.03533173}, {'model': 'resnet50_trained_on_SIN', 'val': 60.18400192, 'imagenet_v2': 48.55999756, 'objectnet': 21.34060711, 'imagenet_r': 40.08333588, 'imagenet_sketch': 30.05561256}, {'model': 'resnet50_clip_zeroshot', 'val': 59.80800247, 'imagenet_v2': 52.52999878, 'objectnet': 40.28302906, 'imagenet_r': 60.4600029, 'imagenet_sketch': 35.3769989}, {'model': 'squeezenet1_1', 'val': 58.1780014, 'imagenet_v2': 45.36999893, 'objectnet': 16.94933564, 'imagenet_r': 22.81000137, 'imagenet_sketch': 11.58010578}, {'model': 'squeezenet1_0', 'val': 58.09200287, 'imagenet_v2': 44.90999985, 'objectnet': 16.38759857, 'imagenet_r': 22.57999992, 'imagenet_sketch': 10.87268448}, {'model': 'resnet50_l2_eps3_robust', 'val': 57.90000153, 'imagenet_v2': 45.31999969, 'objectnet': 17.40844766, 'imagenet_r': 36.19000244, 'imagenet_sketch': 19.39515495}, {'model': 'alexnet_lpf2', 'val': 57.23200226, 'imagenet_v2': 43.98999786, 'objectnet': 14.27028195, 'imagenet_r': 22.08333397, 'imagenet_sketch': 10.46591663}, {'model': 'alexnet_lpf3', 'val': 56.89400101, 'imagenet_v2': 43.64999771, 'objectnet': 14.18926218, 'imagenet_r': 21.70000076, 'imagenet_sketch': 10.53862381}, {'model': 'alexnet_lpf5', 'val': 56.57800293, 'imagenet_v2': 43.34999847, 'objectnet': 14.04882791, 'imagenet_r': 21.86000061, 'imagenet_sketch': 10.55041409}, {'model': 'alexnet', 'val': 56.52200317, 'imagenet_v2': 43.43000031, 'objectnet': 13.85978179, 'imagenet_r': 21.7733345, 'imagenet_sketch': 10.71940899}]
slope, icpt = (1.107560413, -19.85217312)

def imagenet(path, k, size=(5.2, 2.15)):
    fig, ax = plt.subplots(figsize=size)
    ax.plot([40, 95], [40, 95], '--', color=C['neutral'], lw=1.5)
    ax.text(58, 63.5, 'equal accuracy', color=C['neutral'], fontsize=11, rotation=33)
    ax.scatter([r['val'] for r in IV2], [r['imagenet_v2'] for r in IV2], s=16, color=C['memory'], alpha=0.75, zorder=3)
    if k >= 2:
        xx = np.array([55, 90])
        ax.plot(xx, slope * xx + icpt, color=C['danger'], lw=2.5, zorder=4)
        lab = {'alexnet': ('AlexNet', (25, -6)), 'resnet50': ('ResNet-50', (22, -14)), 'efficientnet-l2-noisystudent': ('EfficientNet-L2', (-8, 10))}
        for r in IV2:
            if r['model'] in lab:
                t, o = lab[r['model']]
                ax.plot([r['val']], [r['imagenet_v2']], 'o', ms=10, mfc='none', mec=C['text'], mew=1.5, zorder=5)
                ax.annotate(t, (r['val'], r['imagenet_v2']), xytext=o, textcoords='offset points', ha='left' if o[0] > 0 else 'right', fontsize=12, fontweight='bold')
        ax.text(91.5, 41, '212 models all lose\n7 to 14 points, median 12', ha='right', va='bottom', fontsize=12, fontweight='bold', color=C['danger'])
    ax.set_xlim(54, 92)
    ax.set_ylim(38, 90)
    ax.set_xlabel('ImageNet top-1 accuracy (%)')
    ax.set_ylabel('ImageNet-V2 top-1 (%)')
    ax.grid(color=C['grid'], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, path)
for k in (1, 2):
    imagenet(out / f'imagenet_step{k}.svg', k)
SE, BASE, OBS = (0.009486832981, 0.5, 0.55)

def meantest(path):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    xs = np.linspace(0.465, 0.535, 400)
    ys = np.exp(-0.5 * ((xs - BASE) / SE) ** 2)
    ax.fill_between(xs, 0, ys, color=C['memory_l'])
    ax.plot(xs, ys, color=C['memory'], lw=2.5)
    ax.axvline(OBS, color=C['danger'], lw=3)
    ax.text(0.548, 0.75, 'observed\n0.55', ha='right', fontsize=12, fontweight='bold', color=C['danger'])
    ax.annotate('', xy=(OBS, 1.1), xytext=(BASE, 1.1), arrowprops=dict(arrowstyle='<->', color=C['text'], lw=1.8))
    ax.text((BASE + OBS) / 2, 1.16, '5.3 standard\nerrors', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.466, 0.95, 'baseline\n0.50', fontsize=11.5, color=C['memory'], va='top')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_ylim(0, 1.5)
    ax.set_xlim(0.46, 0.56)
    ax.set_xticks([0.47, 0.5, 0.53, 0.56])
    ax.set_xlabel('Mean over 1,000 requests')
    sp.save(fig, path)
meantest(out / 'meantest.svg')
sp.hbars(out / 'psi_country.svg', [('Other', 0.04702293319, 'danger', '0.047 (72%)'), ('USA', 0.0118353431, 'memory', '0.012'), ('France', 0.003646431136, 'memory', '0.004'), ('UK', 0.002107210313, 'memory', '0.002'), ('Germany', 0.0006899287149, 'memory', '0.001')], '', log=False, size=(3.3, 2.2), xlabel='PSI contribution (total 0.065)')

def grid(path):
    b = []
    x0, y0, w, h = (120, 10, 375, 155)
    cells = [(0, 0, 'Probe deeper', 'error up, inputs flat:', 'concept drift or a fault', V['red_l'], V['red']), (1, 0, 'Isolate slices', 'error and drift together:', 'test the update', V['amber_l'], V['amber']), (0, 1, 'Keep serving', 'nothing moved', '', V['frame_l'], V['frame']), (1, 1, 'Monitor', 'inputs moved, error flat:', 'find the source', V['blue_l'], V['blue'])]
    for cx, cy, t, s1, s2, f, s in cells:
        x, y = (x0 + cx * (w + 10), y0 + cy * (h + 10))
        b.append(rect(x, y, w, h, f, s, 2.5, 8))
        b.append(text(x + w / 2, y + 50, t, s if s != V['frame'] else V['gray'], 27, 700))
        b.append(text(x + w / 2, y + 92, s1, V['ink'], 22))
        b.append(text(x + w / 2, y + 122, s2, V['ink'], 22))
    b.append(arrow(x0, 352, x0 + 2 * w + 10, 352, 'gray', 3))
    b.append(text(x0 + w + 5, 385, 'Input drift statistic (PSI, KS)', V['ink'], 22, 700))
    b.append(arrow(95, y0 + 2 * h + 10, 95, y0, 'gray', 3))
    b.append(f'''<text x="70" y="{y0 + h + 5}" font-size="22" font-weight="700" fill="{V['ink']}" text-anchor="middle" transform="rotate(-90 70 {y0 + h + 5})">Measured error</text>''')
    svg(path, b, 900, 395)
grid(out / 'drift_grid.svg')

def calib(path):
    rows = [('CIFAR-100, LeNet 5', 'ece_lenet', 'ece_lenet_ts'), ('CIFAR-100, ResNet 110', 'ece_r110', 'ece_r110_ts'), ('ImageNet, ResNet 152', 'ece_r152', 'ece_r152_ts')]
    fig, ax = plt.subplots(figsize=(4.8, 1.75))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, uk, tk) in zip(y, rows):
        u, t = ({'CIFAR-100, LeNet 5': 4.85, 'CIFAR-100, ResNet 110': 16.53, 'ImageNet, ResNet 152': 5.48}[lab], {'CIFAR-100, LeNet 5': 2.02, 'CIFAR-100, ResNet 110': 1.26, 'ImageNet, ResNet 152': 1.86}[lab])
        ax.barh(yi + 0.19, u, height=0.36, color=C['danger'])
        ax.barh(yi - 0.19, t, height=0.36, color=C['data'])
        ax.text(u + 0.3, yi + 0.19, {'CIFAR-100, LeNet 5': '4.85', 'CIFAR-100, ResNet 110': '16.53', 'ImageNet, ResNet 152': '5.48'}[lab], va='center', fontsize=12, fontweight='bold', color=C['danger'])
        ax.text(t + 0.3, yi - 0.19, {'CIFAR-100, LeNet 5': '2.02', 'CIFAR-100, ResNet 110': '1.26', 'ImageNet, ResNet 152': '1.86'}[lab], va='center', fontsize=12, fontweight='bold', color=C['data'])
        ax.text(-0.4, yi, lab, ha='right', va='center', fontsize=12, fontweight='bold')
    ax.text(19.5, 0.25, 'as trained', color=C['danger'], fontsize=12, fontweight='bold', ha='right')
    ax.text(19.5, -0.25, 'temperature scaled', color=C['data'], fontsize=12, fontweight='bold', ha='right')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 19.5)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel('Expected calibration error (percentage points)')
    sp.save(fig, path)
calib(out / 'calib.svg')

def riskcov(path, k):
    cov = np.linspace(0.3, 1.0, 200)
    risk = np.array([0.0120729, 0.01207818135, 0.01208377777, 0.01208970412, 0.01209597578, 0.01210260866, 0.01210961922, 0.01211702446, 0.01212484197, 0.01213308992, 0.01214178704, 0.01215095271, 0.0121606069, 0.01217077023, 0.01218146395, 0.01219270998, 0.01220453091, 0.01221695001, 0.01222999126, 0.01224367934, 0.01225803966, 0.0122730984, 0.01228888245, 0.01230541951, 0.01232273804, 0.0123408673, 0.01235983739, 0.01237967921, 0.01240042452, 0.01242210594, 0.01244475696, 0.01246841196, 0.01249310622, 0.01251887597, 0.01254575836, 0.01257379147, 0.0126030144, 0.0126334672, 0.01266519093, 0.01269822767, 0.01273262056, 0.01276841376, 0.01280565251, 0.01284438316, 0.01288465313, 0.01292651098, 0.01297000642, 0.01301519031, 0.01306211466, 0.01311083272, 0.01316139891, 0.01321386891, 0.01326829964, 0.01332474926, 0.01338327725, 0.01344394439, 0.01350681277, 0.01357194582, 0.01363940834, 0.01370926651, 0.01378158792, 0.01385644156, 0.01393389786, 0.01401402874, 0.01409690755, 0.01418260919, 0.01427121005, 0.01436278807, 0.01445742274, 0.01455519515, 0.01465618797, 0.01476048552, 0.01486817374, 0.01497934025, 0.01509407435, 0.01521246706, 0.01533461111, 0.01546060101, 0.01559053301, 0.01572450519, 0.01586261742, 0.01600497144, 0.01615167082, 0.01630282105, 0.01645852949, 0.01661890547, 0.01678406025, 0.01695410707, 0.01712916118, 0.01730933985, 0.0174947624, 0.01768555022, 0.01788182678, 0.0180837177, 0.01829135072, 0.01850485576, 0.01872436492, 0.01895001253, 0.01918193516, 0.01942027163, 0.01966516306, 0.0199167529, 0.02017518693, 0.02044061328, 0.02071318251, 0.02099304757, 0.02128036385, 0.02157528923, 0.02187798406, 0.02218861125, 0.02250733621, 0.02283432697, 0.02316975412, 0.02351379091, 0.02386661322, 0.02422839963, 0.02459933143, 0.02497959262, 0.02536936998, 0.0257688531, 0.02617823435, 0.02659770897, 0.02702747507, 0.02746773366, 0.02791868869, 0.02838054705, 0.02885351863, 0.02933781634, 0.02983365611, 0.03034125698, 0.03086084107, 0.03139263362, 0.03193686306, 0.03249376099, 0.03306356224, 0.03364650489, 0.03424283028, 0.03485278309, 0.03547661131, 0.03611456632, 0.0367669029, 0.03743387925, 0.03811575703, 0.0388128014, 0.03952528104, 0.0402534682, 0.04099763868, 0.04175807192, 0.04253505102, 0.04332886273, 0.04413979753, 0.04496814965, 0.04581421707, 0.0466783016, 0.04756070888, 0.04846174843, 0.04938173367, 0.05032098195, 0.05127981461, 0.05225855696, 0.0532575384, 0.05427709234, 0.05531755633, 0.05637927205, 0.05746258534, 0.05856784626, 0.05969540911, 0.06084563243, 0.0620188791, 0.06321551633, 0.0644359157, 0.06568045321, 0.06694950929, 0.06824346887, 0.06956272138, 0.07090766079, 0.07227868569, 0.07367619924, 0.07510060931, 0.07655232841, 0.07803177383, 0.07953936757, 0.08107553647, 0.0826407122, 0.08423533127, 0.08585983514, 0.08751467019, 0.0892002878, 0.09091714434, 0.09266570128, 0.09444642514, 0.0962597876, 0.09810626549, 0.09998634086, 0.101900501, 0.1038492385, 0.1058330511, 0.1078524423, 0.1099079206, 0.112])
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    ax.plot(cov * 100, risk * 100, color=C['memory'], lw=3)
    ax.set_xlim(30, 102)
    ax.set_ylim(0, 12.8)
    ax.set_xlabel('Coverage: % answered')
    ax.set_ylabel('Error on answered (%)')
    ax.grid(color=C['grid'], lw=0.5)
    ax.plot([100], [11.2], 'o', ms=10, color=C['neutral'])
    ax.annotate('answer all', (100, 11.2), xytext=(-8, -3), textcoords='offset points', ha='right', fontsize=11.5, fontweight='bold', color=C['neutral'])
    if k >= 2:
        x, y = (90.0, 6.5)
        ax.plot([x], [y], 'o', ms=11, color=C['danger'], zorder=5)
        ax.annotate('90% answered,\n6.5% error', (x, y), xytext=(-14, 4), textcoords='offset points', ha='right', fontsize=11.5, fontweight='bold', color=C['danger'])
    ax.text(32, 12.3, 'illustrative', fontsize=10.5, color=C['neutral'], va='top')
    sp.save(fig, path)
for k in (1, 2):
    riskcov(out / f'riskcov_step{k}.svg', k)

def conformal(path):
    n, rank = (1000, 901)
    rng = np.random.default_rng(3)
    scores = np.sort(rng.gamma(2.0, 0.15, n))
    q = scores[rank - 1]
    fig, ax = plt.subplots(figsize=(4.8, 1.8))
    ax.plot(np.arange(1, n + 1), scores, color=C['memory'], lw=3)
    ax.axvline(rank, color=C['danger'], lw=2.5, ls='--')
    ax.axhline(q, color=C['danger'], lw=1.5, ls=':')
    ax.text(rank - 20, scores[-1] * 0.95, 'rank 901\nsets the cutoff', ha='right', fontsize=12, fontweight='bold', color=C['danger'], va='top')
    ax.text(30, q + 0.04, 'keep labels scoring below', fontsize=12, color=C['danger'])
    ax.set_xlabel('Calibration examples sorted by score (n = 1,000)')
    ax.set_ylabel('Score')
    ax.set_yticks([])
    ax.text(30, scores[-1] * 0.98, 'illustrative scores', fontsize=10.5, color=C['neutral'], va='top')
    sp.save(fig, path)
conformal(out / 'conformal.svg')

def zou(path, k):
    rows = ['Request only', '+ “Sure, here’s”', '+ GCG suffix', '+ GCG ensemble']
    keys = ('only', 'sure', 'gcg', 'ens')
    g35 = [_NUMS[f'gpt35_{k_}'] for k_ in keys]
    g4 = [_NUMS[f'gpt4_{k_}'] for k_ in keys]
    fig, ax = plt.subplots(figsize=(5.3, 1.95))
    y = np.arange(4)[::-1]
    for i, yi in enumerate(y):
        a = 1.0 if i < 2 or k >= 2 else 0.0
        ax.barh(yi + 0.2, g35[i], height=0.38, color=C['danger'], alpha=a)
        ax.barh(yi - 0.2, g4[i], height=0.38, color=C['compute'], alpha=a)
        ax.text(g35[i] + 1, yi + 0.2, {3.0: '1.8', 2.0: '5.7', 1.0: '47.4', 0.0: '86.6'}[yi], va='center', fontsize=11.5, fontweight='bold', color=C['danger'], alpha=a)
        ax.text(g4[i] + 1, yi - 0.2, {3.0: '8.0', 2.0: '13.1', 1.0: '29.1', 0.0: '46.9'}[yi], va='center', fontsize=11.5, fontweight='bold', color=C['compute'], alpha=a)
        ax.text(-1.5, yi, rows[i], ha='right', va='center', fontsize=12, fontweight='bold')
    ax.text(99, 3.3, 'GPT-3.5', color=C['danger'], fontsize=12.5, fontweight='bold', ha='right')
    ax.text(99, 2.65, 'GPT-4', color=C['compute'], fontsize=12.5, fontweight='bold', ha='right')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Attack success, % of 388 harmful requests')
    sp.save(fig, path)
for k in (1, 2):
    zou(out / f'zou_step{k}.svg', k)

def access(path, k):
    rows = [('White-box', 'weights, gradients', 'one backward pass', 'adaptive white-box', V['red'], V['red_l']), ('Transfer', 'a surrogate model', 'craft offline, submit', "other models' attacks", V['amber'], V['amber_l']), ('Query', 'returned labels', 'thousands of queries', 'query budgets', V['violet'], V['violet_l']), ('Physical', 'the scene itself', 'survive the camera', 'drive-by tests', V['blue'], V['blue_l'])]
    b = [text(20, 26, 'Attacker has', V['gray'], 20, 700, 'start'), text(495, 26, 'Attack cost', V['gray'], 20, 700), text(770, 26, 'Test it demands', V['gray'], 20, 700)]
    for i, (n, has, cost, ev, s, f) in enumerate(rows):
        y = 40 + i * 88
        op = 1.0 if k >= i + 1 else 0.0
        b.append(rect(10, y, 880, 78, f, s, 2.5, 8, op))
        b.append(text(25, y + 34, n, s, 25, 700, 'start', op))
        b.append(text(25, y + 64, has, V['ink'], 20, 400, 'start', op))
        b.append(text(495, y + 47, cost, V['ink'], 21, 400, 'middle', op))
        b.append(text(770, y + 47, ev, V['ink'], 21, 700, 'middle', op))
    svg(path, b, 900, 395)
for k in (1, 2, 3, 4):
    access(out / f'access_step{k}.svg', k)

def pgd(path, k):
    rows = [('Standard step', [1], ['update']), ('FGSM, clean + adv.', [1, 2], ['attack', f'{2}× batch']), ('PGD-7 training', [7, 1], [f'{7} attack steps', 'update'])]
    fig, ax = plt.subplots(figsize=(5.4, 1.55))
    for i, (lab, parts, names) in enumerate(rows):
        y = 2 - i
        shown = i < 2 or k >= 2
        left = 0
        for p_, nm in zip(parts, names):
            c = C['danger'] if 'attack' in nm else C['memory']
            ax.barh(y, p_, left=left, height=0.62, color=c, edgecolor='white', lw=1.5, alpha=1.0 if shown else 0.0)
            if p_ >= 2:
                ax.text(left + p_ / 2, y, nm, ha='center', va='center', color='white', fontsize=11, fontweight='bold', alpha=1.0 if shown else 0.0)
            left += p_
        total = sum(parts)
        ax.text(total + 0.15, y, f'{total} passes' if total > 1 else '1 pass', va='center', fontsize=12, fontweight='bold', color=C['text'], alpha=1.0 if shown else 0.0)
        ax.text(-0.15, y, lab, ha='right', va='center', fontsize=12, fontweight='bold')
        if i == 2 and k == 1:
            ax.text(0.2, y, '???', va='center', fontsize=14, fontweight='bold', color=C['danger'])
    ax.set_xlim(0, 9.6)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Forward and backward passes per batch')
    sp.save(fig, path)
for k in (1, 2):
    pgd(out / f'pgd_step{k}.svg', k)
RB = [{'model_id': 'Addepalli2021Towards_RN18', 'venue': 'ECCV 2022', 'year': 2022.0, 'architecture': 'ResNet-18', 'additional_data': False, 'clean_acc': 80.24, 'autoattack_acc': 51.06, 'unreliable': False}, {'model_id': 'Addepalli2021Towards_WRN34', 'venue': 'ECCV 2022', 'year': 2022.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 85.32, 'autoattack_acc': 58.04, 'unreliable': False}, {'model_id': 'Addepalli2022Efficient_RN18', 'venue': 'NeurIPS 2022', 'year': 2022.0, 'architecture': 'ResNet-18', 'additional_data': False, 'clean_acc': 85.71, 'autoattack_acc': 52.48, 'unreliable': False}, {'model_id': 'Addepalli2022Efficient_WRN_34_10', 'venue': 'NeurIPS 2022', 'year': 2022.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 88.71, 'autoattack_acc': 57.81, 'unreliable': False}, {'model_id': 'Alayrac2019Labels', 'venue': 'NeurIPS 2019', 'year': 2019.0, 'architecture': 'WideResNet-28-10', 'additional_data': True, 'clean_acc': 86.46, 'autoattack_acc': 56.03, 'unreliable': None}, {'model_id': 'Alfarra2020ClusTR', 'venue': 'arXiv, Jun 2020', 'year': 2020.0, 'architecture': 'WideResNet-28-10', 'additional_data': False, 'clean_acc': 91.03, 'autoattack_acc': 0.0, 'unreliable': None}, {'model_id': 'Amini2024MeanSparse_Ra_WRN_70_16', 'venue': 'arXiv, Jun 2024', 'year': 2024.0, 'architecture': 'MeanSparse RaWideResNet-70-16', 'additional_data': True, 'clean_acc': 93.24, 'autoattack_acc': 72.08, 'unreliable': False}, {'model_id': 'Amini2024MeanSparse_S-WRN-94-16', 'venue': 'arXiv, Jun 2024', 'year': 2024.0, 'architecture': 'MeanSparse WideResNet-94-16', 'additional_data': True, 'clean_acc': 93.6, 'autoattack_acc': 75.28, 'unreliable': False}, {'model_id': 'Andriushchenko2020Understanding', 'venue': 'NeurIPS 2020', 'year': 2020.0, 'architecture': 'PreActResNet-18', 'additional_data': False, 'clean_acc': 79.84, 'autoattack_acc': 43.93, 'unreliable': None}, {'model_id': 'Bai2023Improving_edm', 'venue': 'SIMODS 2024', 'year': 2024.0, 'architecture': 'ResNet-152 + WideResNet-70-16 + mixing network', 'additional_data': True, 'clean_acc': 95.23, 'autoattack_acc': 68.06, 'unreliable': False}, {'model_id': 'Bai2024MixedNUTS', 'venue': 'TMLR, Aug 2024', 'year': 2024.0, 'architecture': 'ResNet-152 + WideResNet-70-16', 'additional_data': True, 'clean_acc': 95.19, 'autoattack_acc': 70.08, 'unreliable': False}, {'model_id': 'Bartoldson2024Adversarial_WRN-82-8', 'venue': 'ICML 2024', 'year': 2024.0, 'architecture': 'WideResNet-82-8', 'additional_data': False, 'clean_acc': 93.11, 'autoattack_acc': 71.59, 'unreliable': False}, {'model_id': 'Bartoldson2024Adversarial_WRN-94-16', 'venue': 'ICML 2024', 'year': 2024.0, 'architecture': 'WideResNet-94-16', 'additional_data': False, 'clean_acc': 93.68, 'autoattack_acc': 73.71, 'unreliable': False}, {'model_id': 'Carmon2019Unlabeled', 'venue': 'NeurIPS 2019', 'year': 2019.0, 'architecture': 'WideResNet-28-10', 'additional_data': True, 'clean_acc': 89.69, 'autoattack_acc': 59.53, 'unreliable': False}, {'model_id': 'Chan2020Jacobian', 'venue': 'ICLR 2020', 'year': 2020.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 93.79, 'autoattack_acc': 0.26, 'unreliable': None}, {'model_id': 'Chen2020Adversarial', 'venue': 'CVPR 2020', 'year': 2020.0, 'architecture': 'ResNet-50', 'additional_data': False, 'clean_acc': 86.04, 'autoattack_acc': 51.56, 'unreliable': None}, {'model_id': 'Chen2020Efficient', 'venue': 'arXiv, Oct 2020', 'year': 2020.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 85.32, 'autoattack_acc': 51.12, 'unreliable': None}, {'model_id': 'Chen2021LTD_WRN34_10', 'venue': 'arXiv, Nov 2021', 'year': 2021.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 85.21, 'autoattack_acc': 56.94, 'unreliable': False}, {'model_id': 'Chen2021LTD_WRN34_20', 'venue': 'arXiv, Nov 2021', 'year': 2021.0, 'architecture': 'WideResNet-34-20', 'additional_data': False, 'clean_acc': 86.03, 'autoattack_acc': 57.71, 'unreliable': False}, {'model_id': 'Chen2024Data_WRN_34_10', 'venue': 'Pattern Recognition 2024', 'year': 2024.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 86.54, 'autoattack_acc': 57.3, 'unreliable': False}, {'model_id': 'Chen2024Data_WRN_34_20', 'venue': 'Pattern Recognition 2024', 'year': 2024.0, 'architecture': 'WideResNet-34-20', 'additional_data': False, 'clean_acc': 86.1, 'autoattack_acc': 58.09, 'unreliable': False}, {'model_id': 'Cui2023Decoupled_WRN-28-10', 'venue': 'NeurIPS 2024', 'year': 2024.0, 'architecture': 'WideResNet-28-10', 'additional_data': False, 'clean_acc': 92.16, 'autoattack_acc': 67.73, 'unreliable': False}, {'model_id': 'Cui2023Decoupled_WRN-34-10', 'venue': 'NeurIPS 2024', 'year': 2024.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 85.31, 'autoattack_acc': 57.09, 'unreliable': False}, {'model_id': 'Dai2021Parameterizing', 'venue': 'arXiv, Oct 2021', 'year': 2021.0, 'architecture': 'WideResNet-28-10-PSSiLU', 'additional_data': False, 'clean_acc': 87.02, 'autoattack_acc': 61.55, 'unreliable': False}, {'model_id': 'Debenedetti2022Light_XCiT-L12', 'venue': 'arXiv, Sep 2022', 'year': 2022.0, 'architecture': 'XCiT-L12', 'additional_data': True, 'clean_acc': 91.73, 'autoattack_acc': 57.58, 'unreliable': False}, {'model_id': 'Debenedetti2022Light_XCiT-M12', 'venue': 'arXiv, Sep 2022', 'year': 2022.0, 'architecture': 'XCiT-M12', 'additional_data': True, 'clean_acc': 91.3, 'autoattack_acc': 57.27, 'unreliable': False}, {'model_id': 'Debenedetti2022Light_XCiT-S12', 'venue': 'arXiv, Sep 2022', 'year': 2022.0, 'architecture': 'XCiT-S12', 'additional_data': True, 'clean_acc': 90.06, 'autoattack_acc': 56.14, 'unreliable': False}, {'model_id': 'Ding2020MMA', 'venue': 'ICLR 2020', 'year': 2020.0, 'architecture': 'WideResNet-28-4', 'additional_data': False, 'clean_acc': 84.36, 'autoattack_acc': 41.44, 'unreliable': None}, {'model_id': 'Engstrom2019Robustness', 'venue': 'GitHub,<br>Oct 2019', 'year': 2019.0, 'architecture': 'ResNet-50', 'additional_data': False, 'clean_acc': 87.03, 'autoattack_acc': 49.25, 'unreliable': None}, {'model_id': 'Gowal2020Uncovering_28_10_extra', 'venue': 'arXiv, Oct 2020', 'year': 2020.0, 'architecture': 'WideResNet-28-10', 'additional_data': True, 'clean_acc': 89.48, 'autoattack_acc': 62.8, 'unreliable': False}, {'model_id': 'Gowal2020Uncovering_34_20', 'venue': 'arXiv, Oct 2020', 'year': 2020.0, 'architecture': 'WideResNet-34-20', 'additional_data': False, 'clean_acc': 85.64, 'autoattack_acc': 56.86, 'unreliable': False}, {'model_id': 'Gowal2020Uncovering_70_16', 'venue': 'arXiv, Oct 2020', 'year': 2020.0, 'architecture': 'WideResNet-70-16', 'additional_data': False, 'clean_acc': 85.29, 'autoattack_acc': 57.2, 'unreliable': False}, {'model_id': 'Gowal2020Uncovering_70_16_extra', 'venue': 'arXiv, Oct 2020', 'year': 2020.0, 'architecture': 'WideResNet-70-16', 'additional_data': True, 'clean_acc': 91.1, 'autoattack_acc': 65.88, 'unreliable': False}, {'model_id': 'Gowal2021Improving_28_10_ddpm_100m', 'venue': 'NeurIPS 2021', 'year': 2021.0, 'architecture': 'WideResNet-28-10', 'additional_data': False, 'clean_acc': 87.5, 'autoattack_acc': 63.44, 'unreliable': False}, {'model_id': 'Gowal2021Improving_70_16_ddpm_100m', 'venue': 'NeurIPS 2021', 'year': 2021.0, 'architecture': 'WideResNet-70-16', 'additional_data': False, 'clean_acc': 88.74, 'autoattack_acc': 66.11, 'unreliable': False}, {'model_id': 'Gowal2021Improving_R18_ddpm_100m', 'venue': 'NeurIPS 2021', 'year': 2021.0, 'architecture': 'PreActResNet-18', 'additional_data': False, 'clean_acc': 87.35, 'autoattack_acc': 58.63, 'unreliable': False}, {'model_id': 'Hendrycks2019Using', 'venue': 'ICML 2019', 'year': 2019.0, 'architecture': 'WideResNet-28-10', 'additional_data': True, 'clean_acc': 87.11, 'autoattack_acc': 54.92, 'unreliable': False}, {'model_id': 'Huang2021Exploring', 'venue': 'NeurIPS 2021', 'year': 2021.0, 'architecture': 'WideResNet-34-R', 'additional_data': True, 'clean_acc': 90.56, 'autoattack_acc': 61.56, 'unreliable': False}, {'model_id': 'Huang2021Exploring_ema', 'venue': 'NeurIPS 2021', 'year': 2021.0, 'architecture': 'WideResNet-34-R', 'additional_data': True, 'clean_acc': 91.23, 'autoattack_acc': 62.54, 'unreliable': False}, {'model_id': 'Huang2022Revisiting_WRN-A4', 'venue': 'arXiv, Dec. 2022', 'year': 2022.0, 'architecture': 'WideResNet-A4', 'additional_data': True, 'clean_acc': 91.58, 'autoattack_acc': 65.79, 'unreliable': False}, {'model_id': 'Jang2019Adversarial', 'venue': 'ICCV 2019', 'year': 2019.0, 'architecture': 'ResNet-20', 'additional_data': False, 'clean_acc': 78.91, 'autoattack_acc': 34.95, 'unreliable': None}, {'model_id': 'Jia2022LAS-AT_34_10', 'venue': 'arXiv, Mar 2022', 'year': 2022.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 84.98, 'autoattack_acc': 56.26, 'unreliable': False}, {'model_id': 'Jia2022LAS-AT_70_16', 'venue': 'arXiv, Mar 2022', 'year': 2022.0, 'architecture': 'WideResNet-70-16', 'additional_data': False, 'clean_acc': 85.66, 'autoattack_acc': 57.61, 'unreliable': False}, {'model_id': 'JinRinard2020Manifold', 'venue': 'arXiv, Mar 2020', 'year': 2020.0, 'architecture': 'ResNet-18', 'additional_data': False, 'clean_acc': 90.84, 'autoattack_acc': 1.35, 'unreliable': None}, {'model_id': 'KimWang2020Sensible', 'venue': 'OpenReview, Sep 2019', 'year': 2019.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 91.51, 'autoattack_acc': 34.22, 'unreliable': None}, {'model_id': 'Kumari2019Harnessing', 'venue': 'IJCAI 2019', 'year': 2019.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 87.8, 'autoattack_acc': 49.12, 'unreliable': None}, {'model_id': 'Kundu2020Tunable', 'venue': 'ASP-DAC 2021', 'year': 2021.0, 'architecture': 'ResNet-18', 'additional_data': False, 'clean_acc': 87.32, 'autoattack_acc': 40.41, 'unreliable': False}, {'model_id': 'Madry2018Towards', 'venue': 'ICLR 2018', 'year': 2018.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 87.14, 'autoattack_acc': 44.04, 'unreliable': None}, {'model_id': 'Mao2019Metric', 'venue': 'NeurIPS 2019', 'year': 2019.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 86.21, 'autoattack_acc': 47.41, 'unreliable': None}, {'model_id': 'Moosavi-Dezfooli2019Robustness', 'venue': 'CVPR 2019', 'year': 2019.0, 'architecture': 'ResNet-18', 'additional_data': False, 'clean_acc': 83.11, 'autoattack_acc': 38.5, 'unreliable': None}, {'model_id': 'Mustafa2019Adversarial', 'venue': 'ICCV 2019', 'year': 2019.0, 'architecture': 'ResNet-110', 'additional_data': False, 'clean_acc': 89.16, 'autoattack_acc': 0.28, 'unreliable': None}, {'model_id': 'Pang2020Bag', 'venue': 'ICLR 2021', 'year': 2021.0, 'architecture': 'WideResNet-34-20', 'additional_data': False, 'clean_acc': 86.43, 'autoattack_acc': 54.39, 'unreliable': None}, {'model_id': 'Pang2020Boosting', 'venue': 'NeurIPS 2020', 'year': 2020.0, 'architecture': 'WideResNet-34-20', 'additional_data': False, 'clean_acc': 85.14, 'autoattack_acc': 53.74, 'unreliable': False}, {'model_id': 'Pang2020Rethinking', 'venue': 'ICLR 2020', 'year': 2020.0, 'architecture': 'ResNet-32', 'additional_data': False, 'clean_acc': 80.89, 'autoattack_acc': 43.48, 'unreliable': None}, {'model_id': 'Pang2022Robustness_WRN28_10', 'venue': 'ICML 2022', 'year': 2022.0, 'architecture': 'WideResNet-28-10', 'additional_data': False, 'clean_acc': 88.61, 'autoattack_acc': 61.04, 'unreliable': False}, {'model_id': 'Pang2022Robustness_WRN70_16', 'venue': 'ICML 2022', 'year': 2022.0, 'architecture': 'WideResNet-70-16', 'additional_data': False, 'clean_acc': 89.01, 'autoattack_acc': 63.35, 'unreliable': False}, {'model_id': 'Peng2023Robust', 'venue': 'BMVC 2023', 'year': 2023.0, 'architecture': 'RaWideResNet-70-16', 'additional_data': False, 'clean_acc': 93.27, 'autoattack_acc': 71.07, 'unreliable': False}, {'model_id': 'Qin2019Adversarial', 'venue': 'NeurIPS 2019', 'year': 2019.0, 'architecture': 'WideResNet-40-8', 'additional_data': False, 'clean_acc': 86.28, 'autoattack_acc': 52.84, 'unreliable': None}, {'model_id': 'Rade2021Helper_R18_ddpm', 'venue': 'OpenReview, Jun 2021', 'year': 2021.0, 'architecture': 'PreActResNet-18', 'additional_data': False, 'clean_acc': 86.86, 'autoattack_acc': 57.09, 'unreliable': False}, {'model_id': 'Rade2021Helper_R18_extra', 'venue': 'OpenReview, Jun 2021', 'year': 2021.0, 'architecture': 'PreActResNet-18', 'additional_data': True, 'clean_acc': 89.02, 'autoattack_acc': 57.67, 'unreliable': False}, {'model_id': 'Rade2021Helper_ddpm', 'venue': 'OpenReview, Jun 2021', 'year': 2021.0, 'architecture': 'WideResNet-28-10', 'additional_data': False, 'clean_acc': 88.16, 'autoattack_acc': 60.97, 'unreliable': False}, {'model_id': 'Rade2021Helper_extra', 'venue': 'OpenReview, Jun 2021', 'year': 2021.0, 'architecture': 'WideResNet-34-10', 'additional_data': True, 'clean_acc': 91.47, 'autoattack_acc': 62.83, 'unreliable': False}, {'model_id': 'Rebuffi2021Fixing_106_16_cutmix_ddpm', 'venue': 'arXiv, Mar 2021', 'year': 2021.0, 'architecture': 'WideResNet-106-16', 'additional_data': False, 'clean_acc': 88.5, 'autoattack_acc': 64.64, 'unreliable': False}, {'model_id': 'Rebuffi2021Fixing_28_10_cutmix_ddpm', 'venue': 'arXiv, Mar 2021', 'year': 2021.0, 'architecture': 'WideResNet-28-10', 'additional_data': False, 'clean_acc': 87.33, 'autoattack_acc': 60.75, 'unreliable': False}, {'model_id': 'Rebuffi2021Fixing_70_16_cutmix_ddpm', 'venue': 'arXiv, Mar 2021', 'year': 2021.0, 'architecture': 'WideResNet-70-16', 'additional_data': False, 'clean_acc': 88.54, 'autoattack_acc': 64.25, 'unreliable': False}, {'model_id': 'Rebuffi2021Fixing_70_16_cutmix_extra', 'venue': 'arXiv, Mar 2021', 'year': 2021.0, 'architecture': 'WideResNet-70-16', 'additional_data': True, 'clean_acc': 92.23, 'autoattack_acc': 66.58, 'unreliable': False}, {'model_id': 'Rebuffi2021Fixing_R18_ddpm', 'venue': 'arXiv, Mar 2021', 'year': 2021.0, 'architecture': 'PreActResNet-18', 'additional_data': False, 'clean_acc': 83.53, 'autoattack_acc': 56.66, 'unreliable': False}, {'model_id': 'Rice2020Overfitting', 'venue': 'ICML 2020', 'year': 2020.0, 'architecture': 'WideResNet-34-20', 'additional_data': False, 'clean_acc': 85.34, 'autoattack_acc': 53.42, 'unreliable': False}, {'model_id': 'Sehwag2020Hydra', 'venue': 'NeurIPS 2020', 'year': 2020.0, 'architecture': 'WideResNet-28-10', 'additional_data': True, 'clean_acc': 88.98, 'autoattack_acc': 57.14, 'unreliable': False}, {'model_id': 'Sehwag2021Proxy', 'venue': 'ICLR 2022', 'year': 2022.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 86.68, 'autoattack_acc': 60.27, 'unreliable': False}, {'model_id': 'Sehwag2021Proxy_R18', 'venue': 'ICLR 2022', 'year': 2022.0, 'architecture': 'ResNet-18', 'additional_data': False, 'clean_acc': 84.59, 'autoattack_acc': 55.54, 'unreliable': False}, {'model_id': 'Sehwag2021Proxy_ResNest152', 'venue': 'ICLR 2022', 'year': 2022.0, 'architecture': 'ResNest152', 'additional_data': False, 'clean_acc': 87.3, 'autoattack_acc': 62.79, 'unreliable': False}, {'model_id': 'Shafahi2019Adversarial', 'venue': 'NeurIPS 2019', 'year': 2019.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 86.11, 'autoattack_acc': 41.47, 'unreliable': None}, {'model_id': 'Sitawarin2020Improving', 'venue': 'arXiv, Mar 2020', 'year': 2020.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 86.84, 'autoattack_acc': 50.72, 'unreliable': None}, {'model_id': 'Sridhar2021Robust', 'venue': 'ACC 2022', 'year': 2022.0, 'architecture': 'WideResNet-28-10', 'additional_data': True, 'clean_acc': 89.46, 'autoattack_acc': 59.66, 'unreliable': False}, {'model_id': 'Sridhar2021Robust_34_15', 'venue': 'ACC 2022', 'year': 2022.0, 'architecture': 'WideResNet-34-15', 'additional_data': True, 'clean_acc': 86.53, 'autoattack_acc': 60.41, 'unreliable': False}, {'model_id': 'Wang2020Improving', 'venue': 'ICLR 2020', 'year': 2020.0, 'architecture': 'WideResNet-28-10', 'additional_data': True, 'clean_acc': 87.5, 'autoattack_acc': 56.29, 'unreliable': False}, {'model_id': 'Wang2023Better_WRN-28-10', 'venue': 'ICML 2023', 'year': 2023.0, 'architecture': 'WideResNet-28-10', 'additional_data': False, 'clean_acc': 92.44, 'autoattack_acc': 67.31, 'unreliable': False}, {'model_id': 'Wang2023Better_WRN-70-16', 'venue': 'ICML 2023', 'year': 2023.0, 'architecture': 'WideResNet-70-16', 'additional_data': False, 'clean_acc': 93.25, 'autoattack_acc': 70.69, 'unreliable': False}, {'model_id': 'WangZhang2019Bilateral', 'venue': 'ICCV 2019', 'year': 2019.0, 'architecture': 'WideResNet-28-10', 'additional_data': False, 'clean_acc': 92.8, 'autoattack_acc': 29.35, 'unreliable': None}, {'model_id': 'Wong2020Fast', 'venue': 'ICLR 2020', 'year': 2020.0, 'architecture': 'PreActResNet-18', 'additional_data': False, 'clean_acc': 83.34, 'autoattack_acc': 43.21, 'unreliable': None}, {'model_id': 'Wu2020Adversarial', 'venue': 'NeurIPS 2020', 'year': 2020.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 85.36, 'autoattack_acc': 56.17, 'unreliable': False}, {'model_id': 'Wu2020Adversarial_extra', 'venue': 'NeurIPS 2020', 'year': 2020.0, 'architecture': 'WideResNet-28-10', 'additional_data': True, 'clean_acc': 88.25, 'autoattack_acc': 60.04, 'unreliable': False}, {'model_id': 'Wu2020Do', 'venue': 'arXiv, Oct 2020', 'year': 2020.0, 'architecture': 'WideResNet-34-15', 'additional_data': True, 'clean_acc': 87.67, 'autoattack_acc': 60.65, 'unreliable': None}, {'model_id': 'Xu2023Exploring_WRN-28-10', 'venue': 'ICLR 2023', 'year': 2023.0, 'architecture': 'WideResNet-28-10', 'additional_data': False, 'clean_acc': 93.69, 'autoattack_acc': 63.89, 'unreliable': False}, {'model_id': 'Zhang2019You', 'venue': 'NeurIPS 2019', 'year': 2019.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 87.2, 'autoattack_acc': 44.83, 'unreliable': None}, {'model_id': 'Zhang2020Attacks', 'venue': 'ICML 2020', 'year': 2020.0, 'architecture': 'WideResNet-34-10', 'additional_data': False, 'clean_acc': 84.52, 'autoattack_acc': 53.51, 'unreliable': False}, {'model_id': 'Zhang2020Geometry', 'venue': 'ICLR 2021', 'year': 2021.0, 'architecture': 'WideResNet-28-10', 'additional_data': True, 'clean_acc': 89.36, 'autoattack_acc': 59.64, 'unreliable': False}, {'model_id': 'Zhang2020Towards', 'venue': 'ICLR 2020', 'year': 2020.0, 'architecture': '5-layer-CNN', 'additional_data': False, 'clean_acc': 44.73, 'autoattack_acc': 32.64, 'unreliable': None}, {'model_id': 'ZhangWang2019Defense', 'venue': 'NeurIPS 2019', 'year': 2019.0, 'architecture': 'WideResNet-28-10', 'additional_data': False, 'clean_acc': 89.98, 'autoattack_acc': 36.64, 'unreliable': None}, {'model_id': 'ZhangXu2020Adversarial', 'venue': 'OpenReview, Sep 2019', 'year': 2019.0, 'architecture': 'WideResNet-28-10', 'additional_data': False, 'clean_acc': 90.25, 'autoattack_acc': 36.45, 'unreliable': None}]
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
BROKEN = {4653550208, 4653553920, 4653557056, 4653560448}

def robustbench(path, k):
    rng = np.random.default_rng(0)
    jit = [rng.uniform(-0.2, 0.2) for _ in RB]
    fig, ax = plt.subplots(figsize=(5.3, 2.0))
    xs = [int(r['year']) + j for r, j in zip(RB, jit)]
    ax.scatter(xs, [r['clean_acc'] for r in RB], s=20, color=C['neutral'], alpha=0.6, zorder=2)
    ax.text(2017.45, 90, 'clean', color=C['neutral'], fontsize=12.5, fontweight='bold', ha='left')
    if k >= 2:
        ax.scatter(xs, [r['autoattack_acc'] for r in RB], s=22, color=C['danger'], zorder=3)
        bx, by = ([], [])
        for y in YEARS:
            run = {2018: 44.04, 2019: 59.53, 2020: 65.88, 2021: 66.58, 2022: 66.58, 2023: 71.07, 2024: 75.28}[y]
            bx.append(y - 0.5)
            by.append(run)
        bx.append(YEARS[-1] + 0.5)
        by.append(run)
        ax.step(bx, by, where='post', color=C['danger'], lw=2.2, zorder=4)
        ax.text(2017.45, 14, 'under attack', color=C['danger'], fontsize=12.5, fontweight='bold')
        ax.text(YEARS[-1] + 0.65, 75.0, 'top\n75%', ha='left', va='center', color=C['danger'], fontsize=12.5, fontweight='bold')
    if k >= 3:
        for r, x in zip(RB, xs):
            if id(r) in BROKEN:
                ax.plot([x], [r['autoattack_acc']], 'o', ms=13, mfc='none', mec=C['text'], mew=1.8, zorder=5)
        ax.text(2020.9, 10, '4 published defenses\nbelow 5% under AutoAttack', fontsize=12, fontweight='bold', color=C['text'], va='bottom')
    ax.set_xticks(YEARS)
    ax.set_xlim(2017.4, 2025.4)
    ax.set_ylim(-3, 100)
    ax.set_ylabel('CIFAR-10 accuracy (%)')
    ax.set_xlabel('Publication year')
    ax.grid(axis='y', color=C['grid'], lw=0.5)
    sp.save(fig, path)
for k in (1, 2, 3):
    robustbench(out / f'robustbench_step{k}.svg', k)
sp.hbars(out / 'certified.svg', [('Certified radius', 1.545116153, 'data', '1.5'), ('8/255 box corner', 1.738843164, 'danger', '1.74')], '', log=False, size=(3.4, 1.25), xlabel='Euclidean length')
sp.hbars(out / 'badnets.svg', [('Honest model, clean signs', 90.0, 'neutral', '90.0% correct'), ('Backdoored, clean signs', 89.3, 'memory', '89.3% correct'), ('Backdoored, sticker on stop', 90.3, 'danger', '90.3% speed limit')], '', log=False, size=(5.4, 1.6), xlabel='Percent (BadNets, Table 4)')

def layers(path, k):
    stages = [('Ingress', 'source identity,', 'least privilege', 'trusted source', 'turns bad', V['blue'], V['blue_l']), ('Sanitize', 'outliers, clusters,', 'spectral filters', 'clean-label poison', 'is not an outlier', V['violet'], V['violet_l']), ('Robust training', 'Huber loss,', 'robust aggregation', 'bounds a residual,', 'not an update', V['amber'], V['amber_l']), ('Release checks', 'triggers, canaries,', 'target slices', 'a trigger nobody', 'tested', V['green'], V['green_l'])]
    b = []
    for i, (n, d1, d2, b1, b2, s, f) in enumerate(stages):
        x = 10 + i * 222
        op = 1.0 if k >= i + 1 else 0.0
        b.append(rect(x, 10, 208, 250, f, s, 2.5, 8, op))
        b.append(text(x + 104, 46, n, s, 24, 700, op=op))
        b.append(text(x + 104, 88, d1, V['ink'], 20, op=op))
        b.append(text(x + 104, 116, d2, V['ink'], 20, op=op))
        b.append(text(x + 104, 170, 'blind spot', V['red'], 20, 700, op=op))
        b.append(text(x + 104, 200, b1, V['ink'], 20, op=op, italic=True))
        b.append(text(x + 104, 228, b2, V['ink'], 20, op=op, italic=True))
        if i < 3:
            b.append(arrow(x + 208, 135, x + 221, 135, 'gray', 3, op))
    op = 1.0 if k >= 5 else 0.0
    b.append(f'''<path d="M 790 265 L 790 300 L 114 300 L 114 268" fill="none" stroke="{V['red']}" stroke-width="3" stroke-dasharray="9,6" marker-end="url(#a-red)" opacity="{op}"/>''')
    b.append(text(450, 338, 'Lineage traces a bad behavior to the batches that taught it', V['red'], 22, 700, op=op))
    svg(path, b, 900, 350)
for k in (1, 2, 3, 4, 5):
    layers(out / f'layers_step{k}.svg', k)
bits = struct.unpack('>I', struct.pack('>f', 0.042))[0]

def bitflip(path, k):
    b = []
    word = format(bits, '032b')
    flipped = format(bits ^ 1 << 30, '032b')

    def row(y, s, hl, op=1.0):
        out_ = []
        cells = [('sign', s[0])] + [('exp', c) for c in s[1:1 + 8]] + [('man', '')]
        x = 10
        for j, (kind, ch) in enumerate(cells):
            wd = 290 if kind == 'man' else 56
            f = V['red_l'] if hl and j == 1 else V['blue_l'] if kind == 'exp' else V['frame_l']
            st = V['red'] if hl and j == 1 else V['frame']
            out_.append(rect(x, y, wd, 52, f, st, 3 if hl and j == 1 else 1.5, 3, op))
            t = ch if kind != 'man' else 'mantissa, 23 bits'
            out_.append(text(x + wd / 2, y + 35, t, V['red'] if hl and j == 1 else V['ink'], 25 if kind != 'man' else 20, 700 if kind != 'man' else 400, op=op))
            x += wd + 4
        return out_
    b.append(text(10, 30, 'w = 0.042', V['ink'], 24, 700, 'start'))
    b.append(text(160, 30, 'sign | exponent, 8 bits | mantissa', V['gray'], 20, 400, 'start'))
    b += row(45, word, False)
    op = 1.0 if k >= 2 else 0.0
    b.append(arrow(98, 102, 98, 160, 'red', 3.5, op))
    b.append(text(125, 140, 'bit 30 flips: value × 2<tspan baseline-shift="super" font-size="15">128</tspan>', V['red'], 23, 700, 'start', op))
    b += row(165, flipped, True, op)
    b.append(text(10, 248, 'w′ ≈ 1.4 × 10<tspan baseline-shift="super" font-size="15">37</tspan>: finite, so every NaN check passes', V['red'], 23, 700, 'start', op))
    op = 1.0 if k >= 3 else 0.0
    checks = [('Range check', '|w| ≤ bound, one compare per value', 'compute'), ('Golden canary', 'replay fixed inputs, compare outputs', '+ communication'), ('Cross-replica', 'run a sampled request twice', '2× on the sample')]
    for i, (n, d, c) in enumerate(checks):
        y = 285 + i * 38
        b.append(text(10, y + 22, n, V['green'], 21, 700, 'start', op))
        b.append(text(185, y + 22, d, V['ink'], 21, 400, 'start', op))
        b.append(text(890, y + 22, c, V['violet'], 20, 700, 'end', op))
    svg(path, b, 900, 400)
for k in (1, 2, 3):
    bitflip(out / f'bitflip_step{k}.svg', k)

def replay(path, k):
    b = []
    b.append(rect(5, 125, 120, 100, V['frame_l'], V['frame'], 2))
    b.append(text(65, 168, 'golden', V['ink'], 21, 700))
    b.append(text(65, 196, 'requests', V['ink'], 21, 700))
    for i, (lab, col, fill) in enumerate([('pipeline v1', V['blue'], V['blue_l']), ('pipeline v2', V['amber'], V['amber_l'])]):
        y = 20 + i * 220
        b.append(arrow(125, 175, 160, y + 45, 'gray', 2.5))
        b.append(rect(163, y, 150, 90, fill, col, 2.5))
        b.append(text(238, y + 38, lab, col, 21, 700))
        b.append(text(238, y + 68, 'raw: equal', V['ink'], 19))
        b.append(arrow(313, y + 45, 340, y + 45, 'gray', 2.5))
        b.append(rect(343, y, 110, 90, '#fff', col, 2.5))
        b.append(text(398, y + 52, 'tensor', V['ink'], 21, 700))
    op = 1.0 if k >= 2 else 0.0
    b.append(rect(470, 115, 125, 120, V['red_l'], V['red'], 2.5, 8, op))
    b.append(text(532, 155, 'differ by', V['red'], 20, 700, op=op))
    b.append(text(532, 182, 'a constant', V['red'], 20, 700, op=op))
    b.append(text(532, 209, 'factor', V['red'], 20, 700, op=op))
    b.append(arrow(453, 65, 500, 113, 'red', 2.5, op))
    b.append(arrow(453, 285, 500, 237, 'red', 2.5, op))
    svg(path, b, 600, 335)
for k in (1, 2):
    replay(out / f'replay_step{k}.svg', k)
print('figures written to', out)
