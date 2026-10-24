#!/usr/bin/env python3
"""Figures for Lecture 2 (The Foundation Model). Run from this folder: python3 make_figs.py

Every number drawn here comes from calcs.py, which reads MLSysIM (registry,
Literature, and the Scenarios.* the chapter also reads), or from
data/model_runs.json, the deck's own model run, which calcs.py also reads:
plotted geometry uses the scenario values, labels use the formatted strings
the slides print (deckvals val/num). The incident dates on the hook timeline
are the postmortem's, typed with their source. Diagram coordinates, axis
ranges, and tick positions are layout. Diagrams use the Volume III stack
palette (rules/volumes/vol3/palette.md): sky blue = model and generation,
green = context, amber = tools and actions, violet = runtime records, red = faults.
"""
import mlsysim
_VALS = {'pm_initial_pct': '0.8', 'pm_worst_pct': '16', 'pm_users_pct': '30', 'pm_topk_n': '12,000', 'pm_topk_fail': '10,240', 'rec_h': '30', 'rec_step': '0.99', 'rec_step_pct': '99', 'rec_success_pct': '74.0', 'rec_attempt': '2.10', 'rec_acc': '0.40', 'rec_acc_pct': '40', 'rec_ceff': '5.25', 'run_done_tokens': '35', 'run_done_max': '128', 'run_cut_tokens': '24', 'run_cut_max': '24', 'wire_tokens': '176', 'visible_tokens': '23', 'visible_sys': '10', 'visible_user': '13', 'framing_tokens': '153', 'wire_ratio': '7.7', 'vocab': '151,936', 'tok_flat_n': '5', 'tok_flat_id0': '755', 'tok_ind_n': '6', 'tok_ind_id0': '262', 'tok_ind_id1': '711', 'tok_path_n': '4', 'adm_window': '8,192', 'adm_reserve': '1,024', 'adm_text': '6,000', 'adm_image': '2,048', 'adm_ceiling': '7,168', 'adm_headroom': '1,168', 'adm_total': '9,072', 'adm_excess': '880', 'adm_half_reserve': '512', 'adm_half_total': '8,560', 'adm_half_excess': '368', 'p_path_t1': '34.6', 'p_c_t1': '16.3', 'p_src_t1': '11.3', 'p_path_t05': '69.2', 'p_src_t05': '7.4', 'run_nucleus': '32', 'run_top_n': '8', 'nm_z1': '12', 'nm_z2': '10.5', 'nm_z3': '9', 'nm_z4': '6', 'nm_z5': '3', 'nm_t_base': '1', 'nm_t_sharp': '0.5', 'nm_top_p': '0.90', 'nm_p1_pct': '78.4', 'nm_p1': '0.7840', 'nm_sharp_pct': '95.0', 'nm_sharp_round': '95', 'nm_tail1_pct': '21.6', 'nm_tail05_pct': '5.0', 'nm_tail_ratio': '4.3', 'nm_nuc_mass': '0.9589', 'nm_nuc1_pct': '81.8', 'nm_nuc2_pct': '18.2', 'nm_nuc1': '0.8176', 'nm_nuc2': '0.1824', 'batch_other': '471', 'batch_pad': '295', 'batch_fp32_changed': '150,078', 'batch_bf16_changed': '140,555', 'batch_fp32_pct': '98.8', 'batch_fp32_mant': '3.9', 'batch_fp32_exp': '-5', 'batch_bf16_max': '0.25', 'cancel_cap': '4,096', 'cancel_step': '96', 'cancel_skipped': '4,000', 'cancel_s': '84.3', 'huang_gsm_std': '95.5', 'huang_gsm_sc': '89.0', 'huang_gsm_orc': '97.5', 'huang_csqa_std': '82.0', 'huang_csqa_sc': '80.0', 'huang_csqa_orc': '85.5', 'huang_hotpot_std': '49.0', 'huang_hotpot_sc': '43.0', 'huang_hotpot_orc': '59.0', 'chk_defect_pct': '18', 'chk_defect': '0.18', 'chk_valid_share': '0.82', 'chk_fa': '0.65', 'chk_va': '0.85', 'chk_post_pct2': '14.37', 'chk_post_pct': '14.4', 'chk_catch_pct': '35', 'chk_vreject_pct': '15', 'chk_fa_half': '0.325', 'chk_half_pct': '7.7', 'chk_joint_pct': '11.7', 'fl_params_b': '70.6', 'fl_weights_gb': '70.6', 'fl_bytes_per_param': '1', 'fl_peak_tflops': '1,979', 'fl_bw_tbs': '3.35', 'fl_prefill_us': '71.35', 'fl_decode_ms': '21.075', 'fl_ratio': '295', 't1_prompt': '8,000', 't1_output': '200', 't1_prefill_s': '0.571', 't1_decode_s': '4.215', 't1_tok_pct': '2.4', 't1_time_pct': '88.1', 'lap_prompt': '2,048', 'lap_prefill_s': '2.05', 'lap_prefill_ms': '1.0', 'lap_prefill_ms2': '1.00', 'lap_decode_ms': '35.1', 'lap_ratio': '35', 'lap_ratio_lo': '25', 'lap_ratio_hi': '41', 'lap_prefill_runs': '5', 'lap_decode_runs': '3', 'az_requests': '8,819', 'az_med_prompt': '1,469', 'az_med_output': '13', 'az_input_pct': '98.7', 'az_time_pct': '80', 'rs_s0': '8,000', 'rs_delta': '1,500', 'rs_k': '200', 'rs_turns': '30', 'rs_turns2': '60', 'rs_resent': '892,500', 'rs_resent2': '3,135,000', 'rs_growth': '3.5', 'rs_app_growth': '4.07', 'rs_first_growth': '2.0', 'rs_fresh': '51,500', 'rs_fresh2': '96,500', 'rs_fresh_growth': '1.9', 'rs_reduction': '17.3', 'rs_pre_s': '63.68', 'rs_pre_reuse_s': '3.67', 'rs_dec_s': '126.45', 'rs_resent_k': '892.5', 'rs_resent2_k': '3,135', 'rs_fresh2_k': '96.5', 'ed_deadline_s': '15', 'ed_full_dm': '350', 'ed_full_k': '1,850', 'ed_diff_dm': '120', 'ed_diff_k': '80', 'ed_full_s': '39.58', 'ed_full_s1': '39.6', 'ed_diff_s': '2.27', 'ed_diff_s1': '2.3', 'ed_speedup': '17.5', 'ed_full_dec_pct': '98.5', 'ed_over_s': '24.6', 'ed_margin': '6.6', 'ed_apply': '0.95', 'ed_pass': '0.60', 'ed_apply_pct': '95', 'ed_pass_pct': '60', 'ed_yield_pct': '57'}
_NUMS = {'pm_initial_pct': 0.8, 'pm_worst_pct': 16.0, 'pm_users_pct': 30.0, 'pm_topk_n': 12000.0, 'pm_topk_fail': 10240.0, 'rec_h': 30.0, 'rec_step': 0.99, 'rec_step_pct': 99.0, 'rec_success_pct': 74.0, 'rec_attempt': 2.1, 'rec_acc': 0.4, 'rec_acc_pct': 40.0, 'rec_ceff': 5.25, 'run_done_tokens': 35.0, 'run_done_max': 128.0, 'run_cut_tokens': 24.0, 'run_cut_max': 24.0, 'wire_tokens': 176.0, 'visible_tokens': 23.0, 'visible_sys': 10.0, 'visible_user': 13.0, 'framing_tokens': 153.0, 'wire_ratio': 7.7, 'vocab': 151936.0, 'tok_flat_n': 5.0, 'tok_flat_id0': 755.0, 'tok_ind_n': 6.0, 'tok_ind_id0': 262.0, 'tok_ind_id1': 711.0, 'tok_path_n': 4.0, 'adm_window': 8192.0, 'adm_reserve': 1024.0, 'adm_text': 6000.0, 'adm_image': 2048.0, 'adm_ceiling': 7168.0, 'adm_headroom': 1168.0, 'adm_total': 9072.0, 'adm_excess': 880.0, 'adm_half_reserve': 512.0, 'adm_half_total': 8560.0, 'adm_half_excess': 368.0, 'p_path_t1': 34.6, 'p_c_t1': 16.3, 'p_src_t1': 11.3, 'p_path_t05': 69.2, 'p_src_t05': 7.4, 'run_nucleus': 32.0, 'run_top_n': 8.0, 'nm_z1': 12.0, 'nm_z2': 10.5, 'nm_z3': 9.0, 'nm_z4': 6.0, 'nm_z5': 3.0, 'nm_t_base': 1.0, 'nm_t_sharp': 0.5, 'nm_top_p': 0.9, 'nm_p1_pct': 78.4, 'nm_p1': 0.784, 'nm_sharp_pct': 95.0, 'nm_sharp_round': 95.0, 'nm_tail1_pct': 21.6, 'nm_tail05_pct': 5.0, 'nm_tail_ratio': 4.3, 'nm_nuc_mass': 0.9589, 'nm_nuc1_pct': 81.8, 'nm_nuc2_pct': 18.2, 'nm_nuc1': 0.8176, 'nm_nuc2': 0.1824, 'batch_other': 471.0, 'batch_pad': 295.0, 'batch_fp32_changed': 150078.0, 'batch_bf16_changed': 140555.0, 'batch_fp32_pct': 98.8, 'batch_fp32_mant': 3.9, 'batch_fp32_exp': -5.0, 'batch_bf16_max': 0.25, 'cancel_cap': 4096.0, 'cancel_step': 96.0, 'cancel_skipped': 4000.0, 'cancel_s': 84.3, 'huang_gsm_std': 95.5, 'huang_gsm_sc': 89.0, 'huang_gsm_orc': 97.5, 'huang_csqa_std': 82.0, 'huang_csqa_sc': 80.0, 'huang_csqa_orc': 85.5, 'huang_hotpot_std': 49.0, 'huang_hotpot_sc': 43.0, 'huang_hotpot_orc': 59.0, 'chk_defect_pct': 18.0, 'chk_defect': 0.18, 'chk_valid_share': 0.82, 'chk_fa': 0.65, 'chk_va': 0.85, 'chk_post_pct2': 14.37, 'chk_post_pct': 14.4, 'chk_catch_pct': 35.0, 'chk_vreject_pct': 15.0, 'chk_fa_half': 0.325, 'chk_half_pct': 7.7, 'chk_joint_pct': 11.7, 'fl_params_b': 70.6, 'fl_weights_gb': 70.6, 'fl_bytes_per_param': 1.0, 'fl_peak_tflops': 1979.0, 'fl_bw_tbs': 3.35, 'fl_prefill_us': 71.35, 'fl_decode_ms': 21.075, 'fl_ratio': 295.0, 't1_prompt': 8000.0, 't1_output': 200.0, 't1_prefill_s': 0.571, 't1_decode_s': 4.215, 't1_tok_pct': 2.4, 't1_time_pct': 88.1, 'lap_prompt': 2048.0, 'lap_prefill_s': 2.05, 'lap_prefill_ms': 1.0, 'lap_prefill_ms2': 1.0, 'lap_decode_ms': 35.1, 'lap_ratio': 35.0, 'lap_ratio_lo': 25.0, 'lap_ratio_hi': 41.0, 'lap_prefill_runs': 5.0, 'lap_decode_runs': 3.0, 'az_requests': 8819.0, 'az_med_prompt': 1469.0, 'az_med_output': 13.0, 'az_input_pct': 98.7, 'az_time_pct': 80.0, 'rs_s0': 8000.0, 'rs_delta': 1500.0, 'rs_k': 200.0, 'rs_turns': 30.0, 'rs_turns2': 60.0, 'rs_resent': 892500.0, 'rs_resent2': 3135000.0, 'rs_growth': 3.5, 'rs_app_growth': 4.07, 'rs_first_growth': 2.0, 'rs_fresh': 51500.0, 'rs_fresh2': 96500.0, 'rs_fresh_growth': 1.9, 'rs_reduction': 17.3, 'rs_pre_s': 63.68, 'rs_pre_reuse_s': 3.67, 'rs_dec_s': 126.45, 'rs_resent_k': 892.5, 'rs_resent2_k': 3135.0, 'rs_fresh2_k': 96.5, 'ed_deadline_s': 15.0, 'ed_full_dm': 350.0, 'ed_full_k': 1850.0, 'ed_diff_dm': 120.0, 'ed_diff_k': 80.0, 'ed_full_s': 39.58, 'ed_full_s1': 39.6, 'ed_diff_s': 2.27, 'ed_diff_s1': 2.3, 'ed_speedup': 17.5, 'ed_full_dec_pct': 98.5, 'ed_over_s': 24.6, 'ed_margin': 6.6, 'ed_apply': 0.95, 'ed_pass': 0.6, 'ed_apply_pct': 95.0, 'ed_pass_pct': 60.0, 'ed_yield_pct': 57.0}
import json
import sys
from pathlib import Path
import matplotlib.patches as mp
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'assets' / 'kit'))
import slideplot as sp
sys.path.insert(0, str(HERE))
from mlsysim.physics.agents import calc_append_only_context, calc_reused_prefix_prompt_tokens
plt = sp.plt
C = sp.C
out = HERE
RUN = json.loads((HERE / 'data' / 'model_runs.json').read_text())
V = {'model': '#0284c7', 'model_l': '#e0f2fe', 'model_t': '#0369a1', 'ctx': '#16a34a', 'ctx_l': '#dcfce7', 'ctx_t': '#15803d', 'tool': '#d97706', 'tool_l': '#fef3c7', 'tool_t': '#b45309', 'rt': '#7c3aed', 'rt_l': '#f3e8ff', 'rt_t': '#6d28d9', 'bad': '#dc2626', 'bad_l': '#fee2e2', 'bad_t': '#b91c1c', 'st': '#94a3b8', 'st_l': '#f8fafc', 'ink': '#0f172a', 'gray': '#64748b'}

def canvas(w=9.0, h=4.6, phys=None):
    """900 units wide; phys is the drawn width in inches (the slot size)."""
    phys = phys or w
    fig = plt.figure(figsize=(phys, h * phys / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 900)
    ax.set_ylim(0, h / w * 900)
    ax.axis('off')
    return (fig, ax)

def box(ax, x, y, w, h, fill, edge, text='', tc=None, fs=17, bold=True, ls='-', lw=2.2, alpha=1.0):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=8', fc=fill, ec=edge, lw=lw, ls=ls, alpha=alpha))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc or V['ink'], fontweight='bold' if bold else 'normal', alpha=alpha, linespacing=1.25)

def arrow(ax, x0, y0, x1, y1, color, lw=2.4, ls='-', alpha=1.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls, mutation_scale=20, alpha=alpha))

def save(fig, name):
    sp.save(fig, out / name)

def boundary(k):
    fig, ax = canvas(9.0, 4.4, phys=6.4)
    H = 440
    box(ax, 15, 270, 220, 95, V['model_l'], V['model'], 'Foundation\nmodel', V['model_t'], fs=14)
    box(ax, 15, 140, 220, 95, V['model_l'], V['model'], 'Inference\nservice', V['model_t'], fs=14)
    ax.text(125, 385, 'Generates text', ha='center', fontsize=13, color=V['model_t'], fontweight='bold')
    box(ax, 255, 210, 195, 90, 'white', V['tool'], 'read_file(\n"src/pool.py")', V['tool_t'], fs=11.5, ls='--')
    ax.text(352, 198, 'proposal:\nno authority', ha='center', va='top', fontsize=12, color=V['tool_t'])
    arrow(ax, 235, 215, 255, 235, V['model'])
    a2 = 1.0 if k >= 2 else 0.0
    a3 = 1.0 if k >= 3 else 0.0
    ax.add_patch(mp.FancyBboxPatch((470, 40), 420, 365, boxstyle='round,pad=0,rounding_size=10', fc=V['rt_l'], ec=V['rt'], lw=2.2, alpha=a2))
    ax.text(680, 375, 'Agent runtime decides', ha='center', fontsize=14, color=V['rt_t'], fontweight='bold', alpha=a2)
    box(ax, 485, 260, 185, 85, 'white', V['rt'], 'Validate\nschema', V['rt_t'], fs=13, alpha=a2)
    box(ax, 690, 260, 190, 85, 'white', V['rt'], 'Authorize\nthe path', V['rt_t'], fs=13, alpha=a2)
    arrow(ax, 450, 255, 485, 295, V['tool'], alpha=a2)
    arrow(ax, 670, 302, 690, 302, V['rt'], alpha=a2)
    box(ax, 690, 140, 190, 85, V['tool_l'], V['tool'], 'Execute in\nsandbox', V['tool_t'], fs=13, alpha=a3)
    box(ax, 485, 140, 185, 85, V['ctx_l'], V['ctx'], 'Observation:\nfile excerpt', V['ctx_t'], fs=13, alpha=a3)
    arrow(ax, 785, 260, 785, 225, V['rt'], alpha=a3)
    arrow(ax, 690, 182, 670, 182, V['tool'], alpha=a3)
    ax.plot([577, 577], [140, 95], color=V['ctx'], lw=2.4, ls='--', alpha=a3)
    arrow(ax, 577, 95, 125, 95, V['ctx'], ls='--', alpha=a3)
    arrow(ax, 125, 95, 125, 140, V['ctx'], ls='--', alpha=a3)
    ax.text(270, 62, 'observation joins the next call', ha='center', fontsize=12, color=V['ctx_t'], alpha=a3)
    ax.set_ylim(30, H)
    save(fig, f'boundary_step{k}.svg')
for k in (1, 2, 3):
    boundary(k)

def token_row(ax, y, label, pieces, ids, color, fill):
    ax.text(10, y + 40, label, fontsize=13, color=V['ink'], fontweight='bold', va='center')
    x = 140
    for pc, i in zip(pieces, ids):
        shown = pc.replace(' ', '␣')
        w = max(62, 22 * len(shown) + 26)
        box(ax, x, y + 15, w, 52, fill, color, shown, color, fs=14, bold=False, lw=1.8)
        ax.text(x + w / 2, y + 8, str(i), ha='center', va='top', fontsize=12, color=V['gray'])
        x += w + 10
    return x
fig, ax = canvas(9.0, 2.8, phys=6.4)
f, ind = (RUN['cl100k_flat'], RUN['cl100k_indented'])
token_row(ax, 175, 'flush', f['pieces'], f['ids'], V['model'], V['model_l'])
token_row(ax, 60, 'indented', ind['pieces'], ind['ids'], V['tool'], V['tool_l'])
ax.text(880, 15, '␣ = space.  Gray: token id.', ha='right', fontsize=13, color=V['gray'])
ax.set_ylim(0, 280)
save(fig, 'tokens.svg')

def admit(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    text, image = (6000.0, 2048.0)
    reserve, half = (1024.0, 512.0)
    window = 8192.0
    rows = [('Text only', [(text, '6,000', C['memory']), (reserve, '1,024', C['neutral'])], '1,168 to spare', False), ('+ image', [(text, '', C['memory']), (image, '2,048', C['compute']), (reserve, '1,024', C['neutral'])], '880 over', True), ('+ image,\nhalf reserve', [(text, '', C['memory']), (image, '2,048', C['compute']), (half, '512', C['neutral'])], '368 over', True)]
    for i, (lab, segs, msg, over) in enumerate(rows):
        y = 2 - i
        vis = i < k
        x = 0
        for w, t, c in segs:
            ax.barh(y, w, left=x, color=c, height=0.6, alpha=1 if vis else 0)
            if t:
                ax.text(x + w / 2, y, t, ha='center', va='center', fontsize=10.5, color='white', fontweight='bold', alpha=1 if vis else 0)
            x += w
        ax.text(-150, y, lab, ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'], alpha=1)
        ax.text(max(x, window) + 150, y, msg, va='center', fontsize=12, fontweight='bold', color=C['danger'] if over else C['data'], alpha=1 if vis else 0)
    ax.axvline(window, color=C['danger'], lw=2, ls='--')
    ax.text(window, 2.55, 'window 8,192', ha='center', fontsize=11, color=C['danger'], fontweight='bold')
    ax.text(0, 2.55, 'prompt', fontsize=11, color=C['memory'], fontweight='bold')
    ax.text(1500, 2.55, 'image', fontsize=11, color=C['compute'], fontweight='bold')
    ax.text(2900, 2.55, 'output reserve', fontsize=11, color=C['neutral'], fontweight='bold')
    ax.set_xlim(0, 11600)
    ax.set_ylim(-0.5, 2.85)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Tokens (scenario values, Ch. 2.3)')
    sp.save(fig, out / f'admit_step{k}.svg')
for k in (1, 2, 3):
    admit(k)
pl = RUN['path_logits']

def probs(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    toks = [t if t.strip() else repr(t) for t in pl['tokens']]
    xs = range(len(toks))
    w = 0.38
    for i, (t, a, b) in enumerate(zip(toks, pl['p_T1'], pl['p_T05'])):
        right = t == 'src'
        c1 = C['data'] if right else C['memory']
        ax.bar(i - (w / 2 if k >= 2 else 0), a, width=w, color=c1)
        ax.text(i - (w / 2 if k >= 2 else 0), a + 0.015, f'{100 * a:.0f}' if a >= 0.01 else '<1', ha='center', fontsize=10, color=c1, fontweight='bold')
        ax.bar(i + w / 2, b if k >= 2 else 0, width=w, color=C['compute'], alpha=1 if k >= 2 else 0)
        ax.text(i + w / 2, b + 0.015, f'{100 * b:.0f}' if b >= 0.01 else '<1', ha='center', fontsize=10, color=C['compute'], fontweight='bold', alpha=1 if k >= 2 else 0)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(toks, fontsize=10.5)
    ax.get_xticklabels()[toks.index('src')].set_color(C['data'])
    ax.get_xticklabels()[toks.index('src')].set_fontweight('bold')
    ax.set_ylim(0, 0.82)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.text(0.99, 0.95, 'T = 1 (blue)', transform=ax.transAxes, ha='right', fontsize=12, color=C['memory'], fontweight='bold')
    ax.text(0.99, 0.82, 'T = 0.5 (orange)', transform=ax.transAxes, ha='right', fontsize=12, color=C['compute'], fontweight='bold', alpha=1 if k >= 2 else 0)
    ax.text(0.99, 0.69, 'top-p 0.90 at T = 1 keeps 32 tokens', transform=ax.transAxes, ha='right', fontsize=12, color=C['text'], fontweight='bold', alpha=1 if k >= 3 else 0)
    ax.annotate('the file the\nprompt names', (toks.index('src') - 0.2, pl['p_T1'][2] + 0.06), xytext=(3.4, 0.33), fontsize=11, color=C['data'], fontweight='bold', arrowprops=dict(arrowstyle='-|>', color=C['data'], lw=1.4))
    ax.set_ylabel('Probability (%)', fontsize=11)
    sp.save(fig, out / f'probs_step{k}.svg')
for k in (1, 2, 3):
    probs(k)

def nm21(k):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    p1, p5 = ([0.7839943536, 0.1749327857, 0.03903278048, 0.00194332771, 9.675258958e-05], [0.950324649, 0.04731387827, 0.002355619292, 5.838996448e-06, 1.447342515e-08])
    w = 0.38
    for i in range(5):
        ax.bar(i - w / 2, p1[i], width=w, color=C['memory'])
        ax.bar(i + w / 2, p5[i] if k >= 2 else 0, width=w, color=C['compute'], alpha=1 if k >= 2 else 0)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"v{i + 1}\n{_VALS[f'nm_z{i + 1}']}" for i in range(5)], fontsize=13)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '50', '100'], fontsize=12)
    ax.text(4.5, 1.0, 'T = 1: v1 78.4%', ha='right', fontsize=14, color=C['memory'], fontweight='bold')
    ax.text(4.5, 0.84, 'T = 0.5: v1 95.0%', ha='right', fontsize=14, color=C['compute'], fontweight='bold', alpha=1 if k >= 2 else 0)
    ax.set_ylabel('Probability (%)', fontsize=13)
    sp.save(fig, out / f'nm21_step{k}.svg')
for k in (1, 2):
    nm21(k)
fig, ax = canvas(9.0, 3.6, phys=6.0)
box(ax, 330, 270, 240, 55, V['st_l'], V['st'], '/tmp/scratch/', V['ink'], fs=14)
box(ax, 120, 140, 230, 55, V['st_l'], V['st'], 'cache_dir/', V['ink'], fs=14)
box(ax, 400, 140, 160, 55, V['st_l'], V['st'], 'build/', V['ink'], fs=14)
box(ax, 610, 140, 190, 55, V['st_l'], V['st'], 'results.db', V['ink'], fs=14)
for x in (235, 480, 705):
    ax.plot([450, x], [270, 195], color=V['st'], lw=2)
ax.add_patch(mp.FancyBboxPatch((105, 127), 260, 81, boxstyle='round,pad=0,rounding_size=10', fc='none', ec=V['ctx'], lw=3, ls='--'))
ax.add_patch(mp.FancyBboxPatch((80, 112), 745, 230, boxstyle='round,pad=0,rounding_size=12', fc='none', ec=V['bad'], lw=3))
ax.text(235, 95, 'COMPLETED\nrm -rf /tmp/scratch/cache_dir', ha='center', va='top', fontsize=12, color=V['ctx_t'], fontweight='bold', family='monospace', linespacing=1.4)
ax.text(640, 95, 'TRUNCATED\nrm -rf /tmp/scratch', ha='center', va='top', fontsize=12, color=V['bad_t'], fontweight='bold', family='monospace', linespacing=1.4)
ax.set_ylim(15, 360)
save(fig, 'rmrf.svg')
fig, ax = plt.subplots(figsize=(5.8, 1.9))
cap, rej = (4096, 96)
ax.barh(0, rej, color=C['danger'], height=0.5)
ax.barh(0, cap - rej, left=rej, color=C['neutral_l'], height=0.5, hatch='//', edgecolor=C['neutral'])
ax.text(rej + 60, 0.45, 'rejected at step 96', fontsize=11, color=C['danger'], fontweight='bold')
ax.text(rej + (cap - rej) / 2 + 60, 0, '4,000 steps (84.3 s at 21.075 ms) skipped if the cancel is honored', ha='center', va='center', fontsize=10, color=C['text'], fontweight='bold', bbox=dict(fc='white', ec='none', pad=2))
ax.set_xlim(0, cap)
ax.set_ylim(-0.5, 0.8)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Decode step (output allowance 4,096)')
sp.save(fig, out / 'stream.svg')
layers = [('Syntax and schema', 'parser, validator', 'fields and types', 'permission, values'), ('Authorization', 'grant check', 'target in the grant', 'serves the goal'), ('Static analysis', 'compiler, linter', 'analyzed properties', 'runtime behavior'), ('Behavioral tests', 'sandboxed run', 'exercised inputs', 'untested inputs')]

def ladder(k):
    fig, ax = canvas(9.0, 4.2, phys=6.4)
    ax.text(390, 395, 'Certifies', ha='center', fontsize=14, color=V['ctx_t'], fontweight='bold')
    ax.text(702, 395, 'Leaves open', ha='center', fontsize=14, color=V['bad_t'], fontweight='bold')
    for i, (name, mech, yes, no) in enumerate(layers):
        y = 300 - i * 88
        a = 1.0 if i < k else 0.0
        box(ax, 5, y, 250, 76, V['rt_l'], V['rt'], '', alpha=a)
        ax.text(130, y + 52, name, ha='center', va='center', fontsize=12.5, color=V['rt_t'], fontweight='bold', alpha=a)
        ax.text(130, y + 22, mech, ha='center', va='center', fontsize=11, color=V['gray'], alpha=a)
        box(ax, 270, y + 8, 240, 60, V['ctx_l'], V['ctx'], yes, V['ctx_t'], fs=12, bold=False, alpha=a)
        ax.text(548, y + 38, 'not', ha='center', va='center', fontsize=12, color=V['gray'], alpha=a)
        box(ax, 585, y + 8, 235, 60, V['bad_l'], V['bad'], no, V['bad_t'], fs=12, bold=False, alpha=a)
    ax.text(895, 300 + 52, 'cheap', ha='right', fontsize=11, color=V['gray'])
    ax.text(895, 300 - 3 * 88 + 2, 'costly', ha='right', fontsize=11, color=V['gray'])
    ax.annotate('', xy=(865, 300 - 3 * 88 + 28), xytext=(865, 300 + 38), arrowprops=dict(arrowstyle='-|>', color=V['gray'], lw=1.8))
    ax.set_ylim(25, 420)
    save(fig, f'ladder_step{k}.svg')
for k in (1, 2, 3, 4):
    ladder(k)
huang = [(name, _NUMS[f'huang_{key}_std'], _NUMS[f'huang_{key}_sc'], _NUMS[f'huang_{key}_orc']) for name, key in (('GSM8K', 'gsm'), ('CommonSenseQA', 'csqa'), ('HotpotQA', 'hotpot'))]

def huang_fig(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    w = 0.27
    for i, (name, std, sc, orc) in enumerate(huang):
        for j, (v, c, show) in enumerate([(std, C['neutral'], True), (sc, C['danger'], True), (orc, C['data'], k >= 2)]):
            x = i + (j - 1) * w
            ax.bar(x, v if show else 0, width=w * 0.95, color=c)
            ax.text(x, v + 2, f'{v:.1f}', ha='center', fontsize=10.5, color=c, fontweight='bold', alpha=1 if show else 0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([h[0] for h in huang], fontsize=12)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_xlim(-0.5, 2.6)
    ax.text(-0.42, 116, 'one answer', color=C['neutral'], fontsize=10.5, fontweight='bold')
    ax.text(0.42, 116, '2 rounds of self-review', color=C['danger'], fontsize=10.5, fontweight='bold')
    ax.text(1.66, 116, 'stopped by true label', color=C['data'], fontsize=11, fontweight='bold', alpha=1 if k >= 2 else 0)
    sp.save(fig, out / f'huang_step{k}.svg')
for k in (1, 2):
    huang_fig(k)

def bayes(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    rows = [('Generated', 0.18, '18'), ('Passed the check', 0.1437346437, '14.4'), ('Passed, FA halved', 0.07743216413, '7.7')]
    for i, (lab, d, pct) in enumerate(rows):
        y = 2 - i
        a = 1 if i < k else 0
        ax.barh(y, d, color=C['danger'], height=0.55, alpha=a)
        ax.barh(y, 1 - d, left=d, color=C['data_l'], height=0.55, alpha=a)
        ax.text(d + 0.01, y, f'{pct}% defective', va='center', fontsize=12, color=C['danger'], fontweight='bold', alpha=a)
        ax.text(-0.01, y, lab, ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    sp.save(fig, out / f'bayes_step{k}.svg')
for k in (1, 2, 3):
    bayes(k)
fig, ax = canvas(9.0, 3.4, phys=6.4)
ax.text(10, 305, 'Parser state: inside the string value of "path"', fontsize=13, color=V['rt_t'], fontweight='bold')
x = 10
for t, p in zip(pl['tokens'], pl['p_T1']):
    w = max(72, 15 * len(t) + 30)
    box(ax, x, 185, w, 78, V['ctx_l'], V['ctx'], f'{t}\n{100 * p:.0f}%', V['ctx_t'], fs=12)
    x += w + 8
ax.text(10, 150, 'All eight stay legal inside a string, so the mask keeps them.', fontsize=13, color=V['ctx_t'], va='top')
for i, t in enumerate(['newline', 'tab', '\\q']):
    box(ax, 10 + i * 125, 30, 112, 60, V['bad_l'], V['bad'], t, V['bad_t'], fs=13)
ax.text(395, 60, 'Illegal here: logit −∞', fontsize=13, color=V['bad_t'], va='center')
ax.set_ylim(15, 340)
save(fig, 'mask.svg')
_sp_save = sp.save

def _plain_ticks_save(fig, path):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
    _sp_save(fig, path)
sp.save = _plain_ticks_save
sp.hbars(out / 'laptop.svg', [('prompt', 0.001003086202 * 1000.0, 'memory', '1.00 ms'), ('output', 0.03507180795 * 1000.0, 'compute', '35.1 ms')], 'ms', log=True, size=(3.6, 1.9), xlabel='ms per token (log scale)')
sp.hbars(out / 'azure.svg', [('prompt', 1469.0, 'memory', '1,469'), ('output', 13.0, 'compute', '13')], 'tokens', log=True, size=(3.6, 1.9), xlabel='median tokens per request (log)')
sp.save = _sp_save

def resend60(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    S0, D = (8000, 1500)
    h1, h2 = (30, 60)
    t = list(range(1, h2 + 1))
    cum_resent = [calc_append_only_context(S0, D, n)[1] / 1000000.0 for n in t]
    cum_fresh = [calc_reused_prefix_prompt_tokens(n, S0, D) / 1000000.0 for n in t]
    ax.plot(t[:h1], cum_resent[:h1], color=C['danger'], lw=3)
    ax.plot(t[h1 - 1:], cum_resent[h1 - 1:], color=C['danger'], lw=3, alpha=1 if k >= 2 else 0)
    ax.plot(t, cum_fresh, color=C['data'], lw=3, alpha=1 if k >= 3 else 0)
    ax.plot([h1], [cum_resent[h1 - 1]], 'o', color=C['danger'], ms=7)
    ax.text(h1 + 1.5, cum_resent[h1 - 1], '30 turns: 892.5k', ha='left', va='top', fontsize=12, color=C['danger'], fontweight='bold')
    ax.text(h2 - 1.5, cum_resent[-1], '60 turns: 3,135k (3.5×)', ha='right', va='bottom', fontsize=12, color=C['danger'], fontweight='bold', alpha=1 if k >= 2 else 0)
    ax.text(h2 - 1.5, cum_fresh[-1] + 0.12, 'fresh prefill 96.5k (1.9×)', ha='right', va='bottom', fontsize=12, color=C['data'], fontweight='bold', alpha=1 if k >= 3 else 0)
    ax.set_xlim(1, h2)
    ax.set_ylim(0, 3.6)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Cumulative input\n(million tokens)', fontsize=11)
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    sp.save(fig, out / f'resend60_step{k}.svg')
for k in (1, 2, 3):
    resend60(k)

def deadline(k):
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    rows = [('Whole file', 0.5957655382, 38.9880597, '39.6'), ('Local edit', 0.5793552299, 1.685970149, '2.3')]
    deadline = 15.0
    for i, (lab, a, b, total) in enumerate(rows):
        y = 1 - i
        al = 1 if i < k else 0
        ax.barh(y, a, color=C['memory_l'], height=0.55, alpha=al, edgecolor=C['memory'])
        ax.barh(y, b, left=a, color=C['memory'], height=0.55, alpha=al)
        ax.text(a + b + 0.6, y, f'{total} s', va='center', fontsize=13, color=C['memory'], fontweight='bold', alpha=al)
        ax.text(-0.5, y, lab, ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
    ax.axvline(deadline, color=C['danger'], lw=2, ls='--')
    ax.axvspan(deadline, 46, color=C['danger_l'], alpha=0.5, zorder=0)
    ax.text(deadline + 0.4, 1.55, '15 s deadline', fontsize=11, color=C['danger'], fontweight='bold')
    ax.text(45, -0.42, 'light: prompt   dark: output', ha='right', fontsize=10, color=C['memory'])
    ax.set_xlim(0, 46)
    ax.set_ylim(-0.5, 1.85)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Service time (s), reference floors')
    sp.save(fig, out / f'deadline_step{k}.svg')
for k in (1, 2):
    deadline(k)
import datetime as _dt

def day(m, d):
    return (_dt.date(2025, m, d) - _dt.date(2025, 8, 1)).days

def hook_timeline(k, size=(6.0, 2.4), fs=13.5, labels=True):
    fig, ax = plt.subplots(figsize=size)
    lanes = [('Routing to\nlong-context servers', day(8, 5), day(9, 4), C['memory']), ('Corrupted output\n(misconfigured servers)', day(8, 25), day(9, 2), C['compute']), ('Miscompiled\napproximate top-k', day(8, 26), day(9, 12), C['network'])]
    for i, (lab, a, b, c) in enumerate(lanes):
        y = 2 - i
        al = 1 if i < k else 0
        ax.plot([a, b], [y, y], color=c, lw=6, solid_capstyle='butt', alpha=al)
        ax.plot([a], [y], 'o', ms=9, color=c, alpha=al)
        ax.plot([b], [y], 'o', ms=9, color=C['data'], alpha=al)
        if labels:
            ax.text(-1, y, lab, ha='right', va='center', fontsize=fs - 1, color=c, fontweight='bold', alpha=1)
    ax.plot([day(8, 31)], [2], 'o', ms=12, color=C['danger'], alpha=1 if k >= 1 else 0)
    if labels:
        ax.annotate('16% of requests\nat the worst hour', (day(8, 31), 2), xytext=(day(8, 31), 2.55), ha='center', fontsize=fs - 1, color=C['danger'], fontweight='bold')
        ax.text(day(8, 5), 2.3, '0.8% at first', ha='left', fontsize=fs - 1, color=C['memory'])
    ticks = [(8, 5), (8, 25), (9, 2), (9, 12)]
    ax.set_xticks([day(m, d) for m, d in ticks])
    ax.set_xticklabels(['Aug 5', 'Aug 25', 'Sep 2', 'Sep 12'], fontsize=fs - 1)
    ax.set_xlim(day(8, 1), day(9, 15))
    ax.set_ylim(-0.6, 3.1)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    sp.save(fig, out / 'hook_timeline.svg')
hook_timeline(3)
print('figures written')
