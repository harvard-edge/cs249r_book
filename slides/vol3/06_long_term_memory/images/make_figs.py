#!/usr/bin/env python3
"""Figures for Lecture 6 (Volume III), Long-Term Memory.

Every number drawn here comes from calcs.py, which reads MLSysIM: the
Scenarios.* the chapter also reads (Napkin Math 6.1 to 6.4, Checkpoint 6.4),
Literature (Jimenez et al. 2024, SWE-bench; Yang et al. 2024, SWE-agent; Zou
et al. 2025, PoisonedRAG), and this deck's own measurement (scan_bench.py,
scan_bench.csv). Plotted geometry uses the scenario values and results;
labels use the formatted strings the slides print (deckvals val/num).
Diagram coordinates, axis ranges, tick positions, and label offsets are
layout; the diagrams' example records (paths, counts in the inverted index,
"retrieved 30 times") are the chapter's illustrative text.

Diagrams are hand-written SVG in the Volume III palette (volumes/vol3/
palette.md): green memory and context, violet storage, amber generated or
tool-side records, red faults, slate structure. Full-width diagrams use a
900 x 500 canvas with text of 20 px or more; column diagrams 540 wide.
Plots use the shared slideplot helper.

Run from this folder: python3 make_figs.py
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
_VALS = {'mof_days': '90', 'mof_damages': '650.88', 'mof_interest': '36.14', 'mof_fees': '125', 'mof_total': '812.02', 'rp_window': '32,768', 'rp_reserve': '2,048', 'rp_other': '24,576', 'rp_envelope': '128', 'rp_body': '512', 'rp_meta': '64', 'rp_allowance': '6,144', 'rp_record': '576', 'rp_records': '10', 'rp_used': '5,888', 'rp_remaining': '256', 'rp_body_only': '12', 'rp_body_only_used': '7,040', 'rp_overshoot': '896', 'rp_meta2': '128', 'rp_meta2_records': '9', 'swb_ctx1': '13,000', 'swb_rec1': '29.58', 'swb_res1': '1.96', 'swb_ctx2': '27,000', 'swb_rec2': '44.41', 'swb_res2': '1.87', 'swb_ctx3': '50,000', 'swb_rec3': '51.06', 'swb_res3': '1.22', 'swa_rag': '2.67', 'swa_shell': '11.00', 'swa_agent': '18.00', 'swa_ratio': '6.7', 'swa_cost_lo': '8', 'swa_cost_hi': '13', 'bm_short': '14.58', 'bm_long': '8.46', 'bm_short_tf_words': 'one', 'bm_long_tf_words': 'six', 'doc_shared': '7', 'doc_union': '13', 'sc_n_m': '10', 'sc_dim': '768', 'sc_bytes': '2', 'sc_bw': '50', 'sc_peak': '1', 'sc_gb': '15.36', 'sc_gflop': '15.36', 'sc_ms': '307.2', 'sc_math_ms': '15.36', 'sc_intensity': '1', 'sc_ridge': '20', 'sc_traj_s': '6.1', 'sc_budget_ms': '10', 'sc_budget_vec': '325,521', 'sc_budget_vec_k': '326,000', 'sc_filtered': '10,000', 'sc_filtered_ms': '0.31', 'lap_bw': '126', 'lap_gflops_lo': '44', 'lap_gflops_hi': '64', 'lap_budget_k': '411,000', 'lap_bytes': '4', 'lap_top_m': '3', 'lap_top_ms': '72', 'ddr_rate': '4,800', 'ddr_bytes': '8', 'ddr_bw': '38.4', 'ann_links': '32', 'ann_link_bytes_words': 'eight', 'ann_graph_gb': '2.56', 'ann_total_gb': '17.92', 'ann_sub': '64', 'ann_code_bytes_words': 'one', 'ann_codes_gb': '0.64', 'ann_ratio': '24', 'ann_lut_kb': '65.5', 'ann_vec_bytes': '1,536', 'gn_allowance': '6,144', 'gn_depth': '3', 'gn_shallow_depth': '2', 'gn_children': '3', 'gn_node': '180', 'gn_env': '100', 'gn_nodes0': '1', 'gn_tok0': '280', 'gn_level0': '1', 'gn_nodes1': '4', 'gn_tok1': '820', 'gn_level1': '3', 'gn_nodes2': '13', 'gn_tok2': '2,440', 'gn_level2': '9', 'gn_nodes3': '40', 'gn_tok3': '7,300', 'gn_level3': '27', 'gn_fit': '33', 'pr_n': '5', 'pr_kb1': '2.7', 'pr_asr1': '97', 'pr_kb2': '5.2', 'pr_asr2': '99', 'pr_kb3': '8.8', 'pr_asr3': '91', 'pr_per_m': '1.86', 'pr_per_m_round': '2', 'pf_shared': '5', 'pf_words': '6', 'ix_files': '8', 'ix_chunks': '16', 'ix_cands': '50', 'ix_local': '88', 'ix_remote': '210', 'ix_serial': '1.26', 'ix_write_us': '0.4', 'ix_check_us': '1.5', 'ix_ratio': '220,000', 'st_ratio_a': '0.36', 'st_ratio_b': '8.52', 'pc_pa_pct': '50', 'pc_pb_pct': '60', 'pc_pa': '0.5', 'pc_pb': '0.6', 'pc_maint': '0.1', 'pc_attempt_b': '1.1', 'pc_ceff_a': '2.00', 'pc_ceff_b': '1.83', 'pc_change': '8.3', 'pc_breakeven': '0.55', 'liu_docs': '20'}
_NUMS = {'mof_days': 90.0, 'mof_damages': 650.88, 'mof_interest': 36.14, 'mof_fees': 125.0, 'mof_total': 812.02, 'rp_window': 32768.0, 'rp_reserve': 2048.0, 'rp_other': 24576.0, 'rp_envelope': 128.0, 'rp_body': 512.0, 'rp_meta': 64.0, 'rp_allowance': 6144.0, 'rp_record': 576.0, 'rp_records': 10.0, 'rp_used': 5888.0, 'rp_remaining': 256.0, 'rp_body_only': 12.0, 'rp_body_only_used': 7040.0, 'rp_overshoot': 896.0, 'rp_meta2': 128.0, 'rp_meta2_records': 9.0, 'swb_ctx1': 13000.0, 'swb_rec1': 29.58, 'swb_res1': 1.96, 'swb_ctx2': 27000.0, 'swb_rec2': 44.41, 'swb_res2': 1.87, 'swb_ctx3': 50000.0, 'swb_rec3': 51.06, 'swb_res3': 1.22, 'swa_rag': 2.67, 'swa_shell': 11.0, 'swa_agent': 18.0, 'swa_ratio': 6.7, 'swa_cost_lo': 8.0, 'swa_cost_hi': 13.0, 'bm_short': 14.58, 'bm_long': 8.46, 'doc_shared': 7.0, 'doc_union': 13.0, 'sc_n_m': 10.0, 'sc_dim': 768.0, 'sc_bytes': 2.0, 'sc_bw': 50.0, 'sc_peak': 1.0, 'sc_gb': 15.36, 'sc_gflop': 15.36, 'sc_ms': 307.2, 'sc_math_ms': 15.36, 'sc_intensity': 1.0, 'sc_ridge': 20.0, 'sc_traj_s': 6.1, 'sc_budget_ms': 10.0, 'sc_budget_vec': 325521.0, 'sc_budget_vec_k': 326000.0, 'sc_filtered': 10000.0, 'sc_filtered_ms': 0.31, 'lap_bw': 126.0, 'lap_gflops_lo': 44.0, 'lap_gflops_hi': 64.0, 'lap_budget_k': 411000.0, 'lap_bytes': 4.0, 'lap_top_m': 3.0, 'lap_top_ms': 72.0, 'ddr_rate': 4800.0, 'ddr_bytes': 8.0, 'ddr_bw': 38.4, 'ann_links': 32.0, 'ann_graph_gb': 2.56, 'ann_total_gb': 17.92, 'ann_sub': 64.0, 'ann_codes_gb': 0.64, 'ann_ratio': 24.0, 'ann_lut_kb': 65.5, 'ann_vec_bytes': 1536.0, 'gn_allowance': 6144.0, 'gn_depth': 3.0, 'gn_shallow_depth': 2.0, 'gn_children': 3.0, 'gn_node': 180.0, 'gn_env': 100.0, 'gn_nodes0': 1.0, 'gn_tok0': 280.0, 'gn_level0': 1.0, 'gn_nodes1': 4.0, 'gn_tok1': 820.0, 'gn_level1': 3.0, 'gn_nodes2': 13.0, 'gn_tok2': 2440.0, 'gn_level2': 9.0, 'gn_nodes3': 40.0, 'gn_tok3': 7300.0, 'gn_level3': 27.0, 'gn_fit': 33.0, 'pr_n': 5.0, 'pr_kb1': 2.7, 'pr_asr1': 97.0, 'pr_kb2': 5.2, 'pr_asr2': 99.0, 'pr_kb3': 8.8, 'pr_asr3': 91.0, 'pr_per_m': 1.86, 'pr_per_m_round': 2.0, 'pf_shared': 5.0, 'pf_words': 6.0, 'ix_files': 8.0, 'ix_chunks': 16.0, 'ix_cands': 50.0, 'ix_local': 88.0, 'ix_remote': 210.0, 'ix_serial': 1.26, 'ix_write_us': 0.4, 'ix_check_us': 1.5, 'ix_ratio': 220000.0, 'st_ratio_a': 0.36, 'st_ratio_b': 8.52, 'pc_pa_pct': 50.0, 'pc_pb_pct': 60.0, 'pc_pa': 0.5, 'pc_pb': 0.6, 'pc_maint': 0.1, 'pc_attempt_b': 1.1, 'pc_ceff_a': 2.0, 'pc_ceff_b': 1.83, 'pc_change': 8.3, 'pc_breakeven': 0.55, 'liu_docs': 20.0}
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
C = sp.C
plt = sp.plt
out = Path(__file__).parent
ROLE = {'memory': ('#dcfce7', '#16a34a', '#15803d'), 'storage': ('#f3e8ff', '#7c3aed', '#6d28d9'), 'tools': ('#fef3c7', '#d97706', '#b45309'), 'compute': ('#e0f2fe', '#0284c7', '#0369a1'), 'fault': ('#fee2e2', '#dc2626', '#b91c1c'), 'struct': ('#f8fafc', '#94a3b8', '#0f172a'), 'dim': ('#f1f5f9', '#cbd5e1', '#94a3b8')}
FONT = 'Helvetica, Arial, sans-serif'
MONO = 'Courier New, Courier, monospace'

class Svg:

    def __init__(self, w=900, h=500):
        self.w, self.h, self.parts = (w, h, [])
        self.parts.append('<defs>' + ''.join((f'<marker id="a-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{v[1]}"/></marker>' for k, v in ROLE.items())) + '</defs>')

    def rect(self, x, y, w, h, role='struct', dash=False, sw=2, rx=8):
        f, s, _ = ROLE[role]
        d = ' stroke-dasharray="7,5"' if dash else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{f}" stroke="{s}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=20, role='struct', bold=False, anchor='start', mono=False, color=None):
        c = color or ROLE[role][2]
        fam = MONO if mono else FONT
        wt = ' font-weight="bold"' if bold else ''
        s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" fill="{c}" text-anchor="{anchor}"{wt}>{s}</text>')

    def lines(self, x, y, rows, size=20, role='struct', gap=None, **kw):
        gap = gap or int(size * 1.25)
        for i, r in enumerate(rows):
            self.text(x, y + i * gap, r, size=size, role=role, **kw)

    def arrow(self, x1, y1, x2, y2, role='struct', sw=2.5, dash=False):
        s = ROLE[role][1]
        d = ' stroke-dasharray="6,5"' if dash else ''
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{s}" stroke-width="{sw}" marker-end="url(#a-{role})"{d}/>')

    def save(self, path):
        body = '\n'.join(self.parts)
        Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">\n<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')
ROLES = [('Source record', 'storage', ['src/pool.py at r18;', "the user's stated preference"], ['source, version,', 'access conditions']), ('Observation log', 'memory', ['waiter test failed at r17,', 'passed at r18 (config c18)'], ['what, when, under', 'which conditions']), ('Agent-written', 'tools', ['note: resume at the', 'integration checks'], ['supporting records,', 'scope, uncertainty']), ('Derived index', 'compute', ['search entries over files,', 'logs, and notes'], ['the source generation', 'it was built from'])]

def roles_fig(path, k):
    s = Svg(900, 395)
    s.text(18, 26, 'Role', size=21, bold=True, color='#475569')
    s.text(283, 26, 'Repair-agent example', size=21, bold=True, color='#475569')
    s.text(643, 26, 'A later read must keep', size=21, bold=True, color='#475569')
    for i, (name, role, ex, keep) in enumerate(ROLES):
        if i >= k:
            continue
        y = 40 + i * 89
        s.rect(12, y, 250, 74, role, sw=2.5)
        if name == 'Agent-written':
            s.text(26, y + 31, name, size=23, role=role, bold=True)
            s.text(26, y + 60, 'memory', size=23, role=role, bold=True)
        else:
            s.text(26, y + 45, name, size=23, role=role, bold=True)
        s.rect(275, y, 352, 74, 'struct')
        s.lines(288, y + 31, ex, size=21, gap=27)
        s.rect(638, y, 252, 74, 'struct')
        s.lines(651, y + 31, keep, size=21, gap=27)
        if i > 0:
            s.arrow(137, y - 1, 137, y - 14, role='struct', sw=2.5)
    s.save(path)
for k in range(1, 5):
    roles_fig(out / f'roles_step{k}.svg', k)

def contract_fig(path, k):
    s = Svg(900, 395)
    s.rect(8, 8, 285, 315, 'storage', sw=2.5)
    s.text(150, 40, 'Runtime sets bounds', size=23, role='storage', bold=True, anchor='middle')
    s.lines(24, 80, ['need: notification logic', 'scope: repair workspace', 'principal: repair agent', 'version: r18', 'limits: items, tokens', 'deadline, work caps'], size=21, gap=43)
    if k >= 2:
        s.arrow(295, 165, 320, 165, role='struct')
        s.rect(322, 8, 250, 315, 'compute', sw=2.5)
        s.text(447, 40, 'Service searches', size=23, role='compute', bold=True, anchor='middle')
        s.lines(338, 80, ['checks access', 'picks an index', 'stops at a limit', 'cancels on timeout'], size=21, gap=43)
    if k >= 3:
        s.arrow(574, 165, 599, 165, role='struct')
        s.rect(601, 8, 291, 315, 'memory', sw=2.5)
        s.text(746, 40, 'Envelope reports', size=23, role='memory', bold=True, anchor='middle')
        s.lines(615, 80, ['status: complete,', '  partial, unavailable', 'searched: repair, r18', 'each item: path, lines,', '  revision, hash, role', 'score: method named'], size=21, gap=43)
        s.rect(8, 335, 884, 54, 'fault', sw=2)
        s.text(450, 370, 'Zero hits prove absence only for a complete scan of the declared scope.', size=22, role='fault', bold=True, anchor='middle')
    s.save(path)
for k in range(1, 4):
    contract_fig(out / f'contract_step{k}.svg', k)

def fit_fig(path, k):
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    a = 6144.0
    body, meta, env = (512.0, 64.0, 128.0)
    body_only, fit = (12.0, 10.0)
    y1, y0 = (1, 0)
    ax.barh(y1, body_only * body, color=C['memory'], height=0.55)
    if k >= 2:
        ax.barh(y1, body_only * meta + env, left=body_only * body, color=C['danger'], height=0.55)
        ax.text(7040.0 + 80, y1, '7,040: over by 896', va='center', fontsize=12, fontweight='bold', color=C['danger'])
        ax.barh(y0, fit * body, color=C['memory'], height=0.55)
        ax.barh(y0, fit * meta + env, left=fit * body, color=C['network'], height=0.55)
        ax.text(a + 80, y0, '5,888: fits, 256 left', va='center', fontsize=12, fontweight='bold', color=C['data'])
    else:
        ax.text(body_only * body + 80, y1, '???', va='center', fontsize=14, fontweight='bold', color=C['danger'])
    ax.text(-60, y1, 'Count bodies:\n12 records', ha='right', va='center', fontsize=12, fontweight='bold')
    ax.text(-60, y0, 'Count serialized:\n10 records', ha='right', va='center', fontsize=12, fontweight='bold', alpha=1.0 if k >= 2 else 0.0)
    ax.axvline(a, color=C['text'], lw=1.6, ls='--')
    ax.text(a, 1.5, ' allowance 6,144', fontsize=11, fontweight='bold', va='bottom', ha='center')
    ax.set_xlim(0, 10200)
    ax.set_ylim(-0.5, 1.75)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Serialized tokens (bodies blue; metadata and envelope red or violet)')
    sp.save(fig, path)
for k in (1, 2):
    fit_fig(out / f'fit_step{k}.svg', k)
SWEB_CONTEXT = [13.0, 27.0, 50.0]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.5))
for ax, key, ck, lab in ((a1, 'swb_rec', 'memory', 'BM25 recall of edited files (%)'), (a2, 'swb_res', 'danger', 'Issues resolved (%)')):
    ys = [_NUMS[f'{key}{i}'] for i in range(1, len(SWEB_CONTEXT) + 1)]
    ax.plot(SWEB_CONTEXT, ys, '-o', color=C[ck], lw=2.5, ms=8)
    for i, (x, y) in enumerate(zip(SWEB_CONTEXT, ys), start=1):
        below = key == 'swb_rec' and i == 1
        ax.annotate(_VALS[f'{key}{i}'], (x, y), xytext=(18, -12) if below else (0, 9), textcoords='offset points', ha='center', fontsize=12, fontweight='bold', color=C[ck])
    ax.set_xticks(SWEB_CONTEXT)
    ax.set_xticklabels([f'{c:.0f}K' for c in SWEB_CONTEXT])
    ax.set_xlim(5, 58)
    ax.set_ylim(min(ys) - (max(ys) - min(ys)) * 0.3, max(ys) + (max(ys) - min(ys)) * 0.35)
    ax.set_title(lab, fontsize=12, fontweight='bold', color=C[ck])
    ax.set_xlabel('Retrieved context, tokens')
fig.tight_layout()
sp.save(fig, out / 'swebench_context.svg')
sp.hbars(out / 'swe_agent.svg', [('Pre-staged\n(BM25)', 2.67, 'neutral', '2.67%'), ('Shell\ncommands', 11.0, 'memory', '11.00%'), ('Search tools\n(SWE-agent)', 18.0, 'data', '18.00%')], '', log=False, size=(3.8, 2.3), xlabel='SWE-bench Lite resolved (%), same model')

def inverted_fig(path):
    s = Svg(540, 345)
    s.text(10, 30, 'term', size=20, bold=True, color='#475569')
    s.text(155, 30, 'postings: file, count', size=20, bold=True, color='#475569')
    rows = [('notify', [('src/pool.py', 2), ('r17.log', 1)]), ('waiter', [('test_pool.py', 3), ('src/pool.py', 1)]), ('return', [('src/pool.py', 4), ('pool.md', 1)])]
    for i, (term, posts) in enumerate(rows):
        y = 48 + i * 82
        s.rect(10, y, 115, 56, 'compute')
        s.text(67, y + 36, term, size=21, role='compute', anchor='middle', mono=True, bold=True)
        s.arrow(127, y + 28, 150, y + 28)
        x = 152
        for name, cnt in posts:
            red = name == 'r17.log'
            s.rect(x, y, 186, 56, 'fault' if red else 'memory')
            s.text(x + 93, y + 36, f'{name}  {cnt}', size=20, role='fault' if red else 'memory', anchor='middle')
            x += 194
    s.text(270, 330, 'A match marks an occurrence; its role is unknown.', size=21, role='fault', bold=True, anchor='middle')
    s.save(path)
inverted_fig(out / 'inverted_index.svg')

def scan_floor(path, k):
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    xs = np.logspace(3, 8, 100)
    ms_per_vector = (Q_(1536, 'B') / Q_(50.0, 'GB / s')).to('ms').magnitude
    budget_ms = 10.0
    ax.plot(xs, xs * ms_per_vector, color=C['memory'], lw=2.5)
    ax.plot([10000000], [307.2], 'o', ms=11, color=C['danger'], mec='white')
    ax.annotate('10 million: 307.2 ms', (10000000, 307.2), xytext=(-12, 4), textcoords='offset points', ha='right', fontsize=12, fontweight='bold', color=C['danger'])
    if k >= 2:
        ax.axhline(budget_ms, color=C['neutral'], lw=1.5, ls='--')
        ax.text(1200.0, budget_ms * 1.25, '10 ms search budget', fontsize=11, color=C['neutral'], fontweight='bold')
        t_filtered = 0.3072
        ax.plot([10000], [t_filtered], 'o', ms=11, color=C['data'], mec='white')
        ax.annotate('10,000: 0.31 ms', (10000, t_filtered), xytext=(10, -14), textcoords='offset points', fontsize=12, fontweight='bold', color=C['data'])
        ax.axvline(325520.8333, color=C['neutral'], lw=1.2, ls=':')
        ax.text(325520.8333 * 1.25, 0.02, 'ceiling about 326,000', fontsize=11, color=C['text'], fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1000.0, 100000000.0)
    ax.set_ylim(0.01, 5000)
    ax.set_xlabel('Eligible vectors (768 dims, FP16)')
    ax.set_ylabel('Scan floor (ms)')
    ax.text(0.01, 0.97, 'derived, 50 GB/s host', transform=ax.transAxes, ha='left', va='top', fontsize=10, color=C['neutral'])
    sp.save(fig, path)
for k in (1, 2):
    scan_floor(out / f'scan_floor_step{k}.svg', k)

def scan_measured(path):
    import numpy as np
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ns = [10000, 30000, 100000, 300000, 1000000, 3000000]
    ts = [0.303, 1.04, 2.89, 7.373, 27.318, 71.896]
    xs = np.logspace(3.8, 6.7, 50)
    ax.plot(xs, xs * 768 * 4 / 126375047600.0 * 1000.0, color=C['memory'], lw=2, label='fit')
    ax.plot(ns, ts, 'o', ms=9, color=C['compute'], mec='white', zorder=5)
    ax.text(13000.0, 30, 'fit: 126 GB/s\nstreamed', fontsize=13, color=C['memory'], fontweight='bold')
    ax.axhline(10.0, color=C['neutral'], lw=1.5, ls='--')
    ax.text(8000.0, 10.0 * 1.25, '10 ms budget', fontsize=12, color=C['neutral'], fontweight='bold')
    ax.annotate('3 million: 72 ms', (ns[-1], ts[-1]), xytext=(-8, 8), textcoords='offset points', ha='right', fontsize=13, fontweight='bold', color=C['compute'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(7000.0, 6000000.0)
    ax.set_ylim(0.1, 300)
    ax.set_xlabel('Vectors (768 dims, FP32)', fontsize=13)
    ax.set_ylabel('Measured scan (ms)', fontsize=13)
    ax.tick_params(labelsize=13)
    sp.save(fig, path)
scan_measured(out / 'scan_measured.svg')
sp.hbars(out / 'ann_bytes.svg', [('Exact scan', 15.36, 'memory', '15.36 GB of vectors'), ('HNSW', 17.92, 'network', '17.92 GB with links'), ('IVF-PQ', 0.64, 'data', '0.64 GB of codes')], '', log=False, size=(5.6, 1.9), xlabel='Resident bytes, 10 million 768-dim FP16 vectors (derived)')

def depth_fig(path, k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    cols = [C['data_l'], C['data_l'], C['data'], C['danger']]
    last = 3
    for d, (n, t) in enumerate(zip([1, 4, 13, 40], [280, 820, 2440, 7300])):
        shown = d < last or k >= 2
        ax.bar(d, t, color=cols[d], width=0.62, alpha=1.0 if shown else 0.0)
        if shown:
            ax.text(d, t + 150, {1: '280', 4: '820', 13: '2,440', 40: '7,300'}[n], ha='center', fontsize=13, fontweight='bold', color=C['danger'] if d == last else C['text'])
        else:
            ax.text(d, 900, '???', ha='center', fontsize=15, fontweight='bold', color=C['danger'])
        ax.text(d, -1050, f"depth {d}\n{ {1: '1', 4: '4', 13: '13', 40: '40'}[n]} node{('' if n == 1 else 's')}", ha='center', va='top', fontsize=12)
    allowance = 6144.0
    ax.axhline(allowance, color=C['text'], lw=1.6, ls='--')
    ax.text(-0.45, allowance + 180, 'allowance 6,144 tokens', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 8300)
    ax.set_ylabel('Serialized tokens')
    ax.text(-0.5, 3300, '3 children per node\n180 tokens per node\n(illustrative)', ha='left', fontsize=11, color=C['neutral'])
    sp.save(fig, path)
for k in (1, 2):
    depth_fig(out / f'depth_step{k}.svg', k)

def write_fig(path, k):
    s = Svg(900, 395)
    rows = [('1  Commit the raw', 'observation', 'memory', ['tool output, revision, task id: durable before', 'any summary or embedding runs']), ('2  Admission gate', '', 'storage', ['permission and consent, record kind and scope,', 'evidence; a fidelity check for prose']), ('3  Identity, then', 'update', 'tools', ['same scope: supersede if its base version is', 'unchanged; other scope: keep both; unclear: flag']), ('4  Only then', 'deduplicate', 'compute', ['similarity may merge storage only when the', 'new record changes no state and no scope'])]
    shown = {1: 1, 2: 2, 3: 4}[k]
    for i, (h1, h2, role, det) in enumerate(rows[:shown]):
        y = 6 + i * 82
        s.rect(10, y, 265, 68, role, sw=2.5)
        s.text(24, y + 29 if h2 else y + 42, h1, size=22, role=role, bold=True)
        if h2:
            s.text(50, y + 57, h2, size=22, role=role, bold=True)
        s.rect(288, y, 602, 68, 'struct')
        s.lines(302, y + 29, det, size=21, gap=26)
        if i > 0:
            s.arrow(142, y - 1, 142, y - 13)
    if k >= 3:
        s.rect(10, 334, 880, 57, 'fault', sw=2)
        s.text(450, 358, '10:00 to 11:00, same calendar: supersede. Similarity-first', size=21, role='fault', bold=True, anchor='middle')
        s.text(450, 383, 'deduplication drops it (5 of 6 words shared).', size=21, role='fault', bold=True, anchor='middle')
    s.save(path)
for k in (1, 2, 3):
    write_fig(out / f'write_step{k}.svg', k)

def deletion_fig(path):
    s = Svg(540, 340)
    s.rect(8, 6, 524, 46, 'fault', sw=2.5)
    s.text(270, 37, '1  Tombstone: exclude from every read', size=21, role='fault', bold=True, anchor='middle')
    s.text(8, 80, '2  Find and purge each derived copy', size=20, bold=True, color='#475569')
    copies = ['primary record', 'keyword postings', 'vector entry', 'episode summary', 'cached prompt prefix', 'prompt template']
    for i, c in enumerate(copies):
        x = 8 + i % 2 * 266
        y = 92 + i // 2 * 52
        s.rect(x, y, 258, 44, 'memory')
        s.text(x + 129, y + 29, c, size=20, role='memory', anchor='middle')
    s.rect(8, 252, 524, 82, 'dim', dash=True)
    s.text(20, 277, 'Outside the perimeter; the receipt says so:', size=20, bold=True, color='#64748b')
    s.lines(20, 302, ['cold backups under a retention period,', 'tokens already sent to a model provider'], size=20, gap=25, color='#64748b')
    s.save(path)
deletion_fig(out / 'deletion_copies.svg')
sp.hbars(out / 'poisonedrag.svg', [('NQ\n2.7M texts', 97.0, 'danger', '97%'), ('HotpotQA\n5.2M texts', 99.0, 'danger', '99%'), ('MS-MARCO\n8.8M texts', 91.0, 'danger', '91%')], '', log=False, size=(5.4, 2.4), xlabel='Target questions answered as the attacker chose (%), 5 texts each')

def cascade_fig(path, k):
    s = Svg(900, 395)
    chain = [('Observation: reproducer passed at r18 (c18)', 'memory'), ('Lesson: timeouts always come from a missed notify', 'tools'), ('Episode summary repeats the lesson', 'compute'), ('Weekly report cites the summary', 'compute'), ('Cross-project heuristic', 'compute')]
    for i, (label, role) in enumerate(chain):
        y = 6 + i * 78
        r = 'fault' if k >= 2 and i >= 1 else role
        s.rect(10, y, 570, 60, r, sw=2.5)
        s.text(24, y + 38, label, size=22, role=r, bold=i == 1)
        if i > 0:
            s.arrow(120, y - 1, 120, y - 17, role='fault' if k >= 2 and i >= 2 else 'struct')
    s.text(598, 102, 'unsupported generalization', size=20, role='tools', bold=True)
    s.text(598, 190, 'retrieved 30 times:', size=20, color='#475569')
    s.text(598, 215, 'no new evidence', size=20, color='#475569')
    if k >= 2:
        s.rect(595, 245, 297, 140, 'fault', sw=2)
        s.lines(610, 278, ['Invalidate the lesson:', 'every record derived', 'from it is marked', 'suspect or retracted'], size=21, role='fault', gap=27, bold=True)
    s.save(path)
for k in (1, 2):
    cascade_fig(out / f'cascade_step{k}.svg', k)

def generations_fig(path):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    gens = ['r16', 'r17', 'r18', 'r19']
    rows = [('summary of r16 runs  [r16, r18)', 0, 2, 'neutral'), ('pool.py chunk, r17 text  [r17, r18)', 1, 2, 'neutral'), ('pool.py chunk, r18 text  [r18, open)', 2, 3.9, 'data')]
    for i, (lab, b, d, ck) in enumerate(rows):
        y = 2 - i
        ax.plot([b, d], [y, y], color=C[ck], lw=9, solid_capstyle='butt')
        ax.text(-0.15, y + 0.3, lab, fontsize=14, color=C[ck], fontweight='bold', bbox=dict(fc='white', ec='none', pad=1), zorder=5)
    ax.axvline(2.25, color=C['data'], lw=2, ls='--')
    ax.axvline(1.25, color=C['neutral'], lw=1.5, ls=':')
    ax.text(2.32, -0.62, 'query at r18:\nadmits one', fontsize=14, color=C['data'], fontweight='bold')
    ax.text(1.18, -0.62, 'query at r17:\nadmits two', fontsize=14, color=C['neutral'], ha='right')
    ax.set_xticks(range(4))
    ax.set_xticklabels(gens, fontsize=15)
    ax.set_yticks([])
    ax.set_xlim(-0.2, 3.9)
    ax.set_ylim(-1.1, 2.6)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Source generation', fontsize=15)
    sp.save(fig, path)
generations_fig(out / 'generations.svg')
_save = sp.save

def _time_ticks_save(fig, path):
    from matplotlib.ticker import FixedLocator, NullFormatter
    names = {1e-06: '1 µs', 0.001: '1 ms', 1.0: '1 s'}
    for ax in fig.axes:
        lo, hi = ax.get_xlim()
        ticks = [t for t in sorted(names) if lo <= t <= hi]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([names[t] for t in ticks])
        ax.xaxis.set_minor_locator(FixedLocator([10.0 ** e for e in range(-7, 2)]))
        ax.xaxis.set_minor_formatter(NullFormatter())
    _save(fig, path)
sp.save = _time_ticks_save
sp.hbars(out / 'exclusion.svg', [('Exclude 8 entries', 4e-07, 'data', '0.4 µs'), ('Check 50 candidates', 1.5e-06, 'data', '1.5 µs'), ('Re-index, local', 0.088, 'danger', '88 ms'), ('Re-index, remote batch', 0.21, 'danger', '210 ms'), ('Re-index, remote per file', 1.26, 'danger', '1.26 s')], '', log=True, size=(5.8, 2.7), xlabel='Added work per edit (log; stipulated costs)')
sp.save = _save
sp.hbars_steps(str(out / 'ceff'), [('Policy A', 2.0, 'neutral', '2.00'), ('Policy B', 1.833333333, 'data', '1.83')], '', log=False, size=(4.6, 1.7), xlabel="Cost per accepted task (units of A's attempt cost)")
print('figures written to', out)
