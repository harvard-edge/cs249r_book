#!/usr/bin/env python3
"""Data plots and diagrams for Volume II, Lecture 4 (Data Storage).

Every number drawn here comes from calcs.py, which reads MLSysIM (registry,
Literature, and the Scenarios.* the chapter also reads): plotted geometry uses
the scenario values, labels use the formatted strings the slides print
(deckvals val/num). Diagram coordinates, axis ranges, and tick positions are
layout. The tier table's capacity column, the archive rate, the latency
column except NVMe, and the "7 to 25 GB/s" upper bound are the chapter's
table 2 entries, typed there and here. The 3FS trace is the DeepSeek 3FS
README read stress test (MIT license), cropped and relabeled on the slide.

Run from this folder: python3 make_figs.py
"""
_VALS = {'params_b': '175', 'run_gpus': '2,048', 'run_nodes': '256', 'gpus_per_node_word': 'eight', 'run_days': '30', 'tokens_t': '1.5', 'token_bytes': '4', 'save_min': '10', 'save_s': '600', 'saves_per_hour_word': 'six', 'nvme_drives_word': 'four', 'nvme_drives': '4', 'corpus_tb': '6', 'ckpt_gb': '2,450', 'ckpt_tb': '2.45', 'saves': '4,320', 'written_pb': '10.6', 'written_tb': '10,584', 'write_read': '1,764', 'ckpt_bpp': '14', 'bpp_fp16': '2', 'bpp_fp32': '4', 'bpp_adam': '8', 'bpp_train': '16', 'bpp_grad': '2', 'weights_gb': '350', 'master_gb': '700', 'adam_gb': '1,400', 'avg_read_mbs': '2.3', 'shards': '2,000', 'shard_gb': '3', 'archive_tb': '116.8', 'source_tb': '3', 'archive_rate': '0.004', 'hbm_tbs': '3.35', 'hbm_gbs': '3,350', 'hbm_cap_gb': '80', 'dram_gbs': '200', 'gen4_gbs': '7', 'gen5_gbs': '14', 'nvme_lat_us': '100', 'hbm_lat_ns': '273', 'dram_lat_ns': '100', 'raid_gbs': '28', 'pfs_tbs': '1', 'obj_agg_gbs': '100', 'pfs_share_gbs': '3.9', 'r_hbm_dram': '16.8', 'r_dram_nvme': '7.1', 'r_nvme_pfs': '7.2', 'r_min': '7', 'r_max': '17', 'hot_hit_pct': '95', 'hot_miss_pct': '5', 'hot_hit': '0.95', 'hot_miss': '0.05', 'hot_penalty': '100', 'hot_mean': '5.95', 'hot_frac_pct': '17', 'hot_lost_pct': '83', 'hot_miss_over_hit_word': 'five', 'mm_corpus_tb': '35', 'mm_local_tb': '30.7', 'mm_hit_pct': '88', 'mm_penalty': '7', 'mm_cold_gbs': '4', 'mm_rate_gbs': '16', 'mm_frac_pct': '58', 'wall_plateau': '4', 'step_ms': '200', 'step_s': '0.2', 'seq_tokens': '4,096', 'img_per_gpu': '256', 'img_kb': '116', 'img_kb1': '115.7', 'text_mbs': '168', 'text_mbs1': '167.8', 'img_gbs': '303', 'img_text': '1,800', 'video_mb': '1', 'video_img': '8.6', 'mlperf_results': '130', 'mlp_unet3d_a100': '1.39', 'mlp_unet3d_h100': '2.76', 'mlp_cosmoflow_a100': '0.35', 'mlp_cosmoflow_h100': '0.55', 'mlp_resnet50_a100': '0.09', 'mlp_resnet50_h100': '0.18', 'mlp_gen_min': '1.6', 'mlp_gen_max': '2', 'ex_accels': '64', 'ex_hosts': '16', 'ex_gbs': '33.87', 'ex_busy': '70.23', 'io_ms': '250', 'seq_ms': '450', 'seq_stall_pct': '55.6', 'ovl_ms': '250', 'ovl_wait_ms': '50', 'ovl_stall_pct': '20', 'p99_obj_ms': '500', 'p99_nvme_ms': '50', 'span_obj': '2.5', 'depth_obj': '3', 'depth_safety': '5', 'depth_safety_word': 'five', 'depth_nvme_word': 'one', 'batch_mb': '40', 'prefetch_node_gb': '1.6', 'gds_rate': '64,000', 'gds_host_us': '120', 'gds_gds_us': '30', 'gds_host_cpu': '7.68', 'gds_gds_cpu': '1.92', 'gds_host_cores': '8', 'gds_gds_cores': '2', 'gds_freed': '6', 'files_m': '200', 'tar_shards': '50,000', 'files_per_shard': '4,000', 'fsr': '500', 'fs1': '5', 'oss': '100', 'oss_gbs': '10', 'tail_servers': '100', 'tail_slow_pct': '1', 'tail_fast': '0.99', 'tail_all_fast': '0.366', 'tail_some_slow_pct': '63', 'fs3_tibs': '6.6', 'fs3_nodes': '180', 'fs3_ssds': '16', 'fs3_ssds_word': 'sixteen', 'fs3_ssd_tib': '14', 'fs3_nics': '2', 'fs3_nics_word': 'two', 'fs3_nic_gbps': '200', 'fs3_clients': '500', 'fs3_per_node_gib': '37.5', 'tib_tb': '1.1', 'shard_node_gb': '9.6', 'pause_s': '0.34', 'drain_s': '2.45', 'both_s': '2.8', 'pause_pct': '0.06', 'storm_params_b': '70', 'storm_state_gb': '980', 'storm_replicas': '1,024', 'storm_ingest_gbs': '100', 'storm_tb': '1,004', 'storm_s': '10,035', 'storm_min': '167', 'once_s': '9.8', 'storm_ratio': '1,024', 'storm_train_s': '1,000', 'syn_tb': '1', 'syn_rest_tb': '1.4', 'syn_moved_tb': '4.4', 'syn_passes_word': 'three', 'ds_tb': '100', 'obj_rate': '0.02', 'egress_rate': '0.09', 'nvme_rate': '0.10', 'store_year': '24,000', 'egress_one': '9,000', 'egress_reads': '10', 'egress_all': '90,000', 'files_get_m': '100', 'get_fees': '40', 'restore': '11,000', 'idle_gpus': '1,024', 'gpu_hour': '2', 'fleet_day': '49,152', 'util_high': '90', 'util_low': '70', 'util_drop': '20', 'idle_day': '9,830', 'ss_tb': '50', 'ss_epochs': '20', 'ss_runs': '4', 'ss_runs_word': 'four', 'ss_source': '12,000', 'ss_per_run': '90,000', 'ss_stream_egress': '360,000', 'ss_stream': '372,000', 'ss_local': '60,000', 'ss_stage_egress': '18,000', 'ss_stage': '90,000', 'ss_saving': '282,000', 'ss_per_epoch': '18,000', 'ss_break_even': '4.3', 'load_obj_gbs': '1', 'load_pfs_gbs': '4', 'load_obj_s': '350', 'load_pfs_s': '87.5', 'load_gen5_s': '25', 'load_stripe_s': '6.2', 'load_dram_s': '1.8', 'load_drives_word': 'eight', 'rollout_replicas': '100', 'rollout_tb': '35', 'rollout_pfs_s': '35', 'codec_dense': '4', 'codec_light': '3', 'codec_dense_gbs': '0.5', 'codec_light_gbs': '1.5'}
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
plt = sp.plt
C = sp.C
out = Path(__file__).parent
V = {'c1_fill': '#EAF1F8', 'c1': '#1F407A', 'c1_text': '#14294E', 'c2_fill': '#fef3c7', 'c2': '#d97706', 'c2_text': '#b45309', 'c3_fill': '#f3e8ff', 'c3': '#7c3aed', 'c3_text': '#6d28d9', 'st_fill': '#dcfce7', 'st': '#16a34a', 'st_text': '#15803d', 'bad_fill': '#fee2e2', 'bad': '#dc2626', 'bad_text': '#b91c1c', 'fr_fill': '#f8fafc', 'fr': '#94a3b8', 'text': '#0f172a', 'mute': '#475569'}
FONT = 'Helvetica, Arial, sans-serif'

class Svg:

    def __init__(self, w=900, h=500):
        self.w, self.h, self.items = (w, h, [])

    def rect(self, x, y, w, h, fill, stroke, sw=2, dash=None, rx=6, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{op}"/>')

    def text(self, x, y, s, size=22, color=None, weight='normal', anchor='middle', italic=False, op=1.0):
        st = ' font-style="italic"' if italic else ''
        self.items.append(f'''<text x="{x}" y="{y}" font-size="{size}" fill="{color or V['text']}" font-weight="{weight}" text-anchor="{anchor}"{st} opacity="{op}">{s}</text>''')

    def line(self, x1, y1, x2, y2, color, sw=3, dash=None, arrow=False, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        mk = f' marker-end="url(#a{color[1:]})"' if arrow else ''
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{mk} opacity="{op}"/>')
        if arrow:
            self.markers.add(color)

    def poly(self, pts, color, sw=3, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        mk = f' marker-end="url(#a{color[1:]})"' if arrow else ''
        p = ' '.join((f'{x},{y}' for x, y in pts))
        self.items.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"{d}{mk}/>')
        if arrow:
            self.markers.add(color)

    def polygon(self, pts, fill, stroke, sw=2):
        p = ' '.join((f'{x},{y}' for x, y in pts))
        self.items.append(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def dots(self, x, y, color):
        """A vertical ellipsis drawn as three dots (no font glyph needed)."""
        for dy in (-10, 0, 10):
            self.items.append(f'<circle cx="{x}" cy="{y + dy}" r="3" fill="{color}"/>')

    def cross(self, x, y, r, color):
        """A bold X drawn as two lines."""
        self.line(x - r, y - r, x + r, y + r, color, 6)
        self.line(x - r, y + r, x + r, y - r, color, 6)
    markers = set()

    def save(self, path):
        defs = ''.join((f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in sorted(self.markers)))
        body = '\n'.join(self.items)
        Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" font-family="{FONT}">\n<defs>{defs}</defs>\n<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')
        Svg.markers = set()

def box(s, x, y, w, h, role, title, sub=None, tsize=22, ssize=19, op=1.0, dash=None):
    s.rect(x, y, w, h, V[role + '_fill'], V[role], dash=dash, op=op)
    tc = V.get(role + '_text', V['text'])
    if sub:
        s.text(x + w / 2, y + h / 2 - 4, title, tsize, tc, 'bold', op=op)
        s.text(x + w / 2, y + h / 2 + 22, sub, ssize, V['mute'], op=op)
    else:
        s.text(x + w / 2, y + h / 2 + 8, title, tsize, tc, 'bold', op=op)
sp.hbars(out / 'hook_bytes.svg', [('Read: one epoch of tokens', 6.0, 'data', '6 TB'), ('Write: checkpoints, 30 days', 10584.0, 'memory', '10.6 PB')], 'TB', log=True, size=(4.1, 1.05), xlabel='Terabytes moved (log scale)')
speed = np.array([1, 2, 4, 8, 16])
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.35, 1.85))
    ax.plot(speed, speed, color=C['neutral'], lw=2, ls='--', marker='o', ms=6)
    ax.text(1.0, 9.5, 'if storage kept up', color=C['neutral'], fontsize=12, fontweight='bold', ha='left')
    shown = speed if k == 2 else speed[:3]
    ax.plot(shown, {1: [1.0, 2.0, 4.0], 2: [1.0, 2.0, 4.0, 4.0, 4.0]}[k], color=C['danger'], lw=3, marker='o', ms=9)
    a = 1.0 if k == 2 else 0.0
    ax.axhline(4.0, color=C['data'], lw=1.5, ls=':', alpha=a)
    ax.text(17, 3.3, 'plateau: 4× at 16× speed', color=C['danger'], fontsize=12.5, fontweight='bold', ha='right', va='top', alpha=a)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(speed, [f'{v}×' for v in speed])
    ax.set_yticks([1, 2, 4, 8, 16], ['1×', '2×', '4×', '8×', '16×'])
    ax.minorticks_off()
    ax.set_xlim(0.8, 20)
    ax.set_ylim(0.8, 20)
    ax.set_xlabel('Accelerator speed (relative)')
    ax.set_ylabel('Batches per second')
    ax.grid(color=C['grid'], lw=0.6)
    sp.save(fig, out / f'io_wall_step{k}.svg')
import csv as _csv, io as _io
# MLPerf Storage v1.0 training, all Closed passing submissions (the book's figure data).
_src = Path(__file__).resolve().parents[4] / 'books/vol2/04_data_storage/data/mlperf_storage_v1_training.csv'
rows = [dict(r, num_accelerators=float(r['num_accelerators']), io_GBps=float(r['io_GBps']))
        for r in _csv.DictReader(_io.StringIO(''.join(l for l in open(_src) if not l.startswith('#'))))
        if r['division'] == 'Closed' and r['au_check'] == 'pass']
assert len(rows) == 130, len(rows)
WL = [('unet3d', '3D U-Net', 'memory'), ('cosmoflow', 'CosmoFlow', 'compute'), ('resnet50', 'ResNet-50', 'data')]
fig, ax = plt.subplots(figsize=(3.9, 1.95))
for w, name, ck in WL:
    for acc, mk in zip(('A100', 'H100'), ('o', '^')):
        pts = [(r['num_accelerators'], r['io_GBps']) for r in rows if r['workload'] == w and r['accelerator'] == acc]
        x, y = zip(*pts)
        ax.scatter(x, y, s=34, marker=mk, color=C[ck], alpha=0.85, edgecolor='white', linewidth=0.5)
for (w, name, ck), (tx, ty) in zip(WL, [(200, 2.6), (200, 1.0), (200, 0.38)]):
    ax.text(tx, ty, name, color=C[ck], fontsize=12.5, fontweight='bold')
ax.text(1.05, 580, 'circle: A100  |  triangle: H100', fontsize=10.5, color=C['text'], bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(top=1500)
ax.set_xlabel('Emulated accelerators')
ax.set_ylabel('Read throughput (GB/s)')
ax.grid(color=C['grid'], lw=0.6)
ax.minorticks_off()
sp.save(fig, out / 'mlperf_scatter.svg')
tiers = [('GPU HBM', 'c1', '80 GB', '3.35 TB/s per GPU', '~273 ns'), ('Host DRAM', 'c1', '0.5 to 2 TB', '200 GB/s per node', '~100 ns'), ('Local NVMe', 'st', '4 to 30 TB', '7 to 25 GB/s', '~100 µs'), ('Parallel file system', 'st', '100+ PB', '1+ TB/s shared', '~1 ms'), ('Object storage', 'st', 'exabytes', '100 GB/s shared', '~50 ms'), ('Archive', 'fr', 'exabytes+', '1 GB/s', 'min to h')]
for k in (1, 2):
    s = Svg(900, 345)
    top, rh, gap = (36, 47, 4)
    s.text(270, 24, 'Tier (wider = more capacity)', 20, V['mute'], 'bold')
    if k == 2:
        for x, lab in ((555, 'Capacity'), (710, 'Bandwidth'), (857, 'Latency')):
            s.text(x, 24, lab, 20, V['mute'], 'bold')
    for i, (name, role, cap, bw, lat) in enumerate(tiers):
        y = top + i * (rh + gap)
        half = 78 + i * 28
        s.polygon([(270 - half, y + rh), (270 + half, y + rh), (270 + half - 22, y), (270 - half + 22, y)], V[role + '_fill'], V[role])
        s.text(270, y + rh / 2 + 8, name, 22, V.get(role + '_text', V['text']), 'bold')
        s.text(4, y + rh / 2 + 8, f'{i}', 22, V['mute'], 'bold', anchor='start')
        if k == 2:
            s.text(555, y + rh / 2 + 8, cap, 20)
            s.text(710, y + rh / 2 + 8, bw, 20, V['c1_text'], 'bold')
            s.text(857, y + rh / 2 + 8, lat, 20)
    s.save(out / f'tiers_step{k}.svg')
fig, ax = plt.subplots(figsize=(2.7, 1.65))
lad = [('GPU HBM (one GPU)', 3350.0, 'memory', '3.35 TB/s'), ('Host DRAM (node)', 200.0, 'memory', '200 GB/s'), ('4 NVMe, RAID 0 (node)', 28.0, 'data', '28 GB/s'), ('File system share (node)', 3.90625, 'data', '3.9 GB/s')]
n = len(lad)
for i, (lab, v, ck, txt) in enumerate(lad):
    y = n - 1 - i
    ax.barh(y, v, color=C[ck], height=0.6)
    ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12.5, fontweight='bold', color=C['text'])
    ax.text(v * 1.2, y, txt, va='center', fontsize=12.5, fontweight='bold', color=C[ck])
for i, r in enumerate([f'÷ {_VALS[k]}' for k in ('r_hbm_dram', 'r_dram_nvme', 'r_nvme_pfs')]):
    ax.text(2000000.0, n - 1.5 - i, r, va='center', ha='center', fontsize=12.5, fontweight='bold', color=C['danger'])
ax.set_xscale('log')
ax.set_xlim(1, 10000000.0)
ax.set_ylim(-0.6, n - 0.4)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('GB/s per client (log scale)')
ax.grid(axis='x', color=C['grid'], lw=0.6)
ax.set_axisbelow(True)
ax.minorticks_off()
sp.save(fig, out / 'ladder.svg')
h = np.linspace(0.5, 1.0, 400)
fig, ax = plt.subplots(figsize=(4.85, 1.5))
for kmiss, ck in ((7.0, 'data'), (100.0, 'danger')):
    ax.plot(h * 100, 1 / (h + (1 - h) * kmiss) * 100, color=C[ck], lw=3)
ax.plot([0.95 * 100], [0.1680672269 * 100], 'o', ms=9, color=C['danger'], mec='white', mew=1.5)
ax.annotate('95% hits, misses 100× slower: 17%', (0.95 * 100, 0.1680672269 * 100), xytext=(53, 10), ha='left', fontsize=11.5, fontweight='bold', color=C['danger'], arrowprops=dict(arrowstyle='-', color=C['danger'], lw=1.2))
ax.plot([0.8777142857 * 100], [0.5767963085 * 100], 'o', ms=9, color=C['data'], mec='white', mew=1.5)
ax.annotate('88% hits, misses 7× slower: 58%', (0.8777142857 * 100, 0.5767963085 * 100), xytext=(51, 86), ha='left', fontsize=11.5, fontweight='bold', color=C['data'], arrowprops=dict(arrowstyle='-', color=C['data'], lw=1.2))
ax.set_xlim(50, 100)
ax.set_ylim(0, 105)
ax.set_xlabel('Hit rate in the hot tier (%)')
ax.set_ylabel('% of all-hot rate')
ax.grid(color=C['grid'], lw=0.6)
sp.save(fig, out / 'cache_hit.svg')
sp.hbars(out / 'demand.svg', [('Images, 256 per GPU', 303.1887864, 'danger', '303 GB/s'), ('Text, 4,096 tokens per GPU', 0.16777216, 'data', '168 MB/s'), ('175B job, 30-day average', 0.002314814815, 'neutral', '2.3 MB/s')], 'GB/s', log=True, size=(4.1, 1.05), xlabel='GB/s for 2,048 GPUs (log scale)')
for k in (1, 2, 3, 4):
    s = Svg(960, 392)
    # Band 1: Sequential (250 ms Read + 200 ms Train = 450 ms/step, 55.6% stalled)
    s.text(10, 22, '1. Sequential: read 250 ms, then compute 200 ms', 21, V['text'], 'bold', anchor='start')
    s.text(98, 52, 'Read', 20, V['st_text'], 'bold', anchor='end')
    s.text(98, 86, 'GPU', 20, V['c1_text'], 'bold', anchor='end')
    u = 0.62  # px per ms; 900 ms = 558 px (x: 110..668)
    x0 = 110
    for b in range(2):
        bx = x0 + b * 450 * u
        box(s, bx, 32, 250 * u - 3, 30, 'st', f'R{b+1} (250)', tsize=19)
        s.rect(bx, 66, 250 * u - 3, 30, V['bad_fill'], V['bad'], sw=1.5, dash='5,4')
        s.text(bx + 125 * u, 88, 'GPU idle', 19, V['bad_text'], 'bold')
        box(s, bx + 250 * u, 66, 200 * u - 3, 30, 'c1', f'T{b+1}', tsize=19)
    s.text(685, 56, '450 ms/step', 21, V['bad_text'], 'bold', anchor='start')
    s.text(685, 84, '55.6% stalled', 21, V['bad_text'], 'bold', anchor='start')
    # Band 2: Overlapped with I/O > compute (250 ms/step, 20% stalled)
    op2 = 1.0 if k >= 2 else 0.0
    s.text(10, 126, '2. Overlapped (I/O 250 ms &gt; compute 200 ms)', 21, V['text'], 'bold', anchor='start', op=op2)
    s.text(98, 160, 'Read', 20, V['st_text'], 'bold', anchor='end', op=op2)
    s.text(98, 194, 'GPU', 20, V['c1_text'], 'bold', anchor='end', op=op2)
    for b in range(3):
        rx = x0 + b * 250 * u
        box(s, rx, 140, 250 * u - 3, 30, 'st', f'R{b+1}', tsize=19, op=op2)
        if b < 2:
            tx = x0 + (b + 1) * 250 * u
            box(s, tx, 174, 200 * u - 3, 30, 'c1', f'T{b+1}', tsize=19, op=op2)
            s.rect(tx + 200 * u, 174, 50 * u - 3, 30, V['bad_fill'], V['bad'], sw=1.5, op=op2)
    s.rect(x0, 174, 250 * u - 3, 30, V['bad_fill'], V['bad'], sw=1.5, dash='5,4', op=op2)
    s.text(x0 + 125 * u, 196, 'fill', 19, V['bad_text'], 'bold', op=op2)
    if k >= 4:
        s.rect(x0 + 250 * u - 4, 136, 500 * u + 4, 72, 'none', V['c3'], sw=2.5, dash='7,4', rx=6)
    s.text(685, 164, '250 ms/step', 21, V['bad_text'], 'bold', anchor='start', op=op2)
    s.text(685, 192, '20% stalled (50 ms)', 20, V['bad_text'], 'bold', anchor='start', op=op2)
    # Band 3: 3-stage pipeline when Read & Prep <= 200 ms (0% input stall in steady state)
    op3 = 1.0 if k >= 3 else 0.0
    s.text(10, 240, '3. 3-stage pipeline (Read, Prep ≤ 200 ms): 3 in flight', 21, V['text'], 'bold', anchor='start', op=op3)
    s.text(98, 297, 'Read', 19, V['st_text'], 'bold', anchor='end', op=op3)
    s.text(98, 326, 'Prep', 19, V['c2_text'], 'bold', anchor='end', op=op3)
    s.text(98, 355, 'GPU', 19, V['c1_text'], 'bold', anchor='end', op=op3)
    cw = 112
    # Show initial pipeline fill explicitly in empty GPU slots before T1
    s.rect(x0, 337, 2 * cw - 4, 24, V['bad_fill'], V['bad'], sw=1.4, dash='5,4', op=op3)
    s.text(x0 + cw - 2, 355, 'pipeline fill', 17.5, V['bad_text'], 'bold', op=op3)
    for b in range(5):
        if b < 4:
            box(s, x0 + b * cw, 279, cw - 4, 24, 'st', f'R{b+1}', tsize=19, op=op3)
        if 1 <= b <= 4:
            box(s, x0 + b * cw, 308, cw - 4, 24, 'c2', f'P{b}', tsize=19, op=op3)
        if 2 <= b <= 4:
            box(s, x0 + b * cw, 337, cw - 4, 24, 'c1', f'T{b-1}', tsize=19, op=op3)
    s.text(685, 300, '200 ms/step', 21, V['st_text'], 'bold', anchor='start', op=op3)
    s.text(685, 327, '0% input stall', 20.5, V['st_text'], 'bold', anchor='start', op=op3)
    s.text(685, 354, '(steady state)' if k < 4 else '(steady state: 3 active)', 19, V['c3_text'] if k >= 4 else V['st_text'], 'bold', anchor='start', op=op3)
    # Step 4: highlight steady-state interval with explicit header above box
    if k >= 4:
        s.text(x0 + 3 * cw - 2, 267, 'After pipeline fill', 19, V['c3_text'], 'bold')
        s.rect(x0 + 2 * cw - 4, 274, 2 * cw + 4, 92, 'none', V['c3'], sw=3, dash='7,4', rx=6)
    s.save(out / f'stall_step{k}.svg')
WIN, P99 = (200, 500)
fig, ax = plt.subplots(figsize=(5.0, 0.85))
for i in range(3):
    ax.broken_barh([(i * WIN + 3, WIN - 6)], (0.55, 0.4), color=C['memory'])
    ax.text(i * WIN + WIN / 2, 0.75, f'batch {i + 1}', ha='center', va='center', color='white', fontsize=11.5, fontweight='bold')
ax.broken_barh([(0, P99)], (0.0, 0.4), color=C['danger'])
ax.text(P99 / 2, 0.2, 'one slow read: 500 ms (p99)', ha='center', va='center', color='white', fontsize=11.5, fontweight='bold')
ax.text(610, 0.5, '500 / 200 = 2.5\nround up: 3', ha='left', va='center', fontsize=14, fontweight='bold', color=C['text'])
for x in (WIN, 2 * WIN, 3 * WIN):
    ax.axvline(x, color=C['grid'], lw=1, zorder=0)
ax.set_xlim(0, 860)
ax.set_ylim(-0.1, 1.05)
ax.set_yticks([])
ax.set_xticks([0, WIN, 2 * WIN, 3 * WIN])
ax.set_xlabel('Time (ms); one compute window = 200 ms')
ax.spines['left'].set_visible(False)
sp.save(fig, out / 'prefetch.svg')
for k in (1, 2):
    s = Svg(900, 290)
    s.text(10, 24, '(a) Through host DRAM', 22, V['text'], 'bold', anchor='start')
    box(s, 10, 42, 120, 60, 'fr', 'CPU')
    s.rect(165, 36, 270, 100, V['c1_fill'], V['c1'])
    s.text(300, 60, 'Host DRAM', 21, V['c1_text'], 'bold')
    box(s, 178, 74, 115, 48, 'fr', 'kernel', tsize=20)
    box(s, 308, 74, 115, 48, 'fr', 'pinned', tsize=20)
    s.line(130, 72, 172, 98, V['mute'], 3)
    s.text(110, 132, 'copy', 20, V['mute'], 'bold')
    box(s, 10, 222, 150, 60, 'st', 'NVMe SSD')
    box(s, 285, 222, 150, 60, 'c1', 'GPU HBM')
    s.poly([(85, 222), (85, 180), (235, 180), (235, 128)], V['st'], 4, arrow=True)
    s.text(150, 172, 'DMA 1', 20, V['st_text'], 'bold')
    s.poly([(365, 122), (365, 216)], V['st'], 4, arrow=True)
    s.text(382, 180, 'DMA 2', 20, V['st_text'], 'bold', anchor='start')
    if k >= 2:
        s.line(452, 36, 452, 285, V['fr'], 2, dash='6,6')
        s.text(470, 24, '(b) GPUDirect Storage', 22, V['text'], 'bold', anchor='start')
        box(s, 470, 42, 120, 60, 'fr', 'CPU')
        box(s, 625, 36, 270, 100, 'c1', 'Host DRAM', 'not on the path', ssize=20)
        box(s, 470, 222, 150, 60, 'st', 'NVMe SSD')
        box(s, 745, 222, 150, 60, 'c1', 'GPU HBM')
        s.line(530, 102, 530, 216, V['mute'], 3, dash='7,6', arrow=True)
        s.text(540, 165, 'control', 20, V['mute'], 'bold', anchor='start')
        s.poly([(585, 222), (585, 190), (820, 190), (820, 216)], V['st'], 4, arrow=True)
        s.text(702, 180, 'one DMA', 20, V['st_text'], 'bold')
    s.save(out / f'gds_step{k}.svg')
for k in (1, 2):
    s = Svg(780, 320)
    s.text(95, 24, 'GPU nodes', 22, V['mute'], 'bold')
    s.text(435, 24, 'Servers', 22, V['mute'], 'bold')
    for i, y in enumerate((120, 190, 262)):
        box(s, 10, y, 165, 52, 'c1', f"Client {['1', '2', 'M'][i]}", tsize=22)
    box(s, 325, 34, 215, 66, 'c3', 'Metadata server', 'open, stat, lock', tsize=22, ssize=21)
    s.poly([(92, 120), (92, 67), (319, 67)], V['c3'], 3, arrow=True)
    s.text(205, 56, '1. look up', 21, V['c3_text'], 'bold')
    s.text(665, 58, 'few hundred', 21, V['c3_text'], 'bold')
    s.text(665, 84, 'thousand ops/s', 21, V['c3_text'], 'bold')
    if k >= 2:
        for i, y in enumerate((120, 190, 262)):
            box(s, 340, y, 195, 52, 'st', f"OSS {['1', '2', '100'][i]}: 10 GB/s", tsize=21)
            s.line(265, y + 26, 334, y + 26, V['st'], 3, arrow=True)
        s.line(265, 146, 265, 288, V['st'], 3)
        for y in (146, 216, 288):
            s.line(175, y, 265, y, V['st'], 3)
        s.text(220, 112, '2. read', 21, V['st_text'], 'bold')
        s.text(665, 206, '100 × 10 GB/s', 21, V['st_text'], 'bold')
        s.text(665, 234, '= 1 TB/s data', 21, V['st_text'], 'bold')
    s.save(out / f'pfs_step{k}.svg')
for k in (1, 2):
    s = Svg(900, 290)
    # Row 1: Symptom (1 file per sample -> MDS meltdown)
    s.text(10, 24, 'Symptom: one file per sample (200M JPEGs)', 22, V['bad_text'], 'bold', anchor='start')
    box(s, 10, 40, 205, 76, 'fr', '200M JPEGs', '1 open() per image', tsize=21, ssize=19)
    s.line(215, 78, 285, 78, V['bad'], 4, arrow=True)
    box(s, 292, 40, 265, 76, 'bad', '1 Metadata server', 'open() queue full', tsize=21, ssize=19)
    s.line(557, 78, 622, 78, V['fr'], 3, dash='6,5', arrow=True)
    box(s, 628, 40, 262, 76, 'fr', '100 Data disks', 'idle: 1% (5 GB/s)', tsize=21, ssize=19)
    # Row 2: Fix (WebDataset tar shards + manifest)
    if k >= 2:
        s.line(10, 140, 890, 140, V['fr'], 1.5, dash='6,6')
        s.text(10, 168, 'Fix: bundle into shards + precomputed manifest', 22, V['st_text'], 'bold', anchor='start')
        box(s, 10, 184, 225, 86, 'st', '50,000 tar shards', '4,000× fewer opens', tsize=21, ssize=19)
        s.line(235, 227, 298, 227, V['st'], 4, arrow=True)
        box(s, 305, 184, 240, 86, 'c3', '1 Metadata server', 'manifest skips stat', tsize=21, ssize=19)
        s.line(545, 227, 610, 227, V['st'], 4, arrow=True)
        box(s, 616, 184, 274, 86, 'st', '100 Data disks', 'full streaming rate', tsize=21, ssize=19)
    s.save(out / f'metadata_step{k}.svg')
fig, ax = plt.subplots(figsize=(5.8, 1.7))
x = 0
for i, (key, txt, ck) in enumerate([('weights_gb', 'FP16\nweights', 'memory'), ('master_gb', 'FP32 master\nweights', 'neutral'), ('adam_gb', 'Adam moments\n(two FP32)', 'compute')]):
    w = {'FP16\nweights': 350.0, 'FP32 master\nweights': 700.0, 'Adam moments\n(two FP32)': 1400.0}[txt]
    ax.barh(0, w, left=x, color=C[ck], height=0.7, edgecolor='white', lw=2)
    ax.text(x + w / 2, 0, {'FP16\nweights': '350', 'FP32 master\nweights': '700', 'Adam moments\n(two FP32)': '1,400'}[txt], ha='center', va='center', color='white', fontsize=13, fontweight='bold')
    if i == 0:
        ax.text(0, 0.42, 'FP16 weights', ha='left', va='bottom', color=C[ck], fontsize=12, fontweight='bold')
    else:
        ax.text(x + w / 2, -0.45, txt, ha='center', va='top', color=C[ck], fontsize=12, fontweight='bold', linespacing=1.05)
    x += w
ax.text(2470, 0, ' = 2,450 GB', va='center', fontsize=14, fontweight='bold', color=C['text'])
ax.text(2490, -0.45, ' 2 + 4 + 8 bytes\n per parameter', va='top', fontsize=12, color=C['text'], linespacing=1.05)
ax.set_xlim(0, 3200)
ax.set_ylim(-1.35, 0.85)
ax.axis('off')
sp.save(fig, out / 'ckpt_bytes.svg')
sp.hbars(out / 'storm.svg', [('Every replica writes its copy', 10035.2, 'danger', '10,035 s = 167 min'), ('Each unique byte written once', 9.8, 'data', '9.8 s')], 's', log=True, size=(3.6, 0.95), xlabel='Seconds to save (log scale)')
for k in (1, 2, 3):
    s = Svg(900, 330)
    lanes = [('Training', 8), ('Local NVMe', 72), ('File system', 136), ('Coordinator', 200)]
    for name, y in lanes:
        s.text(140, y + 36, name, 21, V['text'], 'bold', anchor='end')
    t0, t1, t2, t3, tend = (222, 362, 640, 760, 890)
    ticks = [(t0, 't0'), (t1, 't1')] + ([(t2, 't2')] if k >= 2 else []) + ([(t3, 't3')] if k >= 3 else [])
    for x, lab in ticks:
        s.line(x, 8, x, 268, V['bad'] if x in (t0, t1) else V['c3'], 1.5, dash='5,5')
    bh = 54
    box(s, 150, 8, t0 - 150, bh, 'c1', 'step i', tsize=20)
    box(s, t0, 8, t1 - t0, bh, 'bad', 'pause 0.34 s', tsize=19)
    box(s, t1, 8, tend - t1, bh, 'c1', 'training continues on live state', tsize=20)
    box(s, t0, 72, t1 - t0, bh, 'st', '9.6 GB', tsize=20)
    if k >= 2:
        s.rect(t1, 72, t3 - t1, bh, '#f0fdf4', V['st'])
        s.text((t1 + t3) / 2, 106, 'staged: lost if this node fails', 20, V['st_text'])
        box(s, t1, 136, t2 - t1, bh, 'st', 'copy, about 2.45 s', tsize=20, dash='7,5')
        box(s, t2, 136, tend - t2, bh, 'st', 'durable', tsize=20)
    if k >= 3:
        box(s, t1, 200, t2 - t1, bh, 'fr', 'wait for every shard', tsize=20)
        box(s, t2, 200, t3 - t2, bh, 'c3', 'verify', tsize=20)
        box(s, t3, 200, tend - t3, bh, 'c3', 'manifest', tsize=20)
    s.line(150, 268, tend, 268, V['mute'], 2, arrow=True)
    for x, lab in ticks:
        s.text(x, 293, lab, 22, V['text'], 'bold')
    if k >= 3:
        s.text((t1 + t3) / 2, 322, 'staged, not yet restorable', 20, V['mute'], 'bold')
        s.text((t3 + tend) / 2 + 4, 322, 'restorable', 20, V['st_text'], 'bold')
    s.save(out / f'staging_step{k}.svg')
s = Svg(900, 130)
stages = [('raw corpus', '3 TB'), ('cleaned', 'dedup v4'), ('tokenized', 'tok v2'), ('shards', '6 TB')]
for i, (t, sub) in enumerate(stages):
    x = 5 + i * 175
    box(s, x, 15, 145, 96, 'st', t, sub, tsize=22, ssize=20)
    s.line(x + 145, 63, x + 172, 63, V['st'], 3, arrow=True)
s.rect(705, 5, 190, 116, V['c3_fill'], V['c3'])
s.text(800, 38, 'manifest', 22, V['c3_text'], 'bold')
s.text(800, 68, 'shard hashes,', 20, V['mute'])
s.text(800, 96, 'code, seed', 20, V['mute'])
s.save(out / 'versions.svg')
for k in (1, 2):
    s = Svg(900, 330)
    s.text(150, 40, 'Worker', 22, V['text'], 'bold', anchor='end')
    box(s, 165, 8, 150, 54, 'fr', 'read e42', tsize=21)
    s.line(315, 35, 342, 35, V['mute'], 3, arrow=True)
    box(s, 347, 8, 165, 54, 'st', 'write row', tsize=21)
    s.cross(545, 35, 13, V['bad'])
    box(s, 578, 8, 312, 54, 'c3', 'record commit: never', tsize=21, dash='7,5')
    s.text(150, 112, 'Replacement', 22, V['text'], 'bold', anchor='end')
    box(s, 165, 80, 260, 54, 'fr', 'replay e42: committed?', tsize=21)
    s.text(660, 112, 'crash: written, not committed', 21, V['bad_text'], 'bold')
    s.rect(10, 150, 430, 175, V['fr_fill'], V['fr'])
    s.text(25, 180, '(a) Append the row', 22, V['text'], 'bold', anchor='start')
    s.rect(25, 195, 400, 44, 'white', V['fr'])
    s.text(40, 224, 'e42, click, first write', 20, anchor='start')
    s.rect(25, 247, 400, 44, V['bad_fill'], V['bad'])
    s.text(40, 276, 'e42, click, replay', 20, anchor='start')
    s.text(412, 276, 'duplicate', 20, V['bad_text'], 'bold', anchor='end')
    s.text(25, 316, 'the click counts twice', 20, V['bad_text'], 'bold', anchor='start')
    if k == 2:
        s.rect(460, 150, 430, 175, V['fr_fill'], V['fr'])
        s.text(475, 180, '(b) Write under a stable key', 22, V['text'], 'bold', anchor='start')
        s.rect(475, 195, 400, 44, V['st_fill'], V['st'])
        s.text(490, 224, '(feature_version, e42), click', 20, anchor='start')
        s.text(475, 270, 'the replay overwrites the same row:', 20, V['st_text'], 'bold', anchor='start')
        s.text(475, 298, 'the click counts once', 20, V['st_text'], 'bold', anchor='start')
    s.save(out / f'ackgap_step{k}.svg')
sp.hbars(out / 'egress.svg', [('Store 100 TB for a year', 24000.0 / 1000, 'neutral', '$24,000'), ('Read it 10 times across regions', 90000.0 / 1000, 'danger', '$90,000')], '$', log=False, size=(3.55, 0.8), xlabel='Thousand US dollars')
ep = np.linspace(0, 20, 200)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.9, 1.5))
    a = 1.0 if k == 2 else 0.0
    base = 12000.0
    per_epoch = 90000.0 / 20 * 4
    ax.plot(ep, (base + per_epoch * ep) / 1000, color=C['danger'], lw=3, alpha=a)
    ax.plot(ep, np.full_like(ep, 90000.0 / 1000), color=C['data'], lw=3, alpha=a)
    ax.text(5.5, 330, 'stream every epoch', color=C['danger'], ha='left', fontsize=12.5, fontweight='bold', alpha=a)
    ax.text(19.5, 110, 'stage once per run', color=C['data'], ha='right', fontsize=12.5, fontweight='bold', alpha=a)
    ax.plot([4.333333333], [90000.0 / 1000], 'o', ms=10, color=C['text'], mec='white', mew=1.5, alpha=a)
    ax.annotate('break-even: 4.3 epochs', (4.333333333, 90000.0 / 1000), xytext=(6, 8), fontsize=12, fontweight='bold', color=C['text'], alpha=a, arrowprops=dict(arrowstyle='-', color=C['text'], lw=1.2, alpha=a))
    ax.plot([20] * 2, [90000.0 / 1000, 372000.0 / 1000], color=C['neutral'], lw=1, ls=':', alpha=a)
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Epochs per training run (4 runs a year)')
    ax.set_ylabel('Thousand $ per year')
    ax.grid(color=C['grid'], lw=0.6)
    sp.save(fig, out / f'stage_cost_step{k}.svg')
_cs_rows = [('Object store, 1 GB/s', 350.0, 'danger', '350 s'), ('File system share, 4 GB/s', 87.5, 'compute', '87.5 s'), ('One NVMe, 14 GB/s', 25.0, 'data', '25 s'), ('Eight NVMe, 7 GB/s each', 6.25, 'data', '6.25 s'), ('Host DRAM, 200 GB/s', 1.75, 'memory', '1.8 s')]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.4, 1.75))
    n = len(_cs_rows)
    for i, (lab, val, ck, txt) in enumerate(_cs_rows):
        y = n - 1 - i
        shown = i < 2 or k >= 2
        a = 1.0 if shown else 0.0
        ax.barh(y, val, color=C[ck], height=0.58, alpha=a)
        ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=11.5, fontweight='bold', color=C['text'], alpha=a)
        ax.text(val * 1.18, y, txt, va='center', fontsize=11.5, fontweight='bold', color=C[ck], alpha=a)
    ax.set_xscale('log')
    ax.set_xlim(1, 1600)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Seconds to load 350 GB FP16 weights (log scale)')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    ax.minorticks_off()
    sp.save(fig, out / f'coldstart_step{k}.svg')
try:
    from PIL import Image
    im = Image.open(out / '3fs-read-throughput.jpg')
    w, hgt = im.size
    from PIL import ImageDraw
    cr = im.crop((int(0.054 * w), int(0.135 * hgt), int(0.963 * w), int(0.855 * hgt)))
    cw, ch = cr.size
    ImageDraw.Draw(cr).rectangle((int(0.86 * cw), int(0.84 * ch), cw, ch), fill='white')
    cr.save(out / '3fs_trace_crop.png')
except FileNotFoundError:
    pass
for k in (1, 2):
    s = Svg(900, 225)
    s.line(190, 182, 880, 182, V['mute'], 2, arrow=True)
    s.text(880, 212, 'time', 26, V['mute'], anchor='end')
    s.text(175, 62, 'First read', 28, V['text'], 'bold', anchor='end')
    s.rect(195, 34, 640, 48, V['bad_fill'], V['bad'])
    s.text(634, 67, 'server A: slow, in the tail', 26, V['bad_text'], 'bold')
    if k >= 2:
        s.line(430, 22, 430, 182, V['c1'], 2.5, dash='7 5')
        s.text(436, 212, 'p99 delay', 26, V['c1'], 'bold', anchor='middle')
        s.text(175, 134, 'Second read', 28, V['text'], 'bold', anchor='end')
        s.rect(432, 106, 180, 48, V['st_fill'], V['st'])
        s.text(522, 139, 'replica B', 26, V['st_text'], 'bold')
        s.text(626, 139, 'first response wins', 27, V['st_text'], 'bold', anchor='start')
    s.save(out / f'hedge_timeline_step{k}.svg')

# Recall F5: RDMA path mini-diagram (GPU -> NIC -> Switch -> NIC -> GPU, bypassing CPU)
s = Svg(900, 200)
box(s, 125, 8, 245, 48, 'fr', 'Host CPU bypassed', tsize=24, dash='6,4')
box(s, 530, 8, 245, 48, 'fr', 'Host CPU bypassed', tsize=24, dash='6,4')
box(s, 10, 76, 146, 66, 'c1', 'GPU HBM', tsize=25)
s.line(156, 109, 196, 109, V['st'], 3.5, arrow=True)
box(s, 202, 76, 114, 66, 'st', 'NIC', tsize=26)
s.line(316, 109, 366, 109, V['c2'], 3.5, arrow=True)
box(s, 372, 76, 156, 66, 'c2', 'Switch', 'shared buffer', tsize=24, ssize=21)
s.line(528, 109, 578, 109, V['c2'], 3.5, arrow=True)
box(s, 584, 76, 114, 66, 'st', 'NIC', tsize=26)
s.line(698, 109, 738, 109, V['st'], 3.5, arrow=True)
box(s, 744, 76, 146, 66, 'c1', 'GPU HBM', tsize=25)
s.text(450, 178, 'shared switch queues: slowest flow sets collective finish', 25, V['c2_text'], 'bold')
s.save(out / 'rdma_recall_strip.svg')
print('figures written to', out)
