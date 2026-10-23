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
_VALS = {'params_b': '175', 'run_gpus': '2,048', 'run_nodes': '256', 'gpus_per_node_word': 'eight', 'run_days': '30', 'tokens_t': '1.5', 'token_bytes': '4', 'save_min': '10', 'save_s': '600', 'saves_per_hour_word': 'six', 'nvme_drives_word': 'four', 'nvme_drives': '4', 'corpus_tb': '6', 'ckpt_gb': '2,450', 'ckpt_tb': '2.45', 'saves': '4,320', 'written_pb': '10.6', 'written_tb': '10,584', 'write_read': '1,764', 'ckpt_bpp': '14', 'bpp_fp16': '2', 'bpp_fp32': '4', 'bpp_adam': '8', 'bpp_train': '16', 'bpp_grad': '2', 'weights_gb': '350', 'master_gb': '700', 'adam_gb': '1,400', 'avg_read_mbs': '2.3', 'shards': '2,000', 'shard_gb': '3', 'archive_tb': '116.8', 'source_tb': '3', 'archive_rate': '0.004', 'hbm_tbs': '3.35', 'hbm_gbs': '3,350', 'hbm_cap_gb': '80', 'dram_gbs': '200', 'gen4_gbs': '7', 'gen5_gbs': '14', 'nvme_lat_us': '100', 'hbm_lat_ns': '273', 'dram_lat_ns': '100', 'raid_gbs': '28', 'pfs_tbs': '1', 'obj_agg_gbs': '100', 'pfs_share_gbs': '3.9', 'r_hbm_dram': '16.8', 'r_dram_nvme': '7.1', 'r_nvme_pfs': '7.2', 'r_min': '7', 'r_max': '17', 'hot_hit_pct': '95', 'hot_miss_pct': '5', 'hot_hit': '0.95', 'hot_miss': '0.05', 'hot_penalty': '100', 'hot_mean': '5.95', 'hot_frac_pct': '17', 'hot_lost_pct': '83', 'hot_miss_over_hit_word': 'five', 'mm_corpus_tb': '35', 'mm_local_tb': '30.7', 'mm_hit_pct': '88', 'mm_penalty': '7', 'mm_cold_gbs': '4', 'mm_rate_gbs': '16', 'mm_frac_pct': '58', 'wall_plateau': '4', 'step_ms': '200', 'step_s': '0.2', 'seq_tokens': '4,096', 'img_per_gpu': '256', 'img_kb': '116', 'img_kb1': '115.7', 'text_mbs': '168', 'text_mbs1': '167.8', 'img_gbs': '303', 'img_text': '1,800', 'video_mb': '1', 'video_img': '8.6', 'mlperf_results': '124', 'mlp_unet3d_a100': '1.39', 'mlp_unet3d_h100': '2.75', 'mlp_cosmoflow_a100': '0.35', 'mlp_cosmoflow_h100': '0.55', 'mlp_resnet50_a100': '0.09', 'mlp_resnet50_h100': '0.18', 'mlp_gen_min': '1.6', 'mlp_gen_max': '2', 'ex_accels': '64', 'ex_hosts': '16', 'ex_gbs': '33.87', 'ex_busy': '70.23', 'io_ms': '250', 'seq_ms': '450', 'seq_stall_pct': '55.6', 'ovl_ms': '250', 'ovl_wait_ms': '50', 'ovl_stall_pct': '20', 'p99_obj_ms': '500', 'p99_nvme_ms': '50', 'span_obj': '2.5', 'depth_obj': '3', 'depth_safety': '5', 'depth_safety_word': 'five', 'depth_nvme_word': 'one', 'batch_mb': '40', 'prefetch_node_gb': '1.6', 'gds_rate': '64,000', 'gds_host_us': '120', 'gds_gds_us': '30', 'gds_host_cpu': '7.68', 'gds_gds_cpu': '1.92', 'gds_host_cores': '8', 'gds_gds_cores': '2', 'gds_freed': '6', 'files_m': '200', 'tar_shards': '50,000', 'files_per_shard': '4,000', 'fsr': '500', 'fs1': '5', 'oss': '100', 'oss_gbs': '10', 'tail_servers': '100', 'tail_slow_pct': '1', 'tail_fast': '0.99', 'tail_all_fast': '0.366', 'tail_some_slow_pct': '63', 'fs3_tibs': '6.6', 'fs3_nodes': '180', 'fs3_ssds': '16', 'fs3_ssds_word': 'sixteen', 'fs3_ssd_tib': '14', 'fs3_nics': '2', 'fs3_nics_word': 'two', 'fs3_nic_gbps': '200', 'fs3_clients': '500', 'fs3_per_node_gib': '37.5', 'tib_tb': '1.1', 'shard_node_gb': '9.6', 'pause_s': '0.34', 'drain_s': '2.45', 'both_s': '2.8', 'pause_pct': '0.06', 'storm_params_b': '70', 'storm_state_gb': '980', 'storm_replicas': '1,024', 'storm_ingest_gbs': '100', 'storm_tb': '1,004', 'storm_s': '10,035', 'storm_min': '167', 'once_s': '9.8', 'storm_ratio': '1,024', 'storm_train_s': '1,000', 'syn_tb': '1', 'syn_rest_tb': '1.4', 'syn_moved_tb': '4.4', 'syn_passes_word': 'three', 'ds_tb': '100', 'obj_rate': '0.02', 'egress_rate': '0.09', 'nvme_rate': '0.10', 'store_year': '24,000', 'egress_one': '9,000', 'egress_reads': '10', 'egress_all': '90,000', 'files_get_m': '100', 'get_fees': '40', 'restore': '11,000', 'idle_gpus': '1,024', 'gpu_hour': '2', 'fleet_day': '49,152', 'util_high': '90', 'util_low': '70', 'util_drop': '20', 'idle_day': '9,830', 'ss_tb': '50', 'ss_epochs': '20', 'ss_runs': '4', 'ss_runs_word': 'four', 'ss_source': '12,000', 'ss_per_run': '90,000', 'ss_stream_egress': '360,000', 'ss_stream': '372,000', 'ss_local': '60,000', 'ss_stage_egress': '18,000', 'ss_stage': '90,000', 'ss_saving': '282,000', 'ss_per_epoch': '18,000', 'ss_break_even': '4.3', 'load_obj_gbs': '1', 'load_pfs_gbs': '4', 'load_obj_s': '350', 'load_pfs_s': '87.5', 'load_gen5_s': '25', 'load_stripe_s': '6.2', 'load_dram_s': '1.8', 'load_drives_word': 'eight', 'rollout_replicas': '100', 'rollout_tb': '35', 'rollout_pfs_s': '35', 'codec_dense': '4', 'codec_light': '3', 'codec_dense_gbs': '0.5', 'codec_light_gbs': '1.5'}
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
rows = [{'division': 'Closed', 'submitter': 'DDN', 'system': 'AI400X2T_24x14TiB_nvme_8xNDR200_16clients/cosmoflow-h100', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 64.0, 'num_hosts': 16.0, 'num_runs': 5.0, 'au_mean_pct': 70.23, 'io_GBps': 33.87, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'DDN', 'system': 'AI400X2T_24x14TiB_nvme_8xNDR200_18clients/unet3d-a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 72.0, 'num_hosts': 18.0, 'num_runs': 5.0, 'au_mean_pct': 92.49, 'io_GBps': 99.71, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'DDN', 'system': 'AI400X2T_24x14TiB_nvme_8xNDR200_18clients/unet3d-h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 36.0, 'num_hosts': 18.0, 'num_runs': 5.0, 'au_mean_pct': 93.52, 'io_GBps': 99.02, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'DDN', 'system': 'AI400X2T_24x14TiB_nvme_8xNDR200_30clients/cosmoflow-a100', 'workload': 'cosmoflow', 'accelerator': 'A100', 'num_accelerators': 150.0, 'num_hosts': 30.0, 'num_runs': 5.0, 'au_mean_pct': 70.52, 'io_GBps': 50.11, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'DDN', 'system': 'AI400X2T_24x14TiB_nvme_8xNDR200_32clients/resnet50-a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 864.0, 'num_hosts': 32.0, 'num_runs': 5.0, 'au_mean_pct': 91.3, 'io_GBps': 79.06, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'DDN', 'system': 'AI400X2T_24x14TiB_nvme_8xNDR200_32clients/resnet50-h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 512.0, 'num_hosts': 32.0, 'num_runs': 5.0, 'au_mean_pct': 92.39, 'io_GBps': 92.07, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'HPE', 'system': 'Cray_C500_4xHDR200_1host/resnet50_h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 20.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 92.2, 'io_GBps': 3.57, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'HPE', 'system': 'Cray_C500_4xHDR200_1host/unet3d_h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 2.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 92.2, 'io_GBps': 5.55, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'HPE', 'system': 'Cray_C500_4xHDR200_3hosts/unet3d_h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 3.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 90.78, 'io_GBps': 8.19, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'HPE', 'system': 'Cray_ClusterStor_E1000_10xHDR200_1host/resnet50_a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 50.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 97.73, 'io_GBps': 4.91, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'HPE', 'system': 'Cray_ClusterStor_E1000_10xHDR200_1host/resnet50_h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 34.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 92.17, 'io_GBps': 6.12, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'HPE', 'system': 'Cray_ClusterStor_E1000_10xHDR200_1host/unet3d_a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 9.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 96.1, 'io_GBps': 12.84, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'HPE', 'system': 'Cray_ClusterStor_E1000_10xHDR200_1host/unet3d_h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 5.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 92.72, 'io_GBps': 13.95, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'HPE', 'system': 'Cray_ClusterStor_E1000_6xSlingShot200_5hosts/unet3d_h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 15.0, 'num_hosts': 5.0, 'num_runs': 5.0, 'au_mean_pct': 91.19, 'io_GBps': 40.98, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Hammerspace', 'system': 'Hammerspace_resnet50_1Client_A100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 80.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 94.09, 'io_GBps': 7.56, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Hammerspace', 'system': 'Hammerspace_resnet50_1Client_H100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 27.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 98.37, 'io_GBps': 5.18, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Hammerspace', 'system': 'Hammerspace_resnet50_5Client_A100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 370.0, 'num_hosts': 5.0, 'num_runs': 5.0, 'au_mean_pct': 90.91, 'io_GBps': 33.74, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Hammerspace', 'system': 'Hammerspace_resnet50_5Client_H100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 130.0, 'num_hosts': 5.0, 'num_runs': 5.0, 'au_mean_pct': 92.11, 'io_GBps': 23.36, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Hammerspace', 'system': 'Hammerspace_unet3d_1Client_A100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 7.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 97.43, 'io_GBps': 10.35, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Hammerspace', 'system': 'Hammerspace_unet3d_1Client_H100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 2.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 96.55, 'io_GBps': 5.79, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Hammerspace', 'system': 'Hammerspace_unet3d_5Client_A100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 35.0, 'num_hosts': 5.0, 'num_runs': 5.0, 'au_mean_pct': 92.83, 'io_GBps': 49.15, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Hammerspace', 'system': 'Hammerspace_unet3d_5Client_H100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 10.0, 'num_hosts': 5.0, 'num_runs': 5.0, 'au_mean_pct': 96.06, 'io_GBps': 28.83, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Huawei', 'system': 'OceanStor_A800_51_hosts/unet3d-h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 255.0, 'num_hosts': 51.0, 'num_runs': 5.0, 'au_mean_pct': 91.75, 'io_GBps': 695.48, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_AMDHost/cosmoflow-a100/1host@37GPUs', 'workload': 'cosmoflow', 'accelerator': 'A100', 'num_accelerators': 37.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 72.15, 'io_GBps': 13.04, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_AMDHost/cosmoflow-a100/6hosts@25GPUs', 'workload': 'cosmoflow', 'accelerator': 'A100', 'num_accelerators': 150.0, 'num_hosts': 6.0, 'num_runs': 5.0, 'au_mean_pct': 71.6, 'io_GBps': 52.09, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_AMDHost/cosmoflow-h100/1host@23GPUs', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 23.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 73.67, 'io_GBps': 13.03, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_AMDHost/cosmoflow-h100/6hosts@17GPUs', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 102.0, 'num_hosts': 6.0, 'num_runs': 5.0, 'au_mean_pct': 70.63, 'io_GBps': 54.98, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_AMDHost/unet3d-a100/12hosts@22GPUs', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 264.0, 'num_hosts': 12.0, 'num_runs': 5.0, 'au_mean_pct': 90.43, 'io_GBps': 360.39, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_AMDHost/unet3d-a100/1host@31GPUs', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 31.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.72, 'io_GBps': 42.54, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_AMDHost/unet3d-h100/10hosts@13GPUs', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 130.0, 'num_hosts': 10.0, 'num_runs': 5.0, 'au_mean_pct': 93.66, 'io_GBps': 362.52, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_AMDHost/unet3d-h100/1host@16GPUs', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 16.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.03, 'io_GBps': 42.91, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_IntelHost/cosmoflow-a100/1hosts@51GPUs', 'workload': 'cosmoflow', 'accelerator': 'A100', 'num_accelerators': 51.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 70.83, 'io_GBps': 17.61, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_IntelHost/cosmoflow-h100/1host@30GPUs', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 30.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 70.25, 'io_GBps': 16.18, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_IntelHost/unet3d-a100/1host@48GPUs', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 48.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 93.31, 'io_GBps': 67.59, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'IEI', 'system': 'AS13000_16x7.68TiB_NVMe_4xNDR400_3xStorageNode_IntelHost/unet3d-h100/1host@26GPUs', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 26.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 91.33, 'io_GBps': 70.95, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Lightbits_Labs', 'system': 'Lightbits-3-SappRapid-Clients-3-icelake-target/cosmoflow_a100', 'workload': 'cosmoflow', 'accelerator': 'A100', 'num_accelerators': 45.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 72.8, 'io_GBps': 15.94, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Lightbits_Labs', 'system': 'Lightbits-3-SappRapid-Clients-3-icelake-target/cosmoflow_h100', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 33.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 71.54, 'io_GBps': 18.07, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Lightbits_Labs', 'system': 'Lightbits-6-IceLake-Clients-3-SappRapid-target/resnet50_a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 282.0, 'num_hosts': 6.0, 'num_runs': 5.0, 'au_mean_pct': 94.72, 'io_GBps': 26.74, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Lightbits_Labs', 'system': 'Lightbits-6-IceLake-Clients-3-SappRapid-target/resnet50_h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 198.0, 'num_hosts': 6.0, 'num_runs': 5.0, 'au_mean_pct': 95.67, 'io_GBps': 36.92, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Lightbits_Labs', 'system': 'Lightbits-6-IceLake-Clients-3-SappRapid-target/unet3d_a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 24.0, 'num_hosts': 6.0, 'num_runs': 5.0, 'au_mean_pct': 92.45, 'io_GBps': 31.43, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Lightbits_Labs', 'system': 'Lightbits-6-IceLake-Clients-3-SappRapid-target/unet3d_h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 12.0, 'num_hosts': 6.0, 'num_runs': 5.0, 'au_mean_pct': 94.97, 'io_GBps': 33.76, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-amd-nvme-tcp-dpu/cosmoflow-a100', 'workload': 'cosmoflow', 'accelerator': 'A100', 'num_accelerators': 38.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 71.29, 'io_GBps': 13.21, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-amd-nvme-tcp-dpu/cosmoflow-h100', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 21.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 70.91, 'io_GBps': 11.43, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-amd-nvme-tcp-dpu/resnet50-a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 144.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.35, 'io_GBps': 13.06, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-amd-nvme-tcp-dpu/resnet50-h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 128.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.28, 'io_GBps': 22.48, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-amd-nvme-tcp-dpu/unet3d-a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 21.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.89, 'io_GBps': 29.14, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-amd-nvme-tcp-dpu/unet3d-h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 10.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.28, 'io_GBps': 26.74, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-intel-nvme-tcp-dpu/cosmoflow-a100', 'workload': 'cosmoflow', 'accelerator': 'A100', 'num_accelerators': 40.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 70.42, 'io_GBps': 13.74, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-intel-nvme-tcp-dpu/cosmoflow-h100', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 22.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 71.02, 'io_GBps': 12.0, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-intel-nvme-tcp-dpu/resnet50-a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 144.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.7, 'io_GBps': 13.11, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-intel-nvme-tcp-dpu/resnet50-h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 128.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.32, 'io_GBps': 22.48, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-intel-nvme-tcp-dpu/unet3d-a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 21.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.98, 'io_GBps': 29.07, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-intel-nvme-tcp-dpu/unet3d-h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 10.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.85, 'io_GBps': 27.0, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-intel-nvme-tcp-sw/cosmoflow-h100', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 22.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 70.21, 'io_GBps': 11.87, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-intel-nvme-tcp-sw/resnet50-a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 140.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.39, 'io_GBps': 12.7, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-intel-nvme-tcp-sw/resnet50-h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 92.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.22, 'io_GBps': 16.16, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-intel-nvme-tcp-sw/unet3d-a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 15.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 92.04, 'io_GBps': 20.97, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'MangoBoost', 'system': '1-client-intel-nvme-tcp-sw/unet3d-h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 5.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 91.38, 'io_GBps': 13.53, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_6500_Gen4_High_Capacity_NVMe/cosmoflow/a100', 'workload': 'cosmoflow', 'accelerator': 'A100', 'num_accelerators': 15.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 72.44, 'io_GBps': 5.31, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_6500_Gen4_High_Capacity_NVMe/cosmoflow/h100', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 9.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 71.85, 'io_GBps': 4.98, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_6500_Gen4_High_Capacity_NVMe/resnet50/a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 72.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 91.16, 'io_GBps': 6.6, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_6500_Gen4_High_Capacity_NVMe/resnet50/h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 37.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 92.25, 'io_GBps': 6.66, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_6500_Gen4_High_Capacity_NVMe/unet3d/a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 3.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 97.96, 'io_GBps': 4.47, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_6500_Gen4_High_Capacity_NVMe/unet3d/h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 1.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 96.44, 'io_GBps': 2.91, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_9400_Gen4_Performance_NVMe/cosmoflow/a100', 'workload': 'cosmoflow', 'accelerator': 'A100', 'num_accelerators': 18.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 70.84, 'io_GBps': 6.23, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_9400_Gen4_Performance_NVMe/cosmoflow/h100', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 12.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 71.52, 'io_GBps': 6.6, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_9400_Gen4_Performance_NVMe/resnet50/h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 38.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.87, 'io_GBps': 6.73, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_9400_Gen4_Performance_NVMe/unet3d/a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 4.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 96.0, 'io_GBps': 5.69, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_9400_Gen4_Performance_NVMe/unet3d/h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 2.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 96.24, 'io_GBps': 5.71, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_9550_Gen5_Performance_NVMe/cosmoflow/a100', 'workload': 'cosmoflow', 'accelerator': 'A100', 'num_accelerators': 20.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 72.3, 'io_GBps': 7.06, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_9550_Gen5_Performance_NVMe/cosmoflow/h100', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 13.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 72.38, 'io_GBps': 7.24, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_9550_Gen5_Performance_NVMe/resnet50/a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 115.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.66, 'io_GBps': 10.45, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_9550_Gen5_Performance_NVMe/resnet50/h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 58.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.33, 'io_GBps': 10.22, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_9550_Gen5_Performance_NVMe/unet3d/a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 8.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.29, 'io_GBps': 9.26, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Micron', 'system': 'Micron_9550_Gen5_Performance_NVMe/unet3d/h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 4.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.27, 'io_GBps': 9.94, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-32-Nodes-21-Hosts-resnet50-a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 2100.0, 'num_hosts': 21.0, 'num_runs': 5.0, 'au_mean_pct': 91.74, 'io_GBps': 192.9, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-32-Nodes-50-Hosts-unet3d-h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 100.0, 'num_hosts': 50.0, 'num_runs': 5.0, 'au_mean_pct': 91.56, 'io_GBps': 262.99, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-32-Nodes-65-Hosts-unet3d-a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 195.0, 'num_hosts': 65.0, 'num_runs': 5.0, 'au_mean_pct': 93.71, 'io_GBps': 273.23, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-32-Nodes-66-Hosts-resnet50-h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 1056.0, 'num_hosts': 66.0, 'num_runs': 5.0, 'au_mean_pct': 92.88, 'io_GBps': 191.04, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-4-Nodes-1-Host-resnet50-a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 120.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 91.0, 'io_GBps': 10.95, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-4-Nodes-1-Host-resnet50-h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 65.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 92.97, 'io_GBps': 11.76, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-4-Nodes-1-Host-unet3d-a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 8.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 96.83, 'io_GBps': 11.46, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-4-Nodes-1-Host-unet3d-h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 4.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 96.75, 'io_GBps': 11.1, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-4-Nodes-5-Hosts-cosmoflow-a100', 'workload': 'cosmoflow', 'accelerator': 'A100', 'num_accelerators': 40.0, 'num_hosts': 5.0, 'num_runs': 5.0, 'au_mean_pct': 70.67, 'io_GBps': 13.78, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-4-Nodes-5-Hosts-cosmoflow-h100', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 20.0, 'num_hosts': 5.0, 'num_runs': 5.0, 'au_mean_pct': 72.28, 'io_GBps': 11.1, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-7-Nodes-10-Hosts-resnet50-a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 500.0, 'num_hosts': 10.0, 'num_runs': 5.0, 'au_mean_pct': 94.41, 'io_GBps': 47.41, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-7-Nodes-10-Hosts-resnet50-h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 250.0, 'num_hosts': 10.0, 'num_runs': 5.0, 'au_mean_pct': 94.46, 'io_GBps': 46.07, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-7-Nodes-10-Hosts-unet3d-a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 40.0, 'num_hosts': 10.0, 'num_runs': 5.0, 'au_mean_pct': 93.14, 'io_GBps': 54.79, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'Nutanix', 'system': 'Nutanix-Unified-Storage-7-Nodes-10-Hosts-unet3d-h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 20.0, 'num_hosts': 10.0, 'num_runs': 5.0, 'au_mean_pct': 94.73, 'io_GBps': 56.07, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'WEKA', 'system': 'weka-1-client-dual-socket-56-cores/resnet-a100', 'workload': 'resnet50', 'accelerator': 'A100', 'num_accelerators': 90.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 95.96, 'io_GBps': 8.68, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'WEKA', 'system': 'weka-1-client-dual-socket-56-cores/resnet-h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 74.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 94.99, 'io_GBps': 13.72, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'WEKA', 'system': 'weka-1-client-dual-socket-56-cores/unet3d-a100', 'workload': 'unet3d', 'accelerator': 'A100', 'num_accelerators': 24.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.21, 'io_GBps': 31.12, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'WEKA', 'system': 'weka-1-client-dual-socket-56-cores/unet3d-h100', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 13.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 91.49, 'io_GBps': 34.57, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient1/cosmoflow-h100/cosmoflow_1acc', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 1.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 82.8, 'io_GBps': 0.64, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient1/cosmoflow-h100/cosmoflow_26acc', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 26.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 71.99, 'io_GBps': 14.37, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient1/cosmoflow-h100/cosmoflow_8acc', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 8.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 76.64, 'io_GBps': 4.72, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient1/resnet-h100/resnet50_106acc', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 106.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 90.75, 'io_GBps': 18.7, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient1/resnet-h100/resnet50_1acc', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 1.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 99.44, 'io_GBps': 0.19, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient1/resnet-h100/resnet50_32acc', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 32.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 94.53, 'io_GBps': 5.89, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient1/unet3d-h100/unet3d_12acc', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 12.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 97.95, 'io_GBps': 33.4, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient1/unet3d-h100/unet3d_1acc', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 1.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 98.63, 'io_GBps': 2.97, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient1/unet3d-h100/unet3d_8acc', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 8.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 96.39, 'io_GBps': 21.42, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient3/cosmoflow-h100/cosmoflow_18acc', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 18.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 73.03, 'io_GBps': 10.1, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient3/cosmoflow-h100/cosmoflow_3acc', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 3.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 79.38, 'io_GBps': 1.83, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient3/cosmoflow-h100/cosmoflow_60acc', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 60.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 71.94, 'io_GBps': 32.95, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient3/resnet-h100/resnet50_288acc', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 288.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 96.13, 'io_GBps': 53.7, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient3/resnet-h100/resnet50_3acc', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 3.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 99.3, 'io_GBps': 0.58, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient3/resnet-h100/resnet50_96acc', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 96.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 94.24, 'io_GBps': 17.65, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient3/unet3d-h100/unet3d_12acc', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 12.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 93.41, 'io_GBps': 32.45, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient3/unet3d-h100/unet3d_36acc', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 36.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 95.76, 'io_GBps': 97.61, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F8000X_8x7000GiB_nvme_2xHDR200_Nclient3/unet3d-h100/unet3d_3acc', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 3.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 97.51, 'io_GBps': 8.8, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient1/cosmoflow-h100/cosmoflow_30acc', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 30.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 84.53, 'io_GBps': 19.43, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient1/cosmoflow-h100/cosmoflow_60acc', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 60.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 70.51, 'io_GBps': 32.29, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient1/resnet-h100/resnet50_188acc', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 188.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 95.82, 'io_GBps': 35.05, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient1/resnet-h100/resnet50_94acc', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 94.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 99.25, 'io_GBps': 18.2, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient1/unet3d-h100/unet3d_10acc', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 10.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 98.58, 'io_GBps': 28.92, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient1/unet3d-h100/unet3d_20acc', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 20.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 97.42, 'io_GBps': 55.11, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient3/cosmoflow-h100/cosmoflow_120acc', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 120.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 70.55, 'io_GBps': 64.09, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient3/cosmoflow-h100/cosmoflow_60acc', 'workload': 'cosmoflow', 'accelerator': 'H100', 'num_accelerators': 60.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 81.03, 'io_GBps': 37.1, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient3/resnet-h100/resnet50_270acc', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 270.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 97.39, 'io_GBps': 51.13, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient3/resnet-h100/resnet50_540acc', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 540.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 93.17, 'io_GBps': 97.9, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient3/unet3d-h100/unet3d_30acc', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 30.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 97.27, 'io_GBps': 85.32, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'YanRongTech', 'system': 'F9000X_10x7000GiB_nvme_2xNDR400_Nclient3/unet3d-h100/unet3d_60acc', 'workload': 'unet3d', 'accelerator': 'H100', 'num_accelerators': 60.0, 'num_hosts': 3.0, 'num_runs': 5.0, 'au_mean_pct': 94.87, 'io_GBps': 160.89, 'au_check': 'pass'}, {'division': 'Closed', 'submitter': 'simplyblock', 'system': 'resnet-h100', 'workload': 'resnet50', 'accelerator': 'H100', 'num_accelerators': 1.0, 'num_hosts': 1.0, 'num_runs': 5.0, 'au_mean_pct': 99.61, 'io_GBps': 0.19, 'au_check': 'pass'}]
WL = [('unet3d', '3D U-Net', 'memory'), ('cosmoflow', 'CosmoFlow', 'compute'), ('resnet50', 'ResNet-50', 'data')]
fig, ax = plt.subplots(figsize=(3.9, 1.95))
for w, name, ck in WL:
    for acc, mk in zip(('A100', 'H100'), ('o', '^')):
        pts = [(r['num_accelerators'], r['io_GBps']) for r in rows if r['workload'] == w and r['accelerator'] == acc]
        x, y = zip(*pts)
        ax.scatter(x, y, s=34, marker=mk, color=C[ck], alpha=0.85, edgecolor='white', linewidth=0.5)
for (w, name, ck), (tx, ty) in zip(WL, [(200, 2.6), (200, 1.0), (200, 0.38)]):
    ax.text(tx, ty, name, color=C[ck], fontsize=12.5, fontweight='bold')
ax.text(1.1, 400, 'circle: A100   triangle: H100', fontsize=11, color=C['text'])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Emulated accelerators')
ax.set_ylabel('Read throughput (GB/s)')
ax.grid(color=C['grid'], lw=0.6)
ax.minorticks_off()
sp.save(fig, out / 'mlperf_scatter.svg')
tiers = [('GPU HBM', 'c1', '80 GB', '3.35 TB/s per GPU', '~273 ns'), ('Host DRAM', 'c1', '0.5 to 2 TB', '200 GB/s per node', '~100 ns'), ('Local NVMe', 'st', '4 to 30 TB', '7 to 25 GB/s', '~100 µs'), ('Parallel file system', 'st', '100+ PB', '1+ TB/s shared', '~1 ms'), ('Object storage', 'st', 'very large', '100 GB/s shared', '~50 ms'), ('Archive', 'fr', 'very large', '1 GB/s', 'min to h')]
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
T_IO, T_COMP = (250, 200)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.9, 1.65))
    lanes = {'Seq. I/O': 3.0, 'Seq. GPU': 2.4, 'Overlap I/O': 1.0, 'Overlap GPU': 0.4}
    for t0 in (0, 450):
        ax.broken_barh([(t0, T_IO)], (lanes['Seq. I/O'] - 0.22, 0.44), color=C['data'])
        ax.broken_barh([(t0 + T_IO, T_COMP)], (lanes['Seq. GPU'] - 0.22, 0.44), color=C['memory'])
    ax.text(0, 3.32, 'Sequential: read 250 ms, then compute 200 ms', ha='left', va='bottom', fontsize=11.5, fontweight='bold', color=C['text'])
    ax.text(905, 2.7, '450 ms per step\n55.6% stalled', ha='left', va='center', color=C['danger'], fontsize=12, fontweight='bold')
    a = 1.0 if k == 2 else 0.0
    for i in range(4):
        ax.broken_barh([(i * T_IO, T_IO)], (lanes['Overlap I/O'] - 0.22, 0.44), color=C['data'], alpha=a, edgecolor='white')
        if i < 3:
            ax.broken_barh([(T_IO + i * T_IO, T_COMP)], (lanes['Overlap GPU'] - 0.22, 0.44), color=C['memory'], alpha=a)
    ax.text(905, 0.7, '250 ms per step\n20% stalled', ha='left', va='center', color=C['danger'], fontsize=12, fontweight='bold', alpha=a)
    ax.set_yticks(list(lanes.values()), ['read', 'compute', 'read', 'compute'], fontsize=11.5)
    ax.text(0, 1.32, 'Overlapped', ha='left', va='bottom', fontsize=11.5, fontweight='bold', color=C['text'], alpha=a)
    ax.set_xlim(0, 900)
    ax.set_ylim(-0.1, 3.75)
    ax.set_xlabel('Time (ms)')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    sp.save(fig, out / f'stall_step{k}.svg')
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
s = Svg(900, 290)
s.text(10, 24, '(a) Through host DRAM', 22, V['text'], 'bold', anchor='start')
s.text(470, 24, '(b) GPUDirect Storage', 22, V['text'], 'bold', anchor='start')
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
s.line(452, 36, 452, 285, V['fr'], 2, dash='6,6')
box(s, 470, 42, 120, 60, 'fr', 'CPU')
box(s, 625, 36, 270, 100, 'c1', 'Host DRAM', 'not on the path', ssize=20)
box(s, 470, 222, 150, 60, 'st', 'NVMe SSD')
box(s, 745, 222, 150, 60, 'c1', 'GPU HBM')
s.line(530, 102, 530, 216, V['mute'], 3, dash='7,6', arrow=True)
s.text(540, 165, 'control', 20, V['mute'], 'bold', anchor='start')
s.poly([(585, 222), (585, 190), (820, 190), (820, 216)], V['st'], 4, arrow=True)
s.text(702, 180, 'one DMA', 20, V['st_text'], 'bold')
s.save(out / 'gds.svg')
s = Svg(900, 320)
s.text(105, 24, 'GPU nodes', 21, V['mute'], 'bold')
s.text(480, 24, 'Servers', 21, V['mute'], 'bold')
for i, y in enumerate((120, 190, 262)):
    box(s, 15, y, 180, 52, 'c1', f"Client {['1', '2', 'M'][i]}", tsize=21)
box(s, 370, 34, 220, 66, 'c3', 'Metadata server', 'open, stat, lock', ssize=20)
s.poly([(105, 120), (105, 67), (364, 67)], V['c3'], 3, arrow=True)
s.text(232, 58, '1. look up', 20, V['c3_text'], 'bold')
s.text(755, 60, 'a few hundred', 21, V['c3_text'], 'bold')
s.text(755, 86, 'thousand ops/s', 21, V['c3_text'], 'bold')
for i, y in enumerate((120, 190, 262)):
    box(s, 390, y, 180, 52, 'st', f"OSS {['1', '2', '100'][i]}: 10 GB/s", tsize=20)
    s.line(300, y + 26, 384, y + 26, V['st'], 3, arrow=True)
s.line(300, 146, 300, 288, V['st'], 3)
for y in (146, 216, 288):
    s.line(195, y, 300, y, V['st'], 3)
s.text(248, 112, '2. read', 20, V['st_text'], 'bold')
s.text(755, 210, '100 × 10 GB/s', 21, V['st_text'], 'bold')
s.text(755, 236, '= 1 TB/s of data', 21, V['st_text'], 'bold')
s.save(out / 'pfs.svg')
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
sp.hbars(out / 'coldstart.svg', [('Object store, 1 GB/s', 350.0, 'danger', '350 s'), ('File system share, 4 GB/s', 87.5, 'compute', '87.5 s'), ('One NVMe, 14 GB/s', 25.0, 'data', '25 s'), ('Eight NVMe, 7 GB/s each', 6.25, 'data', '6.2 s'), ('Host DRAM, 200 GB/s', 1.75, 'memory', '1.8 s')], 's', log=True, size=(4.05, 1.55), xlabel='Seconds to load (log scale)')
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
print('figures written to', out)
s = Svg(900, 220)
s.line(190, 182, 880, 182, V['mute'], 2, arrow=True)
s.text(880, 210, 'time', 22, V['mute'], anchor='end')
s.text(175, 62, 'First read', 24, V['text'], 'bold', anchor='end')
s.rect(195, 36, 640, 44, V['bad_fill'], V['bad'])
s.text(634, 66, 'server A: slow, in the tail', 22, V['bad_text'], 'bold')
s.line(430, 24, 430, 182, V['c1'], 2.5, dash='7 5')
s.text(436, 210, 'p99 delay', 22, V['c1'], 'bold', anchor='middle')
s.text(175, 132, 'Second read', 24, V['text'], 'bold', anchor='end')
s.rect(432, 106, 180, 44, V['st_fill'], V['st'])
s.text(522, 136, 'replica B', 22, V['st_text'], 'bold')
s.text(626, 136, 'first response wins', 24, V['st_text'], 'bold', anchor='start')
s.save(out / 'hedge_timeline.svg')
