#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 5 (Collective Communication), Volume II.

Every number a figure prints or plots comes from calcs.py, which reads it from
MLSysIM (registry values, Literature, and Scenarios.*); the list below names
the source of each figure. Derived and illustrative values say so on the slide.
  - Horovod hook and trace: slides/vol2/05_collective_communication/images/data/horovod_scaling_128gpu.csv and
    HorovodTimelineTrace, Literature.Communication.Horovod* (Sergeev and Del
    Balso 2018; bar heights and phase edges read from the paper's figures).
  - Traffic map: table 1 (Sec. 5.3). Contracts: Sec. 5.4 and figure 2.
  - Critical size alpha*beta per interconnect: Scenarios.CriticalSizeByInterconnect
    (table 4); bucket chunks: Scenarios.DdpBucketRingNdr (Napkin Math 5.1).
  - Scoreboard: Scenarios.GradientBufferAllReduceNdr (parameter server, simple
    tree, ring) and Scenarios.HierarchicalAllReduce8x8DgxH100.
  - Rounds: ring 2(N-1), tree 2 log2 N, marked at the reference cluster.
  - Crossover: Scenarios.RingVsTreeSmallBuffer; curves are MLSysIM's ring and
    tree time models (Napkin Math 5.3).
  - Bandwidth per GPU: H100 NVLink per direction and the NDR port (Sec. 5.7.1).
  - Hierarchical phases: Scenarios.HierarchicalAllReduce8x8DgxH100 (Napkin Math 5.4).
  - MoE dispatch: Scenarios.MoeDispatch8Gpu64Experts (Napkin Math 5.5, 5.6).
  - Compression: Scenarios.CompressionPayback8x (Napkin Math 5.7) and
    Scenarios.Int8CompressionRun (Sec. 5.9.5).
  - 1-bit Adam: Literature.Communication.OneBitAdam* (Tang et al. 2021).
  - Overlap: Scenarios.OverlapBudget7B64Gpu and its ideal-bucket variant
    (Napkin Math 5.9, Sec. 5.10.4).

Run from this folder: python3 make_figs.py
"""
from mlsysim.core.series_loader import load_series
import mlsysim
_NUMS = {'hvd_gpus': 128.0, 'hvd_gbe': 25.0, 'hvd_eff_pct': 88.0, 'hvd_base_eff_pct': 50.0, 'hvd_ideal_incv3': 17000.0, 'hvd_tf_incv3': 9000.0, 'hvd_hv_incv3': 15000.0, 'hvd_tf_incv3_pct': 53.0, 'hvd_hv_incv3_pct': 88.0, 'hvd_ideal_resnet': 15000.0, 'hvd_tf_resnet': 6200.0, 'hvd_hv_resnet': 13200.0, 'hvd_tf_resnet_pct': 41.0, 'hvd_hv_resnet_pct': 88.0, 'hvd_fusion_pct': 65.0, 'hvd_fusion_mb': 64.0, 'hvd_rdma_lo_pct': 3.0, 'hvd_rdma_hi_pct': 4.0, 'hvd_rdma_vgg_pct': 30.0, 'hvd_t0_ms': 18500.0, 'hvd_ar_start': 163.5, 'hvd_ar_end': 347.6, 'hvd_ar_ms': 184.0, 'hvd_negotiate_ms': 131.0, 'hvd_span_ms': 325.0, 'hvd_nccl_ms': 19.0, 'hvd_nccl_pct': 6.0, 'hvd_fast_ms': 115.0, 'ib_gbs': 50.0, 'nvlink_gbs': 450.0, 'nvlink_bidir_gbs': 900.0, 'nvlink_ib_ratio': 9.0, 'alpha_us': 1.5, 'gpus_per_node': 8.0, 'buf_gb': 1.0, 'grp_gpus': 64.0, 'ref_n': 1024.0, 'run_steps': 100000.0, 'grad70_params_b': 70.0, 'grad70_gb': 140.0, 'bf16_bytes': 2.0, 'fsdp_layers': 32.0, 'fsdp_calls': 96.0, 'nstar_nvlink_kb': 675.0, 'nstar_nvlink_mb': 0.7, 'nstar_pcie_kb': 224.0, 'nstar_roce_kb': 93.7, 'nstar_hdr_kb': 87.5, 'nstar_ndr_kb': 75.0, 'bucket_mb': 25.0, 'bkt_n_small': 64.0, 'bkt_n_large': 1024.0, 'bkt_chunk_small_kb': 390.6, 'bkt_rounds_small': 126.0, 'bkt_bytes_small_ms': 1.0, 'bkt_total_small_ms': 1.2, 'bkt_chunk_large_kb': 24.4, 'bkt_rounds_large': 2046.0, 'bkt_bytes_large_ms': 1.0, 'bkt_total_large_ms': 4.1, 'bkt_chunk_large_kb0': 24.0, 'bkt_round_small_us': 9.3, 'bkt_ratio': 3.5, 'bkt_knee_ratio': 333.0, 'fuse_count': 100.0, 'fuse_msg_kb': 4.0, 'fuse_total_kb': 400.0, 'fuse_sep_us': 150.0, 'fuse_fused_us': 9.5, 'fuse_speedup': 16.0, 'ps_gb': 64.0, 'ps_ms': 1280.0, 'ring_bytes_ms': 39.4, 'tree_bytes_ms': 240.0, 'ps_ring_ratio': 32.5, 'ps_half_n': 32.0, 'tree_transfers_grp': 12.0, 'tv00': 1.0, 'tv01': 5.0, 'tv02': 3.0, 'tv03': 7.0, 'tv10': 2.0, 'tv11': 6.0, 'tv12': 4.0, 'tv13': 8.0, 'tv20': 3.0, 'tv21': 7.0, 'tv22': 5.0, 'tv23': 9.0, 'tv30': 4.0, 'tv31': 8.0, 'tv32': 6.0, 'tv33': 10.0, 'tsum0': 10.0, 'tsum1': 26.0, 'tsum2': 18.0, 'tsum3': 34.0, 'rs1_g0': 17.0, 'rs1_g1': 3.0, 'rs1_g2': 13.0, 'rs1_g3': 11.0, 'rs2_g0': 14.0, 'rs2_g1': 25.0, 'rs2_g2': 6.0, 'rs2_g3': 21.0, 'rs3_g0': 26.0, 'rs3_g1': 18.0, 'rs3_g2': 34.0, 'rs3_g3': 10.0, 'ag1_g0': 10.0, 'ag1_g1': 26.0, 'ag1_g2': 18.0, 'ag1_g3': 34.0, 'ag2_g0': 34.0, 'ag2_g1': 10.0, 'ag2_g2': 26.0, 'ag2_g3': 18.0, 'ag3_g0': 18.0, 'ag3_g1': 34.0, 'ag3_g2': 10.0, 'ag3_g3': 26.0, 'trace_steps': 6.0, 'est_n': 64.0, 'est_n_minus1': 63.0, 'est_factor': 1.96875, 'est_bytes_ms': 5512.5, 'est_rounds': 126.0, 'est_startup_ms': 0.2, 'est_total_s': 5.5, 'est_startup_pct': 0.0034, 'est_saved_s': 2.76, 'est_saved_gpuh': 4900.0, 'ring_rounds_large': 2046.0, 'tree_rounds_large': 20.0, 'round_ratio': 100.0, 'ref_cross_mb': 7.7, 'rt_n': 64.0, 'rt_m_mb': 1.0, 'rt_alpha_us': 10.0, 'rt_beta_gbs': 10.0, 'rt_ring_rounds': 126.0, 'rt_tree_rounds': 12.0, 'rt_ring_lat_us': 1260.0, 'rt_tree_lat_us': 120.0, 'rt_ring_bw_us': 196.9, 'rt_tree_bw_us': 1200.0, 'rt_ring_total_us': 1456.9, 'rt_ring_total_us0': 1457.0, 'rt_tree_total_us': 1320.0, 'rt_margin_pct': 9.0, 'rt_cross_mb': 1.1, 'rt_cross_exact_mb': 1.14, 'hier_nodes': 8.0, 'hier_gpus': 64.0, 'hier_intra_mb': 875.0, 'hier_rs_ms': 1.94, 'hier_shard_mb': 125.0, 'hier_inter_ms': 4.4, 'hier_total_ms': 8.3, 'hier_flat_ms': 40.0, 'hier_speedup': 4.8, 'ref_params_b': 175.0, 'ref_grad_gb': 350.0, 'ref_flat_s': 14.0, 'ref_hier_s': 3.1, 'moe_gpus': 8.0, 'moe_experts': 64.0, 'moe_experts_per_gpu': 8.0, 'moe_tokens_per_gpu': 512.0, 'moe_pair_tokens': 64.0, 'moe_peers': 7.0, 'moe_hidden': 2048.0, 'moe_token_kb': 4.096, 'moe_sent_mb': 1.84, 'moe_rx_us': 36.7, 'moe_naive_gbs': 350.0, 'moe_skew': 1.5, 'moe_uniform_tokens': 448.0, 'moe_hot_tokens': 672.0, 'moe_hot_mb': 2.75, 'moe_hot_us': 55.1, 'cp_comm_ms': 40.0, 'cp_ratio': 8.0, 'cp_xfer_ms': 5.0, 'cp_codec_ms': 2.0, 'cp_total_ms': 7.0, 'cp_speedup': 5.7, 'cp_fast_ms': 1.0, 'enc_fp16': 2.0, 'enc_int8': 4.0, 'enc_topk': 50.0, 'enc_topk_pct': 1.0, 'enc_1bit': 32.0, 'cr_steps': 100000.0, 'cr_steps_c': 110000.0, 'cr_compute_ms': 300.0, 'cr_ratio': 4.0, 'cr_penalty_pct': 10.0, 'cr_slow_comm_ms': 200.0, 'cr_slow_step_ms': 500.0, 'cr_slow_base_s': 50000.0, 'cr_slow_comp_s': 38500.0, 'cr_fast_comm_ms': 30.0, 'cr_fast_step_ms': 330.0, 'cr_fast_base_s': 33000.0, 'cr_fast_comp_s': 33825.0, 'cr_slow_comm_c_ms': 50.0, 'cr_slow_step_c_ms': 350.0, 'cr_fast_saved_ms': 22.5, 'cr_fast_step_c_ms': 307.5, 'cr_slow_gain_pct': 23.0, 'cr_fast_loss_pct': 2.5, 'ef_g1': 0.4, 'ef_s1': 0.4, 'ef_v1': 0.0, 'ef_n1': 0.4, 'ef_g2': 0.3, 'ef_s2': 0.7, 'ef_v2': 1.0, 'ef_n2': -0.3, 'ef_e2': 0.4, 'ef_g3': 0.2, 'ef_s3': -0.1, 'ef_v3': 0.0, 'ef_n3': -0.1, 'ef_e3': -0.3, 'ef_g4': 0.4, 'ef_s4': 0.3, 'ef_v4': 0.0, 'ef_n4': 0.3, 'ef_e4': -0.1, 'ef_g5': 0.3, 'ef_s5': 0.6, 'ef_v5': 1.0, 'ef_n5': -0.4, 'ef_e5': 0.3, 'ef_e1': 0.0, 'ef_naive': 0.0, 'ef_true': 1.6, 'ef_sent': 2.0, 'ef_final': -0.4, 'efc_g': 0.3, 'oba_volume': 5.0, 'oba_bert': 3.3, 'oba_squad': 2.9, 'oba_gpus': 256.0, 'ov_params_b': 7.0, 'ov_layers': 32.0, 'ov_layers_minus1': 31.0, 'ov_gpus': 64.0, 'ov_bucket_mb': 100.0, 'ov_bucket_ms': 3.0, 'ov_bwd_ms': 15.0, 'ov_comm_ms': 27.0, 'ov_total_bwd_ms': 480.0, 'ov_seq_ms': 1344.0, 'ov_exposed_ms': 12.0, 'ov_overlap_ms': 879.0, 'ov_savings_pct': 34.6, 'ov_ideal_bucket_ms': 0.85, 'ov_ideal_layer_ms': 7.6, 'ov_fast_ms': 487.6}
import sys
from pathlib import Path
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from mlsysim.core.series_loader import load_series
from mlsysim import Literature
from mlsysim.physics import calc_ring_allreduce_time, calc_tree_allreduce_time
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
plt = sp.plt
C = dict(sp.C)
C.update({'c1': '#1F407A', 'c1_l': '#EAF1F8', 'c2': '#d97706', 'c2_l': '#fef3c7', 'c2_t': '#b45309', 'c3': '#7c3aed', 'c3_l': '#f3e8ff', 'ok': '#16a34a', 'ok_l': '#dcfce7', 'bad': '#dc2626', 'bad_l': '#fee2e2', 'frame': '#94a3b8', 'frame_l': '#f8fafc'})
sp.C.update(C)
out = Path(__file__).parent

def label(ax, x, y, text, color, shown=True, **kw):
    """Text drawn in every step (invisible until shown) so the bounding box,
    and so the slide position, never changes between steps."""
    kw.setdefault('fontsize', 12)
    kw.setdefault('fontweight', 'bold')
    return ax.text(x, y, text, color=color, alpha=1.0 if shown else 0.0, **kw)

def blank(size):
    fig, ax = plt.subplots(figsize=size)
    ax.set_axis_off()
    return (fig, ax)

def horovod(k, col=False):
    models = ['Inception V3', 'ResNet-101']
    keys = ['incv3', 'resnet']
    ideal = [_NUMS[f'hvd_ideal_{k}'] for k in keys]
    tf = [_NUMS[f'hvd_tf_{k}'] for k in keys]
    hv = [_NUMS[f'hvd_hv_{k}'] for k in keys]
    fig, ax = plt.subplots(figsize=(3.5, 2.6) if col else (5.8, 2.0))
    w = 0.34
    for i in range(2):
        ax.bar(i - w / 2, ideal[i], w, facecolor='none', edgecolor=C['text'], lw=1.6, ls='--')
        ax.bar(i + w / 2, ideal[i], w, facecolor='none', edgecolor=C['text'], lw=1.6, ls='--')
        ax.bar(i - w / 2, tf[i], w, color=C['bad'], alpha=1.0 if k >= 2 else 0.0)
        ax.bar(i + w / 2, hv[i], w, color=C['c2'], alpha=1.0 if k >= 3 else 0.0)
        label(ax, i - w / 2, tf[i] + 500, f"{ {0: '53', 1: '41'}[i]}%", C['bad'], k >= 2, ha='center', fontsize=13)
        label(ax, i + w / 2, hv[i] + 500, f"{ {0: '88', 1: '88'}[i]}%", C['c2_t'], k >= 3, ha='center', fontsize=13)
        label(ax, i, ideal[i] + 1500, f"ideal { {0: '17,000', 1: '15,000'}[i]}", C['text'], True, ha='center', fontsize=11, fontweight='normal')
    if col:
        label(ax, -0.55, 23500, 'TensorFlow', C['bad'], k >= 2, fontsize=12)
        label(ax, 0.55, 23500, 'Horovod', C['c2_t'], k >= 3, fontsize=12)
    else:
        label(ax, 1.62, 11000, 'distributed\nTensorFlow', C['bad'], k >= 2, fontsize=12, va='center')
        label(ax, 1.62, 5000, 'Horovod\n(ring AllReduce)', C['c2_t'], k >= 3, fontsize=12, va='center')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.6, 1.6 if col else 2.35)
    ax.set_ylim(0, 25500 if col else 19500)
    ax.set_ylabel('Images per second')
    ax.set_yticks([0, 5000, 10000, 15000])
    ax.set_yticklabels(['0', '5,000', '10,000', '15,000'])
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / ('horovod_col.svg' if col else f'horovod_step{k}.svg'))

def traffic(k):
    rows = [('Data parallel', 'AllReduce', 'bandwidth'), ('Sharded (FSDP)', 'AllGather +\nReduceScatter', 'bandwidth, many calls'), ('Tensor parallel', 'AllReduce', 'critical path, in node'), ('Pipeline', 'Send / Recv', 'latency'), ('Expert (MoE)', 'AllToAll', 'skew, narrowest cut')]
    fig, ax = blank((5.8, 2.1))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(-0.5, 5.6)
    heads = [(0.0, 'Split'), (4.4, 'Primitive'), (8.7, 'Binds on')]
    for j, (x, h) in enumerate(heads):
        label(ax, x, 5.15, h, C['neutral'], j == 0 or k > j, fontsize=12)
    for i, (sname, prim, bind) in enumerate(rows):
        y = 4.0 - i * 1.0
        label(ax, 0.0, y, sname, C['c1'], True, fontsize=12, va='center')
        a = 1.0 if k >= 2 else 0.0
        ax.annotate('', xy=(4.3, y), xytext=(3.75, y), arrowprops=dict(arrowstyle='-|>', color=C['c2'], lw=1.8, alpha=a))
        label(ax, 4.4, y, prim, C['c2_t'], k >= 2, fontsize=12, va='center', linespacing=0.95)
        label(ax, 8.7, y, bind, C['text'], k >= 3, fontsize=11, va='center', fontweight='normal')
    sp.save(fig, out / f'traffic_step{k}.svg')
RANK = ['#c7d7ee', '#fde3b0', '#cdeed7', '#e7d9fb']
RANK_T = ['#1F407A', '#b45309', '#15803d', '#6d28d9']

def cell(ax, x, y, fill, text='', tcolor=None, a=1.0, w=0.3, h=0.26):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor='#475569', lw=0.8, alpha=a))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=10, fontweight='bold', color=tcolor or C['text'], alpha=a)

def contracts(k):
    names = ['AllReduce', 'AllGather', 'ReduceScatter', 'AllToAll']
    fig, ax = blank((5.8, 2.35))
    ax.set_xlim(-0.95, 8.1)
    ax.set_ylim(-2.45, 1.75)
    for r in range(4):
        ax.text(-0.15, 1.13 - r * 0.3, f'R{r}', ha='right', va='center', fontsize=11, fontweight='bold', color=C['neutral'])
        ax.text(-0.15, -0.95 - r * 0.3, f'R{r}', ha='right', va='center', fontsize=11, fontweight='bold', color=C['neutral'])
    ax.text(-0.75, 0.68, 'before', fontsize=11, color=C['neutral'], va='center', ha='center', rotation=90)
    ax.text(-0.75, -1.4, 'after', fontsize=11, color=C['neutral'], va='center', ha='center', rotation=90)
    for p, name in enumerate(names):
        x0 = p * 2.05
        a = 1.0 if p < k else 0.0
        ax.text(x0 + 0.6, 1.62, name, ha='center', va='center', fontsize=11, fontweight='bold', color=C['c2_t'], alpha=a)
        for r in range(4):
            yb = 1.0 - r * 0.3
            ya = -1.08 - r * 0.3
            for c in range(4):
                x = x0 + c * 0.3
                if name == 'AllReduce':
                    cell(ax, x, yb, RANK[r], a=a)
                    cell(ax, x, ya, '#94a3b8', 'Σ', 'white', a=a)
                elif name == 'AllGather':
                    if c == r:
                        cell(ax, x, yb, RANK[r], a=a)
                    cell(ax, x, ya, RANK[c], a=a)
                elif name == 'ReduceScatter':
                    cell(ax, x, yb, RANK[r], a=a)
                    if c == r:
                        cell(ax, x, ya, '#94a3b8', f'Σ{r}', 'white', a=a)
                else:
                    cell(ax, x, yb, RANK[r], f'{c}', RANK_T[r], a=a)
                    cell(ax, x, ya, RANK[c], f'{r}', RANK_T[c], a=a)
        ax.annotate('', xy=(x0 + 0.6, -0.27), xytext=(x0 + 0.6, 0.0), arrowprops=dict(arrowstyle='-|>', color=C['c2'], lw=2, alpha=a))
    sp.save(fig, out / f'contracts_step{k}.svg')

def chunk(k):
    rows = [('NVLink 4.0', 'nstar_nvlink_kb'), ('PCIe Gen5', 'nstar_pcie_kb'), ('Ethernet RoCE', 'nstar_roce_kb'), ('InfiniBand HDR', 'nstar_hdr_kb'), ('InfiniBand NDR', 'nstar_ndr_kb')]
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    n = len(rows)
    for i, (nm, key) in enumerate(rows):
        y = n - 1 - i
        v = {'nstar_nvlink_kb': 675.0, 'nstar_pcie_kb': 224.0, 'nstar_roce_kb': 93.7, 'nstar_hdr_kb': 87.5, 'nstar_ndr_kb': 75.0}[key]
        ax.barh(y, v, color=C['c2'], height=0.6)
        ax.text(-0.02, y, nm, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
        xl = v * 0.93
        for line_x in (390.6, 24.4):
            if xl / 2.2 < line_x < xl:
                xl = line_x * 0.93
        ax.text(xl, y, f"{ {'nstar_nvlink_kb': '675', 'nstar_pcie_kb': '224', 'nstar_roce_kb': '93.7', 'nstar_hdr_kb': '87.5', 'nstar_ndr_kb': '75'}[key]} KB", va='center', ha='right', fontsize=11, fontweight='bold', color='white')
    a = 1.0 if k >= 2 else 0.0
    for x, txt, ck in [(390.6, '64 ranks:\n390.6 KB chunk', 'ok'), (24.4, '1,024 ranks:\n24.4 KB chunk', 'bad')]:
        ax.axvline(x, color=C[ck], lw=2, ls='--', alpha=a)
        ax.text(x * 1.06, n - 0.35, txt, color=C[ck], fontsize=11, fontweight='bold', va='bottom', ha='left', alpha=a)
    ax.set_xscale('log')
    ax.set_xlim(8, 6000)
    ax.set_ylim(-0.6, n + 0.9)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(['10 KB', '100 KB', '1 MB'])
    ax.set_xlabel('Critical size $n^* = \\alpha\\beta$ (bytes where startup = transfer)')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'chunk_step{k}.svg')

def scoreboard():
    rows = [('Parameter server', 1280.0, 'bad', '1,280 ms (derived)'), ('Simple tree', 240.0, 'neutral', '240 ms'), ('Flat ring', 39.4, 'c2', '39.4 ms'), ('Hierarchical', 8.3, 'ok', '8.3 ms')]
    sp.hbars_steps(str(out / 'scoreboard'), rows, 'ms', log=True, xlabel='Time for one 1 GB AllReduce, 64 GPUs (ms, log scale)', size=(5.8, 2.3))

def rounds():
    n_ref = int(1024.0)
    N = np.arange(2, n_ref + 1)
    fig, ax = plt.subplots(figsize=(5.8, 2.15))
    ax.plot(N, 2 * (N - 1), color=C['c2'], lw=3)
    ax.plot(N, 2 * np.log2(N), color=C['c1'], lw=3)
    ax.text(120, 1500, 'ring: 2(N−1)', color=C['c2_t'], fontsize=13, fontweight='bold')
    ax.text(560, 170, 'tree: 2 log₂ N', color=C['c1'], fontsize=13, fontweight='bold')
    ax.plot([n_ref], [2046.0], 'o', color=C['c2'], ms=9)
    ax.plot([n_ref], [20.0], 'o', color=C['c1'], ms=9)
    ax.text(1000, 2046.0, '2,046 ', ha='right', va='center', fontsize=12, fontweight='bold', color=C['c2_t'])
    ax.text(1000, 120, '20 ', ha='right', va='bottom', fontsize=12, fontweight='bold', color=C['c1'])
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 2300)
    ax.set_xlabel('Ranks N')
    ax.set_ylabel('Startup rounds')
    ax.set_xticks([0, 256, 512, 768, n_ref])
    ax.grid(color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'rounds.svg')

def crossover():
    N, a, b = (64, Q_(10, 'µs'), Q_(10, 'GB / s'))
    M = np.logspace(3, 8, 300)
    ring = np.array([calc_ring_allreduce_time(Q_(x, 'byte'), N, b, a).to('second').magnitude for x in M])
    tree = np.array([calc_tree_allreduce_time(Q_(x, 'byte'), N, b, a).to('second').magnitude for x in M])
    mx = 1136448.598
    fig, ax = plt.subplots(figsize=(4.3, 2.7))
    ax.plot(M, ring * 1000000.0, color=C['c2'], lw=3)
    ax.plot(M, tree * 1000000.0, color=C['c1'], lw=3)
    ax.axvline(mx, color=C['neutral'], lw=1.4, ls='--')
    ax.text(mx * 1.25, 130, 'crossover\n≈ 1.1 MB', color=C['neutral'], fontsize=12.5, fontweight='bold', va='bottom')
    ax.text(2000.0, 2000.0, 'ring', color=C['c2_t'], fontsize=13, fontweight='bold')
    ax.text(15000.0, 300, 'tree', color=C['c1'], fontsize=13, fontweight='bold')
    ax.text(40000000.0, 20000.0, 'tree', color=C['c1'], fontsize=13, fontweight='bold')
    ax.text(40000000.0, 2200.0, 'ring', color=C['c2_t'], fontsize=13, fontweight='bold')
    m_b = 1000000.0
    ax.plot([m_b], [1456.9], 'o', color=C['c2'], ms=8)
    ax.plot([m_b], [1320.0], 'o', color=C['c1'], ms=8)
    ax.text(1300.0, 8000.0, '1 MB: ring 1,457 μs\n          tree 1,320 μs', ha='left', fontsize=12.5, fontweight='bold', color=C['text'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1000.0, 100000000.0)
    ax.set_ylim(80, 200000.0)
    ax.set_xticks([1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0])
    ax.set_xticklabels(['1 KB', '10 KB', '100 KB', '1 MB', '10 MB', '100 MB'])
    ax.set_yticks([100, 1000, 10000, 100000])
    ax.set_yticklabels(['0.1 ms', '1 ms', '10 ms', '100 ms'])
    ax.minorticks_off()
    ax.set_xlabel('Buffer size M', fontsize=13)
    ax.tick_params(labelsize=12.5)
    ax.set_ylabel('Modeled time', fontsize=13)
    ax.grid(color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'crossover.svg')

def bw_tiers():
    fig, ax = plt.subplots(figsize=(2.9, 2.3))
    nv, ib = (450.0, 50.0)
    ax.bar([0, 1], [nv, ib], color=[C['c2'], C['c1']], width=0.62)
    ax.text(0, nv + 15, '450', ha='center', fontsize=14, fontweight='bold', color=C['c2_t'])
    ax.text(1, ib + 15, '50', ha='center', fontsize=14, fontweight='bold', color=C['c1'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['NVLink\nin node', 'InfiniBand\nport'], fontsize=12, fontweight='bold')
    ax.set_ylim(0, 540)
    ax.set_yticks([0, 200, 400])
    ax.set_ylabel('GB/s per GPU,\none direction', fontsize=11)
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'bw_tiers.svg')

def hier(k):
    fig, ax = blank((5.8, 2.3))
    ax.set_xlim(-0.2, 10.4)
    ax.set_ylim(-0.35, 4.3)
    nodes, gpus = (4, 4)
    for n in range(nodes):
        x0 = n * 1.6
        ax.add_patch(FancyBboxPatch((x0, 0), 1.25, 3.55, boxstyle='round,pad=0.02,rounding_size=0.08', facecolor=C['frame_l'], edgecolor=C['frame'], lw=1.2))
        ax.text(x0 + 0.62, 3.85, f'node {n}', ha='center', fontsize=11, fontweight='bold', color=C['neutral'])
        for g in range(gpus):
            y = 2.75 - g * 0.85
            ax.add_patch(Rectangle((x0 + 0.08, y), 0.98, 0.55, facecolor=C['c1_l'], edgecolor=C['c1'], lw=1.2))
            ax.text(x0 + 0.57, y + 0.27, f'GPU {g}', ha='center', va='center', fontsize=9.5, fontweight='bold', color=C['c1'])
        ax.plot([x0 + 1.17, x0 + 1.17], [0.3, 3.2], color=C['c2'], lw=3.5, alpha=1.0 if k in (1, 3) else 0.25 if k == 2 else 0)
    for g in range(gpus):
        y = 2.75 - g * 0.85 + 0.27
        ax.plot([1.1, 3 * 1.6 + 0.15], [y, y], color=C['c2'], lw=2.4, alpha=1.0 if k == 2 else 0.2 if k == 3 else 0.0, zorder=0)
    texts = ['1  ReduceScatter, NVLink\n   each GPU keeps 1/G: 1.94 ms', '2  AllReduce of the shard,\n   InfiniBand, one ring per\n   GPU position: 4.40 ms', '3  AllGather, NVLink: 1.94 ms']
    for i, t in enumerate(texts):
        label(ax, 6.55, 3.4 - i * 1.3, t, C['c2_t'] if k == i + 1 else C['neutral'], k >= i + 1, fontsize=11, va='top')
    sp.save(fig, out / f'hier_step{k}.svg')

def a2a_matrix():
    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    n = int(8.0)
    for s in range(n):
        for d in range(n):
            local = s == d
            ax.add_patch(Rectangle((d, n - 1 - s), 1, 1, facecolor=C['neutral_l'] if local else C['c2_l'], edgecolor='white', lw=1.5))
            ax.text(d + 0.5, n - 0.5 - s, '64', ha='center', va='center', fontsize=9, color=C['neutral'] if local else C['c2_t'], fontweight='bold')
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=11)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels([str(i) for i in range(n)][::-1], fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Receiving GPU', fontsize=12)
    ax.set_ylabel('Sending GPU', fontsize=12)
    ax.tick_params(length=0)
    for s in ['left', 'bottom', 'top', 'right']:
        ax.spines[s].set_visible(False)
    sp.save(fig, out / 'a2a_matrix.svg')

def a2a_hot(k):
    rows = [('Uniform receiver', 36.7, 'c2', '36.7 μs  (1.84 MB)'), ('Hot receiver, 1.5×', 55.1, 'bad', '55.1 μs  (2.75 MB)')]
    fig, ax = plt.subplots(figsize=(5.8, 1.15))
    for i, (nm, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.55, alpha=1.0 if i < k else 0.0)
        ax.text(-1, y, nm, ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
        label(ax, v + 1, y, txt, C['c2_t'] if i == 0 else C['bad'], i < k, fontsize=12, va='center')
    ax.set_xlim(0, 95)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Receive bound on dispatch at 50 GB/s (μs)')
    sp.save(fig, out / f'a2a_hot_step{k}.svg')

def payback():
    fig, ax = plt.subplots(figsize=(5.8, 1.35))
    comm, xfer, codec = (40.0, 5.0, 2.0)
    ax.barh(1, comm, color=C['c2'], height=0.55)
    ax.text(comm + 0.8, 1, '40 ms uncompressed', va='center', fontsize=12, fontweight='bold', color=C['c2_t'])
    ax.barh(0, xfer, color=C['c2'], height=0.55)
    ax.barh(0, codec, left=xfer, color=C['c1'], height=0.55)
    ax.text(7.0 + 0.8, 0, '5 ms transfer + 2 ms encode = 7 ms', va='center', fontsize=12, fontweight='bold', color=C['ok'])
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('ms per synchronization (8× effective, assumed)')
    sp.save(fig, out / 'payback.svg')

def runlevel(k):
    fig, ax = plt.subplots(figsize=(5.8, 1.95))
    groups = ['Slow network\n200 ms comm per step', 'Fast network\n30 ms comm per step']
    base = [50000.0, 33000.0]
    comp = [38500.0, 33825.0]
    base_t = ['50,000', '33,000']
    comp_t = ['38,500', '33,825']
    w = 0.34
    for i in range(2):
        ax.bar(i - w / 2, base[i], w, color=C['neutral'])
        ax.text(i - w / 2, base[i] + 900, f'{base_t[i]} s', ha='center', fontsize=11, fontweight='bold', color=C['neutral'])
        a = 1.0 if k >= 2 and i == 0 or (k >= 3 and i == 1) else 0.0
        ax.bar(i + w / 2, comp[i], w, color=C['ok'] if i == 0 else C['bad'], alpha=a)
        label(ax, i + w / 2, comp[i] + 900, f'{comp_t[i]} s', C['ok'] if i == 0 else C['bad'], a > 0, ha='center', fontsize=11)
    label(ax, 0.0, 60500, 'INT8 + error feedback: 23% faster', C['ok'], k >= 2, ha='center', fontsize=12)
    label(ax, 1.0, 60500, '2.5% slower', C['bad'], k >= 3, ha='center', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylim(0, 66000)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel('Time to target (s)')
    ax.set_yticks([0, 20000, 40000])
    ax.set_yticklabels(['0', '20,000', '40,000'])
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'runlevel_step{k}.svg')

def onebit():
    rows = [('Communication volume', 5.0, 'c2', 'up to 5× less'), ('BERT-Large pretraining', 3.3, 'ok', 'up to 3.3× throughput'), ('SQuAD fine-tuning', 2.9, 'ok', 'up to 2.9× throughput')]
    sp.hbars(str(out / 'onebit.svg'), rows, '', log=False, xlabel='1-bit Adam against uncompressed Adam, up to 256 GPUs', size=(5.8, 1.8))

def overlap(k):
    L, tb = (int(32.0), 15.0)
    tc, t_bwd, t_fast = (27.0, 480.0, 7.6)
    fig, ax = plt.subplots(figsize=(5.8, 2.35))
    rows = []
    seq_comm = [(t_bwd + i * tc, tc) for i in range(L)]
    ov_comm = [(tb + i * tc, tc) for i in range(L)]
    fast_comm = [(tb * (i + 1), t_fast) for i in range(L)]
    sets = [('Sequential', seq_comm, 'ov_seq_ms', 1), ('Overlapped', ov_comm, 'ov_overlap_ms', 2), ('Overlapped,\nfaster buckets', fast_comm, 'ov_fast_ms', 3)]
    for j, (nm, comm, end_key, step) in enumerate(sets):
        end = {0: 1344.0, 1: 879.0, 2: 487.6}[j]
        y = 2 - j
        a = 1.0 if k >= step else 0.0
        for i in range(L):
            ax.add_patch(Rectangle((i * tb, y + 0.05), tb, 0.33, facecolor=C['c1'], edgecolor='white', lw=0.4, alpha=a))
        for s, w in comm:
            ax.add_patch(Rectangle((s, y - 0.38), w, 0.33, facecolor=C['c2'], edgecolor='white', lw=0.4, alpha=a))
        ax.text(-20, y, nm, ha='right', va='center', fontsize=11, fontweight='bold', color=C['text'], alpha=a)
        ax.text(end + 20, y - 0.2, f"{ {0: '1,344', 1: '879', 2: '487.6'}[j]} ms", va='center', fontsize=12, fontweight='bold', color=[C['bad'], C['c2_t'], C['ok']][j], alpha=a)
    ax.text(240, 2.6, 'backward, 15 ms per layer', ha='center', fontsize=11, color=C['c1'], fontweight='bold')
    ax.text(1150, 0.15, 'reduction, 27 ms\nper layer', ha='center', va='center', fontsize=11, color=C['c2_t'], fontweight='bold')
    ax.set_xlim(0, 1550)
    ax.set_ylim(-0.6, 2.9)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('ms, one backward pass plus its reductions (7B model, 64 GPUs)')
    sp.save(fig, out / f'overlap_step{k}.svg')

def ring():
    fig, ax = blank((2.6, 2.6))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    pos = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for i, (x, y) in enumerate(pos):
        ax.add_patch(FancyBboxPatch((x - 0.45, y - 0.22), 0.9, 0.44, boxstyle='round,pad=0.02,rounding_size=0.06', facecolor=C['c1_l'], edgecolor=C['c1'], lw=1.5))
        ax.text(x, y, f'GPU {i}', ha='center', va='center', fontsize=11, fontweight='bold', color=C['c1'])
    for i in range(4):
        (x0, y0), (x1, y1) = (pos[i], pos[(i + 1) % 4])
        ax.add_patch(FancyArrowPatch((x0 * 0.7 + x1 * 0.3 + 0.12 * (x1 - x0), y0 * 0.7 + y1 * 0.3), (x0 * 0.3 + x1 * 0.7, y0 * 0.3 + y1 * 0.7 - 0.12 * (y1 - y0)), arrowstyle='-|>', mutation_scale=16, color=C['c2'], lw=2.2, connectionstyle='arc3,rad=-0.25'))
    sp.save(fig, out / 'ring.svg')

def horovod_trace():
    trace = load_series('slides/vol2/05_collective_communication/images/data/horovod_timeline_trace.csv').entries
    fast = [(r['phase'], r['start_ms'], r['end_ms']) for r in trace if r['tensors'] == '0-1']
    slow = [(r['phase'], r['start_ms'], r['end_ms']) for r in trace if r['tensors'] == '2-4']
    color = {'negotiate': C['c3'], 'wait': C['neutral_l'], 'schedule': C['c3_l'], 'NCCL': C['c2'], 'copy': C['c2']}
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for row in range(5):
        y = 4 - row
        for name, a, b in fast if row < 2 else slow:
            ax.barh(y, b - a, left=a, height=0.62, color=color[name], edgecolor='white', lw=0.6)
        ax.text(-6, y, f'tensor {row}', ha='right', va='center', fontsize=12)
    span = (163.5, 347.6)
    ax.plot(span, [-1.25, -1.25], color=C['text'], lw=1.4)
    for x in span:
        ax.plot([x, x], [-1.35, -1.15], color=C['text'], lw=1.4)
    ax.text(255.5, -1.45, 'AllReduce 184 ms', ha='center', va='top', fontsize=11.5)
    ax.text(88, -0.45, 'negotiate 131 ms', ha='center', va='top', fontsize=12, fontweight='bold', color=C['c3'])
    ax.text(219, -0.45, 'wait', ha='center', va='top', fontsize=12, color=C['neutral'])
    ax.text(352, -0.45, 'NCCL 19 ms', ha='right', va='top', fontsize=12, fontweight='bold', color=C['c2_t'])
    ax.text(144, 4, 'NCCL', ha='left', va='center', fontsize=12, fontweight='bold', color=C['c2_t'])
    ax.set_xlim(0, 360)
    ax.set_ylim(-2.2, 4.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel('Time in the trace (ms after 18,500 ms)', fontsize=12)
    sp.save(fig, out / 'horovod_trace.svg')
if __name__ == '__main__':
    horovod_trace()
    for k in (1, 2, 3):
        horovod(k)
        traffic(k)
        hier(k)
        runlevel(k)
        overlap(k)
    for k in (1, 2, 3, 4):
        contracts(k)
    for k in (1, 2):
        chunk(k)
        a2a_hot(k)
    scoreboard()
    horovod(3, col=True)
    rounds()
    crossover()
    bw_tiers()
    a2a_matrix()
    payback()
    onebit()
    ring()
    print('figures written to', out)
