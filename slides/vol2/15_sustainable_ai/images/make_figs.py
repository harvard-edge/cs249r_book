#!/usr/bin/env python3
"""Figures for Volume II, Lecture 15 (Sustainable AI).

Every value comes from MLSysIM through calcs.py: figures import calcs (which
from mlsysim.core.series_loader import load_series  # noqa: E402
records every slide value) and read the formatted strings with val() and the
numbers with num(), or the same MLSysIM objects calcs selects (the running
case's footprint, the Literature series behind the outside tables). Nothing
here is typed except layout: positions, sizes, colors, and axis ranges.

Run from this folder: python3 make_figs.py
"""
from mlsysim.core.series_loader import load_series
import mlsysim
import datetime as dt
_NUMS = {'gpus': 16384.0, 'tdp_w': 700.0, 'card_gpuh_m': 30.84, 'card_lb_t': 8930.0, 'card_mb_t': 0.0, 'p_acc_mw': 11.47, 'p_it_mw': 15.77, 'p_fac_mw': 17.66, 'p_fac_mw1': 17.7, 'it_ratio': 1.375, 'it_ratio2': 1.38, 'pue_run': 1.12, 'run_hours': 1882.0, 'run_days': 78.0, 'e_acc_mwh': 21588.0, 'e_it_mwh': 29684.0, 'e_fac_mwh': 33246.0, 'e_acc_gwh': 21.6, 'e_it_gwh': 29.7, 'e_fac_gwh': 33.2, 'g_norway': 10.0, 'co2_norway': 332.0, 'g_quebec': 20.0, 'co2_quebec': 665.0, 'g_france': 50.0, 'co2_france': 1662.0, 'g_eu': 270.0, 'co2_eu': 8976.0, 'g_germany': 385.0, 'co2_germany': 12800.0, 'g_us': 429.0, 'co2_us': 14262.0, 'g_china': 555.0, 'co2_china': 18451.0, 'g_india': 720.0, 'co2_india': 23937.0, 'g_poland': 820.0, 'co2_poland': 27261.0, 'region_spread': 82.0, 'card_implied_g': 414.0, 'fac_over_card': 1.6, 'emb_kg': 164.0, 'hgx_kg': 1312.0, 'emb_made_t': 2687.0, 'emb_alloc_t': 192.0, 'life_h': 26280.0, 'life_y': 3.0, 'emb_share_us': 1.3, 'emb_share_qc': 22.0, 'water_lo_ml': 34.0, 'water_hi_ml': 53.0, 'wue_google': 1.15, 'wue_evap': 1.8, 'lh_gpus': 25000.0, 'lh_mw': 17.5, 'rack_gpus': 32.0, 'rack_gpu_kw': 22.4, 'rack_host_kw': 3.2, 'rack_nvs_kw': 1.6, 'rack_ib_kw': 0.8, 'rack_conv_kw': 2.8, 'rack_support_kw': 8.4, 'rack_it_kw': 30.8, 'rack_cool_kw': 2.7, 'rack_total_kw': 33.5, 'rack_pue': 1.09, 'rack_total_ratio': 1.5, 'rack_total_ratio1': 1.5, 'rack_gpu_share': 67.0, 'feed_gpus': 14000.0, 'feed_acc_mw': 9.8, 'feed_it_mw': 13.5, 'substation_months': 24.0, 'gpu_lead_months': 6.0, 'cluster10k_gpus': 10000.0, 'cap_req_mw': 120.0, 'cap_feed_mw': 80.0, 'cap_pct': 67.0, 'cap_days': 4.5, 'cap_plan_days': 3.0, 'pue_it_mw': 2.0, 'pue_legacy': 1.58, 'pue_survey': 1.55, 'pue_soa': 1.1, 'pue_google': 1.09, 'pue_fac_legacy': 3.16, 'pue_fac_survey': 3.1, 'pue_fac_soa': 2.2, 'pue_fac_google': 2.18, 'pue_saved_mw': 0.96, 'hours_year': 8760.0, 'pue_saved_mwh': 8409.6, 'pue_saved_mwh0': 8410.0, 'pue_saved_usd': 588672.0, 'price_mwh': 70.0, 'rb_min_lo': 30.0, 'rb_min_hi': 60.0, 'rb_mwh_lo': 3.5, 'rb_mwh_hi': 7.0, 'prompt_narrow_wh': 0.1, 'prompt_full_wh': 0.24, 'prompt_accel_wh': 0.14, 'prompt_accel_x_wh': 0.1, 'prompt_host_wh': 0.06, 'prompt_host_x_wh': 0.04, 'prompt_idle_wh': 0.02, 'prompt_idle_x_wh': 0.02, 'prompt_ovh_wh': 0.02, 'prompt_ovh_x_wh': 0.01, 'prompt_ratio': 2.4, 'prompt_accel_pct': 58.0, 'lb_g': 345.0, 'mb_g': 94.0, 'lb_mb_ratio': 3.7, 'prompt_water_ml': 0.26, 'prompt_cut': 33.0, 'dram_pj': 160.0, 'flop_pj': 10.0, 'operand_bytes': 4.0, 'operand_pj': 640.0, 'operand_ratio': 64.0, 'crossover_fpb': 16.0, 'fp16_bytes': 2.0, 'decode_fpb': 1.0, 'move_pj': 320.0, 'compute_pj': 20.0, 'move_pct': 94.0, 'ib_pj_bit': 35.0, 'sync_j': 196.0, 'tpu3_tdp': 450.0, 'tpu3_meas_lo': 245.0, 'tpu3_meas_hi': 310.0, 'tpu3_lo_pct': 54.0, 'tpu3_hi_pct': 69.0, 'site_it_mw': 10.0, 'site_a_pue': 1.08, 'site_b_pue': 1.35, 'site_a_g': 450.0, 'site_b_g': 50.0, 'site_a_t': 4.86, 'site_b_t': 0.675, 'site_ratio': 7.2, 'gshard_t': 4.3, 'gshard_act_pct': 0.25, 't5_t': 46.7, 't5_params_b': 11.0, 'gpt3_t': 552.1, 'v6e_emb_kg': 692.0, 'v6e_lb_kg': 5759.0, 'v6e_mb_kg': 2141.0, 'v6e_years': 6.0, 'v6e_lb_pct': 11.0, 'v6e_mb_pct': 24.0, 'bloom_g': 57.0, 'bloom_emb_t': 11.2, 'bloom_total_t': 50.5, 'bloom_emb_pct': 22.2, 'saved_kw': 0.35, 'hours_day': 24.0, 'saved_kwh_day': 8.4, 'avoid_us_kg': 3.6, 'avoid_qc_kg': 0.168, 'pay_us_weeks': 6.5, 'pay_qc_years': 2.7, 'life_long_y': 5.0, 'life_cut_pct': 40.0, 'bloom_final_t': 24.69, 'bloom_project_t': 66.29, 'bloom_final_pct': 37.0, 'bloom_project_mult': 2.7, 'bloom_models_t': 35.8, 'sweep_n': 50.0, 'sweep_pct': 10.0, 'sweep_runs': 5.0, 'prompts_b': 1.0, 'serve_mwh_day': 240.0, 'serve_cross_day': 139.0, 'lab_cross_days': 1.3, 'radio_lo_pj': 100000.0, 'radio_hi_pj': 500000.0, 'fp32_pj': 3.7, 'int32_add_pj': 0.1, 'radio_x_lo': 27027.0, 'radio_x_hi': 135135.0, 'radio_x_floor': 27000.0, 'wake_vad_uw': 10.0, 'wake_stage_uw': 100.0, 'wake_full_mw': 10.0, 'wake_full_ms': 50.0, 'wake_avg_uw': 15.5, 'wake_x': 645.0, 'cas_edge_mj': 0.5, 'cas_esc': 0.1, 'cas_esc_pct': 10.0, 'cas_tx_mj': 10.0, 'cas_cloud_mj': 50.0, 'cas_mean_mj': 6.5, 'cas_only_mj': 60.0, 'cas_save_pct': 89.0, 'geo_mwh': 10000.0, 'geo_cut_pct': 15.0, 'geo_pl_t': 8200.0, 'geo_pl_cut_t': 6970.0, 'geo_qc_t': 200.0, 'geo_ratio': 41.0, 'carbon_price': 100.0, 'priced_tco_musd': 25.3, 'priced_gap_musd': 0.5, 'eu_min_pct': 20.0, 'eu_max_pct': 38.0, 'eu_hi_pct': 36.0, 'eu_lo_pct': 23.0, 'eu_ratio': 1.57, 'wind_min_gw': 27.0, 'wind_max_gw': 111.0, 'water_10mw': 10.0, 'water_10mw_ml': 158.0, 'latent_kj': 2260.0, 'jev_unit_pct': 50.0, 'jev_unit': 0.5, 'jev_vol': 4.0, 'jev_vol_pct': 300.0, 'jev_net': 2.0, 'jev3_unit_x': 30.0, 'jev3_vol': 100.0, 'jev3_net': 3.3, 'kaplan_alpha': 0.05, 'kaplan_pct': 10.0, 'kaplan_x': 8.2, 'kaplan_x2': 68.0, 'us_2014_twh': 58.0, 'us_2024_twh': 192.0, 'us_2030_twh': 649.0, 'us_2030_lo': 521.0, 'us_2030_hi': 843.0, 'us_growth': 3.3, 'cso_2015_pct': 5.0, 'cso_2023_pct': 21.0, 'cso_2015_gwh': 1238.0, 'cso_2023_gwh': 6334.0, 'cso_growth': 5.1, 'pgd_passes': 8.0, 'pgd_attack': 7.0, 'pgd_update': 1.0}
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from matplotlib.patches import FancyBboxPatch, Rectangle
P_ACC = 11.4688
P_IT = 15.7696
P_FAC = 17.661952
E_FAC = 33245.52
HOURS = 1882.324219
EMB_T = 192.456621
EMBODIED_KG = 164.0
CARBON = {'Norway': 332.4552, 'Quebec': 664.9104, 'France': 1662.276, 'EU average': 8976.2904, 'Germany': 12799.5252, 'US average': 14262.32808, 'China': 18451.2636, 'India': 23936.7744, 'Poland': 27261.3264}
GRID = {'Norway': 10.0, 'Quebec': 20.0, 'France': 50.0, 'EU average': 270.0, 'Germany': 385.0, 'US average': 429.0, 'China': 555.0, 'India': 720.0, 'Poland': 820.0}
C = sp.C
plt = sp.plt
out = Path(__file__).parent

def hide_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def note(ax, text, x=0.0, y=-0.02, fs=10, ha='left', va='top'):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, color=C['neutral'], ha=ha, va=va)
CSO = [(2015, 5.0), (2016, 6.0), (2017, 7.0), (2018, 8.0), (2019, 9.0), (2020, 11.0), (2021, 14.0), (2022, 18.0), (2023, 21.0)]

def hook_fig(path, size=(5.4, 3.6), fs=15, annotate=False):
    fig, ax = plt.subplots(figsize=size)
    years = [y for y, _ in CSO]
    vals = [v for _, v in CSO]
    cols = [C['memory'] if y < 2022 else C['danger'] for y in years]
    ax.bar(years, vals, color=cols, width=0.72)
    for y, v in CSO:
        if y in (2015, 2019, 2022, 2023):
            ax.text(y, v + 0.5, f'{v:.0f}%', ha='center', va='bottom', fontsize=fs, fontweight='bold', color=C['danger'] if y >= 2022 else C['memory'])
    ax.text(2014.6, 22.5, '2022: grid statement', fontsize=fs - 1, fontweight='bold', color=C['danger'], va='center', alpha=1.0 if annotate else 0.0)
    ax.set_xticks([2015, 2017, 2019, 2021, 2023])
    ax.tick_params(labelsize=fs - 2)
    ax.set_ylim(0, 25)
    ax.set_yticks([0, 10, 20])
    ax.set_yticklabels(['0%', '10%', '20%'])
    ax.set_ylabel("Data centers' share of\nmetered electricity", fontsize=fs - 2)
    sp.save(fig, path)
hook_fig(out / 'hook_step1.svg', size=(4.4, 2.2))
hook_fig(out / 'hook_step2.svg', size=(4.4, 2.2), annotate=True)
hook_fig(out / 'hook_thumb.svg', size=(4.0, 2.1), fs=17)

def power_energy(k, size=(5.6, 3.4), fs=16):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.06, 0.16, 0.3, 0.66])
    bx = fig.add_axes([0.5, 0.16, 0.47, 0.66])
    segs = [('GPUs at\n700 W', P_ACC, C['compute']), ('host,\nfabric', P_IT - P_ACC, C['memory']), ('PUE', P_FAC - P_IT, C['neutral'])]
    bottom = 0
    for lab, v, col in segs:
        ax.bar(0, v, bottom=bottom, color=col, width=0.6)
        ax.text(0.36, bottom + v / 2, lab, va='center', ha='left', fontsize=fs - 3, color=col, fontweight='bold')
        bottom += v
    ax.text(0, P_FAC + 0.5, '17.7 MW', ha='center', va='bottom', fontsize=fs, fontweight='bold')
    ax.set_xlim(-0.45, 1.25)
    ax.set_ylim(0, 22)
    ax.set_ylabel('MW', fontsize=fs - 2)
    ax.set_xticks([])
    ax.tick_params(labelsize=fs - 3)
    ax.set_title('Power: every instant', fontsize=fs, loc='left', fontweight='bold', color=C['danger'], x=-0.2)
    a = 1.0 if k >= 2 else 0.0
    t = np.linspace(0, HOURS / 24, 400)
    p = P_FAC * np.ones_like(t)
    bx.fill_between(t, 0, p, color=C['compute_l'], alpha=a, lw=0)
    bx.plot(t, p, color=C['compute'], lw=2.5, alpha=a)
    bx.text(HOURS / 48, P_FAC / 2, '33.2 GWh', ha='center', va='center', fontsize=fs + 2, fontweight='bold', color=C['compute'], alpha=a)
    bx.text(HOURS / 48, P_FAC / 2 - 4.2, '17.7 MW × 1,882 h', ha='center', va='center', fontsize=fs - 3, color=C['text'], alpha=a)
    bx.set_xlim(0, 90)
    bx.set_ylim(0, 22)
    bx.set_xlabel('Days', fontsize=fs - 2, alpha=max(a, 0.0))
    bx.tick_params(labelsize=fs - 3, colors=C['text'] if a else 'white')
    for s in ('bottom', 'left'):
        bx.spines[s].set_alpha(a)
    bx.set_title('Energy: an account over time', fontsize=fs, loc='left', fontweight='bold', color=C['compute'], alpha=a)
    sp.save(fig, out / f'power_energy_step{k}.svg')
for k in (1, 2):
    power_energy(k)
RACK_ROWS = [('GPUs (32 × 700 W)', 22.4, 'compute'), ('Host CPUs, DRAM', 3.2, 'memory'), ('NVSwitch', 1.6, 'network'), ('InfiniBand', 0.8, 'network'), ('Conversion in IT', 2.8, 'neutral')]

def rack_fig(k, size=(5.8, 2.3), fs=14):
    fig, ax = plt.subplots(figsize=size)
    left = 0
    for i, (lab, v, ck) in enumerate(RACK_ROWS):
        show = i == 0 or k >= 2
        ax.barh(0, v, left=left, color=C[ck], height=0.7, alpha=1.0 if show else 0.0, edgecolor='white', lw=1)
        left += v
    ax.text(11.2, 0, 'GPUs 22.4 kW', ha='center', va='center', fontsize=fs, color='white', fontweight='bold')
    a2 = 1.0 if k >= 2 else 0.0
    ax.text(26.6, 0.5, 'host, switches, conversion: 8.4 kW', ha='center', va='bottom', fontsize=fs - 1, color=C['memory'], fontweight='bold', alpha=a2)
    a3 = 1.0 if k >= 3 else 0.0
    ax.barh(0, 2.7, left=30.8, color=C['neutral_l'], edgecolor=C['neutral'], hatch='//', height=0.7, lw=1, alpha=a3)
    ax.text(34.0, 0, '+ PUE 2.7 kW\n= 33.5 kW (1.50 ×)', ha='left', va='center', fontsize=fs - 1, fontweight='bold', color=C['text'], alpha=a3)
    ax.annotate('', xy=(0, -0.55), xytext=(30.8, -0.55), arrowprops=dict(arrowstyle='|-|', color=C['memory'], lw=1.5, alpha=a2))
    ax.text(15.4, -0.85, 'IT load 30.8 kW = 1.38 × the GPUs', ha='center', va='top', fontsize=fs, color=C['memory'], fontweight='bold', alpha=a2)
    ax.set_xlim(0, 47)
    ax.set_ylim(-1.55, 1.05)
    hide_axis(ax)
    sp.save(fig, out / f'rack_step{k}.svg')
for k in (1, 2, 3):
    rack_fig(k)

def chain_fig(k, size=(5.9, 2.4), fs=11.5):
    fig, ax = plt.subplots(figsize=size)
    w, gap = (2.3, 0.45)
    boxes = [('GPU', '21.6 GWh', '30.84M GPU-h\n× 700 W', 'compute'), ('IT', '29.7 GWh', '× 1.375 host,\nfabric', 'memory'), ('Facility', '33.2 GWh', '× PUE 1.12', 'neutral'), ('CO$_2$', '14,262 t', '× 429 g/kWh\n(US average)', 'danger')]
    for i, (title, value, how, ck) in enumerate(boxes):
        x = i * (w + gap)
        a = 1.0 if k > i else 0.0
        ax.add_patch(FancyBboxPatch((x, 0.75), w, 1.05, boxstyle='round,pad=0.02,rounding_size=0.08', facecolor=C[ck + '_l'], edgecolor=C[ck], lw=1.8, alpha=a))
        ax.text(x + w / 2, 1.55, title, ha='center', va='center', fontsize=fs, fontweight='bold', color=C[ck], alpha=a)
        ax.text(x + w / 2, 1.08, value, ha='center', va='center', fontsize=fs + 1.5, fontweight='bold', color=C['text'], alpha=a)
        ax.text(x + w / 2, 0.45, how, ha='center', va='center', fontsize=fs - 1, color=C['text'], alpha=a)
        if i > 0:
            ax.annotate('', xy=(x - 0.04, 1.27), xytext=(x - gap + 0.04, 1.27), arrowprops=dict(arrowstyle='-|>', color=C['text'], lw=1.8, alpha=a))
    ax.set_xlim(-0.05, 4 * w + 3 * gap + 0.05)
    ax.set_ylim(0.05, 1.9)
    hide_axis(ax)
    sp.save(fig, out / f'chain_step{k}.svg')
for k in (1, 2, 3, 4):
    chain_fig(k)

def pue_fig(size=(5.8, 2.5), fs=12):
    fig, ax = plt.subplots(figsize=size)
    rows = [('Legacy air', 'legacy'), ('Survey average, 2022', 'survey'), ('State of the art', 'soa'), ('Google fleet, 2024', 'google')]
    it = 2.0
    for i, (lab, key) in enumerate(rows):
        y = 3 - i
        fac = {'Legacy air': 3.16, 'Survey average, 2022': 3.1, 'State of the art': 2.2, 'Google fleet, 2024': 2.18}[lab]
        ax.barh(y, it, color=C['memory'], height=0.6)
        ax.barh(y, fac - it, left=it, color=C['neutral_l'], edgecolor=C['neutral'], hatch='//', height=0.6, lw=1)
        ax.text(-0.05, y, f"{lab}\nPUE { {'Legacy air': '1.58', 'Survey average, 2022': '1.55', 'State of the art': '1.10', 'Google fleet, 2024': '1.09'}[lab]}", ha='right', va='center', fontsize=fs, fontweight='bold')
        ax.text(fac + 0.04, y, f"{ {'Legacy air': '3.16', 'Survey average, 2022': '3.10', 'State of the art': '2.20', 'Google fleet, 2024': '2.18'}[lab]} MW", va='center', fontsize=fs, fontweight='bold')
    ax.text(1.0, 3, 'IT 2 MW', ha='center', va='center', color='white', fontsize=fs, fontweight='bold')
    ax.text(2.6, -0.75, '1.58 to 1.10 saves 0.96 MW, 8,410 MWh a year', ha='center', va='center', fontsize=fs, color=C['data'], fontweight='bold')
    ax.set_xlim(0, 3.6)
    ax.set_ylim(-1.1, 3.5)
    hide_axis(ax)
    sp.save(fig, out / 'pue.svg')
pue_fig()

def prompt_fig(size=(5.8, 2.3), fs=12):
    fig, ax = plt.subplots(figsize=size)
    parts = [('accelerators', 'compute'), ('host CPU,\nDRAM', 'memory'), ('idle\nmachines', 'network'), ('overhead', 'neutral')]
    keys = ('accel', 'host', 'idle', 'ovh')
    narrow = [_NUMS[f'prompt_{k}_x_wh'] for k in keys]
    full = [_NUMS[f'prompt_{k}_wh'] for k in keys]
    rows = [('Accelerators only', narrow, [True, False, False, False], 'prompt_narrow_wh'), ('Full stack', full, [True, True, True, True], 'prompt_full_wh')]
    for i, (lab, vals, counted, total) in enumerate(rows):
        y = 1 - i
        left = 0
        for (pname, ck), v, c in zip(parts, vals, counted):
            if c:
                ax.barh(y, v, left=left, color=C[ck], height=0.55, edgecolor='white', lw=1)
            else:
                ax.barh(y, v, left=left, color='white', edgecolor=C[ck], hatch='//', height=0.55, lw=1.2)
            left += v
        ax.text(-0.004, y, lab, ha='right', va='center', fontsize=fs, fontweight='bold')
        ax.text(left + 0.004, y, f"{ {'prompt_narrow_wh': '0.10', 'prompt_full_wh': '0.24'}[total]} Wh" + ('' if i else ' counted'), va='center', fontsize=fs + 1, fontweight='bold', color=C['danger'] if i else C['text'])
    x = 0
    label_x = [0.07, 0.17, 0.255, 0.33]
    for (pname, ck), v, lx in zip(parts, full, label_x):
        ax.annotate(pname, xy=(x + v / 2, -0.29), xytext=(lx, -0.72), ha='center', va='center', fontsize=fs - 1, color=C[ck], fontweight='bold', arrowprops=dict(arrowstyle='-', color=C[ck], lw=1))
        x += v
    ax.text(0.0, 1.45, 'Hatched parts are excluded from the 0.10 Wh', ha='left', fontsize=fs - 1, color=C['neutral'])
    ax.set_xlim(0, 0.37)
    ax.set_ylim(-1.05, 1.65)
    hide_axis(ax)
    sp.save(fig, out / 'prompt_boundary.svg')
prompt_fig()
SHORT = {'Switch Transformer': 'Switch'}
PAT = [(SHORT.get(r['model'], r['model']), r['params_b'], r['net_tco2e']) for r in load_series('vol2/15_sustainable_ai/data/patterson_training_footprints.csv').entries]

def patterson_fig(size=(4.4, 2.6), fs=16):
    fig, ax = plt.subplots(figsize=size)
    for name, params, t in PAT:
        ck = 'danger' if name == 'GPT-3' else 'data' if name == 'GShard-600B' else 'memory'
        ax.plot(params, t, 'o', ms=12, color=C[ck], mec='white', mew=1.5)
        off = {'T5': (-8, -30), 'Meena': (8, 4), 'GShard-600B': (-8, 8), 'Switch': (-8, 8), 'GPT-3': (10, -4)}[name]
        ha = 'right' if off[0] < 0 else 'left'
        ax.annotate(f'{name}\n{t:g} t', (params, t), xytext=off, textcoords='offset points', ha=ha, fontsize=fs - 2, fontweight='bold', color=C[ck])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 8000)
    ax.set_ylim(2, 2000)
    ax.set_xlabel('Parameters (billions)', fontsize=fs - 1)
    ax.set_ylabel('Net t CO$_2$e, final run', fontsize=fs - 1)
    ax.tick_params(labelsize=fs - 3)
    ax.minorticks_off()
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1', '10', '100', '1,000'])
    ax.set_yticks([10, 100, 1000])
    ax.set_yticklabels(['10', '100', '1,000'])
    ax.grid(color=C['grid'], lw=0.6)
    sp.save(fig, out / 'patterson.svg')
patterson_fig()
sp.hbars(out / 'energy_ladder.svg', [('DRAM operand (FP32)', 640.0, 'memory', '640 pJ'), ('GPU FLOP', 10.0, 'compute', '10 pJ')], 'pJ', log=False, size=(5.0, 1.25), xlabel=None)

def embodied_fig(k, size=(5.8, 2.9), fs=13):
    fig, ax = plt.subplots(figsize=size)
    us = CARBON['US average']
    qc = CARBON['Quebec']
    rows = [('US average grid', us, '1.3%'), ('Quebec grid', qc, '22%')]
    ax.text(0, 4.62, 'Llama 3.1 405B run (derived)', fontsize=fs, fontweight='bold', color=C['text'], va='center')
    for i, (lab, op, share) in enumerate(rows):
        y = 4 - i
        ax.barh(y, 100 * op / (op + EMB_T), color=C['compute'], height=0.6)
        ax.barh(y, 100 * EMB_T / (op + EMB_T), left=100 * op / (op + EMB_T), color='#8B5A2B', height=0.6)
        ax.text(-1, y, lab, ha='right', va='center', fontsize=fs, fontweight='bold')
        ax.text(101, y, f'embodied {share}', va='center', fontsize=fs, fontweight='bold', color='#8B5A2B')
    a = 1.0 if k >= 2 else 0.0
    ax.text(0, 2.12, 'Measured lifecycle studies', fontsize=fs, fontweight='bold', color=C['text'], va='center', alpha=a)
    meas = [('TPU v6e, 6 years', 10.72701907, 24.42640311), ('BLOOM final run', 22.2, None)]
    for i, (lab, lb, mb) in enumerate(meas):
        y = 1.5 - i
        ax.barh(y, 100 - lb, color=C['compute'], height=0.6, alpha=a)
        ax.barh(y, lb, left=100 - lb, color='#8B5A2B', height=0.6, alpha=a)
        ax.text(-1, y, lab, ha='right', va='center', fontsize=fs, fontweight='bold', alpha=a)
        txt = 'embodied 11% (location)\nor 24% (market)' if mb else f'embodied {lb:.0f}%\n(57 g/kWh grid)'
        ax.text(101, y, txt, va='center', fontsize=fs - 1, fontweight='bold', color='#8B5A2B', alpha=a)
    ax.text(45, 4, 'operational', ha='center', va='center', fontsize=fs, color='white', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 4.9)
    hide_axis(ax)
    sp.save(fig, out / f'embodied_step{k}.svg')
for k in (1, 2):
    embodied_fig(k)

def breakeven_fig(k, size=(4.4, 2.6), fs=16):
    fig, ax = plt.subplots(figsize=size)
    saved_kwh_day = 8.4
    life_days = 1095.0
    days = np.logspace(0, np.log10(1500), 300)
    ax.axhline(EMBODIED_KG, color='#8B5A2B', lw=2.2, ls='--')
    ax.text(1.15, EMBODIED_KG - 12, '164 kg to make', va='top', fontsize=fs - 2, color='#8B5A2B', fontweight='bold')
    ax.axvline(life_days, color=C['neutral'], lw=1.2, ls=':')
    ax.text(life_days * 0.93, 8, '3-year\nlife', ha='right', va='bottom', fontsize=fs - 3, color=C['neutral'])
    if k >= 2:
        us = saved_kwh_day * GRID['US average'] / 1000 * days
        ax.plot(days, us, color=C['danger'], lw=3)
        d = EMBODIED_KG / (saved_kwh_day * GRID['US average'] / 1000)
        ax.plot(d, EMBODIED_KG, 'o', ms=10, color=C['danger'])
        ax.text(1.15, 215, 'US average:\n6.5 weeks', ha='left', fontsize=fs - 1, fontweight='bold', color=C['danger'])
    if k >= 3:
        qc = saved_kwh_day * GRID['Quebec'] / 1000 * days
        ax.plot(days, qc, color=C['data'], lw=3)
        d = EMBODIED_KG / (saved_kwh_day * GRID['Quebec'] / 1000)
        ax.plot(d, EMBODIED_KG, 'o', ms=10, color=C['data'])
        ax.text(d * 0.6, 215, 'Quebec:\n2.7 years', ha='right', fontsize=fs - 1, fontweight='bold', color=C['data'])
    ax.set_xscale('log')
    ax.set_xlim(1, 1500)
    ax.set_ylim(0, 300)
    ax.set_xticks([1, 7, 30, 365])
    ax.set_xticklabels(['1 day', '1 week', '1 month', '1 year'])
    ax.minorticks_off()
    ax.set_ylabel('kg CO$_2$ avoided', fontsize=fs - 1)
    ax.tick_params(labelsize=fs - 3)
    sp.save(fig, out / f'breakeven_step{k}.svg')
for k in (1, 2, 3):
    breakeven_fig(k)
BLOOM_NAMES = {'BLOOM-176B (final run)': ('176B (final run)', 'danger'), 'Miscellaneous processes': ('Misc. processes', 'neutral'), 'Evaluation': ('Evaluation', 'neutral')}
BLOOM = [('176B (final run)', 'danger', 24.69), ('104B', 'memory', 15.19), ('1B', 'memory', 9.06), ('13B', 'memory', 4.97), ('Other models', 'memory', 3.66), ('Misc. processes', 'neutral', 3.3), ('6B', 'memory', 2.95), ('Evaluation', 'neutral', 2.46)]
BLOOM = [(n, v, ck) for n, ck, v in BLOOM]
sp.hbars(out / 'bloom_dev.svg', [(n, v, ck, f'{v:g} t') for n, v, ck in BLOOM], 't', log=False, size=(5.8, 2.7), xlabel='t CO$_2$eq, dynamic GPU energy, French grid 57 g/kWh')

def crossover_fig(size=(5.4, 3.3), fs=14):
    fig, ax = plt.subplots(figsize=size)
    days = np.linspace(0, 200, 300)
    daily = 240.0
    serve = daily * days
    ax.axhline(E_FAC / 1000, color=C['neutral'], lw=2.5)
    ax.text(3, E_FAC / 1000 + 1.2, 'training once: 33.2 GWh', fontsize=fs - 1, color=C['neutral'], fontweight='bold')
    ax.plot(days, serve / 1000, color=C['compute'], lw=3)
    d = 138.523
    ax.plot(d, E_FAC / 1000, 'o', ms=11, color=C['compute'])
    ax.text(d + 6, E_FAC / 1000 - 6, 'day 139', fontsize=fs, fontweight='bold', color=C['compute'])
    ax.text(110, 10, 'serving: 1 billion\nprompts a day\nat 0.24 Wh', fontsize=fs - 0.5, color=C['compute'], fontweight='bold')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 50)
    ax.set_xlabel('Days of service', fontsize=fs - 1)
    ax.set_ylabel('Facility GWh', fontsize=fs - 1)
    ax.tick_params(labelsize=fs - 3)
    sp.save(fig, out / 'crossover.svg')
crossover_fig()

def edge_fig(size=(5.6, 2.0), fs=14):
    fig, ax = plt.subplots(figsize=size)
    rows = [('Radio, 1 bit', 100000.0, 500000.0, 'network', '100,000 to 500,000 pJ'), ('FP32 multiply', 3.7, None, 'compute', '3.7 pJ'), ('INT32 add', 0.1, None, 'data', '0.1 pJ')]
    for i, (lab, lo, hi, ck, txt) in enumerate(rows):
        y = 2 - i
        ax.barh(y, lo, color=C[ck], height=0.6)
        if hi:
            ax.barh(y, hi - lo, left=lo, color=C[ck + '_l'], height=0.6)
        ax.text((hi or lo) * 1.5, y, txt, va='center', fontsize=fs, fontweight='bold', color=C[ck])
        ax.text(0.045, y, lab, ha='right', va='center', fontsize=fs, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(0.05, 30000000000.0)
    ax.set_ylim(-0.6, 2.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([1, 1000.0, 1000000.0])
    ax.set_xticklabels(['1 pJ', '1,000 pJ', '10$^6$ pJ'])
    ax.minorticks_off()
    ax.tick_params(labelsize=fs - 2)
    sp.save(fig, out / 'edge_ladder.svg')
edge_fig()

def regions_fig(size=(5.8, 3.0), fs=13):
    fig, ax = plt.subplots(figsize=size)
    items = sorted(GRID.items(), key=lambda kv: kv[1])
    n = len(items)
    for i, (name, g) in enumerate(items):
        t = CARBON[name]
        y = n - 1 - i
        ck = {'Norway': 'data', 'Quebec': 'data', 'Poland': 'danger', 'US average': 'compute'}.get(name, 'neutral')
        ax.barh(y, t, color=C[ck] if ck != 'neutral' else C['neutral_l'], edgecolor=C[ck], height=0.7)
        ax.text(-300, y, f'{name} ({g:g} g/kWh)', ha='right', va='center', fontsize=fs, fontweight='bold')
        ax.text(t + 300, y, f'{t:,.0f} t', va='center', fontsize=fs, fontweight='bold', color=C[ck])
    ax.set_xlim(0, 33000)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 10000, 20000, 30000])
    ax.set_xticklabels(['0', '10,000', '20,000', '30,000'])
    ax.tick_params(labelsize=fs - 1)
    ax.set_xlabel('t CO$_2$ for the same 33.2 GWh (derived)', fontsize=fs - 1)
    sp.save(fig, out / 'regions.svg')
regions_fig()

def eu_week(k, size=(5.8, 2.5), fs=13.5):
    t = np.array([dt.datetime.fromisoformat('2023-11-20T00:00:00'), dt.datetime.fromisoformat('2023-11-20T01:00:00'), dt.datetime.fromisoformat('2023-11-20T02:00:00'), dt.datetime.fromisoformat('2023-11-20T03:00:00'), dt.datetime.fromisoformat('2023-11-20T04:00:00'), dt.datetime.fromisoformat('2023-11-20T05:00:00'), dt.datetime.fromisoformat('2023-11-20T06:00:00'), dt.datetime.fromisoformat('2023-11-20T07:00:00'), dt.datetime.fromisoformat('2023-11-20T08:00:00'), dt.datetime.fromisoformat('2023-11-20T09:00:00'), dt.datetime.fromisoformat('2023-11-20T10:00:00'), dt.datetime.fromisoformat('2023-11-20T11:00:00'), dt.datetime.fromisoformat('2023-11-20T12:00:00'), dt.datetime.fromisoformat('2023-11-20T13:00:00'), dt.datetime.fromisoformat('2023-11-20T14:00:00'), dt.datetime.fromisoformat('2023-11-20T15:00:00'), dt.datetime.fromisoformat('2023-11-20T16:00:00'), dt.datetime.fromisoformat('2023-11-20T17:00:00'), dt.datetime.fromisoformat('2023-11-20T18:00:00'), dt.datetime.fromisoformat('2023-11-20T19:00:00'), dt.datetime.fromisoformat('2023-11-20T20:00:00'), dt.datetime.fromisoformat('2023-11-20T21:00:00'), dt.datetime.fromisoformat('2023-11-20T22:00:00'), dt.datetime.fromisoformat('2023-11-20T23:00:00'), dt.datetime.fromisoformat('2023-11-21T00:00:00'), dt.datetime.fromisoformat('2023-11-21T01:00:00'), dt.datetime.fromisoformat('2023-11-21T02:00:00'), dt.datetime.fromisoformat('2023-11-21T03:00:00'), dt.datetime.fromisoformat('2023-11-21T04:00:00'), dt.datetime.fromisoformat('2023-11-21T05:00:00'), dt.datetime.fromisoformat('2023-11-21T06:00:00'), dt.datetime.fromisoformat('2023-11-21T07:00:00'), dt.datetime.fromisoformat('2023-11-21T08:00:00'), dt.datetime.fromisoformat('2023-11-21T09:00:00'), dt.datetime.fromisoformat('2023-11-21T10:00:00'), dt.datetime.fromisoformat('2023-11-21T11:00:00'), dt.datetime.fromisoformat('2023-11-21T12:00:00'), dt.datetime.fromisoformat('2023-11-21T13:00:00'), dt.datetime.fromisoformat('2023-11-21T14:00:00'), dt.datetime.fromisoformat('2023-11-21T15:00:00'), dt.datetime.fromisoformat('2023-11-21T16:00:00'), dt.datetime.fromisoformat('2023-11-21T17:00:00'), dt.datetime.fromisoformat('2023-11-21T18:00:00'), dt.datetime.fromisoformat('2023-11-21T19:00:00'), dt.datetime.fromisoformat('2023-11-21T20:00:00'), dt.datetime.fromisoformat('2023-11-21T21:00:00'), dt.datetime.fromisoformat('2023-11-21T22:00:00'), dt.datetime.fromisoformat('2023-11-21T23:00:00'), dt.datetime.fromisoformat('2023-11-22T00:00:00'), dt.datetime.fromisoformat('2023-11-22T01:00:00'), dt.datetime.fromisoformat('2023-11-22T02:00:00'), dt.datetime.fromisoformat('2023-11-22T03:00:00'), dt.datetime.fromisoformat('2023-11-22T04:00:00'), dt.datetime.fromisoformat('2023-11-22T05:00:00'), dt.datetime.fromisoformat('2023-11-22T06:00:00'), dt.datetime.fromisoformat('2023-11-22T07:00:00'), dt.datetime.fromisoformat('2023-11-22T08:00:00'), dt.datetime.fromisoformat('2023-11-22T09:00:00'), dt.datetime.fromisoformat('2023-11-22T10:00:00'), dt.datetime.fromisoformat('2023-11-22T11:00:00'), dt.datetime.fromisoformat('2023-11-22T12:00:00'), dt.datetime.fromisoformat('2023-11-22T13:00:00'), dt.datetime.fromisoformat('2023-11-22T14:00:00'), dt.datetime.fromisoformat('2023-11-22T15:00:00'), dt.datetime.fromisoformat('2023-11-22T16:00:00'), dt.datetime.fromisoformat('2023-11-22T17:00:00'), dt.datetime.fromisoformat('2023-11-22T18:00:00'), dt.datetime.fromisoformat('2023-11-22T19:00:00'), dt.datetime.fromisoformat('2023-11-22T20:00:00'), dt.datetime.fromisoformat('2023-11-22T21:00:00'), dt.datetime.fromisoformat('2023-11-22T22:00:00'), dt.datetime.fromisoformat('2023-11-22T23:00:00'), dt.datetime.fromisoformat('2023-11-23T00:00:00'), dt.datetime.fromisoformat('2023-11-23T01:00:00'), dt.datetime.fromisoformat('2023-11-23T02:00:00'), dt.datetime.fromisoformat('2023-11-23T03:00:00'), dt.datetime.fromisoformat('2023-11-23T04:00:00'), dt.datetime.fromisoformat('2023-11-23T05:00:00'), dt.datetime.fromisoformat('2023-11-23T06:00:00'), dt.datetime.fromisoformat('2023-11-23T07:00:00'), dt.datetime.fromisoformat('2023-11-23T08:00:00'), dt.datetime.fromisoformat('2023-11-23T09:00:00'), dt.datetime.fromisoformat('2023-11-23T10:00:00'), dt.datetime.fromisoformat('2023-11-23T11:00:00'), dt.datetime.fromisoformat('2023-11-23T12:00:00'), dt.datetime.fromisoformat('2023-11-23T13:00:00'), dt.datetime.fromisoformat('2023-11-23T14:00:00'), dt.datetime.fromisoformat('2023-11-23T15:00:00'), dt.datetime.fromisoformat('2023-11-23T16:00:00'), dt.datetime.fromisoformat('2023-11-23T17:00:00'), dt.datetime.fromisoformat('2023-11-23T18:00:00'), dt.datetime.fromisoformat('2023-11-23T19:00:00'), dt.datetime.fromisoformat('2023-11-23T20:00:00'), dt.datetime.fromisoformat('2023-11-23T21:00:00'), dt.datetime.fromisoformat('2023-11-23T22:00:00'), dt.datetime.fromisoformat('2023-11-23T23:00:00'), dt.datetime.fromisoformat('2023-11-24T00:00:00'), dt.datetime.fromisoformat('2023-11-24T01:00:00'), dt.datetime.fromisoformat('2023-11-24T02:00:00'), dt.datetime.fromisoformat('2023-11-24T03:00:00'), dt.datetime.fromisoformat('2023-11-24T04:00:00'), dt.datetime.fromisoformat('2023-11-24T05:00:00'), dt.datetime.fromisoformat('2023-11-24T06:00:00'), dt.datetime.fromisoformat('2023-11-24T07:00:00'), dt.datetime.fromisoformat('2023-11-24T08:00:00'), dt.datetime.fromisoformat('2023-11-24T09:00:00'), dt.datetime.fromisoformat('2023-11-24T10:00:00'), dt.datetime.fromisoformat('2023-11-24T11:00:00'), dt.datetime.fromisoformat('2023-11-24T12:00:00'), dt.datetime.fromisoformat('2023-11-24T13:00:00'), dt.datetime.fromisoformat('2023-11-24T14:00:00'), dt.datetime.fromisoformat('2023-11-24T15:00:00'), dt.datetime.fromisoformat('2023-11-24T16:00:00'), dt.datetime.fromisoformat('2023-11-24T17:00:00'), dt.datetime.fromisoformat('2023-11-24T18:00:00'), dt.datetime.fromisoformat('2023-11-24T19:00:00'), dt.datetime.fromisoformat('2023-11-24T20:00:00'), dt.datetime.fromisoformat('2023-11-24T21:00:00'), dt.datetime.fromisoformat('2023-11-24T22:00:00'), dt.datetime.fromisoformat('2023-11-24T23:00:00'), dt.datetime.fromisoformat('2023-11-25T00:00:00'), dt.datetime.fromisoformat('2023-11-25T01:00:00'), dt.datetime.fromisoformat('2023-11-25T02:00:00'), dt.datetime.fromisoformat('2023-11-25T03:00:00'), dt.datetime.fromisoformat('2023-11-25T04:00:00'), dt.datetime.fromisoformat('2023-11-25T05:00:00'), dt.datetime.fromisoformat('2023-11-25T06:00:00'), dt.datetime.fromisoformat('2023-11-25T07:00:00'), dt.datetime.fromisoformat('2023-11-25T08:00:00'), dt.datetime.fromisoformat('2023-11-25T09:00:00'), dt.datetime.fromisoformat('2023-11-25T10:00:00'), dt.datetime.fromisoformat('2023-11-25T11:00:00'), dt.datetime.fromisoformat('2023-11-25T12:00:00'), dt.datetime.fromisoformat('2023-11-25T13:00:00'), dt.datetime.fromisoformat('2023-11-25T14:00:00'), dt.datetime.fromisoformat('2023-11-25T15:00:00'), dt.datetime.fromisoformat('2023-11-25T16:00:00'), dt.datetime.fromisoformat('2023-11-25T17:00:00'), dt.datetime.fromisoformat('2023-11-25T18:00:00'), dt.datetime.fromisoformat('2023-11-25T19:00:00'), dt.datetime.fromisoformat('2023-11-25T20:00:00'), dt.datetime.fromisoformat('2023-11-25T21:00:00'), dt.datetime.fromisoformat('2023-11-25T22:00:00'), dt.datetime.fromisoformat('2023-11-25T23:00:00'), dt.datetime.fromisoformat('2023-11-26T00:00:00'), dt.datetime.fromisoformat('2023-11-26T01:00:00'), dt.datetime.fromisoformat('2023-11-26T02:00:00'), dt.datetime.fromisoformat('2023-11-26T03:00:00'), dt.datetime.fromisoformat('2023-11-26T04:00:00'), dt.datetime.fromisoformat('2023-11-26T05:00:00'), dt.datetime.fromisoformat('2023-11-26T06:00:00'), dt.datetime.fromisoformat('2023-11-26T07:00:00'), dt.datetime.fromisoformat('2023-11-26T08:00:00'), dt.datetime.fromisoformat('2023-11-26T09:00:00'), dt.datetime.fromisoformat('2023-11-26T10:00:00'), dt.datetime.fromisoformat('2023-11-26T11:00:00'), dt.datetime.fromisoformat('2023-11-26T12:00:00'), dt.datetime.fromisoformat('2023-11-26T13:00:00'), dt.datetime.fromisoformat('2023-11-26T14:00:00'), dt.datetime.fromisoformat('2023-11-26T15:00:00'), dt.datetime.fromisoformat('2023-11-26T16:00:00'), dt.datetime.fromisoformat('2023-11-26T17:00:00'), dt.datetime.fromisoformat('2023-11-26T18:00:00'), dt.datetime.fromisoformat('2023-11-26T19:00:00'), dt.datetime.fromisoformat('2023-11-26T20:00:00'), dt.datetime.fromisoformat('2023-11-26T21:00:00'), dt.datetime.fromisoformat('2023-11-26T22:00:00'), dt.datetime.fromisoformat('2023-11-26T23:00:00'), dt.datetime.fromisoformat('2023-11-27T00:00:00')])
    hrs = np.array([(x - t[0]).total_seconds() / 3600 for x in t])
    fos = np.array([22.68733372, 22.996377, 23.19125151, 23.39794328, 24.3322664, 26.00400795, 29.43166865, 31.59256606, 31.49101424, 31.1445591, 30.6075944, 30.19263108, 29.81990104, 30.49207592, 31.74000857, 33.46262068, 35.02546228, 35.2638532, 35.84360128, 36.11885036, 36.04715464, 36.0570828, 35.67901321, 34.6651305, 33.03078621, 32.60450186, 33.00354586, 32.66229889, 33.00237436, 34.29600909, 36.88810617, 37.87156571, 37.17757033, 36.61078234, 35.95722791, 35.4874733, 35.44372327, 36.09073162, 36.57955215, 37.27227092, 37.59975621, 37.5185899, 37.62880841, 37.23112769, 36.67665352, 36.03510666, 35.33014088, 34.28319069, 33.48424363, 32.59954367, 31.89834025, 32.06587949, 32.53386181, 33.34131345, 35.41473293, 35.95772017, 34.62788936, 33.71127403, 32.64599315, 31.59735102, 30.76434976, 30.765676, 31.57929897, 32.77727814, 33.2749466, 33.10659532, 32.97467069, 32.40084637, 30.28344483, 30.04172545, 27.86946012, 25.96112153, 23.60898247, 22.27778789, 21.7223794, 21.12388556, 21.29560118, 22.10098546, 23.37006014, 25.00560574, 24.82069591, 24.05374247, 22.98968693, 21.7646066, 21.11060872, 21.29002494, 22.2011261, 23.32433103, 24.37197944, 25.42607351, 25.46675292, 24.73436232, 24.23382842, 23.43810817, 22.21098402, 22.05131055, 20.6464677, 20.61275992, 20.00057715, 20.29252731, 20.68312793, 21.77371389, 23.91609524, 24.90191603, 24.61372546, 23.68717848, 22.97446301, 21.97252936, 21.13693521, 21.42655375, 22.64182856, 24.31546381, 25.70479048, 25.92622131, 26.07045286, 25.56435739, 25.43699091, 24.55492776, 23.45457532, 22.76021929, 22.4065453, 21.48966974, 21.70150475, 22.14712479, 22.27475035, 22.53290467, 22.85720584, 23.7002966, 23.97159589, 23.58849274, 22.40292216, 22.15628403, 22.21266534, 22.51751204, 23.19101037, 25.15621156, 27.41970112, 28.9764489, 29.80703167, 29.59123955, 30.01921431, 31.01627081, 31.59712531, 31.64076657, 31.82904657, 31.90449802, 31.95913381, 32.52128638, 32.66060934, 32.86720876, 33.11074301, 32.84963849, 32.27368178, 31.96754462, 30.83886834, 29.8912878, 29.80140909, 30.18027032, 31.75861819, 33.81435312, 35.65543071, 36.01485392, 36.33453925, 36.20396384, 36.53184966, 36.77133195, 37.09060167, 36.34710324, 35.92256322])
    fig, ax = plt.subplots(figsize=size)
    ax.plot(hrs, fos, color=C['text'], lw=2)
    roll = np.array([30.72016533, 31.15114251, 31.55148105, 31.96032665, 32.34634147, 32.70759596, 33.05309601, 33.36378091, 33.62540589, 33.86234573, 34.09010503, 34.31300643, 34.53362485, 34.76795078, 35.0012281, 35.20287575, 35.36161118, 35.46887342, 35.56282079, 35.63720442, 35.68354931, 35.70977843, 35.70886275, 35.69432641, 35.67841225, 35.69730631, 35.69709972, 35.65104948, 35.62619867, 35.60667732, 35.56689833, 35.50550778, 35.42576422, 35.31952751, 35.19871466, 35.06074655, 34.89865812, 34.70368422, 34.48180691, 34.27346302, 34.08617166, 33.90597126, 33.72213815, 33.52821574, 33.32695402, 33.06057033, 32.81084611, 32.49998441, 32.15323153, 31.74176232, 31.31168916, 30.88769079, 30.43177438, 29.96351352, 29.49516652, 28.99330515, 28.53696705, 28.12833399, 27.72593684, 27.32359075, 26.91389306, 26.51165385, 26.11683506, 25.72607786, 25.33220506, 24.96124809, 24.64122635, 24.32839645, 24.00895961, 23.75689226, 23.48174154, 23.2459717, 23.08306291, 22.9596248, 22.89024863, 22.81850687, 22.78386694, 22.75834722, 22.74471091, 22.76746237, 22.76314197, 22.7545182, 22.7392447, 22.73861037, 22.74727382, 22.74837076, 22.75405946, 22.77242206, 22.81371926, 22.86925305, 22.89009254, 22.91524671, 22.94982984, 22.99996161, 23.04649576, 23.09831206, 23.12784992, 23.20118649, 23.2377244, 23.30859638, 23.38587128, 23.45218888, 23.48382183, 23.43970144, 23.38963396, 23.36287856, 23.35876665, 23.33495245, 23.3426089, 23.38743099, 23.43288758, 23.45577016, 23.49080131, 23.56225592, 23.68934874, 23.84503952, 24.01282628, 24.20375225, 24.47297488, 24.8122478, 25.1822706, 25.57487482, 26.008826, 26.43622721, 26.86848394, 27.30122807, 27.73182407, 28.15905479, 28.54027737, 28.88619761, 29.23532477, 29.58682253, 29.90911435, 30.22531201, 30.5445936, 30.90157726, 31.26233316, 31.60548856, 31.89875544, 32.17073492, 32.4462651, 32.7176249, 32.95741912, 33.18631397, 33.38241133, 33.55297452])
    lo, hi = (83, 21)
    span = 24 - 1
    if k >= 2:
        for i0, ck, lab in ((hi, 'danger', '36% fossil'), (lo, 'data', '23% fossil')):
            ax.axvspan(hrs[i0], hrs[i0 + span], color=C[ck + '_l'], zorder=0)
            ax.text(hrs[i0] + 11.5, 46.5, '24-h job\n' + lab, ha='center', va='top', fontsize=fs - 1, fontweight='bold', color=C[ck])
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xticks([12 + 24 * i for i in range(7)])
    ax.set_xticklabels(days)
    ax.tick_params(labelsize=fs - 1)
    ax.set_xlim(0, 168)
    ax.set_ylim(15, 47)
    ax.set_yticks([20, 30, 40])
    ax.set_yticklabels(['20%', '30%', '40%'])
    ax.set_ylabel('Fossil share\nof generation', fontsize=fs - 1)
    sp.save(fig, out / f'euweek_step{k}.svg')
    return (roll[lo], roll[hi])
for k in (1, 2):
    EU_LO, EU_HI = eu_week(k)

def jevons_fig(k, size=(5.4, 3.2), fs=14):
    fig, ax = plt.subplots(figsize=size)
    bars = [('Before', 1.0, 'neutral'), ('Energy\nper query', 0.5, 'data'), ('Queries', 4.0, 'memory'), ('Fleet\nenergy', 2.0, 'danger')]
    for i, (lab, v, ck) in enumerate(bars):
        a = 1.0 if i < 3 or k >= 2 else 0.0
        ax.bar(i, v, color=C[ck], width=0.62, alpha=a)
        ax.text(i, v + 0.08, f'{v:g}×', ha='center', va='bottom', fontsize=fs, fontweight='bold', color=C[ck], alpha=a)
        ax.text(i, -0.15, lab, ha='center', va='top', fontsize=fs - 2, fontweight='bold')
    if k < 2:
        ax.text(3, 0.35, '???', ha='center', fontsize=fs + 4, fontweight='bold', color=C['danger'])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 4.8)
    hide_axis(ax)
    sp.save(fig, out / f'jevons_step{k}.svg')
for k in (1, 2):
    jevons_fig(k)

def lbnl_fig(size=(5.4, 3.3), fs=14):
    fig, ax = plt.subplots(figsize=size)
    y14, y24 = (58.0, 192.0)
    lo, ref, hi = (521.0, 649.0, 843.0)
    ax.bar(0, y14, color=C['neutral'], width=0.6)
    ax.bar(1, y24, color=C['memory'], width=0.6)
    ax.bar(2, lo, color=C['danger_l'], edgecolor=C['danger'], hatch='//', width=0.6, lw=1)
    ax.add_patch(Rectangle((1.7, lo), 0.6, hi - lo, fill=False, edgecolor=C['danger'], ls='--', lw=1.5))
    ax.plot([1.7, 2.3], [ref, ref], color=C['danger'], lw=3)
    for x, key, ck in ((0, 'us_2014_twh', 'neutral'), (1, 'us_2024_twh', 'memory')):
        ax.text(x, {'us_2014_twh': 58.0, 'us_2024_twh': 192.0}[key] + 15, f"{ {'us_2014_twh': '58', 'us_2024_twh': '192'}[key]} TWh", ha='center', fontsize=fs, fontweight='bold', color=C[ck])
    ax.text(2.36, ref, '649 TWh\nreference', va='center', fontsize=fs - 2, fontweight='bold', color=C['danger'])
    ax.text(2.36, hi, '843', va='center', fontsize=fs - 3, color=C['danger'])
    ax.text(2.36, lo, '521', va='center', fontsize=fs - 3, color=C['danger'])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['2014', '2024', '2030\nprojected'])
    ax.tick_params(labelsize=fs - 2)
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(0, 920)
    ax.set_ylabel('US data-center TWh', fontsize=fs - 2)
    sp.save(fig, out / 'lbnl.svg')
lbnl_fig()
if __name__ == '__main__':
    print(f'EU 24-h fossil share {EU_LO:.1f} to {EU_HI:.1f}%')
