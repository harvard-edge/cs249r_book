#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 9 (Data Selection).

Every number comes from MLSysIM through calcs.py: val(key) is the string a
slide prints and num(key) its value, and bar lengths read the Scenarios.*
quantities directly. The notes below say where each figure's inputs live.
  - Data wall: reported pretraining tokens and the stock of public human text
    (Villalobos et al. 2024), read from MLSysIM's literature series
    (vol1/04_data_engineering/data/pretraining_tokens.csv and PublicTextStock), the same data
    as the Data Engineering chapter's figure; the fit is recomputed here
    exactly as the book does (least squares on log10 tokens), and its label
    is Scenarios.PretrainingCorpusGrowth's growth.
  - GPT-2 on WebText and tokens per parameter: Scenarios.Gpt2XlWebTextCorpus
    (GPT-2 XL, WebText, Chinchilla constants), Models GPT-3 and Llama 2 70B.
  - Selection-inequality scoreboard: Scenarios.ResNet50TenPercentCoresetH100
    (Example 9.2 and Sec. 9.7.1), at the chapter's unrounded minutes.
  - Deduplication: Lee et al. 2022 near-duplicate shares
    (Literature.DataSelectionResults) on GPT-2's run,
    Scenarios.Gpt2WebTextNearDuplicates.
  - MinHash page text: the C4 sentence Lee et al. 2022 found repeated
    (arXiv:2107.06499v2, footnote 1); signature values are illustrative.
  - Coverage floor: Scenarios.RareClassTenPercentCoreset (Sec. 9.3.4, 9.10).
  - Curriculum warmup: Scenarios.Cifar10LinearCurriculum (W illustrative).
  - Selection via proxy and FixMatch: Coleman et al. 2020 and Sohn et al.
    2020 (table 2, best error per label budget), Literature.DataSelectionResults.
  - ResNet-50 data path and echoing: Scenarios.ResNet50HalfCoresetEpochH100
    and ResNet50EchoingH100 (Napkin Math 9.4 and 9.5).
  - Cost breakdown and amortized ROI: Scenarios.ImageNetDataCostBreakdown
    and MinHashPipelineAmortization (Napkin Math 9.6; tables 13 and 14).

Run from this folder with the MLSysBook venv (it imports mlsysim):
    python3 make_figs.py
"""
from mlsysim.core.series_loader import load_series
import mlsysim
_VALS = {'corpus_growth': '2.9', 'first_corpus_b': '300', 'first_corpus_year': '2020', 'last_corpus_t': '36', 'last_corpus_year': '2025', 'stock_low_t': '100', 'stock_median_t': '300', 'stock_high_q': '1', 'stock_year': '2028', 'gpt2_tokens_b': '167', 'webtext_tokens_b': '10', 'webtext_gb': '40', 'gpt2_passes': '17', 'optimal_tokens_b': '30', 'gpt2_params_b': '1.5', 'gpt2_flops_m': '1.5', 'gpt2_flops_e': '21', 'flops_per_pt': '6', 'flops_per_pt_word': 'six', 'node_gpus_word': 'eight', 'mfu_pct': '50', 'node_days': '4.4', 'node_kwh': '590', 'node_hours': '105', 'one_h100_days': '35', 'shum_ppl': '34', 'shum_loss_lo': '20', 'shum_loss_hi': '28', 'shum_real_pct': '10', 'imagenet_m': '1.28', 'gflop_per_image': '24.6', 'icr_gain_full': '5', 'icr_gain_subset': '4.5', 'icr_gain_share': '90', 'epoch_flops_m': '3.2', 'epoch_flops_e': '16', 'half_flops_m': '1.6', 'half_flops_e': '16', 'icr_full_m': '1.6', 'icr_full_e': '-16', 'icr_half_m': '2.9', 'icr_half_e': '-16', 'icr_ratio': '1.8', 'paul_epoch': '20', 'sorscher_discard_pct': '20', 'sorscher_kept_pct': '80', 'gpt3_tokens_b': '300', 'gpt3_params_b': '175', 'gpt3_tpp': '1.7', 'llama2_tpp': '28.6', 'gpt2_unique_tpp': '6.7', 'gpt2_trained_tpp': '111', 'tpp_ref': '20', 'processed_over_optimal': '5.6', 'alloc_exp': '0.50', 'alloc_exp_short': '0.5', 'chin_models': '400', 'gopher_params_b': '280', 'gopher_tokens_b': '300', 'chin_params_b': '70', 'chin_tokens_t': '1.4', 'chin_token_mult': '4.7', 'chin_tpp': '20', 'doubling_pct': '41', 'c_over_n2': '120', 'gpt2_nopt_b': '3.5', 'gpt2_dopt_b': '71', 'dopt_over_webtext': '7.1', 'util_pct': '25', 'keep10_pct': '10', 'train_epochs': '100', 'base_min': '211', 'score_s': '42', 'score_proxy_s': '18', 'score_diff_s': '24', 'save_a_min': '188.8', 'save_a_pct': '89.7', 'save_b_min': '189.1', 'save_b_pct': '89.9', 'rescore_min': '70', 'subset_min': '21', 'rescore_ratio': '3.3', 'rescore_save_pct': '57', 'rescore_lost_pct': '37', 'est_pool': '1,000,000', 'est_kept': '100,000', 'est_cut_pct': '90', 'est_full_h': '10', 'est_subset_h': '1', 'est_score_h': '12', 'est_total_h': '13', 'est_loss_h': '3', 'est_breakeven_h': '9', 'lee_repeats': '61,036', 'lee_words': '61', 'minhash_hashes': '128', 'webtext_docs_m': '8', 'all_pairs_m': '3.2', 'all_pairs_e': '13', 'read_s': '6', 'sig_gb': '4.1', 'fw_tokens_t': '15', 'fw_snapshots': '96', 'fw_bands': '14', 'fw_rows': '8', 'fw_shingle': '5', 'fw_hashes': '112', 'fw_thresh': '0.72', 'fw_thresh_pct': '72', 'fw_probe_lo': '0.5', 'fw_probe_hi': '0.8', 'fw_p_lo_pct': '5.3', 'fw_p_hi_pct': '92.4', 'lee_c4_pct': '3.04', 'lee_rn_pct': '13.63', 'dedup_c4_h': '3', 'dedup_rn_h': '14', 'dedup_c4_kwh': '18', 'dedup_rn_kwh': '80', 'lee_mem_word': 'ten', 'lee_overlap_pct': '4', 'proxy_epochs': '5', 'std_epochs': '90', 'cov_pool': '1,000,000', 'cov_pool_m': '1', 'cov_rare_pct': '0.1', 'cov_keep_pct': '10', 'cov_floor': '150', 'cov_rare': '1,000', 'cov_kept': '100', 'cov_extra': '50', 'warm_w': '10', 'warm_served': '5.5', 'cur_epochs': '115', 'cur_base_epochs': '150', 'svp_speedup': '10', 'svp_speedup_word': 'ten', 'svp_err_pct': '0.1', 'svp_end2end': '1.6', 'svp_removed_pct': '50', 'svp_datasets_word': 'five', 'al_price': '10', 'al_start': '1,000', 'al_per_round': '100', 'al_rounds': '10', 'al_round_usd': '50', 'al_random': '5,000', 'al_active': '2,000', 'al_random_usd': '50,000', 'al_active_usd': '20,500', 'al_roi': '143.9', 'al_turnaround_d': '14', 'hosp_scans_m': '1', 'hosp_price': '5', 'hosp_all_m': '5', 'hosp_budget': '500,000', 'hosp_random': '100,000', 'hosp_active': '50,000', 'hosp_mult': '2', 'autoaug_gpuh': '15,000', 'cutmix_pts': '2.28', 'fm_acc_40': '88.6', 'fm_acc_250': '94.9', 'fm_acc_4k': '95.7', 'cifar_train': '50,000', 'fm_labels_4k': '4,000', 'fm_labels_250': '250', 'fm_share_pct': '8', 'fm_labels_40': '40', 'fm_per_class_word': 'four', 'ssl_sup_usd': '4,050', 'ssl_usd': '500', 'ssl_compute_mult': '5', 'pretrain_gpuh': '10,000', 'ft_gpuh': '50', 'scratch_gpuh': '1,000', 'crossover_tasks': '11', 'keep50_pct': '50', 'epoch_s': '126', 'epoch_kj': '88', 'pruned_s': '63', 'pruned_kj': '44', 'path_saved_h': '1.6', 'path_saved_h_round': '2', 'path_saved_mj': '4', 'path_ratio': '135', 'archive_gb': '148', 'nvme_gbs': '7', 'read_archive_s': '21', 'half_gb': '74', 'link_gb': '193', 'link_s': '3', 'demand_ips': '10,100', 'host_ips': '5,250', 'host_epoch_s': '244', 'tf_input_pct': '81.4', 'tf_input_round': '81', 'tf_step_s': '1.1', 'tf_compute_s': '0.2', 'echo_idle_pct': '48', 'echo_busy_pct': '52', 'echo_r': '1.92', 'echo_e': '2', 'echo_h_no': '6.1', 'echo_h_e': '3.2', 'choi_speedup': '3.25', 'choi_acc': '75', 'choi_ratio_word': 'six', 'choi_e_min': '2', 'choi_e_max': '5', 'choi_runs_word': 'five', 'cost_acq': '50,000', 'cost_labels': '64,058', 'cost_label_cents': '5', 'cost_label_usd': '0.05', 'cost_storage': '36', 'storage_months': '12', 'storage_cents': '2', 'cost_runs': '30', 'cost_run_h': '3.5', 'cost_gpuh': '105', 'gpu_rate': '4', 'cost_compute': '421', 'cost_total': '114,515', 'cost_compute_pct': '0.4', 'cost_data_pct': '99.6', 'roi_invest': '55,000', 'roi_build': '50,000', 'roi_operate': '5,000', 'roi_save': '10,000', 'roi_mult_1': '0.2', 'roi_runs_1_word': 'one', 'roi_mult_5': '0.9', 'roi_runs_5_word': 'five', 'roi_mult_10': '1.8', 'roi_runs_10_word': 'ten', 'roi_mult_50': '9.1', 'roi_runs_50_word': 'fifty', 'roi_breakeven': '5.5', 'roi_payback_word': 'six', 'data_scale': '10', 'data_scale_word': 'ten', 'data_gain': '4', 'pitfall_sel_h': '10', 'pitfall_full_h': '8'}
_NUMS = {'corpus_growth': 2.9, 'first_corpus_b': 300.0, 'first_corpus_year': 2020.0, 'last_corpus_t': 36.0, 'last_corpus_year': 2025.0, 'stock_low_t': 100.0, 'stock_median_t': 300.0, 'stock_high_q': 1.0, 'stock_year': 2028.0, 'gpt2_tokens_b': 167.0, 'webtext_tokens_b': 10.0, 'webtext_gb': 40.0, 'gpt2_passes': 17.0, 'optimal_tokens_b': 30.0, 'gpt2_params_b': 1.5, 'gpt2_flops_m': 1.5, 'gpt2_flops_e': 21.0, 'flops_per_pt': 6.0, 'mfu_pct': 50.0, 'node_days': 4.4, 'node_kwh': 590.0, 'node_hours': 105.0, 'one_h100_days': 35.0, 'shum_ppl': 34.0, 'shum_loss_lo': 20.0, 'shum_loss_hi': 28.0, 'shum_real_pct': 10.0, 'imagenet_m': 1.28, 'gflop_per_image': 24.6, 'icr_gain_full': 5.0, 'icr_gain_subset': 4.5, 'icr_gain_share': 90.0, 'epoch_flops_m': 3.2, 'epoch_flops_e': 16.0, 'half_flops_m': 1.6, 'half_flops_e': 16.0, 'icr_full_m': 1.6, 'icr_full_e': -16.0, 'icr_half_m': 2.9, 'icr_half_e': -16.0, 'icr_ratio': 1.8, 'paul_epoch': 20.0, 'sorscher_discard_pct': 20.0, 'sorscher_kept_pct': 80.0, 'gpt3_tokens_b': 300.0, 'gpt3_params_b': 175.0, 'gpt3_tpp': 1.7, 'llama2_tpp': 28.6, 'gpt2_unique_tpp': 6.7, 'gpt2_trained_tpp': 111.0, 'tpp_ref': 20.0, 'processed_over_optimal': 5.6, 'alloc_exp': 0.5, 'alloc_exp_short': 0.5, 'chin_models': 400.0, 'gopher_params_b': 280.0, 'gopher_tokens_b': 300.0, 'chin_params_b': 70.0, 'chin_tokens_t': 1.4, 'chin_token_mult': 4.7, 'chin_tpp': 20.0, 'doubling_pct': 41.0, 'c_over_n2': 120.0, 'gpt2_nopt_b': 3.5, 'gpt2_dopt_b': 71.0, 'dopt_over_webtext': 7.1, 'util_pct': 25.0, 'keep10_pct': 10.0, 'train_epochs': 100.0, 'base_min': 211.0, 'score_s': 42.0, 'score_proxy_s': 18.0, 'score_diff_s': 24.0, 'save_a_min': 188.8, 'save_a_pct': 89.7, 'save_b_min': 189.1, 'save_b_pct': 89.9, 'rescore_min': 70.0, 'subset_min': 21.0, 'rescore_ratio': 3.3, 'rescore_save_pct': 57.0, 'rescore_lost_pct': 37.0, 'est_pool': 1000000.0, 'est_kept': 100000.0, 'est_cut_pct': 90.0, 'est_full_h': 10.0, 'est_subset_h': 1.0, 'est_score_h': 12.0, 'est_total_h': 13.0, 'est_loss_h': 3.0, 'est_breakeven_h': 9.0, 'lee_repeats': 61036.0, 'lee_words': 61.0, 'minhash_hashes': 128.0, 'webtext_docs_m': 8.0, 'all_pairs_m': 3.2, 'all_pairs_e': 13.0, 'read_s': 6.0, 'sig_gb': 4.1, 'fw_tokens_t': 15.0, 'fw_snapshots': 96.0, 'fw_bands': 14.0, 'fw_rows': 8.0, 'fw_shingle': 5.0, 'fw_hashes': 112.0, 'fw_thresh': 0.72, 'fw_thresh_pct': 72.0, 'fw_probe_lo': 0.5, 'fw_probe_hi': 0.8, 'fw_p_lo_pct': 5.3, 'fw_p_hi_pct': 92.4, 'lee_c4_pct': 3.04, 'lee_rn_pct': 13.63, 'dedup_c4_h': 3.0, 'dedup_rn_h': 14.0, 'dedup_c4_kwh': 18.0, 'dedup_rn_kwh': 80.0, 'lee_overlap_pct': 4.0, 'proxy_epochs': 5.0, 'std_epochs': 90.0, 'cov_pool': 1000000.0, 'cov_pool_m': 1.0, 'cov_rare_pct': 0.1, 'cov_keep_pct': 10.0, 'cov_floor': 150.0, 'cov_rare': 1000.0, 'cov_kept': 100.0, 'cov_extra': 50.0, 'warm_w': 10.0, 'warm_served': 5.5, 'cur_epochs': 115.0, 'cur_base_epochs': 150.0, 'svp_speedup': 10.0, 'svp_err_pct': 0.1, 'svp_end2end': 1.6, 'svp_removed_pct': 50.0, 'al_price': 10.0, 'al_start': 1000.0, 'al_per_round': 100.0, 'al_rounds': 10.0, 'al_round_usd': 50.0, 'al_random': 5000.0, 'al_active': 2000.0, 'al_random_usd': 50000.0, 'al_active_usd': 20500.0, 'al_roi': 143.9, 'al_turnaround_d': 14.0, 'hosp_scans_m': 1.0, 'hosp_price': 5.0, 'hosp_all_m': 5.0, 'hosp_budget': 500000.0, 'hosp_random': 100000.0, 'hosp_active': 50000.0, 'hosp_mult': 2.0, 'autoaug_gpuh': 15000.0, 'cutmix_pts': 2.28, 'fm_acc_40': 88.6, 'fm_acc_250': 94.9, 'fm_acc_4k': 95.7, 'cifar_train': 50000.0, 'fm_labels_4k': 4000.0, 'fm_labels_250': 250.0, 'fm_share_pct': 8.0, 'fm_labels_40': 40.0, 'ssl_sup_usd': 4050.0, 'ssl_usd': 500.0, 'ssl_compute_mult': 5.0, 'pretrain_gpuh': 10000.0, 'ft_gpuh': 50.0, 'scratch_gpuh': 1000.0, 'crossover_tasks': 11.0, 'keep50_pct': 50.0, 'epoch_s': 126.0, 'epoch_kj': 88.0, 'pruned_s': 63.0, 'pruned_kj': 44.0, 'path_saved_h': 1.6, 'path_saved_h_round': 2.0, 'path_saved_mj': 4.0, 'path_ratio': 135.0, 'archive_gb': 148.0, 'nvme_gbs': 7.0, 'read_archive_s': 21.0, 'half_gb': 74.0, 'link_gb': 193.0, 'link_s': 3.0, 'demand_ips': 10100.0, 'host_ips': 5250.0, 'host_epoch_s': 244.0, 'tf_input_pct': 81.4, 'tf_input_round': 81.0, 'tf_step_s': 1.1, 'tf_compute_s': 0.2, 'echo_idle_pct': 48.0, 'echo_busy_pct': 52.0, 'echo_r': 1.92, 'echo_e': 2.0, 'echo_h_no': 6.1, 'echo_h_e': 3.2, 'choi_speedup': 3.25, 'choi_acc': 75.0, 'choi_e_min': 2.0, 'choi_e_max': 5.0, 'cost_acq': 50000.0, 'cost_labels': 64058.0, 'cost_label_cents': 5.0, 'cost_label_usd': 0.05, 'cost_storage': 36.0, 'storage_months': 12.0, 'storage_cents': 2.0, 'cost_runs': 30.0, 'cost_run_h': 3.5, 'cost_gpuh': 105.0, 'gpu_rate': 4.0, 'cost_compute': 421.0, 'cost_total': 114515.0, 'cost_compute_pct': 0.4, 'cost_data_pct': 99.6, 'roi_invest': 55000.0, 'roi_build': 50000.0, 'roi_operate': 5000.0, 'roi_save': 10000.0, 'roi_mult_1': 0.2, 'roi_mult_5': 0.9, 'roi_mult_10': 1.8, 'roi_mult_50': 9.1, 'roi_breakeven': 5.5, 'data_scale': 10.0, 'data_gain': 4.0, 'pitfall_sel_h': 10.0, 'pitfall_full_h': 8.0}
import datetime as dt
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from mlsysim.core.series_loader import load_series
plt = sp.plt
C = sp.C
out = Path(__file__).parent

def note(ax, text, x=0.0, y=-0.3):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, color=C['neutral'])

def data_wall():
    root = Path(__file__).resolve().parents[4] / 'mlsysim'
    sys.path.insert(0, str(root))
    from mlsysim import Literature
    corpora = load_series('vol1/04_data_engineering/data/pretraining_tokens.csv').entries
    stock = {r['bound']: r['tokens'] for r in load_series('vol1/04_data_engineering/data/public_text_stock.csv').entries}

    def year(d):
        d = dt.date.fromisoformat(d)
        return d.year + (d.timetuple().tm_yday - 1) / 365.25
    xs = np.array([year(r['date']) for r in corpora])
    ys = np.array([r['tokens'] for r in corpora])
    slope, icept = np.polyfit(xs, np.log10(ys), 1)
    growth = 10 ** slope
    assert 2.8 <= growth <= 3.0, growth
    show = {'GPT-3': (4, 8, 'left'), 'Chinchilla': (-6, 6, 'right'), 'Llama 2': (6, -4, 'left'), 'Qwen3': (-8, 4, 'right')}
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.9))
        if k == 2:
            ax.axhspan(stock['low'], stock['high'], color=C['compute_l'], zorder=0)
            ax.axhline(stock['median'], color=C['compute'], lw=1.2, ls=':')
            ax.text(2019.95, stock['median'] * 1.35, 'Stock of public human text', color=C['compute'], fontsize=12, fontweight='bold', va='bottom')
            ax.text(2019.95, stock['median'] * 0.75, 'estimate, 90% interval', color=C['compute'], fontsize=10, va='top')
        xx = np.linspace(2020, 2028.5, 50)
        ax.plot(xx, 10 ** (slope * xx + icept), color=C['memory'], lw=2, ls='--')
        ax.scatter(xs, ys, s=55, color=C['memory'], edgecolors='white', zorder=3)
        for r, x, y in zip(corpora, xs, ys):
            if r['model'] in show:
                dx, dy, ha = show[r['model']]
                ax.annotate(r['model'], (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=11, ha=ha, color=C['text'])
        ax.text(2026.1, 10 ** (slope * 2026.1 + icept) / 6, '2.9× per year', color=C['memory'], fontsize=12, fontweight='bold', ha='left')
        ax.set_yscale('log')
        ax.set_ylim(100000000000.0, 3000000000000000.0)
        ax.set_xlim(2019.8, 2028.5)
        ax.set_xticks(range(2020, 2029, 2))
        ax.set_ylabel('Training tokens')
        ax.grid(axis='y', color=C['grid'], lw=0.6)
        sp.save(fig, out / f'data_wall_step{k}.svg')

def gpt2_tokens():
    uniq, opt, done = (10000000000.0 / 10 ** 9, 30000000000.0 / 10 ** 9, 166666666700.0 / 10 ** 9)
    sp.hbars(out / 'gpt2_tokens.svg', [('Unique in WebText', uniq, 'data', '10 B'), ('Compute-optimal', opt, 'neutral', '30 B'), ('Processed in training', done, 'compute', '167 B: 17 passes')], 'B', log=False, xlabel='Tokens (billions)', size=(4.2, 1.7))

def icr_bars():
    full = float(getattr(1.586460099e-16, 'magnitude', 1.586460099e-16))
    half = float(getattr(2.855628178e-16, 'magnitude', 2.855628178e-16))
    sp.hbars(out / 'icr_bars.svg', [('All data', full / full, 'neutral', 'baseline'), ('Half, EL2N', half / full, 'data', '1.8×')], '', log=False, size=(3.9, 1.0), fontsize=15)

def warmup():
    """Linear pacing (chapter Sec. 9.4.1): epoch j of a W-epoch warmup serves
    the easiest j/W of the D samples, so the warmup serves D(W+1)/2 in place
    of W*D. W = 10 is an illustrative choice: 5.5 D of 10 D."""
    W = 10
    j = np.arange(1, W + 1)
    frac = j / W
    assert abs(frac.sum() - (W + 1) / 2) < 1e-09
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.bar(j, np.ones(W), color='white', edgecolor=C['neutral'], lw=1.2, ls='--', width=0.75)
    ax.bar(j, frac, color=C['data'], width=0.75)
    post = np.arange(W + 1, W + 4)
    ax.bar(post, np.ones(3), color=C['neutral_l'], edgecolor=C['neutral'], lw=1, width=0.75)
    ax.text(W + 2, 1.06, 'full data', ha='center', fontsize=11, color=C['neutral'])
    ax.text(1.0, 1.24, 'Warmup serves 5.5 D samples in place of 10 D', fontsize=13, fontweight='bold', color=C['data'])
    ax.text(1.0, 1.1, 'dashed: what full-data epochs would serve', fontsize=11, color=C['neutral'])
    ax.set_xticks(list(j) + list(post))
    ax.set_xticklabels([str(v) for v in j] + ['', '', ''])
    ax.set_xlim(0.4, W + 3.6)
    ax.set_ylim(0, 1.38)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', 'D/2', 'D'])
    ax.set_xlabel(f'Epoch (warmup W = {W}, illustrative)')
    ax.set_ylabel('Samples served')
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'warmup.svg')

def tokens_per_param():
    """GPT-2 XL appears twice: its unique tokens (WebText) and its training
    tokens (the run repeated WebText). Chinchilla's reference of about 20
    counts training tokens, each seen about once in that study."""
    rows = [('GPT-3 (175 B)', 1.714285714, 'neutral', '1.7'), ('GPT-2 XL, unique in WebText', 6.666666667, 'danger', '6.7'), ('GPT-2 XL, training tokens', 111.1111111, 'compute', '111: WebText 17 times'), ('Llama 2 70B', 28.57142857, 'neutral', '28.6')]
    ref = 20.0
    n = len(rows)
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.4))
        for i, (lab, v, ck, txt) in enumerate(rows):
            y = n - 1 - i
            col = C[ck] if k == 2 or ck == 'neutral' else C['neutral']
            ax.barh(y, v, color=col, height=0.6)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
            ax.text(v + 1.5, y, txt, va='center', fontsize=13, fontweight='bold', color=col)
        alpha = 1.0 if k == 2 else 0.0
        ax.axvline(ref, color=C['data'], lw=2, ls='--', alpha=alpha)
        ax.text(ref + 1.5, n - 0.55, 'Chinchilla reference: about 20', color=C['data'], fontsize=11, fontweight='bold', alpha=alpha, va='bottom')
        ax.text(ref + 12, 2.0, 'unique: a third of 20', color=C['danger'], fontsize=12, fontweight='bold', va='center', alpha=alpha)
        ax.set_yticks([])
        ax.set_xlim(0, 150)
        ax.set_ylim(-0.5, n - 0.1)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('Tokens per parameter')
        sp.save(fig, out / f'tokens_per_param_step{k}.svg')

def scoreboard():
    base, sub = (210.5051946, 21.05051946)
    sa, sb = (0.7016839821, 0.3080563824)
    rows = [('Full data', 0.0, base, '211 min'), ('Coreset, target scores once', sa, sub, 'saves 89.7%'), ('Coreset, proxy scores once', sb, sub, 'saves 89.9%'), ('Coreset, rescore every epoch', 70.16839821, sub, '70 + 21 min: saves 57%')]
    show_upto = {1: 1, 2: 3, 3: 4}
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.4))
        for i, (lab, score, train, txt) in enumerate(rows):
            y = 3 - i
            vis = i < show_upto[k]
            a = 1.0 if vis else 0.0
            ax.barh(y, train, left=score, color=C['neutral'] if i == 0 else C['data'], height=0.6, alpha=a)
            if score:
                ax.barh(y, score, color=C['compute'], height=0.6, alpha=a)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
            ck = 'danger' if i == 3 else 'neutral' if i == 0 else 'data'
            ax.text(score + train + 3, y, txt, va='center', fontsize=12, fontweight='bold', color=C[ck], alpha=a)
        a3 = 1.0 if k == 3 else 0.0
        ax.text(35, -0.62, 'scoring', color=C['compute'], fontsize=11, fontweight='bold', ha='center', va='top', alpha=a3)
        ax.set_yticks([])
        ax.set_xlim(0, 300)
        ax.set_ylim(-1.0, 3.5)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('Minutes on one H100 (100 epochs)')
        sp.save(fig, out / f'scoreboard_step{k}.svg')

def dedup_hours():
    rows = [('C4: 3.04% near-duplicates', 3.201887428, '3 h, 18 kWh'), ('RealNews: 13.63% near-duplicates', 14.35583081, '14 h, 80 kWh')]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 1.8))
        for i, (lab, h, txt) in enumerate(rows):
            y = 1 - i
            a = 1.0 if i < k else 0.0
            ax.barh(y, h, color=C['data'], height=0.6, alpha=a)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
            ax.text(h + 0.3, y, txt, va='center', fontsize=13, fontweight='bold', color=C['data'], alpha=a)
        ax.set_yticks([])
        ax.set_xlim(0, 22)
        ax.set_ylim(-0.5, 1.5)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Hours of GPT-2's 105-hour run saved")
        sp.save(fig, out / f'dedup_hours_step{k}.svg')

def coverage():
    fl = 150.0
    rows = [('Rare examples in the 1M pool', 1000.0, 'neutral', '1,000'), ('Kept by an unstratified 10% coreset', 100.0, 'danger', 'about 100'), ('Kept with the floor reserved first', fl, 'data', '150')]
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.1))
        for i, (lab, amount, ck, txt) in enumerate(rows):
            y = 2 - i
            a = 1.0 if i < k else 0.0
            ax.barh(y, amount, color=C[ck], height=0.6, alpha=a)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
            ax.text(max(amount, fl) + 25, y, txt, va='center', fontsize=13, fontweight='bold', color=C[ck], alpha=a)
        ax.axvline(fl, color=C['data'], lw=1.5, ls='--')
        ax.text(fl, 2.45, ' floor: 150 (illustrative)', color=C['data'], fontsize=11, fontweight='bold', va='bottom')
        ax.set_yticks([])
        ax.set_xlim(0, 1300)
        ax.set_ylim(-0.5, 2.9)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('Rare-class examples')
        sp.save(fig, out / f'coverage_step{k}.svg')

def svp():
    sp.hbars(out / 'svp_coreset.svg', [('Target trains on all of CIFAR10', 1.0, 'neutral', '1.0'), ('Proxy removes 50%, target trains', 1 / 1.6, 'data', '1.6× faster, same final accuracy')], '', log=False, xlabel='End-to-end training time (relative)', size=(5.8, 1.6))

def fixmatch():
    sp.hbars(out / 'fixmatch.svg', [(f"{_VALS[f'fm_labels_{k}']} labels", _NUMS[f'fm_acc_{k}'], 'data', f"{_VALS[f'fm_acc_{k}']}%") for k in ('40', '250', '4k')], '', log=False, xlabel='CIFAR-10 test accuracy (%); full labeled set: 50,000', size=(5.8, 2.0))

def echo():
    r, e2 = (1.923809524, 2.0)
    e = np.linspace(1, 3.2, 200)
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        if k >= 2:
            ax.plot(e, np.minimum(1, e / r) * 100, color=C['compute'], lw=3)
            ax.axvline(r, color=C['neutral'], lw=1, ls='--')
            ax.text(r + 0.04, 60, 'e = r = 1.92', color=C['neutral'], fontsize=11)
            ax.plot([e2], [100], 'o', ms=11, color=C['compute'], mec='white', mew=1.5, zorder=5)
            ax.annotate('e = 2: 90 epochs, 6.1 h to 3.2 h', (e2, 100), xytext=(8, -26), textcoords='offset points', fontsize=12, fontweight='bold', color=C['compute'])
        if k == 3:
            ax.plot(e, [100] * len(e), color=C['memory'], lw=2.5, ls=(0, (4, 2)))
            ax.text(2.35, 103, 'host twice as fast (r = 0.5)', color=C['memory'], fontsize=11, fontweight='bold')
        ax.plot([1], [100 / r], 'o', ms=11, color=C['danger'], mec='white', mew=1.5, zorder=5)
        ax.annotate('no echo: busy 52%', (1, 100 / r), xytext=(10, -4), textcoords='offset points', fontsize=12, fontweight='bold', color=C['danger'])
        ax.set_xlim(0.9, 3.2)
        ax.set_ylim(30, 112)
        ax.set_xticks([1, 1.5, 2, 2.5, 3])
        ax.set_xlabel('Echo factor e (uses of each prepared image)')
        ax.set_ylabel('Accelerator busy (%)')
        ax.grid(axis='y', color=C['grid'], lw=0.6)
        sp.save(fig, out / f'echo_step{k}.svg')

def cost():
    sp.hbars(out / 'cost_bars.svg', [('Acquisition (licensed images)', 50000, 'data', '$50,000'), ('Labels (1.28M at $0.05)', 64058.35, 'data', '$64,058'), ('Storage (148 GB, 12 months)', 35.56232921, 'neutral', '$36'), ('Compute (30 runs on one H100)', 421.0103893, 'compute', '$421 = 0.4%')], '', log=True, xlabel='Dollars (log scale); total $114,515', size=(5.8, 2.3))

def roi():
    inv, save = (55000, 10000)
    runs = np.array([1, 5, 10, 50])
    vals = np.array([0.1818181818, 0.9090909091, 1.818181818, 9.090909091])
    breakeven = 5.5
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    xs = np.linspace(1, 50, 200)
    ax.plot(xs, save * xs / inv, color=C['data'], lw=3)
    ax.axhline(1, color=C['neutral'], lw=1)
    ax.axvline(breakeven, color=C['neutral'], lw=1, ls='--')
    ax.text(breakeven * 1.05, 5.2, 'break-even:\n5.5 runs', color=C['neutral'], fontsize=11)
    for n, v in zip(runs, vals):
        ck = 'data' if v > 1 else 'danger'
        ax.plot([n], [v], 'o', ms=9, color=C[ck], zorder=5)
        off, ha = {50: ((-8, 4), 'right'), 5: ((-6, -20), 'right')}.get(n, ((9, -20), 'left'))
        ax.annotate(f"{ {0.1818181818: '0.2', 0.9090909091: '0.9', 1.818181818: '1.8', 9.090909091: '9.1'}[v]}$\\times$", (n, v), xytext=off, textcoords='offset points', fontsize=12, fontweight='bold', color=C[ck], ha=ha)
    ax.set_xscale('log')
    ax.set_xticks([1, 5, 10, 50])
    ax.set_xticklabels(['1', '5', '10', '50'])
    ax.minorticks_off()
    ax.set_xlim(0.85, 60)
    ax.set_ylim(-1.2, 10.5)
    ax.set_xlabel('Training runs that reuse the deduplicated data (log scale)')
    ax.set_ylabel('Savings / cost')
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    sp.save(fig, out / 'roi_runs.svg')
FONT = 'font-family="Helvetica, Arial, sans-serif"'

def svg_doc(w, h, body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" {FONT}>\n<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>\n{body}</svg>\n'

def text(x, y, s, size=18, color=None, weight='normal', anchor='middle', op=1.0):
    color = color or C['text']
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}" opacity="{op}">{s}</text>\n'

def box(x, y, w, h, stroke, fill, sw=2.5, op=1.0, rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>\n'

def arrow(x1, y, x2, op=1.0, color='#666666'):
    return f'<g opacity="{op}"><line x1="{x1}" y1="{y}" x2="{x2 - 10}" y2="{y}" stroke="{color}" stroke-width="3"/><polygon points="{x2},{y} {x2 - 12},{y - 7} {x2 - 12},{y + 7}" fill="{color}"/></g>\n'

def minhash():
    """Document, shingles, signature, bands and buckets, candidate check.
    The page text is the start of the 61-word C4 sentence that Lee et al.
    2022 found 61,036 times (arXiv:2107.06499v2, footnote 1); signature
    values are illustrative. 128 hashes, 8M documents come from the chapter
    (Sec. 9.3.2); pairs, read time, and signature size are computed in
    calcs.py."""
    mant = '3.2'
    sup = '¹³'
    W, H = (900, 360)
    xs = [20, 240, 485, 700]
    for k in range(1, 5):
        b = ''
        vis = [1.0 if i < k else 0.0 for i in range(4)]
        b += text(xs[0] + 89, 40, 'Web page', 18, weight='bold', op=vis[0])
        b += box(xs[0], 60, 178, 150, C['neutral'], '#F4F4F4', op=vis[0])
        for i, line in enumerate(['by combining', 'fantastic ideas,', 'interesting', 'arrangements, ...']):
            b += text(xs[0] + 14, 95 + 30 * i, line, 17, anchor='start', op=vis[0])
        b += text(xs[0] + 89, 236, '61,036 copies in C4', 16, color=C['danger'], weight='bold', op=vis[0])
        b += arrow(xs[0] + 184, 135, xs[1] - 20, op=vis[1])
        b += text(xs[1] + 100, 40, 'Shingles', 18, weight='bold', op=vis[1])
        for i, sh in enumerate(['by combining fantastic', 'combining fantastic ideas', 'fantastic ideas, interesting', 'ideas, interesting arr...']):
            b += box(xs[1] - 14, 62 + 37 * i, 228, 30, C['memory'], C['memory_l'], sw=1.5, op=vis[1], rx=4)
            b += text(xs[1] + 100, 83 + 37 * i, sh, 15, op=vis[1])
        b += arrow(xs[1] + 220, 135, xs[2] - 6, op=vis[2])
        b += text(xs[2] + 95, 40, 'Signature', 18, weight='bold', op=vis[2])
        vals = ['17', '4', '92', '33', '8', '61', '25', '70']
        for i, v in enumerate(vals):
            yy = 62 + 18.5 * i
            band = i // 2
            fill = [C['compute_l'], C['data_l'], C['compute_l'], C['data_l']][band]
            b += box(xs[2] + 45, yy, 100, 18, C['neutral'], fill, sw=1, op=vis[2], rx=1)
            b += text(xs[2] + 95, yy + 15, v, 15, op=vis[2])
        b += text(xs[2] + 95, 230, '128 minimum hashes', 16, color=C['neutral'], op=vis[2])
        b += text(xs[2] + 95, 250, '(8 of them shown)', 15, color=C['neutral'], op=vis[2])
        b += arrow(xs[2] + 196, 135, xs[3] - 6, op=vis[3])
        b += text(xs[3] + 95, 40, 'Bands to buckets', 18, weight='bold', op=vis[3])
        for i in range(3):
            hl = i == 1
            b += box(xs[3] + 2 + 66 * i, 70, 60, 110, C['data'] if hl else C['neutral'], C['data_l'] if hl else '#F4F4F4', sw=3 if hl else 1.5, op=vis[3], rx=4)
        b += text(xs[3] + 95, 115, 'page A', 15, color=C['data'], weight='bold', op=vis[3])
        b += text(xs[3] + 95, 140, 'page B', 15, color=C['data'], weight='bold', op=vis[3])
        b += text(xs[3] + 95, 210, 'same bucket:', 15, color=C['data'], weight='bold', op=vis[3])
        b += text(xs[3] + 95, 230, 'compare only these', 15, color=C['data'], weight='bold', op=vis[3])
        b += text(450, 295, f'All pairs of 8M pages: {mant} × 10{sup} comparisons', 19, color=C['neutral'], op=vis[3])
        b += f'''<line x1="225" y1="289" x2="675" y2="289" stroke="{C['danger']}" stroke-width="2.5" opacity="{vis[3]}"/>\n'''
        b += text(450, 335, 'Instead: one 40 GB read (6 s) and 4.1 GB of signatures', 19, color=C['data'], weight='bold', op=vis[3])
        (out / f'minhash_step{k}.svg').write_text(svg_doc(W, H, b))

def data_path():
    """ResNet-50 on one H100: four stages (book figure, redrawn at slide
    size). Numbers: chapter, Napkin Math 9.4 and 9.5 and its data-path
    figure. The 126 s epoch is the chapter's; derived values come from
    calcs.py."""
    W, H = (900, 330)
    stages = [('Local NVMe', 'storage', C['memory'], C['memory_l'], '148 GB read', '21 s per epoch'), ('Host CPUs', 'decode, augment', C['data'], C['data_l'], 'needs 10,100', 'images/s'), ('PCIe link', '', C['network'], C['network_l'], '193 GB', '3 s per epoch'), ('H100', 'accelerator', C['compute'], C['compute_l'], '126 s, 88 kJ', 'per epoch')]
    notes = {3: ['Coreset: 63 s, 44 kJ', 'scoring: 42 s, once'], 0: ['Halves only if the', '74 GB subset is written'], 1: ['Supplies only 5,250/s:', 'the slow stage']}
    order = [3, 0, 1]
    xs = [10, 240, 470, 700]
    for k in range(1, 4):
        b = ''
        active = order[k - 1]
        for i, (t1, t2, st, fl, n1, n2) in enumerate(stages):
            x = xs[i]
            w = 190
            hl = i == active
            red = i == 1 and k == 3
            b += box(x, 140, w, 110, C['danger'] if red else st, fl, sw=6 if hl else 2.5)
            b += text(x + w / 2, 188 if t2 else 202, t1, 21, weight='bold')
            if t2:
                b += text(x + w / 2, 214, t2, 17)
            b += text(x + w / 2, 285, n1, 17, color=C['text'])
            b += text(x + w / 2, 308, n2, 17, color=C['text'])
            if i < 3:
                b += arrow(x + w + 4, 195, xs[i + 1] - 4)
            shown = i in order[:k]
            if i in notes:
                for j, line in enumerate(notes[i]):
                    col = C['danger'] if i == 1 else st
                    b += text(x + w / 2, 80 + 26 * j, line, 17, color=col, weight='bold', op=1.0 if shown else 0.0)
        (out / f'data_path_step{k}.svg').write_text(svg_doc(W, H, b))
if __name__ == '__main__':
    data_wall()
    gpt2_tokens()
    icr_bars()
    warmup()
    tokens_per_param()
    scoreboard()
    dedup_hours()
    coverage()
    svp()
    fixmatch()
    echo()
    cost()
    roi()
    minhash()
    data_path()
    print('figures written')
