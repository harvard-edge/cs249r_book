#!/usr/bin/env python3
"""Figures for Lecture 19 (Agent Economics), Volume III.

Every value comes from calcs.py, which this script imports and which reads
MLSysIM's scenarios, so figures and slides agree; labels use deckvals.val. "AE" is the rendered chapter, Agent Economics; "NM 19.k" its
Napkin Math notebooks. The book's margin SVGs (rejected per accepted, queue
wait, break-even) and its full-page figures (critical-path waterfall,
cascade flowchart, hierarchical budget) are page-sized and fall below the
legibility floor on a slide, so each is redrawn here at the size of its slot:
  - multiplier: AE vol3_economics_rejected_per_accepted_margin.svg
  - kingman_step*: AE vol3_economics_queue_wait_margin.svg
  - breakeven: AE vol3_economics_breakeven_margin.svg
  - waterfall_step*: AE fig-vol3-critical-path-waterfall.svg
  - cascade_step*: AE fig-vol3-model-cascade-flowchart
  - budget_step*: AE fig-vol3-hierarchical-budget
Real-data plots: price_trend (Epoch AI 2025), swe_cohort (SWE-bench
trajectories), frontierswe (FrontierSWE V2),
inferencex (SemiAnalysis InferenceX). Full-width plots are drawn at about
5.8 x 2.5 in with 11 to 13 pt type; column plots at 5.4 x 3.6 in with 15 pt.

Run from this folder: python3 make_figs.py   (then make deck converts SVG)
"""
from types import SimpleNamespace
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
import datetime as dt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
C = sp.C
plt = sp.plt
plt.rcParams['text.parse_math'] = False
out = Path(__file__).parent
P = {'store': ('#f3e8ff', '#7c3aed', '#6d28d9'), 'tool': ('#fef3c7', '#d97706', '#b45309'), 'fault': ('#fee2e2', '#dc2626', '#b91c1c'), 'ok': ('#dcfce7', '#16a34a', '#15803d'), 'think': ('#e0f2fe', '#0284c7', '#0369a1'), 'gray': ('#f1f5f9', '#94a3b8', '#475569')}
VIO, RED, GRN, AMB, SKY, GRY, INK = ('#7c3aed', '#dc2626', '#16a34a', '#d97706', '#0284c7', '#64748b', '#0f172a')
FONT = 'Helvetica, Arial, sans-serif'
COL = (5.4, 3.6)
WIDE = (5.8, 2.5)

def money(v, nd=2):
    return f'${v:,.{nd}f}'

def tidy(ax):
    ax.grid(axis='both', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)

def hook(step):
    """Hook reveal: cost per accepted repair, frontier against low price."""
    fig, ax = plt.subplots(figsize=COL)
    rows = [('Frontier', 0.5466666667, P['think'][1]), ('Low token\nprice', 1.31552, RED)]
    for i, (lab, v, col) in enumerate(rows):
        shown = step >= 2
        ax.bar(i, v if shown else 0, width=0.6, color=col)
        ax.text(i, (v if shown else 0) + 0.05, money(v) if shown else '?', ha='center', va='bottom', fontsize=20, fontweight='bold', color=col if shown else C['neutral'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r[0] for r in rows], fontsize=16, fontweight='bold')
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.75)
    ax.set_ylabel('Dollars per accepted repair', fontsize=15)
    ax.tick_params(axis='y', labelsize=14)
    if step >= 2:
        ax.text(0.5, 1.6, '2.4× more per accepted repair', ha='center', fontsize=16, color=RED, fontweight='bold')
    ax.grid(axis='y', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'hook_step{step}.svg')
for k in (1, 2):
    hook(k)

def bill(step):
    """Per-attempt bill by term (NM 19.1): model tokens, environment, assessment."""
    fig, ax = plt.subplots(figsize=WIDE)
    for i, (x, lab) in enumerate([(SimpleNamespace(attempt_cost=0.41, cohort_cost=41.0, cohort_cost_with_triage=291.0, cost_per_accepted=0.5466666667, cost_per_accepted_with_triage=3.88, environment_cost=0.12, model_cost=0.24, rejected=25, rejected_per_accepted=0.3333333333), 'Frontier\n4 turns'), (SimpleNamespace(attempt_cost=0.32888, cohort_cost=32.888, cohort_cost_with_triage=782.888, cost_per_accepted=1.31552, cost_per_accepted_with_triage=31.31552, environment_cost=0.21, model_cost=0.06888, rejected=75, rejected_per_accepted=3.0), 'Low token price\n7 turns')]):
        y = 1 - i
        parts = [('model tokens', {'Frontier\n4 turns': 0.24, 'Low token price\n7 turns': 0.06888}[lab], SKY), ('environment', {'Frontier\n4 turns': 0.12, 'Low token price\n7 turns': 0.21}[lab], AMB), ('assessment', {'Frontier\n4 turns': 0.41, 'Low token price\n7 turns': 0.32888}[lab] - {'Frontier\n4 turns': 0.24, 'Low token price\n7 turns': 0.06888}[lab] - {'Frontier\n4 turns': 0.12, 'Low token price\n7 turns': 0.21}[lab], GRY)]
        left = 0
        for j, (pn, v, col) in enumerate(parts):
            show = step >= 2 or j == 0
            ax.barh(y, v, left=left, height=0.58, color=col, alpha=1 if show else 0)
            if show and v > 0.08:
                ax.text(left + v / 2, y, money(v, 3), ha='center', va='center', fontsize=12, color='white', fontweight='bold')
            elif show and j == 0:
                ax.text(left + v / 2, y + 0.42, money(v, 3), ha='center', va='center', fontsize=12, color=SKY, fontweight='bold')
            left += v
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
        if step >= 2:
            ax.text(left + 0.008, y, f"{money({'Frontier\n4 turns': 0.41, 'Low token price\n7 turns': 0.32888}[lab], 3)} per attempt", va='center', fontsize=12, fontweight='bold', color=C['text'])
    if step >= 1:
        ax.text(0.0, 1.42, 'model tokens', color=SKY, fontsize=12, fontweight='bold')
    if step >= 2:
        ax.text(0.22, 1.42, 'environment', color=AMB, fontsize=12, fontweight='bold')
        ax.text(0.44, 1.42, 'assessment', color=GRY, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 0.66)
    ax.set_ylim(-0.55, 1.7)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Dollars per attempt (NM 19.1, illustrative)', fontsize=11)
    ax.tick_params(labelsize=11)
    sp.save(fig, out / f'bill_step{step}.svg')
for k in (1, 2):
    bill(k)
PR_ROWS = ({'benchmark': 'MMLU', 'threshold_model': 'GPT-3', 'model': 'GPT-3', 'release_date': '2021-11-20', 'usd_per_mtok': 60.0, 'score': 43.9}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3', 'model': 'GPT-3', 'release_date': '2022-09-01', 'usd_per_mtok': 20.0, 'score': 43.9}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3', 'model': 'GPT-3.5 Turbo', 'release_date': '2023-03-06', 'usd_per_mtok': 2.0, 'score': 68.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3', 'model': 'Llama-2-Chat-13B', 'release_date': '2023-07-18', 'usd_per_mtok': 0.56, 'score': 45.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3', 'model': 'Llama 2-7B', 'release_date': '2023-12-01', 'usd_per_mtok': 0.2, 'score': 45.3}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3', 'model': 'Llama 2-7B', 'release_date': '2024-04-14', 'usd_per_mtok': 0.13, 'score': 45.3}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3', 'model': 'Llama-3.1-Instruct-8B', 'release_date': '2024-07-23', 'usd_per_mtok': 0.1, 'score': 71.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3', 'model': 'Llama-3.2-Instruct-3B', 'release_date': '2024-09-24', 'usd_per_mtok': 0.08, 'score': 64.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3', 'model': 'Gemini-1.5-Flash-8B', 'release_date': '2024-10-03', 'usd_per_mtok': 0.07, 'score': 75.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5', 'model': 'GPT-3.5', 'release_date': '2022-11-30', 'usd_per_mtok': 20.0, 'score': 64.8}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5', 'model': 'GPT-3.5 Turbo', 'release_date': '2023-03-06', 'usd_per_mtok': 2.0, 'score': 68.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5', 'model': 'GPT-3.5-Turbo-2023-11', 'release_date': '2023-11-06', 'usd_per_mtok': 0.75, 'score': 68.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5', 'model': 'Claude-3-Haiku', 'release_date': '2024-03-04', 'usd_per_mtok': 0.5, 'score': 71.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5', 'model': 'Gemini-1.5-Flash-2024-05', 'release_date': '2024-05-10', 'usd_per_mtok': 0.13, 'score': 79.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5', 'model': 'Llama-3.1-Instruct-8B', 'release_date': '2024-07-23', 'usd_per_mtok': 0.1, 'score': 71.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5', 'model': 'Gemini-1.5-Flash-8B', 'release_date': '2024-10-03', 'usd_per_mtok': 0.07, 'score': 75.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'GPT-3.5 Turbo', 'release_date': '2023-03-06', 'usd_per_mtok': 2.0, 'score': 68.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'GPT-3.5-Turbo-2023-11', 'release_date': '2023-11-06', 'usd_per_mtok': 0.75, 'score': 68.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'Claude-3-Haiku', 'release_date': '2024-03-04', 'usd_per_mtok': 0.5, 'score': 71.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'Gemini-1.5-Flash-2024-05', 'release_date': '2024-05-10', 'usd_per_mtok': 0.13, 'score': 79.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'Llama-3.1-Instruct-8B', 'release_date': '2024-07-23', 'usd_per_mtok': 0.1, 'score': 71.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'Gemini-1.5-Flash-8B', 'release_date': '2024-10-03', 'usd_per_mtok': 0.07, 'score': 75.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-4-0314', 'model': 'GPT-4-0314', 'release_date': '2023-03-14', 'usd_per_mtok': 37.5, 'score': 86.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-4-0314', 'model': 'GPT-4 Turbo', 'release_date': '2023-11-06', 'usd_per_mtok': 15.0, 'score': 87.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-4-0314', 'model': 'GPT-4o-2024-05', 'release_date': '2024-05-13', 'usd_per_mtok': 7.5, 'score': 87.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-4-0314', 'model': 'Gemini-1.5-Pro-2024-05', 'release_date': '2024-05-23', 'usd_per_mtok': 2.19, 'score': 86.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-4-0314', 'model': 'Gemini 2.0 Flash', 'release_date': '2025-02-05', 'usd_per_mtok': 0.175, 'score': 88.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4 Turbo', 'release_date': '2023-11-06', 'usd_per_mtok': 15.0, 'score': 87.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4o-2024-05', 'release_date': '2024-05-13', 'usd_per_mtok': 7.5, 'score': 87.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-4 Turbo', 'model': 'Claude-3.5-Sonnet-2024-06', 'release_date': '2024-06-20', 'usd_per_mtok': 6.0, 'score': 88.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-4 Turbo', 'model': 'Llama-3.1-Instruct-405B', 'release_date': '2024-07-23', 'usd_per_mtok': 3.5, 'score': 87.0}, {'benchmark': 'MMLU', 'threshold_model': 'GPT-4 Turbo', 'model': 'Gemini 2.0 Flash', 'release_date': '2025-02-05', 'usd_per_mtok': 0.175, 'score': 88.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4-0314', 'model': 'GPT-4-0314', 'release_date': '2023-03-14', 'usd_per_mtok': 37.5, 'score': 33.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4-0314', 'model': 'GPT-4 Turbo', 'release_date': '2023-11-06', 'usd_per_mtok': 15.0, 'score': 50.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4-0314', 'model': 'Mistral-Large-2024-02', 'release_date': '2024-02-26', 'usd_per_mtok': 6.0, 'score': 36.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4-0314', 'model': 'Claude-3-Haiku', 'release_date': '2024-03-04', 'usd_per_mtok': 0.5, 'score': 33.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4-0314', 'model': 'Gemini-1.5-Flash-2024-05', 'release_date': '2024-05-10', 'usd_per_mtok': 0.13, 'score': 39.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4-0314', 'model': 'Phi 4', 'release_date': '2024-12-13', 'usd_per_mtok': 0.1225, 'score': 53.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4 Turbo', 'release_date': '2023-11-06', 'usd_per_mtok': 15.0, 'score': 50.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4o-2024-05', 'release_date': '2024-05-13', 'usd_per_mtok': 7.5, 'score': 53.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4 Turbo', 'model': 'Claude-3.5-Sonnet-2024-06', 'release_date': '2024-06-20', 'usd_per_mtok': 6.0, 'score': 56.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4 Turbo', 'model': 'Llama-3.1-Instruct-405B', 'release_date': '2024-07-23', 'usd_per_mtok': 3.5, 'score': 50.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4 Turbo', 'model': 'Gemini-1.5-Pro-2024-09', 'release_date': '2024-09-24', 'usd_per_mtok': 2.19, 'score': 61.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4 Turbo', 'model': 'Phi 4', 'release_date': '2024-12-13', 'usd_per_mtok': 0.1225, 'score': 53.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4o-2024-05', 'model': 'GPT-4o-2024-05', 'release_date': '2024-05-13', 'usd_per_mtok': 7.5, 'score': 53.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4o-2024-05', 'model': 'Claude-3.5-Sonnet-2024-06', 'release_date': '2024-06-20', 'usd_per_mtok': 6.0, 'score': 56.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4o-2024-05', 'model': 'Gemini-1.5-Pro-2024-09', 'release_date': '2024-09-24', 'usd_per_mtok': 2.19, 'score': 61.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'GPT-4o-2024-05', 'model': 'Phi 4', 'release_date': '2024-12-13', 'usd_per_mtok': 0.1225, 'score': 53.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'Claude-3.5-Sonnet-2024-06', 'model': 'Claude-3.5-Sonnet-2024-06', 'release_date': '2024-06-20', 'usd_per_mtok': 6.0, 'score': 56.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'Claude-3.5-Sonnet-2024-06', 'model': 'Gemini-1.5-Pro-2024-09', 'release_date': '2024-09-24', 'usd_per_mtok': 2.19, 'score': 61.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'Claude-3.5-Sonnet-2024-06', 'model': 'DeepSeek-V3', 'release_date': '2024-12-26', 'usd_per_mtok': 0.4775, 'score': 57.0}, {'benchmark': 'GPQA Diamond', 'threshold_model': 'Claude-3.5-Sonnet-2024-06', 'model': 'Gemini 2.0 Flash', 'release_date': '2025-02-05', 'usd_per_mtok': 0.175, 'score': 62.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'GPT-3.5-Turbo-2023-06', 'release_date': '2023-06-13', 'usd_per_mtok': 3.25, 'score': 44.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'GPT-3.5-Turbo-2023-11', 'release_date': '2023-11-06', 'usd_per_mtok': 0.75, 'score': 44.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'Gemini-1.5-Flash-2024-05', 'release_date': '2024-05-10', 'usd_per_mtok': 0.13, 'score': 55.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'Llama-3.1-Instruct-8B', 'release_date': '2024-07-23', 'usd_per_mtok': 0.1, 'score': 50.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'Llama-3.2-Instruct-3B', 'release_date': '2024-09-24', 'usd_per_mtok': 0.08, 'score': 50.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'Gemini-1.5-Flash-8B', 'release_date': '2024-10-03', 'usd_per_mtok': 0.07, 'score': 70.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4 Turbo', 'release_date': '2023-11-06', 'usd_per_mtok': 15.0, 'score': 74.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4o-2024-05', 'release_date': '2024-05-13', 'usd_per_mtok': 7.5, 'score': 79.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4o-mini', 'release_date': '2024-07-18', 'usd_per_mtok': 0.26, 'score': 79.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-4 Turbo', 'model': 'DeepSeek-Coder-V2 236B', 'release_date': '2024-09-11', 'usd_per_mtok': 0.175, 'score': 74.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-4 Turbo', 'model': 'Gemini-1.5-Flash-2024-09', 'release_date': '2024-09-24', 'usd_per_mtok': 0.13, 'score': 83.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-4 Turbo', 'model': 'Phi 4', 'release_date': '2024-12-13', 'usd_per_mtok': 0.1225, 'score': 81.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-4o-2024-05', 'model': 'GPT-4o-2024-05', 'release_date': '2024-05-13', 'usd_per_mtok': 7.5, 'score': 79.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-4o-2024-05', 'model': 'GPT-4o-mini', 'release_date': '2024-07-18', 'usd_per_mtok': 0.26, 'score': 79.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-4o-2024-05', 'model': 'Gemini-1.5-Flash-2024-09', 'release_date': '2024-09-24', 'usd_per_mtok': 0.13, 'score': 83.0}, {'benchmark': 'MATH-500', 'threshold_model': 'GPT-4o-2024-05', 'model': 'Phi 4', 'release_date': '2024-12-13', 'usd_per_mtok': 0.1225, 'score': 81.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4-0613', 'model': 'GPT-4-0613', 'release_date': '2023-06-13', 'usd_per_mtok': 37.5, 'score': 23.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4-0613', 'model': 'GPT-4 Turbo', 'release_date': '2023-11-06', 'usd_per_mtok': 15.0, 'score': 36.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4-0613', 'model': 'Mistral-8x22', 'release_date': '2024-04-17', 'usd_per_mtok': 1.2, 'score': 23.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4-0613', 'model': 'Gemini-1.5-Flash-2024-05', 'release_date': '2024-05-10', 'usd_per_mtok': 0.13, 'score': 23.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4-0613', 'model': 'Phi 4', 'release_date': '2024-12-13', 'usd_per_mtok': 0.1225, 'score': 65.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4 Turbo', 'release_date': '2023-11-06', 'usd_per_mtok': 15.0, 'score': 36.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4o-2024-05', 'release_date': '2024-05-13', 'usd_per_mtok': 7.5, 'score': 48.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4 Turbo', 'model': 'Gemini-1.5-Pro-2024-05', 'release_date': '2024-05-23', 'usd_per_mtok': 2.19, 'score': 41.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4o-mini', 'release_date': '2024-07-18', 'usd_per_mtok': 0.26, 'score': 48.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4 Turbo', 'model': 'Gemini-1.5-Flash-2024-09', 'release_date': '2024-09-24', 'usd_per_mtok': 0.13, 'score': 58.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4 Turbo', 'model': 'Phi 4', 'release_date': '2024-12-13', 'usd_per_mtok': 0.1225, 'score': 65.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4o-2024-05', 'model': 'GPT-4o-2024-05', 'release_date': '2024-05-13', 'usd_per_mtok': 7.5, 'score': 48.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4o-2024-05', 'model': 'GPT-4o-mini', 'release_date': '2024-07-18', 'usd_per_mtok': 0.26, 'score': 48.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4o-2024-05', 'model': 'Gemini-1.5-Flash-2024-09', 'release_date': '2024-09-24', 'usd_per_mtok': 0.13, 'score': 58.0}, {'benchmark': 'MATH 5', 'threshold_model': 'GPT-4o-2024-05', 'model': 'Phi 4', 'release_date': '2024-12-13', 'usd_per_mtok': 0.1225, 'score': 65.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-4-0314', 'model': 'GPT-4-0314', 'release_date': '2023-03-14', 'usd_per_mtok': 37.5, 'score': 67.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-4-0314', 'model': 'GPT-3.5-Turbo-2023-06', 'release_date': '2023-06-13', 'usd_per_mtok': 3.25, 'score': 71.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-4-0314', 'model': 'GPT-3.5-Turbo-2023-11', 'release_date': '2023-11-06', 'usd_per_mtok': 0.75, 'score': 71.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-4-0314', 'model': 'Claude-3-Haiku', 'release_date': '2024-03-04', 'usd_per_mtok': 0.5, 'score': 77.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-4-0314', 'model': 'Gemma-2-27B', 'release_date': '2024-06-24', 'usd_per_mtok': 0.26, 'score': 76.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-4-0314', 'model': 'Mistral-NeMo', 'release_date': '2024-07-18', 'usd_per_mtok': 0.13, 'score': 71.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-4-0314', 'model': 'Llama-3.1-Instruct-8B', 'release_date': '2024-07-23', 'usd_per_mtok': 0.1, 'score': 67.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'GPT-3.5-Turbo-2023-06', 'release_date': '2023-06-13', 'usd_per_mtok': 3.25, 'score': 71.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'GPT-3.5-Turbo-2023-11', 'release_date': '2023-11-06', 'usd_per_mtok': 0.75, 'score': 71.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'Claude-3-Haiku', 'release_date': '2024-03-04', 'usd_per_mtok': 0.5, 'score': 77.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'Gemma-2-27B', 'release_date': '2024-06-24', 'usd_per_mtok': 0.26, 'score': 76.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'Mistral-NeMo', 'release_date': '2024-07-18', 'usd_per_mtok': 0.13, 'score': 71.0}, {'benchmark': 'HumanEval', 'threshold_model': 'GPT-3.5-Turbo-2023-06', 'model': 'Phi 4', 'release_date': '2024-12-13', 'usd_per_mtok': 0.1225, 'score': 87.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'GPT-3.5 Turbo', 'release_date': '2023-03-06', 'usd_per_mtok': 2.0, 'score': 1106.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'GPT-3.5-Turbo-2023-11', 'release_date': '2023-11-06', 'usd_per_mtok': 0.75, 'score': 1107.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'Claude-3-Haiku', 'release_date': '2024-03-04', 'usd_per_mtok': 0.5, 'score': 1179.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'Llama-3-Instruct-8B', 'release_date': '2024-04-18', 'usd_per_mtok': 0.15, 'score': 1152.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'Gemini-1.5-Flash-2024-05', 'release_date': '2024-05-10', 'usd_per_mtok': 0.13, 'score': 1227.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'Llama-3.1-Instruct-8B', 'release_date': '2024-07-23', 'usd_per_mtok': 0.1, 'score': 1172.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-3.5 Turbo', 'model': 'Gemini-1.5-Flash-8B', 'release_date': '2024-10-03', 'usd_per_mtok': 0.07, 'score': 1211.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4-0314', 'model': 'GPT-4-0314', 'release_date': '2023-03-14', 'usd_per_mtok': 37.5, 'score': 1186.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4-0314', 'model': 'GPT-4 Turbo', 'release_date': '2023-11-06', 'usd_per_mtok': 15.0, 'score': 1256.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4-0314', 'model': 'Claude-3-Sonnet', 'release_date': '2024-03-04', 'usd_per_mtok': 6.0, 'score': 1201.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4-0314', 'model': 'Llama-3-Instruct-70B', 'release_date': '2024-04-18', 'usd_per_mtok': 0.89, 'score': 1206.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4-0314', 'model': 'Gemini-1.5-Flash-2024-05', 'release_date': '2024-05-10', 'usd_per_mtok': 0.13, 'score': 1227.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4-0314', 'model': 'Gemini-1.5-Flash-8B', 'release_date': '2024-10-03', 'usd_per_mtok': 0.07, 'score': 1211.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4 Turbo', 'release_date': '2023-11-06', 'usd_per_mtok': 15.0, 'score': 1256.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4o-2024-05', 'release_date': '2024-05-13', 'usd_per_mtok': 7.5, 'score': 1285.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4 Turbo', 'model': 'Gemini-1.5-Pro-2024-05', 'release_date': '2024-05-23', 'usd_per_mtok': 2.19, 'score': 1260.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4 Turbo', 'model': 'GPT-4o-mini', 'release_date': '2024-07-18', 'usd_per_mtok': 0.26, 'score': 1273.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4 Turbo', 'model': 'Gemini-1.5-Flash-2024-09', 'release_date': '2024-09-24', 'usd_per_mtok': 0.13, 'score': 1271.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4o-2024-05', 'model': 'GPT-4o-2024-05', 'release_date': '2024-05-13', 'usd_per_mtok': 7.5, 'score': 1285.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4o-2024-05', 'model': 'GPT-4o-2024-08', 'release_date': '2024-08-06', 'usd_per_mtok': 4.38, 'score': 1337.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4o-2024-05', 'model': 'Gemini-1.5-Pro-2024-09', 'release_date': '2024-09-24', 'usd_per_mtok': 2.19, 'score': 1301.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4o-2024-05', 'model': 'DeepSeek-V3', 'release_date': '2024-12-26', 'usd_per_mtok': 0.4775, 'score': 1318.0}, {'benchmark': 'LMSys Chatbot Arena ELO', 'threshold_model': 'GPT-4o-2024-05', 'model': 'Gemini 2.0 Flash', 'release_date': '2025-02-05', 'usd_per_mtok': 0.175, 'score': 1358.0})

def price_trend():
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    end = dt.datetime(2025, 3, 1)
    style = [(('MMLU', 'GPT-3'), 'GPT-3-level knowledge (MMLU)', C['neutral']), (('MMLU', 'GPT-4-0314'), 'GPT-4-level knowledge (MMLU)', C['memory']), (('GPQA Diamond', 'GPT-4-0314'), 'GPT-4-level science (GPQA)', VIO)]
    ylab = {('MMLU', 'GPT-3'): 8, ('MMLU', 'GPT-4-0314'): 70, ('GPQA Diamond', 'GPT-4-0314'): 24}
    short = {('MMLU', 'GPT-3'): 'GPT-3 level, MMLU', ('MMLU', 'GPT-4-0314'): 'GPT-4 level, MMLU', ('GPQA Diamond', 'GPT-4-0314'): 'GPT-4 level, GPQA'}
    for key, lab, col in style:
        pts = sorted((p for p in PR_ROWS if (p['benchmark'], p['threshold_model']) == key), key=lambda p: p['release_date'])
        d = [dt.datetime.fromisoformat(p['release_date']) for p in pts]
        v = [float(p['usd_per_mtok']) for p in pts]
        ax.step(d + [end], v + [v[-1]], where='post', color=col, lw=2.4)
        ax.scatter(d, v, color=col, s=26, zorder=4)
        ax.text(dt.datetime(2024, 6, 15), ylab[key], f"{short[key]}: { {'GPT-3-level knowledge (MMLU)': '12', 'GPT-4-level knowledge (MMLU)': '16', 'GPT-4-level science (GPQA)': '42'}[lab]}× per year", color=col, fontsize=11, fontweight='bold', va='center')
    ax.set_yscale('log')
    ax.set_ylim(0.02, 150)
    ax.set_xlim(dt.datetime(2021, 9, 1), dt.datetime(2025, 3, 15))
    ax.set_ylim(0.03, 150)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:g}'))
    ax.set_ylabel('Cheapest $ per M tokens', fontsize=11)
    ax.tick_params(labelsize=11)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    tidy(ax)
    sp.save(fig, out / 'price_trend.svg')
price_trend()
sw = [(SimpleNamespace(cost_limit=None, cost_per_attempt=1.713719808, cost_per_resolved=5.100356571, input_output_ratio=104.2579578, max_cost=4.201617, max_overshoot=None, mean_calls=33.63, mean_overshoot=None, mean_resolved_cost=1.156898054, mean_unresolved_cost=1.995485033, median_calls=27.0, median_input_output_ratio=102.131244, median_input_tokens=387195.5, median_output_tokens=3852.5, output_token_share=0.009500469336, over_limit_runs=None, resolve_rate=0.336, resolved=168, survivor_understatement=4.408648243, tasks=500, unresolved_share=0.7731731032), '2024 run A'), (SimpleNamespace(cost_limit=None, cost_per_attempt=1.163070661, cost_per_resolved=1.638127692, input_output_ratio=13.88317394, max_cost=3.00501825, max_overshoot=None, mean_calls=59.96, mean_overshoot=None, mean_resolved_cost=1.041257051, mean_unresolved_cost=1.461303983, median_calls=54.5, median_input_output_ratio=14.33224632, median_input_tokens=32082.5, median_output_tokens=2050.5, output_token_share=0.06718996928, over_limit_runs=None, resolve_rate=0.71, resolved=71, survivor_understatement=1.573221223, tasks=100, unresolved_share=0.3643614864), '2025 run B'), (SimpleNamespace(cost_limit=3.0, cost_per_attempt=0.5515226445, cost_per_resolved=0.7295273075, input_output_ratio=62.92777844, max_cost=3.2862315, max_overshoot=0.2862315, mean_calls=28.932, mean_overshoot=0.05713146875, mean_resolved_cost=0.4464282348, mean_unresolved_cost=0.8771430287, median_calls=23.0, median_input_output_ratio=47.04383618, median_input_tokens=183674.5, median_output_tokens=3925.5, output_token_share=0.01564265214, over_limit_runs=8, resolve_rate=0.756, resolved=378, survivor_understatement=1.634142401, tasks=500, unresolved_share=0.388058226), '2026 run C'), (SimpleNamespace(cost_limit=None, cost_per_attempt=0.4735670177, cost_per_resolved=0.6505041452, input_output_ratio=48.19999609, max_cost=2.6613013, max_overshoot=None, mean_calls=35.04, mean_overshoot=None, mean_resolved_cost=0.3981253827, mean_unresolved_cost=0.6754843349, median_calls=31.0, median_input_output_ratio=37.82420344, median_input_tokens=399758.5, median_output_tokens=10431.0, output_token_share=0.02032520487, over_limit_runs=None, resolve_rate=0.728, resolved=364, survivor_understatement=1.633917789, tasks=500, unresolved_share=0.3879741034), '2026 run D')]

def swe_cohort(step):
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    import numpy as np
    x = np.arange(len([(SimpleNamespace(cost_limit=None, cost_per_attempt=1.713719808, cost_per_resolved=5.100356571, input_output_ratio=104.2579578, max_cost=4.201617, max_overshoot=None, mean_calls=33.63, mean_overshoot=None, mean_resolved_cost=1.156898054, mean_unresolved_cost=1.995485033, median_calls=27.0, median_input_output_ratio=102.131244, median_input_tokens=387195.5, median_output_tokens=3852.5, output_token_share=0.009500469336, over_limit_runs=None, resolve_rate=0.336, resolved=168, survivor_understatement=4.408648243, tasks=500, unresolved_share=0.7731731032), '2024 run A'), (SimpleNamespace(cost_limit=None, cost_per_attempt=1.163070661, cost_per_resolved=1.638127692, input_output_ratio=13.88317394, max_cost=3.00501825, max_overshoot=None, mean_calls=59.96, mean_overshoot=None, mean_resolved_cost=1.041257051, mean_unresolved_cost=1.461303983, median_calls=54.5, median_input_output_ratio=14.33224632, median_input_tokens=32082.5, median_output_tokens=2050.5, output_token_share=0.06718996928, over_limit_runs=None, resolve_rate=0.71, resolved=71, survivor_understatement=1.573221223, tasks=100, unresolved_share=0.3643614864), '2025 run B'), (SimpleNamespace(cost_limit=3.0, cost_per_attempt=0.5515226445, cost_per_resolved=0.7295273075, input_output_ratio=62.92777844, max_cost=3.2862315, max_overshoot=0.2862315, mean_calls=28.932, mean_overshoot=0.05713146875, mean_resolved_cost=0.4464282348, mean_unresolved_cost=0.8771430287, median_calls=23.0, median_input_output_ratio=47.04383618, median_input_tokens=183674.5, median_output_tokens=3925.5, output_token_share=0.01564265214, over_limit_runs=8, resolve_rate=0.756, resolved=378, survivor_understatement=1.634142401, tasks=500, unresolved_share=0.388058226), '2026 run C'), (SimpleNamespace(cost_limit=None, cost_per_attempt=0.4735670177, cost_per_resolved=0.6505041452, input_output_ratio=48.19999609, max_cost=2.6613013, max_overshoot=None, mean_calls=35.04, mean_overshoot=None, mean_resolved_cost=0.3981253827, mean_unresolved_cost=0.6754843349, median_calls=31.0, median_input_output_ratio=37.82420344, median_input_tokens=399758.5, median_output_tokens=10431.0, output_token_share=0.02032520487, over_limit_runs=None, resolve_rate=0.728, resolved=364, survivor_understatement=1.633917789, tasks=500, unresolved_share=0.3879741034), '2026 run D')]))
    w = 0.4
    a = [1.156898054, 1.041257051, 0.4464282348, 0.3981253827]
    b = [5.100356571, 1.638127692, 0.7295273075, 0.6505041452]
    ax.bar(x - w / 2, a, w, color=C['neutral_l'], edgecolor=C['neutral'], lw=1.2)
    for i, v in enumerate(a):
        ax.text(x[i] - w / 2, v + 0.08, money(v), ha='center', fontsize=12, color=C['neutral'], fontweight='bold')
    if step >= 2:
        ax.bar(x + w / 2, b, w, color=VIO)
        for i, v in enumerate(b):
            ax.text(x[i] + w / 2, v + 0.08, money(v), ha='center', fontsize=12, color=VIO, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['2024 A\n168/500', '2025 B\n71/100', '2026 C\n378/500', '2026 D\n364/500'], fontsize=13)
    ax.set_ylabel('Dollars', fontsize=13)
    ax.set_ylim(0, 6.2)
    ax.tick_params(axis='y', labelsize=13)
    ax.text(1.0, 5.6, 'mean cost of a resolved run', ha='left', fontsize=14, color=C['neutral'], fontweight='bold')
    if step >= 2:
        ax.text(1.0, 4.9, 'cohort cost per resolved task', ha='left', fontsize=14, color=VIO, fontweight='bold')
    ax.grid(axis='y', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'swe_step{step}.svg')
for k in (1, 2):
    swe_cohort(k)

def multiplier():
    import numpy as np
    fig, ax = plt.subplots(figsize=COL)
    a = np.linspace(0.12, 0.98, 200)
    ax.plot(a * 100, (1 - a) / a, color=VIO, lw=3)
    for acc, col in ((25 / 100, RED), (75 / 100, GRN)):
        m = (1 - acc) / acc
        ax.plot([acc * 100], [m], 'o', ms=12, color=col, mec='white', mew=1.5, zorder=5)
        ax.annotate(f'{m:.2f}' if m < 1 else f'{m:.0f}', (acc * 100, m), xytext=(10, 10), textcoords='offset points', fontsize=17, fontweight='bold', color=col)
    ax.text(30, 4.3, 'low price: 25 of 100 accepted', fontsize=14, color=RED, va='center')
    ax.text(58, 1.9, 'frontier: 75 of 100', fontsize=14, color=GRN, va='center')
    ax.set_xlabel('Acceptance rate α (percent)', fontsize=15)
    ax.set_ylabel('Rejected per accepted', fontsize=15)
    ax.set_ylim(0, 7.5)
    ax.set_xlim(10, 100)
    ax.tick_params(labelsize=14)
    tidy(ax)
    sp.save(fig, out / 'multiplier.svg')
multiplier()

def resend(step):
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    t = np.arange(1, len([0.027, 0.0315, 0.036, 0.0405, 0.045, 0.0495, 0.054, 0.0585, 0.063, 0.0675, 0.072, 0.0765, 0.081, 0.0855, 0.09, 0.0945, 0.099, 0.1035, 0.108, 0.1125, 0.117, 0.1215, 0.126, 0.1305, 0.135, 0.1395, 0.144, 0.1485, 0.153, 0.1575]) + 1)
    ax.bar(t - 0.2, [0.027, 0.0315, 0.036, 0.0405, 0.045, 0.0495, 0.054, 0.0585, 0.063, 0.0675, 0.072, 0.0765, 0.081, 0.0855, 0.09, 0.0945, 0.099, 0.1035, 0.108, 0.1125, 0.117, 0.1215, 0.126, 0.1305, 0.135, 0.1395, 0.144, 0.1485, 0.153, 0.1575], 0.4, color=C['neutral_l'], edgecolor=C['neutral'], lw=0.8)
    if step >= 2:
        ax.bar(t + 0.2, [0.027, 0.0099, 0.01035, 0.0108, 0.01125, 0.0117, 0.01215, 0.0126, 0.01305, 0.0135, 0.01395, 0.0144, 0.01485, 0.0153, 0.01575, 0.0162, 0.01665, 0.0171, 0.01755, 0.018, 0.01845, 0.0189, 0.01935, 0.0198, 0.02025, 0.0207, 0.02115, 0.0216, 0.02205, 0.0225], 0.4, color=SKY)
    ax.text(1, 0.15, 'resend all, fresh price: $2.768', fontsize=12.5, color=C['neutral'], fontweight='bold')
    if step >= 2:
        ax.text(1, 0.12, 'cached prefix: $0.497 (5.6× less)', fontsize=12.5, color=SKY, fontweight='bold')
    ax.set_xlabel('Turn of a 30-turn trajectory (NM 19.2, illustrative)', fontsize=11)
    ax.set_ylabel('Dollars per call', fontsize=11)
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 0.17)
    ax.tick_params(labelsize=11)
    ax.grid(axis='y', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'resend_step{step}.svg')
for k in (1, 2):
    resend(k)

def payback():
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    for c, col, lab, ytxt in ((SimpleNamespace(breakeven_calls=12, compact_cost=Q_(0.19035, 'dollar'), compact_total=Q_(0.22605, 'dollar'), context_at_compaction=30500, gap_before=Q_(0.003, 'dollar'), keep_cost=Q_(0.24525, 'dollar'), net_by_calls={1: 0.0585, 2: 0.05295, 3: 0.0474, 4: 0.04185, 5: 0.0363, 6: 0.03075, 7: 0.0252, 8: 0.01965, 9: 0.0141, 10: 0.00855, 11: 0.003, 12: -0.00255, 13: -0.0081, 14: -0.01365, 15: -0.0192, 16: -0.02475, 17: -0.0303, 18: -0.03585, 19: -0.0414, 20: -0.04695, 21: -0.0525, 22: -0.05805, 23: -0.0636, 24: -0.06915, 25: -0.0747, 26: -0.08025, 27: -0.0858, 28: -0.09135, 29: -0.0969, 30: -0.10245}, per_call_saving=Q_(0.00555, 'dollar'), remaining_calls=15, saving_at=Q_(0.00255, 'dollar'), summary_cost=Q_(0.0357, 'dollar'), upfront=Q_(0.0585, 'dollar')), SKY, 'cached read at 10%', 10.4), (SimpleNamespace(breakeven_calls=23, compact_cost=Q_(0.144675, 'dollar'), compact_total=Q_(0.176025, 'dollar'), context_at_compaction=30500, gap_before=Q_(0.000225, 'dollar'), keep_cost=Q_(0.156375, 'dollar'), net_by_calls={1: 0.0585, 2: 0.055725, 3: 0.05295, 4: 0.050175, 5: 0.0474, 6: 0.044625, 7: 0.04185, 8: 0.039075, 9: 0.0363, 10: 0.033525, 11: 0.03075, 12: 0.027975, 13: 0.0252, 14: 0.022425, 15: 0.01965, 16: 0.016875, 17: 0.0141, 18: 0.011325, 19: 0.00855, 20: 0.005775, 21: 0.003, 22: 0.000225, 23: -0.00255, 24: -0.005325, 25: -0.0081, 26: -0.010875, 27: -0.01365, 28: -0.016425, 29: -0.0192, 30: -0.021975}, per_call_saving=Q_(0.002775, 'dollar'), remaining_calls=15, saving_at=Q_(0.00255, 'dollar'), summary_cost=Q_(0.03135, 'dollar'), upfront=Q_(0.0585, 'dollar')), AMB, 'cached read at 5%', 8.0)):
        n = sorted({'cached read at 10%': {1: 0.0585, 2: 0.05295, 3: 0.0474, 4: 0.04185, 5: 0.0363, 6: 0.03075, 7: 0.0252, 8: 0.01965, 9: 0.0141, 10: 0.00855, 11: 0.003, 12: -0.00255, 13: -0.0081, 14: -0.01365, 15: -0.0192, 16: -0.02475, 17: -0.0303, 18: -0.03585, 19: -0.0414, 20: -0.04695, 21: -0.0525, 22: -0.05805, 23: -0.0636, 24: -0.06915, 25: -0.0747, 26: -0.08025, 27: -0.0858, 28: -0.09135, 29: -0.0969, 30: -0.10245}, 'cached read at 5%': {1: 0.0585, 2: 0.055725, 3: 0.05295, 4: 0.050175, 5: 0.0474, 6: 0.044625, 7: 0.04185, 8: 0.039075, 9: 0.0363, 10: 0.033525, 11: 0.03075, 12: 0.027975, 13: 0.0252, 14: 0.022425, 15: 0.01965, 16: 0.016875, 17: 0.0141, 18: 0.011325, 19: 0.00855, 20: 0.005775, 21: 0.003, 22: 0.000225, 23: -0.00255, 24: -0.005325, 25: -0.0081, 26: -0.010875, 27: -0.01365, 28: -0.016425, 29: -0.0192, 30: -0.021975}}[lab])
        ax.plot(n, {'cached read at 10%': [5.85, 5.295, 4.74, 4.185, 3.63, 3.075, 2.52, 1.965, 1.41, 0.855, 0.3, -0.255, -0.81, -1.365, -1.92, -2.475, -3.03, -3.585, -4.14, -4.695, -5.25, -5.805, -6.36, -6.915, -7.47, -8.025, -8.58, -9.135, -9.69, -10.245], 'cached read at 5%': [5.85, 5.5725, 5.295, 5.0175, 4.74, 4.4625, 4.185, 3.9075, 3.63, 3.3525, 3.075, 2.7975, 2.52, 2.2425, 1.965, 1.6875, 1.41, 1.1325, 0.855, 0.5775, 0.3, 0.0225, -0.255, -0.5325, -0.81, -1.0875, -1.365, -1.6425, -1.92, -2.1975]}[lab], color=col, lw=2.8)
        ax.plot([{'cached read at 10%': 12, 'cached read at 5%': 23}[lab]], [{'cached read at 10%': -0.00255, 'cached read at 5%': -0.00255}[lab] * 100], 'o', ms=9, color=col, mec='white', zorder=5)
        ax.text(0.0, ytxt, f"{lab}: payback { {'cached read at 10%': 12, 'cached read at 5%': 23}[lab]} calls", fontsize=14, color=col, fontweight='bold', clip_on=False)
    ax.axhline(0, color=INK, lw=1.2)
    rest = 15.0
    ax.plot([rest, rest], [-3.5, 6.5], color=RED, lw=1.6, ls='--')
    ax.text(rest + 0.3, 5.4, '15 calls remain', color=RED, fontsize=14, fontweight='bold')
    ax.set_xlabel('Calls remaining after compaction (NM 19.3, illustrative)', fontsize=13)
    ax.set_ylabel('Compact minus keep (¢)', fontsize=13)
    ax.set_xlim(0, 30)
    ax.set_ylim(-11, 6.5)
    ax.tick_params(labelsize=13)
    tidy(ax)
    sp.save(fig, out / 'payback.svg')
payback()
c5 = {'tests': 245.6, 'sandbox': 37.35, 'runtime': 0.2, 'model': 29.25}

def waterfall(step):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    floor = c5['tests'] + c5['sandbox'] + c5['runtime']
    rows = [('Baseline', c5['model'], f'{312.4:.1f} s'), ('Model 4× faster', c5['model'] / 4.0, f'{290.4625:.1f} s, {1.075526101:.2f}×'), ('Speculative decoding', c5['model'] / 3.285714286, f'{292.0521739:.1f} s, {1.069671887:.2f}×'), ('Instant model calls', 0.0, f'{283.15:.1f} s, {1.103302137:.2f}×')]
    for i, (lab, m, txt) in enumerate(rows):
        y = 3 - i
        show = i < step
        a = 1 if show else 0
        ax.barh(y, c5['tests'], height=0.6, color=AMB, alpha=a)
        ax.barh(y, c5['sandbox'], left=c5['tests'], height=0.6, color=P['tool'][0], edgecolor=AMB, hatch='///', lw=0.8, alpha=a)
        ax.barh(y, m, left=floor, height=0.6, color=SKY, alpha=a)
        ax.text(-4, y, lab, ha='right', va='center', fontsize=14, fontweight='bold', color=C['text'])
        if show:
            ax.text(floor + m + 3, y, txt, va='center', fontsize=14, fontweight='bold', color=C['text'])
    ax.text(c5['tests'] / 2, 3.47, 'required tests', ha='center', fontsize=14, color=P['tool'][2], fontweight='bold')
    ax.text(c5['tests'] + c5['sandbox'] / 2, 3.47, 'sandbox', ha='center', fontsize=14, color=P['tool'][2], fontweight='bold')
    ax.text(floor + 32, 3.47, 'model', ha='left', fontsize=14, color=SKY, fontweight='bold')
    ax.axvline(floor, color=RED, ls='--', lw=1.8)
    ax.set_xlim(0, 430)
    ax.set_ylim(-0.5, 3.8)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Elapsed seconds, one serial episode (NM 19.5, illustrative)', fontsize=13)
    ax.tick_params(labelsize=11)
    sp.save(fig, out / f'waterfall_step{step}.svg')
for k in range(1, 5):
    waterfall(k)
fs = {'pts': sorted(({'usd': r['usd_per_trial'], 'mean': 100 * r['mean5'], 'best': 100 * r['best5'], 'worst': 100 * r['worst5']} for r in ({'model': 'Inkling', 'release_date': '2026-07-15', 'mean5': 0.04118886471, 'best5': 0.1079, 'worst5': 0.0013, 'usd_per_trial': 9.15, 'hours_per_trial': 1.16}, {'model': 'Muse Spark 1.2', 'release_date': '2026-08-05', 'mean5': 0.1197025647, 'best5': 0.1836, 'worst5': 0.0684, 'usd_per_trial': 27.81, 'hours_per_trial': 4.59}, {'model': 'Qwen3.8-Max', 'release_date': '2026-08-02', 'mean5': 0.1580935059, 'best5': 0.2427, 'worst5': 0.087, 'usd_per_trial': 55.14, 'hours_per_trial': 18.48}, {'model': 'Qwen3.8-Max-0902', 'release_date': '2026-09-01', 'mean5': 0.1780964941, 'best5': 0.2678, 'worst5': 0.0807, 'usd_per_trial': 51.27, 'hours_per_trial': 19.36}, {'model': 'GLM-5.3 Flash', 'release_date': '2026-08-20', 'mean5': 0.1812520588, 'best5': 0.2845, 'worst5': 0.0861, 'usd_per_trial': 11.03, 'hours_per_trial': 16.39}, {'model': 'Gemini 3.8 Flash', 'release_date': '2026-09-02', 'mean5': 0.1962690294, 'best5': 0.3164, 'worst5': 0.0951, 'usd_per_trial': 38.75, 'hours_per_trial': 7.18}, {'model': 'Gemini 3.7 Flash', 'release_date': '2026-08-13', 'mean5': 0.2026059059, 'best5': 0.3123, 'worst5': 0.1028, 'usd_per_trial': 34.14, 'hours_per_trial': 7.87}, {'model': 'Grok 4.6', 'release_date': '2026-08-12', 'mean5': 0.2528619588, 'best5': 0.3729, 'worst5': 0.1286, 'usd_per_trial': 243.43, 'hours_per_trial': 13.8}, {'model': 'Kimi K3', 'release_date': '2026-07-16', 'mean5': 0.2586503206, 'best5': 0.3755, 'worst5': 0.1387, 'usd_per_trial': 109.71, 'hours_per_trial': 18.4}, {'model': 'Grok 4.7', 'release_date': '2026-09-21', 'mean5': 0.2950476118, 'best5': 0.4106, 'worst5': 0.1587, 'usd_per_trial': 318.77, 'hours_per_trial': 12.06}, {'model': 'GLM-5.3', 'release_date': '2026-08-14', 'mean5': 0.3018186412, 'best5': 0.4064, 'worst5': 0.1772, 'usd_per_trial': 97.22, 'hours_per_trial': 17.04}, {'model': 'GPT-5.6', 'release_date': '2026-07-09', 'mean5': 0.3220246706, 'best5': 0.4417, 'worst5': 0.222, 'usd_per_trial': 179.64, 'hours_per_trial': 8.6}, {'model': 'Claude Fable 5', 'release_date': '2026-06-09', 'mean5': 0.4696449353, 'best5': 0.6177, 'worst5': 0.339, 'usd_per_trial': 301.28, 'hours_per_trial': 15.82}, {'model': 'Claude Opus 5', 'release_date': '2026-07-24', 'mean5': 0.5201176765, 'best5': 0.6065, 'worst5': 0.3912, 'usd_per_trial': 196.71, 'hours_per_trial': 16.07}, {'model': 'Gemini 4 Argon', 'release_date': '2026-09-30', 'mean5': 0.5496118588, 'best5': 0.643, 'worst5': 0.4445, 'usd_per_trial': 129.36, 'hours_per_trial': 10.56}, {'model': 'Claude Fable 5.1', 'release_date': '2026-09-01', 'mean5': 0.5629079, 'best5': 0.6665, 'worst5': 0.4447, 'usd_per_trial': 138.55, 'hours_per_trial': 11.62}, {'model': 'Claude Sonnet 5.5', 'release_date': '2026-09-28', 'mean5': 0.6188929941, 'best5': 0.7169, 'worst5': 0.5093, 'usd_per_trial': 110.45, 'hours_per_trial': 16.01}, {'model': 'Claude Opus 5.5', 'release_date': '2026-09-22', 'mean5': 0.6233098824, 'best5': 0.7152, 'worst5': 0.5199, 'usd_per_trial': 98.87, 'hours_per_trial': 14.25}, {'model': 'GPT-6 Astra', 'release_date': '2026-09-03', 'mean5': 0.6550757456, 'best5': 0.73, 'worst5': 0.5511, 'usd_per_trial': 1029.65, 'hours_per_trial': 12.1})), key=lambda p: -p['mean'])}

def frontierswe():
    fig, ax = plt.subplots(figsize=(5.8, 2.75))
    for p in fs['pts']:
        ax.plot([p['usd'], p['usd']], [p['worst'], p['best']], color=C['memory_l'], lw=4, solid_capstyle='butt')
        ax.plot([p['usd']], [p['mean']], 'o', ms=7, color=C['memory'], zorder=4)
    top = fs['pts'][0]
    ax.annotate(f"top score: {top['usd']:,.0f} dollars per trial", (top['usd'], top['mean']), xytext=(-12, 22), textcoords='offset points', ha='right', fontsize=11.5, color=C['memory'], fontweight='bold', arrowprops=dict(arrowstyle='-', color=C['memory'], lw=1))
    ax.set_xscale('log')
    ax.set_xlim(6, 2000)
    ax.set_ylim(0, 80)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:g}'))
    ax.set_xlabel('Average cost of one trial (log scale)', fontsize=11)
    ax.set_ylabel('Score (%), worst to best', fontsize=11)
    ax.tick_params(labelsize=11)
    tidy(ax)
    sp.save(fig, out / 'frontierswe.svg')
frontierswe()
c6 = {'reached': [100000, 40000, 12000], 'accepted': [60000, 28000, 11400]}

class Svg:

    def __init__(self, w, h):
        self.w, self.h, self.items = (w, h, [])

    def add(self, s, show=True):
        self.items.append(s if show else f'<g opacity="0">{s}</g>')

    def text(self, x, y, s, size=19, color=INK, bold=False, anchor='middle', show=True):
        w = ' font-weight="bold"' if bold else ''
        self.add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{w}>{s}</text>', show)

    def rect(self, x, y, w, h, fill, stroke='none', sw=2, rx=4, show=True, dash=False):
        d = ' stroke-dasharray="7,5"' if dash else ''
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>', show)

    def line(self, x1, y1, x2, y2, color=GRY, width=2.4, show=True, dash=False):
        d = ' stroke-dasharray="7,5"' if dash else ''
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>', show)

    def save(self, name):
        (out / name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">\n<rect width="{self.w}" height="{self.h}" fill="white"/>\n' + '\n'.join(self.items) + '\n</svg>\n')

def cascade(step):
    s = Svg(900, 430)
    x0, wmax = (250, 260)
    tiers = [('Tier 1: small', '$0.0004 + $0.0010 gate'), ('Tier 2: mid', '$0.0020 + $0.0010 gate'), ('Tier 3: frontier', '$0.0600 + $0.0010 gate')]
    total = 100000
    for i, (name, price) in enumerate(tiers):
        y = 14 + i * 112
        reach, acc = (c6['reached'][i], c6['accepted'][i])
        s.text(8, y + 36, name, 24, INK, bold=True, anchor='start')
        s.text(8, y + 68, price, 20, '#334155', anchor='start')
        wr = wmax * reach / total
        wa = wmax * acc / total
        s.rect(x0, y + 10, wa, 66, P['ok'][0], P['ok'][1])
        s.rect(x0 + wa, y + 10, wr - wa, 66, P['fault'][0], P['fault'][1])
        if wa > 66:
            s.text(x0 + wa / 2, y + 51, f'{acc:,}', 21, P['ok'][2], bold=True)
        if wr - wa > 80:
            s.text(x0 + wa + (wr - wa) / 2, y + 51, f'{reach - acc:,}', 21, P['fault'][2], bold=True)
        s.text(x0 + wr + 10, y + 51, f'of {reach:,}', 20, '#334155', anchor='start')
        s.line(x0 - 8, y + 4, x0 - 8, y + 82, VIO, 5)
    s.text(8, 378, 'green: accepted and exits;  red: rejected and escalates', 20, '#334155', anchor='start')
    if step >= 2:
        s.rect(670, 14, 222, 320, '#f8fafc', '#cbd5e1', rx=6)
        s.text(781, 52, 'Per accepted', 22, INK, bold=True)
        s.text(781, 80, 'result', 22, INK, bold=True)
        s.text(781, 128, 'cascade', 21, VIO, bold=True)
        s.text(781, 160, '$0.00998', 25, VIO, bold=True)
        s.text(781, 212, 'direct frontier', 21, RED, bold=True)
        s.text(781, 244, '$0.06289', 25, RED, bold=True)
        s.text(781, 300, '6.3× apart', 23, INK, bold=True)
    if step >= 3:
        s.text(8, 416, 'Time: mean 1.72 s; through all tiers 6.75 s; direct 4.90 s', 21, RED, bold=True, anchor='start')
    s.save(f'cascade_step{step}.svg')
for k in (1, 2, 3):
    cascade(k)
ix = {'pts': [(4.0, 101.57, 50.01), (8.0, 81.41, 79.74), (16.0, 60.84, 119.76), (32.0, 44.73, 177.27), (64.0, 31.77, 252.74)]}

def inferencex():
    fig, ax = plt.subplots(figsize=COL)
    xs = [p[1] for p in ix['pts']]
    ys = [p[2] for p in ix['pts']]
    ax.plot(xs, ys, color=SKY, lw=3, marker='o', ms=10)
    for c, u, g in ix['pts']:
        ax.annotate(f'{c:.0f} users', (u, g), xytext=(10, -18), textcoords='offset points', fontsize=14, color=P['think'][2], fontweight='bold')
    ax.set_xlabel('Tokens/s each user sees', fontsize=15)
    ax.set_ylabel('Tokens/s per GPU', fontsize=15)
    ax.set_ylim(0, 290)
    ax.set_xlim(125, 15)
    ax.tick_params(labelsize=14)
    tidy(ax)
    sp.save(fig, out / 'inferencex.svg')
inferencex()
RHO = [x / 100 for x in range(5, 96)]

def kingman(step):
    fig, ax = plt.subplots(figsize=COL)
    ax.plot([u * 100 for u in RHO], [0.1578947368, 0.1914893617, 0.2258064516, 0.2608695652, 0.2967032967, 0.3333333333, 0.3707865169, 0.4090909091, 0.4482758621, 0.488372093, 0.5294117647, 0.5714285714, 0.6144578313, 0.6585365854, 0.7037037037, 0.75, 0.7974683544, 0.8461538462, 0.8961038961, 0.9473684211, 1.0, 1.054054054, 1.109589041, 1.166666667, 1.225352113, 1.285714286, 1.347826087, 1.411764706, 1.47761194, 1.545454545, 1.615384615, 1.6875, 1.761904762, 1.838709677, 1.918032787, 2.0, 2.084745763, 2.172413793, 2.263157895, 2.357142857, 2.454545455, 2.555555556, 2.660377358, 2.769230769, 2.882352941, 3.0, 3.12244898, 3.25, 3.382978723, 3.52173913, 3.666666667, 3.818181818, 3.976744186, 4.142857143, 4.317073171, 4.5, 4.692307692, 4.894736842, 5.108108108, 5.333333333, 5.571428571, 5.823529412, 6.090909091, 6.375, 6.677419355, 7.0, 7.344827586, 7.714285714, 8.111111111, 8.538461538, 9.0, 9.5, 10.04347826, 10.63636364, 11.28571429, 12.0, 12.78947368, 13.66666667, 14.64705882, 15.75, 17.0, 18.42857143, 20.07692308, 22.0, 24.27272727, 27.0, 30.33333333, 34.5, 39.85714286, 47.0, 57.0], color=VIO, lw=3)
    if step >= 2:
        for u, w, col in ((0.7, 7.0, SKY), (0.9, 27.0, RED)):
            ax.plot([u * 100], [w], 'o', ms=13, color=col, mec='white', mew=1.5, zorder=5)
            ax.annotate(f'{w:.0f} s', (u * 100, w), xytext=(-56, 4), textcoords='offset points', fontsize=20, fontweight='bold', color=col)
    ax.set_xlabel('Utilization ρ (percent)', fontsize=18)
    ax.set_ylabel('Mean wait (s)', fontsize=18)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.tick_params(labelsize=17)
    tidy(ax)
    sp.save(fig, out / f'kingman_step{step}.svg')
for k in (1, 2):
    kingman(k)

def breakeven():
    import numpy as np
    fig, ax = plt.subplots(figsize=COL)
    u = np.linspace(0, 1, 100)
    api = [0.0, 0.8378181818, 1.675636364, 2.513454545, 3.351272727, 4.189090909, 5.026909091, 5.864727273, 6.702545455, 7.540363636, 8.378181818, 9.216, 10.05381818, 10.89163636, 11.72945455, 12.56727273, 13.40509091, 14.24290909, 15.08072727, 15.91854545, 16.75636364, 17.59418182, 18.432, 19.26981818, 20.10763636, 20.94545455, 21.78327273, 22.62109091, 23.45890909, 24.29672727, 25.13454545, 25.97236364, 26.81018182, 27.648, 28.48581818, 29.32363636, 30.16145455, 30.99927273, 31.83709091, 32.67490909, 33.51272727, 34.35054545, 35.18836364, 36.02618182, 36.864, 37.70181818, 38.53963636, 39.37745455, 40.21527273, 41.05309091, 41.89090909, 42.72872727, 43.56654545, 44.40436364, 45.24218182, 46.08, 46.91781818, 47.75563636, 48.59345455, 49.43127273, 50.26909091, 51.10690909, 51.94472727, 52.78254545, 53.62036364, 54.45818182, 55.296, 56.13381818, 56.97163636, 57.80945455, 58.64727273, 59.48509091, 60.32290909, 61.16072727, 61.99854545, 62.83636364, 63.67418182, 64.512, 65.34981818, 66.18763636, 67.02545455, 67.86327273, 68.70109091, 69.53890909, 70.37672727, 71.21454545, 72.05236364, 72.89018182, 73.728, 74.56581818, 75.40363636, 76.24145455, 77.07927273, 77.91709091, 78.75490909, 79.59272727, 80.43054545, 81.26836364, 82.10618182, 82.944]
    ded = [28.0, 28.08727273, 28.17454545, 28.26181818, 28.34909091, 28.43636364, 28.52363636, 28.61090909, 28.69818182, 28.78545455, 28.87272727, 28.96, 29.04727273, 29.13454545, 29.22181818, 29.30909091, 29.39636364, 29.48363636, 29.57090909, 29.65818182, 29.74545455, 29.83272727, 29.92, 30.00727273, 30.09454545, 30.18181818, 30.26909091, 30.35636364, 30.44363636, 30.53090909, 30.61818182, 30.70545455, 30.79272727, 30.88, 30.96727273, 31.05454545, 31.14181818, 31.22909091, 31.31636364, 31.40363636, 31.49090909, 31.57818182, 31.66545455, 31.75272727, 31.84, 31.92727273, 32.01454545, 32.10181818, 32.18909091, 32.27636364, 32.36363636, 32.45090909, 32.53818182, 32.62545455, 32.71272727, 32.8, 32.88727273, 32.97454545, 33.06181818, 33.14909091, 33.23636364, 33.32363636, 33.41090909, 33.49818182, 33.58545455, 33.67272727, 33.76, 33.84727273, 33.93454545, 34.02181818, 34.10909091, 34.19636364, 34.28363636, 34.37090909, 34.45818182, 34.54545455, 34.63272727, 34.72, 34.80727273, 34.89454545, 34.98181818, 35.06909091, 35.15636364, 35.24363636, 35.33090909, 35.41818182, 35.50545455, 35.59272727, 35.68, 35.76727273, 35.85454545, 35.94181818, 36.02909091, 36.11636364, 36.20363636, 36.29090909, 36.37818182, 36.46545455, 36.55272727, 36.64]
    ax.plot(u * 100, api, color=AMB, lw=3)
    ax.plot(u * 100, ded, color=SKY, lw=3)
    ax.text(58, 76, 'per-token API', color=AMB, fontsize=15, fontweight='bold')
    ax.text(72, 24, 'dedicated', color=SKY, fontsize=15, fontweight='bold')
    us = 0.3768303187 * 100
    ax.axvline(us, color=GRY, ls='--', lw=1.6)
    ax.plot([us], [31.25581395], 'o', ms=12, color=VIO, mec='white', zorder=5)
    ax.text(us - 2, 76, 'u* = 37.7%', color=VIO, fontsize=15, fontweight='bold', ha='right')
    load = 0.52 * 100
    ax.plot([load], [43.13088], 'o', ms=10, color=AMB, mec='white', zorder=5)
    ax.plot([load], [32.4928], 'o', ms=10, color=SKY, mec='white', zorder=5)
    ax.text(load - 2, 50, '$43.13', ha='right', color=AMB, fontsize=14, fontweight='bold')
    ax.text(load + 2, 32.4928 - 10, '$32.49', color=SKY, fontsize=14, fontweight='bold')
    ax.set_xlabel('Delivered load (% of reference rate)', fontsize=15)
    ax.set_ylabel('Dollars per hour', fontsize=15)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 90)
    ax.tick_params(labelsize=14)
    tidy(ax)
    sp.save(fig, out / 'breakeven.svg')
breakeven()
b = {'table': [(None, 0.042, 0.0, 4.958), (None, 0.042, 3.7, 1.258), (None, 0.422, 2.5, 2.078), (None, 0.422, 2.5, 2.078), (None, 1.972, 0.0, 3.028)]}

def budget(step):
    """Root balances at the five boundaries of AE table 5: settled (gray),
    reserved (amber), free (green); every bar sums to the $5 budget."""
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    labs = ['1 Planning settled', '2 Children granted', '3 Trace child closed', '4 Fuzzing stopped', '5 Late bill settled']
    for i, (name, s_, r_, f_) in enumerate(b['table']):
        y = 4 - i
        show = i < step
        a = 1 if show else 0
        ax.barh(y, s_, height=0.66, color=GRY, alpha=a)
        ax.barh(y, r_, left=s_, height=0.66, color=AMB, alpha=a)
        ax.barh(y, f_, left=s_ + r_, height=0.66, color=P['ok'][0], edgecolor=GRN, lw=1.2, alpha=a)
        ax.text(-0.06, y, labs[i], ha='right', va='center', fontsize=14, fontweight='bold', color=C['text'])
        if show:
            if s_ > 0.5:
                ax.text(s_ / 2, y, money(s_, 3), ha='center', va='center', fontsize=13.5, color='white', fontweight='bold')
            if r_ > 0.4:
                ax.text(s_ + r_ / 2, y, money(r_), ha='center', va='center', fontsize=13.5, color='white', fontweight='bold')
            ax.text(s_ + r_ + f_ / 2, y, money(f_, 3), ha='center', va='center', fontsize=13.5, color=P['ok'][2], fontweight='bold')
    ax.text(0.0, 4.62, 'settled', color=GRY, fontsize=14, fontweight='bold')
    ax.text(1.0, 4.62, 'reserved', color=AMB, fontsize=14, fontweight='bold')
    ax.text(3.2, 4.62, 'free', color=GRN, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 5.0)
    ax.set_ylim(-0.5, 5.0)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Root budget of $5 (NM 19.10, illustrative)', fontsize=13)
    ax.tick_params(labelsize=13)
    sp.save(fig, out / f'budget_step{step}.svg')
for k in range(1, 6):
    budget(k)
a11 = {'d': {'Rule-based rewriter': {'per_auto': 0.009375, 'per_res': 25.53969957}, 'Single model call': {'per_auto': 0.06911764706, 'per_res': 11.61880165}, 'Bounded 3-turn loop': {'per_auto': 0.01674782821, 'per_res': 6.527444409}}}

def arch(step):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.95))
    names = list(a11['d'])
    short = ['Rewriter', 'Single call', f'{len([0.55, 0.42, 0.3])}-turn loop']
    cols = [GRY, AMB, VIO]
    for j, (ax, key, title, nd) in enumerate(((axes[0], 'per_auto', 'Automated spend per\nautomated acceptance', 3), (axes[1], 'per_res', 'Spend per resolved ticket,\nhuman fallback included', 2))):
        show = j == 0 or step >= 2
        vals = [a11['d'][n][key] for n in names]
        for i, v in enumerate(vals):
            y = 2 - i
            ax.barh(y, v if show else 0, height=0.6, color=cols[i])
            if show:
                ax.text(v + max(vals) * 0.03, y, money(v, nd), va='center', fontsize=13.5, fontweight='bold', color=cols[i])
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(short if j == 0 else ['', '', ''], fontsize=13.5)
        ax.set_xlim(0, max(vals) * 1.55)
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
        ax.set_title(title, fontsize=13.5, color=C['text'] if show else 'white', loc='left')
    sp.save(fig, out / f'arch_step{step}.svg')
for k in (1, 2):
    arch(k)

def thumb_recall():
    """Text-free cue for Lecture 18's token multiples (two bars, 4 and 15)."""
    s = Svg(240, 90)
    s.rect(20, 50, 52, 30, P['think'][1])
    s.rect(100, 10, 52, 70, VIO)
    s.line(10, 80, 230, 80, GRY, 2)
    s.save('thumb_recall.svg')
thumb_recall()
print('figures written to', out)
