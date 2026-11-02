#!/usr/bin/env python3
"""Lecture 13 (Agent Evaluation): every figure the deck draws.

Every value comes from calcs.py, which reads MLSysIM (and the deck's own run
in data/); this script imports it and prints the recorded values (val, num),
so figures and slides agree. Plots use the shared slideplot helper; diagrams
are SVG drawn at the size of their slot.

Run: python3 make_figs.py   (writes SVGs here; `make deck V=3 D=13_evaluation`
converts them to PDF).
"""
import mlsysim
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import slideplot as sp
from slideplot import plt
from mlsysim.physics import calc_independent_success, calc_unchecked_success
BLUE, BLUE_L = ('#0284c7', '#e0f2fe')
GREEN, GREEN_L = ('#16a34a', '#dcfce7')
AMBER, AMBER_L = ('#d97706', '#fef3c7')
PURPLE, PURPLE_L = ('#7c3aed', '#f3e8ff')
RED, RED_L = ('#dc2626', '#fee2e2')
GRAY, GRAY_L = ('#64748b', '#f1f5f9')
INK = '#0f172a'
FONT = 'Helvetica, Arial, sans-serif'

def axis_type(ax, size):
    """Set tick and axis-label type for a plot drawn at its slot size."""
    ax.tick_params(labelsize=size)
    ax.xaxis.label.set_size(size)
    ax.yaxis.label.set_size(size)

def svg(path, w, h, body):
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">\n<defs>' + ''.join((f'<marker id="a{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="{c}"/></marker>' for n, c in [('g', GRAY), ('p', PURPLE), ('a', AMBER), ('r', RED), ('b', BLUE)])) + f'</defs>\n<rect width="{w}" height="{h}" fill="white"/>\n' + body + '\n</svg>\n')

def box(x, y, w, h, fill, stroke, rx=10, dash=False, sw=2.5):
    d = ' stroke-dasharray="8 6"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

def text(x, y, s, size=22, color=INK, anchor='middle', weight='normal'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'

def arrow(x1, y1, x2, y2, m='g', color=GRAY, dash=False, sw=2.5):
    d = ' stroke-dasharray="7 5"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#a{m})"/>'

def hook():
    """pass^k against pass@k at 90 percent per run (NM 13.2)."""
    p, k = (0.9, 8)
    ks = list(range(1, k + 1))
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.6, 2.9))
        ax.plot(ks, [100 * calc_independent_success(p, j) for j in ks], color=BLUE, lw=3, alpha=1 if step == 2 else 0)
        ax.plot(ks, [100 * calc_unchecked_success(p, j) for j in ks], color=PURPLE, lw=3, alpha=1 if step == 2 else 0)
        ax.plot([1], [90.0], 'o', color=INK, ms=9)
        ax.annotate('one run: 90%', (1, 90.0), xytext=(1.6, 50), fontsize=13, color=INK, arrowprops=dict(arrowstyle='-', color=INK, lw=1))
        if step == 1:
            ax.text(k, 50, 'all 8 pass?', fontsize=15, color=PURPLE, ha='right', fontweight='bold')
        else:
            pow_k = 100 * 0.43046721
            ax.plot([k], [pow_k], 'o', color=PURPLE, ms=10)
            ax.text(7.8, pow_k - 12, 'pass$^8$ = 43%', fontsize=14, color=PURPLE, ha='right', fontweight='bold')
            ax.text(7.8, 104, 'pass@8: at least one of 8', fontsize=13, color=BLUE, ha='right')
            ax.text(4.6, 74, 'all of k pass', fontsize=13, color=PURPLE)
        ax.set_xlim(0.7, 8.4)
        ax.set_ylim(0, 112)
        ax.set_xticks(ks)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Runs of the same task, k')
        ax.set_ylabel('Probability (%)')
        ax.grid(axis='y', color=sp.C['grid'], lw=0.6)
        sp.save(fig, HERE / f'hook_step{step}.svg')

def scaffold():
    """Terminal-Bench 2.0: same model, many scaffolds (MLSysIM)."""
    by = {}
    for r in ({'model': 'gemini-2.5-flash', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.1707865169, 'accuracy_se': 0.02491558276, 'release_date': '2025-06-17'}, {'model': 'gemini-2.5-pro', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.2606741573, 'accuracy_se': 0.0254928543, 'release_date': '2025-06-17'}, {'model': 'gemini-2.5-pro', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.3264044944, 'accuracy_se': 0.03021576243, 'release_date': '2025-06-17'}, {'model': 'gemini-2.5-pro', 'harness': 'Gemini CLI', 'harness_org': 'Google', 'accuracy': 0.195505618, 'accuracy_se': 0.0285868312, 'release_date': '2025-06-17'}, {'model': 'gemini-2.5-flash', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.1685393258, 'accuracy_se': 0.02371892814, 'release_date': '2025-06-17'}, {'model': 'gemini-2.5-pro', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.1640449438, 'accuracy_se': 0.0275060586, 'release_date': '2025-06-17'}, {'model': 'gemini-2.5-flash', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.1640449438, 'accuracy_se': 0.02392252782, 'release_date': '2025-06-17'}, {'model': 'gemini-2.5-flash', 'harness': 'Gemini CLI', 'harness_org': 'Google', 'accuracy': 0.154494382, 'accuracy_se': 0.02315852543, 'release_date': '2025-06-17'}, {'model': 'grok-4-0709', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.2719101124, 'accuracy_se': 0.0305152322, 'release_date': '2025-07-09'}, {'model': 'grok-4-0709', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.2539325843, 'accuracy_se': 0.02921611051, 'release_date': '2025-07-09'}, {'model': 'grok-4-0709', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.2314606742, 'accuracy_se': 0.02904963559, 'release_date': '2025-07-09'}, {'model': 'Kimi-K2-Instruct', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.2779026217, 'accuracy_se': 0.02521655844, 'release_date': '2025-07-12'}, {'model': 'Kimi-K2-Instruct', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.2674157303, 'accuracy_se': 0.02697190953, 'release_date': '2025-07-12'}, {'model': 'Qwen3-Coder-480B-A35B-Instruct', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.2539325843, 'accuracy_se': 0.02587054679, 'release_date': '2025-07-31'}, {'model': 'Qwen3-Coder-480B-A35B-Instruct', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.2389513109, 'accuracy_se': 0.02801079902, 'release_date': '2025-07-31'}, {'model': 'Qwen3-Coder-480B-A35B-Instruct', 'harness': 'Dakou Agent', 'harness_org': 'iflow', 'accuracy': 0.2719101124, 'accuracy_se': 0.02568239486, 'release_date': '2025-07-31'}, {'model': 'gpt-oss-120b', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.1415730337, 'accuracy_se': 0.02267352256, 'release_date': '2025-08-05'}, {'model': 'gpt-oss-20b', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.03370786517, 'accuracy_se': 0.01357556342, 'release_date': '2025-08-05'}, {'model': 'claude-opus-4-1-20250805', 'harness': 'Claude Code', 'harness_org': 'Anthropic', 'accuracy': 0.3483146067, 'accuracy_se': 0.02899393154, 'release_date': '2025-08-05'}, {'model': 'claude-opus-4-1-20250805', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.3505617978, 'accuracy_se': 0.02510948127, 'release_date': '2025-08-05'}, {'model': 'gpt-oss-120b', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.1868913858, 'accuracy_se': 0.02659972661, 'release_date': '2025-08-05'}, {'model': 'gpt-oss-20b', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.03071161049, 'accuracy_se': 0.01518681472, 'release_date': '2025-08-05'}, {'model': 'claude-opus-4-1-20250805', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.3685393258, 'accuracy_se': 0.02697190953, 'release_date': '2025-08-05'}, {'model': 'claude-opus-4-1-20250805', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.3797752809, 'accuracy_se': 0.026242804, 'release_date': '2025-08-05'}, {'model': 'gpt-5-mini-2025-08-07', 'harness': 'spoox-o-m', 'harness_org': 'TUM', 'accuracy': 0.3483146067, 'accuracy_se': 0.02715112685, 'release_date': '2025-08-07'}, {'model': 'gpt-5-nano-2025-08-07', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.06966292135, 'accuracy_se': 0.01919874591, 'release_date': '2025-08-07'}, {'model': 'gpt-5-mini-2025-08-07', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.2921348315, 'accuracy_se': 0.0275060586, 'release_date': '2025-08-07'}, {'model': 'gpt-5-2025-08-07', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.4382022472, 'accuracy_se': 0.02987273431, 'release_date': '2025-08-07'}, {'model': 'gpt-5-mini-2025-08-07', 'harness': 'Codex CLI', 'harness_org': 'OpenAI', 'accuracy': 0.3185393258, 'accuracy_se': 0.02998751301, 'release_date': '2025-08-07'}, {'model': 'gpt-5-nano-2025-08-07', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.07865168539, 'accuracy_se': 0.01868668707, 'release_date': '2025-08-07'}, {'model': 'gpt-5-mini-2025-08-07', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.2404494382, 'accuracy_se': 0.0253018939, 'release_date': '2025-08-07'}, {'model': 'gpt-5-2025-08-07', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.3393258427, 'accuracy_se': 0.02904963559, 'release_date': '2025-08-07'}, {'model': 'gpt-5-mini-2025-08-07', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.2224719101, 'accuracy_se': 0.026242804, 'release_date': '2025-08-07'}, {'model': 'gpt-5-2025-08-07', 'harness': 'Codex CLI', 'harness_org': 'OpenAI', 'accuracy': 0.4960674157, 'accuracy_se': 0.02897301491, 'release_date': '2025-08-07'}, {'model': 'gpt-5-nano-2025-08-07', 'harness': 'spoox-o-m', 'harness_org': 'TUM', 'accuracy': 0.2179775281, 'accuracy_se': 0.02803003061, 'release_date': '2025-08-07'}, {'model': 'gpt-5-nano-2025-08-07', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.09887640449, 'accuracy_se': 0.021123213, 'release_date': '2025-08-07'}, {'model': 'gpt-5-2025-08-07', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.3516853933, 'accuracy_se': 0.03055493975, 'release_date': '2025-08-07'}, {'model': 'gpt-5-nano-2025-08-07', 'harness': 'Codex CLI', 'harness_org': 'OpenAI', 'accuracy': 0.1146067416, 'accuracy_se': 0.02267352256, 'release_date': '2025-08-07'}, {'model': 'grok-code-fast-1', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.1415730337, 'accuracy_se': 0.0254928543, 'release_date': '2025-08-28'}, {'model': 'grok-code-fast-1', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.2584269663, 'accuracy_se': 0.026242804, 'release_date': '2025-08-28'}, {'model': 'gpt-5-codex', 'harness': 'Codex CLI', 'harness_org': 'OpenAI', 'accuracy': 0.443258427, 'accuracy_se': 0.02705419817, 'release_date': '2025-09-15'}, {'model': 'gpt-5-codex', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.4134831461, 'accuracy_se': 0.02768181795, 'release_date': '2025-09-15'}, {'model': 'gpt-5-codex', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.4342696629, 'accuracy_se': 0.02873489569, 'release_date': '2025-09-15'}, {'model': 'claude-sonnet-4-5-20250929', 'harness': 'Claude Code', 'harness_org': 'Anthropic', 'accuracy': 0.4005617978, 'accuracy_se': 0.02915379382, 'release_date': '2025-09-29'}, {'model': 'claude-sonnet-4-5-20250929', 'harness': 'Goose', 'harness_org': 'Block', 'accuracy': 0.4314606742, 'accuracy_se': 0.02587054679, 'release_date': '2025-09-29'}, {'model': 'claude-sonnet-4-5-20250929', 'harness': 'MAYA-V2', 'harness_org': 'ADYA', 'accuracy': 0.4269662921, 'accuracy_se': None, 'release_date': '2025-09-29'}, {'model': 'claude-sonnet-4-5-20250929', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.4280898876, 'accuracy_se': 0.02788547043, 'release_date': '2025-09-29'}, {'model': 'claude-sonnet-4-5-20250929', 'harness': 'CAMEL-AI', 'harness_org': 'CAMEL-AI', 'accuracy': 0.4651685393, 'accuracy_se': 0.02351356559, 'release_date': '2025-09-29'}, {'model': 'claude-sonnet-4-5-20250929', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.4252808989, 'accuracy_se': 0.02805165038, 'release_date': '2025-09-29'}, {'model': 'claude-sonnet-4-5-20250929', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.4258426966, 'accuracy_se': 0.0277110028, 'release_date': '2025-09-29'}, {'model': 'glm-4.6', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.2453183521, 'accuracy_se': 0.02444615624, 'release_date': '2025-09-30'}, {'model': 'claude-haiku-4-5-20251001', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.1393258427, 'accuracy_se': 0.02732916893, 'release_date': '2025-10-15'}, {'model': 'claude-haiku-4-5-20251001', 'harness': 'Goose', 'harness_org': 'Block', 'accuracy': 0.3550561798, 'accuracy_se': 0.02888220114, 'release_date': '2025-10-15'}, {'model': 'claude-haiku-4-5-20251001', 'harness': 'Claude Code', 'harness_org': 'Anthropic', 'accuracy': 0.2752808989, 'accuracy_se': 0.02775472252, 'release_date': '2025-10-15'}, {'model': 'claude-haiku-4-5-20251001', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.2831460674, 'accuracy_se': 0.02875598553, 'release_date': '2025-10-15'}, {'model': 'claude-haiku-4-5-20251001', 'harness': 'Mini-SWE-Agent', 'harness_org': 'Princeton', 'accuracy': 0.2983146067, 'accuracy_se': 0.02513361341, 'release_date': '2025-10-15'}, {'model': 'MiniMax-M2', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.3003745318, 'accuracy_se': 0.0274079275, 'release_date': '2025-10-27'}, {'model': 'gpt-5-2025-08-07_medium', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.352, 'accuracy_se': 0.031, 'release_date': '2025-08-07'}, {'model': 'gpt-5-nano-2025-08-07_medium', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.079, 'accuracy_se': 0.019, 'release_date': '2025-08-07'}, {'model': 'grok-4-0709', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.231, 'accuracy_se': 0.029, 'release_date': '2025-07-09'}, {'model': 'claude-opus-4-1-20250805', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.38, 'accuracy_se': 0.026, 'release_date': '2025-08-05'}, {'model': 'claude-sonnet-4-5-20250929', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.428, 'accuracy_se': 0.028, 'release_date': '2025-09-29'}, {'model': 'gemini-2.5-pro', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.326, 'accuracy_se': 0.03, 'release_date': '2025-06-17'}, {'model': 'gpt-5-mini-2025-08-07_medium', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.24, 'accuracy_se': 0.025, 'release_date': '2025-08-07'}, {'model': 'grok-code-fast-1', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.142, 'accuracy_se': 0.025, 'release_date': '2025-08-28'}, {'model': 'claude-haiku-4-5-20251001', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.283, 'accuracy_se': 0.029, 'release_date': '2025-10-15'}, {'model': 'gpt-5-codex', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.434, 'accuracy_se': 0.029, 'release_date': '2025-09-15'}, {'model': 'gemini-2.5-flash-preview-09-2025', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.169, 'accuracy_se': 0.024, 'release_date': '2025-09-25'}, {'model': None, 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.3, 'accuracy_se': 0.027, 'release_date': None}, {'model': 'glm-4.6', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.245, 'accuracy_se': 0.024, 'release_date': '2025-09-30'}, {'model': 'Qwen3-Coder-480B-A35B-Instruct', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.239, 'accuracy_se': 0.028, 'release_date': '2025-07-31'}, {'model': 'gpt-oss-120b', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.187, 'accuracy_se': 0.027, 'release_date': '2025-08-05'}, {'model': 'Kimi-K2-Instruct', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.278, 'accuracy_se': 0.025, 'release_date': '2025-07-12'}, {'model': 'gpt-oss-20b', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.031, 'accuracy_se': 0.015, 'release_date': '2025-08-05'}, {'model': 'claude-opus-4-1-20250805', 'harness': ' OpenHands      ', 'harness_org': ' OpenHands ', 'accuracy': 0.369, 'accuracy_se': 0.027, 'release_date': '2025-08-05'}, {'model': 'gpt-5-2025-08-07_medium', 'harness': ' OpenHands      ', 'harness_org': ' OpenHands ', 'accuracy': 0.438, 'accuracy_se': 0.03, 'release_date': '2025-08-07'}, {'model': 'gpt-5-nano-2025-08-07_medium', 'harness': ' OpenHands      ', 'harness_org': ' OpenHands ', 'accuracy': 0.099, 'accuracy_se': 0.021, 'release_date': '2025-08-07'}, {'model': 'gemini-2.5-flash-preview-09-2025', 'harness': ' OpenHands      ', 'harness_org': ' OpenHands ', 'accuracy': 0.164, 'accuracy_se': 0.024, 'release_date': '2025-09-25'}, {'model': 'Kimi-K2-Instruct', 'harness': ' OpenHands      ', 'harness_org': ' OpenHands ', 'accuracy': 0.267, 'accuracy_se': 0.027, 'release_date': '2025-07-12'}, {'model': 'Qwen3-Coder-480B-A35B-Instruct', 'harness': ' OpenHands      ', 'harness_org': ' OpenHands ', 'accuracy': 0.254, 'accuracy_se': 0.026, 'release_date': '2025-07-31'}, {'model': 'gpt-5-mini-2025-08-07_medium', 'harness': ' OpenHands      ', 'harness_org': ' OpenHands ', 'accuracy': 0.292, 'accuracy_se': 0.028, 'release_date': '2025-08-07'}, {'model': 'grok-4-0709', 'harness': ' OpenHands      ', 'harness_org': ' OpenHands ', 'accuracy': 0.272, 'accuracy_se': 0.031, 'release_date': '2025-07-09'}, {'model': 'claude-sonnet-4-5-20250929', 'harness': ' OpenHands      ', 'harness_org': ' OpenHands ', 'accuracy': 0.426, 'accuracy_se': 0.028, 'release_date': '2025-09-29'}, {'model': 'gemini-2.5-pro', 'harness': ' OpenHands      ', 'harness_org': ' OpenHands ', 'accuracy': 0.164, 'accuracy_se': 0.028, 'release_date': '2025-06-17'}, {'model': 'claude-haiku-4-5-20251001', 'harness': ' OpenHands      ', 'harness_org': ' OpenHands ', 'accuracy': 0.139, 'accuracy_se': 0.027, 'release_date': '2025-10-15'}, {'model': 'gpt-5-nano-2025-08-07_medium', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.07, 'accuracy_se': 0.019, 'release_date': '2025-08-07'}, {'model': 'gpt-oss-20b', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.034, 'accuracy_se': 0.014, 'release_date': '2025-08-05'}, {'model': 'gemini-2.5-pro', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.261, 'accuracy_se': 0.025, 'release_date': '2025-06-17'}, {'model': 'gemini-2.5-flash-preview-09-2025', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.171, 'accuracy_se': 0.025, 'release_date': '2025-09-25'}, {'model': 'gpt-5-mini-2025-08-07_medium', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.222, 'accuracy_se': 0.026, 'release_date': '2025-08-07'}, {'model': 'gpt-5-2025-08-07_medium', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.339, 'accuracy_se': 0.029, 'release_date': '2025-08-07'}, {'model': 'claude-haiku-4-5-20251001', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.298, 'accuracy_se': 0.025, 'release_date': '2025-10-15'}, {'model': 'grok-4-0709', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.254, 'accuracy_se': 0.029, 'release_date': '2025-07-09'}, {'model': 'grok-code-fast-1', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.258, 'accuracy_se': 0.026, 'release_date': '2025-08-28'}, {'model': 'gpt-oss-120b', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.142, 'accuracy_se': 0.023, 'release_date': '2025-08-05'}, {'model': 'claude-opus-4-1-20250805', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.351, 'accuracy_se': 0.025, 'release_date': '2025-08-05'}, {'model': 'gpt-5-codex', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.413, 'accuracy_se': 0.028, 'release_date': '2025-09-15'}, {'model': 'claude-sonnet-4-5-20250929', 'harness': ' Mini-SWE-Agent ', 'harness_org': ' Princeton ', 'accuracy': 0.425, 'accuracy_se': 0.028, 'release_date': '2025-09-29'}, {'model': 'claude-opus-4-1-20250805', 'harness': ' Claude Code    ', 'harness_org': ' Anthropic ', 'accuracy': 0.348, 'accuracy_se': 0.029, 'release_date': '2025-08-05'}, {'model': 'gpt-5-nano-2025-08-07_medium', 'harness': ' Codex CLI      ', 'harness_org': ' OpenAI    ', 'accuracy': 0.115, 'accuracy_se': 0.023, 'release_date': '2025-08-07'}, {'model': 'claude-sonnet-4-5-20250929', 'harness': ' Claude Code    ', 'harness_org': ' Anthropic ', 'accuracy': 0.401, 'accuracy_se': 0.029, 'release_date': '2025-09-29'}, {'model': 'gpt-5-2025-08-07_medium', 'harness': ' Codex CLI      ', 'harness_org': ' OpenAI    ', 'accuracy': 0.496, 'accuracy_se': 0.029, 'release_date': '2025-08-07'}, {'model': 'gemini-2.5-pro', 'harness': ' Gemini CLI     ', 'harness_org': ' Google    ', 'accuracy': 0.196, 'accuracy_se': 0.029, 'release_date': '2025-06-17'}, {'model': 'claude-haiku-4-5-20251001', 'harness': ' Claude Code    ', 'harness_org': ' Anthropic ', 'accuracy': 0.275, 'accuracy_se': 0.028, 'release_date': '2025-10-15'}, {'model': 'gemini-2.5-flash-preview-09-2025', 'harness': ' Gemini CLI     ', 'harness_org': ' Google    ', 'accuracy': 0.154, 'accuracy_se': 0.023, 'release_date': '2025-09-25'}, {'model': 'gpt-5-codex', 'harness': ' Codex CLI      ', 'harness_org': ' OpenAI    ', 'accuracy': 0.443, 'accuracy_se': 0.027, 'release_date': '2025-09-15'}, {'model': 'gpt-5-mini-2025-08-07_medium', 'harness': ' Codex CLI      ', 'harness_org': ' OpenAI    ', 'accuracy': 0.319, 'accuracy_se': 0.03, 'release_date': '2025-08-07'}, {'model': 'kimi-k2-thinking', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.3573033708, 'accuracy_se': 0.02820252473, 'release_date': '2025-11-06'}, {'model': 'kimi-k2-thinking', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.357, 'accuracy_se': 0.028, 'release_date': '2025-11-06'}, {'model': 'gpt-5.1-2025-11-13', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.4758426966, 'accuracy_se': 0.02761603928, 'release_date': '2025-11-13'}, {'model': 'gpt-5.1-codex', 'harness': 'Crux', 'harness_org': 'Roam', 'accuracy': 0.5775280899, 'accuracy_se': 0.02938164221, 'release_date': '2025-11-12'}, {'model': 'gpt-5.1-codex-mini', 'harness': 'Crux', 'harness_org': 'Roam', 'accuracy': 0.4314606742, 'accuracy_se': 0.03003464733, 'release_date': '2025-11-12'}, {'model': 'gpt-5.1-codex', 'harness': 'Letta Code', 'harness_org': 'Letta', 'accuracy': 0.5348314607, 'accuracy_se': 0.02837397023, 'release_date': '2025-11-12'}, {'model': 'gpt-5.1-codex-mini', 'harness': 'hookele', 'harness_org': 'Dmitry Barakhov', 'accuracy': 0.6157303371, 'accuracy_se': 0.01894444666, 'release_date': '2025-11-12'}, {'model': 'gpt-5.1-codex', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.3685393258, 'accuracy_se': 0.03236629143, 'release_date': '2025-11-12'}, {'model': 'gpt-5.1-codex', 'harness': 'Codex CLI', 'harness_org': ' OpenAI      ', 'accuracy': 0.578, 'accuracy_se': 0.029, 'release_date': '2025-11-12'}, {'model': 'gpt-5.1-2025-11-13_medium', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.476, 'accuracy_se': 0.028, 'release_date': '2025-11-13'}, {'model': 'gpt-5.1-2025-11-13_medium', 'harness': 'Terminus 2', 'harness_org': 'Stanford', 'accuracy': 0.476, 'accuracy_se': 0.028, 'release_date': '2025-11-13'}, {'model': 'gpt-5.1-codex', 'harness': ' Terminus 2     ', 'harness_org': ' Stanford  ', 'accuracy': 0.369, 'accuracy_se': 0.032, 'release_date': '2025-11-12'}, {'model': 'gpt-5.1-codex-mini', 'harness': ' Codex CLI      ', 'harness_org': ' OpenAI    ', 'accuracy': 0.431, 'accuracy_se': 0.03, 'release_date': '2025-11-12'}, {'model': 'gemini-3-pro-preview', 'harness': 'Droid', 'harness_org': 'Factory', 'accuracy': 0.6112359551, 'accuracy_se': 0.02785646838, 'release_date': '2025-11-18'}, {'model': 'gemini-3-pro-preview', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.5685393258, 'accuracy_se': 0.02472016341, 'release_date': '2025-11-18'}, {'model': 'gemini-3-pro-preview', 'harness': 'CodeBrain-1.5', 'harness_org': 'Feeling AI', 'accuracy': 0.6224719101, 'accuracy_se': 0.02605734017, 'release_date': '2025-11-18'}, {'model': 'gemini-3-pro-preview', 'harness': 'SageAgent', 'harness_org': 'OpenSage', 'accuracy': 0.6516853933, 'accuracy_se': 0.02065890986, 'release_date': '2025-11-18'}, {'model': 'gemini-3-pro-preview', 'harness': 'Ante', 'harness_org': 'Antigma Labs', 'accuracy': 0.6942571785, 'accuracy_se': 0.02088661871, 'release_date': '2025-11-18'}, {'model': 'gemini-3-pro-preview', 'harness': 'II-Agent', 'harness_org': 'Intelligent Internet', 'accuracy': 0.6179775281, 'accuracy_se': 0.02820252473, 'release_date': '2025-11-18'}, {'model': 'gemini-3-pro-preview', 'harness': 'Letta Code', 'harness_org': 'Letta', 'accuracy': 0.5595505618, 'accuracy_se': 0.0297099389, 'release_date': '2025-11-18'}, {'model': 'gpt-5.1-codex-max', 'harness': 'Codex CLI', 'harness_org': 'OpenAI', 'accuracy': 0.604494382, 'accuracy_se': 0.02660985405, 'release_date': '2025-11-19'}, {'model': 'claude-opus-4-5-20251101', 'harness': 'Letta Code', 'harness_org': 'Letta', 'accuracy': 0.591011236, 'accuracy_se': 0.02412440927, 'release_date': '2025-11-24'}, {'model': 'claude-opus-4-5-20251101', 'harness': 'Mux', 'harness_org': 'Coder', 'accuracy': 0.5842696629, 'accuracy_se': None, 'release_date': '2025-11-24'}, {'model': 'claude-opus-4-5-20251101', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.5775280899, 'accuracy_se': 0.0253018939, 'release_date': '2025-11-24'}, {'model': 'claude-opus-4-5-20251101', 'harness': 'Goose', 'harness_org': 'Block', 'accuracy': 0.543258427, 'accuracy_se': 0.02626589491, 'release_date': '2025-11-24'}, {'model': 'claude-opus-4-5-20251101', 'harness': 'Claude Code', 'harness_org': 'Anthropic', 'accuracy': 0.5213483146, 'accuracy_se': 0.0253018939, 'release_date': '2025-11-24'}, {'model': 'claude-opus-4-5-20251101', 'harness': 'OpenCode', 'harness_org': 'Anomaly Innovations', 'accuracy': 0.5168539326, 'accuracy_se': None, 'release_date': '2025-11-24'}, {'model': 'claude-opus-4-5-20251101', 'harness': 'OpenHands', 'harness_org': 'OpenHands', 'accuracy': 0.5191011236, 'accuracy_se': 0.02888220114, 'release_date': '2025-11-24'}, {'model': 'deepseek/deepseek-v3.2', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.395505618, 'accuracy_se': 0.0275060586, 'release_date': '2025-12-01'}, {'model': 'gpt-5.2-2025-12-11', 'harness': 'Codex CLI', 'harness_org': 'OpenAI', 'accuracy': 0.6292134831, 'accuracy_se': 0.03003464733, 'release_date': '2025-12-11'}, {'model': 'gpt-5.2-2025-12-11', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.5404494382, 'accuracy_se': 0.02891017422, 'release_date': '2025-12-11'}, {'model': 'gpt-5.2-2025-12-11', 'harness': 'Droid', 'harness_org': 'Factory', 'accuracy': 0.6494382022, 'accuracy_se': 0.02768181795, 'release_date': '2025-12-11'}, {'model': 'claude-opus-4-5-20251101', 'harness': 'Droid', 'harness_org': 'Factory', 'accuracy': 0.631, 'accuracy_se': 0.027, 'release_date': '2025-11-24'}, {'model': 'gpt-5.2-2025-12-11', 'harness': 'Mux', 'harness_org': 'Coder', 'accuracy': 0.606741573, 'accuracy_se': None, 'release_date': '2025-12-11'}, {'model': None, 'harness': 'Warp', 'harness_org': 'Warp', 'accuracy': 0.612, 'accuracy_se': 0.03, 'release_date': None}, {'model': 'gemini-3-flash-preview', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.5168539326, 'accuracy_se': 0.03129981347, 'release_date': '2025-12-17'}, {'model': 'gemini-3-flash-preview', 'harness': 'Gemini CLI', 'harness_org': 'Google', 'accuracy': 0.4741573034, 'accuracy_se': 0.03035588264, 'release_date': '2025-12-17'}, {'model': 'claude-opus-4-5-20251101_128K', 'harness': 'Letta Code', 'harness_org': 'Letta', 'accuracy': 0.591, 'accuracy_se': 0.024, 'release_date': '2025-11-24'}, {'model': 'gemini-3-flash-preview', 'harness': 'Junie CLI', 'harness_org': 'JetBrains', 'accuracy': 0.6426966292, 'accuracy_se': 0.02785646838, 'release_date': '2025-12-17'}, {'model': 'gpt-5.2-2025-12-11_medium', 'harness': 'Codex CLI', 'harness_org': 'OpenAI', 'accuracy': 0.629, 'accuracy_se': 0.03, 'release_date': '2025-12-11'}, {'model': 'glm-4.7', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.3338951311, 'accuracy_se': 0.02841429946, 'release_date': '2025-12-22'}, {'model': 'glm-4.7', 'harness': 'Crux', 'harness_org': 'Roam', 'accuracy': 0.3325842697, 'accuracy_se': 0.02452318686, 'release_date': '2025-12-22'}, {'model': 'MiniMax-M2.1', 'harness': 'Crux', 'harness_org': 'Roam', 'accuracy': 0.3662921348, 'accuracy_se': 0.02888220114, 'release_date': '2025-12-23'}, {'model': 'MiniMax-M2.1', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.2921348315, 'accuracy_se': 0.02904963559, 'release_date': '2025-12-23'}, {'model': 'gemini-3-flash-preview', 'harness': ' Gemini CLI     ', 'harness_org': 'Google', 'accuracy': 0.51, 'accuracy_se': 0.03, 'release_date': '2025-12-17'}, {'model': 'gpt-5.2-2025-12-11_medium', 'harness': 'Droid', 'harness_org': 'Factory', 'accuracy': 0.649, 'accuracy_se': 0.028, 'release_date': '2025-12-11'}, {'model': 'gpt-5.2-codex', 'harness': 'Deep Agents', 'harness_org': 'LangChain', 'accuracy': 0.6651685393, 'accuracy_se': 0.03062100456, 'release_date': '2025-12-18'}, {'model': 'kimi-k2.5', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.4320224719, 'accuracy_se': 0.02877705991, 'release_date': '2026-01-27'}, {'model': 'claude-opus-4-6', 'harness': 'MAYA-V2', 'harness_org': 'ADYA', 'accuracy': 0.7207865169, 'accuracy_se': 0.02208661072, 'release_date': '2026-02-05'}, {'model': 'claude-opus-4-6', 'harness': 'TongAgents', 'harness_org': 'Bigai', 'accuracy': 0.7191011236, 'accuracy_se': 0.02679149339, 'release_date': '2026-02-05'}, {'model': 'gpt-5.3-codex', 'harness': 'spoox-o-m', 'harness_org': 'TUM', 'accuracy': 0.7146067416, 'accuracy_se': 0.02491558276, 'release_date': '2026-02-05'}, {'model': 'claude-opus-4-6', 'harness': 'Droid', 'harness_org': 'Factory', 'accuracy': 0.6988764045, 'accuracy_se': 0.02472016341, 'release_date': '2026-02-05'}, {'model': 'gpt-5.3-codex', 'harness': 'SageAgent', 'harness_org': 'OpenSage', 'accuracy': 0.7842696629, 'accuracy_se': 0.02202247191, 'release_date': '2026-02-05'}, {'model': 'gpt-5.3-codex', 'harness': 'IndusAGI Coding Agent', 'harness_org': 'Varun Israni (SoloVpx)', 'accuracy': 0.6913857678, 'accuracy_se': 0.02279204674, 'release_date': '2026-02-05'}, {'model': 'claude-opus-4-6', 'harness': 'Crux', 'harness_org': 'Roam', 'accuracy': 0.6687265918, 'accuracy_se': None, 'release_date': '2026-02-05'}, {'model': 'gpt-5.3-codex', 'harness': 'CodeBrain-1.5', 'harness_org': 'Feeling AI', 'accuracy': 0.7584269663, 'accuracy_se': 0.02030371588, 'release_date': '2026-02-05'}, {'model': 'claude-opus-4-6', 'harness': 'Capy', 'harness_org': 'Capy', 'accuracy': 0.7528089888, 'accuracy_se': 0.02392252782, 'release_date': '2026-02-05'}, {'model': 'gpt-5.3-codex', 'harness': 'Simple Codex', 'harness_org': 'OpenAI', 'accuracy': 0.7505617978, 'accuracy_se': 0.02392252782, 'release_date': '2026-02-05'}, {'model': 'gpt-5.3-codex', 'harness': 'Droid', 'harness_org': 'Factory', 'accuracy': 0.7730337079, 'accuracy_se': 0.02157752762, 'release_date': '2026-02-05'}, {'model': 'gpt-5.3-codex', 'harness': 'Codelia', 'harness_org': 'kousw', 'accuracy': 0.7573033708, 'accuracy_se': 0.02157752762, 'release_date': '2026-02-05'}, {'model': 'gpt-5.3-codex', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.6471910112, 'accuracy_se': 0.02715112685, 'release_date': '2026-02-05'}, {'model': 'gpt-5.3-codex', 'harness': 'Mux', 'harness_org': 'Coder', 'accuracy': 0.7460674157, 'accuracy_se': 0.02510948127, 'release_date': '2026-02-05'}, {'model': 'claude-opus-4-6', 'harness': 'Meta-Harness', 'harness_org': 'Stanford IRIS', 'accuracy': 0.7640449438, 'accuracy_se': 0.02432461527, 'release_date': '2026-02-05'}, {'model': 'claude-opus-4-6', 'harness': 'Claude Code', 'harness_org': 'Anthropic', 'accuracy': 0.5797752809, 'accuracy_se': 0.02904963559, 'release_date': '2026-02-05'}, {'model': 'claude-opus-4-6', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.6292134831, 'accuracy_se': 0.02660985405, 'release_date': '2026-02-05'}, {'model': 'claude-opus-4-6', 'harness': 'Mux', 'harness_org': 'Coder', 'accuracy': 0.6651685393, 'accuracy_se': 0.0253018939, 'release_date': '2026-02-05'}, {'model': 'claude-opus-4-6', 'harness': 'Terminus-KIRA', 'harness_org': 'KRAFTON AI', 'accuracy': 0.7471910112, 'accuracy_se': 0.02596411146, 'release_date': '2026-02-05'}, {'model': 'glm-5', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.5241573034, 'accuracy_se': 0.02626589491, 'release_date': '2026-02-11'}, {'model': 'MiniMax-M2.5', 'harness': 'Terminus 2', 'harness_org': 'Terminal-Bench', 'accuracy': 0.4224719101, 'accuracy_se': 0.026242804, 'release_date': '2026-02-12'}, {'model': 'MiniMax-M2.5', 'harness': 'cchuter', 'harness_org': 'teamblobfish.com', 'accuracy': 0.4269662921, 'accuracy_se': 0.02824548367, 'release_date': '2026-02-12'}, {'model': 'claude-sonnet-4-6', 'harness': 'Simplai Agent', 'harness_org': 'SimplAI', 'accuracy': 0.5342696629, 'accuracy_se': 0.02830028074, 'release_date': '2026-02-17'}, {'model': 'gemini-3.1-pro-preview', 'harness': 'TongAgents', 'harness_org': 'BIGAI', 'accuracy': 0.802247191, 'accuracy_se': 0.026242804, 'release_date': '2026-02-19'}, {'model': 'gemini-3.1-pro-preview', 'harness': 'Gemini CLI', 'harness_org': 'Google', 'accuracy': 0.5936329588, 'accuracy_se': 0.04200978532, 'release_date': '2026-02-19'}, {'model': 'gemini-3.1-pro-preview', 'harness': 'Terminus-KIRA', 'harness_org': 'KRAFTON AI', 'accuracy': 0.7483146067, 'accuracy_se': 0.02568239486, 'release_date': '2026-02-19'}, {'model': 'qwen3.5-9B', 'harness': 'little-coder', 'harness_org': 'Itay Inbar', 'accuracy': 0.09213483146, 'accuracy_se': 0.02351356559, 'release_date': '2026-02-24'}, {'model': 'gemini-3.1-pro-preview', 'harness': 'ForgeCode', 'harness_org': 'ForgeCode', 'accuracy': 0.784, 'accuracy_se': 0.018, 'release_date': '2026-02-19'}, {'model': 'gemini-3.1-pro-preview', 'harness': 'Forge Code', 'harness_org': 'Forge Code', 'accuracy': 0.784, 'accuracy_se': 0.018, 'release_date': '2026-02-19'}, {'model': 'grok-4-20', 'harness': 'Grok CLI', 'harness_org': 'Superagent', 'accuracy': 0.5730337079, 'accuracy_se': None, 'release_date': '2026-02-17'}, {'model': 'claude-opus-4-6', 'harness': 'ForgeCode', 'harness_org': 'ForgeCode', 'accuracy': 0.798, 'accuracy_se': 0.016, 'release_date': '2026-02-05'}, {'model': 'gpt-5.4-2026-03-05', 'harness': 'ForgeCode', 'harness_org': 'ForgeCode', 'accuracy': 0.818, 'accuracy_se': 0.02, 'release_date': '2026-03-05'}, {'model': 'MiniMax-M2.7', 'harness': 'Harness Agent', 'harness_org': 'lazyFrogLOL', 'accuracy': 0.4292134831, 'accuracy_se': 0.02938164221, 'release_date': '2026-03-18'}, {'model': 'MiniMax-M2.7', 'harness': 'IndusAGI Coding Agent', 'harness_org': 'Varun Israni (SoloVpx)', 'accuracy': 0.4505617978, 'accuracy_se': None, 'release_date': '2026-03-18'}, {'model': 'grok-4-20', 'harness': 'grok-cli', 'harness_org': 'Vibe Kit', 'accuracy': 0.573, 'accuracy_se': None, 'release_date': '2026-02-17'}, {'model': 'claude-opus-4-7', 'harness': 'WOZCODE', 'harness_org': 'WOZCODE', 'accuracy': 0.802247191, 'accuracy_se': 0.021123213, 'release_date': '2026-04-16'}, {'model': 'qwen3.6-35b-a3b', 'harness': 'little-coder', 'harness_org': 'Itay Inbar', 'accuracy': 0.2303370787, 'accuracy_se': 0.03205049478, 'release_date': '2026-04-14'}, {'model': 'gpt-5.5', 'harness': 'Capy', 'harness_org': 'Capy', 'accuracy': 0.8314606742, 'accuracy_se': 0.021123213, 'release_date': '2026-04-23'}, {'model': 'gpt-5.5', 'harness': 'clnkr', 'harness_org': 'clnkr', 'accuracy': 0.6606741573, 'accuracy_se': 0.02491558276, 'release_date': '2026-04-23'}, {'model': 'gpt-5.5', 'harness': 'Codex', 'harness_org': 'OpenAI', 'accuracy': 0.82, 'accuracy_se': 0.022, 'release_date': '2026-04-23'}, {'model': 'gpt-5.5', 'harness': 'Codex CLI', 'harness_org': 'OpenAI', 'accuracy': 0.8224719101, 'accuracy_se': 0.02224160638, 'release_date': '2026-04-23'}, {'model': 'gpt-5.5', 'harness': 'NexAU-AHE', 'harness_org': 'china-qijizhifeng', 'accuracy': 0.8471910112, 'accuracy_se': 0.02089235128, 'release_date': '2026-04-23'}):
        by.setdefault(r['model'], []).append(100 * r['accuracy'])
    by = {m: v for m, v in by.items() if m in {'gemini-2.5-pro': 0.1624044944, 'grok-4-0709': 0.041, 'Qwen3-Coder-480B-A35B-Instruct': 0.0329588015, 'claude-opus-4-1-20250805': 0.032, 'gpt-5-mini-2025-08-07': 0.1258426966, 'gpt-5-nano-2025-08-07': 0.1483146067, 'gpt-5-codex': 0.03025842697, 'claude-sonnet-4-5-20250929': 0.06460674157, 'claude-haiku-4-5-20251001': 0.2160561798, 'gpt-5.1-codex': 0.2094606742, 'gemini-3-pro-preview': 0.1347066167, 'claude-opus-4-5-20251101': 0.1141460674, 'claude-opus-4-6': 0.2182247191, 'gpt-5.3-codex': 0.1370786517, 'gemini-3.1-pro-preview': 0.2086142322, 'gpt-5.5': 0.1865168539}}
    order = sorted(by, key=lambda m: sorted(by[m])[len(by[m]) // 2])
    fig, ax = plt.subplots(figsize=(6.2, 2.75))
    for y, m in enumerate(order):
        v = by[m]
        hi = m == 'claude-opus-4-6'
        ax.plot([min(v), max(v)], [y, y], color=PURPLE if hi else BLUE_L, lw=4 if hi else 3.5, solid_capstyle='round', alpha=0.5 if hi else 1)
        ax.plot(v, [y] * len(v), 'o', color=PURPLE if hi else BLUE, ms=4.5)
        ax.text(101, y, f"{100 * {'gemini-2.5-pro': 0.1624044944, 'grok-4-0709': 0.041, 'Qwen3-Coder-480B-A35B-Instruct': 0.0329588015, 'claude-opus-4-1-20250805': 0.032, 'gpt-5-mini-2025-08-07': 0.1258426966, 'gpt-5-nano-2025-08-07': 0.1483146067, 'gpt-5-codex': 0.03025842697, 'claude-sonnet-4-5-20250929': 0.06460674157, 'claude-haiku-4-5-20251001': 0.2160561798, 'gpt-5.1-codex': 0.2094606742, 'gemini-3-pro-preview': 0.1347066167, 'claude-opus-4-5-20251101': 0.1141460674, 'claude-opus-4-6': 0.2182247191, 'gpt-5.3-codex': 0.1370786517, 'gemini-3.1-pro-preview': 0.2086142322, 'gpt-5.5': 0.1865168539}[m]:.0f} pts, n={len(v)}", va='center', fontsize=10, color=PURPLE if hi else GRAY, fontweight='bold' if hi else 'normal')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10)
    for t, m in zip(ax.get_yticklabels(), order):
        if m == 'claude-opus-4-6':
            t.set_color(PURPLE)
            t.set_fontweight('bold')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Terminal-Bench 2.0 accuracy (%), one dot per scaffold')
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis='x', color=sp.C['grid'], lw=0.6)
    ax.tick_params(axis='y', length=0)
    axis_type(ax, 10)
    sp.save(fig, HERE / 'scaffold.svg')

def verdicts():
    """Dropping unresolved trials inflates the rate (illustrative, 60/25/15)."""
    rows = [('Reported, default rule', 'vd_rate', 0.6, PURPLE), ('If every open trial passes', 'vd_upper', 0.75, GRAY), ('Open trials dropped', 'vd_drop', 0.7058823529, RED)]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(2.6, 1.3))
        for i, (lab, key, rate, c) in enumerate(rows):
            show = step == 2 or i < 2
            v = 100 * rate
            ax.barh(2 - i, v, color=c, height=0.6, alpha=1 if show else 0)
            ax.text(v + 1.5, 2 - i, f"{ {'#7c3aed': '60', '#64748b': '75', '#dc2626': '70.6'}[c]}%", va='center', fontsize=9.4, color=c, fontweight='bold', alpha=1 if show else 0)
            ax.text(-1.5, 2 - i, lab, ha='right', va='center', fontsize=8.6, color=INK)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Acceptance rate of 100 trials (%)')
        axis_type(ax, 8.5)
        sp.save(fig, HERE / f'verdicts_step{step}.svg')

def coverage():
    """Our run: visible test pass vs sealed test pass, per configuration."""
    rows = [('baseline_v1', 'b'), ('candidate_v2', 'c')]
    fig, ax = plt.subplots(figsize=(2.3, 1.25))
    for i, (lab, tag) in enumerate(rows):
        y = 1 - i
        vis = 100 * {'baseline_v1': 88, 'candidate_v2': 88}[lab] / {'baseline_v1': 96, 'candidate_v2': 96}[lab]
        seal = 100 * {'baseline_v1': 0.6458333333, 'candidate_v2': 0.6875}[lab]
        ax.barh(y + 0.18, vis, height=0.34, color=GRAY_L, edgecolor=GRAY)
        ax.barh(y - 0.18, seal, height=0.34, color=PURPLE)
        ax.text(vis + 1.5, y + 0.18, f"visible test { {'baseline_v1': '91.7', 'candidate_v2': '91.7'}[lab]}%", va='center', fontsize=8.6, color=GRAY)
        ax.text(seal - 1.5, y - 0.18, f"sealed { {'baseline_v1': '64.6', 'candidate_v2': '68.8'}[lab]}%", va='center', ha='right', fontsize=8.6, color='white', fontweight='bold')
        ax.text(-2, y, lab, ha='right', va='center', fontsize=8.6, color=INK, fontweight='bold')
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Trials passing (%), 96 per configuration')
    axis_type(ax, 8.5)
    sp.save(fig, HERE / 'coverage.svg')

def zheng():
    """Zheng et al. 2023, Table 2, default prompt: position bias."""
    rows = [('GPT-4', 65.0, 30.0, 5.0, 0.0), ('GPT-3.5', 46.2, 50.0, 1.2, 2.5), ('Claude-v1', 23.8, 75.0, 0.0, 1.2)]
    fig, ax = plt.subplots(figsize=(5.4, 2.1))
    for i, (j, cons, first, second, err) in enumerate(rows):
        y = 2 - i
        left = 0
        for v, c in ((cons, GREEN), (first, RED), (second, AMBER), (err, GRAY)):
            ax.barh(y, v, left=left, color=c, height=0.62)
            if v >= 12:
                ax.text(left + v / 2, y, f'{v:.1f}%', ha='center', va='center', fontsize=11.5, color='white', fontweight='bold')
            left += v
        ax.text(-2, y, j, ha='right', va='center', fontsize=12, color=INK, fontweight='bold')
    ax.text(1, 2.52, 'same verdict after swap', fontsize=11, color=GREEN, fontweight='bold')
    ax.text(66, 2.52, 'favors first position', fontsize=11, color=RED, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 2.85)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Answer pairs judged twice, order swapped (%)')
    sp.save(fig, HERE / 'zheng.svg')

def designs():
    """Tasks or repeats: standard error of 600-run designs (exercise 1)."""
    free = [('600 tasks x 1', 2.0), ('200 tasks x 3', 2.581988897), ('100 tasks x 6', 3.265986324)]
    paid = [('54 tasks x 1', 6.666666667), ('50 tasks x 2', 5.656854249), ('40 tasks x 5', 5.291502622), ('37 tasks x 6', 5.369248442)]
    for step, rows, name in ((1, free, 'designs_step1'), (2, paid, 'designs_step2')):
        fig, ax = plt.subplots(figsize=(5.0, 1.9))
        best = min((v for _, v in rows))
        for i, (lab, v) in enumerate(rows):
            y = len(rows) - 1 - i
            c = PURPLE if v == best else GRAY
            ax.barh(y, v, color=c, height=0.62)
            ax.text(v + 0.12, y, f'{v:.2f}', va='center', fontsize=13, color=c, fontweight='bold')
            ax.text(-0.12, y, lab.replace(' x ', ' × '), ha='right', va='center', fontsize=13, color=INK)
        ax.set_xlim(0, 8)
        ax.set_ylim(-0.6, 3.6)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Standard error (points)')
        axis_type(ax, 12)
        sp.save(fig, HERE / f'{name}.svg')

def wilson_fig():
    """Wald against Wilson at 50 tasks (§13.5.2, figure 4 panel B)."""
    n = 50
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.axvspan(-8, 0, color=RED_L)
    ax.text(-4, 2.62, 'impossible', ha='center', fontsize=11, color=RED)
    for i, x in enumerate([0, 2, 25]):
        y = 2 - i
        wl, wu = {0: (0.0, 0.0), 1: (-0.01431611524, 0.09431611524), 2: (0.3614096176, 0.6385903824)}[i]
        sl, su = {0: (0.0, 0.07134759913), 1: (0.01103888433, 0.1346009069), 2: (0.3664451432, 0.6335548568)}[i]
        ax.plot([100 * wl, 100 * wu], [y + 0.17] * 2, color=GRAY, lw=5, solid_capstyle='butt')
        if x == 0:
            ax.plot([0], [y + 0.17], '|', color=GRAY, ms=14, mew=3)
        ax.plot([100 * sl, 100 * su], [y - 0.17] * 2, color=PURPLE, lw=5, solid_capstyle='butt')
        ax.text(-9, y, f'{x} of {n}', ha='right', va='center', fontsize=12.5, fontweight='bold', color=INK)
        ax.text(100 * max(wu, su) + 2, y + 0.17, f'Wald {100 * wl:.2f} to {100 * wu:.2f}', va='center', fontsize=11, color=GRAY)
        ax.text(100 * max(wu, su) + 2, y - 0.17, f'Wilson {100 * sl:.2f} to {100 * su:.2f}', va='center', fontsize=11, color=PURPLE, fontweight='bold')
    ax.set_xlim(-8, 100)
    ax.set_ylim(-0.6, 2.85)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Acceptance rate (%), 95 percent interval')
    sp.save(fig, HERE / 'wilson.svg')

def ranks():
    """SWE-bench Verified, Epoch AI runs, top 25 by score (MLSysIM)."""
    rows = sorted(({'model': 'claude-opus-4-7_max', 'release_date': '2026-04-16', 'mean_score': 0.8347107438, 'stderr': 0.01690116918}, {'model': 'gpt-5.5-pre-release_xhigh', 'release_date': '2026-04-23', 'mean_score': 0.805785124, 'stderr': 0.01800019887}, {'model': 'gemini-3.5-flash_high', 'release_date': '2026-05-19', 'mean_score': 0.7933884298, 'stderr': 0.01842242025}, {'model': 'claude-opus-4-6', 'release_date': '2026-02-05', 'mean_score': 0.7871900826, 'stderr': 0.0186235373}, {'model': 'glm-5.2_max', 'release_date': '2026-06-16', 'mean_score': 0.787, 'stderr': 0.0186569119}, {'model': 'deepseek-v4-pro_max', 'release_date': '2026-04-24', 'mean_score': 0.7763975155, 'stderr': 0.01897828052}, {'model': 'qwen3.7-max', 'release_date': '2026-05-19', 'mean_score': 0.7727272727, 'stderr': 0.01906835447}, {'model': 'gpt-5.4-2026-03-05_high', 'release_date': '2026-03-05', 'mean_score': 0.7685950413, 'stderr': 0.019189407}, {'model': 'claude-opus-4-5-20251101', 'release_date': '2025-11-24', 'mean_score': 0.7665289256, 'stderr': 0.01924895909}, {'model': 'kimi-k2.6', 'release_date': '2026-04-20', 'mean_score': 0.7665289256, 'stderr': 0.01924895909}, {'model': 'qwen3.6-max-preview', 'release_date': '2026-04-20', 'mean_score': 0.7665289256, 'stderr': 0.01924895909}, {'model': 'gemini-3.1-pro-preview-customtools', 'release_date': '2026-02-19', 'mean_score': 0.7561983471, 'stderr': 0.01953721336}, {'model': 'gemini-3-flash-preview', 'release_date': '2025-12-17', 'mean_score': 0.7541322314, 'stderr': 0.01959300214}, {'model': 'claude-sonnet-4-6', 'release_date': '2026-02-17', 'mean_score': 0.7520661157, 'stderr': 0.0196481827}, {'model': 'gpt-5.3-codex_high', 'release_date': '2026-02-05', 'mean_score': 0.7479338843, 'stderr': 0.0197567395}, {'model': 'glm-5.1', 'release_date': '2026-04-07', 'mean_score': 0.7417, 'stderr': 0.0199}, {'model': 'gpt-5.2-2025-12-11_high', 'release_date': '2025-12-11', 'mean_score': 0.7376033058, 'stderr': 0.02001783206}, {'model': 'kimi-k2.5', 'release_date': '2026-01-27', 'mean_score': 0.7376033058, 'stderr': 0.02001783206}, {'model': 'gpt-5-2025-08-07_high', 'release_date': '2025-08-07', 'mean_score': 0.7355371901, 'stderr': 0.02006832182}, {'model': 'claude-opus-4-1-20250805', 'release_date': '2025-08-05', 'mean_score': 0.7334710744, 'stderr': 0.02011824555}, {'model': 'gemini-3-pro-preview', 'release_date': '2025-11-18', 'mean_score': 0.729338843, 'stderr': 0.02021641171}, {'model': 'glm-5', 'release_date': '2026-02-11', 'mean_score': 0.7207792208, 'stderr': 0.0208941416}, {'model': 'gpt-5-2025-08-07_medium', 'release_date': '2025-08-07', 'mean_score': 0.7148760331, 'stderr': 0.020542754}, {'model': 'claude-sonnet-4-5-20250929', 'release_date': '2025-09-29', 'mean_score': 0.7128099174, 'stderr': 0.02058723494}, {'model': 'claude-opus-4-20250514', 'release_date': '2025-05-22', 'mean_score': 0.7066115702, 'stderr': 0.02071754535}, {'model': 'gpt-5.1-2025-11-13_high', 'release_date': '2025-11-13', 'mean_score': 0.6797520661, 'stderr': 0.02122974956}, {'model': 'gpt-5-mini-2025-08-07_medium', 'release_date': '2025-08-07', 'mean_score': 0.6466942149, 'stderr': 0.02174960114}, {'model': 'o3-2025-04-16_medium', 'release_date': '2025-04-16', 'mean_score': 0.6231884058, 'stderr': 0.02207233566}, {'model': 'claude-3-7-sonnet-20250219', 'release_date': '2025-02-24', 'mean_score': 0.6095041322, 'stderr': 0.02219846697}, {'model': 'qwen3.6-plus', 'release_date': '2026-03-31', 'mean_score': 0.5785123967, 'stderr': 0.02246855652}, {'model': 'gemini-2.5-pro', 'release_date': '2025-06-17', 'mean_score': 0.5755693582, 'stderr': 0.02251275674}, {'model': 'gpt-4.1-2025-04-14', 'release_date': '2025-04-14', 'mean_score': 0.4854166667, 'stderr': 0.02283586375}, {'model': 'gpt-4o-2024-11-20', 'release_date': '2024-11-20', 'mean_score': 0.3099173554, 'stderr': 0.02104262598}), key=lambda r: -r['mean_score'])[:25]
    z = 1.959963985
    fig, ax = plt.subplots(figsize=(5.9, 2.5))
    xs = range(1, len(rows) + 1)
    ax.errorbar(list(xs), [100 * r['mean_score'] for r in rows], yerr=[100 * z * r['stderr'] for r in rows], fmt='o', color=BLUE, ecolor=BLUE, elinewidth=2, capsize=2.5, ms=5)
    a, b = (rows[0], rows[1])
    ax.annotate('ranks 1 and 2 overlap', (1.5, 100 * a['mean_score'] + 4), xytext=(5, 89), fontsize=12, color=PURPLE, fontweight='bold', arrowprops=dict(arrowstyle='-', color=PURPLE, lw=1.2))
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_xlim(0, 26)
    ax.set_ylim(60, 92)
    ax.set_xlabel('Leaderboard rank (one evaluator, one scaffold)')
    ax.set_ylabel('Resolved (%)')
    ax.grid(axis='y', color=sp.C['grid'], lw=0.6)
    sp.save(fig, HERE / 'ranks.svg')

def run_curves():
    """Our run: pass@k and pass^k from 8 graded runs per task."""
    runs = 8
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for cfg, ls in (('baseline_v1', '--'), ('candidate_v2', '-')):
        ks = list(range(1, runs + 1))
        at = {'--': [64.58333333, 68.45238095, 69.79166667, 70.83333333, 71.875, 72.91666667, 73.95833333, 75.0], '-': [68.75, 76.19047619, 80.50595238, 83.92857143, 86.75595238, 88.98809524, 90.625, 91.66666667]}[ls]
        pw = {'--': [64.58333333, 60.71428571, 58.18452381, 55.95238095, 54.01785714, 52.38095238, 51.04166667, 50.0], '-': [68.75, 61.30952381, 58.18452381, 55.95238095, 54.01785714, 52.38095238, 51.04166667, 50.0]}[ls]
        ax.plot(ks, at, ls, color=BLUE, lw=3)
        ax.plot(ks, pw, ls, color=PURPLE, lw=3)
        if cfg == 'candidate_v2':
            ax.text(8.15, at[-1] + 1, f'pass@{runs}\n91.7%', va='center', fontsize=13, color=BLUE, fontweight='bold')
            ax.text(8.15, pw[-1] + 2, f'pass$^{runs}$\n50.0%', va='center', fontsize=13, color=PURPLE, fontweight='bold')
        else:
            ax.text(8.15, at[-1] - 6, '75.0%', va='center', fontsize=13, color=BLUE)
    ax.text(1.2, 36, 'solid: candidate_v2\ndashed: baseline_v1', fontsize=12.5, color=GRAY)
    ax.set_xlim(0.7, runs + 0.1)
    ax.set_ylim(30, 100)
    ax.set_xticks([1, 2, 4, 6, 8])
    ax.set_xlabel('k (graded runs per task)')
    ax.set_ylabel('Tasks (%)')
    axis_type(ax, 13)
    ax.grid(axis='y', color=sp.C['grid'], lw=0.6)
    sp.save(fig, HERE / 'run_curves.svg')

def kapoor():
    """Kapoor et al. 2024, Table A1: HumanEval accuracy against total cost."""
    role_color = {'published agent': GRAY, 'author baseline': PURPLE, 'single call': BLUE}
    label = {'GPT-4 (baseline)': 'GPT-4, one call', 'GPT-3.5 (baseline)': 'GPT-3.5, one call'}
    pts = [('LATS (GPT-4)', 88.0, 134.5, '#dc2626'), ('LATS (GPT-3.5)', 80.4, 9.49, '#64748b'), ('LDB (GPT-4)', 93.3, 6.36, '#64748b'), ('Reflexion (GPT-4)', 87.8, 3.9, '#64748b'), ('Retry (GPT-4)', 92.0, 2.51, '#7c3aed'), ('Warming (GPT-4)', 93.2, 2.45, '#7c3aed'), ('GPT-4, one call', 89.6, 1.93, '#0284c7'), ('Escalation', 85.0, 0.27, '#7c3aed'), ('GPT-3.5, one call', 73.9, 0.05, '#0284c7')]
    offs = {'LATS (GPT-4)': (-8, 7, 'right'), 'LATS (GPT-3.5)': (6, -14, 'left'), 'LDB (GPT-4)': (6, 4, 'left'), 'Reflexion (GPT-4)': (6, -12, 'left'), 'Retry (GPT-4)': (-8, -6, 'right'), 'Warming (GPT-4)': (-8, 6, 'right'), 'GPT-4, one call': (-8, -6, 'right'), 'Escalation': (6, -12, 'left'), 'GPT-3.5, one call': (6, 2, 'left')}
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for lab, acc, cost, c in pts:
        ax.plot([cost], [acc], 'o', color=c, ms=8)
        dx, dy, ha = offs[lab]
        ax.annotate(lab, (cost, acc), xytext=(dx, dy), textcoords='offset points', ha=ha, fontsize=10.5, color=c, fontweight='bold' if c != GRAY else 'normal')
    ax.set_xscale('log')
    ax.set_xlim(0.03, 400)
    ax.set_ylim(70, 97)
    ax.set_xlabel('Total cost of the 164 problems (US dollars, log)')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(color=sp.C['grid'], lw=0.6)
    sp.save(fig, HERE / 'kapoor.svg')

def batch():
    """NM 13.3: cost per accepted task, batch vs sequential."""
    b, s = (0.1530612245, 0.1)
    fig, ax = plt.subplots(figsize=(4.4, 1.6))
    ax.barh(1, b, color=GRAY, height=0.6)
    ax.barh(0, s, color=PURPLE, height=0.6)
    ax.text(b + 0.004, 1, '$0.153', va='center', fontsize=13, color=GRAY, fontweight='bold')
    ax.text(s + 0.004, 0, '$0.100', va='center', fontsize=13, color=PURPLE, fontweight='bold')
    ax.text(-0.003, 1, 'batch of 3', ha='right', va='center', fontsize=12.5)
    ax.text(-0.003, 0, 'sequential', ha='right', va='center', fontsize=12.5)
    ax.set_xlim(0, 0.2)
    ax.set_xticks([0, 0.05, 0.1, 0.15])
    ax.set_xticklabels(['$0', '$0.05', '$0.10', '$0.15'])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Cost per accepted task')
    sp.save(fig, HERE / 'batch.svg')

def dims():
    """Four evidence dimensions (table 1), then the criteria they supply."""
    cols = [('Proposal', 'validity', 'schema rejected,', 'then repaired', BLUE, BLUE_L), ('Execution', 'behavior', 'timeout leaves', 'a test unresolved', AMBER, AMBER_L), ('Task', 'outcome', 'checks on the', 'exact ws_r19 bytes', GREEN, GREEN_L), ('Constraint', 'compliance', 'authority, budget,', 'cleanup owners', RED, RED_L)]
    for step in (1, 2):
        b = []
        for i, (t1, t2, e1, e2, s, f) in enumerate(cols):
            x = 20 + i * 220
            b.append(box(x, 20, 200, 190, f, s))
            b.append(text(x + 100, 68, t1, 26, s, weight='bold'))
            b.append(text(x + 100, 100, t2, 26, s, weight='bold'))
            b.append(text(x + 100, 150, e1, 19, INK))
            b.append(text(x + 100, 176, e2, 19, INK))
        if step == 2:
            b.append(text(450, 262, 'The accepted-task criteria they supply', 22, GRAY, weight='bold'))
            spans = [(20, 420, 'process', PURPLE), (460, 640, 'outcome', PURPLE), (670, 880, 'resource, residual duty', PURPLE)]
            for x1, x2, lab, c in spans:
                b.append(box(x1, 280, x2 - x1, 54, PURPLE_L, c, rx=8))
                b.append(text((x1 + x2) / 2, 316, lab, 22 if len(lab) < 12 else 17, c, weight='bold'))
            b.append(text(450, 378, 'Exposure each audits', 22, GRAY, weight='bold'))
            b.append(text(220, 414, 'authority exercised', 21, INK))
            b.append(text(550, 414, 'state the result rests on', 21, INK))
            b.append(text(770, 414, 'horizon consumed', 21, INK))
        svg(HERE / f'dims_step{step}.svg', 900, 430, '\n'.join(b))

def gym():
    """Trial sandbox and grading boundary (after figure 2, §13.3.4)."""
    for step in (1, 2, 3):
        b = []
        b.append(box(15, 15, 420, 450, '#fffbeb', AMBER))
        b.append(text(225, 50, 'Task sandbox, one per trial', 22, AMBER, weight='bold'))
        b.append(box(45, 75, 360, 70, PURPLE_L, PURPLE))
        b.append(text(225, 108, 'agent under test', 22, PURPLE, weight='bold'))
        b.append(text(225, 134, 'candidate_v2', 19, INK))
        b.append(arrow(225, 145, 225, 190, 'g'))
        b.append(box(45, 195, 360, 52, GREEN_L, GREEN))
        b.append(text(225, 229, 'copy-on-write layer (this trial)', 20, INK))
        b.append(box(45, 252, 360, 52, GRAY_L, GRAY))
        b.append(text(225, 286, 'base image, read-only, shared', 20, INK))
        b.append(box(45, 325, 220, 52, BLUE_L, BLUE))
        b.append(text(155, 358, 'service fixtures', 20, INK))
        b.append(box(285, 325, 120, 52, RED_L, RED))
        b.append(text(345, 349, 'egress', 19, RED, weight='bold'))
        b.append(text(345, 370, 'blocked', 19, RED, weight='bold'))
        b.append(text(225, 425, 'reset: kill processes, revoke', 19, GRAY))
        b.append(text(225, 450, 'credentials, discard the layer', 19, GRAY))
        if step >= 2:
            b.append(arrow(405, 221, 495, 221, 'a', AMBER, sw=3.5))
            b.append(text(450, 205, 'ws_r19', 19, AMBER, weight='bold'))
            b.append(box(500, 15, 385, 450, '#faf5ff', PURPLE))
            b.append(text(692, 50, 'Grading boundary', 22, PURPLE, weight='bold'))
            b.append(box(530, 75, 325, 150, 'white', PURPLE))
            b.append(text(692, 106, 'trusted controller', 21, PURPLE, weight='bold'))
            for x, w, chip in ((540, 130, 'hidden tests'), (678, 85, 'criteria'), (771, 78, 'results')):
                b.append(box(x, 124, w, 44, PURPLE_L, PURPLE, rx=6, sw=1.5))
                b.append(text(x + w / 2, 152, chip, 19, INK))
            b.append(text(692, 202, "outside the candidate's reach", 20, GRAY))
        if step >= 3:
            b.append(arrow(640, 225, 640, 275, 'p', PURPLE))
            b.append(text(632, 258, 'selected tests', 19, PURPLE, anchor='end'))
            b.append(box(530, 280, 325, 70, RED_L, RED, dash=True))
            b.append(text(692, 309, 'candidate execution, limited', 19, RED, weight='bold'))
            b.append(text(692, 336, 'the remaining attack surface', 20, INK))
            b.append(arrow(760, 280, 760, 230, 'r', RED, dash=True))
            b.append(text(772, 258, 'untrusted', 19, RED, anchor='start'))
            b.append(box(530, 375, 325, 75, 'white', PURPLE))
            b.append(text(692, 404, 'pass, fail, pending, unknown', 19, INK, weight='bold'))
            b.append(text(692, 434, 'each bound to sha256(ws_r19)', 20, PURPLE))
        svg(HERE / f'gym_step{step}.svg', 900, 480, '\n'.join(b))

def confusion():
    """Grader error rates condition on different populations (exercise 3)."""
    cells = {('fails', 'accepts'): '16', ('fails', 'rejects'): '64', ('passes', 'accepts'): '114', ('passes', 'rejects'): '6'}
    for step in (1, 2):
        b = []
        x0, y0, cw, ch = (240, 95, 180, 110)
        b.append(text(x0 + cw, 40, 'Grader under study', 22, GRAY, weight='bold'))
        b.append(text(x0 + cw / 2, 78, 'accepts', 22, INK, weight='bold'))
        b.append(text(x0 + 1.5 * cw, 78, 'rejects', 22, INK, weight='bold'))
        b.append(text(80, y0 + 40, 'Reference', 21, GRAY, weight='bold', anchor='middle'))
        b.append(text(80, y0 + 66, 'assessment', 21, GRAY, weight='bold', anchor='middle'))
        b.append(text(x0 - 22, y0 + ch / 2 + 8, 'fails', 22, INK, anchor='end', weight='bold'))
        b.append(text(x0 - 22, y0 + 1.5 * ch + 8, 'passes', 22, INK, anchor='end', weight='bold'))
        for i, ref in enumerate(('fails', 'passes')):
            for j, gr in enumerate(('accepts', 'rejects')):
                hit = (ref, gr) == ('fails', 'accepts')
                b.append(box(x0 + j * cw, y0 + i * ch, cw, ch, RED_L if hit else 'white', GRAY, rx=0, sw=1.5))
                b.append(text(x0 + j * cw + cw / 2, y0 + i * ch + ch / 2 + 12, str(cells[ref, gr]), 34, RED if hit else INK, weight='bold'))
        b.append(f'<rect x="{x0 - 6}" y="{y0 - 6}" width="{2 * cw + 12}" height="{ch + 12}" fill="none" stroke="{PURPLE}" stroke-width="4" rx="6"/>')
        b.append(text(x0 + 2 * cw + 25, y0 + 45, 'false-acceptance rate', 21, PURPLE, anchor='start', weight='bold'))
        b.append(text(x0 + 2 * cw + 25, y0 + 75, '16 / 80 = 20%', 21, PURPLE, anchor='start'))
        if step == 2:
            b.append(f'<rect x="{x0 - 12}" y="{y0 - 12}" width="{cw + 24}" height="{2 * ch + 24}" fill="none" stroke="{BLUE}" stroke-width="4" stroke-dasharray="10 6" rx="6"/>')
            b.append(text(x0 + 2 * cw + 25, y0 + ch + 60, 'failure among accepted', 21, BLUE, anchor='start', weight='bold'))
            b.append(text(x0 + 2 * cw + 25, y0 + ch + 90, '16 / 130 = 12.3%', 21, BLUE, anchor='start'))
        svg(HERE / f'confusion_step{step}.svg', 900, 340, '\n'.join(b))

def kappa():
    """Two judges that each pass 90 percent at random (§13.4.3)."""
    b = []
    x0, y0, c = (140, 70, 150)
    b.append(text(x0 + c, 30, 'Judge B', 30, GRAY, weight='bold'))
    b.append(text(x0 + c / 2, 60, 'pass', 28, INK))
    b.append(text(x0 + 1.5 * c, 60, 'fail', 28, INK))
    b.append(text(52, y0 + c - 10, 'Judge', 30, GRAY, weight='bold'))
    b.append(text(52, y0 + c + 22, 'A', 30, GRAY, weight='bold'))
    b.append(text(x0 - 12, y0 + c / 2 + 8, 'pass', 28, INK, anchor='end'))
    b.append(text(x0 - 12, y0 + 1.5 * c + 8, 'fail', 28, INK, anchor='end'))
    vals = [['81', '9'], ['9', '1']]
    for i in range(2):
        for j in range(2):
            agree = i == j
            b.append(box(x0 + j * c, y0 + i * c, c, c, GREEN_L if agree else 'white', GRAY, rx=0, sw=1.5))
            b.append(text(x0 + j * c + c / 2, y0 + i * c + c / 2 + 14, f'{vals[i][j]}%', 38, GREEN if agree else INK, weight='bold'))
    svg(HERE / 'kappa.svg', 470, 390, '\n'.join(b))

def recall_thumbs():
    """Text-free cues: a saga with compensations (Lecture 12) and a staged
    release (Lecture 14). Shapes only; the slide text carries the words."""
    b = []
    for i in range(4):
        x = 20 + i * 120
        b.append(box(x, 40, 80, 50, GREEN_L if i < 3 else RED_L, GREEN if i < 3 else RED))
        if i < 3:
            b.append(arrow(x + 80, 65, x + 118, 65, 'g'))
    for i in range(2, 0, -1):
        x = 20 + i * 120
        b.append(f'<path d="M{x + 40},92 C{x + 30},140 {x - 50},140 {x - 80 + 40},92" fill="none" stroke="{PURPLE}" stroke-width="3" stroke-dasharray="7 5" marker-end="url(#ap)"/>')
    svg(HERE / 'thumb_saga.svg', 500, 150, '\n'.join(b))
    b = [box(20, 30, 460, 40, GRAY_L, GRAY, rx=6), box(20, 30, 46, 40, PURPLE, PURPLE, rx=6), box(20, 90, 460, 40, GRAY_L, GRAY, rx=6), box(20, 90, 230, 40, PURPLE, PURPLE, rx=6), arrow(250, 80, 470, 80, 'p', PURPLE)]
    svg(HERE / 'thumb_canary.svg', 500, 150, '\n'.join(b))

def thumbs_src():
    """Text-free sources for goals and takeaways thumbnails are made by
    tools/greek_thumbs.py from these figures."""
    return ['dims_step2.svg', 'gym_step3.svg', 'confusion_step2.svg', 'hook_step2.svg', 'batch.svg']

def run_hsa():
    """One trial of the running case, tied to its three exposures."""
    b = [box(170, 168, 220, 96, PURPLE_L, PURPLE), text(280, 206, 'trial ws_r19', 27, PURPLE, weight='bold'), text(280, 240, 'pool_repair_01', 23, INK)]
    spokes = [('H', 'horizon', 'many turns: tokens,', 'time, dollars', 20, 20, BLUE, BLUE_L, 'b'), ('S', 'state', 'workspace ws_r19,', 'test environment', 20, 304, AMBER, AMBER_L, 'a'), ('A', 'authority', 'edits the repo,', 'runs the tests', 300, 304, RED, RED_L, 'r')]
    for k, name, l1, l2, x, y, c, cl, m in spokes:
        b.append(box(x, y, 240, 110, cl, c))
        b.append(text(x + 120, y + 34, f'{k}: {name}', 27, c, weight='bold'))
        b.append(text(x + 120, y + 68, l1, 23))
        b.append(text(x + 120, y + 96, l2, 23))
    b.append(arrow(230, 168, 170, 136, 'b', BLUE, sw=3))
    b.append(arrow(220, 264, 170, 300, 'a', AMBER, sw=3))
    b.append(arrow(340, 264, 390, 300, 'r', RED, sw=3))
    b.append(text(420, 50, 'one run exercises', 23, GRAY))
    b.append(text(420, 78, 'all three at once', 23, GRAY))
    svg(HERE / 'run_hsa.svg', 560, 424, '\n'.join(b))

def seq_batch():
    """Batch generates and checks every candidate; sequential stops at a pass."""
    n = int(3.0)
    b = [text(6, 26, 'Batch: all 3, every trial', 25, GRAY, 'start', 'bold')]
    for i in range(n):
        b.append(box(6 + i * 146, 38, 126, 46, GRAY_L, GRAY))
        b.append(text(69 + i * 146, 70, f'cand. {i + 1}', 24))
    b.append(text(6, 122, 'Sequential: stop at a pass', 25, PURPLE, 'start', 'bold'))
    for i in range(n):
        b.append(box(6 + i * 146, 134, 126, 46, PURPLE_L, PURPLE, dash=i > 0))
        b.append(text(69 + i * 146, 166, f'cand. {i + 1}', 24, PURPLE))
        if i < n - 1:
            b.append(arrow(132 + i * 146, 157, 150 + i * 146, 157, 'p', PURPLE))
    b.append(text(6, 212, '1.96 candidates on average', 23, INK, 'start'))
    svg(HERE / 'seq_batch.svg', 440, 222, '\n'.join(b))
if __name__ == '__main__':
    for f in (hook, scaffold, verdicts, coverage, zheng, designs, wilson_fig, ranks, run_curves, kapoor, batch, dims, gym, confusion, kappa, recall_thumbs, run_hsa, seq_batch):
        f()
        print('wrote', f.__name__)
