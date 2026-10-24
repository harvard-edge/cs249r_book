#!/usr/bin/env python3
"""Figures for Lecture 11 (Durable Execution), Volume III.

Diagrams are SVG in the Volume III agentic-stack palette (runtime violet,
tools amber, model sky blue, memory green, faults red, structure slate).
Plots use slides/assets/kit/slideplot.py. Every number drawn comes from
calcs.py, which names its source (DE = the chapter, Durable Execution):

  restarts          measured: data/resample.json (30 restarts from the prompt)
  lost_step1-2      DE Sec. 11.1 (what a crash erases), drawn for the slide
  fold_step1-3      DE Sec. 11.2.1, figure 1 (event sourcing), redrawn
  intent_step1-2    DE Sec. 11.3.2, figure 2 (crash position), redrawn
  crashrun_step1-2  measured: data/summary_Repair*.json (Temporal crash runs)
  acks              measured: data/ack_latency.json
  replica_step1-2   DE Sec. 11.3.3 (completeness rule), drawn for the slide
  groupcommit       measured: data/ack_latency.json (group commit)
  ckpt_step1-3      DE Sec. 11.4.1 (consistent recovery boundary), drawn
  snapshots         DE equation 3 with Sec. 11.4.2 inputs (illustrative)
  cadence           DE NM 11.1 and equation 5 (illustrative)
  resume_step1-3    DE Sec. 11.5 (five stages), drawn for the slide
  fence_step1-3     DE Sec. 11.5.1 (fencing tokens), drawn for the slide
  prefill           measured: data/resume_prefill.json
  feynman           Thinking Machines measurement (Literature.InferenceDeterminism), drawn
  replay_step1-2    DE Sec. 11.6, figure 3 (live vs replay), redrawn
  diverge_step1-2   DE Sec. 11.7 (first divergence, r18 example), drawn
  retention         DE NM 11.2 (illustrative)
  thumb_*           text-free thumbnails for the goals and takeaways (log,
                    intent, cadence, replay)

Run from this folder with the MLSysBook venv: python3 make_figs.py
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import slideplot as sp
C = sp.C
plt = sp.plt
OUT = HERE
ROLE = {'model': ('#e0f2fe', '#0284c7', '#0369a1'), 'memory': ('#dcfce7', '#16a34a', '#15803d'), 'tool': ('#fef3c7', '#d97706', '#b45309'), 'sys': ('#f3e8ff', '#7c3aed', '#6d28d9'), 'fault': ('#fee2e2', '#dc2626', '#b91c1c'), 'struct': ('#f8fafc', '#94a3b8', '#0f172a'), 'white': ('#ffffff', '#94a3b8', '#0f172a')}
INK = '#0f172a'
GRAY = '#64748b'
FONT = 'Helvetica, Arial, sans-serif'
MONO = 'Courier New, Courier, monospace'
FLOOR = 22
VIOLET, RED, BLUE, AMBER, GREEN = (ROLE['sys'][1], ROLE['fault'][1], ROLE['model'][1], ROLE['tool'][1], ROLE['memory'][1])

class Svg:

    def __init__(self, w, h):
        self.w, self.h, self.parts = (w, h, [])

    def add(self, s):
        self.parts.append(s)

    def save(self, name):
        markers = ''.join((f'<marker id="a{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>' for k, col in (('blue', BLUE), ('violet', VIOLET), ('amber', AMBER), ('green', GREEN), ('red', RED), ('gray', GRAY), ('ink', INK))))
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" font-family="{FONT}"><defs>{markers}<pattern id="hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="10" stroke="#fca5a5" stroke-width="4"/></pattern></defs><rect x="0" y="0" width="{self.w}" height="{self.h}" fill="white"/>' + ''.join(self.parts) + '</svg>'
        (OUT / f'{name}.svg').write_text(svg)

    def rect(self, x, y, w, h, role='struct', dash=False, rx=8, sw=2, fill=None, stroke=None, opacity=1.0):
        f, s, _ = ROLE[role]
        d = ' stroke-dasharray="7 5"' if dash else ''
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill or f}" stroke="{stroke or s}" stroke-width="{sw}"{d} opacity="{opacity}"/>')

    def text(self, x, y, s, fs=22, color=INK, anchor='middle', weight='normal', mono=False, italic=False, opacity=1.0, floor=None):
        fs = max(fs, floor if floor is not None else FLOOR)
        fam = f' font-family="{MONO}"' if mono else ''
        it = ' font-style="italic"' if italic else ''
        s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        self.add(f'<text x="{x}" y="{y}" font-size="{fs}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}"{fam}{it} opacity="{opacity}">{s}</text>')

    def box(self, x, y, w, h, role, title, sub=None, fs=24, sfs=22, dash=False, mono_sub=False, title_color=None, sw=2, mono_title=False, opacity=1.0):
        self.rect(x, y, w, h, role, dash=dash, sw=sw, opacity=opacity)
        _, _, tc = ROLE[role]
        if sub:
            self.text(x + w / 2, y + h / 2 - 5, title, fs, title_color or tc, weight='bold', mono=mono_title, opacity=opacity)
            self.text(x + w / 2, y + h / 2 + sfs + 1, sub, sfs, GRAY, mono=mono_sub, opacity=opacity)
        else:
            self.text(x + w / 2, y + h / 2 + fs * 0.35, title, fs, title_color or tc, weight='bold', mono=mono_title, opacity=opacity)

    def arrow(self, x1, y1, x2, y2, color='gray', dash=False, sw=2.5, path=None):
        col = {'blue': BLUE, 'violet': VIOLET, 'amber': AMBER, 'green': GREEN, 'red': RED, 'gray': GRAY, 'ink': INK}[color]
        d = ' stroke-dasharray="8 6"' if dash else ''
        geom = path or f'M{x1},{y1} L{x2},{y2}'
        self.add(f'<path d="{geom}" fill="none" stroke="{col}" stroke-width="{sw}"{d} marker-end="url(#a{color})"/>')

    def line(self, x1, y1, x2, y2, color=GRAY, sw=2, dash=False):
        d = ' stroke-dasharray="7 5"' if dash else ''
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>')

    def cross(self, x, y, r=11, color=None, sw=5):
        col = color or RED
        self.add(f'<path d="M{x - r},{y - r} L{x + r},{y + r} M{x - r},{y + r} L{x + r},{y - r}" stroke="{col}" stroke-width="{sw}"/>')

    def check(self, x, y, r=12, color=None):
        col = color or GREEN
        self.add(f'<path d="M{x - r},{y} L{x - r / 3},{y + r * 0.7} L{x + r},{y - r * 0.8}" fill="none" stroke="{col}" stroke-width="5"/>')

    def dot(self, x, y, r=9, color=INK):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    def num(self, x, y, k, color=None):
        col = color or VIOLET
        self.add(f'<circle cx="{x}" cy="{y}" r="16" fill="{col}"/>')
        self.text(x, y + 7, str(k), 21, 'white', weight='bold')

def restarts():
    """Column figure: first steps chosen by 30 restarts from the prompt."""
    r = {'temp': 0.8, 'n': 30, 'distinct': 8, 'by_tool': {'read_file': 16, 'grep': 9, 'run_tests': 5}, 'n0': 30, 'distinct0': 1}
    order = [('read_file', 'read a file'), ('grep', 'search the code'), ('run_tests', 'run the tests')]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ys = np.arange(len(order))[::-1]
    for y, (key, lab) in zip(ys, order):
        v = r['by_tool'].get(key, 0)
        ax.barh(y, v, color=C['memory'] if key != 'run_tests' else C['danger'], height=0.6)
        ax.text(v + 0.4, y, f'{v}', va='center', fontsize=16, fontweight='bold', color=C['memory'] if key != 'run_tests' else C['danger'])
        ax.text(-0.4, y, lab, va='center', ha='right', fontsize=15, fontweight='bold', color=C['text'])
    ax.set_yticks([])
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 10, 20])
    ax.tick_params(labelsize=13)
    ax.set_xlabel(f"Restarts, of {r['n']}", fontsize=14)
    ax.spines['left'].set_visible(False)
    ax.set_title(f"{r['distinct']} distinct first calls", fontsize=15, color=C['text'], fontweight='bold', loc='left')
    sp.save(fig, OUT / 'restarts.svg')

def lost():
    """What one crash erases: worker memory vs the context window vs a log."""
    for k in (1, 2):
        s = Svg(900, 470)
        s.rect(20, 20, 520, 300, 'struct', rx=12, sw=2, dash=False)
        s.text(40, 56, 'Worker process (memory)', 24, INK, anchor='start', weight='bold')
        items = [('test_r18 dispatched, no result', 'tool'), ('$0.50 reserved for the test', 'sys'), ('grants and control_epoch', 'sys'), ('approval pending for r19', 'sys')]
        for i, (lab, role) in enumerate(items):
            s.rect(40, 76 + i * 56, 480, 46, role, rx=6, sw=1.5, opacity=1.0 if k == 1 else 0.35)
            s.text(56, 107 + i * 56, lab, 23, INK, anchor='start', opacity=1.0 if k == 1 else 0.35)
        s.rect(580, 20, 300, 300, 'model', rx=12, sw=2)
        s.text(600, 56, 'Context window', 24, ROLE['model'][2], anchor='start', weight='bold')
        for i, lab in enumerate(('prompt', 'turns 1 to 17,', 'compacted', 'last observation')):
            s.text(600, 108 + i * 40, lab, 23, INK, anchor='start')
        s.text(600, 290, 'no operations, no budget', 21, GRAY, anchor='start')
        if k >= 2:
            s.cross(280, 190, 70, RED, sw=10)
            s.text(280, 370, 'a crash erases all of it', 25, ROLE['fault'][2], weight='bold')
            s.text(730, 370, 'a transcript, not a record', 23, ROLE['model'][2], weight='bold')
            s.rect(20, 400, 860, 56, 'sys', rx=10, sw=2.5)
            s.text(450, 437, 'Only what reached durable storage survives: the trajectory log', 23, ROLE['sys'][2], weight='bold')
        s.save(f'lost_step{k}')

def fold():
    """Event sourcing: the log is folded into the record (DE figure 1)."""
    ev = [('TaskStarted', None), ('ContextBuilt', None), ('ModelReplied', None), ('Authorized', '+ reserve'), ('ToolIntent', 'test_r18')]
    for k in (1, 2, 3):
        s = Svg(900, 470)
        s.rect(10, 286, 880, 178, 'white', rx=12, stroke='#94a3b8', sw=2)
        s.text(28, 318, 'Trajectory log: durable, append-only', 24, INK, anchor='start', weight='bold')
        x = 20
        w = 148
        for i, (a, b) in enumerate(ev):
            s.rect(x + i * w, 366, w, 84, 'sys', rx=0, sw=2)
            s.text(x + i * w + w / 2, 356, f'e{i + 1}', 22, VIOLET, weight='bold')
            if b:
                s.text(x + i * w + w / 2, 400, a, 22, INK)
                s.text(x + i * w + w / 2, 430, b, 22, INK, mono=b == 'test_r18')
            else:
                s.text(x + i * w + w / 2, 416, a, 22, INK)
        s.line(768, 340, 768, 456, VIOLET, 4)
        s.rect(776, 366, 108, 84, 'white', dash=True, rx=6)
        s.text(830, 400, 'result', 22, GRAY)
        s.text(830, 430, 'missing', 22, GRAY)
        if k >= 2:
            s.box(330, 40, 250, 120, 'struct', 'Reducer', fs=26)
            s.text(455, 128, 'apply(R, e)', 23, INK, mono=True)
            s.text(455, 196, 'no model call, no tool call', 22, GRAY)
            s.arrow(455, 284, 455, 210, 'violet', sw=3.5)
            s.text(470, 258, 'fold, in log order', 22, VIOLET, anchor='start')
            s.box(20, 50, 240, 100, 'memory', 'R0', 'task, grants, limits', fs=26, sfs=22)
            s.arrow(262, 100, 328, 100, 'gray')
        if k >= 3:
            s.rect(650, 30, 240, 200, 'memory', rx=10, sw=2)
            s.text(770, 62, 'Rebuilt record', 24, ROLE['memory'][2], weight='bold')
            s.text(770, 104, 'admission held', 22, INK)
            s.text(770, 140, '$0.50 still reserved', 22, INK)
            s.rect(664, 162, 212, 50, 'fault', dash=True, rx=6)
            s.text(770, 195, 'test_r18 unresolved', 22, ROLE['fault'][2], weight='bold')
            s.arrow(582, 100, 648, 100, 'gray')
        s.save(f'fold_step{k}')

def intent():
    """Durable intent before dispatch, and what each crash position proves."""
    chain = [('propose', 'model reply', 'sys'), ('admit', 'grant, reserve', 'sys'), ('intent ack', 'durable', 'sys'), ('dispatch', 'request leaves', 'tool'), ('evidence', 'result logged', 'sys')]
    for k in (1, 2):
        s = Svg(900, 460)
        w, gap, y = (158, 22, 66)
        for i, (a, b, role) in enumerate(chain):
            x = 10 + i * (w + gap)
            s.box(x, y, w, 96, role, a, b, fs=25, sfs=22)
            if i < 4:
                s.arrow(x + w + 1, y + 48, x + w + gap - 1, y + 48, 'amber' if i == 2 else 'gray')
        s.line(539, 42, 539, 300 if k == 1 else 440, VIOLET, 3.5, dash=True)
        s.text(539, 30, 'durability barrier', 22, VIOLET, weight='bold')
        s.line(10, 216, 890, 216, GRAY, 2)
        s.arrow(860, 216, 890, 216, 'gray')
        s.text(890, 244, 'time', 22, GRAY, anchor='end')
        s.text(268, 198, 'runtime can still withhold', 22, GRAY)
        s.text(712, 198, 'request may have left', 22, GRAY)
        if k >= 2:
            zones = [(10, 439, 'struct', 'Not dispatched', 'absence proves it only', 'in a complete log'), (449, 281, 'fault', 'In doubt', 'reconcile before', 'dependent work'), (730, 160, 'memory', 'Established', 'per endpoint', 'contract')]
            for x, w2, role, a, b, c in zones:
                s.rect(x, 262, w2, 180, role, rx=0, sw=2.5)
                s.text(x + w2 / 2, 312, a, 25, ROLE[role][2], weight='bold')
                s.text(x + w2 / 2, 362, b, 22, INK)
                s.text(x + w2 / 2, 394, c, 22, INK)
        s.save(f'intent_step{k}')

def crashrun():
    """The two measured Temporal runs as lanes on one time axis."""
    c = {'RepairNaive': {'model_calls': 1, 'launches': 2, 'reused': 0, 'ledger': [('model_call', None, 1, 0.0), ('job_launched', 'job_tests_1', 1, 0.007030010223), ('job_launched', 'job_tests_2', 2, 3.889447212)]}, 'RepairKeyed': {'model_calls': 1, 'launches': 1, 'reused': 1, 'ledger': [('model_call', None, 1, 0.0), ('job_launched', 'job_tests_1', 1, 0.008343219757), ('job_reused', 'job_tests_1', 2, 3.888708115)]}}
    for k in (1, 2):
        s = Svg(900, 420)
        x0, sc = (250, 140)
        lanes = [('RepairNaive', 'Unkeyed test', 130), ('RepairKeyed', 'Keyed test', 275)]
        s.line(x0, 380, x0 + 4.6 * sc, 380, GRAY, 2)
        for t in range(0, 5):
            s.line(x0 + t * sc, 376, x0 + t * sc, 386, GRAY, 2)
            s.text(x0 + t * sc, 412, f'{t} s', 22, GRAY)
        for name, lab, y in lanes:
            run = c[name]
            s.text(10, y + 8, lab, 22, INK, anchor='start', weight='bold')
            s.line(x0, y, x0 + 4.6 * sc, y, '#cbd5e1', 3)
            for ev, job, att, t in run['ledger']:
                if ev == 'model_call':
                    continue
                if att == 1:
                    s.rect(x0 + t * sc - 6, y - 26, 26, 52, 'tool', rx=4)
                    s.text(x0 + t * sc + 7, y + 52, 'job_tests_1', 22, ROLE['tool'][2])
                    s.text(x0 + t * sc + 7, y + 78, 'launched', 22, ROLE['tool'][2])
                elif k >= 2:
                    if ev == 'job_launched':
                        s.rect(x0 + t * sc - 6, y - 26, 26, 52, 'fault', rx=4)
                        s.text(x0 + t * sc + 24, y - 36, f'{job} launched again', 22, ROLE['fault'][2], anchor='end', weight='bold')
                    else:
                        s.rect(x0 + t * sc - 6, y - 26, 26, 52, 'memory', rx=4)
                        s.text(x0 + t * sc + 24, y - 36, 'job_tests_1 found, reused', 22, ROLE['memory'][2], anchor='end', weight='bold')
        kx = x0 + 1.0 * sc
        s.line(kx, 46, kx, 365, RED, 3, dash=True)
        s.text(kx, 36, 'worker killed (SIGKILL)', 22, ROLE['fault'][2], weight='bold')
        if k == 1:
            s.text(x0 + 2.75 * sc, 210, 'new worker: what runs?', 24, INK, weight='bold')
        s.save(f'crashrun_step{k}')

def acks():
    a = {'write': 0.024125, 'fsync': 0.0430625, 'full': 3.0078955, 'group': {1: {'flush_ms_median': 3.053792, 'per_record_ms': 3.053792}, 2: {'flush_ms_median': 3.0020835, 'per_record_ms': 1.50104175}, 4: {'flush_ms_median': 3.04875, 'per_record_ms': 0.7621875}, 8: {'flush_ms_median': 3.071854, 'per_record_ms': 0.38398175}, 16: {'flush_ms_median': 3.048104, 'per_record_ms': 0.1905065}, 32: {'flush_ms_median': 3.554021, 'per_record_ms': 0.1110631563}, 64: {'flush_ms_median': 3.6461045, 'per_record_ms': 0.05697038281}}, 'n': 300, 'record_bytes': 512}
    rows = [('write()', a['write'], 'neutral', '0.024 ms'), ('fsync()', a['fsync'], 'compute', '0.043 ms'), ('F_FULLFSYNC', a['full'], 'danger', '3.0 ms')]
    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    n = len(rows)
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(v * 1.25, y, txt, va='center', fontsize=15, fontweight='bold', color=C[ck])
        ax.text(0.0045, y, lab, va='center', ha='right', fontsize=15, fontweight='bold', color=C['text'], family='monospace')
    ax.set_xscale('log')
    ax.set_xlim(0.005, 60)
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(['0.01', '0.1', '1', '10'])
    ax.minorticks_off()
    ax.tick_params(labelsize=13)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Median time to return, ms (log scale)', fontsize=13)
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    sp.save(fig, OUT / 'acks.svg')

def replica():
    """Completeness rule: a lagging replica cannot prove non-dispatch."""
    for k in (1, 2):
        s = Svg(900, 420)
        s.text(10, 40, 'Authoritative log', 24, INK, anchor='start', weight='bold')
        s.text(10, 200, 'Asynchronous replica', 24, INK, anchor='start', weight='bold')
        for i in range(7):
            x = 10 + i * 100
            s.rect(x, 60, 92, 70, 'sys', rx=4)
            s.text(x + 46, 104, f'e{i + 1}', 23, VIOLET, weight='bold')
            if i < 6:
                s.rect(x, 220, 92, 70, 'sys', rx=4)
                s.text(x + 46, 264, f'e{i + 1}', 23, VIOLET, weight='bold')
        s.rect(610, 60, 92, 70, 'tool', rx=4, sw=3)
        s.text(656, 104, 'intent', 22, ROLE['tool'][2], weight='bold')
        s.rect(610, 220, 92, 70, 'white', dash=True, rx=4)
        s.text(656, 264, '?', 26, GRAY, weight='bold')
        s.arrow(656, 134, 656, 214, 'gray', dash=True)
        s.text(670, 182, 'still in flight', 22, GRAY, anchor='start')
        s.text(720, 104, 'test_r18 sent', 22, ROLE['tool'][2], anchor='start')
        if k >= 2:
            s.rect(10, 316, 880, 92, 'fault', rx=8)
            s.text(450, 352, 'Reading the replica: “no intent, so never sent”', 24, ROLE['fault'][2], weight='bold')
            s.text(450, 388, 'Dispatching again launches a second test', 23, INK)
        s.save(f'replica_step{k}')

def groupcommit():
    a = {'write': 0.024125, 'fsync': 0.0430625, 'full': 3.0078955, 'group': {1: {'flush_ms_median': 3.053792, 'per_record_ms': 3.053792}, 2: {'flush_ms_median': 3.0020835, 'per_record_ms': 1.50104175}, 4: {'flush_ms_median': 3.04875, 'per_record_ms': 0.7621875}, 8: {'flush_ms_median': 3.071854, 'per_record_ms': 0.38398175}, 16: {'flush_ms_median': 3.048104, 'per_record_ms': 0.1905065}, 32: {'flush_ms_median': 3.554021, 'per_record_ms': 0.1110631563}, 64: {'flush_ms_median': 3.6461045, 'per_record_ms': 0.05697038281}}, 'n': 300, 'record_bytes': 512}
    ks = sorted(a['group'])
    per = [a['group'][k]['per_record_ms'] for k in ks]
    flush = [a['group'][k]['flush_ms_median'] for k in ks]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.plot(ks, flush, 'o-', color=C['danger'], lw=2.4, ms=7, mec='white')
    ax.plot(ks, per, 'o-', color=C['memory'], lw=2.4, ms=7, mec='white')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks([0.05, 0.1, 0.5, 1, 5])
    ax.set_yticklabels(['0.05', '0.1', '0.5', '1', '5'])
    ax.minorticks_off()
    ax.set_ylim(0.03, 9)
    ax.tick_params(labelsize=12)
    ax.text(ks[-1], flush[-1] * 1.35, 'every record waits the flush', ha='right', fontsize=12.5, fontweight='bold', color=C['danger'])
    ax.text(ks[0], 0.075, 'storage work per record:\n3.1 ms to 0.057 ms', ha='left', va='center', fontsize=12.5, fontweight='bold', color=C['memory'])
    ax.set_xlabel('Records per flush (batch size)')
    ax.set_ylabel('ms')
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    sp.save(fig, OUT / 'groupcommit.svg')

def ckpt():
    """A checkpoint binds record and workspace to one log position c."""
    for k in (1, 2, 3):
        s = Svg(900, 450)
        s.text(10, 40, 'Log', 24, INK, anchor='start', weight='bold')
        for i in range(13):
            x = 70 + i * 62
            s.rect(x, 18, 54, 40, 'sys' if i <= 7 else 'white', rx=3, sw=1.5, stroke=None if i <= 7 else VIOLET)
        cx = 70 + 7 * 62 + 58
        s.line(cx, 6, cx, 300, VIOLET, 4)
        s.text(cx, 330, 'position c', 23, VIOLET, weight='bold')
        s.text(cx + 214, 92, 'suffix: folded at restore', 22, VIOLET)
        s.arrow(cx + 8, 70, cx + 330, 70, 'violet')
        s.box(40, 110, 300, 90, 'memory', 'record_R_ckpt', 'reservations, operations', fs=24, sfs=22, mono_title=True)
        s.box(40, 220, 300, 90, 'tool', 'ws_r18', 'workspace snapshot', fs=24, sfs=22, mono_title=True)
        s.arrow(342, 155, cx - 6, 155, 'gray')
        s.arrow(342, 265, cx - 6, 265, 'gray')
        if k >= 2:
            s.rect(380, 110, 150, 200, 'white', rx=8, sw=2)
            s.text(455, 146, 'manifest', 23, INK, weight='bold')
            for i, lab in enumerate(('position c', 'hashes', 'reducer v3')):
                s.text(455, 190 + i * 36, lab, 22, GRAY)
            s.num(612, 210, 1)
            s.text(640, 202, 'bytes durable first,', 22, INK, anchor='start')
            s.text(640, 232, 'then publish manifest', 22, INK, anchor='start')
        if k >= 3:
            s.rect(40, 360, 820, 80, 'fault', rx=8)
            s.cross(80, 400, 16)
            s.text(110, 392, 'Torn checkpoint: record at c, files copied seconds later', 23, ROLE['fault'][2], anchor='start', weight='bold')
            s.text(110, 424, 'restores a state that never existed', 22, INK, anchor='start')
        s.save(f'ckpt_step{k}')

def snapshots():
    ks, full, delta = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100], [0.0, 0.00064, 0.00192, 0.00384, 0.0064, 0.0096, 0.01344, 0.01792, 0.02304, 0.0288, 0.0352, 0.04224, 0.04992, 0.05824, 0.0672, 0.0768, 0.08704, 0.09792, 0.10944, 0.1216, 0.1344, 0.14784, 0.16192, 0.17664, 0.192, 0.208, 0.22464, 0.24192, 0.25984, 0.2784, 0.2976, 0.31744, 0.33792, 0.35904, 0.3808, 0.4032, 0.42624, 0.44992, 0.47424, 0.4992, 0.5248, 0.55104, 0.57792, 0.60544, 0.6336, 0.6624, 0.69184, 0.72192, 0.75264, 0.784, 0.816, 0.84864, 0.88192, 0.91584, 0.9504, 0.9856, 1.02144, 1.05792, 1.09504, 1.1328, 1.1712, 1.21024, 1.24992, 1.29024, 1.3312, 1.3728, 1.41504, 1.45792, 1.50144, 1.5456, 1.5904, 1.63584, 1.68192, 1.72864, 1.776, 1.824, 1.87264, 1.92192, 1.97184, 2.0224, 2.0736, 2.12544, 2.17792, 2.23104, 2.2848, 2.3392, 2.39424, 2.44992, 2.50624, 2.5632, 2.6208, 2.67904, 2.73792, 2.79744, 2.8576, 2.9184, 2.97984, 3.04192, 3.10464, 3.168, 3.232], [0.0, 0.00064, 0.00128, 0.00192, 0.00256, 0.0032, 0.00384, 0.00448, 0.00512, 0.00576, 0.0064, 0.00704, 0.00768, 0.00832, 0.00896, 0.0096, 0.01024, 0.01088, 0.01152, 0.01216, 0.0128, 0.01344, 0.01408, 0.01472, 0.01536, 0.016, 0.01664, 0.01728, 0.01792, 0.01856, 0.0192, 0.01984, 0.02048, 0.02112, 0.02176, 0.0224, 0.02304, 0.02368, 0.02432, 0.02496, 0.0256, 0.02624, 0.02688, 0.02752, 0.02816, 0.0288, 0.02944, 0.03008, 0.03072, 0.03136, 0.032, 0.03264, 0.03328, 0.03392, 0.03456, 0.0352, 0.03584, 0.03648, 0.03712, 0.03776, 0.0384, 0.03904, 0.03968, 0.04032, 0.04096, 0.0416, 0.04224, 0.04288, 0.04352, 0.04416, 0.0448, 0.04544, 0.04608, 0.04672, 0.04736, 0.048, 0.04864, 0.04928, 0.04992, 0.05056, 0.0512, 0.05184, 0.05248, 0.05312, 0.05376, 0.0544, 0.05504, 0.05568, 0.05632, 0.05696, 0.0576, 0.05824, 0.05888, 0.05952, 0.06016, 0.0608, 0.06144, 0.06208, 0.06272, 0.06336, 0.064])
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    ax.plot(ks, full, color=C['danger'], lw=2.8)
    ax.plot(ks, delta, color=C['neutral'], lw=2.8)
    ax.text(3, 3.2, 'full snapshots: 3.2 GB', ha='left', va='center', fontsize=13, fontweight='bold', color=C['danger'])
    ax.text(ks[-1], delta[-1] + 0.15, 'deltas: 64 MB', ha='right', va='bottom', fontsize=13, fontweight='bold', color=C['neutral'])
    ax.set_xlabel('Checkpoints taken (one every 5 turns)')
    ax.set_ylabel('Bytes written (GB)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3.6)
    ax.tick_params(labelsize=12)
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    sp.save(fig, OUT / 'snapshots.svg')

def cadence():
    k = {'s_log': 22.36067977, 's_unlog': 1.58113883}
    st = np.logspace(0, np.log10(80), 300)
    log_c = [0.501, 0.4937403951, 0.4865868411, 0.4795378014, 0.4725917619, 0.4657472308, 0.4590027378, 0.4523568343, 0.4458080928, 0.4393551068, 0.4329964901, 0.4267308771, 0.4205569218, 0.4144732983, 0.4084786997, 0.4025718386, 0.3967514462, 0.3910162722, 0.385365085, 0.3797966705, 0.3743098328, 0.3689033934, 0.363576191, 0.3583270814, 0.3531549371, 0.3480586471, 0.343037117, 0.338089268, 0.3332140374, 0.3284103781, 0.3236772583, 0.3190136613, 0.3144185855, 0.3098910439, 0.3054300639, 0.3010346876, 0.2967039706, 0.2924369829, 0.2882328079, 0.2840905427, 0.2800092975, 0.2759881956, 0.2720263734, 0.26812298, 0.2642771769, 0.260488138, 0.2567550496, 0.2530771098, 0.2494535285, 0.2458835276, 0.2423663401, 0.2389012107, 0.235487395, 0.2321241599, 0.2288107828, 0.2255465521, 0.2223307668, 0.219162736, 0.2160417793, 0.2129672264, 0.2099384169, 0.2069547002, 0.2040154354, 0.2011199913, 0.1982677458, 0.1954580864, 0.1926904096, 0.1899641208, 0.1872786346, 0.1846333741, 0.1820277712, 0.1794612661, 0.1769333076, 0.1744433527, 0.1719908666, 0.1695753226, 0.1671962018, 0.1648529931, 0.1625451934, 0.1602723068, 0.1580338452, 0.1558293278, 0.1536582811, 0.1515202388, 0.1494147416, 0.1473413373, 0.1452995806, 0.1432890328, 0.1413092622, 0.1393598435, 0.137440358, 0.1355503934, 0.1336895438, 0.1318574094, 0.1300535968, 0.1282777185, 0.126529393, 0.1248082448, 0.1231139044, 0.1214460076, 0.1198041963, 0.1181881178, 0.116597425, 0.1150317763, 0.1134908353, 0.1119742711, 0.110481758, 0.1090129752, 0.1075676075, 0.1061453442, 0.10474588, 0.1033689143, 0.1020141512, 0.1006812998, 0.0993700738, 0.09808019157, 0.09681137605, 0.09556335472, 0.0943358595, 0.09312862674, 0.09194139714, 0.0907739157, 0.08962593165, 0.08849719841, 0.08738747355, 0.0862965187, 0.08522409954, 0.08416998572, 0.08313395083, 0.08211577233, 0.08111523154, 0.08013211354, 0.07916620718, 0.07821730497, 0.07728520312, 0.0763697014, 0.07547060319, 0.07458771536, 0.07372084827, 0.07286981573, 0.07203443495, 0.0712145265, 0.07040991426, 0.06962042541, 0.06884589039, 0.06808614281, 0.06734101951, 0.06661036043, 0.06589400863, 0.06519181026, 0.06450361447, 0.06382927346, 0.06316864238, 0.06252157934, 0.06188794535, 0.06126760432, 0.06066042299, 0.06006627096, 0.0594850206, 0.05891654707, 0.05836072826, 0.0578174448, 0.05728657998, 0.05676801979, 0.05626165284, 0.05576737036, 0.0552850662, 0.05481463675, 0.05435598098, 0.05390900036, 0.0534735989, 0.05304968306, 0.0526371618, 0.05223594652, 0.05184595103, 0.05146709156, 0.05109928675, 0.05074245758, 0.05039652743, 0.05006142198, 0.04973706925, 0.04942339958, 0.0491203456, 0.04882784221, 0.04854582659, 0.04827423815, 0.04801301857, 0.04776211175, 0.04752146377, 0.04729102297, 0.04707073983, 0.04686056705, 0.04666045949, 0.04647037415, 0.04629027022, 0.04612010901, 0.04595985397, 0.04580947067, 0.04566892682, 0.04553819223, 0.04541723882, 0.04530604061, 0.04520457372, 0.04511281635, 0.04503074878, 0.04495835341, 0.04489561466, 0.04484251908, 0.04479905525, 0.04476521384, 0.04474098759, 0.04472637128, 0.04472136177, 0.044725958, 0.04474016095, 0.04476397366, 0.04479740126, 0.04484045093, 0.0448931319, 0.0449554555, 0.04502743512, 0.0451090862, 0.0452004263, 0.04530147503, 0.04541225409, 0.04553278728, 0.04566310049, 0.04580322171, 0.04595318103, 0.04611301067, 0.04628274495, 0.04646242033, 0.04665207541, 0.04685175092, 0.04706148975, 0.04728133695, 0.04751133973, 0.04775154751, 0.04800201188, 0.04826278663, 0.04853392778, 0.04881549357, 0.04910754446, 0.0494101432, 0.04972335478, 0.05004724647, 0.05038188784, 0.05072735077, 0.05108370947, 0.05145104046, 0.05182942267, 0.05221893734, 0.05261966816, 0.0530317012, 0.05345512494, 0.05389003035, 0.05433651084, 0.0547946623, 0.05526458315, 0.05574637431, 0.05624013927, 0.05674598409, 0.05726401742, 0.05779435053, 0.05833709732, 0.05889237438, 0.05946030097, 0.06004099908, 0.06063459344, 0.06124121154, 0.06186098368, 0.06249404298, 0.06314052542, 0.06380056986, 0.06447431807, 0.06516191476, 0.06586350762, 0.06657924734, 0.06730928767, 0.06805378541, 0.06881290046, 0.06958679588, 0.0703756379, 0.07117959595, 0.07199884271, 0.07283355416, 0.07368390957, 0.0745500916, 0.07543228629, 0.07633068313, 0.07724547509, 0.07817685866, 0.07912503389, 0.08009020444, 0.08107257762, 0.08207236444, 0.08308977963, 0.08412504173, 0.0851783731, 0.08625]
    unl_c = [0.7, 0.6956783371, 0.6915060994, 0.6874823908, 0.683606347, 0.6798771354, 0.6762939552, 0.6728560365, 0.6695626411, 0.6664130615, 0.6634066212, 0.6605426744, 0.6578206061, 0.6552398314, 0.6527997962, 0.6504999763, 0.6483398777, 0.6463190365, 0.6444370186, 0.6426934198, 0.6410878655, 0.6396200109, 0.6382895407, 0.6370961691, 0.6360396399, 0.635119726, 0.63433623, 0.6336889834, 0.6331778474, 0.632802712, 0.6325634968, 0.6324601503, 0.6324926503, 0.6326610039, 0.6329652471, 0.6334054454, 0.6339816933, 0.6346941145, 0.6355428621, 0.6365281183, 0.6376500949, 0.6389090328, 0.6403052024, 0.6418389035, 0.6435104657, 0.6453202479, 0.6472686389, 0.6493560572, 0.6515829511, 0.6539497989, 0.6564571091, 0.6591054201, 0.6618953008, 0.6648273504, 0.6679021988, 0.6711205063, 0.6744829643, 0.6779902949, 0.6816432515, 0.6854426187, 0.6893892126, 0.6934838809, 0.697727503, 0.7021209905, 0.706665287, 0.7113613687, 0.7162102441, 0.7212129549, 0.7263705754, 0.7316842136, 0.7371550107, 0.7427841418, 0.7485728161, 0.7545222768, 0.7606338019, 0.766908704, 0.7733483309, 0.7799540659, 0.7867273277, 0.7936695712, 0.8007822875, 0.8080670044, 0.8155252866, 0.8231587359, 0.8309689921, 0.8389577327, 0.8471266736, 0.8554775694, 0.8640122138, 0.87273244, 0.8816401209, 0.89073717, 0.9000255411, 0.9095072292, 0.9191842711, 0.9290587451, 0.9391327723, 0.9494085165, 0.9598881847, 0.970574028, 0.9814683416, 0.9925734654, 1.003891785, 1.015425731, 1.02717778, 1.039150459, 1.051346337, 1.063768034, 1.076418219, 1.089299608, 1.102414969, 1.115767118, 1.129358924, 1.143193305, 1.157273233, 1.171601734, 1.186181883, 1.201016813, 1.21610971, 1.231463816, 1.24708243, 1.262968905, 1.279126653, 1.295559147, 1.312269914, 1.329262545, 1.346540689, 1.364108057, 1.381968423, 1.400125623, 1.418583557, 1.437346189, 1.45641755, 1.475801736, 1.49550291, 1.515525304, 1.535873219, 1.556551026, 1.577563165, 1.598914149, 1.620608566, 1.642651074, 1.665046409, 1.687799379, 1.710914874, 1.734397857, 1.758253372, 1.782486544, 1.807102577, 1.83210676, 1.857504461, 1.883301137, 1.909502329, 1.936113664, 1.963140858, 1.990589717, 2.018466135, 2.046776102, 2.075525697, 2.104721095, 2.134368568, 2.164474483, 2.195045307, 2.226087607, 2.257608049, 2.289613405, 2.322110548, 2.355106458, 2.388608224, 2.422623041, 2.457158214, 2.492221162, 2.527819416, 2.563960622, 2.600652542, 2.637903059, 2.675720172, 2.714112006, 2.753086805, 2.792652942, 2.832818914, 2.87359335, 2.914985006, 2.957002775, 2.99965568, 3.042952882, 3.086903683, 3.131517522, 3.176803982, 3.222772789, 3.269433819, 3.316797092, 3.364872782, 3.413671215, 3.463202873, 3.513478395, 3.564508579, 3.616304387, 3.668876942, 3.722237539, 3.776397637, 3.831368871, 3.887163046, 3.943792148, 4.00126834, 4.059603967, 4.118811559, 4.178903833, 4.239893697, 4.301794251, 4.364618789, 4.428380808, 4.493094001, 4.558772269, 4.625429718, 4.693080667, 4.761739646, 4.831421402, 4.902140902, 4.973913336, 5.046754121, 5.120678901, 5.195703556, 5.271844199, 5.349117185, 5.427539111, 5.507126823, 5.587897414, 5.669868233, 5.753056887, 5.837481243, 5.923159437, 6.01010987, 6.098351218, 6.187902435, 6.278782757, 6.371011702, 6.464609081, 6.559594997, 6.655989854, 6.753814355, 6.853089512, 6.95383665, 7.056077406, 7.159833742, 7.265127944, 7.371982627, 7.480420744, 7.590465585, 7.702140787, 7.815470337, 7.930478578, 8.047190211, 8.165630306, 8.285824302, 8.407798016, 8.531577646, 8.65718978, 8.784661397, 8.914019878, 9.045293007, 9.17850898, 9.313696412, 9.450884339, 9.590102228, 9.731379981, 9.874747944, 10.02023691, 10.16787813, 10.31770332, 10.46974465, 10.62403478, 10.78060686, 10.93949452, 11.10073187, 11.26435356, 11.43039472, 11.59889103, 11.76987868, 11.94339438, 12.11947541, 12.29815959, 12.4794853, 12.66349149, 12.85021768, 13.03970398, 13.23199108, 13.42712029, 13.62513352, 13.82607331, 14.0299828, 14.23690581, 14.44688676, 14.65997078, 14.87620362, 15.09563173, 15.31830224, 15.54426298, 15.77356248, 16.00625]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(st, unl_c, color=C['danger'], lw=2.6)
    ax.plot(st, log_c, color=C['memory'], lw=2.6)
    for x, curve_per, ck in ((k['s_unlog'], Q_(20.0, 's'), 'danger'), (k['s_log'], Q_(0.1, 's'), 'memory')):
        y = {20.0: 0.632455532, 0.1: 0.04472135955}[curve_per.magnitude]
        ax.plot([x], [y], 'o', ms=10, color=C[ck], mec='white', mew=1.5, zorder=5)
    ax.text(0.03, 0.96, 'turns re-executed: optimum 1.6', transform=ax.transAxes, va='top', fontsize=13.5, fontweight='bold', color=C['danger'])
    ax.annotate('turns logged: optimum 22', (k['s_log'], 0.04472135955), xytext=(0, -32), textcoords='offset points', ha='center', fontsize=13.5, fontweight='bold', color=C['memory'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1, 2, 5, 10, 25, 50])
    ax.set_xticklabels(['1', '2', '5', '10', '25', '50'])
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.set_yticklabels(['0.01', '0.1', '1', '10'])
    ax.minorticks_off()
    ax.set_ylim(0.008, 30)
    ax.tick_params(labelsize=13)
    ax.set_xlabel('Checkpoint stride (turns)', fontsize=14)
    ax.set_ylabel('Overhead (s per turn)', fontsize=14)
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    sp.save(fig, OUT / 'cadence.svg')

def resume():
    """Resume in stages: prove, own, restore, reconcile, admit."""
    stages = [('Prove', 'full log', 'struct'), ('Own', 'epoch 7 to 8', 'sys'), ('Restore', 'fold + ws_r18', 'memory'), ('Reconcile', 'test_r18', 'tool'), ('Admit', 'new work', 'model')]
    show = {1: 2, 2: 4, 3: 5}
    for k in (1, 2, 3):
        s = Svg(900, 330)
        w, gap = (164, 17)
        for i, (a, b, role) in enumerate(stages):
            x = 8 + i * (w + gap)
            on = i < show[k]
            s.box(x, 40, w, 110, role, a, b, fs=27, sfs=21, opacity=1.0 if on else 0.25, title_color=INK if role == 'struct' else None)
            s.num(x + 20, 40, i + 1, VIOLET if on else '#cbd5e1')
            if i < 4:
                s.arrow(x + w + 1, 95, x + w + gap - 1, 95, 'gray')
        cx = [8 + i * (w + gap) + w / 2 for i in range(5)]
        notes = [('replica lags:', 'say unavailable'), ('stale writes', 'are rejected'), ('no model call,', 'no tool call'), ('probe by id,', 'never resend'), ('recheck limits,', 'then prefill')]
        for i, (a, b) in enumerate(notes):
            if i < show[k]:
                s.text(cx[i], 200, a, 22, GRAY)
                s.text(cx[i], 230, b, 22, GRAY)
        if k >= 3:
            s.rect(8, 268, 884, 54, 'sys', rx=8)
            s.text(450, 304, 'Missing evidence or ownership: hold the trajectory, do not guess', 23, ROLE['sys'][2], weight='bold')
        s.save(f'resume_step{k}')

def fence():
    """Fencing: the resource checks the owner epoch on every write."""
    for k in (1, 2, 3):
        s = Svg(900, 400)
        s.text(10, 48, 'Worker A', 24, INK, anchor='start', weight='bold')
        s.text(10, 168, 'Worker B', 24, INK, anchor='start', weight='bold')
        s.text(10, 310, 'Log or gateway', 24, INK, anchor='start', weight='bold')
        s.text(10, 340, '(checks epoch)', 22, GRAY, anchor='start')
        s.line(200, 40, 890, 40, '#cbd5e1', 3)
        s.line(200, 160, 890, 160, '#cbd5e1', 3)
        s.rect(200, 280, 690, 70, 'sys', rx=8)
        s.box(210, 18, 150, 44, 'sys', 'epoch 7', fs=22)
        s.rect(380, 22, 260, 36, 'fault', dash=True, rx=6)
        s.text(510, 48, 'stalled, lease expires', 22, ROLE['fault'][2])
        if k >= 2:
            s.box(500, 138, 150, 44, 'sys', 'epoch 8', fs=22)
            s.arrow(575, 184, 575, 276, 'violet', sw=3)
            s.text(562, 236, 'write, epoch 8: accepted', 22, ROLE['sys'][2], anchor='end')
            s.check(575, 316, 14)
        if k >= 3:
            s.arrow(740, 62, 740, 276, 'red', sw=3)
            s.text(752, 110, 'late write,', 22, ROLE['fault'][2], anchor='start')
            s.text(752, 136, 'epoch 7', 22, ROLE['fault'][2], anchor='start')
            s.cross(740, 316, 14)
            s.text(766, 324, 'rejected', 22, ROLE['fault'][2], anchor='start', weight='bold')
        s.save(f'fence_step{k}')

def prefill():
    p = {'tokens': 10202, 'cold': 8.696280956, 'warm': 0.120057106, 'load': 0.778227584, 'n': 3}
    rows = [('Same server,\nprefix held', p['warm'], 'memory', '0.12 s'), ('Model reloaded,\nfull prefill', p['cold'], 'danger', '8.7 s')]
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(v + 0.15, y, txt, va='center', fontsize=15, fontweight='bold', color=C[ck])
        ax.text(-0.15, y, lab, va='center', ha='right', fontsize=13.5, fontweight='bold', color=C['text'])
    ax.set_xlim(0, 11)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.set_xlabel('Time to the first token of the next call, s (10,202 tokens)', fontsize=12)
    sp.save(fig, OUT / 'prefill.svg')

def feynman():
    """1,000 temperature-0 completions split at token 103 (Thinking Machines).
    Drawn for a column slot (about 7.7 by 3.4 cm)."""
    s = Svg(600, 270)
    s.rect(4, 82, 236, 106, 'model', rx=8)
    s.text(122, 116, 'tokens 1 to 102', 24, ROLE['model'][2], weight='bold')
    s.text(122, 146, 'identical', 24, ROLE['model'][2], weight='bold')
    s.text(122, 174, '“...1918, in”', 22, INK)
    s.arrow(242, 118, 318, 66, 'gray', sw=3)
    s.arrow(242, 152, 318, 204, 'gray', sw=3)
    s.rect(322, 10, 274, 104, 'memory', rx=8)
    s.text(459, 50, '“Queens, New York”', 24, ROLE['memory'][2], weight='bold')
    s.text(459, 88, '992 of 1,000', 24, INK)
    s.rect(322, 156, 274, 104, 'fault', rx=8)
    s.text(459, 196, '“New York City”', 24, ROLE['fault'][2], weight='bold')
    s.text(459, 234, '8 of 1,000', 24, INK)
    s.save('feynman')

def replay():
    """Live execution vs replay (DE figure 3), redrawn for the slide."""
    nodes = [('Harness', 'records', 'struct'), ('Model', 'sampled', 'model'), ('Gate', 'authorizes', 'struct'), ('Tools', 'real effects', 'tool')]
    log = ['ContextBuilt', 'ModelReplied', 'ToolIntent', 'ToolResult']
    global FLOOR
    FLOOR = 26
    for k in (1, 2):
        s = Svg(900, 490)
        w = 190
        xs = [10 + i * 230 for i in range(4)]
        s.text(10, 30, 'Live', 24, INK, anchor='start', weight='bold')
        for i, (a, b, role) in enumerate(nodes):
            s.box(xs[i], 44, w, 84, role, a, b, fs=28, sfs=26, title_color=INK if role == 'struct' else None)
            if i < 3:
                s.arrow(xs[i] + w + 1, 86, xs[i + 1] - 1, 86, 'gray')
            s.arrow(xs[i] + w / 2, 130, xs[i] + w / 2, 160, 'violet')
        for i, lab in enumerate(log):
            s.rect(xs[i], 162, w + (40 if i < 3 else 0), 44, 'sys', rx=0)
            s.text(xs[i] + w / 2, 194, lab, 26, INK)
        if k >= 2:
            s.line(10, 232, 890, 232, '#cbd5e1', 2)
            s.text(10, 270, 'Replay', 24, INK, anchor='start', weight='bold')
            for i, lab in enumerate(log):
                s.rect(xs[i], 284, w + (40 if i < 3 else 0), 44, 'sys', rx=0)
                s.text(xs[i] + w / 2, 316, lab, 26, INK)
            states = [('Harness', 'replaying', 'struct', False), ('Model', 'stubbed', 'white', True), ('Gate', 'authorizes', 'struct', False), ('Tools', 'disconnected', 'white', True)]
            for i, (a, b, role, dash) in enumerate(states):
                s.box(xs[i], 396, w, 84, role, a, b, fs=28, sfs=26, dash=dash, title_color=GRAY if dash else INK)
            s.arrow(0, 0, 0, 0, 'violet', sw=3, path='M780,330 L780,346 L105,346 L105,394')
            s.text(240, 382, 'recorded replies stream in, in order', 26, VIOLET, anchor='start')
            s.line(xs[2] + w + 4, 438, xs[3] - 6, 438, RED, 4)
            s.line(xs[3] - 6, 420, xs[3] - 6, 456, RED, 5)
        s.save(f'replay_step{k}')
    FLOOR = 22

def diverge():
    """Two runs compared boundary by boundary; the first mismatch localizes."""
    bounds = ['context', 'model', 'decision', 'tool']
    for k in (1, 2):
        s = Svg(900, 400)
        s.text(10, 120, 'Original', 24, INK, anchor='start', weight='bold')
        s.text(10, 240, 'Replay', 24, INK, anchor='start', weight='bold')
        s.text(10, 270, 'new harness', 22, GRAY, anchor='start')
        x0 = 180
        cols = []
        for t, turn in enumerate(('turn 18', 'turn 19')):
            s.text(x0 + t * 350 + 160, 40, turn, 23, GRAY, weight='bold')
            for j, b in enumerate(bounds):
                x = x0 + t * 350 + j * 82
                cols.append((t, j, x))
                s.text(x + 36, 78, b, 21, GRAY, floor=21)
                s.rect(x, 96, 72, 40, 'sys', rx=4)
        for t, j, x in cols:
            idx = t * 4 + j
            first = t == 1 and j == 0
            if idx < 4:
                role = 'sys'
            elif first:
                role = 'fault'
            else:
                role = 'white'
            s.rect(x, 216, 72, 40, role, rx=4, dash=role == 'white')
            if idx < 4 or (k >= 1 and idx < 4):
                s.check(x + 36, 182, 11)
        if k >= 1:
            s.rect(x0 + 350, 216, 72, 40, 'fault', rx=4, sw=3)
            s.cross(x0 + 350 + 36, 182, 11)
        if k >= 2:
            s.text(x0 + 330, 306, 'first mismatch: compaction dropped', 22, ROLE['fault'][2], weight='bold', anchor='start')
            s.text(x0 + 330, 336, '“regression suite omitted”', 22, ROLE['fault'][2], anchor='start')
            s.text(x0 + 350 + 164 + 36, 190, 'may be consequences', 22, GRAY)
        s.save(f'diverge_step{k}')

def retention():
    t = {'log_mb': 3.96, 'snap_mb': 50}
    rows = [('Log (30 turns)', t['log_mb'], 'network', '3.96 MB'), ('Workspace snapshot', t['snap_mb'], 'compute', '50 MB')]
    fig, ax = plt.subplots(figsize=(5.8, 1.8))
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(v + 1, y, txt, va='center', fontsize=14, fontweight='bold', color=C[ck])
        ax.text(-1, y, lab, va='center', ha='right', fontsize=14, fontweight='bold', color=C['text'])
    ax.set_xlim(0, 65)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.set_xlabel('Bytes kept for recovery and replay (MB)', fontsize=12)
    sp.save(fig, OUT / 'retention.svg')

def thumbs():
    s = Svg(260, 100)
    for i in range(5):
        s.rect(6 + i * 34, 60, 30, 30, 'sys', rx=2, sw=1.5)
    s.arrow(90, 56, 90, 34, 'violet', sw=3)
    s.rect(60, 4, 60, 28, 'memory', rx=4, sw=1.5)
    s.rect(190, 60, 30, 30, 'white', dash=True, rx=2, sw=1.5)
    s.save('thumb_log')
    s = Svg(260, 100)
    for i in range(5):
        s.rect(6 + i * 52, 30, 40, 40, 'tool' if i == 3 else 'sys', rx=4, sw=1.5)
        if i < 4:
            s.arrow(47 + i * 52, 50, 56 + i * 52, 50, 'gray', sw=2)
    s.line(152, 8, 152, 92, VIOLET, 3, dash=True)
    s.save('thumb_intent')
    s = Svg(260, 100)
    pts = ' '.join((f'{20 + i * 11},{82 - 0.68 * (i - 10) ** 2:.1f}' for i in range(21)))
    s.add(f'<polyline points="{pts}" fill="none" stroke="{GREEN}" stroke-width="4"/>')
    s.dot(130, 82, 7, GREEN)
    s.line(10, 94, 250, 94, GRAY, 2)
    s.save('thumb_cadence')
    s = Svg(260, 100)
    for j in range(6):
        s.rect(8 + j * 41, 14, 34, 28, 'sys', rx=3, sw=1.5)
        s.rect(8 + j * 41, 58, 34, 28, 'sys' if j < 3 else 'fault' if j == 3 else 'white', rx=3, sw=1.5, dash=j > 3)
    s.save('thumb_replay')
if __name__ == '__main__':
    restarts()
    lost()
    fold()
    intent()
    crashrun()
    acks()
    replica()
    groupcommit()
    ckpt()
    snapshots()
    cadence()
    resume()
    fence()
    prefill()
    feynman()
    replay()
    diverge()
    retention()
    thumbs()
    print('figures written')
