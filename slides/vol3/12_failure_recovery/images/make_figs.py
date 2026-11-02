#!/usr/bin/env python3
"""Figures for Lecture 12 (Failure Recovery), Volume III.

Every value comes from calcs.py, which reads MLSysIM; this script imports it
and prints the recorded values (val, num), so figures and slides agree. "FR" is the rendered chapter, Failure Recovery. Diagrams
redraw the chapter's figures at slide size (the book SVGs are page-sized and
fall below the legibility floor on a slide):
  - saga_step*: FR figure 1 (vol3-saga-flow.svg), split into three builds.
  - decide_step*: FR figure 2 (recovery-decision-tree.svg).
  - digest: FR figure 3 (semantic-watchdog-oscillation.svg), top panel.
  - watch_resp: FR figure 3, bottom panel, and table 6.
  - reorder_step*: FR table 4 drawn as two plans.
  - cleanup_step*: FR Exercise 1 durations (hypothetical givens).
Plots use slideplot. Thumbnails (thumb_*) carry no text.

Run from this folder: python3 make_figs.py   (then make deck converts SVG)
"""
from types import SimpleNamespace
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
C = sp.C
plt = sp.plt
out = Path(__file__).parent
P = PALETTE = {'store': ('#f3e8ff', '#7c3aed', '#6d28d9'), 'tool': ('#fef3c7', '#d97706', '#b45309'), 'fault': ('#fee2e2', '#dc2626', '#b91c1c'), 'ok': ('#dcfce7', '#16a34a', '#15803d'), 'think': ('#e0f2fe', '#0284c7', '#0369a1'), 'plain': ('#ffffff', '#94a3b8', '#0f172a'), 'band': ('#f8fafc', '#cbd5e1', '#0f172a'), 'gray': ('#f1f5f9', '#cbd5e1', '#94a3b8')}
FONT = 'Helvetica, Arial, sans-serif'
MONO = 'Courier New, Courier, monospace'

class Svg:

    def __init__(self, w, h):
        self.w, self.h, self.items = (w, h, [])

    def add(self, s, show=True):
        if show:
            self.items.append(s)
        else:
            self.items.append(f'<g opacity="0">{s}</g>')

    def box(self, x, y, w, h, title, sub=None, role='plain', show=True, dash=False, mono_sub=False, tsize=21, ssize=18, bold=True, rx=6):
        fill, stroke, text = P[role]
        d = ' stroke-dasharray="7,5"' if dash else ''
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2.2"{d}/>'
        cx = x + w / 2
        if sub is None:
            s += self._t(cx, y + h / 2 + tsize * 0.35, title, tsize, text, bold=bold)
        else:
            s += self._t(cx, y + h / 2 - 4, title, tsize, text, bold=bold)
            s += self._t(cx, y + h / 2 + ssize + 2, sub, ssize, '#334155', mono=mono_sub)
        self.add(s, show)

    def _t(self, x, y, s, size, color, bold=False, anchor='middle', mono=False, italic=False):
        fam = MONO if mono else FONT
        w = ' font-weight="bold"' if bold else ''
        it = ' font-style="italic"' if italic else ''
        return f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{w}{it}>{s}</text>'

    def text(self, x, y, s, size=19, color='#0f172a', show=True, **kw):
        self.add(self._t(x, y, s, size, color, **kw), show)

    def arrow(self, pts, color='#64748b', show=True, dash=False, width=2.4):
        d = ' '.join((f"{('M' if i == 0 else 'L')}{x},{y}" for i, (x, y) in enumerate(pts)))
        da = ' stroke-dasharray="7,5"' if dash else ''
        (x1, y1), (x2, y2) = (pts[-2], pts[-1])
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        L, Wd = (13, 6.5)
        bx, by = (x2 - L * math.cos(a), y2 - L * math.sin(a))
        p1 = (bx + Wd * math.sin(a), by - Wd * math.cos(a))
        p2 = (bx - Wd * math.sin(a), by + Wd * math.cos(a))
        s = f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{da}/><polygon points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>'
        self.add(s, show)

    def rect(self, x, y, w, h, fill, stroke='none', show=True, rx=0, sw=2, dash=False, opacity=1.0):
        d = ' stroke-dasharray="7,5"' if dash else ''
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>', show)

    def raw(self, s, show=True):
        self.add(s, show)

    def save(self, name):
        body = '\n'.join(self.items)
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">\n<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n'
        (out / name).write_text(svg)

def saga(k):
    s = Svg(900, 400)
    s.rect(10, 8, 880, 174, P['band'][0], P['band'][1], rx=8)
    s.text(28, 34, 'Forward operation', 20, bold=True, anchor='start')
    s.rect(10, 198, 880, 194, P['band'][0], P['band'][1], rx=8)
    s.text(28, 224, 'Registered cleanup', 20, bold=True, anchor='start')
    s.box(30, 46, 240, 68, 'Register obligation', 'target unbound', 'store')
    s.box(330, 270, 260, 70, 'Release', 'release_test_env_r18', 'tool', mono_sub=True, ssize=18, dash=k == 1)
    s.arrow([(150, 114), (150, 190), (440, 190), (440, 268)], P['store'][1])
    s.text(162, 152, 'registers first', 19, P['store'][2], anchor='start')
    sh2 = k >= 2
    s.arrow([(270, 80), (318, 80)], show=sh2)
    s.box(320, 46, 260, 68, 'Dispatch', 'provision_test_env_r18', 'tool', mono_sub=True, ssize=18, show=sh2)
    s.arrow([(580, 80), (628, 80)], show=sh2)
    s.box(630, 46, 240, 68, 'Bind on evidence', 'env_r18_01', 'store', mono_sub=True, show=sh2)
    s.arrow([(450, 114), (450, 128)], P['fault'][1], show=sh2)
    s.text(462, 128, 'reply lost', 18, P['fault'][2], anchor='start', show=sh2)
    s.box(320, 130, 260, 44, 'Outcome unknown: owned', None, 'fault', show=sh2, tsize=19, dash=True)
    s.arrow([(750, 114), (750, 190), (560, 190), (560, 268)], P['store'][1], dash=True, show=sh2)
    s.text(572, 226, 'binds target', 19, P['store'][2], anchor='start', show=sh2)
    sh3 = k >= 3
    s.box(30, 238, 240, 48, 'Test ended', None, 'plain', tsize=20, show=sh3)
    s.box(30, 294, 240, 48, 'Artifacts preserved', None, 'plain', tsize=20, show=sh3)
    s.box(30, 350, 240, 36, 'Scratch removed', None, 'gray', tsize=18, show=sh3, bold=False)
    s.arrow([(270, 262), (300, 262), (300, 292), (328, 292)], show=sh3)
    s.arrow([(270, 318), (328, 318)], show=sh3)
    s.arrow([(590, 292), (640, 292), (640, 261), (658, 261)], P['ok'][1], show=sh3)
    s.arrow([(590, 318), (640, 318), (640, 345), (658, 345)], P['fault'][1], show=sh3)
    s.box(660, 236, 212, 50, 'Closed on evidence', None, 'ok', tsize=19, show=sh3)
    s.box(660, 320, 212, 50, 'Retained, owned', None, 'fault', tsize=19, show=sh3, dash=True)
    s.save(f'saga_step{k}.svg')
for k in (1, 2, 3):
    saga(k)

def cleanup(k):
    c = {'end_test': '30', 'preserve': '20', 'release': '40', 'revoke': '5', 'scratch': '10'}
    s = Svg(900, 290)
    blocked = k >= 2
    s.box(30, 34, 240, 64, 'End the test', f"{c['end_test']} s", 'fault' if blocked else 'plain', dash=blocked)
    s.box(30, 122, 240, 64, 'Preserve artifacts', f"{c['preserve']} s", 'ok' if blocked else 'plain')
    s.box(340, 78, 240, 64, 'Release environment', f"{c['release']} s", 'gray' if blocked else 'tool')
    s.box(650, 78, 220, 64, 'Revoke credential', f"{c['revoke']} s", 'gray' if blocked else 'plain')
    s.box(30, 214, 240, 60, 'Delete scratch', f"{c['scratch']} s", 'ok' if blocked else 'plain')
    s.arrow([(270, 66), (305, 66), (305, 100), (338, 100)])
    s.arrow([(270, 154), (305, 154), (305, 120), (338, 120)])
    s.arrow([(580, 110), (648, 110)])
    s.text(300, 250, 'no dependency: runs at once', 19, '#475569', anchor='start')
    if blocked:
        s.text(150, 24, 'cancel reply lost', 19, P['fault'][2], bold=True)
        s.text(605, 180, 'blocked: waits on evidence', 20, P['fault'][2], bold=True)
        s.text(760, 30, 'settle now:', 19, P['ok'][2], bold=True)
        s.text(760, 56, 'artifacts, scratch', 19, P['ok'][2])
    s.save(f'cleanup_step{k}.svg')
for k in (1, 2):
    cleanup(k)

def decide(k):
    s = Svg(900, 400)
    s.box(250, 6, 400, 42, 'Failure evidence recorded', None, 'plain', tsize=20)
    s.arrow([(450, 48), (450, 62)])
    s.box(40, 64, 820, 60, '1 · Evidence class', 'what is known about the effect?', 'store')
    sh2, sh3 = (k >= 2, k >= 3)
    s.box(40, 142, 250, 44, 'Outcome unknown', None, 'fault', tsize=19, dash=True)
    s.box(330, 142, 530, 44, 'Rejected before dispatch, or established', None, 'plain', tsize=19)
    s.arrow([(165, 124), (165, 140)])
    s.arrow([(595, 124), (595, 140)])
    s.box(40, 204, 820, 58, '2 · Admission', 'authority · allowance and reserve · deadline', 'store', show=sh2)
    for x in (165, 420, 595, 770):
        s.arrow([(x, 186), (x, 202)], show=sh2)
    s.box(40, 280, 680, 56, '3 · Expected cost, same endpoint', 'repair first when p > C_repair / C_fallback', 'store', show=sh3)
    for x in (110, 270, 430, 590):
        s.arrow([(x, 262), (x, 278)], show=sh3)
    labels = [('Retry', 'tool'), ('New proposal', 'think'), ('Repair', 'think'), ('Compensate', 'tool')]
    for i, (t, r) in enumerate(labels):
        x = 40 + i * 170
        s.arrow([(x + 70, 336), (x + 70, 350)], show=sh3)
        s.box(x, 352, 150, 44, t, None, r, tsize=19, show=sh3)
    s.arrow([(860, 233), (882, 233), (882, 342), (810, 342), (810, 350)], P['fault'][1], show=sh2)
    s.box(740, 352, 140, 44, 'Hold, escalate', None, 'fault', tsize=18, show=sh2)
    s.text(800, 294, 'none', 18, P['fault'][2], show=sh2, anchor='middle')
    s.text(800, 316, 'admissible', 18, P['fault'][2], show=sh2, anchor='middle')
    s.save(f'decide_step{k}.svg')
for k in (1, 2, 3):
    decide(k)

def reorder(k):
    s = Svg(900, 395)
    s.text(215, 28, 'First plan', 21, bold=True)
    first = [('Publish artifact', 'fault'), ('Validate signature', 'plain'), ('Prepare notes', 'plain'), ('Notify recipients', 'fault')]
    for i, (t, r) in enumerate(first):
        s.box(40, 44 + i * 72, 350, 56, f'{i + 1}  {t}', None, r, tsize=20, bold=r == 'fault')
    s.text(215, 350, 'a failed check finds a public defect', 19, P['fault'][2])
    sh = k >= 2
    s.text(680, 28, 'Reordered plan', 21, bold=True, show=sh)
    second = [('Validate locally', 'ok'), ('Stage notes privately', 'ok'), ('Confirm approval; record duty', 'ok'), ('Publish artifact', 'fault'), ('Deliver notification', 'tool')]
    for i, (t, r) in enumerate(second):
        s.box(470, 44 + i * 62, 410, 50, f'{i + 1}  {t}', None, r, tsize=19, show=sh, bold=r == 'fault')
    s.raw('<line x1="460" y1="226" x2="890" y2="226" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="8,6"/>', sh)
    s.text(675, 382, 'checks before the pivot; one pivot per region', 19, P['ok'][2], show=sh)
    s.save(f'reorder_step{k}.svg')
for k in (1, 2):
    reorder(k)

def region():
    s = Svg(560, 400)
    s.box(20, 20, 520, 80, 'Stage change on canary', 'compensable: revert it', 'ok', tsize=23, ssize=20)
    s.arrow([(280, 100), (280, 128)])
    s.box(20, 130, 520, 54, 'Checks, approval, intent recorded', None, 'store', tsize=21)
    s.arrow([(280, 184), (280, 212)])
    s.box(20, 214, 520, 80, 'Fleet rollout and restart', 'the pivot: no undo', 'fault', tsize=23, ssize=20)
    s.arrow([(280, 294), (280, 322)])
    s.box(20, 324, 520, 66, 'Follow-up duties, owned', None, 'plain', tsize=22)
    s.save('region.svg')
region()

def digest():
    s = Svg(600, 360)
    names = ['A', 'B', 'A', 'B']
    for i, n in enumerate(names):
        x = 20 + i * 145
        role = 'think' if n == 'A' else 'plain'
        s.box(x, 20, 125, 70, f'Patch {n}', 'same failure', role, tsize=21, ssize=19)
        s.text(x + 62, 120, f"t{('' if i == 3 else ' - ' + str(3 - i))}", 19, '#475569', italic=True)
        hot = i in (1, 3)
        s.box(x + 12, 140, 101, 52, f'digest {n.lower()}', None, 'fault' if hot else 'plain', tsize=19, bold=hot)
    s.raw('<path d="M 227 192 L 227 222 L 517 222 L 517 192" fill="none" stroke="#dc2626" stroke-width="2.5"/>')
    s.text(372, 252, 'same digest two turns back: period 2', 20, P['fault'][2], bold=True)
    s.box(130, 284, 340, 62, 'Gate holds new work', None, 'store', tsize=22)
    s.save('digest.svg')
digest()

def watch_resp():
    s = Svg(900, 300)
    s.box(20, 110, 200, 80, 'Finding', 'evidence, rule', 'store')
    s.arrow([(220, 150), (268, 150)], P['store'][1])
    s.box(270, 100, 230, 100, 'Admission hold', 'new work withheld', 'store')
    s.arrow([(500, 150), (540, 150), (540, 55), (578, 55)])
    s.arrow([(540, 150), (578, 150)])
    s.arrow([(540, 150), (540, 245), (578, 245)])
    s.box(580, 20, 300, 70, 'Bounded continuation', 'a factual observation', 'think')
    s.box(580, 115, 300, 70, 'Controlled backtrack', 'restore a clean candidate', 'plain')
    s.box(580, 210, 300, 70, 'End model-directed repair', 'finalize; owe cleanup', 'fault', tsize=20)
    s.save('watch_resp.svg')
watch_resp()

def retry_tree():
    s = Svg(600, 400)
    xs = [50, 190, 330, 480]
    layers = ['agent loop', 'gateway', 'client library']
    s.rect(30, 185, 40, 40, P['fault'][1], rx=4)
    ys1 = [95, 205, 315]
    for y in ys1:
        s.raw(f'<line x1="70" y1="205" x2="{xs[1]}" y2="{y}" stroke="#a78bfa" stroke-width="2"/>')
        s.raw(f'<circle cx="{xs[1]}" cy="{y}" r="13" fill="#7c3aed"/>')
    ys2 = []
    for y in ys1:
        for d in (-30, 0, 30):
            y2 = y + d
            ys2.append(y2)
            s.raw(f'<line x1="{xs[1]}" y1="{y}" x2="{xs[2]}" y2="{y2}" stroke="#c4b5fd" stroke-width="1.6"/>')
            s.raw(f'<circle cx="{xs[2]}" cy="{y2}" r="9" fill="#7c3aed"/>')
    for y in ys2:
        for d in (-9, 0, 9):
            s.raw(f'<line x1="{xs[2]}" y1="{y}" x2="{xs[3]}" y2="{y + d}" stroke="#ddd6fe" stroke-width="1.2"/>')
            s.raw(f'<circle cx="{xs[3]}" cy="{y + d}" r="4.5" fill="#7c3aed"/>')
    for x, n in zip(xs, [1, 3, 9, 27]):
        s.text(x, 385, str(n), 26, '#0f172a', bold=True)
    for x, lab in zip(xs[1:], layers):
        s.text(x, 40, lab, 22, '#6d28d9', bold=True)
    s.save('retry_tree.svg')
retry_tree()

def incident():
    s = Svg(900, 385)
    s.box(20, 8, 260, 74, 'Supervisor', 'outside the agent', 'store', tsize=22, ssize=20)
    s.box(330, 8, 250, 74, 'Suspect trajectory', 'model work held', 'fault', dash=True, tsize=21, ssize=20)
    s.arrow([(280, 45), (328, 45)], P['store'][1])
    s.text(300, 106, 'hold enforced at every dispatch gate', 19, P['store'][2], anchor='start')
    gates = ['tool gateway', 'dispatch adapter', 'memory retrieval']
    for i, g in enumerate(gates):
        y = 120 + i * 58
        s.box(330, y, 250, 46, g, None, 'plain', tsize=20, bold=False)
        s.raw(f'<rect x="588" y="{y + 10}" width="26" height="26" fill="#dc2626" rx="3"/>')
    s.arrow([(150, 82), (150, 143), (328, 143)], P['store'][1])
    s.arrow([(150, 143), (150, 201), (328, 201)], P['store'][1])
    s.arrow([(150, 201), (150, 259), (328, 259)], P['store'][1])
    s.box(640, 8, 240, 74, 'Cleanup service', 'separate narrow grant', 'ok', tsize=21, ssize=20)
    s.arrow([(760, 82), (760, 118)], P['ok'][1])
    s.box(640, 120, 240, 56, 'release env_r18_01', None, 'tool', tsize=19)
    s.rect(640, 196, 240, 108, P['fault'][0], P['fault'][1], rx=6, sw=2.2)
    s.text(760, 226, 'Quarantine', 21, P['fault'][2], bold=True)
    s.text(760, 256, 'summary · memory', 19, '#334155')
    s.text(760, 284, 'patch · approvals', 19, '#334155')
    s.box(20, 322, 860, 54, 'Record kept: inputs, versions, open operations', None, 'band', tsize=20)
    s.save('incident.svg')
incident()

def tag(ax, text, x=0.0, y=-0.32):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10, color=C['neutral'], ha='left', va='top')
R = SimpleNamespace(break_even_cost=0.06092592593, break_even_time=0.04722863741, cost_saved_fraction=0.4390740741, expected_cost=Q_(0.09087, 'dollar'), expected_time=Q_(47.39, 's'), fallback_cost=Q_(0.162, 'dollar'), fallback_time=Q_(86.6, 's'), one_in=16, repair_cost=Q_(0.00987, 'dollar'), repair_time=Q_(4.09, 's'), time_saved_fraction=0.4527713626)
rt, rd = (0.04722863741, 0.06092592593)
p0 = 0.5
import numpy as np
fig, ax = plt.subplots(figsize=(4.5, 3.1))
ps = np.linspace(0, 1, 200)
ax.axhline(1.0, color=C['danger'], lw=2.2, ls='--')
ax.text(0.99, 1.03, 'unwind and rebuild now', color=C['danger'], fontsize=12, ha='right', va='bottom', fontweight='bold')
ax.plot(ps, rt + (1 - ps), color=C['memory'], lw=2.6)
ax.plot(ps, rd + (1 - ps), color=C['compute'], lw=2.6)
ax.plot([p0], [1 - 0.4527713626], 'o', ms=9, color=C['memory'])
ax.annotate('p = 0.5: 47.39 s,\n45.3% less', (p0, 1 - 0.4527713626), xytext=(0.56, 0.66), fontsize=12, color=C['memory'], fontweight='bold')
ax.annotate('break-even p*:\n4.7% (s), 6.1% ($)', (0.06, 1.0), xytext=(0.1, 0.12), fontsize=12, color=C['text'], arrowprops=dict(arrowstyle='->', color=C['neutral']))
ax.text(0.12, 0.66, 'seconds', color=C['memory'], fontsize=12, fontweight='bold')
ax.text(0.3, 0.86, 'dollars', color=C['compute'], fontsize=12, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.25)
ax.set_xlabel('Measured repair success rate p')
ax.set_ylabel('Cost / fallback cost')
ax.grid(color=C['grid'], lw=0.6)
sp.save(fig, out / 'repair_curve.svg')
fig, ax = plt.subplots(figsize=(4.6, 1.7))
t0 = 4.0 * 60 + 9.0
t1 = 5.0 * 60 + 27.0
ax.axvspan(t0, t1, color=C['danger_l'])
ax.axvline(t0, color=C['danger'], lw=2.2)
ax.axvline(t1, color=C['data'], lw=2.2)
ax.text(t0 + 2, 0.78, '04:09 published\nfleet-wide', fontsize=14, color=C['danger'], va='top', fontweight='bold')
ax.text(t1 + 2, 0.78, '05:27\nreverted', fontsize=14, color=C['data'], va='top', fontweight='bold')
ax.text((t0 + t1) / 2 + 12, 0.08, '78 min', fontsize=16, color=C['danger'], ha='center', fontweight='bold')
ax.set_xlim(4 * 60 - 5, 6 * 60)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xticks([240, 300, 360])
ax.set_xticklabels(['04:00', '05:00', '06:00'], fontsize=14)
ax.set_xlabel('19 July 2024, UTC', fontsize=14)
ax.spines['left'].set_visible(False)
sp.save(fig, out / 'cs_window.svg')
P = SimpleNamespace(attempt_cost=Q_(20.45, 'dollar'), loss_caught=Q_(38.4, 'dollar'), loss_caught_raised=Q_(43.2, 'dollar'), loss_missed=Q_(96.0, 'dollar'), loss_missed_raised=Q_(48.0, 'dollar'), loss_per_catch_step=Q_(43.2, 'dollar'), loss_vs_attempt=6.572127139, loss_with_check=Q_(134.4, 'dollar'), loss_with_check_raised=Q_(91.2, 'dollar'), loss_without_check=Q_(480.0, 'dollar'), missed_share=0.7142857143, raised_catch_rate=0.9, reduction=3.571428571, step_vs_attempt=2.112469438)
L = dict(rollout_first=480.0, caught=38.4, missed=96.0, total=134.4)
L90 = dict(caught=43.2, missed=48.0, total=91.2)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    rows = [('Rollout first', 2), ('Canary first', 1), ('Catch 90%', 0)]
    ax.barh(2, L['rollout_first'], color=C['danger'], height=0.6)
    ax.text(L['rollout_first'] + 6, 2, '\\$480', va='center', fontsize=13, color=C['danger'], fontweight='bold')
    a = 1.0 if k >= 2 else 0.0
    ax.barh(1, L['caught'], color=C['neutral'], height=0.6, alpha=a)
    ax.barh(1, L['missed'], left=L['caught'], color=C['danger'], height=0.6, alpha=a)
    ax.text(L['total'] + 6, 1, '\\$134.40: 3.6× less; missed 71.4%', va='center', fontsize=12, color=C['text'], fontweight='bold', alpha=a)
    ax.text(L['caught'] / 2, 1.36, 'caught', ha='center', fontsize=10.5, color=C['neutral'], alpha=a)
    ax.text(L['caught'] + L['missed'] / 2, 1.36, 'missed', ha='center', fontsize=10.5, color=C['danger'], alpha=a)
    b = 1.0 if k >= 3 else 0.0
    ax.barh(0, L90['caught'], color=C['neutral'], height=0.6, alpha=b)
    ax.barh(0, L90['missed'], left=L90['caught'], color=C['danger'], height=0.6, alpha=b)
    ax.text(L90['total'] + 6, 0, '\\$91.20: \\$43.20 less per task', va='center', fontsize=12, color=C['data'], fontweight='bold', alpha=b)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=12, fontweight='bold')
    ax.set_xlim(0, 560)
    ax.set_ylim(-0.6, 2.5)
    ax.set_xlabel('Expected irreversible loss per task ($), illustrative')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'loss_step{k}.svg')
cap, stop = (int(50.0), int(4.0))
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    hs = np.arange(0, cap + 1)
    cum = [0.0, 0.001, 0.0031, 0.0063, 0.0106, 0.016, 0.0225, 0.0301, 0.0388, 0.0486, 0.0595, 0.0715, 0.0846, 0.0988, 0.1141, 0.1305, 0.148, 0.1666, 0.1863, 0.2071, 0.229, 0.252, 0.2761, 0.3013, 0.3276, 0.355, 0.3835, 0.4131, 0.4438, 0.4756, 0.5085, 0.5425, 0.5776, 0.6138, 0.6511, 0.6895, 0.729, 0.7696, 0.8113, 0.8541, 0.898, 0.943, 0.9891, 1.0363, 1.0846, 1.134, 1.1845, 1.2361, 1.2888, 1.3426, 1.3975]
    ax.plot(hs, cum, color=C['network'], lw=2.8)
    ax.plot([stop], [10600 / 1000000.0], 'o', ms=10, color=C['data'], zorder=5)
    ax.plot([cap], [1397500 / 1000000.0], 'o', ms=10, color=C['danger'], zorder=5)
    ax.text(2, 0.62, 'stop after turn 4\n10,600 tokens, $0.050', color=C['data'], fontsize=12, fontweight='bold', va='bottom')
    ax.text(43, 1.52, '50-turn ceiling\n1,397,500 tokens\n$4.42', color=C['danger'], fontsize=12, fontweight='bold', ha='right', va='top')
    if k >= 2:
        ax.fill_between(hs[stop:], 0, cum[stop:], color=C['danger_l'], alpha=0.8)
        ax.text(36, 0.22, '88.7× the dollars', color=C['danger'], fontsize=15, fontweight='bold', ha='center')
    ax.set_xlim(0, 52)
    ax.set_ylim(0, 1.55)
    ax.set_xlabel('Turns the loop runs')
    ax.set_ylabel('Input tokens sent (M)')
    ax.grid(color=C['grid'], lw=0.6)
    sp.save(fig, out / f'loop_step{k}.svg')
j = dict(burst_rate=64000.0, mean_rate=64.0)
fig, ax = plt.subplots(figsize=(5.8, 1.9))
ax.barh(2, j['burst_rate'], color=C['danger'], height=0.55)
ax.barh(1, j['mean_rate'], color=C['data'], height=0.55)
ax.barh(0, j['burst_rate'], color='none', edgecolor=C['data'], ls='--', lw=2, height=0.55)
labels = ['Synchronized: 64,000/s', 'Jittered mean: 64/s', 'Jittered peak: up to 64,000/s']
cols = [C['danger'], C['data'], C['data']]
xs = [j['burst_rate'] * 1.3, j['mean_rate'] * 1.3, j['burst_rate'] * 1.3]
for y, lab, col, x in zip([2, 1, 0], labels, cols, xs):
    ax.text(x, y, lab, va='center', fontsize=12, color=col, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim(1, 30000000.0)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Retry arrivals per second, 128 retries (log)')
ax.minorticks_off()
ax.set_xticks([1, 100.0, 10000.0, 1000000.0])
ax.set_xticklabels(['1', '100', '10,000', '1,000,000'], fontsize=12)
sp.save(fig, out / 'jitter_bars.svg')
fig, ax = plt.subplots(figsize=(6.2, 2.75))
rows = [('IBM1', 73.53), ('AWS1', 66.7), ('GGL2', 21.5), ('AWS4', 9.3), ('SPF2', 8.33), ('AZR4', 7.92), ('AWS3', 4.55), ('AWS2', 4.23), ('GGL3', 3.2), ('AZR2', 1.15), ('CAS2', 0.16)]
for i, (n, v) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, v, color=C['danger'], height=0.62)
    ax.text(v * 1.15, y, f'{v:g} h', va='center', fontsize=13, color=C['danger'], fontweight='bold')
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([n for n, _ in rows][::-1], fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim(0.1, 600)
ax.minorticks_off()
ax.set_xticks([0.1, 1, 10, 100])
ax.set_xticklabels(['0.1', '1', '10', '100'], fontsize=13)
ax.set_xlabel('Duration of each retry-sustained outage (hours, log scale)', fontsize=13)
ax.grid(axis='x', color=C['grid'], lw=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / 'metastable_bars.svg')
o = {'timeout': 14, 'server_error': 6, 'validation': 12, 'denied': 3, 'success': 15}
n_out, trip = (50.0, 0.5 * 50.0)
fig, ax = plt.subplots(figsize=(5.0, 1.9))
segs = [(o['timeout'], 'timeouts', C['danger']), (o['server_error'], '5xx', C['danger']), (o['validation'], 'validation', C['neutral_l']), (o['denied'], 'denied', C['neutral_l']), (o['success'], 'success', C['data_l'])]
for row, counted in ((1, {'timeouts', '5xx'}), (0, {'timeouts', '5xx', 'validation', 'denied'})):
    left = 0
    for n, lab, col in segs:
        c = C['danger'] if lab in counted else C['data_l'] if lab == 'success' else C['neutral_l']
        ax.barh(row, n, left=left, color=c, height=0.62, edgecolor='white')
        ax.text(left + n / 2, row, str(n), ha='center', va='center', fontsize=13, color='white' if lab in counted else C['text'], fontweight='bold')
        left += n
ax.axvline(trip, color=C['text'], lw=1.6, ls='--')
ax.text(trip, 1.5, 'trip at 0.50', ha='center', fontsize=13, color=C['text'])
ax.text(n_out + 1, 1, '0.40: closed', va='center', fontsize=14, color=C['data'], fontweight='bold')
ax.text(n_out + 1, 0, '0.70: opens', va='center', fontsize=14, color=C['danger'], fontweight='bold')
ax.set_yticks([1, 0])
ax.set_yticklabels(['Classified', 'Count all'], fontsize=14, fontweight='bold')
ax.set_xlim(0, 68)
ax.set_ylim(-0.5, 1.7)
ax.set_xticks([0, trip, n_out])
ax.tick_params(labelsize=13)
ax.set_xlabel('Last 50 outcomes; red counts as failure', fontsize=13)
for sname in ('left',):
    ax.spines[sname].set_visible(False)
sp.save(fig, out / 'breaker_window.svg')

def thumb(name, body):
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 100" width="260" height="100"><rect width="260" height="100" fill="white"/>' + body + '</svg>\n'
    (out / name).write_text(svg)
V, R, G, A, B = ('#7c3aed', '#dc2626', '#16a34a', '#d97706', '#0284c7')
thumb('thumb_settle.svg', f'<rect x="8" y="12" width="64" height="30" rx="4" fill="#f3e8ff" stroke="{V}" stroke-width="3"/><rect x="98" y="12" width="64" height="30" rx="4" fill="#fef3c7" stroke="{A}" stroke-width="3"/><rect x="188" y="12" width="64" height="30" rx="4" fill="#f3e8ff" stroke="{V}" stroke-width="3"/><line x1="72" y1="27" x2="98" y2="27" stroke="#64748b" stroke-width="3"/><line x1="162" y1="27" x2="188" y2="27" stroke="#64748b" stroke-width="3"/><path d="M40 42 V72 H120" stroke="{V}" stroke-width="3" fill="none"/><rect x="120" y="60" width="70" height="30" rx="4" fill="#fef3c7" stroke="{A}" stroke-width="3"/><rect x="208" y="60" width="44" height="30" rx="4" fill="#dcfce7" stroke="{G}" stroke-width="3"/>')
thumb('thumb_repair.svg', f'<line x1="10" y1="22" x2="250" y2="22" stroke="{R}" stroke-width="4" stroke-dasharray="10,6"/><line x1="10" y1="18" x2="250" y2="88" stroke="{B}" stroke-width="5"/><line x1="10" y1="12" x2="250" y2="84" stroke="{A}" stroke-width="5"/><circle cx="130" cy="53" r="8" fill="{B}"/>')
thumb('thumb_pivot.svg', ''.join((f'<rect x="{10 + i * 42}" y="35" width="32" height="32" rx="4" fill="#dcfce7" stroke="{G}" stroke-width="3"/>' for i in range(3))) + f'<polygon points="152,20 178,51 152,82 126,51" fill="#fee2e2" stroke="{R}" stroke-width="3"/>' + ''.join((f'<rect x="{190 + i * 36}" y="35" width="28" height="32" rx="4" fill="#f1f5f9" stroke="#94a3b8" stroke-width="3"/>' for i in range(2))))
xs_ = np.linspace(0, 1, 40)
pts = ' '.join((f'{10 + 240 * x:.1f},{90 - 78 * x * x:.1f}' for x in xs_))
thumb('thumb_loop.svg', f'''<polygon points="{10 + 240 * 0.08:.1f},90 {pts.split(' ', 3)[-1]} 250,90" fill="#fee2e2"/><polyline points="{pts}" fill="none" stroke="{V}" stroke-width="5"/><circle cx="{10 + 240 * 0.08:.1f}" cy="{90 - 78 * 0.0064:.1f}" r="8" fill="{G}"/><circle cx="250" cy="12" r="8" fill="{R}"/>''')
tree = f'<rect x="8" y="40" width="20" height="20" fill="{R}"/>'
for y1 in (20, 50, 80):
    tree += f'<line x1="28" y1="50" x2="100" y2="{y1}" stroke="#a78bfa" stroke-width="2.5"/><circle cx="100" cy="{y1}" r="7" fill="{V}"/>'
    for d in (-9, 0, 9):
        tree += f'<line x1="100" y1="{y1}" x2="180" y2="{y1 + d}" stroke="#c4b5fd" stroke-width="2"/><circle cx="180" cy="{y1 + d}" r="4.5" fill="{V}"/>'
        for e in (-3, 0, 3):
            tree += f'<line x1="180" y1="{y1 + d}" x2="245" y2="{y1 + d + e}" stroke="#ddd6fe" stroke-width="1.2"/><circle cx="245" cy="{y1 + d + e}" r="2.2" fill="{V}"/>'
thumb('thumb_contain.svg', tree)
thumb('thumb_recall.svg', ''.join((f'<rect x="{8 + i * 41}" y="34" width="35" height="32" rx="3" fill="#f3e8ff" stroke="{V}" stroke-width="2.5"/>' for i in range(5))) + f'<rect x="213" y="34" width="38" height="32" rx="3" fill="white" stroke="{R}" stroke-width="3" stroke-dasharray="6,4"/>')
grid = ''
import random
rng = random.Random(13)
for i in range(10):
    for jj in range(4):
        ok = rng.random() < 0.7
        grid += f'''<rect x="{12 + i * 24}" y="{10 + jj * 21}" width="18" height="16" rx="2" fill="{('#dcfce7' if ok else '#fee2e2')}" stroke="{(G if ok else R)}" stroke-width="2"/>'''
thumb('thumb_next.svg', grid)
P = PALETTE

def knight_servers():
    """Knight, 1 Aug 2012: the rollback moved all servers onto Power Peg."""
    s = Svg(620, 290)
    for gx, label, after in ((10, 'Before rollback', False), (350, 'After rollback', True)):
        s.text(gx, 30, label, 24, '#0f172a', bold=True, anchor='start')
        for i in range(int(8.0)):
            x = gx + i % 4 * 66
            y = 48 + i // 4 * 62
            peg = after or i == int(8.0) - 1
            s.box(x, y, 58, 52, 'Peg' if peg else 'RLP', None, 'fault' if peg else 'think', tsize=21)
    s.arrow([(280, 104), (340, 104)], P['fault'][1], width=4)
    s.text(10, 210, 'RLP: the new code', 22, P['think'][2], anchor='start', bold=True)
    s.text(350, 210, 'Peg: the retired path', 22, P['fault'][2], anchor='start', bold=True)
    s.text(310, 270, 'Uninstalling RLP from 7 servers put all 8 on Peg', 23, P['fault'][2], bold=True)
    s.save('knight_servers.svg')
knight_servers()

def timeout_seq():
    """Client-token sequence: request acts, reply is lost, lookup settles."""
    s = Svg(560, 430)
    for x, name in ((110, 'Agent'), (450, 'Cloud API')):
        s.box(x - 90, 10, 180, 50, name, None, 'store' if name == 'Agent' else 'tool', tsize=23)
        s.raw(f'<line x1="{x}" y1="60" x2="{x}" y2="420" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6,5"/>')
    s.arrow([(110, 100), (446, 100)], P['store'][1])
    s.text(280, 90, 'create, token t1', 21, '#0f172a')
    s.box(370, 116, 160, 44, 'instance i-1', None, 'ok', tsize=20)
    s.arrow([(450, 196), (114, 196)], P['fault'][1], dash=True)
    s.text(280, 172, 'reply lost', 21, P['fault'][2], bold=True)
    s.raw('<path d="M 268 184 L 292 208 M 292 184 L 268 208" stroke="#dc2626" stroke-width="4"/>')
    s.box(20, 216, 180, 50, 'unknown', 'timeout', 'fault', tsize=21, ssize=18, dash=True)
    s.arrow([(110, 310), (446, 310)], P['store'][1])
    s.text(280, 300, 'retry, same token t1', 21, '#0f172a')
    s.arrow([(450, 360), (114, 360)], P['ok'][1])
    s.text(280, 350, 'returns i-1, no new instance', 21, P['ok'][2])
    s.box(20, 374, 180, 46, 'established', None, 'ok', tsize=21)
    s.save('timeout_seq.svg')
timeout_seq()

def comp_states():
    """Before, after provision, after release: predicates in P vs records."""
    s = Svg(900, 236)
    cols = [(300, 'Before'), (500, 'After provision'), (705, 'After release')]
    for x, h in cols:
        s.text(x + 93, 26, h, 23, '#0f172a', bold=True)
    rows = [('env_r18_01 allocated', 'in P', [('no', 'gray'), ('yes', 'tool'), ('no', 'ok')]), ('A test depends on it', 'in P', [('no', 'gray'), ('yes', 'tool'), ('no', 'ok')]), ('Billing, audit records', 'not in P', [('none', 'gray'), ('written', 'tool'), ('remain', 'fault')])]
    for r, (name, inp, cells) in enumerate(rows):
        y = 42 + r * 64
        s.text(6, y + 26, name, 23, '#0f172a', anchor='start', bold=True)
        s.text(6, y + 50, inp, 19, '#64748b', anchor='start')
        for (x, _), (t, role) in zip(cols, cells):
            s.box(x, y, 186, 54, t, None, role, tsize=24, bold=role != 'gray', dash=role == 'gray')
    s.save('comp_states.svg')
comp_states()

def contracts():
    """Three recovery contracts as three small pictures."""
    s = Svg(900, 330)
    heads = [('Restore selected state', 'ok'), ('Compensate a known effect', 'tool'), ('Manage a commitment', 'fault')]
    for i, (h, role) in enumerate(heads):
        x = 10 + i * 300
        s.rect(x, 8, 280, 314, '#ffffff', P[role][1], rx=8, sw=2.4)
        s.rect(x, 8, 280, 46, P[role][0], P[role][1], rx=8, sw=2.4)
        s.text(x + 140, 39, h, 20, P[role][2], bold=True)
    s.box(30, 72, 110, 50, 'snapshot', None, 'store', tsize=19)
    s.box(170, 72, 100, 50, 'files', None, 'ok', tsize=19)
    s.arrow([(140, 97), (168, 97)], P['ok'][1])
    s.box(70, 142, 160, 46, 'remote env', None, 'gray', tsize=19, dash=True, bold=False)
    s.text(150, 216, 'untouched', 19, '#64748b', italic=True)
    s.box(330, 72, 240, 50, 'env released', None, 'ok', tsize=19)
    s.box(330, 142, 240, 46, 'logs, charges remain', None, 'fault', tsize=19, bold=False)
    s.text(450, 216, 'release itself can fail', 19, '#64748b', italic=True)
    s.box(625, 72, 110, 50, 'config', None, 'fault', tsize=19)
    for j in range(6):
        hx, hy = (790 + j % 2 * 46, 64 + j // 2 * 40)
        s.rect(hx, hy, 36, 30, P['fault'][0], P['fault'][1], rx=3)
    s.arrow([(735, 97), (782, 97)], P['fault'][1], width=2.4)
    s.text(750, 216, 'the effect itself remains', 19, '#64748b', italic=True)
    for i, ex in enumerate(['snapshot ws_r18', 'release env_r18_01', 'fleet config rollout']):
        s.text(150 + i * 300, 262, 'example:', 18, '#64748b')
        s.text(150 + i * 300, 292, ex, 19, '#0f172a', mono=True)
    s.save('contracts.svg')
contracts()

def stuck_pairs():
    """OpenHands scenario 4: steps 1, 3, 5 match and 2, 4, 6 match."""
    s = Svg(900, 250)
    for i in range(6):
        x = 12 + i * 148
        a = 'A' if i % 2 == 0 else 'B'
        role = 'think' if a == 'A' else 'plain'
        s.text(x + 64, 26, f'step {i + 1}', 22, '#475569')
        s.box(x, 38, 128, 50, f'patch {a}', None, role, tsize=23)
        s.box(x, 96, 128, 50, 'test fails', None, 'fault', tsize=22, bold=False)
    for k, (col, y) in enumerate(((P['think'][1], 170), ('#64748b', 200))):
        xs = [12 + 64 + (k + 2 * j) * 148 for j in range(3)]
        s.raw(f'<path d="M {xs[0]} 150 L {xs[0]} {y} L {xs[2]} {y} L {xs[2]} 150 M {xs[1]} 150 L {xs[1]} {y}" fill="none" stroke="{col}" stroke-width="3"/>')
    s.text(450, 240, 'steps 1 = 3 = 5 and 2 = 4 = 6: period 2, stop the run', 24, P['fault'][2], bold=True)
    s.save('stuck_pairs.svg')
stuck_pairs()
print('figures written')
