#!/usr/bin/env python3
"""Toy request ledger and source geometry for the rollback-bundle Explore view.

The owning section (14_ml_ops.qmd, Stateful vs. stateless rollback) states the
mechanism but gives no request counts, rates, or durations, and it has no LEGO
cell. Every count below is therefore a declared toy value (six labeled request
markers). This script derives each state's counters from the per-request
ledger, asserts the lifetime invariants, checks the anchors against the
verified artwork, and verifies the constants embedded in the factory.

Run: python3 books/_extensions/mlsysbook-ext/figure-explore/original-v1-rollback-compatible-bundles.calc.py
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE / 'original-v1-rollback-compatible-bundles.js'
ART_INDEX = Path('/tmp/figures-animation-art-current/index.json')
KEY = 'vol1:fig-rollback-compatible-bundles'

# Declared toy scenario: R1-R4 admitted to the candidate before cutover,
# R5-R6 admitted after it. Each event is one causal change.
REQUESTS = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
EVENTS = [
    ('start', {}),
    ('cutover', {}),
    ('admit', {'R5': 1, 'R6': 1}),
    ('finish', ['R1', 'R2']),
    ('refuse', 'R3'),
    ('cancel_request', 'R4'),
    ('cancel_confirm', 'R4'),
    ('finish', ['R3']),
    ('release', None),
]
LIVE = {'in-flight', 'cancel-requested'}


def ledger():
    req = {r: {'bundle': 2, 'status': 'in-flight', 'admitted': 'before'} for r in REQUESTS[:4]}
    req.update({r: {'bundle': None, 'status': 'not-arrived', 'admitted': None} for r in REQUESTS[4:]})
    target, b2_state, released, refused = 2, 'serving', False, None
    log = [r for r in REQUESTS if req[r]['bundle'] == 2]
    states = []
    for kind, arg in EVENTS:
        refused = None
        if kind == 'cutover':
            target, b2_state = 1, 'retained'
        elif kind == 'admit':
            for r, bundle in arg.items():
                assert bundle == target, 'admission must use the current router target'
                req[r] = {'bundle': bundle, 'status': 'in-flight', 'admitted': 'after'}
        elif kind == 'finish':
            for r in arg:
                assert req[r]['status'] in LIVE
                req[r]['status'] = 'done'
        elif kind == 'refuse':
            refused = arg  # weights 1 + namespace 2 is invalid: no ledger change
        elif kind == 'cancel_request':
            req[arg]['status'] = 'cancel-requested'
        elif kind == 'cancel_confirm':
            assert req[arg]['status'] == 'cancel-requested', 'confirmation follows a request'
            req[arg]['status'] = 'cancel-confirmed'
        elif kind == 'release':
            released = True
        b1 = sum(1 for r in REQUESTS if req[r]['bundle'] == 1 and req[r]['status'] in LIVE)
        b2 = sum(1 for r in REQUESTS if req[r]['bundle'] == 2 and req[r]['status'] in LIVE)
        if target == 2:
            b2_state = 'serving'
        elif released:
            assert b2 == 0, 'Bundle 2 may not be released while any pinned request is live'
            b2_state = 'released'
        elif b2 == 0:
            b2_state = 'releasable'
        states.append({
            'event': kind, 'target': target, 'b1': b1, 'b2': b2, 'b2_state': b2_state,
            'b1_state': 'standby' if b1 == 0 else 'serving', 'refused': refused,
            'log': list(log), 'log_flagged': True,
            'requests': {r: dict(v) for r, v in req.items()},
        })
    return states


def check(states):
    # Pinned bundles never change after admission (no migration across vector spaces).
    for r in REQUESTS:
        seen = {s['requests'][r]['bundle'] for s in states if s['requests'][r]['bundle'] is not None}
        assert len(seen) == 1, (r, seen)
    # Admissions after cutover go to Bundle 1 only; before it, to Bundle 2 only.
    for s in states:
        for v in s['requests'].values():
            if v['admitted'] == 'after': assert v['bundle'] == 1
            if v['admitted'] == 'before': assert v['bundle'] == 2
    # Degraded-window records come only from Bundle 2 admissions and are never repaired by routing.
    assert all(s['log'] == ['R1', 'R2', 'R3', 'R4'] and s['log_flagged'] for s in states)
    # Release happens only after the count reaches zero, and never before.
    counts = [s['b2'] for s in states]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] == 4 and counts[-1] == 0
    first_zero = counts.index(0)
    assert all(s['b2_state'] != 'released' for s in states[:first_zero + 1])
    assert states[-1]['b2_state'] == 'released' and states[-2]['b2_state'] == 'releasable'
    # Cancellation requested still counts; only confirmation decrements.
    i = [s['event'] for s in states].index('cancel_request')
    assert states[i]['b2'] == states[i - 1]['b2'] and states[i + 1]['b2'] == states[i]['b2'] - 1
    # The refused move leaves every counter unchanged.
    j = [s['event'] for s in states].index('refuse')
    assert {k: states[j][k] for k in ('b1', 'b2', 'requests')} == {k: states[j - 1][k] for k in ('b1', 'b2', 'requests')}
    assert [s['b1'] for s in states] == [0, 0, 2, 2, 2, 2, 2, 2, 2]


def geometry():
    """Read the source rectangles and arrow positions from the verified SVG."""
    art = json.loads(ART_INDEX.read_text())[KEY]['asset'] if ART_INDEX.exists() else None
    if not art or not Path(art).exists():
        return None
    s = Path(art).read_text()
    assert 'viewBox="0 0 357.564 145.496"' in s
    rects = []
    for m in re.finditer(r'<path ([^>]*)/>', s[s.index('</defs>'):]):
        a = dict(re.findall(r'([\w-]+)="([^"]*)"', m.group(1)))
        n = list(map(float, re.findall(r'-?\d+\.?\d*', a['d'])))
        A, B, C, D, E, F = map(float, re.findall(r'-?\d+\.?\d*', a.get('transform', 'matrix(1,0,0,1,0,0)')))
        xs = [A * x + C * y + E for x, y in zip(n[0::2], n[1::2])]
        ys = [B * x + D * y + F for x, y in zip(n[0::2], n[1::2])]
        rects.append([round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)])
    return {'router': rects[0], 'bundle1': rects[1], 'bundle2': rects[2], 'green_x': rects[3][0],
            'dashed_x': rects[5][0], 'invalid': rects[7], 'log': rects[8]}


if __name__ == '__main__':
    states = ledger()
    check(states)
    out = {'toy': True, 'requests': REQUESTS, 'states': states}
    g = geometry()
    if MODULE.exists():
        text = MODULE.read_text()
        embedded = json.loads(re.search(r'const D=(\{.*?\});\n', text).group(1))
        assert embedded == {'requests': REQUESTS, 'states': states}, 'factory ledger is stale; paste the new D'
        G = json.loads(re.search(r'const G=(\{.*?\});\n', text).group(1))
        if g:
            assert G['source'] == g, (G['source'], g)
            # Token slots sit in the gaps beside each source arrow, clear of boxes.
            w, h = G['token']
            for x, y in G['slots'].values():
                assert g['router'][3] < y - h / 2 and y + h / 2 < g['bundle1'][1] - 4
            assert all(g['green_x'] < x - w / 2 < x + w / 2 < g['invalid'][0] for r, (x, y) in G['slots'].items() if r in ('R5', 'R6'))
            assert all(g['invalid'][2] < x - w / 2 < x + w / 2 < g['dashed_x'] for r, (x, y) in G['slots'].items() if r not in ('R5', 'R6'))
            # Same-row tokens never overlap.
            assert all(abs(G['slots'][a][0] - G['slots'][b][0]) > w for a, b in (('R1', 'R2'), ('R3', 'R4'), ('R5', 'R6')))
    final = states[-1]
    print(json.dumps({'final': {k: final[k] for k in ('target', 'b1', 'b2', 'b2_state', 'log', 'log_flagged')},
                      'b2_counts': [s['b2'] for s in states], 'source_geometry': g}, indent=1))
    if '--ledger' in sys.argv:
        print(json.dumps({'requests': REQUESTS, 'states': states}, separators=(',', ':')))
