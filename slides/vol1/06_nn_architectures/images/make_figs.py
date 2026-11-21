#!/usr/bin/env python3
"""Data plots for Lecture 6 (Network Architectures, Part 1).

Every value comes from calcs.py in this folder, which reads MLSysIM (the
layer scenarios, the DS-CNN model, and the nRF52840 microcontroller); labels
print the values the slides print (deckvals.val), and bars are drawn at those
values (deckvals.num). Nothing here types a number. Part 2 (06b) owns the
attention, KV-cache, recommender, and lighthouse plots; their generators were
removed from this file on 2026-10-18. The unused energy_weight plot was
removed on 2026-10-20 (no slide includes it).

Run from this folder: python3 make_figs.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
out = Path(__file__).parent
C = sp.C
_tight_save = sp.save
MARGINS = {}

def _fixed_save(fig, path):
    if MARGINS:
        fig.subplots_adjust(**MARGINS)
        fig.savefig(path, format='svg', facecolor='white')
        sp.plt.close(fig)
    else:
        _tight_save(fig, path)
sp.save = _fixed_save
sys.path.insert(0, str(out))

def comma(v):
    return f'{v:,.0f}'
MARGINS.update(left=0.26, right=0.66, bottom=0.3, top=0.97)
sp.hbars_steps(out / 'params', [('Dense layer', 205520896, 'danger', 'about 205.5 million'), ('3×3 convolution', 36864.0, 'compute', '36,864: 5,575× fewer')], 'weights', xlabel='Weights for 64 features from a 224×224×64 input (log scale)', size=sp.SHORT)
MARGINS.update(left=0.37, right=0.7, bottom=0.3, top=0.97)
sp.hbars_steps(out / 'intensity_cnn', [('Dense layer, batch one', 0.5, 'memory', '0.5'), ('ResNet-50', 80.2, 'compute', '80.2: 160× more')], 'FLOP/byte', xlabel='Arithmetic intensity (FLOP per weight byte, FP32, log scale)', size=sp.SHORT)
MARGINS.update(left=0.33, right=0.7, bottom=0.3, top=0.97)
sp.hbars_steps(out / 'mnist_detectors', [('32 dense features', 25088.0, 'danger', '25,088'), ('32 shared 3×3 filters', 288.0, 'compute', '288: 87× fewer')], 'weights', xlabel='Weights for 32 feature detectors on a 28×28 image (log scale)', size=sp.SHORT)
MARGINS.clear()

def depthwise_steps():
    """3-step build matching vol1:fig-depthwise-pointwise-channels:
    standard 3x3 (36,864) -> depthwise 3x3 (576, shared per ch.) -> + pointwise 1x1 (4,096; total 4,672 = 7.9x fewer)."""
    rows = [
        ('Standard 3×3', 36864.0, 'danger', '36,864'),
        ('Depthwise 3×3\n(per channel)', 576.0, 'memory', '576  (shared spatial)'),
        ('Pointwise 1×1\n(mix channels)', 4096.0, 'compute', '4,096  (total 4,672: 7.9× fewer)'),
    ]
    for step in (1, 2, 3):
        fig, ax = sp.plt.subplots(figsize=(5.8, 2.35))
        n = len(rows)
        for i, (lab, v, ck, txt) in enumerate(rows):
            y = n - 1 - i
            shown = i < step
            ax.barh(y, v, color=C[ck], height=0.58, alpha=1.0 if shown else 0.0)
            ax.text(-900, y, lab, ha='right', va='center', fontsize=11.5, fontweight='bold', color=C['text'] if shown else 'white')
            ax.text(v + 700, y, txt, ha='left', va='center', fontsize=11.5, fontweight='bold', color=C[ck], alpha=1.0 if shown else 0.0)
        ax.set_xlim(0, 48000)
        ax.set_ylim(-0.6, n - 0.35)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xticks([0, 10000, 20000, 30000, 40000])
        ax.set_xticklabels(['0', '10K', '20K', '30K', '40K'])
        ax.set_xlabel('Multiply-adds per output (3×3 filter, 64 channels)')
        ax.grid(axis='x', color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.29, right=0.97, bottom=0.25, top=0.96)
        fig.savefig(out / f'depthwise_step{step}.svg', format='svg', facecolor='white')
        sp.plt.close(fig)

depthwise_steps()

def conv_slide():
    """3-step isometric SVG matching vol1:fig-cnn (7x7 input -> 5x5 output map)."""
    FONT = 'font-family="Helvetica, Arial, sans-serif"'
    def iso(r, c, ox, oy, sx=28.0, sy=14.0):
        return (ox + (c - r) * sx, oy + (c + r) * sy)
    def poly(pts, fill, stroke='#1A1A2E', sw=1.4, dash='', op=1.0):
        p = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        d = f' stroke-dasharray="{dash}"' if dash else ''
        fo = f' fill-opacity="{op}"' if op < 1.0 else ''
        return f'<polygon points="{p}" fill="{fill}"{fo} stroke="{stroke}" stroke-width="{sw}"{d}/>'
    def cell(r, c, ox, oy, sx, sy, fill, stroke='#1A1A2E', sw=1.2, dash='', op=1.0):
        return poly([iso(r, c, ox, oy, sx, sy), iso(r, c + 1, ox, oy, sx, sy),
                     iso(r + 1, c + 1, ox, oy, sx, sy), iso(r + 1, c, ox, oy, sx, sy)],
                    fill, stroke, sw, dash, op)

    in_ox, in_oy, in_sx, in_sy = 240.0, 232.0, 29.0, 14.5
    out_ox, out_oy, out_sx, out_sy = 240.0, 26.0, 25.0, 12.5
    for step in (1, 2, 3):
        s = []
        # 7x7 input grid
        for r in range(7):
            for c in range(7):
                s.append(cell(r, c, in_ox, in_oy, in_sx, in_sy, '#FFFFFF', '#666666', 1.1, '5,4'))
        # Window position: step 1 at (1,1), step 2 & 3 slid to (1,3)
        wr, wc = (1, 1) if step == 1 else (1, 3)
        if step >= 2:
            # Ghost of first window at (1,1)
            for r in range(1, 4):
                for c in range(1, 4):
                    s.append(cell(r, c, in_ox, in_oy, in_sx, in_sy, '#FFE7CE', '#CC5500', 1.0, '4,3', 0.55))
        # Active 3x3 orange window
        for r in range(wr, wr + 3):
            for c in range(wc, wc + 3):
                s.append(cell(r, c, in_ox, in_oy, in_sx, in_sy, '#F0801A', '#7A3500', 1.5))
        # 5x5 output grid above
        for r in range(5):
            for c in range(5):
                if step == 3:
                    fill = '#8B3A9C' if (r == wr and c == wc) else '#DEC2E6'
                elif step == 2:
                    fill = '#8B3A9C' if (r == wr and c == wc) else ('#C89FD4' if (r == 1 and c == 1) else '#F5ECF7')
                else:
                    fill = '#8B3A9C' if (r == 1 and c == 1) else '#F5ECF7'
                s.append(cell(r, c, out_ox, out_oy, out_sx, out_sy, fill, '#1A1A2E', 1.2))
        # 4 projection rays from active 3x3 corners to active output cell corners
        in_corners = [iso(wr, wc, in_ox, in_oy, in_sx, in_sy),
                      iso(wr, wc + 3, in_ox, in_oy, in_sx, in_sy),
                      iso(wr + 3, wc + 3, in_ox, in_oy, in_sx, in_sy),
                      iso(wr + 3, wc, in_ox, in_oy, in_sx, in_sy)]
        out_corners = [iso(wr, wc, out_ox, out_oy, out_sx, out_sy),
                       iso(wr, wc + 1, out_ox, out_oy, out_sx, out_sy),
                       iso(wr + 1, wc + 1, out_ox, out_oy, out_sx, out_sy),
                       iso(wr + 1, wc, out_ox, out_oy, out_sx, out_sy)]
        for (x1, y1), (x2, y2) in zip(in_corners, out_corners):
            s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#006395" stroke-width="1.6"/>')
        if step >= 2:
            # Curved slide arrow from (2.5, 2.5) to (2.5, 4.5) on input grid
            p0 = iso(2.5, 2.3, in_ox, in_oy, in_sx, in_sy)
            p1 = iso(2.5, 4.3, in_ox, in_oy, in_sx, in_sy)
            s.append(f'<defs><marker id="arrO" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,1 L7,4 L0,7 Z" fill="#A51C30"/></marker></defs>')
            s.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" stroke="#A51C30" stroke-width="3.2" marker-end="url(#arrO)"/>')
        # Labels
        s.append(f'<text x="240" y="19" {FONT} font-size="25" font-weight="bold" fill="#7030A0" text-anchor="middle">Output map (5×5)</text>')
        cap = {
            1: 'Step 1: 3×3 window, 1 cell',
            2: 'Step 2: same 9 weights slide',
            3: 'Step 3: 25 steps fill map',
        }[step]
        s.append(f'<text x="240" y="458" {FONT} font-size="25" font-weight="bold" fill="#1A1A2E" text-anchor="middle">{cap}</text>')
        body = '\n'.join(s)
        (out / f'conv_slide_step{step}.svg').write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 478" width="480" height="478">\n'
            f'<rect width="480" height="478" fill="white"/>\n{body}\n</svg>\n'
        )

conv_slide()

def kws_plot():
    """Column-size plot for the KWS frame: FP32 and INT8 weight bytes against
    the 1 MB flash of an Arduino Nano 33 BLE Sense Rev2 (decimal KB)."""
    fp32 = 800.0
    int8 = 200.0
    flash = 1048.576
    fig, ax = sp.plt.subplots(figsize=(3.6, 2.1))
    rows = [('FP32', fp32, 'danger', '800 KB, 76% of flash'), ('INT8', int8, 'data', '200 KB, 19%')]
    for i, (label, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(-15, y, label, ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
        inside = v > flash / 2
        ax.text(v - 20 if inside else v + 20, y, txt, ha='right' if inside else 'left', va='center', fontsize=12, fontweight='bold', color='white' if inside else C[ck])
    ax.axvline(flash, color=C['neutral'], lw=2, ls='--')
    ax.text(flash - 15, 1.62, '1 MB flash', ha='right', va='center', fontsize=11, fontweight='bold', color=C['neutral'])
    ax.set_xlim(0, flash * 1.05)
    ax.set_ylim(-0.5, 1.85)
    ax.set_yticks([])
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xticklabels([comma(t) for t in (0, 250, 500, 750, 1000)])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('KWS DS-CNN weights (KB)')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    _tight_save(fig, out / 'kws_bytes.svg')
kws_plot()
print('figures written')
