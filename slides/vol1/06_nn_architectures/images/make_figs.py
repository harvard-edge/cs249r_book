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
MARGINS.update(left=0.36, right=0.72, bottom=0.3, top=0.97)
sp.hbars_steps(out / 'depthwise', [('Standard 3×3', 36864.0, 'danger', '36,864'), ('Depthwise + pointwise', 4672.0, 'compute', '576 + 4,096 = 4,672')], 'MACs', log=False, xlabel='Multiply-adds per output position (3×3 filter, 64 channels in and out)', size=sp.SHORT)
MARGINS.clear()

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
