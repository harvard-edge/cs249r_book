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


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

out = Path(__file__).parent
C = sp.C

# Step builds must keep one canvas size, or the slide rescales the figure
# between steps. The shared save() crops tightly, and a value label that
# appears only in a later step widens that crop, so plots whose labels
# extend past the axes are saved here with fixed margins instead.
_tight_save = sp.save
MARGINS = {}


def _fixed_save(fig, path):
    if MARGINS:
        fig.subplots_adjust(**MARGINS)
        fig.savefig(path, format="svg", facecolor="white")
        sp.plt.close(fig)
    else:
        _tight_save(fig, path)


sp.save = _fixed_save

sys.path.insert(0, str(out))
import calcs as cc  # noqa: E402,F401  (records every value)
from deckvals import val, num  # noqa: E402


def comma(v):
    return f"{v:,.0f}"


# --- Convolution segment: weight sharing ----------------------------------
MARGINS.update(left=0.26, right=0.66, bottom=0.30, top=0.97)
sp.hbars_steps(out / "params", [
    ("Dense layer", cc.CONV.dense_weights, "danger", f"about {val('dense_w_m')} million"),
    (f"{val('k')}×{val('k')} convolution", num("conv_w"), "compute",
     f"{val('conv_w')}: {val('share_ratio')}× fewer"),
], "weights", xlabel=f"Weights for {val('ch')} features from a {val('img')}×{val('img')}×{val('ch')} input (log scale)",
    size=sp.SHORT)

# --- Convolution segment: reuse raises intensity --------------------------
MARGINS.update(left=0.37, right=0.70, bottom=0.30, top=0.97)
sp.hbars_steps(out / "intensity_cnn", [
    ("Dense layer, batch one", num("dense_i"), "memory", val("dense_i")),
    ("ResNet-50", num("resnet_i"), "compute",
     f"{val('resnet_i')}: {val('resnet_vs_dense')}× more"),
], "FLOP/byte", xlabel="Arithmetic intensity (FLOP per weight byte, FP32, log scale)",
    size=sp.SHORT)

# --- Dense segment: weights per detector, dense against shared ------------
MARGINS.update(left=0.33, right=0.70, bottom=0.30, top=0.97)
sp.hbars_steps(out / "mnist_detectors", [
    (f"{val('first_feat')} dense features", num("dense_det"), "danger", val("dense_det")),
    (f"{val('first_feat')} shared {val('k')}×{val('k')} filters", num("conv_det"), "compute",
     f"{val('conv_det')}: {val('det_ratio')}× fewer"),
], "weights", xlabel=f"Weights for {val('first_feat')} feature detectors on a {val('mnist_side')}×{val('mnist_side')} image (log scale)",
    size=sp.SHORT)

# --- Depth and efficiency segment: factoring a 3x3 convolution ------------
MARGINS.update(left=0.36, right=0.72, bottom=0.30, top=0.97)
sp.hbars_steps(out / "depthwise", [
    (f"Standard {val('k')}×{val('k')}", num("conv_w"), "danger", val("conv_w")),
    ("Depthwise + pointwise", num("sep"), "compute",
     f"{val('dw')} + {val('pw')} = {val('sep')}"),
], "MACs", log=False,
    xlabel=f"Multiply-adds per output position ({val('k')}×{val('k')} filter, {val('ch')} channels in and out)",
    size=sp.SHORT)
MARGINS.clear()


# --- Depth and efficiency segment: KWS weight bytes against the board ------
def kws_plot():
    """Column-size plot for the KWS frame: FP32 and INT8 weight bytes against
    the 1 MB flash of an Arduino Nano 33 BLE Sense Rev2 (decimal KB)."""
    fp32 = num("kws_fp32_kb")
    int8 = num("kws_int8_kb")
    flash = cc.m(cc.MCU.memory.flash_capacity, "KB")
    fig, ax = sp.plt.subplots(figsize=(3.6, 2.1))
    rows = [("FP32", fp32, "danger", f"{val('kws_fp32_kb')} KB, {val('kws_fp32_pct')}% of flash"),
            ("INT8", int8, "data", f"{val('kws_int8_kb')} KB, {val('kws_int8_pct')}%")]
    for i, (label, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(-15, y, label, ha="right", va="center", fontsize=13,
                fontweight="bold", color=C["text"])
        inside = v > flash / 2
        ax.text(v - 20 if inside else v + 20, y, txt, ha="right" if inside else "left",
                va="center", fontsize=12, fontweight="bold",
                color="white" if inside else C[ck])
    ax.axvline(flash, color=C["neutral"], lw=2, ls="--")
    ax.text(flash - 15, 1.62, f"{val('flash_mb')} MB flash", ha="right",
            va="center", fontsize=11, fontweight="bold", color=C["neutral"])
    ax.set_xlim(0, flash * 1.05)
    ax.set_ylim(-0.5, 1.85)
    ax.set_yticks([])
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xticklabels([comma(t) for t in (0, 250, 500, 750, 1000)])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("KWS DS-CNN weights (KB)")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    _tight_save(fig, out / "kws_bytes.svg")


kws_plot()
print("figures written")
