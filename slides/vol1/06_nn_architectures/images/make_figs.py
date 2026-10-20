#!/usr/bin/env python3
"""Data plots for Lecture 6 (Network Architectures, Part 1).

Every input and derived value comes from calcs.py in this folder (inputs
carry their sources there: the rendered chapter, Horowitz 2014 via the
chapter, and the Arduino Nano 33 BLE Sense Rev2 tech specs). Nothing here
types a derived number. Part 2 (06b) owns the attention, KV-cache,
recommender, and lighthouse plots; their generators were removed from this
file on 2026-10-18.

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
import calcs as cc  # noqa: E402


def comma(v):
    return f"{v:,.0f}"


# --- Dense segment: moving a weight against computing with it -------------
e_weight = cc.DRAM_PJ_PER_BYTE * cc.FP32_B
MARGINS.update(left=0.42, right=0.78, bottom=0.30, top=0.97)
sp.hbars_steps(out / "energy_weight", [
    ("FP32 multiply-add", cc.MAC_FP32_PJ, "compute", f"{cc.MAC_FP32_PJ} pJ"),
    ("Read FP32 weight, DRAM", e_weight, "memory",
     f"{e_weight:.0f} pJ = {e_weight / cc.MAC_FP32_PJ:.0f}×"),
], "pJ", xlabel="Energy (picojoules, log scale; 45 nm, Horowitz 2014)",
    size=sp.SHORT)

# --- Convolution segment: weight sharing ----------------------------------
dense_w = cc.IMG * cc.IMG * cc.C_IN * cc.C_OUT
conv_w = cc.K3 * cc.K3 * cc.C_IN * cc.C_OUT
MARGINS.update(left=0.26, right=0.66, bottom=0.30, top=0.97)
sp.hbars_steps(out / "params", [
    ("Dense layer", dense_w, "danger", f"about {dense_w / 1e6:.1f} million"),
    ("3×3 convolution", conv_w, "compute",
     f"{comma(conv_w)}: {comma(dense_w / conv_w)}× fewer"),
], "weights", xlabel="Weights for 64 features from a 224×224×64 input (log scale)",
    size=sp.SHORT)

# --- Convolution segment: reuse raises intensity --------------------------
i_dense = 2 / cc.FP32_B
MARGINS.update(left=0.37, right=0.70, bottom=0.30, top=0.97)
sp.hbars_steps(out / "intensity_cnn", [
    ("Dense layer, batch one", i_dense, "memory", f"{i_dense}"),
    ("ResNet-50", cc.RESNET50_INT, "compute",
     f"{cc.RESNET50_INT}: {cc.RESNET50_INT / i_dense:.0f}× more"),
], "FLOP/byte", xlabel="Arithmetic intensity (FLOP per weight byte, FP32, log scale)",
    size=sp.SHORT)

# --- Dense segment: weights per detector, dense against shared ------------
dense_det = cc.N_FEAT_MNIST * cc.MNIST_HW
conv_det = cc.N_FEAT_MNIST * cc.K3 * cc.K3
MARGINS.update(left=0.33, right=0.70, bottom=0.30, top=0.97)
sp.hbars_steps(out / "mnist_detectors", [
    ("32 dense features", dense_det, "danger", comma(dense_det)),
    ("32 shared 3×3 filters", conv_det, "compute",
     f"{conv_det}: {dense_det / conv_det:.0f}× fewer"),
], "weights", xlabel="Weights for 32 feature detectors on a 28×28 image (log scale)",
    size=sp.SHORT)

# --- Depth and efficiency segment: factoring a 3x3 convolution ------------
dw = cc.K3 * cc.K3 * cc.C_IN
pw = cc.C_IN * cc.C_OUT
MARGINS.update(left=0.36, right=0.72, bottom=0.30, top=0.97)
sp.hbars_steps(out / "depthwise", [
    ("Standard 3×3", conv_w, "danger", comma(conv_w)),
    ("Depthwise + pointwise", dw + pw, "compute",
     f"{comma(dw)} + {comma(pw)} = {comma(dw + pw)}"),
], "MACs", log=False,
    xlabel="Multiply-adds per output position (3×3 filter, 64 channels in and out)",
    size=sp.SHORT)
MARGINS.clear()


# --- Depth and efficiency segment: KWS weight bytes against the board ------
def kws_plot():
    """Column-size plot for the KWS frame: FP32 and INT8 weight bytes against
    the 1 MB flash of an Arduino Nano 33 BLE Sense Rev2 (decimal KB)."""
    fp32 = cc.KWS_PARAMS * cc.FP32_B / 1e3
    int8 = cc.KWS_PARAMS * cc.INT8_B / 1e3
    flash = cc.FLASH_B / 1e3
    fig, ax = sp.plt.subplots(figsize=(3.6, 2.1))
    rows = [("FP32", fp32, "danger", f"{fp32:.0f} KB, {100 * fp32 * 1e3 / cc.FLASH_B:.0f}% of flash"),
            ("INT8", int8, "data", f"{int8:.0f} KB, {100 * int8 * 1e3 / cc.FLASH_B:.0f}%")]
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
    ax.text(flash - 15, 1.62, f"{cc.FLASH_B // 2**20} MB flash", ha="right",
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
