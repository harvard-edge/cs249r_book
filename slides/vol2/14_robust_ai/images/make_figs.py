#!/usr/bin/env python3
"""Figures for Lecture 14 (Volume II): Robust AI.

Every number is read from MLSysIM through calcs.py (the same Scenarios.* and
Literature entries the chapter reads); the list below records each source.
"RA" is the rendered chapter, Robust AI. Derived values are labeled "derived"
on the figure or its slide; scenario values are labeled "illustrative".
Typed here: layout coordinates, the illustrative shift points (RA figure 3),
and the illustrative conformal score draw.

  - Hook (Eykholt et al., CVPR 2018, arXiv 1707.08945): LISA-CNN 91 percent
    accuracy on the LISA test set (Sec. 1); targeted success in 100 percent
    of lab images (abstract); drive-by field test, camouflage abstract art,
    84.8 percent of frames (Table 3). RA Example 14.1 quotes 100 and 84.8.
  - Tempe, 2018 (RA Sec. 14.2.2, NTSB HAR-19/03): detected 5.6 s before
    impact; reclassified among vehicle, other, bicycle; collision recognized
    1.2 s before impact; action suppressed 1 s; then alert and gradual
    slowdown. The report's table of ADS parameters (HAR-19/03, p. 15 to 16)
    logs the alert at -0.2 s, which equals 1.2 - 1.0 (checked in calcs.py).
    The preliminary report (HWY18MH010, May 2018) gave 1.3 s for the
    recognition; the final report's 1.2 s is used here and in Lecture 16.
  - Fleet silent corruption (RA Sec. 14.3.1 and its self-check 2):
    Pr(>= 1) = 1 - (1 - p)^N with p = 1e-4 per device-hour (assumed).
    Derived: N = 1,000 gives 0.095; N = 10,000 gives 0.632.
  - Shift panels: redrawn from RA figure 3 (illustrative points).
  - ImageNet-V2 (RA figure 4): MLSysIM vol2/14_robust_ai/data/imagenet_testbed_shift.csv
    (Taori et al. 2020 testbed; Recht et al. 2019), models with val >= 55.
  - Mean-shift test (RA Napkin Math 14.1): sigma 0.3, n 1,000, 0.50 -> 0.55;
    standard error 0.0095 and 5.3 standard errors (derived).
  - Fraud PSI (RA table 3): per-country contributions 0.0118, 0.0021,
    0.0007, 0.0036, 0.0470; sum 0.065. Other's share 72 percent (derived).
  - Calibration (Guo et al., ICML 2017, Table 1, ECE with 15 bins):
    CIFAR-100 LeNet 5 4.85 -> 2.02, CIFAR-100 ResNet 110 16.53 -> 1.26,
    ImageNet ResNet 152 5.48 -> 1.86 (uncalibrated -> temperature scaling).
  - Risk-coverage and conformal ranks: illustrative curves; conformal rank
    ceil((n + 1)(1 - alpha)) with n = 1,000 and alpha = 0.1 is 901 (derived).
  - Jailbreak transfer (Zou et al. 2023, arXiv 2307.15043, Table 2, 388
    behaviors): GPT-3.5 1.8, 5.7, 47.4, 86.6; GPT-4 8.0, 13.1, 29.1, 46.9
    (behavior only; + "Sure, here's"; + GCG on Vicuna and Guanacos;
    + ensemble).
  - PGD-7 work model (RA Napkin Math 14.2): 1 + 7 = 8 standard passes;
    FGSM listing about 3 (RA listing 1 and Napkin Math 14.2).
  - RobustBench (RA figure 10): MLSysIM
    vol2/14_robust_ai/data/robustbench_cifar10_linf.csv, dated and not flagged
    unreliable; l-infinity 8/255.
  - Certified radius (RA Sec. 14.6.1.1): sigma 0.5, pA 0.999, R = sigma *
    Phi^-1(pA) = 1.5. Derived: an l-infinity ball of 8/255 on a 32x32x3
    image reaches l2 length 8/255 * sqrt(3,072) = 1.74.
  - BadNets (Gu et al. 2017, arXiv 1708.06733, Table 4, yellow-square
    trigger, single target): baseline F-RCNN average clean 90.0 percent;
    BadNet average clean 89.3 percent; backdoored stop signs classified as
    speed limit 90.3 percent.
  - Bit flip (RA figure 14): 0.042 in FP32, bit 30 flips, x 2^128,
    about 1.4e37.

Run from this folder: python3 make_figs.py (derived slide numbers: calcs.py)
"""

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
import calcs as K  # noqa: E402  (records every value; figures read them)
from deckvals import val, num  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent

# Volume II schematic roles (volumes/vol2/palette.md)
V = {"blue": "#1F407A", "blue_l": "#EAF1F8", "amber": "#d97706", "amber_l": "#fef3c7",
     "violet": "#7c3aed", "violet_l": "#f3e8ff", "green": "#16a34a", "green_l": "#dcfce7",
     "red": "#dc2626", "red_l": "#fee2e2", "frame": "#94a3b8", "frame_l": "#f8fafc",
     "ink": "#0f172a", "gray": "#64748b"}


# ------------------------------------------------------------------ SVG
def svg(path, body, w=900, h=500):
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="Helvetica, Arial, sans-serif">\n'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>\n'
            '<defs>' + "".join(
                f'<marker id="a-{k}" markerWidth="10" markerHeight="8" refX="9" refY="4" '
                f'orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="{V[k]}"/></marker>'
                for k in ("blue", "amber", "violet", "green", "red", "gray")) + '</defs>\n')
    Path(path).write_text(head + "\n".join(body) + "\n</svg>\n")


def rect(x, y, w, h, fill, stroke, sw=2, r=6, op=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>')


def text(x, y, s, color=V["ink"], size=20, weight=400, anchor="middle", op=1.0, italic=False):
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}" opacity="{op}"{st}>{s}</text>')


def arrow(x1, y1, x2, y2, color="gray", sw=2.5, op=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{V[color]}" '
            f'stroke-width="{sw}" marker-end="url(#a-{color})" opacity="{op}"{d}/>')


def circle(cx, cy, r, fill, stroke, sw=2.5, op=1.0):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{op}"/>')


def tag(ax, s, x=0.0, y=-0.28):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, color=C["neutral"],
            ha="left", va="top")


# =========================================================== HOOK (Eykholt)
HOOK_ROWS = [("Clean test set", num("lisa_clean_pct"), "neutral", f"{val('lisa_clean_pct')}% correct"),
             ("Lab photos", num("sticker_lab_pct"), "compute", f"{val('sticker_lab_pct')}% Speed Limit 45"),
             ("Drive-by frames", num("sticker_drive_pct"), "danger", f"{val('sticker_drive_pct')}% Speed Limit 45")]


def hook(path, k, size=(4.0, 1.85), fs=12.5):
    rows = HOOK_ROWS
    fig, ax = plt.subplots(figsize=size)
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 2 - i
        shown = (i < 2 and k >= i + 1) or (i == 2 and k >= 3)
        ax.barh(y, v, color=C[ck], height=0.66, alpha=1.0 if shown else 0.0)
        ax.text(-2, y, lab, ha="right", va="center", fontsize=fs, fontweight="bold",
                alpha=1.0 if (i <= k or i == 2) else 0.0)
        ax.text(3, y, txt, va="center", fontsize=fs - 0.5, fontweight="bold", color="white",
                alpha=1.0 if shown else 0.0)
        if i == 2 and k == 2:
            ax.text(2, y, "???", va="center", fontsize=fs + 2, fontweight="bold", color=C["danger"])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Percent, LISA-CNN road-sign classifier")
    sp.save(fig, path)


for k in (1, 2, 3):
    hook(out / f"hook_step{k}.svg", k)


def hook_synth(path, size=(3.4, 2.7), fs=15):
    """The hook's final step at the size of the closing slide's narrow column:
    each row's name sits above its bar, and the values print inside the bars."""
    rows = HOOK_ROWS
    fig, ax = plt.subplots(figsize=size)
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 2 - i
        ax.barh(y, v, color=C[ck], height=0.44)
        ax.text(0, y + 0.27, lab, ha="left", va="bottom", fontsize=fs, fontweight="bold")
        ax.text(2, y, txt, va="center", fontsize=fs - 1, fontweight="bold", color="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.35, 2.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sd in ax.spines.values():
        sd.set_visible(False)
    ax.text(0, -0.4, "Percent, LISA-CNN road-sign classifier", ha="left", va="top",
            fontsize=fs - 2, color=C["neutral"])
    sp.save(fig, path)


hook_synth(out / "hook_synth.svg")


# ================================================== TEMPE TIMELINE (NTSB)
def tempe(path, k):
    fig, ax = plt.subplots(figsize=(4.1, 2.3))
    ax.set_xlim(-6.5, 0.9)
    ax.set_ylim(-1.55, 1.35)
    ax.axhline(0, color=C["neutral"], lw=2)
    for t in (-6, -4, -2, 0):
        ax.plot([t, t], [-0.08, 0.08], color=C["neutral"], lw=1.5)
        ax.text(t, -0.3, f"{t} s" if t else "impact", ha="center", va="top", fontsize=11.5)
    detect, recog, hold, alert = (num("tempe_detect_s"), num("tempe_recog_s"),
                                  num("tempe_hold_s"), num("tempe_alert_s"))
    ax.plot([-detect], [0], "o", ms=12, color=C["memory"], zorder=5)
    ax.text(-detect, -0.72, f"first\ndetected\n{val('tempe_detect_s')} s out", ha="center", va="top", fontsize=11.5,
            fontweight="bold", color=C["memory"])
    ax.add_patch(sp.plt.Rectangle((-5.4, -0.18), 4.2, 0.36, color=C["memory_l"], zorder=1))
    ax.text(-3.4, 0.32, "label flips: vehicle,\nother, bicycle", ha="center", va="bottom",
            fontsize=11.5, color=C["memory"])
    a = 1.0 if k >= 2 else 0.0
    ax.plot([-recog], [0], "o", ms=12, color=C["danger"], zorder=5, alpha=a)
    ax.add_patch(sp.plt.Rectangle((-recog, -0.18), hold, 0.36, color=C["danger_l"], zorder=2, alpha=a))
    ax.text(-1.5, -0.72, f"collision\nrecognized\n{val('tempe_recog_s')} s out", ha="center", va="top", fontsize=11.5,
            fontweight="bold", color=C["danger"], alpha=a)
    ax.text(-0.5, 0.32, f"held {val('tempe_hold_s')} s,\nno alert", ha="center", va="bottom", fontsize=11.5,
            fontweight="bold", color=C["danger"], alpha=a)
    ax.plot([-alert], [0], "v", ms=10, color=C["danger"], zorder=5, alpha=a)
    ax.text(0.35, -0.72, f"alert\n{val('tempe_alert_s')} s out", ha="center", va="top", fontsize=11.5,
            color=C["danger"], alpha=a)
    ax.axis("off")
    sp.save(fig, path)


for k in (1, 2):
    tempe(out / f"tempe_step{k}.svg", k)


# ===================================================== THREE PILLARS (SVG)
def pillars(path, k):
    cols = [("Environmental shift", "the world moves", "slices, drift tests",
             "recalibrate, retrain", V["blue"], V["blue_l"]),
            ("Input-level attack", "an adversary picks", "attack suites",
             "harden, filter", V["amber"], V["amber_l"]),
            ("System-level fault", "hardware or code errs", "replay, canaries",
             "isolate, roll back", V["violet"], V["violet_l"])]
    b = [text(450, 30, "Latency, uptime, error rate: all green", V["green"], 24, 700)]
    for i, (name, what, ev, resp, s, f) in enumerate(cols):
        x = 20 + i * 293
        op = 1.0 if k >= i + 1 else 0.0
        b.append(rect(x, 48, 274, 228, f, s, 2.5, 8, op))
        b.append(text(x + 137, 84, name, s, 25, 700, op=op))
        b.append(text(x + 137, 118, what, V["ink"], 21, 400, op=op))
        b.append(text(x + 137, 164, "evidence: " + ev, V["ink"], 20, 400, op=op))
        b.append(text(x + 137, 204, "response:", V["gray"], 20, 700, op=op))
        b.append(text(x + 137, 234, resp, V["ink"], 20, 400, op=op))
    op = 1.0 if k >= 4 else 0.0
    b.append(rect(20, 292, 860, 84, V["red_l"], V["red"], 2.5, 8, op, dash="8,5"))
    b.append(text(450, 325, "Software and pipeline defects cut across all three", V["red"], 23, 700, op=op))
    b.append(text(450, 358, "a preprocessing bug looks like shift; an overflow looks like an attack",
                  V["ink"], 20, 400, op=op))
    svg(path, b, 900, 385)


for k in (1, 2, 3, 4):
    pillars(out / f"pillars_step{k}.svg", k)


# ================================================== FLEET SDC PROBABILITY
def sdc(path, k):
    p = K.SDC1K.event_rate
    n = np.logspace(1, 5, 200)
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.plot(n, 1 - (1 - p) ** n, color=C["danger"], lw=3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(10, 1e5)
    ax.set_xlabel("Accelerators in the job, N")
    ax.set_ylabel("P(silent error\nin an hour)")
    ax.grid(color=C["grid"], lw=0.6)
    pts = [(K.SDC1K.n_devices, f"{val('sdc_n_lo')}: {val('sdc_p_lo')}"),
           (K.SDC10K.n_devices, f"{val('sdc_n_hi')}: {val('sdc_p_hi')}")]
    for i, (x, lab) in enumerate(pts):
        if k >= 2 or i == 0:
            y = 1 - (1 - p) ** x
            ax.plot([x], [y], "o", ms=10, color=C["danger"], mec="white", mew=1.5, zorder=5)
            ax.annotate(lab, (x, y), xytext=(-8, 8), textcoords="offset points",
                        ha="right", fontsize=12, fontweight="bold", color=C["danger"])
    sp.save(fig, path)


for k in (1, 2):
    sdc(out / f"sdc_step{k}.svg", k)


# ===================================================== SHIFT PANELS (SVG)
def shift(path, k):
    b = []
    titles = [("Covariate shift", "P(x) changes", "P(y | x) fixed", V["blue"]),
              ("Label shift", "P(y) changes", "P(x | y) fixed", V["amber"]),
              ("Concept drift", "P(y | x) changes", "P(x) can stay fixed", V["red"])]
    for i, (t, a1, a2, col) in enumerate(titles):
        x0 = 15 + i * 295
        op = 1.0 if k >= i + 1 else 0.0
        b.append(rect(x0, 8, 280, 372, V["frame_l"], V["frame"], 2, 8, op))
        b.append(text(x0 + 140, 42, t, col, 26, 700, op=op))
        b.append(text(x0 + 140, 74, a1, V["ink"], 22, 700, op=op))
        b.append(text(x0 + 140, 102, a2, V["gray"], 21, 400, op=op))
    # panel a
    x0 = 15
    op = 1.0 if k >= 1 else 0
    b.append(f'<line x1="{x0+40}" y1="360" x2="{x0+250}" y2="130" stroke="{V["ink"]}" '
             f'stroke-width="2.5" stroke-dasharray="8,6" opacity="{op}"/>')
    for (cx, cy, filled) in [(65, 200, 0), (95, 245, 0), (125, 215, 0), (90, 330, 1), (130, 345, 1)]:
        b.append(circle(x0 + cx, cy, 10, V["blue"] if filled else "#fff", V["blue"], 3, op))
    for (cx, cy, filled) in [(200, 145, 0), (235, 170, 0), (215, 280, 1), (255, 262, 1), (240, 315, 1)]:
        b.append(circle(x0 + cx, cy, 10, V["amber"] if filled else "#fff", V["amber"], 3, op))
    b.append(text(x0 + 75, 168, "train", V["blue"], 21, 700, op=op))
    b.append(text(x0 + 225, 360, "deploy", V["amber"], 21, 700, op=op))
    # panel b
    x0 = 310
    op = 1.0 if k >= 2 else 0
    base = 340
    for j, (h_tr, h_dep) in enumerate([(100, 45), (100, 175)]):
        bx = x0 + 45 + j * 115
        b.append(f'<rect x="{bx}" y="{base-h_tr}" width="40" height="{h_tr}" fill="{V["blue"]}" opacity="{op}"/>')
        b.append(f'<rect x="{bx+45}" y="{base-h_dep}" width="40" height="{h_dep}" fill="{V["amber"]}" opacity="{op}"/>')
        b.append(text(bx + 42, 368, f"y = {j}", V["ink"], 21, 400, op=op))
    b.append(f'<line x1="{x0+30}" y1="{base}" x2="{x0+250}" y2="{base}" stroke="{V["ink"]}" stroke-width="2" opacity="{op}"/>')
    b.append(text(x0 + 95, 145, "train", V["blue"], 21, 700, op=op))
    b.append(text(x0 + 210, 145, "deploy", V["amber"], 21, 700, op=op))
    # panel c
    x0 = 605
    op = 1.0 if k >= 3 else 0
    b.append(f'<line x1="{x0+40}" y1="350" x2="{x0+250}" y2="130" stroke="{V["ink"]}" '
             f'stroke-width="2.5" stroke-dasharray="8,6" opacity="{op}"/>')
    b.append(f'<line x1="{x0+30}" y1="160" x2="{x0+255}" y2="335" stroke="{V["green"]}" '
             f'stroke-width="3.5" opacity="{op}"/>')
    b.append(text(x0 + 215, 135, "old", V["ink"], 21, 700, op=op))
    b.append(text(x0 + 60, 145, "new", V["green"], 21, 700, op=op))
    for (cx, cy) in [(55, 205), (90, 255), (150, 340), (215, 345)]:
        b.append(circle(x0 + cx, cy, 10, V["blue"], V["blue"], 3, op))
    for (cx, cy) in [(170, 185), (215, 215), (245, 245)]:
        b.append(circle(x0 + cx, cy, 10, "#fff", V["blue"], 3, op))
    for (cx, cy) in [(120, 215), (180, 270), (222, 292)]:
        b.append(circle(x0 + cx, cy, 10, V["red"], V["red"], 3, op))
    svg(path, b, 900, 388)


for k in (1, 2, 3):
    shift(out / f"shift_step{k}.svg", k)


# ======================================================== IMAGENET-V2
IV2 = K.IV2.rows
slope, icpt = K.IV2.result.slope, K.IV2.result.intercept


def imagenet(path, k, size=(5.2, 2.15)):
    fig, ax = plt.subplots(figsize=size)
    ax.plot([40, 95], [40, 95], "--", color=C["neutral"], lw=1.5)
    ax.text(58, 63.5, "equal accuracy", color=C["neutral"], fontsize=11, rotation=33)
    ax.scatter([r["val"] for r in IV2], [r["imagenet_v2"] for r in IV2], s=16,
               color=C["memory"], alpha=0.75, zorder=3)
    if k >= 2:
        xx = np.array([55, 90])
        ax.plot(xx, slope * xx + icpt, color=C["danger"], lw=2.5, zorder=4)
        lab = {"alexnet": ("AlexNet", (25, -6)), "resnet50": ("ResNet-50", (22, -14)),
               "efficientnet-l2-noisystudent": ("EfficientNet-L2", (-8, 10))}
        for r in IV2:
            if r["model"] in lab:
                t, o = lab[r["model"]]
                ax.plot([r["val"]], [r["imagenet_v2"]], "o", ms=10, mfc="none",
                        mec=C["text"], mew=1.5, zorder=5)
                ax.annotate(t, (r["val"], r["imagenet_v2"]), xytext=o, textcoords="offset points",
                            ha="left" if o[0] > 0 else "right", fontsize=12, fontweight="bold")
        ax.text(91.5, 41, f"{val('iv2_models')} models all lose\n{val('iv2_gap_min')} to {val('iv2_gap_max')} "
                f"points, median {val('iv2_gap_median')}", ha="right", va="bottom",
                fontsize=12, fontweight="bold", color=C["danger"])
    ax.set_xlim(54, 92)
    ax.set_ylim(38, 90)
    ax.set_xlabel("ImageNet top-1 accuracy (%)")
    ax.set_ylabel("ImageNet-V2 top-1 (%)")
    ax.grid(color=C["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2):
    imagenet(out / f"imagenet_step{k}.svg", k)


# ===================================================== MEAN-SHIFT TEST
SE, BASE, OBS = K.MEAN.standard_error, K.MEAN.reference_mean, K.MEAN.observed_mean


def meantest(path):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    xs = np.linspace(0.465, 0.535, 400)
    ys = np.exp(-0.5 * ((xs - BASE) / SE) ** 2)
    ax.fill_between(xs, 0, ys, color=C["memory_l"])
    ax.plot(xs, ys, color=C["memory"], lw=2.5)
    ax.axvline(OBS, color=C["danger"], lw=3)
    ax.text(0.548, 0.75, f"observed\n{val('ms_obs')}", ha="right", fontsize=12, fontweight="bold",
            color=C["danger"])
    ax.annotate("", xy=(OBS, 1.1), xytext=(BASE, 1.1),
                arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.8))
    ax.text((BASE + OBS) / 2, 1.16, f"{val('ms_z')} standard\nerrors", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.466, 0.95, f"baseline\n{val('ms_base')}", fontsize=11.5, color=C["memory"], va="top")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(0, 1.5)
    ax.set_xlim(0.46, 0.56)
    ax.set_xticks([0.47, 0.50, 0.53, 0.56])
    ax.set_xlabel(f"Mean over {val('ms_n')} requests")
    sp.save(fig, path)


meantest(out / "meantest.svg")


# ========================================================== FRAUD PSI
sp.hbars(out / "psi_country.svg", [
    ("Other", K.PSI.term("Other"), "danger", f"{val('psi_other')} ({val('psi_other_pct')}%)"),
    ("USA", K.PSI.term("USA"), "memory", val("psi_usa")),
    ("France", K.PSI.term("France"), "memory", val("psi_france")),
    ("UK", K.PSI.term("UK"), "memory", val("psi_uk")),
    ("Germany", K.PSI.term("Germany"), "memory", val("psi_germany")),
], "", log=False, size=(3.3, 2.2),
    xlabel=f"PSI contribution (total {val('psi_total')})")


# ================================================ DRIFT x ERROR GRID (SVG)
def grid(path):
    b = []
    x0, y0, w, h = 120, 10, 375, 155
    cells = [(0, 0, "Probe deeper", "error up, inputs flat:", "concept drift or a fault", V["red_l"], V["red"]),
             (1, 0, "Isolate slices", "error and drift together:", "test the update", V["amber_l"], V["amber"]),
             (0, 1, "Keep serving", "nothing moved", "", V["frame_l"], V["frame"]),
             (1, 1, "Monitor", "inputs moved, error flat:", "find the source", V["blue_l"], V["blue"])]
    for (cx, cy, t, s1, s2, f, s) in cells:
        x, y = x0 + cx * (w + 10), y0 + cy * (h + 10)
        b.append(rect(x, y, w, h, f, s, 2.5, 8))
        b.append(text(x + w / 2, y + 50, t, s if s != V["frame"] else V["gray"], 27, 700))
        b.append(text(x + w / 2, y + 92, s1, V["ink"], 22))
        b.append(text(x + w / 2, y + 122, s2, V["ink"], 22))
    b.append(arrow(x0, 352, x0 + 2 * w + 10, 352, "gray", 3))
    b.append(text(x0 + w + 5, 385, "Input drift statistic (PSI, KS)", V["ink"], 22, 700))
    b.append(arrow(95, y0 + 2 * h + 10, 95, y0, "gray", 3))
    b.append(f'<text x="70" y="{y0 + h + 5}" font-size="22" font-weight="700" fill="{V["ink"]}" '
             f'text-anchor="middle" transform="rotate(-90 70 {y0 + h + 5})">Measured error</text>')
    svg(path, b, 900, 395)


grid(out / "drift_grid.svg")


# ========================================================== CALIBRATION
def calib(path):
    rows = [("CIFAR-100, LeNet 5", "ece_lenet", "ece_lenet_ts"),
            ("CIFAR-100, ResNet 110", "ece_r110", "ece_r110_ts"),
            ("ImageNet, ResNet 152", "ece_r152", "ece_r152_ts")]
    fig, ax = plt.subplots(figsize=(4.8, 1.75))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, uk, tk) in zip(y, rows):
        u, t = num(uk), num(tk)
        ax.barh(yi + 0.19, u, height=0.36, color=C["danger"])
        ax.barh(yi - 0.19, t, height=0.36, color=C["data"])
        ax.text(u + 0.3, yi + 0.19, val(uk), va="center", fontsize=12, fontweight="bold",
                color=C["danger"])
        ax.text(t + 0.3, yi - 0.19, val(tk), va="center", fontsize=12, fontweight="bold",
                color=C["data"])
        ax.text(-0.4, yi, lab, ha="right", va="center", fontsize=12, fontweight="bold")
    ax.text(19.5, 0.25, "as trained", color=C["danger"], fontsize=12, fontweight="bold", ha="right")
    ax.text(19.5, -0.25, "temperature scaled", color=C["data"], fontsize=12, fontweight="bold", ha="right")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 19.5)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("Expected calibration error (percentage points)")
    sp.save(fig, path)


calib(out / "calib.svg")


# ====================================================== RISK-COVERAGE
def riskcov(path, k):
    cov = np.linspace(0.3, 1.0, 200)
    risk = K.SEL.risk_at(cov)            # illustrative shape (Scenarios.SelectiveAnswering90)
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    ax.plot(cov * 100, risk * 100, color=C["memory"], lw=3)
    ax.set_xlim(30, 102)
    ax.set_ylim(0, 12.8)
    ax.set_xlabel("Coverage: % answered")
    ax.set_ylabel("Error on answered (%)")
    ax.grid(color=C["grid"], lw=0.5)
    ax.plot([100], [num("sel_full_pct")], "o", ms=10, color=C["neutral"])
    ax.annotate("answer all", (100, num("sel_full_pct")), xytext=(-8, -3), textcoords="offset points",
                ha="right", fontsize=11.5, fontweight="bold", color=C["neutral"])
    if k >= 2:
        x, y = num("sel_cov_pct"), num("sel_risk_pct")
        ax.plot([x], [y], "o", ms=11, color=C["danger"], zorder=5)
        ax.annotate(f"{val('sel_cov_pct')}% answered,\n{val('sel_risk_pct')}% error", (x, y), xytext=(-14, 4),
                    textcoords="offset points", ha="right", fontsize=11.5, fontweight="bold",
                    color=C["danger"])
    ax.text(32, 12.3, "illustrative", fontsize=10.5, color=C["neutral"], va="top")
    sp.save(fig, path)


for k in (1, 2):
    riskcov(out / f"riskcov_step{k}.svg", k)


# ============================================================ CONFORMAL
def conformal(path):
    n, rank = K.CONF.calibration_count, K.CONF.rank
    rng = np.random.default_rng(3)
    scores = np.sort(rng.gamma(2.0, 0.15, n))
    q = scores[rank - 1]
    fig, ax = plt.subplots(figsize=(4.8, 1.8))
    ax.plot(np.arange(1, n + 1), scores, color=C["memory"], lw=3)
    ax.axvline(rank, color=C["danger"], lw=2.5, ls="--")
    ax.axhline(q, color=C["danger"], lw=1.5, ls=":")
    ax.text(rank - 20, scores[-1] * 0.95, f"rank {val('conf_rank')}\nsets the cutoff", ha="right",
            fontsize=12, fontweight="bold", color=C["danger"], va="top")
    ax.text(30, q + 0.04, "keep labels scoring below", fontsize=12, color=C["danger"])
    ax.set_xlabel(f"Calibration examples sorted by score (n = {val('conf_n')})")
    ax.set_ylabel("Score")
    ax.set_yticks([])
    ax.text(30, scores[-1] * 0.98, "illustrative scores", fontsize=10.5, color=C["neutral"], va="top")
    sp.save(fig, path)


conformal(out / "conformal.svg")


# ======================================================== JAILBREAK (Zou)
def zou(path, k):
    rows = ["Request only", "+ \u201cSure, here\u2019s\u201d", "+ GCG suffix", "+ GCG ensemble"]
    keys = ("only", "sure", "gcg", "ens")
    g35 = [num(f"gpt35_{k_}") for k_ in keys]
    g4 = [num(f"gpt4_{k_}") for k_ in keys]
    fig, ax = plt.subplots(figsize=(5.3, 1.95))
    y = np.arange(4)[::-1]
    for i, yi in enumerate(y):
        a = 1.0 if (i < 2 or k >= 2) else 0.0
        ax.barh(yi + 0.2, g35[i], height=0.38, color=C["danger"], alpha=a)
        ax.barh(yi - 0.2, g4[i], height=0.38, color=C["compute"], alpha=a)
        ax.text(g35[i] + 1, yi + 0.2, val(f"gpt35_{keys[i]}"), va="center", fontsize=11.5, fontweight="bold",
                color=C["danger"], alpha=a)
        ax.text(g4[i] + 1, yi - 0.2, val(f"gpt4_{keys[i]}"), va="center", fontsize=11.5, fontweight="bold",
                color=C["compute"], alpha=a)
        ax.text(-1.5, yi, rows[i], ha="right", va="center", fontsize=12, fontweight="bold")
    ax.text(99, 3.3, "GPT-3.5", color=C["danger"], fontsize=12.5, fontweight="bold", ha="right")
    ax.text(99, 2.65, "GPT-4", color=C["compute"], fontsize=12.5, fontweight="bold", ha="right")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Attack success, % of {val('gcg_behaviors')} harmful requests")
    sp.save(fig, path)


for k in (1, 2):
    zou(out / f"zou_step{k}.svg", k)


# ===================================================== ATTACKER ACCESS (SVG)
def access(path, k):
    rows = [("White-box", "weights, gradients", "one backward pass",
             "adaptive white-box", V["red"], V["red_l"]),
            ("Transfer", "a surrogate model", "craft offline, submit",
             "other models' attacks", V["amber"], V["amber_l"]),
            ("Query", "returned labels", "thousands of queries",
             "query budgets", V["violet"], V["violet_l"]),
            ("Physical", "the scene itself", "survive the camera",
             "drive-by tests", V["blue"], V["blue_l"])]
    b = [text(20, 26, "Attacker has", V["gray"], 20, 700, "start"),
         text(495, 26, "Attack cost", V["gray"], 20, 700),
         text(770, 26, "Test it demands", V["gray"], 20, 700)]
    for i, (n, has, cost, ev, s, f) in enumerate(rows):
        y = 40 + i * 88
        op = 1.0 if k >= i + 1 else 0.0
        b.append(rect(10, y, 880, 78, f, s, 2.5, 8, op))
        b.append(text(25, y + 34, n, s, 25, 700, "start", op))
        b.append(text(25, y + 64, has, V["ink"], 20, 400, "start", op))
        b.append(text(495, y + 47, cost, V["ink"], 21, 400, "middle", op))
        b.append(text(770, y + 47, ev, V["ink"], 21, 700, "middle", op))
    svg(path, b, 900, 395)


for k in (1, 2, 3, 4):
    access(out / f"access_step{k}.svg", k)


# ============================================================ PGD PASSES
def pgd(path, k):
    F, P = K.FGSM, K.PGD
    rows = [("Standard step", [P.standard_passes], ["update"]),
            ("FGSM, clean + adv.", [F.attack_passes, F.update_passes], ["attack", f"{F.update_passes}\u00d7 batch"]),
            ("PGD-7 training", [P.attack_passes, P.update_passes], [f"{P.attack_passes} attack steps", "update"])]
    fig, ax = plt.subplots(figsize=(5.4, 1.55))
    for i, (lab, parts, names) in enumerate(rows):
        y = 2 - i
        shown = i < 2 or k >= 2
        left = 0
        for p_, nm in zip(parts, names):
            c = C["danger"] if "attack" in nm else C["memory"]
            ax.barh(y, p_, left=left, height=0.62, color=c, edgecolor="white", lw=1.5,
                    alpha=1.0 if shown else 0.0)
            if p_ >= 2:
                ax.text(left + p_ / 2, y, nm, ha="center", va="center", color="white",
                        fontsize=11, fontweight="bold", alpha=1.0 if shown else 0.0)
            left += p_
        total = sum(parts)
        ax.text(total + 0.15, y, f"{total} passes" if total > 1 else "1 pass", va="center",
                fontsize=12, fontweight="bold", color=C["text"], alpha=1.0 if shown else 0.0)
        ax.text(-0.15, y, lab, ha="right", va="center", fontsize=12, fontweight="bold")
        if i == 2 and k == 1:
            ax.text(0.2, y, "???", va="center", fontsize=14, fontweight="bold", color=C["danger"])
    ax.set_xlim(0, 9.6)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Forward and backward passes per batch")
    sp.save(fig, path)


for k in (1, 2):
    pgd(out / f"pgd_step{k}.svg", k)


# ========================================================== ROBUSTBENCH
RB = K.RB.rows
YEARS = K.RB.years
BROKEN = {id(r) for r in K.RB.broken}


def robustbench(path, k):
    rng = np.random.default_rng(0)
    jit = [rng.uniform(-0.2, 0.2) for _ in RB]
    fig, ax = plt.subplots(figsize=(5.3, 2.0))
    xs = [int(r["year"]) + j for r, j in zip(RB, jit)]
    ax.scatter(xs, [r["clean_acc"] for r in RB], s=20, color=C["neutral"], alpha=0.6, zorder=2)
    ax.text(2017.45, 90, "clean", color=C["neutral"], fontsize=12.5, fontweight="bold", ha="left")
    if k >= 2:
        ax.scatter(xs, [r["autoattack_acc"] for r in RB], s=22, color=C["danger"], zorder=3)
        bx, by = [], []
        for y in YEARS:
            run = K.RB.best_so_far(y)
            bx.append(y - 0.5)
            by.append(run)
        bx.append(YEARS[-1] + 0.5)
        by.append(run)
        ax.step(bx, by, where="post", color=C["danger"], lw=2.2, zorder=4)
        ax.text(2017.45, 14, "under attack", color=C["danger"], fontsize=12.5, fontweight="bold")
        ax.text(YEARS[-1] + 0.65, num("rb_last_pct"), f"top\n{val('rb_last_pct')}%", ha="left", va="center", color=C["danger"], fontsize=12.5, fontweight="bold")
    if k >= 3:
        for r, x in zip(RB, xs):
            if id(r) in BROKEN:
                ax.plot([x], [r["autoattack_acc"]], "o", ms=13, mfc="none", mec=C["text"], mew=1.8, zorder=5)
        ax.text(2020.9, 10, f"{val('rb_broken')} published defenses\nbelow {val('rb_floor_pct')}% under AutoAttack", fontsize=12,
                fontweight="bold", color=C["text"], va="bottom")
    ax.set_xticks(YEARS)
    ax.set_xlim(2017.4, 2025.4)
    ax.set_ylim(-3, 100)
    ax.set_ylabel("CIFAR-10 accuracy (%)")
    ax.set_xlabel("Publication year")
    ax.grid(axis="y", color=C["grid"], lw=0.5)
    sp.save(fig, path)


for k in (1, 2, 3):
    robustbench(out / f"robustbench_step{k}.svg", k)


# ======================================================== CERTIFIED RADIUS
sp.hbars(out / "certified.svg", [
    ("Certified radius", K.CERT.radius, "data", val("cert_r")),
    (f"{val('eps_levels')}/{val('eps_scale')} box corner", K.CERT.box_corner_l2, "danger", val("cert_corner")),
], "", log=False, size=(3.4, 1.25), xlabel="Euclidean length")


# ============================================================== BADNETS
sp.hbars(out / "badnets.svg", [
    ("Honest model, clean signs", num("bn_clean"), "neutral", f"{val('bn_clean')}% correct"),
    ("Backdoored, clean signs", num("bn_back_clean"), "memory", f"{val('bn_back_clean')}% correct"),
    ("Backdoored, sticker on stop", num("bn_attack"), "danger", f"{val('bn_attack')}% speed limit"),
], "", log=False, size=(5.4, 1.6), xlabel="Percent (BadNets, Table 4)")


# ===================================================== POISON LAYERS (SVG)
def layers(path, k):
    stages = [("Ingress", "source identity,", "least privilege", "trusted source", "turns bad", V["blue"], V["blue_l"]),
              ("Sanitize", "outliers, clusters,", "spectral filters", "clean-label poison", "is not an outlier", V["violet"], V["violet_l"]),
              ("Robust training", "Huber loss,", "robust aggregation", "bounds a residual,", "not an update", V["amber"], V["amber_l"]),
              ("Release checks", "triggers, canaries,", "target slices", "a trigger nobody", "tested", V["green"], V["green_l"])]
    b = []
    for i, (n, d1, d2, b1, b2, s, f) in enumerate(stages):
        x = 10 + i * 222
        op = 1.0 if k >= i + 1 else 0.0
        b.append(rect(x, 10, 208, 250, f, s, 2.5, 8, op))
        b.append(text(x + 104, 46, n, s, 24, 700, op=op))
        b.append(text(x + 104, 88, d1, V["ink"], 20, op=op))
        b.append(text(x + 104, 116, d2, V["ink"], 20, op=op))
        b.append(text(x + 104, 170, "blind spot", V["red"], 20, 700, op=op))
        b.append(text(x + 104, 200, b1, V["ink"], 20, op=op, italic=True))
        b.append(text(x + 104, 228, b2, V["ink"], 20, op=op, italic=True))
        if i < 3:
            b.append(arrow(x + 208, 135, x + 221, 135, "gray", 3, op))
    op = 1.0 if k >= 5 else 0.0
    b.append(f'<path d="M 790 265 L 790 300 L 114 300 L 114 268" fill="none" stroke="{V["red"]}" '
             f'stroke-width="3" stroke-dasharray="9,6" marker-end="url(#a-red)" opacity="{op}"/>')
    b.append(text(450, 338, "Lineage traces a bad behavior to the batches that taught it",
                  V["red"], 22, 700, op=op))
    svg(path, b, 900, 350)


for k in (1, 2, 3, 4, 5):
    layers(out / f"layers_step{k}.svg", k)


# ============================================================ BIT FLIP (SVG)
FLIP = K.FLIP
bits = struct.unpack(">I", struct.pack(">f", FLIP.value))[0]   # the stored word, for its bit pattern


def bitflip(path, k):
    b = []
    word = format(bits, "032b")
    flipped = format(bits ^ (1 << FLIP.flipped_bit), "032b")

    def row(y, s, hl, op=1.0):
        out_ = []
        cells = [("sign", s[0])] + [("exp", c) for c in s[1:1 + FLIP.exponent_bits]] + [("man", "")]
        x = 10
        for j, (kind, ch) in enumerate(cells):
            wd = 290 if kind == "man" else 56
            f = V["red_l"] if (hl and j == 1) else (V["blue_l"] if kind == "exp" else V["frame_l"])
            st = V["red"] if (hl and j == 1) else V["frame"]
            out_.append(rect(x, y, wd, 52, f, st, 3 if (hl and j == 1) else 1.5, 3, op))
            t = ch if kind != "man" else f"mantissa, {val('flip_frac_bits')} bits"
            out_.append(text(x + wd / 2, y + 35, t, V["red"] if (hl and j == 1) else V["ink"],
                             25 if kind != "man" else 20, 700 if kind != "man" else 400, op=op))
            x += wd + 4
        return out_

    b.append(text(10, 30, f"w = {val('flip_w')}", V["ink"], 24, 700, "start"))
    b.append(text(160, 30, f"sign | exponent, {val('flip_exp_bits')} bits | mantissa", V["gray"], 20, 400, "start"))
    b += row(45, word, False)
    op = 1.0 if k >= 2 else 0.0
    b.append(arrow(98, 102, 98, 160, "red", 3.5, op))
    b.append(text(125, 140, f'bit {val("flip_bit")} flips: value \u00d7 2<tspan baseline-shift="super" font-size="15">{val("flip_scale")}</tspan>', V["red"], 23, 700, "start", op))
    b += row(165, flipped, True, op)
    b.append(text(10, 248, f'w\u2032 \u2248 {val("flip_mant")} \u00d7 10<tspan baseline-shift="super" font-size="15">{val("flip_exp")}</tspan>: finite, so every NaN check passes', V["red"], 23, 700, "start", op))
    op = 1.0 if k >= 3 else 0.0
    checks = [("Range check", "|w| \u2264 bound, one compare per value", "compute"),
              ("Golden canary", "replay fixed inputs, compare outputs", "+ communication"),
              ("Cross-replica", "run a sampled request twice", "2\u00d7 on the sample")]
    for i, (n, d, c) in enumerate(checks):
        y = 285 + i * 38
        b.append(text(10, y + 22, n, V["green"], 21, 700, "start", op))
        b.append(text(185, y + 22, d, V["ink"], 21, 400, "start", op))
        b.append(text(890, y + 22, c, V["violet"], 20, 700, "end", op))
    svg(path, b, 900, 400)


for k in (1, 2, 3):
    bitflip(out / f"bitflip_step{k}.svg", k)


# ======================================================== REPLAY HARNESS (SVG)
def replay(path, k):
    b = []
    b.append(rect(5, 125, 120, 100, V["frame_l"], V["frame"], 2))
    b.append(text(65, 168, "golden", V["ink"], 21, 700))
    b.append(text(65, 196, "requests", V["ink"], 21, 700))
    for i, (lab, col, fill) in enumerate([("pipeline v1", V["blue"], V["blue_l"]),
                                          ("pipeline v2", V["amber"], V["amber_l"])]):
        y = 20 + i * 220
        b.append(arrow(125, 175, 160, y + 45, "gray", 2.5))
        b.append(rect(163, y, 150, 90, fill, col, 2.5))
        b.append(text(238, y + 38, lab, col, 21, 700))
        b.append(text(238, y + 68, "raw: equal", V["ink"], 19))
        b.append(arrow(313, y + 45, 340, y + 45, "gray", 2.5))
        b.append(rect(343, y, 110, 90, "#fff", col, 2.5))
        b.append(text(398, y + 52, "tensor", V["ink"], 21, 700))
    op = 1.0 if k >= 2 else 0.0
    b.append(rect(470, 115, 125, 120, V["red_l"], V["red"], 2.5, 8, op))
    b.append(text(532, 155, "differ by", V["red"], 20, 700, op=op))
    b.append(text(532, 182, "a constant", V["red"], 20, 700, op=op))
    b.append(text(532, 209, "factor", V["red"], 20, 700, op=op))
    b.append(arrow(453, 65, 500, 113, "red", 2.5, op))
    b.append(arrow(453, 285, 500, 237, "red", 2.5, op))
    svg(path, b, 600, 335)


for k in (1, 2):
    replay(out / f"replay_step{k}.svg", k)


print("figures written to", out)
