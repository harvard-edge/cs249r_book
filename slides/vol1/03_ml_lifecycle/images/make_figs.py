#!/usr/bin/env python3
"""Figures for Lecture 3 (ML Lifecycle).

Every value comes from calcs.py, which reads MLSysIM: the retinopathy-screener
scenarios (Scenarios.RetinopathyScreenerRequirements, ...LateMemoryLimit,
AlwaysNormalScreener, FundusPhotoAt224, RetinopathyScreeningCloudVersusEdge,
RetinopathyMonitoringCadences), Literature.Surveys (CrowdFlower 2016), and
ReferenceStats.TorchvisionParameterCounts. Figure code holds only layout
(coordinates, sizes, colors) and labels; the chapter sections each figure
redraws are noted at the figure.
  - Six stages and their artifacts: Sec. 3.1. Stage contracts: table 2;
    audit case: Checkpoint 3.3 (Sec. 3.11).
  - ROC: the floors and the illustrative clinic point drawn over the measured
    curve of Gulshan et al., JAMA 2016, Fig. 2A (inset).
  - 224-pixel demonstration: fundus photo by Hao et al. (CC BY 4.0, Wikimedia
    Commons); the scenario records the photo's size and the measured lesion
    widths, and shrink_demo checks the file against it.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as K  # noqa: E402  (records every value; figures read them)
from deckvals import val, num  # noqa: E402

out = Path(__file__).parent
C = sp.C
FONT = "Helvetica, Arial, sans-serif"


# -----------------------------------------------------------------------------
# SVG helpers for diagrams (900 x 500 canvas, text kept as text)
# -----------------------------------------------------------------------------
def svg_open(w=900, h=500):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">',
            '<defs>'
            + "".join(
                f'<marker id="ar{k}" viewBox="0 0 10 10" refX="8" refY="5" '
                f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
                for k, col in (("n", "#555555"), ("d", C["data"]),
                               ("r", C["danger"]), ("b", C["memory"]),
                               ("l", "#BBBBBB")))
            + '</defs>',
            f'<rect width="{w}" height="{h}" fill="white"/>']


def text(x, y, s, size=20, color=None, weight="normal", anchor="middle",
         opacity=1.0):
    color = color or C["text"]
    s = s.replace("&", "&amp;")
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
            f'opacity="{opacity}">{s}</text>')


def line(x1, y1, x2, y2, color="#555555", w=3, marker="n", dash=None,
         opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#ar{marker})"' if marker else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{w}"{d}{m} opacity="{opacity}"/>')


def save_svg(parts, name):
    (out / name).write_text("\n".join(parts + ["</svg>"]) + "\n")


# -----------------------------------------------------------------------------
# 1. The lifecycle loop. Six stages in two rows; the flow runs 1-2-3 along the
#    top, down the right, 4-5-6 back along the bottom, and monitoring returns
#    evidence up the left side.
# -----------------------------------------------------------------------------
BW, BH = 250, 100
TOP, BOT = 140, 380
POS = {1: (150, TOP), 2: (450, TOP), 3: (750, TOP),
       4: (750, BOT), 5: (450, BOT), 6: (150, BOT)}
NAMES = {1: ("Define the", "problem"), 2: ("Collect and", "prepare data"),
         3: ("Develop and", "train model"), 4: ("Evaluate and", "validate"),
         5: ("Deploy and", "integrate"), 6: ("Monitor and", "maintain")}
DAM = {1: ("sets all three", C["text"], "#EEEEEE"), 2: ("Data", C["data"], C["data_l"]),
       3: ("Algorithm", C["compute"], C["compute_l"]), 4: ("verifies", C["neutral"], "#F2F2F2"),
       5: ("Machine", C["memory"], C["memory_l"]), 6: ("verifies", C["neutral"], "#F2F2F2")}
LOOP_H = 500


def stage_box(k, fill="#F4F4F4", stroke="#777777", tcol=None, opacity=1.0,
              sw=2.5):
    x, y = POS[k]
    a, b = NAMES[k]
    tcol = tcol or C["text"]
    return [f'<rect x="{x - BW / 2}" y="{y - BH / 2}" width="{BW}" height="{BH}" '
            f'rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'opacity="{opacity}"/>',
            text(x - BW / 2 + 18, y - BH / 2 + 28, str(k), 24, stroke, "bold",
                 opacity=opacity),
            text(x + 10, y - 4, a, 26, tcol, "bold", opacity=opacity),
            text(x + 10, y + 27, b, 26, tcol, "bold", opacity=opacity)]


def flow_arrows(color="#555555", opacity=1.0, marker="n"):
    p = []
    g = BW / 2 + 4
    for a, b in ((1, 2), (2, 3)):
        p.append(line(POS[a][0] + g, TOP, POS[b][0] - g - 3, TOP, color, 3.5, marker, opacity=opacity))
    for a, b in ((4, 5), (5, 6)):
        p.append(line(POS[a][0] - g, BOT, POS[b][0] + g + 3, BOT, color, 3.5, marker, opacity=opacity))
    p.append(line(750, TOP + BH / 2 + 6, 750, BOT - BH / 2 - 10, color, 3.5, marker, opacity=opacity))
    return p


ARTIFACTS = [  # (x, y, line1, line2, anchor)
    (300, 40, "targets", "and limits", "middle"),
    (600, 40, "versioned", "dataset", "middle"),
    (768, 252, "trained", "weights", "start"),
    (600, 463, "release", "manifest", "middle"),
    (300, 463, "running", "service", "middle"),
]


def artifacts(opacity=1.0, color=None):
    color = color or C["neutral"]
    p = []
    for x, y, a, b, align in ARTIFACTS:
        p.append(text(x, y, a, 23, color, "normal", align, opacity))
        p.append(text(x, y + 26, b, 23, color, "normal", align, opacity))
    return p


def return_arrow(opacity=1.0, label=True):
    p = [line(150, BOT - BH / 2 - 6, 150, TOP + BH / 2 + 10, C["data"], 4.5, "d",
              dash="10,6", opacity=opacity)]
    if label:
        p.append(text(168, 254, "monitoring", 23, C["data"], "bold", "start", opacity))
        p.append(text(168, 281, "evidence", 23, C["data"], "bold", "start", opacity))
    return p


def loop(name, show_art=True, show_return=True, hl=None, dam=False):
    p = svg_open()
    for k in range(1, 7):
        if dam:
            tag, col, fill = DAM[k]
            p += stage_box(k, fill=fill, stroke=col)
            x, y = POS[k]
            p.append(text(x + (105 if k == 1 else 0), y + BH / 2 + 28, tag, 23, col, "bold"))
        elif hl is None:
            p += stage_box(k)
        elif k == hl:
            p += stage_box(k, fill="#FBE3E6", stroke="#A51C30", sw=4)
        else:
            p += stage_box(k, opacity=0.35)
    op = 1.0 if hl is None else 0.35
    p += flow_arrows(opacity=op)
    if show_art:
        p += artifacts(opacity=op)
    if show_return:
        p += return_arrow(opacity=op)
    save_svg(p, name)


loop("loop_step1.svg", show_art=False, show_return=False)
loop("loop_step2.svg", show_art=True, show_return=False)
loop("loop_step3.svg")


# A compact loop for a column slot (the diagnose and closing slides): the same
# six stages with the short names the slides use, in three rows of two, drawn
# at the column's shape so the names read at slide scale. The flow runs down
# the left (1-2-3), across the bottom, up the right (4-5-6), and monitoring
# evidence returns across the top. tint colors stages by the slide's key.
CPOS = {1: (130, 115), 2: (130, 275), 3: (130, 435),
        4: (430, 435), 5: (430, 275), 6: (430, 115)}
CNAMES = {1: "Define", 2: "Data", 3: "Train", 4: "Validate", 5: "Deploy", 6: "Monitor"}
CW, CH = 210, 84


def loop_compact(name, tint=None):
    tint = tint or {}
    p = svg_open(560, 500)
    for k in range(1, 7):
        x, y = CPOS[k]
        col, fill = tint.get(k, ("#777777", "#F4F4F4"))
        tcol = col if k in tint else C["text"]
        p.append(f'<rect x="{x - CW / 2}" y="{y - CH / 2}" width="{CW}" height="{CH}" '
                 f'rx="12" fill="{fill}" stroke="{col}" stroke-width="{4 if k in tint else 2.5}"/>')
        p.append(text(x, y + 12, f"{k}  {CNAMES[k]}", 34, tcol, "bold"))
    g = CH / 2 + 4
    for a, b in ((1, 2), (2, 3)):
        p.append(line(130, CPOS[a][1] + g, 130, CPOS[b][1] - g - 4, "#555555", 3.5))
    for a, b in ((4, 5), (5, 6)):
        p.append(line(430, CPOS[a][1] - g, 430, CPOS[b][1] + g + 4, "#555555", 3.5))
    p.append(line(130 + CW / 2 + 4, 435, 430 - CW / 2 - 8, 435, "#555555", 3.5))
    p.append(line(430 - CW / 2 - 4, 115, 130 + CW / 2 + 10, 115, C["data"], 4.5, "d",
                  dash="10,6"))
    p.append(text(280, 40, "monitoring evidence", 28, C["data"], "bold"))
    save_svg(p, name)


loop_compact("loop_compact.svg")
loop_compact("loop_compact_fix.svg", tint={1: (C["compute"], C["compute_l"]),
                                           2: (C["data"], C["data_l"]),
                                           5: (C["memory"], C["memory_l"])})


# Propagation on the loop: the 500 MB limit found at stage 5 invalidates the
# artifacts behind it; it belongs at stage 1.
def badge(x, y, s, col, fill):
    w = 14 * len(s) + 30
    return [f'<rect x="{x - w / 2}" y="{y - 21}" width="{w}" height="40" rx="20" '
            f'fill="{fill}" stroke="{col}" stroke-width="2.5"/>',
            text(x, y + 8, s, 23, col, "bold")]


def prop(step):
    p = svg_open()
    redo = {2, 3, 4, 5} if step >= 2 else set()
    for k in range(1, 7):
        if k in redo:
            p += stage_box(k, fill=C["danger_l"], stroke=C["danger"])
        elif step == 3 and k == 1:
            p += stage_box(k, fill=C["data_l"], stroke=C["data"], sw=4)
        else:
            p += stage_box(k)
    p += flow_arrows()
    p += return_arrow(label=False)
    # the limit appears at deployment
    p += badge(450, 474, f"{val('budget_mb')} MB limit found here", C["danger"], "white")
    if step >= 2:
        # backward arrow inside the loop from 5 to 2 (the rework path)
        p.append(line(450, BOT - BH / 2 - 8, 450, TOP + BH / 2 + 12, C["danger"], 4.5,
                      "r", dash="9,6"))
        p.append(text(468, 268, "redo", 24, C["danger"], "bold", "start"))
    if step == 3:
        p += badge(170, 26, f"state it here, day {val('day_request')}", C["data"], "white")
    save_svg(p, f"propagation_step{step}.svg")


for s in (1, 2, 3):
    prop(s)


# -----------------------------------------------------------------------------
# 2. Constraints travel both ways (Sec. 3.2)
# -----------------------------------------------------------------------------
def chain_box(x, y, a, b, col, fill, tag=None, w=266):
    p = [f'<rect x="{x - w / 2}" y="{y - 46}" width="{w}" height="92" rx="10" '
         f'fill="{fill}" stroke="{col}" stroke-width="3"/>',
         text(x, y - 6, a, 25, C["text"], "bold"),
         text(x, y + 24, b, 25, C["text"], "bold")]
    if tag:
        p.append(text(x, y - 56, tag, 22, col, "bold"))
    return p


def directions(step):
    p = svg_open(900, 420)
    xs = (138, 450, 762)
    # backward: a deployment limit reaches the training data (right to left)
    p.append(text(5, 32, "Backward", 26, C["danger"], "bold", "start"))
    p += chain_box(xs[2], 120, f"{val('up_uplink_mbit')} Mb/s uplink", "at the clinic", C["danger"], C["danger_l"], "stage 5")
    p += chain_box(xs[1], 120, "compress photos", "before upload", C["danger"], C["danger_l"])
    p += chain_box(xs[0], 120, "train on", "compressed photos", C["danger"], C["danger_l"], "stage 2")
    p.append(line(626, 120, 589, 120, C["danger"], 5, "r"))
    p.append(line(314, 120, 277, 120, C["danger"], 5, "r"))
    if step >= 2:
        p.append(text(5, 242, "Forward", 26, C["memory"], "bold", "start"))
        p += chain_box(xs[0], 340, "lesions need", "high resolution", C["memory"], C["memory_l"], "stage 2")
        p += chain_box(xs[1], 340, "larger input", "to the model", C["memory"], C["memory_l"], "stage 3")
        p += chain_box(xs[2], 340, "more operations", "per inference", C["memory"], C["memory_l"], "stage 5")
        p.append(line(274, 340, 311, 340, C["memory"], 5, "b"))
        p.append(line(586, 340, 623, 340, C["memory"], 5, "b"))
    save_svg(p, f"directions_step{step}.svg")


directions(1)
directions(2)


# -----------------------------------------------------------------------------
# 3. The day-153 timeline (Sec. 3.1.2). Events are evenly spaced, not to scale.
# -----------------------------------------------------------------------------
EVENTS = [  # (day key, line 1, line 2, kind)
    ("day_request", "Request:", "one sentence", "good"),
    ("day_val_a", f"{val('val_a_pct')}%", "sensitivity", "good"),
    ("day_val_b", f"{val('val_b_pct')}%", "after tuning", "good"),
    ("day_handoff", "Handed to", "deployment", "warn"),
    ("day_found", "Model", f"needs {val('model_gb')} GB", "bad"),
    ("day_check", "Reserved:", f"{val('budget_mb')} MB", "bad"),
    ("day_discard", f"{val('months_lost_word').capitalize()} months", "discarded", "bad"),
]


def timeline(step):
    show = {1: 3, 2: 6, 3: 7}[step]
    p = svg_open(900, 215)
    xs = [70, 196, 322, 448, 574, 700, 826]
    p.append(line(10, 85, 890, 85, "#999999", 5, None))
    for i, ((day, a, b, kind), x) in enumerate(zip(EVENTS, xs)):
        if i >= show:
            continue
        col = {"good": C["data"], "warn": C["neutral"], "bad": C["danger"]}[kind]
        big = day == "day_discard"
        p.append(f'<circle cx="{x}" cy="85" r="{19 if big else 13}" fill="{col}" '
                 f'stroke="white" stroke-width="3"/>')
        p.append(text(x, 45, f"Day {val(day)}", 24, col, "bold"))
        p.append(text(x, 140, a, 21, C["text"], "bold" if big else "normal"))
        p.append(text(x, 166, b, 21, C["text"], "bold" if big else "normal"))
    p.append(text(450, 207, "Events evenly spaced, not to scale", 16, C["neutral"]))
    save_svg(p, f"day153_step{step}.svg")


for s in (1, 2, 3):
    timeline(s)


# -----------------------------------------------------------------------------
# 4. Data plots
# -----------------------------------------------------------------------------
# Model memory against the clinic reservation (Sec. 3.1.2, Sec. 3.4)
# Weights of three torchvision models: exact parameter counts of a torchvision
# 0.28.0 run (ReferenceStats.TorchvisionParameterCounts) at FP32, via calcs.py.
_mb = {k: K.m(n * K.mlsysim.BYTES_FP32, "MB") for k, n in K.TORCHVISION_RUN}
sp.hbars(out / "memory_fit.svg", [
    (f"Model on day {val('day_found')}", K.m(K.LATE.model_memory, "MB"), "danger",
     f"{val('model_gb')} GB: {val('overrun_x')}× the reservation"),
    ("Reserved for the model", K.m(K.REQ.memory_budget, "MB"), "neutral", f"about {val('budget_mb')} MB"),
    ("VGG-16 weights", _mb["vgg16"], "neutral", f"{val('mb_vgg16')} MB"),
    ("ResNet-50 weights", _mb["resnet50"], "compute", f"{val('mb_resnet50')} MB"),
    ("MobileNetV2 weights", _mb["mobilenet_v2"], "compute",
     f"{val('mb_mobilenet_v2')} MB"),
], "MB", size=(5.8, 2.15), xlabel="Memory (MB, log scale; weights in FP32)")


# Where practitioners say the time goes (CrowdFlower 2016, chapter figure 4).
# Step 1 hides the algorithms bar for the predict question; step 2 shows it.
def time_sinks(step):
    rows = []
    for name, v in K.SURVEY:
        ck = "data" if v in K.DATA_WORK else ("compute" if "algorithms" in name
                                               else "neutral")
        if step == 1 and "algorithms" in name:
            rows.append((name, 0, ck, "?"))
        else:
            rows.append((name, v, ck, f"{v:.0f}%"))
    sp.hbars(out / f"time_sinks_step{step}.svg", rows, "%", log=False,
             size=(5.8, 2.9),
             xlabel="Share of respondents naming it their biggest time sink (%)")


time_sinks(1)
time_sinks(2)


# Repair schedule (Example 3.1): each task waits for the one before
def gantt():
    tasks, start = [], 0          # each task starts when the one before ends
    for (name, d), key in zip(K.REPAIR, ("rep_prep", "rep_train", "rep_val", "rep_pack")):
        tasks.append((name, start, d, key))
        start += d
    total = num("rep_total")
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.3))
    for i, (name, s, d, key) in enumerate(tasks):
        y = len(tasks) - 1 - i
        ax.barh(y, d, left=s, color=C["danger"], height=0.6)
        ax.text(-0.15, y, name, ha="right", va="center", fontsize=13,
                fontweight="bold", color=C["text"])
        ax.text(s + d / 2, y, f"{val(key)} d", ha="center", va="center", fontsize=12,
                fontweight="bold", color="white")
    ax.annotate("", xy=(total, -0.75), xytext=(0, -0.75),
                arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.5))
    ax.text(total / 2, -1.1, f"{val('rep_total')} days in sequence", ha="center", va="top", fontsize=13,
            fontweight="bold", color=C["text"])
    ax.set_xlim(0, total + 0.2)
    ax.set_ylim(-1.6, len(tasks) - 0.4)
    ax.set_yticks([])
    ax.set_xticks(range(0, int(total) + 1))
    ax.set_xlabel("Days")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "repair_gantt.svg")


gantt()


# Accuracy trap: 100 patients, 10 with disease, a model that says "normal"
def patients(step):
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.7))
    for i in range(K.TRAP.population):
        r, c = divmod(i, 10)              # a 10-wide grid
        sick = i >= K.TRAP.healthy
        if step == 1:
            col = C["danger"] if sick else "#BBBBBB"
        else:
            col = C["danger"] if sick else C["data"]
        ax.add_patch(sp.plt.Rectangle((c * 1.1, (9 - r) * 1.1), 0.9, 0.9,
                                      color=col))
        if step == 2 and sick:
            ax.text(c * 1.1 + 0.45, (9 - r) * 1.1 + 0.42, "×", ha="center",
                    va="center", color="white", fontsize=12, fontweight="bold")
    x0 = 12.3
    if step == 1:
        ax.text(x0, 8.3, f"{val('trap_healthy')} healthy", fontsize=15, fontweight="bold", color="#888888")
        ax.text(x0, 6.3, f"{val('trap_sick')} with disease", fontsize=15, fontweight="bold",
                color=C["danger"])
    else:
        ax.text(x0, 8.6, "Model says “normal” to all", fontsize=13, color=C["text"])
        ax.text(x0, 6.4, f"Accuracy  {val('trap_acc')}%", fontsize=17, fontweight="bold",
                color=C["data"])
        ax.text(x0, 4.2, f"Sensitivity  {val('trap_sens')}%", fontsize=17, fontweight="bold",
                color=C["danger"])
        ax.text(x0, 2.3, f"{val('trap_missed')} of {val('trap_sick')} cases missed", fontsize=13,
                color=C["danger"])
    ax.text(0, -0.9, "Illustrative population", fontsize=9, color=C["neutral"])
    ax.set_xlim(-0.2, 23)
    ax.set_ylim(-1.3, 11.2)
    ax.set_aspect("equal")
    ax.axis("off")
    sp.save(fig, out / f"patients_step{step}.svg")


patients(1)
patients(2)


# The measured ROC curve (Gulshan et al., JAMA 2016, Fig. 2A inset: EyePACS-1,
# 1 - specificity 0 to 30 percent, sensitivity 70 to 100 percent), cropped from
# the published PDF page rendered at 300 dpi. The floors from stage 1 and the
# illustrative clinic point are drawn over it. Axis calibration: tick marks
# located on the 300 dpi page render (x = 0 at column 571, x = 30 at 1125;
# y = 70 at row 1116, y = 100 at row 551.5), shifted by the crop origin.
CROP0 = (505, 525)
X0, X30, Y70, Y100 = 571, 1125, 1116, 551.5
CLINIC_POINT = tuple(100 * x for x in K.REQ.field_operating_point)   # illustrative (spec, sens), %
SPEC_FLOOR, SENS_FLOOR = num("floor_spec"), num("floor_sens")


def roc_px(spec, sens, k):
    fpr = 100 - spec
    x = X0 + (X30 - X0) * fpr / 30 - CROP0[0]
    y = Y70 - (Y70 - Y100) * (sens - 70) / 30 - CROP0[1]
    return x * k, y * k


def roc_real(step):
    from PIL import Image, ImageDraw
    k = 3                                      # upsample so overlays are crisp
    base = Image.open(out / "gulshan2016_fig2a_inset.png").convert("RGB")
    im = base.resize((base.width * k, base.height * k), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    red = (165, 28, 48)
    if step >= 2:
        # pass corner: specificity and sensitivity at or above their floors
        # (axes: 1 - specificity 0 to 30 percent, sensitivity 70 to 100 percent)
        x20, y90 = roc_px(SPEC_FLOOR, SENS_FLOOR, k)
        x0, ytop = roc_px(100, 100, k)
        d.rectangle([x0, ytop, x20, y90], fill=(46, 125, 50, 40))
        _, ybot = roc_px(100, 70, k)
        xr, _ = roc_px(70, SENS_FLOOR, k)
        for a, b in (((x0, y90), (xr, y90)), ((x20, ytop), (x20, ybot))):
            n = 24
            for i in range(0, n, 2):
                p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                q = (a[0] + (b[0] - a[0]) * (i + 1) / n,
                     a[1] + (b[1] - a[1]) * (i + 1) / n)
                d.line([p, q], fill=red, width=12)
    if step >= 3:
        cx, cy = roc_px(*CLINIC_POINT, k)
        r = 34
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=red,
                  outline="white", width=8)
    im.thumbnail((1400, 1400))
    im.save(out / f"roc_real_step{step}.png")


for s in (1, 2, 3):
    roc_real(s)


# Cloud against edge cumulative cost (Napkin Math 3.3), values from calcs.py
CLOUD_K = K.m(K.COST.cloud_annual, "kilodollar/year")
EDGE_RUN_K = K.m(K.COST.edge_annual, "kilodollar/year")
CAPEX_K = K.m(K.COST.edge_capex, "kilodollar")


def cost(step):
    import numpy as np
    t = np.linspace(0, 5, 100)
    cloud = CLOUD_K * t
    edge = CAPEX_K + EDGE_RUN_K * t
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.6))
    ax.plot(t, cloud, color=C["neutral"], lw=3.5)
    ax.text(5.05, cloud[-1], f"Cloud\n\\${val('cost_cloud_k')}k per year", va="center", fontsize=12,
            fontweight="bold", color=C["neutral"])
    if step >= 2:
        ax.plot(t, edge, color=C["memory"], lw=3.5)
        ax.text(5.05, edge[-1] - 20, f"Edge\n\\${val('cost_capex_k')}k + \\${val('cost_edge_k')}k per year", va="center",
                fontsize=12, fontweight="bold", color=C["memory"])
    if step >= 3:
        tc = K.m(K.COST.payback, "year")
        ax.plot([tc], [CLOUD_K * tc], "o", ms=11, color=C["text"], mec="white",
                mew=1.5, zorder=5)
        ax.axvline(tc, color=C["text"], lw=1, ls=":")
        ax.text(tc + 0.08, 30, f"pays back at {val('cost_payback')} years", fontsize=12,
                fontweight="bold", color=C["text"])
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 720)
    ax.set_xlabel("Years after deployment")
    ax.set_ylabel("Cumulative cost (\\$k)")
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"cost_step{step}.svg")


for s in (1, 2, 3):
    cost(s)

# Monitoring cadences (Sec. 3.11.1), in minutes on a log scale
_cm = K.CADENCE.minutes
# Tick labels at 14 pt so the powers of ten stay legible in the column slot.
with sp.plt.rc_context({"xtick.labelsize": 14}):
    sp.hbars(out / "cadences.svg", [
        ("Quarter", _cm["Quarter"], "neutral", "architecture review"),
        ("Month", _cm["Month"], "neutral", "population coverage"),
        ("Week", _cm["Week"], "memory", "labeled accuracy, drift tests"),
        ("Day", _cm["Day"], "memory", "referral rate, confidence"),
        ("Minute", _cm["Minute"], "danger", "camera misconfigured"),
    ], "min", size=(5.4, 2.6),
        xlabel="How long each signal takes to trust (minutes, log scale)")


# -----------------------------------------------------------------------------
# 5. What the machine sees at 224 x 224 (Sec. 3.7.2)
# -----------------------------------------------------------------------------
def shrink_demo():
    from PIL import Image, ImageDraw
    im = Image.open(out / "fundus-dr-hao.png").convert("RGB")
    w, h = im.size
    assert (w, h) == (K.PHOTO.source_width, K.PHOTO.source_height), "photo differs from FundusPhotoAt224"
    n = K.PHOTO.target_size
    left = 30                                    # keep the panel label out
    sq = im.crop((left, 0, left + h, h))          # square, as a 224 x 224 input
    # One exudate cluster, in square coords. Measured on the source with a
    # local-contrast threshold, these exudates are the scenario's feature
    # widths; the n x n input scales every length by n / h (FundusPhotoAt224). Both panels are drawn at the same size
    # (nearest-neighbor for the small image) so its pixel grid shows.
    box = (546, 150, 656, 250)
    full = sq.crop(box).resize((440, 400), Image.LANCZOS)
    small = sq.resize((n, n), Image.LANCZOS)
    f = K.PHOTO.scale
    sbox = tuple(round(v * f) for v in box)
    low = small.crop(sbox).resize((440, 400), Image.NEAREST)
    full.save(out / "retina_full_crop.png")
    low.save(out / "retina_224_crop.png")
    ov = sq.copy()
    d = ImageDraw.Draw(ov)
    d.rectangle(box, outline=(255, 255, 255), width=5)
    ov.resize((500, 500), Image.LANCZOS).save(out / "retina_whole.png")


shrink_demo()


# -----------------------------------------------------------------------------
# 6. Stage contracts on the loop (Table 2, Sec. 3.2; audit case Sec. 3.11).
#    Invariant chips on three hand-offs; step 2 shows a blocked hand-off; step
#    4 traces a monitoring signal to the stage 4 contract that missed it.
# -----------------------------------------------------------------------------
def chip_box(x, y, a, b, col, fill, w=290):
    return [f'<rect x="{x - w / 2}" y="{y - 34}" width="{w}" height="68" rx="10" '
            f'fill="{fill}" stroke="{col}" stroke-width="3"/>',
            text(x, y - 7, a, 22, col, "bold"),
            text(x, y + 22, b, 23, C["text"])]


def contracts(step):
    p = svg_open(900, 560)
    p.append('<g transform="translate(0,30)">')
    for k in range(1, 7):
        p += stage_box(k)
    p += flow_arrows()
    p += return_arrow(label=False)
    p.append('</g>')
    ok = (C["data"], C["data_l"])
    p.append(line(300, 106, 300, 160, C["data"], 3, None))
    p += chip_box(300, 70, "1 → 2", "criteria are numbers", *ok)
    if step >= 2:
        p.append(line(287, 155, 313, 185, C["danger"], 6, None))
        p.append(line(287, 185, 313, 155, C["danger"], 6, None))
        p.append(text(300, 248, "blocked:", 24, C["danger"], "bold"))
        p.append(text(300, 276, "target “TBD”", 24, C["danger"], "bold"))
    if step >= 3:
        p.append(line(600, 106, 600, 160, C["data"], 3, None))
        p += chip_box(600, 70, "2 → 3", "provenance per image", *ok)
        bad = step >= 4
        col, fill = (C["danger"], C["danger_l"]) if bad else ok
        p.append(line(600, 470, 600, 418, col, 3, None))
        p += chip_box(600, 506, "4 → 5", "no subgroup below floor", col, fill,
                      w=320)
    if step >= 4:
        p.append(text(560, 320, "no result for", 24, C["danger"], "bold"))
        p.append(text(560, 348, "patients over 70", 24, C["danger"], "bold"))
    save_svg(p, f"contracts_step{step}.svg")


for s in (1, 2, 3, 4):
    contracts(s)


# -----------------------------------------------------------------------------
# 7. The release gate (Sec. 3.8, Sec. 3.8.1, Table 2, Sec. 3.10). The slice
#    sensitivities are the requirements scenario's subgroup example; the
#    device rows are requirements, not measurements.
# -----------------------------------------------------------------------------
def gate(step):
    p = svg_open(980, 430)
    p.append('<rect x="8" y="8" width="694" height="414" rx="12" fill="#FAFAFA" '
             'stroke="#999999" stroke-width="2"/>')
    p.append(text(28, 50, "Release manifest, one model version", 25, C["text"],
                  "bold", "start"))
    p.append(text(450, 92, "Floor", 21, C["neutral"], "bold", "start"))
    p.append(text(572, 92, "Result", 21, C["neutral"], "bold", "start"))
    p.append('<line x1="22" y1="104" x2="688" y2="104" stroke="#BBBBBB" stroke-width="2"/>')
    rows = [
        (1, "Sensitivity, all clinic photos", f"{val('floor_sens')}%", f"{val('gate_all_pct')}% pass", C["data"]),
        (2, "Sensitivity, cataract patients", f"{val('floor_sens')}%", f"{val('gate_cataract_pct')}% fail",
         C["danger"]),
        (3, "Peak memory on clinic device", f"{val('budget_mb')} MB", "measure", C["neutral"]),
        (3, "Latency on clinic device", "in visit", "measure", C["neutral"]),
        (3, "Confidence matches accuracy", "per use", "measure", C["neutral"]),
    ]
    for i, (k, name, floor, res, col) in enumerate(rows):
        if step < k:
            continue
        y = 146 + i * 54
        p.append(text(28, y, name, 24, C["text"], "normal", "start"))
        p.append(text(450, y, floor, 24, C["text"], "normal", "start"))
        p.append(text(572, y, res, 24, col, "bold", "start"))
    if step >= 3:
        p.append(text(28, 405, "Also recorded: checkpoint hash, preprocessing, slices",
                      20, C["neutral"], "normal", "start"))
    if step >= 2:
        y = 200
        p.append(line(706, y - 8, 736, y - 8, C["danger"], 4, "r"))
        p.append(f'<rect x="742" y="{y - 92}" width="232" height="214" rx="12" '
                 f'fill="{C["danger_l"]}" stroke="{C["danger"]}" stroke-width="3"/>')
        for j, (s_, wgt) in enumerate((("Gate halts.", "bold"),
                                       ("Back to stage 2:", "bold"),
                                       ("collect cataract", "normal"),
                                       ("images, retrain", "normal"),
                                       ("with lens-opacity", "normal"),
                                       ("augmentation", "normal"))):
            p.append(text(858, y - 58 + j * 32, s_, 21, C["danger"] if wgt == "bold"
                          else C["text"], wgt))
    save_svg(p, f"gate_step{step}.svg")


for s in (1, 2, 3):
    gate(s)
print("figures written")
