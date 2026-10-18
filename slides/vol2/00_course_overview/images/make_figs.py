#!/usr/bin/env python3
"""Figures for Volume II, Lecture 0 (course overview).

Every value is sourced. "Derived" marks arithmetic done here from sourced
inputs; the slide note shows the same arithmetic.

  - GPT-3: 3.14e23 FLOPs (Brown et al. 2020); one A100 at its 312 TFLOP/s
    peak needs about 32 years; the run used about 10,000 V100s at about
    24.6 TFLOP/s each for about 15 days (Patterson et al. 2021). Introduction
    Sec. 1.1 and footnote 6.
  - Landmark accelerator counts: MLSysIM Literature.TrainingRuns and the
    model registry, as plotted in the Introduction's cluster-size figure
    (Sec. 1.1, figure 2). GPT-3's point is an estimate (Langston 2020).
  - Growth decomposition, 146 runs 2012 to 2024: compute 3.9x per year =
    count 1.6x x duration 1.3x x delivered rate per accelerator 2x; more
    accelerators running longer supply 51 percent of the exponent
    (Introduction Sec. 1.1, figure 1b; Epoch AI 2026).
  - Scaling price: Kaplan et al. (2020) compute fit, loss proportional to
    compute^-0.050; one doubling multiplies loss by 0.966 (Conclusion
    Sec. 17, citing Kaplan). The 1,024x point (0.71) is derived.
  - Serving cost: $2M training against 18.25 billion requests x $0.001
    = $18.25M first-year serving (Serving at Scale, Napkin Math 10.1; the
    $0.001 per request is a scenario input).
  - Staircase: 10 GB at HBM 3,350 GB/s = 3.0 ms; NVLink 450 GB/s = 22.2 ms;
    InfiniBand 50 GB/s = 200 ms (Compute Infrastructure, Napkin Math 2.3).
  - Failure clock: MTTF 50,000 h per GPU (Introduction Sec. 1.1, an assumed
    value); 50,000 / 16,384 = 3.05 h (derived); Llama 3 405B measured 419
    interruptions in 54 days, 54 x 24 / 419 = 3.1 h (Fault Tolerance,
    War Story 7.2; Dubey et al. 2024).
  - Llama 3 interruption causes: Table 5 of Dubey et al. 2024, as charted
    in Fault Tolerance Sec. 7.3, figure 2.
  - All-node availability a^N: 99.9 percent nodes give 36.8 percent at
    1,000 nodes and 0.0045 percent at 10,000 (Introduction Sec. 1.4.1).
  - Rack power: traditional 12 kW, four-node DGX H100 rack 33.5 kW,
    liquid-cooled class 100 kW (Compute Infrastructure Sec. 2.6.2).
  - Step breakdown: a 2,048-GPU step of 1.5 s = 0.9 s matrix math + 0.4 s
    ring transfers + 0.2 s barrier waits (Introduction Sec. 1.5 self-check
    1); halving compute gives 1.05 s (derived).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

out = Path(__file__).parent
FONT = "Helvetica, Arial, sans-serif"

# Volume II C3 roles (volumes/vol2/palette.md): compute blue, communication
# amber, coordination violet. Used only where a figure encodes C3.
C1, C1_L = "#1F407A", "#EAF1F8"
C2, C2_L = "#d97706", "#fef3c7"
C3, C3_L = "#7c3aed", "#f3e8ff"
INK, GRAY, GRAY_L, RED = "#1A1A2E", "#666666", "#DDDDDD", "#CB202D"


# -----------------------------------------------------------------------------
# 1. GPT-3: one accelerator against the fleet (training time, days, log)
# -----------------------------------------------------------------------------
sp.hbars_steps(out / "gpt3_days", [
    ("One A100", 11648, "neutral", "32 years"),
    ("10,000 V100s", 14.8, "memory", "15 days"),
], "days", xlabel="GPT-3 training time (days, log)", size=(3.8, 2.2))


# -----------------------------------------------------------------------------
# 2. Landmark accelerator counts, 2012 to 2024
# -----------------------------------------------------------------------------
def cluster_counts():
    rows = [("AlexNet", 2012, 2, False, (8, -3, "left")),
            ("Megatron-LM", 2019, 512, False, (-8, 4, "right")),
            ("GPT-3 (estimate)", 2020, 10000, True, (-10, 0, "right")),
            ("PaLM", 2022, 6144, False, (-8, -13, "right")),
            ("LLaMA 1", 2023, 2048, False, (-4, -17, "right")),
            ("DeepSeek-V3", 2024, 2048, False, (4, -17, "left")),
            ("Llama 3", 2024, 16384, False, (-9, 6, "right"))]
    fig, ax = sp.plt.subplots(figsize=(5.7, 2.4))
    ver = [r for r in rows if not r[3]]
    coeffs = np.polyfit([r[1] for r in ver], np.log10([r[2] for r in ver]), 1)
    xs = np.linspace(2011.5, 2024.5, 100)
    ax.plot(xs, 10 ** np.polyval(coeffs, xs), "--", color=sp.C["neutral"], lw=1.5)
    ax.text(2021.0, 40,
            f"fit: about {10 ** coeffs[0]:.0f}× per year", color=sp.C["neutral"],
            fontsize=12, fontweight="bold", rotation=0, ha="center")
    for name, yr, n, est, (dx, dy, ha) in rows:
        ax.plot([yr], [n], "o", ms=10, mfc="white" if est else sp.C["memory"],
                mec=sp.C["memory"], mew=2, zorder=5)
        ax.annotate(name, (yr, n), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=12, color=sp.C["memory"],
                    fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1, 1e5)
    ax.set_xlim(2011.3, 2026.3)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.set_xticks([2012, 2014, 2016, 2018, 2020, 2022, 2024])
    ax.set_ylabel("Accelerators")
    ax.grid(axis="y", color=sp.C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "cluster_counts.svg")


cluster_counts()


# -----------------------------------------------------------------------------
# 3. Where the growth came from: annual factors on a log axis, so each bar's
#    length is its share of the exponent (Introduction figure 1b)
# -----------------------------------------------------------------------------
def growth_factors():
    rows = [("Training compute", 3.9, sp.C["neutral"]),
            ("Accelerator count", 1.6, sp.C["memory"]),
            ("Training duration", 1.3, sp.C["memory"]),
            ("Delivered rate per accelerator", 2.0, sp.C["compute"])]
    for k in range(1, 5):
        fig, ax = sp.plt.subplots(figsize=(5.8, 2.4))
        for i, (label, v, col) in enumerate(rows):
            y = 3 - i
            shown = i < k
            ax.barh(y, np.log10(v), color=col, height=0.6, alpha=1 if shown else 0)
            ax.text(-0.005, y, label, ha="right", va="center", fontsize=13,
                    fontweight="bold", color=sp.C["text"])
            ax.text(np.log10(v) + 0.01, y, f"{v:.1f}× per year", va="center",
                    fontsize=13, fontweight="bold", color=col, alpha=1 if shown else 0)
        # the punchline: count and duration together supply about half
        ax.text(0.47, 1.5,
                "more accelerators, running longer:\n51 percent of the growth",
                va="center", ha="left", fontsize=12, color=sp.C["memory"],
                fontweight="bold", alpha=1 if k >= 4 else 0, linespacing=1.0)
        ax.set_xlim(0, 0.95)
        ax.set_ylim(-0.5, 3.5)
        ticks = [1, 1.5, 2, 3, 4]
        ax.set_xticks(np.log10(ticks))
        ax.set_xticklabels([f"{t:g}×" for t in ticks])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Growth per year (log scale)")
        ax.grid(axis="x", color=sp.C["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, out / f"growth_factors_step{k}.svg")


growth_factors()


# -----------------------------------------------------------------------------
# 4. The scaling price: loss relative to the start, against compute multiple
# -----------------------------------------------------------------------------
def scaling_price():
    m = 2.0 ** np.arange(0, 11)
    for k in (1, 2):
        fig, ax = sp.plt.subplots(figsize=(4.2, 2.6))
        xs = np.logspace(0, np.log10(1024), 200)
        ax.plot(xs, xs ** -0.05, color=sp.C["compute"], lw=3)
        ax.plot(m, m ** -0.05, "o", ms=8, color=sp.C["compute"], mec="white", mew=1.2)
        ax.annotate("each doubling: loss × 0.966", (2, 2 ** -0.05),
                    xytext=(14, 10), textcoords="offset points", fontsize=12,
                    fontweight="bold", color=sp.C["compute"])
        a = 1 if k == 2 else 0
        ax.annotate("1,024× the compute:\nloss × 0.71", (1024, 1024 ** -0.05),
                    xytext=(-70, -6), textcoords="offset points", ha="right", va="top",
                    fontsize=12, fontweight="bold", color=sp.C["danger"], alpha=a,
                    arrowprops=dict(arrowstyle="-|>", color=sp.C["danger"], alpha=a,
                                    lw=1.5))
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 256, 1024])
        ax.set_xticklabels(["1×", "4×", "16×", "64×", "256×", "1,024×"])
        ax.minorticks_off()
        ax.set_ylim(0.55, 1.04)
        ax.set_xlim(0.85, 1300)
        ax.set_xlabel("Training compute (multiple of start, log)")
        ax.set_ylabel("Loss (start = 1)")
        ax.grid(color=sp.C["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, out / f"scaling_price_step{k}.svg")


scaling_price()


# -----------------------------------------------------------------------------
# 5. Serving against training, first year (Napkin Math 10.1)
# -----------------------------------------------------------------------------
sp.hbars_steps(out / "serving_cost", [
    ("Training, once", 2.0, "neutral", "\\$2.0M"),
    ("Serving, first year", 18.25, "compute", "\\$18.25M"),
], "M", log=False, xlabel="Cost (millions of US dollars)", size=(4.3, 1.8))


# -----------------------------------------------------------------------------
# 6. The synchronous step: on time, then with one slow worker (SVG)
# -----------------------------------------------------------------------------
def svg_text(x, y, s, size=24, fill=INK, weight=600, anchor="start", extra=""):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
            f'dominant-baseline="central"{extra}>{s}</text>\n')


def rect(x, y, w, h, fill, stroke="none", sw=0, dash="", rx=4):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>\n')


def sync_step():
    x0, cw, ar, lane, gap = 175, 330, 150, 26, 6
    slow = int(cw * 1.3)

    def block(y0, title, slow_worker):
        s = svg_text(20, y0 - 22, title, size=26, weight=700)
        end = x0 + (slow if slow_worker else cw)
        for i in range(4):
            y = y0 + i * (lane + gap)
            s += svg_text(20, y + lane / 2, f"Worker {i + 1}", size=24, weight=600, fill=GRAY)
            w = slow if (slow_worker and i == 2) else cw
            if slow_worker and i == 2:
                s += rect(x0, y, w, lane, C1_L, RED, 3, "8 5")
                s += svg_text(x0 + w / 2, y + lane / 2, "slow", size=24, fill=RED, weight=700, anchor="middle")
            else:
                s += rect(x0, y, w, lane, C1)
                if slow_worker:
                    s += rect(x0 + w, y, end - x0 - w, lane, GRAY_L)
                    s += svg_text(x0 + w + (end - x0 - w) / 2, y + lane / 2, "wait",
                                  size=22, fill=GRAY, weight=700, anchor="middle")
        h = 4 * lane + 3 * gap
        s += rect(end, y0, ar, h, C2_L, C2, 2)
        s += svg_text(end + ar / 2, y0 + h / 2, "AllReduce", size=25, fill="#b45309",
                      weight=700, anchor="middle")
        s += svg_text(x0 + cw / 2, y0 + h + 22, "compute", size=22, fill=C1,
                      weight=700, anchor="middle")
        return s, end + ar

    top, top_end = block(45, "All workers on time", False)
    bot, bot_end = block(258, "One worker 30% slower (illustrative)", True)
    # the step length of the on-time case, carried down as a dashed line
    mark = (f'<line x1="{top_end}" y1="40" x2="{top_end}" y2="470" stroke="{GRAY}" '
            f'stroke-width="2" stroke-dasharray="6 5"/>\n')
    brace = svg_text(bot_end + 12, 258 + 61, "+30%", size=26, fill=RED, weight=700)
    head = ('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="410" '
            'viewBox="0 0 900 410">\n<rect width="900" height="410" fill="white"/>\n')
    (out / "sync_step_step1.svg").write_text(head + top + "</svg>\n")
    (out / "sync_step_step2.svg").write_text(head + top + bot + brace + "</svg>\n")


sync_step()


# -----------------------------------------------------------------------------
# 7. The bandwidth staircase: one 10 GB buffer, one hop per tier
# -----------------------------------------------------------------------------
sp.hbars_steps(out / "staircase", [
    ("HBM", 3.0, "memory", "3.0 ms"),
    ("NVLink, in node", 22.2, "network", "22.2 ms"),
    ("InfiniBand, across", 200, "danger", "200 ms: 9× NVLink"),
], "ms", xlabel="Time to move 10 GB (ms, log scale)", value_fmt="{:g}", size=(4.5, 2.3))


# -----------------------------------------------------------------------------
# 8. The failure clock (hours, log)
# -----------------------------------------------------------------------------
sp.hbars_steps(out / "failure_clock", [
    ("One GPU, assumed", 50000, "neutral", "50,000 h"),
    ("16,384 GPUs, model", 3.05, "danger", "3.05 h"),
    ("Llama 3, measured", 3.1, "danger", "3.1 h"),
], "h", xlabel="Hours between interruptions (log)", size=(3.9, 2.3))


# -----------------------------------------------------------------------------
# 9. What stopped Llama 3: interruption causes (Dubey et al. 2024, table 5)
# -----------------------------------------------------------------------------
sp.hbars(out / "llama3_causes.svg", [
    ("Faulty GPU", 148, "danger"),
    ("GPU HBM3 memory", 72, "danger"),
    ("Software bug", 54, "neutral"),
    ("Network switch or cable", 35, "network"),
    ("Host maintenance", 32, "neutral"),
    ("GPU SRAM memory", 19, "danger"),
    ("GPU system processor", 17, "danger"),
    ("11 other causes", 42, "neutral"),
], "", log=False, xlabel="Interruptions in 54 days (419 in all)", size=(5.0, 2.8))


# -----------------------------------------------------------------------------
# 10. All-node availability, a^N (Introduction Sec. 1.4.1)
# -----------------------------------------------------------------------------
def availability():
    n = np.logspace(0, 4, 300)
    for k in (1, 2):
        fig, ax = sp.plt.subplots(figsize=(4.4, 2.5))
        ax.plot(n, 100 * 0.999 ** n, color=sp.C["memory"], lw=3)
        ax.text(2.0, 62, "99.9% per node", color=sp.C["memory"], fontsize=12,
                fontweight="bold")
        ax.plot([1000], [36.8], "o", ms=9, color=sp.C["memory"], mec="white", zorder=5)
        ax.annotate("36.8% at 1,000 nodes", (1000, 36.8), xytext=(-12, -4),
                    textcoords="offset points", ha="right", fontsize=12,
                    fontweight="bold", color=sp.C["memory"])
        a = 1 if k == 2 else 0
        ax.plot(n, 100 * 0.9999 ** n, color=sp.C["data"], lw=3, alpha=a)
        ax.plot([10000], [36.8], "o", ms=9, color=sp.C["data"], mec="white", zorder=5, alpha=a)
        ax.text(1.2, 103, "99.99% per node: 36.8% at 10,000", color=sp.C["data"],
                fontsize=12, fontweight="bold", alpha=a, va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(1, 14000)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks([1, 10, 100, 1000, 10000])
        ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
        ax.minorticks_off()
        ax.set_xlabel("Nodes needed at once (log scale)")
        ax.set_ylabel("All nodes up (%)")
        ax.grid(color=sp.C["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, out / f"availability_step{k}.svg")


availability()


# -----------------------------------------------------------------------------
# 11. Rack power (Compute Infrastructure Sec. 2.6.2)
# -----------------------------------------------------------------------------
sp.hbars_steps(out / "rack_power", [
    ("Traditional", 12, "neutral", "12 kW"),
    ("DGX H100, 4 nodes", 33.5, "compute", "33.5 kW"),
    ("Liquid-cooled", 100, "danger", "100 kW"),
], "kW", log=False, xlabel="Power per rack (kW)", value_fmt="{:g}", size=(3.9, 2.2))


# -----------------------------------------------------------------------------
# 12. The C3 axes, one circle per step (column figure, 600 x 430)
# -----------------------------------------------------------------------------
def c3():
    rows = [(C1, C1_L, "Compute (C₁)", "the math: kernels, HBM, % of peak"),
            (C2, C2_L, "Communication (C₂)", "the wire: links, collectives"),
            (C3, C3_L, "Coordination (C₃)", "the logic: barriers, recovery")]
    for k in range(1, 4):
        s = ('<svg xmlns="http://www.w3.org/2000/svg" width="600" height="430" '
             'viewBox="0 0 600 430">\n<rect width="600" height="430" fill="white"/>\n')
        for i, (stroke, fill, name, ev) in enumerate(rows):
            op = 1 if i < k else 0
            y = 15 + i * 140
            s += (f'<rect x="10" y="{y}" width="580" height="120" rx="10" fill="{fill}" '
                  f'stroke="{stroke}" stroke-width="3" fill-opacity="{op}" stroke-opacity="{op}"/>\n')
            s += svg_text(36, y + 40, name, size=30, fill=stroke, weight=700,
                          extra=f' fill-opacity="{op}"')
            s += svg_text(36, y + 85, ev, size=26, fill=INK, weight=500,
                          extra=f' fill-opacity="{op}"')
        s += "</svg>\n"
        (out / f"c3_step{k}.svg").write_text(s)


c3()


# -----------------------------------------------------------------------------
# 13. One 1.5 s step, then the same step on accelerators twice as fast
# -----------------------------------------------------------------------------
def step_bars():
    rows = [("2,048 GPUs,\nmeasured", [0.9, 0.4, 0.2]),
            ("Accelerators\n2× faster", [0.45, 0.4, 0.2])]
    names = ["Compute", "Communication", "Coordination"]
    cols, lights = [C1, C2, C3], [C1_L, C2_L, C3_L]
    for k in (1, 2, 3):
        fig, ax = sp.plt.subplots(figsize=(4.6, 2.4))
        for r, (label, parts) in enumerate(rows):
            y = 1 - r
            shown = r == 0 or k == 3
            left = 0
            for j, w in enumerate(parts):
                colored = k >= 2
                ax.barh(y, w, left=left, height=0.55,
                        color=cols[j] if colored else sp.C["neutral_l"],
                        edgecolor="white", lw=2, alpha=1 if shown else 0)
                ax.text(left + w / 2, y, f"{w:g} s", ha="center", va="center",
                        fontsize=12 if w > 0.25 else 10.5, fontweight="bold",
                        color="white" if colored else sp.C["text"], alpha=1 if shown else 0)
                left += w
            ax.text(left + 0.02, y, f"{left:.2f} s".replace("1.50", "1.5"), va="center",
                    fontsize=13, fontweight="bold", color=sp.C["text"],
                    alpha=1 if shown else 0)
            ax.text(-0.02, y, label, ha="right", va="center", fontsize=12,
                    fontweight="bold", color=sp.C["text"], alpha=1 if shown else 0,
                    linespacing=1.0)
        # C3 names over the first bar, once the colors are on
        left = 0
        for j, w in enumerate(rows[0][1]):
            ax.text(left + w / 2, 1.38 if j != 1 else 1.68, names[j], ha="center",
                    va="bottom", fontsize=11,
                    fontweight="bold", color=cols[j], alpha=1 if k >= 2 else 0)
            left += w
        a = 1 if k == 3 else 0
        ax.axvline(0.75, ymin=0.122, ymax=0.175, color=sp.C["danger"], ls="--", lw=1.8, alpha=a)
        ax.text(0.75, -0.45, "half of 1.5 s", ha="center", va="top", fontsize=11,
                fontweight="bold", color=sp.C["danger"], alpha=a)
        ax.set_xlim(0, 1.85)
        ax.set_ylim(-0.75, 1.95)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Time per step (s)")
        sp.save(fig, out / f"step_bars_step{k}.svg")


step_bars()


# -----------------------------------------------------------------------------
# 14. The stack as the map of the course: the kit's final build step (same
#     canvas, so nothing moves) with each lecture on its row, following the
#     kit's roadmap. Lectures 0, 1, and 17 span the stack; the slide says so.
# -----------------------------------------------------------------------------
kit = Path(__file__).resolve().parents[3] / "assets" / "kit" / "vol2"
stack_svg = (kit / "stack_build8.svg").read_text()
ACCENT = "#1F407A"
ROW_Y = {"Facility": 418.25, "Fabric": 362.5, "Execution": 306.75, "Control": 251.0,
         "Serving": 195.25, "Operations": 139.5, "Assurance": 83.75, "Governance": 28.0}
ON_ROW = {"Facility": ["L2"], "Fabric": ["L3", "L4"], "Execution": ["L5", "L6"],
          "Control": ["L7", "L8"], "Serving": ["L9", "L10", "L11"],
          "Operations": ["L12"], "Assurance": ["L13", "L14"],
          "Governance": ["L15", "L16"]}


def badge(x, y, w, h, label, size=22):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" '
            f'fill="#FFFFFF" stroke="{ACCENT}" stroke-width="1.5"/>\n'
            f'<text x="{x + w / 2}" y="{y + h / 2}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="700" fill="{ACCENT}" '
            f'text-anchor="middle" dominant-baseline="central">{label}</text>\n')


badges = ""
for row, labels in ON_ROW.items():
    right = 710
    for label in reversed(labels):
        w = 58 if len(label) <= 2 else 70
        badges += badge(right - w, ROW_Y[row] + 6, w, 36, label)
        right -= w + 6
(out / "stack_lectures.svg").write_text(stack_svg.replace("</svg>", badges + "</svg>"))

print("figures written")
