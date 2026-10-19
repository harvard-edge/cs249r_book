#!/usr/bin/env python3
"""Data plots for Lecture 0 (course overview).

Every value is sourced, and every derived value comes from calcs.py:
  - Google's measured median Gemini text prompt, 0.24 Wh = 864 J
    (Elsworth et al. 2025); charging a phone, about 40,000 J (Model
    Serving, Napkin Math 13.5).
  - Daily chat prompts: about 2.5 billion (OpenAI to Axios, July 2025);
    GPT-3 training, 1,287 MWh (Patterson et al. 2021).
  - Datacenter electricity: 415 TWh in 2024, about 945 TWh projected for 2030
    (IEA, Energy and AI, 2025, executive summary).
  - Deployment power: cloud 3 MW, edge 200 W, mobile 5 W, TinyML 1 mW
    (System Paradigms, Sec. 2.1).
  - The stack with each lecture on its layer (kit stack_build8 canvas).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

out = Path(__file__).parent

# Linear axes print thousands with a comma separator ("10,000", not "10000").
# slideplot does not take a tick formatter, so the deck wraps its save step.
_save = sp.save


def _save_with_commas(fig, path):
    for ax in fig.axes:
        if ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _save(fig, path)


sp.save = _save_with_commas

# One measured prompt against charging a phone, on a linear axis so the
# ratio reads true (two steps). Values from calcs.py.
import calcs  # noqa: E402

sp.hbars_steps(out / "energy_answer", [
    ("One prompt (Google, measured)", round(calcs.google_j), "compute",
     f"{calcs.google_j:,.0f} J"),
    ("Charging a phone", calcs.PHONE_J, "neutral",
     f"{calcs.PHONE_J:,} J = {calcs.PHONE_J / calcs.google_j:.0f} prompts"),
], "J", log=False, xlabel="Energy (joules)")

# Fleet scale: about 2.5 billion ChatGPT prompts a day (OpenAI to Axios,
# July 2025) at Google's measured 0.24 Wh each, against the energy to train
# GPT-3 once (Patterson et al. 2021). Estimate: the per-prompt figure is
# Gemini's median, applied to another service.
sp.hbars_steps(out / "fleet_daily", [
    ("Chat prompts, one day (estimate)", round(calcs.fleet_mwh), "compute",
     f"about {calcs.fleet_mwh:,.0f} MWh"),
    ("Training GPT-3, once", calcs.GPT3_MWH, "neutral", f"{calcs.GPT3_MWH:,} MWh"),
], "MWh", log=False, xlabel="Energy (megawatt-hours)", size=sp.SHORT)

# Datacenter electricity (IEA, Energy and AI, 2025)
sp.hbars_steps(out / "datacenter_electricity", [
    ("2024", calcs.DC_TWH_2024, "neutral"),
    ("2030 (projected)", calcs.DC_TWH_2030, "danger"),
], "TWh", log=False, xlabel="Datacenter electricity per year (TWh)",
    size=sp.SHORT)

# One field, four machines: power envelope by deployment tier
# (System Paradigms, Sec. 2.1). Tick labels at 14 pt for the column slot.
with sp.plt.rc_context({"xtick.labelsize": 14}):
    sp.hbars_steps(out / "power_tiers", [
        ("Cloud", 3e6, "neutral", "3 MW"),
        ("Edge", 200, "neutral", "200 W"),
        ("Mobile", 5, "neutral", "5 W"),
        ("TinyML", 1e-3, "neutral", "1 mW"),
    ], "W", xlabel="Power (watts, log scale)", value_fmt="{:g}")

# The stack as the map of the course: the kit's final stack (same canvas, so
# nothing moves after the eighth build step) with each lecture placed on its
# layer, following the kit's course_order figure. Lectures 1 to 3 and 16 span
# the whole stack and are named on the slide instead.
kit_vol1 = Path(__file__).resolve().parents[3] / "assets" / "kit" / "vol1"
stack_svg = (kit_vol1 / "stack_build8.svg").read_text()
ACCENT = "#A51C30"
LAYER_Y = {"Hardware": 412, "Frameworks": 348, "Models": 284, "Training": 220,
           "Serving": 156, "Operations": 92, "Applications": 28}
ON_LAYER = {"Hardware": ["L11", "L11b", "L12", "L12b"], "Frameworks": ["L7"],
            "Models": ["L5", "L6", "L6b", "L10", "L10b"],
            "Training": ["L8", "L8b"], "Serving": ["L13", "L13b"],
            "Operations": ["L14", "L14b"], "Applications": ["L15", "L15b"]}


def badge(x, y, w, h, label, size=24):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" '
            f'fill="#FFFFFF" stroke="{ACCENT}" stroke-width="1.5"/>\n'
            f'<text x="{x + w / 2}" y="{y + h / 2}" font-family="Helvetica, Arial, '
            f'sans-serif" font-size="{size}" font-weight="700" fill="{ACCENT}" '
            f'text-anchor="middle" dominant-baseline="central">{label}</text>\n')


badges = ""
for layer, labels in ON_LAYER.items():
    y0 = LAYER_Y[layer]
    right = 688
    for label in reversed(labels):
        w = 66 if len(label) <= 3 else 80
        badges += badge(right - w, y0 + 7, w, 40, label)
        right -= w + 6
# Data lectures sit inside the data bar, below its label
for label, y in (("L4", 300), ("L4b", 352), ("L9", 404)):
    badges += badge(715, y, 54, 40, label, 24 if len(label) <= 2 else 20)
(out / "stack_lectures.svg").write_text(stack_svg.replace("</svg>", badges + "</svg>"))

print("figures written")
