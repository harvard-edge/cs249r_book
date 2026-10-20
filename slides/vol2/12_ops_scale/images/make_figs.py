#!/usr/bin/env python3
"""Figures for Volume II, Lecture 12 (ML Operations at Scale).

Every value is sourced. "OPS" is the rendered chapter, ML Operations at
Scale. Values computed here are labeled "derived" on the figure or its
slide; constructed scenarios from the chapter are labeled "illustrative".

  - Hook timeline (OPS Example 12.2, illustrative scenario): 25,000 QPS
    ranking model, latency gate < 12 ms, an upstream schema change makes a
    user-affinity feature null for 8 percent of queries, imputed as 0.0;
    HTTP errors and latency normal; conversion down 3.2 percent for 48 h.
  - Operational load sketch (OPS Sec. 12.2, figure 1, same code shapes):
    alerts 20 N, deployment N ln N, conflicts N^2/2, team capacity 2,500;
    the total crosses capacity near 50 models. Shapes, not measurements.
  - FBLearner Flow (Meta Engineering blog, 9 May 2016): more than 1 million
    models trained; more than 500,000 workflow runs in one month (April);
    fewer than 150 workflow authors; used by more than 25 percent of the
    engineering team; more than 6 million predictions per second.
  - Registry graph (OPS figure 3): embedding E -> retrieval R -> rankers
    K1, K2, K3 -> ensemble M; fraud model F has no recorded edge.
  - Feature store (OPS Sec. 12.4.1, 12.4.6): offline store scans terabytes;
    online lookup under 10 ms; 1 billion recommendations x 100 features a
    day = 1.16 million lookups/s before peak; 5 to 10 ms of a 50 ms budget.
  - Point-in-time join (OPS figure 6): total_clicks_today is 10 at
    midnight, 4 clicks logged before the noon prediction; 2.5x inflation.
  - Data validation (Breck et al., MLSys 2019, Table 1): over 700 Google
    pipelines, 30 days, share of pipelines on which an anomaly fired:
    new feature column 10%, out-of-domain categorical values 6%, missing
    feature column 6%, feature present in too few examples 3%, data
    completely missing 3%, feature vector too small 2%.
  - Broken-feature cost (OPS Sec. 12.4.7, constructed): 8 percent of
    $15M/week for 5 days = $857K; a 4-hour window = $28.6K (derived), 30x.
  - Rollout exposure (OPS Napkin Math 12.5): 100 percent against 5 percent,
    20x under the same rollback time.
  - Canary schedule (OPS Sec. 12.5.5): 10,000 samples per variant at
    1 million requests/h: 1 percent stage 1 h minimum, 5 percent 12 min;
    configured 1% 2 h, 5% 30 min, 25% 30 min, 50% 1 h, then 100%: about 4 h.
  - Sample size (OPS equation 4, Sec. 12.5.3): n = (1.96 + 0.84)^2 2p(1-p)
    / delta^2; 5% baseline, 0.5 pt: 29,792; 1% baseline, 0.05 pt: 620,800.
  - Model loading (Baylor et al., KDD 2017, Sec. 6.1, TFX): 99.9th
    percentile inference latency during model loads 500 to 1,500 ms;
    75 to 150 ms with a dedicated model-loading threadpool.
  - False alerts (OPS Napkin Math 12.7, equation 9): 1 - (1 - p)^M for
    p = 0.003 (3-sigma) and p = 0.05 (2-sigma).
  - Monitoring hierarchy (OPS table 17): business / incident lead,
    portfolio / domain team, model / model owner, infrastructure / platform.
  - SLO vector (OPS figure 12, illustrative): availability 0.40, latency
    0.60, freshness 1.30, model quality 0.50; mean 0.70.
  - Meta outage, 4 October 2021 (Cloudflare blog, "Understanding how
    Facebook disappeared from the Internet"; times UTC): routing-change
    peak about 15:40; facebook.com unavailable on 1.1.1.1 from about 15:50
    to 21:20; renewed BGP activity peaking 21:17; reconnected by 21:28.
  - Philly (Jeon et al., USENIX ATC 2019, Tables 3 and 6; Microsoft,
    two-month trace, 96,260 jobs): jobs passed 69.3%, killed 13.5%,
    unsuccessful 17.2%; GPU time 44.53%, 37.69%, 17.76%; mean utilization
    of in-use GPUs 52.32%.
  - Own against rent (OPS figure 13): 1,024 H100s, $59.80M upfront plus
    $1.5M/year owned; rent $4/GPU-hour at 80 percent = $2.39M/month; the
    lines cross at 26.4 months. The 50 percent rental line is derived
    (1,024 x $4 x 730 h x 0.5 = $1.50M/month; crosses at 43.6 months).
  - TCO (OPS table 31, constructed budgets): startup train 5.0K, infer
    2.0K, data 0.5K, iterate 40K per month; production company 150K,
    400K, 80K, 350K per month.

Every plotted value and label number is read from MLSysIM through calcs.py
(val, num, and its scenario handles); the list above records the sources.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402  (records every value; figures read them through val/num)
from deckvals import val, num  # noqa: E402
from mlsysim.physics import calc_two_proportion_sample_size  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent

# Volume II C3 roles for schematics (palette.md): C1 blue, C2 amber, C3 violet.
V = {"c1": "#1F407A", "c1_l": "#EAF1F8", "c2": "#d97706", "c2_l": "#fef3c7",
     "c3": "#7c3aed", "c3_l": "#f3e8ff", "ok": "#16a34a", "ok_l": "#dcfce7",
     "bad": "#dc2626", "bad_l": "#fee2e2", "frame": "#94a3b8", "frame_l": "#f8fafc"}


def tag(ax, text, x=0.0, y=-0.30, fs=9.5):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs,
            color=C["neutral"], ha="left", va="top")


def hide_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def box(ax, x, y, w, h, text, fc, ec, fs=12, tc=None, alpha=1.0, bold=True, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=fc, edgecolor=ec, lw=lw, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or ec, fontweight="bold" if bold else "normal", alpha=alpha)


def arrow(ax, p, q, color, alpha=1.0, lw=1.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, alpha=alpha))


# ------------------------------------------------------------------ hook
T_CHANGE = 6  # layout: hour on the 60 h axis where the upstream change lands


def hook(k, size=(5.9, 2.8), fs=12.5, name=None):
    """Rows of signals over 60 h. k=1: only service health shown; k=2:
    feature and product rows revealed; k=3 (hook_synth): controls marked."""
    fig, ax = plt.subplots(figsize=size)
    t = np.linspace(0, 60, 400)
    t_end = T_CHANGE + num("hook_hours")
    rows = [("HTTP errors", "neutral", lambda t: np.zeros_like(t), "normal"),
            ("p99 latency", "neutral", lambda t: np.zeros_like(t), f"under {val('hook_gate_ms')} ms"),
            ("feature nulls", "danger", lambda t: np.where((t >= T_CHANGE) & (t < t_end), 1.0, 0.0),
             f"{val('hook_null_pct')}% of queries"),
            ("conversion", "danger", lambda t: np.where((t >= T_CHANGE) & (t < t_end), -1.0, 0.0),
             f"down {val('hook_conv_pct')}%")]
    for i, (lab, ck, f, note) in enumerate(rows):
        y = 3 - i
        a = 1.0 if (i < 2 or k >= 2) else 0.0
        ax.plot(t, y + 0.28 * f(t), color=C[ck], lw=2.6, alpha=a)
        ax.text(-1.5, y, lab, ha="right", va="center", fontsize=fs, fontweight="bold")
        ax.text(61, y, note, ha="left", va="center", fontsize=fs - 1, color=C[ck],
                fontweight="bold", alpha=a)
        if i >= 2 and k == 1:
            ax.text(30, y, "???", ha="center", va="center", fontsize=15,
                    fontweight="bold", color=C["danger"])
    ax.axvline(T_CHANGE, color=C["neutral"], lw=1.2, ls=":")
    ax.text(6.5, 3.55, "schema change", fontsize=13, color=C["neutral"])
    a3 = 1.0 if k >= 3 else 0.0
    if k >= 2:
        ax.annotate("", xy=(t_end, 0.55), xytext=(T_CHANGE, 0.55),
                    arrowprops=dict(arrowstyle="<->", color=C["danger"], lw=1.5))
    ax.text((T_CHANGE + t_end) / 2, 0.62, f"{val('hook_hours')} h unalerted", ha="center", va="bottom", fontsize=fs - 1.5,
            color=C["danger"], fontweight="bold", alpha=1.0 if k >= 2 else 0.0)
    for x, lab in ((7.5, "null-rate gate"), (10.0, "freshness and quality SLO")):
        ax.plot([x], [1.0 if lab.startswith("null") else 0.0], "v", ms=11,
                color=V["ok"], alpha=a3, zorder=6)
    ax.text(8.5, 1.65, "a data contract stops it here", fontsize=fs - 0.5, color=V["ok"],
            fontweight="bold", alpha=a3)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.6, 3.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.set_xlabel("Hours after the upstream change (illustrative timeline)", fontsize=13.5)
    sp.save(fig, out / (name or f"hook_step{k}.svg"))


for k in (1, 2):
    hook(k, fs=14)
# Synthesis: the same timeline drawn at column size so labels stay legible.
hook(3, size=(4.4, 3.0), fs=14, name="hook_synth.svg")

# ------------------------------------------------------- operational load
LOAD = calcs.LOAD
n = LOAD.model_counts()
alerts = LOAD.alerts(n)
deploy = LOAD.deployments(n)
conflicts = LOAD.conflicts(n)
total = LOAD.total(n)
cap = LOAD.team_capacity
cross = LOAD.crossing_models
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.9, 2.8))
    ax.axhline(cap, color=C["danger"], lw=1.6, ls="--")
    ax.text(1.15, cap * 1.25, "one team's capacity", color=C["danger"], fontsize=11,
            fontweight="bold")
    ax.plot(n, alerts, color=C["memory"], lw=2.4)
    ax.text(520, alerts[-1], "alerts: N", color=C["memory"], fontsize=11.5,
            fontweight="bold", ha="left", va="center", clip_on=False)
    a2 = 1.0 if k >= 2 else 0.0
    ax.plot(n[1:], deploy[1:], color=C["compute"], lw=2.4, ls="--", alpha=a2)
    ax.text(520, deploy[-1], "deploys: N log N", color=C["compute"],
            fontsize=11.5, fontweight="bold", ha="left", va="center", alpha=a2, clip_on=False)
    a3 = 1.0 if k >= 3 else 0.0
    ax.plot(n, conflicts, color=C["network"], lw=2.4, ls="-.", alpha=a3)
    ax.plot(n, total, color=C["text"], lw=3.0, alpha=a3)
    ax.text(520, conflicts[-1], "conflicts: N²/2\nand the total", color=C["network"],
            fontsize=11.5, fontweight="bold", ha="left", va="center", alpha=a3, clip_on=False)
    ax.plot([cross], [cap], "o", ms=9, color=C["danger"], alpha=a3, zorder=6)
    ax.text(40, cap * 4.0, f"total crosses\nnear {val('load_cross_round')} models",
            color=C["danger"], fontsize=11, fontweight="bold", ha="right", alpha=a3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 500)
    ax.set_ylim(10, 4e5)
    ax.set_xticks([1, 10, 50, 100, 500])
    ax.set_xticklabels(["1", "10", "50", "100", "500"])
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xlabel("Models in production")
    ax.set_ylabel("Operational load")
    tag(ax, "Assumed growth shapes (chapter figure 1), not measurements", y=-0.26, fs=11)
    sp.save(fig, out / f"load_step{k}.svg")

# ------------------------------------------------------- FBLearner Flow
# Axis type set larger than the kit default: the plot is shown at about half size.
LARGE_AXIS = {"xtick.labelsize": 14, "axes.labelsize": 14}
with plt.rc_context(LARGE_AXIS):
    sp.hbars(out / "fbl.svg", [
        ("Models trained", num("fbl_models"), "memory", f"> {val('fbl_models')}"),
        ("Workflow runs, one month", num("fbl_runs"), "memory", f"> {val('fbl_runs')}"),
        ("Workflow authors", num("fbl_authors"), "danger", f"< {val('fbl_authors')}"),
    ], "", log=True, size=(5.8, 2.0),
        xlabel="Count, log scale (Meta FBLearner Flow, 2016)")

# ------------------------------------------------------- registry graph
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.9, 2.8))
    ax.set_xlim(0, 9.9)
    ax.set_ylim(0, 4.6)
    hide_axis(ax)
    a2 = 1.0 if k >= 2 else 0.0
    a3 = 1.0 if k >= 3 else 0.0
    ec = V["c3"] if k >= 2 else V["frame"]
    ax.add_patch(FancyBboxPatch((2.55, 0.3), 7.3, 3.85, boxstyle="round,pad=0,rounding_size=0.1",
                                facecolor=V["c3_l"], edgecolor=V["c3"], lw=1.6, ls="--", alpha=a2))
    ax.text(9.75, 0.42, "impact set: re-test before release", ha="right", va="bottom",
            fontsize=11.5, color=V["c3"], fontweight="bold", alpha=a2)
    box(ax, 0.05, 2.0, 2.05, 1.05, "Embedding\nE v3.2", V["c3_l"], V["c3"], fs=11)
    box(ax, 2.65, 2.15, 1.75, 0.75, "Retrieval", V["c1_l"], V["c1"], fs=11)
    for j, yy in enumerate((3.15, 2.15, 1.15)):
        box(ax, 4.95, yy, 1.75, 0.75, f"Ranker {j + 1}", V["c1_l"], V["c1"], fs=11)
        arrow(ax, (4.4, 2.52), (4.95, yy + 0.37), ec)
        arrow(ax, (6.7, yy + 0.37), (7.3, 2.52), ec)
    box(ax, 7.3, 2.15, 2.3, 0.75, "Ensemble", V["c1_l"], V["c1"], fs=11)
    arrow(ax, (2.1, 2.52), (2.65, 2.52), ec)
    for x, h in ((1.07, "hop 0"), (3.52, "hop 1"), (5.82, "hop 2"), (8.45, "hop 3")):
        ax.text(x, 4.3, h, ha="center", fontsize=11, color=C["neutral"])
    box(ax, 0.05, 0.3, 2.05, 0.8, "Fraud model", V["frame_l"], V["frame"], fs=10.5, tc=C["neutral"])
    ax.plot([1.07, 1.07], [1.15, 1.95], color=V["bad"], lw=1.6, ls=":", alpha=a3)
    ax.text(1.12, 1.55, "no edge?", fontsize=10.5, color=V["bad"],
            fontweight="bold", va="center", alpha=a3)
    sp.save(fig, out / f"registry_step{k}.svg")

# ------------------------------------------------------- feature store
fig, ax = plt.subplots(figsize=(5.9, 2.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
hide_axis(ax)
box(ax, 0.0, 3.3, 1.7, 0.8, "batch", V["ok_l"], V["ok"], fs=11)
box(ax, 0.0, 1.2, 1.7, 0.8, "streams", V["ok_l"], V["ok"], fs=11)
box(ax, 2.3, 1.55, 2.3, 2.2, "versioned\ndefinition", V["c3_l"], V["c3"], fs=11)
arrow(ax, (1.7, 3.7), (2.3, 3.2), V["ok"])
arrow(ax, (1.7, 1.6), (2.3, 2.1), V["ok"])
box(ax, 5.2, 3.2, 2.0, 1.1, "offline\nstore", V["c1_l"], V["c1"], fs=11)
box(ax, 5.2, 0.9, 2.0, 1.1, "online\nstore", V["c1_l"], V["c1"], fs=11)
arrow(ax, (4.6, 3.1), (5.2, 3.6), V["c3"])
arrow(ax, (4.6, 2.2), (5.2, 1.6), V["c3"])
box(ax, 7.9, 3.2, 2.0, 1.1, "training", V["frame_l"], V["frame"], fs=11, tc=C["text"])
box(ax, 7.9, 0.9, 2.0, 1.1, "serving", V["frame_l"], V["frame"], fs=11, tc=C["text"])
arrow(ax, (7.2, 3.75), (7.9, 3.75), V["c1"])
arrow(ax, (7.2, 1.45), (7.9, 1.45), V["c1"])
ax.text(6.55, 4.5, "full history: scans terabytes", ha="center", fontsize=11.5,
        color=V["c1"], fontweight="bold")
ax.text(6.55, 0.35, f"latest value: {val('fs_lookups_m')} M lookups/s", ha="center", fontsize=11.5,
        color=V["c1"], fontweight="bold")
sp.save(fig, out / "fstore.svg")

# ------------------------------------------------------- point in time
# Layout: where the clicks fall on the day axis; their counts are the scenario's.
click_hours = [1.5, 4.0, 7.5, 10.0, 13.0, 14.5, 16.0, 18.5, 20.0, 22.5]
assert len(click_hours) == calcs.PIT.events_by_day_end
assert sum(h < 12 for h in click_hours) == calcs.PIT.events_before_prediction
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.9, 2.5))
    ax.plot([0, 24], [1, 1], color=C["neutral"], lw=1.5)
    for h in click_hours:
        early = h < 12
        ax.plot([h], [1], "o", ms=12, color=V["c1"] if early else C["neutral_l"],
                mec=V["c1"] if early else C["neutral"], mew=1.5, zorder=5)
    ax.axvline(12, color=C["danger"], lw=2.2)
    ax.text(12, 1.95, "prediction at noon", ha="center", fontsize=12, color=C["danger"],
            fontweight="bold", zorder=10, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    a = 1.0 if k >= 2 else 0.0
    ax.annotate("", xy=(0, 0.45), xytext=(12, 0.45),
                arrowprops=dict(arrowstyle="<->", color=V["c1"], lw=2, alpha=a))
    ax.text(6, 0.18, f"point-in-time join: {val('pit_before')} clicks", ha="center", fontsize=12,
            color=V["c1"], fontweight="bold", alpha=a,
            bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=a), zorder=10)
    ax.annotate("", xy=(0, -0.35), xytext=(24, -0.35),
                arrowprops=dict(arrowstyle="<->", color=C["danger"], lw=2, alpha=a))
    ax.text(12, -0.62, f"naive join to the midnight table: {val('pit_day')} clicks ({val('pit_ratio')}×)", ha="center",
            fontsize=12, color=C["danger"], fontweight="bold", alpha=a,
            bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=a), zorder=10)
    if k == 1:
        ax.text(18, 0.45, "total_clicks_today = ???", ha="center", fontsize=12.5,
                fontweight="bold", color=C["danger"])
    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(-0.9, 2.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])
    sp.save(fig, out / f"pit_step{k}.svg")

# ------------------------------------------------------- data validation
# Drawn at the size of its slide column (about 3.7 x 2.1 in), with the long
# check names on two lines so the type stays at 14 to 15 pt.
breck_rows = [("New feature\ncolumn", "breck_new_pct", "data"),
              ("Missing feature\ncolumn", "breck_missing_pct", "danger"),
              ("Unexpected\ncategory value", "breck_category_pct", "data"),
              ("Feature in too\nfew examples", "breck_few_pct", "danger"),
              ("Data completely\nmissing", "breck_gone_pct", "danger")]
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for i, (lab, key, ck) in enumerate(breck_rows):
    v = num(key)
    y = len(breck_rows) - 1 - i
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=14, fontweight="bold", color=C["text"], linespacing=1.0)
    ax.text(v + 0.15, y, f"{val(key)}%", va="center", fontsize=15, fontweight="bold", color=C[ck])
ax.set_yticks([])
ax.set_xlim(0, 13)
ax.set_xticks(range(0, 13, 2))
ax.set_ylim(-0.6, len(breck_rows) - 0.4)
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelsize=14)
ax.set_xlabel(f"Share of {val('breck_pipelines')}+ Google pipelines\nwhere the check fired in {val('breck_days')} days (%)", fontsize=14)
sp.save(fig, out / "breck.svg")

# ------------------------------------------------------- broken-feature cost
fig, ax = plt.subplots(figsize=(5.2, 2.4))
for y, lab, v, ck, txt in ((1, f"No alert,\n{val('fc_days')} days", calcs.INCIDENT.undetected_loss.to("USD").magnitude / 1e3, "danger",
                            f"\\${val('fc_loss_k')}K"),
                           (0, f"Alert and\ncontain, {val('fc_alert_h')} h", calcs.INCIDENT.alerted_loss.to("USD").magnitude / 1e3, "data",
                            f"\\${val('fc_alert_loss_k')}K")):
    ax.barh(y, v, color=C[ck], height=0.55)
    ax.text(v + 15, y, txt, va="center", fontsize=14, fontweight="bold", color=C[ck])
    ax.text(-15, y, lab, ha="right", va="center", fontsize=13, fontweight="bold")
ax.text(500, 0.0, f"{val('fc_window_ratio')}× shorter", va="center", fontsize=14, fontweight="bold",
        color=C["data"])
ax.set_xlim(0, 1100)
ax.set_ylim(-0.6, 1.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Lost revenue, \\$K (constructed scenario)")
sp.save(fig, out / "feature_cost.svg")

# ------------------------------------------------------- exposure 20x
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    full, canary = num("exp_full_pct"), num("exp_canary_pct")
    ax.bar(0, full, color=C["danger"], width=0.6)
    ax.text(0, full + 3, f"{val('exp_full_pct')}%", ha="center", fontsize=15, fontweight="bold", color=C["danger"])
    a = 1.0 if k >= 2 else 0.0
    ax.bar(1, canary, color=C["data"], width=0.6, alpha=a)
    ax.text(1, canary + 3, f"{val('exp_canary_pct')}%", ha="center", fontsize=15, fontweight="bold", color=C["data"], alpha=a)
    ax.text(1, 45, f"{val('exp_ratio')}× less\nexposure", ha="center", fontsize=15, fontweight="bold",
            color=C["data"], alpha=a)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["all at once", f"{val('exp_canary_pct')}% canary"], fontsize=14, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("Users exposed until rollback (%)", fontsize=13)
    sp.save(fig, out / f"exposure_step{k}.svg")

# ------------------------------------------------------- canary schedule
stages = [(100 * share, dur.to("hour").magnitude) for share, dur in calcs.CANARY.stages]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    t0 = 0.0
    for share, dur in stages:
        ax.add_patch(Rectangle((t0, 0), dur, share, facecolor=C["memory_l"],
                               edgecolor=C["memory"], lw=1.5))
        ax.text(t0 + dur / 2, share + 3, f"{share:.0f}%", ha="center", fontsize=14,
                fontweight="bold", color=C["memory"])
        t0 += dur
    ax.add_patch(Rectangle((t0, 0), 0.6, 100, facecolor=C["data_l"], edgecolor=C["data"], lw=1.5))
    ax.text(t0 + 0.3, 103, "100%", ha="center", fontsize=14, fontweight="bold", color=C["data"])
    a = 1.0 if k >= 2 else 0.0
    ax.annotate("", xy=(0, 30), xytext=(num("canary_t1_h"), 30),
                arrowprops=dict(arrowstyle="<->", color=C["danger"], lw=1.6, alpha=a))
    ax.text(0.05, 36, f"{val('canary_t1_h')} h: {val('canary_samples')} samples\n"
            f"at {val('canary_p1_pct')}% of {val('canary_rate_m')} M/h", fontsize=13, color=C["danger"],
            fontweight="bold", alpha=a)
    ax.text(t0 + 0.75, 55, f"about {val('canary_total_h')} h\nin all", fontsize=14, color=C["text"],
            fontweight="bold", alpha=a)
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Hours since release", fontsize=13)
    ax.set_ylabel("Traffic share (%)", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_yticks([0, 50, 100])
    sp.save(fig, out / f"stages_step{k}.svg")

# ------------------------------------------------------- sample size
fig, ax = plt.subplots(figsize=(5.2, 3.3))
d = np.linspace(0.0003, 0.01, 400)
for test, ck, xy in ((calcs.TEST, "memory", (0.0078, 1.8e4)),
                     (calcs.SMALL, "compute", (0.0062, 6.5e2))):
    nn = [calc_two_proportion_sample_size(test.baseline_rate, x, test.z_alpha, test.z_beta) for x in d]
    ax.plot(d * 100, nn, color=C[ck], lw=2.8)
    ax.text(xy[0] * 100, xy[1], f"{100 * test.baseline_rate:.0f}% baseline", color=C[ck], fontsize=13,
            fontweight="bold")
ax.plot([num("mde_d_pt")], [num("mde_n")], "o", ms=10, color=C["memory"], mec="white", zorder=5)
ax.annotate(f"{val('mde_n')} at {val('mde_d_pt')} pt", (num("mde_d_pt"), num("mde_n")), xytext=(8, 30),
            textcoords="offset points", fontsize=13, fontweight="bold", color=C["memory"])
ax.plot([num("mde2_d_pt")], [num("mde2_n")], "o", ms=10, color=C["compute"], mec="white", zorder=5)
ax.annotate(f"{val('mde2_n')} at {val('mde2_d_pt')} pt", (num("mde2_d_pt"), num("mde2_n")), xytext=(40, 6),
            textcoords="offset points", fontsize=13, fontweight="bold", color=C["compute"])
ax.set_yscale("log")
ax.set_xlim(0, 1.0)
ax.set_ylim(5e2, 3e6)
ax.tick_params(labelsize=12)
ax.set_xlabel("Smallest lift to detect (percentage points)", fontsize=13)
ax.set_ylabel("Users per variant", fontsize=13)
sp.save(fig, out / "mde.svg")

# ------------------------------------------------------- model loading (TFX)
fig, ax = plt.subplots(figsize=(4.6, 2.4))
for y, lab, lo, hi, ck in ((1, "Shared\nthreads", num("tfx_load_lo"), num("tfx_load_hi"), "danger"),
                           (0, "Own load\nthread", num("tfx_own_lo"), num("tfx_own_hi"), "data")):
    ax.barh(y, hi - lo, left=lo, color=C[ck], height=0.5)
    ax.text(hi + 40 if y == 0 else (lo + hi) / 2, y + (0 if y == 0 else 0.42),
            f"{lo:,.0f} to {hi:,.0f} ms", va="center", ha="left" if y == 0 else "center",
            fontsize=14, fontweight="bold", color=C[ck])
    ax.text(-30, y, lab, ha="right", va="center", fontsize=14, fontweight="bold")
ax.set_xlim(0, 1600)
ax.set_ylim(-0.5, 1.8)
ax.set_yticks([])
ax.set_xticks([0, 500, 1000, 1500])
ax.tick_params(labelsize=12)
ax.spines["left"].set_visible(False)
ax.set_xlabel("p99.9 latency during a model load (ms)", fontsize=13)
sp.save(fig, out / "tfx_load.svg")

# ------------------------------------------------------- false alerts
# Drawn for its slot under the dashboard panel: about 7.8 x 3.3 cm.
fig, ax = plt.subplots(figsize=(5.2, 2.3))
M = np.logspace(0, 3, 300)
for fleet, ck in ((calcs.SIGMA3, "memory"), (calcs.SIGMA2, "danger")):
    ax.plot(M, fleet.any_false_alert(M), color=C[ck], lw=2.8)
ax.text(1.3, 0.42, "2-sigma", color=C["danger"], fontsize=13, fontweight="bold")
ax.text(950, 0.04, "3-sigma", color=C["memory"], fontsize=13, fontweight="bold", ha="right")
ax.axhline(1, color=C["neutral"], lw=1, ls=":")
ax.plot([num("al_monitors")], [calcs.SIGMA3.any_false_alert(calcs.SIGMA3.monitors)], "o", ms=9,
        color=C["memory"], mec="white", zorder=5)
ax.text(1.3, 0.95, f"3-sigma reaches {val('al_any3')}\nat {val('al_monitors')} checks", va="top", fontsize=12,
        fontweight="bold", color=C["memory"], linespacing=1.0)
ax.set_xscale("log")
ax.set_xlim(1, 1000)
ax.set_ylim(0, 1.05)
ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(["1", "10", "100", "1,000"])
ax.tick_params(labelsize=12)
ax.set_xlabel("Independent checks, M", fontsize=13)
ax.set_ylabel("P(any false alert)", fontsize=13)
sp.save(fig, out / "alerts.svg")

# ------------------------------------------------------- monitoring pyramid
fig, ax = plt.subplots(figsize=(5.9, 2.9))
ax.set_xlim(0, 10)
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 4.4)
hide_axis(ax)
levels = [("Infrastructure", "platform team", V["c1_l"], V["c1"]),
          ("Model", "model owner", V["c1_l"], V["c1"]),
          ("Portfolio", "domain team", V["c3_l"], V["c3"]),
          ("Business", "incident lead", V["bad_l"], V["bad"])]
for i, (lab, own, fc, ec) in enumerate(levels):
    y0, y1 = i * 1.0 + 0.1, i * 1.0 + 1.0
    w0, w1 = 7.0 - 1.3 * i, 7.0 - 1.3 * (i + 1)
    cx = 3.7
    ax.add_patch(Polygon([(cx - w0 / 2, y0), (cx + w0 / 2, y0), (cx + w1 / 2 + 0.01, y1),
                          (cx - w1 / 2 - 0.01, y1)], closed=True, facecolor=fc, edgecolor=ec, lw=1.6))
    ax.text(cx, (y0 + y1) / 2, lab, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=ec)
    ax.text(7.9, (y0 + y1) / 2, own, ha="left", va="center", fontsize=12.5,
            fontweight="bold", color=ec)
ax.text(7.9, 4.25, "first owner paged", ha="left", fontsize=11, color=C["neutral"])
ax.annotate("", xy=(7.6, 3.8), xytext=(7.6, 0.2),
            arrowprops=dict(arrowstyle="<->", color=C["neutral"], lw=1.3))
sp.save(fig, out / "pyramid.svg")

# ------------------------------------------------------- SLO vector
fig, ax = plt.subplots(figsize=(5.9, 2.6))
SLO = calcs.SLO
rows = [(lab, SLO.consumption[dim], SLO.owners[dim])
        for lab, dim in (("Availability", "availability"), ("Latency", "latency"),
                         ("Freshness", "freshness"), ("Model quality", "quality"))]
for i, (lab, v, own) in enumerate(rows):
    y = 3 - i
    bad = v > 1
    ax.barh(y, v, color=C["danger"] if bad else C["memory_l"],
            edgecolor=C["danger"] if bad else C["memory"], height=0.6, lw=1.5)
    ax.text(v - 0.03 if bad else v + 0.03, y, f"{v:.2f}", va="center", fontsize=12.5,
            fontweight="bold", ha="right" if bad else "left",
            color="white" if bad else C["memory"],
            bbox=None if bad else dict(facecolor="white", edgecolor="none", pad=0.5), zorder=5)
    ax.text(-0.03, y, f"{lab}\n({own} owner)", ha="right", va="center", fontsize=11.5,
            fontweight="bold")
ax.axvline(1.0, color=C["text"], lw=2)
ax.text(1.02, 3.55, "budget spent: B = 1", ha="left", fontsize=11.5, fontweight="bold")
ax.axvline(SLO.mean, color=C["neutral"], lw=1.5, ls="--")
ax.text(SLO.mean - 0.02, 3.55, f"mean {val('slo_mean')}", ha="right", fontsize=11.5, color=C["neutral"], fontweight="bold")
ax.text(1.36, 1.0, "max: stop\npromotion", va="center", fontsize=12, color=C["danger"],
        fontweight="bold")
ax.set_xlim(0, 1.75)
ax.set_ylim(-0.5, 3.8)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Budget consumed B = observed / allowed violations (illustrative)")
sp.save(fig, out / "slo_vector.svg")

# ------------------------------------------------------- Meta outage timeline
fig, ax = plt.subplots(figsize=(5.4, 3.0))


def hh(key):
    """Hour of the day for a clock time recorded as key_h and key_m."""
    return num(f"{key}_h") + num(f"{key}_m") / 60


def hhmm(key):
    return f"{val(key + '_h')}:{val(key + '_m')}"


ax.add_patch(Rectangle((hh("meta_start"), 0.6), hh("meta_end") - hh("meta_start"), 0.8,
                       facecolor=C["danger_l"], edgecolor=C["danger"], lw=1.5))
ax.text((hh("meta_start") + hh("meta_end")) / 2, 1.0, f"facebook.com unreachable\nabout {val('meta_hours')} h",
        ha="center", va="center", fontsize=13, fontweight="bold", color=C["danger"])
for t, lab, y, ha in (("meta_peak", f"{hhmm('meta_peak')} routing\nchanges peak", 2.15, "center"),
                      ("meta_return", f"{hhmm('meta_return')} routes\nreturn", 2.15, "center")):
    ax.plot([hh(t)], [1.6], "v", ms=12, color=C["network"])
    ax.text(hh(t), y, lab, ha=ha, va="center", fontsize=12, fontweight="bold",
            color=C["network"])
ax.text(18.6, 0.15, f"resolvers saw {val('meta_dns')}× normal DNS queries", ha="center", fontsize=12,
        fontweight="bold", color=C["neutral"])
ax.set_xlim(14.5, 22.5)
ax.set_ylim(-0.1, 2.7)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([15, 16, 17, 18, 19, 20, 21, 22])
ax.set_xticklabels([f"{h}:00" for h in range(15, 23)], fontsize=11)
ax.set_xlabel("4 October 2021, UTC (Cloudflare measurements)", fontsize=12)
sp.save(fig, out / "meta_timeline.svg")

# ------------------------------------------------------- Philly
fig, ax = plt.subplots(figsize=(5.4, 2.4))
parts = [("passed", "data"), ("killed", "compute"), ("failed", "danger")]
FLEET = calcs.FLEET
data = {"Jobs": [100 * float(FLEET.PhillyJobsPassedFraction), 100 * float(FLEET.PhillyJobsKilledFraction),
                 100 * float(FLEET.PhillyJobsUnsuccessfulFraction)],
        "GPU time": [100 * float(FLEET.PhillyGpuTimePassedFraction), 100 * float(FLEET.PhillyGpuTimeKilledFraction),
                     100 * float(FLEET.PhillyGpuTimeUnsuccessfulFraction)]}
for y, key in ((1, "Jobs"), (0, "GPU time")):
    left = 0
    for (lab, ck), v in zip(parts, data[key]):
        ax.barh(y, v, left=left, color=C[ck], height=0.6)
        ax.text(left + v / 2, y, f"{lab}\n{v:.0f}%", ha="center", va="center", fontsize=13.5,
                fontweight="bold", color="white")
        left += v
    ax.text(-1.5, y, key, ha="right", va="center", fontsize=13, fontweight="bold")
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 1.5)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel(f"Share (%), Microsoft Philly, {val('philly_jobs')} jobs over {val('philly_months_word')} months")
sp.save(fig, out / "philly.svg")

# ------------------------------------------------------- own against rent
OWN, HALF = calcs.OWN1K, calcs.HALF1K
life = num("own_life_months")
months = np.linspace(0, life, 200)
own = OWN.upfront.to("USD").magnitude / 1e6 + OWN.owned_cost_per_month.to("USD").magnitude / 1e6 * months
rent80 = OWN.rent_per_month.to("USD").magnitude / 1e6 * months
rent50 = HALF.rent_per_month.to("USD").magnitude / 1e6 * months
xm = OWN.crossover_months
assert HALF.crossover_months > life  # the 50 percent line does not cross within the life
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    ax.plot(months, own, color=C["memory"], lw=3)
    ax.text(1, 65, f"own: \\${val('own_upfront_m')}M up front + \\${val('own_opex_m')}M/yr", color=C["memory"], fontsize=11.5,
            fontweight="bold")
    ax.plot(months, rent80, color=C["compute"], lw=3)
    ax.text(29, 88, f"rent, {val('own_util_pct')}% busy", color=C["compute"], fontsize=11.5, fontweight="bold",
            ha="right")
    ax.plot([xm], [np.interp(xm, months, own)], "o", ms=9, color=C["danger"], zorder=6)
    ax.text(14, 76, f"cross at {val('own_cross_months')} months", ha="left", color=C["danger"], fontsize=11.5,
            fontweight="bold")
    a = 1.0 if k >= 2 else 0.0
    ax.plot(months, rent50, color=C["compute"], lw=2.4, ls="--", alpha=a)
    ax.text(1, 56, f"rent, {val('half_util_pct')}% busy:\nno crossing in {val('own_life_months')}\nmonths (derived)", color=C["compute"],
            fontsize=11, fontweight="bold", ha="left", va="top", alpha=a)
    ax.set_xlim(0, life)
    ax.set_ylim(0, 110)
    ax.set_xticks(np.arange(0, life + 1, 6))
    ax.set_xlabel(f"Months, {val('own_gpus')} H100s at \\${val('rent_rate')}/GPU-hour")
    ax.set_ylabel("Cumulative \\$M")
    sp.save(fig, out / f"buyrent_step{k}.svg")

# ------------------------------------------------------- TCO
fig, ax = plt.subplots(figsize=(5.9, 2.2))
comp = [("train", "memory"), ("infer", "compute"), ("data", "data"), ("iterate", "network")]
TERMS = ("training", "inference", "data", "iteration")
budgets = {"Startup": [calcs.STARTUP.monthly[t].to("USD/month").magnitude for t in TERMS],
           "50-model company": [calcs.COMPANY.monthly[t].to("USD/month").magnitude for t in TERMS]}
for y, key in ((1, "Startup"), (0, "50-model company")):
    vals = np.array(budgets[key])
    sh = vals / vals.sum() * 100
    left = 0
    for (lab, ck), s in zip(comp, sh):
        ax.barh(y, s, left=left, color=C[ck], height=0.6)
        if s > 7.5:
            ax.text(left + s / 2, y, f"{lab}\n{s:.0f}%", ha="center", va="center",
                    fontsize=11.5 if s > 9 else 10, fontweight="bold", color="white")
        left += s
    ax.text(-1.5, y, key, ha="right", va="center", fontsize=12.5, fontweight="bold")
_sh = calcs.STARTUP.shares
ax.text(11, 1.42, f"infer {100 * _sh['inference']:.0f}%, data {100 * _sh['data']:.0f}%", fontsize=10.5,
        color=C["neutral"], fontweight="bold")
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 1.75)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Share of monthly TCO (%), constructed budgets")
sp.save(fig, out / "tco.svg")

print("done")
