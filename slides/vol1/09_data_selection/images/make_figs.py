#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 9 (Data Selection).

Every number comes from MLSysIM through calcs.py: val(key) is the string a
slide prints and num(key) its value, and bar lengths read the Scenarios.*
quantities directly. The notes below say where each figure's inputs live.
  - Data wall: reported pretraining tokens and the stock of public human text
    (Villalobos et al. 2024), read from MLSysIM's literature series
    (Literature.Series.PretrainingTokens and PublicTextStock), the same data
    as the Data Engineering chapter's figure; the fit is recomputed here
    exactly as the book does (least squares on log10 tokens), and its label
    is Scenarios.PretrainingCorpusGrowth's growth.
  - GPT-2 on WebText and tokens per parameter: Scenarios.Gpt2XlWebTextCorpus
    (GPT-2 XL, WebText, Chinchilla constants), Models GPT-3 and Llama 2 70B.
  - Selection-inequality scoreboard: Scenarios.ResNet50TenPercentCoresetH100
    (Example 9.2 and Sec. 9.7.1), at the chapter's unrounded minutes.
  - Deduplication: Lee et al. 2022 near-duplicate shares
    (Literature.DataSelectionResults) on GPT-2's run,
    Scenarios.Gpt2WebTextNearDuplicates.
  - MinHash page text: the C4 sentence Lee et al. 2022 found repeated
    (arXiv:2107.06499v2, footnote 1); signature values are illustrative.
  - Coverage floor: Scenarios.RareClassTenPercentCoreset (Sec. 9.3.4, 9.10).
  - Curriculum warmup: Scenarios.Cifar10LinearCurriculum (W illustrative).
  - Selection via proxy and FixMatch: Coleman et al. 2020 and Sohn et al.
    2020 (table 2, best error per label budget), Literature.DataSelectionResults.
  - ResNet-50 data path and echoing: Scenarios.ResNet50HalfCoresetEpochH100
    and ResNet50EchoingH100 (Napkin Math 9.4 and 9.5).
  - Cost breakdown and amortized ROI: Scenarios.ImageNetDataCostBreakdown
    and MinHashPipelineAmortization (Napkin Math 9.6; tables 13 and 14).

Run from this folder with the MLSysBook venv (it imports mlsysim):
    python3 make_figs.py
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402
from deckvals import val, num  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent


def note(ax, text, x=0.0, y=-0.30):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, color=C["neutral"])


# -----------------------------------------------------------------------------
# 1. The data wall (slide variant of Data Engineering's figure, from MLSysIM)
# -----------------------------------------------------------------------------
def data_wall():
    root = Path(__file__).resolve().parents[4] / "mlsysim"
    sys.path.insert(0, str(root))
    from mlsysim import Literature
    corpora = Literature.Series.PretrainingTokens.entries
    stock = {r["bound"]: r["tokens"] for r in Literature.Series.PublicTextStock.entries}

    def year(d):
        d = dt.date.fromisoformat(d)
        return d.year + (d.timetuple().tm_yday - 1) / 365.25

    xs = np.array([year(r["date"]) for r in corpora])
    ys = np.array([r["tokens"] for r in corpora])
    slope, icept = np.polyfit(xs, np.log10(ys), 1)
    growth = 10 ** slope
    assert 2.8 <= growth <= 3.0, growth
    show = {"GPT-3": (4, 8, "left"), "Chinchilla": (-6, 6, "right"),
            "Llama 2": (6, -4, "left"), "Qwen3": (-8, 4, "right")}
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.9))
        if k == 2:
            ax.axhspan(stock["low"], stock["high"], color=C["compute_l"], zorder=0)
            ax.axhline(stock["median"], color=C["compute"], lw=1.2, ls=":")
            ax.text(2019.95, stock["median"] * 1.35, "Stock of public human text",
                    color=C["compute"], fontsize=12, fontweight="bold", va="bottom")
            ax.text(2019.95, stock["median"] * 0.75, "estimate, 90% interval",
                    color=C["compute"], fontsize=10, va="top")
        xx = np.linspace(2020, 2028.5, 50)
        ax.plot(xx, 10 ** (slope * xx + icept), color=C["memory"], lw=2, ls="--")
        ax.scatter(xs, ys, s=55, color=C["memory"], edgecolors="white", zorder=3)
        for r, x, y in zip(corpora, xs, ys):
            if r["model"] in show:
                dx, dy, ha = show[r["model"]]
                ax.annotate(r["model"], (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=11, ha=ha, color=C["text"])
        ax.text(2026.1, 10 ** (slope * 2026.1 + icept) / 6, f"{val('corpus_growth')}× per year",
                color=C["memory"], fontsize=12, fontweight="bold", ha="left")
        ax.set_yscale("log")
        ax.set_ylim(1e11, 3e15)
        ax.set_xlim(2019.8, 2028.5)
        ax.set_xticks(range(2020, 2029, 2))
        ax.set_ylabel("Training tokens")
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        sp.save(fig, out / f"data_wall_step{k}.svg")


# -----------------------------------------------------------------------------
# 2. GPT-2 on WebText: unique, compute-optimal, processed tokens
# -----------------------------------------------------------------------------
def gpt2_tokens():
    g = calcs.GPT2
    uniq, opt, done = (g.unique_tokens.magnitude / 10 ** 9, g.optimal_tokens.magnitude / 10 ** 9,
                       g.tokens_processed.magnitude / 10 ** 9)
    sp.hbars(out / "gpt2_tokens.svg", [
        ("Unique in WebText", uniq, "data", f"{val('webtext_tokens_b')} B"),
        ("Compute-optimal", opt, "neutral", f"{val('optimal_tokens_b')} B"),
        ("Processed in training", done, "compute",
         f"{val('gpt2_tokens_b')} B: {val('gpt2_passes')} passes"),
    ], "B", log=False, xlabel="Tokens (billions)", size=(4.2, 1.7))


# -----------------------------------------------------------------------------
# 3c. Curriculum warmup: linear pacing serves about half the samples
# -----------------------------------------------------------------------------
def warmup():
    """Linear pacing (chapter Sec. 9.4.1): epoch j of a W-epoch warmup serves
    the easiest j/W of the D samples, so the warmup serves D(W+1)/2 in place
    of W*D. W = 10 is an illustrative choice: 5.5 D of 10 D."""
    W = calcs.CURRIC.warmup_epochs
    j = np.arange(1, W + 1)
    frac = j / W
    assert abs(frac.sum() - (W + 1) / 2) < 1e-9
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.bar(j, np.ones(W), color="white", edgecolor=C["neutral"], lw=1.2, ls="--", width=0.75)
    ax.bar(j, frac, color=C["data"], width=0.75)
    post = np.arange(W + 1, W + 4)
    ax.bar(post, np.ones(3), color=C["neutral_l"], edgecolor=C["neutral"], lw=1, width=0.75)
    ax.text(W + 2, 1.06, "full data", ha="center", fontsize=11, color=C["neutral"])
    ax.text(1.0, 1.24, f"Warmup serves {val('warm_served')} D samples in place of {val('warm_w')} D",
            fontsize=13, fontweight="bold", color=C["data"])
    ax.text(1.0, 1.10, "dashed: what full-data epochs would serve", fontsize=11,
            color=C["neutral"])
    ax.set_xticks(list(j) + list(post))
    ax.set_xticklabels([str(v) for v in j] + ["", "", ""])
    ax.set_xlim(0.4, W + 3.6)
    ax.set_ylim(0, 1.38)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "D/2", "D"])
    ax.set_xlabel(f"Epoch (warmup W = {W}, illustrative)")
    ax.set_ylabel("Samples served")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "warmup.svg")


# -----------------------------------------------------------------------------
# 4. Tokens per parameter
# -----------------------------------------------------------------------------
def tokens_per_param():
    """GPT-2 XL appears twice: its unique tokens (WebText) and its training
    tokens (the run repeated WebText). Chinchilla's reference of about 20
    counts training tokens, each seen about once in that study."""
    g, gpt3, llama2 = calcs.GPT2, calcs.Models.Language.GPT3, calcs.Models.Language.Llama2_70B

    def tpp(model):
        return calcs.m(model.training_tokens, "count") / calcs.m(model.parameters, "param")
    rows = [(f"GPT-3 ({val('gpt3_params_b')} B)", tpp(gpt3), "neutral", val("gpt3_tpp")),
            ("GPT-2 XL, unique in WebText", g.unique_tokens_per_param, "danger", val("gpt2_unique_tpp")),
            ("GPT-2 XL, training tokens", g.trained_tokens_per_param, "compute",
             f"{val('gpt2_trained_tpp')}: WebText {val('gpt2_passes')} times"),
            ("Llama 2 70B", tpp(llama2), "neutral", val("llama2_tpp"))]
    ref = g.tokens_per_param
    n = len(rows)
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.4))
        for i, (lab, v, ck, txt) in enumerate(rows):
            y = n - 1 - i
            col = C[ck] if (k == 2 or ck == "neutral") else C["neutral"]
            ax.barh(y, v, color=col, height=0.6)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=13, fontweight="bold", color=C["text"])
            ax.text(v + 1.5, y, txt, va="center", fontsize=13, fontweight="bold", color=col)
        alpha = 1.0 if k == 2 else 0.0
        ax.axvline(ref, color=C["data"], lw=2, ls="--", alpha=alpha)
        ax.text(ref + 1.5, n - 0.55, f"Chinchilla reference: about {val('tpp_ref')}",
                color=C["data"], fontsize=11, fontweight="bold", alpha=alpha, va="bottom")
        ax.text(ref + 12, 2.0, f"unique: a third of {val('tpp_ref')}", color=C["danger"],
                fontsize=12, fontweight="bold", va="center", alpha=alpha)
        ax.set_yticks([])
        ax.set_xlim(0, 150)
        ax.set_ylim(-0.5, n - 0.1)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Tokens per parameter")
        sp.save(fig, out / f"tokens_per_param_step{k}.svg")


# -----------------------------------------------------------------------------
# 5. Selection inequality scoreboard (stacked: scoring + training)
# -----------------------------------------------------------------------------
def scoreboard():
    cs = calcs.CORESET
    base, sub = calcs.m(cs.full_time, "minute"), calcs.m(cs.subset_time, "minute")
    sa, sb = calcs.m(cs.score_time(), "minute"), calcs.m(cs.score_time(cs.proxy), "minute")
    rows = [("Full data", 0.0, base, f"{val('base_min')} min"),
            ("Coreset, target scores once", sa, sub, f"saves {val('save_a_pct')}%"),
            ("Coreset, proxy scores once", sb, sub, f"saves {val('save_b_pct')}%"),
            ("Coreset, rescore every epoch", calcs.m(cs.rescoring_time, "minute"), sub,
             f"{val('rescore_min')} + {val('subset_min')} min: saves {val('rescore_save_pct')}%")]
    show_upto = {1: 1, 2: 3, 3: 4}
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.4))
        for i, (lab, score, train, txt) in enumerate(rows):
            y = 3 - i
            vis = i < show_upto[k]
            a = 1.0 if vis else 0.0
            ax.barh(y, train, left=score, color=C["neutral"] if i == 0 else C["data"],
                    height=0.6, alpha=a)
            if score:
                ax.barh(y, score, color=C["compute"], height=0.6, alpha=a)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=12, fontweight="bold", color=C["text"])
            ck = "danger" if i == 3 else ("neutral" if i == 0 else "data")
            ax.text(score + train + 3, y, txt, va="center", fontsize=12, fontweight="bold",
                    color=C[ck], alpha=a)
        a3 = 1.0 if k == 3 else 0.0
        ax.text(35, -0.62, "scoring", color=C["compute"], fontsize=11, fontweight="bold",
                ha="center", va="top", alpha=a3)
        ax.set_yticks([])
        ax.set_xlim(0, 300)
        ax.set_ylim(-1.0, 3.5)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"Minutes on one H100 ({val('train_epochs')} epochs)")
        sp.save(fig, out / f"scoreboard_step{k}.svg")


# -----------------------------------------------------------------------------
# 6. Deduplication returns hours of GPT-2's run
# -----------------------------------------------------------------------------
def dedup_hours():
    d = calcs.DEDUP
    rows = [(f"C4: {val('lee_c4_pct')}% near-duplicates", calcs.m(d.time_saved("C4"), "hour"),
             f"{val('dedup_c4_h')} h, {val('dedup_c4_kwh')} kWh"),
            (f"RealNews: {val('lee_rn_pct')}% near-duplicates", calcs.m(d.time_saved("RealNews"), "hour"),
             f"{val('dedup_rn_h')} h, {val('dedup_rn_kwh')} kWh")]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 1.8))
        for i, (lab, h, txt) in enumerate(rows):
            y = 1 - i
            a = 1.0 if i < k else 0.0
            ax.barh(y, h, color=C["data"], height=0.6, alpha=a)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=12, fontweight="bold", color=C["text"])
            ax.text(h + 0.3, y, txt, va="center", fontsize=13,
                    fontweight="bold", color=C["data"], alpha=a)
        ax.set_yticks([])
        ax.set_xlim(0, 22)
        ax.set_ylim(-0.5, 1.5)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"Hours of GPT-2's {val('node_hours')}-hour run saved")
        sp.save(fig, out / f"dedup_hours_step{k}.svg")


# -----------------------------------------------------------------------------
# 7. Coverage floor
# -----------------------------------------------------------------------------
def coverage():
    cv = calcs.COVER
    fl = cv.floor
    rows = [(f"Rare examples in the {val('cov_pool_m')}M pool", cv.rare, "neutral", val("cov_rare")),
            (f"Kept by an unstratified {val('cov_keep_pct')}% coreset", cv.expected_kept, "danger",
             f"about {val('cov_kept')}"),
            ("Kept with the floor reserved first", fl, "data", val("cov_floor"))]
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.1))
        for i, (lab, amount, ck, txt) in enumerate(rows):
            y = 2 - i
            a = 1.0 if i < k else 0.0
            ax.barh(y, amount, color=C[ck], height=0.6, alpha=a)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=13, fontweight="bold", color=C["text"])
            ax.text(max(amount, fl) + 25, y, txt, va="center", fontsize=13,
                    fontweight="bold", color=C[ck], alpha=a)
        ax.axvline(fl, color=C["data"], lw=1.5, ls="--")
        ax.text(fl, 2.45, f" floor: {val('cov_floor')} (illustrative)", color=C["data"], fontsize=11,
                fontweight="bold", va="bottom")
        ax.set_yticks([])
        ax.set_xlim(0, 1300)
        ax.set_ylim(-0.5, 2.9)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Rare-class examples")
        sp.save(fig, out / f"coverage_step{k}.svg")


# -----------------------------------------------------------------------------
# 8. Selection via proxy (Coleman et al. 2020)
# -----------------------------------------------------------------------------
def svp():
    sp.hbars(out / "svp_coreset.svg", [
        ("Target trains on all of CIFAR10", 1.0, "neutral", "1.0"),
        (f"Proxy removes {val('svp_removed_pct')}%, target trains", 1 / num("svp_end2end"), "data",
         f"{val('svp_end2end')}× faster, same final accuracy"),
    ], "", log=False, xlabel="End-to-end training time (relative)", size=(5.8, 1.6))


# -----------------------------------------------------------------------------
# 9. FixMatch on CIFAR-10
# -----------------------------------------------------------------------------
def fixmatch():
    sp.hbars(out / "fixmatch.svg", [
        (f"{val(f'fm_labels_{k}')} labels", num(f"fm_acc_{k}"), "data", f"{val(f'fm_acc_{k}')}%")
        for k in ("40", "250", "4k")
    ], "", log=False, xlabel=f"CIFAR-10 test accuracy (%); full labeled set: {val('cifar_train')}",
        size=(5.8, 2.0))


# -----------------------------------------------------------------------------
# 10. Data echoing
# -----------------------------------------------------------------------------
def echo():
    r, e2 = calcs.ECHO.ratio, calcs.ECHO.echo_factor
    e = np.linspace(1, 3.2, 200)
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        if k >= 2:
            ax.plot(e, np.minimum(1, e / r) * 100, color=C["compute"], lw=3)
            ax.axvline(r, color=C["neutral"], lw=1, ls="--")
            ax.text(r + 0.04, 60, f"e = r = {val('echo_r')}", color=C["neutral"], fontsize=11)
            ax.plot([e2], [100], "o", ms=11, color=C["compute"], mec="white", mew=1.5, zorder=5)
            ax.annotate(f"e = {val('echo_e')}: {val('std_epochs')} epochs, {val('echo_h_no')} h to {val('echo_h_e')} h",
                        (e2, 100), xytext=(8, -26),
                        textcoords="offset points", fontsize=12, fontweight="bold",
                        color=C["compute"])
        if k == 3:
            ax.plot(e, [100] * len(e), color=C["memory"], lw=2.5, ls=(0, (4, 2)))
            ax.text(2.35, 103, "host twice as fast (r = 0.5)", color=C["memory"],
                    fontsize=11, fontweight="bold")
        ax.plot([1], [100 / r], "o", ms=11, color=C["danger"], mec="white", mew=1.5, zorder=5)
        ax.annotate(f"no echo: busy {val('echo_busy_pct')}%", (1, 100 / r), xytext=(10, -4),
                    textcoords="offset points", fontsize=12, fontweight="bold",
                    color=C["danger"])
        ax.set_xlim(0.9, 3.2)
        ax.set_ylim(30, 112)
        ax.set_xticks([1, 1.5, 2, 2.5, 3])
        ax.set_xlabel("Echo factor e (uses of each prepared image)")
        ax.set_ylabel("Accelerator busy (%)")
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        sp.save(fig, out / f"echo_step{k}.svg")


# -----------------------------------------------------------------------------
# 11. Cost breakdown and amortized ROI
# -----------------------------------------------------------------------------
def cost():
    b, usd = calcs.BILL, lambda q: calcs.m(q, "USD")
    sp.hbars(out / "cost_bars.svg", [
        ("Acquisition (licensed images)", usd(b.acquisition), "data", f"${val('cost_acq')}"),
        (f"Labels ({val('imagenet_m')}M at ${val('cost_label_usd')})", usd(b.label_cost), "data",
         f"${val('cost_labels')}"),
        (f"Storage ({val('archive_gb')} GB, {val('storage_months')} months)", usd(b.storage_cost), "neutral",
         f"${val('cost_storage')}"),
        (f"Compute ({val('cost_runs')} runs on one H100)", usd(b.compute_cost), "compute",
         f"${val('cost_compute')} = {val('cost_compute_pct')}%"),
    ], "", log=True, xlabel=f"Dollars (log scale); total ${val('cost_total')}", size=(5.8, 2.3))


def roi():
    pl = calcs.PIPELINE
    inv, save = calcs.m(pl.investment, "USD"), calcs.m(pl.saving_per_run, "USD")
    runs = np.array(pl.run_counts)
    vals = np.array([pl.savings_multiple(n) for n in pl.run_counts])
    breakeven = pl.breakeven_runs
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    xs = np.linspace(1, 50, 200)
    ax.plot(xs, save * xs / inv, color=C["data"], lw=3)
    ax.axhline(1, color=C["neutral"], lw=1)
    ax.axvline(breakeven, color=C["neutral"], lw=1, ls="--")
    ax.text(breakeven * 1.05, 5.2, f"break-even:\n{val('roi_breakeven')} runs", color=C["neutral"], fontsize=11)
    for n, v in zip(runs, vals):
        ck = "data" if v > 1 else "danger"
        ax.plot([n], [v], "o", ms=9, color=C[ck], zorder=5)
        off, ha = {50: ((-8, 4), "right"), 5: ((-6, -20), "right")}.get(n, ((9, -20), "left"))
        ax.annotate(f"{val(f'roi_mult_{n}')}$\\times$", (n, v), xytext=off,
                    textcoords="offset points", fontsize=12, fontweight="bold",
                    color=C[ck], ha=ha)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 50])
    ax.set_xticklabels(["1", "5", "10", "50"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 60)
    ax.set_ylim(-1.2, 10.5)
    ax.set_xlabel("Training runs that reuse the deduplicated data (log scale)")
    ax.set_ylabel("Savings / cost")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    sp.save(fig, out / "roi_runs.svg")


# -----------------------------------------------------------------------------
# 12. Hand-drawn SVG diagrams (MinHash build, ResNet-50 data path)
# -----------------------------------------------------------------------------
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def svg_doc(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" {FONT}>\n'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>\n{body}</svg>\n')


def text(x, y, s, size=18, color=None, weight="normal", anchor="middle", op=1.0):
    color = color or C["text"]
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
            f'text-anchor="{anchor}" opacity="{op}">{s}</text>\n')


def box(x, y, w, h, stroke, fill, sw=2.5, op=1.0, rx=8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>\n')


def arrow(x1, y, x2, op=1.0, color="#666666"):
    return (f'<g opacity="{op}"><line x1="{x1}" y1="{y}" x2="{x2 - 10}" y2="{y}" stroke="{color}" '
            f'stroke-width="3"/><polygon points="{x2},{y} {x2 - 12},{y - 7} {x2 - 12},{y + 7}" '
            f'fill="{color}"/></g>\n')


def minhash():
    """Document, shingles, signature, bands and buckets, candidate check.
    The page text is the start of the 61-word C4 sentence that Lee et al.
    2022 found 61,036 times (arXiv:2107.06499v2, footnote 1); signature
    values are illustrative. 128 hashes, 8M documents come from the chapter
    (Sec. 9.3.2); pairs, read time, and signature size are computed in
    calcs.py."""
    mant = val("all_pairs_m")
    sup = val("all_pairs_e").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    W, H = 900, 360
    xs = [20, 240, 485, 700]
    for k in range(1, 5):
        b = ""
        vis = [1.0 if i < k else 0.0 for i in range(4)]
        # column 1: document
        b += text(xs[0] + 89, 40, "Web page", 18, weight="bold", op=vis[0])
        b += box(xs[0], 60, 178, 150, C["neutral"], "#F4F4F4", op=vis[0])
        for i, line in enumerate(["by combining", "fantastic ideas,", "interesting", "arrangements, ..."]):
            b += text(xs[0] + 14, 95 + 30 * i, line, 17, anchor="start", op=vis[0])
        b += text(xs[0] + 89, 236, f"{val('lee_repeats')} copies in C4", 16, color=C["danger"],
                  weight="bold", op=vis[0])
        # column 2: shingles
        b += arrow(xs[0] + 184, 135, xs[1] - 20, op=vis[1])
        b += text(xs[1] + 100, 40, "Shingles", 18, weight="bold", op=vis[1])
        for i, sh in enumerate(["by combining fantastic", "combining fantastic ideas",
                                "fantastic ideas, interesting", "ideas, interesting arr..."]):
            b += box(xs[1] - 14, 62 + 37 * i, 228, 30, C["memory"], C["memory_l"], sw=1.5, op=vis[1], rx=4)
            b += text(xs[1] + 100, 83 + 37 * i, sh, 15, op=vis[1])
        # column 3: signature
        b += arrow(xs[1] + 220, 135, xs[2] - 6, op=vis[2])
        b += text(xs[2] + 95, 40, "Signature", 18, weight="bold", op=vis[2])
        vals = ["17", "4", "92", "33", "8", "61", "25", "70"]
        for i, v in enumerate(vals):
            yy = 62 + 18.5 * i
            band = i // 2
            fill = [C["compute_l"], C["data_l"], C["compute_l"], C["data_l"]][band]
            b += box(xs[2] + 45, yy, 100, 18, C["neutral"], fill, sw=1, op=vis[2], rx=1)
            b += text(xs[2] + 95, yy + 15, v, 15, op=vis[2])
        b += text(xs[2] + 95, 230, f"{val('minhash_hashes')} minimum hashes", 16, color=C["neutral"], op=vis[2])
        b += text(xs[2] + 95, 250, "(8 of them shown)", 15, color=C["neutral"], op=vis[2])
        # column 4: bands and buckets
        b += arrow(xs[2] + 196, 135, xs[3] - 6, op=vis[3])
        b += text(xs[3] + 95, 40, "Bands to buckets", 18, weight="bold", op=vis[3])
        for i in range(3):
            hl = i == 1
            b += box(xs[3] + 2 + 66 * i, 70, 60, 110, C["data"] if hl else C["neutral"],
                     C["data_l"] if hl else "#F4F4F4", sw=3 if hl else 1.5, op=vis[3], rx=4)
        b += text(xs[3] + 95, 115, "page A", 15, color=C["data"], weight="bold", op=vis[3])
        b += text(xs[3] + 95, 140, "page B", 15, color=C["data"], weight="bold", op=vis[3])
        b += text(xs[3] + 95, 210, "same bucket:", 15, color=C["data"], weight="bold", op=vis[3])
        b += text(xs[3] + 95, 230, "compare only these", 15, color=C["data"], weight="bold", op=vis[3])
        # bottom line: the all-pairs cost, crossed out in the last step
        b += text(450, 295, f"All pairs of {val('webtext_docs_m')}M pages: {mant} × 10{sup} comparisons", 19,
                  color=C["neutral"], op=vis[3])
        b += (f'<line x1="225" y1="289" x2="675" y2="289" stroke="{C["danger"]}" '
              f'stroke-width="2.5" opacity="{vis[3]}"/>\n')
        b += text(450, 335, f"Instead: one {val('webtext_gb')} GB read ({val('read_s')} s) and "
                  f"{val('sig_gb')} GB of signatures", 19,
                  color=C["data"], weight="bold", op=vis[3])
        (out / f"minhash_step{k}.svg").write_text(svg_doc(W, H, b))


def data_path():
    """ResNet-50 on one H100: four stages (book figure, redrawn at slide
    size). Numbers: chapter, Napkin Math 9.4 and 9.5 and its data-path
    figure. The 126 s epoch is the chapter's; derived values come from
    calcs.py."""
    W, H = 900, 330
    stages = [("Local NVMe", "storage", C["memory"], C["memory_l"],
               f"{val('archive_gb')} GB read", f"{val('read_archive_s')} s per epoch"),
              ("Host CPUs", "decode, augment", C["data"], C["data_l"],
               f"needs {val('demand_ips')}", "images/s"),
              ("PCIe link", "", C["network"], C["network_l"], f"{val('link_gb')} GB",
               f"{val('link_s')} s per epoch"),
              ("H100", "accelerator", C["compute"], C["compute_l"],
               f"{val('epoch_s')} s, {val('epoch_kj')} kJ", "per epoch")]
    notes = {3: [f"Coreset: {val('pruned_s')} s, {val('pruned_kj')} kJ",
                 f"scoring: {val('score_s')} s, once"],
             0: ["Halves only if the", f"{val('half_gb')} GB subset is written"],
             1: [f"Supplies only {val('host_ips')}/s:", "the slow stage"]}
    order = [3, 0, 1]
    xs = [10, 240, 470, 700]
    for k in range(1, 4):
        b = ""
        active = order[k - 1]
        for i, (t1, t2, st, fl, n1, n2) in enumerate(stages):
            x = xs[i]
            w = 190
            hl = i == active
            red = (i == 1 and k == 3)
            b += box(x, 140, w, 110, C["danger"] if red else st, fl, sw=6 if hl else 2.5)
            b += text(x + w / 2, 188 if t2 else 202, t1, 21, weight="bold")
            if t2:
                b += text(x + w / 2, 214, t2, 17)
            b += text(x + w / 2, 285, n1, 17, color=C["text"])
            b += text(x + w / 2, 308, n2, 17, color=C["text"])
            if i < 3:
                b += arrow(x + w + 4, 195, xs[i + 1] - 4)
            shown = i in order[:k]
            if i in notes:
                for j, line in enumerate(notes[i]):
                    col = C["danger"] if i == 1 else st
                    b += text(x + w / 2, 80 + 26 * j, line, 17, color=col, weight="bold",
                              op=1.0 if shown else 0.0)
        (out / f"data_path_step{k}.svg").write_text(svg_doc(W, H, b))


if __name__ == "__main__":
    data_wall()
    gpt2_tokens()
    warmup()
    tokens_per_param()
    scoreboard()
    dedup_hours()
    coverage()
    svp()
    fixmatch()
    echo()
    cost()
    roi()
    minhash()
    data_path()
    print("figures written")
