"""Lecture 15 (Trajectory Curation): every figure the deck draws.

Run: python3 make_figs.py   (writes SVGs here; `make deck V=3 D=15_trajectory_curation`
converts them to PDF). Plots use the shared slideplot helper and are drawn at
the size of their slot; diagrams are SVG at 900 units wide. Every value comes
from calcs.py, which reads MLSysIM and data/corpus_stats.json; this script
imports it and prints the recorded values (val, num), so figures and slides
agree.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import slideplot as sp  # noqa: E402
from slideplot import plt  # noqa: E402
import calcs  # noqa: E402
from deckvals import val, num  # noqa: E402

STATS = json.loads((HERE / "data" / "corpus_stats.json").read_text())

# Volume III roles (volumes/vol3/palette.md)
BLUE, BLUE_L = "#0284c7", "#e0f2fe"
GREEN, GREEN_L = "#16a34a", "#dcfce7"
AMBER, AMBER_L = "#d97706", "#fef3c7"
PURPLE, PURPLE_L = "#7c3aed", "#f3e8ff"
RED, RED_L = "#dc2626", "#fee2e2"
GRAY, GRAY_L = "#64748b", "#f8fafc"
LINE = "#94a3b8"
INK = "#0f172a"
FONT = "Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------- SVG helpers
def svg(path, w, h, body):
    marks = "".join(
        f'<marker id="a{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="{c}"/></marker>'
        for n, c in [("g", GRAY), ("p", PURPLE), ("a", AMBER), ("r", RED), ("b", BLUE), ("n", GREEN)])
    (HERE / path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{FONT}">\n<defs>{marks}</defs>\n<rect width="{w}" height="{h}" fill="white"/>\n'
        + body + "\n</svg>\n")


def g(body, o=1.0):
    return f'<g opacity="{o}">{body}</g>'


def box(x, y, w, h, fill, stroke, rx=8, dash=False, sw=2.5):
    d = ' stroke-dasharray="8 6"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=20, color=INK, anchor="middle", weight="normal", family=None):
    f = f' font-family="{family}"' if family else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{f}>{s}</text>')


def arrow(x1, y1, x2, y2, m="g", color=GRAY, dash=False, sw=2.5):
    d = ' stroke-dasharray="7 5"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} '
            f'marker-end="url(#a{m})"/>')


def card(x, y, w, h, title, sub, fill, stroke, ts=21, ss=18, dash=False, tcolor=INK):
    out = box(x, y, w, h, fill, stroke, dash=dash)
    if sub:
        out += text(x + w / 2, y + h / 2 - 4, title, ts, tcolor, weight="bold")
        out += text(x + w / 2, y + h / 2 + 21, sub, ss, GRAY)
    else:
        out += text(x + w / 2, y + h / 2 + 7, title, ts, tcolor, weight="bold")
    return out


# ---------------------------------------------------------------- hook (NM 15.1)
def hook():
    """3,000 failures a month, split by the probe that removed them."""
    t = calcs.TRI
    W, H = 900, 300
    x0, x1 = 40, 860
    scale = (x1 - x0) / t.failures
    parts = [("schema field missing", t.removed["interface"], val("tri_iface"), AMBER, AMBER_L),
             ("sandbox timeout", t.removed["runtime"], val("tri_rt"), PURPLE, PURPLE_L),
             ("residual", t.residual, val("tri_residual"), RED, RED_L)]
    for step in (1, 2, 3):
        b = text(450, 40, f"{val('tri_fail')} failed trajectories a month (illustrative, NM 15.1)", 22,
                 INK, weight="bold")
        if step == 1:
            b += box(x0, 70, x1 - x0, 80, "#e2e8f0", LINE, rx=6)
            b += text(450, 120, "?   what fixes them   ?", 24, GRAY, weight="bold")
        else:
            x = x0
            for label, n, shown, s, f in parts:
                w = n * scale
                b += box(x, 70, w, 80, f, s, rx=6)
                b += text(x + w / 2, 105, shown, 24, s, weight="bold")
                lab_x = x + w / 2 if n > 400 else x1 - 4
                anchor = "middle" if n > 400 else "end"
                b += text(lab_x, 135 if n > 400 else 185, label, 18, s, anchor=anchor)
                x += w
        if step == 3:
            sx = x0 + t.system_failures * scale
            b += f'<path d="M{x0},210 L{x0},225 L{sx - 6},225 L{sx - 6},210" fill="none" stroke="{GREEN}" stroke-width="3"/>'
            b += text((x0 + sx) / 2, 258, f"repair below the weights: {val('tri_system')} gone for every model",
                      20, GREEN, weight="bold")
            b += f'<path d="M{sx},210 L{sx},225 L{x1},225 L{x1},210" fill="none" stroke="{BLUE}" stroke-width="3"/>'
            b += text(x1, 285, f"weights: at most {val('tri_residual')}", 20, BLUE, anchor="end", weight="bold")
        svg(f"hook_step{step}.svg", W, H, b)


# ---------------------------------------------------------------- ladder (§15.2.2)
def ladder():
    """Drawn 900 x 420 for a full-width slot about 12 cm wide: type 18 units or more."""
    W, H = 900, 430
    rows = [8, 88, 168, 248]
    for step in (1, 2, 3):
        o2 = 1 if step >= 2 else 0
        o3 = 1 if step >= 3 else 0
        b = text(70, 26, "cheap,", 18, GRAY) + text(70, 46, "reversible", 18, GRAY)
        b += f'<polygon points="62,58 78,58 90,378 50,378" fill="#e2e8f0" stroke="{LINE}"/>'
        b += text(70, 402, "costly,", 18, GRAY) + text(70, 422, "global", 18, GRAY)
        b += card(150, rows[0], 300, 60, "Measured failure", "recorded failing trace", RED_L, RED)
        b += arrow(300, 68, 300, 86, "r", RED)
        b += card(150, rows[1], 300, 60, "Context probe", "supply the missing fact", GRAY_L, LINE)
        b += text(522, rows[1] + 22, "disappears", 18, GRAY) + arrow(450, rows[1] + 30, 592, rows[1] + 30, "n", GREEN)
        b += card(596, rows[1], 290, 60, "Repair context", "retrieval or assembly", GREEN_L, GREEN)
        b += g(arrow(300, 148, 300, 166, "r", RED)
               + card(150, rows[2], 300, 60, "Schema probe", "corrected tool description", GRAY_L, LINE)
               + text(522, rows[2] + 22, "disappears", 18, GRAY) + arrow(450, rows[2] + 30, 592, rows[2] + 30, "a", AMBER)
               + card(596, rows[2], 290, 60, "Repair tool or schema", "schema, docs, adapter", AMBER_L, AMBER)
               + arrow(300, 228, 300, 246, "r", RED)
               + card(150, rows[3], 300, 60, "Runtime probe", "clean sandbox, raised limits", GRAY_L, LINE)
               + text(522, rows[3] + 22, "disappears", 18, GRAY) + arrow(450, rows[3] + 30, 592, rows[3] + 30, "p", PURPLE)
               + card(596, rows[3], 290, 60, "Repair runtime", "bounds, sandbox pool", PURPLE_L, PURPLE), o2)
        b += g(f'<line x1="120" y1="320" x2="890" y2="320" stroke="{LINE}" stroke-width="2" stroke-dasharray="8 6"/>'
               + arrow(300, 308, 300, 350, "r", RED) + text(312, 340, "survives every probe", 18, GRAY, "start")
               + card(150, 354, 300, 60, "Residual failure", "a capability gap", RED_L, RED)
               + text(522, 376, "on evidence", 18, GRAY) + arrow(450, 384, 592, 384, "b", BLUE)
               + card(596, 348, 290, 36, "Supervised fine-tuning", None, BLUE_L, BLUE, ts=19)
               + card(596, 390, 290, 36, "Reinforcement learning", None, BLUE_L, BLUE, ts=19), o3)
        svg(f"ladder_step{step}.svg", W, H, b)


# ---------------------------------------------------------------- artifacts (§15.2.3)
def artifacts():
    """Drawn 640 x 380 for a column slot about 9.5 cm wide."""
    W, H = 640, 380
    rows = [("Imitation material", "mistake as context;", "only the fix as target", GREEN, GREEN_L),
            ("Preference comparison", "same start, tools, state;", "a recorded reason", AMBER, AMBER_L),
            ("Rollout fixture", "resettable environment", "and a grader", PURPLE, PURPLE_L)]
    for step in (1, 2, 3):
        b = card(6, 140, 190, 96, "One recorded run", "trace_r19_01", GRAY_L, LINE, ts=20, ss=18)
        b += text(101, 264, "outcome: accepted", 18, GRAY)
        for i, (t, l1, l2, s, f) in enumerate(rows):
            y = 6 + i * 126
            o = 1 if i < step else 0
            b += g(arrow(196, 188, 262, y + 56, "g", GRAY)
                   + box(266, y, 368, 114, f, s)
                   + text(282, y + 34, t, 22, s, "start", "bold")
                   + text(282, y + 66, l1, 20, INK, "start")
                   + text(282, y + 96, l2, 20, INK, "start"), o)
        svg(f"artifacts_step{step}.svg", W, H, b)


# ---------------------------------------------------------------- splits (§15.4)
def straddle():
    k_max = int(num("sp_k"))
    ks = list(range(1, k_max + 1))
    vals = [100 * calcs.straddle(k) for k in ks]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.7))
        shown = ks if step == 2 else ks[:2]
        ax.plot(shown, vals[:len(shown)], color=PURPLE, lw=3, marker="o", ms=6)
        if step == 1:
            ax.text(k_max, 50, f"{val('sp_k')} attempts:\n?", fontsize=15, color=PURPLE, ha="center", fontweight="bold")
        else:
            ax.plot([k_max], [vals[-1]], "o", color=RED, ms=11, zorder=5)
            ax.annotate(f"{val('sp_straddle')}% of tasks\non both sides", (k_max, vals[-1]), xytext=(5.0, 20),
                        fontsize=14, color=RED, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=RED, lw=1.2))
        ax.set_xlim(0.6, 8.6)
        ax.set_ylim(0, 100)
        ax.set_xticks(ks)
        ax.set_xlabel(f"Attempts per task (random {val('sp_f_pct')}/{val('sp_q_pct')} split of traces)", fontsize=13)
        ax.set_ylabel("Tasks straddling (%)", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
        sp.save(fig, HERE / f"straddle_step{step}.svg")


def corpus_runs():
    """Runs per issue in the public corpus, and what each split does."""
    hist = {int(k): v for k, v in STATS["runs_per_task_hist"].items()}
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    xs = sorted(hist)
    ax.bar(xs, [hist[x] for x in xs], color=BLUE, width=0.85)
    ax.set_xlabel(f"Runs per issue (last bar: {val('c_cap_bar')} or more)", fontsize=13)
    ax.set_ylabel("Issues", fontsize=13)
    ax.tick_params(labelsize=12)
    med = STATS["runs_per_task"]["median"]
    ax.axvline(med, color=INK, lw=1.5, ls="--")
    ax.text(med + 0.8, max(hist.values()) * 0.92, f"median {med}", fontsize=13, color=INK)
    ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
    sp.save(fig, HERE / "corpus_runs.svg")

    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    gs = STATS["group_split"]
    rows = [("random split of runs", 100 * STATS["straddle"]["measured"], val("c_straddle_pct"), RED,
             "issues on both sides"),
            ("split by issue", 0.0, f"{0.0:.1f}", GREEN, "issues on both sides"),
            ("same split, by repo", 100 * gs["test_repo_shared"] / gs["test_tasks"], val("c_test_repo_pct"), AMBER,
             "test issues whose repo is in training")]
    for i, (lab, v, shown, c, what) in enumerate(rows):
        y = 2 - i
        ax.barh(y, max(v, 0.4), color=c, height=0.62)
        ax.text(-2, y, lab, ha="right", va="center", fontsize=13, fontweight="bold", color=INK)
        ax.text(max(v, 0.4) + 1.5, y, f"{shown}%  {what}", va="center", fontsize=12.5, color=c,
                fontweight="bold")
    ax.set_xlim(0, 175)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.tick_params(labelsize=12)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Percent of {val('c_tasks')} issues (or of {val('c_test_tasks')} test issues)", fontsize=12)
    sp.save(fig, HERE / "corpus_split.svg")


def groups():
    """Drawn 660 x 400 for a column slot about 9 cm wide."""
    W, H = 660, 400
    for step in (1, 2):
        b = text(180, 26, "task group g = pool_repair_01", 20, PURPLE, weight="bold")
        b += box(6, 40, 350, 352, PURPLE_L, PURPLE, rx=12)
        items = [("attempts 1 to 8", "same fixture, new seeds"),
                 ("paraphrased prompt", "same problem, new words"),
                 ("fork with the same bug", "found by similarity screen"),
                 ("trace_r19_01_redacted_v1", "derived record")]
        for i, (t, s) in enumerate(items):
            y = 54 + i * 84
            b += card(20, y, 322, 72, t, s, "white", PURPLE, ts=19, ss=18)
        b += arrow(358, 216, 420, 216, "p", PURPLE)
        b += text(389, 202, "s(g)", 20, PURPLE, weight="bold")
        b += g(card(424, 92, 230, 92, "training", "all of g", GREEN_L, GREEN)
               + card(424, 252, 230, 92, "final test", "none of g", "#f1f5f9", LINE, dash=True), 1 if step == 2 else 0)
        svg(f"groups_step{step}.svg", W, H, b)


def lsh():
    import numpy as np
    js = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.plot(js, [calcs.lsh_p(j) for j in js], color=PURPLE, lw=3)
    for jk, pk, tx, ty in (("mh_j_lo", "mh_p_lo", 0.56, 0.62), ("mh_j_hi", "mh_p_hi", 0.66, 0.80)):
        j = num(jk)
        ax.plot([j], [calcs.lsh_p(j)], "o", color=RED, ms=9, zorder=5)
        ax.text(tx, ty, f"J = {val(jk)}: {val(pk)}%", fontsize=13, color=RED, fontweight="bold")
    ax.set_xlabel("Jaccard similarity of two records", fontsize=13)
    ax.set_ylabel("Sent to review", fontsize=13)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "50%", "100%"])
    ax.tick_params(labelsize=12)
    ax.text(0.02, 0.9, f"{val('mh_bands')} bands of {val('mh_rows')} rows", fontsize=13, color=PURPLE)
    ax.grid(color=sp.C["grid"], lw=0.6)
    sp.save(fig, HERE / "lsh.svg")


# ---------------------------------------------------------------- admission (§15.5)
def cascade():
    """Drawn 900 x 390 for a full-width slot about 12 cm wide."""
    W, H = 900, 390
    heads = [("R", "record"), ("U", "use"), ("P", "process"), ("O", "outcome"), ("T", "target")]
    rows = [("record A", ["pass"] * 5, ("Admitted", "this route", PURPLE_L, PURPLE)),
            ("record B", ["pass", "pass", "pass", "fail", "-"], ("Rejected", "record kept", RED_L, RED)),
            ("record C", ["pass", "pending", "-", "-", "-"], ("Held", "evidence due", "#f1f5f9", GRAY))]
    for step in (1, 2, 3):
        b = ""
        for i, (L, sub) in enumerate(heads):
            x = 160 + i * 106
            b += text(x + 48, 26, L, 24, INK, weight="bold") + text(x + 48, 50, sub, 18, GRAY)
        for r, (name, cells, end) in enumerate(rows):
            y = 66 + r * 100
            o = 1 if r < step else 0
            body = card(4, y, 140, 66, name, None, PURPLE_L, PURPLE, ts=20)
            for i, c in enumerate(cells):
                x = 160 + i * 106
                if c == "-":
                    body += box(x, y, 96, 66, "white", "#cbd5e1", dash=True, sw=2)
                    body += text(x + 48, y + 40, "not run", 18, GRAY)
                else:
                    fill, st = {"pass": ("white", LINE), "fail": (RED_L, RED),
                                "pending": ("#f1f5f9", GRAY)}[c]
                    body += box(x, y, 96, 66, fill, st, dash=(c == "pending"))
                    body += text(x + 48, y + 40, c, 19, RED if c == "fail" else INK,
                                 weight="bold" if c != "pass" else "normal")
            body += arrow(690, y + 33, 708, y + 33, "g")
            body += card(712, y, 184, 66, end[0], end[1], end[2], end[3], ts=20, ss=18)
            b += g(body, o)
        b += g(text(450, 380, "a missing check never counts as a pass", 21, INK, weight="bold"),
               1 if step == 3 else 0)
        svg(f"cascade_step{step}.svg", W, H, b)


def order():
    t_best, t_worst = (calcs.AD.time_per_attempted[i].to("second").magnitude
                       for i in (calcs.AD.best_order, 1 - calcs.AD.best_order))
    rows = [("process audit first", t_best, "data", val("ad_t_best")),
            ("tests first", t_worst, "danger", val("ad_t_worst"))]
    sp.hbars(str(HERE / "order.svg"), [(a, v, c, f"{shown} s") for a, v, c, shown in rows], "s", log=False,
             xlabel="Slot time per attempted record (s)", size=(3.6, 1.1))


# ---------------------------------------------------------------- recovery (§15.6)
def typology():
    """Drawn 900 x 420 for a full-width slot about 12 cm wide."""
    W, H = 900, 420
    rows = [("Direct-success demonstration",
             [("task and", "context", "c"), ("inspect the", "failing check", "t"), ("edit the", "pool wakeup", "t"),
              ("required", "checks pass", "c"), ("report the", "result", "t")]),
            ("Recovery demonstration",
             [("earlier", "context", "c"), ("e2 malformed", "edit, refused", "x"), ("e3 inspect", "the schema", "t"),
              ("e4 corrected", "edit", "t"), ("e5 check", "summary", "c"), ("e6 report", "and scope", "t")]),
            ("Hard-negative candidate",
             [("task and", "context", "c"), ("plausible", "edit", "x"), ("targeted", "check passes", "c"),
              ("required", "check fails", "c")])]
    for step in (1, 2, 3):
        b = ""
        for r, (title, cells) in enumerate(rows):
            y = 4 + r * 126
            body = text(6, y + 20, title, 21, INK, "start", "bold")
            for i, (l1, l2, kind) in enumerate(cells):
                x = 6 + i * 148
                fill, st, dash = {"c": (GRAY_L, LINE, False), "t": (BLUE_L, BLUE, False),
                                  "x": (RED_L, RED, True)}[kind]
                body += box(x, y + 30, 140, 66, fill, st, dash=dash)
                body += text(x + 70, y + 58, l1, 18) + text(x + 70, y + 82, l2, 18)
                bar = BLUE if kind == "t" else "#e2e8f0"
                body += f'<rect x="{x}" y="{y + 102}" width="140" height="9" fill="{bar}"/>'
            b += g(body, 1 if r < step else 0)
        b += (box(6, 394, 26, 18, BLUE_L, BLUE, rx=3) + text(40, 409, "selected target", 18, INK, "start")
              + box(230, 394, 26, 18, GRAY_L, LINE, rx=3) + text(264, 409, "context only", 18, INK, "start")
              + box(430, 394, 26, 18, RED_L, RED, rx=3, dash=True) + text(464, 409, "never imitated", 18, INK, "start")
              + f'<rect x="650" y="398" width="26" height="9" fill="{BLUE}"/>' + text(684, 409, "in the loss", 18, INK, "start"))
        svg(f"typology_step{step}.svg", W, H, b)


def denominators():
    """NM 15.3 (illustrative) beside the same split measured in the corpus."""
    dn = calcs.DN
    ill = [100 * dn.record_share, 100 * dn.stored_share, 100 * dn.target_share]
    rc = STATS["recovery"]
    mea = [100 * rc["recovery"] / (rc["recovery"] + rc["direct"]),
           100 * rc["recovery_chars"] / (rc["recovery_chars"] + rc["direct_chars"]),
           100 * rc["recovery_ai"] / (rc["recovery_ai"] + rc["direct_ai"])]
    for name, vals, labs, color, title in (
            ("denom_nm", ill, ["records", "stored tokens", "selected targets"], PURPLE,
             "NM 15.3 (illustrative)"),
            ("denom_corpus", mea, ["resolved runs", "characters", "model-written chars"], BLUE,
             "Corpus: runs with a refused edit")):
        fig, ax = plt.subplots(figsize=(3.6, 2.5))
        for i, (v, lab) in enumerate(zip(vals, labs)):
            y = (2 - i) * 1.5
            ax.barh(y, v, color=color, height=0.55)
            ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=14, color=color, fontweight="bold")
            ax.text(1, y + 0.36, lab, fontsize=12.5, color=INK, va="bottom")
        ax.set_xlim(0, 62)
        ax.set_ylim(-0.5, 3.95)
        ax.set_yticks([])
        ax.set_xticks([0, 25, 50])
        ax.tick_params(labelsize=12)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=13, color=color, loc="left")
        sp.save(fig, HERE / f"{name}.svg")


def plateau():
    import numpy as np
    ps = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ax.plot(ps, [calcs.yield_any(p) for p in ps], color=BLUE, lw=2)
    ax.plot(ps, [calcs.mixed(p) for p in ps], color=PURPLE, lw=3.5)
    ax.text(0.30, 1.03, "a success", fontsize=13, color=BLUE)
    ax.text(0.33, 0.62, "mixed group", fontsize=13, color=PURPLE, fontweight="bold")
    p_lo, _, p_hi = calcs.GC_S.pass_rates
    for p in (p_lo, p_hi):
        ax.plot([p], [calcs.mixed(p)], "o", color=RED, ms=8, zorder=5)
    ax.text(0.09, 0.22, val("gc_mixed_lo"), fontsize=13, color=RED, fontweight="bold")
    ax.text(0.73, 0.22, val("gc_mixed_hi"), fontsize=13, color=RED, fontweight="bold")
    ax.set_xlabel(f"Per-attempt pass rate ({val('gc_k')} attempts)", fontsize=13)
    ax.set_ylabel("Probability", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=12)
    ax.grid(color=sp.C["grid"], lw=0.6)
    sp.save(fig, HERE / "plateau.svg")

    mix = STATS["outcome_mix"]
    n = sum(mix.values())
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    rows = [("all runs fail", mix["all_fail"], RED), ("mixed", mix["mixed"], PURPLE),
            ("all runs pass", mix["all_pass"], GREEN)]
    for i, (lab, v, c) in enumerate(rows):
        y = (2 - i) * 1.5
        ax.barh(y, 100 * v / n, color=c, height=0.55)
        ax.text(100 * v / n + 2, y, f"{100 * v / n:.1f}%", va="center", fontsize=14, color=c, fontweight="bold")
        ax.text(1, y + 0.36, lab, fontsize=12.5, color=INK, va="bottom")
    ax.set_xlim(0, 115)
    ax.set_ylim(-0.5, 3.95)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.tick_params(labelsize=12)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Share of {n:,} issues", fontsize=12)
    sp.save(fig, HERE / "outcome_mix.svg")


# ---------------------------------------------------------------- collection (§15.8)
def staging():
    store = num("st_store_gb")
    t_full = calcs.ST.time_to_full.to("second").magnitude
    wm = calcs.ST.high_watermark * store
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.6))
        ax.plot([0, t_full], [0, store], color=RED, lw=3)
        ax.axhline(store, color=INK, lw=1.5)
        ax.text(5, store + 1.0, f"store full: {val('st_store_gb')} GB", fontsize=13, color=INK)
        ax.plot([t_full], [store], "o", color=RED, ms=9)
        ax.text(t_full + 6, store + 1.0, f"{val('st_tfull')} s", fontsize=14, color=RED,
                fontweight="bold", ha="left")
        ax.text(110, 6, f"verifiers stall: +{val('st_growth_r')} MB/s", fontsize=13, color=RED)
        if step == 2:
            ax.axhspan(wm, store, color=AMBER_L)
            ax.axhline(wm, color=AMBER, lw=2, ls="--")
            ax.text(4, wm + 1.0, f"stop at {val('st_mark_gb')} GB ({val('st_mark_pct')}%)",
                    fontsize=12.5, color=AMBER, fontweight="bold")
            ax.text(222, wm + 6.2, f"{val('st_inflight')} GB in flight", fontsize=12, color=AMBER)
            ax.text(222, wm + 2.2, f"+{val('st_react_gb')} GB in {val('st_react_s')} s", fontsize=12, color=AMBER)
        ax.set_xlim(0, 310)
        ax.set_ylim(0, 37)
        ax.set_xlabel("Seconds after the fixture breaks (illustrative, NM 15.4)", fontsize=12)
        ax.set_ylabel("Reserved staging (GB)", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.grid(color=sp.C["grid"], lw=0.6)
        sp.save(fig, HERE / f"staging_step{step}.svg")


def lineage():
    """Drawn 900 x 400 for a full-width slot about 11 cm wide."""
    W, H = 900, 400
    src = ["collection identity", "generation config", "environment, capture", "assessment",
           "use and partition", "reverse references"]
    for step in (1, 2):
        b = box(4, 4, 280, 352, "white", PURPLE, rx=10)
        b += text(144, 32, "Source record", 21, INK, weight="bold")
        b += text(144, 58, "trace_r19_01", 19, PURPLE, family="Courier, monospace")
        for i, s in enumerate(src):
            hl = s == "reverse references"
            b += box(18, 72 + i * 46, 252, 38, PURPLE_L if hl else GRAY_L, PURPLE if hl else LINE, rx=5, sw=2)
            b += text(144, 98 + i * 46, s, 19)
        b += card(306, 150, 184, 72, "Redaction", "versioned policy", GRAY_L, LINE, ts=20, ss=18)
        b += arrow(270, 186, 302, 186) + arrow(490, 186, 520, 186)
        der = [("parent reference", PURPLE_L, PURPLE), ("changed fields", GRAY_L, LINE),
               ("same split group", GREEN_L, GREEN), ("same use permission", GREEN_L, GREEN),
               ("assessment: pending", "#f1f5f9", GRAY)]
        body = box(524, 4, 372, 300, "white", PURPLE, rx=10)
        body += text(710, 32, "Derived record", 21, INK, weight="bold")
        body += text(710, 58, "trace_r19_01_redacted_v1", 19, PURPLE, family="Courier, monospace")
        for i, (s, f, st) in enumerate(der):
            body += box(540, 72 + i * 45, 340, 37, f, st, rx=5, sw=2, dash=(i == 4))
            body += text(710, 97 + i * 45, s, 19)
        body += arrow(540, 91, 274, 91, "p", PURPLE)
        b += body
        b += g(f'<path d="M284,330 L710,330 L710,308" fill="none" stroke="{RED}" stroke-width="2.5" '
               f'stroke-dasharray="7 5" marker-end="url(#ar)"/>'
               + text(560, 388, "withdraw the source: every descendant is found", 20, RED, weight="bold"),
               1 if step == 2 else 0)
        svg(f"lineage_step{step}.svg", W, H, b)


def dispositions():
    from matplotlib.ticker import FixedLocator, FixedFormatter
    rows = [("selected, emitted", "co_emitted", GREEN), ("assessed, not selected", "co_other", GRAY),
            ("ineligible for this use", "co_inelig", RED), ("pending at cutoff", "co_pending", PURPLE),
            ("unknown at cutoff", "co_unknown", PURPLE)]
    fig, ax = plt.subplots(figsize=(3.9, 2.4))
    for i, (lab, key, c) in enumerate(rows):
        y = 4 - i
        v = num(key)
        ax.barh(y, v, color=c, height=0.6)
        ax.text(v * 1.15, y, val(key), va="center", fontsize=12, color=c, fontweight="bold")
        ax.text(0.9, y, lab + " ", va="center", ha="right", fontsize=11.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, 2000)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1", "10", "100", "1,000"]))
    ax.minorticks_off()
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Attempts at the cutoff (log scale)", fontsize=11)
    sp.save(fig, HERE / "dispositions.svg")


def synthesis():
    """Drawn 900 x 200 for a full-width strip about 13 cm wide."""
    W, H = 900, 200
    steps = [(val("tri_fail"), "failures", RED_L, RED), (val("tri_system"), "fixed below", GREEN_L, GREEN),
             (val("tri_residual"), "residual", BLUE_L, BLUE), ("split", "group fixed", PURPLE_L, PURPLE),
             ("R U P O T", "all pass", PURPLE_L, PURPLE), ("e3 e4 e6", "targets", BLUE_L, BLUE),
             ("redacted", "re-admitted", PURPLE_L, PURPLE)]
    b = ""
    for i, (t, s, f, st) in enumerate(steps):
        x = 4 + i * 128
        b += card(x, 40, 118, 92, t, s, f, st, ts=20, ss=18)
        if i < len(steps) - 1:
            b += arrow(x + 118, 86, x + 128, 86)
    b += text(450, 26, "what reaches the weights, and with what evidence", 21, INK, weight="bold")
    b += text(450, 172, "published as repair_demonstrations_v1 with its manifest", 20, PURPLE, weight="bold")
    svg("synthesis.svg", W, H, b)


def recall_thumb():
    """Text-free: one saved prefix and three continuation branches (Lecture 14's postmortem)."""
    b = f'<line x1="20" y1="110" x2="260" y2="110" stroke="{GRAY}" stroke-width="10" stroke-linecap="round"/>'
    b += f'<circle cx="260" cy="110" r="16" fill="{PURPLE}"/>'
    for y, c in ((40, GREEN), (110, GREEN), (180, RED)):
        b += f'<path d="M276,110 C330,110 340,{y} 400,{y} L560,{y}" fill="none" stroke="{c}" stroke-width="9" stroke-linecap="round"/>'
        b += f'<circle cx="580" cy="{y}" r="16" fill="{c}"/>'
    svg("recall_branches.svg", 620, 220, b)


if __name__ == "__main__":
    recall_thumb()
    hook()
    ladder()
    artifacts()
    straddle()
    corpus_runs()
    groups()
    lsh()
    cascade()
    order()
    typology()
    denominators()
    plateau()
    staging()
    lineage()
    dispositions()
    synthesis()
