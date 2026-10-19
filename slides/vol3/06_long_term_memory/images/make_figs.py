#!/usr/bin/env python3
"""Figures for Lecture 6 (Volume III), Long-Term Memory.

Every value comes from calcs.py, which names its source: "LTM" is the
rendered chapter (Long-Term Memory, dev text read 2026-10-18); outside data
are Jimenez et al. 2024 (SWE-bench), Yang et al. 2024 (SWE-agent), Zou et al.
2025 (PoisonedRAG), Liu et al. 2024 (Lost in the Middle), and this deck's own
measurement (scan_bench.py, scan_bench.csv).

Diagrams are hand-written SVG in the Volume III palette (volumes/vol3/
palette.md): green memory and context, violet storage, amber generated or
tool-side records, red faults, slate structure. Full-width diagrams use a
900 x 500 canvas with text of 20 px or more; column diagrams 540 wide.
Plots use the shared slideplot helper.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

import calcs as K  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent

# Volume III palette roles (fill, stroke, text)
ROLE = {
    "memory": ("#dcfce7", "#16a34a", "#15803d"),
    "storage": ("#f3e8ff", "#7c3aed", "#6d28d9"),
    "tools": ("#fef3c7", "#d97706", "#b45309"),
    "compute": ("#e0f2fe", "#0284c7", "#0369a1"),
    "fault": ("#fee2e2", "#dc2626", "#b91c1c"),
    "struct": ("#f8fafc", "#94a3b8", "#0f172a"),
    "dim": ("#f1f5f9", "#cbd5e1", "#94a3b8"),
}
FONT = "Helvetica, Arial, sans-serif"
MONO = "Courier New, Courier, monospace"


# ------------------------------------------------------------ SVG helpers
class Svg:
    def __init__(self, w=900, h=500):
        self.w, self.h, self.parts = w, h, []
        self.parts.append(
            '<defs>' + "".join(
                f'<marker id="a-{k}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{v[1]}"/></marker>'
                for k, v in ROLE.items()) + '</defs>')

    def rect(self, x, y, w, h, role="struct", dash=False, sw=2, rx=8):
        f, s, _ = ROLE[role]
        d = ' stroke-dasharray="7,5"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                          f'fill="{f}" stroke="{s}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=20, role="struct", bold=False, anchor="start",
             mono=False, color=None):
        c = color or ROLE[role][2]
        fam = MONO if mono else FONT
        wt = ' font-weight="bold"' if bold else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
                          f'fill="{c}" text-anchor="{anchor}"{wt}>{s}</text>')

    def lines(self, x, y, rows, size=20, role="struct", gap=None, **kw):
        gap = gap or int(size * 1.25)
        for i, r in enumerate(rows):
            self.text(x, y + i * gap, r, size=size, role=role, **kw)

    def arrow(self, x1, y1, x2, y2, role="struct", sw=2.5, dash=False):
        s = ROLE[role][1]
        d = ' stroke-dasharray="6,5"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{s}" '
                          f'stroke-width="{sw}" marker-end="url(#a-{role})"{d}/>')

    def save(self, path):
        body = "\n".join(self.parts)
        Path(path).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}">\n'
            f'<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')


# ------------------------------------------------------- record roles build
ROLES = [
    ("Source record", "storage", ["src/pool.py at r18;", "the user's stated preference"],
     ["source, version,", "access conditions"]),
    ("Observation log", "memory", ["waiter test failed at r17,", "passed at r18 (config c18)"],
     ["what, when, under", "which conditions"]),
    ("Agent-written", "tools", ["note: resume at the", "integration checks"],
     ["supporting records,", "scope, uncertainty"]),
    ("Derived index", "compute", ["search entries over files,", "logs, and notes"],
     ["the source generation", "it was built from"]),
]


def roles_fig(path, k):
    s = Svg(900, 395)
    s.text(18, 26, "Role", size=21, bold=True, color="#475569")
    s.text(283, 26, "Repair-agent example", size=21, bold=True, color="#475569")
    s.text(643, 26, "A later read must keep", size=21, bold=True, color="#475569")
    for i, (name, role, ex, keep) in enumerate(ROLES):
        if i >= k:
            continue
        y = 40 + i * 89
        s.rect(12, y, 250, 74, role, sw=2.5)
        if name == "Agent-written":
            s.text(26, y + 31, name, size=23, role=role, bold=True)
            s.text(26, y + 60, "memory", size=23, role=role, bold=True)
        else:
            s.text(26, y + 45, name, size=23, role=role, bold=True)
        s.rect(275, y, 352, 74, "struct")
        s.lines(288, y + 31, ex, size=21, gap=27)
        s.rect(638, y, 252, 74, "struct")
        s.lines(651, y + 31, keep, size=21, gap=27)
        if i > 0:
            s.arrow(137, y - 1, 137, y - 14, role="struct", sw=2.5)
    s.save(path)


for k in range(1, 5):
    roles_fig(out / f"roles_step{k}.svg", k)


# --------------------------------------------------- retrieval contract build
def contract_fig(path, k):
    s = Svg(900, 395)
    s.rect(8, 8, 285, 315, "storage", sw=2.5)
    s.text(150, 40, "Runtime sets bounds", size=23, role="storage", bold=True, anchor="middle")
    s.lines(24, 80, ["need: notification logic", "scope: repair workspace",
                     "principal: repair agent", "version: r18",
                     "limits: items, tokens", "deadline, work caps"], size=21, gap=43)
    if k >= 2:
        s.arrow(295, 165, 320, 165, role="struct")
        s.rect(322, 8, 250, 315, "compute", sw=2.5)
        s.text(447, 40, "Service searches", size=23, role="compute", bold=True, anchor="middle")
        s.lines(338, 80, ["checks access", "picks an index", "stops at a limit",
                          "cancels on timeout"], size=21, gap=43)
    if k >= 3:
        s.arrow(574, 165, 599, 165, role="struct")
        s.rect(601, 8, 291, 315, "memory", sw=2.5)
        s.text(746, 40, "Envelope reports", size=23, role="memory", bold=True, anchor="middle")
        s.lines(615, 80, ["status: complete,", "  partial, unavailable", "searched: repair, r18",
                          "each item: path, lines,", "  revision, hash, role", "score: method named"],
                size=21, gap=43)
        s.rect(8, 335, 884, 54, "fault", sw=2)
        s.text(450, 370, "Zero hits prove absence only for a complete scan of the declared scope.",
               size=22, role="fault", bold=True, anchor="middle")
    s.save(path)


for k in range(1, 4):
    contract_fig(out / f"contract_step{k}.svg", k)


# ----------------------------------------------- NM 6.1 records that fit
def fit_fig(path, k):
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    a = K.allowance
    # row 1 (top): count bodies only
    y1, y0 = 1, 0
    ax.barh(y1, K.body_only * K.BODY, color=C["memory"], height=0.55)
    if k >= 2:
        ax.barh(y1, K.body_only * K.META + K.ENVELOPE, left=K.body_only * K.BODY,
                color=C["danger"], height=0.55)
        ax.text(K.body_only_tokens + 80, y1, f"{K.body_only_tokens:,}: over by {K.overshoot:,}",
                va="center", fontsize=12, fontweight="bold", color=C["danger"])
        ax.barh(y0, K.fit * K.BODY, color=C["memory"], height=0.55)
        ax.barh(y0, K.fit * K.META + K.ENVELOPE, left=K.fit * K.BODY, color=C["network"],
                height=0.55)
        ax.text(K.allowance + 80, y0, f"{K.used:,}: fits, {K.remainder} left",
                va="center", fontsize=12, fontweight="bold", color=C["data"])
    else:
        ax.text(K.body_only * K.BODY + 80, y1, "???", va="center", fontsize=14,
                fontweight="bold", color=C["danger"])
    ax.text(-60, y1, f"Count bodies:\n{K.body_only} records", ha="right", va="center",
            fontsize=12, fontweight="bold")
    ax.text(-60, y0, f"Count serialized:\n{K.fit} records", ha="right", va="center",
            fontsize=12, fontweight="bold", alpha=1.0 if k >= 2 else 0.0)
    ax.axvline(a, color=C["text"], lw=1.6, ls="--")
    ax.text(a, 1.5, f" allowance {a:,}", fontsize=11, fontweight="bold", va="bottom",
            ha="center")
    ax.set_xlim(0, 10_200)
    ax.set_ylim(-0.5, 1.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Serialized tokens (bodies blue; metadata and envelope red or violet)")
    sp.save(fig, path)


for k in (1, 2):
    fit_fig(out / f"fit_step{k}.svg", k)


# ---------------------------------------------- SWE-bench: context vs result
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.5))
for ax, ys, ck, lab in ((a1, K.SWEB_RECALL, "memory", "BM25 recall of edited files (%)"),
                        (a2, K.SWEB_RESOLVED, "danger", "Issues resolved (%)")):
    ax.plot(K.SWEB_CONTEXT, ys, "-o", color=C[ck], lw=2.5, ms=8)
    for x, y in zip(K.SWEB_CONTEXT, ys):
        below = ys is K.SWEB_RECALL and x == K.SWEB_CONTEXT[0]
        ax.annotate(f"{y:.2f}", (x, y), xytext=(18, -12) if below else (0, 9),
                    textcoords="offset points",
                    ha="center", fontsize=12, fontweight="bold", color=C[ck])
    ax.set_xticks(K.SWEB_CONTEXT)
    ax.set_xticklabels([f"{c}K" for c in K.SWEB_CONTEXT])
    ax.set_xlim(5, 58)
    ax.set_ylim(min(ys) - (max(ys) - min(ys)) * 0.3, max(ys) + (max(ys) - min(ys)) * 0.35)
    ax.set_title(lab, fontsize=12, fontweight="bold", color=C[ck])
    ax.set_xlabel("Retrieved context, tokens")
fig.tight_layout()
sp.save(fig, out / "swebench_context.svg")

# ------------------------------------------------- SWE-agent vs pre-staging
sp.hbars(out / "swe_agent.svg", [
    ("Pre-staged\n(BM25)", K.LITE_RAG, "neutral", f"{K.LITE_RAG:.2f}%"),
    ("Shell\ncommands", K.LITE_SHELL, "memory", f"{K.LITE_SHELL:.2f}%"),
    ("Search tools\n(SWE-agent)", K.LITE_AGENT, "data", f"{K.LITE_AGENT:.2f}%"),
], "", log=False, size=(3.8, 2.3),
    xlabel="SWE-bench Lite resolved (%), same model")


# ------------------------------------------------------- inverted index
def inverted_fig(path):
    s = Svg(540, 345)
    s.text(10, 30, "term", size=20, bold=True, color="#475569")
    s.text(155, 30, "postings: file, count", size=20, bold=True, color="#475569")
    rows = [("notify", [("src/pool.py", 2), ("r17.log", 1)]),
            ("waiter", [("test_pool.py", 3), ("src/pool.py", 1)]),
            ("return", [("src/pool.py", 4), ("pool.md", 1)])]
    for i, (term, posts) in enumerate(rows):
        y = 48 + i * 82
        s.rect(10, y, 115, 56, "compute")
        s.text(67, y + 36, term, size=21, role="compute", anchor="middle", mono=True, bold=True)
        s.arrow(127, y + 28, 150, y + 28)
        x = 152
        for name, cnt in posts:
            red = name == "r17.log"
            s.rect(x, y, 186, 56, "fault" if red else "memory")
            s.text(x + 93, y + 36, f"{name}  {cnt}", size=20,
                   role="fault" if red else "memory", anchor="middle")
            x += 194
    s.text(270, 330, "A match is an occurrence, not a role.", size=21, role="fault",
           bold=True, anchor="middle")
    s.save(path)


inverted_fig(out / "inverted_index.svg")


# ------------------------------------------------------ exact-scan floor
def scan_floor(path, k):
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    xs = np.logspace(3, 8, 100)
    ax.plot(xs, xs * K.DIM * K.B_FP16 / K.BW * 1e3, color=C["memory"], lw=2.5)
    ax.plot([K.N_VEC], [K.t_stream * 1e3], "o", ms=11, color=C["danger"], mec="white")
    ax.annotate(f"10 million: {K.t_stream * 1e3:.1f} ms", (K.N_VEC, K.t_stream * 1e3),
                xytext=(-12, 4), textcoords="offset points", ha="right", fontsize=12,
                fontweight="bold", color=C["danger"])
    if k >= 2:
        ax.axhline(K.BUDGET_S * 1e3, color=C["neutral"], lw=1.5, ls="--")
        ax.text(1.2e3, K.BUDGET_S * 1e3 * 1.25, "10 ms search budget", fontsize=11,
                color=C["neutral"], fontweight="bold")
        ax.plot([1e4], [K.t_10k * 1e3], "o", ms=11, color=C["data"], mec="white")
        ax.annotate(f"10,000: {K.t_10k * 1e3:.2f} ms", (1e4, K.t_10k * 1e3),
                    xytext=(10, -14), textcoords="offset points", fontsize=12,
                    fontweight="bold", color=C["data"])
        ax.axvline(K.ceiling, color=C["neutral"], lw=1.2, ls=":")
        ax.text(K.ceiling * 1.25, 0.02, f"ceiling about {round(K.ceiling, -3):,.0f}",
                fontsize=11, color=C["text"], fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e3, 1e8)
    ax.set_ylim(0.01, 5000)
    ax.set_xlabel("Eligible vectors (768 dims, FP16)")
    ax.set_ylabel("Scan floor (ms)")
    ax.text(0.01, 0.97, "derived, 50 GB/s host", transform=ax.transAxes, ha="left",
            va="top", fontsize=10, color=C["neutral"])
    sp.save(fig, path)


for k in (1, 2):
    scan_floor(out / f"scan_floor_step{k}.svg", k)


# -------------------------------------------- measured scan on one laptop
def scan_measured(path):
    import numpy as np
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ns = [r[0] for r in K.rows]
    ts = [r[2] * 1e3 for r in K.rows]
    xs = np.logspace(3.8, 6.7, 50)
    ax.plot(xs, xs * K.DIM * K.B_FP32 / K.bw_fit * 1e3, color=C["memory"], lw=2,
            label="fit")
    ax.plot(ns, ts, "o", ms=9, color=C["compute"], mec="white", zorder=5)
    ax.text(1.3e4, 30, f"fit: {K.bw_fit / 1e9:.0f} GB/s\nstreamed", fontsize=13,
            color=C["memory"], fontweight="bold")
    ax.axhline(K.BUDGET_S * 1e3, color=C["neutral"], lw=1.5, ls="--")
    ax.text(8e3, K.BUDGET_S * 1e3 * 1.25, "10 ms budget", fontsize=12, color=C["neutral"],
            fontweight="bold")
    ax.annotate(f"3 million: {ts[-1]:.0f} ms", (ns[-1], ts[-1]), xytext=(-8, 8),
                textcoords="offset points", ha="right", fontsize=13, fontweight="bold",
                color=C["compute"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7e3, 6e6)
    ax.set_ylim(0.1, 300)
    ax.set_xlabel("Vectors (768 dims, FP32)", fontsize=13)
    ax.set_ylabel("Measured scan (ms)", fontsize=13)
    ax.tick_params(labelsize=13)
    sp.save(fig, path)


scan_measured(out / "scan_measured.svg")

# ------------------------------------------------------- ANN resident bytes
sp.hbars(out / "ann_bytes.svg", [
    ("Exact scan", K.scan_bytes / 1e9, "memory", f"{K.scan_bytes / 1e9:.2f} GB of vectors"),
    ("HNSW", K.hnsw_total / 1e9, "network", f"{K.hnsw_total / 1e9:.2f} GB with links"),
    ("IVF-PQ", K.pq_codes / 1e9, "data", f"{K.pq_codes / 1e9:.2f} GB of codes"),
], "", log=False, size=(5.6, 1.9),
    xlabel="Resident bytes, 10 million 768-dim FP16 vectors (derived)")


# --------------------------------------------------- NM 6.3 depth budget
def depth_fig(path, k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    cols = [C["data_l"], C["data_l"], C["data"], C["danger"]]
    for d, (n, t) in enumerate(zip(K.depth_nodes, K.depth_tokens)):
        shown = d < 3 or k >= 2
        ax.bar(d, t, color=cols[d], width=0.62, alpha=1.0 if shown else 0.0)
        if shown:
            ax.text(d, t + 150, f"{t:,}", ha="center", fontsize=13, fontweight="bold",
                    color=C["danger"] if d == 3 else C["text"])
        else:
            ax.text(d, 900, "???", ha="center", fontsize=15, fontweight="bold",
                    color=C["danger"])
        ax.text(d, -1050, f"depth {d}\n{n} node{'' if n == 1 else 's'}", ha="center", va="top", fontsize=12)
    ax.axhline(K.ALLOWANCE, color=C["text"], lw=1.6, ls="--")
    ax.text(-0.45, K.ALLOWANCE + 180, f"allowance {K.ALLOWANCE:,} tokens", fontsize=12,
            fontweight="bold")
    ax.set_xticks([])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 8300)
    ax.set_ylabel("Serialized tokens")
    ax.text(-0.5, 3300, "3 children per node\n180 tokens per node\n(illustrative)",
            ha="left", fontsize=11, color=C["neutral"])
    sp.save(fig, path)


for k in (1, 2):
    depth_fig(out / f"depth_step{k}.svg", k)


# ------------------------------------------------------- write path build
def write_fig(path, k):
    s = Svg(900, 395)
    rows = [
        ("1  Commit the raw", "observation", "memory",
         ["tool output, revision, task id: durable before", "any summary or embedding runs"]),
        ("2  Admission gate", "", "storage",
         ["permission and consent, record kind and scope,", "evidence; a fidelity check for prose"]),
        ("3  Identity, then", "update", "tools",
         ["same scope: supersede if its base version is", "unchanged; other scope: keep both; unclear: flag"]),
        ("4  Only then", "deduplicate", "compute",
         ["similarity may merge storage only when the", "new record changes no state and no scope"]),
    ]
    shown = {1: 1, 2: 2, 3: 4}[k]
    for i, (h1, h2, role, det) in enumerate(rows[:shown]):
        y = 6 + i * 82
        s.rect(10, y, 265, 68, role, sw=2.5)
        s.text(24, y + 29 if h2 else y + 42, h1, size=22, role=role, bold=True)
        if h2:
            s.text(50, y + 57, h2, size=22, role=role, bold=True)
        s.rect(288, y, 602, 68, "struct")
        s.lines(302, y + 29, det, size=21, gap=26)
        if i > 0:
            s.arrow(142, y - 1, 142, y - 13)
    if k >= 3:
        s.rect(10, 334, 880, 57, "fault", sw=2)
        s.text(450, 358, "10:00 to 11:00, same calendar: supersede. Similarity-first", size=21,
               role="fault", bold=True, anchor="middle")
        s.text(450, 383, f"deduplication drops it ({K.pref_shared} of {K.pref_total} words shared).",
               size=21, role="fault", bold=True, anchor="middle")
    s.save(path)


for k in (1, 2, 3):
    write_fig(out / f"write_step{k}.svg", k)


# -------------------------------------------------------- deletion copies
def deletion_fig(path):
    s = Svg(540, 340)
    s.rect(8, 6, 524, 46, "fault", sw=2.5)
    s.text(270, 37, "1  Tombstone: exclude from every read", size=21, role="fault",
           bold=True, anchor="middle")
    s.text(8, 80, "2  Find and purge each derived copy", size=20, bold=True,
           color="#475569")
    copies = ["primary record", "keyword postings", "vector entry", "episode summary",
              "cached prompt prefix", "prompt template"]
    for i, c in enumerate(copies):
        x = 8 + (i % 2) * 266
        y = 92 + (i // 2) * 52
        s.rect(x, y, 258, 44, "memory")
        s.text(x + 129, y + 29, c, size=20, role="memory", anchor="middle")
    s.rect(8, 252, 524, 82, "dim", dash=True)
    s.text(20, 277, "Outside the perimeter; the receipt says so:", size=20, bold=True,
           color="#64748b")
    s.lines(20, 302, ["cold backups under a retention period,",
                      "tokens already sent to a model provider"], size=20, gap=25,
            color="#64748b")
    s.save(path)


deletion_fig(out / "deletion_copies.svg")

# -------------------------------------------------------- PoisonedRAG bars
sp.hbars(out / "poisonedrag.svg", [
    (f"{n}\n{size / 1e6:.1f}M texts", asr * 100, "danger", f"{asr * 100:.0f}%")
    for n, size, asr in K.POISON
], "", log=False, size=(5.4, 2.4),
    xlabel=f"Target questions answered as the attacker chose (%), {K.POISON_N} texts each")


# --------------------------------------------------------- poisoning cascade
def cascade_fig(path, k):
    s = Svg(900, 395)
    chain = [("Observation: reproducer passed at r18 (c18)", "memory"),
             ("Lesson: timeouts always come from a missed notify", "tools"),
             ("Episode summary repeats the lesson", "compute"),
             ("Weekly report cites the summary", "compute"),
             ("Cross-project heuristic", "compute")]
    for i, (label, role) in enumerate(chain):
        y = 6 + i * 78
        r = "fault" if (k >= 2 and i >= 1) else role
        s.rect(10, y, 570, 60, r, sw=2.5)
        s.text(24, y + 38, label, size=22, role=r, bold=(i == 1))
        if i > 0:
            s.arrow(120, y - 1, 120, y - 17, role="fault" if (k >= 2 and i >= 2) else "struct")
    s.text(598, 102, "unsupported generalization", size=20, role="tools", bold=True)
    s.text(598, 190, "retrieved 30 times:", size=20, color="#475569")
    s.text(598, 215, "no new evidence", size=20, color="#475569")
    if k >= 2:
        s.rect(595, 245, 297, 140, "fault", sw=2)
        s.lines(610, 278, ["Invalidate the lesson:", "every record derived", "from it is marked",
                           "suspect or retracted"], size=21, role="fault", gap=27, bold=True)
    s.save(path)


for k in (1, 2):
    cascade_fig(out / f"cascade_step{k}.svg", k)


# --------------------------------------------------- source generations
def generations_fig(path):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    gens = ["r16", "r17", "r18", "r19"]
    rows = [("summary of r16 runs  [r16, r18)", 0, 2, "neutral"),
            ("pool.py chunk, r17 text  [r17, r18)", 1, 2, "neutral"),
            ("pool.py chunk, r18 text  [r18, open)", 2, 3.9, "data")]
    for i, (lab, b, d, ck) in enumerate(rows):
        y = 2 - i
        ax.plot([b, d], [y, y], color=C[ck], lw=9, solid_capstyle="butt")
        ax.text(-0.15, y + 0.3, lab, fontsize=14, color=C[ck], fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1), zorder=5)
    ax.axvline(2.25, color=C["data"], lw=2, ls="--")
    ax.axvline(1.25, color=C["neutral"], lw=1.5, ls=":")
    ax.text(2.32, -0.62, "query at r18:\nadmits one", fontsize=14, color=C["data"],
            fontweight="bold")
    ax.text(1.18, -0.62, "query at r17:\nadmits two", fontsize=14, color=C["neutral"],
            ha="right")
    ax.set_xticks(range(4))
    ax.set_xticklabels(gens, fontsize=15)
    ax.set_yticks([])
    ax.set_xlim(-0.2, 3.9)
    ax.set_ylim(-1.1, 2.6)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Source generation", fontsize=15)
    sp.save(fig, path)


generations_fig(out / "generations.svg")

# -------------------------------------------------- exclusion vs rebuild
# Log ticks are written as times (1 µs, 1 ms, 1 s): a 10 with a superscript
# exponent is too small to read on the slide.
_save = sp.save


def _time_ticks_save(fig, path):
    from matplotlib.ticker import FixedLocator, NullFormatter
    names = {1e-6: "1 µs", 1e-3: "1 ms", 1.0: "1 s"}
    for ax in fig.axes:
        lo, hi = ax.get_xlim()
        ticks = [t for t in sorted(names) if lo <= t <= hi]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([names[t] for t in ticks])
        ax.xaxis.set_minor_locator(FixedLocator([10.0 ** e for e in range(-7, 2)]))
        ax.xaxis.set_minor_formatter(NullFormatter())
    _save(fig, path)


sp.save = _time_ticks_save
sp.hbars(out / "exclusion.svg", [
    ("Exclude 8 entries", K.excl_write, "data", f"{K.excl_write * 1e6:.1f} µs"),
    ("Check 50 candidates", K.excl_query, "data", f"{K.excl_query * 1e6:.1f} µs"),
    ("Re-index, local", K.sync_local, "danger", f"{K.sync_local * 1e3:.0f} ms"),
    ("Re-index, remote batch", K.sync_remote_batch, "danger",
     f"{K.sync_remote_batch * 1e3:.0f} ms"),
    ("Re-index, remote per file", K.sync_remote_serial, "danger",
     f"{K.sync_remote_serial:.2f} s"),
], "", log=True, size=(5.8, 2.7), xlabel="Added work per edit (log; stipulated costs)")
sp.save = _save

# ------------------------------------------------ Liu et al. position curve
fig, ax = plt.subplots(figsize=(5.8, 2.4))
ax.plot(K.LIU_POS, K.LIU_ACC, "-o", color=C["memory"], lw=2.5, ms=9)
for x, y in zip(K.LIU_POS, K.LIU_ACC):
    if x == 5:
        ax.text(x + 0.5, y + 0.8, f"{y:.1f}", ha="left", va="bottom", fontsize=12,
                fontweight="bold", color=C["memory"])
    elif x in (10, 15):
        ax.text(x, y - 1.8, f"{y:.1f}", ha="center", va="top", fontsize=12,
                fontweight="bold", color=C["memory"])
    else:
        ax.text(x, y + 2.2, f"{y:.1f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=C["memory"])
ax.axhline(K.LIU_CLOSED, color=C["danger"], lw=1.5, ls="--")
ax.text(21.3, K.LIU_CLOSED, f"no documents\n{K.LIU_CLOSED}", fontsize=11,
        color=C["danger"], fontweight="bold", va="center", ha="left", clip_on=False)
ax.set_xticks(K.LIU_POS)
ax.set_xlim(0, 21)
ax.set_ylim(45, 85)
ax.set_xlabel("Position of the answer among 20 retrieved documents")
ax.set_ylabel("Accuracy (%)")
sp.save(fig, out / "liu_position.svg")

# --------------------------------------- Checkpoint 6.4 cost per accepted task
sp.hbars_steps(str(out / "ceff"), [
    ("Policy A", K.ceff_a, "neutral", f"{K.ceff_a:.2f}"),
    ("Policy B", K.ceff_b, "data", f"{K.ceff_b:.2f}"),
], "", log=False, size=(4.6, 1.7),
    xlabel="Cost per accepted task (units of A's attempt cost)")

print("figures written to", out)
