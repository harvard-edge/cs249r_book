#!/usr/bin/env python3
"""Generate LaTeX macros from corpus_stats.json.

Pipeline: corpus.json → analyze_corpus.py → corpus_stats.json → THIS → macros.tex
                                                                      → paper.tex includes macros.tex

Run: python3 generate_macros.py
Reads: corpus_stats.json, ../vault/corpus.json
Writes: macros.tex (auto-generated, do not edit)
"""

import json
from pathlib import Path

PAPER_DIR = Path(__file__).parent
STATS_PATH = PAPER_DIR / "corpus_stats.json"
CORPUS_PATH = PAPER_DIR.parent / "vault" / "corpus.json"
OUTPUT_PATH = PAPER_DIR / "macros.tex"


def fmt_num(n):
    """Format number with LaTeX thousands separator."""
    s = f"{int(n):,}"
    return s.replace(",", "{,}")


def main():
    stats = json.load(open(STATS_PATH))
    corpus = json.load(open(CORPUS_PATH))
    pub = [q for q in corpus if q.get("status") == "published"]

    s = stats["summary"]
    val = stats.get("validation", {})

    # Track distribution
    from collections import Counter
    tracks = Counter(q.get("track", "") for q in pub)
    total = len(pub)

    # Validated %
    validated = sum(1 for q in pub if q.get("validated"))
    val_pct = f"{100 * validated / total:.1f}" if total > 0 else "0"

    # Math verified
    math_errors = sum(1 for q in pub if q.get("math_status") == "ERROR")

    lines = [
        "% ═══════════════════════════════════════════════════════════════",
        "% AUTO-GENERATED — do not edit. Run: python3 generate_macros.py",
        "% ═══════════════════════════════════════════════════════════════",
        "",
        f"\\newcommand{{\\numquestions}}{{{fmt_num(s['published'])}}}",
        f"\\newcommand{{\\numpublished}}{{{fmt_num(s['published'])}}}",
        f"\\newcommand{{\\numtracks}}{{{s['tracks']}}}",
        f"\\newcommand{{\\numlevels}}{{{s['levels']}}}",
        f"\\newcommand{{\\numtopics}}{{{s['topics']}}}",
        f"\\newcommand{{\\numareas}}{{{s['areas']}}}",
        f"\\newcommand{{\\numzones}}{{{s['zones']}}}",
        f"\\newcommand{{\\numedges}}{{{stats.get('taxonomy_graph', {}).get('total_edges', 123)}}}",
        f"\\newcommand{{\\numchains}}{{{s['chains_total']}}}",
        f"\\newcommand{{\\numfullchains}}{{{s['chains_full']}}}",
        f"\\newcommand{{\\numinvariants}}{{19}}",
        f"\\newcommand{{\\numvalidated}}{{{val_pct}\\%}}",
        f"\\newcommand{{\\nummatherrors}}{{{math_errors}}}",
        "",
        "% Track distribution",
    ]

    for track in ["cloud", "edge", "mobile", "tinyml", "global"]:
        n = tracks.get(track, 0)
        pct = f"{100 * n / total:.1f}" if total > 0 else "0"
        lines.append(f"\\newcommand{{\\track{track.capitalize()}Count}}{{{fmt_num(n)}}}")
        lines.append(f"\\newcommand{{\\track{track.capitalize()}Pct}}{{{pct}}}")

    # Applicability matrix
    matrix_path = PAPER_DIR.parent / "vault" / "data" / "applicable_cells.json"
    if matrix_path.exists():
        matrix = json.load(open(matrix_path))
        ms = matrix.get("stats", {})
        app = ms.get("applicable_topic_track_pairs", 233)
        exc = ms.get("excluded_topic_track_pairs", 83)
        lines.append(f"\\newcommand{{\\numapplicablepairs}}{{{app}}}")
        lines.append(f"\\newcommand{{\\numexcludedpairs}}{{{exc}}}")
        lines.append(f"\\newcommand{{\\numapplicablecells}}{{{fmt_num(app * s['zones'])}}}")
    else:
        lines.append("\\newcommand{\\numapplicablepairs}{233}")
        lines.append("\\newcommand{\\numexcludedpairs}{83}")
        lines.append("\\newcommand{\\numapplicablecells}{2{,}563}")

    lines.append("")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {OUTPUT_PATH}")
    print(f"  Questions: {s['published']}")
    print(f"  Topics: {s['topics']}, Zones: {s['zones']}")
    print(f"  Validated: {val_pct}%")
    print(f"  Math errors: {math_errors}")


if __name__ == "__main__":
    main()
