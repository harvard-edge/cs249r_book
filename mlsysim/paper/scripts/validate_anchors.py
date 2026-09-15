#!/usr/bin/env python3
"""Check the paper's seven validation anchors against their published values.

The anchor configurations live in ``paper_scenarios.py`` and are the same ones
``generate_paper_values.py`` turns into the paper's macros, so this check and
the paper cannot disagree about what was run.

Each anchor carries an explicit tolerance (``TOLERANCES`` below). The script
prints one row per anchor and exits non-zero if any anchor falls outside its
tolerance, if a run the anchor depends on is infeasible, or if a solver raises.

Run from the mlsysim project root (the directory holding the ``mlsysim``
package)::

    python paper/scripts/validate_anchors.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_scenarios as ps  # noqa: E402

# Tolerance per anchor: ("relative", r) passes when |pred - rep| / rep <= r;
# ("exact", None) requires an exact match; ("none", None) has no reported value.
# A1/A3/A4 use the paper's stated first-order accuracy envelope of +/-20%
# (Discussion: "within +/-20% of a cycle-accurate trace"). A5 is an identity the
# paper states to better than 1%. A6 is an identity up to the rounding of the
# published 552 t. A7 must recover Meta's TP and PP exactly.
TOLERANCES = {
    "A1": ("relative", 0.20),
    "A2": ("none", None),
    "A3": ("relative", 0.20),
    "A4": ("relative", 0.20),
    "A5": ("relative", 0.01),
    "A6": ("relative", 0.005),
    "A7": ("exact", None),
}


def main() -> int:
    warnings.simplefilter("ignore")
    rows: list[tuple[str, str, str, str, str, str]] = []
    failures: list[str] = []
    notes: list[str] = []

    def relative_row(key, label, pred_txt, rep_txt, rel_error):
        _, tol = TOLERANCES[key]
        ok = rel_error <= tol
        rows.append((key, label, pred_txt, rep_txt, f"{rel_error:.1%} (tol {tol:.1%})", "PASS" if ok else "FAIL"))
        if not ok:
            failures.append(f"{key}: relative error {rel_error:.1%} exceeds tolerance {tol:.1%}")

    try:
        a1 = ps.anchor_one()
        if not a1["feasible"]:
            failures.append("A1: ResNet-50 training at batch 256 does not fit on one A100")
        relative_row("A1", "ResNet-50 8x A100 (img/s)", f"{a1['node']:,.0f}", f"{a1['reported']:,}", a1["rel_error"])

        a2 = ps.anchor_two()
        rows.append(("A2", "Llama-2 70B TP=2 decode floor", f"{a2['floor_ms']:.1f} ms", "none", "no reported value", "INFO"))

        a3 = ps.anchor_three()
        relative_row("A3", "Llama 3 405B 16K H100 MFU", f"{a3['mfu']:.1%}", f"{a3['reported']:.1%}", a3["rel_error"])
        if not a3["result"].node_profile.feasible:
            notes.append("A3: DistributedModel's per-accelerator profile is infeasible (it prices the unsharded "
                         "405B model on one H100); the anchor is not gated on it.")

        a4 = ps.anchor_four()
        relative_row("A4", "PaLM-540B 6,144 TPU v4 MFU", f"{a4['mfu']:.1%}", f"{a4['reported']:.1%}", a4["rel_error"])
        if not a4["result"].node_profile.feasible:
            notes.append("A4: DistributedModel's per-accelerator profile is infeasible (unsharded 540B on one "
                         "TPU v4); the anchor is not gated on it.")

        a5 = ps.anchor_five()
        relative_row("A5", "Chinchilla P* (params)", f"{a5['p_star'] / 1e9:.2f}B", f"{a5['reported'] / 1e9:.0f}B", a5["rel_error"])

        a6 = ps.anchor_six()
        relative_row("A6", "GPT-3 carbon (t CO2)", f"{a6['tonnes']:.1f}", f"{a6['reported']}", a6["rel_error"])

        a7 = ps.anchor_seven(a3["fleet"])
        cfg = a7["result"].best_config
        ok = a7["match"]
        rows.append(("A7", "Llama 3 405B parallelism", f"TP={cfg['tp']} PP={cfg['pp']} DP={cfg['dp']}",
                     f"TP={a7['reported_tp']} PP={a7['reported_pp']}", "exact TP, PP", "PASS" if ok else "FAIL"))
        if not ok:
            failures.append(f"A7: optimizer chose {cfg}, expected TP={a7['reported_tp']} PP={a7['reported_pp']}")
    except Exception as exc:  # any solver error is a failed check
        print(f"validate_anchors: solver raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    widths = [max(len(r[i]) for r in rows + [("Anchor", "Scenario", "Predicted", "Reported", "Error", "Status")])
              for i in range(6)]
    header = ("Anchor", "Scenario", "Predicted", "Reported", "Error", "Status")
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    for n in notes:
        print(f"note: {n}")
    if failures:
        print(f"\n{len(failures)} anchor check(s) failed:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nAll anchors within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
