#!/usr/bin/env python3
"""Iterative coverage loop: analyze → generate → render → judge → apply.

The loop keeps tightening corpus balance by re-analyzing after every
generation pass. It stops automatically when the corpus reaches a
steady state — no big gaps remain, hallucination rate spikes, or
budget is exhausted. The user does not pick a "total questions to
generate"; the loop self-paces against measurable saturation.

Pipeline per iteration:

  1. analyze_coverage_gaps.py         → top priority cells
  2. gemini_cli_generate_questions.py → batched draft generation
  3. render_visuals.py                → rebuild any visual SVGs
  4. gemini_cli_llm_judge.py          → multi-criteria validation
  5. apply judgments                  → drop DROP, keep PASS as draft
  6. log to history; check saturation criteria

Stop conditions (any one halts the loop):

  - Top priority gap drops below `--gap-threshold` (default 1.0)
  - LLM-as-judge DROP rate exceeds `--max-drop-rate` (default 0.3)
  - Total Gemini API calls exceed `--max-calls` (default 60)
  - Iteration count reaches `--max-iters` (default 20)
  - Same priority cell appears in two consecutive iterations
    (analyzer can't find new gaps — convergence)

Usage:

    # Run the loop with default budgets:
    python3 iterate_coverage_loop.py

    # Conservative: 5 iterations, 30 calls max:
    python3 iterate_coverage_loop.py --max-iters 5 --max-calls 30

    # Plan only (no API calls):
    python3 iterate_coverage_loop.py --dry-run

The loop logs every iteration to
``interviews/vault/_validation_results/coverage_loop/<timestamp>/`` so
the operator can inspect what happened on each pass and at what point
saturation was reached.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VAULT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = VAULT_DIR / "scripts"
QUESTIONS_DIR = VAULT_DIR / "questions"
VISUALS_DIR = VAULT_DIR / "visuals"
DEFAULT_OUTPUT_DIR = VAULT_DIR / "_validation_results" / "coverage_loop"


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    """Run a subprocess; return (returncode, stdout). stderr passes through."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stderr.strip():
        # Surface stderr but don't kill the loop on warnings
        print(result.stderr.rstrip(), file=sys.stderr)
    return result.returncode, result.stdout


def analyze(plan_size: int, want_visual: bool, out_dir: Path) -> dict[str, Any]:
    """Run analyze_coverage_gaps.py and return parsed report.json."""
    cmd = [
        sys.executable, str(SCRIPTS / "analyze_coverage_gaps.py"),
        "--total", str(plan_size),
    ]
    if want_visual:
        cmd.append("--visual")
    rc, _ = run(cmd)
    if rc != 0:
        raise RuntimeError("analyze step failed")
    # Find latest report
    cgaps_dir = VAULT_DIR / "_validation_results" / "coverage_gaps"
    latest = max(cgaps_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    report = json.loads((latest / "report.json").read_text())
    # Copy report into the loop's iteration dir for traceability
    shutil.copy(latest / "report.md", out_dir / "report.md")
    shutil.copy(latest / "report.json", out_dir / "report.json")
    return report


def generate(plan: list[dict[str, Any]], batch_size: int, max_calls: int,
             want_visual: bool, dry_run: bool) -> tuple[int, list[Path]]:
    """Generate via batched Gemini calls. Return (calls_used, new_yaml_paths)."""
    if not plan:
        return 0, []
    # Snapshot which YAMLs exist before — anything new after is what we just
    # generated.
    before = {p for p in QUESTIONS_DIR.glob("**/*.yaml")}

    # Targets: the analyzer's recommended cells, formatted for --target.
    targets: list[str] = []
    for cell in plan:
        targets.append(f"{cell['track']}:{cell['topic']}:{cell['zone']}:{cell['level']}")

    cmd = [
        sys.executable, str(SCRIPTS / "gemini_cli_generate_questions.py"),
        "--batch-size", str(batch_size),
        "--max-calls", str(max_calls),
    ]
    for t in targets:
        cmd += ["--target", t]
    if want_visual:
        cmd.append("--visual")
    if dry_run:
        cmd.append("--dry-run")

    rc, _ = run(cmd)
    if rc != 0:
        print("  ! generate step returned non-zero; continuing")

    after = {p for p in QUESTIONS_DIR.glob("**/*.yaml")}
    new = sorted(after - before)
    expected_calls = (len(plan) + batch_size - 1) // batch_size
    return min(expected_calls, max_calls), new


def render_visuals_step() -> int:
    """Render any new/stale visuals."""
    cmd = [sys.executable, str(SCRIPTS / "render_visuals.py")]
    rc, _ = run(cmd)
    return rc


def judge(new_yaml_paths: list[Path], chunk_size: int, max_calls: int,
          dry_run: bool, out_dir: Path) -> dict[str, Any]:
    """Run LLM-as-judge on the just-generated drafts."""
    if not new_yaml_paths:
        return {"verdicts": {}, "details": [], "drop_rate": 0.0,
                "pass_rate": 0.0, "calls_used": 0}
    if dry_run:
        return {"verdicts": {"PASS": len(new_yaml_paths)}, "details": [],
                "drop_rate": 0.0, "pass_rate": 1.0, "calls_used": 0}

    files_from = out_dir / "judge_inputs.txt"
    files_from.write_text(
        "\n".join(str(p.relative_to(VAULT_DIR.parent.parent)) for p in new_yaml_paths),
        encoding="utf-8",
    )
    cmd = [
        sys.executable, str(SCRIPTS / "gemini_cli_llm_judge.py"),
        "--files-from", str(files_from),
        "--chunk-size", str(chunk_size),
        "--max-calls", str(max_calls),
    ]
    rc, _ = run(cmd)

    judge_dir = VAULT_DIR / "_validation_results" / "llm_judge"
    if not judge_dir.exists():
        return {"verdicts": {}, "details": [], "drop_rate": 0.0,
                "pass_rate": 0.0, "calls_used": 0}
    latest = max(judge_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    summary = json.loads((latest / "summary.json").read_text())
    summary["calls_used"] = (len(new_yaml_paths) + chunk_size - 1) // chunk_size
    # Copy into iteration dir
    shutil.copy(latest / "summary.json", out_dir / "judge_summary.json")
    return summary


def apply_judgments(judge_summary: dict[str, Any]) -> dict[str, int]:
    """Drop DROP-verdict YAMLs (and their visual sources). PASS items stay
    as draft; NEEDS_FIX items also stay (a human edits them)."""
    counts = {"dropped": 0, "kept_pass": 0, "kept_needs_fix": 0}
    for item in judge_summary.get("details", []):
        verdict = item.get("verdict")
        qid = item.get("id")
        if not qid:
            continue
        if verdict == "DROP":
            # Find and delete the YAML + any sibling source files
            for p in QUESTIONS_DIR.glob(f"**/{qid}.yaml"):
                p.unlink()
                counts["dropped"] += 1
            for ext in (".dot", ".py", ".svg"):
                for sp in VISUALS_DIR.glob(f"**/{qid}{ext}"):
                    sp.unlink()
        elif verdict == "PASS":
            counts["kept_pass"] += 1
        elif verdict == "NEEDS_FIX":
            counts["kept_needs_fix"] += 1
    return counts


def saturation_reached(history: list[dict[str, Any]], current: dict[str, Any],
                       gap_threshold: float, max_drop_rate: float) -> str | None:
    """Return reason string if saturated; None to continue."""
    top = current.get("top_priority", 0.0)
    if top < gap_threshold:
        return f"top priority gap {top:.2f} below threshold {gap_threshold}"
    if current.get("drop_rate", 0.0) > max_drop_rate:
        return (f"DROP rate {current['drop_rate']:.1%} exceeds "
                f"{max_drop_rate:.0%} — likely hallucination")
    if len(history) >= 2:
        prev = history[-1]
        if (prev.get("top_cell") == current.get("top_cell")
                and prev.get("top_priority") == current.get("top_priority")):
            return "same top-priority cell two iterations in a row — converged"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--max-calls", type=int, default=60,
                        help="Hard cap on TOTAL Gemini API calls (gen + judge).")
    parser.add_argument("--gen-batch-size", type=int, default=12)
    parser.add_argument("--gen-calls-per-iter", type=int, default=3,
                        help="Gen calls per iteration. With batch_size=12 → 36 q/iter.")
    parser.add_argument("--judge-chunk-size", type=int, default=15)
    parser.add_argument("--gap-threshold", type=float, default=1.0,
                        help="Stop when top priority gap drops below this.")
    parser.add_argument("--max-drop-rate", type=float, default=0.3,
                        help="Stop if LLM-as-judge DROP rate exceeds this.")
    parser.add_argument("--visual-each-iter", action="store_true",
                        help="Bias half of each iteration's plan toward visuals.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loop run dir: {run_dir}")

    history: list[dict[str, Any]] = []
    calls_used_total = 0

    for it in range(args.max_iters):
        iter_dir = run_dir / f"iter_{it:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Iteration {it+1} of {args.max_iters} ===")

        # 1. Analyze
        plan_size = args.gen_batch_size * args.gen_calls_per_iter
        report = analyze(plan_size,
                         want_visual=(args.visual_each_iter and it % 2 == 0),
                         out_dir=iter_dir)
        plan = report.get("recommended_plan", [])
        if not plan:
            print("  ! analyzer returned empty plan; halting")
            break
        top_cell = (plan[0]["track"], plan[0]["zone"], plan[0]["level"])
        top_priority = plan[0]["priority"]
        print(f"  top priority: {top_cell} @ {top_priority}")

        # 2. Generate
        calls_remaining = args.max_calls - calls_used_total
        gen_calls = min(args.gen_calls_per_iter, max(0, calls_remaining // 2))
        if gen_calls == 0:
            print("  ! API call budget exhausted; halting")
            break
        plan_subset = plan[: gen_calls * args.gen_batch_size]
        used, new_paths = generate(
            plan_subset, args.gen_batch_size, gen_calls,
            want_visual=(args.visual_each_iter and it % 2 == 0),
            dry_run=args.dry_run,
        )
        calls_used_total += used
        print(f"  generated {len(new_paths)} drafts ({used} calls)")

        # 3. Render visuals
        render_visuals_step()

        # 4. Judge
        judge_calls_left = args.max_calls - calls_used_total
        judge_calls = min(5, max(0, judge_calls_left))
        judgment = judge(new_paths, args.judge_chunk_size, judge_calls,
                         dry_run=args.dry_run, out_dir=iter_dir)
        calls_used_total += judgment.get("calls_used", 0)
        print(f"  judge verdicts: {judgment.get('verdicts', {})}")

        # 5. Apply judgments
        applied = apply_judgments(judgment) if not args.dry_run else {
            "dropped": 0, "kept_pass": 0, "kept_needs_fix": 0}
        print(f"  applied: dropped={applied['dropped']} "
              f"pass={applied['kept_pass']} fix={applied['kept_needs_fix']}")

        # 6. Record + check saturation
        record = {
            "iter": it, "top_cell": list(top_cell), "top_priority": top_priority,
            "generated": len(new_paths), "calls_used_total": calls_used_total,
            "drop_rate": judgment.get("drop_rate", 0.0),
            "pass_rate": judgment.get("pass_rate", 0.0),
            "applied": applied,
        }
        history.append(record)
        (iter_dir / "iter_record.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")

        reason = saturation_reached(history[:-1], record,
                                    args.gap_threshold, args.max_drop_rate)
        if reason:
            print(f"  ✓ STOP: {reason}")
            break

    # Final summary
    final = {
        "iterations": len(history),
        "calls_used_total": calls_used_total,
        "history": history,
        "stopped_reason": (history and reason) or "max iterations reached",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "loop_summary.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8")
    print(f"\n=== Loop complete ===")
    print(f"Iterations:      {len(history)}")
    print(f"Calls used:      {calls_used_total}")
    print(f"Total generated: {sum(h['generated'] for h in history)}")
    print(f"Total dropped:   {sum(h['applied']['dropped'] for h in history)}")
    print(f"Summary: {run_dir}/loop_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
