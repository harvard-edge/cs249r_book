#!/usr/bin/env python3
"""Interactive accept/reject for Gemini-proposed corrections.

Reads a 01_audit.json file produced by audit_corpus_batched.py
--propose-fixes, walks each row that has a non-empty
``suggested_corrections`` block, and prompts the operator to:

  [a]ccept  — apply the proposed correction(s) to the YAML
  [r]eject  — leave the YAML untouched
  [e]dit    — open $EDITOR with the correction loaded; save to apply
  [s]kip    — defer to a later session
  [q]uit    — stop reviewing; persist dispositions and exit
  [h]elp    — print this menu

Per the CORPUS_HARDENING_PLAN.md correction policy:

  - Math errors: when math_correct=fail, the proposed correction
    typically rewrites BOTH napkin_math AND realistic_solution as a
    unit (the solution often depends on the napkin number). Review
    them together; accept-or-reject the pair, never split.

  - Level inflation: when level_fit=fail, the proposed correction
    relabels DOWN (e.g. L4 → L2) — never attempts to rewrite the
    question to match a higher claimed level.

  - Format markers: when format_compliance=fail, the proposed
    correction adds the missing markers but should NOT change the
    underlying prose semantics. If the diff shows prose changes
    beyond marker insertion, that's a sign the prompt drifted —
    reject and re-run propose-fixes with a tighter prompt.

CORPUS_HARDENING_PLAN.md Phase 5.

Usage:

    python3 interviews/vault-cli/scripts/apply_corrections.py \\
        --input interviews/vault/_pipeline/runs/<dir>/01_audit.json

    # Filter to a specific track or gate:
    apply_corrections.py --input <path> --filter-track cloud
    apply_corrections.py --input <path> --filter-gate format_compliance

    # Auto-accept format-marker-only corrections (low-risk; review
    # everything else):
    apply_corrections.py --input <path> --auto-accept-format

    # Resume an earlier session: dispositions persist to a sidecar so
    # already-accepted corrections aren't re-prompted.
    apply_corrections.py --input <path>
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# Locate vault_cli for round-trip-safe YAML I/O + Pydantic validation.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "interviews" / "vault-cli" / "src"))

from vault_cli.models import Question  # noqa: E402
from vault_cli.yaml_io import dump_str, load_file  # noqa: E402

VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"

# Disposition statuses written to the sidecar.
DISPOSITION_ACCEPTED = "accepted"
DISPOSITION_REJECTED = "rejected"
DISPOSITION_SKIPPED = "skipped"
DISPOSITION_EDITED = "edited"
DISPOSITION_FAILED = "failed-to-apply"


# ─── locating + loading ──────────────────────────────────────────────────


def find_question_file(qid: str) -> Path | None:
    """Locate a question YAML by id."""
    for path in QUESTIONS_DIR.rglob(f"{qid}.yaml"):
        return path
    return None


def load_dispositions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d.get("dispositions", {}) if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def save_dispositions(path: Path, dispositions: dict[str, dict],
                      input_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_audit": str(input_path),
        "dispositions": dispositions,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ─── correction application ──────────────────────────────────────────────


def apply_correction_to_dict(body: dict, correction: dict) -> dict:
    """Apply a correction dict to a question body in-place style.

    Returns a NEW dict (doesn't mutate input). Fields supported:
      - title              → top-level
      - level              → top-level (relabel-down per Q3)
      - common_mistake     → details.common_mistake
      - napkin_math        → details.napkin_math
      - realistic_solution → details.realistic_solution

    Other keys in correction are ignored with a warning.
    """
    out = json.loads(json.dumps(body))  # deep copy via JSON round-trip
    details = out.setdefault("details", {})

    if "title" in correction:
        out["title"] = correction["title"]
    if "level" in correction:
        out["level"] = correction["level"]
    if "common_mistake" in correction:
        details["common_mistake"] = correction["common_mistake"]
    if "napkin_math" in correction:
        details["napkin_math"] = correction["napkin_math"]
    if "realistic_solution" in correction:
        details["realistic_solution"] = correction["realistic_solution"]

    unknown = set(correction.keys()) - {
        "title", "level", "common_mistake", "napkin_math", "realistic_solution",
    }
    if unknown:
        print(f"  WARN: ignoring unknown correction keys: {sorted(unknown)}",
              file=sys.stderr)

    return out


def validate_proposed(body: dict) -> tuple[bool, str]:
    """Run Pydantic validation on the proposed body. Returns (ok, error)."""
    try:
        Question.model_validate(body)
        return True, ""
    except Exception as e:
        return False, str(e)[:300]


def write_yaml(path: Path, body: dict) -> None:
    """Atomic write: temp file then os.replace."""
    text = dump_str(body)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ─── diff rendering ──────────────────────────────────────────────────────


def render_field_diff(label: str, before: str | None, after: str | None) -> str:
    """Return a unified-diff snippet for one field; empty string if unchanged."""
    if before == after:
        return ""
    before_lines = (before or "").splitlines(keepends=True)
    after_lines = (after or "").splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        before_lines, after_lines,
        fromfile=f"{label} (current)",
        tofile=f"{label} (proposed)",
        n=2,
    ))
    if not diff:
        return ""
    return "".join(diff)


def render_correction(body: dict, correction: dict) -> str:
    """Pretty-print the diff between current YAML body and proposed correction."""
    parts: list[str] = []
    details = body.get("details") or {}

    for field, label in [
        ("title", "title"),
        ("level", "level"),
        ("common_mistake", "common_mistake"),
        ("napkin_math", "napkin_math"),
        ("realistic_solution", "realistic_solution"),
    ]:
        if field not in correction:
            continue
        current = (body.get(field) if field in ("title", "level")
                   else details.get(field))
        proposed = correction.get(field)
        if current == proposed:
            continue
        diff = render_field_diff(label, str(current or ""), str(proposed or ""))
        if diff:
            parts.append(diff)
        else:
            parts.append(f"--- {label}: {current!r} → {proposed!r}\n")
    return "".join(parts) if parts else "(no field changes)\n"


# ─── interactive prompt ──────────────────────────────────────────────────


HELP_MENU = """\
  [a]ccept   apply the proposed correction(s) to the YAML
  [r]eject   leave the YAML untouched
  [e]dit     open the proposed YAML in $EDITOR; save to apply, exit empty to reject
  [s]kip     defer to a later session
  [q]uit     stop reviewing; save dispositions and exit
  [h]elp     show this menu
"""


def prompt_choice() -> str:
    while True:
        try:
            ans = input("  [a/r/e/s/q/h]> ").strip().lower()
        except EOFError:
            return "q"
        if ans in {"a", "r", "e", "s", "q", "h", "accept", "reject",
                    "edit", "skip", "quit", "help"}:
            return ans[0]


def edit_in_editor(initial_body: dict) -> dict | None:
    """Open the proposed body in $EDITOR. Return the edited body, or None
    if the user emptied the file or didn't change it."""
    editor = os.environ.get("EDITOR", "vi")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(dump_str(initial_body))
        tmp_path = Path(f.name)
    try:
        subprocess.run([editor, str(tmp_path)], check=False)
        text = tmp_path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        # Round-trip through yaml_io to apply the same hardening + Pydantic
        # to ensure the edited file is valid before applying.
        try:
            return load_file(tmp_path)
        except Exception as e:
            print(f"  edited file failed to parse: {e}", file=sys.stderr)
            return None
    finally:
        tmp_path.unlink(missing_ok=True)


# ─── auto-accept heuristic ──────────────────────────────────────────────


def is_format_only_correction(correction: dict, body: dict) -> bool:
    """True iff the correction touches ONLY common_mistake and/or
    napkin_math, AND the proposed text contains the canonical markers
    while the current text does not.

    Used by --auto-accept-format. Lower-risk than auto-accepting other
    correction types because format markers are mechanical structure
    additions, not semantic rewrites.
    """
    keys = set(correction.keys())
    if not keys.issubset({"common_mistake", "napkin_math"}):
        return False

    if "common_mistake" in correction:
        new = correction["common_mistake"] or ""
        for marker in ("**The Pitfall:**", "**The Rationale:**", "**The Consequence:**"):
            if marker not in new:
                return False
    if "napkin_math" in correction:
        new = correction["napkin_math"] or ""
        if not all(m in new for m in ("**Calculations:**",)):
            return False
        # accepts both "Assumptions:" and "Assumptions & Constraints:"
        if "**Assumptions" not in new:
            return False
        if "**Conclusion" not in new:
            return False
    return True


# ─── main loop ───────────────────────────────────────────────────────────


def review_one(
    row: dict,
    *,
    auto_accept_format: bool,
) -> tuple[str, str]:
    """Returns (disposition, message). May write to disk."""
    qid = row.get("qid")
    correction = row.get("suggested_corrections") or {}
    if not correction:
        return DISPOSITION_SKIPPED, "no correction proposed"

    yaml_path = find_question_file(qid)
    if not yaml_path:
        return DISPOSITION_FAILED, f"YAML not found for {qid}"

    try:
        body = load_file(yaml_path)
    except Exception as e:
        return DISPOSITION_FAILED, f"YAML load failed: {e}"

    if not isinstance(body, dict):
        return DISPOSITION_FAILED, "YAML did not parse to a dict"

    # Print summary of the row's gate verdicts, then the diff.
    print(f"\n─── {qid} ─── [{body.get('track')}/{body.get('level')}] "
          f"{body.get('title', '')[:60]}")
    gate_summary = ", ".join(
        f"{g}={row.get(g, '?')}"
        for g in ("format_compliance", "level_fit", "coherence",
                   "math_correct", "title_quality")
    )
    print(f"  gates: {gate_summary}")
    if row.get("level_fit_rationale") or row.get("coherence_rationale"):
        for k in ("level_fit_rationale", "coherence_rationale"):
            if row.get(k):
                print(f"  {k}: {row[k]}")
    print()
    print(render_correction(body, correction))

    if auto_accept_format and is_format_only_correction(correction, body):
        print("  [auto-accept] format-marker-only correction")
        choice = "a"
    else:
        choice = prompt_choice()

    if choice == "h":
        print(HELP_MENU)
        return review_one(row, auto_accept_format=auto_accept_format)
    if choice == "q":
        return "quit", ""
    if choice == "s":
        return DISPOSITION_SKIPPED, ""
    if choice == "r":
        return DISPOSITION_REJECTED, ""

    if choice == "e":
        proposed_body = apply_correction_to_dict(body, correction)
        edited = edit_in_editor(proposed_body)
        if edited is None:
            return DISPOSITION_REJECTED, "editor returned empty"
        ok, why = validate_proposed(edited)
        if not ok:
            return DISPOSITION_FAILED, f"edited body fails validation: {why}"
        write_yaml(yaml_path, edited)
        return DISPOSITION_EDITED, ""

    # accept
    proposed_body = apply_correction_to_dict(body, correction)
    ok, why = validate_proposed(proposed_body)
    if not ok:
        return DISPOSITION_FAILED, f"proposed body fails validation: {why}"
    write_yaml(yaml_path, proposed_body)
    return DISPOSITION_ACCEPTED, ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", type=Path, required=True,
                    help="path to 01_audit.json from a --propose-fixes run")
    ap.add_argument("--dispositions-out", type=Path, default=None,
                    help="sidecar JSON to persist accept/reject decisions "
                         "(default: <input-dir>/02_dispositions.json)")
    ap.add_argument("--filter-track", type=str, default=None,
                    help="only review qids in this track")
    ap.add_argument("--filter-gate", type=str, default=None,
                    help="only review rows where this gate failed "
                         "(format_compliance / level_fit / coherence / "
                         "math_correct / title_quality)")
    ap.add_argument("--auto-accept-format", action="store_true",
                    help="auto-accept format-marker-only corrections "
                         "(lower-risk: just adds the canonical markers)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap how many corrections to review this session")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1

    audit = json.loads(args.input.read_text(encoding="utf-8"))
    rows = audit.get("rows", [])
    print(f"loaded {len(rows)} audit rows from {args.input}")

    # Filter to rows that have a correction.
    candidates = [r for r in rows if r.get("suggested_corrections")]
    print(f"  with proposed corrections: {len(candidates)}")

    if args.filter_track:
        # Need to look up track from the YAML since rows don't always carry it.
        before = len(candidates)
        filtered = []
        for r in candidates:
            yp = find_question_file(r.get("qid"))
            if not yp:
                continue
            try:
                b = load_file(yp)
                if b.get("track") == args.filter_track:
                    filtered.append(r)
            except Exception:
                continue
        candidates = filtered
        print(f"  after --filter-track={args.filter_track}: "
              f"{len(candidates)} (was {before})")

    if args.filter_gate:
        candidates = [r for r in candidates if r.get(args.filter_gate) == "fail"]
        print(f"  after --filter-gate={args.filter_gate}: {len(candidates)}")

    # Load prior dispositions; skip qids already accepted/rejected.
    disp_path = args.dispositions_out or (args.input.parent / "02_dispositions.json")
    dispositions = load_dispositions(disp_path)
    candidates = [r for r in candidates
                   if dispositions.get(r.get("qid"), {}).get("disposition")
                   not in {DISPOSITION_ACCEPTED, DISPOSITION_REJECTED,
                            DISPOSITION_EDITED, DISPOSITION_FAILED}]
    print(f"  remaining (not yet accepted/rejected/edited): {len(candidates)}")

    if args.limit:
        candidates = candidates[: args.limit]
        print(f"  capped at --limit={args.limit}")

    if not candidates:
        print("nothing to review.")
        return 0

    counters = {DISPOSITION_ACCEPTED: 0, DISPOSITION_REJECTED: 0,
                 DISPOSITION_EDITED: 0, DISPOSITION_SKIPPED: 0,
                 DISPOSITION_FAILED: 0}

    for i, row in enumerate(candidates, start=1):
        print(f"\n[{i}/{len(candidates)}]", end="")
        result, msg = review_one(row, auto_accept_format=args.auto_accept_format)
        if result == "quit":
            print("\n[quit] stopping; persisting dispositions...")
            break
        counters[result] = counters.get(result, 0) + 1
        dispositions[row["qid"]] = {
            "disposition": result,
            "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "message": msg,
        }
        save_dispositions(disp_path, dispositions, args.input)

    print()
    print("session summary:")
    for k, v in counters.items():
        print(f"  {k:20s} {v}")
    print(f"\nwrote dispositions to {disp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
