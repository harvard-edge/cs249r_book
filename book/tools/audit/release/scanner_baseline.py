#!/usr/bin/env python3
"""Phase A6 — re-run the audit infrastructure against current dev HEAD.

Captures three measurements that together constitute the audit baseline:

1. ``book/tools/audit/scan.py --scope vol1|vol2`` ledger counts
2. The 6 detector self-tests (123/123 expected)
3. ``book/tools/bib_lint.py --all --check`` (36 grandfathered, 0 new)

Compares to the post-Pass 16 expected end state recorded in
``PASS_16_COMPLETION_REPORT.md`` §3 and reports any deviation as a
regression.

Output: ``ledgers/scanner-baseline.json`` plus per-run logs under
``logs/scanner-baseline/``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/VJ/GitHub/MLSysBook-release-audit")
OUT_DIR = Path.home() / "Desktop/MIT_Press_Feedback/16_release_audit/ledgers"
LOG_DIR = Path.home() / "Desktop/MIT_Press_Feedback/16_release_audit/logs/scanner-baseline"

EXPECTED_SCANNER = {
    "vol1": {"total": 4, "accepted": 3, "open": 1},
    "vol2": {"total": 7, "accepted": 2, "open": 5},
}
EXPECTED_SELF_TESTS = {
    "h3_titlecase": 41,
    "concept_term_capitalization": 36,
    "abbreviation_first_use": 17,
    "latin_running_text": 14,
    "alt_text_style": 7,
    "bibliography_hygiene": 8,
}
EXPECTED_BIB_LINT = {"grandfathered": 36, "new_errors": 0}


def run(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        env=full_env,
    )
    output = proc.stdout + proc.stderr
    log_path.write_text(output)
    return (proc.returncode, output)


def parse_scan_output(output: str) -> dict:
    """Pull totals out of the scanner's text output.

    Scanner output (post-Pass 16 format) looks like::

        Total: 3 issues across 35 files (5.2s)

        AUDIT LEDGER SUMMARY (scope=vol1)
        ============================================================
        Total issues: 3

        By category:
          abbreviation-first-use                   3

        By status:
          accepted                                 3
          open                                     0
    """
    out = {"raw_total": None, "accepted": 0, "open": 0, "by_category": {}}
    m = re.search(r"Total issues?:\s*(\d+)", output)
    if m:
        out["raw_total"] = int(m.group(1))
    # By status block.
    in_status = False
    in_category = False
    for line in output.splitlines():
        if "By status" in line:
            in_status, in_category = True, False
            continue
        if "By category" in line:
            in_category, in_status = True, False
            continue
        if not line.strip() or line.startswith("="):
            in_status = False
            in_category = False
            continue
        if in_status:
            m = re.match(r"\s*(\w+)\s+(\d+)", line)
            if m:
                out[m.group(1)] = int(m.group(2))
        elif in_category:
            m = re.match(r"\s*([a-z][a-z0-9-]*)\s+(\d+)", line)
            if m:
                out["by_category"][m.group(1)] = int(m.group(2))
    return out


def parse_selftest(output: str) -> dict:
    """Self-test outputs typically end with 'X/Y tests passed'."""
    m = re.search(r"(\d+)\s*/\s*(\d+)\s*(?:tests?\s*)?(?:passed|PASSED|OK)", output)
    if m:
        return {"passed": int(m.group(1)), "total": int(m.group(2))}
    # Fallback: count lines starting with 'PASS' / 'FAIL'.
    pass_count = sum(1 for line in output.splitlines() if line.strip().startswith("PASS"))
    fail_count = sum(1 for line in output.splitlines() if line.strip().startswith("FAIL"))
    if pass_count or fail_count:
        return {"passed": pass_count, "total": pass_count + fail_count}
    return {"passed": None, "total": None, "raw_tail": output.splitlines()[-3:]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = {"scanner": {}, "self_tests": {}, "bib_lint": {}, "regressions": []}

    # 1. Scanner runs.
    for scope in ("vol1", "vol2"):
        log = LOG_DIR / f"scan-{scope}.log"
        rc, out = run(
            ["python3", "book/tools/audit/scan.py", "--scope", scope, "-v",
             "--output", str(LOG_DIR / f"scan-{scope}-ledger.json")],
            log,
        )
        parsed = parse_scan_output(out)
        parsed["return_code"] = rc
        parsed["log"] = str(log)
        results["scanner"][scope] = parsed

        exp = EXPECTED_SCANNER[scope]
        if parsed["raw_total"] is None:
            results["regressions"].append(
                f"scanner {scope}: could not parse output (rc={rc}); see {log}"
            )
        else:
            # A regression is "more open issues than the pass-16 expected
            # baseline". If we have FEWER open issues, that's an improvement
            # and is recorded as an info note, not a regression.
            if parsed.get("open", 0) > exp["open"]:
                results["regressions"].append(
                    f"scanner {scope}: open={parsed.get('open')} > expected={exp['open']} (REGRESSION)"
                )
            elif parsed.get("open", 0) < exp["open"]:
                results.setdefault("improvements", []).append(
                    f"scanner {scope}: open={parsed.get('open')} < expected={exp['open']} (improvement vs pass-16)"
                )

    # 2. Detector self-tests.
    for module in EXPECTED_SELF_TESTS:
        log = LOG_DIR / f"selftest-{module}.log"
        rc, out = run(
            ["python3", f"book/tools/audit/checks/{module}.py"],
            log,
            env={"PYTHONPATH": str(REPO / "book/tools")},
        )
        parsed = parse_selftest(out)
        parsed["return_code"] = rc
        parsed["log"] = str(log)
        results["self_tests"][module] = parsed
        exp_count = EXPECTED_SELF_TESTS[module]
        if parsed.get("passed") != exp_count:
            results["regressions"].append(
                f"self-test {module}: passed={parsed.get('passed')}/"
                f"{parsed.get('total')} expected={exp_count}/{exp_count}"
            )

    # 3. bib_lint --all --check.
    log = LOG_DIR / "bib-lint.log"
    rc, out = run(["python3", "book/tools/bib_lint.py", "--all", "--check"], log)
    # Output format: "Total: 0 NEW errors (0 grandfathered), 62 warnings"
    grandfathered = None
    new_errors = None
    warnings = None
    m = re.search(r"(\d+)\s+NEW\s+errors?\s*\((\d+)\s+grandfathered\)", out)
    if m:
        new_errors = int(m.group(1))
        grandfathered = int(m.group(2))
    m = re.search(r"(\d+)\s+warnings?", out)
    if m:
        warnings = int(m.group(1))
    results["bib_lint"] = {
        "return_code": rc,
        "grandfathered": grandfathered,
        "new_errors": new_errors,
        "warnings": warnings,
        "log": str(log),
    }
    if rc != 0:
        results["regressions"].append(f"bib_lint: return code {rc} (expected 0)")
    if new_errors is not None and new_errors > EXPECTED_BIB_LINT["new_errors"]:
        results["regressions"].append(
            f"bib_lint: new_errors={new_errors} > expected={EXPECTED_BIB_LINT['new_errors']}"
        )
    if grandfathered is not None and grandfathered < EXPECTED_BIB_LINT["grandfathered"]:
        results.setdefault("improvements", []).append(
            f"bib_lint: grandfathered={grandfathered} < expected={EXPECTED_BIB_LINT['grandfathered']} "
            f"(improvement: {EXPECTED_BIB_LINT['grandfathered'] - grandfathered} entries newly resolved)"
        )

    out_path = OUT_DIR / "scanner-baseline.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps({
        "scanner_open": {s: results["scanner"][s].get("open") for s in ("vol1", "vol2")},
        "scanner_accepted": {s: results["scanner"][s].get("accepted") for s in ("vol1", "vol2")},
        "scanner_total": {s: results["scanner"][s].get("raw_total") for s in ("vol1", "vol2")},
        "self_tests": {m: f"{v.get('passed')}/{v.get('total')}"
                       for m, v in results["self_tests"].items()},
        "bib_lint": {k: results["bib_lint"].get(k) for k in ("return_code", "grandfathered", "new_errors", "warnings")},
        "regressions": results["regressions"],
        "improvements": results.get("improvements", []),
    }, indent=2))


if __name__ == "__main__":
    main()
