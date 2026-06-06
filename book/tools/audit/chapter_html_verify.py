#!/usr/bin/env python3
"""Build chapter HTML, audit raw output, and track pass/fail in a ledger.

Usage (repo root)::

    python3 book/tools/audit/chapter_html_verify.py --list
    python3 book/tools/audit/chapter_html_verify.py --vol1 introduction
    python3 book/tools/audit/chapter_html_verify.py --vol1 --all
    python3 book/tools/audit/chapter_html_verify.py --report

Ledger: book/tools/audit/artifacts/chapter_html_audit.json
Table:   book/tools/audit/artifacts/chapter_html_audit.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_HTML = REPO_ROOT / "book/tools/audit/fmt/audit_html.py"
LEDGER_JSON = REPO_ROOT / "book/tools/audit/artifacts/chapter_html_audit.json"
LEDGER_MD = REPO_ROOT / "book/tools/audit/artifacts/chapter_html_audit.md"

CHAPTERS: dict[str, list[str]] = {
    "vol1": [
        "introduction/introduction",
        "ml_systems/ml_systems",
        "ml_workflow/ml_workflow",
        "data_engineering/data_engineering",
        "nn_computation/nn_computation",
        "nn_architectures/nn_architectures",
        "frameworks/frameworks",
        "training/training",
        "data_selection/data_selection",
        "model_compression/model_compression",
        "hw_acceleration/hw_acceleration",
        "benchmarking/benchmarking",
        "model_serving/model_serving",
        "ml_ops/ml_ops",
        "responsible_engr/responsible_engr",
        "conclusion/conclusion",
        "backmatter/appendix_algorithm",
        "backmatter/appendix_assumptions",
        "backmatter/appendix_dam",
        "backmatter/appendix_data",
        "backmatter/appendix_machine",
    ],
    "vol2": [
        "introduction/introduction",
        "compute_infrastructure/compute_infrastructure",
        "network_fabrics/network_fabrics",
        "data_storage/data_storage",
        "distributed_training/distributed_training",
        "collective_communication/collective_communication",
        "fault_tolerance/fault_tolerance",
        "fleet_orchestration/fleet_orchestration",
        "performance_engineering/performance_engineering",
        "inference/inference",
        "edge_intelligence/edge_intelligence",
        "ops_scale/ops_scale",
        "security_privacy/security_privacy",
        "robust_ai/robust_ai",
        "sustainable_ai/sustainable_ai",
        "responsible_ai/responsible_ai",
        "conclusion/conclusion",
        "backmatter/appendix_dam",
        "backmatter/appendix_fleet",
        "backmatter/appendix_communication",
        "backmatter/appendix_reliability",
        "backmatter/appendix_inference",
        "backmatter/appendix_c3",
        "backmatter/appendix_assumptions",
    ],
}

HTML_ERROR_PATTERNS = [
    re.compile(r"NameError:", re.I),
    re.compile(r"KeyError:", re.I),
    re.compile(r"AttributeError:", re.I),
    re.compile(r"ModuleNotFoundError:", re.I),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"Error rendering"),
]

HTML_MATH_RENDER_PATTERNS = [
    (
        "spaced ASCII multiplication",
        re.compile(r"\b\d[\d,]*(?:\.\d+)?\s+x\s+(?:\d|\$|USD|[A-Za-z_(])"),
    ),
    (
        "literal dollar-times",
        re.compile(r"\$\s*\\times\s*\$"),
    ),
]


@dataclass
class ChapterResult:
    vol: str
    chapter: str
    qmd: str
    html: str
    status: str  # pass | fail | pending | skip
    build_ok: bool = False
    build_seconds: float = 0.0
    html_spurious_clean: bool | None = None
    html_error_hits: list[str] = field(default_factory=list)
    html_math_artifacts_clean: bool | None = None
    html_math_artifact_hits: list[str] = field(default_factory=list)
    lego_focal_ok: bool | None = None
    prose_ok: bool | None = None
    registry_ok: bool | None = None
    notes: str = ""
    updated_at: str = ""


def _chapter_id(vol: str, ch_path: str) -> str:
    return f"{vol}/{ch_path.split('/')[-1]}"


def _qmd_path(vol: str, ch_path: str) -> Path:
    return REPO_ROOT / "book/quarto/contents" / vol / f"{ch_path}.qmd"


def _build_html(vol: str, ch_path: str) -> tuple[bool, float, str]:
    name = ch_path.split("/")[-1]
    binder_vol = f"--{vol}"
    binder_ch = f"{vol}/{name}"
    log = Path(f"/tmp/render_{vol}_{name}.log")
    t0 = time.monotonic()
    proc = subprocess.run(
        ["./book/binder", "build", "html", binder_vol, binder_ch,
         "--skip-hygiene", "--skip-validate"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - t0
    log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    return proc.returncode == 0, elapsed, str(log)


def _live_html(vol: str, ch_path: str) -> Path:
    build_dir = REPO_ROOT / "book/quarto/_build" / f"html-{vol}" / "contents" / vol
    return build_dir / f"{ch_path}.html"


def _archive_html(vol: str, ch_path: str, live: Path) -> Path:
    name = ch_path.split("/")[-1]
    archive_dir = REPO_ROOT / "book/quarto/_build/html-audit" / vol
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / f"{name}.html"
    dest.write_bytes(live.read_bytes())
    return dest


def _audit_spurious(html: Path) -> tuple[bool, list[str]]:
    proc = subprocess.run(
        [sys.executable, str(AUDIT_HTML), str(html)],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 and "CLEAN" in proc.stdout:
        return True, []
    issues = [ln for ln in (proc.stdout or proc.stderr).splitlines() if ln.strip()]
    return False, issues[:10]


def _scan_html_errors(html: Path) -> list[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    soup = BeautifulSoup(html.read_text(encoding="utf-8", errors="replace"), "html.parser")
    content = soup.find("main") or soup.body
    if not content:
        return []
    for tag in content(["script", "style", "pre", "code"]):
        tag.decompose()
    text = content.get_text(separator=" ")
    hits: list[str] = []
    for pat in HTML_ERROR_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    return hits


def _visible_text(html: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""
    soup = BeautifulSoup(html.read_text(encoding="utf-8", errors="replace"), "html.parser")
    content = soup.find("main") or soup.body
    if not content:
        return ""
    for tag in content(["script", "style", "annotation", "mjx-assistive-mml"]):
        tag.decompose()
    return re.sub(r"\s+", " ", content.get_text(separator=" "))


def _scan_math_render_artifacts(html: Path) -> list[str]:
    text = _visible_text(html)
    if not text:
        return []
    hits: list[str] = []
    for label, pat in HTML_MATH_RENDER_PATTERNS:
        for match in pat.finditer(text):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()
            hits.append(f"{label}: {context}")
            if len(hits) >= 10:
                return hits
    return hits


def _lego_focal(qmd: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "book/tools/audit/lego_focal_verify.py"), str(qmd)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0 and not out.strip():
        return True, "no inline Python refs"
    ok = proc.returncode == 0 and "issues=0" in out
    return ok, out.strip().splitlines()[-1] if out.strip() else "no output"


def _prose_exec(qmd: Path, timeout_s: int = 120) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "mlsysim"), "MPLBACKEND": "Agg"}
    try:
        proc = subprocess.run(
            [sys.executable, "-W", "ignore::UserWarning",
             str(REPO_ROOT / "book/tools/audit/fmt/audit_prose.py"), str(qmd)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"prose exec timed out after {timeout_s}s"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "prose exec failed").strip()
        return False, err[:300]
    return True, "ok"


def _registry_scan(qmd: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "book/tools/audit/book_check_registry_sources.py"), str(qmd)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, (proc.stdout or proc.stderr or "").strip()[-200:]


def verify_chapter(vol: str, ch_path: str, skip_build: bool = False) -> ChapterResult:
    qmd = _qmd_path(vol, ch_path)
    name = ch_path.split("/")[-1]
    res = ChapterResult(
        vol=vol,
        chapter=name,
        qmd=str(qmd.relative_to(REPO_ROOT)),
        html="",
        status="pending",
        updated_at=datetime.now(timezone.utc).isoformat(),
    )

    if not qmd.is_file():
        res.status = "skip"
        res.notes = "QMD missing"
        return res

    reg_ok, reg_msg = _registry_scan(qmd)
    res.registry_ok = reg_ok
    if not reg_ok:
        res.notes += f"registry: {reg_msg}; "

    lego_ok, lego_msg = _lego_focal(qmd)
    res.lego_focal_ok = lego_ok
    if not lego_ok:
        res.notes += f"lego: {lego_msg}; "

    prose_ok, prose_msg = _prose_exec(qmd)
    res.prose_ok = prose_ok
    if not prose_ok:
        res.notes += f"prose: {prose_msg}; "

    if skip_build:
        archive = REPO_ROOT / "book/quarto/_build/html-audit" / vol / f"{name}.html"
        if not archive.is_file():
            res.status = "pending"
            res.notes += "no archived HTML (--skip-build)"
            return res
        html = archive
        res.build_ok = True
        res.build_seconds = 0.0
    else:
        build_ok, secs, log = _build_html(vol, ch_path)
        res.build_ok = build_ok
        res.build_seconds = round(secs, 1)
        if not build_ok:
            res.status = "fail"
            res.notes += f"build failed ({log}); "
            return res
        live = _live_html(vol, ch_path)
        if not live.is_file():
            res.status = "fail"
            res.notes += f"HTML missing at {live}; "
            return res
        html = _archive_html(vol, ch_path, live)

    res.html = str(html.relative_to(REPO_ROOT))
    res.html_error_hits = _scan_html_errors(html)
    res.html_math_artifact_hits = _scan_math_render_artifacts(html)
    res.html_math_artifacts_clean = not res.html_math_artifact_hits
    clean, spurious = _audit_spurious(html)
    res.html_spurious_clean = clean
    if res.html_error_hits:
        res.notes += f"html errors: {res.html_error_hits}; "
    if res.html_math_artifact_hits:
        res.notes += f"math artifacts: {res.html_math_artifact_hits[:2]}; "
    if not clean:
        res.notes += f"spurious .0: {spurious[:2]}; "

    passed = (
        res.build_ok
        and not res.html_error_hits
        and res.html_math_artifacts_clean
        and res.html_spurious_clean
        and res.lego_focal_ok
        and res.prose_ok
        and res.registry_ok
    )
    res.status = "pass" if passed else "fail"
    if passed:
        res.notes = "OK — build + HTML clean + LEGO/registry consistent"
    return res


def load_ledger() -> dict:
    if LEDGER_JSON.is_file():
        return json.loads(LEDGER_JSON.read_text(encoding="utf-8"))
    return {"updated_at": "", "chapters": {}}


def save_ledger(ledger: dict) -> None:
    LEDGER_JSON.parent.mkdir(parents=True, exist_ok=True)
    ledger["updated_at"] = datetime.now(timezone.utc).isoformat()
    LEDGER_JSON.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    _write_markdown_table(ledger)


def _write_markdown_table(ledger: dict) -> None:
    rows = []
    for key in sorted(ledger.get("chapters", {}).keys()):
        c = ledger["chapters"][key]
        rows.append(c)

    def yn(v):
        if v is True:
            return "✅"
        if v is False:
            return "❌"
        return "—"

    lines = [
        "# Chapter HTML audit ledger",
        "",
        f"Updated: {ledger.get('updated_at', '')}",
        "",
        "Registry migration chapter verification: build HTML → raw HTML scan → LEGO/prose/registry checks.",
        "",
        "| Vol | Chapter | Status | Build | HTML | Math | Spurious | LEGO | Prose | Registry | s | Notes |",
        "|-----|---------|--------|-------|------|------|----------|------|-------|----------|---|-------|",
    ]
    pass_n = fail_n = pending_n = 0
    for c in rows:
        st = c.get("status", "pending")
        if st == "pass":
            pass_n += 1
        elif st == "fail":
            fail_n += 1
        else:
            pending_n += 1
        notes = (c.get("notes") or "").replace("|", "\\|")[:80]
        lines.append(
            f"| {c['vol']} | {c['chapter']} | **{st}** | {yn(c.get('build_ok'))} | "
            f"{yn(not c.get('html_error_hits')) if c.get('html') else '—'} | "
            f"{yn(c.get('html_math_artifacts_clean'))} | "
            f"{yn(c.get('html_spurious_clean'))} | {yn(c.get('lego_focal_ok'))} | "
            f"{yn(c.get('prose_ok'))} | {yn(c.get('registry_ok'))} | "
            f"{c.get('build_seconds', '')} | {notes} |"
        )
    lines.extend([
        "",
        f"**Summary:** {pass_n} pass / {fail_n} fail / {pending_n} pending / {len(rows)} total",
        "",
        "Re-run one chapter: `python3 book/tools/audit/chapter_html_verify.py --vol1 training`",
        "",
    ])
    LEDGER_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vol1", nargs="*", metavar="CHAPTER", help="Vol1 chapter(s) by short name")
    parser.add_argument("--vol2", nargs="*", metavar="CHAPTER", help="Vol2 chapter(s) by short name")
    parser.add_argument("--all", action="store_true", help="All chapters in selected volume(s)")
    parser.add_argument("--list", action="store_true", help="List chapters")
    parser.add_argument("--report", action="store_true", help="Regenerate markdown table only")
    parser.add_argument("--skip-build", action="store_true", help="Audit archived HTML only")
    args = parser.parse_args()

    if args.list:
        for vol, paths in CHAPTERS.items():
            print(f"\n{vol}:")
            for p in paths:
                print(f"  {p.split('/')[-1]}")
        return 0

    if args.report:
        save_ledger(load_ledger())
        print(f"Wrote {LEDGER_MD}")
        return 0

    targets: list[tuple[str, str]] = []
    if args.all:
        if args.vol1 is not None or (not args.vol1 and not args.vol2):
            targets.extend(("vol1", p) for p in CHAPTERS["vol1"])
        if args.vol2 is not None or (not args.vol1 and not args.vol2):
            targets.extend(("vol2", p) for p in CHAPTERS["vol2"])
    else:
        for vol_flag, vol in [(args.vol1, "vol1"), (args.vol2, "vol2")]:
            if vol_flag is None:
                continue
            names = vol_flag if vol_flag else []
            if not names:
                parser.error(f"Provide chapter names or --all for {vol}")
            path_map = {p.split("/")[-1]: p for p in CHAPTERS[vol]}
            for name in names:
                if name not in path_map:
                    parser.error(f"Unknown {vol} chapter: {name}")
                targets.append((vol, path_map[name]))

    if not targets:
        parser.print_help()
        return 2

    ledger = load_ledger()
    ledger.setdefault("chapters", {})
    fail = 0
    for vol, ch_path in targets:
        cid = _chapter_id(vol, ch_path)
        print(f"\n=== {cid} ===")
        result = verify_chapter(vol, ch_path, skip_build=args.skip_build)
        ledger["chapters"][cid] = asdict(result)
        save_ledger(ledger)
        print(f"  {result.status.upper()} ({result.build_seconds}s) — {result.notes[:120]}")
        if result.status == "fail":
            fail += 1

    print(f"\nLedger: {LEDGER_JSON}")
    print(f"Table:  {LEDGER_MD}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
