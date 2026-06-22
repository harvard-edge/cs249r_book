#!/usr/bin/env python3
"""Run render-guarantee stack on binder debug HTML artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "audit"))
from audit_math_rendering import visible_text_from_html, scan_html  # noqa: E402

INLINE_PY = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")
ARTIFACTS = REPO / "book/tools/audit/artifacts"


@dataclass
class LayerResult:
    layer: str
    ok: bool
    passed: int = 0
    total: int = 0
    failures: list = field(default_factory=list)


def _latest_debug_artifacts(vol: str) -> Path:
    root = REPO / "book/quarto/_build/debug" / vol / "html"
    runs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if not runs:
        raise SystemExit(f"No binder debug HTML run for {vol}")
    return runs[-1] / "phase1" / "artifacts"


def _stem(name: str) -> str:
    return re.sub(r"^\d+_", "", Path(name).stem)


def sync_artifacts_to_html_audit(vol: str, artifact_dir: Path) -> Path:
    dest_root = REPO / "book/quarto/_build/html-audit" / vol
    dest_root.mkdir(parents=True, exist_ok=True)
    for html in artifact_dir.glob("*.html"):
        (dest_root / f"{_stem(html.name)}.html").write_bytes(html.read_bytes())
    return dest_root


def _chapters_with_inline_python(vol: str) -> list[Path]:
    base = REPO / "book/quarto/contents" / vol
    out = []
    for qmd in sorted(base.rglob("*.qmd")):
        if qmd.name.startswith("_") and "notation" not in qmd.name:
            continue
        if INLINE_PY.search(qmd.read_text(encoding="utf-8", errors="replace")):
            out.append(qmd)
    return out


def run_prose_layer(vol: str) -> LayerResult:
    res = LayerResult(layer="audit_prose", ok=True)
    env = {**os.environ, "PYTHONPATH": str(REPO / "mlsysim"), "MPLBACKEND": "Agg"}
    script = REPO / "book/tools/audit/fmt/audit_prose.py"
    chapters = _chapters_with_inline_python(vol)
    res.total = len(chapters)
    for qmd in chapters:
        proc = subprocess.run(
            [sys.executable, "-W", "ignore::UserWarning", str(script), str(qmd), "--flagged-only"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            res.ok = False
            res.failures.append({"chapter": qmd.parent.name, "error": (proc.stderr or proc.stdout or "")[:300]})
        else:
            res.passed += 1
    return res


def run_static_html_layer(vol: str, html_dir: Path) -> LayerResult:
    res = LayerResult(layer="static_html", ok=True)
    audit_html = REPO / "book/tools/audit/fmt/audit_html.py"
    html_files = sorted(html_dir.glob("*.html"))
    res.total = len(html_files)
    for html in html_files:
        stem = _stem(html.name)
        issues = []
        proc = subprocess.run([sys.executable, str(audit_html), str(html)], capture_output=True, text=True)
        if proc.returncode != 0 or "CLEAN" not in (proc.stdout or ""):
            issues.append("spurious_.0")
        text = visible_text_from_html(html.read_text(encoding="utf-8", errors="replace"))
        leaks = scan_html(text)
        if leaks:
            issues.append(f"math_leaks:{len(leaks)}")
        for pat in (r"\{python\}", r"Traceback \(most recent call last\)", r"NameError:"):
            if re.search(pat, text, re.I):
                issues.append(pat)
        if issues:
            res.ok = False
            res.failures.append({"chapter": stem, "issues": issues})
        else:
            res.passed += 1
    return res


def run_lego_html_layer(vol: str) -> LayerResult:
    res = LayerResult(layer="audit_lego_html", ok=True)
    env = {**os.environ, "PYTHONPATH": str(REPO / "mlsysim"), "MPLBACKEND": "Agg"}
    report_path = ARTIFACTS / f"lego_html_{vol}.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "book/tools/audit/fmt/audit_lego_html.py"), "--report", str(report_path)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=3600,
    )
    data = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else []
    vol_rows = [r for r in data if r.get("vol") == vol]
    res.total = len(vol_rows)
    for row in vol_rows:
        if row.get("status") == "PASS":
            res.passed += 1
        else:
            res.ok = False
            miss = []
            for b in row.get("blocks", []):
                miss.extend(x["ref"] for x in b.get("refs", []) if x.get("status") == "FAIL")
            res.failures.append({"chapter": row.get("chapter"), "status": row.get("status"), "missing": miss[:6]})
    if proc.returncode != 0 and res.passed < res.total:
        res.ok = False
    return res


def run_playwright_layer(vol: str, artifact_dir: Path) -> LayerResult:
    res = LayerResult(layer="playwright", ok=True)
    report = ARTIFACTS / f"playwright_{vol}.json"
    proc = subprocess.run(
        ["uv", "run", "--with", "playwright", "python3",
         str(REPO / "book/tools/audit/render_playwright_verify.py"),
         "--vol", vol, "--artifact-dir", str(artifact_dir), "--report", str(report)],
        cwd=REPO, capture_output=True, text=True, timeout=3600,
    )
    data = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else []
    res.total = len(data)
    res.passed = sum(1 for r in data if r.get("ok"))
    for row in data:
        if not row.get("ok"):
            res.ok = False
            res.failures.append({
                "chapter": row.get("chapter"),
                "leaks": row.get("leak_count"),
                "ref_failures": len(row.get("ref_failures", [])),
            })
    if proc.returncode != 0 and res.passed < res.total:
        res.ok = False
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol1", action="store_true")
    ap.add_argument("--vol2", action="store_true")
    ap.add_argument("--sync-artifacts", action="store_true")
    ap.add_argument("--artifact-dir", type=Path)
    ap.add_argument("--skip-playwright", action="store_true")
    ap.add_argument("--skip-prose", action="store_true")
    ap.add_argument("--skip-lego-html", action="store_true")
    args = ap.parse_args()

    vols = []
    if args.vol2:
        vols.append("vol2")
    if args.vol1 or not vols:
        vols.insert(0, "vol1")

    overall_ok = True
    all_layers = {}
    t0 = time.time()

    for vol in vols:
        print(f"\n=== {vol} ===", flush=True)
        artifact_dir = args.artifact_dir or _latest_debug_artifacts(vol)
        if args.sync_artifacts:
            sync_artifacts_to_html_audit(vol, artifact_dir)
        layers = []
        if not args.skip_prose:
            print("Layer 1: audit_prose", flush=True)
            layers.append(run_prose_layer(vol))
        print("Layer 2: static HTML", flush=True)
        layers.append(run_static_html_layer(vol, artifact_dir))
        if not args.skip_lego_html:
            print("Layer 3: audit_lego_html", flush=True)
            layers.append(run_lego_html_layer(vol))
        if not args.skip_playwright:
            print("Layer 4: playwright", flush=True)
            layers.append(run_playwright_layer(vol, artifact_dir))
        for lr in layers:
            print(f"  {'PASS' if lr.ok else 'FAIL'} {lr.layer}: {lr.passed}/{lr.total}")
            for f in lr.failures[:8]:
                print(f"    {f}")
            if not lr.ok:
                overall_ok = False
        all_layers[vol] = [asdict(lr) for lr in layers]

    ledger = ARTIFACTS / "render_guarantee.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"ok": overall_ok, "volumes": all_layers, "elapsed_s": round(time.time()-t0,1)}, indent=2))
    print(f"\nLedger: {ledger} -> {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
