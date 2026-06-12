#!/usr/bin/env python3
"""Build chapter PDF, audit TeX/PDF output, and track pass/fail in a ledger.

Usage (repo root)::

    python3 book/tools/audit/chapter_pdf_verify.py --list
    python3 book/tools/audit/chapter_pdf_verify.py --vol1 introduction
    python3 book/tools/audit/chapter_pdf_verify.py --vol1 --all
    python3 book/tools/audit/chapter_pdf_verify.py --report

Ledger: book/tools/audit/artifacts/chapter_pdf_audit.json
Table:   book/tools/audit/artifacts/chapter_pdf_audit.md

Builds one chapter at a time via ``./book/binder build pdf --volN <chapter>``.
Archives PDF + keep-tex output under ``book/quarto/_build/pdf-audit/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BOOK_DIR = REPO_ROOT / "book/quarto"
LEDGER_JSON = REPO_ROOT / "book/tools/audit/artifacts/chapter_pdf_audit.json"
LEDGER_MD = REPO_ROOT / "book/tools/audit/artifacts/chapter_pdf_audit.md"

PDF_NAMES = {
    "vol1": "Machine-Learning-Systems-Vol1.pdf",
    "vol2": "Machine-Learning-Systems-Vol2.pdf",
}
TEX_NAMES = {
    "vol1": "Machine-Learning-Systems-Vol1.tex",
    "vol2": "Machine-Learning-Systems-Vol2.tex",
}

# Same inline-Python chapter list as chapter_html_verify.py
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
        "backmatter/appendix_c3",
        "backmatter/appendix_assumptions",
    ],
}

TEX_ERROR_PATTERNS = [
    re.compile(r"! LaTeX Error", re.I),
    re.compile(r"Undefined control sequence", re.I),
    re.compile(r"Missing \$ inserted", re.I),
    re.compile(r"Package .+ Error", re.I),
    re.compile(r"Emergency stop", re.I),
    re.compile(r"Runaway argument", re.I),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"NameError:", re.I),
    re.compile(r"AttributeError:", re.I),
    re.compile(r"ModuleNotFoundError:", re.I),
    re.compile(r"Error rendering", re.I),
    re.compile(r"`\{python\}"),
]

PDF_TEXT_ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"NameError:", re.I),
    re.compile(r"AttributeError:", re.I),
    re.compile(r"ModuleNotFoundError:", re.I),
    re.compile(r"Error rendering", re.I),
    re.compile(r"\b(?:Figure|Table|Section|Equation)\s+\?\?+", re.I),
]

PDF_TEXT_WARN_PATTERNS = [
    re.compile(r"\?@(?:fig|sec|tbl|eq)-"),
    re.compile(r"@(?:fig|sec|tbl|eq)-[A-Za-z0-9_:-]+"),
]

MATH_ENVS = (
    "equation", "equation*", "align", "align*", "aligned", "gather", "gather*",
    "multline", "multline*", "split", "flalign", "flalign*", "eqnarray", "eqnarray*",
)


@dataclass
class ChapterResult:
    vol: str
    chapter: str
    qmd: str
    pdf: str
    tex: str
    status: str  # pass | fail | pending | skip
    build_ok: bool = False
    build_seconds: float = 0.0
    pdf_bytes: int = 0
    tex_error_hits: list[str] = field(default_factory=list)
    pdf_text_hits: list[str] = field(default_factory=list)
    pdf_text_warnings: list[str] = field(default_factory=list)
    math_imbalance: list[str] = field(default_factory=list)
    display_math_blocks: int = 0
    inline_math_lines: int = 0
    prose_ok: bool | None = None
    registry_ok: bool | None = None
    notes: str = ""
    updated_at: str = ""


def _chapter_id(vol: str, ch_path: str) -> str:
    return f"{vol}/{ch_path.split('/')[-1]}"


def _qmd_path(vol: str, ch_path: str) -> Path:
    return REPO_ROOT / "book/quarto/contents" / vol / f"{ch_path}.qmd"


def _build_pdf(vol: str, ch_path: str) -> tuple[bool, float, str]:
    name = ch_path.split("/")[-1]
    log = Path(f"/tmp/render_pdf_{vol}_{name}.log")
    t0 = time.monotonic()
    proc = subprocess.run(
        ["./book/binder", "build", "pdf", f"--{vol}", name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - t0
    log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    return proc.returncode == 0, elapsed, str(log)


def _live_pdf(vol: str) -> Path:
    return BOOK_DIR / "_build" / f"pdf-{vol}" / PDF_NAMES[vol]


def _live_tex(vol: str) -> Path:
    return BOOK_DIR / TEX_NAMES[vol]


def _archive_artifacts(vol: str, ch_path: str, live_pdf: Path, live_tex: Path) -> tuple[Path, Path]:
    name = ch_path.split("/")[-1]
    archive_dir = BOOK_DIR / "_build/pdf-audit" / vol
    archive_dir.mkdir(parents=True, exist_ok=True)
    pdf_dest = archive_dir / f"{name}.pdf"
    tex_dest = archive_dir / f"{name}.tex"
    shutil.copy2(live_pdf, pdf_dest)
    shutil.copy2(live_tex, tex_dest)
    return pdf_dest, tex_dest


def _scan_tex(tex: Path) -> tuple[list[str], list[str], int, int]:
    text = tex.read_text(encoding="utf-8", errors="replace")
    active_text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("%")
    )
    hits: list[str] = []
    for pat in TEX_ERROR_PATTERNS:
        if pat.search(active_text):
            hits.append(pat.pattern)

    imbalances: list[str] = []
    for env in MATH_ENVS:
        begin_n = len(re.findall(rf"\\begin\{{{re.escape(env)}\}}", active_text))
        end_n = len(re.findall(rf"\\end\{{{re.escape(env)}\}}", active_text))
        if begin_n != end_n:
            imbalances.append(f"{env}: begin={begin_n} end={end_n}")

    display_start = len(
        re.findall(
            r"\\begin\{(?:equation|align|gather|multline|flalign|eqnarray)",
            active_text,
        )
    )
    bracket_begin = len(re.findall(r"(?<!\\)\\\[", active_text))
    bracket_end = len(re.findall(r"(?<!\\)\\\]", active_text))
    if bracket_begin != bracket_end:
        imbalances.append(f"display brackets: [={bracket_begin} ]={bracket_end}")

    inline_paren = sum(1 for ln in active_text.splitlines() if "\\(" in ln or "\\)" in ln)
    inline_dollar = sum(1 for ln in active_text.splitlines() if re.search(r"(?<!\$)\$(?!\$)", ln))

    return hits, imbalances, display_start + max(0, active_text.count("\\[")), inline_paren + inline_dollar


def _scan_pdf_text(pdf: Path) -> tuple[list[str], list[str]]:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return ["pdftotext failed"], []
    body = proc.stdout or ""
    hits: list[str] = []
    warnings: list[str] = []
    for pat in PDF_TEXT_ERROR_PATTERNS:
        if pat.search(body):
            hits.append(pat.pattern)
    for pat in PDF_TEXT_WARN_PATTERNS:
        if pat.search(body):
            warnings.append(pat.pattern)
    return hits, warnings


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
        pdf="",
        tex="",
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

    prose_ok, prose_msg = _prose_exec(qmd)
    res.prose_ok = prose_ok
    if not prose_ok:
        res.notes += f"prose: {prose_msg}; "

    if skip_build:
        archive_pdf = BOOK_DIR / "_build/pdf-audit" / vol / f"{name}.pdf"
        archive_tex = BOOK_DIR / "_build/pdf-audit" / vol / f"{name}.tex"
        if not archive_pdf.is_file() or not archive_tex.is_file():
            res.status = "pending"
            res.notes += "no archived PDF/TEX (--skip-build)"
            return res
        pdf, tex = archive_pdf, archive_tex
        res.build_ok = True
    else:
        build_ok, secs, log = _build_pdf(vol, ch_path)
        res.build_ok = build_ok
        res.build_seconds = round(secs, 1)
        if not build_ok:
            res.status = "fail"
            res.notes += f"build failed ({log}); "
            return res
        live_pdf = _live_pdf(vol)
        live_tex = _live_tex(vol)
        if not live_pdf.is_file() or not live_tex.is_file():
            res.status = "fail"
            res.notes += "PDF or TEX missing after build; "
            return res
        pdf, tex = _archive_artifacts(vol, ch_path, live_pdf, live_tex)

    res.pdf = str(pdf.relative_to(REPO_ROOT))
    res.tex = str(tex.relative_to(REPO_ROOT))
    res.pdf_bytes = pdf.stat().st_size

    tex_hits, math_imb, display_n, inline_n = _scan_tex(tex)
    res.tex_error_hits = tex_hits
    res.math_imbalance = math_imb
    res.display_math_blocks = display_n
    res.inline_math_lines = inline_n

    if shutil.which("pdftotext"):
        pdf_hits, pdf_warn = _scan_pdf_text(pdf)
        res.pdf_text_hits = pdf_hits
        res.pdf_text_warnings = pdf_warn
        if pdf_warn:
            res.notes += f"xref warnings (single-chapter PDF): {len(pdf_warn)} patterns; "
    else:
        res.notes += "pdftotext unavailable; "

    if res.tex_error_hits:
        res.notes += f"tex errors: {res.tex_error_hits}; "
    if res.math_imbalance:
        res.notes += f"math imbalance: {res.math_imbalance}; "
    if res.pdf_text_hits:
        res.notes += f"pdf errors: {res.pdf_text_hits}; "

    passed = (
        res.build_ok
        and not res.tex_error_hits
        and not res.math_imbalance
        and not res.pdf_text_hits
        and res.prose_ok
        and res.registry_ok
    )
    res.status = "pass" if passed else "fail"
    if passed:
        res.notes = (
            f"OK — PDF+TeX clean; display_math={display_n}, inline_math_lines={inline_n}"
        )
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
    rows = [ledger["chapters"][k] for k in sorted(ledger.get("chapters", {}).keys())]

    def yn(v):
        if v is True:
            return "✅"
        if v is False:
            return "❌"
        return "—"

    lines = [
        "# Chapter PDF audit ledger",
        "",
        f"Updated: {ledger.get('updated_at', '')}",
        "",
        "Per-chapter PDF build → keep-tex scan (math env balance + error patterns) → pdftotext scan.",
        "",
        "| Vol | Chapter | Status | Build | TeX | Math | PDF text | Prose | Registry | s | KB | Notes |",
        "|-----|---------|--------|-------|-----|------|----------|-------|----------|---|----|-------|",
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
        kb = round(c.get("pdf_bytes", 0) / 1024) if c.get("pdf_bytes") else ""
        notes = (c.get("notes") or "").replace("|", "\\|")[:70]
        lines.append(
            f"| {c['vol']} | {c['chapter']} | **{st}** | {yn(c.get('build_ok'))} | "
            f"{yn(not c.get('tex_error_hits')) if c.get('tex') else '—'} | "
            f"{yn(not c.get('math_imbalance')) if c.get('tex') else '—'} | "
            f"{yn(not c.get('pdf_text_hits')) if c.get('pdf') else '—'} | "
            f"{yn(c.get('prose_ok'))} | {yn(c.get('registry_ok'))} | "
            f"{c.get('build_seconds', '')} | {kb} | {notes} |"
        )
    lines.extend([
        "",
        f"**Summary:** {pass_n} pass / {fail_n} fail / {pending_n} pending / {len(rows)} total",
        "",
        "Re-run one chapter: `python3 book/tools/audit/chapter_pdf_verify.py --vol1 training`",
        "",
        "Archived artifacts: `book/quarto/_build/pdf-audit/<vol>/<chapter>.{pdf,tex}`",
        "",
    ])
    LEDGER_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vol1", nargs="*", metavar="CHAPTER")
    parser.add_argument("--vol2", nargs="*", metavar="CHAPTER")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
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
    requested_any_vol = args.vol1 is not None or args.vol2 is not None
    if args.all:
        if args.vol1 is not None or not requested_any_vol:
            targets.extend(("vol1", p) for p in CHAPTERS["vol1"])
        if args.vol2 is not None or not requested_any_vol:
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
        print(
            f"  {result.status.upper()} ({result.build_seconds}s, "
            f"math={result.display_math_blocks}d+{result.inline_math_lines}i) — "
            f"{result.notes[:100]}"
        )
        if result.status == "fail":
            fail += 1

    print(f"\nLedger: {LEDGER_JSON}")
    print(f"Table:  {LEDGER_MD}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
