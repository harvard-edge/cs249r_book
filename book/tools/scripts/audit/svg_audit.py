#!/usr/bin/env python3
"""svg_audit.py v2 — batched, phase-based SVG quality audit for 238 figures.

Architecture (Approach C, iteration-based, holistic):

  PHASE 1: BATCHED REVIEW  (8 figures per Gemini call)
    Input:  rendered PNGs of N figures
    Prompt: "For each figure answer design-sound + aesthetic-clean +
             top-3 problems. Apply the book's SVG style rules."
    Output: per-figure JSON with {clean, design_sound, problems, recommendation}

  PHASE 2: INDIVIDUAL REWRITE  (1 figure per Gemini call)
    Input:  SVG source + problems + rejected_attempts memory
    Prompt: "Content-preserving, geometry-fluid rewrite. Preserve every text
             label, colour, and semantic element; you have full latitude
             over positions, sizes, and layout."
    Output: complete revised SVG

  GATE: Content audit (local, no Gemini)
    - Extract all <text> content from BEFORE; same from AFTER.
    - If any text was changed/dropped/reordered, REJECT the candidate.
    - Check canonical SVG hash; if identical to cycle-2-ago, mark oscillating.

  PHASE 3: BATCHED VERIFY  (8 before/after pairs per Gemini call)
    Input:  16 PNG attachments (BEFORE and AFTER for 8 figures)
    Prompt: "For each figure: were the listed problems resolved? Any regression?"
    Output: per-figure JSON with {next_action: accept|revert|iterate, regression}

  APPLY per figure:
    accept  -> git add + commit atomically; candidate becomes baseline
    iterate -> candidate stays, keep trying next cycle
    revert  -> git checkout SVG, log rejected_attempts, try different approach
              next cycle

  LOOP PHASES 1-3 up to MAX_CYCLES (4). Terminate when figure is clean or cap hit.

State persists to .audit/state.json atomically after every figure transition,
so a killed run resumes exactly where it left off.

Hard-pinned model: gemini-3.1-pro-preview.
Budget cap:        12 h wall-time at .audit/gemini-budget.txt.

Usage:
    python3 svg_audit.py audit --scope smoke --skip-drafts     # diagnose + fix
    python3 svg_audit.py audit --scope full  --skip-drafts
    python3 svg_audit.py diagnose --scope full                  # review only
    python3 svg_audit.py fix --scope full  --skip-drafts        # fix after diagnose
    python3 svg_audit.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMINI_MODEL          = "gemini-3.1-pro-preview"
GEMINI_TIMEOUT_SEC    = 420      # rewrite prompts carry full SVG source
RETRY_BACKOFFS_SEC    = (2, 8, 24)
WALL_TIME_BUDGET_SEC  = 24 * 60 * 60   # 24h safety cap for overnight full sweep
RENDER_WIDTH_PX       = 600
REVIEW_BATCH_SIZE     = 8
VERIFY_BATCH_SIZE     = 8        # 16 images per call
MAX_CYCLES            = 2        # full-sweep default; smoke showed cycles 1-2 capture bulk of improvements

TERMINAL_STATUSES = {"clean", "fixed", "needs-human", "render-failed", "skipped-draft"}

SVG_NS = "http://www.w3.org/2000/svg"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def worktree_root() -> Path:
    return Path(__file__).resolve().parents[4]

def audit_dir() -> Path:      return worktree_root() / ".audit"
def png_dir() -> Path:        return audit_dir() / "png"
def before_dir() -> Path:     return audit_dir() / "png_before"
def after_dir() -> Path:      return audit_dir() / "png_after"
def backups_dir() -> Path:    return audit_dir() / "svg_backups"
def logs_dir() -> Path:       return audit_dir() / "logs"
def state_path() -> Path:     return audit_dir() / "state.json"
def budget_path() -> Path:    return audit_dir() / "gemini-budget.txt"
def inventory_path() -> Path: return audit_dir() / "svg-inventory.txt"
def report_path() -> Path:    return audit_dir() / "SVG_AUDIT_REPORT.md"

def ensure_dirs() -> None:
    for d in (audit_dir(), png_dir(), before_dir(), after_dir(),
              backups_dir(), logs_dir()):
        d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Figure:
    svg_path: str
    chapter: str
    is_draft: bool = False
    # Rendering
    png_path: str = ""
    render_ok: bool = False
    render_error: str = ""
    # Current review state
    clean_this_cycle: bool = False
    design_sound: bool = True
    current_problems: List[str] = field(default_factory=list)
    recommendation: str = "pending"     # pending | polish | redraw-required | clean
    overall_assessment: str = ""
    # Loop state
    cycles_completed: int = 0
    rejected_attempts: List[str] = field(default_factory=list)
    canonical_hashes: List[str] = field(default_factory=list)  # for oscillation detection
    # Commits
    commit_shas: List[str] = field(default_factory=list)
    # Terminal
    status: str = "pending"
    final_notes: str = ""
    # Audit trail
    iterations: List[Dict[str, Any]] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
CHAPTER_RE = re.compile(r"book/quarto/contents/(vol[12]/[^/]+)/images/svg/")

def _chapter_of(svg_path: str) -> str:
    m = CHAPTER_RE.search(svg_path)
    return m.group(1) if m else "unknown"

def build_inventory() -> List[Figure]:
    p = inventory_path()
    if not p.exists():
        raise FileNotFoundError(f"Missing inventory: {p}")
    figs = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line: continue
        name = Path(line).name
        figs.append(Figure(svg_path=line, chapter=_chapter_of(line),
                           is_draft=name.startswith("_")))
    return figs

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def _rsvg(svg: Path, png: Path, width: int) -> Optional[str]:
    try:
        cp = subprocess.run(
            ["rsvg-convert", f"--width={width}", "--keep-aspect-ratio",
             "--output", str(png), str(svg)],
            check=False, capture_output=True, text=True, timeout=30)
    except Exception as e:
        return f"rsvg exception: {e}"
    if cp.returncode != 0:
        return f"rsvg rc={cp.returncode} {cp.stderr[:160]}"
    if not png.exists() or png.stat().st_size == 0:
        return "rsvg empty"
    return None

def _inkscape(svg: Path, png: Path, width: int) -> Optional[str]:
    if not shutil.which("inkscape"): return "no inkscape"
    try:
        cp = subprocess.run(
            ["inkscape", str(svg), "--export-type=png",
             f"--export-width={width}", f"--export-filename={png}"],
            check=False, capture_output=True, text=True, timeout=90)
    except Exception as e:
        return f"inkscape exception: {e}"
    if cp.returncode != 0:
        return f"inkscape rc={cp.returncode} {cp.stderr[:160]}"
    if not png.exists() or png.stat().st_size == 0:
        return "inkscape empty"
    return None

def render_svg(svg_rel: str, png_rel: str, width: int = RENDER_WIDTH_PX) -> Optional[str]:
    svg_abs = worktree_root() / svg_rel
    png_abs = worktree_root() / png_rel
    png_abs.parent.mkdir(parents=True, exist_ok=True)
    err = _rsvg(svg_abs, png_abs, width)
    if err is None: return None
    err2 = _inkscape(svg_abs, png_abs, width)
    if err2 is None: return None
    return f"{err}; {err2}"

def png_rel_for(fig: Figure, variant: str = "") -> str:
    subdir = {"before": before_dir(), "after": after_dir()}.get(variant, png_dir())
    name = f"{fig.chapter.replace('/', '__')}__{Path(fig.svg_path).stem}.png"
    return str((subdir / name).relative_to(worktree_root()))

# ---------------------------------------------------------------------------
# Gemini budget + invocation
# ---------------------------------------------------------------------------
def _read_budget() -> float:
    if not budget_path().exists(): return 0.0
    try: return float(budget_path().read_text().strip() or "0")
    except Exception: return 0.0

def _add_budget(secs: float) -> float:
    budget_path().parent.mkdir(parents=True, exist_ok=True)
    total = _read_budget() + max(0.0, secs)
    budget_path().write_text(f"{total:.2f}\n")
    return total

def _budget_exhausted() -> bool: return _read_budget() >= WALL_TIME_BUDGET_SEC

def gemini_call(prompt: str, timeout: int = GEMINI_TIMEOUT_SEC,
                log_prefix: str = "") -> Dict[str, Any]:
    """Returns {stdout, error, wall_time_sec}."""
    if _budget_exhausted():
        return {"stdout": "", "error": "budget-exhausted", "wall_time_sec": 0.0}
    if log_prefix:
        (logs_dir() / f"{log_prefix}.prompt.txt").write_text(prompt)
    started = time.monotonic()
    last_err = ""
    for attempt, backoff in enumerate(RETRY_BACKOFFS_SEC, start=1):
        try:
            cp = subprocess.run(
                ["gemini", "-m", GEMINI_MODEL, "-o", "json",
                 "--approval-mode", "plan", "-p", prompt],
                cwd=str(worktree_root()),
                check=False, text=True, capture_output=True, timeout=timeout)
            if cp.returncode == 0 and cp.stdout.strip():
                wall = time.monotonic() - started
                _add_budget(wall)
                if log_prefix:
                    (logs_dir() / f"{log_prefix}.response.json").write_text(cp.stdout)
                return {"stdout": cp.stdout, "error": "", "wall_time_sec": wall}
            last_err = f"rc={cp.returncode} {cp.stderr[:300]}"
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        except Exception as e:
            last_err = f"exception:{e}"
        if attempt < len(RETRY_BACKOFFS_SEC):
            time.sleep(backoff)
    wall = time.monotonic() - started
    _add_budget(wall)
    if log_prefix:
        (logs_dir() / f"{log_prefix}.response.json").write_text(
            json.dumps({"error": last_err, "wall": wall}, indent=2))
    return {"stdout": "", "error": last_err, "wall_time_sec": wall}

def _unwrap_response(stdout: str) -> str:
    s = stdout.strip()
    try:
        env = json.loads(s)
        if isinstance(env, dict):
            return (env.get("response") or env.get("text") or env.get("output") or s)
    except Exception:
        pass
    return s

def _parse_json_body(body: str) -> Any:
    body = body.strip()
    fence = re.match(r"^```(?:json|xml|svg)?\s*(.*?)\s*```$", body, re.S)
    if fence: body = fence.group(1)
    try: return json.loads(body)
    except Exception: pass
    # Find first {...} or [...] block
    for opener, closer in (("{", "}"), ("[", "]")):
        m = re.search(re.escape(opener) + r".*" + re.escape(closer), body, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: continue
    return None

# ---------------------------------------------------------------------------
# Content audit (LOCAL, no Gemini) — THE critical content-preservation gate
# ---------------------------------------------------------------------------
def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag

def extract_text_content(svg_src: str) -> List[str]:
    """Return a sorted list of all text content from <text> and <tspan> nodes.
    Normalises whitespace so trivial reflow doesn't trip the audit."""
    try:
        root = ET.fromstring(svg_src)
    except Exception:
        return []
    texts = []
    for el in root.iter():
        if _strip_ns(el.tag) in ("text", "tspan"):
            raw = "".join(el.itertext()) if el.text is None else (el.text or "")
            # Normalise: collapse whitespace, strip
            norm = " ".join(raw.split()).strip()
            if norm: texts.append(norm)
    texts.sort()
    return texts

def content_audit_ok(before_svg: str, after_svg: str) -> (bool, str):
    """Hard gate: every text element in BEFORE must survive into AFTER.
    Returns (ok, reason-if-not)."""
    before_texts = extract_text_content(before_svg)
    after_texts  = extract_text_content(after_svg)
    if not before_texts:
        # Trivial case: no text to preserve
        return (True, "")
    missing = [t for t in before_texts if t not in after_texts]
    if missing:
        sample = "; ".join(missing[:3])
        more = f" (and {len(missing)-3} more)" if len(missing) > 3 else ""
        return (False, f"text content lost: {sample}{more}")
    # Optional: check added content (Gemini shouldn't add new text either)
    added = [t for t in after_texts if t not in before_texts]
    if len(added) > 2:  # tolerate tiny additions (e.g. SVG title/desc meta)
        sample = "; ".join(added[:3])
        return (False, f"unexpected new text: {sample}")
    return (True, "")

def canonical_svg_hash(svg_src: str) -> str:
    """Hash a canonicalised form of the SVG so whitespace/attribute-order
    changes don't create false non-convergence signals."""
    try:
        root = ET.fromstring(svg_src)
    except Exception:
        return hashlib.sha256(svg_src.encode()).hexdigest()[:16]
    # Sort attribute keys per element
    def _canon(el: ET.Element) -> str:
        tag = _strip_ns(el.tag)
        attrs = ",".join(f"{k}={v}" for k, v in sorted(el.attrib.items()))
        text = (el.text or "").strip()
        children = "".join(_canon(c) for c in el)
        return f"<{tag} {attrs}>{text}{children}</{tag}>"
    canon = _canon(root)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# PHASE 1: Batched review prompt
# ---------------------------------------------------------------------------
RULES_SUMMARY = """
ML SYSTEMS TEXTBOOK SVG STYLE RULES (condensed — full rules in .claude/rules/svg-style.md):

SEMANTIC PALETTE (any colour outside this is a defect unless domain-justified):
  Compute blue    fill=#cfe2f3 stroke=#4a90c4   — GPU, forward/backward, inference
  Data green      fill=#d4edda stroke=#3d9e5a   — data flow, healthy paths, memory
  Routing orange  fill=#fdebd0 stroke=#c87b2a   — scheduler, load balancer, batching
  Error red       fill=#f9d6d5 stroke=#c44      — bottleneck, waste, decode-bound
  MIT red accent  #a31f34                       — GPU badges, SLA pills, critical annotations
  Neutral         fill=#f7f7f7 stroke=#bbb      — background regions, idle state
  AllReduce green fill=#e8f5e9 stroke=#2d7a2d   — deeper sync green

SHAPE DISCIPLINE:
  - Rectangles are the default primitive (rx=4). Do not mix shape languages.
  - Uniform sizing within a role: three "Worker" boxes are identical width & height.
  - No decorative shapes (stars, banners, speech bubbles, scrolls).

ARROW DISCIPLINE:
  - Every arrow either has a head or explicitly does not. No ambiguity.
  - Arrows land at box borders — never inside, never short.
  - Orthogonal routing (H/V). Diagonals only for physical network topologies.
  - Parallel arrows evenly spaced (10px) along the source edge.
  - Stroke-width 1.5 primary / 1.2 secondary / 0.8 tertiary — semantic, not decorative.

ALIGNMENT & GRID:
  - All elements snap to 10px grid. Coordinates are integers or one decimal place.
  - Baselines match across panels. Column widths match in comparison figures.

WHITESPACE:
  - >=20px padding between any box and panel edge.
  - >=40px gutter between side-by-side panels.
  - >=4px clearance between text bounding box and any line or border. Labels NEVER touch strokes.

BORDER SEMANTICS:
  - Solid = realised/present. Dashed = optional/planned. Dotted = approximate/boundary.
  - Fill and stroke come from the same semantic family (blue fill + blue stroke).

CHARTJUNK AVOIDANCE:
  - No drop-shadows, gradients, 3-D, bevels, glows. Textbook, not keynote.
  - Every element carries meaning or does not belong.

ACCESSIBILITY:
  - Colour is never the only carrier of meaning; add label/shape/pattern cue.
  - Text on coloured fills meets WCAG AA contrast (4.5:1).

ITERATIVE IMPROVEMENT — "CONTENT-PRESERVING, GEOMETRY-FLUID":
  - Content is INVARIANT: every text label, colour role, and semantic element must survive.
  - Geometry is FLUID: positions, sizes, viewBox, spacings may be freely adjusted.
  - If a label overlaps a line, move the label. If cramped, widen the viewBox.
  - Two-pass: (1) is the DESIGN sound? (2) is the figure AESTHETICALLY clean?
    A figure passes only when both are true at the same time.
"""

REVIEW_PROMPT_HEADER = (
    "You are the definitive figure-quality reviewer for an MIT Press machine "
    "learning systems textbook. Apply the book's SVG style rules (below) "
    "rigorously. For EACH figure in the batch, answer both questions: "
    "(1) is the DESIGN sound — does the layout serve the message, or does "
    "the structure itself need restructuring? (2) is the figure "
    "AESTHETICALLY clean — no overlaps, boxes balanced, labels clear of "
    "strokes, whitespace used purposefully?\n\n"
    "Return ONLY a JSON array (no prose, no fence) with one object per "
    "figure in input order:\n\n"
    '[\n'
    '  {\n'
    '    "figure": "<svg filename>",\n'
    '    "design_sound": true | false,\n'
    '    "design_critique": "<one sentence if not sound, else empty string>",\n'
    '    "aesthetic_clean": true | false,\n'
    '    "problems": [ "<specific problem 1>", "<specific problem 2>", ... up to 3 most impactful ],\n'
    '    "recommendation": "clean" | "polish" | "redraw-required",\n'
    '    "overall_assessment": "<one sentence summary>"\n'
    '  }\n'
    ']\n\n'
    "Recommendation:\n"
    "  \"clean\"           — design sound AND aesthetic clean; no edit needed.\n"
    "  \"polish\"          — content preservation-friendly edits can resolve problems.\n"
    "  \"redraw-required\" — figure is structurally so broken that human redraw is warranted.\n\n"
    "Be specific about WHICH element is the problem (not 'there is overlap' but 'arrow from Worker B to Shared Storage crosses text label Worker C Task partition 3'). Limit problems to the 3 most impactful — the ones whose resolution most improves the figure as a whole.\n\n"
    + RULES_SUMMARY
)

def build_review_prompt(batch: List[Figure]) -> str:
    lines = [REVIEW_PROMPT_HEADER, "", "--- FIGURES IN THIS BATCH ---"]
    for i, fig in enumerate(batch, 1):
        if not fig.render_ok: continue
        lines.extend(["",
            f"Figure {i}: {Path(fig.svg_path).name}",
            f"Chapter: {fig.chapter}",
            f"@{fig.png_path}"])
    return "\n".join(lines)

def phase_review_batch(batch: List[Figure], batch_id: str) -> None:
    """PHASE 1 — batched review for up to REVIEW_BATCH_SIZE figures."""
    # Render first
    for fig in batch:
        png_rel = png_rel_for(fig)
        err = render_svg(fig.svg_path, png_rel)
        if err:
            fig.render_ok = False
            fig.render_error = err
            fig.status = "render-failed"
            fig.final_notes = f"render failed: {err[:120]}"
        else:
            fig.render_ok = True
            fig.png_path = png_rel

    rendered = [f for f in batch if f.render_ok]
    if not rendered: return

    prompt = build_review_prompt(rendered)
    result = gemini_call(prompt, log_prefix=f"review-{batch_id}")
    if result["error"]:
        for fig in rendered:
            fig.iterations.append({"phase": "review", "batch": batch_id,
                                   "error": result["error"]})
        return

    parsed = _parse_json_body(_unwrap_response(result["stdout"]))
    if not isinstance(parsed, list):
        for fig in rendered:
            fig.iterations.append({"phase": "review", "batch": batch_id,
                                   "error": "parse-failed"})
        return

    by_name = {Path(f.svg_path).name: f for f in rendered}
    for entry in parsed:
        fname = entry.get("figure", "")
        fig = by_name.get(fname)
        if not fig: continue
        fig.design_sound = bool(entry.get("design_sound", True))
        fig.clean_this_cycle = bool(entry.get("aesthetic_clean", False)) and fig.design_sound
        fig.current_problems = entry.get("problems", []) or []
        if entry.get("design_critique"):
            fig.current_problems = [f"DESIGN: {entry['design_critique']}"] + fig.current_problems
        fig.recommendation = entry.get("recommendation", "polish")
        fig.overall_assessment = entry.get("overall_assessment", "")
        fig.iterations.append({
            "phase": "review", "batch": batch_id,
            "design_sound": fig.design_sound,
            "aesthetic_clean": fig.clean_this_cycle,
            "problems_count": len(fig.current_problems),
            "recommendation": fig.recommendation,
        })

# ---------------------------------------------------------------------------
# PHASE 2: Individual rewrite
# ---------------------------------------------------------------------------
REWRITE_PROMPT_HEADER = (
    "You are editing an SVG figure for an MIT Press machine learning systems "
    "textbook. Resolve the listed problems. Follow this invariant strictly:\n\n"
    "CONTENT IS INVARIANT. Every text label, numeric value, colour role, "
    "and semantic element MUST survive unchanged. Do NOT reword, truncate, "
    "abbreviate, reorder, or change ANY text. Do NOT change colour "
    "semantics (compute blue stays compute blue).\n\n"
    "GEOMETRY IS FLUID. Positions, sizes, spacings, groupings, alignments, "
    "and the viewBox itself may be freely adjusted. Move labels out of the "
    "way of lines. Widen the canvas if cramped. Rebalance heavy columns. "
    "Re-route arrows orthogonally if they cross content.\n\n"
    "The goal is a figure that, as a whole, reads cleanly — not a minimum "
    "diff. Apply the full SVG style rules (below).\n\n"
    "Return ONLY the complete revised SVG — nothing else, no prose, no "
    "markdown fence. Your response must start with <?xml or <svg.\n"
    + RULES_SUMMARY
)

def build_rewrite_prompt(fig: Figure, svg_src: str) -> str:
    parts = [REWRITE_PROMPT_HEADER, "",
             "--- PROBLEMS TO RESOLVE ---"]
    for i, p in enumerate(fig.current_problems, 1):
        parts.append(f"{i}. {p}")
    if fig.rejected_attempts:
        parts.append("")
        parts.append("--- PREVIOUS ATTEMPTS THAT FAILED (do NOT repeat these approaches) ---")
        for i, r in enumerate(fig.rejected_attempts[-4:], 1):
            parts.append(f"{i}. {r}")
    parts.extend(["", "--- ORIGINAL SVG SOURCE ---", svg_src])
    return "\n".join(parts)

def phase_rewrite_one(fig: Figure, cycle: int) -> Optional[str]:
    """PHASE 2 — generate a candidate revised SVG. Returns new SVG or None."""
    svg_abs = worktree_root() / fig.svg_path
    try:
        svg_src = svg_abs.read_text()
    except Exception as e:
        fig.iterations.append({"phase": "rewrite", "cycle": cycle,
                               "error": f"read-failed:{e}"})
        return None

    prompt = build_rewrite_prompt(fig, svg_src)
    log_id = f"rewrite-c{cycle}-{Path(fig.svg_path).stem}"
    result = gemini_call(prompt, log_prefix=log_id)
    if result["error"]:
        fig.iterations.append({"phase": "rewrite", "cycle": cycle,
                               "error": result["error"]})
        return None

    body = _unwrap_response(result["stdout"]).strip()
    fence = re.match(r"^```(?:xml|svg)?\s*(.*?)\s*```$", body, re.S)
    if fence: body = fence.group(1).strip()
    if not (body.startswith("<?xml") or body.startswith("<svg")):
        m = re.search(r"(<\?xml.*?</svg>|<svg.*?</svg>)", body, re.S)
        if not m:
            fig.iterations.append({"phase": "rewrite", "cycle": cycle,
                                   "error": "no-svg-in-response",
                                   "response_head": body[:200]})
            return None
        body = m.group(1)

    fig.iterations.append({"phase": "rewrite", "cycle": cycle,
                           "wall_sec": round(result["wall_time_sec"], 1)})
    return body

# ---------------------------------------------------------------------------
# PHASE 3: Batched verify
# ---------------------------------------------------------------------------
VERIFY_PROMPT_HEADER = (
    "You are verifying figure fixes for an MIT Press ML systems textbook. "
    "For each figure in the batch you are shown TWO renderings: BEFORE "
    "(original) and AFTER (with the proposed fix applied). For each figure, "
    "determine: were the listed problems resolved? Was any regression "
    "introduced elsewhere?\n\n"
    "Return ONLY a JSON array (no prose, no fence), one entry per figure "
    "in input order:\n\n"
    '[\n'
    '  {\n'
    '    "figure": "<svg filename>",\n'
    '    "resolved": true | false,\n'
    '    "regression": "<description of new problem introduced, or null>",\n'
    '    "next_action": "accept" | "revert" | "iterate"\n'
    '  }\n'
    ']\n\n'
    "next_action rules:\n"
    "  \"accept\"  — problems resolved AND no regression. Commit the fix.\n"
    "  \"revert\"  — regression worse than the original problems. Roll back.\n"
    "  \"iterate\" — improvement present but not complete. Keep the new SVG "
    "as baseline, propose further polish next cycle.\n"
)

def build_verify_prompt(batch: List[Dict[str, Any]]) -> str:
    lines = [VERIFY_PROMPT_HEADER, "", "--- FIGURES TO VERIFY ---"]
    for i, item in enumerate(batch, 1):
        lines.extend(["",
            f"Figure {i}: {item['name']}",
            f"Original problems: {'; '.join(item['problems'])}",
            f"BEFORE:  @{item['before_png']}",
            f"AFTER:   @{item['after_png']}"])
    return "\n".join(lines)

def phase_verify_batch(items: List[Dict[str, Any]], batch_id: str) -> Dict[str, Dict]:
    """PHASE 3 — batched before/after verify. Returns {filename: verdict_dict}."""
    if not items: return {}
    prompt = build_verify_prompt(items)
    result = gemini_call(prompt, log_prefix=f"verify-{batch_id}")
    if result["error"]:
        return {item["name"]: {"error": result["error"]} for item in items}
    parsed = _parse_json_body(_unwrap_response(result["stdout"]))
    if not isinstance(parsed, list):
        return {item["name"]: {"error": "parse-failed"} for item in items}
    out = {}
    for entry in parsed:
        name = entry.get("figure", "")
        if name:
            out[name] = entry
    return out

# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------
def git_commit(fig: Figure, cycle: int, description: str) -> Optional[str]:
    try:
        subprocess.run(["git", "-C", str(worktree_root()), "add", fig.svg_path],
                       check=True, capture_output=True)
        short = description.strip().split("\n")[0][:60] or "quality fix"
        msg = f"style({fig.chapter}/svg): [c{cycle}] {short} ({Path(fig.svg_path).name})"
        cp = subprocess.run(
            ["git", "-C", str(worktree_root()), "commit", "-m", msg],
            check=False, capture_output=True, text=True)
        if cp.returncode != 0: return None
        rev = subprocess.run(
            ["git", "-C", str(worktree_root()), "rev-parse", "--short", "HEAD"],
            check=False, capture_output=True, text=True)
        return rev.stdout.strip() if rev.returncode == 0 else None
    except Exception:
        return None

def git_revert(fig: Figure) -> None:
    subprocess.run(["git", "-C", str(worktree_root()),
                    "checkout", "--", fig.svg_path],
                   check=False, capture_output=True)

# ---------------------------------------------------------------------------
# Main fix loop
# ---------------------------------------------------------------------------
def run_fix_cycles(figures: List[Figure], scope_set: set, skip_drafts: bool,
                   meta: Dict) -> None:
    """PHASE 1-3 loop for up to MAX_CYCLES. Mutates figures in-place, saves state."""

    def active_queue() -> List[Figure]:
        out = []
        for f in figures:
            if f.svg_path not in scope_set: continue
            if f.status in TERMINAL_STATUSES: continue
            if skip_drafts and f.is_draft:
                if f.status == "pending":
                    f.status = "skipped-draft"; f.final_notes = "draft, skipped"
                continue
            out.append(f)
        return out

    for cycle in range(1, MAX_CYCLES + 1):
        queue = active_queue()
        if not queue:
            print(f"[cycle {cycle}] queue empty — stopping.")
            break
        if _budget_exhausted():
            print(f"[cycle {cycle}] budget exhausted — stopping.")
            break

        print(f"\n━━━ CYCLE {cycle} / {MAX_CYCLES} ━━━ "
              f"queue={len(queue)} budget={_read_budget():.0f}s")

        # ── PHASE 1: review in batches ─────────────────────────────────────
        for i in range(0, len(queue), REVIEW_BATCH_SIZE):
            batch = queue[i:i + REVIEW_BATCH_SIZE]
            bid = f"c{cycle}-review-b{i//REVIEW_BATCH_SIZE + 1}"
            print(f"[{bid}] reviewing {len(batch)} figures ...", flush=True)
            phase_review_batch(batch, bid)
            save_state(figures, meta)
            cleans = sum(1 for f in batch if f.clean_this_cycle)
            polish = sum(1 for f in batch if f.recommendation == "polish")
            redraw = sum(1 for f in batch if f.recommendation == "redraw-required")
            print(f"  clean={cleans} polish={polish} redraw-req={redraw} "
                  f"budget={_read_budget():.0f}s", flush=True)

        # ── Finalise clean / redraw-required figures ──────────────────────
        for fig in queue:
            if fig.clean_this_cycle:
                fig.status = "fixed" if fig.cycles_completed > 0 else "clean"
                fig.final_notes = fig.final_notes or f"converged at cycle {cycle}"
            elif fig.recommendation == "redraw-required":
                fig.status = "needs-human"
                fig.final_notes = "redraw-required (structural issue beyond polish)"
        save_state(figures, meta)

        # ── PHASE 2: individual rewrites ──────────────────────────────────
        polish_queue = [f for f in queue
                        if f.status not in TERMINAL_STATUSES
                        and f.recommendation == "polish"
                        and f.current_problems]
        if not polish_queue:
            print(f"[cycle {cycle}] nothing to polish.")
            save_state(figures, meta)
            continue

        verify_items = []  # populated as we rewrite + render
        for j, fig in enumerate(polish_queue, 1):
            if _budget_exhausted(): break
            print(f"[c{cycle} rewrite {j}/{len(polish_queue)}] "
                  f"{Path(fig.svg_path).name} ...", flush=True)

            svg_abs = worktree_root() / fig.svg_path
            before_content = svg_abs.read_text()

            # Snapshot backup + BEFORE render (frozen per-cycle)
            backup_file = backups_dir() / f"{Path(fig.svg_path).stem}.cycle{cycle}.svg"
            backup_file.write_text(before_content)
            before_rel = png_rel_for(fig, "before")
            before_abs = worktree_root() / before_rel
            shutil.copy2(worktree_root() / fig.png_path, before_abs)

            # Request rewrite
            candidate = phase_rewrite_one(fig, cycle)
            if candidate is None:
                fig.rejected_attempts.append(f"c{cycle}: no-rewrite-produced")
                save_state(figures, meta); continue

            # LOCAL GATE: content audit
            ok, reason = content_audit_ok(before_content, candidate)
            if not ok:
                fig.rejected_attempts.append(f"c{cycle}: content-audit-failed ({reason})")
                fig.iterations.append({"phase": "content-audit", "cycle": cycle,
                                       "rejected": reason})
                save_state(figures, meta); continue

            # LOCAL GATE: convergence / oscillation
            h = canonical_svg_hash(candidate)
            if h in fig.canonical_hashes[:-1]:  # seen this exact form before
                fig.status = "needs-human"
                fig.final_notes = f"oscillating at cycle {cycle} (hash repeat)"
                save_state(figures, meta); continue
            fig.canonical_hashes.append(h)
            if len(fig.canonical_hashes) > 6:
                fig.canonical_hashes = fig.canonical_hashes[-6:]

            # Write + render AFTER
            svg_abs.write_text(candidate)
            after_rel = png_rel_for(fig, "after")
            err = render_svg(fig.svg_path, after_rel)
            if err:
                git_revert(fig)
                fig.rejected_attempts.append(f"c{cycle}: render-failed ({err[:80]})")
                fig.iterations.append({"phase": "render-after", "cycle": cycle,
                                       "error": err})
                save_state(figures, meta); continue

            # Queue for verify
            verify_items.append({
                "fig": fig,
                "name": Path(fig.svg_path).name,
                "problems": list(fig.current_problems),
                "before_png": before_rel,
                "after_png":  after_rel,
                "before_content": before_content,
                "candidate": candidate,
                "cycle": cycle,
            })

        save_state(figures, meta)

        # ── PHASE 3: batched verify ───────────────────────────────────────
        for k in range(0, len(verify_items), VERIFY_BATCH_SIZE):
            if _budget_exhausted(): break
            vbatch = verify_items[k:k + VERIFY_BATCH_SIZE]
            bid = f"c{cycle}-verify-b{k//VERIFY_BATCH_SIZE + 1}"
            print(f"[{bid}] verifying {len(vbatch)} before/after pairs ...", flush=True)
            verdicts = phase_verify_batch(vbatch, bid)

            # APPLY verdicts
            for item in vbatch:
                fig = item["fig"]
                v = verdicts.get(item["name"], {})
                action = v.get("next_action", "revert")
                regression = v.get("regression")

                fig.iterations.append({
                    "phase": "verify", "cycle": cycle,
                    "action": action, "regression": regression,
                    "resolved": v.get("resolved"),
                })

                if action == "accept":
                    # Commit the fix atomically
                    desc = (fig.current_problems[0] if fig.current_problems
                            else "aesthetic polish")
                    sha = git_commit(fig, cycle, desc)
                    if sha: fig.commit_shas.append(sha)
                    fig.cycles_completed += 1
                    # Stay in queue — may still have more problems to find next cycle
                elif action == "iterate":
                    # Keep candidate SVG as baseline; don't revert, don't commit yet
                    # The next cycle's review will re-evaluate and propose further polish.
                    # But we SHOULD commit the partial progress so it's not lost on Ctrl-C.
                    desc = (fig.current_problems[0] if fig.current_problems
                            else "partial polish")
                    sha = git_commit(fig, cycle, f"[iter] {desc}")
                    if sha: fig.commit_shas.append(sha)
                    fig.cycles_completed += 1
                else:  # revert (or error/unknown)
                    git_revert(fig)
                    reason = regression or "verify-rejected"
                    fig.rejected_attempts.append(f"c{cycle}: {reason[:120]}")

            save_state(figures, meta)
            print(f"  -> processed {len(vbatch)} pairs "
                  f"budget={_read_budget():.0f}s", flush=True)

    # ── Post-cycles: close out anything still pending ─────────────────────
    for fig in figures:
        if fig.svg_path not in scope_set: continue
        if fig.status in TERMINAL_STATUSES: continue
        if skip_drafts and fig.is_draft:
            fig.status = "skipped-draft"
            fig.final_notes = "draft, skipped"
            continue
        # Still pending after MAX_CYCLES
        if fig.cycles_completed > 0:
            fig.status = "fixed"
            fig.final_notes = (f"partial — {fig.cycles_completed} cycle(s) of edits "
                               f"accepted but still has open problems")
        else:
            fig.status = "needs-human"
            fig.final_notes = (f"exhausted {MAX_CYCLES} cycles, {len(fig.current_problems)} "
                               f"problems remaining")
    save_state(figures, meta)

# ---------------------------------------------------------------------------
# State IO
# ---------------------------------------------------------------------------
def load_state() -> Optional[Dict[str, Any]]:
    if not state_path().exists(): return None
    try: return json.loads(state_path().read_text())
    except Exception: return None

def save_state(figures: List[Figure], meta: Dict) -> None:
    state = {
        "meta": meta,
        "figures": [asdict(f) for f in figures],
        "totals": compute_totals(figures),
        "gemini_wall_time_sec": _read_budget(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = state_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path())

def compute_totals(figures: List[Figure]) -> Dict[str, int]:
    t = {"total": len(figures), "live": 0, "draft": 0,
         "pending": 0, "clean": 0, "fixed": 0,
         "needs-human": 0, "render-failed": 0, "skipped-draft": 0}
    for f in figures:
        t[f.status] = t.get(f.status, 0) + 1
        if f.is_draft: t["draft"] += 1
        else: t["live"] += 1
    return t

def figures_from_state(state: Dict) -> List[Figure]:
    return [Figure(**f) for f in state.get("figures", [])]

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _scope_paths(figs: List[Figure], scope: str) -> set:
    if scope == "full":  return {f.svg_path for f in figs}
    if scope == "smoke": return {f.svg_path for f in figs if f.chapter == "vol2/introduction"}
    return {f.svg_path for f in figs if f.chapter == scope}

def mode_audit(scope: str, skip_drafts: bool) -> int:
    ensure_dirs()
    state = load_state()
    if state:
        figures = figures_from_state(state)
        meta = state.get("meta", {})
    else:
        figures = build_inventory()
        meta = {
            "model": GEMINI_MODEL,
            "render_width_px": RENDER_WIDTH_PX,
            "max_cycles": MAX_CYCLES,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "scope": scope,
            "skip_drafts": skip_drafts,
        }
    scope_set = _scope_paths(figures, scope)
    print(f"[audit] scope={scope} figures_in_scope={len(scope_set)} "
          f"skip_drafts={skip_drafts}", flush=True)

    run_fix_cycles(figures, scope_set, skip_drafts, meta)

    totals = compute_totals(figures)
    print(f"\n[audit] done. totals={totals}")
    return 0

def mode_diagnose(scope: str, skip_drafts: bool) -> int:
    """Single review pass, no rewrites. Useful for pre-audit surveys."""
    ensure_dirs()
    figures = build_inventory()
    meta = {
        "model": GEMINI_MODEL, "mode": "diagnose-only",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    scope_set = _scope_paths(figures, scope)
    queue = [f for f in figures
             if f.svg_path in scope_set and not (skip_drafts and f.is_draft)]
    print(f"[diagnose-only] figures={len(queue)}", flush=True)

    for i in range(0, len(queue), REVIEW_BATCH_SIZE):
        batch = queue[i:i + REVIEW_BATCH_SIZE]
        bid = f"diag-b{i//REVIEW_BATCH_SIZE + 1}"
        print(f"[{bid}] reviewing {len(batch)} ...", flush=True)
        phase_review_batch(batch, bid)
        save_state(figures, meta)
        print(f"  budget={_read_budget():.0f}s", flush=True)
    print(f"\n[diagnose-only] done. totals={compute_totals(figures)}")
    return 0

def mode_report() -> int:
    state = load_state()
    if not state:
        print("No state.json.", file=sys.stderr); return 2
    figures = figures_from_state(state)
    meta = state.get("meta", {})
    totals = compute_totals(figures)

    lines = [
        "# SVG Figure Audit Report",
        "",
        f"- Run started: {meta.get('started_at','?')}",
        f"- Model: `{meta.get('model','?')}`",
        f"- Scope: `{meta.get('scope','?')}`",
        f"- Skip drafts: {meta.get('skip_drafts','?')}",
        f"- Max cycles: {meta.get('max_cycles', MAX_CYCLES)}",
        f"- Gemini wall time: {_read_budget():.0f} s "
        f"({_read_budget()/60:.1f} min)",
        "",
        "## Totals", "",
        f"- Total figures: {totals['total']}",
        f"- Live / Draft: {totals['live']} / {totals['draft']}",
        f"- Fixed (committed): **{totals.get('fixed',0)}**",
        f"- Clean (no edit needed): {totals.get('clean',0)}",
        f"- Needs human: **{totals.get('needs-human',0)}**",
        f"- Render-failed: {totals.get('render-failed',0)}",
        f"- Skipped drafts: {totals.get('skipped-draft',0)}",
        f"- Still pending: {totals.get('pending',0)}",
        "",
        "## Fixed (with commit SHAs)", "",
    ]
    fixed = sorted([f for f in figures if f.status == "fixed"],
                   key=lambda x: (x.chapter, x.svg_path))
    if not fixed: lines.append("_None._")
    else:
        lines.append("| Figure | Chapter | Cycles | Commits | Notes |")
        lines.append("|---|---|---|---|---|")
        for f in fixed:
            shas = ", ".join(f.commit_shas) or "-"
            lines.append(f"| `{Path(f.svg_path).name}` | {f.chapter} | "
                         f"{f.cycles_completed} | {shas} | {f.final_notes} |")

    lines += ["", "## Needs human review", ""]
    needs = sorted([f for f in figures if f.status == "needs-human"],
                   key=lambda x: (x.chapter, x.svg_path))
    if not needs: lines.append("_None._")
    else:
        for f in needs:
            lines.append(f"### `{Path(f.svg_path).name}` ({f.chapter})")
            lines.append("")
            lines.append(f"**Notes:** {f.final_notes}")
            lines.append("")
            lines.append(f"**Assessment:** {f.overall_assessment or '—'}")
            lines.append("")
            if f.current_problems:
                lines.append("**Remaining problems:**")
                for p in f.current_problems:
                    lines.append(f"- {p}")
                lines.append("")
            if f.rejected_attempts:
                lines.append("**Rejected fix attempts:**")
                for r in f.rejected_attempts[-5:]:
                    lines.append(f"- {r}")
                lines.append("")
            if f.png_path:
                lines.append(f"**Rendered PNG:** `{f.png_path}`")
                lines.append("")

    lines += ["", "## Render-failed", ""]
    rf = [f for f in figures if f.status == "render-failed"]
    if not rf: lines.append("_None._")
    else:
        for f in rf:
            lines.append(f"- `{f.svg_path}` — {f.render_error}")

    report_path().write_text("\n".join(lines))
    print(f"report -> {report_path().relative_to(worktree_root())}")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("audit", "diagnose", "fix"):
        p = sub.add_parser(name)
        p.add_argument("--scope", default="smoke",
                       help="smoke | full | <chapter slug>")
        p.add_argument("--skip-drafts", action="store_true")
    sub.add_parser("report")
    args = ap.parse_args()

    if args.mode in ("audit", "fix"):
        return mode_audit(args.scope, getattr(args, "skip_drafts", False))
    if args.mode == "diagnose":
        return mode_diagnose(args.scope, getattr(args, "skip_drafts", False))
    if args.mode == "report":
        return mode_report()
    return 2

if __name__ == "__main__":
    sys.exit(main())
