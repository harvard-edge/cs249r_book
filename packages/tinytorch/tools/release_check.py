#!/usr/bin/env python3
r"""
TinyTorch release check
=======================

Runs every gate a release must pass, in the order a failure is cheapest to find.
Each check is independent and prints PASS/FAIL with the specific offenders, so a
failure tells you what to fix rather than that something is wrong.

    python3 tools/release_check.py            # all gates
    python3 tools/release_check.py --fast     # skip the two slow gates
    python3 tools/release_check.py --list     # show the gates and exit

Written 2026-09 during the pre-release cleanup. Every gate here corresponds to a
real defect found that day; none of them are hypothetical.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import contextlib
import collections

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
MODULES = ROOT / "modules"
TESTS = ROOT / "tests"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

_registry: list = []


def gate(name, slow=False, advisory=False):
    def deco(fn):
        _registry.append((name, fn, slow, advisory))
        return fn
    return deco


# ---------------------------------------------------------------- helpers ---
def module_files():
    out = []
    for d in sorted(SRC.glob("[0-9][0-9]_*")):
        py = next(d.glob("[0-9]*.py"), None)
        if py:
            out.append((int(d.name[:2]), d.name, py))
    return out


def cells(text):
    out, cur, hdr = [], [], None
    for line in text.splitlines():
        if line.startswith("# %%"):
            if hdr is not None:
                out.append((hdr, "\n".join(cur)))
            hdr, cur = line, []
        elif hdr is not None:
            cur.append(line)
    if hdr is not None:
        out.append((hdr, "\n".join(cur)))
    return out


def export_targets():
    t = {}
    for num, name, py in module_files():
        m = re.search(r"^#\|\s*default_exp\s+([\w.]+)", py.read_text(), re.M)
        if m:
            t[f"tinytorch.{m.group(1)}"] = (num, name)
    return t


# =============================== STRUCTURE ==================================
@gate("structure: 20 modules, numbered 1-20, each with an export target")
def g_module_set():
    mods = module_files()
    targets = export_targets()
    errs = []
    if len(mods) != 20:
        errs.append(f"expected 20 modules, found {len(mods)}")
    nums = [n for n, _, _ in mods]
    if nums != list(range(1, 21)):
        errs.append(f"numbering gap: {nums}")
    if len(targets) != 20:
        errs.append(f"expected 20 '#| default_exp' directives, found {len(targets)}")
    return errs


@gate("structure: every module carries the 13-section spine, in order")
def g_spine():
    SPINE = ["🔗 Prerequisites & Progress", "🎯 Learning Objectives",
             "📦 Where This Code Lives", "📋 Module Dependencies",
             "💡 Introduction", "📐 Foundations", "🏗️", "🔧 Integration",
             "📊 Systems Analysis", "🧪 Module Integration Test",
             "🤔 ML Systems Reflection Questions", "⭐ Aha Moment",
             "🚀 MODULE SUMMARY"]
    EXEMPT = {("20_capstone", "📊 Systems Analysis")}
    errs = []
    for _, name, py in module_files():
        heads = re.findall(r"^## (.+)$", py.read_text(), re.M)
        for sec in SPINE:
            if (name, sec) in EXEMPT:
                continue
            if not any(h.startswith(sec) for h in heads):
                errs.append(f"{name}: missing section '{sec}'")
    return errs


@gate("structure: no duplicate ## headings inside a module")
def g_dup_heads():
    errs = []
    for _, name, py in module_files():
        heads = re.findall(r"^## (.+)$", py.read_text(), re.M)
        for h, c in collections.Counter(heads).items():
            if c > 1:
                errs.append(f"{name}: '{h}' appears {c}x")
    return errs


@gate("structure: subtitled headings use ': ', never ' - '")
def g_head_sep():
    errs = []
    for _, name, py in module_files():
        for m in re.finditer(r"^## ([^\w\s]\S*\s+.+)$", py.read_text(), re.M):
            if " - " in m.group(1):
                errs.append(f"{name}: '## {m.group(1)}'")
    return errs


@gate("structure: MODULE SUMMARY has its canonical subsections in order")
def g_summary():
    ORDER = ["Key Accomplishments", "Systems Insights Discovered",
             "Ready for Next Steps", "Export with:", "**Next**:"]
    errs = []
    for num, name, py in module_files():
        t = py.read_text()
        if "## 🚀 MODULE SUMMARY" not in t:
            errs.append(f"{name}: no MODULE SUMMARY")
            continue
        s = t[t.index("## 🚀 MODULE SUMMARY"):]
        seq = [(m.group(1) or m.group(2) or m.group(3))
               for m in re.finditer(r"^### (.+)$|^(Export with:)|^(\*\*Next\*\*:)", s, re.M)]
        core = [x for x in seq if x in ORDER]
        want = ORDER[:-1] if num == 20 else ORDER
        if core != want:
            errs.append(f"{name}: got {core}")
    return errs


# ============================== NBGRADER ====================================
@gate("nbgrader: exactly three canonical cell-header shapes")
def g_headers():
    ok = {
        'exercise': re.compile(r'^# %% nbgrader=\{"grade": false, "grade_id": "[^"]+", "solution": true\}$'),
        'test':     re.compile(r'^# %% nbgrader=\{"grade": true, "grade_id": "[^"]+", "locked": true, "points": \d+\}$'),
        'given':    re.compile(r'^# %% nbgrader=\{"grade": false, "grade_id": "[^"]+", "solution": false\}$'),
    }
    errs = []
    for _, name, py in module_files():
        for line in py.read_text().splitlines():
            if line.startswith("# %% nbgrader=") and not any(p.match(line) for p in ok.values()):
                errs.append(f"{name}: {line[:88]}")
    return errs


@gate("nbgrader: a cell marked solution:true actually has BEGIN/END SOLUTION")
def g_solution_markers():
    errs = []
    for _, name, py in module_files():
        for h, b in cells(py.read_text()):
            if '"solution": true' in h and "### BEGIN SOLUTION" not in b:
                gid = re.search(r'"grade_id":\s*"([^"]+)"', h)
                errs.append(f"{name}: {gid.group(1) if gid else '?'}")
    return errs


@gate("nbgrader: grade_ids are unique within a module")
def g_gid_unique():
    errs = []
    for _, name, py in module_files():
        ids = re.findall(r'"grade_id":\s*"([^"]+)"', py.read_text())
        for g, c in collections.Counter(ids).items():
            if c > 1:
                errs.append(f"{name}: grade_id '{g}' used {c}x")
    return errs


# ============================== PEDAGOGY ====================================
@gate("pedagogy: every exercise has a markdown cell explaining it first")
def g_no_orphans():
    errs = []
    for _, name, py in module_files():
        cs = cells(py.read_text())
        for i, (h, b) in enumerate(cs):
            if '"solution": true' in h:
                if not (i > 0 and cs[i-1][0].startswith("# %% [markdown]")):
                    gid = re.search(r'"grade_id":\s*"([^"]+)"', h)
                    errs.append(f"{name}: {gid.group(1) if gid else '?'} has no lead-in")
    return errs


@gate("pedagogy: every unit test has a What/Why/Expected header")
def g_test_headers():
    errs = []
    for _, name, py in module_files():
        cs = cells(py.read_text())
        for i, (h, b) in enumerate(cs):
            if '"grade": true' not in h:
                continue
            gid = re.search(r'"grade_id":\s*"([^"]+)"', h)
            gid = gid.group(1) if gid else "?"
            # The Module Integration Test is framed by its own '## 🧪 Module
            # Integration Test' section heading in all 20 modules, so it does
            # not carry the per-test header. Its grade_id varies by module.
            if gid in ("test-module", "test_module", "module-test",
                       "module-integration", "module_integration"):
                continue
            prev = cs[i-1][1] if i > 0 and cs[i-1][0].startswith("# %% [markdown]") else ""
            missing = [f for f in ("**What we're testing**", "**Why it matters**", "**Expected**")
                       if f not in prev]
            if missing:
                errs.append(f"{name}: {gid} missing {', '.join(missing)}")
    return errs


@gate("pedagogy: docstring scaffold present on exercises (TODO/APPROACH/HINTS)")
def g_scaffold():
    errs = []
    for _, name, py in module_files():
        for h, b in cells(py.read_text()):
            if '"solution": true' not in h:
                continue
            gid = re.search(r'"grade_id":\s*"([^"]+)"', h)
            gid = gid.group(1) if gid else "?"
            for field in ("TODO:", "APPROACH:"):
                if field not in b:
                    errs.append(f"{name}: {gid} missing {field}")
    return errs


@gate("pedagogy: Reflection Questions render as markdown, not a code cell")
def g_reflection_markdown():
    errs = []
    for _, name, py in module_files():
        for h, b in cells(py.read_text()):
            if "## 🤔 ML Systems Reflection Questions" in b and not h.startswith("# %% [markdown]"):
                errs.append(f"{name}: reflection section is a code cell")
    return errs


# ========================= PROGRESSIVE DISCLOSURE ===========================
@gate("disclosure: no module imports from a later-numbered module")
def g_disclosure():
    targets = export_targets()
    errs = []
    for num, name, py in module_files():
        for i, line in enumerate(py.read_text().splitlines(), 1):
            m = re.match(r"\s*from (tinytorch\.[\w.]+) import ", line)
            if m and m.group(1) in targets and targets[m.group(1)][0] > num:
                errs.append(f"{name}:{i} imports {m.group(1)} (module {targets[m.group(1)][1]})")
    return errs


@gate("disclosure: forward references to later modules are framed as previews",
      advisory=True)
def g_forward_refs():
    PREVIEW = ("Next", "will ", "you'll", "You'll", "prepare", "Coming", "→", "->", "Looking Ahead")
    errs = []
    for num, name, py in module_files():
        for i, line in enumerate(py.read_text().splitlines(), 1):
            for m in re.finditer(r"Module\s+(\d{1,2})\b", line):
                if int(m.group(1)) > num and not any(p in line for p in PREVIEW):
                    errs.append(f"{name}:{i} {line.strip()[:74]}")
    return errs


# ============================ PACKAGE SURFACE ===============================
@gate("package: every export target imports")
def g_imports():
    errs = []
    for target in sorted(export_targets()):
        try:
            importlib.import_module(target)
        except Exception as e:
            errs.append(f"{target}: {type(e).__name__}: {e}")
    return errs


@gate("package: every documented import resolves")
def g_documented_imports():
    errs = []
    for _, name, py in module_files():
        for line in py.read_text().splitlines():
            m = re.match(r"\s*from (tinytorch\.[\w.]+) import (.+?)\s*(?:#.*)?$", line)
            if not m or "(" in m.group(2) or m.group(2).strip() == "*":
                continue
            try:
                mod = importlib.import_module(m.group(1))
            except Exception as e:
                errs.append(f"{name}: {m.group(1)} ({e})")
                continue
            for sym in (s.strip() for s in m.group(2).split(",")):
                if sym and not hasattr(mod, sym):
                    errs.append(f"{name}: {m.group(1)} has no '{sym}'")
    return errs


@gate("package: no unreferenced zero-arg demo functions (dead code)")
def g_dead_demos():
    errs = []
    for _, name, py in module_files():
        src = py.read_text()
        tree = ast.parse(src)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            n = node.name
            if not re.match(r"^(analyze_|explore_|run_\w+_example)", n):
                continue
            if node.args.args:            # takes arguments -> a utility, not a demo
                continue
            if not re.search(rf"^\s*{n}\(\)", src, re.M):
                errs.append(f"{name}: {n}() is defined but never called")
    return errs


@gate("package: no SyntaxWarnings (invalid escapes in ASCII-art docstrings)")
def g_syntax_warnings():
    import warnings
    errs = []
    for _, name, py in module_files():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            try:
                compile(py.read_text(), str(py), "exec")
            except SyntaxError as e:
                errs.append(f"{name}: SyntaxError line {e.lineno}: {e.msg}")
                continue
            for w in caught:
                if issubclass(w.category, SyntaxWarning):
                    errs.append(f"{name}: {w.message}")
    return errs


# ================================ TESTS =====================================
@gate("tests: no test signals failure with a bare return")
def g_test_returns():
    errs = []
    for f in sorted(TESTS.rglob("test_*.py")):
        src = f.read_text()
        for m in re.finditer(r"^def (test_\w+)\(", src, re.M):
            nxt = re.search(r"^(?:def |class |@)", src[m.end():], re.M)
            body = src[m.end(): m.end() + (nxt.start() if nxt else len(src))]
            if re.search(r"^\s{4}return (True|False)\b", body, re.M):
                errs.append(f"{f.relative_to(ROOT)}::{m.group(1)}")
    return errs


@gate("tests: no bare 'except:' swallowing a failure")
def g_bare_except():
    errs = []
    for f in sorted(TESTS.rglob("test_*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.match(r"\s*except:\s*$", line):
                errs.append(f"{f.relative_to(ROOT)}:{i}")
    return errs


@gate("tests: every test file imports and collects")
def g_collect():
    p = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                        str(TESTS), "--ignore", str(TESTS / "environment")],
                       capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        return [l for l in (p.stdout + p.stderr).splitlines() if "error" in l.lower()][:10]
    return []


# ============================== SLOW GATES ==================================
@gate("student journey: all 20 notebooks run end-to-end as __main__", slow=True)
def g_journey():
    errs = []
    for path in sorted(MODULES.glob("*/*.ipynb")):
        src = "\n".join("".join(c["source"]) for c in json.load(open(path))["cells"]
                        if c["cell_type"] == "code")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(src, str(path), "exec"), {"__name__": "__main__"})
        except Exception as e:
            errs.append(f"{path.parent.name}: {type(e).__name__}: {str(e)[:70]}")
    return errs


@gate("pytest: full suite green", slow=True)
def g_pytest():
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", str(TESTS),
                        "--ignore", str(TESTS / "environment")],
                       capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        return [l for l in (p.stdout + p.stderr).splitlines()
                if l.startswith(("FAILED", "ERROR"))][:15] or ["pytest exited non-zero"]
    return []


# ================================= MAIN =====================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="skip the slow gates")
    ap.add_argument("--list", action="store_true", help="list gates and exit")
    ap.add_argument("-k", metavar="SUBSTR", help="run only gates matching SUBSTR")
    args = ap.parse_args()

    if args.list:
        for name, _, slow, _adv in _registry:
            print(f"  {'[slow] ' if slow else '       '}{name}")
        return 0

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    failed = 0
    width = max(len(n) for n, _, _, _ in _registry) + 2
    for name, fn, slow, advisory in _registry:
        if args.fast and slow:
            print(f"  {DIM}SKIP{RESET}  {name}")
            continue
        if args.k and args.k not in name:
            continue
        try:
            errs = fn()
        except Exception as e:
            errs = [f"gate itself crashed: {type(e).__name__}: {e}"]
        if errs and advisory:
            print(f"  {YELLOW}WARN{RESET}  {name.ljust(width)} {YELLOW}{len(errs)} to review{RESET}")
            for e in errs[:6]:
                print(f"          {DIM}{e}{RESET}")
            if len(errs) > 6:
                print(f"          {DIM}... and {len(errs)-6} more{RESET}")
        elif errs:
            failed += 1
            print(f"  {RED}FAIL{RESET}  {name.ljust(width)} {RED}{len(errs)} issue(s){RESET}")
            for e in errs[:12]:
                print(f"          {DIM}{e}{RESET}")
            if len(errs) > 12:
                print(f"          {DIM}... and {len(errs)-12} more{RESET}")
        else:
            print(f"  {GREEN}PASS{RESET}  {name}")

    total = len([1 for n, _, s, a in _registry if not (args.fast and s)])
    print()
    if failed:
        print(f"  {RED}{failed} of {total} gates failed{RESET}")
    else:
        print(f"  {GREEN}all {total} gates passed{RESET}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
