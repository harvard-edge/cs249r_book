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
import json
import os
import pathlib
import re
import subprocess
import sys
import collections
import shutil
import tempfile

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


def section_text(text, heading):
    """Body of one ## section: from its heading to the next ## heading."""
    i = text.find(heading)
    if i < 0:
        return ""
    j = text.find("\n## ", i + len(heading))
    return text[i:] if j < 0 else text[i:j]


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


SPINE = ["🔗 Prerequisites & Progress", "🎯 Learning Objectives",
         "📦 Where This Code Lives", "📋 Module Dependencies",
         "💡 Introduction", "📐 Foundations", "🏗️", "🔧 Integration",
         "📊 Systems Analysis", "🧪 Module Integration Test",
         "🤔 ML Systems Reflection Questions", "⭐ Aha Moment",
         "🚀 MODULE SUMMARY"]
# The capstone builds a submission pipeline rather than a component, so it has no
# Systems Analysis section; every other module carries all 13.
SPINE_EXEMPT = {("20_capstone", "📊 Systems Analysis")}


def spine_index(heading):
    for k, sec in enumerate(SPINE):
        if heading.startswith(sec):
            return k
    return None


@gate("structure: every module carries the 13-section spine, in order")
def g_spine():
    # Until 2026-09-08 this gate checked presence only; the order check was added
    # when the anatomy pass found five modules with Systems Analysis before
    # Integration and eight with extra top-level sections. See MODULE_ANATOMY.md.
    errs = []
    for _, name, py in module_files():
        heads = re.findall(r"^## (.+)$", py.read_text(), re.M)
        for sec in SPINE:
            if (name, sec) in SPINE_EXEMPT:
                continue
            if not any(h.startswith(sec) for h in heads):
                errs.append(f"{name}: missing section '{sec}'")
        seq = []
        for h in heads:
            k = spine_index(h)
            if k is None:
                errs.append(f"{name}: '## {h}' is not a spine section (make it a ### subsection)")
            else:
                seq.append((k, h))
        for (a, ha), (b, hb) in zip(seq, seq[1:]):
            if b < a or (b == a and a != SPINE.index("🏗️")):
                errs.append(f"{name}: '## {hb}' comes after '## {ha}'")
                break
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
        # Extended from ## to ### on 2026-09-08; the ### form had 30 offenders.
        for m in re.finditer(r"^(##|###) (.+)$", py.read_text(), re.M):
            if " - " in m.group(2):
                errs.append(f"{name}: '{m.group(1)} {m.group(2)}'")
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
        if seq != ORDER:
            errs.append(f"{name}: got {seq}")
        # The **Next** line is the last thing a student reads (2026-09-08).
        tail = [ln for ln in s.rstrip().splitlines() if ln.strip()][-2:]
        if not (tail[-1] == '"""' and tail[0].startswith("**Next**:")):
            errs.append(f"{name}: summary does not end on its **Next** line")
    return errs


@gate("structure: Module Dependencies carries its four labels, in order")
def g_dependencies():
    LABELS = ["**Prerequisites**", "**External Dependencies**",
              "**TinyTorch Dependencies**", "**Dependency Flow**"]
    errs = []
    for _, name, py in module_files():
        sec = section_text(py.read_text(), "## 📋 Module Dependencies")
        found = [l for l in re.findall(r"^(\*\*[^*]+\*\*)", sec, re.M) if l in LABELS]
        if found != LABELS:
            errs.append(f"{name}: got {found}")
    return errs


@gate("structure: reflection questions are numbered '### Question N:'")
def g_reflection_numbering():
    # Standardized 2026-09-08 from four heading styles (bare, numbered, emoji, bonus).
    errs = []
    for _, name, py in module_files():
        sec = section_text(py.read_text(), "## 🤔 ML Systems Reflection Questions")
        heads = re.findall(r"^### (.+)$", sec, re.M)
        nums = []
        for h in heads:
            m = re.match(r"Question (\d+): ", h)
            if m:
                nums.append(int(m.group(1)))
            elif not h.startswith("Bonus Question: "):
                errs.append(f"{name}: '### {h}'")
        if nums != list(range(1, len(nums) + 1)):
            errs.append(f"{name}: questions numbered {nums}")
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
        # Ungraded tests carry the same lead-in; key on the heading too (2026-09-08).
        for h, b in cs:
            if h.startswith("# %% [markdown]") and "### 🧪 Unit Test: " in b:
                missing = [f for f in ("**What we're testing**", "**Why it matters**", "**Expected**")
                           if f not in b]
                if missing:
                    title = re.search(r"### 🧪 Unit Test: (.+)", b).group(1)
                    errs.append(f"{name}: '{title}' missing {', '.join(missing)}")
    return errs


@gate("pedagogy: unit-test headings and prints use the 🧪/✅ grammar")
def g_test_grammar():
    # One grammar for every graded test (2026-09-08): the markdown lead-in is
    # '### 🧪 Unit Test: <Name>', the cell prints '🧪 Unit Test: ...' on entry and
    # a '✅ ...' line on success. The Module Integration Test is the one exception:
    # it is framed by its own ## heading and prints a per-module banner.
    errs = []
    for _, name, py in module_files():
        text = py.read_text()
        for m in re.finditer(r"^(#+) (.*(?:Unit|Integration) Test.*)$", text, re.M):
            level, h = m.groups()
            if h.startswith("🧪 Module Integration Test"):
                continue
            if not (level == "###" and h.startswith("🧪 Unit Test: ")):
                errs.append(f"{name}: '{level} {h}'")
        cs = cells(text)
        for i, (h, b) in enumerate(cs):
            if '"grade": true' not in h:
                continue
            gid = re.search(r'"grade_id":\s*"([^"]+)"', h).group(1)
            if "module" in gid:
                continue
            if 'print("🧪 Unit Test: ' not in b:
                errs.append(f"{name}: {gid} does not print '🧪 Unit Test: ...'")
            if "✅" not in b:
                errs.append(f"{name}: {gid} has no ✅ success line")
            # The docstring carries the marker too, so a reader scanning
            # definitions sees the same grammar as the notebook output.
            # Added 2026-09-09 after 19 tests across four modules were found
            # opening with a bare 'Test ...' docstring.
            for fn in re.finditer(r'^def (test_unit_\w+)\(.*\n\s+"""(.*?)"""', b, re.M):
                if not fn.group(2).startswith("🧪"):
                    errs.append(f"{name}: {fn.group(1)} docstring does not open with 🧪")
    return errs


@gate("package: every solution cell opens with an export directive")
def g_solution_exports():
    # A solution cell is student-written code that must reach the package, so its
    # first line is '#| export' (public) or '#| exporti' (module-private). Found
    # two unexported solutions in 15 and 16 on 2026-09-08.
    errs = []
    for _, name, py in module_files():
        for h, b in cells(py.read_text()):
            if '"solution": true' not in h:
                continue
            first = b.split("\n", 1)[0]
            if first not in ("#| export", "#| exporti"):
                gid = re.search(r'"grade_id":\s*"([^"]+)"', h).group(1)
                errs.append(f"{name}: {gid} starts with {first!r}")
    return errs


@gate("tests: test_module() runs only from the module's tail cell")
def g_test_module_tail():
    # The last code cell of every module is the same three-line runner:
    # test_module(), a blank line, demo_<module>(). The test_module cell itself
    # must not self-run, or the notebook runs the suite twice (2026-09-08).
    errs = []
    for _, name, py in module_files():
        cs = cells(py.read_text())
        for h, b in cs:
            if "def test_module" in b and re.search(r"^if __name__", b, re.M):
                errs.append(f"{name}: test_module cell runs itself")
        code = [b for h, b in cs if not h.startswith("# %% [markdown]")]
        tail = [ln for ln in code[-1].splitlines() if ln.strip()]
        if not (tail and tail[0] == 'if __name__ == "__main__":' and tail[1].strip() == "test_module()"
                and tail[2].strip() == 'print("\\n")' and tail[-1].strip().startswith("demo")):
            errs.append(f"{name}: tail cell is {tail[:4]}")
    return errs


@gate("structure: __main__ runners only call names defined in earlier cells")
def g_runner_order():
    # A notebook executes top to bottom, so a `if __name__ == "__main__":` block
    # may only use names already defined. Added 2026-09-08 after a cell move put
    # module 13's create_causal_mask() below the test that calls it; the slow
    # notebook gate caught it, this catches it in under a second.
    errs = []
    for _, name, py in module_files():
        code = [b for h, b in cells(py.read_text()) if not h.startswith("# %% [markdown]")]
        trees = []
        for b in code:
            try:
                trees.append(ast.parse(b))
            except SyntaxError:
                trees.append(None)
        def defs(tree):
            out = set()
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                    out.add(n.name)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    out.update((a.asname or a.name).split(".")[0] for a in n.names)
                elif isinstance(n, ast.Assign):
                    out.update(t.id for t in n.targets if isinstance(t, ast.Name))
            return out
        all_defs = set().union(*(defs(t) for t in trees if t))
        seen, bodies = set(), {}
        for i, tree in enumerate(trees):
            if tree is None:
                continue
            seen |= defs(tree)
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                    bodies[n.name] = n
            for n in tree.body:
                if not (isinstance(n, ast.If) and "__main__" in ast.dump(n.test)):
                    continue
                # Follow calls transitively through plain functions: the runner
                # calls a test, the test calls a helper; the helper must already
                # exist. Classes are treated as leaves and names bound inside a
                # function (arguments, local imports, assignments) are ignored.
                def local_names(fn):
                    out = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
                    for s in ast.walk(fn):
                        if isinstance(s, (ast.Import, ast.ImportFrom)):
                            out.update((a.asname or a.name).split(".")[0] for a in s.names)
                        elif isinstance(s, ast.Name) and isinstance(s.ctx, ast.Store):
                            out.add(s.id)
                    return out
                todo = [(n, set())]; visited = set(); bad = set()
                while todo:
                    node, local = todo.pop()
                    for s in ast.walk(node):
                        if not (isinstance(s, ast.Call) and isinstance(s.func, ast.Name)):
                            continue
                        callee = s.func.id
                        if callee in local or callee not in all_defs:
                            continue
                        if callee not in seen:
                            bad.add(callee)
                        elif isinstance(bodies.get(callee), ast.FunctionDef) and callee not in visited:
                            visited.add(callee)
                            todo.append((bodies[callee], local_names(bodies[callee])))
                for u in sorted(bad):
                    errs.append(f"{name}: cell {i} runner reaches {u}(), defined in a later cell")
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
    PREVIEW = ("Next", "will ", "you'll", "You'll", "prepare", "Coming", "→", "->", "Looking Ahead",
               "implements", "adds", "teaches", "written in", "completes", "is the same", "exists to")
    errs = []
    for num, name, py in module_files():
        for i, line in enumerate(py.read_text().splitlines(), 1):
            # A dependency-diagram label row such as "(Module 06)  (Module 07)" is not
            # prose and needs no preview framing (2026-09-08).
            if not re.sub(r"\([^)]*\)|\s", "", line):
                continue
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


@gate("tests: no graded cell swallows its own failure")
def g_graded_except():
    """A handler that only passes (or only prints) inside a locked graded cell
    makes the points unreachable: the assertion it guards can fail and the cell
    still reports success. Added 2026-09-09, after module 19's 10-point
    plotting test was found to catch every exception and pass regardless.

    The expected-raise idiom is not a violation, so a try body containing an
    assert is skipped:  try: f(); assert False, "should raise"; except E: pass
    """
    errs = []
    for _, name, path in module_files():
        lines = path.read_text().splitlines()
        graded = False
        for i, line in enumerate(lines):
            if line.startswith("# %%"):
                graded = '"grade": true' in line
                continue
            if not graded:
                continue
            m = re.match(r"(\s*)except\b.*:\s*$", line)
            if not m:
                continue
            indent = len(m.group(1))

            # Walk back to the matching `try:` at the same indent.
            start = None
            for j in range(i - 1, -1, -1):
                cand = lines[j]
                if not cand.strip():
                    continue
                ci = len(cand) - len(cand.lstrip())
                if ci < indent:
                    break
                if ci == indent and cand.strip() == "try:":
                    start = j
                    break
            if start is not None and "assert" in "\n".join(lines[start + 1:i]):
                continue

            handler = []
            for nxt in lines[i + 1:]:
                if not nxt.strip():
                    continue
                if len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                handler.append(nxt.strip())
            if not handler:
                continue
            # Strip trailing comments: `pass  # optional` is still a swallow.
            stmts = [c for c in (h.split("#")[0].strip() for h in handler) if c]
            if all(st == "pass" or st.startswith("print(") for st in stmts):
                errs.append(f"{name}:{i + 1}: graded handler body is only {handler[0]!r}")
    return errs


@gate("pedagogy: an exercise is followed by its test, not by more exercises")
def g_exercise_runs():
    """The flow a student works through is implement, then test, then implement.
    A long run of exercises with no graded cell between them means the student
    writes several components before any of them is checked, and it is how
    module 14 came to test its nine profiling helpers 1,100 lines after they
    were written. Added 2026-09-09.

    A family of small classes taught together may share one test, so the limit
    is a run length rather than strict alternation. Modules 06 and 14 sat at 10
    and 9 when this gate was written; the rest were at 3 or below.
    """
    LIMIT = 5
    errs = []
    sol = re.compile(r'^# %% nbgrader=\{"grade": false, "grade_id": "([^"]+)", "solution": true\}')
    tst = re.compile(r'^# %% nbgrader=\{"grade": true, "grade_id": "[^"]+", "locked": true, "points": \d+\}')
    for _, name, path in module_files():
        run, first = 0, None
        for i, line in enumerate(path.read_text().splitlines(), 1):
            m = sol.match(line)
            if m:
                run += 1
                if run == 1:
                    first = (i, m.group(1))
                if run == LIMIT + 1:
                    errs.append(f"{name}:{first[0]} {first[1]}: "
                                f"{run} exercises with no test between them")
            elif tst.match(line):
                run, first = 0, None
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
def notebook_paths():
    """Canonical notebook names from the numbered source modules."""
    return [MODULES / name / f"{name.split('_', 1)[1]}.ipynb"
            for _, name, _ in module_files()]


@gate("package: every source module has exactly its expected notebook")
def g_notebook_set():
    # 2026-09-11: an empty modules/ directory previously passed the journey gate.
    expected = set(notebook_paths())
    if not expected:
        return ["no source modules found; cannot validate notebook inventory"]
    actual = set(MODULES.glob("*/*.ipynb"))
    return ([f"missing notebook: {p.relative_to(MODULES)}" for p in sorted(expected - actual)]
            + [f"unexpected notebook: {p.relative_to(MODULES)}" for p in sorted(actual - expected)])


@gate("reference: source-built numerical and training regressions")
def g_reference_regressions():
    # 2026-09-11: green reference notebooks missed graph-lifetime, accumulation,
    # fractional-constant, and distillation-training defects. Build from source
    # in a temporary package so stale local exports cannot make this check pass.
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "check_reference.py")],
                       capture_output=True, text=True, cwd=ROOT)
    if p.returncode:
        return (p.stdout + p.stderr).splitlines()[-20:] or [f"reference check exited {p.returncode}"]
    return []


@gate("student journey: all 20 notebooks run end-to-end as __main__", slow=True)
def g_journey():
    errs = g_notebook_set()
    if errs:
        return errs
    from nbdev.export import nb_export

    # 2026-09-11: each notebook gets a new interpreter and only earlier exports.
    # Reusing the release-check process hid dependencies on later modules and
    # leaked monkey-patched classes and RNG state between notebooks.
    runner = (
        "import json,sys; from pathlib import Path; "
        "p=Path(sys.argv[1]); nb=json.loads(p.read_text(encoding='utf-8')); "
        "code='\\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'); "
        "exec(compile(code,str(p),'exec'),{'__name__':'__main__'})"
    )
    with tempfile.TemporaryDirectory(prefix="tinytorch-progression-") as tmp:
        package = pathlib.Path(tmp) / "tinytorch"
        for subdir in ("", "core", "perf"):
            target = package / subdir
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "tinytorch" / subdir / "__init__.py", target / "__init__.py")
        env = os.environ.copy()
        env.update(PYTHONPATH=tmp, MPLBACKEND="Agg", TINYTORCH_QUIET="1")
        for path in notebook_paths():
            print(f"          Running {path.parent.name} with earlier modules only", flush=True)
            try:
                p = subprocess.run([sys.executable, "-c", runner, str(path)], cwd=tmp,
                                   env=env, capture_output=True, text=True, timeout=300)
                if p.returncode:
                    errs.append(f"{path.parent.name}: " + (p.stderr or p.stdout)[-1500:])
                    break
                nb_export(str(path), lib_path=str(package))
            except subprocess.TimeoutExpired:
                errs.append(f"{path.parent.name}: notebook exceeded 300 seconds")
                break
    return errs


@gate("pytest: full suite green", slow=True)
def g_pytest():
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", str(TESTS),
                        "--ignore", str(TESTS / "environment")],
                       capture_output=True, text=True, cwd=ROOT)
    for line in p.stdout.splitlines():
        if re.search(r"\d+ passed|\d+ failed|\d+ skipped", line) and " in " in line:
            print(f"          {line}", flush=True)
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
