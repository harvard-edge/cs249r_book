r"""Strict notation: every symbol inside math is a registered macro.

The single source of truth for symbols is the mlsysim notation registry
(``mlsysim/mlsysim/notation/data/``; guide: ``data/README.md``). Each entry has
a macro (``\LinkBandwidth``), the LaTeX readers see (``\beta``), a meaning,
and a kind. This module checks the book against it and helps convert it.

A math span may contain registry macros, LaTeX built-ins that are not
symbols (``\frac``, ``\sum``, ``\log``, spacing), numerals, punctuation, and
the role-based exemptions in ``data/exempt.yaml``. Everything else is a
finding:

``letter``         a Latin letter written raw (``p``, ``T``)
``greek``          a Greek letter written raw (``\alpha``)
``form``           a font or accent applied in the chapter (``\mathbf{q}``)
``decorated``      a macro given a new subscript, superscript or prime in the
                   chapter (``\Bandwidth_{\text{bisect}}``): a new meaning
                   needs its own entry
``unknown``        a command that is neither a registry macro nor a LaTeX
                   built-in (a typo, or a deleted macro)
``alias``          a retired macro name: use the canonical one
``out-of-scope``   a registry macro the chapter's volume does not declare
                   (not in that volume's notation chapter, not in the
                   chapter's ``notation-local`` list)

Where math is read: body math (``$...$``, ``$$...$$``, exactly as Pandoc
finds it), figure/table/listing captions in div attributes, quiz JSON strings,
and the LaTeX strings passed to ``fmt_math(...)`` in Python cells.

The module never guesses about its inputs: if mlsysim cannot be imported it
raises, because a check that sees an empty registry reports nonsense.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_HINT = Path(__file__).resolve().parents[3]
TOOLS = REPO_HINT / "binder" / "tools" / "notation"
RULE_DOC = "mlsysim/mlsysim/notation/data/README.md"

# Commands whose argument is text, a label or a dimension, never a symbol.
TEXT_ARG_COMMANDS = {
    "text", "textrm", "textit", "textbf", "texttt", "textsf", "mathrm", "operatorname",
    "mbox", "label", "tag", "hspace", "vspace", "color", "textcolor", "href", "url",
    "begin", "end", "eqref", "ref",
}
# Commands whose argument is a symbol that receives a font or an accent.
FORM_COMMANDS = {
    "mathbf", "boldsymbol", "bm", "mathcal", "mathscr", "mathbb", "mathfrak", "mathsf",
    "mathtt", "mathit", "vec", "hat", "bar", "tilde", "dot", "ddot", "overline",
    "widehat", "widetilde", "check", "breve", "underline", "acute", "grave",
}
# Commands followed by a dimension (\mkern-3mu, \kern 2pt).
DIMENSION_COMMANDS = {"mkern", "mskip", "kern", "hskip", "vskip", "mspace"}
ENVS_WITH_SPEC = {"array", "tabular", "alignat", "alignedat"}
GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta",
    "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "rho", "varrho",
    "sigma", "varsigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega", "varpi",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi",
    "Omega", "ell", "pi", "digamma", "varkappa",
}
ESCAPE_HATCH = re.compile(r"<!--\s*notation-ok:\s*(?P<reason>[^>]+?)\s*-->")


# ===========================================================================
# Registry, exemptions, built-ins
# ===========================================================================


@dataclass(frozen=True)
class RegistryEntry:
    macro: str          # without backslash
    latex: str
    meaning: str
    kind: str = "scalar"
    domain: str = ""


@dataclass
class Exemptions:
    index: Set[str] = field(default_factory=lambda: {"i", "j", "k"})
    differential: Set[str] = field(default_factory=lambda: {"d"})
    euler: Set[str] = field(default_factory=lambda: {"e"})
    transpose: Set[str] = field(default_factory=lambda: {"T"})
    commands: Set[str] = field(default_factory=lambda: {"pi", "infty", "ell"})
    forms: Set[str] = field(default_factory=lambda: {
        r"\mathbb{R}", r"\mathbb{N}", r"\mathbb{Z}", r"\mathbb{Q}", r"\mathbb{C}",
        r"\mathbb{E}", r"\mathbb{P}", r"\mathbb{1}", r"\mathcal{O}"})

    @classmethod
    def load(cls, repo: Path) -> "Exemptions":
        import yaml  # local: binder imports this module eagerly
        path = repo / "mlsysim" / "mlsysim" / "notation" / "data" / "exempt.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        roles = data.get("letters_by_role", {})
        return cls(set(roles.get("index", [])), set(roles.get("differential", [])),
                   set(roles.get("euler", [])), set(roles.get("transpose", [])),
                   set(data.get("commands", [])), set(data.get("forms", [])))


class Registry:
    """The registry plus everything the check needs to judge a span."""

    def __init__(self, entries: Sequence[RegistryEntry], aliases: Dict[str, str],
                 builtins: Optional[Set[str]] = None, exempt: Optional[Exemptions] = None):
        self.entries = {e.macro: e for e in entries}
        self.aliases = dict(aliases)              # old name -> canonical (no backslashes)
        self.builtins = set(builtins or ())       # LaTeX commands KaTeX defines
        self.exempt = exempt or Exemptions()
        self._by_latex: Dict[Tuple[str, ...], List[RegistryEntry]] = {}
        for e in entries:
            key = tuple(t for t in tokenize(e.latex) if not t.isspace())
            self._by_latex.setdefault(key, []).append(e)
        self.max_latex_tokens = max((len(k) for k in self._by_latex), default=1)

    @classmethod
    def load(cls, repo: Path) -> "Registry":
        sys.path.insert(0, str(repo / "mlsysim"))
        try:
            from mlsysim.notation import Notation
        except Exception as exc:  # pragma: no cover - environment failure
            raise RuntimeError(
                "notation check cannot import mlsysim (" + type(exc).__name__ + ": " + str(exc)
                + "). Run binder with the project virtual environment on PATH; an empty "
                "registry would make every result wrong.") from exc
        entries = [RegistryEntry(s.macro[1:], s.latex, str(s.definition), s.kind, s.domain)
                   for s in Notation.list()]
        aliases = {a[1:]: c[1:] for a, c in Notation.aliases().items()}
        bfile = repo / "binder" / "tools" / "notation" / "katex_builtins.json"
        builtins = set(json.loads(bfile.read_text())["commands"]) if bfile.exists() else set()
        return cls(entries, aliases, builtins, Exemptions.load(repo))

    def is_macro(self, name: str) -> bool:
        return name in self.entries or name in self.aliases

    def canonical(self, name: str) -> str:
        return self.aliases.get(name, name)

    def macro_table(self) -> Dict[str, str]:
        """\\Name -> LaTeX for every macro and alias (what the book compiles)."""
        table = {"\\" + m: e.latex for m, e in self.entries.items()}
        for a, c in self.aliases.items():
            table["\\" + a] = "\\" + c
        return table

    def candidates(self, tokens: Sequence[str]) -> List[RegistryEntry]:
        return list(self._by_latex.get(tuple(tokens), []))


# ===========================================================================
# Math extraction (Pandoc tex_math_dollars rules; cross-checked by a test)
# ===========================================================================


@dataclass
class MathSpan:
    line: int
    col: int
    text: str
    display: bool
    source: str = "body"     # body | caption | quiz | fmt_math
    section: str = ""
    hatch: str = ""


_FENCE = re.compile(r"^\s*(```+|~~~+)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_DIV_ATTR = re.compile(r"^\s*:{3,}\s*\{.*\}\s*$")
_CAPTION_ATTR = re.compile(r"\b(fig-cap|tbl-cap|lst-cap|title)=\"((?:[^\"\\]|\\.)*)\"")
_FMT_MATH = re.compile(r"fmt_math\(\s*([rfRF]{0,2})([\"'])(.*?)(?<!\\)\2", re.S)


def _blank(text: str, pattern: re.Pattern) -> str:
    return pattern.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _scan_dollars(text: str, original: str) -> List[Tuple[int, int, str, bool]]:
    """(line, col, contents, display) for each $-delimited span, by Pandoc's rules:
    an opener is an unescaped $ not followed by space; it pairs with the FIRST
    unescaped $ after it, which closes only if not preceded by space and not
    followed by a digit; otherwise there is no math at that opener."""
    starts = [0]
    for ln in text.split("\n"):
        starts.append(starts[-1] + len(ln) + 1)

    def where(pos: int) -> Tuple[int, int]:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1, pos - starts[lo] + 1

    def escaped(pos: int) -> bool:
        b = 0
        while pos - 1 - b >= 0 and text[pos - 1 - b] == "\\":
            b += 1
        return b % 2 == 1

    out = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "$" or escaped(i):
            i += 1
            continue
        if text.startswith("$$", i):
            end = text.find("$$", i + 2)
            while end != -1 and escaped(end):
                end = text.find("$$", end + 1)
            if end == -1:
                i += 2
                continue
            out.append((*where(i), original[i + 2:end], True))
            i = end + 2
            continue
        if i + 1 >= n or text[i + 1].isspace():
            i += 1
            continue
        j = i + 1
        while j < n:
            if text[j] == "$" and not escaped(j):
                break
            if text[j] == "\n" and text[j + 1:j + 2] == "\n":
                j = n
                break
            j += 1
        if j < n and not text[j - 1].isspace() and not (j + 1 < n and text[j + 1].isdigit()):
            out.append((*where(i), original[i + 1:j], False))
            i = j + 1
        else:
            i += 1
    return out


def extract_math(source: str, include_captions: bool = True) -> List[MathSpan]:
    """Math in a .qmd file: body math as Pandoc sees it, plus caption attributes."""
    lines = source.split("\n")
    masked: List[str] = []
    in_fence, fence_marker = False, ""
    front = lines[:1] == ["---"]
    attr_lines: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        if front:
            masked.append(" " * len(line))
            if idx > 0 and line.strip() in ("---", "..."):
                front = False
            continue
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)[0] * 3
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif line.strip().startswith(fence_marker):
                in_fence = False
            masked.append(" " * len(line))
            continue
        if in_fence:
            masked.append(" " * len(line))
            continue
        if _DIV_ATTR.match(line):
            attr_lines.append((idx, line))
            masked.append(" " * len(line))
            continue
        masked.append(line)
    text = "\n".join(masked)
    text = _blank(text, re.compile(r"<!--.*?-->", re.S))
    text = _blank(text, re.compile(r"`[^`\n]*`"))
    text = _blank(text, re.compile(r"\{[^{}\n]*=\"[^\"\n]*\"[^{}\n]*\}"))
    original = "\n".join(lines)

    section_at, current = [], ""
    for raw_line, seen in zip(lines, masked):
        if _HEADING.match(seen):
            current = raw_line.strip()
        section_at.append(current)

    spans = [MathSpan(ln, col, body, disp) for ln, col, body, disp in _scan_dollars(text, original)]
    if include_captions:
        for idx, line in attr_lines:
            for m in _CAPTION_ATTR.finditer(line):
                cap = m.group(2).replace('\\"', '"')
                for _, col, body, disp in _scan_dollars(cap, cap):
                    spans.append(MathSpan(idx + 1, m.start(2) + col, body, disp, "caption"))
    for s in spans:
        s.section = section_at[s.line - 1] if s.line - 1 < len(section_at) else ""
        m = ESCAPE_HATCH.search(lines[s.line - 1])
        if m:
            s.hatch = m.group("reason")
    spans.sort(key=lambda s: (s.line, s.col))
    return spans


def _fstring_literal(body: str) -> str:
    """The literal text of an f-string: {{ and }} become braces, each {field} becomes 1.

    A numeral stands in for a computed value, so it can never read as a symbol.
    """
    out, i, n = [], 0, len(body)
    while i < n:
        if body.startswith("{{", i) or body.startswith("}}", i):
            out.append(body[i])
            i += 2
        elif body[i] == "{":
            depth, j = 1, i + 1
            while j < n and depth:
                depth += {"{": 1, "}": -1}.get(body[j], 0)
                j += 1
            out.append("1")
            i = j
        else:
            out.append(body[i])
            i += 1
    return "".join(out)


def extract_fmt_math(source: str) -> List[MathSpan]:
    """LaTeX passed to fmt_math(...) inside Python cells (f-string fields become 1)."""
    out = []
    for m in _FMT_MATH.finditer(source):
        prefix, body = m.group(1).lower(), m.group(3)
        if "f" in prefix:
            body = _fstring_literal(body)
        if "r" not in prefix:
            body = body.replace("\\\\", "\\")
        out.append(MathSpan(source.count("\n", 0, m.start()) + 1, 1, body, False, "fmt_math"))
    return out


def extract_quiz_math(json_text: str) -> List[MathSpan]:
    """Math in every string of a quiz JSON file."""
    out: List[MathSpan] = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + [str(k)])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + [str(i)])
        elif isinstance(node, str) and "$" in node:
            for _, col, body, disp in _scan_dollars(node, node):
                out.append(MathSpan(0, col, body, disp, "quiz", section="/".join(path)))

    walk(json.loads(json_text), [])
    return out


# ===========================================================================
# Judging one span
# ===========================================================================

_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|[A-Za-z]|\d|\s+|.", re.S)


def tokenize(math: str) -> List[str]:
    return _TOKEN.findall(math)


def _skip_space(toks: List[str], k: int) -> int:
    while k < len(toks) and toks[k].isspace():
        k += 1
    return k


def _read_group(toks: List[str], k: int) -> Tuple[List[str], int]:
    """One argument at toks[k]: a {group} or a single token. Returns (contents, next index)."""
    k = _skip_space(toks, k)
    if k >= len(toks):
        return [], k
    if toks[k] != "{":
        return [toks[k]], k + 1
    depth, j = 0, k
    while j < len(toks):
        if toks[j] == "{":
            depth += 1
        elif toks[j] == "}":
            depth -= 1
            if depth == 0:
                return toks[k + 1:j], j + 1
        j += 1
    return toks[k + 1:], len(toks)


def _read_bracket(toks: List[str], k: int) -> int:
    """Skip an optional [..] argument starting at toks[k]."""
    k2 = _skip_space(toks, k)
    if k2 < len(toks) and toks[k2] == "[":
        j = k2
        while j < len(toks) and toks[j] != "]":
            j += 1
        return j + 1
    return k


@dataclass
class Finding:
    raw: str
    kind: str
    start: int      # token index
    end: int


def _decoration_ok(group: List[str], ex: Exemptions) -> bool:
    """Decorations that do not create a new symbol: powers, inverse, transpose, an index."""
    body = "".join(t for t in group if not t.isspace())
    if body in ("\\top", "-1") or body in ex.transpose:
        return True
    if re.fullmatch(r"\d+", body):
        return True
    idx = "|".join(map(re.escape, sorted(ex.index)))
    # an index, an index expression (k+1, t-1), or a parenthesized one ((i), (\ell))
    if re.fullmatch(r"\(?(?:%s|\\ell)(?:[+-]\d+)?\)?" % idx, body):
        return True
    return False


def judge(math: str, registry: Registry, scope: Optional[Set[str]] = None) -> List[Finding]:
    """Every non-registry symbol in one math span.

    ``scope`` is the set of macro names the chapter may use (its volume's
    notation chapter plus its notation-local list); None skips the scope test.
    """
    ex = registry.exempt
    toks = tokenize(math)
    found: List[Finding] = []
    sub_stack: List[int] = []      # 1 for a subscript group, 0 otherwise
    depth_sub = 0
    prev = ""                      # previous significant token
    prev2 = ""                     # the one before it
    k = 0
    while k < len(toks):
        t = toks[k]
        if t.isspace():
            k += 1
            continue
        if t == "{":
            prev2 = prev
            is_sub = 1 if prev == "_" else 0
            sub_stack.append(is_sub)
            depth_sub += is_sub
            prev, k = t, k + 1
            continue
        if t == "}":
            if sub_stack:
                depth_sub -= sub_stack.pop()
            prev, k = t, k + 1
            continue
        if t == "\\\\":                           # line break, maybe with [4pt]
            k, prev = _read_bracket(toks, k + 1), t
            continue
        if t.startswith("\\") and len(t) > 1 and t[1].isalpha():
            name = t[1:]
            nk = k + 1
            if nk < len(toks) and toks[nk] == "*":   # \operatorname*{...}
                nk += 1
            if name in TEXT_ARG_COMMANDS:
                arg, after = _read_group(toks, nk)
                body = "".join(a for a in arg if not a.isspace())
                if name == "begin" and body in ENVS_WITH_SPEC:
                    _, after = _read_group(toks, after)
                k, prev = after, "}"
                continue
            if name in DIMENSION_COMMANDS:
                j = _skip_space(toks, nk)
                while j < len(toks) and re.fullmatch(r"[-+.\d]|[a-z]", toks[j]):
                    j += 1
                k, prev = j, "}"
                continue
            if name in FORM_COMMANDS:
                arg, after = _read_group(toks, nk)
                raw = f"\\{name}{{{''.join(arg).strip()}}}"
                if raw not in ex.forms:
                    found.append(Finding(raw, "form", k, after))
                k, prev = after, "}"
                continue
            if registry.is_macro(name):
                if name in registry.aliases:
                    found.append(Finding(t, "alias", k, nk))
                elif scope is not None and name not in scope:
                    found.append(Finding(t, "out-of-scope", k, nk))
                j = _skip_space(toks, nk)
                while j < len(toks) and toks[j] in ("_", "^", "'"):
                    if toks[j] == "'":
                        found.append(Finding(t + "'", "decorated", k, j + 1))
                        j += 1
                    else:
                        grp, after = _read_group(toks, j + 1)
                        if not _decoration_ok(grp, ex):
                            found.append(Finding(t + toks[j] + "{" + "".join(grp).strip() + "}",
                                                 "decorated", k, after))
                        j = after
                    j = _skip_space(toks, j)
                k, prev = j, "}"
                continue
            if name in GREEK:
                if name not in ex.commands:
                    found.append(Finding(t, "greek", k, nk))
                k, prev = nk, t
                continue
            if name not in registry.builtins:
                found.append(Finding(t, "unknown", k, nk))
            k, prev = nk, t
            continue
        if len(t) == 1 and t.isalpha():
            j = _skip_space(toks, k + 1)
            nxt = toks[j] if j < len(toks) else ""
            exempt = (
                (t in ex.index and (depth_sub > 0 or prev == "_")) or
                (t in ex.transpose and (prev == "^" or (prev == "{" and prev2 == "^" and nxt == "}"))) or
                (t in ex.euler and nxt == "^") or
                (t in ex.differential and (bool(re.fullmatch(r"[A-Za-z]", nxt))
                                           or (nxt.startswith("\\") and registry.is_macro(nxt[1:]))))
            )
            if not exempt:
                found.append(Finding(t, "letter", k, k + 1))
        prev = t
        k += 1
    return found


def suggest(math: str, f: Finding, registry: Registry) -> List[RegistryEntry]:
    """Registry entries whose LaTeX matches the symbol as written, most specific first."""
    toks = tokenize(math)
    for end in range(min(len(toks), f.start + 4 * registry.max_latex_tokens), f.start, -1):
        cand = registry.candidates([t for t in toks[f.start:end] if not t.isspace()])
        if cand:
            return cand
    return []


# ===========================================================================
# Scope: which macros a chapter may use
# ===========================================================================

_VOLUME = re.compile(r"books/(vol[1-4])/")
_LOCAL_DECL = re.compile(r"^notation-local:\s*\[(.*?)\]\s*$", re.M)


def volume_of(rel: str) -> str:
    m = _VOLUME.search(rel.replace("\\", "/"))
    return m.group(1) if m else ""


def notation_table_macros(repo: Path, volume: str) -> Set[str]:
    """Macros listed in the first column of the volume's notation chapter tables."""
    path = repo / "books" / volume / "frontmatter" / "notation.qmd"
    if not path.exists():
        return set()
    names: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = line.split("|")
        if line.lstrip().startswith("|") and len(cells) > 2:
            names |= set(re.findall(r"\\([A-Z][A-Za-z]*)", cells[1]))
    return names


def local_macros(source: str) -> Set[str]:
    """The chapter's own symbols, from `notation-local: [\\A, \\B]` in its front matter."""
    if not source.startswith("---"):
        return set()
    head = source.split("\n---", 1)[0]
    m = _LOCAL_DECL.search(head)
    return set(re.findall(r"\\?([A-Z][A-Za-z]*)", m.group(1))) if m else set()


# ===========================================================================
# Findings for a file, grouped into a work package
# ===========================================================================


@dataclass
class Violation:
    file: str
    line: int
    col: int
    section: str
    source: str
    raw: str
    kind: str
    math: str
    token_start: int
    token_end: int
    suggestions: List[Dict[str, str]] = field(default_factory=list)
    hatch: str = ""

    def fingerprint(self) -> str:
        key = f"{self.raw}\x00{' '.join(self.math.split())}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]


def _entry_dict(e: RegistryEntry) -> Dict[str, str]:
    return {"macro": e.macro, "latex": e.latex, "meaning": e.meaning, "kind": e.kind, "domain": e.domain}


def scan_spans(spans: Iterable[MathSpan], rel: str, registry: Registry,
               scope: Optional[Set[str]] = None) -> List[Violation]:
    out: List[Violation] = []
    for span in spans:
        for f in judge(span.text, registry, scope):
            if f.kind == "alias":
                sug = [_entry_dict(registry.entries[registry.canonical(f.raw[1:])])]
            elif f.kind == "out-of-scope":
                sug = [_entry_dict(registry.entries[f.raw[1:]])]
            elif f.kind == "unknown":
                sug = []
            else:
                sug = [_entry_dict(e) for e in suggest(span.text, f, registry)]
            out.append(Violation(rel, span.line, span.col, span.section, span.source, f.raw, f.kind,
                                 span.text, f.start, f.end, sug, span.hatch))
    return out


def scan_path(path: Path, repo: Path, registry: Registry, scoped: bool = True) -> List[Violation]:
    rel = str(path.resolve().relative_to(repo))
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        return scan_spans(extract_quiz_math(text), rel, registry)
    scope = None
    vol = volume_of(rel)
    if scoped and vol and not rel.endswith("frontmatter/notation.qmd"):
        scope = notation_table_macros(repo, vol) | local_macros(text)
    return scan_spans(extract_math(text) + extract_fmt_math(text), rel, registry, scope)


def work_package(violations: List[Violation], rel: str) -> Dict:
    """One file's findings ordered by section, plus a per-symbol summary."""
    sections: Dict[str, Dict] = {}
    symbols: Dict[Tuple[str, str], Dict] = {}
    for v in violations:
        head = v.section or "(before the first heading)"
        sec = sections.setdefault(head, {"heading": head, "sites": []})
        sec["sites"].append({"line": v.line, "col": v.col, "source": v.source, "raw": v.raw, "kind": v.kind,
                             "token_start": v.token_start, "token_end": v.token_end, "math": v.math})
        sym = symbols.setdefault((v.raw, v.kind), {"raw": v.raw, "kind": v.kind, "count": 0, "lines": [],
                                                    "candidates": v.suggestions})
        sym["count"] += 1
        if v.line not in sym["lines"]:
            sym["lines"].append(v.line)
    return {"file": rel, "volume": volume_of(rel), "raw_symbols": len(violations),
            "hatches": sorted({v.hatch for v in violations if v.hatch}),
            "sections": list(sections.values()),
            "symbols": sorted(symbols.values(), key=lambda s: -s["count"])}


_KIND_ADVICE = {
    "letter": "write the macro whose meaning fits",
    "greek": "write the macro whose meaning fits",
    "form": "register the bold/calligraphic/accented symbol; its LaTeX carries the form",
    "decorated": "a new subscript, superscript or prime makes a new symbol: register it",
    "unknown": "neither a registry macro nor a LaTeX built-in: fix the spelling or register the symbol",
    "alias": "retired name: write the canonical macro",
    "out-of-scope": "not declared for this chapter: add it to the volume's notation chapter, or to this "
                    "chapter's notation-local list if only this chapter uses it",
}


def render_text(pkg: Dict) -> str:
    out = [f"{pkg['file']}: {pkg['raw_symbols']} raw symbol(s) in {len(pkg['sections'])} section(s)"]
    for sec in pkg["sections"]:
        counts = collections.Counter(s["raw"] for s in sec["sites"])
        lines = sorted({s["line"] for s in sec["sites"]})
        shown = ", ".join(map(str, lines[:10])) + (" ..." if len(lines) > 10 else "")
        out.append(f"  {sec['heading'][:72]}")
        out.append(f"      lines {shown}:  " + "  ".join(f"{r} x{n}" if n > 1 else r for r, n in counts.most_common()))
    if pkg["symbols"]:
        out.append("  Symbols to resolve:")
        for sym in pkg["symbols"]:
            c = sym["candidates"]
            if c and sym["kind"] in ("letter", "greek", "form"):
                hint = "the registry renders this as " + "; ".join(f"\\{x['macro']} ({x['meaning']})" for x in c[:4])
            elif c:
                hint = f"\\{c[0]['macro']} ({c[0]['meaning']})"
            else:
                hint = "no registry entry renders this"
            out.append(f"    {sym['raw']:<20} x{sym['count']:<4} [{sym['kind']}] {_KIND_ADVICE[sym['kind']]}; {hint}")
        out.append(f"  Next: read each symbol's sentence and choose by meaning, or register a new symbol "
                   f"({RULE_DOC}). Full site list: --json.")
    if pkg["hatches"]:
        out.append(f"  Escape hatches on this file: {', '.join(pkg['hatches'])}")
    return "\n".join(out)


# ===========================================================================
# Ratchet (fingerprints, not counts)
# ===========================================================================


def baseline_of(violations: Iterable[Violation]) -> Dict[str, Dict[str, int]]:
    base: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    for v in violations:
        fp = v.fingerprint()
        base[v.file][fp] = base[v.file].get(fp, 0) + 1
    return {f: dict(sorted(d.items())) for f, d in sorted(base.items())}


def ratchet(current: Dict[str, Dict[str, int]], baseline: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
    """A new fingerprint, or more of an existing one, is a regression."""
    worse: Dict[str, List[str]] = {}
    for f, fps in current.items():
        old = baseline.get(f, {})
        new = [fp for fp, n in fps.items() if n > old.get(fp, 0)]
        if new:
            worse[f] = new
    return worse


# ===========================================================================
# Rendering check (KaTeX MathML) and the deterministic applier
# ===========================================================================


def render_mathml(items: List[Tuple[str, str, bool]], macros: Dict[str, str]) -> Dict[str, str]:
    """id -> MathML (TeX annotation stripped) for (id, math, display) items."""
    req = {"macros": macros, "items": [{"id": i, "math": m, "display": d} for i, m, d in items]}
    proc = subprocess.run(["node", str(TOOLS / "render_math.js")], input=json.dumps(req),
                          capture_output=True, text=True, check=True, cwd=str(TOOLS))
    return json.loads(proc.stdout)


@dataclass
class RenderMismatch:
    file: str
    line: int
    reason: str
    before: str
    after: str


def verify_rendering(before_text: str, after_text: str, rel: str,
                     macros_before: Dict[str, str], macros_after: Dict[str, str]) -> List[RenderMismatch]:
    """Every math span renders identically before and after an edit.

    The old side renders with the registry as it was (``macros_before``) and the
    new side with the registry as it is (``macros_after``), so editing an entry's
    LaTeX is caught too. Spans are aligned in document order.
    """
    old, new = extract_math(before_text), extract_math(after_text)
    ro = render_mathml([(f"o{i}", s.text, s.display) for i, s in enumerate(old)], macros_before)
    rn = render_mathml([(f"n{i}", s.text, s.display) for i, s in enumerate(new)], macros_after)
    ko = [ro[f"o{i}"] for i in range(len(old))]
    kn = [rn[f"n{i}"] for i in range(len(new))]
    out: List[RenderMismatch] = []
    for tag, a0, a1, b0, b1 in difflib.SequenceMatcher(a=ko, b=kn, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        for off in range(max(a1 - a0, b1 - b0)):
            o = old[a0 + off] if a0 + off < a1 else None
            w = new[b0 + off] if b0 + off < b1 else None
            reason = "renders differently" if (o and w) else ("math removed" if o else "math added")
            out.append(RenderMismatch(rel, (w or o).line, reason, o.text if o else "", w.text if w else ""))
    return out


def apply_assignments(source: str, assignments: List[Dict]) -> Tuple[str, List[str]]:
    """Replace symbols at exact positions in body math.

    Each assignment is {"line", "col", "token_start", "token_end", "raw", "macro"},
    as produced from a work package. An assignment whose tokens no longer read
    ``raw`` is refused, never guessed. Returns (new source, refusals).
    """
    spans = {(s.line, s.col): s for s in extract_math(source) if s.source == "body"}
    lines = source.split("\n")
    line_starts = [0]
    for ln in lines:
        line_starts.append(line_starts[-1] + len(ln) + 1)
    by_span: Dict[Tuple[int, int], List[Dict]] = collections.defaultdict(list)
    for a in assignments:
        by_span[(a["line"], a["col"])].append(a)
    refusals: List[str] = []
    edits: List[Tuple[int, int, str]] = []
    for key, items in by_span.items():
        span = spans.get(key)
        if span is None:
            refusals.append(f"line {key[0]} col {key[1]}: no body math span starts here")
            continue
        toks = tokenize(span.text)
        offsets = [0]
        for t in toks:
            offsets.append(offsets[-1] + len(t))
        body_start = line_starts[span.line - 1] + span.col - 1 + (2 if span.display else 1)
        taken: List[Tuple[int, int]] = []
        for a in sorted(items, key=lambda a: a["token_start"]):
            s0, s1 = a["token_start"], a["token_end"]
            got = "".join(toks[s0:s1])
            if " ".join(got.split()) != " ".join(a["raw"].split()):
                refusals.append(f"line {span.line}: expected {a['raw']!r} at tokens {s0}-{s1}, found {got!r}")
                continue
            if any(not (s1 <= x0 or s0 >= x1) for x0, x1 in taken):
                refusals.append(f"line {span.line}: assignments overlap at tokens {s0}-{s1}")
                continue
            taken.append((s0, s1))
            rep = "\\" + a["macro"].lstrip("\\")
            j = s1
            while j < len(toks) and toks[j].isspace():
                j += 1
            if s1 < len(toks) and toks[s1][:1].isalpha():
                rep += " "        # \Macro directly followed by a letter would merge into one control word
            edits.append((body_start + offsets[s0], body_start + offsets[s1], rep))
    for s, e, rep in sorted(edits, key=lambda x: -x[0]):
        source = source[:s] + rep + source[e:]
    return source, refusals


# ===========================================================================
# Command line (binder wires the same functions into `check notation`)
# ===========================================================================


def _git_show(repo: Path, ref: str, rel: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{rel}"],
                          capture_output=True, text=True, check=True).stdout


def registry_table_at(repo: Path, ref: str) -> Dict[str, str]:
    """Macro table (\\Name -> LaTeX) of the registry as it was at a git ref."""
    import yaml
    base = "mlsysim/mlsysim/notation/data/"
    names = subprocess.run(["git", "-C", str(repo), "ls-tree", "--name-only", f"{ref}:{base}"],
                           capture_output=True, text=True, check=True).stdout.split()
    table: Dict[str, str] = {}
    for n in names:
        if not n.endswith(".yaml") or n == "exempt.yaml":
            continue
        data = yaml.safe_load(_git_show(repo, ref, base + n)) or {}
        for e in data.get("symbols", []) or []:
            table[e["macro"]] = e["latex"]
            for a in e.get("aliases", []) or []:
                table[a] = e["macro"]
    return table


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Strict notation check (see module docstring).")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--repo", default=str(REPO_HINT))
    ap.add_argument("--json", action="store_true", help="print work packages as JSON")
    ap.add_argument("--find", metavar="WORDS", help="search the registry by meaning")
    ap.add_argument("--verify-rendering", metavar="REF", help="compare each file's rendered math with REF")
    ap.add_argument("--apply", metavar="ASSIGNMENTS_JSON", help="apply assignments to the single path given")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    reg = Registry.load(repo)

    if args.find:
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", args.find)]

        def score(e: RegistryEntry) -> int:
            hay = (re.sub(r"(?<!^)(?=[A-Z])", " ", e.macro) + " " + e.meaning).lower()
            return sum(1 for w in words if re.search(r"\b" + re.escape(w), hay))

        hits = sorted((e for e in reg.entries.values() if score(e) == len(words)), key=lambda e: e.macro)
        hits += sorted((e for e in reg.entries.values() if 0 < score(e) < len(words)), key=lambda e: (-score(e), e.macro))
        for e in hits[:25]:
            print(f"\\{e.macro:<34} {e.latex:<26} {e.kind:<9} {e.domain:<12} {e.meaning}")
        print(f"{len(hits)} match(es)" + (" (first 25; matches on every word come first)" if len(hits) > 25 else ""))
        return 0

    if args.apply:
        (path,) = [Path(p).resolve() for p in args.paths]
        new, refusals = apply_assignments(path.read_text(encoding="utf-8"), json.loads(Path(args.apply).read_text()))
        for r in refusals:
            print("refused:", r)
        path.write_text(new, encoding="utf-8")
        return 1 if refusals else 0

    if args.verify_rendering:
        before_macros = registry_table_at(repo, args.verify_rendering)
        after_macros = reg.macro_table()
        bad: List[RenderMismatch] = []
        for p in args.paths:
            rel = str(Path(p).resolve().relative_to(repo))
            bad += verify_rendering(_git_show(repo, args.verify_rendering, rel),
                                    (repo / rel).read_text(encoding="utf-8"), rel, before_macros, after_macros)
        if args.json:
            print(json.dumps([asdict(b) for b in bad], indent=1))
        else:
            for b in bad:
                print(f"{b.file}:{b.line}: {b.reason}\n    before: {b.before}\n    after:  {b.after}")
            print(f"{len(bad)} math span(s) render differently" if bad else
                  "rendering check passed: every math span renders exactly as before")
        return 1 if bad else 0

    pkgs = []
    for p in args.paths:
        path = Path(p).resolve()
        pkgs.append(work_package(scan_path(path, repo, reg), str(path.relative_to(repo))))
    print(json.dumps(pkgs, indent=1) if args.json else "\n\n".join(render_text(pk) for pk in pkgs))
    return 1 if any(pk["raw_symbols"] for pk in pkgs) else 0


if __name__ == "__main__":
    sys.exit(main())
