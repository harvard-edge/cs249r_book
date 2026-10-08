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
    "Glyph",  # a letter mentioned as a letter, not used as a quantity (README)
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
        table["\\Glyph"] = "#1"
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

# An inline `{python} ...` value is one token: Quarto substitutes a number or a
# formatted string before Pandoc sees the math, so its letters are not symbols.
_TOKEN = re.compile(r"`\{python\}[^`]*`|\\[A-Za-z]+|\\.|[A-Za-z]|\d|\s+|.", re.S)


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
    # several indices, letters or digits (ij, i0, 0j, i,j): a matrix entry's position
    if re.fullmatch(r"(?:%s|\d)(?:,?(?:%s|\d))+" % (idx, idx), body):
        return True
    return False


def _index_like(body: str, registry: "Registry") -> bool:
    """Digits, index letters, index or count macros, joined by + - , (an index expression).

    A rounded rank (\\lceil B q \\rceil) is an index too; inside the brackets any
    registered quantity may appear.
    """
    ex = registry.exempt
    rounded = re.search(r"\\l(ceil|floor)", body) is not None
    rest = re.sub(r"\\(lceil|rceil|lfloor|rfloor|left|right)\b", "", body)
    for m in re.findall(r"\\([A-Za-z]+)", rest):
        e = registry.entries.get(registry.canonical(m))
        if m in ("ell", "cdot", "times"):
            pass
        elif e is None or (e.kind not in ("index", "count") and not rounded):
            return False
        rest = rest.replace("\\" + m, "", 1)
    letters = set(re.findall(r"[A-Za-z]", rest))
    return letters <= set(ex.index) and re.fullmatch(r"[\dA-Za-z+\-,\s]*", rest) is not None


def _top_level_scripts(latex: str) -> Set[str]:
    """The script marks (_ ^) at brace depth 0 of an entry's LaTeX."""
    depth, marks = 0, set()
    for c in latex:
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif depth == 0 and c in "_^":
            marks.add(c)
    return marks


def _macro_decoration_ok(mark: str, grp: List[str], registry: "Registry", entry: Optional["RegistryEntry"]) -> bool:
    """Decorations a registered macro may carry without becoming a new symbol (README).

    Subscript: an index (``_t``, ``_{\\LayerIndex}``, ``_0``). Superscript: a
    power, including a power by another macro (``D^{\\DataScalingExponent}``);
    the inverse and the transpose; a parenthesized index naming a layer, block or
    iteration (``W^{(\\LayerIndex)}``, ``m^{(1)}``); and ``*`` for the optimal value.
    Anything else, a text label above all, is a new symbol.
    """
    # thin spaces and the like only nudge the position; they are not part of the symbol
    grp = [x for x in grp if x not in ("\\,", "\\;", "\\:", "\\!")]
    if _decoration_ok(grp, registry.exempt):
        return True
    body = re.sub(r"\\[,;:!]", "", "".join(x for x in grp if not x.isspace()))
    if mark == "_":
        # an index, or an order statistic's parenthesized rank (p_{(1)})
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1]
        return _index_like(body, registry)
    if body == "*":
        return True
    # the optimum or a bootstrap replicate of an indexed quantity: ^{*(b)}
    if body.startswith("*(") and body.endswith(")"):
        return _index_like(body[2:-1], registry)
    if body.startswith("(") and body.endswith(")"):
        return _index_like(body[1:-1], registry)
    if "\\text" in body or "\\mathrm" in body or "\\operatorname" in body:
        return False
    # a power: macros, digits and arithmetic only
    rest = re.sub(r"\\(?:frac|cdot|times|left|right|over)\b", "", body)
    for m in re.findall(r"\\([A-Za-z]+)", rest):
        if not registry.is_macro(m):
            return False
    rest = re.sub(r"\\[A-Za-z]+", "", rest)
    return re.fullmatch(r"[\d+\-./(){}\s]*", rest) is not None


def judge(math: str, registry: Registry, scope: Optional[Set[str]] = None) -> List[Finding]:
    """Every non-registry symbol in one math span.

    ``scope`` is the set of macro names the chapter may use (its volume's
    notation chapter plus its notation-local list); None skips the scope test.
    """
    ex = registry.exempt
    toks = tokenize(math)
    found: List[Finding] = []
    # x_\Macro reads only the first token of the macro's LaTeX: with \beta_{\text{lat}}
    # the label lands on x, or LaTeX stops on a double subscript. Scripts take {\Macro}.
    for i, t in enumerate(toks):
        if t in ("_", "^"):
            j = _skip_space(toks, i + 1)
            if j < len(toks) and toks[j].startswith("\\"):
                e = registry.entries.get(registry.canonical(toks[j][1:]))
                if e is not None and len(tokenize(e.latex)) > 1:
                    found.append(Finding(t + toks[j], "unbraced-script", i, j + 1))
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
                    raw, after = _with_label(toks, raw, after, ex)
                    found.append(Finding(raw, "form", k, after))
                k, prev = after, "}"
                continue
            if registry.is_macro(name):
                if name in registry.aliases:
                    found.append(Finding(t, "alias", k, nk))
                elif scope is not None and name not in scope:
                    found.append(Finding(t, "out-of-scope", k, nk))
                j = _skip_space(toks, nk)
                entry = registry.entries.get(registry.canonical(name))
                while j < len(toks) and toks[j] in ("_", "^", "'"):
                    if toks[j] == "'":
                        # a prime on a function is its derivative; on anything else, a new symbol
                        if entry is None or entry.kind != "function":
                            found.append(Finding(t + "'", "decorated", k, j + 1))
                        j += 1
                    else:
                        grp, after = _read_group(toks, j + 1)
                        if entry is not None and toks[j] in _top_level_scripts(entry.latex):
                            # \Macro_i where the macro is already p_{\text{gate}}: LaTeX
                            # stops on a double subscript; {\Macro}_i is what renders
                            found.append(Finding(t + toks[j] + "{" + "".join(grp).strip() + "}",
                                                 "double-script", k, after))
                        elif not _macro_decoration_ok(toks[j], grp, registry, entry):
                            found.append(Finding(t + toks[j] + "{" + "".join(grp).strip() + "}",
                                                 "decorated", k, after))
                        j = after
                    j = _skip_space(toks, j)
                k, prev = j, "}"
                continue
            if name in GREEK:
                if name not in ex.commands:
                    raw, end = _with_label(toks, t, nk, ex)
                    found.append(Finding(raw, "greek", k, end))
                    k, prev = end, "}"
                    continue
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
                raw, end = _with_label(toks, t, k + 1, ex)
                found.append(Finding(raw, "letter", k, end))
                prev, k = "}", end
                continue
        prev = t
        k += 1
    return found


def _with_label(toks: List[str], raw: str, k: int, ex: "Exemptions") -> Tuple[str, int]:
    """Extend a raw symbol by a descriptive subscript (A_{\\text{component}}).

    A subscript that is an index, power or transpose is not part of the symbol;
    a label subscript is, because the registry entry's LaTeX carries it.
    """
    j = _skip_space(toks, k)
    if j < len(toks) and toks[j] == "_":
        grp, after = _read_group(toks, j + 1)
        if not _decoration_ok(grp, ex):
            return raw + "_{" + "".join(grp).strip() + "}", after
    return raw, k


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
                                 span.text, f.start, f.end, sug))
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
            "sections": list(sections.values()),
            "symbols": sorted(symbols.values(), key=lambda s: -s["count"])}


_KIND_ADVICE = {
    "letter": "write the macro whose meaning fits",
    "greek": "write the macro whose meaning fits",
    "form": "register the bold/calligraphic/accented symbol; its LaTeX carries the form",
    "decorated": "a new subscript, superscript or prime makes a new symbol: register it",
    "unbraced-script": "a macro used as a subscript or superscript must be braced, x^{\\Macro}: unbraced, LaTeX takes only its first token",
    "double-script": "the macro already carries this script, so LaTeX stops on a double subscript or superscript: group it, {\\Macro}_i",
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


def _span_norm(s: str) -> str:
    """Compare LaTeX by content: no whitespace, no braces around one character."""
    s = re.sub(r"\s+", "", s)
    return re.sub(r"([_^])\{(\\[A-Za-z]+|[^{}\\])\}", r"\1\2", s)


def _macro_extent(toks: List[str], s0: int, s1: int, latex: Optional[str]) -> int:
    """End token of the span a macro replaces.

    A macro whose LaTeX is several tokens (``P_t``, ``\\Pr(E)``) replaces the
    whole matching run, not just its head symbol; otherwise ``P_t`` would become
    ``\\Macro_t`` and print the subscript twice. Without a longer match, the
    site's own tokens.
    """
    if not latex:
        return s1
    want = _span_norm(latex)
    best = s1
    acc = ""
    for j in range(s0, len(toks)):
        acc += toks[j]
        got = _span_norm(acc)
        # never end on whitespace: the normalized text is the same, but taking it
        # would swallow the space or line break after the symbol
        if got == want and j + 1 > best and not toks[j].isspace():
            best = j + 1
        # mid-group text (B^{) is not a prefix of the normalized LaTeX, so bound
        # the search by length instead of stopping at the first mismatch
        if len(acc) > 2 * len(latex) + 8:
            break
    return best


def _apply_exemptions() -> Exemptions:
    try:
        return Exemptions.load(REPO_HINT)
    except Exception:          # outside a checkout: the defaults
        return Exemptions()


_EX_FOR_APPLY = _apply_exemptions()


def apply_assignments(source: str, assignments: List[Dict],
                      latex_of: Optional[Dict[str, str]] = None) -> Tuple[str, List[str]]:
    """Replace symbols at exact positions in body math.

    Each assignment is {"line", "col", "token_start", "token_end", "raw", "macro"},
    as produced from a work package. An assignment whose tokens no longer read
    ``raw`` is refused, never guessed. With ``latex_of`` (macro -> LaTeX), a
    macro whose LaTeX spans several tokens replaces all of them. Returns (new
    source, refusals).
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
            if _span_norm(got) != _span_norm(a["raw"]):   # T_N and T_{N} are the same text
                refusals.append(f"line {span.line}: expected {a['raw']!r} at tokens {s0}-{s1}, found {got!r}")
                continue
            keep = ""
            if latex_of:
                lx = latex_of.get("\\" + a["macro"].lstrip("\\"))
                s1 = _macro_extent(toks, s0, s1, lx)
                # a site wider than the macro's LaTeX (A_{i0} for a macro rendering A):
                # replace only the part the macro renders and keep the rest
                if lx and a["macro"].lstrip("\\") != "Glyph":
                    want, acc = _span_norm(lx), ""
                    for j in range(s0, s1):
                        acc += toks[j]
                        if _span_norm(acc) == want and j + 1 < s1:
                            rest = toks[j + 1:s1]
                            r0 = _skip_space(rest, 0)
                            # keep only an index or power; a label belongs to the replaced symbol
                            if r0 < len(rest) and rest[r0] in ("_", "^"):
                                grp, _ = _read_group(rest, r0 + 1)
                                if _decoration_ok(grp, _EX_FOR_APPLY):
                                    keep = "".join(rest)
                            break
            if any(not (s1 <= x0 or s0 >= x1) for x0, x1 in taken):
                refusals.append(f"line {span.line}: assignments overlap at tokens {s0}-{s1}")
                continue
            taken.append((s0, s1))
            rep = "\\" + a["macro"].lstrip("\\")
            if a["macro"].lstrip("\\") == "Glyph":
                rep = "\\Glyph{" + got.strip() + "}"
            j = s1
            while j < len(toks) and toks[j].isspace():
                j += 1
            if keep:
                rep += keep
            elif s1 < len(toks) and toks[s1][:1].isalpha():
                rep += " "        # \Macro directly followed by a letter would merge into one control word
            edits.append((body_start + offsets[s0], body_start + offsets[s1], rep))
    for s, e, rep in sorted(edits, key=lambda x: -x[0]):
        source = source[:s] + rep + source[e:]
    return source, refusals


def glyph_collisions(source: str, registry: Registry) -> List[Dict]:
    """Different symbols that look identical in one section of a chapter.

    Two registered macros may share LaTeX (``p`` is a pass probability in one
    place and a candidate rate in another), but never within one top-level
    (``##``) section: the reader, who reads a section at a time, would see one
    letter mean two things. Returns one record per shared rendering and
    section: the section, the LaTeX, the macros, and the first line of each.
    """
    table = registry.macro_table()

    def expand(s: str, depth: int = 0) -> str:
        out = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])",
                     lambda m: table.get("\\" + m.group(1), m.group(0)) if m.group(1) != "Glyph" else m.group(0), s)
        return out if out == s or depth > 8 else expand(out, depth + 1)

    h2_at, current = [], ""
    in_fence = False
    for line in source.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and line.startswith("## "):
            current = line[3:].strip()
        h2_at.append(current)

    first: Dict[Tuple[str, str], Dict[str, int]] = collections.defaultdict(dict)
    for span in extract_math(source):
        if span.source not in ("body", "caption"):
            continue
        section = h2_at[span.line - 1] if span.line - 1 < len(h2_at) else ""
        for m in re.finditer(r"\\([A-Za-z]+)(?![A-Za-z])", span.text):
            name = m.group(1)
            if not registry.is_macro(name) or name == "Glyph":
                continue
            canon = registry.canonical(name)
            look = _span_norm(expand(registry.entries[canon].latex)) if canon in registry.entries else ""
            if look:
                first[(section, look)].setdefault(canon, span.line)
    allowed = _same_glyph_ok()
    out = []
    for (sec, look), ms in sorted(first.items()):
        names = sorted(ms)
        # pairs the registry declares compatible (a quantity and its special case)
        if len(names) > 1 and all(frozenset((x, y)) in allowed
                                  for i, x in enumerate(names) for y in names[i + 1:]):
            continue
        if len(names) > 1:
            out.append({"section": sec, "latex": look, "macros": names, "lines": [ms[m] for m in names]})
    return out


def _same_glyph_ok() -> Set[frozenset]:
    """Pairs listed under same_glyph_ok in data/vocabulary.yaml."""
    path = REPO_HINT / "mlsysim" / "mlsysim" / "notation" / "data" / "vocabulary.yaml"
    try:
        import yaml  # local: binder imports this module eagerly
        rows = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("same_glyph_ok") or []
    except Exception:
        return set()
    return {frozenset(m.lstrip("\\") for m in r["macros"]) for r in rows if len(r.get("macros", [])) == 2}


def assign_from_glossary(package: Dict, glossary: Dict, registry: Registry) -> Dict:
    """Turn a chapter glossary into exact edits, without judgment.

    A site gets the glossary row whose glyph matches and whose lines include the
    site's line. A retired alias gets its canonical macro unless a glossary row
    claims it (the alias was used with a meaning its canonical entry does not
    have). A site with no matching row is ``unresolved``; a site with several is
    ``ambiguous``: both go back to a person or model, nothing is guessed.
    """
    rows = glossary.get("rows", [])
    out = {"assignments": [], "unresolved": [], "ambiguous": [], "not_body": []}
    for sec in package["sections"]:
        for site in sec["sites"]:
            site = dict(site, section=sec["heading"])
            if site["source"] != "body":
                out["not_body"].append(site)
                continue
            def matches(r: Dict) -> bool:
                if (r.get("symbol") or r.get("glyph")) != site["raw"]:
                    return False
                keys = {(x["line"], x["col"], x["token_start"]) for x in r.get("sites", [])}
                if (site["line"], site["col"], site["token_start"]) in keys:
                    return True
                return site["line"] in r.get("lines", [])

            exact = [r for r in rows if (site["line"], site["col"], site["token_start"])
                     in {(x["line"], x["col"], x["token_start"]) for x in r.get("sites", [])}
                     and (r.get("symbol") or r.get("glyph")) == site["raw"]]
            hits = exact or [r for r in rows if matches(r)]
            if site["kind"] == "alias" and not hits:
                macro = registry.canonical(site["raw"].lstrip("\\"))
                out["assignments"].append(dict(site, macro=macro, via="alias"))
                continue
            if len(hits) == 1:
                out["assignments"].append(dict(site, macro=hits[0]["macro"], via="glossary", meaning=hits[0].get("meaning", "")))
            elif not hits:
                out["unresolved"].append(site)
            else:
                out["ambiguous"].append(dict(site, options=[h["macro"] for h in hits]))
    return out


def concordance(repo: Path, macro: str, registry: Registry, width: int = 220) -> List[Dict]:
    """Every use of one macro in the book, with the sentence around it.

    The reader sees in one place whether all uses mean what the registry says;
    this is how a macro reused for the wrong meaning gets caught.
    """
    name = registry.canonical(macro.lstrip("\\"))
    names = {name} | {a for a, c in registry.aliases.items() if c == name}
    pat = re.compile(r"\\(%s)(?![A-Za-z])" % "|".join(map(re.escape, sorted(names))))
    out: List[Dict] = []
    for path in sorted((repo / "books").glob("vol*/**/*.qmd")):
        if "_build" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for span in extract_math(text):
            if not pat.search(span.text):
                continue
            line = text.split("\n")[span.line - 1]
            pos = max(0, span.col - 1 - width // 2)
            out.append({"file": str(path.relative_to(repo)), "line": span.line,
                        "context": line[pos:pos + width].strip(), "math": span.text.strip()[:160]})
    return out


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
        if not n.endswith(".yaml") or n in ("exempt.yaml", "vocabulary.yaml"):
            continue
        data = yaml.safe_load(_git_show(repo, ref, base + n)) or {}
        for e in data.get("symbols", []) or []:
            table[e["macro"]] = e["latex"]
            for a in e.get("aliases", []) or []:
                table[a] = e["macro"]
    table["\\Glyph"] = "#1"     # defined by the build, not the registry
    return table


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Strict notation check (see module docstring).")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--repo", default=str(REPO_HINT))
    ap.add_argument("--json", action="store_true", help="print work packages as JSON")
    ap.add_argument("--find", metavar="WORDS", help="search the registry by meaning")
    ap.add_argument("--verify-rendering", metavar="REF", help="compare each file's rendered math with REF")
    ap.add_argument("--apply", metavar="ASSIGNMENTS_JSON", help="apply assignments to the single path given")
    ap.add_argument("--assign", metavar="GLOSSARY_JSON", help="derive exact edits for the single path from a glossary")
    ap.add_argument("--concordance", metavar="MACRO", help="every use of one macro in the book, in context")
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

    if args.concordance:
        name = reg.canonical(args.concordance.lstrip("\\"))
        e = reg.entries.get(name)
        uses = concordance(repo, name, reg)
        if args.json:
            print(json.dumps({"macro": name, "meaning": e.meaning if e else "", "uses": uses}, indent=1))
        else:
            print(f"\\{name} = {e.meaning if e else '(not registered)'}  [{len(uses)} uses]")
            for u in uses:
                print(f"  {u['file']}:{u['line']}: {u['context']}")
        return 0

    if args.assign:
        (path,) = [Path(p).resolve() for p in args.paths]
        pkg = work_package(scan_path(path, repo, reg, scoped=False), str(path.relative_to(repo)))
        res = assign_from_glossary(pkg, json.loads(Path(args.assign).read_text()), reg)
        print(json.dumps(res, indent=1))
        sys.stderr.write(f"{len(res['assignments'])} assigned, {len(res['unresolved'])} unresolved, "
                         f"{len(res['ambiguous'])} ambiguous, {len(res['not_body'])} outside body math\n")
        return 0 if not (res["unresolved"] or res["ambiguous"]) else 1

    if args.apply:
        (path,) = [Path(p).resolve() for p in args.paths]
        new, refusals = apply_assignments(path.read_text(encoding="utf-8"), json.loads(Path(args.apply).read_text()),
                                          reg.macro_table())
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
