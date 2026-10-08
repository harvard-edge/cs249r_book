"""Schema for mathematical notation symbols in MLSysIM.

A symbol is an immutable record: the macro the book writes, the LaTeX readers
see, what it means, and what kind of quantity it is. The rules for names and
fields are explained for authors in ``data/README.md``; this module states them
as code, so the registry cannot load if an entry breaks them.

Two levels of rule:

* **Shape rules** (``NotationSymbol`` validators) hold for every entry, always:
  the macro is a valid TeX control word, required fields are present, ``kind``
  is one of the allowed values.
* **Convention rules** (``convention_problems``) describe what a good entry
  looks like: a meaningful name, a plain-language definition, LaTeX that fits
  the kind. The registry test applies them to every entry; there is no
  exemption list.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import yaml

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..fmt import MarkdownStr

Kind = Literal["scalar", "count", "index", "vector", "matrix", "set", "function", "distribution", "operator", "record"]
KINDS: Tuple[str, ...] = ("scalar", "count", "index", "vector", "matrix", "set", "function", "distribution", "operator", "record")

# TeX control words are letters only: "\Foo2" is "\Foo" followed by "2".
_MACRO = re.compile(r"^\\[A-Z][A-Za-z]*$")

# Whole words that never belong in a name (see README, "Naming").
_BANNED_WORDS = {
    "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Other",
    "And", "Or", "Of", "The", "Var", "Symbol", "Value", "Quantity", "Term", "Macro",
    "Bold", "Hat", "Bar", "Tilde", "Dot", "Cal", "Chapter", "Appendix",
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa",
    "Lambda", "Mu", "Nu", "Xi", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi",
    "Psi", "Omega",
}
# Names that are one word on purpose (the series core).
_ONE_WORD_OK = {
    "Time", "Bandwidth", "Latency", "Parameters", "Operations", "Loss", "Energy", "Power",
    "Throughput", "Speedup", "Trajectory", "Policy", "Reward", "Mass", "Gravity", "Pose",
    "Velocity", "Advantage",
}
# Bold anywhere in the symbol: \\hat{\\mathbf{y}} is a bold vector with a hat.
_BOLD = re.compile(r"\\(mathbf|boldsymbol|bm)\b")
# Calligraphic or blackboard anywhere: \\partial \\mathcal{C} is the boundary of a set.
_SET = re.compile(r"\\(mathcal|mathbb|mathscr)\b")


# --- controlled vocabulary (data/vocabulary.yaml) ----------------------------

_VOCAB_PATH = Path(__file__).parent / "data" / "vocabulary.yaml"
_vocab_cache: Optional[Dict] = None


def vocabulary() -> Dict:
    """The units, quantity words and idioms in ``data/vocabulary.yaml``."""
    global _vocab_cache
    if _vocab_cache is None:
        _vocab_cache = yaml.safe_load(_VOCAB_PATH.read_text(encoding="utf-8"))
    return _vocab_cache


_UNIT_TOKEN = re.compile(r"\s*([A-Za-z]+|\d+|[*/^()-])")


def unit_dimension(unit: str) -> Optional[Dict[str, int]]:
    """Base dimensions of a unit expression, or None if it uses an unknown unit.

    Units combine with ``*``, ``/``, ``^n`` and parentheses; ``a/b/c`` is
    ``a/(b*c)``. ``dimensionless`` is the empty dimension.
    """
    u = unit.strip()
    if u == "dimensionless":
        return {}
    table = vocabulary()["units"]
    toks = [m.group(1) for m in _UNIT_TOKEN.finditer(u)]
    if "".join(toks) != u.replace(" ", ""):
        return None
    pos = 0

    def add(a: Dict[str, int], b: Dict[str, int], sign: int) -> Dict[str, int]:
        out = dict(a)
        for k, v in b.items():
            out[k] = out.get(k, 0) + sign * v
        return {k: v for k, v in out.items() if v}

    def factor() -> Optional[Dict[str, int]]:
        nonlocal pos
        if pos >= len(toks):
            return None
        tok = toks[pos]
        if tok == "(":
            pos += 1
            inner = product()
            if inner is None or pos >= len(toks) or toks[pos] != ")":
                return None
            pos += 1
            base = inner
        elif tok == "1":
            pos += 1
            base = {}
        elif tok in table:
            pos += 1
            base = dict(table[tok] or {})
        else:
            return None
        if pos < len(toks) and toks[pos] == "^":
            pos += 1
            neg = pos < len(toks) and toks[pos] == "-"
            pos += neg
            if pos >= len(toks) or not toks[pos].isdigit():
                return None
            n = int(toks[pos]) * (-1 if neg else 1)
            pos += 1
            base = {k: v * n for k, v in base.items()}
        return base

    def product() -> Optional[Dict[str, int]]:
        nonlocal pos
        acc = factor()
        if acc is None:
            return None
        while pos < len(toks) and toks[pos] in "*/":
            op = toks[pos]
            pos += 1
            nxt = factor()
            if nxt is None:
                return None
            acc = add(acc, nxt, 1 if op == "*" else -1)
        return acc

    dim = product()
    return dim if dim is not None and pos == len(toks) else None


def quantity_word(name: str) -> Optional[str]:
    """The vocabulary word a macro name ends in (longest match), if any."""
    words = sorted(vocabulary()["words"], key=len, reverse=True)
    for w in words:
        if name.endswith(w) and (len(name) == len(w) or name[-len(w) - 1].islower() or name[-len(w)].isupper()):
            return w
    return None


def _word_problems(s: "NotationSymbol", name: str) -> List[str]:
    if any(name.endswith(i) for i in (vocabulary().get("idioms") or {})):
        return []
    w = quantity_word(name)
    if w is None:
        return []
    rule = vocabulary()["words"][w]
    probs: List[str] = []
    if s.kind not in rule["kinds"]:
        probs.append(f"a name ending in '{w}' must have kind {' or '.join(rule['kinds'])}, not '{s.kind}' "
                     f"(rename the entry, or fix its kind)")
    unit = str(s.unit).strip()
    dim = unit_dimension(unit) if unit and unit != "varies" else None
    if dim is None:
        return probs
    if "dimension" in rule and dim != rule["dimension"]:
        probs.append(f"a name ending in '{w}' needs a unit of dimension {rule['dimension'] or 'dimensionless'}; "
                     f"'{unit}' is {dim or 'dimensionless'}")
    if rule.get("per_time_or_share") and dim and dim.get("time", 0) != -1:
        probs.append(f"a name ending in '{w}' is per unit of time or a dimensionless share; '{unit}' is neither")
    if rule.get("currency") and dim.get("currency", 0) != 1:
        probs.append(f"a name ending in '{w}' is an amount of money; '{unit}' is not in USD")
    return probs


def split_words(name: str) -> List[str]:
    """Split a PascalCase name into words: 'KvCacheMemory' -> ['Kv', 'Cache', 'Memory']."""
    return re.findall(r"[A-Z][a-z]*", name)


class NotationSymbol(BaseModel):
    """One registered symbol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    macro: str
    latex: str
    definition: MarkdownStr
    kind: Kind = "scalar"
    unit: MarkdownStr = MarkdownStr("")
    notes: MarkdownStr = MarkdownStr("")
    aliases: Tuple[str, ...] = Field(default_factory=tuple)
    # Why the LaTeX keeps a field convention that the kind rule would reject
    # (theta as a non-bold parameter vector, upright SE(3)). Excuses only that rule.
    render_exception: MarkdownStr = MarkdownStr("")
    # Set by the loader from the file the entry sits in (data/<domain>.yaml).
    domain: str = ""
    # Legacy fields kept readable while the registry migrates; never written by new entries.
    category: str = ""
    volume: str = ""

    @field_validator("macro")
    @classmethod
    def _macro_is_control_word(cls, v: str) -> str:
        if not _MACRO.match(v):
            raise ValueError(f"macro {v!r} must be a backslash followed by letters only, starting uppercase (e.g. \\LinkBandwidth)")
        return v

    @field_validator("aliases")
    @classmethod
    def _aliases_are_control_words(cls, v: Tuple[str, ...]) -> Tuple[str, ...]:
        for a in v:
            if not _MACRO.match(a):
                raise ValueError(f"alias {a!r} must be a backslash followed by letters only")
        return tuple(v)

    @field_validator("latex", "definition")
    @classmethod
    def _required(cls, v: str, info) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError(f"field '{info.field_name}' must not be empty")
        return s

    # --- presentation helpers (unchanged API) -------------------------------

    @property
    def symbol(self) -> MarkdownStr:
        """Rendered math symbol (e.g. '$D_{\\text{vol}}$') for Quarto prose."""
        return MarkdownStr(f"${self.latex}$")

    @property
    def with_unit(self) -> MarkdownStr:
        """Symbol with parenthesized unit (e.g. '$D_{\\text{vol}}$ (bytes)')."""
        if self.unit:
            return MarkdownStr(f"${self.latex}$ ({self.unit})")
        return self.symbol

    @property
    def full_spec(self) -> MarkdownStr:
        """Definition, symbol, and unit (e.g. 'Data volume, $D_{\\text{vol}}$ [bytes]')."""
        u = f" [{self.unit}]" if self.unit else ""
        return MarkdownStr(f"{self.definition}, ${self.latex}${u}")

    def _repr_markdown_(self) -> str:
        return str(self.symbol)

    def __str__(self) -> str:
        return str(self.symbol)


def convention_problems(s: NotationSymbol) -> List[str]:
    """Everything about an entry that breaks the naming and field conventions."""
    problems: List[str] = []
    name = s.macro[1:]
    words = split_words(name)
    if "".join(words) != name:
        problems.append("name must be PascalCase words only")
    banned = [w for w in words if w in _BANNED_WORDS]
    if banned:
        problems.append(f"name contains {', '.join(banned)} (numbers, joining words, glyph names and placeholders are not allowed)")
    if len(words) == 1 and name not in _ONE_WORD_OK:
        problems.append("one-word names are reserved for the series core; add a qualifier")
    if len(words) > 5:
        problems.append("name is longer than five words")
    if re.match(r"^A[A-Z]", name):
        problems.append("name starts with the article 'A'")
    d = str(s.definition).strip()
    if d.lower() in {"(none)", "none", "n/a", "tbd"}:
        problems.append("definition is a placeholder")
    if "$" in d or "\\" in d:
        problems.append("definition contains LaTeX (put it in notes)")
    if ";" in d:
        problems.append("definition lists several meanings (one entry per meaning)")
    if d.lower().replace(" ", "") == name.lower():
        problems.append("definition only repeats the name; say what the symbol means")
    if not str(s.render_exception).strip():
        if s.kind in ("vector", "matrix") and not _BOLD.search(s.latex):
            problems.append(f"kind '{s.kind}' must render bold (\\mathbf, \\boldsymbol or \\bm)")
        if s.kind == "set" and not _SET.search(s.latex):
            problems.append("kind 'set' must render calligraphic or blackboard (\\mathcal, \\mathbb, \\mathscr)")
        if s.kind in ("scalar", "count", "index") and _BOLD.search(s.latex):
            problems.append(f"kind '{s.kind}' must not render bold")
    unit = str(s.unit).strip()
    if s.kind == "scalar" and not unit:
        problems.append("a scalar needs a unit ('dimensionless' if it has none)")
    if s.kind in ("count", "index", "record") and unit:
        problems.append(f"kind '{s.kind}' has no unit (say what is counted in the definition)")
    if unit and unit != "varies" and unit_dimension(unit) is None:
        problems.append(f"unit '{unit}' is not built from the units in data/vocabulary.yaml")
    if unit == "varies" and not str(s.notes).strip():
        problems.append("unit 'varies' needs notes saying what it depends on")
    problems.extend(_word_problems(s, name))
    return problems
