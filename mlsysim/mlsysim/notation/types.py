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
from typing import List, Literal, Tuple

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
    if s.kind == "scalar" and not str(s.unit).strip():
        problems.append("a scalar needs a unit ('dimensionless' if it has none)")
    return problems
