"""Pydantic schema and types for mathematical notation in MLSysIM.

Every symbol is an immutable, strictly validated record that represents
the single source of truth for notation across the MLSysBook series.
"""
from __future__ import annotations

import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, field_validator

from ..fmt import MarkdownStr

VolumeLiteral = Literal["core", "vol1", "vol2", "vol3", "vol4"]


class NotationSymbol(BaseModel):
    """A mathematically rigorous, validated notation symbol in MLSysIM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    macro: str
    latex: str
    definition: MarkdownStr
    unit: MarkdownStr = MarkdownStr("")
    notes: MarkdownStr = MarkdownStr("")
    category: str
    volume: str = "core"

    @field_validator("macro")
    @classmethod
    def validate_macro_name(cls, v: str) -> str:
        """Enforce leading backslash and strict PascalCase (e.g. \\WalkingSpeed)."""
        if not re.match(r"^\\[A-Z][a-zA-Z0-9]*$", v):
            raise ValueError(
                f"Macro '{v}' must be strict PascalCase starting with a backslash "
                "(e.g. '\\DataVolume', not '\\data_volume' or '\\Endtoend')."
            )
        return v

    @field_validator("latex", "definition", "category")
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        s = v.strip()
        if not s:
            raise ValueError(f"Field '{info.field_name}' must not be empty.")
        return s

    @property
    def symbol(self) -> MarkdownStr:
        """Rendered math symbol (e.g. '$D_{\\text{vol}}$') verbatim for Quarto."""
        return MarkdownStr(f"${self.latex}$")

    @property
    def with_unit(self) -> MarkdownStr:
        """Symbol with parenthesized unit (e.g. '$D_{\\text{vol}}$ (bytes)')."""
        if self.unit:
            return MarkdownStr(f"${self.latex}$ ({self.unit})")
        return self.symbol

    @property
    def full_spec(self) -> MarkdownStr:
        """Definition, symbol, and unit (e.g. 'Data Volume, $D_{\\text{vol}}$ [bytes]')."""
        u = f" [{self.unit}]" if self.unit else ""
        return MarkdownStr(f"{self.definition}, ${self.latex}${u}")

    def _repr_markdown_(self) -> str:
        """Jupyter/Quarto hook: render symbol verbatim into prose when `{python} Symbol` is used."""
        return str(self.symbol)

    def __str__(self) -> str:
        return str(self.symbol)
