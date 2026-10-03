"""MLSysIM Mathematical Notation Registry.

Exposes canonical, validated notation symbols across all 4 volumes of MLSysBook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List
import yaml

from .types import NotationSymbol

_DATA_FILE = Path(__file__).parent / "data" / "symbols.yaml"


class NotationRegistry:
    """Registry of all canonical notation symbols in MLSysBook."""

    def __init__(self, data_file: Path | None = None) -> None:
        self._symbols: Dict[str, NotationSymbol] = {}
        self._by_macro: Dict[str, NotationSymbol] = {}
        self._load(data_file or _DATA_FILE)

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for item in data.get("symbols", []):
            sym = NotationSymbol(**item)
            attr_name = sym.macro.lstrip("\\")
            self._symbols[attr_name] = sym
            self._by_macro[sym.macro] = sym
            setattr(self, attr_name, sym)

    def __getitem__(self, name: str) -> NotationSymbol:
        clean_name = name.lstrip("\\")
        if clean_name in self._symbols:
            return self._symbols[clean_name]
        raise KeyError(f"Notation symbol '{name}' not found in registry.")

    def __iter__(self) -> Iterator[NotationSymbol]:
        return iter(self._symbols.values())

    def __len__(self) -> int:
        return len(self._symbols)

    def get_by_macro(self, macro: str) -> NotationSymbol | None:
        return self._by_macro.get(macro)

    def list(self) -> List[NotationSymbol]:
        """Return symbols sorted alphabetically by macro name."""
        return sorted(self._symbols.values(), key=lambda s: s.macro)

    def for_category(self, category: str) -> List[NotationSymbol]:
        """Return symbols belonging to a specific notation category."""
        return [s for s in self.list() if s.category == category]

    def for_volume(self, volume: str) -> List[NotationSymbol]:
        """Return symbols belonging to a specific volume (e.g. 'core', 'vol1', 'vol4')."""
        return [s for s in self.list() if s.volume == volume]


# Global singletons
Notation = NotationRegistry()
Symbols = Notation
SYMBOLS = Notation
