"""MLSysIM mathematical notation registry.

Loads every ``data/<domain>.yaml`` file into one validated registry. The domain
of an entry is the name of the file it sits in. Loading fails loudly: a
duplicate macro, an alias that collides with a name, or an entry that breaks the
schema raises immediately, because a half-loaded registry would make every
notation check wrong.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List

import yaml

from .types import NotationSymbol

DATA_DIR = Path(__file__).parent / "data"
_NOT_SYMBOL_FILES = {"exempt.yaml"}


class NotationRegistry:
    """All canonical notation symbols of MLSysBook."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._symbols: Dict[str, NotationSymbol] = {}
        self._by_macro: Dict[str, NotationSymbol] = {}
        self._aliases: Dict[str, str] = {}
        self._load(data_dir or DATA_DIR)

    def _load(self, data_dir: Path) -> None:
        files = sorted(p for p in data_dir.glob("*.yaml") if p.name not in _NOT_SYMBOL_FILES)
        if not files:
            raise RuntimeError(f"no notation data files found in {data_dir}")
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            domain = path.stem
            for item in data.get("symbols", []) or []:
                item = dict(item)
                item.setdefault("domain", domain)
                sym = NotationSymbol(**item)
                if sym.macro in self._by_macro:
                    other = self._by_macro[sym.macro].domain
                    raise ValueError(f"{sym.macro} is defined twice ({other}.yaml and {domain}.yaml)")
                attr = sym.macro.lstrip("\\")
                self._symbols[attr] = sym
                self._by_macro[sym.macro] = sym
                setattr(self, attr, sym)
        for sym in self._by_macro.values():
            for alias in sym.aliases:
                if alias in self._by_macro:
                    raise ValueError(f"alias {alias} of {sym.macro} is also a symbol name")
                if alias in self._aliases:
                    raise ValueError(f"alias {alias} points to both {self._aliases[alias]} and {sym.macro}")
                self._aliases[alias] = sym.macro

    # --- lookup ----------------------------------------------------------------

    def __getitem__(self, name: str) -> NotationSymbol:
        key = "\\" + name.lstrip("\\")
        key = self._aliases.get(key, key)
        if key in self._by_macro:
            return self._by_macro[key]
        raise KeyError(f"Notation symbol '{name}' not found in registry.")

    def __iter__(self) -> Iterator[NotationSymbol]:
        return iter(self._by_macro.values())

    def __len__(self) -> int:
        return len(self._by_macro)

    def get_by_macro(self, macro: str) -> NotationSymbol | None:
        return self._by_macro.get(self._aliases.get(macro, macro))

    def list(self) -> List[NotationSymbol]:
        """All symbols, sorted by macro name."""
        return sorted(self._by_macro.values(), key=lambda s: s.macro)

    def aliases(self) -> Dict[str, str]:
        """Retired macro name -> canonical macro name."""
        return dict(self._aliases)

    def for_domain(self, domain: str) -> List[NotationSymbol]:
        return [s for s in self.list() if s.domain == domain]

    # Legacy accessors, kept while consumers migrate.
    def for_category(self, category: str) -> List[NotationSymbol]:
        return [s for s in self.list() if s.category == category]

    def for_volume(self, volume: str) -> List[NotationSymbol]:
        return [s for s in self.list() if s.volume == volume]


Notation = NotationRegistry()
Symbols = Notation
SYMBOLS = Notation
