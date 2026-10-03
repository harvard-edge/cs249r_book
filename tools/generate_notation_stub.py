#!/usr/bin/env python3
"""Generate registry.pyi type stub for MLSysIM Notation."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "mlsysim"))
from mlsysim.notation import Notation

lines = [
    "# Auto-generated type stubs for mlsysim.Notation / mlsysim.Symbols",
    "from typing import Dict, Iterator, List",
    "from .types import NotationSymbol",
    "",
    "class NotationRegistry:",
    "    def __getitem__(self, name: str) -> NotationSymbol: ...",
    "    def __iter__(self) -> Iterator[NotationSymbol]: ...",
    "    def __len__(self) -> int: ...",
    "    def list(self) -> List[NotationSymbol]: ...",
    "    def for_category(self, category: str) -> List[NotationSymbol]: ...",
    "    def for_volume(self, volume: str) -> List[NotationSymbol]: ...",
    "",
]

for s in Notation.list():
    attr = s.macro.lstrip("\\")
    doc = f"{s.definition} (${s.latex}$)"
    if s.unit:
        doc += f" [{s.unit}]"
    lines.append(f"    {attr}: NotationSymbol  # {doc}")

lines.extend([
    "",
    "Notation: NotationRegistry",
    "Symbols: NotationRegistry",
    "SYMBOLS: NotationRegistry",
])

pyi_path = PROJECT_ROOT / "mlsysim/mlsysim/notation/registry.pyi"
pyi_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(Notation)} symbol stubs to {pyi_path}")
