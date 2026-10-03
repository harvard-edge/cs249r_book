import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class NotationSymbol:
    id: str
    macro: str
    latex: str
    definition: str
    volume: str
    unit: str
    notes: str
    category: str

class SymbolNamespace:
    """Namespace for symbols belonging to a specific volume."""
    def __init__(self, name: str):
        self._name = name
        self._symbols = {}

    def _add_symbol(self, symbol: NotationSymbol):
        attr_name = symbol.macro.lstrip('\\')
        setattr(self, attr_name, symbol)
        self._symbols[attr_name] = symbol

class NotationRegistry:
    """Global registry of all notations."""
    def __init__(self):
        self.core = SymbolNamespace("core")
        self.vol1 = SymbolNamespace("vol1")
        self.vol2 = SymbolNamespace("vol2")
        self.vol3 = SymbolNamespace("vol3")
        self.vol4 = SymbolNamespace("vol4")
        self._all_symbols = {}
        self._load()

    def _load(self):
        yaml_path = Path(__file__).parent / "registry.yml"
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        for v in data.get('variables', []):
            sym = NotationSymbol(
                id=v.get('id', ''),
                macro=v.get('macro', ''),
                latex=v.get('latex', ''),
                definition=v.get('definition', ''),
                volume=v.get('volume', 'core'),
                unit=v.get('unit', ''),
                notes=v.get('notes', ''),
                category=v.get('category', '')
            )
            
            # Add to flat registry
            attr_name = sym.macro.lstrip('\\')
            self._all_symbols[attr_name] = sym
            setattr(self, attr_name, sym)
            
            # Add to namespace
            ns = getattr(self, sym.volume, None)
            if ns:
                ns._add_symbol(sym)

# Global singleton
SYMBOLS = NotationRegistry()
