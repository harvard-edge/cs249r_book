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

class NotationRegistry:
    """Global flat registry of all notations."""
    def __init__(self):
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
            
            attr_name = sym.macro.lstrip('\\')
            self._all_symbols[attr_name] = sym
            setattr(self, attr_name, sym)

    def __iter__(self):
        return iter(self._all_symbols.values())

    def render_table(self, category: str):
        """Prints a Markdown table for a specific category, for Quarto's output: asis"""
        vars_list = [v for v in self._all_symbols.values() if v.category == category]
        vars_list.sort(key=lambda x: x.macro)
        
        is_disambig = "disambiguation" in category.lower() or "resolving collisions" in category.lower()
        
        print("::: {tbl-colwidths=\"[10,20,13,57]\"}\n")
        if is_disambig:
            print("| **Symbol** | **Canonical Convention** | **Colliding Meaning** | **Disambiguation Rule** |")
        else:
            print("| **Symbol** | **Definition** | **Unit** | **Notes** |")
        print("|:---|:---|:---|:---|")
        
        for var in vars_list:
            symbol = f"**${var.latex}$**"
            defn = f"**{var.definition}**"
            unit = getattr(var, 'unit', '') or ''
            notes = (getattr(var, 'notes', '') or '').replace('\n', ' ')
            print(f"| {symbol} | {defn} | {unit} | {notes} |")
        
        print("\n:::\n")

# Global singleton
SYMBOLS = NotationRegistry()
