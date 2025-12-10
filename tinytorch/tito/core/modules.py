"""
Module definitions for TinyTorch CLI.

Auto-discovers modules from the src/ directory structure.
This ensures the CLI is always in sync with actual module folders.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Optional


def _find_project_root() -> Path:
    """Find the TinyTorch project root directory."""
    # Start from this file's location and walk up
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / 'pyproject.toml').exists() and (current / 'src').exists():
            return current
        current = current.parent
    # Fallback to cwd
    return Path.cwd()


@lru_cache(maxsize=1)
def _discover_modules() -> Dict[str, str]:
    """
    Auto-discover modules from src/ directory.
    
    Scans for directories matching pattern: NN_name (e.g., 01_tensor, 15_quantization)
    Returns: {"01": "01_tensor", "02": "02_activations", ...}
    """
    project_root = _find_project_root()
    src_dir = project_root / 'src'
    
    if not src_dir.exists():
        return {}
    
    mapping = {}
    pattern = re.compile(r'^(\d{2})_(\w+)$')
    
    for entry in sorted(src_dir.iterdir()):
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                num = match.group(1)  # "01", "15", etc.
                mapping[num] = entry.name
    
    return mapping


def get_module_mapping() -> Dict[str, str]:
    """Get the module number to folder name mapping (auto-discovered)."""
    return _discover_modules().copy()


def get_module_name(module_input: str) -> Optional[str]:
    """Get the folder name for a module number or return None if not found."""
    normalized = normalize_module_number(module_input)
    return _discover_modules().get(normalized)


def get_module_display_name(module_input: str) -> str:
    """
    Get a human-readable display name from module folder name.
    E.g., "15_quantization" -> "Quantization"
    """
    folder = get_module_name(module_input)
    if folder and "_" in folder:
        return folder.split("_", 1)[1].replace("_", " ").title()
    return "Unknown"


def get_next_module(current_module: str) -> Optional[Tuple[str, str, str]]:
    """
    Get the next module after the current one.
    
    Returns: (module_number, folder_name, display_name) or None if no next module.
    """
    mapping = _discover_modules()
    normalized = normalize_module_number(current_module)
    
    try:
        current_num = int(normalized)
        next_num = f"{current_num + 1:02d}"
        if next_num in mapping:
            folder = mapping[next_num]
            display = get_module_display_name(next_num)
            return (next_num, folder, display)
    except ValueError:
        pass
    return None


def normalize_module_number(module_input: str) -> str:
    """
    Normalize module input to 2-digit format.
    
    Examples:
        "1" -> "01"
        "15" -> "15"
        "15_quantization" -> "15"
    """
    # If it's a pure number
    if module_input.isdigit():
        return f"{int(module_input):02d}"
    # If it's a folder name like "15_quantization", extract the number
    if "_" in module_input:
        prefix = module_input.split("_")[0]
        if prefix.isdigit():
            return f"{int(prefix):02d}"
    return module_input


def get_total_modules() -> int:
    """Get the total number of discovered modules."""
    return len(_discover_modules())


def module_exists(module_input: str) -> bool:
    """Check if a module exists."""
    return get_module_name(module_input) is not None


def clear_cache():
    """Clear the module discovery cache (useful after adding new modules)."""
    _discover_modules.cache_clear()
