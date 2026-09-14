"""
Module definitions for TinyTorch CLI.

Auto-discovers modules from the src/ directory structure.
This ensures the CLI is always in sync with actual module folders.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Optional


@dataclass
class ModuleMetadata:
    """Metadata extracted from module.yaml file."""
    title: str
    subtitle: str
    description: str


# Required fields in module.yaml
REQUIRED_METADATA_FIELDS = {'title', 'subtitle', 'description'}


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


def _parse_yaml_file(content: str) -> Dict[str, str]:
    """
    Parse simple YAML content (key: value format).

    This is a lightweight parser that handles the simple module.yaml format
    without requiring the pyyaml dependency.

    Args:
        content: YAML file content

    Returns:
        Dictionary of parsed key-value pairs
    """
    data = {}
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()
    return data


def _validate_module_yaml(data: Dict[str, str], yaml_path: Path) -> Optional[str]:
    """
    Validate module.yaml content has all required fields.

    Args:
        data: Parsed YAML data
        yaml_path: Path to the YAML file (for error messages)

    Returns:
        Error message string if invalid, None if valid
    """
    missing = REQUIRED_METADATA_FIELDS - set(data.keys())
    if missing:
        return f"module.yaml missing required fields: {', '.join(sorted(missing))} in {yaml_path}"

    # Check for empty values
    for field in REQUIRED_METADATA_FIELDS:
        if not data[field]:
            return f"module.yaml has empty '{field}' field in {yaml_path}"

    return None


def get_module_metadata(module_input: str) -> Optional[ModuleMetadata]:
    """
    Get metadata for a module from its module.yaml file.

    Args:
        module_input: Module number ("01") or folder name ("01_tensor")

    Returns:
        ModuleMetadata or None if not found/parseable
    """
    folder = get_module_name(module_input)
    if not folder:
        return None

    project_root = _find_project_root()
    yaml_file = project_root / 'src' / folder / 'module.yaml'

    if not yaml_file.exists():
        return None

    try:
        content = yaml_file.read_text(encoding='utf-8')
        data = _parse_yaml_file(content)

        # Validate
        error = _validate_module_yaml(data, yaml_file)
        if error:
            # Log warning but don't crash
            import sys
            print(f"Warning: {error}", file=sys.stderr)
            return None

        return ModuleMetadata(
            title=data['title'],
            subtitle=data['subtitle'],
            description=data['description']
        )
    except Exception as e:
        import sys
        print(f"Warning: Failed to parse {yaml_file}: {e}", file=sys.stderr)
        return None


@lru_cache(maxsize=1)
def get_all_module_metadata() -> Dict[str, ModuleMetadata]:
    """
    Get metadata for all modules.

    Returns: {"01": ModuleMetadata(...), "02": ModuleMetadata(...), ...}
    Cached for performance.
    """
    mapping = _discover_modules()
    result = {}

    for num in mapping:
        metadata = get_module_metadata(num)
        if metadata:
            result[num] = metadata

    return result


def clear_cache():
    """Clear all module caches (useful after adding new modules)."""
    _discover_modules.cache_clear()
    get_all_module_metadata.cache_clear()
