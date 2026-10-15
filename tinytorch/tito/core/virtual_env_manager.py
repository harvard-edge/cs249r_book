import os, sys, json
from pathlib import Path

DEFAULT_VENV = ".venv"
CONFIG_FILE = ".tinyrc"


def get_venv_bin_dir(venv_path: Path) -> Path:
    """Return the bin directory for a venv (Scripts/ on Windows, bin/ on Unix)."""
    if sys.platform == "win32" or os.name == "nt":
        return venv_path / "Scripts"
    return venv_path / "bin"


def resolve_venv_tool(tool_name: str, venv_path: Path = None) -> str:
    """Resolve a console-script tool to its venv executable, falling back to PATH.

    A bare name like "jupyter" or "jupytext" is only found by subprocess if
    it's on PATH, which isn't true when tito runs outside an activated venv
    shell even though the tool is correctly installed inside .venv. Checking
    the venv's bin/Scripts dir first avoids that false "not installed" result.
    """
    venv_bin_dir = get_venv_bin_dir(venv_path or get_venv_path())
    exe = venv_bin_dir / (f"{tool_name}.exe" if sys.platform == "win32" else tool_name)
    return str(exe) if exe.exists() else tool_name


def get_venv_path() -> Path:
    """
    Fetch venv in case users have a custom path
    """
    # print(f"running this from {os.getcwd()}")  # Debug output - commented out for clean CLI
    if "VENV_PATH" in os.environ:
        return Path(os.environ["VENV_PATH"]).expanduser().resolve()

    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, encoding='utf-8') as f:
                cfg = json.load(f)
            return Path(cfg.get("venv_path", DEFAULT_VENV)).expanduser().resolve()
        except Exception:
            pass

    return Path(DEFAULT_VENV).resolve()
