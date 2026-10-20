import os, sys, json, shutil
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
    shell even though the tool is correctly installed inside .venv.

    Search order:
      1. venv_path, when the caller names one explicitly.
      2. The directory of the interpreter running tito. The tools are installed
         beside tito itself, and this holds whatever the current directory is.
         Do not .resolve() sys.executable: a venv's python is a symlink to the
         base interpreter, and resolving it leaves the venv.
      3. The configured venv (VENV_PATH, .tinyrc, or ./.venv). These are
         relative to the current directory, so they only help from the
         project root.
      4. PATH, and finally the bare name so the caller's FileNotFoundError
         message still fires.
    """
    exe_name = f"{tool_name}.exe" if sys.platform == "win32" else tool_name
    candidates = []
    if venv_path is not None:
        candidates.append(get_venv_bin_dir(Path(venv_path)))
    candidates.append(Path(sys.executable).parent)
    if venv_path is None:
        candidates.append(get_venv_bin_dir(get_venv_path()))

    for bin_dir in candidates:
        exe = bin_dir / exe_name
        if exe.exists():
            return str(exe)
    return shutil.which(tool_name) or tool_name


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
