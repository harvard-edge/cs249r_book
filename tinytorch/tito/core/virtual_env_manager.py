import os, sys, json
from pathlib import Path

DEFAULT_VENV = ".venv"
CONFIG_FILE = ".tinyrc"


def get_venv_bin_dir(venv_path: Path) -> Path:
    """Return the bin directory for a venv (Scripts/ on Windows, bin/ on Unix)."""
    if sys.platform == "win32" or os.name == "nt":
        return venv_path / "Scripts"
    return venv_path / "bin"


def get_venv_path() -> Path:
    """
    Fetch venv in case users have a custom path
    """
    # print(f"running this from {os.getcwd()}")  # Debug output - commented out for clean CLI
    if "VENV_PATH" in os.environ:
        return Path(os.environ["VENV_PATH"]).expanduser().resolve()

    if Path(CONFIG_FILE).exists():
        try:
            cfg = json.load(open(CONFIG_FILE))
            return Path(cfg.get("venv_path", DEFAULT_VENV)).expanduser().resolve()
        except Exception:
            pass

    return Path(DEFAULT_VENV).resolve()
