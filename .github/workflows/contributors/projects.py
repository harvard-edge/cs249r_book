"""Single source of truth helpers for contribution projects.

Loads ``.github/workflows/contributors/projects.json`` and exposes both a
small Python API and a CLI used by the GitHub Actions workflows.

Edit ``projects.json`` (and only ``projects.json``) to add, rename, or
reorder a project. Every consumer — generators, workflow env, trigger
paths, commit lists, bot replies — reads from there.

CLI commands (used from workflows that don't import Python directly):

    python3 projects.py keys             # PROJECTS env value
    python3 projects.py aliases          # PROJECT_ALIASES env value
    python3 projects.py dirs-overrides   # PROJECT_DIRS env value
    python3 projects.py update-files     # newline-separated file list
    python3 projects.py json             # pretty-printed config dump

The Python API (``projects()``, ``keys()``, ``dirs()`` …) is what the
generator scripts use directly via ``from projects import ...``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "projects.json"


def _config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def projects() -> list[dict]:
    """Return the ordered list of project entries from ``projects.json``."""
    return _config()["projects"]


def extra_files() -> dict:
    """Return the ``extra_files`` block (root README, root config, etc.)."""
    return _config().get("extra_files", {})


def keys() -> list[str]:
    """Canonical project keys, in render order."""
    return [p["key"] for p in projects()]


def dirs() -> dict[str, str]:
    """Map project key → on-disk directory."""
    return {p["key"]: p["dir"] for p in projects()}


def alias_pairs() -> list[tuple[str, str]]:
    """List of (alias, canonical_key) tuples."""
    out: list[tuple[str, str]] = []
    for p in projects():
        for a in p.get("aliases", []):
            out.append((a, p["key"]))
    return out


def dir_overrides() -> list[tuple[str, str]]:
    """(project_key, dir) pairs only where they differ.

    Used by workflow env to keep PROJECT_DIRS minimal — projects whose key
    already equals their dir are implicit and need no override.
    """
    return [(p["key"], p["dir"]) for p in projects() if p["key"] != p["dir"]]


def env_projects() -> str:
    return ",".join(keys())


def env_aliases() -> str:
    return ",".join(f"{a}:{k}" for a, k in alias_pairs())


def env_dirs_overrides() -> str:
    return ",".join(f"{k}:{d}" for k, d in dir_overrides())


def update_files() -> list[str]:
    """All file paths the update-contributors workflow should stage and check."""
    extra = extra_files()
    out: list[str] = []
    if "root_config" in extra:
        out.append(extra["root_config"])
    if "root_readme" in extra:
        out.append(extra["root_readme"])
    if "site_contributors_json" in extra:
        out.append(extra["site_contributors_json"])
    for p in projects():
        out.append(f"{p['dir']}/.all-contributorsrc")
        out.append(f"{p['dir']}/README.md")
    return out


_COMMANDS = {
    "keys": lambda: print(env_projects()),
    "aliases": lambda: print(env_aliases()),
    "dirs-overrides": lambda: print(env_dirs_overrides()),
    "update-files": lambda: print("\n".join(update_files())),
    "json": lambda: print(json.dumps(_config(), indent=2)),
}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in _COMMANDS:
        cmds = "|".join(_COMMANDS)
        print(f"usage: projects.py <{cmds}>", file=sys.stderr)
        return 2
    _COMMANDS[argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
