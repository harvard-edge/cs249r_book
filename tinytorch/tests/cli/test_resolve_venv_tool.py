"""
resolve_venv_tool() search-order tests.

tito launches console scripts (jupyter, jupytext, nbdev-*) by path so they
work outside an activated venv. The first version looked only in the
configured venv, which is resolved relative to the current directory, so
running tito from modules/ or any other subdirectory fell back to a bare
name and hit the same "not installed" error it was meant to fix. The
interpreter running tito is the reliable anchor: its bin dir is where the
tools were installed.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.core import virtual_env_manager as vem


def _fake_venv(root: Path, tools):
    bin_dir = vem.get_venv_bin_dir(root)
    bin_dir.mkdir(parents=True)
    for tool in tools:
        exe = bin_dir / (f"{tool}.exe" if sys.platform == "win32" else tool)
        exe.write_text("")
        exe.chmod(0o755)
    python = bin_dir / ("python.exe" if sys.platform == "win32" else "python")
    python.write_text("")
    return bin_dir, python


def test_finds_tool_beside_interpreter_from_a_subdirectory(tmp_path, monkeypatch):
    bin_dir, python = _fake_venv(tmp_path / ".venv", ["jupyter"])
    subdir = tmp_path / "modules"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    monkeypatch.delenv("VENV_PATH", raising=False)
    with patch.object(vem.sys, "executable", str(python)):
        assert Path(vem.resolve_venv_tool("jupyter")).parent == bin_dir


def test_explicit_venv_path_wins_over_interpreter(tmp_path):
    explicit_bin, _ = _fake_venv(tmp_path / "explicit", ["jupytext"])
    _, python = _fake_venv(tmp_path / "running", ["jupytext"])
    with patch.object(vem.sys, "executable", str(python)):
        resolved = Path(vem.resolve_venv_tool("jupytext", tmp_path / "explicit"))
    assert resolved.parent == explicit_bin


def test_falls_back_to_bare_name_when_nowhere(tmp_path, monkeypatch):
    _, python = _fake_venv(tmp_path / ".venv", [])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    with patch.object(vem.sys, "executable", str(python)):
        assert vem.resolve_venv_tool("no-such-tool-xyz") == "no-such-tool-xyz"


def test_does_not_follow_interpreter_symlink_out_of_venv(tmp_path):
    if os.name == "nt":
        return
    base_bin = tmp_path / "base" / "bin"
    base_bin.mkdir(parents=True)
    (base_bin / "python3.12").write_text("")
    venv_bin, _ = _fake_venv(tmp_path / "venv", ["jupyter"])
    link = venv_bin / "python3"
    link.symlink_to(base_bin / "python3.12")
    with patch.object(vem.sys, "executable", str(link)):
        assert Path(vem.resolve_venv_tool("jupyter")).parent == venv_bin
