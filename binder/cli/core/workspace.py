"""Disposable git worktrees that keep concurrent binder builds apart.

A build writes at the Quarto project root (``_quarto.yml``, ``index.qmd``,
LaTeX intermediates, the ``.quarto`` cache), so two builds in one checkout
collide. A workspace is a detached ``git worktree`` of a snapshot of the
invoking checkout, created under the system temporary directory and removed
when the run ends. The snapshot carries staged and unstaged edits, and
untracked, non-ignored files are copied in, so a workspace builds what is on
disk rather than only what is committed.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

#: Parent directory of every workspace binder creates. Workspaces are deleted
#: recursively, so :func:`create_workspace` and :func:`remove_workspace`
#: refuse paths outside it.
WORKSPACE_ROOT = Path(tempfile.gettempdir()) / "binder-workspaces"


def workspace_path(run_id: str, name: str) -> Path:
    """Return the directory for workspace *name* of run *run_id* under :data:`WORKSPACE_ROOT`."""
    return WORKSPACE_ROOT / run_id / name


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    """Run ``git -C repo_root <args>`` and return its standard output.

    Raises:
        RuntimeError: If *check* is set and git exits non-zero.
    """
    result = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


@dataclass(frozen=True)
class Snapshot:
    """Content that every workspace of one run is created from.

    Attributes:
        repo_root: Checkout the snapshot was taken from.
        commit: Commit holding the tracked content, uncommitted edits included.
        untracked: Repository-relative paths of untracked, non-ignored files.
    """

    repo_root: Path
    commit: str
    untracked: Tuple[str, ...] = ()


def take_snapshot(repo_root: Path) -> Snapshot:
    """Capture *repo_root*'s working tree for workspace creation.

    ``git stash create`` records staged and unstaged edits as an unreferenced
    commit without touching the shared stash list; a clean tree falls back to
    ``HEAD``. Untracked files are listed, not committed, and copied into each
    workspace by :func:`create_workspace`.
    """
    repo_root = Path(repo_root).resolve()
    commit = _git(repo_root, "stash", "create").strip() or _git(repo_root, "rev-parse", "HEAD").strip()
    listed = _git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    return Snapshot(repo_root, commit, tuple(path for path in listed.split("\0") if path))


@dataclass
class Workspace:
    """A worktree created by :func:`create_workspace`."""

    path: Path
    snapshot: Snapshot


def create_workspace(snapshot: Snapshot, path: Path) -> Workspace:
    """Check out *snapshot* as a detached worktree at *path*.

    Repository hooks are disabled for the checkout; they set up developer
    tooling that a throwaway build tree does not need. Untracked files from the
    snapshot are copied in afterwards.

    Raises:
        ValueError: If *path* is not inside :data:`WORKSPACE_ROOT`.
        RuntimeError: If ``git worktree add`` fails.
    """
    path = _checked_path(path)
    no_hooks = path.parent / ".no-hooks"
    no_hooks.mkdir(parents=True, exist_ok=True)
    _git(snapshot.repo_root, "-c", f"core.hooksPath={no_hooks}",
         "worktree", "add", "--detach", "--force", str(path), snapshot.commit)
    for relative in snapshot.untracked:
        source = snapshot.repo_root / relative
        if source.is_file():
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return Workspace(path, snapshot)


def remove_workspace(workspace: Workspace) -> None:
    """Delete a workspace directory and its worktree registration.

    Anything left inside the workspace, including build output, is discarded;
    callers collect what they need first.

    Raises:
        ValueError: If the workspace path is not inside :data:`WORKSPACE_ROOT`.
    """
    path = _checked_path(workspace.path)
    repo_root = workspace.snapshot.repo_root
    _git(repo_root, "worktree", "remove", "--force", str(path), check=False)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    _git(repo_root, "worktree", "prune", check=False)


def _checked_path(path: Path) -> Path:
    """Return *path* resolved, refusing anything outside :data:`WORKSPACE_ROOT`.

    Workspaces are deleted recursively; confining them to one directory keeps
    a bad argument from ever pointing that deletion at a real checkout.
    """
    resolved = Path(path).resolve()
    root = WORKSPACE_ROOT.resolve()
    if root not in resolved.parents:
        raise ValueError(f"Workspace path must be inside {root}: {resolved}")
    return resolved
