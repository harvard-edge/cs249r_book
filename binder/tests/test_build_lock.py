"""The per-checkout build lock stops two builds from colliding in one worktree."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from binder.cli.core.process import BUILD_LOCK_NAME, BuildInProgress, build_lock  # noqa: E402

pytestmark = pytest.mark.skipif(os.name != "posix", reason="flock is POSIX only")

HOLDER = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {repo!r})
    from pathlib import Path
    from binder.cli.core.process import build_lock
    with build_lock(Path({root!r})):
        print("held", flush=True)
        time.sleep(60)
    """
)


def _start_holder(root: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLDER.format(repo=str(REPO_ROOT), root=str(root))],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "held"
    return proc


def test_lock_is_reentrant_across_sequential_builds(tmp_path: Path) -> None:
    with build_lock(tmp_path):
        pass
    with build_lock(tmp_path):
        pass


def test_second_build_in_same_checkout_fails_and_names_the_holder(tmp_path: Path) -> None:
    holder = _start_holder(tmp_path)
    try:
        with pytest.raises(BuildInProgress) as err:
            with build_lock(tmp_path):
                pytest.fail("the lock should not have been granted")
        assert str(holder.pid) in str(err.value)
        assert "separate worktree" in str(err.value)
    finally:
        holder.kill()
        holder.wait()


def test_killed_build_leaves_no_stale_lock(tmp_path: Path) -> None:
    holder = _start_holder(tmp_path)
    holder.send_signal(signal.SIGKILL)
    holder.wait()
    deadline = time.time() + 5
    while True:
        try:
            with build_lock(tmp_path):
                break
        except BuildInProgress:
            assert time.time() < deadline, "lock was not released after the holder died"
            time.sleep(0.1)
    assert (tmp_path / BUILD_LOCK_NAME).exists()


def test_separate_worktrees_do_not_contend(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    holder = _start_holder(a)
    try:
        with build_lock(b):
            pass
    finally:
        holder.kill()
        holder.wait()
