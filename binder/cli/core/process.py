"""Process-group helpers shared by the build, debug, and parallel runners.

Quarto renders through a tree of child processes (Pandoc, LaTeX, Python
kernels). Binder starts each renderer in its own session so the whole tree
can be stopped at once, and gives the renderer a ``PYTHONPATH`` that imports
this checkout's sources ahead of any installed copy.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence


def local_render_env(root_dir: Path, base: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return an environment whose ``PYTHONPATH`` imports *root_dir*'s sources first.

    The repository root and its nested ``mlsysim`` package directory lead the
    path, so a globally installed or stale editable checkout never shadows the
    worktree being rendered. An existing ``PYTHONPATH`` is kept after them.

    Args:
        root_dir: Repository root of the checkout being rendered.
        base: Environment to start from; defaults to ``os.environ``.

    Returns:
        A new environment dictionary.
    """
    env = dict(os.environ if base is None else base)
    root = Path(root_dir).resolve()
    paths = [str(root), str((root / "mlsysim").resolve())]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def start_process_group(cmd: Sequence[str], **popen_kwargs) -> subprocess.Popen:
    """Start *cmd* as the leader of a new session.

    Every process the command spawns joins that session's process group, so
    :func:`stop_process_group` can stop the whole tree. Keyword arguments are
    passed to :class:`subprocess.Popen`.
    """
    return subprocess.Popen(list(cmd), start_new_session=True, **popen_kwargs)


def stop_process_group(process: Optional[subprocess.Popen], grace_seconds: float = 0.0) -> None:
    """Stop a process started with :func:`start_process_group` and wait for it.

    With a grace period the group first receives SIGTERM, which lets a child
    ``binder`` stop its own renderer and restore generated files; anything
    still running when the grace period ends is killed. Without a grace period
    the group is killed immediately. Where process groups are unavailable
    (Windows), the process itself is terminated or killed instead.

    Args:
        process: Process to stop; ``None`` or an already exited process is a no-op.
        grace_seconds: Seconds to wait after SIGTERM before killing.
    """
    if process is None or process.poll() is not None:
        return
    if grace_seconds > 0:
        _signal_group(process, terminate=True)
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
    _signal_group(process, terminate=False)
    process.wait()


def _signal_group(process: subprocess.Popen, terminate: bool) -> None:
    """Send SIGTERM (when *terminate*) or SIGKILL to *process*'s group, ignoring exit races."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM if terminate else signal.SIGKILL)
        elif terminate:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass  # The group exited between poll() and the signal.


@contextlib.contextmanager
def interrupts_as_keyboard_interrupt() -> Iterator[None]:
    """Raise ``KeyboardInterrupt`` for SIGINT and SIGTERM inside the block.

    Build code then unwinds through its own ``finally`` blocks, which stop the
    renderer and restore generated files, instead of exiting mid-cleanup. The
    previous handlers are reinstated on exit. Python only allows signal
    handlers on the main thread, so on other threads the block runs unchanged.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(signum, frame):
        """Signal handler that raises ``KeyboardInterrupt`` naming the signal."""
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, _raise)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
