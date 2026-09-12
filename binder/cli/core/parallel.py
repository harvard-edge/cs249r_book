"""Run binder builds side by side, each in its own git worktree.

``binder build ... --parallel N`` and ``binder debug ... --parallel N`` turn a
request into :class:`BuildJob` objects and hand them to :func:`run_jobs`. The
runner uses N worker threads; each worker owns one :class:`BuildSession`, a
workspace (see :mod:`workspace`) that it reuses for every job it picks up. A
job runs ``binder build`` inside the workspace, and its log and output
directory are moved into the invoking checkout under the run directory, so
results outlive the workspace.

``binder debug`` also drives a :class:`BuildSession` directly for its section
bisection, which needs a single workspace and chooses each build from the
previous result.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import ConfigManager, get_output_file
from .discovery import VOLUME_DIRS
from .process import start_process_group, stop_process_group
from .workspace import (
    Snapshot,
    Workspace,
    create_workspace,
    remove_workspace,
    take_snapshot,
    workspace_path,
)

FORMATS = ("html", "pdf", "epub")

#: Default worker count: a quarter of the cores, clamped to 1-4. Each worker
#: renders a whole Quarto project, which is CPU- and memory-heavy.
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 4) // 4))

#: Seconds a stopped build gets after SIGTERM to clean up before it is killed.
STOP_GRACE_SECONDS = 15

#: Per-job time limit, matching the renderer timeout inside ``binder build``.
DEFAULT_TIMEOUT_SECONDS = 1800


@dataclass(frozen=True)
class BuildJob:
    """One ``binder build`` invocation to run in a workspace.

    Attributes:
        format_type: ``html``, ``pdf``, or ``epub``.
        volume: Volume directory name such as ``vol1``.
        chapter: Canonical chapter file stem; ``None`` builds the whole volume.
        extra_args: Additional ``binder build`` flags, such as ``--skip-validate``.
        label: Name that distinguishes jobs with identical arguments, such as
            the steps of a section bisection.
    """

    format_type: str
    volume: str
    chapter: Optional[str] = None
    extra_args: Tuple[str, ...] = ()
    label: str = ""

    @property
    def name(self) -> str:
        """Identifier for the job's log and output directory."""
        if self.label:
            return self.label
        return "-".join(part for part in (self.volume, self.format_type, self.chapter) if part)

    def binder_args(self) -> List[str]:
        """Arguments for ``binder`` that perform this job."""
        args = ["build", self.format_type]
        if self.chapter:
            args.append(self.chapter)
        return [*args, f"--{self.volume}", *self.extra_args]

    def output_dir(self, workspace_root: Path) -> Path:
        """Directory this job's build writes to inside the checkout at *workspace_root*.

        Mirrors ``BuildCommand``: a volume build uses the configured
        ``project.output-dir`` and a chapter build writes to
        ``chapters/<stem>`` beneath it.
        """
        base = ConfigManager(workspace_root).get_output_dir(self.format_type, self.volume)
        return base / "chapters" / self.chapter if self.chapter else base


@dataclass
class JobResult:
    """Outcome of one :class:`BuildJob`.

    Attributes:
        job: The job that ran.
        ok: True when ``binder build`` exited 0 and produced its artifact.
        returncode: Exit status, or ``None`` if the build never ran or was stopped.
        seconds: Wall-clock duration, including workspace preparation.
        log_path: Combined standard output and error of the build.
        output_dir: Collected output in the invoking checkout, if any was written.
        artifact: Primary artifact inside ``output_dir`` (PDF, EPUB, or ``index.html``).
        note: Short explanation when the job did not succeed normally.
    """

    job: BuildJob
    ok: bool
    returncode: Optional[int]
    seconds: float
    log_path: Path
    output_dir: Optional[Path] = None
    artifact: Optional[Path] = None
    note: str = ""

    def log_text(self) -> str:
        """Return the build log, or an empty string when none was written."""
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def to_dict(self) -> dict:
        """JSON-serializable summary used for ``summary.json`` and ``--json`` output."""
        return {
            "job": self.job.name,
            "format": self.job.format_type,
            "volume": self.job.volume,
            "chapter": self.job.chapter,
            "ok": self.ok,
            "returncode": self.returncode,
            "seconds": round(self.seconds, 1),
            "log": str(self.log_path),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "artifact": str(self.artifact) if self.artifact else None,
            "note": self.note,
        }


def new_run_id() -> str:
    """Return a timestamped identifier that is unique to this process."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"


class BuildSession:
    """A workspace that runs build jobs one at a time.

    Use it as a context manager: the workspace is created on the first job and
    removed on exit unless ``keep`` is set.

    Args:
        snapshot: Content the workspace is created from.
        run_dir: Directory in the invoking checkout that receives job logs and output.
        name: Workspace directory name, unique within the run.
        run_id: Groups this run's workspaces under the workspace root.
        isolate_cache: Give builds a private ``XDG_CACHE_HOME``. The Pandoc
            diagram filter writes cache files non-atomically, so builds that
            run at the same time must not share its cache.
        keep: Leave the workspace on disk for inspection.
        timeout_seconds: Per-job limit before the build is stopped.
    """

    def __init__(
        self,
        snapshot: Snapshot,
        run_dir: Path,
        *,
        name: str,
        run_id: str,
        isolate_cache: bool = False,
        keep: bool = False,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        """Store the session settings; the workspace is created lazily by ``path``."""
        self.snapshot = snapshot
        self.run_dir = Path(run_dir)
        self.name = name
        self.run_id = run_id
        self.isolate_cache = isolate_cache
        self.keep = keep
        self.timeout_seconds = timeout_seconds
        self.workspace: Optional[Workspace] = None
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def __enter__(self) -> "BuildSession":
        """Return the session itself."""
        return self

    def __exit__(self, *exc_info) -> bool:
        """Close the session and let any exception propagate."""
        self.close()
        return False

    @property
    def path(self) -> Path:
        """Workspace root, created on first use."""
        if self.workspace is None:
            self.workspace = create_workspace(self.snapshot, workspace_path(self.run_id, self.name))
        return self.workspace.path

    def run(self, job: BuildJob, prepare: Optional[Callable[[Path], None]] = None) -> JobResult:
        """Run *job* in the workspace and collect its log and output.

        Args:
            job: The build to run.
            prepare: Called with the workspace root before the build starts,
                for callers that edit sources inside the workspace.

        Returns:
            The job's result; setup errors are reported as a failed result.

        Raises:
            KeyboardInterrupt: Re-raised after the running build is stopped.
        """
        job_dir = self.run_dir / job.name
        if job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True)
        log_path = job_dir / "build.log"
        start = time.monotonic()

        try:
            root = self.path
            if prepare is not None:
                prepare(root)
            stale = job.output_dir(root)
            if stale.exists():
                shutil.rmtree(stale)
        except Exception as error:
            log_path.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
            return JobResult(job, False, None, time.monotonic() - start, log_path,
                             note=f"setup failed: {error}")

        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if self.isolate_cache:
            env["XDG_CACHE_HOME"] = str(root / ".binder-cache")
        cmd = [sys.executable, str(root / "binder" / "binder"), "-v", *job.binder_args()]
        returncode: Optional[int] = None
        note = ""
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ ./binder/binder {' '.join(cmd[3:])}\n# workspace: {root}\n\n")
            log.flush()
            process = start_process_group(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            with self._lock:
                self._process = process
            try:
                returncode = process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                stop_process_group(process, STOP_GRACE_SECONDS)
                note = f"timed out after {self.timeout_seconds}s"
            except KeyboardInterrupt:
                stop_process_group(process, STOP_GRACE_SECONDS)
                raise
            finally:
                with self._lock:
                    self._process = None

        output_dir = artifact = None
        built = job.output_dir(root)
        if built.is_dir():
            output_dir = job_dir / "output"
            shutil.move(str(built), str(output_dir))
            artifact = get_output_file(output_dir, job.format_type)
        if not note:
            if returncode not in (0, None):
                note = f"exit code {returncode}"
            elif artifact is None:
                note = f"no {job.format_type} artifact produced"
        ok = returncode == 0 and artifact is not None
        return JobResult(job, ok, returncode, time.monotonic() - start, log_path, output_dir, artifact, note)

    def stop(self) -> None:
        """Stop the build that is running, if any. Safe to call from another thread."""
        with self._lock:
            process = self._process
        stop_process_group(process, STOP_GRACE_SECONDS)

    def close(self) -> None:
        """Remove the workspace unless ``keep`` was requested."""
        if self.workspace is not None and not self.keep:
            remove_workspace(self.workspace)
            self.workspace = None


def run_jobs(
    repo_root: Path,
    jobs: Sequence[BuildJob],
    run_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    keep_workspaces: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    on_start: Optional[Callable[[BuildJob], None]] = None,
    on_finish: Optional[Callable[[JobResult], None]] = None,
) -> List[JobResult]:
    """Run *jobs* on up to *workers* workspaces and return the results in job order.

    ``run_dir/summary.json`` records every finished job. On Ctrl-C every
    running build is stopped, queued jobs are cancelled, workspaces are
    removed, and ``KeyboardInterrupt`` propagates.

    Args:
        repo_root: Checkout whose working tree is snapshotted for the workspaces.
        jobs: Jobs to run; their names must be unique.
        run_dir: Directory that receives one subdirectory per job.
        workers: Maximum number of builds running at once.
        keep_workspaces: Leave workspaces on disk for inspection.
        timeout_seconds: Per-job limit.
        on_start: Called from the worker thread when a job starts.
        on_finish: Called from the calling thread as each job finishes.

    Raises:
        ValueError: If two jobs share a name.
    """
    jobs = list(jobs)
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("Build jobs must have unique names")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(workers, len(jobs) or 1))
    snapshot = take_snapshot(repo_root)
    run_id = new_run_id()

    sessions: List[BuildSession] = []
    sessions_lock = threading.Lock()
    local = threading.local()

    def session_for_thread() -> BuildSession:
        """Return this worker thread's session, creating and registering it on first use."""
        if not hasattr(local, "session"):
            with sessions_lock:
                local.session = BuildSession(
                    snapshot, run_dir, name=f"worker-{len(sessions) + 1}", run_id=run_id,
                    isolate_cache=workers > 1, keep=keep_workspaces, timeout_seconds=timeout_seconds)
                sessions.append(local.session)
        return local.session

    def execute(job: BuildJob) -> JobResult:
        """Call ``on_start`` and run *job* in the current thread's session."""
        if on_start is not None:
            on_start(job)
        return session_for_thread().run(job)

    results: Dict[str, JobResult] = {}
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="binder-build")
    futures = {pool.submit(execute, job): job for job in jobs}
    try:
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = JobResult(job, False, None, 0.0, run_dir / job.name / "build.log",
                                   note=f"runner error: {error}")
            results[job.name] = result
            if on_finish is not None:
                on_finish(result)
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        with sessions_lock:
            running = list(sessions)
        for session in running:
            session.stop()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for session in sessions:
            session.close()
        finished = [results[job.name] for job in jobs if job.name in results]
        _write_summary(run_dir, finished, snapshot)
    return [results[job.name] for job in jobs]


def _write_summary(run_dir: Path, results: Sequence[JobResult], snapshot: Snapshot) -> None:
    """Write ``summary.json`` describing the snapshot and each finished job."""
    payload = {
        "commit": snapshot.commit,
        "untracked_files": len(snapshot.untracked),
        "jobs": [result.to_dict() for result in results],
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plan_build_jobs(
    discovery,
    formats: Sequence[str],
    volumes: Sequence[str],
    chapters: Sequence[str] = (),
    *,
    each_chapter: bool = False,
    extra_args: Sequence[str] = (),
    volume_pdf_args: Sequence[str] = (),
) -> List[BuildJob]:
    """Expand a build request into one job per format, volume, and chapter.

    Args:
        discovery: ``ChapterDiscovery`` used to resolve chapter names.
        formats: Output formats to build.
        volumes: Volumes to build.
        chapters: Chapter names, titles, or glob/``re:`` patterns. Each
            resolved chapter becomes its own job; they need a single volume.
        each_chapter: Build every chapter of each volume, in the order of the
            volume's PDF configuration, as separate jobs.
        extra_args: ``binder build`` flags passed to every job.
        volume_pdf_args: Flags that only apply to whole-volume PDF builds,
            such as ``--no-cover`` and ``--print-marks``.

    Returns:
        Jobs ordered by volume, then format, then chapter.

    Raises:
        ValueError: For unknown formats or volumes, a chapter from another
            volume, conflicting options, or an empty plan.
        FileNotFoundError: When a chapter does not resolve.
    """
    unknown_formats = [fmt for fmt in formats if fmt not in FORMATS]
    if unknown_formats:
        raise ValueError(f"Unknown format: {', '.join(unknown_formats)}")
    if not volumes:
        raise ValueError("Select volumes with --volN or --all")
    unknown_volumes = [vol for vol in volumes if vol not in VOLUME_DIRS]
    if unknown_volumes:
        raise ValueError(f"Parallel builds support {', '.join(VOLUME_DIRS)}; got {', '.join(unknown_volumes)}")
    if chapters and each_chapter:
        raise ValueError("Pass chapter names or --each-chapter, not both")
    if chapters and len(set(volumes)) != 1:
        raise ValueError("Chapter builds need exactly one volume")

    def stems_for(volume: str) -> List[Optional[str]]:
        """Return the de-duplicated chapter stems for *volume*; ``[None]`` means the whole volume.

        Raises ValueError when a resolved chapter belongs to another volume.
        """
        if each_chapter:
            return list(dict.fromkeys(discovery.get_chapters_from_config(volume)))
        if not chapters:
            return [None]
        stems: List[Optional[str]] = []
        for name in discovery.expand_chapter_patterns(list(chapters), volume=volume):
            spec = name if "/" in name else f"{volume}/{name}"
            for path in discovery.validate_chapters([spec]):
                owner = discovery._get_volume_from_path(path)
                if owner not in (None, volume):
                    raise ValueError(f"Chapter {name!r} belongs to {owner}, not {volume}")
                stems.append(path.stem)
        return list(dict.fromkeys(stems))

    jobs: List[BuildJob] = []
    for volume in dict.fromkeys(volumes):
        stems = stems_for(volume)
        for format_type in dict.fromkeys(formats):
            for stem in stems:
                args = list(extra_args)
                if stem is None and format_type == "pdf":
                    args.extend(volume_pdf_args)
                jobs.append(BuildJob(format_type, volume, stem, tuple(args)))
    if not jobs:
        raise ValueError("Nothing to build")
    return jobs


def results_table(results: Sequence[JobResult]):
    """Return a Rich table with one row per job: result, duration, and where to look."""
    from rich.markup import escape
    from rich.table import Table

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Job")
    table.add_column("Result")
    table.add_column("Time", justify="right")
    table.add_column("Artifact or log")
    for result in results:
        status = "[green]ok[/green]" if result.ok else f"[red]failed[/red] [dim]{escape(result.note)}[/dim]"
        where = result.artifact if result.ok and result.artifact else result.log_path
        table.add_row(escape(result.job.name), status, f"{result.seconds:.0f}s", escape(str(where)))
    return table
